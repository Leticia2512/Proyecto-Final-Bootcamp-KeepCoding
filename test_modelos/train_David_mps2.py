import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import mlflow
from pathlib import Path
from datetime import datetime

# --- Fix para datasets guardados con eye_pytorch_dataset ---
import sys
import create_dataset.eye_pytorch_dataset as eye_pytorch_dataset
sys.modules["eye_pytorch_dataset"] = eye_pytorch_dataset


# ========================
# CONFIG
# ========================
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 5e-5
NUM_WORKERS = 0   # ⚠️ en Mac/MPS mejor 0
SEED = 42
EXPERIMENT_NAME = "Experimento_David_CustomNet_MPS_v2"

TRAIN_PT = "Data/dataset/train_dataset.pt"
VAL_PT = "Data/dataset/val_dataset.pt"
TEST_PT = "Data/dataset/test_dataset.pt"


# ========================
# UTILS
# ========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_mlflow():
    uri = "file:" + \
        str((Path(__file__).resolve().parent.parent / "mlruns_david").resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ========================
# MODELO PROPIO
# ========================
class MultiModalNet(nn.Module):
    def __init__(self, n_features=6, n_classes=4, activation_function=nn.ReLU):
        super(MultiModalNet, self).__init__()
        self.activation = activation_function()

        # Rama de metadatos (sin dropout)
        self.features_branch = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            self.activation,

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.activation,

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            self.activation,

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            self.activation
        )

        # Rama de imágenes (ahora con 3 capas conv+pool)
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            self.activation,
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.activation,
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            self.activation
        )

        # Clasificador final (dropout 0.2)
        self.classifier = nn.Sequential(
            nn.Linear(16 + 32, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, feats, imgs):
        x_feat = self.features_branch(feats)
        x_img = self.image_branch(imgs)
        x = torch.cat([x_feat, x_img], dim=1)
        return self.classifier(x)


# ========================
# EVALUACIÓN
# ========================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y, all_p = [], []

    for imgs, feats, y in loader:
        imgs, feats, y = imgs.to(device), feats.to(
            device), y.to(device, dtype=torch.long)
        logits = model(feats, imgs)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)

        all_y.append(y.cpu())
        all_p.append(logits.argmax(1).cpu())

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return {"loss": avg_loss, "acc": acc}


# ========================
# MAIN LOOP
# ========================
def main():
    set_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Usando GPU Apple Silicon con Metal (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS no disponible, usando CPU")

    # === Cargar datasets ===
    train_data = torch.load(TRAIN_PT, weights_only=False)
    val_data = torch.load(VAL_PT, weights_only=False)
    test_data = torch.load(TEST_PT, weights_only=False)

    train_ds = train_data["indices"]
    val_ds = val_data["indices"]
    test_ds = test_data["indices"]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    # Detectar dimensiones
    sample_batch = next(iter(train_loader))
    imgs, feats, y = sample_batch
    meta_dim = feats.shape[1]
    n_classes = len(torch.unique(y))

    # === Modelo ===
    model = MultiModalNet(n_features=meta_dim, n_classes=n_classes).to(device)

    out = model(feats.to(device), imgs.to(device))
    print("🔎 Test forward OK → salida:", out.shape)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # === MLflow ===
    init_mlflow()
    run_name = "custom_multimodalnet_mps_v2_" + \
        datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "num_classes": n_classes,
            "meta_dim": meta_dim,
            "device": str(device),
            "backbone": "CustomCNN_v2"
        })

        best_acc = -1.0
        ckpt_path = Path("checkpoints/best_david_custom_mps_v2.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for imgs, feats, y in train_loader:
                imgs, feats, y = imgs.to(device), feats.to(
                    device), y.to(device, dtype=torch.long)
                logits = model(feats, imgs)
                loss = criterion(logits, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)

            train_loss = total_loss / len(train_loader.dataset)
            scheduler.step()

            val_metrics = evaluate(model, val_loader, criterion, device)
            print(f"[{epoch:02d}] train_loss={train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | acc={val_metrics['acc']:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "num_classes": n_classes,
                    "meta_dim": meta_dim
                }, ckpt_path)
                mlflow.log_artifact(str(ckpt_path))

        # Test final
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(
            f"[TEST] loss={test_metrics['loss']:.4f} | acc={test_metrics['acc']:.4f}")
        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        })


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
