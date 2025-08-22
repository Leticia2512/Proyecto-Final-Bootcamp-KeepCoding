import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score
import numpy as np
import mlflow
from pathlib import Path
from datetime import datetime
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import random

# ---- CONFIG ----
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-5 # LR más bajo para transfer learning
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
EXPERIMENT_NAME = "DataAvengersSofia"

# Rutas de los datasets guardados
TRAIN_PT = r"Data\dataset\train_dataset.pt"
VAL_PT = r"Data\dataset\val_dataset.pt"
TEST_PT = r"Data\dataset\test_dataset.pt"


# ---- UTILS ----
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_mlflow():
    uri = "file:" + str(Path("mlruns").resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ---- MODELO ----
class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # ResNet usa `fc` en lugar de `classifier`
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # capa para metadatos
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.LayerNorm(32), # Cambiado a LayerNorm
            nn.Dropout(0.5) # Añadido dropout
        )

        self.head = nn.Sequential(
            nn.Linear(in_feats + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Aumentado a 0.5
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        f = torch.cat([f_img, f_meta], dim=1)
        return self.head(f)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y, all_p = [], []
    for x, m, y in loader:   
        x, m, y = x.to(device), m.to(device), y.to(device, dtype=torch.long)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_y.append(y.cpu())
        all_p.append(logits.argmax(1).cpu())
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return {"loss": avg_loss, "acc": acc}


# ---- MAIN ----
def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar datasets guardados (.pt)
    try:
        train_data = torch.load(TRAIN_PT, weights_only=False)
        val_data = torch.load(VAL_PT, weights_only=False)
        test_data = torch.load(TEST_PT, weights_only=False)

        # ACCEDE AL OBJETO SUBSET DENTRO DEL DICCIONARIO
        train_ds = train_data["indices"]
        val_ds = val_data["indices"]
        test_ds = test_data["indices"]
    except Exception as e:
        print(f"Error al cargar archivos .pt: {e}")
        return


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False)

    num_classes = len(set(train_ds.dataset.targets))
    meta_dim = len(train_ds.dataset.feature_cols) 
    print(f"num_classes={num_classes}, meta_dim={meta_dim}")

    # Modelo
    model = ImageClassifier(
        meta_dim=meta_dim, num_classes=num_classes, pretrained=True).to(device)

    optimizer = SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY) # Cambio el optimizador

    scheduler = StepLR(optimizer, step_size=3, gamma=0.5) # Reduce LR a la mitad cada 3 épocas

    criterion = nn.CrossEntropyLoss()

    # MLflow
    init_mlflow()
    run_name = "ResNet50_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "num_classes": num_classes,
            "backbone": "ResNet50",
            "meta_dim": meta_dim,
            "optimizer": "SGD",
            "scheduler": "StepLR"
        })

        best_acc = -1.0
        ckpt_path = Path("checkpoints/best_singlelabel.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for x, m, y in train_loader:
                x, m, y = x.to(device), m.to(
                    device), y.to(device, dtype=torch.long)
                logits = model(x, m)
                loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * x.size(0)
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
                    "num_classes": num_classes,
                    "meta_dim": meta_dim
                }, ckpt_path)
                mlflow.log_artifact(str(ckpt_path))

        # test final
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