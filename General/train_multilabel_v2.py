from eye_pytorch_dataset import EyeDataset
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow

# ---- RUTAS / CONFIG ----
PARQUET = r"DataNau\dataset_eyes_long.parquet"
IMAGE_DIR = "224x224"
FEATURE_COLS = ["Patient Age", "Patient_Sex_Binario"]

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42

EXPERIMENT_NAME = "odir-eye-multilabel-v2"

# ---- DATASET ----


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_mlflow():
    uri = "file:" + str(Path("mlruns").resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


class MultiModalNet(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(in_feats + 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)  # logits
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        f = torch.cat([f_img, f_meta], dim=1)
        return self.head(f)


def compute_pos_weight(subset: Subset, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    total = 0
    for _, _, y in subset:
        counts += y
        total += 1
    pos = counts.clamp_min(1.0)
    neg = (total - counts).clamp_min(1.0)
    return neg / pos  # para BCEWithLogitsLoss


@torch.no_grad()
def best_thresh_for_f1(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, m, y in loader:
        x, m = x.to(device), m.to(device)
        all_logits.append(model(x, m).cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    probs = logits.sigmoid()
    best_f1, best_t = -1.0, 0.5
    for t in torch.linspace(0.05, 0.95, 19):
        pred = (probs >= t).float()
        f1 = f1_score(y.numpy(), pred.numpy(),
                      average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresh=0.5):
    model.eval()
    total_loss = 0.0
    all_y, all_p = [], []
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_y.append(y.cpu())
        all_p.append(logits.cpu())
    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).sigmoid().numpy()
    pred = (p >= thresh).astype(np.float32)

    avg_loss = total_loss / len(loader.dataset)
    precision = precision_score(y, pred, average="micro", zero_division=0)
    recall = recall_score(y, pred, average="micro", zero_division=0)
    f1 = f1_score(y, pred, average="micro", zero_division=0)
    return {"loss": avg_loss, "precision": precision, "recall": recall, "f1": f1}


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset duplicado para no pisar transforms entre splits
    ds_train = EyeDataset(PARQUET, IMAGE_DIR, FEATURE_COLS,
                          transform=train_tf, num_classes=None)
    ds_eval = EyeDataset(PARQUET, IMAGE_DIR, FEATURE_COLS,
                         transform=eval_tf,  num_classes=ds_train.num_classes)
    num_classes = ds_train.num_classes
    print("num_classes =", num_classes)

    # Splits por índices iguales en ambos datasets
    idx = np.arange(len(ds_train))
    tr, tmp = train_test_split(
        idx, test_size=0.2, random_state=SEED, shuffle=True)
    va, te = train_test_split(
        tmp, test_size=0.5, random_state=SEED, shuffle=True)

    train_set = Subset(ds_train, tr)
    val_set = Subset(ds_eval,  va)
    test_set = Subset(ds_eval,  te)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # pos_weight (suavizado)
    w = compute_pos_weight(train_set, num_classes)
    w = torch.clamp(torch.sqrt(w), max=50)  # suaviza + evita extremos
    print("pos_weight median (sqrt+clamp):", w.median().item())

    model = MultiModalNet(meta_dim=len(FEATURE_COLS),
                          num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=w.to(device))

    # MLflow
    init_mlflow()
    with mlflow.start_run(run_name="resnet18_meta_v2"):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "num_classes": num_classes,
            "backbone": "resnet18_imagenet",
            "meta_dim": len(FEATURE_COLS),
            "parquet": PARQUET,
            "image_dir": IMAGE_DIR,
        })

        best_f1 = -1.0
        best_ckpt = Path("checkpoints/best_v2.pt")
        best_ckpt.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            # train
            model.train()
            total_loss = 0.0
            for x, m, y in train_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
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

            # val @ 0.5
            val_metrics = evaluate(
                model, val_loader, criterion, device, thresh=0.5)
            # sweep de umbral
            best_t, best_f1_epoch = best_thresh_for_f1(
                model, val_loader, device)

            print(f"[{epoch:02d}] train_loss={train_loss:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | F1={val_metrics['f1']:.4f} "
                  f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
                  f"| best_t={best_t:.2f} F1*={best_f1_epoch:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_f1_at_0.5": val_metrics["f1"],
                "val_precision_at_0.5": val_metrics["precision"],
                "val_recall_at_0.5": val_metrics["recall"],
                "best_thresh": best_t,
                "best_f1": best_f1_epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            if best_f1_epoch > best_f1:
                best_f1 = best_f1_epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "feature_cols": FEATURE_COLS
                }, best_ckpt)
                mlflow.log_artifact(str(best_ckpt))

        # test final (umbral 0.5 por simplicidad)
        test_metrics = evaluate(
            model, test_loader, criterion, device, thresh=0.5)
        print(f"[TEST] loss={test_metrics['loss']:.4f} | "
              f"F1={test_metrics['f1']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}")
        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
        })


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
