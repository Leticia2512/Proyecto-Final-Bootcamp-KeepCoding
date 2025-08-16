# train_multilabel.py
from eye_pytorch_dataset import EyeDataset  # multi-label fijo
import os
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18

import mlflow

# -----------------------
# CONFIG
# -----------------------
PARQUET = r"DataNau\dataset_eyes_long.parquet"  # <- tu ruta
IMAGE_DIR = "224x224"
FEATURE_COLS = ["Patient Age", "Patient_Sex_Binario"]

BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
VAL_EVERY = 1
THRESH = 0.5
USE_SAMPLER = False  # pon True si hay clases MUY raras

EXPERIMENT_NAME = "odir-eye-multilabel"

# -----------------------
# DATASET (importa tu clase)
# -----------------------


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_mlflow():
    mlruns_dir = "mlruns"  # local; puedes subirlo con Git LFS si quieres
    uri = "file:" + str(Path(mlruns_dir).resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


class MultiModalNet(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = False):
        super().__init__()
        # Imagen: ResNet18
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # MLP para metadatos
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Fusión + cabeza
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
        logits = self.head(f)
        return logits


def compute_pos_weight(subset: Subset, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    total = 0
    for _, _, y in subset:
        counts += y
        total += 1
    pos = counts.clamp_min(1.0)
    neg = (total - counts).clamp_min(1.0)
    return (neg / pos)  # BCEWithLogitsLoss expects pos_weight for each class


@torch.no_grad()
def evaluate(model, loader, criterion, device, thresh=0.5):
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0.0

    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        pred = (logits.sigmoid() >= thresh).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
        tn += ((1 - pred) * (1 - y)).sum().item()

    n = len(loader.dataset)
    avg_loss = total_loss / n
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"loss": avg_loss, "precision": precision, "recall": recall, "f1": f1}


def make_loaders(ds, batch_size, num_workers, use_sampler, seed):
    # split 80/10/10
    idx = np.arange(len(ds))
    tr, tmp = train_test_split(
        idx, test_size=0.2, random_state=seed, shuffle=True)
    va, te = train_test_split(
        tmp, test_size=0.5, random_state=seed, shuffle=True)

    train_set = Subset(ds, tr)
    val_set = Subset(ds, va)
    test_set = Subset(ds, te)

    # DataLoaders
    if use_sampler:
        sample_weights = []
        for _, _, y in train_set:
            k = max(1, int(y.sum().item()))  # menos positivos => más peso
            sample_weights.append(1.0 / k)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, m, y in train_loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        logits = model(x, m)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    # Dataset completo (inferimos num_classes)
    ds = EyeDataset(
        parquet_path=PARQUET,
        image_dir=IMAGE_DIR,
        feature_cols=FEATURE_COLS,
        transform=None,
        num_classes=None
    )
    num_classes = ds.num_classes
    print(f"num_classes = {num_classes}")

    # Loaders
    train_set, val_set, test_set, train_loader, val_loader, test_loader = make_loaders(
        ds, BATCH_SIZE, NUM_WORKERS, USE_SAMPLER, SEED
    )

    # pos_weight para BCEWithLogitsLoss
    pos_weight = compute_pos_weight(train_set, num_classes)
    print("pos_weight (median):", pos_weight.median().item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalNet(meta_dim=len(FEATURE_COLS),
                          num_classes=num_classes, pretrained=False).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # MLflow
    init_mlflow()
    with mlflow.start_run(run_name="resnet18_meta_baseline"):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "num_classes": num_classes,
            "thresh": THRESH,
            "use_sampler": USE_SAMPLER,
            "backbone": "resnet18",
            "meta_dim": len(FEATURE_COLS),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "parquet": PARQUET,
            "image_dir": IMAGE_DIR,
        })

        best_f1 = -1.0
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt_dir / "best.pt"

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device)
            scheduler.step()

            if epoch % VAL_EVERY == 0:
                val_metrics = evaluate(
                    model, val_loader, criterion, device, THRESH)
                print(f"[{epoch:02d}] train_loss={train_loss:.4f} | "
                      f"val_loss={val_metrics['loss']:.4f} | F1={val_metrics['f1']:.4f} "
                      f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}")

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=epoch)

                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "num_classes": num_classes,
                        "feature_cols": FEATURE_COLS
                    }, best_ckpt)
                    mlflow.log_artifact(str(best_ckpt))

        # Test final con mejor checkpoint
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test_metrics = evaluate(model, test_loader, criterion, device, THRESH)
        print(f"[TEST] loss={test_metrics['loss']:.4f} | F1={test_metrics['f1']:.4f} "
              f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}")
        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"]
        })


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()  # Windows friendly
    main()
