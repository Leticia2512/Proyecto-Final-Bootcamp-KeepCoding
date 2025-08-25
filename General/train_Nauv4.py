# train_singlelabel_1cycle_ema.py
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4                  # LR base (backbone)
HEAD_LR_MULT = 6.0         # LR de cabeza/meta = LR * HEAD_LR_MULT
WEIGHT_DECAY = 2e-4
NUM_WORKERS = 4
SEED = 42
EXPERIMENT_NAME = "ExperimentoNau_singlelabel"

# Datasets guardados (Subsets .pt con (img, meta, y) o (img, meta, y, id))
TRAIN_PT = r"Data\dataloader\train_dataset.pt"
VAL_PT = r"Data\dataloader\val_dataset.pt"
TEST_PT = r"Data\dataloader\test_dataset.pt"

# Sampler balanceado (solo train)
USE_BALANCED_SAMPLER = True

# Early stopping
PATIENCE = 8

# ------------- UTILS -------------


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


def unpack_batch(batch):
    """Admite datasets que devuelven (x,m,y) o (x,m,y,id)."""
    if len(batch) == 4:
        x, m, y, _ = batch
    else:
        x, m, y = batch
    return x, m, y


def get_subset_targets(subset):
    base = subset.dataset
    idxs = subset.indices
    return np.array([int(base.targets[i]) for i in idxs], dtype=int)


def plot_confusion(cm, class_names, out_path: Path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ------------- MODELO -------------


class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )

        self.head = nn.Sequential(
            nn.Linear(in_feats + 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        f = torch.cat([f_img, f_meta], dim=1)
        return self.head(f)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names=None, prefix="val"):
    model.eval()
    total_loss = 0.0
    all_y, all_pred = [], []
    for batch in loader:
        x, m, y = unpack_batch(batch)
        x, m, y = x.to(device), m.to(device), y.to(device, dtype=torch.long)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        all_y.append(y.cpu())
        all_pred.append(pred.cpu())

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_pred).numpy()

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)

    rep_txt = classification_report(y_true, y_pred, digits=3)
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    out_dir = Path(f"reports_{prefix}")
    out_dir.mkdir(exist_ok=True)
    rep_path = out_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep_txt)

    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion(cm, class_names, cm_path)

    return {
        "loss": avg_loss,
        "acc": acc,
        "rep_path": str(rep_path),
        "cm_path": str(cm_path),
    }

# ----- EMA -----


@torch.no_grad()
def ema_init_like(model):
    ema = type(model)(model.meta[0].in_features,
                      model.head[-1].out_features, pretrained=False)
    ema.load_state_dict(model.state_dict())
    return ema


@torch.no_grad()
def ema_update(ema, src, decay=0.999):
    for (k, v_ema), v in zip(ema.state_dict().items(), src.state_dict().values()):
        v_ema.copy_(decay * v_ema + (1.0 - decay) * v)

# ------------- MAIN -------------


def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"

    # Carga datasets (Subsets guardados)
    train_ds = torch.load(TRAIN_PT, weights_only=False)
    val_ds = torch.load(VAL_PT,   weights_only=False)
    test_ds = torch.load(TEST_PT,  weights_only=False)

    # clases/meta_dim
    sample = train_ds[0]
    # con tus feature_cols=["Patient Age","Patient_Sex_Binario"] => 2
    meta_dim = sample[1].shape[0]
    train_targets = get_subset_targets(train_ds)
    num_classes = int(train_targets.max() + 1)
    print(f"num_classes={num_classes}, meta_dim={meta_dim}")

    # Sampler balanceado (opcional)
    if USE_BALANCED_SAMPLER:
        counts = np.bincount(train_targets, minlength=num_classes)
        class_w = 1.0 / np.clip(counts, 1, None)
        sample_w = class_w[train_targets]
        sampler = WeightedRandomSampler(
            sample_w, num_samples=len(sample_w), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=shuffle, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)

    # Modelo
    model = ImageClassifier(
        meta_dim=meta_dim, num_classes=num_classes, pretrained=True).to(device)

    # LR discriminativo
    param_groups = [
        {"params": model.backbone.parameters(), "lr": LR},
        {"params": model.meta.parameters(),     "lr": LR * HEAD_LR_MULT},
        {"params": model.head.parameters(),     "lr": LR * HEAD_LR_MULT},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # OneCycleLR (por batch)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR, LR * HEAD_LR_MULT, LR * HEAD_LR_MULT],
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # EMA
    ema_model = ema_init_like(model).to(device)
    ema_decay = 0.999

    # MLflow
    init_mlflow()
    run_name = "resnet18_singlelabel_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "head_lr_mult": HEAD_LR_MULT,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "num_classes": num_classes,
            "backbone": "resnet18_imagenet",
            "meta_dim": meta_dim,
            "balanced_sampler": USE_BALANCED_SAMPLER,
            "scheduler": "OneCycleLR",
            "ema_decay": ema_decay,
        })

        # AMP (API nueva en tu PyTorch 2.7)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        best_acc = -1.0
        bad_epochs = 0
        ckpt_path = Path("checkpoints/best_singlelabel_1cycle_ema.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                x, m, y = unpack_batch(batch)
                x, m, y = x.to(device), m.to(
                    device), y.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(x, m)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

                # EMA después del step
                ema_update(ema_model, model, ema_decay)

                total_loss += loss.item() * x.size(0)

                # OneCycleLR: step por batch
                scheduler.step()

            train_loss = total_loss / len(train_loader.dataset)

            # Validación con EMA (suele generalizar mejor)
            val_out = evaluate(ema_model, val_loader,
                               criterion, device, prefix="val")
            print(
                f"[{epoch:02d}] train_loss={train_loss:.4f} | val_loss={val_out['loss']:.4f} | acc={val_out['acc']:.4f}")

            # Log métricas + LR del primer grupo
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_out["loss"],
                "val_acc": val_out["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            improved = val_out["acc"] > best_acc + 1e-4
            if improved:
                best_acc = val_out["acc"]
                bad_epochs = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": ema_model.state_dict(),   # guardamos EMA
                    "optimizer_state": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "meta_dim": meta_dim
                }, ckpt_path)
                mlflow.log_artifact(str(ckpt_path))
                mlflow.log_artifact(val_out["rep_path"])
                mlflow.log_artifact(val_out["cm_path"])
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print("Early stopping!")
                    break

        # Test final con EMA
        test_out = evaluate(ema_model, test_loader,
                            criterion, device, prefix="test")
        print(
            f"[TEST] loss={test_out['loss']:.4f} | acc={test_out['acc']:.4f}")
        mlflow.log_metrics({
            "test_loss": test_out["loss"],
            "test_acc": test_out["acc"],
        })
        mlflow.log_artifact(test_out["rep_path"])
        mlflow.log_artifact(test_out["cm_path"])


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
