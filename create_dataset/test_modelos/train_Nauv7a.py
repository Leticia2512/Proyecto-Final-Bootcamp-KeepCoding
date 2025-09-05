
import matplotlib.pyplot as plt
import mlflow
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ============= CONFIG (Run A) =============
BATCH_SIZE = 32
EPOCHS = 35
LR = 1e-4        # LR base (backbone)
HEAD_LR_MULT = 8.0         # LR para meta+head = LR * HEAD_LR_MULT
WEIGHT_DECAY = 5e-4        # ↑ regularización (Run A)
NUM_WORKERS = 4           # en Windows, si da guerra, pon 0
SEED = 42
EXPERIMENT_NAME = "ExperimentoNau_singlelabel"

TRAIN_PT = r"Data\dataloader\train_dataset.pt"
VAL_PT = r"Data\dataloader\val_dataset.pt"
TEST_PT = r"Data\dataloader\test_dataset.pt"

# OneCycleLR con más warmup (Run A)
ONECYCLE_PCT_START = 0.20
DIV_FACTOR = 10.0
FINAL_DIV_FACTOR = 100.0

# EMA
EMA_DECAY = 0.9995

# Label smoothing (dejamos 0.05 para aislar efectos del Run A)
LABEL_SMOOTH = 0.05
# =========================================


# ---------- Helpers top-level (picklables) ----------
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


def unpack_item(item):
    """Admite (x,m,y) o (x,m,y,id). Devuelve (x,m,y)."""
    if len(item) == 4:
        x, m, y, _ = item
    else:
        x, m, y = item
    return x, m, y


class NormalizeCollate:
    """Collate picklable (Windows-safe). Normaliza columna 'edad' (índice 0)."""

    def __init__(self, mu: float, sigma: float, age_index: int = 0):
        self.mu = float(mu)
        self.sigma = float(sigma) if sigma != 0 else 1.0
        self.age_index = int(age_index)

    def __call__(self, batch):
        xs, ms, ys = [], [], []
        for it in batch:
            x, m, y = unpack_item(it)
            xs.append(x)
            ms.append(m)
            ys.append(int(y.item()) if torch.is_tensor(y) else int(y))
        x = torch.stack(xs)
        m = torch.stack(ms)
        y = torch.tensor(ys, dtype=torch.long)
        m[:, self.age_index] = (m[:, self.age_index] - self.mu) / self.sigma
        return x, m, y


def get_subset_targets(subset):
    base, idxs = subset.dataset, subset.indices
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


# ---------------- Modelo ----------------
class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, dropout: float = 0.6, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet34(weights=weights)
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
            nn.Dropout(dropout),     # Run A: 0.6
            nn.Linear(512, num_classes),
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        f = torch.cat([f_img, f_meta], dim=1)
        return self.head(f)


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val"):
    model.eval()
    total_loss = 0.0
    all_y, all_pred = [], []
    correct_top5 = 0

    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device, dtype=torch.long)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        pred = logits.argmax(1)
        all_y.append(y.cpu())
        all_pred.append(pred.cpu())

        k = min(5, logits.shape[1])
        top5 = torch.topk(logits, k=k, dim=1).indices
        correct_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_pred).numpy()
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    top5_acc = correct_top5 / len(loader.dataset)

    # artefactos
    rep_txt = classification_report(y_true, y_pred, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(i) for i in range(cm.shape[0])]

    out_dir = Path(f"reports_{prefix}")
    out_dir.mkdir(exist_ok=True)
    rep_path = out_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep_txt)
    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion(cm, class_names, cm_path)

    return {"loss": avg_loss, "acc": acc, "top5": top5_acc,
            "rep_path": str(rep_path), "cm_path": str(cm_path)}


# ---------------- Train ----------------
def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"

    # Carga datasets (Subsets .pt)
    train_ds = torch.load(TRAIN_PT, weights_only=False)
    val_ds = torch.load(VAL_PT,   weights_only=False)
    test_ds = torch.load(TEST_PT,  weights_only=False)

    meta_dim = train_ds[0][1].shape[0]
    train_targets = get_subset_targets(train_ds)
    num_classes = int(train_targets.max() + 1)
    print(f"num_classes={num_classes}, meta_dim={meta_dim}")

    # Normalización edad (columna 0 de meta) usando SOLO train
    ages = np.array([float(train_ds.dataset.meta[i][0])
                    for i in train_ds.indices])
    mu, sigma = float(ages.mean()), float(ages.std() + 1e-8)
    collate = NormalizeCollate(mu, sigma, age_index=0)

    # Pesos por clase + sampler balanceado
    counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
    class_weights = class_weights / class_weights.mean()
    w_tensor = torch.tensor(class_weights, device=device)

    sample_w = class_weights[train_targets]
    sample_w = (sample_w / sample_w.mean()).astype(np.float32)
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=pin_mem, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=pin_mem, collate_fn=collate)

    # Modelo
    model = ImageClassifier(
        meta_dim=meta_dim, num_classes=num_classes, dropout=0.6, pretrained=True).to(device)

    # Optimizador con LR discriminativo
    param_groups = [
        {"params": model.backbone.parameters(), "lr": LR},
        {"params": list(model.meta.parameters()) + list(model.head.parameters()),
         "lr": LR * HEAD_LR_MULT},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # OneCycleLR por batch (Run A: pct_start=0.2)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR, LR * HEAD_LR_MULT],
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=ONECYCLE_PCT_START,
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
    )

    criterion = nn.CrossEntropyLoss(
        weight=w_tensor, label_smoothing=LABEL_SMOOTH)

    # EMA (modelo clonado)
    ema_model = ImageClassifier(
        meta_dim=meta_dim, num_classes=num_classes, dropout=0.6, pretrained=False).to(device)
    ema_model.load_state_dict(model.state_dict())

    @torch.no_grad()
    def ema_update(decay=EMA_DECAY):
        for (k, v_ema), v in zip(ema_model.state_dict().items(), model.state_dict().values()):
            v_ema.copy_(decay * v_ema + (1.0 - decay) * v)

    # MLflow
    init_mlflow()
    run_name = "runA_resnet34_onecycle_warmup_" + \
        datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "head_lr_mult": HEAD_LR_MULT,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "num_classes": num_classes,
            "backbone": "resnet34_imagenet",
            "meta_dim": meta_dim,
            "balanced_sampler": True,
            "scheduler": "OneCycleLR",
            "onecycle_pct_start": ONECYCLE_PCT_START,
            "ema_decay": EMA_DECAY,
            "label_smoothing": LABEL_SMOOTH,
        })

        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        best_acc, bad_epochs, PATIENCE = -1.0, 0, 10
        ckpt_path = Path("checkpoints/best_runA_resnet34.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for x, m, y in train_loader:
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

                scheduler.step()
                ema_update()

                total_loss += loss.item() * x.size(0)

            train_loss = total_loss / len(train_loader.dataset)

            # Validación con EMA
            val_out = evaluate(ema_model, val_loader,
                               criterion, device, prefix="val")
            print(f"[{epoch:02d}] train_loss={train_loss:.4f} | "
                  f"val_loss={val_out['loss']:.4f} | acc={val_out['acc']:.4f} | top5={val_out['top5']:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_out["loss"],
                "val_acc": val_out["acc"],
                "val_top5": val_out["top5"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            mlflow.log_artifact(val_out["rep_path"])
            mlflow.log_artifact(val_out["cm_path"])

            if val_out["acc"] > best_acc + 1e-4:
                best_acc, bad_epochs = val_out["acc"], 0
                torch.save({
                    "epoch": epoch,
                    "model_state": ema_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "meta_dim": meta_dim,
                }, ckpt_path)
                mlflow.log_artifact(str(ckpt_path))
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print("Early stopping!")
                    break

        # Test final con EMA
        test_out = evaluate(ema_model, test_loader,
                            criterion, device, prefix="test")
        print(
            f"[TEST] loss={test_out['loss']:.4f} | acc={test_out['acc']:.4f} | top5={test_out['top5']:.4f}")
        mlflow.log_metrics({
            "test_loss": test_out["loss"],
            "test_acc": test_out["acc"],
            "test_top5": test_out["top5"],
        })
        mlflow.log_artifact(test_out["rep_path"])
        mlflow.log_artifact(test_out["cm_path"])


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
