
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torch.nn as nn
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ============= CONFIG (Run E - 384px strong) =============
BATCH_SIZE = 24
EPOCHS = 40          # 3 warmup + 37 entrenamiento OneCycle
WARMUP_FREEZE_E = 3           # epochs con backbone congelado
LR = 1e-4        # LR base (backbone)
HEAD_LR_MULT = 8.0         # LR de meta+head = LR * HEAD_LR_MULT
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4           # en Windows, si da problemas, pon 0
SEED = 42
EXPERIMENT_NAME = "ExperimentoNau_singlelabel_top5"

# Datasets 384px
TRAIN_PT = r"Data\dataset\train_dataset384.pt"
VAL_PT = r"Data\dataset\val_dataset384.pt"
TEST_PT = r"Data\dataset\test_dataset384.pt"

# Clases que mantenemos y mapeo -> 0..4
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(
    KEEP_CLASSES)}  # {0:0,1:1,2:2,5:3,6:4}

# OneCycleLR (fase post-warmup)
ONECYCLE_PCT_START = 0.20
DIV_FACTOR = 10.0
FINAL_DIV_FACTOR = 100.0

# EMA y regularización
EMA_DECAY = 0.9995
LABEL_SMOOTH = 0.05
DROPOUT = 0.6

# CutMix
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0
# =========================================================


# ----------------- Utils -----------------
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
    """Admite (x,m,y) o (x,m,y,id) y devuelve (x,m,y)."""
    if len(item) == 4:
        x, m, y, _ = item
    else:
        x, m, y = item
    return x, m, y


class NormalizeCollate:
    """Windows-safe collate. Normaliza 'edad' (columna 0) del vector meta y remapea labels."""

    def __init__(self, mu: float, sigma: float, age_index: int = 0, old2new=None):
        self.mu = float(mu)
        self.sigma = float(sigma) if sigma != 0 else 1.0
        self.age_index = int(age_index)
        self.old2new = old2new or {}

    def __call__(self, batch):
        xs, ms, ys = [], [], []
        for it in batch:
            x, m, y = unpack_item(it)
            y = int(y.item()) if torch.is_tensor(y) else int(y)
            y = self.old2new.get(y, y)
            xs.append(x)
            ms.append(m)
            ys.append(y)
        x = torch.stack(xs)
        m = torch.stack(ms)
        y = torch.tensor(ys, dtype=torch.long)
        m[:, self.age_index] = (m[:, self.age_index] - self.mu) / self.sigma
        return x, m, y


def get_base_and_indices(maybe_subset):
    """Soporta Subset(...) o {'dataset','indices'} guardado."""
    if isinstance(maybe_subset, dict):
        base = maybe_subset["dataset"]
        indices = list(maybe_subset["indices"])
    else:
        base = maybe_subset.dataset
        indices = list(maybe_subset.indices)
    return base, indices


def filter_indices_by_classes(base_dataset, indices, keep):
    keep_set = set(keep)
    kept = [i for i in indices if int(base_dataset.targets[i]) in keep_set]
    return kept


class RemappedSubset(Dataset):
    """Subconjunto filtrado con remapeo de etiquetas old->new."""

    def __init__(self, base_dataset, indices, old2new):
        self.base = base_dataset
        self.indices = list(indices)
        self.old2new = dict(old2new)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.base[real_idx]
        x, m, y = unpack_item(item)
        y = int(y.item()) if torch.is_tensor(y) else int(y)
        y_new = self.old2new[y]
        return x, m, torch.tensor(y_new, dtype=torch.long)


def plot_confusion(cm, class_names, out_path: Path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ----------------- Modelo -----------------
class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, dropout: float = DROPOUT, pretrained: bool = True):
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
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        f = torch.cat([f_img, f_meta], dim=1)
        return self.head(f)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, prefix="val"):
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
    rep_txt = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    out_dir = Path(f"reports_{prefix}")
    out_dir.mkdir(exist_ok=True)
    rep_path = out_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep_txt)
    cm_path = out_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion(cm, class_names, cm_path)

    return {"loss": avg_loss, "acc": acc, "top5": top5_acc,
            "rep_path": str(rep_path), "cm_path": str(cm_path)}


# ----------------- CutMix -----------------
def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


# ----------------- Train -----------------
def main():
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"

    # Carga objetos .pt
    tr_raw = torch.load(TRAIN_PT, weights_only=False)
    va_raw = torch.load(VAL_PT,   weights_only=False)
    te_raw = torch.load(TEST_PT,  weights_only=False)

    # Base dataset + indices originales
    tr_base, tr_idx = get_base_and_indices(tr_raw)
    va_base, va_idx = get_base_and_indices(va_raw)
    te_base, te_idx = get_base_and_indices(te_raw)

    # Filtrar por clases a conservar
    tr_idx = filter_indices_by_classes(tr_base, tr_idx, KEEP_CLASSES)
    va_idx = filter_indices_by_classes(va_base, va_idx, KEEP_CLASSES)
    te_idx = filter_indices_by_classes(te_base, te_idx, KEEP_CLASSES)

    # Subsets remapeados (0..4)
    train_ds = RemappedSubset(tr_base, tr_idx, OLD2NEW)
    val_ds = RemappedSubset(va_base, va_idx, OLD2NEW)
    test_ds = RemappedSubset(te_base, te_idx, OLD2NEW)

    meta_dim = train_ds[0][1].shape[0]
    num_classes = len(KEEP_CLASSES)
    class_names = [f"class_{NEW} (old {OLD})" for OLD, NEW in sorted(
        OLD2NEW.items(), key=lambda kv: kv[1])]
    print(
        f"[Top5-384 strong] num_classes={num_classes}, meta_dim={meta_dim}, kept={KEEP_CLASSES}")

    # Normalización de edad SOLO con train filtrado
    ages = np.array([float(tr_base.meta[i][0]) for i in tr_idx])
    mu, sigma = float(ages.mean()), float(ages.std() + 1e-8)
    collate = NormalizeCollate(mu, sigma, age_index=0, old2new=OLD2NEW)

    # Pesos por clase (balanceo)
    y_train = np.array([OLD2NEW[int(tr_base.targets[i])]
                       for i in tr_idx], dtype=int)
    counts = np.bincount(y_train, minlength=num_classes)
    class_weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
    class_weights = class_weights / class_weights.mean()
    w_tensor = torch.tensor(class_weights, device=device)

    sample_w = class_weights[y_train]
    sample_w = (sample_w / sample_w.mean()).astype(np.float32)
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_mem,
                            collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_mem,
                             collate_fn=collate)

    # Modelo
    model = ImageClassifier(meta_dim=meta_dim, num_classes=num_classes,
                            dropout=DROPOUT, pretrained=True).to(device)

    # Optimizador con LR discriminativo
    param_groups = [
        {"params": model.backbone.parameters(), "lr": LR},
        {"params": list(model.meta.parameters()) + list(model.head.parameters()),
         "lr": LR * HEAD_LR_MULT},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # Criterio base
    criterion = nn.CrossEntropyLoss(
        weight=w_tensor, label_smoothing=LABEL_SMOOTH)

    # EMA
    ema_model = ImageClassifier(
        meta_dim=meta_dim, num_classes=num_classes, dropout=DROPOUT, pretrained=False).to(device)
    ema_model.load_state_dict(model.state_dict())

    @torch.no_grad()
    def ema_update(decay=EMA_DECAY):
        for (k, v_ema), v in zip(ema_model.state_dict().items(), model.state_dict().values()):
            v_ema.copy_(decay * v_ema + (1.0 - decay) * v)

    # MLflow
    init_mlflow()
    run_name = "runE_resnet34_top5_384_strong_" + \
        datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(run_name=run_name):
        mapping_path = Path("top5_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({"keep": KEEP_CLASSES, "old2new": OLD2NEW}, f, indent=2)

        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "warmup_freeze_epochs": WARMUP_FREEZE_E,
            "lr": LR,
            "head_lr_mult": HEAD_LR_MULT,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "num_classes": num_classes,
            "backbone": "resnet34_imagenet",
            "meta_dim": meta_dim,
            "balanced_sampler": True,
            "scheduler": "WarmupFreeze + OneCycleLR",
            "onecycle_pct_start": ONECYCLE_PCT_START,
            "ema_decay": EMA_DECAY,
            "label_smoothing": LABEL_SMOOTH,
            "dropout": DROPOUT,
            "keep_classes": str(KEEP_CLASSES),
            "input_resolution": 384,
            "cutmix_prob": CUTMIX_PROB,
            "cutmix_alpha": CUTMIX_ALPHA,
        })
        mlflow.log_artifact(str(mapping_path))

        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        best_acc, bad_epochs, PATIENCE = -1.0, 0, 10
        ckpt_path = Path("checkpoints/best_runE_top5_384.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        steps_per_epoch = max(len(train_loader), 1)

        # ---------- Warmup: freeze backbone ----------
        for p in model.backbone.parameters():
            p.requires_grad = False

        for epoch in range(1, WARMUP_FREEZE_E + 1):
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

                ema_update()
                total_loss += loss.item() * x.size(0)

            train_loss = total_loss / len(train_loader.dataset)
            val_out = evaluate(ema_model, val_loader, criterion,
                               device, class_names, prefix="valE_warmup")
            print(
                f"[W{epoch:02d}] train_loss={train_loss:.4f} | val_loss={val_out['loss']:.4f} | acc={val_out['acc']:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_out["loss"],
                "val_acc": val_out["acc"],
                "val_top5": val_out["top5"],
            }, step=epoch)

        # ---------- Unfreeze y OneCycleLR ----------
        for p in model.backbone.parameters():
            p.requires_grad = True

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[LR, LR * HEAD_LR_MULT],
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS - WARMUP_FREEZE_E,
            pct_start=ONECYCLE_PCT_START,
            div_factor=DIV_FACTOR,
            final_div_factor=FINAL_DIV_FACTOR,
        )

        # Entrenamiento con CutMix
        for e in range(WARMUP_FREEZE_E + 1, EPOCHS + 1):
            model.train()
            total_loss = 0.0

            for x, m, y in train_loader:
                x, m, y = x.to(device), m.to(
                    device), y.to(device, dtype=torch.long)

                use_cutmix = (np.random.rand() < CUTMIX_PROB)
                optimizer.zero_grad(set_to_none=True)

                if use_cutmix:
                    lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
                    idx_perm = torch.randperm(x.size(0), device=x.device)
                    x2 = x[idx_perm]
                    W = x.size(3)
                    H = x.size(2)
                    x1a, y1a, x2a, y2a = rand_bbox(W, H, lam)
                    x[:, :, y1a:y2a, x1a:x2a] = x2[:, :, y1a:y2a, x1a:x2a]
                    y_a, y_b = y, y[idx_perm]
                    # ajustar lambda exacto por área
                    lam = 1. - ((x2a - x1a) * (y2a - y1a) / (W * H))

                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        logits = model(x, m)
                        loss = lam * criterion(logits, y_a) + \
                            (1. - lam) * criterion(logits, y_b)
                else:
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
            val_out = evaluate(ema_model, val_loader, criterion,
                               device, class_names, prefix="valE")
            print(f"[{e:02d}] train_loss={train_loss:.4f} | val_loss={val_out['loss']:.4f} | acc={val_out['acc']:.4f} | top5={val_out['top5']:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_out["loss"],
                "val_acc": val_out["acc"],
                "val_top5": val_out["top5"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=e)

            mlflow.log_artifact(val_out["rep_path"])
            mlflow.log_artifact(val_out["cm_path"])

            # Early stopping
            if val_out["acc"] > best_acc + 1e-4:
                best_acc, bad_epochs = val_out["acc"], 0
                torch.save({
                    "epoch": e,
                    "model_state": ema_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "num_classes": num_classes,
                    "meta_dim": meta_dim,
                    "keep_classes": KEEP_CLASSES,
                    "old2new": OLD2NEW,
                    "input_resolution": 384
                }, Path("checkpoints/best_runE_top5_384.pt"))
                mlflow.log_artifact("checkpoints/best_runE_top5_384.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= 10:
                    print("Early stopping!")
                    break

        # Test final con EMA
        test_out = evaluate(ema_model, test_loader, criterion,
                            device, class_names, prefix="testE")
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
