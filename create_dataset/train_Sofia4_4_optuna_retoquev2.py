from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
from load_dataloaders import load_dataloaders  # tu Data Augmentation Plus

# ------- CONFIG -------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PT = BASE_DIR / "Data" / "dataset" / "train_dataset.pt"
VAL_PT = BASE_DIR / "Data" / "dataset" / "val_dataset.pt"
TEST_PT = BASE_DIR / "Data" / "dataset" / "test_dataset.pt"

# Hparams (Optuna best + ajustes de convergencia)
BATCH_SIZE = 16
EPOCHS = 18               # ↑ antes 12
LR = 3.385411072501414e-04
WEIGHT_DECAY = 6.473169608362886e-05
DROPOUT = 0.5936083897821617
LABEL_SMOOTH = 0.09597773956829538
ONECYCLE_PCT_START = 0.2909130491341355
EMA_DECAY = 0.995704246471735

NUM_WORKERS = 4           # en Windows, si da guerra: 0
SEED = 42

# Clases top-5 y remapeo
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(KEEP_CLASSES)}

# MLflow (opcional)
EXPERIMENT_NAME = "ExperimentoNau_singlelabel_top5_best"
RUN_NAME = "efficientnetb0_top5_best_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------ UTILS ------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def try_init_mlflow():
    try:
        import mlflow
        uri = "file:" + str(Path("mlruns").resolve())
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow no disponible: {e}")
        return None


def unpack_item(item):
    if len(item) == 4:
        x, m, y, _ = item
    else:
        x, m, y = item
    return x, m, y


class FilterMapDataset(Dataset):
    """Filtra KEEP_CLASSES y remapea etiquetas con OLD2NEW, preservando transforms del dataset base."""

    def __init__(self, base, keep_classes, old2new):
        self.keep = set(keep_classes)
        self.map = dict(old2new)
        if isinstance(base, Subset):
            self.base_ds = base.dataset
            source_indices = base.indices
        else:
            self.base_ds = base
            source_indices = range(len(self.base_ds))
        self.indices = [i for i in source_indices if self._label_in_keep(i)]

    def _label_in_keep(self, i):
        if hasattr(self.base_ds, "targets"):
            y = int(self.base_ds.targets[i])
        else:
            _, _, y = self.base_ds[i]
            y = int(y) if not torch.is_tensor(y) else int(y.item())
        return y in self.keep

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_img, x_meta, y = self.base_ds[real_idx]
        y_val = int(y) if not torch.is_tensor(y) else int(y.item())
        return x_img, x_meta, torch.tensor(self.map[y_val], dtype=torch.long)


def get_all_labels_from_dataset(ds: Dataset, batch_size: int = 512, num_workers: int = 0) -> np.ndarray:
    ys = []
    tmp_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    for _, _, y in tmp_loader:
        ys.append(y.cpu())
    return torch.cat(ys).numpy()


def plot_confusion(cm: np.ndarray, class_names, out_path: Path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("Real")
    plt.xlabel("Pred")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


class NormalizeMetaCollate:
    """Normaliza la columna 'edad' (índice 0) en metadatos."""

    def __init__(self, mu: float, sigma: float, age_index: int = 0):
        self.mu = float(mu)
        self.sigma = float(sigma) if sigma != 0 else 1.0
        self.age_index = int(age_index)

    def __call__(self, batch):
        xs, ms, ys = [], [], []
        for x, m, y in batch:
            xs.append(x)
            ms.append(m)
            ys.append(y)
        x = torch.stack(xs)
        m = torch.stack(ms)
        y = torch.tensor(ys, dtype=torch.long)
        m[:, self.age_index] = (m[:, self.age_index] - self.mu) / self.sigma
        return x, m, y


# ---------------- MODELO ----------------
class ImageClassifier(nn.Module):
    """EfficientNet-B0 + MLP meta + cabeza fusionada."""

    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_feats = self.backbone.classifier[1].in_features  # (Dropout, Linear)
        self.backbone.classifier = nn.Identity()

        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )

        self.head = nn.Sequential(
            nn.Linear(in_feats + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        return self.head(torch.cat([f_img, f_meta], dim=1))


# -------- BUCLES --------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, scheduler, ema_update):
    model.train()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for imgs, metas, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(imgs, metas)
            loss = criterion(logits, ys)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        ema_update()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.detach().argmax(1)
        y_true_all.append(ys.detach().cpu())
        y_pred_all.append(preds.cpu())

    epoch_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return epoch_loss, acc, f1m


def _enable_mc_dropout(module: nn.Module):
    """Activa dropout también en eval (solo capas Dropout)."""
    for m in module.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d) or isinstance(m, nn.Dropout3d):
            m.train()


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val", class_names=None, out_dir=OUT_DIR, tta=True, mc_passes=5):
    """
    Evalúa el modelo. Si tta=True, usa flip horizontal + MC-Dropout (N pases) y promedia logits.
    """
    model.eval()
    # Activamos dropout en eval para MC sampling (solo capas Dropout)
    if tta and mc_passes > 1:
        _enable_mc_dropout(model)

    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for imgs, metas, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True, dtype=torch.long)

        # multi-pass con MC-dropout
        if tta and mc_passes > 1:
            logits_acc = None
            for _ in range(mc_passes):
                logits = model(imgs, metas)
                logits_flip = model(torch.flip(imgs, dims=[-1]), metas)
                logits_t = 0.5 * (logits + logits_flip)
                logits_acc = logits_t if logits_acc is None else (
                    logits_acc + logits_t)
            logits = logits_acc / float(mc_passes)
        else:
            logits = model(imgs, metas)
            if tta:
                logits_flip = model(torch.flip(imgs, dims=[-1]), metas)
                logits = 0.5 * (logits + logits_flip)

        loss = criterion(logits, ys)
        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(1)
        y_true_all.append(ys.cpu())
        y_pred_all.append(preds.cpu())

    epoch_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recm = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Artefactos
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(int(cm.shape[0]))]

    rep_dir = out_dir / f"reports_{prefix}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, digits=3))

    cm_path = rep_dir / f"{prefix}_confusion_matrix.png"
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("Real")
    plt.xlabel("Pred")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "loss": epoch_loss,
        "acc": acc,
        "f1_macro": f1m,
        "precision_macro": precm,
        "recall_macro": recm,
        "rep_path": str(rep_path),
        "cm_path": str(cm_path),
    }


# -------- MAIN ---------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    print(f"Dispositivo: {device}")

    # Carga dataloaders (tu Augmentation Plus)
    base_train, base_val, base_test = load_dataloaders(
        TRAIN_PT, VAL_PT, TEST_PT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        seed=SEED
    )

    # Filtrar y remapear datasets
    train_ds = FilterMapDataset(base_train.dataset, KEEP_CLASSES, OLD2NEW)
    val_ds = FilterMapDataset(base_val.dataset,   KEEP_CLASSES, OLD2NEW)
    test_ds = FilterMapDataset(base_test.dataset,  KEEP_CLASSES, OLD2NEW)

    # Sampler balanceado (mejor que class weights simultáneos)
    y_train = get_all_labels_from_dataset(
        train_ds, batch_size=512, num_workers=NUM_WORKERS)
    num_classes = len(KEEP_CLASSES)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights_sampler = (counts.sum() / np.maximum(counts, 1.0))
    class_weights_sampler = class_weights_sampler / class_weights_sampler.mean()
    sample_w = class_weights_sampler[y_train]
    sample_w = (sample_w / sample_w.mean()).astype(np.float32)
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True)

    # Normaliza edad (metadato 0) con stats de train filtrado
    def get_age_stats(ds: Dataset, age_index: int = 0):
        vals = []
        tmp = DataLoader(ds, batch_size=512, shuffle=False,
                         num_workers=0, pin_memory=False)
        for _, m, _ in tmp:
            vals.append(m[:, age_index].cpu())
        if not vals:
            return 0.0, 1.0
        v = torch.cat(vals).float().numpy()
        return float(v.mean()), float(v.std() + 1e-8)

    mu_age, sigma_age = get_age_stats(train_ds, age_index=0)
    collate = NormalizeMetaCollate(mu_age, sigma_age, age_index=0)

    # DataLoaders
    common_dl_kwargs = dict(num_workers=NUM_WORKERS,
                            pin_memory=pin_mem, collate_fn=collate)
    if NUM_WORKERS > 0:
        common_dl_kwargs.update(
            dict(persistent_workers=True, prefetch_factor=2))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, shuffle=False, **common_dl_kwargs)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                            shuffle=False, **common_dl_kwargs)
    test_loader = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False, **common_dl_kwargs)

    # Dimensiones
    sample = train_ds[0]
    meta_dim = int(sample[1].shape[0])
    class_names = [str(c) for c in KEEP_CLASSES]
    print(f"num_classes={num_classes}, meta_dim={meta_dim}")

    # Modelo
    model = ImageClassifier(meta_dim=meta_dim, num_classes=num_classes,
                            pretrained=True, dropout=DROPOUT).to(device)

    # Pérdida (sin class weights; ya balanceamos con sampler)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # Optimizador + OneCycleLR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=ONECYCLE_PCT_START,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    # EMA
    ema_model = ImageClassifier(
        meta_dim=meta_dim, num_classes=num_classes, pretrained=False, dropout=DROPOUT).to(device)
    ema_model.load_state_dict(model.state_dict())

    @torch.no_grad()
    def ema_update():
        for (k, v_ema), v in zip(ema_model.state_dict().items(), model.state_dict().values()):
            v_ema.copy_(EMA_DECAY * v_ema + (1.0 - EMA_DECAY) * v)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # MLflow (opcional)
    mlflow = try_init_mlflow()
    if mlflow is not None:
        mlflow.start_run(run_name=RUN_NAME)
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "pin_memory": pin_mem,
            "num_classes": num_classes,
            "meta_dim": meta_dim,
            "optimizer": "AdamW",
            "backbone": "efficientnet_b0_imagenet",
            "label_smoothing": LABEL_SMOOTH,
            "dropout": DROPOUT,
            "onecycle_pct_start": ONECYCLE_PCT_START,
            "ema_decay": EMA_DECAY,
        })
        tmp_art_dir = OUT_DIR / \
            f"mlflow_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tmp_art_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_art_dir = OUT_DIR

    # Early stopping por val_acc (más paciencia)
    best_val_acc = -1.0
    bad_epochs = 0
    patience = 6        # ↑ antes 4

    ckpt_path = tmp_art_dir / "best_model_ema.pt"
    torch.save(ema_model.state_dict(), ckpt_path)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, scheduler, ema_update
        )
        # Validación sobre EMA + TTA (MC-dropout 5 pases)
        val_out = evaluate(ema_model, val_loader, criterion, device,
                           prefix="val", class_names=class_names, out_dir=tmp_art_dir,
                           tta=True, mc_passes=5)

        print(f"[Época {epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} f1m={train_f1m:.4f} | "
              f"val_loss={val_out['loss']:.4f} acc={val_out['acc']:.4f} f1m={val_out['f1_macro']:.4f}")

        if mlflow is not None:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "train_f1m":  train_f1m,
                "val_loss":   val_out["loss"],
                "val_acc":    val_out["acc"],
                "val_f1m":    val_out["f1_macro"],
            }, step=epoch)

        if val_out["acc"] > (best_val_acc + 1e-4):
            best_val_acc = val_out["acc"]
            bad_epochs = 0
            torch.save(ema_model.state_dict(), ckpt_path)
            if mlflow is not None:
                mlflow.log_artifact(str(val_out["rep_path"]))
                mlflow.log_artifact(str(val_out["cm_path"]))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping!")
                break

    # Cargar mejor EMA y evaluar en test (con TTA + MC-dropout)
    if ckpt_path.exists():
        ema_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        ema_model.to(device)

    test_out = evaluate(
        ema_model, test_loader, criterion, device,
        prefix="test", class_names=class_names, out_dir=tmp_art_dir,
        tta=True, mc_passes=5
    )
    print(
        f"[TEST] loss={test_out['loss']:.4f} acc={test_out['acc']:.4f} f1m={test_out['f1_macro']:.4f}")

    if mlflow is not None:
        mlflow.log_metrics({
            "test_loss":   test_out["loss"],
            "test_acc":    test_out["acc"],
            "test_f1m":    test_out["f1_macro"],
            "test_prec_m": test_out["precision_macro"],
            "test_rec_m":  test_out["recall_macro"],
        })
        mlflow.log_artifact(str(test_out["rep_path"]))
        mlflow.log_artifact(str(test_out["cm_path"]))
        mlflow.end_run()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
