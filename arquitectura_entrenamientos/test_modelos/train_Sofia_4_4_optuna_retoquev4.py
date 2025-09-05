
"""
Entrenamiento EfficientNet-B0 + metadatos (Top-5 clases) con:
- OneCycleLR
- EMA durante el entrenamiento
- SWA opcional con BN recalculado (multi-entrada)
- TTA en valid/test
- Early Stopping por F1-macro
- MLflow opcional

Usa tus Subsets .pt y filtra/remapea las clases [0,1,2,5,6] -> [0..4]
"""

from load_dataloaders import load_dataloaders
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms.functional as TF

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
plt.switch_backend("agg")  # por si se ejecuta sin display

# ----------- TU IMPORT -----------
# Debe devolver dataloaders base (train/val/test) desde tus .pt
# ---------------------------------

warnings.filterwarnings("ignore")


# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PT = BASE_DIR / "Data" / "dataset" / "train_dataset.pt"
VAL_PT = BASE_DIR / "Data" / "dataset" / "val_dataset.pt"
TEST_PT = BASE_DIR / "Data" / "dataset" / "test_dataset.pt"

# Mejores hiperparámetros (Optuna) + ajustes
BATCH_SIZE = 16
EPOCHS = 18                 # dejamos margen; early stopping decidirá
LR = 3.385e-4
WEIGHT_DECAY = 6.47317e-05
DROPOUT = 0.5936
LABEL_SMOOTH = 0.096
ONECYCLE_PCT_START = 0.291
EMA_DECAY = 0.9957

NUM_WORKERS = 0             # en Windows, mejor 0 salvo que uses if __name__ guard
SEED = 42

KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(
    KEEP_CLASSES)}  # {0:0,1:1,2:2,5:3,6:4}

# MLflow opcional
EXPERIMENT_NAME = "ExperimentoNau_singlelabel_top5_best"
RUN_NAME = "effb0_top5_final_" + datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SWA (opcional)
USE_SWA = True
SWA_START_EPOCH = max(1, EPOCHS - 5)  # promediamos últimas ~5 épocas

# TTA
VAL_TTA = 4
TEST_TTA = 4

# Early Stopping
PATIENCE = 8
# ==========================================


# -------------- Utils --------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # para reproducibilidad estricta:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def try_init_mlflow():
    try:
        import mlflow
        uri = "file:" + str((BASE_DIR / "mlruns").resolve())
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow no disponible: {e}")
        return None


def plot_confusion(cm: np.ndarray, class_names, out_path: Path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("Clase real")
    plt.xlabel("Predicción")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---- dataset wrapper: filtra/remapea ----
class FilterMapDataset(Dataset):
    def __init__(self, base, keep_classes, old2new):
        self.keep = set(keep_classes)
        self.map = dict(old2new)

        if isinstance(base, Subset):
            self.base_ds = base.dataset
            source_indices = base.indices
        else:
            self.base_ds = base
            source_indices = range(len(self.base_ds))

        self.indices = [i for i in source_indices if int(
            self._get_target(i)) in self.keep]

    def _get_target(self, i):
        if hasattr(self.base_ds, "targets"):
            return int(self.base_ds.targets[i])
        item = self.base_ds[i]
        # permite (x,m,y) o (x,m,y, id)
        if isinstance(item, (list, tuple)):
            if len(item) >= 3:
                return int(item[2])
        raise RuntimeError(
            "El dataset base debe devolver (img, meta, y [,...]).")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.base_ds[real_idx]
        x_img, x_meta, y = sample[:3]  # ignora sample_id si existiera
        y_new = torch.as_tensor(self.map[int(y)], dtype=torch.long)
        return x_img, x_meta, y_new


# -------------- Modelo --------------
class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_feats = self.backbone.classifier[1].in_features
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
        f = torch.cat([f_img, f_meta], dim=1)
        return self.head(f)


# -------------- EMA helpers --------------
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float = EMA_DECAY):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def clone_model(model: nn.Module) -> nn.Module:
    import copy
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# -------------- BN reset multi-entrada (SWA) --------------
def _is_bn(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))


@torch.no_grad()
def _reset_bn_stats(module: nn.Module):
    if _is_bn(module):
        if hasattr(module, "running_mean") and module.running_mean is not None:
            module.running_mean.zero_()
        if hasattr(module, "running_var") and module.running_var is not None:
            module.running_var.fill_(1)
        if hasattr(module, "num_batches_tracked") and module.num_batches_tracked is not None:
            module.num_batches_tracked.zero_()


@torch.no_grad()
def update_bn_multi_input(data_loader, model: nn.Module, device=None, max_batches=None):
    if device is None:
        device = next(model.parameters()).device
    model.apply(_reset_bn_stats)
    was_training = model.training
    model.train()

    n = 0
    for batch in data_loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Batch no compatible para update_bn_multi_input")
        x_img, x_meta = batch[:2]
        x_img = x_img.to(device, non_blocking=True)
        x_meta = x_meta.to(device, non_blocking=True)
        model(x_img, x_meta)  # forward para acumular stats BN
        n += 1
        if max_batches is not None and n >= max_batches:
            break

    model.train(was_training)


# -------------- Eval con/sin TTA --------------
@torch.no_grad()
def _forward_tta(model, imgs, metas, tta: int):
    if tta <= 1:
        return model(imgs, metas)
    logits_sum = 0
    for k in range(tta):
        x = imgs
        if k % 2 == 1:
            x = torch.flip(x, dims=[-1])        # H-flip
        if k == 2:
            x = torch.flip(x, dims=[-2])        # V-flip
        if k == 3:
            x = TF.rotate(x, 15)                # rot leve
        logits_sum += model(x, metas)
    return logits_sum / tta


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val", class_names=None,
             out_dir=OUT_DIR, tta: int = 1):
    model.eval()
    total_loss = 0.0
    ys_all, preds_all = [], []

    for imgs, metas, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True, dtype=torch.long)

        logits = _forward_tta(model, imgs, metas, tta)
        loss = criterion(logits, ys)
        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(1)
        ys_all.append(ys.cpu())
        preds_all.append(preds.cpu())

    y_true = torch.cat(ys_all).numpy()
    y_pred = torch.cat(preds_all).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recm = recall_score(y_true, y_pred, average="macro", zero_division=0)
    avg_loss = total_loss / len(loader.dataset)

    # artefactos
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(int(cm.shape[0]))]

    rep_dir = out_dir / f"reports_{prefix}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    rep_path = rep_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred,
                target_names=class_names, digits=3, zero_division=0))

    cm_path = rep_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion(cm, class_names, cm_path)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1_macro": f1m,
        "precision_macro": precm,
        "recall_macro": recm,
        "rep_path": str(rep_path),
        "cm_path": str(cm_path),
    }


# -------------- Train one epoch (AMP + OneCycle + EMA) --------------
def train_one_epoch(model, ema_model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0.0
    ys_all, preds_all = [], []

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

        # EMA update
        ema_update(ema_model, model, decay=EMA_DECAY)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        ys_all.append(ys.detach().cpu())
        preds_all.append(preds.detach().cpu())

    epoch_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(ys_all).numpy()
    y_pred = torch.cat(preds_all).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return epoch_loss, acc, f1m


# ================= MAIN =================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = (device.type == "cuda")
    print(f"Dispositivo: {device}")

    # 1) Carga base de dataloaders (desde tus .pt)
    base_train_loader, base_val_loader, base_test_loader = load_dataloaders(
        TRAIN_PT, VAL_PT, TEST_PT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        seed=SEED,
    )

    # 2) Filtrar/remap datasets a Top-5
    train_ds = FilterMapDataset(
        base_train_loader.dataset, KEEP_CLASSES, OLD2NEW)
    val_ds = FilterMapDataset(base_val_loader.dataset,   KEEP_CLASSES, OLD2NEW)
    test_ds = FilterMapDataset(
        base_test_loader.dataset,  KEEP_CLASSES, OLD2NEW)

    # 3) DataLoaders definitivos
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_mem)

    # 4) Dimensiones y nombres de clases
    meta_dim = int(train_ds[0][1].shape[0])
    num_classes = len(KEEP_CLASSES)
    class_names = [str(c) for c in KEEP_CLASSES]
    print(f"num_classes={num_classes}, meta_dim={meta_dim}")
    print(f"Mapeo old->new: {OLD2NEW}")

    # 5) Modelo + EMA
    model = ImageClassifier(meta_dim=meta_dim, num_classes=num_classes,
                            pretrained=True, dropout=DROPOUT).to(device)
    ema_model = clone_model(model)

    # 6) Criterio (sin weights, ya está balanceado por recorte)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # 7) Optimizador + OneCycleLR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        pct_start=ONECYCLE_PCT_START
    )

    # 8) AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 9) MLflow opcional
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
            "use_swa": USE_SWA,
            "val_tta": VAL_TTA,
            "test_tta": TEST_TTA,
            "keep_classes": str(KEEP_CLASSES),
        })
        tmp_art_dir = Path(tempfile.mkdtemp(prefix="mlflow_artifacts_"))
    else:
        tmp_art_dir = OUT_DIR

    # 10) Guardar mapping como artefacto
    mapping_path = tmp_art_dir / "top5_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"keep": KEEP_CLASSES, "old2new": OLD2NEW}, f, indent=2)
    if mlflow is not None:
        mlflow.log_artifact(str(mapping_path))

    # 11) Loop entrenamiento + Early Stopping (val F1-macro) usando EMA para validar
    best_val_f1 = -1.0
    bad_epochs = 0
    ckpt_path = tmp_art_dir / "best_model_ema.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1m = train_one_epoch(
            model, ema_model, train_loader, criterion, optimizer, scheduler, device, scaler
        )

        # valida con EMA + TTA
        val_out = evaluate(ema_model, val_loader, criterion, device,
                           prefix="val", class_names=class_names,
                           out_dir=tmp_art_dir, tta=VAL_TTA)

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
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            mlflow.log_artifact(val_out["rep_path"])
            mlflow.log_artifact(val_out["cm_path"])

        improved = val_out["f1_macro"] > (best_val_f1 + 1e-6)
        if improved:
            best_val_f1 = val_out["f1_macro"]
            bad_epochs = 0
            torch.save(ema_model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping!")
                break

    # 12) Carga mejor EMA para post-procesos
    if ckpt_path.exists():
        ema_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        ema_model.to(device)

    # 13) SWA opcional: promediar últimos pesos + recalcular BN (multi-entrada)
    best_model = ema_model
    best_tag = "EMA"
    best_val_acc = None
    best_val_f1m = None

    if USE_SWA:
        from torch.optim.swa_utils import AveragedModel
        swa_model = clone_model(model)   # estructura
        swa_avg = AveragedModel(best_model)  # inicializar desde EMA ya buena

        # Nota: si quieres promediar varias épocas, lo ideal es acumular
        # durante el loop. Aquí tomamos la EMA final y recalculamos BN.
        # Recalc BN con datos de train (sin shuffle si quieres ser estricto)
        update_bn_multi_input(train_loader, swa_avg,
                              device=device, max_batches=None)

        # Evalúa SWA en val con TTA
        swa_val = evaluate(swa_avg, val_loader, criterion, device,
                           prefix="val_swa", class_names=class_names,
                           out_dir=tmp_art_dir, tta=VAL_TTA)
        print(
            f"[SWA] val_acc={swa_val['acc']:.4f}  val_f1m={swa_val['f1_macro']:.4f}")

        if mlflow is not None:
            mlflow.log_metrics({
                "val_swa_loss": swa_val["loss"],
                "val_swa_acc":  swa_val["acc"],
                "val_swa_f1m":  swa_val["f1_macro"],
            })
            mlflow.log_artifact(swa_val["rep_path"])
            mlflow.log_artifact(swa_val["cm_path"])

        # Compara EMA vs SWA y elige
        ema_val = evaluate(ema_model, val_loader, criterion, device,
                           prefix="val_ema", class_names=class_names,
                           out_dir=tmp_art_dir, tta=VAL_TTA)
        if mlflow is not None:
            mlflow.log_metrics({
                "val_ema_loss": ema_val["loss"],
                "val_ema_acc":  ema_val["acc"],
                "val_ema_f1m":  ema_val["f1_macro"],
            })

        # Prioriza F1; si empate, prioriza acc
        choose_swa = (swa_val["f1_macro"] > ema_val["f1_macro"] + 1e-6) or (
            abs(swa_val["f1_macro"] - ema_val["f1_macro"]) <= 1e-6 and
            swa_val["acc"] > ema_val["acc"])
        if choose_swa:
            best_model = swa_avg
            best_tag = "SWA"
            best_val_acc = swa_val["acc"]
            best_val_f1m = swa_val["f1_macro"]
        else:
            best_model = ema_model
            best_tag = "EMA"
            best_val_acc = ema_val["acc"]
            best_val_f1m = ema_val["f1_macro"]
    else:
        ema_val = evaluate(ema_model, val_loader, criterion, device,
                           prefix="val_ema", class_names=class_names,
                           out_dir=tmp_art_dir, tta=VAL_TTA)
        best_val_acc = ema_val["acc"]
        best_val_f1m = ema_val["f1_macro"]

    print(
        f"[SELECT] Mejor modelo = {best_tag} (val_acc={best_val_acc:.4f}, val_f1m={best_val_f1m:.4f})")

    # 14) Test final (TTA)
    test_out = evaluate(best_model, test_loader, criterion, device,
                        prefix=f"test_{best_tag.lower()}",
                        class_names=class_names, out_dir=tmp_art_dir, tta=TEST_TTA)
    print(f"[TEST {best_tag}] loss={test_out['loss']:.4f} acc={test_out['acc']:.4f} f1m={test_out['f1_macro']:.4f}")

    # 15) Log MLflow + limpieza
    if mlflow is not None:
        mlflow.log_metrics({
            "test_loss":   test_out["loss"],
            "test_acc":    test_out["acc"],
            "test_f1m":    test_out["f1_macro"],
            "test_prec_m": test_out["precision_macro"],
            "test_rec_m":  test_out["recall_macro"],
        })
        mlflow.log_artifact(test_out["rep_path"])
        mlflow.log_artifact(test_out["cm_path"])
        mlflow.end_run()

        # limpia artefactos temporales si no quieres dejarlos locales
        if tmp_art_dir != OUT_DIR and tmp_art_dir.exists():
            shutil.rmtree(tmp_art_dir, ignore_errors=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
