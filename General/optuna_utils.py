# optuna_tune_top5.py
# Tuning de hiperparámetros (SIN MLflow) para el modelo Top-5 (clases 0,1,2,5,6)

import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import accuracy_score

import optuna
from optuna.exceptions import TrialPruned

# =============== CONFIG BÁSICA ===============
TRAIN_PT = r"Data\dataset\train_dataset.pt"
VAL_PT = r"Data\dataset\val_dataset.pt"
TEST_PT = r"Data\dataset\test_dataset.pt"

# Mantenemos 5 clases y remapeamos a 0..4
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(
    KEEP_CLASSES)}  # {0:0,1:1,2:2,5:3,6:4}

SEED = 42
NUM_WORKERS_DEFAULT = 4   # en Windows pon 0 si te da problemas

# OneCycle factores base
DIV_FACTOR = 10.0
FINAL_DIV_FACTOR = 100.0

OUT_DIR = Path("optuna_out")
OUT_DIR.mkdir(exist_ok=True)
# ============================================


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_item(item):
    # Soporta (x,m,y) o (x,m,y,id)
    if len(item) == 4:
        x, m, y, _ = item
    else:
        x, m, y = item
    return x, m, y


class NormalizeCollate:
    """Collate seguro para Windows. Normaliza 'edad' (col 0 de meta) y remapea labels."""

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
    # Acepta Subset(...) o dict {"dataset":..., "indices":...}
    if isinstance(maybe_subset, dict):
        base = maybe_subset["dataset"]
        indices = list(maybe_subset["indices"])
    else:
        base = maybe_subset.dataset
        indices = list(maybe_subset.indices)
    return base, indices


def filter_indices_by_classes(base_dataset, indices, keep):
    keep_set = set(keep)
    return [i for i in indices if int(base_dataset.targets[i]) in keep_set]


class RemappedSubset(Dataset):
    def __init__(self, base_dataset, indices, old2new):
        self.base = base_dataset
        self.indices = list(indices)
        self.old2new = dict(old2new)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.base[real_idx]
        x, m, y = unpack_item(item)
        y = int(y.item()) if torch.is_tensor(y) else int(y)
        y_new = self.old2new[y]
        return x, m, torch.tensor(y_new, dtype=torch.long)


class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, dropout: float, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet34(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(inplace=True), nn.BatchNorm1d(32)
        )
        self.head = nn.Sequential(
            nn.Linear(in_feats + 32, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, num_classes)
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        return self.head(torch.cat([f_img, f_meta], dim=1))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y, all_pred = [], []
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device, dtype=torch.long)
        logits = model(x, m)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        all_y.append(y.cpu())
        all_pred.append(pred.cpu())
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_pred).numpy()
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc


def build_dataloaders(batch_size: int, num_workers: int, device):
    # Carga .pt
    tr_raw = torch.load(TRAIN_PT, weights_only=False)
    va_raw = torch.load(VAL_PT,   weights_only=False)
    te_raw = torch.load(TEST_PT,  weights_only=False)

    tr_base, tr_idx = get_base_and_indices(tr_raw)
    va_base, va_idx = get_base_and_indices(va_raw)
    te_base, te_idx = get_base_and_indices(te_raw)

    # Filtrar por las 5 clases
    tr_idx = filter_indices_by_classes(tr_base, tr_idx, KEEP_CLASSES)
    va_idx = filter_indices_by_classes(va_base, va_idx, KEEP_CLASSES)
    te_idx = filter_indices_by_classes(te_base, te_idx, KEEP_CLASSES)

    train_ds = RemappedSubset(tr_base, tr_idx, OLD2NEW)
    val_ds = RemappedSubset(va_base, va_idx, OLD2NEW)
    test_ds = RemappedSubset(te_base, te_idx, OLD2NEW)

    meta_dim = train_ds[0][1].shape[0]
    num_classes = len(KEEP_CLASSES)

    # Normalización de edad (solo train)
    ages = np.array([float(tr_base.meta[i][0]) for i in tr_idx])
    mu, sigma = float(ages.mean()), float(ages.std() + 1e-8)
    collate = NormalizeCollate(mu, sigma, age_index=0, old2new=OLD2NEW)

    # Sampler balanceado
    y_train = np.array([OLD2NEW[int(tr_base.targets[i])]
                       for i in tr_idx], dtype=int)
    counts = np.bincount(y_train, minlength=num_classes)
    class_weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
    class_weights = class_weights / class_weights.mean()
    sample_w = class_weights[y_train]
    sample_w = (sample_w / sample_w.mean()).astype(np.float32)
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate)

    return train_loader, val_loader, test_loader, meta_dim, num_classes


def objective(trial: optuna.Trial, num_workers: int, device: torch.device) -> float:
    # ------- ESPACIO DE BÚSQUEDA -------
    lr = trial.suggest_float("lr", 3e-5, 3e-4, log=True)
    head_lr_mult = trial.suggest_float("head_lr_mult", 4.0, 10.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.30, 0.70)
    label_smooth = trial.suggest_float("label_smoothing", 0.00, 0.10)
    onecycle_pct = trial.suggest_float("onecycle_pct_start", 0.05, 0.30)
    ema_decay = trial.suggest_float("ema_decay", 0.995, 0.9999)
    epochs = trial.suggest_int("epochs", 22, 38)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # ------- DATA -------
    train_loader, val_loader, test_loader, meta_dim, num_classes = build_dataloaders(
        batch_size=batch_size, num_workers=num_workers, device=device
    )

    # ------- MODELO/OPT -------
    model = ImageClassifier(meta_dim=meta_dim, num_classes=num_classes,
                            dropout=dropout, pretrained=True).to(device)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": lr},
        {"params": list(model.meta.parameters()) +
         list(model.head.parameters()), "lr": lr * head_lr_mult},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr, lr * head_lr_mult],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=onecycle_pct,
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
    )

    # Ojo: usamos sampler balanceado; no pasamos class_weight a la loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # EMA
    ema_model = ImageClassifier(
        meta_dim=meta_dim, num_classes=num_classes, dropout=dropout, pretrained=False).to(device)
    ema_model.load_state_dict(model.state_dict())

    @torch.no_grad()
    def ema_update():
        for (k, v_ema), v in zip(ema_model.state_dict().items(), model.state_dict().values()):
            v_ema.copy_(ema_decay * v_ema + (1.0 - ema_decay) * v)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ------- Entrenamiento + Pruning -------
    best_val_acc = -1.0
    bad_epochs = 0
    PATIENCE = 8

    for epoch in range(1, epochs + 1):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_update()

            total_loss += loss.item() * x.size(0)

        # Eval con EMA
        val_loss, val_acc = evaluate(ema_model, val_loader, criterion, device)

        # Report a Optuna + posibilidad de prune
        trial.report(val_acc, step=epoch)
        if trial.should_prune():
            raise TrialPruned()

        # Early stopping “manual”
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    # (Opcional) test para ver rendimiento con esos hparams
    test_loss, test_acc = evaluate(ema_model, test_loader, criterion, device)
    # Guardamos algo rápido por trial (se borra si no quieres)
    trial.set_user_attr("test_acc", float(test_acc))
    trial.set_user_attr("val_acc_best", float(best_val_acc))

    return best_val_acc  # objetivo = maximizar val_acc


def save_study_results(study: optuna.Study, out_dir: Path):
    # 1) Mejor trial
    (out_dir / "best_params.json").write_text(
        json.dumps(study.best_params, indent=2), encoding="utf-8"
    )
    (out_dir / "best_value.txt").write_text(str(study.best_value), encoding="utf-8")

    # 2) Todos los trials a CSV (sin pandas)
    with (out_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Cabecera
        header = ["number", "state", "value", "test_acc", "val_acc_best"] + sorted(
            {k for t in study.trials for k in t.params.keys()}
        )
        writer.writerow(header)
        # Filas
        for t in study.trials:
            row = [
                t.number,
                str(t.state),
                t.value if t.value is not None else "",
                t.user_attrs.get("test_acc", ""),
                t.user_attrs.get("val_acc_best", "")
            ]
            for k in header[5:]:
                row.append(t.params.get(k, ""))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None,
                        help="segundos (opcional)")
    parser.add_argument("--study-name", type=str,
                        default="top5_resnet34_tuning")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_top5.db",
                        help="p.ej. sqlite:///optuna_top5.db (persistente)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Estudio Optuna persistente (puedes reanudarlo)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    def _objective(trial):
        return objective(trial, num_workers=args.num_workers, device=device)

    study.optimize(_objective, n_trials=args.trials,
                   timeout=args.timeout, gc_after_trial=True)

    print("\n=== MEJOR TRIAL ===")
    print("Best value (val_acc):", study.best_value)
    print("Best params:", study.best_params)

    # Guardar resultados
    stamp_dir = OUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp_dir.mkdir(parents=True, exist_ok=True)
    save_study_results(study, stamp_dir)
    print(f"Resultados guardados en: {stamp_dir.resolve()}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
