# optuna_tune_top5_efficient.py
# Tuning (SIN MLflow) para EfficientNet-B0 + meta (Top-5 clases 0,1,2,5,6)
# debe devolver (train_loader, val_loader, test_loader)
from load_dataloaders import load_dataloaders
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score

import optuna
from optuna.exceptions import TrialPruned

# ========= RUTAS Y CONFIG =========
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # ajusta si lo tienes en otra carpeta

TRAIN_PT = PROJECT_ROOT / "Data" / "dataset" / "train_dataset.pt"
VAL_PT = PROJECT_ROOT / "Data" / "dataset" / "val_dataset.pt"
TEST_PT = PROJECT_ROOT / "Data" / "dataset" / "test_dataset.pt"

# Mantener 5 clases y remapear a 0..4
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(
    KEEP_CLASSES)}  # {0:0,1:1,2:2,5:3,6:4}

SEED = 42
NUM_WORKERS_DEFAULT = 4  # en Windows usa 0 si da problemas
DIV_FACTOR = 10.0
FINAL_DIV_FACTOR = 100.0
PATIENCE = 4  # early stopping

OUT_DIR = Path("optuna_out_efficient")
OUT_DIR.mkdir(exist_ok=True)

# Importa tus dataloaders con “Augmentation Plus”


# ========= UTILIDADES =========
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


class FilterMapDataset(Dataset):
    """
    Envuelve un dataset/Subset para filtrar KEEP_CLASSES y remapear con OLD2NEW.
    Mantiene las transforms del dataset base.
    """

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
        y_new = self.map[y_val]
        return x_img, x_meta, torch.tensor(y_new, dtype=torch.long)


def get_all_labels(ds: Dataset, num_workers: int = 0) -> np.ndarray:
    ys = []
    tmp = DataLoader(ds, batch_size=512, shuffle=False,
                     num_workers=num_workers, pin_memory=False)
    for _, _, y in tmp:
        ys.append(y.cpu())
    return torch.cat(ys).numpy()


def get_age_stats(ds: Dataset, age_index: int = 0, num_workers: int = 0) -> Tuple[float, float]:
    vals = []
    tmp = DataLoader(ds, batch_size=512, shuffle=False,
                     num_workers=num_workers, pin_memory=False)
    for _, m, _ in tmp:
        vals.append(m[:, age_index].cpu())
    if len(vals) == 0:
        return 0.0, 1.0
    v = torch.cat(vals).float().numpy()
    mu, sigma = float(v.mean()), float(v.std() + 1e-8)
    return mu, sigma


class NormalizeMetaCollate:
    """Collate: normaliza la columna de edad (índice 0) en los metadatos."""

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


# ========= MODELO =========
class ImageClassifier(nn.Module):
    """EfficientNet-B0 + MLP meta + cabeza fusionada."""

    def __init__(self, meta_dim: int, num_classes: int, dropout: float, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        # Linear de EfficientNet
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(inplace=True), nn.BatchNorm1d(32)
        )
        self.head = nn.Sequential(
            nn.Linear(in_feats + 32, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        return self.head(torch.cat([f_img, f_meta], dim=1))


# ========= EVAL =========
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
        all_y.append(y.cpu())
        all_pred.append(logits.argmax(1).cpu())
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_pred).numpy()
    acc = accuracy_score(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc


# ========= DATA BUILD =========
def build_filtered_dataloaders(batch_size: int, num_workers: int, device):
    pin_mem = (device.type == "cuda")
    # Mantiene tu Augmentation Plus
    base_train, base_val, base_test = load_dataloaders(
        TRAIN_PT, VAL_PT, TEST_PT,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_mem, seed=SEED
    )
    # Filtrado + remapeo
    train_ds = FilterMapDataset(base_train.dataset, KEEP_CLASSES, OLD2NEW)
    val_ds = FilterMapDataset(base_val.dataset,   KEEP_CLASSES, OLD2NEW)
    test_ds = FilterMapDataset(base_test.dataset,  KEEP_CLASSES, OLD2NEW)

    # Sampler balanceado
    y_train = get_all_labels(train_ds, num_workers=num_workers)
    num_classes = len(KEEP_CLASSES)
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = (counts.sum() / np.maximum(counts, 1.0))
    class_weights = class_weights / class_weights.mean()
    sample_w = class_weights[y_train]
    sample_w = (sample_w / sample_w.mean()).astype(np.float32)
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True)

    # Normaliza edad desde train filtrado
    mu, sigma = get_age_stats(train_ds, num_workers=num_workers)
    collate = NormalizeMetaCollate(mu, sigma, age_index=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate)
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate)
    meta_dim = int(train_ds[0][1].shape[0])
    return train_loader, val_loader, test_loader, meta_dim


# ========= OBJETIVO OPTUNA =========
def objective(trial: optuna.Trial, num_workers: int, device: torch.device) -> float:
    # --- Espacio de búsqueda (centrado en tu setup) ---
    # alrededor de 1e-4
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-5, 1e-3, log=True)  # around 5e-4
    dropout = trial.suggest_float("dropout", 0.30, 0.70)
    label_smooth = trial.suggest_float("label_smoothing", 0.00, 0.10)
    epochs = trial.suggest_int("epochs", 12, 18)  # alrededor de 15
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # OneCycle y EMA (como en tu “super-optimizado” anterior)
    onecycle_pct = trial.suggest_float("onecycle_pct_start", 0.05, 0.30)
    ema_decay = trial.suggest_float("ema_decay", 0.995, 0.9999)

    # --- Data ---
    train_loader, val_loader, test_loader, meta_dim = build_filtered_dataloaders(
        batch_size=batch_size, num_workers=num_workers, device=device
    )
    num_classes = len(KEEP_CLASSES)

    # --- Modelo / Optimizador / Scheduler / Loss ---
    model = ImageClassifier(meta_dim=meta_dim, num_classes=num_classes,
                            dropout=dropout, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=onecycle_pct,
        div_factor=DIV_FACTOR,
        final_div_factor=FINAL_DIV_FACTOR,
    )
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

    # --- Entrenamiento + EarlyStopping + Pruning ---
    best_val_acc = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(
                device), y.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x, m)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_update()

        # Validación con EMA
        val_loss, val_acc = evaluate(ema_model, val_loader, criterion, device)

        # Report a Optuna + posibilidad de prune
        trial.report(val_acc, step=epoch)
        if trial.should_prune():  # corta trials malos
            raise TrialPruned()

        # Early stopping manual (paciencia = 4)
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    # (Opcional) medimos test con los mismos pesos EMA
    test_loss, test_acc = evaluate(ema_model, test_loader, criterion, device)
    trial.set_user_attr("test_acc", float(test_acc))
    trial.set_user_attr("val_acc_best", float(best_val_acc))

    return best_val_acc  # objetivo: maximizar val_acc


# ========= GUARDADO DE RESULTADOS =========
def save_study_results(study: optuna.Study, out_dir: Path):
    stamp = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp.mkdir(parents=True, exist_ok=True)
    # Mejor trial
    (stamp / "best_params.json").write_text(json.dumps(study.best_params,
                                                       indent=2), encoding="utf-8")
    (stamp / "best_value.txt").write_text(str(study.best_value), encoding="utf-8")
    # Todos los trials a CSV (sin pandas)
    with (stamp / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["number", "state", "value", "test_acc", "val_acc_best"] + sorted(
            {k for t in study.trials for k in t.params.keys()}
        )
        writer.writerow(header)
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
    print(f"Resultados guardados en: {stamp.resolve()}")


# ========= MAIN =========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None,
                        help="segundos (opcional)")
    parser.add_argument("--study-name", type=str,
                        default="top5_efficientb0_tuning")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_top5_efficient.db",
                        help="p.ej. sqlite:///optuna_top5_efficient.db (persistente)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Estudio persistente y reanudable
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

    save_study_results(study, OUT_DIR)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
