from pathlib import Path
from datetime import datetime
import tempfile
import shutil

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
from load_dataloaders_Sofia import load_dataloaders


# ==== CONFIGURACIÓN ======

BASE_DIR = Path(__file__).resolve().parent.parent

# Subsets preprocesados en .pt
TRAIN_PT = BASE_DIR / "Data" / "dataset" / "train_dataset.pt"
VAL_PT   = BASE_DIR / "Data" / "dataset" / "val_dataset.pt"
TEST_PT  = BASE_DIR / "Data" / "dataset" / "test_dataset.pt"

# Hiperparámetros
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 5e-4
NUM_WORKERS  = 0
SEED         = 42

# Clases que mantenemos y su remapeo a 0..4
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(KEEP_CLASSES)}  # {0:0,1:1,2:2,5:3,6:4}

# Configuración de MLflow
EXPERIMENT_NAME = "Experimento_4_Leticia_ResNet18_Top5"
RUN_NAME        = "resnet18_mm_" + datetime.now().strftime("%Y%m%d_%H%M%S")



# ==== FUNCIONES UTILES ===

def set_seed(seed: int = 42):
    """
    Fija semillas para reproducibilidad (NumPy, PyTorch y CUDA determinista).
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def try_init_mlflow():
    """
    Inicializa MLflow en local guardando en ./mlruns.
    Devuelve el módulo mlflow si está disponible.
    """
    try:
        import mlflow
        tracking_path = (BASE_DIR / "mlruns").resolve()
        mlflow.set_tracking_uri(f"file://{tracking_path}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow no disponible: {e}")
        return None


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Calcula pesos de clase inversos a la frecuencia (normalizados).
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    eps = 1e-6
    w = 1.0 / (counts + eps)
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def plot_confusion(cm: np.ndarray, class_names, out_path: Path):
    """
    Guarda en disco la matriz de confusión.
    """
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusión")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("Clase real")
    plt.xlabel("Clase predicha")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


class FilterMapDataset(Dataset):
    """
    Envuelve un dataset o Subset para:
      1) Filtrar sólo las clases KEEP_CLASSES
      2) Remapear etiquetas originales a [0..num_clases-1] con OLD2NEW
    """
    def __init__(self, base, keep_classes, old2new):
        self.keep = set(keep_classes)
        self.map = old2new

        # Desempaquetar Subset si es necesario
        if isinstance(base, Subset):
            self.base_ds = base.dataset
            source_indices = base.indices
        else:
            self.base_ds = base
            source_indices = range(len(self.base_ds))

        # Filtrar por clases permitidas
        self.indices = [i for i in source_indices
                        if int(self._get_target(i)) in self.keep]

    def _get_target(self, i):
        if hasattr(self.base_ds, "targets"):
            return int(self.base_ds.targets[i])
        _, _, y = self.base_ds[i]
        return int(y)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_img, x_meta, y = self.base_ds[real_idx]
        y_new = torch.as_tensor(self.map[int(y)], dtype=torch.long)
        return x_img, x_meta, y_new


def get_all_labels_from_dataset(ds: Dataset, batch_size: int = 512, num_workers: int = 0) -> np.ndarray:
    """
    Extrae todas las etiquetas de un dataset recorriéndolo en batches.
    """
    ys = []
    tmp_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    for _, _, y in tmp_loader:
        ys.append(y.cpu())
    return torch.cat(ys).numpy()


# ======== MODELO =========

class ImageClassifier(nn.Module):
    """
    Clasificador multimodal:
      - Rama de imagen: ResNet18 preentrenada en ImageNet
      - Rama de metadatos: MLP sencillo
      - Fusión: concatenación + MLP final → logits
    """
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        # ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Rama metadatos
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )

        # Cabeza de fusión
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


# === ENTRENAMIENTO/EVAL ==
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Entrena una época completa.
    Devuelve pérdida media, accuracy y F1-macro.
    """
    model.train()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for imgs, metas, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, metas)
        loss = criterion(logits, ys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        y_true_all.append(ys.detach().cpu())
        y_pred_all.append(preds.detach().cpu())

    epoch_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return epoch_loss, acc, f1m


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, prefix: str, class_names, out_dir: Path):
    """
    Evalúa en validación o test:
      - Calcula métricas globales
      - Guarda classification_report y matriz de confusión
    """
    model.eval()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for imgs, metas, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True, dtype=torch.long)

        logits = model(imgs, metas)
        loss = criterion(logits, ys)
        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(1)
        y_true_all.append(ys.detach().cpu())
        y_pred_all.append(preds.detach().cpu())

    epoch_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    num_classes = len(class_names)
    labels_list = list(range(num_classes))

    acc   = accuracy_score(y_true, y_pred)
    f1m   = f1_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)
    precm = precision_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)
    recm  = recall_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)

    rep_dir = out_dir / f"reports_{prefix}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    rep_path = rep_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(classification_report(
            y_true, y_pred, digits=3,
            labels=labels_list, target_names=class_names,
            zero_division=0
        ))

    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    cm_path = rep_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion(cm, class_names, cm_path)

    return {
        "loss": epoch_loss,
        "acc": acc,
        "f1_macro": f1m,
        "precision_macro": precm,
        "recall_macro": recm,
        "rep_path": str(rep_path),
        "cm_path": str(cm_path),
    }


# ========= MAIN ==========
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    print(f"Dispositivo: {device}")

    # Cargar datasets originales (Subsets .pt)
    train_loader, val_loader, test_loader = load_dataloaders(
        TRAIN_PT, VAL_PT, TEST_PT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        seed=SEED
    )

    # Filtrar y remapear datasets
    train_ds = FilterMapDataset(train_loader.dataset, KEEP_CLASSES, OLD2NEW)
    val_ds   = FilterMapDataset(val_loader.dataset,   KEEP_CLASSES, OLD2NEW)
    test_ds  = FilterMapDataset(test_loader.dataset,  KEEP_CLASSES, OLD2NEW)

    # Crear DataLoaders filtrados
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)

    # Meta-dimensión y nº de clases
    sample = train_ds[0]
    meta_dim = int(sample[1].shape[0])
    num_classes = len(KEEP_CLASSES)
    class_names = [str(c) for c in KEEP_CLASSES]

    print(f"num_classes={num_classes}, meta_dim={meta_dim}")
    print(f"Mapeo old->new: {OLD2NEW}")

    # Modelo
    model = ImageClassifier(
        meta_dim=meta_dim,
        num_classes=num_classes,
        pretrained=True,
        dropout=0.5
    ).to(device)

    # Pesos de clase
    y_train = get_all_labels_from_dataset(train_ds, batch_size=512, num_workers=NUM_WORKERS)
    class_weights = compute_class_weights(y_train, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # MLflow
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
            "backbone": "resnet18_imagenet",
            "label_smoothing": 0.05,
            "keep_classes": KEEP_CLASSES,
            "old2new_map": OLD2NEW
        })
        tmp_art_dir = Path(tempfile.mkdtemp(prefix="mlflow_artifacts_"))
    else:
        tmp_art_dir = Path(tempfile.mkdtemp(prefix="local_artifacts_"))

    # Entrenamiento con early stopping
    best_val_f1 = -1.0
    patience = 6
    bad_epochs = 0
    ckpt_path = tmp_art_dir / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_out = evaluate(model, val_loader, criterion, device,
                           prefix="val", class_names=class_names, out_dir=tmp_art_dir)

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

        if val_out["f1_macro"] > (best_val_f1 + 1e-6):
            best_val_f1 = val_out["f1_macro"]
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            if mlflow is not None:
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                mlflow.log_artifact(val_out["rep_path"], artifact_path="validation")
                mlflow.log_artifact(val_out["cm_path"],  artifact_path="validation")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping!")
                break

    # Evaluación final en test
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.to(device)

    criterion_eval = nn.CrossEntropyLoss(weight=class_weights)
    test_out = evaluate(model, test_loader, criterion_eval, device,
                        prefix="test", class_names=class_names, out_dir=tmp_art_dir)
    print(f"[TEST] loss={test_out['loss']:.4f} acc={test_out['acc']:.4f} f1m={test_out['f1_macro']:.4f}")

    if mlflow is not None:
        mlflow.log_metrics({
            "test_loss":   test_out["loss"],
            "test_acc":    test_out["acc"],
            "test_f1m":    test_out["f1_macro"],
            "test_prec_m": test_out["precision_macro"],
            "test_rec_m":  test_out["recall_macro"],
        })
        mlflow.log_artifact(test_out["rep_path"], artifact_path="test")
        mlflow.log_artifact(test_out["cm_path"],  artifact_path="test")

    shutil.rmtree(tmp_art_dir, ignore_errors=True)
    if mlflow is not None:
        mlflow.end_run()


if __name__ == "__main__":
    main()
