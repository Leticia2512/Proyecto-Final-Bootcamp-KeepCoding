# train_4_Leticia.py

from pathlib import Path
from datetime import datetime
import tempfile
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

# ResNet18 y pesos (compatibles con distintas versiones de torchvision)
from torchvision.models import resnet18
try:
    from torchvision.models import ResNet18_Weights
    _HAS_WEIGHTS = True
except Exception:
    _HAS_WEIGHTS = False

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


from load_dataloaders_Leti_Exp_3 import load_dataloaders


# ------------------ CONFIGURACIÓN ------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de los Subsets preprocesados (.pt) generados por create_split_dataset_Leti_IMG.py
TRAIN_PT = BASE_DIR / "Data" / "dataset" / "train_dataset.pt"
VAL_PT   = BASE_DIR / "Data" / "dataset" / "val_dataset.pt"
TEST_PT  = BASE_DIR / "Data" / "dataset" / "test_dataset.pt"

# Hiperparámetros
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 5e-4
NUM_WORKERS  = 0
SEED         = 42
IMG_SIZE     = 224  

# Configuración de MLflow
EXPERIMENT_NAME = "Experimento_3_Leticia_ResNet18"
RUN_NAME        = "resnet18_imgonly_" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ------------------ FUNCIONES AUXILIARES ------------------
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
    Inicializa MLflow en modo local (./mlruns).
    Devuelve el módulo mlflow si está disponible, o None en caso de error.
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


def load_subset(path: Path):
    """
    Carga un archivo .pt que contiene un Subset de PyTorch.
    (No se usa directamente aquí, pero se mantiene por si necesitas utilidades extra.)
    """
    obj = torch.load(path, weights_only=False)
    if isinstance(obj, Subset):
        return obj
    raise TypeError(f"Se esperaba un Subset en {path}, pero se encontró: {type(obj)}")


def get_train_labels(subset: Subset) -> np.ndarray:
    """
    Extrae las etiquetas del Subset sin cargar imágenes.
    Requiere que el Dataset base tenga el atributo `targets` (EyeDatasetIMG lo tiene).
    """
    base = subset.dataset
    idxs = subset.indices
    return np.array([int(base.targets[i]) for i in idxs], dtype=int)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Pesos de clase inversos a la frecuencia, normalizados.
    Útiles para clases desbalanceadas con CrossEntropyLoss.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    eps = 1e-6
    w = 1.0 / (counts + eps)
    w = w * (num_classes / w.sum()) 
    return torch.tensor(w, dtype=torch.float32)


def plot_confusion(cm: np.ndarray, class_names, out_path: Path):
    """
    Guarda en disco una figura de la matriz de confusión.
    """
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


# ------------------ MODELO (solo imágenes) ------------------
class ImageOnlyClassifier(nn.Module):
    """
    Clasificador de imágenes basado en ResNet18 preentrenada (sin metadatos).
    Reemplaza la capa final para ajustar al nº de clases.
    """
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        if pretrained and _HAS_WEIGHTS:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Backbone ResNet18
        self.backbone = resnet18(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Clasificación
        self.head = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img):
        f_img = self.backbone(x_img)      
        logits = self.head(f_img)         
        return logits


# ------------------ BUCLES DE ENTRENAMIENTO/EVAL ------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Entrena una época completa y devuelve:
    - pérdida media
    - accuracy
    - F1-macro
    """
    model.train()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for imgs, ys in loader:  # <<< SOLO (imgs, ys)
        imgs = imgs.to(device, non_blocking=True)
        ys   = ys.to(device, non_blocking=True, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
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
    f1m = f1_score(y_true, y_pred, average="macro")
    return epoch_loss, acc, f1m


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, prefix: str, class_names, out_dir: Path):
    """
    Evalúa el modelo en validación o test:
    - Calcula loss, acc, F1, precision y recall macro.
    - Guarda classification report y matriz de confusión en disco.
    """
    model.eval()
    total_loss = 0.0
    y_true_all, y_pred_all = [], []

    for imgs, ys in loader: 
        imgs = imgs.to(device, non_blocking=True)
        ys   = ys.to(device, non_blocking=True, dtype=torch.long)

        logits = model(imgs)
        loss = criterion(logits, ys)
        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(1)
        y_true_all.append(ys.detach().cpu())
        y_pred_all.append(preds.detach().cpu())

    # Métricas globales
    epoch_loss = total_loss / len(loader.dataset)
    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    acc   = accuracy_score(y_true, y_pred)
    f1m   = f1_score(y_true, y_pred, average="macro")
    precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recm  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Guardar report y matriz de confusión
    rep_dir = out_dir / f"reports_{prefix}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    rep_path = rep_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(int(cm.shape[0]))]
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


# ------------------ MAIN ------------------
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda"
    print(f"Dispositivo: {device}")

    # Cargar DataLoaders (con transforms para train/val/test)
    train_loader, val_loader, test_loader = load_dataloaders(
        TRAIN_PT, VAL_PT, TEST_PT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        seed=SEED,
        img_size=IMG_SIZE,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Inferir nº de clases a partir de las etiquetas de train
    train_subset = train_loader.dataset  
    y_train = get_train_labels(train_subset)
    num_classes = int(y_train.max() + 1)
    class_names = [str(i) for i in range(num_classes)]
    print(f"num_classes={num_classes}")

    # Definir modelo (solo imágenes)
    model = ImageOnlyClassifier(
        num_classes=num_classes,
        pretrained=True,
        dropout=0.5
    ).to(device)

    # Pérdida con class weights + label smoothing
    class_weights = compute_class_weights(y_train, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Optimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Inicializar MLflow (si está disponible)
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
            "optimizer": "AdamW",
            "backbone": "resnet18_imagenet",
            "label_smoothing": 0.05,
            "img_size": IMG_SIZE,
            "dataset_splits": "pt_subsets_80_10_10",
        })
        tmp_art_dir = Path(tempfile.mkdtemp(prefix="mlflow_artifacts_"))
    else:
        tmp_art_dir = Path(tempfile.mkdtemp(prefix="local_artifacts_"))

    # Entrenamiento con early stopping (según F1-macro en validación)
    best_val_f1 = -1.0
    patience = 6
    bad_epochs = 0
    ckpt_path = tmp_art_dir / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path) 

    for epoch in range(1, EPOCHS + 1):
        # ---- Entrenamiento ----
        train_loss, train_acc, train_f1m = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # ---- Validación ----
        val_out = evaluate(
            model, val_loader, criterion, device,
            prefix="val", class_names=class_names, out_dir=tmp_art_dir
        )

        print(
            f"[Época {epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} f1m={train_f1m:.4f} | "
            f"val_loss={val_out['loss']:.4f} acc={val_out['acc']:.4f} f1m={val_out['f1_macro']:.4f}"
        )

        # Log de métricas en MLflow
        if mlflow is not None:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "train_f1m":  train_f1m,
                "val_loss":   val_out["loss"],
                "val_acc":    val_out["acc"],
                "val_f1m":    val_out["f1_macro"],
            }, step=epoch)

        # ---- Early Stopping + guardado del mejor modelo ----
        if val_out["f1_macro"] > (best_val_f1 + 1e-6):
            best_val_f1 = val_out["f1_macro"]
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)

            # Subir artefactos a MLflow
            if mlflow is not None:
                mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                mlflow.log_artifact(val_out["rep_path"], artifact_path="validation")
                mlflow.log_artifact(val_out["cm_path"],  artifact_path="validation")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping!")
                break

    # ---- Evaluación final en test ----
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.to(device)

    criterion_eval = nn.CrossEntropyLoss(weight=class_weights)
    test_out = evaluate(
        model, test_loader, criterion_eval, device,
        prefix="test", class_names=class_names, out_dir=tmp_art_dir
    )
    print(f"[TEST] loss={test_out['loss']:.4f} acc={test_out['acc']:.4f} f1m={test_out['f1_macro']:.4f}")

    # Log de resultados de test en MLflow
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
        mlflow.end_run()

    # Limpieza: eliminar la carpeta temporal de artefactos
    shutil.rmtree(tmp_art_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
