"""
Hiperparámetros
BATCH_SIZE = 32, 
EPOCHS  = 15, 
LR = 1e-4, 
WEIGHT_DECAY = 5e-4


patience = 4

Modelo: efficientnet_b0

Target: Todas las Clases

Data Augmentation: Básico

Tiempo: 1.1H
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import tempfile
import shutil
from torch.utils.data import DataLoader, Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
from load_dataloaders_Sofia import load_dataloaders

# ------- CONFIGURACIÓN -------
# Carpeta raíz del proyecto 
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas a los Subsets guardados
TRAIN_PT = BASE_DIR / "Data" / "dataset" / "train_dataset.pt"
VAL_PT   = BASE_DIR / "Data" / "dataset" / "val_dataset.pt"
TEST_PT  = BASE_DIR / "Data" / "dataset" / "test_dataset.pt"

# Hiperparámetros
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-4
WEIGHT_DECAY = 5e-4 # Voy aumentarlo
NUM_WORKERS  = 0    
SEED         = 42

# MLflow
EXPERIMENT_NAME = "DataAvengersSofia"
RUN_NAME        = "efficientnetb0_mm_" + datetime.now().strftime("%Y%m%d_%H%M%S")

# Carpeta de salida para artefactos locales
OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------ FUNCIONES AUXILIARES ------
def set_seed(seed: int = 42):
    """Fija la semilla aleatoria para reproducibilidad."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def try_init_mlflow():
    """Intenta inicializar MLflow, devuelve el módulo o None si no está instalado."""
    try:
        import mlflow
        uri = "file:" + str(Path("mlruns").resolve())
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow no disponible: {e}")
        return None

def load_subset(path: Path):
    """
    Carga un archivo .pt que contiene un Subset de PyTorch (tal y como lo guardamos).
    """
    obj = torch.load(path)
    if isinstance(obj, Subset):
        return obj
    raise TypeError(f"Se esperaba un Subset en {path}, pero se encontró: {type(obj)}")


def get_train_labels(subset: Subset):
    """Extrae las etiquetas y del Subset (sin necesidad de cargar imágenes)."""
    base = subset.dataset   
    idxs = subset.indices
    return np.array([int(base.targets[i]) for i in idxs], dtype=int)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Calcula pesos inversos a la frecuencia de cada clase.
    Esto ayuda a que la pérdida dé más importancia a clases minoritarias.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    eps = 1e-6
    w = 1.0 / (counts + eps)
    w = w * (num_classes / w.sum())  # normalización
    return torch.tensor(w, dtype=torch.float32)


def plot_confusion(cm: np.ndarray, class_names, out_path: Path):
    """Guarda una figura con la matriz de confusión."""
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



# ---------------- MODELO ----------------
class ImageClassifier(nn.Module):
    """
    Clasificador multimodal: 
    - Rama de imágenes → ResNet18 preentrenada en ImageNet.
    - Rama de metadatos → MLP sencillo.
    - Fusión de ambas → capa final de clasificación.
    """
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # El clasificador de EfficientNet es Sequential(Dropout, Linear)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  

        # Rama de metadatos
        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )

        # Fusión y Clasificación
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



# -------- BUCLES DE ENTRENAMIENTO/EVALUACIÓN --------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Entrena una época completa y devuelve pérdida, accuracy y F1-macro."""
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
    f1m = f1_score(y_true, y_pred, average="macro")
    return epoch_loss, acc, f1m


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val", class_names=None, out_dir=OUT_DIR):
    """
    Evalúa el modelo en un conjunto (val o test).
    Guarda el classification report y la matriz de confusión.
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

    acc   = accuracy_score(y_true, y_pred)
    f1m   = f1_score(y_true, y_pred, average="macro")
    precm = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recm  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Guardar report y matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(int(cm.shape[0]))]

    rep_dir = out_dir / f"reports_{prefix}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    rep_path = rep_dir / f"{prefix}_classification_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, digits=3))

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


# -------- MAIN ---------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = device.type == "cuda" 
    print(f"Dispositivo: {device}")

    # Cargar DataLoaders 
    train_loader, val_loader, test_loader = load_dataloaders(
        TRAIN_PT, VAL_PT, TEST_PT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,             
        pin_memory=pin_mem,                
        seed=SEED
    )

    # Inferir dimensiones y clases 
    train_subset = train_loader.dataset
    sample = train_subset[0]
    meta_dim = int(sample[1].shape[0])     
    y_train = get_train_labels(train_subset) 
    num_classes = int(y_train.max() + 1)
    class_names = [str(i) for i in range(num_classes)]
    print(f"num_classes={num_classes}, meta_dim={meta_dim}")

    # Modelo 
    model = ImageClassifier(
        meta_dim=meta_dim,
        num_classes=num_classes,
        pretrained=True,
        dropout=0.5
    ).to(device)

    # Pérdida con class weights 
    class_weights = compute_class_weights(y_train, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Optimizador 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    #MLflow 
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
            "label_smoothing": 0.05,
        })
        tmp_art_dir = Path(tempfile.mkdtemp(prefix="mlflow_artifacts_"))
    else:
        tmp_art_dir = None

    # Entrenamiento con early stopping (por val F1-macro)
    best_val_f1 = -1.0
    patience = 4 # vamos a reducirlo para intentar que el entrenamiento se detenga antes de lsa épocas donde ya no se generaliza bien
    bad_epochs = 0
    ckpt_path = tmp_art_dir / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_out = evaluate(model, val_loader, criterion, device, prefix="val", class_names=class_names, out_dir=tmp_art_dir)

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

        # Guardar mejor modelo
        if val_out["f1_macro"] > (best_val_f1 + 1e-6):
            best_val_f1 = val_out["f1_macro"]
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            if mlflow is not None:
                mlflow.log_artifact(str(ckpt_path))
                mlflow.log_artifact(val_out["rep_path"])
                mlflow.log_artifact(val_out["cm_path"])
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

    # Manejo de tmp_art_dir: si MLflow no estaba activo, usamos OUT_DIR.
    if tmp_art_dir is None:
        tmp_art_dir = OUT_DIR
        
    test_out = evaluate(
        model, test_loader, criterion_eval, device,
        prefix="test", class_names=class_names, out_dir=tmp_art_dir
    )
    print(f"[TEST] loss={test_out['loss']:.4f} acc={test_out['acc']:.4f} f1m={test_out['f1_macro']:.4f}")

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

       # Limpieza: eliminamos la carpeta temporal de artefactos locales
        if 'tmp_art_dir' in locals() and tmp_art_dir is not None:
            import shutil
            shutil.rmtree(tmp_art_dir, ignore_errors=True)

        # Cerrar el run 
        mlflow.end_run()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
