# create_split_dataset_Leti_Exp_3.py


from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

# Importa el dataset SOLO-IMÁGENES y sus transforms
from Archive.create_dataset_prueba.v2.eye_pytorch_dataset_Exp_3 import EyeDatasetIMG, get_train_transform, get_eval_transform


def _load_targets(parquet_or_csv_path: Path) -> np.ndarray:
    """
    Carga el fichero de anotaciones (parquet o csv) y devuelve el vector y (cod_target).
    Se asume que existe la columna 'cod_target' con enteros 0..K-1.
    """
    if parquet_or_csv_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(parquet_or_csv_path)
    else:
        df = pd.read_csv(parquet_or_csv_path)

    if "cod_target" not in df.columns:
        raise ValueError("No se encuentra la columna 'cod_target' en el fichero de anotaciones.")

    return df["cod_target"].astype(int).to_numpy()


def create_split(
    parquet_file=Path("Data/parquet/dataset_meta_ojouni.parquet"),
    image_dir=Path("224x224"),
    img_size: int = 224,
):
    """
    Crea splits estratificados (80/10/10) y guarda Subsets en .pt.
    - Usa EyeDatasetIMG (solo-imágenes), sin metadatos.
    - Train: augmentations + normalización.
    - Val/Test: preprocesado determinista (sin augmentations).

    Args:
        parquet_file: ruta al .parquet o .csv con columnas ['filename', 'cod_target']
        image_dir: carpeta que contiene las imágenes físicas
        img_size: tamaño al que se redimensionan las imágenes en los transforms
    """
    # ----- Rutas portables relativas al repo -----
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    parquet_file = Path(parquet_file)
    if not parquet_file.is_absolute():
        parquet_file = (repo_root / parquet_file).resolve()

    image_dir = Path(image_dir)
    if not image_dir.is_absolute():
        image_dir = (repo_root / image_dir).resolve()

    out_dir = (repo_root / "Data" / "dataset").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    y = _load_targets(parquet_file)
    idxs = np.arange(len(y))

    # Validación de etiquetas
    if y.min() < 0 or (np.unique(y) != np.arange(y.max() + 1)).any():
        raise ValueError(
            "Las etiquetas 'cod_target' deben ser enteros consecutivos (0..K-1). "
            "Mapéalas antes si es necesario."
        )

    # ----- Splits 80/10/10 estratificados -----
    tr_idx, tmp_idx = train_test_split(
        idxs, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.5, random_state=42, shuffle=True, stratify=y[tmp_idx]
    )

    # ----- Datasets (solo-imágenes) con transforms -----
    # Dataset base para TRAIN con augmentations
    ds_train = EyeDatasetIMG(
        parquet_path=str(parquet_file),
        image_dir=str(image_dir),
        transform=get_train_transform(img_size),
    )

    # Dataset base para EVAL (VAL/TEST) con transform 
    ds_eval = EyeDatasetIMG(
        parquet_path=str(parquet_file),
        image_dir=str(image_dir),
        transform=get_eval_transform(img_size),
    )

    # ----- Subsets por índices -----
    dataset_train = Subset(ds_train, tr_idx)
    dataset_val   = Subset(ds_eval,  va_idx)
    dataset_test  = Subset(ds_eval,  te_idx)

    # ----- Guardado de datasets e índices  -----
    torch.save(dataset_train, out_dir / "train_dataset.pt")
    torch.save(dataset_val,   out_dir / "val_dataset.pt")
    torch.save(dataset_test,  out_dir / "test_dataset.pt")

    # Guardamos también los índices
    np.save(out_dir / "train_idx.npy", tr_idx)
    np.save(out_dir / "val_idx.npy",   va_idx)
    np.save(out_dir / "test_idx.npy",  te_idx)

    print(f"[OK] Guardados en: {out_dir}")
    print(f"  - train_dataset.pt  ({len(tr_idx)} muestras)")
    print(f"  - val_dataset.pt    ({len(va_idx)} muestras)")
    print(f"  - test_dataset.pt   ({len(te_idx)} muestras)")
    print(f"  - train_idx.npy / val_idx.npy / test_idx.npy")


def main():
    create_split()


if __name__ == "__main__":
    main()
