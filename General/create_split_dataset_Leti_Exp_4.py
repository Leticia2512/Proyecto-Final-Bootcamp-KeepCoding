# create_split_dataset_Leti_Exp_4.py
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

# Importa el dataset SOLO-IMÁGENES y sus transforms 
from eye_pytorch_dataset_Leti_Exp_4 import EyeDatasetIMG, get_train_transform, get_eval_transform


def _load_targets_df(parquet_or_csv_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carga el fichero de anotaciones (parquet o csv) y devuelve (df, y).
    Se espera 'filename' (str) y 'cod_target' (int 0..K-1).
    """
    if parquet_or_csv_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(parquet_or_csv_path)
    else:
        df = pd.read_csv(parquet_or_csv_path)

    req = {"filename", "cod_target"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {parquet_or_csv_path}: {sorted(missing)}")

    df = df.dropna(subset=["filename"]).copy()
    df["filename"] = df["filename"].astype(str).str.strip()
    y = df["cod_target"].astype(int).to_numpy()
    return df, y


def create_split(
    parquet_file: Path = Path("Data/parquet/dataset_img_multiclass.parquet"),
    image_dir: Path = Path("300x300"),
    img_size: int = 300,
    random_state: int = 42,
):
    """
    Crea splits 80/10/10 estratificados por 'cod_target' y guarda Subsets en .pt.

    - Usa EyeDatasetIMG (solo-imágenes), sin metadatos.
    - Train: augmentations (get_train_transform(img_size)).
    - Val/Test: transform determinista (get_eval_transform(img_size)).

    """
    # ----- Rutas portables -----
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

    if not parquet_file.exists():
        raise FileNotFoundError(f"No existe el fichero de anotaciones: {parquet_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {image_dir}")

    # ----- Carga anotaciones + y -----
    df, y = _load_targets_df(parquet_file)
    idxs = np.arange(len(y))

    # Validación de etiquetas consecutivas 0..K-1
    classes = np.unique(y)
    if (classes != np.arange(classes.max() + 1)).any() or classes.min() != 0:
        raise ValueError(
            "Las etiquetas 'cod_target' deben ser enteros consecutivos (0..K-1). "
            f"Etiquetas detectadas: {classes.tolist()}"
        )

    # ----- Splits 80/10/10 estratificados -----
    tr_idx, tmp_idx = train_test_split(
        idxs, test_size=0.20, random_state=random_state, shuffle=True, stratify=y
    )
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=random_state, shuffle=True, stratify=y[tmp_idx]
    )

    # ----- Datasets (solo-imágenes) con transforms -----
    ds_train = EyeDatasetIMG(
        parquet_path=str(parquet_file),
        image_dir=str(image_dir),
        transform=get_train_transform(img_size),
        img_size=img_size,
    )

    ds_eval = EyeDatasetIMG(
        parquet_path=str(parquet_file),
        image_dir=str(image_dir),
        transform=get_eval_transform(img_size),
        img_size=img_size,
    )

    # ----- Subsets por índices -----
    dataset_train = Subset(ds_train, tr_idx)
    dataset_val   = Subset(ds_eval,  va_idx)
    dataset_test  = Subset(ds_eval,  te_idx)

    # ----- Guardado -----
    torch.save(dataset_train, out_dir / "train_dataset.pt")
    torch.save(dataset_val,   out_dir / "val_dataset.pt")
    torch.save(dataset_test,  out_dir / "test_dataset.pt")

    # Índices (útiles para reproducibilidad y análisis)
    np.save(out_dir / "train_idx.npy", tr_idx)
    np.save(out_dir / "val_idx.npy",   va_idx)
    np.save(out_dir / "test_idx.npy",  te_idx)

    # ----- Logs útiles -----
    print(f"[OK] Guardados en: {out_dir}")
    print(f"  - train_dataset.pt  ({len(tr_idx)} muestras)")
    print(f"  - val_dataset.pt    ({len(va_idx)} muestras)")
    print(f"  - test_dataset.pt   ({len(te_idx)} muestras)")
    print("Imagenes desde:", image_dir)
    print("Anotaciones   :", parquet_file)
    print("img_size      :", img_size)


def main():
    create_split()


if __name__ == "__main__":
    main()
