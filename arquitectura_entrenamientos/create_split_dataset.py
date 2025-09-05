from pathlib import Path
from sklearn.model_selection import train_test_split
from eye_pytorch_dataset import EyeDataset, get_train_transform, get_eval_transform
import numpy as np
import pandas as pd
from torch.utils.data import Subset
import torch


def create_split(
        parquet_file=Path("Data/parquet/dataset_meta_ojouni.parquet"),
        image_dir=Path("224x224"),
        feature_cols=["Patient Age", "Patient_Sex_Binario"]):
    """
    Crea splits estratificados (80/10/10) y guarda Subsets en .pt.
    """
    # Rutas portables
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

    # 1) y e índices desde parquet
    df = pd.read_parquet(parquet_file)
    y = df["cod_target"].astype(int).to_numpy()
    idxs = np.arange(len(y))

    # 2) splits 80/10/10 estratificados
    tr_idx, tmp_idx = train_test_split(
        idxs, test_size=0.2, random_state=42, shuffle=True, stratify=y)
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.5, random_state=42, shuffle=True, stratify=y[tmp_idx])

    # 3) datasets con transforms separadas
    ds_train = EyeDataset(parquet_path=parquet_file, image_dir=image_dir, feature_cols=feature_cols,
                          transform=get_train_transform(224), num_classes=8)

    ds_eval = EyeDataset(parquet_path=parquet_file, image_dir=image_dir, feature_cols=feature_cols,
                         transform=get_eval_transform(224),  num_classes=8)

    # Subsets
    dataset_train = Subset(ds_train, tr_idx)
    dataset_val = Subset(ds_eval,  va_idx)
    dataset_test = Subset(ds_eval,  te_idx)

    # Guardado
    torch.save(dataset_train, out_dir / "train_dataset.pt")
    torch.save(dataset_val,   out_dir / "val_dataset.pt")
    torch.save(dataset_test,  out_dir / "test_dataset.pt")


def main():
    create_split()


if __name__ == "__main__":
    main()
