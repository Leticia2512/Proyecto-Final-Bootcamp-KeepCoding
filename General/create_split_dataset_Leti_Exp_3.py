from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from eye_pytorch_dataset_Leti_Exp_3 import EyeDataset, get_train_transform, get_eval_transform
from torch.utils.data import Subset


def create_split(
    parquet_file=Path("Data/parquet/dataset_meta_ojouni_quality.parquet"),  
    image_dir=Path("224x224"),
    feature_cols=("Patient Age", "Patient_Sex_Binario"),
    seed: int = 42,
):
    """
    Crea splits estratificados (80/10/10) por clase y AGRUPADOS por paciente.
    Guarda Subsets en .pt (compatibles con tu pipeline actual).
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

    # Leer parquet y construir patient_id desde filename
    df = pd.read_parquet(parquet_file).reset_index(drop=True)
    df["filename"] = df["filename"].astype(str)
    # Extrae el prefijo numérico antes del guion bajo (e.g., "123_right.jpg" -> "123")
    pid = df["filename"].str.extract(r"^(\d+)_")[0]
    if pid.isna().any():
        # Fallback simple: quita sufijos típicos; ajusta a tu convención si difiere
        pid = df["filename"].str.replace(r"_left\.jpg|_right\.jpg", "", regex=True)
    df["patient_id"] = pid.astype(str)

    y = df["cod_target"].astype(int).to_numpy()
    groups = df["patient_id"].to_numpy()

    # Split 80% train / 20% tmp (val+test), estratificado y agrupado por paciente
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_idx, tmp_idx = next(skf.split(df, y=y, groups=groups))

    # Split 10% val / 10% test dentro de tmp, estratificado y agrupado
    df_tmp = df.iloc[tmp_idx].reset_index(drop=True)
    y_tmp  = df_tmp["cod_target"].to_numpy()
    g_tmp  = df_tmp["patient_id"].to_numpy()

    skf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    va_part, te_part = next(skf2.split(df_tmp, y=y_tmp, groups=g_tmp))
    va_idx = tmp_idx[va_part]
    te_idx = tmp_idx[te_part]

    # Info de tamaños
    n = len(df)
    print(f"[Split] train: {len(tr_idx)} ({len(tr_idx)/n:.1%}) | "
          f"val: {len(va_idx)} ({len(va_idx)/n:.1%}) | "
          f"test: {len(te_idx)} ({len(te_idx)/n:.1%})")

    # Datasets con transforms 
    ds_train = EyeDataset(
        parquet_path=parquet_file,
        image_dir=image_dir,
        feature_cols=feature_cols,
        transform=get_train_transform(224),
        num_classes=8 
    )
    ds_eval = EyeDataset(
        parquet_path=parquet_file,
        image_dir=image_dir,
        feature_cols=feature_cols,
        transform=get_eval_transform(224),
        num_classes=8
    )

    # Subsets
    dataset_train = Subset(ds_train, tr_idx)
    dataset_val   = Subset(ds_eval,  va_idx)
    dataset_test  = Subset(ds_eval,  te_idx)

    # Guardado
    torch.save(dataset_train, out_dir / "train_dataset.pt")
    torch.save(dataset_val,   out_dir / "val_dataset.pt")
    torch.save(dataset_test,  out_dir / "test_dataset.pt")

    print("[OK] Guardados Subset en Data/dataset/: train_dataset.pt, val_dataset.pt, test_dataset.pt")


def main():
    create_split()


if __name__ == "__main__":
    main()
