from sklearn.model_selection import train_test_split
from eye_pytorch_dataset import EyeDataset
import numpy as np


def create_split(
        parquet_file=r"Data\parquet\dataset_meta_ojouni.parquet",
        image_dir=r"224x224",
        feature_cols=["Patient Age", "Patient_Sex_Binario", "ID"]):

    ds = EyeDataset(parquet_path=parquet_file,
                    image_dir=image_dir, feature_cols=feature_cols)

    # --- 1) índices y etiquetas (single-label) ---
    idxs = np.arange(len(ds))
    y = np.array(ds.targets)            # vector con cod_target (0..K-1)

    # --- 2) split train (80%) vs temp (20%) estratificado ---
    tr_idx, tmp_idx = train_test_split(
        idxs,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    # --- 3) split temp en val (10%) vs test (10%) estratificado ---
    va_idx, te_idx = train_test_split(
        tmp_idx,
        test_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=y[tmp_idx]
    )

    return tr_idx, va_idx, te_idx, ds
