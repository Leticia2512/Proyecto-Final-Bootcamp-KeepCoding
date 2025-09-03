from sklearn.model_selection import train_test_split
from eye_pytorch_dataset import EyeDataset
import numpy as np
from torchvision import transforms
from torch.utils.data import Subset
import torch


def create_split(
        parquet_file=r"Data\parquet\dataset_meta_ojouni.parquet",
        image_dir=r"224x224",
        feature_cols=["Patient Age", "Patient_Sex_Binario"]):

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

    train_imgs_transforms = transforms.Compose([
                                                transforms.RandomHorizontalFlip(p= 0.2),
                                                transforms.RandomRotation(15)
                                                ])

    dataset_train = Subset(ds, tr_idx)
    dataset_train.transform = train_imgs_transforms

    dataset_val = Subset(ds, va_idx)
    dataset_test = Subset(ds, te_idx)


    torch.save({
    "dataset": ds,          # tu EyeDataset
    "indices": dataset_train    # lista de índices
}, r"Data\dataset\train_dataset.pt")

    torch.save({
        "dataset": ds,          # tu EyeDataset
        "indices": dataset_val    # lista de índices
    }, r"Data\dataset\val_dataset.pt")

    torch.save({
        "dataset": ds,          # tu EyeDataset
        "indices": dataset_test    # lista de índices
    }, r"Data\dataset\test_dataset.pt")


def main():

    create_split()


if __name__ == "__main__":
    
    main()
