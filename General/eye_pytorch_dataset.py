# eye_pytorch_dataset.py  (multi-label only)
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class EyeDataset(Dataset):
    """
    Dataset por ojo multi-label.
    parquet esperado con columnas:
      - image_name : str
      - idx_list   : list[int]
      - feature_cols (ej.: 'Patient Age', 'Patient_Sex_Binario')
    """

    def __init__(self, parquet_path: str, image_dir: str, feature_cols, transform=None, num_classes=None):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
        input_dir = os.path.join(parent_dir, parquet_path)

        self.df = pd.read_parquet(input_dir).reset_index(drop=True)
        self.image_dir = image_dir
        self.feature_cols = list(feature_cols)

        # cache ligera
        self.paths = self.df["filename"].astype(str).tolist()
        self.meta = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.targets = self.df["cod_target"].astype(int).to_numpy()

        # transformaciones
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # imagen
        img_path = os.path.join(self.image_dir, self.paths[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # metadatos
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)

        y = torch.tensor(self.targets[idx], dtype=torch.long)

        return img, meta, y
