# eye_pytorch_dataset_Leti.py 

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# ----- Transforms modulares ------
def get_train_transform(img_size: int = 224):
    return T.Compose([
        T.RandomAffine(
            degrees=7,                 # rotación ±7°
            translate=(0.02, 0.02),    # ±2% desplazamiento
            scale=(0.95, 1.05),        # ±5% escala
            shear=None
        ),
        T.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.01, 0.03), ratio=(0.3, 3.3), value='random')
    ])


def get_eval_transform(img_size: int = 224):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


class EyeDataset(Dataset):
    """
    Dataset por ojo (multiclase: una etiqueta por muestra).

    Se espera un parquet con:
      - 'filename'   : str (nombre de la imagen)
      - 'cod_target' : int (clase diagnóstica)
      - `feature_cols`: metadatos ('Patient Age', 'Patient_Sex_Binario')
    """

    def __init__(self, parquet_path:str, image_dir:str, feature_cols, transform=None, num_classes=None):
       
        BASE_DIR = Path(__file__).resolve().parent
        parent_dir = BASE_DIR.parent

        parquet_path = Path(parquet_path)
        if not parquet_path.is_absolute():
            parquet_path = (parent_dir / parquet_path).resolve()

        image_dir = Path(image_dir)
        if not image_dir.is_absolute():
            image_dir = (parent_dir / image_dir).resolve()

        # Cargar dataframe 
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.image_dir = image_dir
        self.feature_cols = list(feature_cols)

        # Cache ligera
        self.paths = self.df["filename"].astype(str).tolist()
        self.meta = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.targets = self.df["cod_target"].astype(int).to_numpy()

        # Transform por defecto (eval) si no se pasa ninguna
        self.transform = transform or get_eval_transform(224)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Imágenes
        img_path = self.image_dir / self.paths[idx]
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        img = self.transform(img)

        # Metadatos y target
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long) 

        return img, meta, y
