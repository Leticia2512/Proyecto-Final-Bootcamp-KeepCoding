# eye_pytorch_dataset_Leti_Exp_3.py 

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# ----- Transforms modulares ------
def get_train_transform(img_size: int = 224):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02)], p=0.4),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.05, scale=(0.02, 0.06)),
    ])


def get_eval_transform(img_size: int = 224):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


# ----- CLAHE selectivo (determinista) -----
def _apply_clahe_pil(pil_img, clip=2.0, tiles=(8, 8)):
    """
    Aplica CLAHE en canal L (LAB). Devuelve una PIL Image.
    Se usa solo si _HAS_CV2=True.
    """
    if not _HAS_CV2:
        return pil_img
    import numpy as _np
    img = cv2.cvtColor(_np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    Lc = clahe.apply(L)
    lab_c = cv2.merge([Lc, A, B])
    out = cv2.cvtColor(lab_c, cv2.COLOR_Lab2BGR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


class EyeDataset(Dataset):
    """
    Dataset por ojo (multiclase: una etiqueta por muestra).

    Se espera un parquet con:
      - 'filename'   : str (nombre de la imagen)
      - 'cod_target' : int (clase diagnóstica)
      - columnas de metadatos en `feature_cols` (p.ej. 'Patient Age', 'Patient_Sex_Binario')
      - (Opcional) 'is_dark', 'is_low_contrast' para preprocesado selectivo
    """

    def __init__(self, parquet_path: str, image_dir: str, feature_cols, transform=None, num_classes=None, train: bool = False):
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
        self.train = bool(train)

        self.has_quality = all(c in self.df.columns for c in ["is_dark", "is_low_contrast"])

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

        # PREPROCESADO DETERMINISTA: CLAHE selectivo con los flags de calidad
        if self.has_quality and _HAS_CV2:
            is_dark = bool(self.df.at[idx, "is_dark"])
            is_low  = bool(self.df.at[idx, "is_low_contrast"])
            if is_dark or is_low:
                img = _apply_clahe_pil(img, clip=2.0, tiles=(8, 8))

        img = self.transform(img)

        # Metadatos y target
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)

        return img, meta, y
