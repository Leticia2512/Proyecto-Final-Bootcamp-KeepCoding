# eye_pytorch_dataset_Leti_Exp_3.py
# ------------------------------------------------------------
# Dataset a nivel de ojo para clasificación multiclase usando
# ÚNICAMENTE las imágenes (sin metadatos).
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ---------- Transforms para entrenamiento y evaluación ----------
def get_train_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.1)
    ])


def get_eval_transform(img_size: int = 224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


class EyeDatasetIMG(Dataset):
    """
    Dataset por ojo (multiclase: una etiqueta por muestra), solo-imágenes.

    Estructura esperada del fichero de anotaciones (parquet o CSV):
      - 'filename'   : str (nombre del archivo de imagen)
      - 'cod_target' : int (id de clase, p.e. 0..K-1)
    """

    def __init__(self, parquet_path: str, image_dir: str, transform=None, img_size: int = 224):
        """
        Args:
            parquet_path: Ruta al .parquet o .csv con columnas 'filename' y 'cod_target'.
            image_dir: Carpeta donde están las imágenes físicas.
            transform: Transform opcional de torchvision (si None => eval por defecto).
            img_size: Tamaño de imagen si usamos los transforms por defecto.
        """
        base_dir = Path(__file__).resolve().parent
        parent_dir = base_dir.parent

        # Normalizamos rutas relativas a la carpeta padre del script
        parquet_path = Path(parquet_path)
        if not parquet_path.is_absolute():
            parquet_path = (parent_dir / parquet_path).resolve()

        image_dir = Path(image_dir)
        if not image_dir.is_absolute():
            image_dir = (parent_dir / image_dir).resolve()

        # Carga del DataFrame 
        if parquet_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_csv(parquet_path)

        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

        # Cache de columnas 
        self.paths = self.df["filename"].astype(str).tolist()
        self.targets = self.df["cod_target"].astype(int).tolist()

        # Si no se pasa transform, usar el de evaluación (determinista)
        self.transform = transform if transform is not None else get_eval_transform(img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # --- Carga de imagen como PIL y conversión a RGB ---
        img_path = self.image_dir / self.paths[idx]
        with Image.open(img_path) as im:
            img = im.convert("RGB")

        # Aplicar transform si aplicara
        if self.transform is not None:
            img = self.transform(img)

        y = torch.tensor(self.targets[idx], dtype=torch.long)

        return img, y
