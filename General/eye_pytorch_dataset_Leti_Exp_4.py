# eye_pytorch_dataset_Leti_Exp_4.py
# Dataset SOLO-IMÁGENES (multiclase) 
# Pensado para entrenar con EfficientNet-B3


from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# ------- Transforms recomendadas --------
def get_train_transform(img_size: int = 300):

    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        T.RandomAutocontrast(p=0.30),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3))
    ])


def get_eval_transform(img_size: int = 300):
  
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ------- Dataset --------
class EyeDatasetIMG(Dataset):
    """
    Dataset por ojo (multiclase) SOLO-IMÁGENES.

    Estructura esperada del fichero de anotaciones (parquet o CSV):
      - 'filename'   : str (nombre del archivo de imagen)
      - 'cod_target' : int (id de clase, p.e. 0..K-1)
    """

    def __init__(
        self,
        parquet_path: str,
        image_dir: str,
        transform=None,
        img_size: int = 300,
        strict_checks: bool = False,
    ):
        """
        Args:
            parquet_path: Ruta al .parquet o .csv con 'filename' y 'cod_target'.
            image_dir: Carpeta donde están las imágenes físicas.
            transform: Transform opcional (si None => eval por defecto).
            img_size: Tamaño usado si aplicamos transforms por defecto.
            strict_checks: Si True, comprueba existencia de cada archivo y
                           lanza error si falta (más seguro, algo más lento).
        """
        base_dir = Path(__file__).resolve().parent
        parent_dir = base_dir.parent

        # Normalizar rutas relativas
        pq = Path(parquet_path)
        if not pq.is_absolute():
            pq = (parent_dir / pq).resolve()

        imgd = Path(image_dir)
        if not imgd.is_absolute():
            imgd = (parent_dir / imgd).resolve()

        if not pq.exists():
            raise FileNotFoundError(f"No se encuentra el fichero de anotaciones: {pq}")
        if not imgd.exists():
            raise FileNotFoundError(f"No se encuentra la carpeta de imágenes: {imgd}")

        # Cargar anotaciones
        if pq.suffix.lower() == ".parquet":
            df = pd.read_parquet(pq)
        else:
            df = pd.read_csv(pq)

        # Chequeos mínimos
        for col in ["filename", "cod_target"]:
            if col not in df.columns:
                raise ValueError(f"Falta la columna '{col}' en: {pq}")

        # Limpieza básica
        df = df.dropna(subset=["filename"]).copy()
        df["filename"] = df["filename"].astype(str).str.strip()

        # Cache de columnas
        self.df = df.reset_index(drop=True)
        self.image_dir = imgd
        self.paths = self.df["filename"].tolist()
        self.targets = self.df["cod_target"].astype(int).to_numpy()

        # Transform por defecto
        self.transform = transform if transform is not None else get_eval_transform(img_size)

        # verificación de archivos
        if strict_checks:
            missing = [p for p in self.paths if not (self.image_dir / p).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Se han encontrado {len(missing)} archivos ausentes. "
                    f"Ejemplos: {missing[:5]}"
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        # Carga robusta de imagen como PIL y conversión a RGB
        img_path = self.image_dir / self.paths[idx]
        try:
            with Image.open(img_path) as im:
                img = im.convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"No se pudo abrir la imagen '{img_path}': {e}")

        # Aplicar transforms
        img = self.transform(img) if self.transform is not None else img

        # Etiqueta multiclase
        y = torch.tensor(int(self.targets[idx]), dtype=torch.long)

        return img, y
