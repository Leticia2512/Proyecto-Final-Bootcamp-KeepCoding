import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

# ==== Clases y mapeos (top-5) ====
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(KEEP_CLASSES)}
NEW2OLD = {new: old for old, new in OLD2NEW.items()}


DROPOUT = 0.5936083897821617
# ==== Modelo ====


class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = False, dropout: float = 0.5):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_feats = self.backbone.classifier[1].in_features  # (Dropout, Linear)
        self.backbone.classifier = nn.Identity()

        self.meta = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
        )
        self.head = nn.Sequential(
            nn.Linear(in_feats + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        f_img = self.backbone(x_img)
        f_meta = self.meta(x_meta)
        f = torch.cat([f_img, f_meta], dim=1)
        return self.head(f)

# ==== Predictor ====


class OcularPredictor:
    def __init__(self, model_path: Path, fallback_mu_age: float = 0.0, fallback_sigma_age: float = 1.0):
        """
        model_path: ruta al .pt/.pth
        fallback_mu_age/sigma_age: por si el checkpoint no trae estadísticas.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gender_map = {"male": 1, "female": 0}

        # Cargar ckpt + modelo con los mismos hiperparámetros estructurales
        ckpt = torch.load(str(model_path), map_location=self.device)

        # Si el checkpoint es un dict enriquecido (recomendado)
        if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state" in ckpt):
            state = ckpt.get("state_dict", ckpt.get("model_state"))
            self.mu_age = float(ckpt.get("mu_age", fallback_mu_age))
            self.sigma_age = float(ckpt.get("sigma_age", fallback_sigma_age))
        else:
            # Si guardaste solo state_dict
            state = ckpt
            self.mu_age = float(fallback_mu_age)
            self.sigma_age = float(fallback_sigma_age)

        # Construir el modelo (dropout acorde al entrenamiento)
        self.model = ImageClassifier(
            meta_dim=2,
            num_classes=len(KEEP_CLASSES),
            pretrained=False,
            dropout=DROPOUT,  # pon aquí el que usaste al entrenar si cambió
        ).to(self.device)

        self.model.load_state_dict(state)
        self.model.eval()

        # Transforms imagen
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _normalize_meta(self, age: float, gender_str: str):
        # Normaliza edad como en train; protege sigma==0
        sigma = self.sigma_age if self.sigma_age != 0 else 1.0
        age_norm = (float(age) - float(self.mu_age)) / float(sigma)
        gender = self.gender_map[gender_str.lower()]
        return torch.tensor([age_norm, float(gender)], dtype=torch.float32)

    def preprocess(self, image_path: str, meta_data: dict):
        # Imagen
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"El archivo de imagen no existe: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Metadatos normalizados
        meta_tensor = self._normalize_meta(
            meta_data["age"], meta_data["gender"]).unsqueeze(0).to(self.device)
        return image_tensor, meta_tensor

    @torch.no_grad()
    def predict(self, image_path: str, meta_data: dict):
        x_img, x_meta = self.preprocess(image_path, meta_data)
        logits = self.model(x_img, x_meta)
        probabilities = torch.softmax(logits, dim=1)
        pred_new = int(torch.argmax(probabilities, dim=1).item())
        pred_old = NEW2OLD[pred_new]
        return pred_old, probabilities[0].cpu().numpy().tolist()


# ===========================
# Ejemplo de uso
# ===========================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "bestmodel" / "best_model.pt"

    # Si tu checkpoint NO guarda mu/sigma, ajusta aquí a los del train:
    # (pon los reales si los sabes; en su defecto 0 y 1 dejan "sin normalizar")
    predictor = OcularPredictor(
        model_path=MODEL_PATH,
        fallback_mu_age=57.8,   # <-- cámbialo por la media de edad del train si la tienes
        fallback_sigma_age=11.7  # <-- cámbialo por el std del train si lo guardaste
    )

    image_path = r"app\208_right.jpg"
    meta_data = {"age": 45, "gender": "male"}

    pred_class, probs = predictor.predict(image_path, meta_data)
    print(f"La índice de la clase predicha es: {pred_class}")
    print("Probabilidades para cada clase remapeada:")
    print(f"Clases: {[f'clase_{c}' for c in KEEP_CLASSES]}")
    print(f"Probabilities: {probs}")
