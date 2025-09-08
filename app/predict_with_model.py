import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

# Mapeo de clases
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(KEEP_CLASSES)}
NEW2OLD = {new: old for old, new in OLD2NEW.items()}

# Definición del modelo
class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = False, dropout: float = 0.5):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_feats = self.backbone.classifier[1].in_features
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


class OcularPredictor:
    def __init__(self, model_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        self.gender_map = {"male": 1, "female": 0}

    def _load_model(self, model_path: Path):
        """Carga el modelo y sus pesos."""
        model = ImageClassifier(
            meta_dim=2,
            num_classes=len(KEEP_CLASSES),
            pretrained=False
        ).to(self.device)
        model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        model.eval()
        return model

    def _get_transforms(self):
        """Devuelve las transformaciones para la imagen."""
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    

    def preprocess(self, image_path: str, meta_data: dict):
        """
        Preprocesa la imagen y los metadatos para la predicción.
        """
        # 1. Preprocesar la imagen
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"El archivo de imagen no existe: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 2. Preprocesar los metadatos
        meta_tensor = torch.tensor(
            [meta_data["age"], self.gender_map[meta_data["gender"]]],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        return image_tensor, meta_tensor

    def predict(self, image_path: str, meta_data: dict):
        """
        Realiza una predicción sobre una imagen y metadatos.
        Args:
            image_path: Ruta a la imagen.
            meta_data: Diccionario con los metadatos, ej: {"age": 45, "gender": "male"}
        Returns:
            Tupla con la clase original predicha y las probabilidades.
        """
        image_tensor, meta_tensor = self.preprocess(image_path, meta_data)
        
        # 3. Realizar la predicción
        with torch.no_grad():
            logits = self.model(image_tensor, meta_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
        # Obtener el índice de la clase con la probabilidad más alta
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        original_class_index = NEW2OLD[predicted_class_index]

        return original_class_index, probabilities[0].cpu().numpy().tolist()

# Crear una única instancia del predictor

from pathlib import Path
# Obtiene la ruta del directorio actual donde se está ejecutando el script
BASE_DIR = Path(__file__).resolve().parent
# Define la ruta al modelo de forma relativa
MODEL_PATH = BASE_DIR / "bestmodel" / "best_model.pt"

# Crea la instancia del predictor usando la ruta correcta
predictor = OcularPredictor(model_path=MODEL_PATH)