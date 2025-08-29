import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from PIL import Image
import numpy as np
from torchvision import transforms as T

# Mapeo de clases (copiado del script de entrenamiento)
KEEP_CLASSES = [0, 1, 2, 5, 6]
OLD2NEW = {old: new for new, old in enumerate(KEEP_CLASSES)}

# Definición del modelo (copiada del script de entrenamiento)
class ImageClassifier(nn.Module):
    def __init__(self, meta_dim: int, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
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

# Dimensiones del modelo (basadas en los hiperparámetros de entrenamiento)
META_DIM = 2 
NUM_CLASSES = len(KEEP_CLASSES)
MODEL_PATH = "bestmodel/best_model.pt"

# Cargar el modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier(
    meta_dim=META_DIM,
    num_classes=NUM_CLASSES,
    pretrained=False # No se cargam cargar los pesos de imagenet porque usaremos los nuestros
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Poner el modelo en modo de evaluación

def predict(image_path: str, meta_data: dict):
    """
    Realiza una predicción sobre una imagen y metadatos.

    Args:
        image_path: Ruta a la imagen.
        meta_data: Diccionario con los metadatos, ej: {"age": 45, "gender": "male"}
    
    Returns:
        El índice de la clase predicha.
    """
    # 1. Preprocesar la imagen   
    transform = T.Compose([       
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 2. Preprocesar los metadatos
    gender_map = {"male": 1, "female": 0}
    meta_tensor = torch.tensor([meta_data["age"], gender_map[meta_data["gender"]]], 
                               dtype=torch.float32).unsqueeze(0).to(device)
    
    # 3. Realizar la predicción
    with torch.no_grad():
        logits = model(image_tensor, meta_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
    # Obtener el índice de la clase con la probabilidad más alta
    predicted_class_index = torch.argmax(probabilities, dim=1).item()
    
    # El modelo predice los índices nuevos, no los originales
    new2old = {new: old for old, new in OLD2NEW.items()}
    original_class_index = new2old[predicted_class_index]

    return original_class_index, probabilities[0]


if __name__ == "__main__":
    # Asegúrate de que la ruta de la imagen sea correcta
    image_path = "backend_fastAPI\19_right.jpg" 
    meta_data = {"age": 45, "gender": "male"} # Ejemplo de datos de metadatos

    # Llamar a la función de predicción
    predicted_class, probabilities = predict(image_path, meta_data)

    print(f"La clase original predicha es: {predicted_class}")
    print("Probabilidades para cada clase remapeada:")
    print(f"Clases: {[f'clase_{c}' for c in KEEP_CLASSES]}")
    print(f"Probabilidades: {[f'{p:.4f}' for p in probabilities.tolist()]}")
    
    # Puedes usar un diccionario para mapear los índices a nombres de clases más descriptivos si los tienes
    class_names = {0: "A", 1: "C", 2: "D", 5: "M", 6: "N"}
    print(f"\nPredicción final: {class_names[predicted_class]}")
