import dataset_mod as dm
import pandas as pd
import torchvision.transforms as transforms

df = pd.read_parquet("EDA/dataset_meta_final.parquet")

# Cargamos las rutas de las imagenes

image_dir = "224x224"
left_paths = df["Left-Fundus"].tolist()
right_paths = df["Right-Fundus"].tolist()


# Metadatos numéricos preprocesados
""" no añado el embedding de indices de las enfermedades """
feature_cols = ["Patient Age", "Patient_Sex_Binario"]
features = df[feature_cols].values

# definimos el target
targets = df[["N", "D", "G", "C", "A", "H", "M", "O"]].values

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = dm.AvengersDataset(
    left_paths=left_paths,
    right_paths=right_paths,
    features=features,
    targets=targets,
    transform=transform,
    base_dir=image_dir,
    multilabel=True
)
