import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import dataset_mod as dm
from torch.utils.data import DataLoader

# ==== Configuración ====
parquet_path = "EDA/dataset_meta_final.parquet"
image_dir = "224x224"
feature_cols = ["Patient Age", "Patient_Sex_Binario"]
target_cols = ["N", "D", "G", "C", "A", "H", "M", "O"]

test_size = 0.1
val_size = 0.1
random_state = 42

# ==== Cargar DataFrame ====
df = pd.read_parquet(parquet_path)

# ==== Split de filas ====
# Primero test
df_train, df_test = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    shuffle=True
)

# Luego val desde train restante
df_train, df_val = train_test_split(
    df_train,
    test_size=val_size / (1 - test_size),
    random_state=random_state,
    shuffle=True
)

print(
    f"Imágenes → Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# ==== Transformación imágenes ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Función para crear dataset ====


def make_dataset(df_split):
    return dm.AvengersDataset(
        left_paths=df_split["Left-Fundus"].tolist(),
        right_paths=df_split["Right-Fundus"].tolist(),
        features=df_split[feature_cols],
        targets=df_split[target_cols],
        transform=transform,
        base_dir=image_dir,
        multilabel=True
    )


# ==== Crear datasets ====
train_dataset = make_dataset(df_train)
val_dataset = make_dataset(df_val)
test_dataset = make_dataset(df_test)


train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
