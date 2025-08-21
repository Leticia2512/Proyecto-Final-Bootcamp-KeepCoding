from torch.utils.data import DataLoader, Subset, TensorDataset
import torch
from create_split_dataset import create_split
from fun_transform import square_image
from torchvision import transforms
import os

tr_idx, va_idx, te_idx, ds = create_split()

train_imgs_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p= 0.2),
    transforms.RandomRotation(15)
])

dataset_train = Subset(ds, tr_idx)
dataset_train.transform = train_imgs_transforms

# --- 4) DataLoaders (con Subset sobre índices) ---
train_loader = DataLoader(dataset_train, batch_size=32,
                          shuffle=True, num_workers=0, pin_memory=True)  # pin_memory=True si usas GPU
val_loader = DataLoader(Subset(ds, va_idx), batch_size=32,
                        shuffle=False, num_workers=0)
test_loader = DataLoader(Subset(ds, te_idx), batch_size=32,
                         shuffle=False, num_workers=0)

"""

# guardar DataLoaders
subset1 = train_loader.dataset   # esto es un Subset(ds, indices)
subset2 = val_loader.dataset
subset3 = test_loader.dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
input_dir = os.path.join(parent_dir, r"Data/dataloader/")

torch.save(subset1, f"{input_dir}train_dataset.pt")
torch.save(subset2, f"{input_dir}val_dataset.pt")
torch.save(subset3, f"{input_dir}test_dataset.pt")

"""

torch.save({
    "dataset": ds,          # tu EyeDataset
    "indices": train_loader    # lista de índices
}, "train_dataset2.pt")






