from torch.utils.data import DataLoader, Subset, TensorDataset
import torch
from create_split_dataset import create_split
from fun_transform import square_image
import torchvision.transforms as T

tr_idx, va_idx, te_idx, ds = create_split()


# --- 4) DataLoaders (con Subset sobre índices) ---
train_loader = DataLoader(Subset(ds, tr_idx), batch_size=32,
                          shuffle=True, num_workers=0, pin_memory=True)  # pin_memory=True si usas GPU
val_loader = DataLoader(Subset(ds, va_idx), batch_size=32,
                        shuffle=False, num_workers=0)
test_loader = DataLoader(Subset(ds, te_idx), batch_size=32,
                         shuffle=False, num_workers=0)

# guardar DataLoaders
subset1 = train_loader.dataset   # esto es un Subset(ds, indices)
subset2 = val_loader.dataset
subset3 = test_loader.dataset

torch.save(subset1, r"Data\dataloader\train_dataset.pt")
torch.save(subset2, r"Data\dataloader\val_dataset.pt")
torch.save(subset3, r"Data\dataloader\test_dataset.pt")
