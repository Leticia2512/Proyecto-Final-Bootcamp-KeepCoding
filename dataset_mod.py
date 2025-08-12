
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def to_tensor(arr, dtype=torch.float32):
    arr = np.asarray(arr)
    if dtype == torch.float32:
        arr = arr.astype(np.float32, copy=False)
    elif dtype == torch.long:
        arr = arr.astype(np.int64, copy=False)
    return torch.from_numpy(arr).to(dtype).contiguous()


class AvengersDataset(Dataset):
    def __init__(self, left_paths, right_paths, features, targets,
                 transform=None, base_dir=None, multilabel=False):

        n = len(left_paths)
        assert len(right_paths) == n and len(features) == n and len(targets) == n, \
            "Longitudes inconsistentes entre columnas."

        # Guardar rutas absolutas si hay base_dir
        if base_dir is not None:
            self.left_paths = [os.path.join(base_dir, p) for p in left_paths]
            self.right_paths = [os.path.join(base_dir, p) for p in right_paths]
        else:
            self.left_paths = list(left_paths)
            self.right_paths = list(right_paths)

        self.features = to_tensor(features, dtype=torch.float32)
        self.targets = to_tensor(
            targets,  dtype=torch.float32 if multilabel else torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_paths[idx]).convert("RGB")
        right_img = Image.open(self.right_paths[idx]).convert("RGB")

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        meta = self.features[idx]
        y = self.targets[idx]

        return left_img, right_img, meta, y
