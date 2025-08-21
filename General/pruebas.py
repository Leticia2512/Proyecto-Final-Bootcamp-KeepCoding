import torch
from torch.utils.data import Subset
from eye_pytorch_dataset import EyeDataset


data = torch.load(r"Data\dataloader\train_dataset.pt", weights_only=False)

print(data)