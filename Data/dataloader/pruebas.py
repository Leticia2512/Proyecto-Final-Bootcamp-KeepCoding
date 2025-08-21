import torch
from torch.utils.data import Subset

data = torch.load("train_dataset.pt", weights_only=False)
train_ds = Subset(data["dataset"], data["indices"])

print(train_ds)