from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from eye_pytorch_dataset import EyeDataset

parquet_file = "EDA/prepared/dataset_eyes_long.parquet"
image_dir = "224x224"
feature_cols = ["Patient Age", "Patient_Sex_Binario"]

ds = EyeDataset(parquet_path=parquet_file,
                image_dir=image_dir, feature_cols=feature_cols)

idx = list(range(len(ds)))
tr, tmp = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
va, te = train_test_split(tmp, test_size=0.5, random_state=42, shuffle=True)

train_loader = DataLoader(Subset(ds, tr), batch_size=32,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(Subset(ds, va), batch_size=32,
                        shuffle=False, num_workers=4)
test_loader = DataLoader(Subset(ds, te), batch_size=32,
                         shuffle=False, num_workers=4)

# sanity
x, m, y = ds[0]
# -> [3,224,224], [2], [num_classes], torch.float32
print(x.shape, m.shape, y.shape, y.dtype)
