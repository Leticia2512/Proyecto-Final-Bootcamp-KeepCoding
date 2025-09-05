import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
# 1) CONFIGURACIÓN
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "dataset"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)

EXPERIMENT_NAME = "ResNet50_MPS"
RUN_NAME = f"resnet50_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4 #cambiamos antes 1e-4, volvemos a este LR
WEIGHT_DECAY = 2e-4 #Cambiamos antes 1e-4 mas suavizado 
PATIENCE = 6  

# ============================================================
# 2) DISPOSITIVO
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Dispositivo:", device)

# ============================================================
# 3) DATASETS Y DATALOADERS
# ============================================================
def load_dataloaders(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False):
    train_ds = torch.load(DATA_DIR / "train_dataset.pt", weights_only=False)
    val_ds   = torch.load(DATA_DIR / "val_dataset.pt",   weights_only=False)
    test_ds  = torch.load(DATA_DIR / "test_dataset.pt",  weights_only=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

# ============================================================
# 4) MODELO MULTIMODAL
# ============================================================
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, meta_dim):
        super(ImageClassifier, self).__init__()

        # Backbone ResNet50 preentrenada
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Red para metadatos
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        # Cabezal final
        self.classifier = nn.Sequential(
            nn.Linear(in_feats + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4), #Cambio a 0,5 antes 0,3 . De nuevo 0,3 
            nn.Linear(256, num_classes)
        )

    def forward(self, x_img, x_meta):
        f_img = self.resnet(x_img)
        f_meta = self.meta_net(x_meta)
        x = torch.cat([f_img, f_meta], dim=1)
        return self.classifier(x)

# ============================================================
# 5) FUNCIONES DE ENTRENAMIENTO Y VALIDACIÓN
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, feats, labels in loader:
        imgs, feats, labels = imgs.to(device), feats.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for imgs, feats, labels in loader:
            imgs, feats, labels = imgs.to(device), feats.to(device), labels.to(device)

            outputs = model(imgs, feats)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)

# ============================================================
# 6) FUNCIONES AUXILIARES
# ============================================================
def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ============================================================
# 7) MAIN LOOP
# ============================================================
def main():
    # Dataloaders
    train_loader, val_loader, test_loader = load_dataloaders()

    # Inferir dimensiones
    sample_img, sample_meta, sample_label = next(iter(train_loader))
    meta_dim = sample_meta.shape[1]
    num_classes = len(torch.unique(sample_label))

    # Modelo + optimizador + loss
    model = ImageClassifier(num_classes=num_classes, meta_dim=meta_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Configurar MLflow
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):
        # Log de parámetros
        mlflow.log_param("architecture", "ResNet50 + MLP(meta)")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("lr", LR)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("epochs", EPOCHS)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            print(f"[Epoch {epoch}] "
                  f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                  f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            # Early stopping y guardado del mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), BASE_DIR / "best_resnet50.pt")
                mlflow.pytorch.log_model(model, artifact_path="resnet50_model")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping activado.")
                    break

        # Evaluación en test
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        print(f"[TEST] Loss: {test_loss:.4f}, acc: {test_acc:.4f}")
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

        # Classification report
        class_names = [str(i) for i in range(num_classes)]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_txt = classification_report(y_true, y_pred, target_names=class_names)
        with open(BASE_DIR / "classification_report.txt", "w") as f:
            f.write(report_txt)
        mlflow.log_artifact(BASE_DIR / "classification_report.txt")

        # Matriz de confusión
        cm_path = BASE_DIR / "confusion_matrix.png"
        save_confusion_matrix(y_true, y_pred, class_names, cm_path)
        mlflow.log_artifact(cm_path)


if __name__ == "__main__":
    main()


