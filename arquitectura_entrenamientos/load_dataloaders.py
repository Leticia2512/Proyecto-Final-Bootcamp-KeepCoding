import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path

try: 
    from eye_pytorch_dataset import get_train_transform, get_eval_transform

except ImportError:
    print("WARNING: No se pudo importar get_train_transform/get_eval_transform. Asegúrate de que eye_pytorch_dataset esté accesible.")
    # Funciones dummy si falla la importación
    def get_train_transform(size=224): return None
    def get_eval_transform(size=224): return None


def load_dataloaders(train_pt, val_pt, test_pt, batch_size=32, num_workers=4, pin_memory=True, seed=42):
    """
    Carga Subsets guardados directamente (.pt), recrea los Subsets 
    para reasignar la transformación de entrenamiento, y devuelve dataloaders.
    """
    
    # 1. Cargar el objeto Subset (usando str() y weights_only=False para robustez)
    # Aquí train_ds, val_ds y test_ds son directamente objetos Subset.
    try:
        # Usa weights_only=False para cargar objetos complejos como Dataset/Subset
        train_ds = torch.load(str(train_pt), weights_only=False)
        val_ds = torch.load(str(val_pt), weights_only=False)
        test_ds = torch.load(str(test_pt), weights_only=False)

    except Exception as e:
        print(f"Error al cargar datasets: {e}")
        print("Asegúrate de que los archivos .pt existan y que no se hayan guardado usando una versión incompatible de PyTorch.")
        raise

    if not isinstance(train_ds, Subset):
        raise TypeError(
            f"El archivo {train_pt.name} no contiene un objeto Subset. Contiene: {type(train_ds)}")

    # El dataset base es el .dataset dentro del Subset (ds_train.dataset)
    base_dataset = train_ds.dataset

    # 2. Reasignar TRANSFORMACIONES (Crucial para el data augmentation en train)
    train_ds.dataset.transform = get_train_transform(224)

    val_ds.dataset.transform = get_eval_transform(224)
    test_ds.dataset.transform = get_eval_transform(224)

    # 3. Crear DataLoaders

    # Generator reproducible
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, generator=g)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
