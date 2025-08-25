# load_dataloaders_Leti_Exp_3.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Usar los transforms del dataset SOLO-IMÁGENES.
try:
    from eye_pytorch_dataset_Leti_Exp_3 import get_train_transform, get_eval_transform
except Exception:
    try:
        from eye_pytorch_dataset_Leti_Exp_3 import get_train_transform, get_eval_transform
    except Exception:
        import torchvision.transforms as T
        def get_train_transform(size=224):
            return T.Compose([T.Resize((size, size)), T.ToTensor()])
        def get_eval_transform(size=224):
            return T.Compose([T.Resize((size, size)), T.ToTensor()])


def _set_transform(ds, tfm):
    """
    Reasigna el transform al dataset base (si es Subset) o al propio dataset.
    """
    base = getattr(ds, "dataset", None)
    if base is not None and hasattr(base, "transform"):
        base.transform = tfm
    elif hasattr(ds, "transform"):
        ds.transform = tfm


def load_dataloaders(train_pt, val_pt, test_pt, batch_size=32, num_workers=4, pin_memory=True, seed=42, img_size=224,
                     persistent_workers=True, prefetch_factor=2):
    """
    Carga Subsets guardados (.pt) y devuelve dataloaders listos para entrenar.
    Mantiene la firma original para ser un drop-in replacement.
    """
    # Rutas como string por compatibilidad
    train_pt = str(Path(train_pt))
    val_pt   = str(Path(val_pt))
    test_pt  = str(Path(test_pt))

    # 1) Cargar los objetos guardados (Subset/Dataset complejos)
    train_ds = torch.load(train_pt, weights_only=False)
    val_ds   = torch.load(val_pt,   weights_only=False)
    test_ds  = torch.load(test_pt,  weights_only=False)

    # 2) Reinyectar transforms (crucial para convertir PIL -> Tensor)
    _set_transform(train_ds, get_train_transform(224))
    _set_transform(val_ds,   get_eval_transform(224))
    _set_transform(test_ds,  get_eval_transform(224))

    # 3) Generator reproducible para el shuffle de train
    g = torch.Generator()
    g.manual_seed(seed)

    # 4) DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
