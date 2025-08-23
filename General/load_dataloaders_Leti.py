import torch
from torch.utils.data import DataLoader


def load_dataloaders(train_pt, val_pt, test_pt, batch_size=32, num_workers=4, pin_memory=True, seed=42):
    """
    Carga Subsets guardados (.pt) y devuelve dataloaders listos para entrenar.
    """
    train_ds = torch.load(train_pt)
    val_ds = torch.load(val_pt)
    test_ds = torch.load(test_pt)

    # Generator reproducible
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,generator=g)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
