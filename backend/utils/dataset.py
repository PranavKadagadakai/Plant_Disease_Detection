from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from utils.transforms import test_transform, train_transform

from config import BATCH_SIZE, DATA_DIR


def get_dataloaders():
    dataset = ImageFolder(DATA_DIR, transform=train_transform)

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, dataset.classes
