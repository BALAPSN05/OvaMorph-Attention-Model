from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return train_tf, eval_tf


def build_dataloaders(data_dir, batch_size=16, image_size=224, val_fraction=0.2, workers=2, seed=42):
    data_dir = Path(data_dir)
    train_tf, eval_tf = build_transforms(image_size)

    base = datasets.ImageFolder(data_dir, transform=train_tf)
    n_val = max(1, int(len(base) * val_fraction))
    n_train = len(base) - n_val
    generator = __import__('torch').Generator().manual_seed(seed)
    train_set, val_set = random_split(base, [n_train, n_val], generator=generator)

    # Use deterministic evaluation preprocessing for validation.
    val_set.dataset = datasets.ImageFolder(data_dir, transform=eval_tf)
    val_set.indices = train_set.indices.__class__(val_set.indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, base.classes
