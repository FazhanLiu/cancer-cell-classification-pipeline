from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HistopathologyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: str | Path, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.image_dir / f"{row.id}.tif"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(row.label), dtype=torch.long)
        return image, label, row.id


def build_transforms(image_size: int, train: bool):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def make_splits(root: str | Path, val_size: float, test_size: float, seed: int):
    root = Path(root)
    labels = pd.read_csv(root / "train_labels.csv")
    train_df, holdout_df = train_test_split(
        labels,
        test_size=val_size + test_size,
        stratify=labels["label"],
        random_state=seed,
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test,
        stratify=holdout_df["label"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def make_loaders(config: dict):
    ds_cfg = config["dataset"]
    root = Path(ds_cfg["root"])
    seed = int(config.get("seed", 42))
    train_df, val_df, test_df = make_splits(
        root=root,
        val_size=float(ds_cfg["val_size"]),
        test_size=float(ds_cfg["test_size"]),
        seed=seed,
    )
    image_size = int(ds_cfg["image_size"])
    train_ds = HistopathologyDataset(train_df, root / "train", build_transforms(image_size, True))
    val_ds = HistopathologyDataset(val_df, root / "train", build_transforms(image_size, False))
    test_ds = HistopathologyDataset(test_df, root / "train", build_transforms(image_size, False))

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(ds_cfg.get("num_workers", 4))
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, {"train": train_df, "val": val_df, "test": test_df}


def class_weights_from_frame(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["label"].value_counts().sort_index()
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)

