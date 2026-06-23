import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancer_pipeline.config import ensure_dir, load_config
from cancer_pipeline.data import class_weights_from_frame, make_splits
from cancer_pipeline.engine import fit, predict
from cancer_pipeline.models import create_model
from cancer_pipeline.plots import create_training_plots
from cancer_pipeline.utils import count_parameters, get_device, save_json, set_seed


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: str | Path, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(self.image_dir / f"{row.id}.tif").convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(row.label), dtype=torch.long)
        return image, label, row.id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT with hard example mining and stronger augmentation."
    )
    parser.add_argument("--config", default="configs/vit.yaml")
    parser.add_argument("--base-checkpoint", default="outputs/vit_small/best.pt")
    parser.add_argument("--output-run-name", default="vit_small_hard_mining")
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hard-weight", type=float, default=4.0)
    parser.add_argument("--fn-weight", type=float, default=6.0)
    parser.add_argument("--fp-weight", type=float, default=4.0)
    parser.add_argument("--randaugment-num-ops", type=int, default=2)
    parser.add_argument("--randaugment-magnitude", type=int, default=7)
    return parser.parse_args()


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_hard_aug_transform(image_size: int, num_ops: int, magnitude: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.12, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.12, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
        ]
    )


def load_model_from_config(config: dict, checkpoint_path: str | Path, device):
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = create_model(model_name, **model_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model, model_name


def add_hard_mining_columns(frame: pd.DataFrame, predictions: pd.DataFrame, threshold: float):
    pred_cols = predictions.rename(columns={"label": "pred_source_label"})
    merged = frame.merge(pred_cols[["id", "prob_cancer"]], on="id", how="left")
    if merged["prob_cancer"].isna().any():
        missing = int(merged["prob_cancer"].isna().sum())
        raise ValueError(f"Missing hard-mining predictions for {missing} training samples.")

    merged["pred_label"] = (merged["prob_cancer"] >= threshold).astype(int)
    merged["is_fp"] = (merged["label"].astype(int) == 0) & (merged["pred_label"] == 1)
    merged["is_fn"] = (merged["label"].astype(int) == 1) & (merged["pred_label"] == 0)
    merged["is_hard"] = merged["is_fp"] | merged["is_fn"]
    return merged


def make_sample_weights(frame: pd.DataFrame, fp_weight: float, fn_weight: float, hard_weight: float):
    weights = torch.ones(len(frame), dtype=torch.float)
    weights[frame["is_hard"].to_numpy()] = float(hard_weight)
    weights[frame["is_fp"].to_numpy()] = float(fp_weight)
    weights[frame["is_fn"].to_numpy()] = float(fn_weight)
    return weights


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    config["run_name"] = args.output_run_name
    config["train"]["epochs"] = int(args.epochs)
    config["train"]["lr"] = float(args.lr)

    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")) / config["run_name"])
    device = get_device()
    ds_cfg = config["dataset"]
    train_cfg = config["train"]
    root = Path(ds_cfg["root"])
    image_size = int(ds_cfg["image_size"])

    train_df, val_df, test_df = make_splits(
        root=root,
        val_size=float(ds_cfg["val_size"]),
        test_size=float(ds_cfg["test_size"]),
        seed=seed,
    )

    eval_transform = build_eval_transform(image_size)
    train_eval_ds = FrameImageDataset(train_df, root / "train", eval_transform)
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(ds_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    base_model, model_name = load_model_from_config(config, args.base_checkpoint, device)
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    _, train_predictions = predict(base_model, train_eval_loader, device, amp)
    train_predictions.to_csv(output_dir / "train_predictions_before_hard_mining.csv", index=False)

    train_df = add_hard_mining_columns(train_df, train_predictions, args.threshold)
    hard_summary = {
        "threshold": args.threshold,
        "train_samples": int(len(train_df)),
        "hard_samples": int(train_df["is_hard"].sum()),
        "fp_samples": int(train_df["is_fp"].sum()),
        "fn_samples": int(train_df["is_fn"].sum()),
        "fp_weight": args.fp_weight,
        "fn_weight": args.fn_weight,
        "hard_weight": args.hard_weight,
    }
    save_json(hard_summary, output_dir / "hard_mining_summary.json")
    train_df.to_csv(output_dir / "train_hard_examples.csv", index=False)

    train_ds = FrameImageDataset(
        train_df,
        root / "train",
        build_hard_aug_transform(
            image_size=image_size,
            num_ops=args.randaugment_num_ops,
            magnitude=args.randaugment_magnitude,
        ),
    )
    val_ds = FrameImageDataset(val_df, root / "train", eval_transform)
    test_ds = FrameImageDataset(test_df, root / "train", eval_transform)

    sampler = WeightedRandomSampler(
        weights=make_sample_weights(train_df, args.fp_weight, args.fn_weight, args.hard_weight),
        num_samples=len(train_df),
        replacement=True,
    )
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=int(train_cfg["batch_size"]),
            sampler=sampler,
            num_workers=int(ds_cfg.get("num_workers", 4)),
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(ds_cfg.get("num_workers", 4)),
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(ds_cfg.get("num_workers", 4)),
            pin_memory=True,
        ),
    }

    model, _ = load_model_from_config(config, args.base_checkpoint, device)
    weight = None
    if bool(train_cfg.get("class_weight", False)):
        weight = class_weights_from_frame(train_df).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))

    print(f"device={device}")
    print(f"model={model_name} trainable_params={count_parameters(model):,}")
    print(f"hard_mining={hard_summary}")
    summary = fit(model, loaders, criterion, optimizer, scheduler, device, config, output_dir)

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics, test_predictions = predict(model, loaders["test"], device, amp)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json({"summary": summary, "hard_mining": hard_summary, "test": test_metrics}, output_dir / "metrics.json")
    create_training_plots(output_dir)
    print(test_metrics)


if __name__ == "__main__":
    main()
