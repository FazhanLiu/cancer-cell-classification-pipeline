import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancer_pipeline.config import ensure_dir, load_config
from cancer_pipeline.data import build_transforms, make_splits, HistopathologyDataset
from cancer_pipeline.metrics import classification_metrics
from cancer_pipeline.models import create_model
from cancer_pipeline.utils import get_device, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for images, _, _ in tqdm(loader, desc="ensemble eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    args = parse_args()
    config = load_config(args.config)
    ds_cfg = config["dataset"]
    _, _, test_df = make_splits(ds_cfg["root"], ds_cfg["val_size"], ds_cfg["test_size"], seed=42)
    dataset = HistopathologyDataset(
        test_df,
        Path(ds_cfg["root"]) / "train",
        build_transforms(int(ds_cfg["image_size"]), train=False),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=int(ds_cfg.get("num_workers", 4)),
        pin_memory=True,
    )
    device = get_device()
    weighted_probs = []
    weights = []

    for item in config["models"]:
        item = dict(item)
        weight = float(item.pop("weight", 1.0))
        checkpoint_path = item.pop("checkpoint")
        model_name = item.pop("name")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint.get("config", {}).get("model", {})
        model_kwargs = {k: v for k, v in model_config.items() if k != "name"}
        if not model_kwargs:
            model_kwargs = {"num_classes": 2, "pretrained": False}
        else:
            model_kwargs["pretrained"] = False
        model = create_model(model_name, **model_kwargs).to(device)
        model.load_state_dict(checkpoint["model"])
        weighted_probs.append(predict_probs(model, loader, device) * weight)
        weights.append(weight)

    final_prob = np.sum(weighted_probs, axis=0) / np.sum(weights)
    metrics = classification_metrics(test_df["label"].values, final_prob)
    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")) / config["run_name"])
    pd.DataFrame({"id": test_df["id"].values, "label": test_df["label"].values, "prob_cancer": final_prob}).to_csv(
        output_dir / "ensemble_predictions.csv",
        index=False,
    )
    save_json(metrics, output_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()

