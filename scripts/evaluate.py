import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancer_pipeline.config import load_config
from cancer_pipeline.data import make_loaders
from cancer_pipeline.engine import predict
from cancer_pipeline.models import create_model
from cancer_pipeline.utils import get_device, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    loaders, _ = make_loaders(config)
    device = get_device()
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = create_model(model_name, **model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics, predictions = predict(
        model,
        loaders["test"],
        device,
        bool(config["train"].get("amp", True)) and device.type == "cuda",
    )
    output_dir = Path(args.checkpoint).resolve().parent
    predictions.to_csv(output_dir / "eval_predictions.csv", index=False)
    save_json(metrics, output_dir / "eval_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()

