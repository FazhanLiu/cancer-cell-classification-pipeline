import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancer_pipeline.config import ensure_dir, load_config
from cancer_pipeline.data import class_weights_from_frame, make_loaders
from cancer_pipeline.engine import fit, predict
from cancer_pipeline.models import create_model
from cancer_pipeline.utils import count_parameters, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")) / config["run_name"])

    loaders, frames = make_loaders(config)
    device = get_device()
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = create_model(model_name, **model_cfg).to(device)

    train_cfg = config["train"]
    weight = None
    if bool(train_cfg.get("class_weight", False)):
        weight = class_weights_from_frame(frames["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))

    print(f"device={device}")
    print(f"model={model_name} trainable_params={count_parameters(model):,}")
    summary = fit(model, loaders, criterion, optimizer, scheduler, device, config, output_dir)

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics, test_predictions = predict(
        model,
        loaders["test"],
        device,
        bool(train_cfg.get("amp", True)) and device.type == "cuda",
    )
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    save_json({"summary": summary, "test": test_metrics}, output_dir / "metrics.json")
    print(test_metrics)


if __name__ == "__main__":
    main()

