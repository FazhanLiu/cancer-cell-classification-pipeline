import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancer_pipeline.config import ensure_dir, load_config
from cancer_pipeline.data import make_loaders
from cancer_pipeline.engine import fit, predict
from cancer_pipeline.models import create_model
from cancer_pipeline.plots import create_training_plots
from cancer_pipeline.utils import count_parameters, get_device, save_json, set_seed


class FocalLoss(nn.Module):
    """Multi-class focal loss for hard example focused training.

    Focal loss down-weights easy samples and keeps larger gradients on samples
    that the model currently finds difficult. This is useful when FP/FN errors
    are still high after standard cross-entropy training.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_negative: float = 0.5,
        alpha_positive: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "alpha",
            torch.tensor([float(alpha_negative), float(alpha_positive)], dtype=torch.float32),
        )

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        if self.label_smoothing > 0:
            num_classes = logits.size(1)
            smooth = self.label_smoothing / num_classes
            target_dist = torch.full_like(logits, smooth)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
            ce_loss = -(target_dist * log_probs).sum(dim=1)
            pt = (target_dist * probs).sum(dim=1).clamp_min(1e-8)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction="none")
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)

        alpha = self.alpha.to(logits.device).gather(0, targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        return (alpha * focal_weight * ce_loss).mean()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ViT-Small with focal loss to reduce hard FP/FN errors."
    )
    parser.add_argument("--config", default="configs/vit.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override base output directory. Defaults to output_dir in the config file.",
    )
    parser.add_argument("--output-run-name", default="vit_small_focal_loss")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument(
        "--alpha-negative",
        type=float,
        default=0.5,
        help="Class weight for normal/non-cancer samples. Increase this to penalize FP more.",
    )
    parser.add_argument(
        "--alpha-positive",
        type=float,
        default=0.5,
        help="Class weight for cancer samples. Increase this to penalize FN more.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Usually keep this at 0 for focal loss unless validation becomes unstable.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=None,
        help="Optional checkpoint to fine-tune from, e.g. outputs/vit_small/best.pt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    config["run_name"] = args.output_run_name
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    config["loss"] = {
        "name": "focal_loss",
        "gamma": args.gamma,
        "alpha_negative": args.alpha_negative,
        "alpha_positive": args.alpha_positive,
        "label_smoothing": args.label_smoothing,
    }
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        config["train"]["lr"] = float(args.lr)
    config["train"]["class_weight"] = False
    config["train"]["label_smoothing"] = float(args.label_smoothing)

    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")) / config["run_name"])
    loaders, _ = make_loaders(config)
    device = get_device()

    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = create_model(model_name, **model_cfg).to(device)
    if args.pretrained_checkpoint:
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        config["pretrained_checkpoint"] = str(args.pretrained_checkpoint)

    train_cfg = config["train"]
    criterion = FocalLoss(
        gamma=args.gamma,
        alpha_negative=args.alpha_negative,
        alpha_positive=args.alpha_positive,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]))

    print(f"device={device}")
    print(f"model={model_name} trainable_params={count_parameters(model):,}")
    print(
        "loss=focal "
        f"gamma={args.gamma} "
        f"alpha_negative={args.alpha_negative} "
        f"alpha_positive={args.alpha_positive} "
        f"label_smoothing={args.label_smoothing}"
    )
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
    save_json({"summary": summary, "loss": config["loss"], "test": test_metrics}, output_dir / "metrics.json")
    create_training_plots(output_dir)
    print(test_metrics)


if __name__ == "__main__":
    main()
