import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cancer_pipeline.config import load_config
from cancer_pipeline.models import create_model
from cancer_pipeline.utils import get_device


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit.yaml")
    parser.add_argument("--checkpoint", default="outputs/vit_small/best.pt")
    parser.add_argument("--predictions", default="outputs/vit_small/test_predictions.csv")
    parser.add_argument("--output", default="outputs/vit_small/xai")
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--samples-per-group", type=int, default=2)
    parser.add_argument("--lime-samples", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--occlusion-patch", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--attention-weight", type=float, default=0.5)
    parser.add_argument("--lime-weight", type=float, default=0.3)
    parser.add_argument("--occlusion-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_vit(config_path: str, checkpoint_path: str, device):
    config = load_config(config_path)
    model_cfg = dict(config["model"])
    model_name = model_cfg.pop("name")
    model = create_model(model_name, **model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    for block in getattr(model, "blocks", []):
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
    return model, config


def image_path_from_id(dataset_root: str | Path, image_id: str):
    return Path(dataset_root) / "train" / f"{image_id}.tif"


def load_image(path: str | Path, image_size: int):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return image


@torch.no_grad()
def predict_prob(model, tensor, device):
    logits = model(tensor.to(device))
    return logits.softmax(dim=1)[:, 1].detach().cpu().numpy()


def select_samples(predictions_path: str | Path, threshold: float, samples_per_group: int, seed: int):
    predictions = pd.read_csv(predictions_path).copy()
    predictions["pred_label"] = (predictions["prob_cancer"] >= threshold).astype(int)
    predictions["group"] = np.select(
        [
            (predictions["label"] == 1) & (predictions["pred_label"] == 1),
            (predictions["label"] == 0) & (predictions["pred_label"] == 0),
            (predictions["label"] == 0) & (predictions["pred_label"] == 1),
            (predictions["label"] == 1) & (predictions["pred_label"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="OTHER",
    )
    selected = []
    for group in ["TP", "TN", "FP", "FN"]:
        frame = predictions[predictions["group"] == group].copy()
        if frame.empty:
            continue
        frame["distance_from_threshold"] = (frame["prob_cancer"] - threshold).abs()
        if group in {"FP", "FN"}:
            frame = frame.sort_values("distance_from_threshold")
        else:
            frame = frame.sample(frac=1.0, random_state=seed).sort_values("distance_from_threshold")
        selected.append(frame.head(samples_per_group))
    if not selected:
        raise ValueError("No samples were selected. Check predictions and threshold.")
    return pd.concat(selected, ignore_index=True)


def confusion_group(label: int, pred_label: int) -> str:
    if label == 1 and pred_label == 1:
        return "TP"
    if label == 0 and pred_label == 0:
        return "TN"
    if label == 0 and pred_label == 1:
        return "FP"
    if label == 1 and pred_label == 0:
        return "FN"
    return "OTHER"


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    v_min = float(values.min())
    v_max = float(values.max())
    if math.isclose(v_min, v_max):
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def weighted_final_score(attention_map, lime_map, occlusion_map, attention_weight=0.5, lime_weight=0.3, occlusion_weight=0.2):
    total_weight = attention_weight + lime_weight + occlusion_weight
    if total_weight <= 0:
        raise ValueError("At least one XAI fusion weight must be positive.")
    attention = normalize_map(attention_map)
    lime = normalize_map(lime_map)
    occlusion = normalize_map(occlusion_map)
    fused = (
        attention_weight * attention
        + lime_weight * lime
        + occlusion_weight * occlusion
    ) / total_weight
    return normalize_map(fused)


def overlay_heatmap(image, heatmap, alpha=0.42, cmap="jet"):
    base = np.asarray(image).astype(np.float32) / 255.0
    heatmap = normalize_map(heatmap)
    color = plt.get_cmap(cmap)(heatmap)[..., :3]
    overlay = (1 - alpha) * base + alpha * color
    return np.clip(overlay, 0, 1)


def save_heatmap_triplet(image, heatmap, output_path, title, metadata):
    overlay = overlay_heatmap(image, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.text(0.5, 0.02, metadata, ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


class AttentionCapture:
    def __init__(self, model):
        self.attentions = []
        self.handles = []
        for block in model.blocks:
            handle = block.attn.attn_drop.register_forward_hook(self._hook)
            self.handles.append(handle)

    def _hook(self, _module, _inputs, output):
        self.attentions.append(output.detach().cpu())

    def clear(self):
        self.attentions.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attention_rollout(model, tensor, device, discard_ratio=0.0):
    capture = AttentionCapture(model)
    capture.clear()
    with torch.no_grad():
        _ = model(tensor.to(device))
    attentions = [attn.squeeze(0) for attn in capture.attentions]
    capture.close()
    if not attentions:
        raise RuntimeError("No attention maps were captured. The ViT attention implementation may have changed.")

    result = torch.eye(attentions[0].shape[-1])
    for attention in attentions:
        attention_heads = attention.mean(dim=0)
        if discard_ratio > 0:
            flat = attention_heads.view(-1)
            _, indices = flat.topk(int(flat.numel() * discard_ratio), largest=False)
            flat[indices] = 0
            attention_heads = flat.view_as(attention_heads)
        identity = torch.eye(attention_heads.shape[-1])
        attention_heads = attention_heads + identity
        attention_heads = attention_heads / attention_heads.sum(dim=-1, keepdim=True)
        result = torch.matmul(attention_heads, result)

    mask = result[0, 1:].numpy()
    grid = int(math.sqrt(mask.shape[0]))
    mask = mask.reshape(grid, grid)
    mask = torch.tensor(mask)[None, None]
    mask = F.interpolate(mask, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    return normalize_map(mask.squeeze().numpy())


def grid_masks(grid_size, rng, num_samples):
    num_features = grid_size * grid_size
    masks = rng.integers(0, 2, size=(num_samples, num_features))
    masks[0, :] = 1
    return masks


def apply_grid_mask(image_array, mask, grid_size, baseline):
    masked = image_array.copy()
    h, w, _ = image_array.shape
    cell_h = h / grid_size
    cell_w = w / grid_size
    for idx, keep in enumerate(mask):
        if keep:
            continue
        row = idx // grid_size
        col = idx % grid_size
        y0 = int(round(row * cell_h))
        y1 = int(round((row + 1) * cell_h))
        x0 = int(round(col * cell_w))
        x1 = int(round((col + 1) * cell_w))
        masked[y0:y1, x0:x1, :] = baseline
    return masked


def lime_explanation(model, image, transform, device, grid_size=7, num_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    image_array = np.asarray(image).astype(np.uint8)
    baseline = image_array.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    masks = grid_masks(grid_size, rng, num_samples)
    batch_tensors = []
    probs = []
    for mask in masks:
        masked = apply_grid_mask(image_array, mask, grid_size, baseline)
        batch_tensors.append(transform(Image.fromarray(masked)))
        if len(batch_tensors) == 32:
            batch = torch.stack(batch_tensors)
            probs.extend(predict_prob(model, batch, device))
            batch_tensors = []
    if batch_tensors:
        batch = torch.stack(batch_tensors)
        probs.extend(predict_prob(model, batch, device))

    distances = pairwise_distances(masks, np.ones((1, masks.shape[1])), metric="cosine").ravel()
    weights = np.sqrt(np.exp(-(distances**2) / 0.25))
    regressor = Ridge(alpha=1.0)
    regressor.fit(masks, np.asarray(probs), sample_weight=weights)
    coefs = regressor.coef_.reshape(grid_size, grid_size)
    coefs = np.maximum(coefs, 0)
    heatmap = torch.tensor(coefs, dtype=torch.float32)[None, None]
    heatmap = F.interpolate(heatmap, size=image.size[::-1], mode="bilinear", align_corners=False)
    return normalize_map(heatmap.squeeze().numpy())


def occlusion_sensitivity(model, image, transform, device, patch_size=32, stride=16):
    image_array = np.asarray(image).astype(np.uint8)
    baseline = image_array.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    base_tensor = transform(image).unsqueeze(0)
    base_prob = float(predict_prob(model, base_tensor, device)[0])
    h, w, _ = image_array.shape
    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    batch_tensors = []
    coords = []
    drops = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            masked = image_array.copy()
            masked[y : y + patch_size, x : x + patch_size, :] = baseline
            batch_tensors.append(transform(Image.fromarray(masked)))
            coords.append((x, y))
            if len(batch_tensors) == 32:
                probs = predict_prob(model, torch.stack(batch_tensors), device)
                drops.extend(base_prob - probs)
                batch_tensors = []
    if batch_tensors:
        probs = predict_prob(model, torch.stack(batch_tensors), device)
        drops.extend(base_prob - probs)

    for (x, y), drop in zip(coords, drops):
        score_map[y : y + patch_size, x : x + patch_size] += max(float(drop), 0.0)
        count_map[y : y + patch_size, x : x + patch_size] += 1.0
    score_map = score_map / np.maximum(count_map, 1.0)
    return normalize_map(score_map)


def attention_region_summary(heatmap, grid_size=3):
    h, w = heatmap.shape
    regions = []
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = int(round(row * h / grid_size))
            y1 = int(round((row + 1) * h / grid_size))
            x0 = int(round(col * w / grid_size))
            x1 = int(round((col + 1) * w / grid_size))
            score = float(heatmap[y0:y1, x0:x1].mean())
            vertical = ["upper", "central", "lower"][row]
            horizontal = ["left", "middle", "right"][col]
            regions.append({"region": f"{vertical}-{horizontal}", "score": score})
    return sorted(regions, key=lambda item: item["score"], reverse=True)


def high_score_area_ratio(heatmap, threshold=0.7):
    heatmap = normalize_map(heatmap)
    return float((heatmap >= threshold).mean())


def save_summary_grid(records, output_path):
    if not records:
        return
    rows = len(records)
    fig, axes = plt.subplots(rows, 5, figsize=(17, 3.4 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row_idx, record in enumerate(records):
        axes[row_idx, 0].imshow(record["image"])
        axes[row_idx, 0].set_title(f"{record['group']} original")
        axes[row_idx, 1].imshow(overlay_heatmap(record["image"], record["attention"]))
        axes[row_idx, 1].set_title("Attention Rollout")
        axes[row_idx, 2].imshow(overlay_heatmap(record["image"], record["lime"]))
        axes[row_idx, 2].set_title("Grid LIME")
        axes[row_idx, 3].imshow(overlay_heatmap(record["image"], record["occlusion"]))
        axes[row_idx, 3].set_title("Occlusion")
        axes[row_idx, 4].imshow(overlay_heatmap(record["image"], record["final_score"]))
        axes[row_idx, 4].set_title("Final Score")
        for col in range(5):
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set_ylabel(
            f"true={record['label']} pred={record['pred_label']}\nprob={record['prob']:.3f}",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    device = get_device()
    model, config = load_vit(args.config, args.checkpoint, device)
    image_size = int(config["dataset"]["image_size"])
    dataset_root = Path(config["dataset"]["root"])
    transform = build_eval_transform(image_size)

    output_dir = Path(args.output)
    attention_dir = output_dir / "attention_rollout"
    lime_dir = output_dir / "lime"
    occlusion_dir = output_dir / "occlusion"
    final_score_dir = output_dir / "final_score"
    summary_dir = output_dir / "summary"
    for directory in [attention_dir, lime_dir, occlusion_dir, final_score_dir, summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    selected = select_samples(args.predictions, args.threshold, args.samples_per_group, args.seed)
    selected.to_csv(output_dir / "selected_samples.csv", index=False)

    records = []
    explanation_rows = []
    for idx, row in selected.iterrows():
        image_id = row["id"]
        image = load_image(image_path_from_id(dataset_root, image_id), image_size)
        tensor = transform(image).unsqueeze(0)
        prob = float(predict_prob(model, tensor, device)[0])
        pred_label = int(prob >= args.threshold)
        label = int(row["label"])
        group = confusion_group(label, pred_label)
        prefix = f"{idx:02d}_{group}_{image_id[:10]}"
        metadata = f"true={label} predicted={pred_label} prob_cancer={prob:.4f} threshold={args.threshold:.2f}"

        attention_map = attention_rollout(model, tensor, device)
        lime_map = lime_explanation(
            model,
            image,
            transform,
            device,
            grid_size=args.grid_size,
            num_samples=args.lime_samples,
            seed=args.seed + idx,
        )
        occlusion_map = occlusion_sensitivity(
            model,
            image,
            transform,
            device,
            patch_size=args.occlusion_patch,
            stride=args.occlusion_stride,
        )
        final_score_map = weighted_final_score(
            attention_map,
            lime_map,
            occlusion_map,
            attention_weight=args.attention_weight,
            lime_weight=args.lime_weight,
            occlusion_weight=args.occlusion_weight,
        )

        save_heatmap_triplet(image, attention_map, attention_dir / f"{prefix}_attention_rollout.png", "Attention Rollout", metadata)
        save_heatmap_triplet(image, lime_map, lime_dir / f"{prefix}_lime.png", "Grid LIME", metadata)
        save_heatmap_triplet(image, occlusion_map, occlusion_dir / f"{prefix}_occlusion.png", "Occlusion Sensitivity", metadata)
        fusion_metadata = (
            f"{metadata} | final_score={args.attention_weight:.2f}*attention"
            f"+{args.lime_weight:.2f}*lime+{args.occlusion_weight:.2f}*occlusion"
        )
        save_heatmap_triplet(image, final_score_map, final_score_dir / f"{prefix}_final_score.png", "Weighted Final Score", fusion_metadata)

        top_regions = attention_region_summary(attention_map)[:3]
        top_final_regions = attention_region_summary(final_score_map)[:3]
        explanation_rows.append(
            {
                "id": image_id,
                "group": group,
                "true_label": label,
                "pred_label": pred_label,
                "prob_cancer": prob,
                "threshold": args.threshold,
                "attention_weight": args.attention_weight,
                "lime_weight": args.lime_weight,
                "occlusion_weight": args.occlusion_weight,
                "top_attention_regions": "; ".join(f"{r['region']}={r['score']:.3f}" for r in top_regions),
                "top_final_score_regions": "; ".join(f"{r['region']}={r['score']:.3f}" for r in top_final_regions),
                "final_score_area_ratio_0_7": high_score_area_ratio(final_score_map, threshold=0.7),
            }
        )
        records.append(
            {
                "image": image,
                "attention": attention_map,
                "lime": lime_map,
                "occlusion": occlusion_map,
                "final_score": final_score_map,
                "group": group,
                "label": label,
                "pred_label": pred_label,
                "prob": prob,
            }
        )

    pd.DataFrame(explanation_rows).to_csv(output_dir / "xai_explanations.csv", index=False)
    with open(output_dir / "xai_explanations.json", "w", encoding="utf-8") as f:
        json.dump(explanation_rows, f, indent=2)
    save_summary_grid(records, summary_dir / "xai_summary_grid.png")
    print(f"Saved XAI outputs to: {output_dir}")
    print(f"Selected samples: {output_dir / 'selected_samples.csv'}")
    print(f"Structured explanation data: {output_dir / 'xai_explanations.csv'}")
    print(f"Summary grid: {summary_dir / 'xai_summary_grid.png'}")


if __name__ == "__main__":
    main()
