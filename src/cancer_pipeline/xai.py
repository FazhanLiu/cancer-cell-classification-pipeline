"""Post-hoc cancer-score explanations; attention rollout is class-independent."""
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances


@torch.no_grad()
def predict_prob(model, tensor, device):
    return model(tensor.to(device)).float().softmax(dim=1)[:, 1].cpu().numpy()


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Heatmaps must be finite 2D arrays.")
    v_min = float(values.min())
    v_max = float(values.max())
    if math.isclose(v_min, v_max):
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def weighted_final_score(attention_map, lime_map, occlusion_map, attention_weight=0.5, lime_weight=0.3, occlusion_weight=0.2):
    total_weight = attention_weight + lime_weight + occlusion_weight
    if not np.isfinite([attention_weight, lime_weight, occlusion_weight]).all() or min(attention_weight, lime_weight, occlusion_weight) < 0 or not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("At least one XAI fusion weight must be positive.")
    if not (np.shape(attention_map) == np.shape(lime_map) == np.shape(occlusion_map)):
        raise ValueError("All fusion maps must have identical shapes.")
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


def attention_rollout(model, tensor, device, discard_ratio=0.0, return_raw=False):
    capture = AttentionCapture(model)
    capture.clear()
    try:
        with torch.no_grad():
            _ = model(tensor.to(device))
        attentions = [attn.squeeze(0) for attn in capture.attentions]
    finally:
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
    raw = mask.squeeze().numpy()
    return raw if return_raw else normalize_map(raw)


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


def lime_explanation(model, image, transform, device, grid_size=7, num_samples=200, seed=42, return_raw=False, return_coefficients=False):
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
    heatmap = torch.tensor(coefs, dtype=torch.float32)[None, None]
    heatmap = F.interpolate(heatmap, size=image.size[::-1], mode="bilinear", align_corners=False)
    raw = heatmap.squeeze().numpy()
    if return_coefficients:
        return raw, coefs
    return raw if return_raw else normalize_map(np.maximum(raw, 0))


def occlusion_sensitivity(model, image, transform, device, patch_size=32, stride=16, return_raw=False):
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
    if not 1 <= stride <= patch_size <= min(h, w):
        raise ValueError("Require 1 <= stride <= patch_size <= image size.")
    ys = sorted(set(range(0, h - patch_size + 1, stride)) | {h - patch_size})
    xs = sorted(set(range(0, w - patch_size + 1, stride)) | {w - patch_size})
    for y in ys:
        for x in xs:
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
        score_map[y : y + patch_size, x : x + patch_size] += float(drop)
        count_map[y : y + patch_size, x : x + patch_size] += 1.0
    score_map = score_map / np.maximum(count_map, 1.0)
    return score_map if return_raw else normalize_map(np.maximum(score_map, 0))


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
    return sorted([r for r in regions if r["score"] > 0], key=lambda item: item["score"], reverse=True)


def high_score_area_ratio(heatmap, threshold=0.7):
    heatmap = normalize_map(heatmap)
    return float((heatmap >= threshold).mean())


def explain(model, image, transform, device, seed=42, grid_size=7, lime_samples=200,
            occlusion_patch=32, occlusion_stride=16, weights=(0.5, 0.3, 0.2)):
    if grid_size < 1 or grid_size > min(image.size) or lime_samples < 2:
        raise ValueError('Invalid explanation grid/sample count.')
    if not hasattr(model, 'blocks') or getattr(model, 'num_prefix_tokens', 1) != 1:
        raise ValueError('Attention rollout requires a compatible ViT with CLS token.')
    weighted_final_score(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), *weights)
    for block in model.blocks:
        if hasattr(block.attn, 'fused_attn'):
            block.attn.fused_attn = False
    tensor = transform(image).unsqueeze(0)
    maps, timings = {}, {}
    for name, operation in [
        ('attention', lambda: attention_rollout(model, tensor, device, return_raw=True)),
        ('lime', lambda: lime_explanation(model, image, transform, device, grid_size, lime_samples, seed, return_raw=True, return_coefficients=True)),
        ('occlusion', lambda: occlusion_sensitivity(model, image, transform, device, occlusion_patch, occlusion_stride, return_raw=True))]:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = operation()
        if name == 'lime':
            raw, coefficients = result
            maps['lime_coefficients'] = coefficients
        else:
            raw = result
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings[name] = time.perf_counter() - start
        maps[name + '_raw'] = raw
        maps[name] = normalize_map(raw if name == 'attention' else np.maximum(raw, 0))
    fusion_start = time.perf_counter()
    maps['fusion'] = weighted_final_score(maps['attention'], maps['lime'], maps['occlusion'], *weights)
    timings['fusion_overhead'] = time.perf_counter() - fusion_start
    timings['fusion_total'] = sum(timings[k] for k in ('attention', 'lime', 'occlusion')) + timings['fusion_overhead']
    equal_start = time.perf_counter()
    maps['equal_fusion'] = weighted_final_score(maps['attention'], maps['lime'], maps['occlusion'], 1, 1, 1)
    timings['equal_fusion_total'] = sum(timings[k] for k in ('attention', 'lime', 'occlusion')) + time.perf_counter() - equal_start
    maps['disagreement'] = np.std(np.stack([maps[k] for k in ('attention', 'lime', 'occlusion')]), axis=0)
    return maps, timings


def deletion_curves(model, image, transform, device, maps, seed=42, grid_size=7,
                    fractions=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)):
    """Identical grid/baseline for every method; not a localization metric."""
    array = np.asarray(image)
    baseline = array.reshape(-1, 3).mean(0).astype(np.uint8)
    names = ('attention', 'lime', 'occlusion', 'equal_fusion', 'fusion', 'random')
    curves = {}
    rng = np.random.default_rng(seed)
    for name in names:
        if name == 'random':
            ranking = rng.permutation(grid_size ** 2)
        else:
            h, w = maps[name].shape
            scores = []
            for row in range(grid_size):
                for col in range(grid_size):
                    y0, y1 = round(row*h/grid_size), round((row+1)*h/grid_size)
                    x0, x1 = round(col*w/grid_size), round((col+1)*w/grid_size)
                    scores.append(maps[name][y0:y1, x0:x1].mean())
            ranking = np.argsort(-np.asarray(scores), kind='stable')
        tensors, actual_fractions = [], []
        for fraction in fractions:
            mask = np.ones(grid_size ** 2, dtype=int)
            count = round(fraction * len(mask))
            mask[ranking[:count]] = 0
            masked = apply_grid_mask(array, mask, grid_size, baseline)
            tensors.append(transform(Image.fromarray(masked)))
            actual_fractions.append(count / len(mask))
        probs = predict_prob(model, torch.stack(tensors), device)
        curves[name] = {'fractions': actual_fractions, 'probabilities': probs.tolist(),
                        'deletion_auc': float(np.trapz(probs, actual_fractions)),
                        'score_drop_at_half': float(probs[0] - probs[list(fractions).index(0.5)])}
    return curves
