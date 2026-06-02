# Cancer Cell Classification Pipeline

PyTorch training pipeline for the Kaggle Histopathologic Cancer Detection dataset.

The default configuration is tuned for an NVIDIA RTX 5060 Laptop GPU with about 8 GB VRAM:

- mixed precision training enabled
- small and medium models first
- batch size 32 for CNNs, 16 for transformer-style models
- gradient accumulation supported
- stratified train/validation split

## Dataset

Expected dataset layout:

```text
E:\cancer_cell_classification\histopathologic-cancer-detection
  train/
  test/
  train_labels.csv
  sample_submission.csv
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA version from:

```text
https://pytorch.org/get-started/locally/
```

## Quick Start

Run a lightweight baseline:

```powershell
python scripts/train.py --config configs/resnet18.yaml
```

Train EfficientNet-B0:

```powershell
python scripts/train.py --config configs/efficientnet_b0.yaml
```

Train Swin-Tiny:

```powershell
python scripts/train.py --config configs/swin_tiny.yaml
```

Evaluate a checkpoint:

```powershell
python scripts/evaluate.py --config configs/efficientnet_b0.yaml --checkpoint outputs/efficientnet_b0/best.pt
```

Run soft-voting ensemble:

```powershell
python scripts/ensemble.py --config configs/ensemble.yaml
```

## Recommended Experiment Order

1. `resnet18`
2. `efficientnet_b0`
3. `resnet50`
4. `swin_tiny`
5. `vit_tiny_patch16_224`
6. `hybrid_resnet18_transformer`
7. best CNN + best transformer soft-voting ensemble

## Outputs

Each training run writes:

```text
outputs/<run_name>/
  best.pt
  last.pt
  metrics.json
  history.csv
  val_predictions.csv
```
