# Cancer Cell Classification Pipeline

PyTorch training pipeline for the Kaggle Histopathologic Cancer Detection dataset.

The default configuration is tuned for an NVIDIA RTX 5060 Laptop GPU with about 8 GB VRAM:

- mixed precision training enabled
- ResNet50 and ViT-Small only
- gradient accumulation to keep effective batch size reasonable
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

Train ResNet50:

```powershell
python scripts/train.py --config configs/resnet50.yaml
```

Train ViT-Small:

```powershell
python scripts/train.py --config configs/vit.yaml
```

Evaluate a checkpoint:

```powershell
python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint outputs/resnet50/best.pt
```

## Recommended Experiment Order

1. `resnet50`
2. `vit_small_patch16_224`

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
