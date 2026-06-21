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

Train CNN-GNN:

```powershell
python scripts/train.py --config configs/cnn_gnn_resnet50.yaml
```

Evaluate a checkpoint:

```powershell
python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint outputs/resnet50/best.pt
```

Generate ViT XAI visualizations:

```powershell
python scripts/xai_vit.py --config configs/vit.yaml --checkpoint outputs/vit_small/best.pt --predictions outputs/vit_small/test_predictions.csv --output outputs/vit_small/xai --threshold 0.36
```

Generate human-readable LLM explanations from the XAI output:

```powershell
python scripts/llm_explain_xai.py --provider mock --input outputs/vit_small/xai/xai_explanations.json --output outputs/vit_small/xai/llm_report_mock
```

Optional free-tier API providers can be used by setting one API key and changing `--provider`:

```powershell
$env:OPENROUTER_API_KEY="your_key"
python scripts/llm_explain_xai.py --provider openrouter --input outputs/vit_small/xai/xai_explanations.json --output outputs/vit_small/xai/llm_report_openrouter

$env:GROQ_API_KEY="your_key"
python scripts/llm_explain_xai.py --provider groq --input outputs/vit_small/xai/xai_explanations.json --output outputs/vit_small/xai/llm_report_groq

$env:GEMINI_API_KEY="your_key"
python scripts/llm_explain_xai.py --provider gemini --input outputs/vit_small/xai/xai_explanations.json --output outputs/vit_small/xai/llm_report_gemini

$env:HF_TOKEN="your_key"
python scripts/llm_explain_xai.py --provider huggingface --input outputs/vit_small/xai/xai_explanations.json --output outputs/vit_small/xai/llm_report_huggingface
```

## Recommended Experiment Order

1. `resnet50`
2. `vit_small_patch16_224`
3. `cnn_gnn_resnet50`

## Outputs

Each training run writes:

```text
E:\cancer_cell_classification\model_outputs\<run_name>\
  best.pt
  last.pt
  metrics.json
  history.csv
  val_predictions.csv
  loss_curve.png
  metrics_curve.png
  val_confusion_matrix.png
  val_roc_curve.png
  test_confusion_matrix.png
  test_roc_curve.png
```
