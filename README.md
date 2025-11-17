# MentalQA Multi-Label Classification

Minimal, reproducible PyTorch + HuggingFace project for fine-tuning Arabic PLMs on the MentalQA dataset with **multi-label classification**.

## Tasks

1. **Q-types** (7 labels): `["diagnosis","treatment","anatomy","epidemiology","healthy_lifestyle","provider_choice","other"]`
2. **A-types** (3 labels): `["information","direct_guidance","emotional_support"]`

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Data

**Linux/Mac:**
```bash
bash scripts/download_data.sh
```

**Windows:**
```bash
python scripts/prepare_data.py
```

This will process the MentalQA dataset from `../Train_Dev.tsv`, creating train/val/test splits (60/20/20).

## Training

### Train Q-types Model

```bash
bash scripts/run_qtypes.sh
```

Or directly:
```bash
python -m src.train --config configs/marbert_qtypes.yaml
```

### Train A-types Model

```bash
bash scripts/run_atypes.sh
```

Or directly:
```bash
python -m src.train --config configs/marbert_atypes.yaml
```

## Evaluation

```bash
python -m src.evaluate --config configs/marbert_qtypes.yaml --checkpoint outputs/qtypes/best --test data/qtypes_test.jsonl
```

## Achived Results by runinig this configuration

### Q-types Classification Results

| Approach | Samples | F1-Micro | F1-Weighted | Jaccard | Loss | Rank |
|----------|---------|----------|-------------|---------|------|------|
| **Selective Back-Translation** | 229 | **63.4%**  | **59.1%**  | 52.6% | 0.430 | **1st** |
| **Simple Augmentation** | 398 | 62.2% | 56.5% | 51.9% | 0.443 | 2nd |
| **Qwen Augmented** | 329 | 62.1% | 57.0% | 51.8% | 0.437 | 3rd |
| **No Augmentation** | 210 | 61.7% | 58.1% | **52.9%**  | **0.423**  | 4th |
| **Selective + Paraphrasing (BT)** | 311 | 60.4% | 57.3% | 51.2% | 0.436 | 5th |
| **Full Back-Translation** | 423 | 59.6% | 56.8% | 50.5% | 0.534 | 6th |


## Configuration

Hyperparameters (from Table 3):
- Hidden dim: 768
- Batch size: 8
- Dropout: 0.1
- Epochs: 15
- Learning rate: 2e-5
- Optimizer: Adam
- Early stopping patience: 10
- Threshold: 0.5 (for multi-label binarization)

## Model Architecture

- **Head:** Multi-label (sigmoid activation)
- **Loss:** BCEWithLogitsLoss
- **Threshold:** 0.5 for prediction binarization

## Data Split

- Train: 60%
- Validation: 20%
- Test: 20%

Fixed random seed: 42

## Refrence
THis work has been done in an effort to reproduce the high score achived in this paper 
Alhuzali & Alasmari, 2025 — Pre-Trained LMs for Arabic Mental-Health Q&A Classification (MDPI).
https://www.mdpi.com/2227-9032/13/9/985
