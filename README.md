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

## Expected Results

Based on paper (Table 4):

### MARBERT (fine-tuned) — Q-types:
- F1-Micro ≈ **0.85**
- F1-Weighted ≈ **0.85**
- Jaccard ≈ **0.80**

### MARBERT (fine-tuned) — A-types:
- F1-Micro ≈ **0.95**
- F1-Weighted ≈ **0.95**
- Jaccard ≈ **0.94**

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

## Citation

Please cite the original MentalQA paper when using this code.

