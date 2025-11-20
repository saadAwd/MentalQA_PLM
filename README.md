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


## Experimental Results & Data Augmentation Strategies

### Q-types Classification Results

We tested multiple data augmentation strategies to address class imbalance. Results on validation set:

| Approach | Samples | F1-Micro | F1-Weighted | Jaccard | Loss | Rank |
|----------|---------|----------|-------------|---------|------|------|
| **Selective Back-Translation** ⭐ | 229 | **63.4%** 🥇 | **59.1%** 🥇 | 52.6% | 0.430 | **1st** |
| Simple Augmentation | 398 | 62.2% | 56.5% | 51.9% | 0.443 | 2nd |
| Qwen Augmented | 329 | 62.1% | 57.0% | 51.8% | 0.437 | 3rd |
| No Augmentation | 210 | 61.7% | 58.1% | **52.9%** 🥇 | **0.423** 🥇 | 4th |
| Selective + Paraphrasing (BT) | 311 | 60.4% | 57.3% | 51.2% | 0.436 | 5th |
| Full Back-Translation | 423 | 59.6% | 56.8% | 50.5% | 0.534 | 6th |

### Data Augmentation Strategies

1. **No Augmentation (Baseline)**
   - Samples: 210
   - Approach: Use original training data as-is
   - Result: Strong baseline (61.7% F1-Micro)

2. **Simple Augmentation**
   - Samples: 398 (+188, +89%)
   - Methods: Random deletion, insertion, swap, noise
   - Result: Moderate improvement (62.2% F1-Micro)

3. **Full Back-Translation**
   - Samples: 423 (+213, +101%)
   - Method: Back-translate all samples (Arabic→English→Arabic)
   - Result: Underperformed (59.6% F1-Micro) - too much augmentation

4. **Selective Back-Translation** ⭐ **BEST**
   - Samples: 229 (+19, +9%)
   - Method: Only augment underrepresented classes (< median)
   - Target: Reach median of well-represented classes
   - Result: **Best overall performance** (63.4% F1-Micro, 59.1% F1-Weighted)

5. **Selective + Qwen Paraphrasing**
   - Samples: 329 (+119, +57%)
   - Method: Qwen model for prompt-based paraphrasing
   - Result: Good balance but underperformed vs Selective BT (62.1% F1-Micro)

6. **Selective + Paraphrasing (BT)**
   - Samples: 311 (+101, +48%)
   - Method: Combination of selective back-translation and paraphrasing
   - Result: Lower performance (60.4% F1-Micro)

### Key Findings

1. **Selective Back-Translation is Optimal**: Best F1-Micro (63.4%) and F1-Weighted (59.1%) with minimal dataset increase (+9%)

2. **More Augmentation ≠ Better Performance**: Full back-translation (423 samples) underperformed. Optimal point: 229 samples (selective)

3. **Class Balance Matters**: Qwen achieved perfect balance but underperformed. Quality > Quantity for augmentation

4. **Baseline is Strong**: No augmentation achieved 61.7% F1-Micro, showing dataset quality is good

### Comparison with Paper Results

| Metric | Our Best | Paper (MARBERT) | Gap |
|--------|----------|-----------------|-----|
| F1-Micro | 63.4% | ~85% | -21.6% |
| F1-Weighted | 59.1% | ~85% | -25.9% |
| Jaccard | 52.6% | ~80% | -27.4% |

**Reasons for Gap**:
- Smaller dataset: 351 samples vs paper's larger set
- Class imbalance: Severe in our dataset
- Different splits: Paper may use different train/test splits
- Hyperparameter differences: May have different optimal settings

### A-types Classification Results

Training for A-types (answer types) classification is currently in progress. Results will be updated upon completion.

**Expected Results** (based on paper):
- F1-Micro: ~95%
- F1-Weighted: ~95%
- Jaccard: ~94%

**Note**: The same augmentation strategies tested for Q-types (No Augmentation, Simple Augmentation, Full Back-Translation, Selective Back-Translation, Qwen Augmented, etc.) will be evaluated for A-types classification. Results will be added to this section once training completes.

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
