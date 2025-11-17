#!/usr/bin/env python
"""Augment multi-label training data"""
import sys
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import load_jsonl, save_jsonl
from src.augmentation import balance_multilabel_dataset_selective, ArabicDataAugmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Augment multi-label training data")
    parser.add_argument("--task", type=str, required=True, choices=["qtypes", "atypes"], 
                        help="Task: qtypes or atypes")
    parser.add_argument("--input", type=str, default=None, 
                        help="Input JSONL file (default: data/{task}/train.jsonl)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (default: data/{task}/train_augmented.jsonl)")
    parser.add_argument("--target-samples", type=int, default=None,
                        help="Target samples per class for balancing (default: auto = median of all classes)")
    parser.add_argument("--use-paraphrasing", action="store_true", default=True,
                        help="Use paraphrasing to generate multiple variations (default: True)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set default paths
    if args.input is None:
        args.input = f"data/{args.task}/train.jsonl"
    if args.output is None:
        args.output = f"data/{args.task}/train_augmented.jsonl"
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    data = load_jsonl(args.input)
    logger.info(f"Loaded {len(data)} samples")
    
    # Create augmenter with back-translation
    logger.info("Initializing back-translation augmenter...")
    logger.info("This will download translation models on first use (may take a few minutes)...")
    augmenter = ArabicDataAugmenter(seed=args.seed)
    
    # Selectively augment only underrepresented classes to achieve balanced representation
    if args.target_samples:
        logger.info(f"Target: {args.target_samples} samples per class")
    else:
        logger.info("Target: Auto-determined (median of all classes) for balanced representation")
    logger.info("Only underrepresented classes will be augmented")
    logger.info("Well-represented classes will be kept as-is (no augmentation)")
    logger.info("Using back-translation + paraphrasing for multiple variations")
    logger.info("Note: This takes longer but produces higher quality augmentations")
    augmented_data = balance_multilabel_dataset_selective(
        data,
        target_samples_per_class=args.target_samples,
        augmenter=augmenter,
        max_augmentations_per_sample=20,  # Increased to allow reaching targets
        use_paraphrasing=args.use_paraphrasing
    )
    
    # Save augmented data
    output_path = Path(args.output)
    save_jsonl(augmented_data, output_path)
    logger.info(f"Saved {len(augmented_data)} augmented samples to {output_path}")
    
    # Print statistics
    from collections import Counter
    original_labels = Counter()
    augmented_labels = Counter()
    
    for sample in data:
        label_indices = [i for i, val in enumerate(sample['labels']) if val == 1]
        for idx in label_indices:
            original_labels[idx] += 1
    
    for sample in augmented_data:
        label_indices = [i for i, val in enumerate(sample['labels']) if val == 1]
        for idx in label_indices:
            augmented_labels[idx] += 1
    
    logger.info("\nLabel distribution:")
    labels = ["diagnosis", "treatment", "anatomy", "epidemiology", "healthy_lifestyle", "provider_choice", "other"] if args.task == "qtypes" else ["information", "direct_guidance", "emotional_support"]
    for i, label in enumerate(labels):
        orig = original_labels.get(i, 0)
        aug = augmented_labels.get(i, 0)
        logger.info(f"  {label}: {orig} -> {aug} (+{aug - orig})")


if __name__ == "__main__":
    main()

