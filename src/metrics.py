"""Multi-label classification metrics"""
import torch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def compute_multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute multi-label metrics: F1-Micro, F1-Weighted, Jaccard (samples)
    
    Args:
        logits: Raw logits from model (n_samples, n_labels)
        labels: True multi-hot labels (n_samples, n_labels)
        threshold: Threshold for binarization (default 0.5)
    
    Returns:
        Dictionary with f1_micro, f1_weighted, jaccard
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Binarize with threshold
    preds = (probs >= threshold).astype(int)
    
    # Ensure labels are binary
    labels = labels.astype(int)
    
    # Compute F1-Micro (treats each label prediction independently)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    
    # Compute F1-Weighted (weighted by support per label)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Compute Jaccard (IoU) - average over samples
    # Jaccard for multi-label: intersection over union per sample, then average
    jaccard = jaccard_score(labels, preds, average='samples', zero_division=0)
    
    metrics = {
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_weighted),
        'jaccard': float(jaccard),
    }
    
    return metrics


def compute_metrics_fn(threshold: float = 0.5):
    """Create metrics function for HuggingFace Trainer"""
    def metrics_fn(eval_pred):
        predictions, labels = eval_pred
        
        # predictions are logits
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        return compute_multilabel_metrics(predictions, labels, threshold=threshold)
    
    return metrics_fn

