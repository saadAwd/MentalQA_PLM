"""Training utilities for multi-label classification"""
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MultiLabelTrainer(Trainer):
    """Custom trainer for multi-label classification with BCE loss"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # BCEWithLogitsLoss is automatically used for problem_type="multi_label_classification"
        # But we can explicitly set it if needed
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = nn.BCEWithLogitsLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute BCE loss for multi-label classification
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # BCEWithLogitsLoss expects float labels for multi-label
        loss = self.loss_fn(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss

