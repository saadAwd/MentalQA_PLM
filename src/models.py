"""Multi-label classification models"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MultiLabelClassifier(nn.Module):
    """Multi-label sequence classifier with configurable dropout"""
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load config and model
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        config.problem_type = "multi_label_classification"
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        
        # Set dropout
        if hasattr(self.model.config, 'hidden_dropout_prob'):
            self.model.config.hidden_dropout_prob = dropout
        if hasattr(self.model.config, 'attention_probs_dropout_prob'):
            self.model.config.attention_probs_dropout_prob = dropout
        
        # Apply dropout to classifier if it exists
        if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'dropout'):
            self.model.classifier.dropout = nn.Dropout(dropout)
        elif hasattr(self.model, 'classifier'):
            # Wrap classifier with dropout
            original_classifier = self.model.classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                original_classifier
            )
        
        logger.info(f"Initialized {model_name} with {num_labels} labels, dropout={dropout}")
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_name: str, checkpoint_path: str, num_labels: int, dropout: float = 0.1):
        """Load model from checkpoint"""
        model = cls(model_name, num_labels, dropout)
        # Load weights
        state_dict = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location="cpu")
        model.model.load_state_dict(state_dict, strict=False)
        return model

