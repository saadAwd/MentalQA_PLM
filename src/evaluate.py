"""Evaluation script for test set"""
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import logging
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import load_config
from src.data_utils import load_jsonl
from src.models import MultiLabelClassifier
from src.metrics import compute_multilabel_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiLabelDataset(torch.utils.data.Dataset):
    """Dataset for multi-label classification"""
    
    def __init__(self, data: list, tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])
        labels = torch.tensor(item['labels'], dtype=torch.float32)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }


def evaluate(config_path: str, checkpoint_path: str, test_file: str):
    """Evaluate model on test set"""
    # Load config
    config = load_config(config_path)
    model_name = config['model_name']
    num_labels = config['num_labels']
    dropout = config.get('dropout', 0.1)
    threshold = config.get('threshold', 0.5)
    max_length = config.get('max_length', 256)
    
    # Load test data
    logger.info(f"Loading test data from {test_file}")
    test_data = load_jsonl(test_file)
    logger.info(f"Test samples: {len(test_data)}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Create dataset
    test_dataset = MultiLabelDataset(test_data, tokenizer, max_length)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Run inference
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics
    metrics = compute_multilabel_metrics(logits, labels, threshold=threshold)
    
    logger.info("=" * 60)
    logger.info("TEST SET METRICS")
    logger.info("=" * 60)
    logger.info(f"F1-Micro: {metrics['f1_micro']:.4f}")
    logger.info(f"F1-Weighted: {metrics['f1_weighted']:.4f}")
    logger.info(f"Jaccard: {metrics['jaccard']:.4f}")
    logger.info("=" * 60)
    
    # Save metrics
    output_path = Path(checkpoint_path).parent / "test_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {output_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-label classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test", type=str, required=True, help="Path to test JSONL file")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.test)


if __name__ == "__main__":
    main()

