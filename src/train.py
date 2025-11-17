"""Main training script"""
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import logging
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
import random

from src.config import load_config
from src.data_utils import load_jsonl
from src.models import MultiLabelClassifier
from src.metrics import compute_metrics_fn
from src.trainer import MultiLabelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def train(config_path: str):
    """Main training function"""
    # Load config
    config = load_config(config_path)
    task_name = config['task_name']
    model_name = config['model_name']
    num_labels = config['num_labels']
    seed = config['seed']
    batch_size = config['batch_size']
    # Handle learning rate (may be string like "2e-5" or float)
    if isinstance(config['lr'], str):
        try:
            lr = float(config['lr'])
        except ValueError:
            # Handle scientific notation strings
            lr = float(eval(config['lr']))
    else:
        lr = config['lr']
    epochs = config['epochs']
    weight_decay = float(config.get('weight_decay', 0.0)) if isinstance(config.get('weight_decay', 0.0), str) else config.get('weight_decay', 0.0)
    dropout = config.get('dropout', 0.1)
    early_stop_patience = config.get('early_stop_patience', 10)
    threshold = config.get('threshold', 0.5)
    max_length = config.get('max_length', 256)
    fp16 = config.get('fp16', True)
    
    # Set seed
    set_seed(seed)
    
    # Setup paths
    data_dir = Path("data") / task_name
    output_dir = Path("outputs") / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (use augmented if available, otherwise original)
    logger.info(f"Loading data for {task_name}...")
    train_file = data_dir / "train_augmented.jsonl"
    if not train_file.exists():
        train_file = data_dir / "train.jsonl"
        logger.info(f"Using original training data: {train_file}")
    else:
        logger.info(f"Using augmented training data: {train_file}")
    
    train_data = load_jsonl(str(train_file))
    val_data = load_jsonl(str(data_dir / "val.jsonl"))
    
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = MultiLabelDataset(train_data, tokenizer, max_length)
    val_dataset = MultiLabelDataset(val_data, tokenizer, max_length)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = MultiLabelClassifier(model_name, num_labels, dropout=dropout)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=seed,
        report_to="none",
    )
    
    # Create trainer
    trainer = MultiLabelTrainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn(threshold=threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save best model
    best_model_path = output_dir / "best"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    logger.info(f"Saved best model to {best_model_path}")
    
    # Final evaluation
    logger.info("Running final validation evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final validation metrics: {eval_results}")
    
    # Save metrics
    with open(output_dir / "val_metrics.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train multi-label classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

