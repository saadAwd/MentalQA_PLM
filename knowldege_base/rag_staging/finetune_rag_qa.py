"""
Fine-tuning script for RAG QA models using PEFT (LoRA).

This allows fine-tuning quantized models efficiently using LoRA adapters.
The base quantized model stays frozen, only small adapter weights are trained.

Usage:
    python -m knowldege_base.rag_staging.finetune_rag_qa \
        --model-path "knowldege_base/data/models/Sakalti_Saka-14B_4bit" \
        --train-data "path/to/train.jsonl" \
        --output-dir "outputs/saka-14b-lora"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)

from .rag_qa import _get_local_model_path, _model_exists_locally


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def format_prompt(query: str, context: str, answer: str) -> str:
    """Format training example for causal LM."""
    prompt = f"السؤال: {query}\n\nالنص:\n{context}\n\nالإجابة: {answer}"
    return prompt


def prepare_dataset(
    train_data: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
) -> Dataset:
    """Prepare dataset for training."""
    texts = []
    for item in train_data:
        query = item.get("question", item.get("query", ""))
        context = item.get("context", "")
        answer = item.get("answer", item.get("response", ""))
        prompt = format_prompt(query, context, answer)
        texts.append(prompt)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_function, batched=True)
    return tokenized


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune RAG QA model with LoRA")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local quantized model (e.g., knowldege_base/data/models/Sakalti_Saka-14B_4bit)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training JSONL file with 'question', 'context', 'answer' fields",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for LoRA adapters",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size (keep small for large models)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (higher = more parameters, default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor, default: 32)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Fine-tuning RAG QA Model with LoRA")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Training data: {args.train_data}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization (if it was quantized)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Setup LoRA
    print("Setting up LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Common for Qwen-based models
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load training data
    print("Loading training data...")
    train_data = load_training_data(args.train_data)
    print(f"Training examples: {len(train_data)}")

    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(train_data, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✓ Fine-tuning complete!")


if __name__ == "__main__":
    main()

