"""Data loading and preprocessing for MentalQA"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


# Label mappings for Q-types (7 labels)
QTYPES_LABELS = ["diagnosis", "treatment", "anatomy", "epidemiology", "healthy_lifestyle", "provider_choice", "other"]

# Label mappings for A-types (3 labels)
ATYPES_LABELS = ["information", "direct_guidance", "emotional_support"]


def parse_multilabel_qt(row: Dict) -> List[int]:
    """Parse Q-types multi-label from TSV row"""
    labels = []
    qt_str = str(row.get('final_QT', ''))
    
    # Map according to paper: A=treatment, B=diagnosis, C=anatomy, D=healthy_lifestyle, E=epidemiology, F=provider_choice, Z=other
    qt_map = {
        'A': 'treatment',
        'B': 'diagnosis',
        'C': 'anatomy',
        'D': 'healthy_lifestyle',
        'E': 'epidemiology',
        'F': 'provider_choice',
        'Z': 'other'
    }
    
    try:
        import ast
        # Handle string representation of list
        if qt_str.startswith('[') and qt_str.endswith(']'):
            qt_list = ast.literal_eval(qt_str)
        elif qt_str:
            # Try splitting by comma if not a list
            qt_list = [x.strip().strip("'\"") for x in qt_str.split(',')]
        else:
            qt_list = []
        
        for label_code in qt_list:
            label_code = str(label_code).strip().strip("'\"")
            if label_code in qt_map:
                label_name = qt_map[label_code]
                if label_name in QTYPES_LABELS:
                    labels.append(QTYPES_LABELS.index(label_name))
    except Exception as e:
        logger.debug(f"Error parsing QT labels '{qt_str}': {e}")
        pass
    
    return sorted(set(labels))  # Remove duplicates and sort


def parse_multilabel_as(row: Dict) -> List[int]:
    """Parse A-types multi-label from TSV row"""
    labels = []
    as_str = str(row.get('final_AS', ''))
    
    # Map: 1=information, 2=direct_guidance, 3=emotional_support
    as_map = {
        '1': 'information',
        '2': 'direct_guidance',
        '3': 'emotional_support'
    }
    
    try:
        import ast
        # Handle string representation of list
        if as_str.startswith('[') and as_str.endswith(']'):
            as_list = ast.literal_eval(as_str)
        elif as_str:
            # Try splitting by comma if not a list
            as_list = [x.strip().strip("'\"") for x in as_str.split(',')]
        else:
            as_list = []
        
        for label_code in as_list:
            label_code = str(label_code).strip().strip("'\"")
            if label_code in as_map:
                label_name = as_map[label_code]
                if label_name in ATYPES_LABELS:
                    labels.append(ATYPES_LABELS.index(label_name))
    except Exception as e:
        logger.debug(f"Error parsing AS labels '{as_str}': {e}")
        pass
    
    return sorted(set(labels))  # Remove duplicates and sort


def create_multilabel_vector(label_indices: List[int], num_labels: int) -> List[int]:
    """Create multi-hot vector from label indices"""
    vector = [0] * num_labels
    for idx in label_indices:
        if 0 <= idx < num_labels:
            vector[idx] = 1
    return vector


def load_mentalqa_tsv(tsv_path: str, task: str) -> List[Dict]:
    """Load MentalQA TSV and convert to multi-label format"""
    df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
    
    data = []
    if task == "qtypes":
        labels = QTYPES_LABELS
        parse_fn = parse_multilabel_qt
        text_col = 'question'
    elif task == "atypes":
        labels = ATYPES_LABELS
        parse_fn = parse_multilabel_as
        text_col = 'answer'  # A-types use answer text
    else:
        raise ValueError(f"Unknown task: {task}")
    
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ''))
        if not text or text == 'nan':
            continue
        
        label_indices = parse_fn(row)
        if not label_indices:  # Skip samples with no labels
            continue
        
        label_vector = create_multilabel_vector(label_indices, len(labels))
        
        data.append({
            'id': idx,
            'text': text,
            'labels': label_vector,
            'label_indices': label_indices
        })
    
    logger.info(f"Loaded {len(data)} samples for {task}")
    return data


def split_data(data: List[Dict], train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train/val/test (60/20/20)"""
    # First split: train (60%) and temp (40%)
    train, temp = train_test_split(data, test_size=1 - train_ratio, random_state=seed)
    
    # Second split: val (20%) and test (20%) from temp
    val_size = val_ratio / (1 - train_ratio)  # 0.2 / 0.4 = 0.5
    val, test = train_test_split(temp, test_size=1 - val_size, random_state=seed)
    
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_jsonl(data: List[Dict], output_path: Path):
    """Save data to JSONL format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL format"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def prepare_data(tsv_path: str, task: str, output_dir: str = "data", seed: int = 42):
    """Prepare and save train/val/test splits"""
    # Load data
    data = load_mentalqa_tsv(tsv_path, task)
    
    if not data:
        raise ValueError(f"No valid data loaded for task {task}")
    
    # Split
    train, val, test = split_data(data, train_ratio=0.6, val_ratio=0.2, seed=seed)
    
    # Save
    output_path = Path(output_dir) / task
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_jsonl(train, output_path / "train.jsonl")
    save_jsonl(val, output_path / "val.jsonl")
    save_jsonl(test, output_path / "test.jsonl")
    
    # Save label list
    labels = QTYPES_LABELS if task == "qtypes" else ATYPES_LABELS
    with open(output_path / "label_list.json", 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved data to {output_path}")
    logger.info(f"  Train: {len(train)} samples")
    logger.info(f"  Val: {len(val)} samples")
    logger.info(f"  Test: {len(test)} samples")
    
    return train, val, test


def main():
    """CLI entry point for data preparation"""
    import argparse
    parser = argparse.ArgumentParser(description="Prepare MentalQA data for multi-label classification")
    parser.add_argument("--tsv", type=str, required=True, help="Path to Train_Dev.tsv")
    parser.add_argument("--task", type=str, required=True, choices=["qtypes", "atypes"], help="Task: qtypes or atypes")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    prepare_data(args.tsv, args.task, args.output_dir, args.seed)


if __name__ == "__main__":
    main()

