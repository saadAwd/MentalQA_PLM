#!/usr/bin/env python
"""Windows-compatible data preparation script"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import prepare_data

if __name__ == "__main__":
    # Check if Train_Dev.tsv exists in parent directory
    tsv_file = Path(__file__).parent.parent.parent / "Train_Dev.tsv"
    
    if not tsv_file.exists():
        print(f"Error: Train_Dev.tsv not found at {tsv_file}")
        print("Please place Train_Dev.tsv in the parent directory.")
        sys.exit(1)
    
    print(f"Found Train_Dev.tsv at {tsv_file}")
    
    # Prepare Q-types data
    print("\nPreparing Q-types data...")
    prepare_data(str(tsv_file), "qtypes", output_dir="data", seed=42)
    
    # Prepare A-types data
    print("\nPreparing A-types data...")
    prepare_data(str(tsv_file), "atypes", output_dir="data", seed=42)
    
    print("\nData preparation complete!")

