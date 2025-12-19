"""
Helper script to download large models reliably with resume support.

This is more reliable than downloading during model loading, especially for
large models like Saka-14B (~28GB).

Usage:
    python -m knowldege_base.rag_staging.download_model Sakalti/Saka-14B
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def download_model_hf_cli(model_name: str, cache_dir: Optional[str] = None):
    """
    Download model using huggingface-cli (more reliable for large models).
    
    This uses the CLI which has better resume and retry logic.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    print("=" * 80)
    print(f"Downloading model: {model_name}")
    print("=" * 80)
    print("This may take a while for large models (~28GB for Saka-14B)...")
    print("The download will resume if interrupted.")
    print("=" * 80)
    
    try:
        # Use snapshot_download which has better resume support
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,  # Resume if interrupted
            local_files_only=False,
        )
        print(f"\n✓ Model downloaded successfully!")
        print(f"  Location: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: Use huggingface-cli command:")
        print(f"  huggingface-cli download {model_name}")
        raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model with resume support"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="HuggingFace model name (e.g., Sakalti/Saka-14B)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory (default: HuggingFace cache)",
    )
    
    args = parser.parse_args()
    
    download_model_hf_cli(args.model_name, args.cache_dir)


if __name__ == "__main__":
    main()

