"""
Download a model from HuggingFace Hub
Usage:
    python scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import os
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HuggingFace token")
    args = parser.parse_args()

    print(f"📥 Downloading {args.model}...")
    path = snapshot_download(
        repo_id=args.model,
        token=args.token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print(f"✅ Model saved to: {path}")

if __name__ == "__main__":
    main()
