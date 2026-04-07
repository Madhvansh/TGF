"""
Download pre-trained model checkpoints from GitHub Releases.

Usage:
    python scripts/download_models.py
"""

import urllib.request
import os
import sys

REPO = "Madhvansh/TGF"
RELEASE_TAG = "v0.1.0"
CHECKPOINT_DIR = "checkpoints"

MODELS = {
    "moment_model.pkl": f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/moment_model.pkl",
    "moment_tgf_model.pt": f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/moment_tgf_model.pt",
}


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading {os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Saved to {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  Failed: {e}")
        print(f"  You can manually download from: {url}")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for filename, url in MODELS.items():
        dest = os.path.join(CHECKPOINT_DIR, filename)
        if os.path.exists(dest):
            print(f"Skipping {filename} (already exists)")
            continue
        download_file(url, dest)

    print("\nDone. Run TGF with:")
    print(f"  python -m tgf_dosing.main --data data/Parameters_5K.csv --moment-checkpoint {CHECKPOINT_DIR}/moment_model.pkl")


if __name__ == "__main__":
    main()
