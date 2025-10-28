#!/usr/bin/env python3
"""
Script to download and extract brain data to Railway volume.
Run this via: railway run python upload_brain_data.py <URL_TO_BRAIN_DATA>
"""
import sys
import urllib.request
import tarfile
from pathlib import Path

def download_and_extract(url, target_dir="/data/ghsl_brain"):
    """Download and extract brain data archive."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    archive_path = target_path / "ghsl_brain.tar.gz"

    print(f"Downloading brain data from {url}...")
    urllib.request.urlretrieve(url, archive_path)

    print(f"Extracting to {target_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_path.parent)

    print("Cleaning up archive...")
    archive_path.unlink()

    print("Brain data uploaded successfully!")
    print(f"Contents of {target_dir}:")
    for item in sorted(target_path.iterdir()):
        print(f"  - {item.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_brain_data.py <URL_TO_BRAIN_DATA>")
        sys.exit(1)

    download_and_extract(sys.argv[1])
