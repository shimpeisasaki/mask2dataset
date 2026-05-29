"""Download DeepLab Cityscapes weights helper.

Usage:
  PYTHONPATH=. python scripts/download_deeplab_weights.py

It will read `config/deeplab_weights.yaml` or the environment variable
`DEEPLAB_WEIGHTS_URL` and download the file into `models/deeplab_cityscapes_mobilenetv2.pth`.
"""
from pathlib import Path
import os
import urllib.request
import yaml

repo_root = Path(__file__).resolve().parents[1]
models_dir = repo_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
target = models_dir / "deeplab_cityscapes_mobilenetv2.pth"

url = os.environ.get("DEEPLAB_WEIGHTS_URL")
if not url:
    cfg = repo_root / "config" / "deeplab_weights.yaml"
    if cfg.exists():
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        url = raw.get("deeplab_cityscapes_mobilenetv2_url")

if not url:
    print("No URL configured. Set DEEPLAB_WEIGHTS_URL or config/deeplab_weights.yaml")
    raise SystemExit(1)

print(f"Downloading {url} -> {target}")

def _report(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 / total_size)
    print(f"downloaded {pct:.1f}%\r", end="")

urllib.request.urlretrieve(url, filename=str(target), reporthook=_report)
print("\nDone")
