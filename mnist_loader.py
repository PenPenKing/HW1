#!/usr/bin/env python3
"""Download MNIST from HuggingFace (ylecun/mnist) and export it to disk.

Outputs (under --out, default ./data):
    hf/*.parquet                    raw files as downloaded
    mnist/train-images-idx3-ubyte   original IDX format, 60000x28x28 uint8
    mnist/train-labels-idx1-ubyte   original IDX format, 60000 uint8
    mnist/t10k-images-idx3-ubyte    original IDX format, 10000x28x28 uint8
    mnist/t10k-labels-idx1-ubyte    original IDX format, 10000 uint8
    mnist/*.npy                     same data as uint8 arrays (fast to load)

Downloads go through HF_ENDPOINT when it is set (hf-mirror.com here), because
huggingface_hub's HEAD probe does not survive the mirror; plain GET does.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.request
from pathlib import Path

REPO = "ylecun/mnist"
FILES = {
    "train": "mnist/train-00000-of-00001.parquet",
    "test": "mnist/test-00000-of-00001.parquet",
}


def endpoint() -> str:
    ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return ep


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present ({dest.stat().st_size} bytes)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "mnist-loader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r       {done/1e6:8.2f}/{total/1e6:.2f} MB ({pct:5.1f}%)", end="")
        print()
    tmp.replace(dest)
    return dest


def read_parquet(path: Path):
    """Return (images uint8[N,28,28], labels uint8[N])."""
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = table.column_names
    label_col = next((c for c in names if "label" in c.lower()), None)
    img_col = next((c for c in names if c != label_col), None)
    if label_col is None or img_col is None:
        raise SystemExit(f"unexpected parquet schema: {names}")

    labels = np.asarray(table.column(label_col).to_pylist(), dtype=np.uint8)
    raw = table.column(img_col).to_pylist()
    images = np.zeros((len(raw), 28, 28), dtype=np.uint8)
    for i, item in enumerate(raw):
        if isinstance(item, dict) and "bytes" in item and item["bytes"] is not None:
            blob = item["bytes"]
        else:
            raise SystemExit(f"cannot decode image cell of type {type(item)}")
        images[i] = decode_image(blob)
    return images, labels


def decode_image(blob: bytes):
    """Decode a PNG/JPEG byte string to a 28x28 uint8 array."""
    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(blob)).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != (28, 28):
        raise SystemExit(f"unexpected image shape {arr.shape}")
    return arr


def write_idx_images(path: Path, images) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 0x00000803, images.shape[0], 28, 28))
        f.write(images.astype("uint8").tobytes())


def write_idx_labels(path: Path, labels) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 0x00000801, labels.shape[0]))
        f.write(labels.astype("uint8").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output root directory")
    ap.add_argument("--idx", action="store_true", default=True,
                    help="also write the original IDX files (default: on)")
    args = ap.parse_args()

    import numpy as np

    out = Path(args.out)
    hf_dir = out / "hf"
    mn_dir = out / "mnist"
    mn_dir.mkdir(parents=True, exist_ok=True)

    print(f"endpoint: {endpoint()}")
    splits = {}
    for split, rel in FILES.items():
        url = f"{endpoint()}/datasets/{REPO}/resolve/main/{rel}"
        pq_path = download(url, hf_dir / Path(rel).name)
        print(f"[read] {pq_path.name}")
        images, labels = read_parquet(pq_path)
        splits[split] = (images, labels)
        print(f"       {split}: {images.shape} labels {labels.shape} "
              f"classes {np.bincount(labels, minlength=10)}")

    # sanity: a few well-known statistics of MNIST
    tr_x, tr_y = splits["train"]
    te_x, te_y = splits["test"]
    assert tr_x.shape == (60000, 28, 28), tr_x.shape
    assert te_x.shape == (10000, 28, 28), te_x.shape
    assert tr_x.max() == 255 and tr_x.min() == 0
    print(f"train pixel mean {tr_x.mean():.4f} (expect ~33.32)")

    for split, (images, labels) in splits.items():
        np.save(mn_dir / f"{split}_images.npy", images)
        np.save(mn_dir / f"{split}_labels.npy", labels)
        if args.idx:
            tag = "train" if split == "train" else "t10k"
            write_idx_images(mn_dir / f"{tag}-images-idx3-ubyte", images)
            write_idx_labels(mn_dir / f"{tag}-labels-idx1-ubyte", labels)

    print(f"\nwrote {len(list(mn_dir.iterdir()))} files to {mn_dir.resolve()}")
    for p in sorted(mn_dir.iterdir()):
        print(f"  {p.name:32s} {p.stat().st_size/1e6:8.3f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
