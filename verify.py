#!/usr/bin/env python3
"""Compare the CUDA implementation against the NumPy reference.

Two levels of checking:

  layer level  -- build/dump/{conv1,relu1,pool1,conv2,relu2,pool2}.bin produced by
                  nut_lenet --dump-layers must match weights/ref_layers.npz
  end to end   -- build/dump/logits.bin produced by nut_lenet --dump-logits must
                  match weights/ref_test_logits.npy (atol=rtol=1e-5)

usage:
    python verify.py                     # uses build/dump
    python verify.py --dump build/dump --weights weights
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# shapes of the dumped layers: (N images = 8) x C x H x W, matching the C++ side
LAYERS = [
    ("conv1",  (8, 8, 28, 28),  ("conv1",)),
    ("relu1",  (8, 8, 28, 28),  ("relu1",)),
    ("pool1",  (8, 8, 14, 14),  ("pool1",)),
    ("conv2",  (8, 16, 14, 14), ("conv2",)),
    ("relu2",  (8, 16, 14, 14), ("relu2",)),
    ("pool2",  (8, 16, 7, 7),   ("pool2",)),
]


def compare(name, got, ref, atol=1e-5, rtol=1e-5):
    if got.shape != ref.shape:
        print(f"  {name:8s} SHAPE MISMATCH {got.shape} vs {ref.shape}")
        return False
    diff = np.abs(got - ref)
    ok = np.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"  {name:8s} max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="build/dump")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()

    dump = Path(args.dump)
    wdir = Path(args.weights)
    ref = np.load(wdir / "ref_layers.npz")
    ok = True

    print("layer-by-layer (CUDA vs NumPy):")
    for fname, cshape, refkeys in LAYERS:
        p = dump / f"{fname}.bin"
        if not p.exists():
            print(f"  {fname:8s} missing ({p}) -- run nut_lenet --dump-layers {dump}")
            ok = False
            continue
        got = np.fromfile(p, dtype=np.float32).reshape(cshape)
        ok &= compare(fname, got, ref[refkeys[0]], args.atol, args.rtol)

    print("end to end (logits):")
    lp = dump / "logits.bin"
    if lp.exists():
        got = np.fromfile(lp, dtype=np.float32).reshape(-1, 10)
        rl = np.load(wdir / "ref_test_logits.npy")
        if got.shape[0] != rl.shape[0]:
            print(f"  logits   SHAPE MISMATCH {got.shape} vs {rl.shape}")
            ok = False
        else:
            ok &= compare("logits", got, rl, args.atol, args.rtol)
            # classification must be identical
            gp, rp = got.argmax(1), rl.argmax(1)
            same = int((gp == rp).sum())
            print(f"  argmax   identical for {same}/{len(gp)} images "
                  f"{'OK' if same == len(gp) else 'FAIL'}")
            ok &= (same == len(gp))
            lab = np.load(wdir / "ref_test_labels.npy")
            print(f"  accuracy CUDA {100.0*(gp == lab).mean():.2f}%   "
                  f"NumPy {100.0*(rp == lab).mean():.2f}%")
    else:
        print(f"  missing {lp} -- run nut_lenet --dump-logits {lp}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
