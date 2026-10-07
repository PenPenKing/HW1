#!/usr/bin/env python3
"""Train the TinyLeNet used by this homework, in pure NumPy, and export:

  weights/lenet_weights.bin   flat float32 blob, layout described by the .json
  weights/lenet_weights.json  shapes/offsets so the CUDA host can load it blind
  weights/ref_test_logits.npy (10000,10) float32 reference logits for the test set
  weights/ref_layers.npz      per-layer activations for the first N test images
  results/train_log.txt       human readable training log

Network (same as the course guidance's TinyLeNet, 9098 parameters):
  input 1x28x28
    conv 1->8, 3x3, stride 1, pad 1   -> ReLU -> maxpool 2x2   8x14x14
    conv 8->16, 3x3, stride 1, pad 1  -> ReLU -> maxpool 2x2   16x7x7
    flatten 784 -> dense 10 -> argmax
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def im2col(xp: np.ndarray, k: int) -> np.ndarray:
    """(N,C,Hp,Wp) padded input -> (N, C*k*k, H*W) column matrix."""
    n, c, hp, wp = xp.shape
    h, w = hp - k + 1, wp - k + 1
    s = xp.strides
    view = np.lib.stride_tricks.as_strided(
        xp, shape=(n, c, k, k, h, w), strides=(s[0], s[1], s[2], s[3], s[2], s[3])
    )
    return view.reshape(n, c * k * k, h * w)


def col2im(cols: np.ndarray, k: int, hp: int, wp: int) -> np.ndarray:
    n, ckk, hw = cols.shape
    c = ckk // (k * k)
    h = hp - k + 1
    w = wp - k + 1
    out = np.zeros((n, c, hp, wp), dtype=np.float32)
    d = cols.reshape(n, c, k, k, h, w)
    for i in range(k):
        for j in range(k):
            out[:, :, i:i + h, j:j + w] += d[:, :, i, j]
    return out


def conv_forward(x, w, b, pad=1):
    n, c, h, wd = x.shape
    o, _, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    cols = im2col(xp, k)
    out = w.reshape(o, -1) @ cols + b[:, None]
    return out.reshape(n, o, h, wd), cols, xp.shape


def conv_backward(dout, w, cols, in_shape, pad=1):
    n, o, h, wd = dout.shape
    _, c, k, _ = w.shape
    flat = dout.reshape(n, o, h * wd)
    db = flat.sum(axis=(0, 2))
    dw = flat @ cols.transpose(0, 2, 1)          # (n,o,c*k*k) summed later
    dw = dw.sum(axis=0).reshape(w.shape)
    dcols = w.reshape(o, -1).T @ flat            # (n,c*k*k,h*w)
    dxp = col2im(dcols, k, in_shape[2], in_shape[3])
    dx = dxp[:, :, pad:dxp.shape[2] - pad, pad:dxp.shape[3] - pad] if pad else dxp
    return dx, dw, db


def relu_forward(x):
    return np.maximum(x, 0.0)


def relu_backward(dout, x):
    return dout * (x > 0.0)


def pool_forward(x):
    n, c, h, w = x.shape
    v = x.reshape(n, c, h // 2, 2, w // 2, 2)
    return v.max(axis=(3, 5))


def pool_backward(dout, x):
    n, c, h, w = x.shape
    v = x.reshape(n, c, h // 2, 2, w // 2, 2)
    mask = (v == v.max(axis=(3, 5), keepdims=True))
    # split ties so that gradients are not duplicated
    mask = mask / mask.sum(axis=(3, 5), keepdims=True)
    d = (dout[:, :, :, None, :, None] * mask).reshape(n, c, h, w)
    return d


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


class LeNet:
    """conv1 -> relu -> pool -> conv2 -> relu -> pool -> fc -> softmax."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        k = lambda fi: np.sqrt(2.0 / fi).astype(np.float32)  # He scale
        self.p = {
            "conv1_w": (rng.standard_normal((8, 1, 3, 3)) * k(9)).astype(np.float32),
            "conv1_b": np.zeros(8, np.float32),
            "conv2_w": (rng.standard_normal((16, 8, 3, 3)) * k(72)).astype(np.float32),
            "conv2_b": np.zeros(16, np.float32),
            "fc_w": (rng.standard_normal((10, 784)) * k(784)).astype(np.float32),
            "fc_b": np.zeros(10, np.float32),
        }

    def forward(self, x, cache=None):
        """x: (N,1,28,28) float32 -> logits (N,10)."""
        c1, cols1, shp1 = conv_forward(x, self.p["conv1_w"], self.p["conv1_b"])
        r1 = relu_forward(c1)
        p1 = pool_forward(r1)

        c2, cols2, shp2 = conv_forward(p1, self.p["conv2_w"], self.p["conv2_b"])
        r2 = relu_forward(c2)
        p2 = pool_forward(r2)

        flat = p2.reshape(p2.shape[0], -1)
        logits = flat @ self.p["fc_w"].T + self.p["fc_b"]

        if cache is not None:
            cache.update(dict(x=x, c1=c1, r1=r1, p1=p1, c2=c2, r2=r2, p2=p2,
                              flat=flat, logits=logits, cols1=cols1, cols2=cols2,
                              shp1=shp1, shp2=shp2))
        return logits

    def backward(self, cache, labels) -> dict:
        n = labels.shape[0]
        logits = cache["logits"]
        z = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(z)
        prob /= prob.sum(axis=1, keepdims=True)
        dlogits = prob.copy()
        dlogits[np.arange(n), labels] -= 1.0
        dlogits /= n

        g = {}
        g["fc_w"] = dlogits.T @ cache["flat"]
        g["fc_b"] = dlogits.sum(axis=0)
        dflat = dlogits @ self.p["fc_w"]
        dp2 = dflat.reshape(cache["p2"].shape)

        dr2 = pool_backward(dp2, cache["r2"])
        dc2 = relu_backward(dr2, cache["c2"])
        dp1, g["conv2_w"], g["conv2_b"] = conv_backward(
            dc2, self.p["conv2_w"], cache["cols2"], cache["shp2"])

        dr1 = pool_backward(dp1, cache["r1"])
        dc1 = relu_backward(dr1, cache["c1"])
        _, g["conv1_w"], g["conv1_b"] = conv_backward(
            dc1, self.p["conv1_w"], cache["cols1"], cache["shp1"])
        return g

    def predict(self, x, batch=512):
        preds = np.empty(x.shape[0], np.int64)
        for i in range(0, x.shape[0], batch):
            preds[i:i + batch] = self.forward(x[i:i + batch]).argmax(axis=1)
        return preds


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def iterate_batches(x, y, batch, rng):
    idx = rng.permutation(x.shape[0])
    for i in range(0, x.shape[0] - batch + 1, batch):
        sel = idx[i:i + batch]
        yield x[sel], y[sel]


def augment_shift(x, rng, max_shift):
    """Random translation with zero fill, x: (N,1,28,28). Grouped by offset so
    that the whole batch is shifted with 25 vectorised slice assignments."""
    if max_shift <= 0:
        return x
    n = x.shape[0]
    dx = rng.integers(-max_shift, max_shift + 1, n)
    dy = rng.integers(-max_shift, max_shift + 1, n)
    out = np.zeros_like(x)
    for sx in range(-max_shift, max_shift + 1):
        for sy in range(-max_shift, max_shift + 1):
            sel = (dx == sx) & (dy == sy)
            if not sel.any():
                continue
            src = x[sel, 0]
            dst = np.zeros_like(src)
            ysrc = slice(max(0, sy), 28 + min(0, sy))
            ydst = slice(max(0, -sy), 28 + min(0, -sy))
            xsrc = slice(max(0, sx), 28 + min(0, sx))
            xdst = slice(max(0, -sx), 28 + min(0, -sx))
            dst[:, ydst, xdst] = src[:, ysrc, xsrc]
            out[sel, 0] = dst
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mnist")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--augment", type=int, default=0,
                    help="random translation range in pixels (0 = off)")
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay")
    ap.add_argument("--ref-images", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data = Path(args.data)
    tr_x = np.load(data / "train_images.npy").astype(np.float32) / 255.0
    tr_y = np.load(data / "train_labels.npy").astype(np.int64)
    te_x = np.load(data / "test_images.npy").astype(np.float32) / 255.0
    te_y = np.load(data / "test_labels.npy").astype(np.int64)
    tr_x = tr_x[:, None, :, :]
    te_x = te_x[:, None, :, :]
    print(f"train {tr_x.shape}  test {te_x.shape}  dtype {tr_x.dtype}")

    model = LeNet(args.seed)
    n_param = sum(v.size for v in model.p.values())
    print(f"parameters: {n_param}")

    # Adam state
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    v = {k: np.zeros_like(v) for k, v in model.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0

    log_lines = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_loss, ep_correct, ep_seen = 0.0, 0, 0
        # cosine learning-rate decay over the whole run
        frac = (epoch - 1) / max(1, args.epochs)
        lr = args.lr * (0.5 * (1 + np.cos(np.pi * frac)) if args.cosine else 1.0)
        for xb, yb in iterate_batches(tr_x, tr_y, args.batch, rng):
            if args.augment:
                xb = augment_shift(xb, rng, args.augment)
            cache = {}
            logits = model.forward(xb, cache)
            z = logits - logits.max(axis=1, keepdims=True)
            logsumexp = np.log(np.exp(z).sum(axis=1))
            loss = (logsumexp - z[np.arange(len(yb)), yb]).mean()
            ep_loss += loss * len(yb)
            ep_correct += (logits.argmax(axis=1) == yb).sum()
            ep_seen += len(yb)

            g = model.backward(cache, yb)
            step += 1
            for k in model.p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh = m[k] / (1 - b1 ** step)
                vh = v[k] / (1 - b2 ** step)
                model.p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

        te_pred = model.predict(te_x)
        te_acc = (te_pred == te_y).mean()
        line = (f"epoch {epoch:2d}  loss {ep_loss / ep_seen:.4f}  lr {lr:.2e}  "
                f"train_acc {ep_correct / ep_seen * 100:.2f}%  test_acc {te_acc * 100:.2f}%  "
                f"({time.time() - t0:.1f}s)")
        print(line, flush=True)
        log_lines.append(line)

    # ---- final evaluation ------------------------------------------------ #
    ref_logits = np.empty((te_x.shape[0], 10), np.float32)
    for i in range(0, te_x.shape[0], 512):
        ref_logits[i:i + 512] = model.forward(te_x[i:i + 512])
    acc = (ref_logits.argmax(axis=1) == te_y).mean()
    final = f"final test accuracy: {acc * 100:.2f}% ({int(acc * te_x.shape[0])}/{te_x.shape[0]})"
    print(final)
    log_lines.append(final)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- flat weight blob ------------------------------------------------ #
    order = ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"]
    blob, meta, off = [], {"order": order, "tensors": {}}, 0
    for k in order:
        arr = np.ascontiguousarray(model.p[k], dtype=np.float32)
        blob.append(arr.reshape(-1))
        meta["tensors"][k] = {"shape": list(arr.shape), "offset_floats": off,
                              "count": int(arr.size)}
        off += arr.size
    flat = np.concatenate(blob)
    (out / "lenet_weights.bin").write_bytes(flat.tobytes())
    meta["total_floats"] = int(flat.size)
    meta["bytes"] = int(flat.nbytes)
    meta["test_accuracy"] = float(acc)
    (out / "lenet_weights.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out/'lenet_weights.bin'} ({flat.nbytes} bytes, {flat.size} floats)")

    # ---- references used by verify.py ------------------------------------ #
    np.save(out / "ref_test_logits.npy", ref_logits)
    np.save(out / "ref_test_pred.npy", ref_logits.argmax(axis=1).astype(np.int32))
    np.save(out / "ref_test_labels.npy", te_y.astype(np.int32))

    k = args.ref_images
    cache = {}
    model.forward(te_x[:k], cache)
    np.savez(out / "ref_layers.npz",
             input=te_x[:k], conv1=cache["c1"], relu1=cache["r1"], pool1=cache["p1"],
             conv2=cache["c2"], relu2=cache["r2"], pool2=cache["p2"],
             flat=cache["flat"], logits=cache["logits"],
             labels=te_y[:k].astype(np.int32))
    print(f"wrote references for {k} images: {out/'ref_layers.npz'}")

    Path("results").mkdir(exist_ok=True)
    Path("results/train_log.txt").write_text("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
