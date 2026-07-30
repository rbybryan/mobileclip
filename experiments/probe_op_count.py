"""Count aten dispatches (≈ CUDA kernel launches) per training step of the vision tower.

Zero-GPU diagnostic for the 0.67%-MFU question (notes/architecture-round4.md §0.2).
Runs the *training-mode* (non-reparameterized) FastViT tower on CPU at a small
spatial size and counts every aten op that a step dispatches, split into
forward / backward and bucketed by op family. Layout ops (copy_/clone/contiguous)
also get their element counts summed, so the "layout tax" can be scaled to the
real training shape analytically.

Usage:
    PYTHONPATH=. python experiments/probe_op_count.py [--h 64] [--t 8] [--b 1]
"""
import argparse
import collections
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, ".")


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = collections.Counter()
        self.numels = collections.Counter()
        self.enabled = True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.enabled:
            name = str(func).split(".")[0].replace("aten::", "")
            self.counts[name] += 1
            n = 0
            for o in (out if isinstance(out, (list, tuple)) else [out]):
                if isinstance(o, torch.Tensor):
                    n += o.numel()
            self.numels[name] += n
        return out


LAYOUT = {"copy_", "clone", "contiguous", "_to_copy", "as_strided", "cat", "stack"}
CONV = {"convolution", "conv2d", "conv1d", "_convolution", "convolution_backward",
        "cudnn_convolution", "thnn_conv2d"}
NORM = {"native_batch_norm", "batch_norm", "_native_batch_norm_legit",
        "native_batch_norm_backward", "cudnn_batch_norm", "cudnn_batch_norm_backward",
        "_native_batch_norm_legit_functional"}


def build_tower(with_cp, temporal=True):
    from models.mobileviclip_small import FastViT, RepCPE
    from functools import partial
    return FastViT(
        [4, 12, 24, 4],
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
        embed_dims=[80, 160, 320, 640],
        pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
        mlp_ratios=[3, 3, 3, 3],
        downsamples=[True, True, True, True],
        se_downsamples=[False, False, True, True],
        clip_embed_dim=512,
        attn_pool_num_heads=16,
        head_drop_path_rate=0.0,
        projection_dim=512,
        temporal_pooling="mean",
        with_cp=with_cp,
    )


def run(with_cp, t, b, h):
    model = build_tower(with_cp).train()
    # tower input is [B, C, T, H, W] (see MobileViCLIP_Small.encode_vision)
    x = torch.randn(b, 3, t, h, h, requires_grad=True)
    fwd, bwd = OpCounter(), OpCounter()
    with fwd:
        y = model(x)
    loss = y.float().pow(2).mean()
    with bwd:
        loss.backward()
    return fwd, bwd


def report(tag, c):
    tot = sum(c.counts.values())
    layout = sum(v for k, v in c.counts.items() if k in LAYOUT)
    layout_n = sum(v for k, v in c.numels.items() if k in LAYOUT)
    conv = sum(v for k, v in c.counts.items() if k in CONV)
    norm = sum(v for k, v in c.counts.items() if k in NORM)
    print(f"\n=== {tag}: {tot} aten dispatches ===")
    print(f"  layout (copy_/clone/contiguous/_to_copy): {layout} "
          f"({100*layout/max(tot,1):.1f}%), {layout_n/1e6:.1f}M elements moved")
    print(f"  convolution family: {conv}   norm family: {norm}")
    print("  top 20 ops:")
    for k, v in c.counts.most_common(20):
        print(f"    {k:36s} {v:6d}   ({c.numels[k]/1e6:8.1f}M el)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--h", type=int, default=64)
    a = ap.parse_args()
    print(f"torch {torch.__version__}  T={a.t} B={a.b} HxW={a.h}x{a.h}")
    for cp in (False, True):
        f, b = run(cp, a.t, a.b, a.h)
        report(f"FORWARD  with_cp={cp}", f)
        report(f"BACKWARD with_cp={cp}", b)
        print(f"  >>> STEP TOTAL with_cp={cp}: "
              f"{sum(f.counts.values()) + sum(b.counts.values())} dispatches")
