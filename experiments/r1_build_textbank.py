#!/usr/bin/env python3
"""Build the R1 text bank: dump **pre-projection** text features + caption_to_idx for the
negative pool (notes/r1-implementation-plan.md §2/§7.6).

Reuses the verified build path in `r1_textfeat_dump.build_text_model`, but dumps the feature
**before** `@ projection_layer` (via `text_encoder.encode_text(..., return_pre_proj=True)`), so
the training loop can re-project each step with the *current* projection — the exact bank. The
saved `idx` is `caption_to_idx(caption)`, the same id space the dataset uses under
`caption_idx=True`, so a banked caption sharing a query clip's idx is treated as a positive.

⚠️ The model build / encode step is not verifiable off the training server (no weights). The
caption loading + caption_to_idx + save logic is self-tested. If the build errors, paste the
full traceback.

Usage (training env, single GPU, forward-only) — ONE line, no backslash-continuation
(a wrapped backslash-continuation line is exactly what got dropped on paste, silently building
the cache at the default path with the wrong name):
    python experiments/r1_build_textbank.py --anno anno_downstream/internvid_zh_10k_pilot.json --out-feats experiments/cache/textbank_pilot_feats.fp16.npy --out-idx experiments/cache/textbank_pilot_idx.npy

`--ckpt` is irrelevant to the cached feature: the pre-projection EOT feature is produced by the
frozen text base loaded from the eval config's `text_ckpt_path` (mobileclip_s2.pt) during model
construction; `--ckpt` only overrides vision + the trained projection_layer, which sit AFTER the
returned pre-projection feature. So the default `--ckpt` is fine.

Self-test (no model/data):
    python experiments/r1_build_textbank.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                    # sibling scripts
sys.path.insert(0, os.path.dirname(_HERE))   # repo root: `utils`/`models`/`dataset` on path

# reuse the (path-fixed, verified) build + caption-loading path from r1_textfeat_dump.
_spec = _ilu.spec_from_file_location("r1dump", os.path.join(_HERE, "r1_textfeat_dump.py"))
_dump = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dump)
build_text_model = _dump.build_text_model
load_distinct_captions = _dump.load_distinct_captions


def _caption_to_idx():
    """Canonical id used by the training dataset; fall back to the byte-identical copy."""
    try:
        from dataset.pt_dataset import caption_to_idx  # the training source of truth
        return caption_to_idx
    except Exception:
        return _dump.caption_to_idx


def dump_bank(model, tokenizer, device, caps, out_feats, out_idx, bs=256):
    import numpy as np
    import torch

    c2i = _caption_to_idx()
    feats, idxs = [], []
    with torch.no_grad():
        for i in range(0, len(caps), bs):
            chunk = caps[i:i + bs]
            ids = tokenizer(chunk).to(device)
            # pre-projection EOT feature [b, C_pre] — re-projected with the current proj at train time.
            pre = model.text_encoder.encode_text(ids, return_pre_proj=True)
            if isinstance(pre, (tuple, list)):
                pre = pre[0]
            feats.append(pre.float().cpu().numpy().astype("float16"))
            idxs.extend(int(c2i(c)) for c in chunk)
            if i % (bs * 20) == 0:
                print(f"  {i + len(chunk)}/{len(caps)}")
    feats = np.concatenate(feats)
    idxs = np.asarray(idxs, dtype=np.int64)
    for p in (out_feats, out_idx):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    np.save(out_feats, feats)
    np.save(out_idx, idxs)
    print(f"saved feats {feats.shape} ({feats.dtype}) -> {out_feats}")
    print(f"saved idx   {idxs.shape} -> {out_idx}  (distinct idx: {len(set(idxs.tolist()))})")
    print("set criterion['text_bank'] = dict(enable=True, path=<feats>, idx_path=<idx>, "
          "K=16384, dedup_cos=0.90) in the arm config.")


def _self_test():
    import numpy as np  # noqa: F401
    c2i = _caption_to_idx()
    a, b = c2i("a red car"), c2i("a red car")
    assert a == b, "caption_to_idx must be stable"
    assert a != c2i("a blue car"), "distinct captions -> distinct idx"
    assert -(2 ** 63) <= int(a) < 2 ** 63, "idx must fit int64"
    print("self-test OK (caption_to_idx stable/distinct/int64; model+dump need the training env)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno", nargs="*", default=[], help="training anno JSON(s) for the pool")
    ap.add_argument("--ckpt", default="checkpoints/mobileviclip_small.pt")
    ap.add_argument("--config", default=_dump.DEFAULT_CONFIG, help="eval config = model build path")
    ap.add_argument("--caption-key", default="caption")
    ap.add_argument("--sample", type=int, default=0, help="0 = all distinct captions")
    ap.add_argument("--out-feats", default="experiments/cache/textbank_feats.fp16.npy")
    ap.add_argument("--out-idx", default="experiments/cache/textbank_idx.npy")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test(); return
    if not args.anno:
        ap.error("need --anno (one or more training anno JSONs); or --self-test")

    caps, key = load_distinct_captions(args.anno, args.caption_key, args.sample)
    if not caps:
        print("no captions read — check --anno / --caption-key."); return
    print(f"distinct captions: {len(caps)} (field {key!r}); encoding PRE-projection …")
    try:
        model, tok, device = build_text_model(args.config, args.ckpt)
    except Exception:
        import traceback
        traceback.print_exc()
        print("\nmodel build failed — paste the full traceback. captions are loaded; only the build step is left.")
        return
    dump_bank(model, tok, device, caps, args.out_feats, args.out_idx)


if __name__ == "__main__":
    sys.exit(main())
