# `neural_inspired/` — opt-in brain-inspired add-ons

These are **optional research add-ons**, not part of the baseline model. They are
the "may-or-may-not-make-the-cut" ideas from
[`notes/neural-inspired-directions.md`](../notes/neural-inspired-directions.md):
each is motivated by neuroscience but justified only by measured gain. If an
add-on's improvement turns out minor, it is meant to be **removed wholesale**
without touching the baseline.

## Isolation guarantees

- **Self-contained**: torch-only; no import of the baseline model modules.
- **Off by default**: reachable *only* via an explicit config knob.
- **Lazy**: the baseline model imports a class from here **only inside the
  matching config branch** (e.g. `temporal_pooling == "surprise"`), so a baseline
  run never touches this package.
- **No-regression start**: every component is zero-gated to begin **bit-for-bit
  equal** to the baseline it replaces, so enabling it can't hurt step 0.

## Components

| class | replaces | config knob | notes-directions ref |
|---|---|---|---|
| `TemporalSurprisePool` | mean / attention temporal pooling | `model.vision_encoder.temporal_pooling = "surprise"` | N2 (predictive-coding surprise pooling), pool half |

### `TemporalSurprisePool`
Predictive-coding surprise-weighted pooling over the T frame embeddings. The mean
frame is the "prediction"; each frame is weighted by its prediction-error energy.
Gated by `alpha` (init 0 → mean pooling). It is the **brain-inspired twin of
`TemporalAttentionPool`** (the engineering baseline in
`models/mobileviclip_small.py`); its kill-criterion is *must beat
`temporal_pooling="attn"` at equal steps on full data* — that ranking is an
**H100 job**, the V100/10k pilot only screens that it wires up and trains
(`alpha`/`gamma`/`key` leave init, loss falls).

Configs: `experiments/configs/pilot_10k_surprise.py` (V100 plumbing screen, isolates
the pool vs mean) and `experiments/configs/nA1_surprisepool.py` (full-data A/B arm).

## Screening `TemporalSurprisePool` on V100 (10k plumbing go/no-go)

The 10k pilot only checks that the add-on **wires up and trains** (not that it
wins — that ranking is the H100 A/B, `nA1_surprisepool.py`). Run from repo root
(`conda activate mobileviclip`; edit `CUDA_VISIBLE_DEVICES` inside the `.sh` if needed):

```bash
git pull origin main
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_surprise.py 8 debug True   # smoke
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_surprise.py 8               # real
```

**Pass = loss falls AND the pool trained off mean (`alpha` ≠ 0).** The ckpt is a
DeepSpeed directory — weights are in `mp_rank_00_model_states.pt` under `module`
(loading the dir directly raises `IsADirectoryError`):

```bash
python - <<'PY'
import torch
f = "experiments/runs/pilot_10k_surprise/ckpt_09.pth/mp_rank_00_model_states.pt"
sd = torch.load(f, map_location="cpu"); sd = sd.get("module", sd)
a = sd.get("vision_encoder.temporal_pool.alpha")
g = sd.get("vision_encoder.temporal_pool.gamma")
k = sd.get("vision_encoder.temporal_pool.key.weight")
print("alpha =", None if a is None else round(a.float().flatten()[0].item(),5), " (init 0 → want ≠0)")
print("gamma =", None if g is None else round(g.float().flatten()[0].item(),5), " (init 1 → want ≠1)")
print("key.weight L2 =", None if k is None else round(k.float().norm().item(),4))
PY
```

`alpha` ≠ 0 + loss down → wired and training, not silently degenerating to mean
pooling. If `alpha` is stuck at **exactly** 0, the module never left the zero-gradient
init: the first 10k screen (2026-07-24) hit this because `gamma` also started at 0 —
uniform softmax makes `pooled ≡ mean`, which zeroes the gradient to *every* parameter.
Fixed by initializing `gamma` nonzero (so `pooled ≠ mean`, giving `alpha` a live
gradient) while keeping the `alpha=0` no-regression start. See
`experiments/pilot_10k_results.md` for the autopsy. If `alpha` is still 0 after that,
check that `temporal_pool` params are trainable (freeze/lr name-match `"temporal_pool"`
in `models/mobileviclip_small.py`). Unit test (torch-only, no timm/decord):
`python -m pytest tests/test_neural_inspired.py -v` — includes a regression test that
`alpha` receives a nonzero gradient at init.

## How to remove an add-on (if it doesn't earn its place)

For `TemporalSurprisePool`:
1. Delete `neural_inspired/temporal_surprise_pool.py` and its export in
   `neural_inspired/__init__.py` (delete the whole folder if it's the last one).
2. Delete the `elif temporal_pooling == "surprise":` branch in
   `models/mobileviclip_small.py` (the only reference into this package).
3. Delete `experiments/configs/{pilot_10k_surprise,nA1_surprisepool}.py` and
   `tests/test_neural_inspired.py`.

After that the baseline (`temporal_pooling` ∈ {`mean`,`attn`}) is byte-identical.
