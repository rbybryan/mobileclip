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

## How to remove an add-on (if it doesn't earn its place)

For `TemporalSurprisePool`:
1. Delete `neural_inspired/temporal_surprise_pool.py` and its export in
   `neural_inspired/__init__.py` (delete the whole folder if it's the last one).
2. Delete the `elif temporal_pooling == "surprise":` branch in
   `models/mobileviclip_small.py` (the only reference into this package).
3. Delete `experiments/configs/{pilot_10k_surprise,nA1_surprisepool}.py` and
   `tests/test_neural_inspired.py`.

After that the baseline (`temporal_pooling` ∈ {`mean`,`attn`}) is byte-identical.
