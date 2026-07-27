# R1 — SigLIP-regime negative-pool arm: implementation plan

**Date:** 2026-07-27
**Reframed per** [`notes/r1-examination.md`](r1-examination.md) §6: sell this as **"contrastive
training at small batch, fixed with a SigLIP-regime large negative pool"**, *not* as "the
mathematically-exact bank". The exactness is a real but **v2t-side** engineering nicety
(examination §1); the number we lead with (t2v) still rides a stale video queue.

**Status of the pre-checks (examination §7):** Part A done — exact-duplicate FN ≈0, so
**exact-`idx` dedup can be skipped**. Part B (near-duplicate density) decides the
cosine dedup mask; run `experiments/r1_textfeat_dump.py` → `r1_fn_density.py --feats`
(COMMANDS §3.2).

**Go/no-go pilot RAN (2026-07-27) — see §7.** Result: **inconclusive, prior tilted to
KILL.** The probe as first shipped was *mis-specified* (its metrics answer the wrong
question, and its baked-in "saturated ⇒ kill" rule was wrong science). §7 records the
numbers, the correction, the two zero-training checks that actually decide, and why
**near-dup dedup is now a build prerequisite, not optional** (upgrades §2 / examination §3).

---

## 0. The one arm, in order of what actually matters

| # | Component | Why | Cost |
|---|---|---|---|
| **(i)** | **Sigmoid (SigLIP) loss** | The documented fix for small-batch contrastive; already implemented (`vtc_loss_sigmoid`, `r1a_sigmoid.py`). Also **sidesteps the fp16 trap** (SigLIP logit scale ≈`exp(log 10)`≈10, vs softmax's `1/temp`=100 — far from fp16 overflow). | flag flip |
| **(ii)** | **Learnable temperature/bias** | Already in `r1a_sigmoid.py` (`logit_scale`, `logit_bias`, `different_lr`). Fixed 1/100 was inherited from image-CLIP and is on the sharp end. | in r1a |
| **(iii)** | **Large text negative pool** | The lever for bottleneck (d). Cached frozen-text features, re-projected each step. | the new work |
| **(iv)** | **Saturation instrumentation** | **This is the go/no-go**, not the pool itself. | ~10 lines |

**Build order:** turn on (i)+(ii) first (that *is* `r1a_sigmoid.py` — already a queued arm), add
the instrumentation (iv), then the pool (iii). If (iv) shows the batch is already saturated at
512, stop before (iii).

---

## 1. Touchpoints (verified against `main`)

1. **`models/criterions.py`** — the pool logic. `vtc_loss_sigmoid` (`:145`) is the host;
   `get_mask(idx)` (`:287`) is the multi-positive target builder that the pool **must** extend
   (examination §3). `get_sim`/`get_sim_sigmoid` produce the `[B,B]` logits.
2. **`models/mobileviclip_small.py:forward` (`:126-155`)** — must accept the cached bank and pass
   it to the loss. Today it computes loss internally from `(image, text, idx)`.
3. **Dataset → loader → forward plumbing** — the bank features + their `idx` must reach `forward`.
   Options: (a) a module buffer refreshed each step from a cached tensor on GPU (simplest — the
   bank is global, not per-sample), or (b) thread through the batch. **(a) is preferred:** the
   bank is the *same* every step; only the *sampled subset* changes.
4. **`scripts/pretraining/clip/mobileviclip_small/config.py`** — the switch, default-off.

---

## 2. The pool mechanism (reference spec — to wire + test on the server)

**Offline, once** (the "exact" part — valid only because the text tower is frozen; examination
§0/§2): cache the **pre-projection** text features for the caption pool. Use
`experiments/r1_textfeat_dump.py`'s build path, but stop *before* `projection_layer` (dump the
output of `encode_text` **minus** the final matmul; or equivalently dump post-projection and
skip re-projection — see the exactness note). Store `[K, 512]` fp16 on disk (~67 MB at K=65k).

> **Exactness caveat to encode in the loader (examination §2):** the frozen text tower must run
> with dropout off. `requires_grad=False` does **not** disable dropout and training is in
> `model.train()`; the config sets `embed_dropout=0` so it is exact *today*, but **assert every
> text-path dropout p==0** (or `text_encoder.eval()` for the cache pass) so a future config swap
> can't silently break exactness.

**Per step, in `forward`** (reference — note the corrected `get_mask` threading, not
`cross_entropy(arange)`):

```python
# bank_feats: [K, 512] cached PRE-projection text features (constant across the run)
# bank_idx:   [K]      caption_to_idx of each banked caption (for the multi-positive mask)
# sample K' of them per step (K'=16k..65k); draw from a SEPARATE torch.Generator so the
# main data RNG stream is untouched (reproducibility).
t_bank = F.normalize(bank_feats_sub @ self.text_encoder.projection_layer, dim=-1)  # re-project w/ CURRENT proj -> exact
# extend the [B, B] logits to [B, B+K'] by appending video–bankedtext columns:
extra = get_sim_cols(vision_proj, t_bank, temp_or_t_b)          # [B, K']
# EXTEND THE TARGET MASK, do not assume all banked cols are negatives:
#   a banked caption sharing a query clip's idx is a POSITIVE.
extra_mask = (idx.view(-1,1) == bank_idx_sub.view(1,-1))         # [B, K'] positives among the bank
# sigmoid path (preferred): per-pair BCE-with-logits over [B, B+K'], targets = concat(get_mask, extra_mask)
# softmax path: log_softmax over the extended row, soft targets = row-normalized concat(...)
```

- **fp16:** compute the extended-logit matmul + reduction in **fp32** with max-subtraction, and
  chunk over the K′ axis. Under sigmoid this is a safety margin, not a necessity (§0).
- **Asymmetry (examination §1):** this hardens **v2t** (video query, more text candidates).
  The headline **t2v** needs banked *video* features, which drift (`freeze_vision=False`) → a
  standard **momentum/FIFO video queue**, added only if t2v lags. Do **not** claim exactness for
  t2v.
- **Exact-`idx` dedup:** Part A showed it is ≈moot (examination §7) — the `extra_mask` above
  already handles the rare exact collision correctly, so nothing extra is needed there.
- **Near-dup dedup:** a precomputed cosine-threshold mask is a **prerequisite**, not optional
  (revised — see §7.4). A near-duplicate caption with a different `idx` is a *semantic positive
  labeled negative*; under sigmoid its high logit sits in the active region and dominates the
  gradient, so scaling the pool scales up **mislabeled repulsion**. Part B sizes the mask; it
  does not decide whether the mask exists.

---

## 3. Config (the switch — follows the `r1a_sigmoid.py` idiom, default-off)

```python
# experiments/configs/nR1_textbank.py  (create when the loss wiring lands)
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa

model["use_sigmoid_loss"] = True                       # (i) SigLIP regime
for __c in available_corpus["internvid_all"]:
    __c["caption_idx"] = True                          # multi-positive mask active
optimizer["different_lr"] = dict(                      # (ii) learnable t/b need a big lr
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3)
criterion["text_bank"] = dict(                         # (iii) the pool (read by the new loss path)
    enable=True, path="cache/internvid_txtfeat.fp16.npy",
    idx_path="cache/internvid_txtidx.npy", K=16384, dedup_cos=None)  # dedup_cos set from Part B
output_dir = "experiments/runs/nR1_textbank"
```

**Default-off = bitwise-identical:** with `text_bank.enable=False` (the base config default), the
new `forward`/loss branch is an **early return** to the current path (not `+0*extra` — that
executes ops and can change fp16 reduction order; examination §3 / GATE-0 lesson).

---

## 4. The go/no-go instrumentation (build this before the pool pays off)

> **Superseded by §7 (2026-07-27).** The pilot ran; the metrics below (softmax-mass /
> positive-rank) were the right *intuition* but `top1_is_pos`/margin turned out to be the wrong
> instruments and the kill rule was mis-stated. §7 has the measured numbers, the corrected
> metrics now in code (`maxneg_by_pool`, `neg_grad_mass`), and the decision table. Read §7 first.

Log every step (cheap):
- **softmax/sigmoid mass on banked vs in-batch negatives** — <5% ⇒ the pool is inert ⇒ **kill**.
- **rank of the positive among the 512 in-batch negatives** — already rank-1 for most samples ⇒
  batch is saturated at 512 ⇒ a bigger pool cannot help ⇒ **(d) misdiagnosed, stop**.

These can be read from a **short pilot with only (i)+(ii)+instrumentation, no pool yet** — i.e.
run `r1a_sigmoid.py` with the two log lines added. That answers "is (d) real here?" *before*
any pool engineering, which is the cheapest possible go/no-go given the examination §4 warning
that our aligned-init 3-epoch regime may already be near its own saturation.

---

## 5. Sequencing (extends `experiments/ROADMAP.md` 2.4)

```
2.4a  Part B near-dup density (now one command; COMMANDS §3.2)   -> sets dedup_cos
2.4b  r1a_sigmoid.py + 2 saturation log lines, 10k pilot         -> the GO/NO-GO
        kill: banked/positive-rank say 512 already saturates -> stop, (d) misdiagnosed
2.4c  build the pool (§2): offline cache -> forward plumbing -> get_mask extension -> fp32/chunk
        full A/B: r1a (no pool) vs nR1_textbank (pool), matched steps, both DSL+raw
        report BOTH directions; headline t2v honestly = stale-queue side (§1)
2.4d  video FIFO queue only if t2v lags (momentum encoder; not exact)
```

**Kill criterion (unchanged):** 512 → 65k negatives < 0.5 R-Sum ⇒ stop. 2.4b makes that verdict
reachable for ~one pilot's cost, before the pool is built.

---

## 6. Honesty ledger (carry into any writeup)

- Exactness is v2t-side; t2v is a stale queue (§1). Report both directions; don't headline exact.
- The size-of-prize (30–60× below SigLIP saturation) was measured from-scratch at billion scale;
  our aligned 3-epoch fine-tune may sit near its own saturation — 2.4b tests this directly (§4).
- Neuroscience framing (complementary learning systems) is a rationalisation of an engineering
  fact, not a derivation — do not sell R1 as brain-inspired (examination §T1 / round-2 R1).

*Code references checked 2026-07-27 against `models/criterions.py`, `models/mobileviclip_small.py`,
`scripts/pretraining/clip/mobileviclip_small/config.py`, `experiments/configs/r1a_sigmoid.py`.*

---

## 7. Go/no-go pilot — measured (2026-07-27), and the probe corrected

Ran the pilot per COMMANDS §3.3 (`MVC_LOG_SAT=20 bash experiments/run_pretrain.sh
experiments/configs/pilot_10k_r1.py 8`): sigmoid loss, global **B=512**, 10 epochs on the
`internvid_zh_10k` pilot (9000 train / 1000 val), both towers from the aligned `mobileclip_s2`.
`[SAT]` every 20 steps:

| step | top1_is_pos | mean(pos − hardest_neg) |
|---|---|---|
| 20  | 0.754 | 0.431 |
| 40  | 0.670 | 0.253 |
| 60  | 0.686 | 0.315 |
| 80  | 0.744 | 0.466 |
| 100 | 0.740 | 0.526 |
| 120 | 0.752 | 0.555 |
| 140 | 0.758 | 0.604 |
| 160 | 0.773 | 0.595 |

Pilot val retrieval plateaued at ~**t2v R@1 76–77, R@5 92** (started at 79.7 from the aligned
init, dipped, recovered — normal for a sigmoid fine-tune re-learning `logit_scale`/`logit_bias`).

### 7.1 The probe as first shipped was mis-specified — now corrected in code

Two errors, both fixed in `models/criterions.py` this round:

1. **Wrong kill rule.** The v1 log line asserted "`top1_is_pos→1` & big margin ⇒ batch
   saturated ⇒ bigger pool useless." That logic is **false**: the hardest negative in the data
   grows like **`max-neg-similarity ~ log(pool size)`**, so a 32k pool almost always surfaces a
   harder negative than any fixed 512-draw — *even when every 512-batch is perfectly separated*.
   Full in-batch separation and "a bigger pool adds gradient" are **not** mutually exclusive.
2. **Wrong instruments.** `top1_is_pos` and `mean_margin` measure **ranking/separation**, not the
   quantity that decides the lever: **where the negative-logit mass sits relative to the sigmoid
   boundary**. Under `L = −log σ(z·(t·s+b))`, a negative contributes gradient `∝ σ(t·s+b)` — it is
   dead once pushed below the boundary. "More negatives help" requires the larger pool to add
   negatives that land in the **still-active region**; ranking stats cannot see that.

The probe now also logs the two metrics that *can* (both computable from the same `[B,B]`
matrix, no extra training):
- **`maxneg_by_pool[64:… 128:… 256:… 512:…]`** — mean hardest-negative value over nested column
  prefixes. **Still rising at 512 ⇒ a larger pool keeps finding harder negatives** (the direct,
  link-function-free test). Flat by 512 ⇒ pool won't add hard negatives.
- **`neg_grad_mass`** (sigmoid path) — mean over rows of `Σ_j σ(logit_ij)` over negatives =
  effective # of gradient-bearing negatives. **≈0 ⇒ in-batch negatives are dead** (adding *easy*
  ones from a pool is free of signal).

### 7.2 What the measured numbers do and don't say

- `top1_is_pos ≈ 0.73` (random ≈ 1/512 ≈ 0.002) ⇒ the 512-batch is **strongly separated but NOT
  saturated** — ~27% of queries still have an in-batch negative outranking the positive. **The
  clean "512 saturated ⇒ kill" signal did not fire.**
- But this is **not** a build signal either. The residual hard negatives are confounded twice:
  (a) on a 10k / 3-epoch pilot a **growing** margin is mostly **memorization**, not evidence about
  the corpus; (b) **false negatives** — near-duplicate captions with a different `idx` — are
  labeled negatives yet are semantically correct; they *are* high-logit "hard" negatives, so they
  simultaneously **depress `top1_is_pos`** and **inflate the margin** without being real signal.

### 7.3 Verdict — inconclusive, prior tilted to KILL; two zero-training checks decide

The two metrics I *didn't* have (7.1) are exactly what's needed, and both are free:

1. **`maxneg_by_pool` scaling** (now logged) — re-run the same pilot; read whether mean
   hardest-negative is still climbing 256→512. Plateaued ⇒ **kill**.
2. **Near-duplicate density** (Part B; `r1_textfeat_dump.py` → `r1_fn_density.py --feats`,
   COMMANDS §3.2) — off-diagonal high-cosine pairs in the *projected* space.

Decision table:

| `maxneg_by_pool` at 512 | near-dup density | call |
|---|---|---|
| plateaued | any | **kill** (no harder negatives to add) |
| still rising | **high** | **kill / fix labels** ("hard negatives" are false negatives; a bigger un-deduped pool amplifies mislabeled repulsion) |
| still rising | **low** | **build R1** (genuine hard negatives a larger pool would add) — least likely of the three |

**Regime prior (independent of the pilot).** Aligned init + frozen text + only the projection
trainable + tiny data means we start *inside* the flat, post-saturation part of SigLIP's
batch-size curve (that curve is a from-scratch, alignment-from-noise phenomenon). So the
negative-pool lever is expected to be **substantially flatter here** than SigLIP's "batch-limited
below 16k" headline — that number does not transfer to our regime (echoes examination §4).

### 7.4 Consequence for the build (revises §2)

Because **false negatives are themselves hard negatives**, near-dup dedup moves from "add iff
Part B is high" to a **hard prerequisite** (§2 bullet updated). The same mechanism that would make
a pool "work" (surfacing high-logit negatives) is the mechanism that injects mislabeled repulsion;
an un-deduplicated 32k pool is a **net-negative risk**, not a neutral default.

**Do not build the pool on this evidence.** Re-run the pilot to read `maxneg_by_pool`/`neg_grad_mass`,
run Part B, then apply the table above.

*Interpretation pressure-tested via a `deep-learning-researcher` review of the InfoNCE/sigmoid
gradient (2026-07-27); the corrections above are its findings, verified against `criterions.py`.*
