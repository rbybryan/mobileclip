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

**RESOLVED → GO (2026-07-27). Build the pool (2.4c).** The go/no-go pilot (§7.5) showed
`maxneg_by_pool` still climbing ~+0.2/doubling at 512 (pool lever NOT saturated), and Part B
(§7.6) measured near-duplicate density **low** (~2% at cos≥0.95 ⇒ ~0.5 false-neg/query at 65k).
Still-rising + low-near-dup = the **BUILD** row: the harder negatives a bigger pool adds are
genuine, not false-negatives. Dedup mask `dedup_cos=0.90` stays a (cheap, ~free) prerequisite.
GO means "premise holds, build the arm + run the A/B", not "R1 works" — kill criterion
(512→65k < 0.5 R-Sum ⇒ stop) still governs. Trail: §7.1 (probe mis-spec, fixed) → §7.5
(corrected metrics) → §7.6 (Part B → GO).

**A/B RUN + VERDICT (2026-07-28): the pool WINS its kill criterion, CONFIRMED across two seeds —
ΔR-Sum = +11.0 (seed 42) and +13.1 (seed 43)**, bank ahead at all 10 epochs both directions on both
seeds, scale-invariant metric, env parity confirmed (both `torch 2.5.1+cu118`). R1 is validated **at
pilot scale**. Caveat: both arms sit below the frozen-init epoch-0 peak, so at 9000 clips the bank
acts as *reduced-degradation / higher-plateau* rather than absolute gain over init (the control
collapses at ep1, the bank protects it) — see §9/§9.1/§9.2 for the full tables and why it's
trustworthy. **Verdict: GO, scale up** (§9.3 = longer schedule; true payoff regime = more data).

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
    idx_path="cache/internvid_txtidx.npy", K=16384, dedup_cos=0.90)  # 0.90 from Part B (§7.6)
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
2.4a  Part B near-dup density  DONE (§7.6): low -> dedup_cos = 0.90 (a sparse, ~free mask)
2.4b  go/no-go pilot           DONE (§7.5): maxneg still climbing at 512 -> NOT saturated -> GO
2.4c  build the pool (§2): offline cache -> forward plumbing -> get_mask extension -> fp32/chunk
        full A/B: r1a (no pool) vs nR1_textbank (pool), matched steps, both DSL+raw
        report BOTH directions; headline t2v honestly = stale-queue side (§1)
2.4d  video FIFO queue only if t2v lags (momentum encoder; not exact)
```

**Status: 2.4a + 2.4b resolved → GO. Next action is 2.4c (build the pool).**

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

*Interpretation pressure-tested via a `deep-learning-researcher` review of the InfoNCE/sigmoid
gradient (2026-07-27); the corrections above are its findings, verified against `criterions.py`.*

---

## 7.5 Second pilot — the corrected metrics (measured 2026-07-27), and the lean flips

Re-ran the same pilot with the corrected probe. `maxneg_by_pool` = mean hardest-negative logit
over nested column prefixes; `neg_grad_mass` = mean over rows of Σσ(logit) on negatives:

| step | maxneg 64 → 128 → 256 → 512 | Δ / doubling | neg_grad_mass |
|---|---|---|---|
| 20  | −7.24 → −7.07 → −6.91 → −6.77 | ~+0.16 | 0.10 |
| 80  | −4.87 → −4.69 → −4.48 → −4.30 | ~+0.19 | 0.63 |
| 120 | −4.74 → −4.51 → −4.27 → −4.05 | ~+0.23 | 0.72 |
| 160 | −4.77 → −4.48 → −4.19 → **−4.00** | +0.29 / +0.29 / **+0.20** | 0.69 |

(Pilot val plateaued t2v R@1 ≈ 77, unchanged from the first run — the probe is read-only, as designed.)

**Reading — this is the key result, and it flips the prior lean:**

1. **The pool lever is NOT saturated at 512.** The hardest-negative logit is still rising at the
   256→512 step (~+0.2), and the per-doubling increment is roughly **constant** (≈+0.2), not
   decaying toward 0. That is the textbook `max-neg ~ log(pool size)` signature *still in its
   linear regime* — extrapolating 512→65k (≈7 doublings) predicts the hardest negative keeps
   getting ~+1.4 harder. **A larger pool would keep surfacing materially harder negatives.**
   This directly contradicts the regime-prior guess (§7.3) that an aligned-init fine-tune would
   sit on the flat part of the curve. It does not.
2. **`neg_grad_mass` ≈ 0.7, plateaued.** In-batch negatives are alive (not ≈0 ⇒ not the
   "everything dead, nothing to gain" case), but the in-batch hard-negative gradient has
   *stabilised* while harder negatives demonstrably exist just outside the 512-sample (point 1).
   That combination — in-batch gradient saturating **while** the pool still holds harder
   negatives — is precisely the SigLIP "batch-limited" signature. Mildly R1-favourable.

**Updated verdict: leans BUILD — conditional entirely on Part B.** The `maxneg` climb is real,
but the probe cannot tell whether the harder negatives a bigger pool adds are *genuine* or
*near-duplicate false-negatives* (which have high similarity and would inflate `maxneg` identically
— examination §3, §7.4). So:

| `maxneg` at 512 | near-dup density (Part B) | call |
|---|---|---|
| still rising *(measured: yes)* | **low** | **BUILD R1** (SigLIP-regime pool per §2; the climb is genuine hard negatives) |
| still rising *(measured: yes)* | **high** | **KILL / fix labels first** (the climb is mostly false-negatives; an un-deduped pool amplifies mislabeled repulsion) |

**Part B is now the sole decider and it did not run** — `r1_textfeat_dump.py` hit
`ModuleNotFoundError: No module named 'utils'` (run as a plain script, repo root wasn't on
`sys.path`). **Fixed this round** (insert repo root at import time; verified the import now
advances past the `utils` path to a normal dependency load). Re-run Part B (COMMANDS §3.3 ②),
then apply the table. Caveats unchanged: 10k zh pilot; `maxneg` beyond 512 is an extrapolation;
`neg_grad_mass`/margin still partly reflect memorisation on 9k examples.

---

## 7.6 Part B measured (2026-07-27) → **GO**

The path fix landed; `r1_textfeat_dump.py` built the model, encoded 4998 distinct captions with the
frozen text tower, and `r1_fn_density.py` measured near-duplicate density in the **projected,
normalized** space (the space the loss compares captions in):

| cosine ≥ | mean neighbors / caption | % with ≥1 neighbor | per-pair rate | ⇒ false-neg / query at K=65k |
|---|---|---|---|---|
| 0.90 | 0.166 | 7.3% | 3.3e-5 | ~2.2 |
| 0.95 | 0.040 | 2.0% | 8.0e-6 | ~0.5 |
| 0.99 | 0.002 | 0.14% | 4.0e-7 | ~0.03 |

**Density is LOW.** At cos≥0.95 (where two captions are effectively interchangeable) only ~2% of
captions have *any* near-neighbor, and a 65k pool would inject only ~0.5 false-negatives per query
(~0.03 at ≥0.99). Cross-referenced with §7.5: the `maxneg` climb (+0.2/doubling, still rising at
512) is therefore **overwhelmingly genuine hard negatives**, not false-negatives — the high-similarity
mass sits in the 0.3–0.9 "hard but correct" band, not the ≥0.95 "same caption" band. That is the
ideal case for a larger negative pool.

**Decision — GO (build R1, 2.4c).** This is the "still rising + low near-dup" row of the §7.5 table.

**Dedup mask: `dedup_cos = 0.90` (conservative), and it is nearly free.** At 65k the mask flags only
~2 pairs per query (~0.5 at 0.95); a precomputed sparse cosine-threshold mask is cheap to build and
apply, and it keeps the residual false-negative repulsion negligible while preserving the genuine
hard-negative signal. It stays a **prerequisite** (§7.4) — cheap, not optional.

**Caveats carried into the build:** one InternVid shard (5k of the distinct captions), zh-pilot text
distribution; boilerplate captions could repeat more across the full 10M than within 5k, so re-measure
density on the actual pool when the offline cache is built (it is a byproduct of the cache pass). The
**GO is not "R1 works"** — it is "the premise holds, build the arm and run the A/B." The kill criterion
still governs: **512→65k negatives < 0.5 R-Sum ⇒ stop** (§5).

---

## 8. Build landed (2.4c, 2026-07-27) — what to run, what's verified

The pool is implemented as a **config-gated, default-off** feature (same discipline as the probe:
no `criterion.text_bank` ⇒ bitwise-identical to today). Files:

| file | change |
|---|---|
| `models/criterions.py` | `vtc_loss_sigmoid(text_bank=None)` + `_text_bank_nll` — extends the `[B,B]` sigmoid matrix with `[B,K']` video–bankedtext columns; banked cols sharing a query idx are **positives**; near-dup cols (cosine to the query's own text ≥ `dedup_cos`) dropped. fp32 matmul. **All under `if text_bank is not None`** ⇒ off-path unchanged. |
| `models/mobileviclip_small.py` | `_init_text_bank` (loads `feats`/`idx` buffers iff `criterion.text_bank.enable`), `_sample_text_bank` (per-step deterministic subset, **re-projects with the current `projection_layer`** = exact bank), `forward` passes it only when `training`. |
| `models/backbones/.../mobileclip/text_encoder.py` | `encode_text(return_pre_proj=True)` returns the EOT-pooled feature **before** `@ projection_layer` (what the cache stores). |
| `experiments/r1_build_textbank.py` | offline cache: pre-projection feats `[K, C_pre]` fp16 + `caption_to_idx` `[K]` — reuses the Part-B-verified build path. |
| `experiments/configs/nR1_textbank_pilot.py` | pilot A/B **treatment** (control = `pilot_10k_r1.py`); only diff is `criterion.text_bank`. |

**Run:** COMMANDS **§3.3.1** — build the cache, then the A/B (`nR1_textbank_pilot` vs `pilot_10k_r1`,
matched steps). Startup logs `[R1] text bank ON: …`.

**Verified on CPU here:** the `_text_bank_nll` math (positives, dedup drops exactly the near-dup
false-negatives, fp16 round-trip) against an inline reference; all files parse; `caption_to_idx` in the
cache builder is byte-identical to the training `dataset.pt_dataset` one (bank idx = training idx);
`projection_layer` is a direct trainable attr of `text_encoder`. **GPU-unverified (disclosed):**
end-to-end forward wiring, buffer dtype/device movement under deepspeed, and the cache encode pass
(its build path is the one Part B already ran). If the run errors, paste the traceback.

**Report BOTH directions** (v2t hardened directly, t2v indirect — §1); apply the kill criterion (§5).

---

## 9. A/B — treatment arm measured (2026-07-27), control still pending

`nR1_textbank_pilot` (bank ON) ran clean: startup logged
`[R1] text bank ON: pool=(8999, 512) K'=16384 dedup_cos=0.9`, 10 epochs, no crash. Because the
pool holds all 8999 distinct captions and `K'=min(16384, 8999)=8999`, **every step appends the entire
bank** as sigmoid negative columns — the intended "big pool" stress. (Absolute `video-loss_vtc` ≈ 11–14
is *not* comparable to a no-pool run: the bank adds 8999 extra per-row sigmoid terms. Only held-out
recall is comparable.)

Treatment `ret_test/` per epoch:

| ep | txt_r1 | img_r1 | r_mean |
|----|--------|--------|--------|
| 0  | 79.1 | 80.7 | 90.97 |
| 1  | 75.9 | 78.5 | 89.00 |
| 2  | 75.3 | 77.7 | 88.40 |
| 3  | 75.4 | 79.1 | 88.62 |
| 4  | 74.9 | 78.6 | 88.58 |
| 5  | 74.8 | 79.3 | 88.65 |
| 6  | 75.0 | 79.4 | 88.67 |
| 7  | 75.4 | 79.1 | 88.88 |
| 8  | 75.1 | 79.5 | 88.80 |
| 9  | 75.4 | 79.5 | 88.87 |

Shape: epoch-0 (post-warmup, near the frozen init) is the peak at 90.97, then a dip to 88.4 (ep2) and a
slow climb back to ~88.9. This "fine-tune slightly below the strong zero-shot init on an easy in-domain
val" shape is expected to appear in **both** arms — which is exactly why the decision is the *delta*, not
the absolute. `sim_v2t/sim_t2v has nan` printed once during the epoch-5 eval (known benign fp16 eval
quirk, see TROUBLESHOOTING; it did not crash and the epoch produced numbers).

**Cannot call go/kill yet.** The kill criterion (§5) is defined on **pool − no-pool**; the control
(`pilot_10k_r1`, identical setup minus `criterion.text_bank`) has not run. Next: run the control (matched
10 epochs, ~38 min), then compare per-epoch `txt_r1`/`img_r1` and best-epoch R-Sum. Kill if
pool − no-pool < 0.5 R-Sum.

### 9.1 Control measured (2026-07-28) → **A/B verdict: GO, kill criterion cleared**

`pilot_10k_r1` (no bank, seed 42, same 10 epochs) ran clean. Side-by-side `ret_test/`:

| ep | txt_r1 T | txt_r1 C | img_r1 T | img_r1 C | r_mean T | r_mean C | Δr_mean |
|----|---------|---------|---------|---------|---------|---------|--------|
| 0  | 79.1 | 77.7 | 80.7 | 79.6 | 90.97 | 89.83 | +1.14 |
| 1  | 75.9 | 70.2 | 78.5 | 75.4 | 89.00 | 85.35 | +3.65 |
| 2  | 75.3 | 69.7 | 77.7 | 74.8 | 88.40 | 85.08 | +3.32 |
| 3  | 75.4 | 69.7 | 79.1 | 75.3 | 88.62 | 85.52 | +3.10 |
| 4  | 74.9 | 70.7 | 78.6 | 75.3 | 88.58 | 86.12 | +2.46 |
| 5  | 74.8 | 71.6 | 79.3 | 75.5 | 88.65 | 86.57 | +2.08 |
| 6  | 75.0 | 71.8 | 79.4 | 76.4 | 88.67 | 86.70 | +1.97 |
| 7  | 75.4 | 72.9 | 79.1 | 77.3 | 88.88 | 87.12 | +1.76 |
| 8  | 75.1 | 72.3 | 79.5 | 77.1 | 88.80 | 86.98 | +1.82 |
| 9  | 75.4 | 72.6 | 79.5 | 77.1 | 88.87 | 87.03 | +1.84 |

**Final-epoch R-Sum** (txt r1/5/10 + img r1/5/10): T = 75.4+91.8+95.4+79.5+93.9+97.2 = **533.2**;
C = 72.6+90.3+93.6+77.1+92.6+96.0 = **522.2** → **ΔR-Sum = +11.0**. Kill criterion (§5) was
*pool − no-pool < 0.5 R-Sum ⇒ stop*. We are at **+11.0 (~22×)**, and the bank leads at **every
epoch in both directions** (v2t +2.8, t2v +2.4 at ep9). **Verdict: GO — R1 clears its kill
criterion decisively.**

**Why this is trustworthy (not a temperature/noise artifact):**
- Retrieval R@k ranks by **cosine on L2-normalized features** → **invariant to logit_scale /
  logit_bias**. The bank's extra gradient on the SigLIP temperature therefore *cannot* manufacture
  the gap; it reflects genuinely better-separated embeddings.
- The gap is **systematic** (bank ahead all 10 epochs, both directions) — noise fluctuates sign.
- The bank marks each query's own caption (present in the pool) as a **positive** and drops near-dups
  (cos ≥ 0.90), so the model is not being pushed away from its own/paraphrase captions.

**Honest caveats (carry into any writeup):**
- **Both arms sit below their epoch-0 (frozen-init) peak** on this easy in-domain 1k val. At pilot
  scale the bank's effect manifests as *reduced fine-tuning degradation / a higher, faster plateau*,
  **not** an absolute gain over the zero-shot init. The regime where R1 is designed to pay off
  (large data, many steps, in-batch negatives genuinely information-limiting) is **not** what a
  9000-clip / 17-batch pilot tests.
- **Env line to confirm:** the control log printed `torch 2.5.1+cu118` (deepspeed "Skipping import
  of cpp extensions"); confirm the treatment ran the **same** torch (`python -c "import torch;
  print(torch.__version__)"`) so the A/B isn't confounded by an env change between the two runs.
- **Convergence-vs-plateau open question:** control is still climbing at ep9 (85.08→87.03) while the
  bank plateaus flatter/higher (88.87); the LR schedule has decayed to ~0, so neither gains much
  more, but "does no-pool catch up with more steps" is untested.

**Next (recommended before any scale-up):** one **seed-43 confirmation** of both arms
(`nR1_textbank_pilot_s43.py` vs `pilot_10k_r1_s43.py`, committed) to rule out nondeterminism; if the
gap reproduces, scale the A/B to more InternVid data / more steps — the setting R1 actually targets.

### 9.2 Seed-43 confirmation (2026-07-28) → **reproduced, ΔR-Sum = +13.1 (seed-42 was +11.0)**

Both seed-43 twins ran to completion (env parity **confirmed**: both logged `torch 2.5.1+cu118`).
The bank arm again logged `[R1] text bank ON: pool=(8999, 512) K'=16384 dedup_cos=0.9`; the control
carried no `criterion.text_bank`. Per-epoch `ret_test/` `r_mean` (T = treatment/bank, C = control):

| ep | T r_mean | C r_mean | Δ |
|----|----------|----------|-----|
| 0  | 90.85 | 90.20 | +0.65 |
| 1  | 89.52 | 85.43 | **+4.09** |
| 2  | 89.23 | 84.92 | +4.31 |
| 3  | 88.82 | 85.37 | +3.45 |
| 4  | 88.68 | 86.13 | +2.55 |
| 5  | 88.65 | 86.47 | +2.18 |
| 6  | 88.78 | 86.30 | +2.48 |
| 7  | 88.93 | 86.82 | +2.11 |
| 8  | 88.92 | 86.72 | +2.20 |
| 9  | 88.97 | 86.78 | +2.19 |

Final-epoch **R-Sum** (txt r1/r5/r10 + img r1/r5/r10):
T `75.8+92.1+95.3+79.5+93.8+97.3 = 533.8`; C `72.4+90.0+93.3+76.6+92.5+95.9 = 520.7` → **ΔR-Sum
= +13.1** (kill criterion is < 0.5; both seeds clear it by ~20×). ep9 directional deltas: v2t
txt_r1 **+3.4**, t2v img_r1 **+2.9**.

**Two seeds now agree** (Δ = +11.0 and +13.1), and seed-43 makes the *mechanism* visually explicit:
the **control collapses at ep1** (90.20 → 85.43 r_mean) and never fully recovers, while the bank arm
barely dips (90.85 → plateau ~88.9). This is exactly the "reduced fine-tuning degradation / higher,
faster plateau" story from §9.1's honest caveat — the bank protects the frozen-init separation that
naive fine-tuning erodes. The env-parity caveat is now **closed**; the "does no-pool catch up with
more steps" question is addressed by the next experiment (§9.3, longer schedule on the same pilot),
and the genuine payoff regime (more data) still needs the maintainer to confirm what larger InternVid
subset is staged on the server.

**Verdict stands: GO, scale up.** See [`notes/live/from-claude.md`](live/from-claude.md) for the
next server step.

### 9.3 Next experiment — longer schedule (does the control catch up with more steps?)

The cheapest well-scoped follow-up that needs **no new data** (whose staging on the server is
unconfirmed): re-run the matched pair on the **same 9000-clip pilot** with a **40-epoch** cosine
schedule (4× the pilot). This isolates the one open question — whether the control's slow climb
eventually closes the bank's lead, or whether the bank keeps a durable edge. Committed configs
(seed 42, identical to the 10-epoch pair except `scheduler.epochs = 40`):

| arm | config | output_dir |
|-----|--------|------------|
| treatment (bank) | `experiments/configs/nR1_textbank_pilot_long.py` | `experiments/runs/nR1_textbank_pilot_long` |
| control (no bank) | `experiments/configs/pilot_10k_r1_long.py` | `experiments/runs/pilot_10k_r1_long` |

Reuses the existing `textbank_pilot_*` cache — no rebuild. ~4× the ~38-min pilot wall-clock per arm
(~2.5 h each). **Read:** track per-epoch `r_mean` Δ. If the bank's lead **persists or widens** at 40
epochs → durable, not just faster convergence → proceed to the true scale-up (more data). If the
control **closes to < 0.5 R-Sum** by ep~30-40 → the bank mainly *accelerates* and the pilot-scale win
is a convergence-speed effect, which reframes (not kills) the claim.

**True payoff regime — runs on the H100, not the V100:** the setting R1 actually targets is more
InternVid data (so in-batch negatives become genuinely information-limiting). The config defines
`internvid_all` (11 `InternVId-FLT_*` shards) — but **the V100 server (`101vector`) only has the 10k
subsamples; the full InternVid dataset lives only on the H100 server.** So the large-data A/B is an
**H100 experiment** (pointing `train_corpus = "internvid_all"` at the V100 would fail — data absent).
Every pilot-scale run above, including §9.3's 40-epoch pair, stays on the V100. Before wiring the
large-data A/B on the H100, build a matching larger text-bank cache there (the pilot cache
`textbank_pilot_*` is 10k-only). All-server data split recorded in memory (`server-data-split`).
