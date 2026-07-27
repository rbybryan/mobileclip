# R1 — SigLIP-regime negative-pool arm: implementation plan

**Date:** 2026-07-27
**Reframed per** [`notes/r1-examination.md`](r1-examination.md) §6: sell this as **"contrastive
training at small batch, fixed with a SigLIP-regime large negative pool"**, *not* as "the
mathematically-exact bank". The exactness is a real but **v2t-side** engineering nicety
(examination §1); the number we lead with (t2v) still rides a stale video queue.

**Status of the pre-checks (examination §7):** Part A done — exact-duplicate FN ≈0, so
**exact-`idx` dedup can be skipped**. Part B (near-duplicate density) decides whether a
cosine dedup mask is needed; run `experiments/r1_textfeat_dump.py` → `r1_fn_density.py --feats`
(COMMANDS §3.2). **The go/no-go is neither — it is the in-batch saturation probe below.**

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
- **Near-dup dedup:** add a precomputed cosine-threshold mask **iff** Part B shows high near-dup
  density; otherwise skip.

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
