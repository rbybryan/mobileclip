# R1 (exact frozen-text negative bank) — critical examination

**Date:** 2026-07-24
**Examines:** `notes/neural-inspired-round2.md` §3 R1 (the post-GATE-0 recommended next step).
**Verdict:** worth building — but the note **oversells one thing** and **under-specifies two**,
and one **premise is a literature bet, not a near-certain win**. All four points below are
checked against the code on `main`, not against the proposal's own prose.

---

## 0. What R1 gets right (verified, and stronger than stated)

**The exactness design is not just clever — the code makes it *necessary*.**
The training config sets `open_text_projection=True`
(`scripts/pretraining/clip/mobileviclip_small/config.py:53`), so the final text projection
**does** train. Therefore caching the *final* text embeddings would be stale, and caching the
**pre-projection** features + re-applying the current projection each step is the *only* exact
form. R1 picks exactly that. ✅

**And everything before the projection is genuinely frozen** — `freeze_text=True` and only
`projection_layer` is unfrozen (`mobileviclip_small.py:74-80`, verified earlier). So the
pre-projection feature of a caption is, in principle, constant for the whole run.

That much holds. Now the problems.

---

## 1. OVERSOLD: "exact, no momentum encoder" is true — but for the **wrong direction**

R1's marquee property is that the bank is *mathematically exact* (no staleness, no momentum
encoder). **That property attaches to the retrieval direction we do _not_ headline.**

The loss is symmetric (`criterions.py:137-139`): `loss = (loss_i2t + loss_t2i)/2`, both terms
built from the same `[B,B]` matrix `sim_v2t = vision_proj @ text_proj.T` (`get_sim:54`).

- `loss_i2t = −Σ log_softmax(sim_v2t, **dim=1**)·targets` — softmax over the **text** axis:
  each **video** is the query, ranked over texts. Adding banked **text** columns lengthens this
  axis ⇒ **hardens `loss_i2t` = video→text.**
- `loss_t2i` uses `sim_t2v = sim_v2t.T`, softmax over the **video** axis: each **text** is the
  query, ranked over videos. A text bank adds **no video columns** ⇒ **`loss_t2i` = text→video
  is untouched.**

Our headline metric is **t2v R@1 = `img_r1` = text→video** (`experiments/README.md §3.4`).
That is `loss_t2i` — the term the exact text bank **does not** harden. To harden it you need
banked **video** embeddings, and the vision tower is **not** frozen (`freeze_vision=False`,
config `:51`) ⇒ those drift ⇒ you are back to a **stale MoCo/XBM queue with a momentum
encoder** — precisely the well-established, non-novel thing.

**So the clean "no momentum encoder, exact" story covers v2t; the number we lead with (t2v)
gets the ordinary stale-queue treatment.** The note flags the asymmetry in one line but does
not connect it to "the exactness is on the non-headline side." It should be front and center.

**Honest counterweight (do not over-swing the criticism):** both terms train the *same* shared
encoders + projection, so hardening v2t improves the joint geometry and *can* lift t2v
indirectly — CLIP-scale evidence is that scaling negatives helps because it improves
representation quality, not because of the axis bookkeeping. So R1 may still move t2v; it just
does so as an **empirical bet through the shared representation**, not via the clean guaranteed
mechanism the note advertises. Report **both** directions and don't headline the exactness.

---

## 2. FRAGILITY: exactness holds only because every text-path dropout is p=0

`requires_grad=False` does **not** disable dropout, and `tasks_clip/pretrain.py:44` calls
`model.train()`, so the frozen text tower runs in **train mode**. If any dropout in the text
path had `p>0`, the pre-projection feature would vary per step with the dropout mask and the
bank would **not** be exact.

Checked: `mobileclip_s2.json` exposes **no dropout key** (`embed_dropout` defaults to `0.0`,
`text_encoder.py:26,52,160`), and the text RepMixer's `drop` also defaults `0.0`. **So it is
exact today** — but by *configuration*, not by *construction*. A future `embed_dropout>0` (or a
config swap) silently breaks exactness with no error.

**Fix (cheap):** when building the bank, either `assert` every dropout `p` in the text path is
0, or call `self.text_encoder.eval()` for the cached-feature computation. Make the invariant
explicit instead of latent.

---

## 3. UNDER-SPECIFIED: the sketch's loss form contradicts the repo's actual loss

R1's sketch ends with `F.cross_entropy(logits, arange(B))` — a single-positive-on-the-diagonal
loss. **The repo does not use that.** `vtc_loss` uses `get_mask(idx)` soft labels
(`criterions.py:130-139, 287-304`): with `caption_idx=True`, the target is
`eq(idx, idx.T)` row-normalized — **multi-positive**, because InternVid clips sharing a caption
are all positives.

Consequence for the bank, and it is not cosmetic: when you concatenate banked text columns,
**any banked caption sharing the query clip's `idx` is a positive**, not a negative. A plain
`arange` target (or an all-negatives bank) **injects those as false negatives** — which is
exactly the failure the "offline dedup mask" freebie is meant to prevent, but the sketch never
wires the mask into the target. **The bank must extend `get_mask`, not sit behind
`cross_entropy(arange)`.** This is the single most likely way a naive R1 implementation quietly
underperforms.

(The fp16-overflow fix R1 already gives — fp32 logits + max-subtract + chunk over the bank
dim — is correct and confirmed necessary: the repo already casts `sim` to fp32 before
`log_softmax` at `criterions.py:132-133`, but the `v @ t_bank.T` matmul at logit-scale 100 is
where a `[512, 65536]` fp16 tensor overflows first.)

---

## 4. PREMISE RISK: "30–60× below SigLIP saturation ⇒ big lever" was measured in a different regime

R1's size-of-prize argument is SigLIP's batch sweep ("strongly batch-limited below 16k,
saturates ~32k"). That sweep is **from-scratch, billion-scale image-text pretraining.** Our
regime is the opposite corner: **10M pairs, 3 epochs, both towers initialised from an
already-aligned `mobileclip_s2`** (pilot opening loss ≈1.13, not the ~13.9 of a random start —
`config.py:126-136`). The DL-feasibility review (round-2 provenance) already warned that
batch-size effects are largest in from-scratch long-schedule regimes; a short fine-tune from an
aligned init may sit **much closer to its own saturation** than the 30–60× headline implies.

So the expected gain is a **genuine open question**, not the "+1.5 R-Sum, mode" the enthusiasm
suggests. R1's own instrumented kill-check is the right hedge and should gate the spend:
log the **softmax mass on banked vs in-batch negatives** (<5% ⇒ inert) and the **rank of the
positive among the 512 in-batch negatives** (already rank-1 for most ⇒ saturated at 512 ⇒ a
bigger pool cannot help).

---

## 5. Cheapest first experiment — zero training, zero GPU, and it also de-risks §1/§3

> **Runnable now:** `experiments/r1_fn_density.py` (self-tested; `caption_to_idx` verified byte-identical to the repo). Part A (exact-duplicate density) needs only the anno JSONs — commands in `COMMANDS.md` §3.2. Part B (embedding near-duplicate density) consumes a precomputed text-feature `.npy`; `--help-feats` prints the generator snippet.

Before building the bank, one offline measurement sizes both the opportunity and the risk, and
it is only possible **because** the text space is static (so it doubles as a smoke-test of the
exact-bank premise):

> Encode a large InternVid caption sample once with the frozen text tower and measure the
> **false-negative / near-duplicate density** at several cosine thresholds. Measure in the
> **projected, normalized** space — that is the space the loss compares captions in
> (`sim = v @ text_proj.T`), so it is where "the loss can't tell two captions apart" is defined.
> (Pre-projection matters only for *bank exactness* per §0, not for this density measurement.)

It answers three things at once:
- **Opportunity (§4):** how many *distinct* captions exist ⇒ how many genuinely-hard negatives
  a 65k pool would actually contain (vs near-duplicates of each other).
- **Risk (§3):** how large the offline dedup mask must be at K = 16k / 65k ⇒ how bad the
  false-negative problem is if you *don't* wire the mask.
- **Premise (§0/§2):** it exercises exactly the "cache the frozen pre-projection feature" path,
  on a small scale, before committing to the full pipeline.

If near-duplicate density is high, R1 must ship the dedup mask from day one (not "if t2v lags");
if distinct-caption count is low, the effective pool is smaller than 65k and the lever shrinks.

---

## 6. Net recommendation

**Pursue R1, but reframe it and sequence the checks:**

1. **Lead the story with "SigLIP-regime training at small batch", not "exact bank".** Sigmoid
   loss is already implemented (`r1a_sigmoid.py`) and directly addresses the batch-size
   limitation R1 targets; bundle temperature + sigmoid + pool as *one* SigLIP-regime arm rather
   than selling "exactness" as the headline. The exactness is a real but **v2t-side** engineering
   nicety (§1), not the contribution.
2. **Run the §5 offline density probe first** (free), then the instrumented in-batch saturation
   check (cheap), *before* the full bank. Either can kill (d) early.
3. **When implementing:** thread the bank through `get_mask` (§3), and make the no-dropout
   invariant explicit (§2).
4. **For the headline t2v number, be upfront** that R1 = stale video queue there (§1); the exact
   part is v2t. If t2v is the number that must move, R1 alone may not be the lever — that is the
   honest version of "add a video FIFO only if t2v lags."

**Kill criterion unchanged and correct:** 512→65k negatives < 0.5 R-Sum ⇒ (d) misdiagnosed,
stop. The examination's contribution is to make that verdict *cheaply reachable* (§5) and to
stop the two silent-underperformance modes (§2, §3) from muddying it.

---

*All code references checked 2026-07-24 against `models/criterions.py`,
`models/backbones/internvideo2/mobileclip/text_encoder.py`,
`scripts/pretraining/clip/mobileviclip_small/config.py`, `tasks_clip/pretrain.py` on `main`
(`e199f20`).*
