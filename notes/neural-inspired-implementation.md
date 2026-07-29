# Neural-Inspired Implementation Plans (feasible, code-grounded)

**Date:** 2026-07-23

**Scope.** `notes/neural-inspired-directions.md` *ranked* eight neuroscience-inspired ideas
(N1–N8). This note is the executable follow-through: **which of them can actually be built
on this codebase, in what order, with what exact touchpoints, configs, tests, and kill
criteria.** It is written to be run from the server (durable-markdown convention, `CLAUDE.md
§0`) and to slot into the existing code-dev line of `experiments/ROADMAP.md` (第二线) — not
to open a parallel schedule.

> ⚠️ **Read [`notes/neural-inspired-round2.md`](neural-inspired-round2.md) (2026-07-24) before
> executing anything here.** Round 2 corrects three items in this file:
> **(1)** A1 surprise pooling is weaker than claimed — *any* reweighting of a set stays
> order-blind (round 2 F1/F2), so A1 can only exploit appearance saliency, never temporal order;
> **(2)** B1's config sketch sets `criterion["loss_weight"]`, which by round 2 §5 P2 **does
> nothing to the optimization** — `tasks_clip/pretrain.py:114` sums the loss dict unweighted
> and `loss_weight` (`:49`) only builds logging meters, so the weight must be applied inside
> `forward`;
> **(3)** a mandatory zero-training gate (**GATE-0**, round 2 §2) precedes all temporal work,
> because our benchmark suite may not be able to measure these ideas at all.
> **GATE-0 has since run and closed (2026-07-27) — see the ✅ block in §0 for the outcome and
> for how the follow-through diverged from round 2 §2c's prescription.**

Two hard constraints are inherited from the repo and govern every plan below:

- **C-machine (two-level validation, `ROADMAP.md` / `README.md`).** V100 + 10k pilot does
  **link go/no-go only** — does it run, does loss drop, do the new parameters receive
  gradient / leave their init. It **cannot rank methods** (eval saturates, BN is confounded,
  binomial SE too large). Real A/B ranking + winner final-train happen on H100 + full 10M.
  Every item therefore carries *two* gates: a **link-test** and a **full-scale kill criterion**.
- **C-switch (component convention, C3 in `paper-plan.md`).** Every new component is a
  **default-off switch** whose off-path is **bitwise-identical** to the locked baseline, and
  whose on-path is **zero-initialized so step-0 == baseline** (the `TemporalAttentionPool`
  zero-init proj, `mobileviclip_tiny.py:1338-1339`, is the reference pattern). No component
  is allowed to regress the baseline at initialization.

Anchored to code checked 2026-07-23; **line numbers re-verified 2026-07-29 after PR #222**
(the C2 T-agnostic landing shifted most model-file lines by +7…+15). Re-locate by symbol if
they drift again. `mobileviclip_tiny.py` is cited; `_small.py` mirrors it at larger width —
**except A1**, whose `"surprise"` branch exists only in `mobileviclip_small.py:1397-1402`
(`_tiny.py:1258-1263` still raises `ValueError` for anything but `mean`/`attn`).

---

## 0. Feasibility tiers (what can start now vs what is gated)

*(Status column refreshed 2026-07-29. GATE-0 is closed; A1 is built and piloted; B1 remains
the unstarted headline. `notes/neural-inspired-round2.md` supersedes several claims below —
each one is marked inline.)*

| Tier | Items | Depends on | Start? |
|------|-------|-----------|--------|
| **A — built; A1 now a 1M-tier arm** | **A1** surprise-weighted pooling (N2 aggregator), **A2** graded temporal windows (N4) | code: nothing — pure architecture, C3-class components. ~~GATE-0~~ **✅ closed 2026-07-27** | **A1: built, 10k ✅, now an arm in the 1M verification tier (`nA1`). A2: not built** |
| **B — pilot now, full on R2** | **B1** geometry-preserving distillation (N1) | R2 teacher-embedding pipeline (`ROADMAP` 2.2); pilot uses an **online** InternVideo2-S teacher | Loss+wiring now; full A/B rides on 2.2 — **still ⬜, and ROADMAP G1 flags this as the plan's most severe gap** |
| **C — gated (R4-class)** | **C1** predictive-coding structured frame gating (N2 efficiency half); **N3** two-stream fallback | ~~C2 T-agnostic ops (`ROADMAP` 2.1)~~ **✅ satisfied 2026-07-24 (PR #222)** + on-device latency/export harness — ⚠️ **U1 export smoke now says joint attention is NOT an export blocker (ONNX + Core ML pass at all T), so the deployment premise under Tier C has weakened** | Only after convergence + dual trigger |
| **Skip** | N5 (pure adaptive-T), N7 (foveation), N8 (SNN) | — | See `neural-inspired-directions.md §2` |

> ### ✅ GATE-0 closed 2026-07-27 — outcome and what it did to this file
>
> Round 2 §2 made GATE-0 a mandatory zero-training gate "before all temporal work". **It has
> since run and closed** (`experiments/gate0_results.md` §8; ROADMAP 状态速览). Three
> independent lines of evidence agree that **the official MobileViCLIP weights do not encode
> temporal direction**: shuffling the 8 frames costs ≈0 (t2v +0.1/+0.2) while repeating a
> single frame costs −13; the k=3 temporal kernels' antisymmetric fraction is 0.5011 vs a
> random null of 0.4999 (indistinguishable from untrained); and the temporal gate sits at
> 6.1% of the spatial gate from the same 1e-5 init. Order is worth ≈0 to this model; **frame
> diversity is worth ≈13**.
>
> **Consequences for the items here, and they are not the ones round 2 §2c predicted.**
> §2c's matching row prescribed "R2 becomes the *only* temporal proposal worth funding, and
> it must be measured on RTime; everything else temporal is dead." The actual decision taken
> (gate0_results §8, `notes/order-sensitive-eval.md`, ROADMAP) **diverged on both halves**:
> RTime/SSv2-MC were **shelved** (⏸ 暂缓 — buying a whole video-scrape round to measure order
> is not worth it when three cheaper probes already agree), and the freed budget went to
> **R1, the frozen-text negative bank**, which attacks bottleneck **(d)** and *is* measurable
> on plain MSR-VTT retrieval. So the operative reading is "**order is not the lever here —
> stop paying to measure it, and spend on (d) instead**", not "fund R2 and only R2".
> §2c is amended in place in round 2; see the note there.
>
> Net for this file: A1 and A2 are no longer gate-blocked, but the gate's answer also removes
> the temporal-order motivation that round 1 attached to them. They survive on
> appearance-saliency / receptive-field grounds only, which is a much smaller claim.

Recommended order (as executed): **A1/A2 code + link-test** → **GATE-0 ✅** → **R1 text
negative bank** (round 2 namespace — see the ID warning below; now the most-evidenced item,
`notes/r1-implementation-plan.md`) → **B1 with the R2 teacher pass**, still the headline
train-time novelty and still unstarted → **C1 only if triggered**. B1 is the one worth the
most and is now the plan's biggest hole (ROADMAP **G1**: C1/DR is billed as the main line yet
is the only contribution layer with zero code). A1/A2 are cheap insurance — but see A1's
round-2 demotion below before budgeting for either.

---

## Tier A — start now (ROADMAP 第二线, C3-class)

Both A-items are the same shape as the existing R1 component A/Bs
(`experiments/configs/r1a_sigmoid.py`, `r1b_attnpool.py`): one default-off switch, a
zero-init on-path, a 10k link-test, then queued for the full-scale A/B beside R1.

### A1 · Surprise-weighted temporal pooling (N2 aggregator, teacher-free)

> **STATUS 2026-07-29 — A1 is built, piloted and closed; this section is now a record, not a
> build order.** The module ships as `neural_inspired/temporal_surprise_pool.py`, the config as
> `experiments/configs/nA1_surprisepool.py`, and the 10k link-test **passed** on 2026-07-24
> (`alpha=-4.74e-5` ≠ 0, `gamma=1` in place, `key.weight` L2 13.06 ≈ kaiming 13.07 — see
> `experiments/pilot_10k_results.md`, ROADMAP 状态速览). What remains is the ranking arm, now
> scheduled in the **1M verification tier** as `nA1` (4 arms × 2 seeds, `train_corpus=internvid_v1`,
> batch 128, 3 epochs — `notes/verification-tier-1m-zh.md`), not on the 10M full run.
> ⚠️ **Read `verification-tier-1m-zh.md` §6 before trusting any A1 ranking number:** the first
> 1M result showed that 10× more data buys only +0.82 raw r_mean, inside the ±1.6 noise band,
> so **MSR-VTT 1k cannot currently rank arms at this scale.** Paired query-level bootstrap CI
> (ROADMAP **G4**) was the prerequisite for reading A1's arm — and it **landed 2026-07-29**
> (`experiments/paired_bootstrap.py`, `collect_results.py --compare A B`, `COMMANDS.md` §3.9).
> Resolution improves ~11×: paired 95% CI halfwidth on R-Sum is **±1.5** vs ±17 unpaired, so a
> +0.8 R@1 gap is now decidable. **Read A1's arm as "does the CI exclude 0", not against the
> old ±1.6 band.** Caveat: per-query ranks are only dumped from 07-29 onward, so arms evaluated
> before that need one ~1 min re-eval.

**Idea.** The head currently mean-pools 8 frame embeddings (`mobileviclip_small.py:1436-1453`);
P3 adds a learned attention pool (`temporal_pooling="attn"`, `TemporalAttentionPool:1456`).
Add a **third** mode that weights each frame by its *prediction error* — a teacher-free
proxy for "surprise" (predictive-coding motivation, `neural-inspired-directions.md §N2`):
frames that deviate from the temporal context carry more information than static ones.

> ⚠️ **Round-2 correction (§6.1) — read this before believing the motivation above.**
> A1 reweights a *set*. By round 2 F1 the head is unconditionally order-blind, so **any**
> weighting scheme — A1's and the attention pool it must beat alike — is invariant to frame
> permutation. A1 can therefore exploit **appearance saliency only, never temporal order**.
> Round 1 sold it as "the aggregator half of the predictive-coding story, unifying the
> efficiency headline with the mean-pool fix"; that framing is **withdrawn**. The order fix is
> round 2's **R2 order-recoverable readout**, not A1. A1 survives as a cheap appearance-saliency
> ablation and nothing more — which is consistent with the novelty pin below.

**Touchpoint.** `FastViT.forward` pooling branch (`mobileviclip_small.py:1436-1453`) + the
builder that reads `model.vision_encoder.temporal_pooling`. New module beside
`TemporalAttentionPool`.

**As-built module** (`neural_inspired/temporal_surprise_pool.py` — the sketch that used to sit
here was **wrong and is deleted**; it initialized *both* gates to zero, which is a
zero-gradient fixed point the module can never leave. PR #220 fixed it by starting `gamma` at
1. Do not resurrect the old sketch):

```python
class TemporalSurprisePool(nn.Module):
    """Weight T frame embeddings by temporal prediction error (surprise).
    alpha init 0 => output == mean at step 0 (C-switch);
    gamma init 1 (NOT 0) => pooled != mean, so alpha has a live gradient."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)     # learned error space
        self.gamma = nn.Parameter(torch.ones(1))       # sharpness — MUST be nonzero
        self.alpha = nn.Parameter(torch.zeros(1))      # gate — 0 => exactly mean-pool
        self.eps = eps
    def forward(self, x):                              # x: [B, T, C]
        mean = x.mean(1)                               # [B, C]  gist / prediction
        err = self.key(x - mean.unsqueeze(1))          # [B, T, C]  prediction error
        surprise = err.pow(2).mean(-1)                 # [B, T]  error energy
        w = torch.softmax(self.gamma * surprise, dim=1)
        pooled = torch.einsum("bt,btc->bc", w, x)      # [B, C]
        return mean + self.alpha * (pooled - mean)
```

**Config** (as merged, `experiments/configs/nA1_surprisepool.py`):

```python
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa
model["vision_encoder"]["temporal_pooling"] = "surprise"
# different_lr is present but COMMENTED OUT in the merged config (lr=1e-4 if re-enabled);
# the pilot ran on the base lr. Re-enable only with a matched control arm.
output_dir = "experiments/runs/nA1_surprisepool"
```

**Link-test (10k, go/no-go) — PASSED 2026-07-24.** Runs to completion; loss decreases; ckpt
`temporal_pool.alpha` has left 0 (received gradient). Extend
`tests/test_optimization_components.py`: with `temporal_pooling="mean"` the forward is
bitwise-equal to baseline; with `"surprise"` and `alpha=0`, output `allclose` to mean-pool.
Note the pilot-scale caveat recorded in from-claude (8): `gamma`/`key` barely move at 10k
because their gradients are scaled by `alpha`~1e-5, and `alpha`'s **sign** is noise at that
magnitude — only "left 0" is readable, not direction.

**Full-scale kill criterion (T1 — beat the non-neuro twin).** The honest baseline for A1 is
**`temporal_pooling="attn"`** (the learned attention pool). Surprise-pooling is only worth
keeping if it **≥ attn on aggregated R-Sum** at matched steps — otherwise the attention pool
already captures the temporal saliency and A1 is a weaker special case. Cut if not ≥ attn.

**Novelty status (pinned 2026-07-24 — A1 is NOT a contribution).** Every ingredient is
prior art: weighted frame pooling = the attention-pooling family (our own C3 attn pool,
X-Pool, CLIP4Clip seqTransf); "far-from-the-gist frames matter" scoring = novelty /
representativeness scores in video summarization (centroid-distance keyframing, DR-DSN's
representativeness reward) and **Itti & Baldi's Bayesian surprise saliency (2005/2009)** —
the citation a reviewer will reach for first; the predictive-coding story (Rao & Ballard;
PredNet) is borrowed motivation, not mechanism novelty; the zero-init gate is standard
(ReZero/LayerScale). What is ours is only this specific parameterization inside a mobile
video CLIP. **Paper ceiling:** if it wins the T1 A/B — one ablation row + one sentence
("a predictive-coding-inspired pooling variant"), citing the lineage above; if it ties or
loses — remove per the kill criterion. It never enters the contribution list (paper-plan
C1–C4 does not include it, by design).

### A2 · Graded temporal receptive windows (N4, free ablation)

**Idea.** Every RepMixer's temporal branch is a depthwise `Conv1d(k=3)` over time
(`RepMixer.forward:583-606`, `mixer_1d:558`), the *same* short window at every depth. Grow
the temporal window with depth (Hasson hierarchy, `§N4`): short early, long late — via a
**dilation** schedule over the **three RepMixer stages** (`token_mixers=("repmixer",
"repmixer","repmixer","attention")`, `:200` — stage 4 is attention with `temporal_embedding`,
it has no 1D conv), keeping k=3.

**Touchpoint.** The stage builder that instantiates `mixer_1d` (pass `dilation` per stage).

> ⚠️ **Corrected 2026-07-29 — the round-1 claim here was half wrong, and the real defect is
> worse than the one it flagged.** Round 1 said the 2D `MobileOneBlock` "already threads
> `dilation` end-to-end incl. reparam" and that the open question was whether
> `MobileOneBlock_1d` *exposes* `dilation`. Code check: `MobileOneBlock_1d` **does** expose it
> (`mobileone.py:390`, stored `:420`), so that question is answered — but **exposing is not
> threading**. `dilation` reaches only the inference-mode `reparam_conv` (`:177` 2D, `:444` 1D)
> and the reparameterize-time rebuild (`:247`, `:522`). The **train-time branch builder
> `_conv_bn` silently drops it**: `nn.Conv2d(...)` at `mobileone.py:347` and `nn.Conv1d(...)`
> at `:621` are both constructed with no `dilation` argument. So setting `[1,2,4]` today gives
> a train-time receptive field of 3 and a reparameterized one of 3/5/9 — the branches compute
> **different functions**, and the reparam round-trip breaks silently. **A2 is therefore not a
> config-only change**: thread `dilation` (+ dilation-aware padding) into both `_conv_bn`
> bodies first. The round-trip test below is exactly the gate that catches this, so run it
> before trusting any A2 number.

The reparam `id_tensor` centre tap is dilation-invariant, so `RepMixer.reparameterize:622`
stays valid once `_conv_bn` is fixed. The pretrained
`mobileclip_s2.pt` has **no** temporal 1D convs (added fresh), so no checkpoint-load conflict;
the residual `mixer_1d − norm_1d` is ~0 at init, so the baseline path is preserved.

**Config** (create `experiments/configs/nA2_gradedwindows.py`):

```python
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa
model["vision_encoder"]["temporal_window"] = "graded"     # default "fixed"
model["vision_encoder"]["temporal_dilation"] = [1, 2, 4]  # over the 3 RepMixer stages
output_dir = "experiments/runs/nA2_gradedwindows"
```

**Link-test.** `temporal_window="fixed"` off-path bitwise-equal to baseline; **reparam
round-trip** unit test (train-mode forward == reparameterized forward) at each stage's
dilation — extend the existing reparam test.

**Full-scale kill criterion.** Free ablation, low bar: must **not regress** overall R-Sum and
should show its gain where a long temporal receptive field matters (**SSv2-MC** and other
temporal-sensitive sets). If flat everywhere, keep as a one-line negative result, not a knob.
Main engineering risk = **threading `dilation` into both `_conv_bn` bodies** (see the
correction above); the reparam round-trip test is the gate. Round 2 §6.2 adds a second caveat:
A2's expected SSv2-MC win must be **re-read against the GATE-0 audit, not assumed**.
**GATE-0 has now reported (✅ 2026-07-27) and the news for A2 is bad**: order is worth ≈0 to
this model while *frame diversity* is worth ≈13, and SSv2-MC was shelved along with the rest of
the order-sensitive suite (`notes/order-sensitive-eval.md`). A2's headline evidence set no
longer exists, so a "flat everywhere" result would be **uninterpretable, not negative**. Do not
spend a ranking arm on A2 without first naming a benchmark that can actually see it.

---

## Tier B — geometry-preserving distillation (N1), the headline train-time novelty

**Dependency.** N1 distills from a teacher, so it rides on the **R2 teacher-embedding
pipeline** (`ROADMAP` 2.2: offline InternVideo2 pass over InternVid-10M, storing video+text
embeddings). Two facts make it feasible *now* at pilot scale without waiting for the full pass:

1. The **cosine-KD machinery already exists** — `UTA_Loss.uta_vision_loss`
   (`criterions.py:469`) and `new_UTA_Loss` (`:540`) compute `2 − 2·cos(student, teacher)`,
   InternVideo2's unmasked-token-alignment loss. It is wired into `tasks/pretrain.py`
   (the InternVideo2 path, `c_loss`/`uta_all` at `:73-74`) but **not** into
   `tasks_clip/pretrain.py` (MobileViCLIP is VTC-only). The **loss math** is reused, but the
   **teacher-tensor plumbing is genuinely new to `tasks_clip`**: the loss is computed inside
   `model.forward`, whose signature is `(image, text, idx)` (`tasks_clip/pretrain.py:113`), so
   feeding teacher embeddings means threading them through dataset → loader → `forward` →
   `loss_dict`. That plumbing *is* the R2 pipeline work — so B1 = R2 plumbing + reused
   cosine-KD + the new geo term, **not a flag flip**.
2. **InternVideo2 model code is in-repo** (`models/backbones/internvideo2/internvideo2.py`),
   so an **online** InternVideo2-S teacher can run in-step on the 10k pilot for the link-test,
   before the offline R2 storage exists — **but its weights are not in `CHECKPOINTS.md`**;
   sourcing the teacher checkpoint (and the L-vs-1B decision) is part of ROADMAP 2.2, so the
   pilot link-test is gated on that download, not runnable this second.

**The novel term (N1).** Cosine-KD matches *where* each embedding lands; it ignores the
teacher's **relational geometry**, which is what retrieval (a ranking task) depends on. Add a
**geometry-preserving** loss — match the student's and teacher's centered batch-similarity
(Gram) matrices (similarity-preserving KD, Tung & Mori *ICCV* 2019; CKA, Kornblith *ICML*
2019):

```python
class GeometryKD_Loss(nn.Module):
    """Distill the teacher's batch-similarity GEOMETRY, not just per-vector cosine."""
    def geo_loss(self, z_s, z_t):                 # z_*: [B, C]
        z_s = F.normalize(z_s, dim=-1); z_t = F.normalize(z_t, dim=-1)
        Gs, Gt = z_s @ z_s.t(), z_t @ z_t.t()     # [B, B] Gram matrices
        Gs = F.normalize(Gs, p=2, dim=1)          # Tung-Mori row-normalized SP-KD
        Gt = F.normalize(Gt, p=2, dim=1)
        return (Gs - Gt).pow(2).sum(1).mean()
        # CKA variant: return 1 - cka(Gs, Gt) for a scale-invariant target
```

Apply to the video–video, text–text, and cross video–text Grams (ablate which help). This is
**O(B²)** supervision — a denser per-batch signal than InfoNCE's O(B), which also softens the
small-batch weakness (bottleneck (d), the same one P4/sigmoid targets from another angle).

**Wiring.** Port the KD path into `tasks_clip/pretrain.py` and `MobileViCLIP_Small.forward`;
add loss-weight keys.

> ⚠️ **Corrected 2026-07-29 (round 2 §5 P2, §6.3).** Round 1 wrote here that "the loss is
> selected purely by nonzero weight, `tasks_clip/pretrain.py:48`". **That is wrong.** `:49`
> builds *logging meters* (`loss_names = [... if v != 0]`, actually `:49`); the actual objective is
> `loss = sum(loss_dict.values())` at **`tasks_clip/pretrain.py:114`** — an **unweighted** sum.
> `criterion["loss_weight"]` therefore has **zero effect on the optimization**: the config
> below selects nothing and scales nothing. The weights must be applied **inside `forward`**
> where the loss terms are produced (`models/mobileviclip_small.py:222-250`), or the KD terms
> silently enter at weight 1.0 each. Fix this when implementing B1 — and note the same trap
> propagates into `plan-review-upgrades.md` U9, which merges `vtc_kd` and `kd_geo` into one
> ablation table on the assumption that their configured weights are live.

```python
# experiments/configs/nB1_geomkd.py  (lands with ROADMAP 2.2)
# NOTE: loss_weight is logging-only (see the correction above). Apply real weights in forward().
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa
criterion["loss_weight"].update(dict(kd_cos=1.0, kd_geo=1.0))   # kd_sim optional (C1 plan)
model["distill"] = dict(teacher="internvideo2_s", source="online")  # pilot; "offline" at full
for __c in available_corpus["internvid_all"]:
    __c["caption_idx"] = True          # shared-caption multi-positive (audit fix 1)
output_dir = "experiments/runs/nB1_geomkd"
```

**Mandatory baselines (T1 — falsify the neuroscience).** Three arms, matched steps:
`arm-cos` = `kd_cos` only (the existing UTA cosine-KD); `arm-rkd` = RKD-distance
(Park *CVPR* 2019); `arm-geo` = `+ kd_geo`. The **falsifiable figure**: student↔teacher CKA
plotted against retrieval R@k across arms — geometry-KD must move *both*.

**Link-test (10k, online teacher).** `kd_geo` produces a finite loss that decreases and its
gradient reaches the student; teacher forward adds no NaNs. Ranking is **not** read here.

**Full-scale kill criterion.** `arm-geo` must beat **both** `arm-cos` and `arm-rkd` on
aggregated multi-benchmark R-Sum, **same direction in DSL and raw protocols** (paper-plan
§6.4/§6.7), second seed for <0.5 calls. If it does not, N1 collapses to a re-citation and C1
ships with cosine-KD only — a clean, pre-registered negative result.

**Design details to fix at 2.2 (inherited from paper-plan §6.9).** teacher embed-dim ≠ 512
needs a projection; record teacher frame indices + sampling protocol; the Gram loss is
batch-composition-sensitive, so keep the sampler seed logged for reproducibility.

---

## Tier C — predictive-coding structured gating (N2 efficiency half), gated

**This is the R4 slot with a predictive-coding rationale**, not a mainline item. It is the
only idea that attacks the dominant per-frame conv cost (the "~92%" figure is an unprofiled
analytic estimate — **profiling the per-frame vs stage-4 split is itself a prerequisite**),
but it is a genuine research bet and is **doubly gated**:

- **Prerequisites:** C2 T-agnostic ops (`ROADMAP` 2.1) must exist (dynamic frame counts kill
  the hard-coded `reshape(8,…)`), *and* the on-device latency/export harness (paper-plan Wk8)
  must be able to measure worst-case latency — because the whole claim is latency, not MACs (T2).
- **Dual trigger (both, as in paper-plan R4):** ≥2 weeks buffer after R3, **and** the elastic
  model degrades at low T (frame information demonstrably matters).

**Deployable form (T2).** Static, structured **block-skip gating** — a tiny per-frame surprise
signal (reuse A1's error proxy) decides whether flagged-static frames skip whole backbone
blocks — **not** fine delta-sparsity (NPUs do not accelerate unstructured sparsity; it can cut
MACs while raising latency). Encode K=2–3 keyframes fully; propagate through the temporal
RepMixer (`:589`,`:599`) as the cross-frame injector. Train as a **self-distillation
fine-tune** from the full-frame path, reusing B1's loss — no from-scratch run.

**Baselines / kill.** Must beat (a) uniform-K temporal striding + mean-pool and (b)
ResidualViT-style residual encoding (Soldelli *ICCV* 2025) at **equal measured worst-case
on-device latency** (report Core ML / TFLite, not MACs). Win → a paper section (elastic-T ×
asymmetric-compute Pareto). Lose → future work, no second attempt.

**N3 two-stream** (magno/parvo, `§N3`) is the **low-risk substitute**: if C1's gating breaks
on motion-heavy SSv2-MC, a fixed slow-high-res / fast-low-res split lowers to dense convs
(export-clean) and still cuts the ×T cost — reliable but low-novelty ("mobile SlowFast").

---

## Shared harness (applies to every item)

**Config template.** Every item is `from scripts.pretraining.clip.mobileviclip_small.config
import *` then a few overrides + `output_dir` (the `r1a_sigmoid.py` idiom). A/B arms must share
`deepspeed.initial_scale_power` (=8) so overflow-skips don't desynchronize step counts
(base config `:138-139`).

**Link-test (10k V100, go/no-go — mirrors `ROADMAP` 1.3):** (1) runs on the smoke slice; (2)
loss decreases; (3) the new parameters leave their init / receive nonzero gradient. Never used
to rank.

**Full-scale A/B (H100, 10M):** matched steps + data budget; aggregated multi-benchmark R-Sum;
**same direction in both DSL and raw** protocols; second seed for borderline (<0.5) calls
(paper-plan §6.7). Primary protocol = raw cosine (`_emb/`); DSL appendix-only.

**Unit-test discipline:** every default-off switch gets a bitwise-equal off-path assertion in
`tests/test_optimization_components.py`; every reparameterized op gets a train-vs-reparam
round-trip test.

---

## Sequencing — extends `experiments/ROADMAP.md` 第二线 (code dev, no data dependency)

```
2.4  A1 surprise-pool + A2 graded-windows      (C3-class; start now)
       -> 10k link-test -> queue for full A/B beside R1 components
       kill: A1 must beat attn-pool; A2 must not regress (win on SSv2-MC)
2.5  B1 geometry-KD: port UTA cosine-KD into tasks_clip + add GeometryKD_Loss
       (lands with the R2 teacher pipeline, 2.2)
       -> pilot online-teacher link-test -> full A/B as an R2 sub-arm
       kill: arm-geo > arm-cos AND arm-rkd, both protocols; else cosine-KD only
post-convergence, on dual trigger:
     C1 predictive-coding structured gating  (R4+; needs 2.1 + Wk8 latency harness)
       N3 two-stream as the low-risk fallback
```

**Skips** (rationale in `neural-inspired-directions.md §2`): N5 pure adaptive-T (breaks
static-graph export; worst-case unchanged), N7 foveation (collapses into the existing
token-merging line), N8 SNN (no neuromorphic silicon on phones — matches the spikingjelly
removal already in the hygiene gate).

---

## Audit corrections (2026-07-23 red-team pass, code-verified)

A re-review against the source. Five findings, all fixed in place above:

1. **"92% / 8%" MAC split was presented as fact — it is an unprofiled analytic estimate.**
   It traces to the feasibility review, not to any measurement; `model-architecture-improvements.md:73`
   says only that per-frame stages 1–3 dominate ("no measured profile yet") and never states 92%.
   Softened in both notes; **profiling the split is now an explicit Tier-C prerequisite**.
2. **A2 dilation schedule was wrong (4 entries).** `token_mixers=("repmixer","repmixer",
   "repmixer","attention")` (`mobileviclip_tiny.py:200`) → temporal 1D convs live in **stages 1–3
   only**; stage 4 is attention. Fixed to a 3-entry schedule `[1,2,4]`.
3. **A2 reparam risk was mislocated.** The 2D `MobileOneBlock` already threads `dilation` incl.
   reparam (`mobileone.py:123,177`), id_tensor centre is dilation-invariant — so the risk is not
   "id_tensor hardcodes k=3" but "does `MobileOneBlock_1d` expose `dilation`". Relocated.
4. **B1 "port, not new machinery" was too rosy.** The loss math is reused, but `model.forward`
   is `(image,text,idx)` and computes loss internally (`tasks_clip/pretrain.py:113`) — feeding
   teacher tensors needs new dataset→loader→forward→loss_dict plumbing (the R2 work). Reframed.
5. **B1 online-teacher "runnable now" overstated.** InternVideo2-S *code* is in-repo but its
   *weights* are absent from `CHECKPOINTS.md`; the pilot link-test is gated on sourcing that
   checkpoint (part of ROADMAP 2.2). Corrected.

**Verified correct (no change):** A1 — the `temporal_pooling` dispatch (`:1284-1289`) and the
`"temporal_pool"` name match for freeze/`different_lr` (`:67`) are exactly as the plan assumes.

---

## References

Loss / distillation precedents (B1 baselines): Similarity-Preserving KD — Tung & Mori,
*ICCV* 2019; RKD — Park et al., *CVPR* 2019; CKA — Kornblith et al., *ICML* 2019; CLIP-KD —
Yang et al., *CVPR* 2024. Efficiency prior art (C1): ResidualViT — Soldelli et al., *ICCV*
2025; SlowFast — Feichtenhofer et al., *ICCV* 2019. Neuroscience anchors and the full N1–N8
rationale: `notes/neural-inspired-directions.md §5`. Sequencing / gates: `notes/paper-plan.md`,
`experiments/ROADMAP.md`.

*Line/file references checked 2026-07-23 against `models/mobileviclip_tiny.py`,
`models/criterions.py`, `scripts/pretraining/clip/mobileviclip_small/config.py`,
`tasks_clip/pretrain.py`.*
