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
> nothing to the optimization** — `tasks_clip/pretrain.py:94-95` sums the loss dict unweighted
> and `loss_weight` only builds logging meters, so the weight must be applied inside `forward`;
> **(3)** a mandatory zero-training gate (**GATE-0**, round 2 §2) now precedes all temporal work,
> because our benchmark suite may not be able to measure these ideas at all.

Two hard constraints are inherited from the repo and govern every plan below:

- **C-machine (two-level validation, `ROADMAP.md` / `README.md`).** V100 + 10k pilot does
  **link go/no-go only** — does it run, does loss drop, do the new parameters receive
  gradient / leave their init. It **cannot rank methods** (eval saturates, BN is confounded,
  binomial SE too large). Real A/B ranking + winner final-train happen on H100 + full 10M.
  Every item therefore carries *two* gates: a **link-test** and a **full-scale kill criterion**.
- **C-switch (component convention, C3 in `paper-plan.md`).** Every new component is a
  **default-off switch** whose off-path is **bitwise-identical** to the locked baseline, and
  whose on-path is **zero-initialized so step-0 == baseline** (the `TemporalAttentionPool`
  zero-init proj, `mobileviclip_tiny.py:1326-1327`, is the reference pattern). No component
  is allowed to regress the baseline at initialization.

Anchored to code checked 2026-07-23. `mobileviclip_tiny.py` is cited; `_small.py` mirrors it
at larger width.

---

## 0. Feasibility tiers (what can start now vs what is gated)

| Tier | Items | Depends on | Start? |
|------|-------|-----------|--------|
| **A — now** | **A1** surprise-weighted pooling (N2 aggregator), **A2** graded temporal windows (N4) | nothing — pure architecture, C3-class components | **Immediately** (ROADMAP 第二线) |
| **B — pilot now, full on R2** | **B1** geometry-preserving distillation (N1) | R2 teacher-embedding pipeline (`ROADMAP` 2.2); pilot uses an **online** InternVideo2-S teacher | Loss+wiring now; full A/B rides on 2.2 |
| **C — gated (R4-class)** | **C1** predictive-coding structured frame gating (N2 efficiency half); **N3** two-stream fallback | C2 T-agnostic ops (`ROADMAP` 2.1) + on-device latency/export harness (paper-plan Wk8) | Only after convergence + dual trigger |
| **Skip** | N5 (pure adaptive-T), N7 (foveation), N8 (SNN) | — | See `neural-inspired-directions.md §2` |

Recommended order: **A1/A2 now** (free, low-risk, mirror the R1 component A/Bs) →
**B1 with the R2 teacher pass** (the headline train-time novelty) → **C1 only if triggered**
post-convergence. B1 is the one worth the most; A1/A2 are cheap insurance that also
strengthen B1 (a better aggregator + temporal features give the distilled signal capacity).

---

## Tier A — start now (ROADMAP 第二线, C3-class)

Both A-items are the same shape as the existing R1 component A/Bs
(`experiments/configs/r1a_sigmoid.py`, `r1b_attnpool.py`): one default-off switch, a
zero-init on-path, a 10k link-test, then queued for the full-scale A/B beside R1.

### A1 · Surprise-weighted temporal pooling (N2 aggregator, teacher-free)

**Idea.** The head currently mean-pools 8 frame embeddings (`mobileviclip_tiny.py:1296-1306`);
P3 adds a learned attention pool (`temporal_pooling="attn"`, `TemporalAttentionPool:1309`).
Add a **third** mode that weights each frame by its *prediction error* — a teacher-free
proxy for "surprise" (predictive-coding motivation, `neural-inspired-directions.md §N2`):
frames that deviate from the temporal context carry more information than static ones.

**Touchpoint.** `FastViT.forward` pooling branch (`:1296-1306`) + the builder that reads
`model.vision_encoder.temporal_pooling`. New module beside `TemporalAttentionPool`:

```python
class TemporalSurprisePool(nn.Module):
    """Weight T frame embeddings by temporal prediction error (surprise).
    alpha init 0 => uniform weights => step-0 is exactly mean-pool (C-switch)."""
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))      # 0 => uniform => mean
    def forward(self, x):                              # x: [B, T, C]
        ctx = x.mean(1, keepdim=True)                  # temporal context
        surprise = (x - ctx).norm(dim=-1)              # [B, T]  cheap error proxy
        w = torch.softmax(self.alpha * surprise, 1).unsqueeze(-1)  # [B, T, 1]
        return (w * x).sum(1)                           # [B, C]
```

**Config** (create `experiments/configs/nA1_surprisepool.py`):

```python
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa
model["vision_encoder"]["temporal_pooling"] = "surprise"
optimizer["different_lr"] = dict(enable=True, module_names=["temporal_pool"], lr=1e-3)
output_dir = "experiments/runs/nA1_surprisepool"
```

**Link-test (10k, go/no-go).** Runs to completion; loss decreases; ckpt `temporal_pool.alpha`
has left 0 (received gradient). Extend `tests/test_optimization_components.py`: with
`temporal_pooling="mean"` the forward is bitwise-equal to baseline; with `"surprise"` and
`alpha=0`, output `allclose` to mean-pool.

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
The 2D `MobileOneBlock` already threads `dilation` end-to-end incl. reparam
(`mobileone.py:123,177`), and the reparam `id_tensor` centre tap is dilation-invariant, so
`RepMixer.reparameterize:617-676` stays valid. **The one thing to verify/extend:** confirm
`MobileOneBlock_1d` (the 1D variant used for `mixer_1d`, `:562`) exposes `dilation` — if it
only takes `kernel_size`, thread `dilation` + dilation-aware padding into it. The pretrained
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
Main engineering risk = confirming/extending `MobileOneBlock_1d` dilation support; the reparam
round-trip test is the gate.

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
   `model.forward`, whose signature is `(image, text, idx)` (`tasks_clip/pretrain.py:94`), so
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
add loss-weight keys (the loss is selected purely by nonzero weight, `tasks_clip/pretrain.py:48`):

```python
# experiments/configs/nB1_geomkd.py  (lands with ROADMAP 2.2)
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
   is `(image,text,idx)` and computes loss internally (`tasks_clip/pretrain.py:94`) — feeding
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
