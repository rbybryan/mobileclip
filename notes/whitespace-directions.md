# Whitespace Directions — On-Device Video-CLIP (Phase 3 candidates)

**v2 (feasibility-reviewed).** v1 proposed two directions (quantization, streaming)
from a July-2026 literature sweep. v2 adds a **code-verified deployment finding**,
an honest feasibility review that rescopes both, and **two higher-ROI directions**
that v1 missed. Survey basis: `docs/clip-2025-progress-report.md`.

> TL;DR of the review: both original plans are implementable but were oversold.
> The **highest-value, lowest-risk** work is a **data-pipeline upgrade** and
> **temporal-hard-negative fine-tuning**; quantization is a gated engineering
> item; causal streaming should be dropped (wrong bottleneck).

---

## 0. Deployment reality (verified 2026-07) — read this first

A code review resolved an ambiguity that changes the framing:

- **Canonical model = `MobileViCLIP_Small` (with temporal modeling).** Both
  `scripts/pretraining/clip/mobileviclip_small/config.py` and
  `scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py` set
  `model_cls="MobileViCLIP_Small"`; `models/__init__.py` registers only
  `MobileViCLIP_{Small,Tiny}`. So the temporal structural reparameterization
  (1D temporal RepMixer branch + `temporal_embedding` + joint T·H·W attention in
  `models/mobileviclip_small.py`) **is** in the real train/eval model.
- **`models/convert_model.py` is NOT canonical.** `InternVideo2_CLIP_small_mobile`
  is a standalone, apparently-WIP Core ML export (imports `coremltools`, `forward`
  returns `None`, loads `mobileclip_s0.pt` for both towers). Its blocks are
  **purely spatial + mean-pool** — no 1D temporal conv, no `temporal_embedding`.
  It is wired into no config and no `__init__`.
- ~~**⇒ There is no working on-device export of the temporal model.** The only
  export script sidesteps every temporal op — which plausibly means those ops
  (hard-coded 8-frame reshapes, depthwise `Conv1d`, joint T·H·W attention) are
  **hard to lower to Core ML / TFLite**.~~ **⚠️ REFUTED 2026-07-28 — see §0-bis.**
  A model-level `reparameterize_model` util exists (`tasks_clip/shared_utils.py`,
  off by default) and helps on the GPU side.

### §0-bis — the "hard to lower" inference was wrong (measured 2026-07-27)

The struck-through claim above was an **inference**, not a measurement: it went from
"`convert_model.py` sidesteps the temporal ops" to "those ops are hard to lower",
hedged with *plausibly*. **U1's export smoke tested it directly and it does not hold**
(`experiments/export_smoke_results.md`; 101vector, full size `[1,T,3,256,256]`):

| Route | T ∈ {2,4,8,16} | Detail |
|---|---|---|
| **ONNX** | ✅ **all pass** | export + `onnx.checker` + onnxruntime load & inference |
| **Core ML** | ✅ **all pass** | `jit.trace` → `ct.convert(mlprogram)`, ~4343 MIL ops, all 5 stages |
| TFLite | ⏭️ not evaluated | toolchain only (`litert-torch` needs torch ≥ 2.6; box pinned to 2.5.1 for V100 sm_70) |

The exported object is the **vision tower** with architecture params matching
`build_vision_encoder` — i.e. **it contains every temporal op**. `convert_model.py`
sidesteps them because it is an abandoned WIP built from `mobileclip_s0` whose
`forward` returns `None` and which is wired into no config — **not** because the ops
cannot be lowered.

**Consequence — Step 0 is cleared, and the table below is re-scored:** directions
gated on "needs on-device export" (quantization, streaming) **lose that prerequisite**.
Two caveats survive: **TFLite/Android is still unknown**, and **exportable ≠ fast** —
no on-device latency has been measured, so the NPU-realization question (§3) is
untouched. Re-ranking rationale: [`c2-vs-alternatives.md`](c2-vs-alternatives.md) §1.

---

## Priority (reordered by ROI × risk × repo state)

Repo constraint: Phase 1 (replicate) is not finished and there is **no trained
checkpoint yet**. Only **PTQ** (no training) and **data-pipeline** work (folds
into the training they are already standing up) are immediately actionable.

| # | Direction | ROI | Risk | Needs training? | Needs on-device export (Step 0)? | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Data-pipeline upgrade** (ReSpec + recaption) | High | Low | folds into training | No | **Do first** |
| 2 | **Temporal hard-negative fine-tuning** | Med-High | Low | temporal branch only | No | **Do early** |
| 3 | **W8A8 retrieval-aware quantization** (was Plan A) | **Med→High** | Med | No (PTQ) | ~~Yes~~ **cleared (§0-bis)**; latency still unmeasured | **Promoted** — now a "pillar 2" candidate paired with AC5 reparam; still kill-criterion gated |
| 4 | **Frame-selection long-video retrieval** (reframed Plan B) | Med | Med | Yes | Optional | Later — and see `frame_select_results.md`: at K=2 selection *does* matter (uniform − random +3.4/+3.9) but greedy k-center ≈ no selection |
| — | ~~Causal streaming (v1 Plan B)~~ | Low | High | Yes | ~~Yes~~ cleared | **Still drop** — Step 0 is gone but **GATE-0 replaces it**: the model does not encode temporal direction, so causal vs non-causal is unmeasurable on the current suite |

---

## 1. Data-pipeline upgrade (NEW — highest ROI)

**Goal.** Improve the replication itself: better data → better MobileViCLIP,
before touching architecture.

**What.** On `internvid_all` (`configs/data.py`): (a) **ReSpec-style online
filtering** (relevance to target distribution + specificity vs a generic-caption
root; ReSpec reported ~49 avg retrieval with 27.5% of the data), and/or
(b) **MLLM recaptioning** (PE/GVE-style) of a subset. The dataset already exposes
caption filtering hooks (`dataset/pt_dataset.py`: min-caption-length,
`caption_augmentation`, `jump_filter`), so this is mostly a data-prep + config
change, not a model change.

**Eval.** Same harness (§0 of v1); measure R@1 vs the plain-InternVid baseline at
matched compute/data budget.

**Why first.** Low risk, no new inference-time complexity, directly serves the
repo's stated "replicate + improve" goal, and is useful even before the model is
fully reproduced.

---

## 2. Temporal hard-negative fine-tuning (NEW)

**Goal.** Fix the documented **order-blindness** of CLIP-style video models
(TVBench, TemporalBench, VITATECS) *on this on-device model*, cheaply.

**Why cheap & repo-native.** The model already supports training **only the
temporal params**: under `freeze_vision`, everything is frozen except
`mixer_1d / norm_1d / layer_scale_1d` (and the temporal branch), per
`models/mobileviclip_small.py`. So: freeze the image backbone, add **shuffled /
reversed / wrong-order window** hard negatives to the VTC loss
(`VTC_VTM_Loss.vtc_loss`), and fine-tune the temporal branch + `temporal_embedding`.

**Eval.** TVBench / TemporalBench (temporal sensitivity) **and** standard
MSR-VTT/DiDeMo retrieval (guard against regression).

**Novelty.** Order-blindness fixes exist for large models; doing it for an
**on-device** contrastive model and measuring the temporal-vs-retrieval trade-off
is a clean, low-cost contribution that uses the repo's exact structure.

---

## 3. W8A8 retrieval-aware quantization (rescoped Plan A)

**Honest reframing.** v1 oversold this. Most of the model is FastViT (= image
CLIP), whose quantization is well studied; the genuinely new part is the small
temporal branch. So the defensible contribution is **engineering + measurement +
a retrieval-aware calibration**, not a big novelty claim.

**Prerequisite.** Step 0 (§0): an on-device export of the temporal model.

**Kill-criterion FIRST.** Before investing, export INT8 depthwise convs +
attention and **measure latency vs FP16 on the target device**. Depthwise convs
are memory-bound and INT8 kernels are **often not faster** on mobile NPUs/GPUs —
if there is no speedup, the win is **size/memory only** (~4×); decide whether that
alone is worth it, else pivot.

**Then, if it passes:**
- **W8A8 primary** (per-output-channel weights, percentile activations),
  **de-emphasize INT4** (high risk, marginal size gain for a ~50–100M model).
- **Mixed precision** guided by a **sensitivity map** built from *embedding
  cosine drift* (not layer MSE): expect the depthwise convs (incl. the temporal
  `reparam_conv_1d`), the attention stage, and the projection heads
  (`GlobalPool2D.proj`, text projection) to be sensitive.
- **Retrieval-aware calibration:** minimize `λ1·layerMSE + λ2·(1−cos(z_fp,z_q)) +
  λ3·KL(sim_fp‖sim_q)` on a held-out InternVid batch (preserves ranking).
- **QAT recovery** from the FP16 teacher via the repo's VTC loss + similarity
  distillation, if PTQ misses the budget.
- **Cover both towers** — the text encoder is ~63M vs ~36M vision on Small, so it
  is *not* a side concern.

**Eval.** ΔR@1 (MSR-VTT/DiDeMo/ANet ZS) + K400 ZS at W8A8 vs FP16; on-device
latency / energy / size.

---

## 4. Frame-selection efficient long-video retrieval (reframed Plan B)

**Why v1's causal-streaming was wrong.** Real-time / long-video cost is dominated
by **per-frame spatial encode × number of frames**, not by the tiny temporal
branch. Causal-izing the temporal conv addresses the wrong bottleneck, sacrifices
bidirectional context (accuracy), and serves a **niche** use case (retrieval is
usually index-then-query, not live streaming).

**Reframe.** Attack the real bottleneck with **learned / query-aware / adaptive
frame selection** (ProCLIP, AdaFocus lineage) on top of the existing model, so
long videos are encoded with far fewer frames. This helps the **mainstream**
offline long-video indexing case and is still novel on-device.

**Optional cheap demo.** A sliding-window + running-mean + on-device ANN index
gives a "streaming-ish" capability **without** causal retraining (nearly free for
the frame-mean pooling path). Ship it as a demo, not the research contribution.

**Drop:** causal-conv retraining as the primary mechanism.

**Eval.** Long-video retrieval (MomentSeeker, LoVR, ActivityNet); accuracy vs
frames-encoded curve; on-device throughput.

---

## Notes

- Temporal-grounding-via-contrastive is **crowded** (CVA/CVPR'26, GranAlign/AAAI'26,
  DART/ECCV'26, HieraMamba/CVPR'26, MASRA, CoSTL…). Avoid unless the angle is very
  sharp.
- Line/symbol references point at `models/mobileviclip_small.py`,
  `models/convert_model.py`, `configs/data.py`, `dataset/pt_dataset.py`,
  `tasks_clip/shared_utils.py` at the time of writing; re-locate by symbol if files
  drift.
- All external figures cited are author-reported; the point of these directions is
  to produce **independently measured** numbers the field lacks for video-CLIP.
