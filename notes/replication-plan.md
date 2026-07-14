# MobileViCLIP — Replication & Improvement Plan

Source paper: arXiv:2508.07312 (ICCV 2025). Original repo: MCG-NJU/MobileViCLIP.

## Phase 0 — Setup
- [ ] Mirror/study original repo (`../MobileViCLIP`).
- [ ] Reproduce the conda environment (`mobileviclip.yml`).
- [ ] Obtain pretrained checkpoints referenced by the original README.

## Phase 1 — Replicate
- [ ] Zero-shot video classification (e.g. Kinetics-400).
- [ ] Zero-shot video-text retrieval (e.g. MSR-VTT).
- [ ] Confirm reported speed numbers / model sizes (S / B variants).

## Phase 2 — Understand
- [ ] Temporal structural reparameterization module.
- [ ] Image-text -> video-text adaptation and training data pipeline.

## Phase 3 — Improve (candidate directions, prioritized)

> These are **candidates for review, not committed work.** Full detail + the
> feasibility review are in [`whitespace-directions.md`](whitespace-directions.md).
> Ordering is by ROI × risk × repo state (Phase 1 not done, no trained checkpoint
> yet → prefer no-training / data work first). Reviewed against a 2025–2026
> literature sweep ([`../docs/clip-2025-progress-report.md`](../docs/clip-2025-progress-report.md)).

**Cross-cutting blocker — Step 0 for anything on-device.** There is no working
on-device export of the temporal model: `models/convert_model.py` is a
non-canonical, spatial-only WIP (no 1D temporal conv / `temporal_embedding`,
mean-pool only) and is wired into nothing; the canonical train/eval model
(`MobileViCLIP_Small`) is temporal. Any on-device direction (latency, quant,
streaming) must first produce a Core ML / TFLite export of the temporal model —
non-trivial (hard-coded 8-frame reshapes, depthwise `Conv1d`, joint T·H·W
attention op coverage).

- [ ] **1. Data-pipeline upgrade — do first.** ReSpec-style relevance+specificity
      filtering + MLLM recaptioning on `internvid_all`. Data-side only (hooks in
      `dataset/pt_dataset.py`, `configs/data.py`); no on-device dependency;
      improves the replication itself. High ROI, low risk.
- [ ] **2. Temporal hard-negative fine-tuning — do early.** Add shuffled/reversed/
      wrong-order window hard negatives to the VTC loss; train **only the temporal
      branch** via the existing `freeze_vision` path (unfreezes
      `mixer_1d/norm_1d/layer_scale_1d` + `temporal_embedding`). Targets documented
      order-blindness; eval on TVBench/TemporalBench (needs adding a temporal
      benchmark to `configs/data.py`) + guard standard retrieval. Cheap.
- [ ] **3. W8A8 retrieval-aware quantization — gated.** PTQ (no training). Gate on
      a latency **kill-criterion** first (INT8 depthwise is often *not* faster on
      mobile → possibly size-only win). Retrieval-aware calibration (preserve
      embedding cosine + similarity matrix); cover **both** towers; de-emphasize
      INT4. Latency payoff requires Step 0. Novelty is thin — position as an
      engineering + measurement contribution.
- [ ] **4. Adaptive frame selection for long-video retrieval — later.** Real
      bottleneck is frame count (per-frame spatial encode × T), not the temporal
      branch; learned / query-aware frame selection over the existing model.
- [ ] ~~Causal streaming retraining~~ — **dropped**: wrong bottleneck, accuracy
      tradeoff, niche use case.

Avoid temporal-grounding-via-contrastive (crowded in 2026: CVA/CVPR'26,
GranAlign/AAAI'26, DART/ECCV'26, HieraMamba/CVPR'26 …).

## Log
- 2026-07-11: Repo scaffold created.
- 2026-07-14: Added a 2025–2026 image/video-CLIP landscape survey
  (`docs/clip-2025-progress-report`) and a feasibility review of the Phase-3
  candidates (`notes/whitespace-directions.md` v2). Verified via code that the
  canonical model is the temporal `MobileViCLIP_Small`, that `convert_model.py`
  is a non-canonical spatial-only WIP export, and that no on-device export of the
  temporal model exists yet. No implementation started (planning only).
