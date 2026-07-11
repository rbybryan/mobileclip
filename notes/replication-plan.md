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

## Phase 3 — Improve (candidate directions)
- [ ] Efficiency: attention/token reduction, quantization, distillation.
- [ ] Accuracy: better temporal modeling, stronger text encoder, data curation.
- [ ] On-device: latency benchmarking on target mobile hardware.

## Log
- 2026-07-11: Repo scaffold created.
