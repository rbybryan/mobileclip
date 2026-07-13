# mobileclip

> 中文分步运行指南（在新 Linux 机器上从零跑起来）：见 **[README_zh.md](README_zh.md)**。

Replication and improvement of **MobileViCLIP** — *An Efficient Video-Text Model for Mobile Devices* (ICCV 2025).

## Goal
- **Replicate** the original MobileViCLIP results (zero-shot video classification and video-text retrieval).
- **Improve** on efficiency and/or accuracy for on-device deployment.

## References
- Paper: [arXiv:2508.07312](https://arxiv.org/abs/2508.07312) · [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_MobileViCLIP_An_Efficient_Video-Text_Model_for_Mobile_Devices_ICCV_2025_paper.html)
- Original code: [MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)

## Background — 2025 image/video CLIP landscape
A survey report situating MobileViCLIP in the 2025 CLIP landscape (on-device ↔ frontier), covering MobileCLIP2, SigLIP 2, MetaCLIP 2, Perception Encoder, and more — plus a **source-level walk-through of this repo's** temporal structural reparameterization and InternVid-10M-FLT data pipeline.
- **[docs/clip-2025-progress-report.md](docs/clip-2025-progress-report.md)** — Markdown, renders on GitHub
- Interactive version with charts (light/dark): [shareable report](https://claude.ai/code/artifact/4e34bc69-c8bf-46a6-ade1-0f11b23c8288)

## Layout
This repo is seeded with the **upstream MobileViCLIP code** as the working base:
```
configs/         upstream configs
models/          model definitions (MobileCLIP backbone, InternVideo2, BERT, BEATs)
scripts/         training / evaluation entry points
tasks/ tasks_clip/  training + eval task loops
dataset/ preprocess/ tools/ utils/  data + helpers
mobileviclip.yml conda environment
docs/            UPSTREAM_README.md + clip-2025-progress-report (landscape survey)
notes/           replication log, ablations, ideas
CHECKPOINTS.md   download links for weights, annotations, videos
```

## Getting the weights & data
Weights, annotations, and videos live on external hosting (not committed). See **[CHECKPOINTS.md](CHECKPOINTS.md)** for all download links and target paths.

## Environment
```bash
conda env create -f mobileviclip.yml
conda activate mobileviclip
```

## Commands
All setup / download / train / eval command lines are in **[COMMANDS.md](COMMANDS.md)**.

## Status
Upstream code imported. Next: reproduce environment, download checkpoints, replicate zero-shot eval. See `notes/replication-plan.md`.
