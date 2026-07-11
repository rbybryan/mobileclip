# mobileclip

Replication and improvement of **MobileViCLIP** — *An Efficient Video-Text Model for Mobile Devices* (ICCV 2025).

## Goal
- **Replicate** the original MobileViCLIP results (zero-shot video classification and video-text retrieval).
- **Improve** on efficiency and/or accuracy for on-device deployment.

## References
- Paper: [arXiv:2508.07312](https://arxiv.org/abs/2508.07312) · [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_MobileViCLIP_An_Efficient_Video-Text_Model_for_Mobile_Devices_ICCV_2025_paper.html)
- Original code: [MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)

## Layout
This repo is seeded with the **upstream MobileViCLIP code** as the working base:
```
configs/         upstream configs
models/          model definitions (MobileCLIP backbone, InternVideo2, BERT, BEATs)
scripts/         training / evaluation entry points
tasks/ tasks_clip/  training + eval task loops
dataset/ preprocess/ tools/ utils/  data + helpers
mobileviclip.yml conda environment
docs/UPSTREAM_README.md  original README
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
