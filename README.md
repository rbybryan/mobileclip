# mobileclip

Replication and improvement of **MobileViCLIP** — *An Efficient Video-Text Model for Mobile Devices* (ICCV 2025).

## Goal
- **Replicate** the original MobileViCLIP results (zero-shot video classification and video-text retrieval).
- **Improve** on efficiency and/or accuracy for on-device deployment.

## References
- Paper: [arXiv:2508.07312](https://arxiv.org/abs/2508.07312) · [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_MobileViCLIP_An_Efficient_Video-Text_Model_for_Mobile_Devices_ICCV_2025_paper.html)
- Original code: [MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)

## Layout
```
configs/     experiment / model configs
data/        dataset prep and manifests (gitignored contents)
models/      model definitions
scripts/     training / eval / benchmarking entry points
notes/       replication log, ablations, ideas
```

## Status
Project scaffold. See `notes/` for the replication plan and progress log.
