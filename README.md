# MobileViCLIP — replication & improvement

A working replication of **MobileViCLIP** (*An Efficient Video-Text Model for Mobile
Devices*, ICCV 2025) plus an ongoing effort to improve it for on-device deployment.

- Paper: [arXiv:2508.07312](https://arxiv.org/abs/2508.07312) · [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_MobileViCLIP_An_Efficient_Video-Text_Model_for_Mobile_Devices_ICCV_2025_paper.html)
- Upstream code this repo is seeded from: [MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)
  (upstream README kept verbatim at [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md))
- 中文分步运行指南（从零在一台新 Linux 机器上跑起来）：**[README_zh.md](README_zh.md)**

---

## Status

**The baseline is reproduced and the evaluation pipeline is verified.** One round of
model-side changes has been run and scored; the full-scale (10M) training run has not.

| | |
|---|---|
| ✅ Baseline reproduced | MSR-VTT zero-shot **t2v R@1 = 42.50** with the official `mobileviclip_small.pt`, against ≈42.5 reported in the paper |
| ✅ Model-optimization round | 6 schemes A/B-tested at the 1M verification tier — **1 passed, 5 rejected** (below) |
| ✅ Training-data audit | 10k-clip full-coverage audit of the InternVid-zh corpus + a VLM recaption pipeline |
| ⬜ Full-scale run | 10M pretraining (R0→R3) not yet executed |

### What the model-optimization round found

Six single-change variants against the `sub1m_r0` baseline, each bit-for-bit equal to
the baseline at step 0, judged by paired bootstrap (10,000 resamples) on MSR-VTT 1k:

| Scheme | Change vs baseline | Verdict |
|---|---|---|
| **Drop the temporal branch** | remove the cross-frame 1D-conv branch entirely | **✅ non-inferior** — and ~27.8% fewer exported ops, −18% CPU mixer |
| Attention pooling | learned per-frame weights instead of mean | ❌ does not replicate across seeds |
| Second-order stats pooling | add inter-frame std to the mean | ❌ significantly negative |
| Surprise pooling | weight frames by prediction error | ❌ clean zero (degenerates to mean) |
| SigLIP sigmoid loss | softmax → sigmoid | ❌ significantly negative (DSL) |

**Reading:** on this dataset the model is largely insensitive to temporal information,
so the temporal conv block can be deleted for a free efficiency win. A separate audit
explains why — see below. Full numbers, CIs and mechanism analysis:
[`model-optimization/`](model-optimization/README.md).

### Why: the training captions carry almost no temporal signal

A full-coverage audit of 10,000 InternVid-zh clips
([`docs/caption-quality-report.md`](docs/caption-quality-report.md)):

- **69.1%** of videos contain multiple sequential events, but **99.2%** of their captions
  say nothing about order — only 0.55% of the corpus describes event sequence at all.
- **~16%** of the *English source* captions do not match the video; the Chinese
  translation itself is healthy (**3.6%** error rate). The problem is upstream of translation.

That is the motivation for the recaption line in [`recaption/`](recaption/README.md):
rewrite captions from the frames with a VLM rather than re-translate.

---

## Quick start

```bash
# 1. environment (Python 3.10 / PyTorch 1.13.1 + CUDA 11.7)
ENV_NAME=mobileviclip bash scripts/setup_env.sh
conda activate mobileviclip

# 2. weights, annotations and videos — see CHECKPOINTS.md for every link and target path

# 3. zero-shot eval (edit CUDA_VISIBLE_DEVICES inside the .sh first)
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh

# 4. pretraining
bash scripts/pretraining/clip/mobileviclip_tiny/run.sh
```

Two recurring traps: install `open_clip_torch` with `--no-deps` (otherwise torch is
silently upgraded to 2.x), and keep `torch`/`torchvision`/`torchaudio` versions matched.
Both are covered in [`COMMANDS.md`](COMMANDS.md) §1.

Run every command from the repo root unless a doc says otherwise.

## Where things are

| Path | What |
|---|---|
| [`COMMANDS.md`](COMMANDS.md) | **Command cheat sheet** — setup, downloads, eval, training, probes |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | Every weight / annotation / video download link and its target path |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | Known error → cause → fix |
| [`README_zh.md`](README_zh.md) | 中文全流程运行指南（新机器从零开始） |
| [`experiments/`](experiments/README.md) | A/B framework, run configs, and the measured result reports |
| [`model-optimization/`](model-optimization/README.md) | The 6-scheme model-change round: what changed, how to rerun, what it scored |
| [`recaption/`](recaption/README.md) | InternVid-zh VLM recaption + data-audit line (self-contained: docs + scripts) |
| [`docs/`](docs/) | Upstream README, CLIP landscape survey, dataset survey, caption-quality report |
| `models/` `tasks/` `tasks_clip/` `dataset/` | Model definitions and training / eval loops (from upstream, with this repo's changes) |
| `neural_inspired/` | Opt-in research add-ons, off by default |
| `tests/` | `pytest tests/` from the repo root |

Weights, annotations and videos are **gitignored** — they are downloaded on the machine
per [`CHECKPOINTS.md`](CHECKPOINTS.md), never committed.

## Background reading

A survey placing MobileViCLIP in the 2025 CLIP landscape (on-device ↔ frontier: MobileCLIP2,
SigLIP 2, MetaCLIP 2, Perception Encoder …), including a source-level walk-through of this
repo's temporal structural reparameterization and its InternVid-10M-FLT data pipeline:
[`docs/clip-2025-progress-report.md`](docs/clip-2025-progress-report.md).

## License

[MIT](LICENSE), inherited from upstream MobileViCLIP (© Multimedia Computing Group,
Nanjing University). This repo is a fork: most of `models/`, `tasks/`, `tasks_clip/`,
`dataset/` and `configs/` is upstream code with this project's changes on top.
