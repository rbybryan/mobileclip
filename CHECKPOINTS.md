# Checkpoints & Data Links

Download links pulled from the upstream MobileViCLIP repo ([MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)).
These assets are **not** committed here (large binaries / external hosting) — download them into the paths below, which are gitignored.

## Pretrained weights

Place all weights under `checkpoints/`:

```
checkpoints
├── mobileclip_s0.pt        # MobileCLIP base (image-text)
├── mobileclip_s2.pt        # MobileCLIP base (image-text)
├── mobileviclip_tiny.pt    # MobileViCLIP (video-text)
├── mobileviclip_small.pt   # MobileViCLIP (video-text)
└── InternVideo2-stage2_1b-224p-f4.pt   # C1 teacher (video-text) — see below
```

| File | Source |
|------|--------|
| `mobileclip_s0.pt`, `mobileclip_s2.pt` | Apple MobileCLIP — https://github.com/apple/ml-mobileclip (mirror on Google Drive below) |
| `mobileviclip_tiny.pt`, `mobileviclip_small.pt` | Authors' Google Drive |
| `mobileclip2_s2.pt` (optional, U8 init swap) | Apple MobileCLIP2 — https://huggingface.co/apple/MobileCLIP2-S2 |
| `InternVideo2-stage2_1b-224p-f4.pt` (**C1 teacher**) | OpenGVLab — https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4 |

- **MobileViCLIP + MobileCLIP weights (Google Drive):** https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7?usp=drive_link
- **Apple MobileCLIP (official):** https://github.com/apple/ml-mobileclip
- **Apple MobileCLIP2-S2 (optional student init, see `notes/plan-review-upgrades.md` U8):**
  https://huggingface.co/apple/MobileCLIP2-S2 — architecture identical to MobileCLIP-S2
  (verified 2026-07-23: same per-tower param counts, HF card instantiates the
  `MobileCLIP-S2` OpenCLIP config), so it loads wherever `mobileclip_s2.pt` does after
  the same state-dict conversion. Improved arms use it as init; R0 keeps the original
  `mobileclip_s2.pt` to anchor reproduction.

- **InternVideo2-stage2 1B (C1 teacher, required by `experiments/extract_teacher.py`):**
  https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4 — **必须是 stage2**,
  不是 stage1、不是 CLIP-only 变体。理由(评审 U7-4,2026-07-27 定案):stage2 是
  InternVideo2 家族**最后一代保有真对比空间**的模型,`vtc_kd`(相似度矩阵 KD)才算得出来;
  IV3 / IV2.5 / InternVideo-Next 均无对比空间,**不可**作 vtc_kd 教师。参考数字:
  1B 版 ZS MSR-VTT T2V R@1 ≈ **51.9**(6B 版 ≈ 55.9),核验见
  [`notes/caption-kd-review-2026-07.md`](notes/caption-kd-review-2026-07.md)。
  - 该权重原生 **4 帧**;我们按 T=16 抽取,时间位置编码由脚本用
    `interpolate_pos_embed(..., orig_t_size=4)` 插值(与 `mobileviclip_small.py` 载
    stage2 视觉塔的那条分支同一套处理)。改用别的 ckpt 时记得同步 `--vision-ckpt-t-size`。
  - 只有**视觉塔**是提取视频嵌入的必需件。文本塔(LLaMA 系,要 `peft`)只有跑
    `--stage text`(为 `vtc_kd` 存教师文本嵌入)时才需要,见 `COMMANDS.md` §3.10。
  - 载权重时脚本会打印 `missing/unexpected` 键数,`missing > 20` **直接抛错** ——
    防的是「权重没对上但前向照跑、嵌入全是随机初始化」这类静默失败(先例:#344
    目录式 ckpt 被静默跳过,评测跑在未加载的模型上)。

## Annotations

Downstream + InternVid annotations, place under `anno_downstream/`:

```
anno_downstream
├── InternVid
│   ├── InternVid-10M-flt_1.json
│   ├── InternVid-10M-flt_2.json
│   └── ...
├── anet_ret_val.json
└── ...
```

- **Annotations (Google Drive):** https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza?usp=drive_link

## Videos (datasets)

Download from each dataset's official website into `data/`:

```
data
├── anet_1.3_video_val_resize/   # ActivityNet
├── kinetics_400_val_10s_320p/   # Kinetics-400
└── MSRVTT_Videos/               # MSR-VTT
```

### b100: don't download — most of this is already on the box

On the b100 server, `/workspace/mount/b100_zaip_data/video_datasets/` already holds a large
collection of video datasets shared across users. Relevant to this repo:

| repo expects (`configs/data.py`) | on b100 |
|---|---|
| `data/MSRVTT_Videos/` | `video_datasets/MSR-VTT/{TrainValVideo,TestVideo}/` |
| `anno_downstream/msrvtt_1k_test.json` | `video_datasets/MSR-VTT/mobileviclip_annotations/msrvtt_1k_test.json` ← **already in repo format** |
| `data/didemo/processed_videos/test` | `video_datasets/DiDeMo/`, `video_datasets/OpenDataLab___DiDeMo/` |
| `data/anet_1.3_video_val_resize` | `video_datasets/ActivityNetDataset/` |
| `data/kinetics_400_val_10s_320p` | `video_datasets/OpenMMLab___Kinetics-400/` |
| `data/ucf101/videos`, `data/hmdb51/videos_test`, `data/ssv2/…` | not spotted — probe before downloading |

Wire them with **symlinks** (the source dirs are shared and read-only in spirit — never copy or
modify them). Probe + wiring commands: [`experiments/SETUP_B100.md` §4b](experiments/SETUP_B100.md).
Pretraining data (InternVid-10M-FLT) is wired separately in §4.

> ⚠️ **Symlink at the level the anno expects.** Video paths are built as
> `join(data_root, anno["video"])` and nothing ever takes a basename. MSR-VTT's `video` field is
> `"TestVideo/video9770.mp4"` — a *relative path* — so the **subdirectories** get symlinked into
> `data/MSRVTT_Videos/`; flattening the `.mp4` files there puts every path one level off and the
> readiness check reports "抽查 N 条视频全部缺失". Check `anno[0]["video"]` before wiring each
> benchmark. Details: `TROUBLESHOOTING.md` 「预检报"抽查 N 条视频全部缺失"」.

## Downloading from Google Drive

The folders above are Google Drive links. To pull them on a headless machine:

```bash
pip install gdown
# whole folder:
gdown --folder https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7
gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza
```

## Paper & code

- Paper: https://arxiv.org/abs/2508.07312 (ICCV 2025)
- Upstream code: https://github.com/MCG-NJU/MobileViCLIP
