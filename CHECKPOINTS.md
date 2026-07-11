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
└── mobileviclip_small.pt   # MobileViCLIP (video-text)
```

| File | Source |
|------|--------|
| `mobileclip_s0.pt`, `mobileclip_s2.pt` | Apple MobileCLIP — https://github.com/apple/ml-mobileclip (mirror on Google Drive below) |
| `mobileviclip_tiny.pt`, `mobileviclip_small.pt` | Authors' Google Drive |

- **MobileViCLIP + MobileCLIP weights (Google Drive):** https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7?usp=drive_link
- **Apple MobileCLIP (official):** https://github.com/apple/ml-mobileclip

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
