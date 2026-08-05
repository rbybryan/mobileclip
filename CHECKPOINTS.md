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

  ```bash
  wget -O checkpoints/mobileclip2_s2.pt \
    "https://huggingface.co/apple/MobileCLIP2-S2/resolve/main/mobileclip2_s2.pt"
  ```
  ⚠️ Before training with it, run the **key-set identity gate** in
  `notes/next-training-round-zh.md` §0 (the loader is `strict=False` — a key mismatch
  silently loads nothing). Consumed by `experiments/configs/sub1m_mc2init_s42.py`
  (both towers swapped — the frozen text tower defines the output space, so a
  vision-only swap would break step-0 alignment).

- **InternVideo2-stage2 1B (C1 teacher, required by `experiments/extract_teacher.py`):**
  https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4 — **必须是 stage2**,
  不是 stage1、不是 CLIP-only 变体。理由(评审 U7-4,2026-07-27 定案):stage2 是
  InternVideo2 家族**最后一代保有真对比空间**的模型,`vtc_kd`(相似度矩阵 KD)才算得出来;
  IV3 / IV2.5 / InternVideo-Next 均无对比空间,**不可**作 vtc_kd 教师。参考数字:
  1B 版 ZS MSR-VTT T2V R@1 ≈ **51.9**(6B 版 ≈ 55.9),核验见
  [`recaption/docs/archive/kd-review.md`](recaption/docs/archive/kd-review.md)。
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

### SSv2-MC(`ssv2_mc_val`)—— 时序探针,2026-07-30 实测**尚未落地**

`configs/data.py` 里 `available_corpus["ssv2_mc_val"]` 指向
`anno_downstream/ssv2_mc_val.json` + `data/ssv2/20bn-something-something-v2`,
两者在 101vector 上**都不存在**(2026-07-30 实测报
`FileNotFoundError: anno_downstream/ssv2_mc_val.json`)。

它是目前唯一能**证伪**「删 `temporal_embedding` 更好」的基准 —— MSR-VTT 的字幕以外观
为主,而 SSv2 是 174 选 1 的动作多选(随机基线 **0.57%**,上游 `mobileviclip_small`
参考值 **acc 46.07**,见 `eval_ssv2_mc.sh` 末尾注释),必须真的看懂动作演变才能答对。

**① 先探测(别急着下载 —— 机器上可能已经有了)**

```bash
ls -d data/ssv2 data/20bn* 2>/dev/null
ls anno_downstream/ | grep -i ssv2
# 全盘找一找共享副本(b100 的经验:大数据集常在共享目录里)
find / -maxdepth 6 -type d \( -name '20bn-something*' -o -name 'ssv2' \) 2>/dev/null | head
```

**② 标注 JSON** —— 与其它 `anno_downstream/*.json` 同一个 Google Drive 文件夹。
若整目录 `gdown --folder`(`COMMANDS.md` §1)没把它带下来,单独取:

```bash
# 在上面那个 folder 里找到 ssv2_mc_val.json,右键取 file id,然后:
gdown <FILE_ID> -O anno_downstream/ssv2_mc_val.json
python -c "import json;d=json.load(open('anno_downstream/ssv2_mc_val.json'));print(len(d), d[0].keys())"
```

> ⚠️ `gdown --folder` **一次最多 50 个文件**,超出会静默截断。`anno_downstream/`
> 目前约 20 个文件还没到上限,但若以后加了 InternVid 分片要留意这条。

**③ 视频** —— SSv2 **有许可协议,不能匿名 wget**。原 20BN 数据集现由 Qualcomm 托管:
<https://www.qualcomm.com/developer/software/something-something-v-2-dataset>
需要注册账号并接受许可,然后下载分卷 zip(`20bn-something-something-v2-??.zip`,
合计 ~20 GB,220 847 个 webm;验证集 24 777 个)。解包后按仓库预期路径接上:

```bash
mkdir -p data/ssv2
# 解包后目录名通常就是 20bn-something-something-v2/
ln -s /path/to/20bn-something-something-v2 data/ssv2/20bn-something-something-v2
# 抽查 anno 里的前几个 video id 在不在
python - <<'EOF'
import json, os
d = json.load(open('anno_downstream/ssv2_mc_val.json'))
root = 'data/ssv2/20bn-something-something-v2'
miss = [x for x in d[:50] if not os.path.exists(os.path.join(root, x['video']))]
print(f"前 50 条里缺 {len(miss)} 个;示例:{miss[:3]}")
EOF
```

> 💡 **只需要验证集**,但官方只发整包 —— 没有官方的部分下载。若磁盘紧张,解包后
> 可按 `ssv2_mc_val.json` 里的 id 白名单裁剪。
> ⚠️ **不要从来路不明的 HF 镜像拉** —— 该数据集的许可要求注明来源,论文里要写清楚。

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
