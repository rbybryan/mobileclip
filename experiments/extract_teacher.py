"""C1 教师嵌入离线提取(C1 教师管线;评审 U7)。

C1(视频数据强化)的做法:**用 InternVideo2-stage2 教师离线过一遍语料,把嵌入存下来**,
学生训练时只读盘、不跑教师。本文件就是那个「过一遍」。长期标红的
「提取脚本至今未写」指的正是它。

教师身份(U7-4,2026-07-27 定案,勿改):**InternVideo2-stage2**。它有对比式共享空间,
`vtc_kd`(相似度矩阵 KD)才算得出来。Qwen3-VL 搭车方案已被否决(无对比对齐空间);
InternVideo3 / 2.5 / Next 同样不可用作 vtc_kd 教师 —— 更强教师的正道是 IV2-stage2-6B。

--------------------------------------------------------------------------------
产出什么
--------------------------------------------------------------------------------
每个 shard 一个目录,内含定长块(block)+ 一份 manifest:

    <out>/<shard_tag>/block_000000.npz   每块 --block-size 条,原子落盘,可断点续跑
    <out>/<shard_tag>/manifest.json      教师身份 / 采样协议 / 维度 / 代码版本
    <out>/<shard_tag>/video_emb.fp16.npy 由 `merge` 合并出的最终数组(可选)

块内数组(N = 块内条数,D = 教师 clip_embed_dim,默认 768):

  rows          int64  [N]        在 anno 文件里的**行号**(对齐的唯一凭据)
  ok            bool   [N]        该条是否成功解码;失败条**保留占位行**,不删
  video_emb     f16    [N, D]     教师池化视频嵌入(失败条为全 0)
  frame_indices int32  [N, T]     实际抽到的帧号(**可复现性要求**,失败条为 -1)
  vlen/fps      int32/f32 [N]     解码元信息,便于事后复查
  frame_emb     f16    [N, T, D]  可选(--per-frame,U7-2)
  subset_emb    f16    [N, S, D]  可选(--per-frame,低 T 子集池化,见下)

**失败条一律保留占位行、绝不静默丢弃** —— 一旦丢行,`rows` 与嵌入的对齐就悄悄错位,
而这种错位在下游只表现为「蒸馏没效果」,极难归因
(训练内评测被静默跳过,白跑 15 小时)。本脚本另设 `--fail-rate-abort`:
单块失败率超阈值直接抛错,不让一个半空的 shard 混进语料。

--------------------------------------------------------------------------------
三条设计决策在这里的落点
--------------------------------------------------------------------------------
1. **教师维度 ≠ 学生 512** —— 本脚本不投影,原样存 D(IV2-1B 的 `clip_embed_dim=768`)。
   投影方案是**训练侧**的事;manifest 里记了 `embed_dim`,接线时照读即可。
2. **必须记录教师抽帧与采样协议** —— `frame_indices` 逐条存,协议(num_frames /
   sample / img_size / mean / std / 解码后端)写进 manifest。默认 `sample="middle"`,
   **确定性**:同一个视频重跑给同一组帧,复现不依赖随机种子。
3. **低 T 可达性** —— `--per-frame` 时额外存 `subset_emb`:对 T∈{2,4,8} 的
   **确定性均匀子集**,把教师读出头限制在这些帧的 token 上再池化一次。
   ⚠️ **口径要说清楚**:这是「教师的**读出**只看这些帧」,**不等于**「教师**只看**这些帧
   跑一遍」—— 主干注意力仍然跨全部 16 帧。要后者必须按子集重跑主干(需 pos-embed 插值,
   ~3× 成本),本脚本**没做**,manifest 里记作 `subset_pool_mode="readout_restricted"`。
   拿它回答 §六.9 决策 3 时,请按这个口径陈述。

--------------------------------------------------------------------------------
语料范围仍未定(U7-1),脚本不替你定
--------------------------------------------------------------------------------
U7-1 建议 **curated 2–3M 先行**(教师 pass 成本 ~1/3),全量 10M 仅当 R2 显示数据饥饿
再补。本脚本用 `--shard-id/--num-shards/--limit` 切,**不硬编码任何语料规模**;
先用 `--limit 100000` 跑一个 100k 切片实测吞吐,再按实测数外推决定跑多少(U7-1 的方法)。

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
    # 0) 不需要权重、不需要视频的自检(CI / 接线前先跑这个)
    python experiments/extract_teacher.py self-test

    # 1) 100k 成本切片(U7-1 的决策数据)
    python experiments/extract_teacher.py run \\
        --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \\
        --out experiments/teacher/iv2s2_100k \\
        --teacher-ckpt checkpoints/InternVideo2-stage2_1b-224p-f4.pt \\
        --limit 100000 --batch-size 8 --num-workers 8

    # 2) 全语料按 shard 并行(每卡一个 shard;彼此不通信,挂了各自续跑)
    CUDA_VISIBLE_DEVICES=0 python experiments/extract_teacher.py run ... --shard-id 0 --num-shards 4 &
    CUDA_VISIBLE_DEVICES=1 python experiments/extract_teacher.py run ... --shard-id 1 --num-shards 4 &

    # 3) 合并 + 体检
    python experiments/extract_teacher.py merge  --out experiments/teacher/iv2s2_100k
    python experiments/extract_teacher.py verify --out experiments/teacher/iv2s2_100k

命令同步记在 `COMMANDS.md` §3.10。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_VERSION = "1.0"

#: 教师默认帧预算。C2 的全部 T∈{2,4,8,16} 都对齐这一份 T=16 锚点(C2)。
DEFAULT_NUM_FRAMES = 16

#: 默认抽帧协议。`middle` = 把视频均分 T 段、各取**段中点**,确定性、与评测口径一致。
DEFAULT_SAMPLE = "middle"

#: IV2 走 ImageNet 归一化 —— 不是 CLIP 那套。依据是仓库自己的分派规则
#: (`dataset/__init__.py:get_train_transform`:`"internvideo" in name` → ImageNet)。
#: 用错会让教师嵌入整体偏移,而 loss 照样下降,属于最难查的一类错。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: 存 `subset_emb` 的低 T 子集(U7-2 用例 a / §六.9 决策 3)。
DEFAULT_SUBSET_TS = (2, 4, 8)


# ------------------------------------------------------------------ 小工具


def uniform_subset(num_frames: int, t: int) -> List[int]:
    """从 `num_frames` 帧里取 t 帧的**确定性**均匀子集(取每段中点)。

    与 `dataset.video_utils.get_frame_indices(sample="middle")` 同构:均分 t 段、各取中点。
    这样 `subset_emb[T]` 对应的帧,就是学生在该 T 下**真正会看到**的那几帧。
    """
    if t <= 0 or t > num_frames:
        raise ValueError(f"bad subset size {t} for num_frames={num_frames}")
    seg = num_frames / t
    return [min(num_frames - 1, int(seg * i + seg / 2)) for i in range(t)]


def sha256_file(path: str, max_bytes: int = 64 << 20) -> str:
    """文件指纹。大文件只摘前 `max_bytes`(权重几个 GB,全量摘要没必要)。"""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while n < max_bytes:
            b = f.read(min(1 << 20, max_bytes - n))
            if not b:
                break
            h.update(b)
            n += len(b)
    return f"sha256:{h.hexdigest()}:first{n}B"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def save_npz_atomic(path: str, **arrays) -> None:
    """原子落盘:先写 .tmp 再 rename。中途被 kill 不会留下半个块骗过续跑逻辑。"""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def read_anno(path: str) -> List[dict]:
    """读语料标注。返回**全部**记录;切片交给调用方,以免行号语义漂移。

    两种格式自动识别:
      - JSONL(一行一个 JSON 对象)—— b100 上游原始文件的格式;
      - JSON 数组 `[{...}, ...]` —— 仓库 `anno_downstream/InternVid/
        InternVid-10M-flt_*.json` 的格式(`configs/data.py` 指向的就是它)。
    判别看第一个非空白字符:`[` 是数组,其余按 JSONL 逐行读。
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        while head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            out = json.load(f)
            if not isinstance(out, list):
                raise ValueError(f"{path}: 顶层是 JSON 但不是数组")
            return out
        out = []
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


def pick_video_path(rec: dict) -> Optional[str]:
    """取视频路径。b100 的 jsonl 用 `abs_path`(绝对路径);老格式用 `video`。"""
    for k in ("abs_path", "video", "video_path", "path"):
        v = rec.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def pick_captions(rec: dict, key: str, max_captions: int) -> List[str]:
    """取字幕列表。U7-3 要求 K≥2 条/片,故 `caption` 可以是 str 也可以是 list。"""
    v = rec.get(key)
    caps: List[str] = []
    if isinstance(v, str):
        caps = [v]
    elif isinstance(v, list):
        caps = [c for c in v if isinstance(c, str) and c]
    for extra in ("captions", "recaption", "synthetic_captions"):
        ev = rec.get(extra)
        if isinstance(ev, list):
            caps += [c for c in ev if isinstance(c, str) and c]
        elif isinstance(ev, str) and ev:
            caps.append(ev)
    seen, uniq = set(), []
    for c in caps:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:max_captions]


def shard_rows(n_total: int, shard_id: int, num_shards: int, limit: int = 0) -> List[int]:
    """把行号切给某个 shard。**跨步切片**(i::num_shards)而不是切连续块 ——
    语料里相邻行往往来自同一批下载/同一来源,连续切会让各 shard 的失败率与
    时长分布系统性不同,吞吐外推就不准了。"""
    rows = list(range(n_total))
    if limit and limit < n_total:
        rows = rows[:limit]
    if num_shards > 1:
        rows = rows[shard_id::num_shards]
    return rows


# ------------------------------------------------------------------ manifest


def build_manifest(args, embed_dim: int, n_rows: int, subset_ts: Sequence[int]) -> dict:
    """把「这批嵌入是怎么来的」完整写死。复现性要求。"""
    return {
        "script": os.path.basename(__file__),
        "script_version": SCRIPT_VERSION,
        "git_sha": git_sha(),
        "created_unix": int(time.time()),
        "teacher": {
            "identity": "InternVideo2-stage2",  # U7-4 定案
            "ckpt": args.teacher_ckpt,
            "ckpt_fingerprint": (
                sha256_file(args.teacher_ckpt)
                if args.teacher_ckpt and os.path.exists(args.teacher_ckpt)
                else None
            ),
            "embed_dim": embed_dim,
            "vision_ckpt_t_size": args.vision_ckpt_t_size,
        },
        "sampling": {
            "num_frames": args.num_frames,
            "sample": args.sample,
            "img_size": args.img_size,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "decoder": "decord",
            "deterministic": args.sample == "middle",
        },
        "subsets": {
            "mode": "readout_restricted",
            "note": (
                "教师读出头只看子集帧的 token;主干注意力仍跨全部 num_frames 帧。"
                "不等于把教师按子集重跑一遍(那需要 pos-embed 插值)。"
            ),
            "ts": list(subset_ts),
            "indices": {str(t): uniform_subset(args.num_frames, t) for t in subset_ts},
        },
        "corpus": {
            "anno": args.anno,
            "n_rows_total": n_rows,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "limit": args.limit,
            "captions_key": args.captions_key,
            "max_captions": args.max_captions,
        },
        "blocks": {"block_size": args.block_size},
    }


# ------------------------------------------------------------------ 数据


class TeacherVideoDataset:
    """按行号解码 + 抽帧 + 归一化。**解码失败返回占位零帧并标 ok=False,不抛、不丢行。**"""

    def __init__(self, records: List[dict], rows: List[int], num_frames: int,
                 sample: str, img_size: int):
        self.records = records
        self.rows = rows
        self.num_frames = num_frames
        self.sample = sample
        self.img_size = img_size
        self._tf = None

    def __len__(self) -> int:
        return len(self.rows)

    def _transform(self):
        if self._tf is None:
            import torch
            from torchvision import transforms

            self._tf = transforms.Compose([
                transforms.Resize(
                    (self.img_size, self.img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            self._torch = torch
        return self._tf

    def __getitem__(self, i: int):
        import torch

        tf = self._transform()
        row = self.rows[i]
        rec = self.records[row]
        T, S = self.num_frames, self.img_size
        path = pick_video_path(rec)

        try:
            if path is None:
                raise FileNotFoundError("no video path field in record")
            from dataset.video_utils import read_frames_decord

            frames, idx, _dur = read_frames_decord(path, T, sample=self.sample)
            if frames.shape[0] != T:  # 短视频:重复末帧补齐,并如实记录
                pad = frames[-1:].repeat(T - frames.shape[0], 1, 1, 1)
                frames = torch.cat([frames, pad], 0)
                idx = list(idx) + [int(idx[-1])] * (T - len(idx))
            px = tf(frames)  # [T, C, S, S]
            return {
                "row": row, "ok": True, "pixels": px,
                "frame_indices": np.asarray(idx[:T], dtype=np.int32),
                "vlen": int(rec.get("vlen", -1)), "fps": float(rec.get("fps", -1.0)),
            }
        except Exception as e:  # noqa: BLE001 —— 任何解码错都降级为占位行
            logger.debug("decode failed row=%s path=%s: %s", row, path, e)
            return {
                "row": row, "ok": False,
                "pixels": torch.zeros(T, 3, S, S, dtype=torch.float32),
                "frame_indices": np.full((T,), -1, dtype=np.int32),
                "vlen": -1, "fps": -1.0,
            }


def collate(batch):
    import torch

    return {
        "rows": np.asarray([b["row"] for b in batch], dtype=np.int64),
        "ok": np.asarray([b["ok"] for b in batch], dtype=bool),
        "pixels": torch.stack([b["pixels"] for b in batch]),
        "frame_indices": np.stack([b["frame_indices"] for b in batch]),
        "vlen": np.asarray([b["vlen"] for b in batch], dtype=np.int32),
        "fps": np.asarray([b["fps"] for b in batch], dtype=np.float32),
    }


# ------------------------------------------------------------------ 教师模型


def build_teacher(ckpt: str, num_frames: int, img_size: int, device: str,
                  vision_ckpt_t_size: int = 4, use_flash_attn: bool = False):
    """构建 IV2-stage2 视觉塔并载权重。

    权重键的处理与 `models/mobileviclip_small.py` 的
    `load_vision_ckpt_from_internvideo2_stage2` 分支保持一致:剥 `vision_encoder.`
    前缀、跳过 `clip_decoder` / `*_pos_embed` 辅助键、按 `orig_t_size` 插值时间位置编码
    (stage2 原生是 4 帧,我们要 16 帧)。
    """
    import torch
    from models.backbones.internvideo2.internvideo2_clip_vision import InternVideo2
    from models.backbones.internvideo2.pos_embed import interpolate_pos_embed

    if use_flash_attn:
        # 上游断言三开关必须一致(internvideo2_clip_vision.py:379),而
        # `import flash_attn` 成功 ≠ 融合算子可用:dropout-layer-norm / fused-dense
        # 是单独编译的扩展,缺了时模块顶层把 DropoutAddRMSNorm/FusedMLP 置 None,
        # 到 partial(None) 才炸出 "first argument must be callable"(b100 2026-08-02
        # 实测)。这里先探一遍,缺任何一件就整体降级为朴素注意力。
        try:
            from flash_attn.ops.rms_norm import DropoutAddRMSNorm  # noqa: F401
            from flash_attn.modules.mlp import FusedMLP  # noqa: F401
        except ImportError as e:
            logger.warning(
                "--use-flash-attn 被请求,但融合算子不可用(%s)—— 自动降级为朴素"
                "注意力。注意:朴素路径会显式物化注意力矩阵,T=16/224p 下 batch 8 "
                "单次分配约 8GiB,共卡时建议 --batch-size 4 以下。", e)
            use_flash_attn = False

    model = InternVideo2(
        img_size=img_size,
        num_frames=num_frames,
        use_flash_attn=use_flash_attn,
        use_fused_rmsnorm=use_flash_attn,
        use_fused_mlp=use_flash_attn,
        drop_path_rate=0.0,
        head_drop_path_rate=0.0,
    )

    sd = torch.load(ckpt, map_location="cpu")
    for k in ("module", "model", "state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

    interpolate_pos_embed(sd, model, orig_t_size=vision_ckpt_t_size)

    clean = {}
    for k, v in sd.items():
        kk = k[len("vision_encoder."):] if k.startswith("vision_encoder.") else k
        if "clip_decoder" in kk or "final_clip_decoder" in kk:
            continue
        if "clip_pos_embed" in kk or "clip_img_pos_embed" in kk or "img_pos_embed" in kk:
            continue
        clean[kk] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    # 不静默:载权重是最容易「看起来成功其实没加载」的一步(参见 TROUBLESHOOTING
    # 里目录式 ckpt 被静默跳过、评测跑在未加载模型上的先例 #344)。
    logger.info("teacher load_state_dict: missing=%d unexpected=%d",
                len(missing), len(unexpected))
    if len(missing) > 20:
        raise RuntimeError(
            f"教师权重疑似没对上:missing={len(missing)} 个键(前 10: {list(missing)[:10]})。"
            "检查 --teacher-ckpt 是不是 InternVideo2-stage2 的视觉塔,以及 "
            "--vision-ckpt-t-size 是否为该 ckpt 的原生帧数。"
        )
    return model.to(device).eval()


class ProjectorCapture:
    """抓 `clip_projector` 的**输入 token 流**,用来做子集池化。

    forward 末尾是 `x = clip_projector(x); x = fc_norm(x)`,而 `AttentionPoolingBlock`
    的 query 是 token 均值、再对全部 token 做一次 cross-attention —— 也就是说它是
    **token 集合的纯函数**。把它作用在某个帧子集的 token 上,得到的就是
    「教师读出只看这些帧」的嵌入。一次主干前向 + 若干次极廉价的池化,不需要 T× 成本。
    """

    def __init__(self, model):
        self.model = model
        self.tokens = None
        self._frozen = False
        self._h = model.clip_projector.register_forward_pre_hook(self._hook)

    def _hook(self, _m, inputs):
        # `pool()` 自己也会调 clip_projector,从而再次触发本 hook。若不设防,
        # 第一次子集池化就会把 `self.tokens` 覆盖成那个子集,后续切片全部基于
        # 被污染的 token 流 —— 单测 `test_pool_does_not_clobber_capture` 盯的就是这个。
        if not self._frozen:
            self.tokens = inputs[0]

    def close(self):
        self._h.remove()

    def pool(self, tokens):
        """对任意 token 子集跑一遍教师读出头(projector + fc_norm)。不改变已捕获的 token。"""
        self._frozen = True
        try:
            return self.model.fc_norm(self.model.clip_projector(tokens))
        finally:
            self._frozen = False


def frame_token_slice(tokens, t: int, n_t: int):
    """取第 t 帧的 token(含全局 cls,保持与整体池化同构)。

    token 布局:`[cls] + T*L`,第 t 帧占 `1 + t*L : 1 + (t+1)*L`。
    """
    import torch

    n_patch = tokens.shape[1] - 1
    if n_patch % n_t != 0:
        raise RuntimeError(f"token 数 {n_patch} 不能被帧数 {n_t} 整除,布局假设不成立")
    L = n_patch // n_t
    return torch.cat([tokens[:, :1], tokens[:, 1 + t * L: 1 + (t + 1) * L]], dim=1)


def subset_token_slice(tokens, ts: Sequence[int], n_t: int):
    import torch

    n_patch = tokens.shape[1] - 1
    L = n_patch // n_t
    parts = [tokens[:, :1]]
    for t in ts:
        parts.append(tokens[:, 1 + t * L: 1 + (t + 1) * L])
    return torch.cat(parts, dim=1)


# ------------------------------------------------------------------ 视频阶段


def run_video_stage(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    records = read_anno(args.anno)
    rows = shard_rows(len(records), args.shard_id, args.num_shards, args.limit)
    if not rows:
        raise SystemExit("切完没有任何行 —— 检查 --limit / --shard-id / --num-shards")

    # 开跑即失败的键检查(G10 教训:配置键对不上,静默跳过,白跑 15 小时)
    probe = records[rows[0]]
    if pick_video_path(probe) is None:
        raise SystemExit(
            f"anno 首行没有视频路径字段(找过 abs_path/video/video_path/path);"
            f"实际键={sorted(probe.keys())[:12]}"
        )

    tag = args.shard_tag or (
        f"shard{args.shard_id:02d}of{args.num_shards:02d}" if args.num_shards > 1 else "all"
    )
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    model = build_teacher(args.teacher_ckpt, args.num_frames, args.img_size,
                          device, args.vision_ckpt_t_size, args.use_flash_attn)
    embed_dim = int(model.clip_projector.cross_attn.proj.out_features)

    subset_ts = tuple(int(t) for t in args.subset_ts.split(",")) if args.per_frame else ()
    write_json_atomic(os.path.join(out_dir, "manifest.json"),
                      build_manifest(args, embed_dim, len(records), subset_ts))

    n_blocks = (len(rows) + args.block_size - 1) // args.block_size
    logger.info("shard=%s rows=%d blocks=%d D=%d", tag, len(rows), n_blocks, embed_dim)

    cap = ProjectorCapture(model) if args.per_frame else None
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (
        torch.float16 if args.amp == "fp16" else None)

    try:
        for b in range(n_blocks):
            path = os.path.join(out_dir, f"block_{b:06d}.npz")
            if os.path.exists(path):        # 续跑:整块已完成就跳过
                continue
            blk = rows[b * args.block_size:(b + 1) * args.block_size]
            ds = TeacherVideoDataset(records, blk, args.num_frames, args.sample, args.img_size)
            dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collate, shuffle=False, pin_memory=True)

            acc: Dict[str, list] = {k: [] for k in
                                    ("rows", "ok", "emb", "fidx", "vlen", "fps", "femb", "semb")}
            t0 = time.time()
            for batch in dl:
                px = batch["pixels"].to(device, non_blocking=True)
                # IV2 的 PatchEmbed 是 Conv3d,吃 [B, C, T, H, W]
                px = px.permute(0, 2, 1, 3, 4).contiguous()
                with torch.no_grad():
                    ctx = (torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype)
                           if amp_dtype else torch.no_grad())
                    with ctx:
                        emb = model(px)
                        if cap is not None:
                            tok, n_t = cap.tokens, args.num_frames
                            femb = torch.stack(
                                [cap.pool(frame_token_slice(tok, t, n_t)) for t in range(n_t)], 1)
                            semb = torch.stack(
                                [cap.pool(subset_token_slice(tok, uniform_subset(n_t, t), n_t))
                                 for t in subset_ts], 1)
                            acc["femb"].append(femb.float().cpu().numpy().astype(np.float16))
                            acc["semb"].append(semb.float().cpu().numpy().astype(np.float16))
                acc["emb"].append(emb.float().cpu().numpy().astype(np.float16))
                acc["rows"].append(batch["rows"])
                acc["ok"].append(batch["ok"])
                acc["fidx"].append(batch["frame_indices"])
                acc["vlen"].append(batch["vlen"])
                acc["fps"].append(batch["fps"])

            ok = np.concatenate(acc["ok"])
            emb = np.concatenate(acc["emb"])
            emb[~ok] = 0                      # 失败条写零,但**保留行**
            fail_rate = float((~ok).mean())
            if fail_rate > args.fail_rate_abort:
                raise RuntimeError(
                    f"块 {b} 解码失败率 {fail_rate:.1%} 超过 --fail-rate-abort "
                    f"{args.fail_rate_abort:.0%};疑似路径/挂载/权限问题,先查再续跑。"
                    "(不写半空的块 —— 半空的语料在下游只表现为『蒸馏没效果』。)"
                )

            arrays = dict(
                rows=np.concatenate(acc["rows"]), ok=ok, video_emb=emb,
                frame_indices=np.concatenate(acc["fidx"]),
                vlen=np.concatenate(acc["vlen"]), fps=np.concatenate(acc["fps"]),
            )
            if cap is not None:
                fe = np.concatenate(acc["femb"]); fe[~ok] = 0
                se = np.concatenate(acc["semb"]); se[~ok] = 0
                arrays["frame_emb"] = fe
                arrays["subset_emb"] = se
            save_npz_atomic(path, **arrays)

            dt = time.time() - t0
            logger.info("block %d/%d  n=%d  fail=%.1f%%  %.1fs  %.1f clip/s",
                        b + 1, n_blocks, len(ok), 100 * fail_rate, dt, len(ok) / max(dt, 1e-6))
    finally:
        if cap is not None:
            cap.close()

    logger.info("视频阶段完成:%s", out_dir)


# ------------------------------------------------------------------ 文本阶段


def run_text_stage(args) -> None:
    """教师文本嵌入 —— `vtc_kd`(相似度矩阵 KD)需要它。

    ⚠️ **依赖重**:IV2-stage2 的文本塔是 LLaMA 系(`internvideo2_clip_text`),要 `peft`
    与一份分词器权重;仓库刻意不 eager import 它(TROUBLESHOOTING「冒烟 import 崩在
    peft / transformers」)。因此本阶段**独立成一个 stage**,只有显式 `--stage text`
    才会触发那条 import 链 —— 视频阶段完全不受影响。
    """
    records = read_anno(args.anno)
    rows = shard_rows(len(records), args.shard_id, args.num_shards, args.limit)
    tag = args.shard_tag or (
        f"shard{args.shard_id:02d}of{args.num_shards:02d}" if args.num_shards > 1 else "all")
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    # 逐条的字幕数不定(U7-3 要 K≥2),故用 ragged 布局:
    # text_emb[M, D] 拉平存,cap_offsets[N+1] 给出每条 clip 的切片区间。
    flat: List[str] = []
    offsets = [0]
    for r in rows:
        caps = pick_captions(records[r], args.captions_key, args.max_captions)
        flat.extend(caps)
        offsets.append(len(flat))
    if not flat:
        raise SystemExit(f"没抽到任何字幕 —— 检查 --captions-key(当前 {args.captions_key!r})")
    logger.info("文本阶段:%d clips → %d captions(K 均值 %.2f)",
                len(rows), len(flat), len(flat) / max(len(rows), 1))

    import torch
    from models.backbones.internvideo2 import LLaMA, Tokenizer  # 重依赖,延迟到此

    tok = Tokenizer(tokenizer_path=args.text_tokenizer)
    text_model = LLaMA(**json.loads(args.text_model_kwargs or "{}")).to(args.device).eval()

    embs = []
    for i in range(0, len(flat), args.batch_size):
        chunk = flat[i:i + args.batch_size]
        with torch.no_grad():
            e = text_model(tok(chunk).to(args.device))
        embs.append(e.float().cpu().numpy().astype(np.float16))
    text_emb = np.concatenate(embs)

    save_npz_atomic(
        os.path.join(out_dir, "text_emb.npz"),
        rows=np.asarray(rows, dtype=np.int64),
        cap_offsets=np.asarray(offsets, dtype=np.int64),
        text_emb=text_emb,
    )
    logger.info("文本阶段完成:%s", out_dir)


# ------------------------------------------------------------------ merge / verify


def merge_blocks(out_dir: str) -> dict:
    """把 block_*.npz 按 rows 升序拼成最终数组。返回体检摘要。"""
    tags = [d for d in sorted(os.listdir(out_dir))
            if os.path.isdir(os.path.join(out_dir, d))]
    summary = {}
    for tag in tags:
        d = os.path.join(out_dir, tag)
        blocks = sorted(f for f in os.listdir(d)
                        if f.startswith("block_") and f.endswith(".npz"))
        if not blocks:
            continue
        acc: Dict[str, list] = {}
        for b in blocks:
            z = np.load(os.path.join(d, b))
            for k in z.files:
                acc.setdefault(k, []).append(z[k])
        cat = {k: np.concatenate(v) for k, v in acc.items()}
        order = np.argsort(cat["rows"], kind="stable")
        cat = {k: v[order] for k, v in cat.items()}
        for k, v in cat.items():
            np.save(os.path.join(d, f"{k}.npy"), v)
        summary[tag] = {
            "n": int(cat["rows"].shape[0]),
            "n_ok": int(cat["ok"].sum()),
            "fail_rate": float((~cat["ok"]).mean()),
            "dup_rows": int(cat["rows"].shape[0] - np.unique(cat["rows"]).shape[0]),
            "arrays": sorted(cat.keys()),
        }
    return summary


def verify(out_dir: str) -> dict:
    """体检:行号唯一、失败率、零向量数、嵌入范数分布、manifest 在位。"""
    report = {}
    for tag in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, tag)
        if not os.path.isdir(d):
            continue
        rp = os.path.join(d, "rows.npy")
        ep = os.path.join(d, "video_emb.npy")
        if not (os.path.exists(rp) and os.path.exists(ep)):
            report[tag] = {"status": "未 merge,先跑 merge"}
            continue
        rows, emb = np.load(rp), np.load(ep)
        ok = np.load(os.path.join(d, "ok.npy"))
        norms = np.linalg.norm(emb[ok].astype(np.float32), axis=1) if ok.any() else np.zeros(1)
        zero_but_ok = int(((np.abs(emb).sum(1) == 0) & ok).sum())
        report[tag] = {
            "n": int(rows.shape[0]),
            "rows_unique": bool(np.unique(rows).shape[0] == rows.shape[0]),
            "rows_sorted": bool(np.all(np.diff(rows) > 0)),
            "fail_rate": float((~ok).mean()),
            "zero_emb_but_ok": zero_but_ok,   # >0 ⇒ 教师前向出了问题,不是解码问题
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "manifest": os.path.exists(os.path.join(d, "manifest.json")),
        }
    return report


# ------------------------------------------------------------------ self-test


def self_test() -> None:
    """不需要权重、不需要视频、不需要 GPU 的端到端自检。

    覆盖:抽帧子集确定性、token 切片布局、子集池化 = 「把读出限制在这些 token」、
    块的原子落盘与续跑、失败条占位不丢行、merge 后行号对齐。
    """
    import tempfile

    import torch

    # 1) 子集确定性 + 覆盖性
    assert uniform_subset(16, 16) == list(range(16))
    assert uniform_subset(16, 2) == [4, 12], uniform_subset(16, 2)
    assert uniform_subset(16, 4) == [2, 6, 10, 14], uniform_subset(16, 4)
    assert uniform_subset(16, 8) == [1, 3, 5, 7, 9, 11, 13, 15], uniform_subset(16, 8)
    for t in (2, 4, 8):
        s = uniform_subset(16, t)
        assert len(s) == len(set(s)) == t and all(0 <= i < 16 for i in s)

    # 2) token 切片布局:cls + T*L
    B, T, L, C = 2, 4, 3, 8
    tokens = torch.arange(B * (1 + T * L) * C, dtype=torch.float32).reshape(B, 1 + T * L, C)
    fs = frame_token_slice(tokens, 2, T)
    assert fs.shape == (B, 1 + L, C)
    assert torch.equal(fs[:, 1:], tokens[:, 1 + 2 * L: 1 + 3 * L])
    ss = subset_token_slice(tokens, [1, 3], T)
    assert ss.shape == (B, 1 + 2 * L, C)
    assert torch.equal(ss[:, 1:1 + L], tokens[:, 1 + L: 1 + 2 * L])

    # 3) 池化是 token 集合的纯函数 —— 子集池化 == 只喂子集 token
    class _FakePool(torch.nn.Module):          # 与 AttentionPoolingBlock 同构的简化件
        def forward(self, x):
            return x.mean(1) * 2.0

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip_projector = _FakePool()
            self.fc_norm = torch.nn.Identity()

    m = _FakeModel()
    cap = ProjectorCapture(m)
    got = cap.pool(subset_token_slice(tokens, [1, 3], T))
    want = m.fc_norm(m.clip_projector(subset_token_slice(tokens, [1, 3], T)))
    assert torch.allclose(got, want)
    cap.close()

    # 4) 分片:跨步、不重不漏
    rows = shard_rows(100, 0, 4) + shard_rows(100, 1, 4) + shard_rows(100, 2, 4) + shard_rows(100, 3, 4)
    assert sorted(rows) == list(range(100))
    assert shard_rows(100, 0, 1, limit=7) == list(range(7))

    # 5) 块落盘 / 续跑 / merge 对齐,含失败占位行
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "all")
        os.makedirs(d)
        n, D = 5, 4
        ok = np.array([True, True, False, True, True])
        emb = np.random.RandomState(0).randn(n, D).astype(np.float16)
        emb[~ok] = 0
        save_npz_atomic(os.path.join(d, "block_000000.npz"),
                        rows=np.arange(n, dtype=np.int64), ok=ok, video_emb=emb,
                        frame_indices=np.zeros((n, 16), np.int32),
                        vlen=np.zeros(n, np.int32), fps=np.zeros(n, np.float32))
        write_json_atomic(os.path.join(d, "manifest.json"), {"teacher": {"embed_dim": D}})
        assert not os.path.exists(os.path.join(d, "block_000000.npz.tmp.npz"))
        s = merge_blocks(tmp)
        assert s["all"]["n"] == n and s["all"]["dup_rows"] == 0
        assert abs(s["all"]["fail_rate"] - 0.2) < 1e-9
        r = verify(tmp)
        assert r["all"]["rows_unique"] and r["all"]["rows_sorted"]
        assert r["all"]["zero_emb_but_ok"] == 0

    print("self-test OK —— 抽帧/切片/池化/分片/块续跑/merge 对齐 全过")


# ------------------------------------------------------------------ CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="C1 教师嵌入离线提取(InternVideo2-stage2;C1 教师管线)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--out", required=True, help="输出根目录")
        q.add_argument("--anno", help="语料标注(jsonl 或 json 数组,自动识别)")
        q.add_argument("--shard-id", type=int, default=0)
        q.add_argument("--num-shards", type=int, default=1)
        q.add_argument("--shard-tag", default="")
        q.add_argument("--limit", type=int, default=0, help="只取前 N 行(成本切片用)")
        q.add_argument("--batch-size", type=int, default=8)
        q.add_argument("--num-workers", type=int, default=8)
        q.add_argument("--block-size", type=int, default=2048)
        q.add_argument("--device", default="cuda")
        q.add_argument("--captions-key", default="caption")
        q.add_argument("--max-captions", type=int, default=4)

    r = sub.add_parser("run", help="跑提取(视频 / 文本 / 两者)")
    common(r)
    r.add_argument("--stage", choices=["video", "text", "both"], default="video")
    r.add_argument("--teacher-ckpt", default="", help="IV2-stage2 视觉塔权重")
    r.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    r.add_argument("--sample", default=DEFAULT_SAMPLE, choices=["middle", "rand"])
    r.add_argument("--img-size", type=int, default=224)
    r.add_argument("--vision-ckpt-t-size", type=int, default=4,
                   help="ckpt 原生帧数(IV2-stage2 = 4),用于时间位置编码插值")
    r.add_argument("--use-flash-attn", action="store_true")
    r.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    r.add_argument("--per-frame", action="store_true",
                   help="额外存逐帧 + 低 T 子集池化嵌入(U7-2;只加极小算力)")
    r.add_argument("--subset-ts", default=",".join(str(t) for t in DEFAULT_SUBSET_TS))
    r.add_argument("--fail-rate-abort", type=float, default=0.2,
                   help="单块解码失败率超过它就抛错,不写半空的块")
    r.add_argument("--text-tokenizer", default="")
    r.add_argument("--text-model-kwargs", default="")

    m = sub.add_parser("merge", help="把 block_*.npz 合并成最终数组")
    m.add_argument("--out", required=True)

    v = sub.add_parser("verify", help="体检:对齐 / 失败率 / 零向量 / 范数")
    v.add_argument("--out", required=True)

    sub.add_parser("self-test", help="不需要权重与视频的端到端自检")

    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.cmd == "self-test":
        self_test()
        return 0
    if a.cmd == "merge":
        print(json.dumps(merge_blocks(a.out), ensure_ascii=False, indent=2))
        return 0
    if a.cmd == "verify":
        rep = verify(a.out)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        bad = [k for k, r in rep.items()
               if isinstance(r.get("rows_unique"), bool)
               and (not r["rows_unique"] or r.get("zero_emb_but_ok", 0) > 0)]
        return 1 if bad else 0

    if not a.anno:
        raise SystemExit("run 需要 --anno")
    if a.stage in ("video", "both"):
        if not a.teacher_ckpt:
            raise SystemExit("视频阶段需要 --teacher-ckpt(IV2-stage2 视觉塔)")
        run_video_stage(a)
    if a.stage in ("text", "both"):
        run_text_stage(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
