"""闸② —— 训练到底是不是**解码瓶颈**?(零 GPU 训练,~10 分钟)

背景(`notes/architecture-round4.md` §0.2):round-4 实测 **MFU ≈ 0.67%**
(477 GFLOP/样本 × 55.8 样本/s ÷ 4×H100 ÷ 990 TFLOP/s)。另一路佐证:
55.8 ÷ 24 workers = **430 ms/片**,是很典型的 decord 解码耗时 ⇒
**「6.6 天/轮」很可能是 `num_workers` 配小了的产物,而不是模型属性。**

本脚本只做两件事,都不建模型、不前向、不反传:

* **A. 单片解码耗时** —— 直接调 `read_frames_decord`,量纯解码墙钟。
* **B. loader-only 吞吐** —— 用**训练入口自己的** `setup_dataloaders`,
  只迭代 dataloader、不建模型。这是「上限吞吐」:模型再快也不可能超过它。

判读(写在输出末尾,不用回头翻文档):

| loader-only | 训练实测 55.8 样本/s | 结论 |
|---|---|---|
| ≈ 55.8 | 相当 | **坐实解码瓶颈** ⇒ 调 `num_workers`/预解码即可提速,§0.2 成立 |
| ≥ 3× | 远高于 | 不是解码瓶颈 ⇒ round-4 的 FLOP 模型错了(**这本身也是结论**) |

用法(必须走 torchrun —— `setup_main()` 会读 argv 里的 config 并做 dist 初始化,
裸 `python` 或 heredoc 都跑不起来,2026-07-31 之前 COMMANDS.md 里的写法就是错的):

    python -m torch.distributed.run --nproc_per_node=1 --master_port=$((10000+RANDOM%20000)) \
        experiments/probe_decode_bound.py \
        scripts/pretraining/clip/mobileviclip_small/config.py \
        output_dir probe_decode_bound

可选:`--batches N`(默认 50)控制 B 段迭代多少个 batch。
"""
from __future__ import annotations

import os
import sys
import time


def _fmt(x: float) -> str:
    return f"{x:,.1f}"


def probe_decode(cfg, n_reads: int = 20) -> None:
    """A. 单片解码耗时 —— 从**配置自己的** train corpus 里取真实 clip,不硬编码路径。"""
    import json

    from dataset.video_utils import read_frames_decord

    print("=" * 72)
    print("A. 单片解码耗时(纯 decord,不建模型)")
    print("=" * 72)

    tf = cfg.train_file
    # train_file 可能是单个 dict,也可能是 list(internvid_all)
    entries = tf if isinstance(tf, (list, tuple)) else [tf]
    entry = dict(entries[0])
    anno_path, data_root = entry.get("anno_path"), entry.get("data_root")
    print(f"  anno_path = {anno_path}")
    print(f"  data_root = {data_root}")
    if not anno_path or not os.path.isfile(anno_path):
        print(f"  [跳过] 标注文件不存在:{anno_path}")
        return

    with open(anno_path) as f:
        anno = json.load(f)
    print(f"  标注条数 = {len(anno):,}")

    # 记录里视频文件名的键:本仓库 video 也用 "image"(见 dataset/base_dataset.py:139-156)
    def _rel(rec):
        for k in ("image", "video", "abs_path"):
            if k in rec:
                return rec[k]
        return None

    # 找到**第一个真实存在**的文件再计时 —— InternVid 常常只下了一部分分片,
    # 直接拿第 0 条会 FileNotFoundError 而看起来像「脚本坏了」。
    path = None
    for rec in anno[:2000]:
        rel = _rel(rec)
        if rel is None:
            continue
        cand = os.path.join(data_root, rel) if data_root else rel
        if os.path.isfile(cand):
            path = cand
            break
    if path is None:
        print("  [跳过] 前 2000 条标注对应的视频文件一个都不存在 —— 先确认分片下载与 data_root。")
        return

    print(f"  计时对象 = {path}")
    nf = cfg.num_frames
    read_frames_decord(path, nf, sample="middle")          # 预热(文件缓存/解码器初始化)
    t0 = time.time()
    for _ in range(n_reads):
        read_frames_decord(path, nf, sample="middle")
    ms = (time.time() - t0) / n_reads * 1000
    print(f"\n  ⇒ **{ms:.0f} ms/clip**(T={nf};round-4 由 55.8÷24 反推的预测值 ≈430)")
    print(f"     单 worker 理论吞吐 = {1000/ms:.2f} 样本/s;"
          f"{cfg.num_workers} workers ⇒ ≈ {1000/ms*cfg.num_workers:.1f} 样本/s")


def probe_loader(cfg, max_batches: int = 50) -> None:
    """B. loader-only 吞吐 —— 用训练入口自己的构造函数,避免手搓配置。"""
    from tasks_clip.pretrain import setup_dataloaders

    print("\n" + "=" * 72)
    print("B. loader-only 吞吐(只迭代 dataloader,不建模型/不前向)")
    print("=" * 72)
    print(f"  num_workers = {cfg.num_workers}   batch_size = {cfg.batch_size}")

    train_loaders, _, _ = setup_dataloaders(cfg, mode="pt")
    dl = train_loaders[0]

    it = iter(dl)
    next(it)                                   # 预热:第一个 batch 含 worker 启动开销
    t0, n, i = time.time(), 0, 0
    for i, _batch in enumerate(it, start=1):
        n += dl.batch_size
        if i >= max_batches:
            break
    dt = time.time() - t0
    thr = n / dt if dt > 0 else float("nan")
    print(f"\n  ⇒ **{thr:.1f} 样本/s**({i} 个 batch,{dt:.1f}s,不含首个预热 batch)")

    print("\n" + "-" * 72)
    print("  判读(与训练实测 55.8 样本/s 比):")
    if thr < 55.8 * 1.5:
        print(f"    loader-only {thr:.1f} ≈ 训练实测 55.8 ⇒ **坐实解码瓶颈**。")
        print("    ⇒ round4 §0.2 成立,「6.6 天/轮」是可移除的;先调 num_workers"
              f"(当前 {cfg.num_workers},对照 `nproc`),必要时预解码/降分辨率存帧。")
    elif thr >= 55.8 * 3:
        print(f"    loader-only {thr:.1f} ≥ 3× 训练实测 ⇒ **不是解码瓶颈**。")
        print("    ⇒ 瓶颈在算力/通信,round-4 的 FLOP 模型需要修正 —— 这本身也是结论。")
    else:
        print(f"    loader-only {thr:.1f} 介于 1.5×~3× 之间 ⇒ **部分受解码限制**,"
              "调 workers 有收益但拿不满;两边都要动。")
    print("    ⚠ 同时看 `nvidia-smi dmon -s u -c 60` 的 sm 列:持续 <25% 支持解码瓶颈。")
    print("-" * 72)


def main() -> int:
    # setup_main() 会读 argv 里的 config 并做分布式初始化 ⇒ 必须在 torchrun 下跑
    from utils.config_utils import setup_main

    max_batches = 50
    if "--batches" in sys.argv:
        i = sys.argv.index("--batches")
        max_batches = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]          # setup_config 不认识这个参数,先摘掉

    cfg = setup_main()
    try:
        probe_decode(cfg)
    except Exception as e:                      # A 段失败不该挡住 B 段
        print(f"\n[A 段失败,继续跑 B 段] {type(e).__name__}: {e}")
    probe_loader(cfg, max_batches=max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
