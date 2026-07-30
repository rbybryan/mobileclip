"""闸② —— 训练到底卡在哪?(零 GPU 训练,~15 分钟)

背景(`notes/architecture-round4.md` §0.2):round-4 实测 **MFU ≈ 0.67%**
(477 GFLOP/样本 × 55.8 样本/s ÷ 4×H100 ÷ 990 TFLOP/s)。GPU 确实在空转,
但**空转的原因**决定了该怎么修,而 2026-07-30 第一次实测就打出了自相矛盾的两个数:

| 段 | 实测(2026-07-30) | 含义 |
|---|---|---|
| A 热解码(同一文件读 20 次) | **22 ms/片** | ⇒ 6 workers 应有 **274 样本/s** |
| B loader-only(6 workers) | **48.9 样本/s** | 比 A 的外推**低 5.6×** |
| round-4 训练(4 卡 × 6 = 24 workers) | 55.8 样本/s | workers ×4,吞吐只 **+14%** |

A 和 B 差 5.6× ⇒ **「解码慢」解释不了 loader**;workers ×4 只换 +14% ⇒
**某个共享资源已经饱和**。三个互斥假设:

* **H1 核数不够** —— 6 workers 就吃满了核;24 workers 是 3× 超订,于是
  每片从 123 ms 劣化到 430 ms。修法:预解码降 CPU,不是加 workers。
* **H2 挂载盘 I/O** —— 数据在 `/workspace/mount/...`,A 段第 2~20 次读命中
  page cache 所以只要 22 ms;训练读的是 1000 万个**互不相同**的文件,每个都是冷读。
  修法:打包成顺序读的 shard / 落本地盘。
* **H3 decord 线程超订** —— `dataset/video_utils.py:137` 是 `num_threads=0`
  (= 自动 = 多线程)。A 段只有一个 reader,独占整机;loader 里 6 个 worker
  各自开一把线程互相抢。修法:`num_threads=1`,一行改动。

**这三个的修法完全不同,所以必须先判出来是哪一个。** 本脚本四段,都不建模型、不前向:

* **A. 热解码** —— 同一文件反复读。这是**纯解码 CPU 下界**(无 I/O)。
* **A2. 冷读** —— 每个文件只碰一次,且拆成「纯读字节」和「读+解码」两组
  **不相交**的文件。`纯读` 占比高 ⇒ H2。
* **B. loader-only 吞吐** —— 用训练入口自己的 `setup_dataloaders`,只迭代不建模型。
* **C. num_workers 扫描** —— 复用 B 已建好的 dataset,只换 workers。
  **曲线压平 ⇒ H1/H2(加 workers 无用);继续爬 ⇒ round-4 的「免费修法」成立。**

判读写在输出末尾,不用回头翻文档。

用法(必须走 torchrun —— `setup_main()` 会读 argv 里的 config 并做 dist 初始化,
裸 `python` 或 heredoc 都跑不起来,2026-07-30 之前 COMMANDS.md 里的写法就是错的):

    python -m torch.distributed.run --nproc_per_node=1 --master_port=$((10000+RANDOM%20000)) \
        experiments/probe_decode_bound.py \
        scripts/pretraining/clip/mobileviclip_small/config.py \
        output_dir probe_decode_bound

可选:
  `--batches N`     B 段迭代多少个 batch(默认 50)
  `--sweep a,b,c`   C 段扫描的 num_workers 列表(默认 `1,2,6,12,24`;给 `none` 跳过)
  `--cold-files N`  A2 段每组取多少个互不相同的文件(默认 25)
"""
from __future__ import annotations

import os
import statistics
import sys
import time

# 训练实测吞吐(round-4 §0.2,4 卡 × 6 workers = 24 workers)
TRAIN_THROUGHPUT = 55.8
TRAIN_WORKERS = 24

_STATE: dict = {}          # 各段把结论存这里,最后统一判读


def _hdr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _entries(cfg):
    """train_file 可能是单个 dict,也可能是 list(internvid_all)。"""
    tf = cfg.train_file
    return list(tf) if isinstance(tf, (list, tuple)) else [tf]


def _rel(rec):
    """记录里视频文件名的键:本仓库 video 也用 "image"(见 dataset/base_dataset.py:139-156)。"""
    for k in ("image", "video", "abs_path"):
        if k in rec:
            return rec[k]
    return None


def _load_anno(cfg):
    import json

    entry = dict(_entries(cfg)[0])
    anno_path, data_root = entry.get("anno_path"), entry.get("data_root")
    print(f"  anno_path = {anno_path}")
    print(f"  data_root = {data_root}")
    if not anno_path or not os.path.isfile(anno_path):
        print(f"  [跳过] 标注文件不存在:{anno_path}")
        return None, None
    with open(anno_path) as f:
        anno = json.load(f)
    print(f"  标注条数 = {len(anno):,}")
    return anno, data_root


def _find_existing(anno, data_root, want: int, max_checks: int = 20000):
    """按**跨度**取样,找 want 个真实存在的文件。

    InternVid 常常只下了一部分分片,连着取前 N 条既可能全不存在、又会全落在同一个
    分片里(顺序读,不能代表训练的随机访问)。跨度取样两个问题一起躲开。
    """
    n = len(anno)
    stride = max(1, n // max_checks)
    found, checks = [], 0
    for i in range(0, n, stride):
        rel = _rel(anno[i])
        checks += 1
        if rel is None:
            continue
        cand = os.path.join(data_root, rel) if data_root else rel
        if os.path.isfile(cand):
            found.append(cand)
            if len(found) >= want:
                break
    return found, checks


def probe_decode(cfg, n_reads: int = 20) -> None:
    """A. 纯解码 CPU 下界 —— 同一文件反复读,第 2 次起必然命中 page cache。"""
    from dataset.video_utils import read_frames_decord

    _hdr("A. 热解码耗时(同一文件读 %d 次;这是**无 I/O** 的解码 CPU 下界)" % n_reads)

    anno, data_root = _load_anno(cfg)
    if anno is None:
        return
    files, checks = _find_existing(anno, data_root, want=1)
    if not files:
        print(f"  [跳过] 抽查 {checks} 条标注,对应视频一个都不存在 —— 先确认分片下载与 data_root。")
        return
    path = files[0]

    print(f"  计时对象 = {path}")
    nf = cfg.num_frames
    read_frames_decord(path, nf, sample="middle")          # 预热(文件缓存/解码器初始化)
    t0 = time.time()
    for _ in range(n_reads):
        read_frames_decord(path, nf, sample="middle")
    ms = (time.time() - t0) / n_reads * 1000
    _STATE["warm_ms"] = ms
    print(f"\n  ⇒ **{ms:.0f} ms/clip**(T={nf})")
    print(f"     单 worker 外推 = {1000/ms:.2f} 样本/s;"
          f"{cfg.num_workers} workers ⇒ ≈ {1000/ms*cfg.num_workers:.1f} 样本/s"
          "  ← B 段若远低于此,说明瓶颈不是解码 CPU")


def probe_cold(cfg, n_files: int = 25) -> None:
    """A2. 冷读 —— 每个文件只碰一次,拆开 I/O 与解码。

    两组文件**互不相交**:先纯读的那批会进 page cache,拿去解码就不冷了。
    """
    from dataset.video_utils import read_frames_decord

    _hdr("A2. 冷读(每个文件只碰一次;分离「挂载盘 I/O」与「解码 CPU」)")

    anno, data_root = _load_anno(cfg)
    if anno is None:
        return
    files, checks = _find_existing(anno, data_root, want=n_files * 2)
    print(f"  抽查 {checks:,} 条标注,命中 {len(files)} 个真实存在的文件")
    if len(files) < 4:
        print("  [跳过] 命中太少,冷读统计没有意义。")
        return

    half = len(files) // 2
    read_set, dec_set = files[:half], files[half:]

    # 1) 纯读字节:不解码,只量挂载盘把文件搬进内存要多久
    read_ms, nbytes = [], 0
    for p in read_set:
        t0 = time.time()
        with open(p, "rb") as f:
            b = f.read()
        read_ms.append((time.time() - t0) * 1000)
        nbytes += len(b)
    med_read = statistics.median(read_ms)
    avg_mb = nbytes / len(read_set) / 1e6
    mbps = avg_mb / (med_read / 1000) if med_read > 0 else float("nan")
    _STATE["cold_read_ms"] = med_read
    print(f"\n  纯读字节(冷,{len(read_set)} 个文件):中位 **{med_read:.0f} ms/文件**"
          f",平均 {avg_mb:.2f} MB ⇒ ≈ {mbps:.0f} MB/s")

    # 2) 读 + 解码:另一批文件,同样每个只碰一次
    dec_ms = []
    for p in dec_set:
        t0 = time.time()
        read_frames_decord(p, cfg.num_frames, sample="rand")
        dec_ms.append((time.time() - t0) * 1000)
    med_dec = statistics.median(dec_ms)
    _STATE["cold_decode_ms"] = med_dec
    print(f"  读+解码(冷,{len(dec_set)} 个文件):中位 **{med_dec:.0f} ms/clip**")

    if med_dec > 0:
        frac = med_read / med_dec * 100
        print(f"\n  ⇒ I/O 占冷解码的 **{frac:.0f}%**"
              f"({med_read:.0f} / {med_dec:.0f} ms)")
        warm = _STATE.get("warm_ms")
        if warm:
            print(f"  ⇒ 冷 {med_dec:.0f} ms vs 热 {warm:.0f} ms = **{med_dec/warm:.1f}× 冷读惩罚**")


def probe_loader(cfg, max_batches: int = 50):
    """B. loader-only 吞吐 —— 用训练入口自己的构造函数,避免手搓配置。"""
    from tasks_clip.pretrain import setup_dataloaders

    _hdr("B. loader-only 吞吐(只迭代 dataloader,不建模型/不前向)")
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
    _STATE["loader_thr"] = thr
    print(f"\n  ⇒ **{thr:.1f} 样本/s**({i} 个 batch,{dt:.1f}s,不含首个预热 batch)")
    return dl.dataset


def probe_workers(cfg, dataset, workers_list, time_budget: float = 30.0,
                  max_batches: int = 40) -> None:
    """C. num_workers 扫描 —— **判 H1/H2 与「加 workers 就能修」的分水岭**。

    复用 B 已建好的 dataset(重建一次要 ~35 s 读 11 个 JSON),只换 workers。
    用 `shuffle=True` 而不是训练那个 stateful sampler:随机访问模式和训练一致,
    而 stateful sampler 反复取 iter 会带状态,不适合扫描。
    """
    from torch.utils.data import DataLoader

    _hdr("C. num_workers 扫描(每档最多 %.0f s / %d batch)" % (time_budget, max_batches))
    print(f"  batch_size = {cfg.batch_size};配置里的 num_workers = {cfg.num_workers}")
    print(f"  {'workers':>8} | {'样本/s':>9} | {'每 worker':>10} | {'vs 1 worker':>11}")
    print("  " + "-" * 48)

    base = None
    for w in workers_list:
        dl = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=w, shuffle=True,
            drop_last=True, pin_memory=True, persistent_workers=False,
        )
        it = iter(dl)
        try:
            next(it)                            # 预热:含 worker 启动
        except StopIteration:
            print(f"  {w:>8} | 数据不足,跳过")
            continue
        t0, n, i = time.time(), 0, 0
        for i, _b in enumerate(it, start=1):
            n += dl.batch_size
            if i >= max_batches or (time.time() - t0) > time_budget:
                break
        dt = time.time() - t0
        thr = n / dt if dt > 0 else float("nan")
        if base is None:
            base = thr
        per = thr / w
        print(f"  {w:>8} | {thr:>9.1f} | {per:>10.1f} | {thr/base:>10.2f}×")
        _STATE.setdefault("sweep", []).append((w, thr))
        del it, dl                              # 收掉 worker 进程再进下一档

    print("\n  (「每 worker」若随 workers 上升而塌,就是共享资源在饱和 —— 见末尾判读)")


def verdict(cfg) -> None:
    """把四段的数放在一起判 —— 单看任何一段都会判错(2026-07-30 就是这么判错的)。"""
    _hdr("判读")

    ncpu = os.cpu_count()
    try:
        naff = len(os.sched_getaffinity(0))
    except AttributeError:
        naff = ncpu
    print(f"  机器:os.cpu_count() = {ncpu},本进程可用核 = {naff}"
          f";配置 num_workers = {cfg.num_workers}")

    warm = _STATE.get("warm_ms")
    thr = _STATE.get("loader_thr")
    sweep = _STATE.get("sweep") or []

    if warm and thr:
        expect = 1000 / warm * cfg.num_workers
        ratio = thr / expect
        print(f"\n  ① A vs B:热解码外推 {expect:.0f} 样本/s,实测 loader {thr:.1f} "
              f"⇒ 只拿到 **{ratio*100:.0f}%**")
        if ratio < 0.4:
            print("     ⇒ **瓶颈不是解码 CPU 本身**(否则该拿到大部分)。看 ② ③ 定位。")
        else:
            print("     ⇒ loader 基本吃满了解码能力,确是解码瓶颈。")

    cr, cd = _STATE.get("cold_read_ms"), _STATE.get("cold_decode_ms")
    if cr and cd:
        frac = cr / cd
        print(f"\n  ② 冷读:I/O 占冷解码 {frac*100:.0f}%"
              + (f",冷/热 = {cd/warm:.1f}×" if warm else ""))
        if frac > 0.5:
            print("     ⇒ **H2 挂载盘 I/O 主导**。修法:打包顺序读的 shard / 落本地盘;"
                  "加 workers 只会让同一条管子更挤。")
        elif warm and cd < warm * 1.5:
            print("     ⇒ 冷读几乎不比热读贵 ⇒ **排除 H2**,瓶颈在 CPU 侧(H1 或 H3)。")
        else:
            print("     ⇒ I/O 有代价但不主导;CPU 侧仍是大头。")

    if len(sweep) >= 2:
        (w0, t0_), (wN, tN) = sweep[0], sweep[-1]
        gain = tN / t0_ if t0_ else float("nan")
        print(f"\n  ③ 扫描:workers {w0}→{wN}({wN/w0:.0f}×)吞吐 {t0_:.1f}→{tN:.1f}"
              f"(**{gain:.2f}×**)")
        if gain < wN / w0 * 0.4:
            print("     ⇒ **曲线已压平 ⇒ 加 num_workers 修不了。** "
                  "round-4 §0.2 修法阶梯的第一级(6→12–16,「免费」)作废,"
                  "直接上预解码/打包那一级。")
            if naff and cfg.num_workers >= naff * 0.8:
                print(f"     ⇒ 且 num_workers({cfg.num_workers}) 已接近可用核({naff})"
                      " ⇒ **H1 核数不够**。")
            else:
                print(f"     ⇒ 但 num_workers({cfg.num_workers}) 远小于可用核({naff})"
                      " ⇒ 不是核数,看 ② 与 H3(decord `num_threads=0` 线程超订,"
                      "`dataset/video_utils.py:137`)。")
        else:
            best = max(sweep, key=lambda x: x[1])[0]      # 扫描里吞吐最高那一档
            print("     ⇒ 吞吐仍随 workers 上升 ⇒ **round-4 的「免费修法」成立**,"
                  f"把 num_workers 从 {cfg.num_workers} 调到 **{best}**"
                  f"(本次扫描的最优档;可用核 {naff})再测一次。")

    if thr:
        print(f"\n  ④ 对照训练实测:loader-only(1 卡 {cfg.num_workers} workers)"
              f"{thr:.1f} 样本/s vs 训练(4 卡 {TRAIN_WORKERS} workers){TRAIN_THROUGHPUT} 样本/s。")
        print("     loader-only 单卡就逼近 4 卡训练全速 ⇒ **GPU 确实在空转**,§0.2 的 MFU 结论成立;"
              "分歧只在「为什么」。")
    print("\n  ⚠ 另跑 `nvidia-smi dmon -s u -c 60`:真训练时 sm 列持续 <25% 支持上面的判读。")
    print("-" * 72)


def main() -> int:
    # setup_main() 会读 argv 里的 config 并做分布式初始化 ⇒ 必须在 torchrun 下跑
    from utils.config_utils import setup_main

    def _take(flag, default):
        if flag not in sys.argv:
            return default
        i = sys.argv.index(flag)
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]          # setup_config 不认识这些参数,先摘掉
        return val

    max_batches = int(_take("--batches", 50))
    cold_files = int(_take("--cold-files", 25))
    sweep_arg = _take("--sweep", "1,2,6,12,24")
    workers_list = [] if sweep_arg.lower() == "none" else [int(x) for x in sweep_arg.split(",")]

    cfg = setup_main()

    for name, fn in (("A", lambda: probe_decode(cfg)),
                     ("A2", lambda: probe_cold(cfg, n_files=cold_files))):
        try:
            fn()
        except Exception as e:                  # 任一段失败都不该挡住后面的段
            print(f"\n[{name} 段失败,继续] {type(e).__name__}: {e}")

    dataset = None
    try:
        dataset = probe_loader(cfg, max_batches=max_batches)
    except Exception as e:
        print(f"\n[B 段失败,继续] {type(e).__name__}: {e}")

    if dataset is not None and workers_list:
        try:
            probe_workers(cfg, dataset, workers_list)
        except Exception as e:
            print(f"\n[C 段失败,继续] {type(e).__name__}: {e}")

    verdict(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
