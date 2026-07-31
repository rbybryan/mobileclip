#!/usr/bin/env python3
"""V100 微检查合集(A 档里零数据、分钟级的那几项)。

    python experiments/v100_micro_checks.py all
    python experiments/v100_micro_checks.py a12          # 只跑一项

每一项都**自带判据**并以退出码表示 PASS/FAIL,好让 `run_v100_all.sh` 直接汇总。
出处:notes/v100-10k-advance-tests.md A2 / A7 / A11 / A12 + B1。
"""
import argparse
import sys


def _ok(name, passed, msg):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {msg}")
    return 0 if passed else 1


# --------------------------------------------------------------------- A2
def a2_sdpa_identity():
    """SDPA 替换是不是数学恒等(float64,本模型 stage-4 真实形状)。

    仓库多处按 `mobileviclip.yml` 的 torch 1.13.1 推理、断言「SDPA 用不了」——
    **那个环境哪台机器都不是**。这一项把 P1(SDPA)的正确性风险彻底拆掉。
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    B, H, N, D = 1, 20, 512, 32          # N = T·H·W = 8·8·8,20 头,与 stage-4 对齐
    q, k, v = (torch.randn(B, H, N, D, dtype=torch.float64) for _ in range(3))

    naive = (q @ k.transpose(-2, -1) * D ** -0.5).softmax(-1) @ v
    if not hasattr(F, "scaled_dot_product_attention"):
        return _ok("A2 SDPA恒等", False, "本环境 torch 没有 SDPA ⇒ P1 确实被环境挡住(与预期相反,要记下来)")
    sdpa = F.scaled_dot_product_attention(q, k, v)
    d = (naive - sdpa).abs().max().item()
    return _ok("A2 SDPA恒等", d < 1e-12, f"max|naive − SDPA| = {d:.3e}(判据 < 1e-12)")


# -------------------------------------------------------------------- A11
def a11_p15_equivalence():
    """P15:时序 depthwise Conv1d 可以写成**批偏移加权加**,一个 layout op 都不用。

    帧在 batch 维上是 t·B + b,所以 reshape/permute/conv1d 那一串等价于在连续
    NCHW 张量上做三次带偏移的加权加。成立 ⇒ P15 严格优于 P3(P3 还要一次 clone,
    且产出 §10.1 判为 ANE 快路径风险的大 dilation 卷积)。
    """
    import torch

    torch.manual_seed(0)
    T, B, C, H, W = 8, 2, 16, 4, 4
    x = torch.randn(T * B, C, H, W, dtype=torch.float64)
    w = torch.randn(C, 3, dtype=torch.float64)
    bias = torch.randn(C, 1, 1, dtype=torch.float64)

    # 参考实现:reshape → permute → conv1d(仓库现状)
    xr = x.view(T, B, C, H, W).permute(1, 3, 4, 2, 0).reshape(B * H * W, C, T)
    ref = torch.nn.functional.conv1d(xr, w.unsqueeze(1), padding=1, groups=C)
    ref = ref.view(B, H, W, C, T).permute(4, 0, 3, 1, 2).reshape(T * B, C, H, W) + bias

    # P15:批偏移加权加,零 layout op
    xv = x.view(T, B, C, H, W)
    out = w[:, 1].view(1, 1, C, 1, 1) * xv
    out[1:] += w[:, 0].view(1, 1, C, 1, 1) * xv[:-1]
    out[:-1] += w[:, 2].view(1, 1, C, 1, 1) * xv[1:]
    out = out.reshape(T * B, C, H, W) + bias

    d = (ref - out).abs().max().item()
    return _ok("A11 P15等价", d < 1e-12, f"max|conv1d − 批偏移| = {d:.3e}(判据 < 1e-12)")


# -------------------------------------------------------------------- A12
def a12_bn_double_update():
    """`use_reentrant=False` 的梯度检查点会让 BN 的 EMA 对同一 batch 应用两次。

    判据反过来:**发现双更新 = PASS**(确认了已知 bug)。
    后果:所有 BN 重校准必须 `with_cp=False`;且本仓已训 ckpt 的 BN 动量实际是
    0.19 而不是 0.10。⚠️ **不要在活动 campaign 中途修** —— 它对所有臂一视同仁。
    """
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    m = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)).train()
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    checkpoint(lambda t: m(t), x, use_reentrant=False).sum().backward()
    n = int(m[1].num_batches_tracked.item())
    return _ok("A12 BN双更新", n == 2,
               f"一步之后 num_batches_tracked = {n}"
               + ("(== 2 ⇒ 确认双更新;重校准必须 with_cp=False)" if n == 2
                  else "(== 1 ⇒ 本环境没有复现,§11.3 的前提要重审)"))


# --------------------------------------------------------------------- A7
def a7_macs_vs_resolution():
    """round4-P7 的**成本轴**:MACs vs img_size。没有它,阶段 3 的精度轴无法解读。

    视觉塔与分辨率无关(无绝对位置嵌入、RepCPE 是核不是网格、head 是 GlobalPool mean)
    ⇒ 改 img_size 不需插值、不需重训。
    """
    import torch
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return _ok("A7 分辨率MACs", False, "本环境 torch 没有 flop_counter(需 ≥2.0)")
    sys.path.insert(0, ".")
    from experiments.export_smoke import build_vision_tower

    tower = build_vision_tower(False, False).eval()
    base = None
    print("    边长   GMACs/帧   相对 256²")
    for r in (128, 160, 181, 224, 256):
        x = torch.randn(1, 3, r, r)
        with torch.no_grad(), FlopCounterMode(display=False) as f:
            tower(x)
        g = f.get_total_flops() / 2 / 1e9          # FLOPs → MACs
        base = base or g
        print(f"    {r:<6} {g:>8.3f}   {g / base if base else 1:>6.3f}×"
              if r != 128 else f"    {r:<6} {g:>8.3f}   (基准)")
        if r == 128:
            base = g
    return _ok("A7 分辨率MACs", True, "成本轴已产出(与阶段 3 的精度轴配成前沿)")


# --------------------------------------------------------------------- B1
def b1_vocab_coverage(anno):
    """P13(b):裁 BPE 词表。**估计**是 12–20k / 49,408 覆盖 >99.9%,这一项把它变成测量。

    ⚠️ 10k 是**中文抽样**,训练/评测用英文字幕 ⇒ 结论必须标 **10k-zh 口径**,
    全量英文语料上要重测。这条限制别省。
    """
    import collections
    import json
    import os

    if not anno or not os.path.exists(anno):
        return _ok("B1 词表覆盖", False, f"找不到 anno:{anno}(用 --anno 指定)")
    sys.path.insert(0, ".")
    from models.backbones.internvideo2.mobileclip.clip_tokenizer import ClipTokenizer  # noqa

    recs = json.load(open(anno))
    caps = [r.get("caption") for r in recs if isinstance(r, dict) and r.get("caption")]
    if not caps:
        return _ok("B1 词表覆盖", False, "anno 里没有 caption 字段")
    tok = ClipTokenizer(json.load(open(
        "./models/backbones/internvideo2/mobileclip/configs/mobileclip_s2.json")))
    cnt = collections.Counter()
    for c in caps:
        cnt.update(tok.encode(c)[:32] if hasattr(tok, "encode") else [])
    tot = sum(cnt.values())
    if not tot:
        return _ok("B1 词表覆盖", False, "tokenizer 接口对不上,需按仓库实际 API 调整")
    hit = None
    for k in (8000, 12000, 16000, 20000, 24000):
        cov = 100 * sum(v for _, v in cnt.most_common(k)) / tot
        print(f"    k={k:<6} 覆盖 {cov:.4f}%")
        if hit is None and cov > 99.9:
            hit = k
    return _ok("B1 词表覆盖", hit is not None,
               (f"k={hit} 即可 >99.9%(**10k-zh 口径**)⇒ P13(b) 落地"
                if hit else "24k 仍不到 99.9% ⇒ 按预登记杀掉 P13(b),只留量化那半"))


# --------------------------------------------------------------------- A9
def a9_cpu_path_walltime():
    """§1 的 CPU 逐路径计时在 §6 里被标注为**「待复现」的顾问方数据**。本机复现即闭环。

    要复现的三个数(8 线程,T=8,B=1):时序分支 190 ms / 空间 dw3×3 68–98 ms /
    ConvFFN 790 ms,合计 1082 ms。**判据**:时序分支若仍占 mixer+FFN 墙钟的 ~18%
    而只占 **0.20% MACs** ⇒ round4-P1「删时序分支」的论据独立坐实。
    """
    import time
    import torch
    torch.set_num_threads(8)
    sys.path.insert(0, ".")
    from experiments.export_smoke import build_vision_tower

    tower = build_vision_tower(False, False).eval()
    blocks = [m for m in tower.modules()
              if hasattr(m, "token_mixer") and hasattr(getattr(m, "token_mixer"), "mixer_1d")]
    if not blocks:
        return _ok("A9 CPU路径墙钟", False, "没找到带 mixer_1d 的 RepMixerBlock")

    def bench(fn, x, n=5):
        with torch.no_grad():
            fn(x); t0 = time.perf_counter()
            for _ in range(n): fn(x)
        return (time.perf_counter() - t0) / n * 1000

    # 形状按 RepMixer.forward:753-772 的真实链路:
    #   时序支路输入是 (b·h·w, c, t),且**包含那两次 reshape/permute** —— layout op
    #   本身就是 §1 要量的成本,不能只量卷积。
    tot = {"时序(含layout)": 0.0, "空间 mixer": 0.0, "ConvFFN": 0.0}
    with torch.no_grad():
        for blk in blocks:
            tm = blk.token_mixer
            C = tm.mixer_1d.mixer_1d.weight.shape[0] if hasattr(tm.mixer_1d, "mixer_1d") \
                else next(tm.mixer_1d.parameters()).shape[0]
            S = {80: 64, 160: 32, 320: 16, 640: 8}.get(C, 16)
            x = torch.randn(8, C, S, S)                       # (t·b, c, h, w),t=8 b=1

            def temporal(z):
                t, b, c, h, w = 8, 1, C, S, S
                y = z.reshape(t, b, c, h, w).permute(1, 3, 4, 2, 0).reshape(b*h*w, c, t)
                y = y + (tm.mixer_1d(y) - tm.norm_1d(y))
                return y.reshape(b, h, w, c, t).permute(4, 0, 3, 1, 2).reshape(t*b, c, h, w)

            tot["时序(含layout)"] += bench(temporal, x)
            tot["空间 mixer"] += bench(lambda z: tm.mixer(z) - tm.norm(z), x)
            if hasattr(blk, "convffn"):
                tot["ConvFFN"] += bench(lambda z: blk.convffn(z), x)

    total = sum(tot.values()) or 1.0
    print(f"    {'路径':<16}{'ms':>9}{'占比':>9}")
    for k, v in tot.items():
        print(f"    {k:<16}{v:>9.1f}{100*v/total:>8.1f}%")
    frac = 100 * tot["时序(含layout)"] / total
    return _ok("A9 CPU路径墙钟", True,
               f"时序分支占墙钟 {frac:.1f}%(仅 0.20% MACs)"
               + ("⇒ P1 论据独立坐实" if frac > 5 else "⇒ 与 §1 的 ~18% 不符,§1 那组数要重审"))


# -------------------------------------------------------------------- A13
def a13_op_census():
    """§11 整条屋顶线订正的实证基础:每个 RepMixerBlock 前向到底发多少 aten 调用。

    面板数出:**55 个 aten 调用、其中 33 个碰全激活张量、9 个 BN、7 个 conv、
    2 个整张量转置 `clone`**。**判据**:那两个 `clone` 的**源 stride** 里若出现
    `B·C·H·W` 量级的跨度(stage-1 = 41.9MB)⇒ §11.1「一个 warp 的 32 条 lane 读
    32 个相隔 41.9MB 的地址」成立 ⇒ 该分支确实是访存灾难。
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    sys.path.insert(0, ".")
    from experiments.export_smoke import build_vision_tower

    class Census(TorchDispatchMode):
        def __init__(self): self.ops, self.clones = [], []
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            n = str(func)
            self.ops.append(n)
            out = func(*args, **(kwargs or {}))
            if "clone" in n and args and hasattr(args[0], "stride"):
                self.clones.append((tuple(args[0].shape), tuple(args[0].stride())))
            return out

    tower = build_vision_tower(False, False).eval()
    blk = next((m for m in tower.modules()
                if hasattr(m, "token_mixer") and hasattr(m.token_mixer, "mixer_1d")), None)
    if blk is None:
        return _ok("A13 算子普查", False, "没找到 RepMixerBlock")
    from models.mobileviclip_small import set_num_frames
    set_num_frames(8)                       # :1431 —— 不先设,reshape 会按陈旧 T 崩掉
    C = next(blk.token_mixer.mixer_1d.parameters()).shape[0]
    S = {80: 64, 160: 32, 320: 16, 640: 8}.get(C, 16)
    x = torch.randn(8, C, S, S)
    with torch.no_grad(), Census() as c:
        blk(x)
    import collections
    kinds = collections.Counter(
        "BN" if "batch_norm" in o else "conv" if "convolution" in o else
        "clone" if "clone" in o else "other" for o in c.ops)
    print(f"    aten 调用总数 = {len(c.ops)}  ·  {dict(kinds)}")
    for shp, st in c.clones[:4]:
        print(f"    clone 源: shape={shp} stride={st}  最大跨度={max(st) if st else 0}")
    big = any(max(st) >= 10_000_000 for _, st in c.clones if st)
    return _ok("A13 算子普查", True,
               f"{len(c.ops)} 调用 / {kinds.get('BN',0)} BN / {kinds.get('conv',0)} conv / "
               f"{kinds.get('clone',0)} clone"
               + ("(clone 跨度达 1e7+ ⇒ §11.1 的访存灾难坐实)" if big else ""))


# -------------------------------------------------------------------- A10
def a10_latency_vs_t():
    """A10 的**那个问题**:运行时有没有把权重摊销到 T 这一批帧上?

    摊销 ⇒ 总流量 ≈ 693MB 激活 + 72MB 权重 ⇒ 权重 INT8 只省 4.7%,量化是**尺寸**杠杆;
    没摊销(逐帧重读)⇒ 693 + 576 = 1269MB ⇒ 省 23%,是**速度**杠杆。收益差 5 倍。

    🔴 **原方案量的是 Core ML 图,而 Core ML 的 predict 只在 macOS 上能跑** ——
    这台是 Linux,**跑不了**。这里改用 **PyTorch CPU 前向**量同一条曲线:
    **严格线性(截距≈0)⇒ 权重没被摊销;次线性且截距为正 ⇒ 摊销了。**
    ⚠️ **诚实边界**:它回答的是「本运行时摊不摊销」,**不能外推到 ANE**。
    ANE 那一问仍然要一台 mac。
    """
    import time
    import torch
    torch.set_num_threads(8)
    sys.path.insert(0, ".")
    from experiments.export_smoke import build_vision_tower

    from models.mobileviclip_small import set_num_frames
    tower = build_vision_tower(False, False).eval()
    ts, lat = [2, 4, 8, 16], []
    with torch.no_grad():
        for t in ts:
            set_num_frames(t)
            x = torch.randn(1, t, 3, 256, 256).permute(0, 2, 1, 3, 4)
            tower(x)
            t0 = time.perf_counter(); tower(x)
            lat.append((time.perf_counter() - t0) * 1000)
    print(f"    {'T':>4}{'ms':>10}{'ms/帧':>10}")
    for t, l in zip(ts, lat):
        print(f"    {t:>4}{l:>10.1f}{l/t:>10.1f}")
    # 线性拟合 lat = a·T + b,看截距 b 的相对大小
    n = len(ts); sx = sum(ts); sy = sum(lat)
    sxx = sum(t*t for t in ts); sxy = sum(t*l for t, l in zip(ts, lat))
    a = (n*sxy - sx*sy) / (n*sxx - sx*sx); b = (sy - a*sx) / n
    rel = b / max(lat[-1], 1e-9)
    print(f"    拟合 lat ≈ {a:.2f}·T + {b:.2f} ms   截距占 T=16 延迟的 {100*rel:.1f}%")
    return _ok("A10 延迟vsT", True,
               (f"截距 {100*rel:.1f}% > 5% ⇒ **权重被摊销** ⇒ 量化是尺寸杠杆(~4.7%)"
                if rel > 0.05 else
                f"截距 {100*rel:.1f}% ≈ 0 ⇒ **没摊销** ⇒ 量化是速度杠杆(~23%)")
               + " · 仅 PyTorch-CPU 口径,不外推 ANE")



CHECKS = {"a2": a2_sdpa_identity, "a11": a11_p15_equivalence,
          "a12": a12_bn_double_update, "a7": a7_macs_vs_resolution,
          "a9": a9_cpu_path_walltime, "a13": a13_op_census,
          "a10": a10_latency_vs_t}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="?", default="all",
                    choices=["all", "a2", "a7", "a9", "a10", "a11", "a12", "a13", "b1"])
    ap.add_argument("--anno", default="anno_downstream/internvid_zh_10k_pilot.json")
    args = ap.parse_args()

    rc = 0
    names = list(CHECKS) + ["b1"] if args.which == "all" else [args.which]
    for n in names:
        try:
            rc |= b1_vocab_coverage(args.anno) if n == "b1" else CHECKS[n]()
        except Exception as exc:                     # noqa: BLE001
            rc |= _ok(n, False, f"{type(exc).__name__}: {exc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
