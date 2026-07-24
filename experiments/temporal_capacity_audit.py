#!/usr/bin/env python3
"""静态时间维容量审计 —— 只读 checkpoint,不需要 GPU / 数据 / 前向。

回答一个问题:**这个训练好的模型到底有没有编码时间方向?**

背景(见 `notes/neural-inspired-round2.md` §1):视频嵌入是 8 帧嵌入的
**无权平均**,而集合的均值对任意重排都不变 —— 所以 head 是**无条件时间盲**的。
模型全部的时序敏感性只能来自 backbone 内部两处,并且必须以「所有帧共享的
clip 级信号」的形式写进每一帧,才能在平均后幸存:

1. stage 1-3 RepMixer 里的深度可分离时间卷积 `Conv1d(k=3)`;
2. stage 4 joint attention 的 `temporal_embedding`(**零初始化**)。

本脚本量化这两处的容量上界:

* `temporal_embedding` 的范数 —— 它是零初始化的。若训练后仍≈0,stage 4 对帧
  顺序就是 permutation-equivariant,再叠加均值池化 ⇒ 该通路对时间方向完全无贡献。
* 时间卷积核的**反对称度** `||w - flip(w)|| / ||w||` —— k=3 核在时间反转下
  `[a,b,c] -> [c,b,a]`,所以**对称核对时间反转完全不敏感**。这是「时间之箭」
  容量的直接度量,0 = 完全反转盲,1 = 纯差分算子。
* 时间卷积核的 **DC/AC 比** `|sum(w)| / sum(|w|)` —— 1 = 纯低通(把帧间残差
  抹平、只保留时间均值),小 = 保留/放大帧间差异。低通化是「帧残差塌缩」的
  典型征兆(见 notes 中 §2 的梯度平坦性论证)。

用法:

    python experiments/temporal_capacity_audit.py checkpoints/mobileviclip_small.pt
    python experiments/temporal_capacity_audit.py --self-test    # 不需要 checkpoint

⚠️ 本脚本**尚未在真实 checkpoint 上跑过**(Claude 侧无权重)。首次运行若键名
不匹配,它会把找到的候选键打印出来,按提示调整 --kernel-key / --embed-key 即可。
"""
from __future__ import annotations

import argparse
import sys

import torch


# --------------------------------------------------------------------------
# 诊断量(纯函数,--self-test 覆盖)
# --------------------------------------------------------------------------
def kernel_asymmetry(w: torch.Tensor) -> torch.Tensor:
    """每通道**反对称分量占比** ||w_asym|| / ||w|| ∈ [0, 1],w: [..., k]。

    把核分解为对称/反对称两部分 w = w_sym + w_asym,其中
    w_asym = (w - flip(w)) / 2,于是本量 = ||w - flip(w)|| / (2||w||)。

    时间反转把 k=3 核 [a,b,c] 变成 [c,b,a]:
      * 对称核([1,2,1]、[0,1,0]、[1,1,1])=> 0 => **对时间方向完全不敏感**;
      * 纯差分核 [-1,0,1] 是纯反对称 => 1。

    注:必须除以 2||w|| 而非 ||w||。||w - flip(w)|| = 2||w_asym||,
    不除 2 的话纯反对称核给出 2 而不是 1(2026-07-24 自检发现)。
    """
    w = w.float()
    flipped = torch.flip(w, dims=[-1])
    num = (w - flipped).norm(dim=-1)
    den = w.norm(dim=-1).clamp_min(1e-12)
    return num / (2.0 * den)


def kernel_dc_ac(w: torch.Tensor) -> torch.Tensor:
    """每通道 DC/AC 比 |sum(w)| / sum(|w|),w: [..., k]。

    1 = 纯低通(所有抽头同号,等价于时间平滑/取均值);0 = 纯高通/差分。
    """
    w = w.float()
    num = w.sum(dim=-1).abs()
    den = w.abs().sum(dim=-1).clamp_min(1e-12)
    return num / den


def _stats(x: torch.Tensor) -> str:
    x = x.flatten().float()
    q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))
    return (f"mean={x.mean():.4f}  median={q[1]:.4f}  "
            f"p05={q[0]:.4f}  p95={q[2]:.4f}  n={x.numel()}")


# --------------------------------------------------------------------------
# checkpoint 审计
# --------------------------------------------------------------------------
def _unwrap(ckpt):
    """常见容器:{'model': sd} / {'module': sd} / {'state_dict': sd} / 裸 sd。"""
    if not isinstance(ckpt, dict):
        raise SystemExit(f"不认识的 checkpoint 类型: {type(ckpt)}")
    for key in ("model", "module", "state_dict"):
        inner = ckpt.get(key)
        if isinstance(inner, dict) and any(
            isinstance(v, torch.Tensor) for v in inner.values()
        ):
            return inner
    return ckpt


def audit(sd: dict, kernel_key: str, embed_key: str) -> None:
    tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}

    # ---- 1. stage-4 temporal_embedding(零初始化,最具决定性的一个数)----
    print("=" * 72)
    print("1. stage-4 temporal_embedding(零初始化 —— 若仍≈0 则该通路时间盲)")
    print("=" * 72)
    emb = {k: v for k, v in tensors.items() if embed_key in k}
    if not emb:
        print(f"  [未找到] 没有键包含 '{embed_key}'。")
    for k, v in sorted(emb.items()):
        v = v.float()
        # [1, C, T]:逐时间位置的范数,看各位置是否真的被区分开
        per_pos = v.reshape(-1, v.shape[-1]).norm(dim=0)
        rel_spread = (per_pos.std() / per_pos.mean().clamp_min(1e-12)
                      if per_pos.mean() > 0 else torch.tensor(0.0))
        print(f"  {k}  shape={tuple(v.shape)}")
        print(f"      ||E||   = {v.norm():.6f}   (零初始化;≈0 ⇒ 训练从未使用)")
        print(f"      逐位置范数 = {[round(float(x), 4) for x in per_pos]}")
        print(f"      位置间相对离散度 = {float(rel_spread):.4f}"
              "   (≈0 ⇒ 各帧位置无区分 ⇒ 对帧顺序等变)")

    # ---- 2. 时间卷积核 ----
    print()
    print("=" * 72)
    print("2. 时间维深度卷积核(反对称度 = 时间之箭容量;DC/AC = 低通程度)")
    print("=" * 72)
    kernels = {k: v for k, v in tensors.items()
               if kernel_key in k and k.endswith("weight") and v.dim() >= 2}
    if not kernels:
        print(f"  [未找到] 没有键包含 '{kernel_key}' 且以 'weight' 结尾。")
        cand = sorted(k for k in tensors
                      if ("_1d" in k or "temporal" in k) and k.endswith("weight"))
        if cand:
            print("  候选(含 '_1d' / 'temporal' 的权重键),用 --kernel-key 指定:")
            for k in cand[:25]:
                print(f"    {k}   shape={tuple(tensors[k].shape)}")
            if len(cand) > 25:
                print(f"    ... 另有 {len(cand) - 25} 个")
        print()
        print("  注:训练态 checkpoint 里时间分支尚未 reparameterize,等效核是")
        print("      id_tensor + layer_scale_1d*(mixer_1d - norm_1d) 的融合结果。")
        print("      若只找到 mixer_1d/norm_1d 的分支权重,请先把模型 build 起来")
        print("      调用 reparameterize() 后再存一份推理态权重,再跑本脚本。")
        return

    all_asym, all_dc = [], []
    for k, v in sorted(kernels.items()):
        w = v.float().reshape(v.shape[0], -1) if v.dim() > 2 else v.float()
        if w.shape[-1] < 2:
            continue
        a, d = kernel_asymmetry(w), kernel_dc_ac(w)
        all_asym.append(a)
        all_dc.append(d)
        print(f"  {k}  shape={tuple(v.shape)}")
        print(f"      反对称度  {_stats(a)}")
        print(f"      DC/AC     {_stats(d)}")

    if all_asym:
        print()
        print("-" * 72)
        A = torch.cat([x.flatten() for x in all_asym])
        D = torch.cat([x.flatten() for x in all_dc])
        print(f"  全部时间核 反对称度: {_stats(A)}")
        print(f"  全部时间核 DC/AC   : {_stats(D)}")
        print("-" * 72)
        print("  判读:反对称度中位数 <0.05 ⇒ 时间核近似对称 ⇒ 对时间反转基本不敏感;")
        print("        DC/AC 中位数 >0.95 ⇒ 近似纯低通 ⇒ 帧间残差被抹平。")
        print("        两者同时成立 ⇒ 模型几乎不编码时间方向,RTime-Binary 预期≈50%。")


def _self_test() -> None:
    """在合成核上验证两个诊断量的定义(不需要 checkpoint)。"""
    cases = {
        "identity  [0,1,0]": torch.tensor([[0.0, 1.0, 0.0]]),
        "symmetric [1,2,1]": torch.tensor([[1.0, 2.0, 1.0]]),
        "difference[-1,0,1]": torch.tensor([[-1.0, 0.0, 1.0]]),
        "lowpass   [1,1,1]": torch.tensor([[1.0, 1.0, 1.0]]),
        "asym      [0,1,2]": torch.tensor([[0.0, 1.0, 2.0]]),
    }
    print("self-test —— 反对称度 / DC-AC 在已知核上的取值")
    for name, w in cases.items():
        print(f"  {name:20s} asym={float(kernel_asymmetry(w)):.4f}  "
              f"dc/ac={float(kernel_dc_ac(w)):.4f}")
    # 断言:对称核反对称度必须为 0;纯反对称核必须为 1(归一化到 [0,1]);
    #       纯差分核 DC 必须为 0;全同号核 DC/AC 必须为 1。
    assert float(kernel_asymmetry(cases["symmetric [1,2,1]"])) < 1e-6
    assert float(kernel_asymmetry(cases["identity  [0,1,0]"])) < 1e-6
    assert abs(float(kernel_asymmetry(cases["difference[-1,0,1]"])) - 1.0) < 1e-6
    assert float(kernel_dc_ac(cases["difference[-1,0,1]"])) < 1e-6
    assert float(kernel_dc_ac(cases["lowpass   [1,1,1]"])) > 1 - 1e-6
    # 时间反转不变性:反对称度对核翻转不变
    w = cases["asym      [0,1,2]"]
    assert abs(float(kernel_asymmetry(w))
               - float(kernel_asymmetry(torch.flip(w, dims=[-1])))) < 1e-6
    print("self-test 通过。")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", nargs="?", help="checkpoint 路径(.pt)")
    p.add_argument("--kernel-key", default="reparam_conv_1d",
                   help="时间卷积核的键名子串(默认推理态 reparam_conv_1d)")
    p.add_argument("--embed-key", default="temporal_embedding",
                   help="stage-4 时间位置编码的键名子串")
    p.add_argument("--self-test", action="store_true",
                   help="只跑诊断量的自检,不读 checkpoint")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.ckpt:
        p.error("需要 checkpoint 路径,或使用 --self-test")

    print(f"读取 {args.ckpt} ...")
    sd = _unwrap(torch.load(args.ckpt, map_location="cpu"))
    print(f"共 {sum(1 for v in sd.values() if isinstance(v, torch.Tensor))} 个张量\n")
    audit(sd, args.kernel_key, args.embed_key)


if __name__ == "__main__":
    sys.exit(main())
