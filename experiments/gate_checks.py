"""架构第四轮零 GPU 闸门(`notes/architecture-round4.md` §3 的 ①③④⑤)。

**为什么需要这个文件,而不是几段 heredoc。** 2026-07-29 首次在服务器上跑那几段 heredoc,
三条全废,而且**其中一条是静默失败**:

* ③ `layer_scale_1d = 0` —— 匹配到 **0 个键**,脚本照样把 ckpt 存了出去 ⇒ `_ablate.pt`
  与原文件逐位相同,评测会"通过"但什么都没消融;
* ⑤ 读 `layer_scale` —— `torch.cat(): expected a non-empty list`;
* ① 比对空间 `layer_scale` —— 输出 `0 / 0`(一个都没匹配上,却打印得像结论)。

根因一句话:**发布的 `mobileviclip_small.pt` 是重参数化(推理态)权重。**
`RepMixer.reparameterize()`(`models/mobileviclip_small.py:753-824`)会
`__delattr__` 掉 `mixer/norm/layer_scale/mixer_1d/norm_1d/layer_scale_1d`,
把它们折进 `reparam_conv` / `reparam_conv_1d`。所以那些键在发布权重里**根本不存在**。

⇒ 本文件的两条纪律:**(a) 先探测 ckpt 处于哪种形态再动手;(b) 任何一步匹配到 0 个键
就立刻抛错,绝不落盘。**

--------------------------------------------------------------------------------
重参数化形态下,闸门该怎么算(这比原设想更直接)
--------------------------------------------------------------------------------
折叠公式(`:766-777`):`w_1d = id_tensor + layer_scale_1d · (mixer_1d.w − norm_1d.w)`

也就是说**融合核里已经含着 `δ + λ·Δ` 这个和**,不必再去找 λ:

* **⑤ INT8 判据** —— 直接量 `|w_1d − δ|`(δ = 深度可分 Conv1d 的中心抽头单位核)。
  每通道对称 INT8 的尺度由最大元素(≈identity 抽头 1)定 ⇒ `s ≈ 1/127`,
  抹零阈值 `s/2 ≈ 3.94e-3`。`max|w_1d − δ| < 阈值` 的通道,其时序混合项在 INT8 下**被整个抹掉**。
* **③ 删时序分支** —— 令 `reparam_conv_1d.weight := δ`、`bias := 0`。这与
  `layer_scale_1d = 0` **数学等价**,也正是 INT8 对该支路做的事。

⚠️ **一个必须诚实标注的点**:`notes/architecture-round4.md` §2 P2 引用的
「λ 中位 8e-4 / 空间 1.3e-2」**不可能来自这份发布权重**(键不存在)。在 ⑤ 用本文件
复现出真实数字之前,那两个数按**未核实**处理,P2 的 INT8 结论也随之待定。

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
    python experiments/gate_checks.py inspect  --ckpt checkpoints/mobileviclip_small.pt
    python experiments/gate_checks.py gate5    --ckpt checkpoints/mobileviclip_small.pt
    python experiments/gate_checks.py gate3    --ckpt checkpoints/mobileviclip_small.pt --out checkpoints/_ablate_temporal.pt
    python experiments/gate_checks.py gate4    --ckpt checkpoints/mobileviclip_small.pt --out checkpoints/_ablate_tempemb.pt
    python experiments/gate_checks.py gate1    --ckpt checkpoints/mobileviclip_small.pt --image-ckpt checkpoints/mobileclip_s2.pt
    python experiments/gate_checks.py self-test          # 零依赖,合成权重,验证全部逻辑

跑完 gate3/gate4 后按 `COMMANDS.md` §3 的零样本评测跑 MSR-VTT,对比 t2v 42.50 / r_mean 61.55。
判据:③ Δr_mean > 1.0(raw 与 DSL 同向)才保留该支路;④ Δ > 0.5 才保留 embedding。
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import torch

#: 每通道对称 INT8:尺度由 identity 抽头(≈1)定,round-to-nearest 的抹零阈值是 s/2。
INT8_SCALE = 1.0 / 127.0
INT8_ZERO_THRESHOLD = INT8_SCALE / 2.0


def load_state(path: str) -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "module", "state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def require(keys: List[str], what: str, ckpt: str) -> None:
    """匹配到 0 个键 ⇒ 抛错。**这是本文件存在的理由。**"""
    if not keys:
        raise SystemExit(
            f"\n[FAIL] 在 {ckpt} 里没有匹配到任何 {what}。\n"
            f"       这不是「没有效应」,是**键名假设错了** —— 绝不能当作结论,\n"
            f"       更不能继续落盘。先跑:\n"
            f"         python experiments/gate_checks.py inspect --ckpt {ckpt}\n"
        )


def detect_mode(sd: Dict[str, torch.Tensor]) -> str:
    has_reparam = any(".reparam_conv_1d." in k for k in sd)
    has_train = any(k.endswith("layer_scale_1d") for k in sd)
    if has_reparam and not has_train:
        return "reparameterized"
    if has_train and not has_reparam:
        return "training"
    if has_reparam and has_train:
        return "mixed"
    return "unknown"


def identity_kernel_like(w: torch.Tensor) -> torch.Tensor:
    """深度可分 Conv1d 的单位核 δ:形状 [C,1,k],中心抽头 = 1,其余 0。"""
    idk = torch.zeros_like(w)
    idk[:, :, w.shape[-1] // 2] = 1.0
    return idk


def temporal_conv_keys(sd: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(k for k in sd if k.endswith("reparam_conv_1d.weight"))


# ----------------------------------------------------------------- inspect


def cmd_inspect(args) -> int:
    sd = load_state(args.ckpt)
    mode = detect_mode(sd)
    print(f"ckpt      : {args.ckpt}")
    print(f"tensors   : {len(sd)}")
    print(f"形态      : {mode}")
    print()
    buckets = {
        "reparam_conv_1d.weight (融合时序核)": [k for k in sd if k.endswith("reparam_conv_1d.weight")],
        "reparam_conv_1d.bias": [k for k in sd if k.endswith("reparam_conv_1d.bias")],
        "reparam_conv.weight (融合空间核)": [k for k in sd if k.endswith("reparam_conv.weight")],
        "layer_scale_1d (训练态)": [k for k in sd if k.endswith("layer_scale_1d")],
        "layer_scale (训练态)": [k for k in sd if k.endswith("layer_scale")],
        "temporal_embedding": [k for k in sd if "temporal_embedding" in k],
    }
    for name, ks in buckets.items():
        print(f"  {name:38s} {len(ks):4d}" + (f"   e.g. {ks[0]}" if ks else ""))
    print()
    if mode == "reparameterized":
        print("⇒ 推理态权重:layer_scale* 已被 reparameterize() 删除,折进 reparam_conv*。")
        print("  gate3/gate5 会走融合核路径(量 |w_1d − δ|),这是等价且更直接的做法。")
    elif mode == "training":
        print("⇒ 训练态权重:直接读 layer_scale_1d。")
    else:
        print("⚠️ 形态未知 —— 先看上面的键分布再决定,别硬跑。")
    return 0


# ----------------------------------------------------------------- gate 5


def cmd_gate5(args) -> int:
    sd = load_state(args.ckpt)
    mode = detect_mode(sd)
    print(f"形态: {mode}   INT8 抹零阈值 = {INT8_ZERO_THRESHOLD:.3e}\n")

    if mode == "training":
        ks = [k for k in sd if k.endswith("layer_scale_1d")]
        require(ks, "layer_scale_1d", args.ckpt)
        v = torch.cat([sd[k].flatten().abs().float() for k in ks])
        print(f"layer_scale_1d  n={v.numel()}  median={v.median():.3e}  p95={v.quantile(.95):.3e}")
        print("  (训练态只能给 λ;要判 INT8 得看融合后的 |w_1d − δ|,请用推理态权重)")
        return 0

    ks = temporal_conv_keys(sd)
    require(ks, "reparam_conv_1d.weight(融合时序核)", args.ckpt)

    per_ch_max: List[torch.Tensor] = []
    for k in ks:
        w = sd[k].float()                       # [C,1,k]
        dev = (w - identity_kernel_like(w)).abs()
        per_ch_max.append(dev.amax(dim=(1, 2)))  # 每通道最大偏离
    dev_all = torch.cat(per_ch_max)
    erased = (dev_all < INT8_ZERO_THRESHOLD).float().mean().item()

    print(f"融合时序核 |w_1d − δ|,逐通道最大值   (blocks={len(ks)}, channels={dev_all.numel()})")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"    p{int(q*100):02d} = {dev_all.quantile(q):.3e}")
    print(f"\n  低于 INT8 抹零阈值的通道占比: {100*erased:.1f}%")
    print("  ⇒ " + (
        "**坐实**:多数通道的时序混合项在 PTQ INT8 下被整个抹掉,"
        "等价于 layer_scale_1d=0(round4 §2 P2 成立)"
        if erased > 0.5 else
        "**未坐实**:融合核里的时序项在 INT8 下存活 ⇒ round4 §2 P2 的机制不成立,"
        "该结论需要撤回/改写"))

    # 对照:空间核,用来说明这不是「所有核都被抹」的平凡结果
    sk = [k for k in sd if k.endswith("reparam_conv.weight")]
    if sk:
        sv = []
        for k in sk:
            w = sd[k].float()
            if w.shape[-1] != w.shape[-2] or w.shape[1] != 1:
                continue                        # 只看深度可分方核
            idk = torch.zeros_like(w)
            idk[:, :, w.shape[-2] // 2, w.shape[-1] // 2] = 1.0
            sv.append((w - idk).abs().amax(dim=(1, 2, 3)))
        if sv:
            s = torch.cat(sv)
            print(f"\n  对照 · 空间融合核 |w − δ| 中位 = {s.median():.3e}  "
                  f"(低于阈值占比 {100*(s < INT8_ZERO_THRESHOLD).float().mean():.1f}%)")
    return 0


# ----------------------------------------------------------------- gates 3 / 4


def _save(sd_full, out: str) -> None:
    torch.save(sd_full, out)
    print(f"\n[OK] 已写出 {out}")
    print("     按 COMMANDS.md §3 跑 MSR-VTT 零样本评测,与 t2v 42.50 / r_mean 61.55 对比。")


def cmd_gate3(args) -> int:
    """删时序分支:融合核 := δ,bias := 0(等价 layer_scale_1d = 0)。"""
    full = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = full
    for k in ("model", "module", "state_dict"):
        if isinstance(full, dict) and k in full and isinstance(full[k], dict):
            sd = full[k]
            break
    mode = detect_mode({k: v for k, v in sd.items() if isinstance(v, torch.Tensor)})

    n = 0
    if mode == "training":
        ks = [k for k in sd if k.endswith("layer_scale_1d")]
        require(ks, "layer_scale_1d", args.ckpt)
        for k in ks:
            sd[k] = torch.zeros_like(sd[k]); n += 1
    else:
        ks = [k for k in sd if k.endswith("reparam_conv_1d.weight")]
        require(ks, "reparam_conv_1d.weight", args.ckpt)
        for k in ks:
            w = sd[k]
            before = (w - identity_kernel_like(w.float())).abs().max().item()
            sd[k] = identity_kernel_like(w.float()).to(w.dtype)
            bk = k.replace(".weight", ".bias")
            if bk in sd:
                sd[bk] = torch.zeros_like(sd[bk])
            n += 1
            if n == 1:
                print(f"  样例 {k}: 改前 max|w−δ| = {before:.3e} → 改后 0")
    print(f"[gate3] 时序分支已中和,涉及 {n} 处(形态={mode})")
    _save(full, args.out)
    return 0


def cmd_gate4(args) -> int:
    full = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = full
    for k in ("model", "module", "state_dict"):
        if isinstance(full, dict) and k in full and isinstance(full[k], dict):
            sd = full[k]
            break
    ks = [k for k in sd if "temporal_embedding" in k and isinstance(sd[k], torch.Tensor)]
    require(ks, "temporal_embedding", args.ckpt)
    for k in ks:
        print(f"  {k}  shape={tuple(sd[k].shape)}  ‖·‖={sd[k].float().norm():.4f} → 0")
        sd[k] = torch.zeros_like(sd[k])
    print(f"[gate4] temporal_embedding 已置零,涉及 {len(ks)} 处")
    _save(full, args.out)
    return 0


# ----------------------------------------------------------------- gate 1


def cmd_gate1(args) -> int:
    """空间参数是否与图像 ckpt 逐位相同 ⇒ 骨干是否被冻结。

    发布权重是推理态,`layer_scale` 已不存在,所以改比对**融合空间核**
    `reparam_conv.weight`;若图像 ckpt 是训练态则无法直接比,脚本会如实说明而不是打印 0/0。
    """
    a, b = load_state(args.ckpt), load_state(args.image_ckpt)
    print(f"video ckpt 形态: {detect_mode(a)}   image ckpt 形态: {detect_mode(b)}")

    def suffix_index(sd):
        idx = {}
        for k in sd:
            idx.setdefault(k.split("vision_encoder.")[-1], []).append(k)
        return idx

    bi = suffix_index(b)
    cand_tags = ("reparam_conv.weight", "layer_scale")
    total = same = diff = nomatch = 0
    for k, v in a.items():
        if not any(k.endswith(t) for t in cand_tags) or "_1d" in k:
            continue
        total += 1
        hits = bi.get(k.split("vision_encoder.")[-1], [])
        if not hits:
            nomatch += 1
            continue
        w = b[hits[0]]
        if w.shape != v.shape:
            nomatch += 1
            continue
        same += bool(torch.equal(v.float(), w.float())); diff += (not torch.equal(v.float(), w.float()))

    print(f"可比对的空间张量: {total}   逐位相同: {same}   不同: {diff}   对不上名字/形状: {nomatch}")
    if total == 0 or same + diff == 0:
        print("\n⚠️ **无法判定** —— 两份 ckpt 的键空间对不上(很可能一份是推理态、一份是训练态)。")
        print("   不要把这读成『没有证据』。替代做法:在服务器上按同一配置各自 build 模型、")
        print("   `reparameterize()` 后再比,或直接比 `patch_embed`/stem 这类两边都存在的张量。")
        return 2
    frac = same / max(same + diff, 1)
    print("\n⇒ " + (
        f"**{100*frac:.0f}% 逐位相同 = 骨干被冻结** ⇒ GATE-0 腿②③ 混淆,只有腿①(行为学)可用"
        if frac > 0.8 else
        f"仅 {100*frac:.0f}% 相同 ⇒ 空间塔确实训过 ⇒ 腿②③ 的『同初值』前提成立,GATE-0 维持原判"))
    return 0


# ----------------------------------------------------------------- self-test


def cmd_selftest(args) -> int:
    """合成两种形态的权重,验证探测/中和/阈值/抛错四条逻辑。不需要任何真 ckpt。"""
    import tempfile, os

    C, K = 8, 3
    # --- 推理态:w_1d = δ + λΔ,前半通道 λ 极小(应判「抹掉」),后半很大(应判「存活」)
    w = torch.zeros(C, 1, K)
    w[:, :, K // 2] = 1.0
    w[: C // 2] += 1e-5 * torch.randn(C // 2, 1, K)     # 远低于 3.94e-3
    w[C // 2:] += 0.5 * torch.randn(C - C // 2, 1, K)   # 远高于
    rep = {"vision_encoder.b0.token_mixer.reparam_conv_1d.weight": w,
           "vision_encoder.b0.token_mixer.reparam_conv_1d.bias": torch.randn(C),
           "vision_encoder.attn.temporal_embedding": torch.randn(1, 640, 8)}
    assert detect_mode(rep) == "reparameterized"
    dev = (w - identity_kernel_like(w)).abs().amax(dim=(1, 2))
    assert (dev[: C // 2] < INT8_ZERO_THRESHOLD).all(), "小 λ 通道应判为被抹掉"
    assert (dev[C // 2:] > INT8_ZERO_THRESHOLD).all(), "大 λ 通道应判为存活"

    # --- 训练态探测
    trn = {"vision_encoder.b0.token_mixer.layer_scale_1d": torch.full((C, 1), 8e-4)}
    assert detect_mode(trn) == "training"

    # --- gate3 中和 + 落盘往返
    with tempfile.TemporaryDirectory() as d:
        p, o = os.path.join(d, "in.pt"), os.path.join(d, "out.pt")
        torch.save({"model": dict(rep)}, p)
        cmd_gate3(argparse.Namespace(ckpt=p, out=o))
        got = load_state(o)["vision_encoder.b0.token_mixer.reparam_conv_1d.weight"]
        assert torch.equal(got, identity_kernel_like(got)), "gate3 应把融合核置为纯 δ"
        assert load_state(o)["vision_encoder.b0.token_mixer.reparam_conv_1d.bias"].abs().sum() == 0

        # --- 0 匹配必须抛错、且不落盘(这正是服务器上那次静默失败)
        p2, o2 = os.path.join(d, "empty.pt"), os.path.join(d, "never.pt")
        torch.save({"model": {"unrelated": torch.zeros(3)}}, p2)
        try:
            cmd_gate3(argparse.Namespace(ckpt=p2, out=o2))
            raise AssertionError("0 匹配时本应 SystemExit")
        except SystemExit:
            pass
        assert not os.path.exists(o2), "抛错后绝不能留下产物"

    print("self-test OK —— 形态探测 / INT8 阈值 / gate3 中和 / 0 匹配抛错且不落盘 全过")
    return 0


# ----------------------------------------------------------------- CLI


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="架构第四轮零 GPU 闸门(round4 §3)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("inspect", "先跑这个:探测 ckpt 形态与键分布"),
                        ("gate5", "INT8 会不会抹掉融合核里的时序项"),
                        ("gate3", "删时序分支(融合核 := δ)"),
                        ("gate4", "置零 temporal_embedding")):
        q = sub.add_parser(name, help=help_)
        q.add_argument("--ckpt", required=True)
        if name in ("gate3", "gate4"):
            q.add_argument("--out", required=True)
    q = sub.add_parser("gate1", help="空间参数是否与图像 ckpt 逐位相同(骨干是否被冻)")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--image-ckpt", required=True)
    sub.add_parser("self-test", help="合成权重自检,零依赖")

    a = p.parse_args(argv)
    return {"inspect": cmd_inspect, "gate5": cmd_gate5, "gate3": cmd_gate3,
            "gate4": cmd_gate4, "gate1": cmd_gate1, "self-test": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
