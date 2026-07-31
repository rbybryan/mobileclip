"""架构第四轮零 GPU 闸门(`notes/architecture-round4.md` §3 的 ①③④⑤)。

**为什么需要这个文件,而不是几段 heredoc。** 2026-07-29 首次在服务器上跑那几段 heredoc,
三条全废,而且**其中一条是静默失败**:

* ③ `layer_scale_1d = 0` —— 匹配到 **0 个键**,脚本照样把 ckpt 存了出去 ⇒ `_ablate.pt`
  与原文件逐位相同,评测会"通过"但什么都没消融;
* ⑤ 读 `layer_scale` —— `torch.cat(): expected a non-empty list`;
* ① 比对空间 `layer_scale` —— 输出 `0 / 0`(一个都没匹配上,却打印得像结论)。

**根因(2026-07-29 实测更正)**:发布的 `mobileviclip_small.pt` 是**训练态**权重
(`inspect` 实测:`layer_scale_1d` 40 个、`layer_scale` 80 个、`temporal_embedding` 4 个、
`reparam_conv*` 0 个)。原 heredoc 挂掉**不是**因为重参数化 —— 是因为
**顶层包装键不是 `model`**:那几段只写了 `sd.get("model", sd)`,取不到就退回**顶层字典**,
于是遍历的是 `epoch/optimizer/...` 这类键,`endswith("layer_scale_1d")` 自然 0 命中。
本文件的 `load_state()` 依次尝试 `model / module / state_dict` 并过滤非 Tensor,才拿到 2497 个张量。

> ⚠️ 我先前把根因判成「ckpt 是推理态」,**那是错的**。两种形态都要支持,所以
> `detect_mode()` 保留 —— 但请记住实测结论:**发布权重是训练态**。

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

✅ **λ 已实测复现(2026-07-29)**:`layer_scale_1d` n=9920,**median 7.973e-04,p95 2.762e-03**
—— 与 round4 §2 P2 引用的「8e-4 / 2.8e-3」**完全吻合**,该数据来源确认可靠(先前标它
「不可能来自发布权重」是我判断失误,已撤回)。

⚠️ **但 λ 本身仍不是 INT8 判据**:融合核是 `δ + λ·Δ`,要比的是 **λ·|Δ|**。
用 `gate5-fused` 走一遍真正的 `reparameterize()` 再量 `|w_1d − δ|` 才算数。

--------------------------------------------------------------------------------
用法
--------------------------------------------------------------------------------
    python experiments/gate_checks.py inspect  --ckpt checkpoints/mobileviclip_small.pt
    python experiments/gate_checks.py gate5    --ckpt checkpoints/mobileviclip_small.pt
    python experiments/gate_checks.py gate3    --ckpt checkpoints/mobileviclip_small.pt --out checkpoints/_ablate_temporal.pt
    python experiments/gate_checks.py gate4    --ckpt checkpoints/mobileviclip_small.pt --out checkpoints/_ablate_tempemb.pt
    python experiments/gate_checks.py gate1    --ckpt checkpoints/mobileviclip_small.pt --image-ckpt checkpoints/mobileclip_s2.pt
    python experiments/gate_checks.py gate5-fused --ckpt checkpoints/mobileviclip_small.pt \\
        --config scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py
    python experiments/gate_checks.py self-test          # 零依赖,合成权重,验证全部逻辑

跑完 gate3/gate4 后按 `COMMANDS.md` §3 的零样本评测跑 MSR-VTT。基线锚点(勿混口径):
**raw = `ret_test_emb/` = 论文主口径** → t2v R@1 40.8 / r_mean 61.55;
**DSL = `ret_test/`** → t2v R@1 42.50 / r_mean 62.28。
判据:③ Δr_mean > 1.0(raw 与 DSL 同向)才保留该支路;④ Δ > 0.5 才保留 embedding。
"""

from __future__ import annotations

import argparse
import math
import re
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
        t = torch.cat([sd[k].flatten().abs().float() for k in ks])
        # 分母有两种取法,**只有 token_mixer 那一种是可比的**:
        #   · token_mixer.layer_scale  —— 缩放**空间 mixer** 支路,与 layer_scale_1d
        #     缩放时序 mixer 支路一一对应。这是 GATE-0 腿③ / round2 静态审计用的分母
        #     (`experiments/temporal_capacity_audit.py:178`),得到 ≈6.1%。
        #   · 块级 layer_scale     —— 缩放的是 **ConvFFN 残差**,另一个模块,数量级也不同。
        # 2026-07-29 首版把两者混在一起(80 个全池化)⇒ 算出 2.42%,并据此误判
        # 「文档里的 6.1% 复现不出来」。**是这个脚本错了,不是文档错了。**
        sp_mixer = [k for k in sd
                    if k.endswith("layer_scale") and "token_mixer" in k]
        sp_all = [k for k in sd
                  if k.endswith("layer_scale") and not k.endswith("layer_scale_1d")]
        print(f"layer_scale_1d (时序 mixer)  blocks={len(ks):3d}  n={t.numel():6d}  "
              f"median={t.median():.3e}  p95={t.quantile(.95):.3e}")
        if sp_mixer:
            s_ = torch.cat([sd[k].flatten().abs().float() for k in sp_mixer])
            print(f"layer_scale    (空间 mixer)  blocks={len(sp_mixer):3d}  n={s_.numel():6d}  "
                  f"median={s_.median():.3e}  p95={s_.quantile(.95):.3e}   ← 可比分母")
            print(f"\n  **时序/空间 mixer gate 比值(中位)= {t.median()/s_.median():.4f}**"
                  f"   ← GATE-0 腿③ 的那个数(文档记的 ≈6.1%)")
        if sp_all:
            s_all = torch.cat([sd[k].flatten().abs().float() for k in sp_all])
            print(f"  (参考,**勿用于腿③**:对全部 {len(sp_all)} 个 layer_scale 池化后"
                  f" median={s_all.median():.3e},比值 {t.median()/s_all.median():.4f} —— "
                  f"混进了块级 ConvFFN 门,与时序 mixer 门不可比)")
        print(f"\n  λ 中位 {t.median():.2e} vs INT8 抹零阈值 {INT8_ZERO_THRESHOLD:.2e}")
        print("  ⚠️ **但 λ 本身不是判据**:融合核是 δ + λ·Δ,真正要比的是 **λ·|Δ|**"
              "(Δ = BN 折叠后的 mixer_1d − norm_1d)。")
        print("  ⇒ 判 INT8 请跑:  python experiments/gate_checks.py gate5-fused "
              f"--ckpt {args.ckpt} --config <eval config.py>")
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


def cmd_gate5_fused(args) -> int:
    """真 INT8 判据:build 模型 → 载权重 → `reparameterize()` → 量 |w_1d − δ|。

    训练态 ckpt 只给得出 λ,而融合核是 `δ + λ·Δ`。判 INT8 要的是 **λ·|Δ|**,
    也就是重参数化之后的实际偏离量。与其手工复现 BN 折叠(容易错),不如直接走仓库
    自己的 `reparameterize()`,这也正是部署路径。
    """
    import importlib.util

    from models.backbones.internvideo2.mobileclip.modules.common.mobileone import (
        reparameterize_model,
    )

    spec = importlib.util.spec_from_file_location("evalcfg", args.config)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)

    from utils.config import Config
    cfg = Config.from_file(args.config) if hasattr(Config, "from_file") else cfg_mod

    from models.mobileviclip_small import MobileViCLIP_Small
    model = MobileViCLIP_Small(config=cfg, tokenizer=None, is_pretrain=False)

    sd = load_state(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # `text_encoder.*` **本来就不在** video ckpt 里 —— 构造函数已从 mobileclip_s2.pt 单独载入
    # 文本塔(tasks_clip/retrieval.py 也照样报这 148 个 missing 然后正常跑完)。本闸只看视觉塔,
    # 所以只对**非 text_encoder** 的缺失 fail-loud;否则会像 2026-07-29 那次一样自己把自己挡住。
    vis_missing = [k for k in missing if not k.startswith("text_encoder.")]
    print(f"load_state_dict: missing={len(missing)}"
          f"(其中 text_encoder={len(missing) - len(vis_missing)},属预期)"
          f" unexpected={len(unexpected)}")
    if len(vis_missing) > 20:
        raise SystemExit(
            f"[FAIL] 视觉塔权重没对上(non-text missing={len(vis_missing)});"
            f"前 5: {vis_missing[:5]}")

    model.eval()
    model = reparameterize_model(model)

    devs = []
    for name, m in model.named_modules():
        w = getattr(getattr(m, "reparam_conv_1d", None), "weight", None)
        if w is None:
            continue
        w = w.detach().float()
        devs.append((w - identity_kernel_like(w)).abs().amax(dim=(1, 2)))
    require(devs, "reparameterize() 之后的 reparam_conv_1d", args.ckpt)

    d = torch.cat(devs)
    erased = (d < INT8_ZERO_THRESHOLD).float().mean().item()
    print(f"\n融合时序核 |w_1d − δ| 逐通道最大值  (channels={d.numel()})")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        print(f"    p{int(q*100):02d} = {d.quantile(q):.3e}")
    print(f"\n  INT8 抹零阈值 {INT8_ZERO_THRESHOLD:.3e};低于它的通道占比 **{100*erased:.1f}%**")
    print("  ⇒ " + (
        "**坐实** round4 §2 P2:多数通道的时序混合项在 PTQ INT8 下被整个抹掉"
        if erased > 0.5 else
        "**未坐实**:时序项在 INT8 下存活 ⇒ round4 §2 P2 的机制不成立,须改写"))
    return 0


def cmd_gate5_spatial(args) -> int:
    """**A5 / §10.4**:空间 mixer 那一半的 INT8 判据 —— §8 明确标注「未验证」的那条。

    `gate5-fused` 只量过**时序**核。配套那句「空间 mixer 只跑在 1–3 LSB / ~2 有效 bit」
    **至今没人量过**;而时序那次的解析估计错了 1–2 个数量级(假设 |Δ|=O(0.1–1),实际 ≈7.7)
    ⇒ **默认它也是错的,直到量出来**。

    🔴 **别再用固定阈值。** `gate5-fused` 拿 |w−δ| 去比一个固定的 3.937e-3,
    **隐含假设每通道的 INT8 scale 分母恒为 1** —— 那正是它把结论说反的原因
    (§10.0:换成 LSB 单位后中位数是 0.78 LSB,不是「存活」)。这里一律按**每通道** scale 算。

    每通道:s_c = max|w_c|/127 · ŵ_c = s_c·round(w_c/s_c)
        ① a_c = max|w_c|                          真正的 per-channel scale 分母
        ② n_c = 127·max|w_c − δ| / a_c            该支路占多少 LSB(尺度无关)
        ③ ρ_c = ‖ŵ_c − w_c‖₂ / ‖w_c − δ‖₂        **头条**:量化误差相对该支路本身的大小
    """
    from models.backbones.internvideo2.mobileclip.modules.common.mobileone import (
        reparameterize_model,
    )
    from utils.config import Config
    from models.mobileviclip_small import MobileViCLIP_Small

    cfg = Config.from_file(args.config)
    model = MobileViCLIP_Small(config=cfg, tokenizer=None, is_pretrain=False)
    sd = load_state(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    vis_missing = [k for k in missing if not k.startswith("text_encoder.")]
    print(f"load_state_dict: missing={len(missing)} "
          f"(text_encoder={len(missing) - len(vis_missing)},属预期) unexpected={len(unexpected)}")
    if len(vis_missing) > 20:
        raise SystemExit(f"[FAIL] 视觉塔权重没对上;前 5: {vis_missing[:5]}")

    model.eval()
    model = reparameterize_model(model)

    # **按模块选,不按名字**:只取同时有 reparam_conv 与 reparam_conv_1d 的模块,
    # 这样能排掉 stem / downsample / ConvFFN 里同样暴露 reparam_conv 的 MobileOneBlock。
    mods = [m for _, m in model.named_modules()
            if hasattr(m, "reparam_conv") and hasattr(m, "reparam_conv_1d")]
    tot_c = sum(m.reparam_conv.weight.shape[0] for m in mods)
    # 两条断言 —— 当年就是少了它们才出的 40-vs-80 错(§0.1 的 gate5 分母事故)
    assert len(mods) == 40, f"期望 40 个 RepMixer,实际 {len(mods)}"
    assert tot_c == 9920, f"期望 Σ out_channels = 9920,实际 {tot_c}"
    print(f"[OK] 选中 {len(mods)} 个 RepMixer,Σ out_channels = {tot_c}")

    ns, rhos = [], []
    for m in mods:
        w = m.reparam_conv.weight.detach().float()      # (C,1,3,3)
        # ⚠️ 空间核是 4 维 ⇒ 归约必须 amax(dim=(1,2,3));写成 (1,2) 会静默算错
        a = w.abs().amax(dim=(1, 2, 3)).clamp_min(1e-12)
        delta = torch.zeros_like(w)
        delta[:, :, 1, 1] = 1.0                          # 恒等核的 1 在 [1,1],不是一维中心下标
        dev = (w - delta)
        ns.append(127.0 * dev.abs().amax(dim=(1, 2, 3)) / a)
        s = (a / 127.0).view(-1, 1, 1, 1)
        wq = s * torch.round(w / s)
        num = (wq - w).flatten(1).norm(dim=1)
        den = dev.flatten(1).norm(dim=1).clamp_min(1e-12)
        rhos.append(num / den)

    n = torch.cat(ns); rho = torch.cat(rhos)
    print(f"\n空间 mixer,逐通道(channels={n.numel()})")
    print("    分位   n(LSB)   b_eff(bit)   rho")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        nq = n.quantile(q).item()
        print(f"    p{int(q*100):02d}   {nq:8.3f}   {math.log2(max(nq,1e-9)):9.2f}   {rho.quantile(q).item():.4f}")

    med = rho.quantile(0.5).item()
    print(f"\n  **头条:rho 中位数 = {med:.4f}**")
    if med < 0.10:
        print("  ⇒ 空间量化很干净 ⇒ **埋掉 AC9 的重参数化专属立论**。发一段机制说明即可:")
        print("     RepMixer 的恒等抽头是**未经 BN 的精确 δ**,与 RepVGG 不同,所以它对量化是良性的。")
    elif med < 0.35:
        print("  ⇒ AC9 缩成**一行部署规则**:40 个 dw3×3 留 fp16(= 0.59% MACs)。**不做 QAT。**")
    else:
        print("  ⇒ AC9 拿到**新立论**(亚 2-bit 支路编码);修法仍是 fp16-on-depthwise。")
    print("\n  (面板预登记的预测是 n≈12 LSB / b_eff≈4.7 bit / rho≈0.02–0.05 ——")
    print("   即「空间 mixer 只有 1–3 LSB / ~2 bit」是**保守方向上的错**。对照上表看它中没中。)")
    return 0



# ----------------------------------------------------------------- gates 3 / 4


def _save(sd_full, out: str) -> None:
    torch.save(sd_full, out)
    print(f"\n[OK] 已写出 {out}")
    print("     按 COMMANDS.md §3 跑 MSR-VTT 零样本评测。基线:raw(ret_test_emb/,主口径)"
          " t2v 40.8 / r_mean 61.55;DSL(ret_test/) t2v 42.50 / r_mean 62.28。")


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


def block_prefixes(sd: Dict[str, torch.Tensor]) -> List[str]:
    """40 个 RepMixerBlock 的键前缀,按 (阶段, 块) 自然序。

    锚点用 `.token_mixer.layer_scale_1d` —— 每个 RepMixerBlock 恰好一个,
    且 stage 4 的 AttentionBlock 没有,所以这一步顺带把 40/44 的歧义去掉。
    """
    suf = ".token_mixer.layer_scale_1d"
    pre = [k[: -len(suf)] for k in sd if k.endswith(suf)]
    # 自然序:按键里出现的整数排,否则 block 10 会排在 block 2 前面
    return sorted(pre, key=lambda s: [int(t) for t in re.findall(r"\d+", s)])


def cmd_gate6_dropblock(args) -> int:
    """把第 i 个 RepMixerBlock 变成**精确恒等**(P11 逐块深度敏感度扫描)。

    依据(源码核实):
      RepMixer.forward:743-746  x = x + λ_1d·(mixer_1d−norm_1d); x = x + λ·(mixer−norm)
      RepMixerBlock.forward:1075 x = token_mixer(x); x = x + drop_path(layer_scale·convffn(x))
    ⇒ 三个门全置零 ⇒ 该块 f(x) = x。**少置一个就不是恒等。**

    🔴 全模型 `layer_scale` 共 80 个 = 40 个 token_mixer(空间 mixer)+ 40 个块级
    (ConvFFN 门)。混池正是 §0.1 里 gate5 分母取错的同一个坑 —— 所以这里**断言恰好命中 3 个**。
    """
    full = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = full
    for k in ("model", "module", "state_dict"):
        if isinstance(full, dict) and k in full and isinstance(full[k], dict):
            sd = full[k]
            break
    tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    mode = detect_mode(tensors)
    if mode != "training":
        raise SystemExit(
            f"[gate6] 需要 training-态权重(有 layer_scale),实测形态={mode}。\n"
            "        重参数化后的权重里门已被折进核,逐块恒等要改的东西不一样,"
            "本子命令拒绝在该形态下写出任何文件。"
        )

    pre = block_prefixes(tensors)
    if len(pre) != 40:
        raise SystemExit(f"[gate6] 期望 40 个 RepMixerBlock,实际 {len(pre)} —— 先跑 `inspect` 看键名。")
    if not (0 <= args.block < len(pre)):
        raise SystemExit(f"[gate6] --block 必须在 0..{len(pre)-1},收到 {args.block}")

    p = pre[args.block]
    want = [f"{p}.token_mixer.layer_scale_1d",   # 时序 mixer 残差门
            f"{p}.token_mixer.layer_scale",      # 空间 mixer 残差门
            f"{p}.layer_scale"]                  # 块级 ConvFFN 门
    miss = [k for k in want if k not in sd]
    if miss:
        raise SystemExit(f"[gate6] 块 {args.block} ({p}) 缺门:{miss} —— 不写出任何文件。")

    for k in want:
        sd[k] = torch.zeros_like(sd[k])
    assert len(want) == 3, "每块必须恰好 3 个门"
    print(f"[gate6] 块 {args.block}/40 前缀={p}")
    print(f"        已置零 3 个门(通道数 {[tuple(sd[k].shape) for k in want]}) ⇒ 该块现在是恒等")
    _save(full, args.out)
    return 0


def cmd_gate1(args) -> int:
    """空间参数是否与图像 ckpt 逐位相同 ⇒ 骨干是否被冻结。

    发布权重是推理态,`layer_scale` 已不存在,所以改比对**融合空间核**
    `reparam_conv.weight`;若图像 ckpt 是训练态则无法直接比,脚本会如实说明而不是打印 0/0。
    """
    a, b = load_state(args.ckpt), load_state(args.image_ckpt)
    print(f"video ckpt 形态: {detect_mode(a)}   image ckpt 形态: {detect_mode(b)}")

    def norm_key(k: str) -> str:
        """把两边的塔前缀都剥掉再比。

        视频 ckpt 用 `vision_encoder.`;Apple 的 MobileCLIP ckpt 用
        `image_encoder.model.`(见 `models/mobileviclip_small.py:341-342` 的载入路径)。
        首跑就是因为只剥了前者,80 个张量全部「对不上名字」。
        """
        for p in ("vision_encoder.", "image_encoder.model.", "image_encoder.", "module."):
            if k.startswith(p):
                k = k[len(p):]
        return k

    def suffix_index(sd):
        idx = {}
        for k in sd:
            idx.setdefault(norm_key(k), []).append(k)
        return idx

    bi = suffix_index(b)
    cand_tags = ("reparam_conv.weight", "layer_scale")
    total = same = diff = nomatch = 0
    for k, v in a.items():
        if not any(k.endswith(t) for t in cand_tags) or "_1d" in k:
            continue
        total += 1
        hits = bi.get(norm_key(k), [])
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
    q = sub.add_parser("gate5-fused", help="真 INT8 判据:重参数化后量 |w_1d − δ|")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--config", required=True,
                   help="任一 eval config,例如 scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py")
    q = sub.add_parser("gate5-spatial",
                       help="A5:空间 mixer 的 INT8 判据(per-channel scale + rho),决定 AC9 生死")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--config", required=True,
                   help="任一 eval config,例如 scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py")
    q = sub.add_parser("gate1", help="空间参数是否与图像 ckpt 逐位相同(骨干是否被冻)")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--image-ckpt", required=True)
    q = sub.add_parser("gate6-dropblock", help="把第 i 个 RepMixerBlock 变成恒等(P11 逐块扫描)")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--block", required=True, type=int, help="0..39,按 (阶段,块) 自然序")
    q.add_argument("--out", required=True)
    sub.add_parser("self-test", help="合成权重自检,零依赖")

    a = p.parse_args(argv)
    return {"inspect": cmd_inspect, "gate5": cmd_gate5, "gate3": cmd_gate3,
            "gate4": cmd_gate4, "gate1": cmd_gate1, "gate5-fused": cmd_gate5_fused,
            "gate6-dropblock": cmd_gate6_dropblock,
            "gate5-spatial": cmd_gate5_spatial,
            "self-test": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
