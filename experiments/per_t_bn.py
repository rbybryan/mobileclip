"""per-T BatchNorm 诊断与校准工具(计划评审 U3;C2 · T 无关化 验收件)。

背景(评审 U3):时序 Conv1d 分支遍布所有 RepMixer
块,FastViT 主干与时序分支都是 BatchNorm;随机 T 训练会把 BN running stats 变成
「跨 T 的混合物」,推理固定某个 T 时统计失配(FFN, CVPR'23 的 temporal frequency
deviation)。修法 = 训练照旧,**训练后对每个 T 只前向刷新一遍 BN buffer**,评测/
导出按 T 取对应的一套(slimmable networks 的 switchable-BN 先例)。

本文件只提供纯工具函数(单测:tests/test_t_agnostic.py);与数据管线的接线
随 R3 试点落地——用法(伪代码,3 行):

    snap8 = snapshot_bn_stats(model)                       # 训练结束的混合统计
    for T in [2, 4, 8, 16]:
        reset_bn_stats(model)
        recalibrate_bn(model, batches_at_T(T))             # 只前向,~几百 batch
        bank[T] = snapshot_bn_stats(model)
    save_bn_bank("bn_bank.pth", bank)                      # MB 级
    # 评测/导出前:load_bn_stats(model, bank[T]);再 reparameterize 得 per-T 折叠权重

诊断(README §2b 矩阵「per-T BN 漂移诊断」行):对同一 ckpt 分别在纯 T=2 与
纯 T=16 的 batch 上 recalibrate 出两套统计,`bn_drift_report` 量化偏移——
mean_z(均值漂移 / 参考标准差)显著大于 0 即坐实风险。
"""
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import nn

_BN = nn.modules.batchnorm._BatchNorm  # BatchNorm1d/2d/3d 的共同基类


def bn_modules(model: nn.Module):
    """产出 (name, module):模型里所有带 running stats 的 BatchNorm。"""
    for name, m in model.named_modules():
        if isinstance(m, _BN) and m.track_running_stats:
            yield name, m


def snapshot_bn_stats(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """克隆全部 BN 的 running_mean/var/num_batches_tracked(存 CPU)。"""
    snap = {}
    for name, m in bn_modules(model):
        snap[name] = dict(
            running_mean=m.running_mean.detach().cpu().clone(),
            running_var=m.running_var.detach().cpu().clone(),
            num_batches_tracked=m.num_batches_tracked.detach().cpu().clone(),
        )
    return snap


def load_bn_stats(model: nn.Module, snap: Dict[str, Dict[str, torch.Tensor]],
                  strict: bool = True) -> None:
    """把 snapshot 写回模型(按模块名对齐)。"""
    names = {name for name, _ in bn_modules(model)}
    missing = set(snap) - names
    if strict and missing:
        raise KeyError(f"snapshot 里有模型中不存在的 BN:{sorted(missing)[:5]} ...")
    for name, m in bn_modules(model):
        if name not in snap:
            if strict:
                raise KeyError(f"snapshot 缺少 BN:{name}")
            continue
        s = snap[name]
        m.running_mean.data.copy_(s["running_mean"].to(m.running_mean.device))
        m.running_var.data.copy_(s["running_var"].to(m.running_var.device))
        m.num_batches_tracked.data.copy_(
            s["num_batches_tracked"].to(m.num_batches_tracked.device)
        )


def reset_bn_stats(model: nn.Module) -> None:
    """清零 running stats 并把 momentum 置 None(累积均值,标准校准配方)。"""
    for _, m in bn_modules(model):
        m.reset_running_stats()
        m.momentum = None


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    batches: Iterable,
    forward_fn: Optional[Callable[[nn.Module, object], object]] = None,
) -> int:
    """只前向刷新 BN 统计:全模型 eval,仅 BN 置 train;返回用掉的 batch 数。

    ``forward_fn(model, batch)`` 缺省为 ``model(batch)``——真实模型请传入
    只跑视觉塔的闭包(不需要文本塔/损失)。结束后整模型回到 eval。
    """
    model.eval()
    for _, m in bn_modules(model):
        m.train()
    n = 0
    for batch in batches:
        if forward_fn is not None:
            forward_fn(model, batch)
        else:
            model(batch)
        n += 1
    model.eval()
    return n


def bn_drift_report(
    snap_a: Dict[str, Dict[str, torch.Tensor]],
    snap_b: Dict[str, Dict[str, torch.Tensor]],
    eps: float = 1e-5,
    topk: int = 10,
) -> Dict:
    """量化两套 BN 统计的漂移(a 相对参考 b)。

    - ``mean_z``:|μa−μb| / sqrt(σb²+eps) 的逐通道均值(以 b 的标准差为单位);
    - ``logvar_shift``:|log(σa²+eps) − log(σb²+eps)| 的逐通道均值。
    返回 {global: {...}, layers: {name: {...}}, top: [(name, mean_z), ...]}。
    """
    layers = {}
    for name in snap_a:
        if name not in snap_b:
            continue
        ma, va = snap_a[name]["running_mean"], snap_a[name]["running_var"]
        mb, vb = snap_b[name]["running_mean"], snap_b[name]["running_var"]
        mean_z = ((ma - mb).abs() / (vb + eps).sqrt()).mean().item()
        logvar_shift = ((va + eps).log() - (vb + eps).log()).abs().mean().item()
        layers[name] = dict(mean_z=mean_z, logvar_shift=logvar_shift)
    if not layers:
        return dict(global_=dict(mean_z=0.0, logvar_shift=0.0), layers={}, top=[])
    g = dict(
        mean_z=sum(v["mean_z"] for v in layers.values()) / len(layers),
        logvar_shift=sum(v["logvar_shift"] for v in layers.values()) / len(layers),
    )
    top = sorted(((n, v["mean_z"]) for n, v in layers.items()),
                 key=lambda x: -x[1])[:topk]
    return dict(global_=g, layers=layers, top=top)


def save_bn_bank(path: str, bank: Dict[int, Dict]) -> None:
    """保存 {T: snapshot} 的 per-T BN 库(MB 级)。"""
    torch.save(bank, path)


def load_bn_bank(path: str) -> Dict[int, Dict]:
    return torch.load(path, map_location="cpu")
