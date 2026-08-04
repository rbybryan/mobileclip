"""P1 结构性旁路(`RepMixer.skip_temporal`)的等价性单测(round-5 §7 第 1 步)。

三条主张,三个测试:
  1. 训练态:λ_1d ≡ 0 时,旁路前向与原前向**逐位一致**(reshape/permute 往返不改值,
     时序残差恒为 0)。
  2. 重参数化态:λ_1d ≡ 0 折叠出的 reparam_conv_1d 是精确 δ 核 + 零偏置 ⇒ 恒等;
     旁路(直接跳过它连同 layout 链)同样**逐位一致**。
  3. 反向护栏:λ_1d **非零**时旁路必须改变输出 —— 防止上面两条因为某处静默 no-op
     而空洞地通过(house rule:等价性测试必须带一个"能测出不等价"的对照)。

off-path(不设 skip_temporal)走的是原代码路径,未加任何算子 —— 逐位不变由
test_offpath_bitwise 显式覆盖。
"""
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.mobileviclip_small import FastViT, RepCPE, RepMixer, set_num_frames
from models.backbones.internvideo2.mobileclip.modules.common.mobileone import (
    reparameterize_model,
)


def _mini_fastvit():
    """照抄 tests/test_t_agnostic.py:_mini_fastvit(3 个 RepMixer + 1 个 attention 段)。"""
    torch.manual_seed(0)
    return FastViT(
        layers=[1, 1, 1, 1],
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
        embed_dims=[16, 32, 64, 640],
        mlp_ratios=[1, 1, 1, 1],
        downsamples=[True, True, True, True],
        se_downsamples=[False, False, True, True],
        pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
        temporal_pooling="mean",
        with_cp=False,
    ).eval()


def _zero_gates(model) -> int:
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "layer_scale_1d" in name:
                p.zero_()
                n += 1
    return n


def _set_skip(model, flag: bool = True) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, RepMixer):
            m.skip_temporal = flag
            n += 1
    return n


def test_train_form_bitwise_when_gates_zero():
    fv = _mini_fastvit()
    assert _zero_gates(fv) == 3
    set_num_frames(4)
    x = torch.randn(2, 3, 4, 64, 64)
    with torch.no_grad():
        ref = fv(x)
    assert _set_skip(fv) == 3
    with torch.no_grad():
        out = fv(x)
    assert torch.equal(ref, out), "训练态:λ_1d=0 下旁路必须逐位等于原前向"


def test_reparam_form_bitwise_when_gates_zero():
    fv = _mini_fastvit()
    _zero_gates(fv)
    rep = reparameterize_model(fv).eval()          # deepcopy + 各模块 reparameterize
    set_num_frames(4)
    x = torch.randn(2, 3, 4, 64, 64)
    with torch.no_grad():
        ref = rep(x)
    assert _set_skip(rep) == 3
    with torch.no_grad():
        out = rep(x)
    assert torch.equal(ref, out), (
        "重参数化态:λ_1d=0 折叠出的 reparam_conv_1d 应为精确恒等,旁路必须逐位一致")


def test_skip_with_nonzero_gates_changes_output():
    """护栏:门非零时旁路**必须**改变输出,否则上面两条只是在测一个静默 no-op。"""
    fv = _mini_fastvit()                            # λ_1d = 1e-5 初始化,非零
    set_num_frames(4)
    x = torch.randn(2, 3, 4, 64, 64)
    with torch.no_grad():
        ref = fv(x)
    _set_skip(fv)
    with torch.no_grad():
        out = fv(x)
    assert not torch.equal(ref, out), "λ_1d≠0 时旁路应当改变输出;相等说明旁路没生效"


def test_offpath_bitwise():
    """不动 flag(以及显式设 False)都必须与从未碰过该属性的前向逐位一致。"""
    fv = _mini_fastvit()
    set_num_frames(4)
    x = torch.randn(2, 3, 4, 64, 64)
    with torch.no_grad():
        a = fv(x)
    _set_skip(fv, flag=False)
    with torch.no_grad():
        b = fv(x)
    assert torch.equal(a, b)
