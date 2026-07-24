"""C2 · T 无关化(ROADMAP 2.1)的验收单测。

    python -m pytest tests/test_t_agnostic.py -v

覆盖(README §2b 矩阵「C2 T 无关化」行的机械判据):
1. **T=8 逐位一致**:MHSA / RepMixer(训练路径 + reparam 路径)/ AttentionBlock 的
   新动态 T 前向,与「旧代码硬编码 8」的参考实现(下方 _legacy_* 逐行照抄改动前
   代码)在同一权重、同一输入上 torch.equal;
2. **任意 T 可跑**:缩小版 FastViT 在 T∈{2,3,4,8,16} 下前向形状正确、数值有限
   (mean 与 attn 两种时序池化都过);
3. temporal_embedding 插值:t==8 原样返回(同一对象)、端点保持、单调、形状;
4. utils/elastic_t.py:子采样下标约定、跨 rank 决定性、非法参数报错;
5. experiments/per_t_bn.py:snapshot/load 逐位往返、recalibrate 真的把统计拉向
   数据分布、drift report 对分布漂移非零。
"""
import copy
from functools import partial

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
nn = torch.nn

from models.mobileviclip_small import (  # noqa: E402
    MHSA,
    RepMixer,
    AttentionBlock,
    FastViT,
    RepCPE,
    set_num_frames,
    get_num_frames,
    resize_temporal_embedding,
)
from models import mobileviclip_tiny as tiny_mod  # noqa: E402
from utils.elastic_t import choose_step_t, subsample_frame_indices  # noqa: E402
from experiments.per_t_bn import (  # noqa: E402
    bn_drift_report,
    load_bn_stats,
    recalibrate_bn,
    reset_bn_stats,
    snapshot_bn_stats,
)


@pytest.fixture(autouse=True)
def _reset_frame_ctx():
    """每个测试前后都把帧数上下文复位到默认 8(旧行为)。"""
    set_num_frames(8)
    yield
    set_num_frames(8)


def _randomize_bn_stats(module, seed=0):
    """把 BN running stats 随机化,让 eval 路径的 BN 变换非平凡。"""
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.running_mean.data = torch.randn(m.running_mean.shape, generator=g) * 0.1
            m.running_var.data = torch.rand(m.running_var.shape, generator=g) * 0.5 + 0.75


# ---------------------------------------------------------------------------
# 旧代码参考实现(逐行照抄改动前的硬编码-8 前向;权重取自被测模块)
# ---------------------------------------------------------------------------

def _legacy_mhsa_forward(m, x):
    x = x.reshape(8, int(x.shape[0] / 8), x.shape[1], x.shape[2], x.shape[3])
    shape = x.shape
    T, B, C, H, W = shape
    x = x.permute(1, 2, 0, 3, 4)  # B,C,T,W,H
    N = T * H * W
    if len(shape) == 5:
        x = torch.flatten(x, start_dim=2).transpose(-2, -1)  # (B, N, C)
    qkv = (
        m.qkv(x)
        .reshape(B, N, 3, m.num_heads, m.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    attn = (q * m.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = m.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = m.proj(x)
    x = m.proj_drop(x)
    if len(shape) == 5:
        x = x.transpose(-2, -1).reshape(B, C, T, H, W)
    x = x.permute(2, 0, 1, 3, 4).reshape(T * B, C, H, W)
    return x


def _legacy_repmixer_forward(blk, x):
    x = x.reshape(8, int(x.shape[0] / 8), x.shape[1], x.shape[2], x.shape[3])
    t, b, c, h, w = x.shape
    x = x.permute(1, 3, 4, 2, 0).reshape(b * h * w, c, t)
    x = x + blk.layer_scale_1d * (blk.mixer_1d(x) - blk.norm_1d(x))
    x = x.reshape(b, h, w, c, t).permute(4, 0, 3, 1, 2).reshape(t * b, c, h, w)
    x = x + blk.layer_scale * (blk.mixer(x) - blk.norm(x))
    return x


def _legacy_repmixer_reparam_forward(blk, x):
    x = x.reshape(8, int(x.shape[0] / 8), x.shape[1], x.shape[2], x.shape[3])
    t, b, c, h, w = x.shape
    x = x.permute(1, 3, 4, 2, 0).reshape(b * h * w, c, t)
    x = blk.reparam_conv_1d(x)
    x = x.reshape(b, h, w, c, t).permute(4, 0, 3, 1, 2).reshape(t * b, c, h, w)
    x = blk.reparam_conv(x)
    return x


def _legacy_attnblock_forward(blk, x):
    x = x.reshape(8, int(x.shape[0] / 8), x.shape[1], x.shape[2], x.shape[3])
    t, b, c, h, w = x.shape
    x = x.permute(1, 3, 4, 2, 0).reshape(b * h * w, c, t)
    x = x + blk.temporal_embedding
    x = x.reshape(b, h, w, c, t).permute(4, 0, 3, 1, 2).reshape(t * b, c, h, w)
    x = x + blk.drop_path(
        blk.layer_scale_1 * _legacy_mhsa_forward(blk.token_mixer, blk.norm(x))
    )
    x = x + blk.drop_path(blk.layer_scale_2 * blk.convffn(x))
    return x


# ---------------------------------------------------------------------------
# 1. T=8 逐位一致(新动态路径 vs 旧硬编码路径)
# ---------------------------------------------------------------------------

def test_mhsa_t8_bitwise_vs_legacy():
    torch.manual_seed(0)
    m = MHSA(dim=64, head_dim=32).eval()
    x = torch.randn(8 * 2, 64, 4, 4)  # (T*B, C, H, W),T 在外维
    set_num_frames(8)
    with torch.no_grad():
        assert torch.equal(m(x), _legacy_mhsa_forward(m, x))


def test_repmixer_t8_bitwise_vs_legacy_train_path():
    torch.manual_seed(0)
    blk = RepMixer(dim=16, kernel_size=3).eval()
    _randomize_bn_stats(blk)
    x = torch.randn(8 * 2, 16, 6, 6)
    set_num_frames(8)
    with torch.no_grad():
        assert torch.equal(blk(x), _legacy_repmixer_forward(blk, x))


def test_repmixer_t8_bitwise_vs_legacy_reparam_path():
    torch.manual_seed(0)
    blk = RepMixer(dim=16, kernel_size=3).eval()
    _randomize_bn_stats(blk)
    rep = copy.deepcopy(blk)
    rep.reparameterize()
    x = torch.randn(8 * 2, 16, 6, 6)
    set_num_frames(8)
    with torch.no_grad():
        assert torch.equal(rep(x), _legacy_repmixer_reparam_forward(rep, x))


@pytest.mark.parametrize("attn_cls,dim", [(AttentionBlock, 640),
                                          (tiny_mod.AttentionBlock, 512)])
def test_attnblock_t8_bitwise_vs_legacy(attn_cls, dim):
    torch.manual_seed(0)
    blk = attn_cls(dim=dim, mlp_ratio=1.0, with_cp=False).eval()
    _randomize_bn_stats(blk)
    with torch.no_grad():
        blk.temporal_embedding.normal_(std=0.02)  # 零嵌入会让加法测不出来
    x = torch.randn(8 * 1, dim, 2, 2)
    set_num_frames(8)
    with torch.no_grad():
        assert torch.equal(blk(x), _legacy_attnblock_forward(blk, x))


def test_tiny_mhsa_t8_bitwise_vs_legacy():
    torch.manual_seed(0)
    m = tiny_mod.MHSA(dim=64, head_dim=32).eval()
    x = torch.randn(8 * 2, 64, 4, 4)
    set_num_frames(8)
    with torch.no_grad():
        assert torch.equal(m(x), _legacy_mhsa_forward(m, x))


# ---------------------------------------------------------------------------
# 2. 任意 T:缩小版 FastViT 全塔前向
# ---------------------------------------------------------------------------

def _mini_fastvit(temporal_pooling="mean"):
    torch.manual_seed(0)
    return FastViT(
        layers=[1, 1, 1, 1],
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
        embed_dims=[16, 32, 64, 640],  # 末段 640 = AttentionBlock 内嵌 temporal_embedding 的通道数
        mlp_ratios=[1, 1, 1, 1],
        downsamples=[True, True, True, True],
        se_downsamples=[False, False, True, True],
        pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
        temporal_pooling=temporal_pooling,
        with_cp=False,
    ).eval()


@pytest.mark.parametrize("pooling", ["mean", "attn"])
def test_mini_fastvit_forward_any_t(pooling):
    fv = _mini_fastvit(pooling)
    for t in (2, 3, 4, 8, 16):
        x = torch.randn(1, 3, t, 64, 64)  # (B, C, T, H, W)
        with torch.no_grad():
            out = fv(x)
        assert out.shape == (1, 512), f"T={t} 输出形状 {out.shape}"
        assert torch.isfinite(out).all(), f"T={t} 输出含非有限值"
        assert get_num_frames() == t, "FastViT.forward 应已把上下文设为当前 T"


def test_mini_fastvit_t8_deterministic():
    fv = _mini_fastvit()
    x = torch.randn(2, 3, 8, 64, 64)
    with torch.no_grad():
        a, b = fv(x), fv(x)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 3. temporal_embedding 插值
# ---------------------------------------------------------------------------

def test_resize_temporal_embedding_identity_at_native_t():
    emb = torch.randn(1, 640, 8)
    assert resize_temporal_embedding(emb, 8) is emb, "t==T0 必须原样返回(逐位等价)"


@pytest.mark.parametrize("t", [2, 3, 4, 16])
def test_resize_temporal_embedding_shape_and_endpoints(t):
    ramp = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    out = resize_temporal_embedding(ramp, t)
    assert out.shape == (1, 1, t)
    # align_corners=True:首末帧位置保持原值
    assert torch.allclose(out[..., 0], ramp[..., 0])
    assert torch.allclose(out[..., -1], ramp[..., -1])
    # 单调输入 -> 单调输出
    assert (out[0, 0, 1:] >= out[0, 0, :-1]).all()


# ---------------------------------------------------------------------------
# 4. utils/elastic_t.py
# ---------------------------------------------------------------------------

def test_subsample_frame_indices_convention():
    assert subsample_frame_indices(16, 16) == list(range(16))
    assert subsample_frame_indices(16, 8) == [1, 3, 5, 7, 9, 11, 13, 15]
    assert subsample_frame_indices(16, 4) == [2, 6, 10, 14]
    assert subsample_frame_indices(16, 2) == [4, 12]
    for total, t in [(16, 5), (12, 7), (8, 3)]:
        idx = subsample_frame_indices(total, t)
        assert len(idx) == t
        assert all(0 <= i < total for i in idx)
        assert all(b > a for a, b in zip(idx, idx[1:])), "必须严格递增"
    with pytest.raises(ValueError):
        subsample_frame_indices(16, 0)
    with pytest.raises(ValueError):
        subsample_frame_indices(8, 16)  # choices 超过 num_frames 必须炸而不是静默


def test_choose_step_t_deterministic_across_ranks_and_varies():
    choices = [2, 4, 8, 16]
    # 同 (seed, step) -> 同 T(模拟多个 rank 各自独立调用)
    for step in range(5):
        picks = {choose_step_t(choices, seed=42, step=step) for _ in range(4)}
        assert len(picks) == 1
    # 不同 step 覆盖全部选项
    seen = {choose_step_t(choices, seed=42, step=s) for s in range(200)}
    assert seen == set(choices)


# ---------------------------------------------------------------------------
# 5. experiments/per_t_bn.py
# ---------------------------------------------------------------------------

class _ToyBNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))


def _batches(shift, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(4, 3, 8, 8, generator=g) + shift for _ in range(n)]


def test_per_t_bn_roundtrip_and_drift():
    torch.manual_seed(0)
    net = _ToyBNNet()

    reset_bn_stats(net)
    recalibrate_bn(net, _batches(shift=0.0))
    snap0 = snapshot_bn_stats(net)
    assert set(snap0) == {"bn1", "bn2"}

    reset_bn_stats(net)
    recalibrate_bn(net, _batches(shift=3.0))
    snap3 = snapshot_bn_stats(net)

    # 分布漂移必须被量出来;recalibrate 真把统计拉向了数据(bn1 输入均值 ~ 随 shift 移动)
    rep = bn_drift_report(snap3, snap0)
    assert rep["global_"]["mean_z"] > 0.5, rep["global_"]
    assert not net.training, "recalibrate 结束后应回到 eval"

    # load 往返:写回 snap0 后再快照,逐位一致;且与 snap3 明显不同
    load_bn_stats(net, snap0)
    snap0b = snapshot_bn_stats(net)
    for name in snap0:
        for k in ("running_mean", "running_var", "num_batches_tracked"):
            assert torch.equal(snap0[name][k], snap0b[name][k])
    assert bn_drift_report(snap0b, snap0)["global_"]["mean_z"] == 0.0

    with pytest.raises(KeyError):
        load_bn_stats(net, {"不存在的层": snap0["bn1"]})


def test_frame_ctx_set_get():
    set_num_frames(4)
    assert get_num_frames() == 4
    set_num_frames(8)
    assert get_num_frames() == 8
