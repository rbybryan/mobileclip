"""选帧基线的单测(纯 CPU,不需要数据/权重)。

跑法:python -m pytest tests/test_frame_select.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from utils.elastic_t import subsample_frame_indices  # noqa: E402
from utils.frame_select import (  # noqa: E402
    greedy_kcenter,
    pixel_features,
    random_indices,
    select_indices,
)


# --------------------------------------------------------------------------
# uniform:必须与 C2 训练时的切片口径逐位一致(这就是要对照的那条基线)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_uniform_matches_elastic_t(k):
    assert select_indices("uniform", total=16, k=k) == subsample_frame_indices(16, k)


def test_uniform_fixed_table():
    assert select_indices("uniform", total=16, k=2) == [4, 12]
    assert select_indices("uniform", total=16, k=4) == [2, 6, 10, 14]
    assert select_indices("uniform", total=16, k=8) == [1, 3, 5, 7, 9, 11, 13, 15]


# --------------------------------------------------------------------------
# 零操作契约:K >= T 返回 None,调用方据此走"原样返回同一对象"
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["uniform", "divmax", "random"])
@pytest.mark.parametrize("k", [16, 17])
def test_k_ge_total_returns_none(mode, k):
    frames = torch.randn(16, 3, 32, 32)
    assert select_indices(mode, total=16, k=k, frames=frames) is None


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="未知的选帧模式"):
        select_indices("smart", total=16, k=2)


def test_divmax_without_frames_raises():
    with pytest.raises(ValueError, match="需要 frames"):
        select_indices("divmax", total=16, k=2, frames=None)


# --------------------------------------------------------------------------
# greedy_kcenter:构造两个远离的簇,必须每簇各取一帧
# --------------------------------------------------------------------------
def test_kcenter_picks_one_per_cluster():
    a = torch.zeros(3, 8)
    b = torch.ones(3, 8) * 100.0
    feats = torch.cat([a, b], dim=0)  # 下标 0,1,2 一簇;3,4,5 一簇
    sel = greedy_kcenter(feats, 2)
    assert len(sel) == 2
    assert (sel[0] < 3) != (sel[1] < 3), f"两帧落在同一簇:{sel}"


def test_kcenter_seed_is_most_representative():
    # 三个点共线,中间那个离均值最近 ⇒ K=1 必须选它,而不是端点
    feats = torch.tensor([[-10.0], [0.0], [10.0]])
    assert greedy_kcenter(feats, 1) == [1]


def test_kcenter_deterministic_and_wellformed():
    feats = torch.randn(16, 64)
    first = greedy_kcenter(feats, 4)
    assert first == greedy_kcenter(feats, 4), "同输入必须同输出"
    assert first == sorted(first), "下标必须升序"
    assert len(set(first)) == 4, "下标必须互不重复"
    assert all(0 <= i < 16 for i in first)


def test_kcenter_full_k_is_all_frames():
    feats = torch.randn(8, 16)
    assert greedy_kcenter(feats, 8) == list(range(8))


def test_kcenter_rejects_bad_k():
    feats = torch.randn(8, 16)
    with pytest.raises(ValueError):
        greedy_kcenter(feats, 0)
    with pytest.raises(ValueError):
        greedy_kcenter(feats, 9)


# --------------------------------------------------------------------------
# pixel_features:减均值后每维求和为 0(否则距离会被全局亮度主导)
# --------------------------------------------------------------------------
def test_pixel_features_is_mean_centered():
    frames = torch.randn(16, 3, 64, 64)
    feats = pixel_features(frames, grid=8)
    assert feats.shape == (16, 3 * 8 * 8)
    assert torch.allclose(feats.sum(0), torch.zeros(feats.shape[1]), atol=1e-4)


def test_divmax_prefers_the_odd_frame_out():
    # 15 帧几乎相同 + 1 帧完全不同 ⇒ K=2 必须包含那个异类帧
    frames = torch.zeros(16, 3, 32, 32)
    frames += torch.randn(16, 3, 32, 32) * 1e-3
    frames[11] = 50.0
    sel = select_indices("divmax", total=16, k=2, frames=frames)
    assert 11 in sel, f"没选中唯一的异类帧:{sel}"


# --------------------------------------------------------------------------
# random:地板臂,必须按 (seed,index) 决定性,且随 index 变化
# --------------------------------------------------------------------------
def test_random_deterministic_per_index():
    assert random_indices(16, 4, seed=0, index=7) == random_indices(16, 4, seed=0, index=7)


def test_random_varies_across_samples():
    picks = {tuple(random_indices(16, 4, seed=0, index=i)) for i in range(50)}
    assert len(picks) > 1, "所有样本选到同一组帧,种子没起作用"


def test_random_wellformed():
    sel = random_indices(16, 4, seed=0, index=3)
    assert sel == sorted(sel) and len(set(sel)) == 4
    assert all(0 <= i < 16 for i in sel)


# --------------------------------------------------------------------------
# 数据集钩子:未设环境变量时必须是纯 no-op(返回同一个对象)
# --------------------------------------------------------------------------
def _hook():
    # dataset/av_utils.py 硬 import torchaudio(见 TROUBLESHOOTING),
    # 缺它的机器上跳过这一组;V100/B100 的完整环境里会真跑。
    pytest.importorskip("torchaudio")
    pytest.importorskip("decord")
    from dataset.base_dataset import maybe_select_frames

    return maybe_select_frames


def test_hook_is_identity_when_unset(monkeypatch):
    monkeypatch.delenv("MVC_FRAME_SELECT", raising=False)
    frames = torch.randn(16, 3, 32, 32)
    assert _hook()(frames, 0) is frames, "未设开关时必须返回同一个对象"


def test_hook_is_identity_when_none(monkeypatch):
    monkeypatch.setenv("MVC_FRAME_SELECT", "none")
    frames = torch.randn(16, 3, 32, 32)
    assert _hook()(frames, 0) is frames


def test_hook_requires_k(monkeypatch):
    monkeypatch.setenv("MVC_FRAME_SELECT", "uniform")
    monkeypatch.delenv("MVC_FRAME_SELECT_K", raising=False)
    with pytest.raises(ValueError, match="MVC_FRAME_SELECT_K"):
        _hook()(torch.randn(16, 3, 32, 32), 0)


@pytest.mark.parametrize("mode", ["uniform", "divmax", "random"])
@pytest.mark.parametrize("k", [2, 4])
def test_hook_shapes(monkeypatch, mode, k):
    monkeypatch.setenv("MVC_FRAME_SELECT", mode)
    monkeypatch.setenv("MVC_FRAME_SELECT_K", str(k))
    out = _hook()(torch.randn(16, 3, 32, 32), 0)
    assert out.shape == (k, 3, 32, 32)


def test_hook_uniform_equals_direct_slice(monkeypatch):
    monkeypatch.setenv("MVC_FRAME_SELECT", "uniform")
    monkeypatch.setenv("MVC_FRAME_SELECT_K", "4")
    frames = torch.randn(16, 3, 32, 32)
    assert torch.equal(_hook()(frames, 0), frames[[2, 6, 10, 14]])
