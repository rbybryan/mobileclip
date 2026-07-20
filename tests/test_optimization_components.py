"""优化组件的数值不变性 / 正确性单元测试。

在装好训练环境的机器上运行(需要 torch;attention-pool 测试还需要 timm):

    python -m pytest tests/test_optimization_components.py -v

覆盖三件事:
1. sigmoid 损失在 idx=None 时与旧的 torch.eye 公式逐位一致(改动不破坏默认行为);
2. sigmoid 损失在 idx 有重复(共享字幕)时把重复对当正例;
3. TemporalAttentionPool 零初始化时与 mean 池化逐位等价(严格不退化起步)。
"""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _legacy_sigmoid_loss(vision_proj, text_proj, t, b):
    """改动前 vtc_loss_sigmoid 的公式(torch.eye 目标),作为对照。"""
    v = F.normalize(vision_proj, dim=-1)
    x = F.normalize(text_proj, dim=-1)
    logits = v @ x.T * t + b
    eye = torch.eye(v.shape[0])
    mask = -torch.ones_like(logits) + 2 * eye
    return torch.mean(-torch.sum(F.logsigmoid(mask * logits), dim=-1))


def _make_loss():
    from models.criterions import VTC_VTM_Loss
    return VTC_VTM_Loss(False)


def test_sigmoid_loss_matches_legacy_when_idx_none():
    torch.manual_seed(0)
    loss_fn = _make_loss()
    v = torch.randn(16, 64)
    x = torch.randn(16, 64)
    t, b = torch.tensor(10.0), torch.tensor(-10.0)
    new = loss_fn.vtc_loss_sigmoid(v, x, idx=None, temp=1.0, t=t, b=b, all_gather=False)
    old = _legacy_sigmoid_loss(v, x, t, b)
    assert torch.allclose(new, old, atol=1e-6), f"{new} != {old}"


def test_sigmoid_loss_multi_positive_with_duplicate_idx():
    torch.manual_seed(0)
    loss_fn = _make_loss()
    v = torch.randn(8, 32)
    x = torch.randn(8, 32)
    t, b = torch.tensor(10.0), torch.tensor(-10.0)
    idx_unique = torch.arange(8)
    idx_dup = idx_unique.clone()
    idx_dup[1] = idx_dup[0]  # 样本 0/1 共享字幕
    l_unique = loss_fn.vtc_loss_sigmoid(v, x, idx=idx_unique, temp=1.0, t=t, b=b, all_gather=False)
    l_dup = loss_fn.vtc_loss_sigmoid(v, x, idx=idx_dup, temp=1.0, t=t, b=b, all_gather=False)
    # idx 全唯一时目标退化为对角 -> 与 idx=None 一致
    l_none = loss_fn.vtc_loss_sigmoid(v, x, idx=None, temp=1.0, t=t, b=b, all_gather=False)
    assert torch.allclose(l_unique, l_none, atol=1e-6)
    # 有共享字幕时目标不同 -> 损失必须不同(重复对从负例翻成正例)
    assert not torch.allclose(l_dup, l_unique, atol=1e-6)


def test_caption_to_idx_stable_shared_and_bounded():
    pytest.importorskip("decord")  # dataset 模块链会 import decord
    from dataset.pt_dataset import caption_to_idx

    a = caption_to_idx("a dog runs in the park")
    assert a == caption_to_idx("a dog runs in the park"), "同字幕必须同 idx(跨进程稳定)"
    assert a != caption_to_idx("a cat sits on the mat")
    assert 0 <= a < 2 ** 63, "必须能装进 int64(torch.eq 跨 rank 比较)"


def test_temporal_attention_pool_zero_init_equals_mean():
    pytest.importorskip("timm")
    from models.mobileviclip_small import TemporalAttentionPool

    torch.manual_seed(0)
    pool = TemporalAttentionPool(dim=512, num_heads=8).eval()
    x = torch.randn(4, 8, 512)  # [B, T, C]
    with torch.no_grad():
        out = pool(x)
    assert torch.allclose(out, x.mean(1), atol=1e-6), \
        "zero-init attention pool must equal mean pooling at step 0"

    # 扰动输出投影后必须偏离 mean(说明模块真的在参与计算)
    with torch.no_grad():
        pool.proj.weight.add_(0.01)
        out2 = pool(x)
    assert not torch.allclose(out2, x.mean(1), atol=1e-6)
