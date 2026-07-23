"""脑启发 add-on 组件的单元测试(neural_inspired/)。

这些组件 torch-only、与基线模型解耦,所以本文件**不需要** timm / decord /
easydict,只要有 torch 即可跑:

    python -m pytest tests/test_neural_inspired.py -v

覆盖 TemporalSurprisePool 的两条关键性质:
1. 零初始化(alpha=0)时与 mean 池化逐位等价 —— 严格不退化起步;
2. 打开门控(alpha、gamma 非零)后输出偏离 mean —— 说明惊奇加权真的在参与计算。
"""
import pytest

torch = pytest.importorskip("torch")


def test_temporal_surprise_pool_zero_init_equals_mean():
    from neural_inspired import TemporalSurprisePool

    torch.manual_seed(0)
    pool = TemporalSurprisePool(dim=512).eval()
    x = torch.randn(4, 8, 512)  # [B, T, C]
    with torch.no_grad():
        out = pool(x)
    assert torch.allclose(out, x.mean(1), atol=1e-6), \
        "zero-gated surprise pool must equal mean pooling at step 0"


def test_temporal_surprise_pool_deviates_when_gated():
    from neural_inspired import TemporalSurprisePool

    torch.manual_seed(0)
    pool = TemporalSurprisePool(dim=512).eval()
    x = torch.randn(4, 8, 512)
    # 打开惊奇锐度 + 门控后,权重不再均匀,输出必须偏离 mean。
    with torch.no_grad():
        pool.gamma.fill_(2.0)
        pool.alpha.fill_(1.0)
        out = pool(x)
    assert not torch.allclose(out, x.mean(1), atol=1e-6), \
        "with alpha/gamma>0 the surprise pool must differ from mean pooling"


def test_temporal_surprise_pool_shape_and_weight_normalization():
    from neural_inspired import TemporalSurprisePool

    torch.manual_seed(0)
    pool = TemporalSurprisePool(dim=64).eval()
    x = torch.randn(3, 8, 64)
    out = pool(x)
    assert out.shape == (3, 64)  # [B, T, C] -> [B, C]
