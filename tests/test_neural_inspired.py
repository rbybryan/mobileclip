"""脑启发 add-on 组件的单元测试(neural_inspired/)。

这些组件 torch-only、与基线模型解耦,所以本文件**不需要** timm / decord /
easydict,只要有 torch 即可跑:

    python -m pytest tests/test_neural_inspired.py -v

覆盖 TemporalSurprisePool 的关键性质:
1. 零初始化(alpha=0)时与 mean 池化逐位等价 —— 严格不退化起步;
2. 打开门控(alpha、gamma 非零)后输出偏离 mean —— 说明惊奇加权真的在参与计算;
3. **init 处梯度必须能流到 alpha**(回归测试)—— gamma 若也从 0 起,softmax 均匀、
   pooled≡mean,alpha/gamma/key 的梯度全为 0,模块永远走不出 mean 池化(V100 试点
   曾据此把一个「死」模块当活的跑了 10 epoch,见 experiments/pilot_10k_results.md)。
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


def test_temporal_surprise_pool_gradient_flows_at_init():
    """回归测试:init 处 output==mean(不退化)且 alpha 能收到非零梯度(门能打开)。

    这两条必须同时成立。gamma 从 0 起会让二者只满足前者:pooled≡mean ⇒ alpha 梯度
    恒为 0,模块卡在 mean 池化的零梯度不动点上(V100 试点的死模块正是如此)。
    """
    from neural_inspired import TemporalSurprisePool

    torch.manual_seed(0)
    pool = TemporalSurprisePool(dim=64)  # 训练态,保留梯度
    x = torch.randn(4, 8, 64)
    out = pool(x)
    # 不退化:init 处严格等于 mean 池化(alpha=0 保证,与 gamma 取值无关)。
    assert torch.allclose(out, x.mean(1), atol=1e-6), \
        "no-regression: output must equal mean pooling at init"
    # 门能打开:alpha 在 init 处必须有非零梯度,否则永远走不出 mean 池化。
    (out.pow(2).mean()).backward()
    assert pool.alpha.grad is not None and pool.alpha.grad.abs().item() > 0, \
        "alpha must receive a nonzero gradient at init (gamma!=0 makes pooled!=mean)"


def test_temporal_surprise_pool_shape_and_weight_normalization():
    from neural_inspired import TemporalSurprisePool

    torch.manual_seed(0)
    pool = TemporalSurprisePool(dim=64).eval()
    x = torch.randn(3, 8, 64)
    out = pool(x)
    assert out.shape == (3, 64)  # [B, T, C] -> [B, C]
