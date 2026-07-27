"""选帧诊断脚本的统计与判读逻辑测试(不碰视频,纯 CPU)。

诊断脚本的价值全在「(A)/(B) 两个假说给出相反预测」这一点上 —— 如果统计口径
或判读阈值写错,跑出来的表会把两个机制混掉,而那种错误在真实数据上**看不出来**
(没有 ground truth 可对)。所以这里用**构造好的极端下标**把两条预测各自钉死。
"""
import importlib.util
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "frame_select_diag", os.path.join(_ROOT, "experiments", "frame_select_diag.py")
)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


# ------------------------------------------------------------ 解析基线
def test_endpoint_rate_matches_hand_calculation():
    """n=16,k=2:1 − C(14,2)/C(16,2) = 1 − 91/120 = 29/120。"""
    assert diag.analytic_endpoint_rate(16, 2) == pytest.approx(29 / 120)


def test_endpoint_rate_saturates_when_k_large():
    """k > n−2 时不可能避开两个端点。"""
    assert diag.analytic_endpoint_rate(16, 15) == 1.0
    assert diag.analytic_endpoint_rate(16, 16) == 1.0


def test_endpoint_rate_rejects_bad_k():
    for k in (0, -1, 17):
        with pytest.raises(ValueError):
            diag.analytic_endpoint_rate(16, k)


def test_mean_pair_gap_is_n_plus_1_over_3():
    """E|i−j| = (n+1)/3;n=16 ⇒ 17/3。"""
    assert diag.analytic_mean_pair_gap(16) == pytest.approx(17 / 3)


def test_mean_pair_gap_matches_brute_force():
    """小 n 上与穷举一致,防止公式记反。"""
    from itertools import combinations
    for n in (2, 3, 5, 9, 16):
        pairs = list(combinations(range(n), 2))
        brute = sum(b - a for a, b in pairs) / len(pairs)
        assert diag.analytic_mean_pair_gap(n) == pytest.approx(brute)


# ------------------------------------------------------------ 统计
def test_index_stats_on_uniform_k2():
    """uniform 在 16 帧上取 K=2 恒为 {4,12}:不碰端点,间隔 8。"""
    from utils.elastic_t import subsample_frame_indices
    idx = subsample_frame_indices(16, 2)
    assert idx == [4, 12]
    s = diag.index_stats([idx] * 10, 16)
    assert s["endpoint_rate"] == 0.0
    assert s["mean_pair_gap"] == pytest.approx(8.0)
    assert s["mean_span"] == pytest.approx(8.0)


def test_index_stats_counts_either_endpoint():
    s = diag.index_stats([[0, 5], [5, 15], [6, 7]], 16)
    assert s["endpoint_rate"] == pytest.approx(2 / 3)


def test_index_stats_rejects_empty():
    with pytest.raises(ValueError):
        diag.index_stats([], 16)


# ------------------------------------------------------------ 判读
def test_verdict_A_when_divmax_hugs_endpoints():
    """恒选 {0,15}:端点率 1.0、间隔顶满 ⇒ 假说 (A) 离群帧。"""
    stats = diag.index_stats([[0, 15]] * 100, 16)
    d = diag.discriminate(stats, 16, 2)
    assert d["verdict"] == "A"
    assert d["endpoint_ratio_vs_random"] > 4  # 1.0 / 0.2417


def test_verdict_B_when_divmax_looks_random():
    """构造一批间隔≈(n+1)/3 且端点率≈随机的下标 ⇒ 假说 (B) 像素≠时间。"""
    import random
    rng = random.Random(0)
    idx = [sorted(rng.sample(range(16), 2)) for _ in range(4000)]
    d = diag.discriminate(diag.index_stats(idx, 16), 16, 2)
    assert d["verdict"] == "B"


def test_verdict_neither_when_moderately_spread():
    """恒选 {2,13}:间隔 11,既没顶满(15)也不同于随机(5.67)⇒ 两条都不命中。"""
    d = diag.discriminate(diag.index_stats([[2, 13]] * 50, 16), 16, 2)
    assert d["verdict"] == "neither"


def test_every_verdict_has_prose():
    """判读文本必须覆盖所有分支,否则报告阶段 KeyError。"""
    assert set(diag.VERDICT_TEXT) == {"A", "B", "AB", "neither"}


# ------------------------------------------------------------ 守卫
@pytest.mark.parametrize("var", ["MVC_FRAME_SELECT", "MVC_FRAME_PERTURB"])
def test_refuses_to_run_with_hook_enabled(var):
    """钩子开着会改变吐出来的帧,诊断就变成了诊断另一件事 —— 必须拒跑。"""
    env = {**os.environ, var: "divmax" if "SELECT" in var else "shuffle"}
    env.pop("MVC_DATA_DIR", None)
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "experiments", "frame_select_diag.py"),
         "--limit", "1"],
        capture_output=True, text=True, env=env, cwd=_ROOT, timeout=180,
    )
    assert r.returncode != 0
    assert "拒绝执行" in (r.stdout + r.stderr)
    assert var in (r.stdout + r.stderr)
