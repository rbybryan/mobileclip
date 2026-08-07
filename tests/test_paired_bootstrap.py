"""配对 bootstrap 的单测(评审 U10-1 / 评审 U10-1)。

只依赖 numpy —— 本地无 torch/decord 也能全跑。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from paired_bootstrap import (  # noqa: E402
    load_ranks,
    paired_bootstrap,
    recalls_from_ranks,
    save_ranks,
    unpaired_halfwidth,
)


def _ranks(i2t, t2i):
    return {"i2t": np.asarray(i2t), "t2i": np.asarray(t2i)}


# --- 复算的召回必须与 itm_eval 一致 -------------------------------------------

def test_recalls_match_itm_eval_formula():
    # rank 0,1,2,...,9 —— R@1 命中 1 条、R@5 命中 5 条、R@10 命中 10 条
    r = _ranks(np.arange(10), np.arange(10))
    m = recalls_from_ranks(r)
    assert m["txt_r1"] == pytest.approx(10.0)
    assert m["txt_r5"] == pytest.approx(50.0)
    assert m["txt_r10"] == pytest.approx(100.0)
    assert m["img_r1"] == pytest.approx(10.0)
    # R-Sum = 六个召回之和
    assert m["R_sum"] == pytest.approx(
        m["txt_r1"] + m["txt_r5"] + m["txt_r10"] + m["img_r1"] + m["img_r5"] + m["img_r10"]
    )
    # r_mean 是两个方向 r_mean 的平均
    assert m["r_mean"] == pytest.approx((m["txt_r_mean"] + m["img_r_mean"]) / 2)


def test_rank_zero_is_a_hit_at_k1():
    """rank 用的是 0-based(`itm_eval` 的 `ranks < 1`)—— 错一位就全盘皆错。"""
    assert recalls_from_ranks(_ranks([0], [0]))["txt_r1"] == pytest.approx(100.0)
    assert recalls_from_ranks(_ranks([1], [1]))["txt_r1"] == pytest.approx(0.0)


# --- 配对性质 -----------------------------------------------------------------

def test_identical_arms_give_zero_delta_and_zero_width_ci():
    """两条臂完全相同 ⇒ 配对差恒为 0,CI 宽度也是 0(共享方差被完全抵消)。

    这正是配对的意义:无配对做同样的事会得到一个非零宽度的带。
    """
    rng = np.random.default_rng(0)
    r = _ranks(rng.integers(0, 100, 500), rng.integers(0, 100, 500))
    res = paired_bootstrap(r, {k: v.copy() for k, v in r.items()}, n_boot=200, seed=1)
    for k in ("txt_r1", "img_r1", "r_mean", "R_sum"):
        assert res[k]["delta"] == pytest.approx(0.0)
        assert res[k]["lo"] == pytest.approx(0.0)
        assert res[k]["hi"] == pytest.approx(0.0)
        assert res[k]["significant"] is False


def test_uniform_improvement_is_detected_as_significant():
    """臂在每条查询上都不差、且有一批明显变好 ⇒ 必须判显著。"""
    n = 1000
    base_i2t = np.full(n, 5)          # 全部 rank=5:R@1=0, R@10=100
    arm_i2t = base_i2t.copy()
    arm_i2t[:100] = 0                 # 其中 100 条提到 rank 0 ⇒ R@1 +10.0
    r_arm = _ranks(arm_i2t, base_i2t.copy())
    r_base = _ranks(base_i2t, base_i2t.copy())
    res = paired_bootstrap(r_arm, r_base, n_boot=1000, seed=0)
    assert res["txt_r1"]["delta"] == pytest.approx(10.0)
    assert res["txt_r1"]["significant"] is True
    assert res["txt_r1"]["lo"] > 0
    # 没动的方向不该被判显著
    assert res["img_r1"]["delta"] == pytest.approx(0.0)
    assert res["img_r1"]["significant"] is False


def test_paired_ci_is_much_tighter_than_unpaired():
    """本模块存在的理由:同一对臂,配对带必须显著窄于无配对带。"""
    rng = np.random.default_rng(7)
    n = 1000
    base_i2t = rng.integers(0, 50, n)
    base_t2i = rng.integers(0, 50, n)
    # 臂 = 基线上做一点小扰动(共享绝大部分方差)
    arm_i2t, arm_t2i = base_i2t.copy(), base_t2i.copy()
    arm_i2t[:20] = 0
    r_arm, r_base = _ranks(arm_i2t, arm_t2i), _ranks(base_i2t, base_t2i)

    res = paired_bootstrap(r_arm, r_base, n_boot=2000, seed=0)
    half_paired = (res["R_sum"]["hi"] - res["R_sum"]["lo"]) / 2
    half_unpaired = unpaired_halfwidth(r_arm, r_base, n_boot=2000, seed=0)
    assert half_paired < half_unpaired / 3, (
        f"配对({half_paired:.3f})相对无配对({half_unpaired:.3f})没有明显收窄 —— "
        "配对逻辑可能坏了(两臂没用同一组下标?)"
    )


def test_both_arms_use_the_same_resample_indices():
    """直接钉死配对实现:若两臂各抽各的,identical-arms 就不会得到 0 宽 CI。"""
    rng = np.random.default_rng(3)
    r = _ranks(rng.integers(0, 30, 300), rng.integers(0, 30, 300))
    res = paired_bootstrap(r, {k: v.copy() for k, v in r.items()}, n_boot=300, seed=5)
    assert res["R_sum"]["hi"] - res["R_sum"]["lo"] == pytest.approx(0.0)


def test_reproducible_given_seed():
    rng = np.random.default_rng(11)
    a = _ranks(rng.integers(0, 20, 200), rng.integers(0, 20, 200))
    b = _ranks(rng.integers(0, 20, 200), rng.integers(0, 20, 200))
    r1 = paired_bootstrap(a, b, n_boot=200, seed=42)
    r2 = paired_bootstrap(a, b, n_boot=200, seed=42)
    assert r1["R_sum"]["lo"] == pytest.approx(r2["R_sum"]["lo"])
    assert r1["R_sum"]["hi"] == pytest.approx(r2["R_sum"]["hi"])


def test_mismatched_query_counts_raise():
    a = _ranks(np.zeros(10), np.zeros(10))
    b = _ranks(np.zeros(11), np.zeros(10))
    with pytest.raises(ValueError, match="查询数不一致"):
        paired_bootstrap(a, b, n_boot=10)


# --- 落盘 / 读回 --------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    i2t, t2i = np.array([0, 3, 7]), np.array([1, 1, 9])
    save_ranks(str(tmp_path), "ret_test_emb/", i2t, t2i)
    got = load_ranks(str(tmp_path), "ret_test_emb/")
    assert np.array_equal(got["i2t"], i2t)
    assert np.array_equal(got["t2i"], t2i)


def test_dsl_and_raw_go_to_separate_files(tmp_path):
    """两个口径不能互相覆盖 —— 否则 raw/DSL 会串味。"""
    save_ranks(str(tmp_path), "ret_test/", np.array([0]), np.array([0]))
    save_ranks(str(tmp_path), "ret_test_emb/", np.array([5]), np.array([5]))
    assert load_ranks(str(tmp_path), "ret_test/")["i2t"][0] == 0
    assert load_ranks(str(tmp_path), "ret_test_emb/")["i2t"][0] == 5


def test_missing_ranks_error_tells_you_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="重跑一次评测"):
        load_ranks(str(tmp_path), "ret_test_emb/")


# --- 多选口径(SSv2-MC)---------------------------------------------------------
# 与检索口径同构:同一批题目 ⇒ 必须配对。这里钉死的不变式和上面那组一一对应,
# 因为踩过的坑是一样的(裸差值 + 无配对带 = 判不出来)。

from paired_bootstrap import (  # noqa: E402
    load_mc_correct,
    paired_bootstrap_mc,
    save_mc_correct,
    unpaired_halfwidth_mc,
)


def test_mc_identical_arms_give_zero_delta_and_zero_width_ci():
    c = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    r = paired_bootstrap_mc(c, c, n_boot=200, seed=0)["acc"]
    assert r["delta"] == pytest.approx(0.0)
    assert r["lo"] == pytest.approx(0.0)
    assert r["hi"] == pytest.approx(0.0)
    assert r["significant"] is False   # CI 含 0(退化成一个点)⇒ 不显著


def test_mc_uniform_improvement_is_detected_as_significant():
    base = np.zeros(400, dtype=np.int8)
    arm = base.copy()
    arm[:80] = 1                        # 臂多答对 20%
    r = paired_bootstrap_mc(arm, base, n_boot=500, seed=0)["acc"]
    assert r["delta"] == pytest.approx(20.0)
    assert r["significant"] is True
    assert r["lo"] > 0


def test_mc_paired_ci_is_tighter_than_unpaired():
    rng = np.random.default_rng(7)
    base = (rng.random(600) < 0.46).astype(np.int8)   # ≈ 上游 46.07 的量级
    arm = base.copy()
    flip = rng.choice(600, 30, replace=False)          # 只动一小部分题
    arm[flip] = 1 - arm[flip]
    paired = paired_bootstrap_mc(arm, base, n_boot=500, seed=0)["acc"]
    half_paired = (paired["hi"] - paired["lo"]) / 2
    half_unpaired = unpaired_halfwidth_mc(arm, base, n_boot=500, seed=0)
    assert half_paired < half_unpaired      # 配对必须更窄,否则配对逻辑写错了


def test_mc_reproducible_given_seed():
    rng = np.random.default_rng(1)
    base = (rng.random(200) < 0.5).astype(np.int8)
    arm = (rng.random(200) < 0.55).astype(np.int8)
    a = paired_bootstrap_mc(arm, base, n_boot=300, seed=42)["acc"]
    b = paired_bootstrap_mc(arm, base, n_boot=300, seed=42)["acc"]
    assert a == b


def test_mc_mismatched_question_counts_raise():
    with pytest.raises(ValueError, match="题目数不一致"):
        paired_bootstrap_mc(np.ones(10), np.ones(11))


def test_mc_save_and_load_roundtrip(tmp_path):
    c = np.array([1, 0, 0, 1, 1], dtype=np.int8)
    save_mc_correct(str(tmp_path), c)
    back = load_mc_correct(str(tmp_path))
    assert np.array_equal(back, c.astype(np.float64))


def test_mc_missing_file_error_tells_you_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="retrieval_mc.py"):
        load_mc_correct(str(tmp_path))
