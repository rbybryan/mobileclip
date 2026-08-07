"""逐查询**配对** bootstrap —— 评审 U10-1 / 评审 U10-1。

为什么需要它(2026-07-29 的 1M 层首个结果逼出来的):
`experiments/README.md` §6.1 的 **±1.6 带是「无配对」带** —— 它回答的是「一个绝对
R@1 的抽样误差有多大」。但 A/B 两条臂评的是**同一** 1000 条查询,是**配对**比较:
两臂在同一条查询上同时对/同时错的部分是**共享方差**,做差时会抵消掉。

实测把这件事变成了阻塞项:`sub1m_r0_s42` 显示**数据翻 10 倍只买到 +0.82 r_mean**,
落在无配对 ±1.6 带以内 ⇒ 用无配对带去判各臂(效应只会更小)必然什么都判不出来。
配对 bootstrap 是现成的、唯一能把分辨率压到 <1.0 的手段。

用法:
    # 1) 评测时会自动在 <out_dir>/ranks_<口径>.npz 存逐查询 rank(见
    #    tasks_clip/retrieval_utils.py:evaluation_wrapper 的 dump_ranks_dir)
    # 2) 比较两条臂:
    python experiments/collect_results.py --compare <臂目录> <基线目录>

只依赖 numpy —— 没有 torch/decord 也能跑,便于本地开发与单测。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: 与 `itm_eval` 完全一致的六个召回项;R-Sum 即它们之和。
RECALL_SPECS: Tuple[Tuple[str, str, int], ...] = (
    ("txt_r1", "i2t", 1),
    ("txt_r5", "i2t", 5),
    ("txt_r10", "i2t", 10),
    ("img_r1", "t2i", 1),
    ("img_r5", "t2i", 5),
    ("img_r10", "t2i", 10),
)

RANKS_FILENAME_TMPL = "ranks_{tag}.npz"


def _tag_from_prefix(prefix: str) -> str:
    """`ret_test/` -> `ret_test`;`ret_test_emb/` -> `ret_test_emb`(可做文件名)。"""
    return prefix.strip("/").replace("/", "_") or "eval"


def save_ranks(out_dir: str, prefix: str, ranks_i2t: np.ndarray, ranks_t2i: np.ndarray) -> str:
    """把逐查询 rank 存成 npz(每条臂 ~8KB,可以无脑一直开着)。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, RANKS_FILENAME_TMPL.format(tag=_tag_from_prefix(prefix)))
    np.savez_compressed(
        path,
        i2t=np.asarray(ranks_i2t, dtype=np.int32),
        t2i=np.asarray(ranks_t2i, dtype=np.int32),
    )
    return path


def load_ranks(run_dir: str, prefix: str = "ret_test_emb/") -> Dict[str, np.ndarray]:
    """读回某条臂的逐查询 rank。默认读 **raw 主口径**(`ret_test_emb/`)。"""
    path = os.path.join(run_dir, RANKS_FILENAME_TMPL.format(tag=_tag_from_prefix(prefix)))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到逐查询 rank:{path}\n"
            "  —— 该臂是在 rank dump 落地(2026-07-29)之前评的。重跑一次评测即可\n"
            "     (只是前向,MSR-VTT 1k 约 1 分钟),命令见 COMMANDS.md §3.8。"
        )
    with np.load(path) as z:
        return {"i2t": z["i2t"].astype(np.int64), "t2i": z["t2i"].astype(np.int64)}


def recalls_from_ranks(ranks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """由逐查询 rank 复算六个召回 + r_mean + R-Sum,数值与 `itm_eval` 一致。"""
    out: Dict[str, float] = {}
    for name, direction, k in RECALL_SPECS:
        r = ranks[direction]
        out[name] = 100.0 * float(np.count_nonzero(r < k)) / len(r)
    out["txt_r_mean"] = (out["txt_r1"] + out["txt_r5"] + out["txt_r10"]) / 3
    out["img_r_mean"] = (out["img_r1"] + out["img_r5"] + out["img_r10"]) / 3
    out["r_mean"] = (out["txt_r_mean"] + out["img_r_mean"]) / 2
    out["R_sum"] = sum(out[n] for n, _, _ in RECALL_SPECS)
    return out


def _metric_matrix(ranks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把每条查询变成 0/1 命中向量,bootstrap 时只需对它取均值 —— 比重排快得多。"""
    hits: Dict[str, np.ndarray] = {}
    for name, direction, k in RECALL_SPECS:
        hits[name] = (ranks[direction] < k).astype(np.float64)
    return hits


def _aggregate(hits: Dict[str, np.ndarray],
               idx_i2t: np.ndarray,
               idx_t2i: np.ndarray) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    for name, direction, _ in RECALL_SPECS:
        idx = idx_i2t if direction == "i2t" else idx_t2i
        vals[name] = 100.0 * float(hits[name][idx].mean())
    vals["r_mean"] = (
        (vals["txt_r1"] + vals["txt_r5"] + vals["txt_r10"]) / 3
        + (vals["img_r1"] + vals["img_r5"] + vals["img_r10"]) / 3
    ) / 2
    vals["R_sum"] = sum(vals[n] for n, _, _ in RECALL_SPECS)
    return vals


def paired_bootstrap(
    arm: Dict[str, np.ndarray],
    baseline: Dict[str, np.ndarray],
    n_boot: int = 10000,
    seed: int = 0,
    ci: float = 95.0,
    metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """臂 − 基线 的**配对** bootstrap。

    关键在**同一次重采样里,两条臂用同一组查询下标** —— 共享方差因此被抵消,
    这正是配对带远窄于无配对 ±1.6 带的原因。i2t 与 t2i 是两个不同的查询总体,
    各自独立抽样,但同样在两臂间保持一致。

    返回 ``{指标: {delta, lo, hi, significant, arm, baseline}}``。
    ``significant`` = 95% CI 不含 0。
    """
    for d in ("i2t", "t2i"):
        if len(arm[d]) != len(baseline[d]):
            raise ValueError(
                f"两条臂的 {d} 查询数不一致({len(arm[d])} vs {len(baseline[d])})"
                " —— 配对比较要求同一个测试集,请检查是不是评了不同的 benchmark。"
            )
    if n_boot < 1:
        raise ValueError("n_boot 必须 >= 1")

    hits_a, hits_b = _metric_matrix(arm), _metric_matrix(baseline)
    n_i2t, n_t2i = len(arm["i2t"]), len(arm["t2i"])
    rng = np.random.default_rng(seed)

    keys = list(metrics) if metrics else [n for n, _, _ in RECALL_SPECS] + ["r_mean", "R_sum"]
    draws: Dict[str, np.ndarray] = {k: np.empty(n_boot) for k in keys}

    for b in range(n_boot):
        idx_i2t = rng.integers(0, n_i2t, n_i2t)
        idx_t2i = rng.integers(0, n_t2i, n_t2i)
        va = _aggregate(hits_a, idx_i2t, idx_t2i)
        vb = _aggregate(hits_b, idx_i2t, idx_t2i)   # ← 同一组下标,这就是"配对"
        for k in keys:
            draws[k][b] = va[k] - vb[k]

    lo_q, hi_q = (100.0 - ci) / 2, 100.0 - (100.0 - ci) / 2
    point_a, point_b = recalls_from_ranks(arm), recalls_from_ranks(baseline)

    res: Dict[str, Dict[str, float]] = {}
    for k in keys:
        lo, hi = np.percentile(draws[k], [lo_q, hi_q])
        res[k] = {
            "arm": point_a[k],
            "baseline": point_b[k],
            "delta": point_a[k] - point_b[k],
            "lo": float(lo),
            "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0),
        }
    return res


def unpaired_halfwidth(arm: Dict[str, np.ndarray],
                       baseline: Dict[str, np.ndarray],
                       n_boot: int = 10000,
                       seed: int = 0,
                       ci: float = 95.0,
                       metric: str = "R_sum") -> float:
    """同一对臂的**无配对** CI 半宽,仅用于展示配对买到了多少分辨率。

    两臂各抽各的下标 ⇒ 共享方差不抵消。把它和配对半宽并排打出来,就能直观说明
    「为什么不能拿 ±1.6 这种无配对带去判 A/B」。
    """
    hits_a, hits_b = _metric_matrix(arm), _metric_matrix(baseline)
    n_i2t, n_t2i = len(arm["i2t"]), len(arm["t2i"])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        va = _aggregate(hits_a, rng.integers(0, n_i2t, n_i2t), rng.integers(0, n_t2i, n_t2i))
        vb = _aggregate(hits_b, rng.integers(0, n_i2t, n_i2t), rng.integers(0, n_t2i, n_t2i))
        draws[b] = va[metric] - vb[metric]
    lo, hi = np.percentile(draws, [(100.0 - ci) / 2, 100.0 - (100.0 - ci) / 2])
    return float(hi - lo) / 2


# ---------------------------------------------------------------------------
# 多选题口径(SSv2-MC 等)
#
# 为什么要单独一套:多选评测(`tasks_clip/retrieval_mc.py`)只产出一个标量 acc,
# 没有 i2t/t2i 两个查询总体,也没有 rank。但**统计问题是同一个** —— base 与臂评
# 的是同一批题目,是配对比较;不做配对就只能拿两个裸 acc 相减,重蹈 ±1.6 无配对
# 带的覆辙(见本文件顶部的模块说明)。
#
# 逐题 0/1 正确性由 `retrieval_mc.py` 在算完 acc 后落盘(~2KB/臂),与检索口径的
# `ranks_*.npz` 同理:存它几乎不要钱,缺了它就得重跑整个评测。
# ---------------------------------------------------------------------------

MC_FILENAME = "mc_correct.npz"


def save_mc_correct(out_dir: str, correct: Sequence[int]) -> str:
    """把逐题 0/1 正确性存成 npz。`retrieval_mc.py` 在 rank0 上调用一次。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, MC_FILENAME)
    np.savez_compressed(path, correct=np.asarray(correct, dtype=np.int8))
    return path


def load_mc_correct(run_dir: str) -> np.ndarray:
    """读回某条臂的逐题正确性;缺文件时给出可执行的补救指引而不是裸 KeyError。"""
    path = os.path.join(run_dir, MC_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到逐题正确性:{path}\n"
            "  —— 该臂是在 MC rank dump 落地(2026-07-30)之前评的,或者用的是\n"
            "     `retrieval.py` 而不是 `retrieval_mc.py`(多选任务必须用后者)。\n"
            "     重跑一次评测即可,命令见 COMMANDS.md §3.12。"
        )
    with np.load(path) as z:
        return z["correct"].astype(np.float64)


def paired_bootstrap_mc(
    arm: np.ndarray,
    baseline: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
    ci: float = 95.0,
) -> Dict[str, Dict[str, float]]:
    """多选准确率的**配对** bootstrap,返回结构与 `paired_bootstrap` 同构(键为 `acc`)。

    同一次重采样里两条臂用**同一组题目下标** —— 两臂同时答对/同时答错的题在做差时
    抵消掉,这正是配对带远窄于无配对带的原因。
    """
    arm = np.asarray(arm, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if arm.shape != baseline.shape:
        raise ValueError(
            f"两条臂的题目数不一致({len(arm)} vs {len(baseline)})"
            " —— 配对比较要求同一个测试集,请检查是不是评了不同的 benchmark。"
        )
    if arm.ndim != 1:
        raise ValueError(f"逐题正确性应为一维 0/1 向量,收到 shape={arm.shape}")
    if n_boot < 1:
        raise ValueError("n_boot 必须 >= 1")

    n = len(arm)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)          # ← 同一组下标,这就是"配对"
        draws[b] = 100.0 * (arm[idx].mean() - baseline[idx].mean())

    lo_q, hi_q = (100.0 - ci) / 2, 100.0 - (100.0 - ci) / 2
    lo, hi = np.percentile(draws, [lo_q, hi_q])
    point_a, point_b = 100.0 * arm.mean(), 100.0 * baseline.mean()
    return {
        "acc": {
            "arm": float(point_a),
            "baseline": float(point_b),
            "delta": float(point_a - point_b),
            "lo": float(lo),
            "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0),
        }
    }


def unpaired_halfwidth_mc(arm: np.ndarray,
                          baseline: np.ndarray,
                          n_boot: int = 10000,
                          seed: int = 0,
                          ci: float = 95.0) -> float:
    """同一对臂的**无配对** acc CI 半宽,仅用于展示配对买到了多少分辨率。"""
    arm = np.asarray(arm, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    n = len(arm)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = 100.0 * (arm[rng.integers(0, n, n)].mean()
                            - baseline[rng.integers(0, n, n)].mean())
    lo, hi = np.percentile(draws, [(100.0 - ci) / 2, 100.0 - (100.0 - ci) / 2])
    return float(hi - lo) / 2
