#!/usr/bin/env python3
"""选帧诊断:divmax 到底选中了哪些帧?(零 GPU,零训练)

## 为什么要跑

1.8 实测(`experiments/frame_select_results.md`)在 K=2 上得到两条硬结论:
  - **选帧确实要紧**:uniform 相对 random +3.40/+3.90(两口径同向);
  - **但 divmax 是错的判据**:相对 uniform −2.80/−2.40,且**与纯随机无可测差别**。

"divmax 为什么失败"有两个预注册假说(§4.1),它们对**选帧下标**给出**相反**预测:

| 假说 | 端点命中率 | 平均配对间隔 \\|i−j\\| |
|---|---|---|
| (A) k-center 偏好离群帧 | **远高于**随机基线 | 接近最大(≈ n−1) |
| (B) 像素距离 ≠ 时间距离 | 接近随机基线 | 接近随机基线 ≈ (n+1)/3 |

本脚本只做一件事:把 `greedy_kcenter` 在真实测试帧上的**下标**dump 出来,
用上表两行把 (A)/(B) 分开。**不建模型、不过编码器、不需要 GPU。**

结论若是 (A)(我的预期):divmax 在**时间上比 uniform 更分散**却更差
⇒ 可写的结论是"**要有代表性的帧,不要极端帧**",而不是"多样性无用"。
这也与 GATE-0 相容 —— `freeze`(零分散)崩 13 分说明**需要一些分散**,
本实验说明**最大化分散是过头的**。

## 逐位复现评测所见的帧

评测变换是 `Resize((256,256), bilinear)` + `x.float()/255`
(`dataset/__init__.py:get_test_transform`,**Normalize 那行是注释掉的**),
本脚本原样重建它并复用同一个 `VidTxtRetEvalDataset` ⇒ 打分用的像素与 1.8 那轮逐位相同。

用法:

    unset MVC_DATA_DIR                      # 双根规则:评测侧不 export
    python experiments/frame_select_diag.py                  # 全量 1000 条
    python experiments/frame_select_diag.py --limit 50       # 冒烟
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from itertools import combinations

# 双根规则:configs/data.py 在**导入时**读 MVC_DATA_DIR,所以必须在 import 前清掉。
# 评测数据在仓库 data/ 下;若带着训练用的 MVC_DATA_DIR 进来会解析到不存在的路径。
os.environ.pop("MVC_DATA_DIR", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- 纯统计部分
# (下面这些函数不碰视频/torch,单测直接调)

def analytic_endpoint_rate(n: int, k: int) -> float:
    """随机取 k 个不同下标时,集合含端点(0 或 n−1)的概率。

    = 1 − C(n−2, k) / C(n, k)   (两个端点都不选中的补集)
    """
    if not 0 < k <= n:
        raise ValueError(f"需要 0 < k <= n,得到 k={k} n={n}")
    if k > n - 2:
        return 1.0
    return 1.0 - math.comb(n - 2, k) / math.comb(n, k)


def analytic_mean_pair_gap(n: int) -> float:
    """随机取的一对不同下标,E|i−j| = (n+1)/3。

    对 k>2 同样成立:k-子集里任取一对,其边缘分布就是「两个不同下标均匀取」。
    """
    if n < 2:
        raise ValueError(f"需要 n >= 2,得到 {n}")
    return (n + 1) / 3.0


def index_stats(index_lists, n: int) -> dict:
    """一组选帧下标的汇总统计。index_lists: List[List[int]]。"""
    if not index_lists:
        raise ValueError("index_lists 为空")
    endpoint_hits, gaps, spans = 0, [], []
    for idx in index_lists:
        s = set(idx)
        if 0 in s or (n - 1) in s:
            endpoint_hits += 1
        pair_gaps = [abs(a - b) for a, b in combinations(sorted(idx), 2)]
        if pair_gaps:
            gaps.append(sum(pair_gaps) / len(pair_gaps))
        spans.append(max(idx) - min(idx))
    return {
        "n_videos": len(index_lists),
        "endpoint_rate": endpoint_hits / len(index_lists),
        "mean_pair_gap": (sum(gaps) / len(gaps)) if gaps else 0.0,
        "mean_span": sum(spans) / len(spans),
        "max_span": max(spans),
    }


def discriminate(divmax: dict, n: int, k: int, tol: float = 0.15) -> dict:
    """用 (A)/(B) 的相反预测判读 divmax 的下标统计。

    tol 是「接近随机基线」的相对容差。两个假说不互斥 —— 本函数报的是
    **哪一条预测被数据支持**,不是非此即彼的裁决。
    """
    rand_ep = analytic_endpoint_rate(n, k)
    rand_gap = analytic_mean_pair_gap(n)
    max_gap = float(n - 1)

    ep_ratio = divmax["endpoint_rate"] / rand_ep if rand_ep > 0 else float("inf")
    gap_close_to_random = abs(divmax["mean_pair_gap"] - rand_gap) <= tol * rand_gap
    gap_close_to_max = divmax["mean_pair_gap"] >= (1 - tol) * max_gap

    # (A) 的签名:端点命中率显著高于随机 且 间隔顶到最大
    supports_a = ep_ratio >= 1.5 and gap_close_to_max
    # (B) 的签名:间隔与随机基线无异
    supports_b = gap_close_to_random

    if supports_a and not supports_b:
        verdict = "A"
    elif supports_b and not supports_a:
        verdict = "B"
    elif supports_a and supports_b:
        verdict = "AB"      # 数学上不该同时成立(除非 n 很小),留作异常标记
    else:
        verdict = "neither"
    return {
        "random_endpoint_rate": rand_ep,
        "random_mean_pair_gap": rand_gap,
        "max_possible_gap": max_gap,
        "endpoint_ratio_vs_random": ep_ratio,
        "verdict": verdict,
    }


VERDICT_TEXT = {
    "A": (
        "**假说 (A) 成立:k-center 偏好离群帧。**\n"
        "divmax 在时间上比 uniform **更**分散,却在检索上更差 ⇒ 可写的结论是\n"
        "**要有代表性的帧,不要极端帧**,而不是「多样性无用」。与 GATE-0 相容:\n"
        "`freeze`(零分散)崩 13 分说明需要一些分散,本轮说明最大化分散是过头的。"
    ),
    "B": (
        "**假说 (B) 成立:像素距离 ≠ 时间距离。**\n"
        "divmax 的时间分布与随机无异 ⇒ 降采样 RGB 上的远近被镜头运动/曝光主导,\n"
        "根本没在挑「内容不同」的帧。结论是**廉价像素代理不可用**,\n"
        "而不是「多样性判据错」—— 要判后者需换语义特征重跑(那只能当上界报)。"
    ),
    "AB": (
        "⚠️ 两条预测同时命中,这在 n 较大时不该发生 —— 先核对 n/k 与统计口径。"
    ),
    "neither": (
        "两条预测都没命中:divmax 既没顶到最大间隔,也不同于随机基线。\n"
        "⇒ 机制是第三种,先看下标直方图再下结论(原始下标已 dump)。"
    ),
}


# ---------------------------------------------------------------- 数据侧
class AttrDict(dict):
    """属性可访问的 dict。

    `dataset/utils.py:load_anno` 用**属性**读 corpus 条目(`d.data_root`、
    `d.anno_path`、`d.media_type`),而 `configs/data.py` 里存的是普通 dict
    —— 真实链路里这层包装由 `utils/config.py` 的 `Config`(内部 EasyDict)完成,
    直接把 `available_corpus[...]` 塞进数据集会 `AttributeError`。

    这里不 import easydict:它只是 `utils/config.py` 的传递依赖,不保证在任何
    跑本脚本的环境里都装了(本机就没有),而 corpus 条目是扁平的字符串字典,
    等价的最小包装就够。**必须是 dict 子类** —— `load_anno` 用
    `isinstance(ann_file_list, dict)` 决定要不要包成单元素列表。
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:                      # 属性协议要求 AttributeError
            raise AttributeError(name) from exc


def build_eval_dataset(pool: int):
    """复用评测那条链路,逐位重建 1.8 那轮看到的帧。"""
    import torch  # noqa: F401  - torchvision 需要
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    from configs.data import available_corpus
    from dataset.ret_dataset import VidTxtRetEvalDataset

    # 与 dataset/__init__.py:get_test_transform 逐行一致(Normalize 在那里是注释掉的)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.Lambda(lambda x: x.float().div(255.0)),
    ])
    return VidTxtRetEvalDataset(
        AttrDict(available_corpus["msrvtt_1k_test"]),
        transform,
        num_frames=pool,
        video_reader_type="decord",
        sample_type="middle",
        num_tries=1,
    )


def collect_indices(pool: int, ks, limit, workers: int):
    """跑一遍数据集,给每个 K 收集三种模式的选帧下标。"""
    import torch
    from torch.utils.data import DataLoader

    from utils.elastic_t import subsample_frame_indices
    from utils.frame_select import greedy_kcenter, pixel_features, random_indices

    ds = build_eval_dataset(pool)
    total = len(ds) if limit is None else min(limit, len(ds))
    print(f"数据集 {len(ds)} 条,本次跑 {total} 条,母序列 {pool} 帧。", flush=True)

    out = {k: {"uniform": [], "divmax": [], "random": []} for k in ks}
    loader = DataLoader(
        torch.utils.data.Subset(ds, range(total)),
        batch_size=8, num_workers=workers, shuffle=False, collate_fn=lambda b: b,
    )
    done = 0
    for batch in loader:
        for frames, index in batch:
            feats = pixel_features(frames)
            for k in ks:
                out[k]["uniform"].append(subsample_frame_indices(pool, k))
                out[k]["divmax"].append(greedy_kcenter(feats, k))
                out[k]["random"].append(random_indices(pool, k, seed=0, index=int(index)))
            done += 1
        if done % 200 < 8:
            print(f"  ... {done}/{total}", flush=True)
    return out


# ---------------------------------------------------------------- 报告
def report(out, pool, ks):
    lines = []
    for k in ks:
        lines.append(f"\n## K={k}(母序列 {pool} 帧)\n")
        stats = {m: index_stats(v, pool) for m, v in out[k].items()}
        lines.append("| 模式 | 端点命中率 | 平均配对间隔 \\|i−j\\| | 平均跨度 |")
        lines.append("|---|---|---|---|")
        for m in ("uniform", "divmax", "random"):
            s = stats[m]
            lines.append(
                f"| `{m}` | {s['endpoint_rate']:.3f} | "
                f"{s['mean_pair_gap']:.2f} | {s['mean_span']:.2f} |"
            )
        d = discriminate(stats["divmax"], pool, k)
        lines.append(
            f"| *随机理论值* | {d['random_endpoint_rate']:.3f} | "
            f"{d['random_mean_pair_gap']:.2f} | — |"
        )
        lines.append("")
        lines.append(
            f"divmax 端点命中率是随机基线的 **{d['endpoint_ratio_vs_random']:.2f}×**;"
            f"最大可能间隔 {d['max_possible_gap']:.0f}。\n"
        )
        lines.append(VERDICT_TEXT[d["verdict"]])

        # 下标直方图:哪些位置被 divmax 偏爱
        c = Counter(i for idx in out[k]["divmax"] for i in idx)
        hist = "  ".join(f"{i}:{c.get(i, 0)}" for i in range(pool))
        lines.append(f"\ndivmax 各下标被选中次数:\n```\n{hist}\n```")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=16, help="母序列帧数(与 1.8 一致)")
    ap.add_argument("--ks", default="2,4", help="逗号分隔的 K")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条(冒烟用)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="experiments/evals/frame_select/diag_indices.json")
    args = ap.parse_args()

    # 这两个钩子会改变数据集吐出来的帧,开着跑等于诊断了另一件事。
    for var in ("MVC_FRAME_SELECT", "MVC_FRAME_PERTURB"):
        val = os.environ.get(var, "").strip().lower()
        if val and val != "none":
            sys.exit(f"拒绝执行:{var}={val} 已设置。诊断必须在未选帧/未扰动的 "
                     f"{args.pool} 帧母序列上跑,先 unset {var}。")

    ks = [int(x) for x in args.ks.split(",")]
    out = collect_indices(args.pool, ks, args.limit, args.workers)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"pool": args.pool, "indices": {str(k): v for k, v in out.items()}}, f)
    print(f"\n原始下标已写入 {args.out}")
    print(report(out, args.pool, ks))


if __name__ == "__main__":
    main()
