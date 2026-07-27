#!/usr/bin/env python3
"""选帧诊断:divmax 到底选中了哪些帧?(零 GPU,零训练)

## 为什么要跑

1.8 实测(`experiments/frame_select_results.md`)在 K=2 上得到两条硬结论:
  - **选帧确实要紧**:uniform 相对 random +3.40/+3.90(两口径同向);
  - **但 divmax 是错的判据**:相对 uniform −2.80/−2.40,且**与纯随机无可测差别**。

本脚本把 `greedy_kcenter` 在真实测试帧上的**下标**dump 出来,用两条**独立**信号
判读机制。**不建模型、不过编码器、不需要 GPU。**

| 信号 | 度量什么 |
|---|---|
| **端点命中率**(vs 解析随机值 `1 − C(n−2,k)/C(n,k)`) | 是否在抓片段边界 |
| **内部 CV**(下标 1..n−2 被选次数的 std/mean) | 端点之外该指标还有没有区分度 |

## 2026-07-27 实测结论(1000 条 MSR-VTT 测试视频)

两个 K 上一致:**端点被系统性过采、内部近乎平坦**。

    K=2  端点率 0.423(随机 0.242,1.75×)  内部 CV 0.110
    K=4  端点率 0.706(随机 0.450,1.57×)  内部 CV 0.156
    K=2 直方图:下标 0 被选 283 次(均匀期望 125,2.26×)、下标 15 选 158 次,
                中间 1–14 落在 91–140 之间

⇒ 机制:**降采样像素距离≈单调漂移(镜头运动/曝光),k-center 的 argmax 把这点
漂移翻译成「取时间极值」**,于是半个帧预算花在边界帧(淡入淡出、转场、运动模糊)上。

原 §4.1 把 (A)「偏好离群帧」与 (B)「像素≠时间」写成互斥选项是**框架错了** ——
数据显示两者是同一机制的两面。论文里的那句话仍是「**要有代表性的帧,不要极端帧**」。

⚠️ **「平均配对间隔」不要拿来判读**:greedy k-center 的首点是最靠近均值的帧,
纯粹追逐端点也只能得到 `gap ≈ (n−1)/2`。第一版据此判错过一次,详见
`analyze_mechanism` 的文档字符串。

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
    counts = [0] * n
    for idx in index_lists:
        s = set(idx)
        if 0 in s or (n - 1) in s:
            endpoint_hits += 1
        for i in idx:
            counts[i] += 1
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
        "counts": counts,
    }


def interior_cv(counts) -> float:
    """内部下标(1 .. n−2)被选中次数的变异系数 std/mean。

    小 ⇒ 除端点外该指标**几乎不区分帧**;大 ⇒ 内部真的有结构。
    刻意排除两个端点,因为端点偏好由 `endpoint_rate` 单独度量,
    混在一起会让「端点过采」把 CV 顶高、看起来像「内部有结构」。
    """
    interior = counts[1:len(counts) - 1]
    if not interior:
        raise ValueError("n 太小,没有内部下标")
    mean = sum(interior) / len(interior)
    if mean == 0:
        return 0.0
    var = sum((c - mean) ** 2 for c in interior) / len(interior)
    return math.sqrt(var) / mean


# 判据阈值。两条信号都是相对量,不依赖具体 n/k。
ENDPOINT_BIAS_RATIO = 1.3      # 端点命中率 / 随机理论值,超过即算「偏好端点」
FLAT_INTERIOR_CV = 0.25        # 内部下标次数的 CV,低于即算「内部无区分度」


def analyze_mechanism(divmax: dict, n: int, k: int) -> dict:
    """判读 divmax 的下标分布。**两条独立信号,分别报,不互相否决。**

    ## 为什么不再用「平均配对间隔」当判据(2026-07-27 修正)

    第一版把假说 (A)(k-center 偏好离群帧)的签名写成「间隔顶到最大 ≈ n−1」。
    **那是推错的。** `greedy_kcenter` 的第一个点是 `argmin ‖x − 均值‖`,即**最有
    代表性**的帧(通常在中段);第二个点才是最远点。所以即便纯粹追逐端点,
    K=2 也只会得到 `gap ≈ |中段 − 端点| ≈ (n−1)/2`,**够不到 n−1**。

    后果:实测 K=4 时 `gap 6.33` 落在随机值 5.67 附近,旧函数据此判了 (B)
    「时间分布与随机无异」—— 而同一批数据的端点命中率是随机的 1.57×,
    与该结论直接矛盾。**间隔对 greedy k-center 不诊断,端点率才诊断。**

    现在报两条:
      `endpoint_ratio`  端点命中率 / 随机理论值 —— 是否在抓片段边界
      `interior_cv`     内部下标次数的变异系数 —— 端点之外该指标还有没有区分度
    """
    rand_ep = analytic_endpoint_rate(n, k)
    ep_ratio = divmax["endpoint_rate"] / rand_ep if rand_ep > 0 else float("inf")
    cv = interior_cv(divmax["counts"])

    biased = ep_ratio >= ENDPOINT_BIAS_RATIO
    flat = cv <= FLAT_INTERIOR_CV
    verdict = ("endpoint_bias_flat" if biased and flat else
               "endpoint_bias_structured" if biased else
               "uninformative" if flat else
               "structured_no_bias")
    return {
        "random_endpoint_rate": rand_ep,
        "random_mean_pair_gap": analytic_mean_pair_gap(n),
        "endpoint_ratio_vs_random": ep_ratio,
        "interior_cv": cv,
        "verdict": verdict,
    }


VERDICT_TEXT = {
    "endpoint_bias_flat": (
        "**机制:像素距离≈单调漂移 ⇒ argmax 变成「取时间极值」。**\n"
        "端点被系统性过采,而端点之外的下标分布近乎平坦 —— 也就是说降采样 RGB 上的\n"
        "远近主要由**镜头运动/曝光漂移**构成,它携带的时间信息只够把「最远」推到片段\n"
        "边界,内部几乎不区分帧。k-center 的 argmax 把这点漂移直接翻译成挑边界帧\n"
        "(淡入淡出、转场、运动模糊)。\n\n"
        "⇒ 原 §4.1 把 (A)「偏好离群帧」与 (B)「像素≠时间」写成互斥选项是**框架错了**:\n"
        "它们是同一机制的两面。论文里可写的那句仍是**要有代表性的帧,不要极端帧**,\n"
        "但现在有了机制:**最大化像素多样性 = 去挑片段边界**。\n"
        "与 GATE-0 相容 —— `freeze`(零分散)崩 13 分说明需要分散;本轮说明**均匀时间\n"
        "间隔已经把有用的分散拿够了**,再往极值推是有害的。"
    ),
    "endpoint_bias_structured": (
        "**端点被过采,且内部仍有结构。**\n"
        "指标不只是漂移 —— 端点之外也在做有区分度的选择。要判这份结构是否有语义,\n"
        "需要把选帧下标与内容标注对齐再看,本脚本给不出。"
    ),
    "uninformative": (
        "**该指标基本不携带信息**:端点命中率与随机无异,内部也平坦。\n"
        "⇒ 结论收窄为「廉价像素代理不可用」,**不能**推广到「多样性判据错」——\n"
        "要判后者需换语义特征重跑(那只能当上界报,见 §5 成本诚实性)。"
    ),
    "structured_no_bias": (
        "**内部有结构但不追逐端点。**\n"
        "这是三种失败模式之外的第四种,先看下标直方图再下结论(原始下标已 dump)。"
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
        lines.append("| 模式 | 端点命中率(**判据**) | 内部 CV(**判据**) | 平均配对间隔(*不诊断*) |")
        lines.append("|---|---|---|---|")
        for m in ("uniform", "divmax", "random"):
            s = stats[m]
            lines.append(
                f"| `{m}` | {s['endpoint_rate']:.3f} | {interior_cv(s['counts']):.3f} | "
                f"{s['mean_pair_gap']:.2f} |"
            )
        d = analyze_mechanism(stats["divmax"], pool, k)
        lines.append(
            f"| *随机理论值* | {d['random_endpoint_rate']:.3f} | — | "
            f"{d['random_mean_pair_gap']:.2f} |"
        )
        lines.append("")
        lines.append(
            f"divmax 端点命中率 = 随机基线的 **{d['endpoint_ratio_vs_random']:.2f}×**"
            f"(阈值 {ENDPOINT_BIAS_RATIO});内部 CV = **{d['interior_cv']:.3f}**"
            f"(阈值 {FLAT_INTERIOR_CV},低于即算内部无区分度)。\n"
        )
        lines.append(
            "> 「平均配对间隔」只作描述,**不参与判读** —— greedy k-center 的首点是\n"
            "> 最靠近均值的帧,即便纯粹追逐端点,间隔也够不到 n−1(见 `analyze_mechanism`\n"
            "> 的文档字符串:第一版正是在这里判错的)。\n"
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
