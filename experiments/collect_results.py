#!/usr/bin/env python3
"""汇总实验评测结果并与基线对比,输出 markdown 表格。

只依赖标准库。扫描给定目录(默认 experiments/runs 与 experiments/evals)下的
eval_res*.json(由 tasks_clip/{pretrain,retrieval,retrieval_mc}.py 落盘,
pandas orient="columns" 格式:{"<protocol>": {"<metric>": value}}),
按「评测协议 × 指标」整理成对比表。

两个打分口径(见 tasks_clip/retrieval_utils.py 的 raw/DSL 两口径):
  - `.../`      DSL(dual-softmax,测试集级先验)—— 原 repo 主报口径
  - `..._emb/`  原始余弦相似度 —— 无测试集先验
对比结论必须两个口径同时成立才可信,故两个口径都输出。

用法:
    python experiments/collect_results.py
    python experiments/collect_results.py --baseline baseline_official
    python experiments/collect_results.py --dirs experiments/evals --format md
"""
import argparse
import sys
import json
import os
from collections import OrderedDict

PREFERRED_METRICS = [
    "txt_r1", "txt_r5", "txt_r10", "txt_r_mean",
    "img_r1", "img_r5", "img_r10", "img_r_mean",
    "t2v_r1", "t2v_r5", "t2v_r10",
    "v2t_r1", "v2t_r5", "v2t_r10",
    "r_mean", "r_sum", "accuracy",
]

# kill 判据统一用 R-Sum(6 项召回之和);eval json 里没有就现算
R_SUM_COMPONENTS = ["txt_r1", "txt_r5", "txt_r10", "img_r1", "img_r5", "img_r10"]

RESULT_FILES = ("eval_res.json", "eval_res_best.json", "eval_res_latest.json")


def find_results(root):
    """yield (run_name, benchmark, filepath); run/benchmark 由目录层级推断。"""
    for dirpath, _dirnames, filenames in os.walk(root):
        hits = [f for f in RESULT_FILES if f in filenames]
        if not hits:
            continue
        # 同目录多个结果文件时按 RESULT_FILES 的优先级取一个
        fname = hits[0]
        rel = os.path.relpath(dirpath, root)
        parts = [p for p in rel.split(os.sep) if p not in (".", "")]
        run = parts[0] if parts else os.path.basename(root)
        bench = "/".join(parts[1:]) if len(parts) > 1 else "-"
        yield run, bench, os.path.join(dirpath, fname)


def load_result(path):
    """return OrderedDict{protocol: {metric: value}}"""
    with open(path) as f:
        raw = json.load(f)
    out = OrderedDict()
    for protocol, metrics in raw.items():
        if isinstance(metrics, dict):
            if "r_sum" not in metrics and all(c in metrics for c in R_SUM_COMPONENTS):
                metrics["r_sum"] = sum(metrics[c] for c in R_SUM_COMPONENTS)
            out[protocol] = metrics
    return out


def metric_order(metrics):
    known = [m for m in PREFERRED_METRICS if m in metrics]
    rest = sorted(m for m in metrics if m not in PREFERRED_METRICS)
    return known + rest


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)



def run_compare_mc(arm_dir, base_dir, n_boot, seed, out_path=None):
    """多选任务(SSv2-MC 等)的配对 bootstrap。

    多选评测只产出一个标量 acc,但统计问题和检索口径完全一样:两条臂答的是同一批
    题目 ⇒ 是配对比较。不配对就只能拿两个裸 acc 相减,分辨率会差好几倍。
    输入是 `retrieval_mc.py` 落盘的 `mc_correct.npz`(逐题 0/1)。
    """
    from paired_bootstrap import (load_mc_correct, paired_bootstrap_mc,
                                  unpaired_halfwidth_mc)

    arm, base = load_mc_correct(arm_dir), load_mc_correct(base_dir)
    res = paired_bootstrap_mc(arm, base, n_boot=n_boot, seed=seed)["acc"]

    verdict = "**显著**" if res["significant"] else "不显著(CI 含 0)"
    half_paired = (res["hi"] - res["lo"]) / 2
    half_unpaired = unpaired_halfwidth_mc(arm, base, n_boot=n_boot, seed=seed)
    lines = [
        f"# 配对 bootstrap(多选):`{os.path.basename(arm_dir.rstrip('/'))}` − "
        f"`{os.path.basename(base_dir.rstrip('/'))}`",
        "",
        f"题目数 n={len(arm)} · {n_boot} 次重采样 · seed={seed} · 95% CI",
        "",
        "| 指标 | 臂 | 基线 | Δ | 95% CI | 判定 |",
        "|---|---:|---:|---:|---|---|",
        f"| acc | {res['arm']:.2f} | {res['baseline']:.2f} | {res['delta']:+.2f} | "
        f"[{res['lo']:+.2f}, {res['hi']:+.2f}] | {verdict} |",
        "",
        "**配对买到了多少分辨率**(acc 的 95% CI 半宽):",
        f"- 配对:**±{half_paired:.2f}**",
        f"- 无配对:±{half_unpaired:.2f}",
    ]
    if half_paired < 1e-9:
        lines.append(
            "- ⇒ 配对半宽为 **0**:若这是自比对(臂目录 == 基线目录)说明链路正确;"
            "若是两条不同的臂,说明两次评测逐题结果完全相同,先查是不是比错了目录。"
        )
    else:
        lines.append(
            f"- ⇒ 配对把可分辨的最小差异缩到约 **1/{half_unpaired / half_paired:.1f}**。"
        )
    lines += [
        "",
        "判读:**以 CI 是否含 0 为准**。SSv2-MC 是**时序**探针(174 选 1,随机基线 0.57%),",
        "MSR-VTT 上「删 `temporal_embedding` 更好」的结论要在这里也不变号才算数。",
    ]
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
        print(f"\n[written to {out_path}]")
    return res


def run_compare(arm_dir, base_dir, prefix, n_boot, seed, out_path=None):
    """配对 bootstrap 比较两条臂(评审 U10-1 / 评审 U10-1)。

    为什么不能直接拿 ±1.6 判:那是**无配对**带。两条臂评的是同一批查询,
    配对做差会抵消共享方差 —— 2026-07-29 的 1M 结果显示「10× 数据」的效应
    只有 +0.82 r_mean,不配对就根本分辨不出来。
    """
    from paired_bootstrap import load_ranks, paired_bootstrap, unpaired_halfwidth

    arm, base = load_ranks(arm_dir, prefix), load_ranks(base_dir, prefix)
    res = paired_bootstrap(arm, base, n_boot=n_boot, seed=seed)

    lines = [
        f"# 配对 bootstrap:`{os.path.basename(arm_dir.rstrip('/'))}` − "
        f"`{os.path.basename(base_dir.rstrip('/'))}`",
        "",
        f"口径 `{prefix}` · 查询数 i2t={len(arm['i2t'])} / t2i={len(arm['t2i'])} · "
        f"{n_boot} 次重采样 · seed={seed} · 95% CI",
        "",
        "| 指标 | 臂 | 基线 | Δ | 95% CI | 判定 |",
        "|---|---:|---:|---:|---|---|",
    ]
    order = ["txt_r1", "txt_r5", "txt_r10", "img_r1", "img_r5", "img_r10", "r_mean", "R_sum"]
    for k in order:
        r = res[k]
        verdict = "**显著**" if r["significant"] else "不显著(CI 含 0)"
        lines.append(
            f"| {k} | {r['arm']:.2f} | {r['baseline']:.2f} | {r['delta']:+.2f} | "
            f"[{r['lo']:+.2f}, {r['hi']:+.2f}] | {verdict} |"
        )

    half_paired = (res["R_sum"]["hi"] - res["R_sum"]["lo"]) / 2
    half_unpaired = unpaired_halfwidth(arm, base, n_boot=n_boot, seed=seed)
    lines += [
        "",
        "**配对买到了多少分辨率**(R-Sum 的 95% CI 半宽):",
        f"- 配对:**±{half_paired:.2f}**",
        f"- 无配对:±{half_unpaired:.2f}",
    ]
    if half_paired < 1e-9:
        # 自比对(臂 == 基线)时共享方差被完全抵消,半宽恰为 0 —— 这正是
        # 配对生效的判据本身,算比值只会打印一个无意义的大数。
        lines.append(
            "- ⇒ 配对半宽为 **0**:两侧逐次抽到同一组下标、差值恒等于 0。"
            "若这是自比对(臂目录 == 基线目录),说明配对链路正确;"
            "若这是两条**不同**的臂,说明两次评测产出了逐查询完全相同的 rank,先查是不是比错了目录。"
        )
    else:
        lines.append(
            f"- ⇒ 配对把可分辨的最小差异缩到约 **1/{half_unpaired / half_paired:.1f}**。"
        )
    lines += [
        "",
        "判读:**以 CI 是否含 0 为准**,不要再套 `README.md` §6.1 的 ±1.6 —— 那是无配对带,",
        "用在 A/B 上过于保守。两口径同向(`ret_test/` 与 `ret_test_emb/` 都跑一遍)的纪律照旧。",
    ]
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
        print(f"\n[written to {out_path}]")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="*",
                    default=["experiments/runs", "experiments/evals"],
                    help="要扫描的根目录")
    ap.add_argument("--baseline", default=None,
                    help="作为对比基线的 run 名(目录名);给出后附加 Δ 列")
    ap.add_argument("--out", default=None, help="同时写入该 markdown 文件")
    ap.add_argument("--compare", nargs=2, metavar=("ARM_DIR", "BASE_DIR"), default=None,
                    help="配对 bootstrap 比较两条臂(U10-1/G4)。需要两边都有 ranks_*.npz;"
                         "缺了就重跑一次评测(纯前向,MSR-VTT 1k 约 1 分钟)")
    ap.add_argument("--compare-mc", nargs=2, metavar=("ARM_DIR", "BASE_DIR"), default=None,
                    help="多选任务(SSv2-MC 等)的配对 bootstrap。需要两边都有 mc_correct.npz"
                         "(由 tasks_clip/retrieval_mc.py 落盘)")
    ap.add_argument("--prefix", default="ret_test_emb/",
                    help="比较用的口径,默认 raw 主口径;DSL 用 ret_test/")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap 重采样次数")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap 随机种子(可复现)")
    args = ap.parse_args()

    if args.compare and args.compare_mc:
        raise SystemExit("[FAIL] --compare 与 --compare-mc 只能给一个")

    if args.compare:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        run_compare(args.compare[0], args.compare[1], args.prefix,
                    args.n_boot, args.seed, args.out)
        return

    if args.compare_mc:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        run_compare_mc(args.compare_mc[0], args.compare_mc[1],
                       args.n_boot, args.seed, args.out)
        return

    # rows[(bench, protocol)][run] = {metric: value}
    rows = OrderedDict()
    contributing = []  # 真正提供了结果的根目录,用于在产物里标明出处
    for root in args.dirs:
        if not os.path.isdir(root):
            continue
        found_here = False
        for run, bench, path in sorted(find_results(root)):
            for protocol, metrics in load_result(path).items():
                rows.setdefault((bench, protocol), OrderedDict())[run] = metrics
                found_here = True
        if found_here:
            contributing.append(root)

    if not rows:
        print("没有找到任何 eval_res*.json;先跑 experiments/run_eval.sh / run_pretrain.sh。")
        return

    runs_found = sorted({run for per_run in rows.values() for run in per_run})

    # 评测全崩时 evals/ 下不会有结果,但 runs/ 下的训练自评测照样会被捞进来,
    # 于是产出一个名叫 BASELINE.md、内容却是试点数字的文件(2026-07-22 真实事故,
    # 见 TROUBLESHOOTING.md)。所以出处必须写进产物、基线缺失必须报警。
    if args.baseline and args.baseline not in runs_found:
        print(f"\n⚠️  指定的基线 run `{args.baseline}` 不在结果里!"
              f"实际找到的 run:{', '.join(runs_found)}"
              f"\n    ——(该基线的评测没跑成功?下面的表里没有基线,Δ 列已省略。)\n")

    lines = [
        "<!-- generated by experiments/collect_results.py -->",
        f"> 扫描目录:{', '.join(contributing)}",
        f"> 包含的 run:{', '.join(runs_found)}",
    ]
    for (bench, protocol), per_run in rows.items():
        kind = "raw-cosine" if protocol.rstrip("/").endswith("_emb") else "DSL"
        lines.append(f"\n## {bench} · `{protocol}` ({kind})\n")
        all_metrics = OrderedDict()
        for metrics in per_run.values():
            for m in metric_order(metrics):
                all_metrics[m] = None
        cols = list(all_metrics)
        base = per_run.get(args.baseline) if args.baseline else None

        delta_metric = "r_sum" if (base is not None and "r_sum" in base) else "r_mean"
        header = ["run"] + cols
        if base is not None:
            header += [f"Δ{delta_metric}(vs {args.baseline})"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for run, metrics in per_run.items():
            cells = [run] + [fmt(metrics.get(m, "-")) for m in cols]
            if base is not None:
                if run != args.baseline and delta_metric in metrics and delta_metric in base:
                    cells.append(f"{metrics[delta_metric] - base[delta_metric]:+.2f}")
                else:
                    cells.append("-")
            lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
