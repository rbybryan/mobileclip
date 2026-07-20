#!/usr/bin/env python3
"""汇总实验评测结果并与基线对比,输出 markdown 表格。

只依赖标准库。扫描给定目录(默认 experiments/runs 与 experiments/evals)下的
eval_res*.json(由 tasks_clip/{pretrain,retrieval,retrieval_mc}.py 落盘,
pandas orient="columns" 格式:{"<protocol>": {"<metric>": value}}),
按「评测协议 × 指标」整理成对比表。

两个打分口径(见 notes/script-level-optimizations-zh.md H3):
  - `.../`      DSL(dual-softmax,测试集级先验)—— 原 repo 主报口径
  - `..._emb/`  原始余弦相似度 —— 无测试集先验
对比结论必须两个口径同时成立才可信,故两个口径都输出。

用法:
    python experiments/collect_results.py
    python experiments/collect_results.py --baseline baseline_official
    python experiments/collect_results.py --dirs experiments/evals --format md
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="*",
                    default=["experiments/runs", "experiments/evals"],
                    help="要扫描的根目录")
    ap.add_argument("--baseline", default=None,
                    help="作为对比基线的 run 名(目录名);给出后附加 Δ 列")
    ap.add_argument("--out", default=None, help="同时写入该 markdown 文件")
    args = ap.parse_args()

    # rows[(bench, protocol)][run] = {metric: value}
    rows = OrderedDict()
    for root in args.dirs:
        if not os.path.isdir(root):
            continue
        for run, bench, path in sorted(find_results(root)):
            for protocol, metrics in load_result(path).items():
                rows.setdefault((bench, protocol), OrderedDict())[run] = metrics

    if not rows:
        print("没有找到任何 eval_res*.json;先跑 experiments/run_eval.sh / run_pretrain.sh。")
        return

    lines = []
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
