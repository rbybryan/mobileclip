"""Render the documented paired differences; does not rerun model evaluation."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "docs/results/reported_ablation_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 11,
                         "svg.hashsalt": "mobileclip-ablation-summary"})
    fig, ax = plt.subplots(figsize=(10.5, 4.1), layout="constrained")
    for i, row in enumerate(rows):
        value, lo, hi = (float(row[k]) for k in ("delta_r_sum", "ci_low", "ci_high"))
        ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt="o", capsize=5,
                    color="#315a73", markersize=7, linewidth=2)
    ax.set_yticks(range(len(rows)), [f"{r['variant']} · {r['protocol']} · seed {r['seed']}" for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlabel("Difference in R-sum from the matched baseline (95% paired bootstrap CI)")
    ax.set_title("Reported ablations · 1M training tier · MSR-VTT 1k", loc="left", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    fig.supxlabel("Transcribed from retrospective-brief.md; protocols differ. Intervals crossing zero do not prove equivalence.",
                  fontsize=9, color="#475569")
    path = ROOT / "docs/assets/ablation-summary.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
