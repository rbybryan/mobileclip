"""Run the repository's paired evaluation on synthetic ranks, without downloads.

These invented observations demonstrate the evaluator, not model performance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.paired_bootstrap import paired_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo/paired_comparison.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    baseline = {d: rng.integers(0, 30, 200) for d in ("i2t", "t2i")}
    candidate = {d: ranks.copy() for d, ranks in baseline.items()}
    # Deliberately improve a subset of queries. Keep query order identical.
    candidate["t2i"][:30] = 0
    comparison = paired_bootstrap(candidate, baseline, n_boot=1000, seed=7)
    identity = paired_bootstrap(baseline, baseline, n_boot=100, seed=7)
    assert identity["R_sum"]["delta"] == 0
    assert identity["R_sum"]["lo"] == identity["R_sum"]["hi"] == 0
    assert comparison["img_r1"]["delta"] > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "data": "synthetic; not a benchmark result", "seed": 7,
        "queries_per_direction": 200, "bootstrap_resamples": 1000,
        "comparison": comparison,
    }, indent=2) + "\n")
    metric = comparison["img_r1"]
    print("SYNTHETIC demonstration — no model weights or datasets used")
    print(f"Text-to-video R@1 difference: {metric['delta']:+.2f} percentage points")
    print(f"Paired 95% CI: [{metric['lo']:+.2f}, {metric['hi']:+.2f}]")
    print(f"Identity comparison: zero difference and zero-width CI. Saved {args.output}")


if __name__ == "__main__":
    main()
