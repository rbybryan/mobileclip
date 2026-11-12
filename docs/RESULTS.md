# Experiment notes

These comparisons use a 1M-clip training subset and the MSR-VTT 1k retrieval test set.
I used 10,000 paired bootstrap resamples, keeping the same query indices in each
baseline–variant comparison.

## Main comparisons

| Change | Score | Seed | Δ R-sum | 95% paired CI |
|---|---|---|---:|---|
| Remove temporal branch | raw | 42 | +1.40 | [−1.10, +3.90] |
| Attention pooling | raw | 43 | +0.10 | [−3.10, +3.40] |
| Second-order pooling | raw | 42 | −4.30 | [−8.30, −0.30] |
| Sigmoid loss | dual-softmax | 42 | −5.70 | [−10.10, −1.20] |

`R_sum` is the sum of R@1/5/10 in both retrieval directions. In the evaluator,
`txt_*` is video-to-text and `img_*` is text-to-video. Raw and dual-softmax scores
use different scoring procedures, so I've kept the protocol beside each comparison.

The values above come from §3 of the
[concise retrospective](../model-optimization/docs/retrospective-brief.md).
The [CSV](results/reported_ablation_summary.csv) and
[plot script](../scripts/plot_reported_results.py) reproduce the README figure.
Per-query rank files are not included in the repo; recomputing the intervals needs
those files or another checkpoint evaluation.

## What I take from this

Removing the temporal branch is worth testing further. The point estimate is slightly
positive, but the interval also allows a small loss. Earlier notes called this
“non-inferior.” That was too strong: the δ=2.5 R-sum threshold came from observed seed
variation, rather than a prespecified acceptable-loss margin. I now treat it as a
screening result for the next experiment.

Attention pooling looked better in the first seed, but the second-seed difference
was close to zero. Second-order pooling and the sigmoid loss were less promising in
the comparisons above. Surprise pooling converged to mean pooling; I left it out of
the figure because the summary doesn't contain a paired interval for it.

Some earlier tables use in-training evaluations and others use offline reevaluations.
Their point estimates differ slightly. The figure uses the paired-comparison table
throughout, without mixing estimates and intervals from different evaluation runs.
Older notes also use “six schemes” inconsistently; the README names the five distinct
changes summarized here.

## Efficiency and caption analysis

The temporal-branch experiment reports about 27.8% fewer exported operators and an
18% reduction in the CPU mixer measurement. These are component-level results.
I still need an end-to-end latency measurement on a target device.

The model-assisted caption audit covers a 10,000-clip sample. It suggests that temporal
descriptions are scarce, which motivates the recaptioning work. It doesn't yet tell
me whether changing the captions will make temporal modeling useful. That needs a
controlled training comparison, along with checks on the caption judgments.

## Next experiments

I'd first repeat the branch-removal comparison across seeds and retrieval datasets,
then measure full-model latency on a target device. The larger 10M-clip training run
is still pending, and no trained checkpoint for these changes is distributed here.

For the full experiment commands, see the
[ablation runbook](../model-optimization/docs/run.md). The CPU example in the
[README](../README.md) is a way to try the evaluator with synthetic data.
