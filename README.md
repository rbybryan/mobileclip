# MobileViCLIP experiments: efficient video–text representations

Research code exploring **which temporal components help video–text retrieval, what they cost,
and how caption quality affects the questions we can test**. Built on
[MobileViCLIP (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_MobileViCLIP_An_Efficient_Video-Text_Model_for_Mobile_Devices_ICCV_2025_paper.html).

**Status:** experimental replication and ablation study. A 1M-clip experiment round is
reported; the proposed 10M-clip training and end-to-end mobile latency evaluation are not
established by this release. [Results and limitations](docs/RESULTS.md).

[CPU demo](#try-the-evaluator-on-a-cpu) · [Reported results](docs/RESULTS.md) ·
[Full reproduction](#reproduce-model-experiments) · [中文指南](README_zh.md)

## Contributions in this repository

The base architecture, core training code, and original paper belong to
[MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP).
This repository adds and evaluates:

- **Controlled model changes:** temporal-branch removal, attention/statistics/surprise
  pooling, and sigmoid-loss variants, with isolated experiment configurations.
- **Evaluation tooling:** saved per-query ranks, paired bootstrap comparisons,
  cross-seed checks, and diagnostics for frame selection and temporal sensitivity.
- **Data analysis:** a model-assisted audit of a 10,000-clip InternVid-zh sample and a
  frame-based recaptioning pipeline.
- **Engineering checks:** regression tests for model components and evaluation utilities,
  plus export and component-cost experiments.

See [model changes](model-optimization/docs/model-changes.md),
[evaluator](experiments/paired_bootstrap.py), and [recaptioning](recaption/README.md)
for the implementation. The [upstream README](docs/UPSTREAM_README.md) is retained for attribution.

## What the experiments found

![Reported paired differences from the ablation study](docs/assets/ablation-summary.svg)

Temporal-branch removal is a candidate for further efficiency testing: the reported
raw R-sum change is **+1.40, 95% CI [−1.10, +3.90]** at the 1M tier.
This does not demonstrate improved accuracy, equivalence, or formal non-inferiority.
Attention pooling's initial gain did not persist in a second seed; second-order pooling
and sigmoid loss have negative reported comparisons. These findings help identify which
changes deserve more compute and which need a different hypothesis.

The chart transcribes existing reports; it does not rerun the benchmark. Raw and
DSL scores use different protocols. Operator-count and CPU-component savings are not
end-to-end mobile speedups. See [the evidence, provenance, and limits](docs/RESULTS.md)
before quoting a result.

## Try the evaluator on a CPU

Python 3.10–3.12. No GPU, credentials, model weights, or dataset downloads are needed.
Run from the repository root:

```bash
git clone https://github.com/rbybryan/mobileclip.git
cd mobileclip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python examples/paired_bootstrap_demo.py
python -m pytest -q tests/test_paired_bootstrap.py tests/test_collect_frame_select.py tests/test_release_environment.py
```

The demo creates deterministic **synthetic** query ranks, compares a deliberately improved
candidate with its baseline using the real evaluator, and writes
`outputs/demo/paired_comparison.json`. It also checks that comparing a model with itself
returns a zero difference and a zero-width interval. These numbers are not research results.

To redraw the reported figure from the supplied summary table:

```bash
python scripts/plot_reported_results.py
```

## Reproduce model experiments

Full training/evaluation uses a separate NVIDIA/Linux environment and requires licensed
weights, annotations, and videos obtained from their original providers.

```bash
ENV_NAME=mobileviclip bash scripts/setup_env.sh
conda activate mobileviclip
```

Then follow [CHECKPOINTS.md](CHECKPOINTS.md) for data and weight preparation,
[the experiment guide](experiments/README.md) for evaluation, and
[the ablation runbook](model-optimization/docs/run.md) for training and paired comparisons.
The CPU demo environment is not a replacement for the historical GPU environment.
`requirements.lock.txt` records a historical machine environment; it is not a portable
installation recipe. The setup script is the documented training recipe.

## Repository guide

| Area | Contents |
|---|---|
| `models/`, `tasks_clip/`, `dataset/` | Upstream model/training components with this study's changes |
| `experiments/` | Configurations, evaluation, paired statistics, and diagnostic reports |
| `model-optimization/` | Model-change explanations and experiment retrospectives |
| `recaption/` | Caption audit and recaption pipeline |
| `tests/` | CPU utility tests and model-component tests with heavier dependencies |
| `docs/RESULTS.md` | Public result interpretation and reproducibility limits |
| `COMMANDS.md`, `TROUBLESHOOTING.md` | Detailed historical runbooks |

## Attribution and license

[MIT license](LICENSE), retaining the upstream copyright notice. Changes and experiment
tooling in this repository are by Boyan Rong. Third-party model weights, datasets, and
vendored components remain subject to their own terms and notices. No dataset or checkpoint
license is granted by this repository's code license.

For the underlying method, cite the [MobileViCLIP paper](https://arxiv.org/abs/2508.07312).
For this experimental extension, cite the repository and the exact commit used.

[Release-preparation scope and publication notes](docs/RELEASE_PREPARATION.md).
