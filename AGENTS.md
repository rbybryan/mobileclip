# Repository Guidelines

## Project Structure & Module Organization

Core model definitions live in `models/`, with reusable vision and text components under `models/backbones/`. Dataset readers, samplers, and media decoding are in `dataset/`. Training and retrieval entry points are split between `tasks/` and the CLIP-specific `tasks_clip/`; shared configuration and runtime helpers live in `configs/`, `utils/`, and `tools/`. Experiment configurations and launchers are organized by model size and task under `scripts/pretraining/clip/` and `scripts/evaluation/clip/zero_shot/`. Use `preprocess/` for data preparation utilities and `tests/` for lightweight checks. Keep downloaded datasets, annotations, checkpoints, and generated logs outside source directories.

## Build, Test, and Development Commands

- `conda env create -f mobileviclip.yml` creates the pinned Python 3.10/PyTorch environment (`internvideo`).
- `conda activate internvideo` activates that environment.
- `PYTHONPATH=. python tests/test_cfg.py scripts/pretraining/clip/mobileviclip_tiny/config.py` loads and prints a configuration as a quick smoke test.
- `bash scripts/pretraining/clip/mobileviclip_tiny/run.sh` launches distributed tiny-model pretraining; adjust GPU counts and data paths in the script/config first.
- `bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh` runs a representative one-GPU retrieval evaluation after checkpoints and annotations are configured.

Run commands from the repository root so relative imports and configuration paths resolve correctly.

## Coding Style & Naming Conventions

Use four-space indentation and broadly follow PEP 8. Name functions, variables, and modules with `snake_case`; use `CapWords` for classes and `UPPER_CASE` for constants. Keep imports grouped as standard library, third-party, then local modules. Configuration files should use descriptive lowercase names such as `config_msrvtt.py`. No formatter or linter is currently enforced, so preserve nearby style and avoid unrelated formatting changes.

## Testing Guidelines

The current suite is a configuration smoke check rather than a full unit-test framework, and there is no coverage threshold. Add focused `test_*.py` files under `tests/` for deterministic logic. For model or data-pipeline changes, report the exact config, hardware assumptions, command, and relevant metric or log excerpt used for validation.

## Commit & Pull Request Guidelines

History uses short, imperative, single-line subjects (for example, `Update README.md`) without Conventional Commit prefixes. Keep commits focused. Pull requests should explain the motivation and behavior change, list validation commands, link related issues, and document required data or checkpoint setup. Never commit credentials, machine-specific paths, datasets, checkpoints, or generated logs.
