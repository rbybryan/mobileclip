# Troubleshooting

## `libtorchaudio.so` / `libtorch_cuda_cpp.so: cannot open shared object file`

### Symptom

Running an evaluation (e.g. `scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh`)
fails at import time with:

```
OSError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
...
OSError: Could not load this library: .../torchaudio/lib/libtorchaudio.so
```

### Cause

`torchaudio` was built against a **different torch version** than the one installed.
The env is pinned to `torch 1.13.1+cu117`, whose runtime uses `libtorch_cuda.so`; the
split-out `libtorch_cuda_cpp.so` only existed in torch ≤ 1.12. When a later
`pip install` pulls in a mismatched `torchaudio` build, its shared libs can no longer
load, and the failure happens at *import* time — taking down even video/image-only
tasks (like MSRVTT retrieval) that never touch audio.

### Fix — reinstall torch / torchvision / torchaudio so they match

```bash
conda activate mobileviclip
pip install --no-cache-dir --force-reinstall --no-deps \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

Verify all three load and report matching versions:

```bash
python -c "import torch, torchaudio, torchvision; print(torch.__version__, torchaudio.__version__, torchvision.__version__)"
# expected: 1.13.1+cu117 0.13.1+cu117 0.14.1+cu117
```

### Notes

- `scripts/setup_env.sh` pins `torchaudio==0.13.1+cu117` (matching
  `requirements.lock.txt`) to keep the torch/torchaudio builds in sync and avoid
  this drift on fresh setups.
- `dataset/utils.py` imports `torchaudio` lazily (inside `load_audio_from_path`),
  so a broken/missing audio stack only affects audio tasks — video/image evals
  import cleanly regardless.
