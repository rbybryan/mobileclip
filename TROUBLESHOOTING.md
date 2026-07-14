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

---

## `ImportError: cannot import name 'Cache' from 'transformers'`

### Symptom

An evaluation fails while importing the model backbone:

```
File ".../models/backbones/internvideo2/internvideo2_clip_text.py", line 5, in <module>
    from peft import get_peft_model, LoraConfig, TaskType
...
File ".../peft/peft_model.py", line 38, in <module>
    from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel
ImportError: cannot import name 'Cache' from 'transformers'
```

### Cause

`peft >= 0.8` imports the `Cache` class from `transformers`, which only exists in
`transformers >= 4.36`. This repo pins `transformers==4.28.1` (alongside
`tokenizers==0.13.3` and the InternVideo2 text code), so a newer peft (the lock file
previously pinned `peft==0.12.0`) can never import.

The repo only uses `get_peft_model`, `LoraConfig`, and `TaskType` — all stable since
peft 0.2 — so the fix is to downgrade peft rather than bump transformers.

### Fix — downgrade peft to a version compatible with transformers 4.28.1

Install with `--no-deps` (see the warning below):

```bash
conda activate mobileviclip
pip install --no-cache-dir --no-deps "peft==0.4.0"
```

Verify it imports:

```bash
python -c "from peft import get_peft_model, LoraConfig, TaskType; print('peft OK')"
```

`scripts/setup_env.sh` and `requirements.lock.txt` now pin `peft==0.4.0` so fresh
setups avoid this mismatch.

> **Warning — always use `--no-deps` for peft (and open_clip).** Installing
> `peft` without `--no-deps` lets pip's resolver UPGRADE torch (e.g. to a 2.x
> wheel) to "satisfy" peft's unpinned `torch` requirement. That uninstalls the
> `torch 1.13.1+cu117` build and breaks `torchvision` / `torchaudio` (which are
> built for 1.13.1) — see the next section. peft's real dependencies
> (`transformers`, `accelerate`, `safetensors`, `huggingface-hub`, etc.) are
> already present in this env, so `--no-deps` is safe.

---

## `ImportError: cannot import name '_cast_Long' from 'torch.onnx.symbolic_opset9'`

### Symptom

```
UserWarning: Failed to load image Python extension: Could not load this library:
  .../torchvision/image.so
...
ImportError: cannot import name '_cast_Long' from 'torch.onnx.symbolic_opset9'
```

Often preceded by a pip warning like:

```
torchvision 0.14.1+cu117 requires torch==1.13.1, but you have torch 2.x which is incompatible.
torchaudio 0.13.1+cu117 requires torch==1.13.1, but you have torch 2.x which is incompatible.
```

### Cause

`torch` was upgraded to a 2.x wheel (usually by a `pip install <pkg>` that ran
**without** `--no-deps` — commonly `peft` or `open_clip`), which uninstalled the
`torch 1.13.1+cu117` build. `torchvision 0.14.1` / `torchaudio 0.13.1` are compiled
against torch 1.13.1 and can no longer load their C++ extensions against torch 2.x.

### Fix — restore the matching torch stack, then reinstall the offending pkg with `--no-deps`

```bash
conda activate mobileviclip

# Restore the cu117 trio
pip install --no-cache-dir --force-reinstall --no-deps \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Reinstall whatever pulled torch up, this time isolated from its deps
pip install --no-cache-dir --no-deps "peft==0.4.0" -i https://pypi.org/simple

# Verify all three agree and load
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)"
# expected: 1.13.1+cu117 0.14.1+cu117 0.13.1+cu117
```
