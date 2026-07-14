#!/usr/bin/env bash
# Reproduce the MobileViCLIP conda environment (CPU/login-node friendly).
#
# This is the VALIDATED recipe: all 7 repo entry modules (incl. tasks_clip/retrieval.py
# and tasks_clip/pretrain.py) import cleanly with it. flash-attn and apex are left OUT
# (optional; only needed on Ampere+ GPUs — the repo degrades gracefully without them).
#
# Deviations from mobileviclip.yml and WHY:
#   - av 11.0.0 -> 12.0.0     : 11.0.0 has no py3.10 wheel; 12.0.0 wheel bundles ffmpeg.
#   - open_clip_torch 2.20.0  : required by the MobileCLIP backbone; NOT in the yml.
#                               MUST install with --no-deps or pip silently upgrades
#                               torch to 2.x and breaks torchvision/onnx.
#   - deepspeed installed with DS_BUILD_OPS=0 (ops JIT-compile only when actually used).
#   - peft 0.12.0 -> 0.4.0    : peft>=0.8 imports `Cache` from transformers, which only
#                               exists in transformers>=4.36. This repo pins transformers
#                               4.28.1, so peft 0.12 fails with
#                               "ImportError: cannot import name 'Cache' from 'transformers'".
#                               The repo only uses get_peft_model/LoraConfig/TaskType, all
#                               stable since peft 0.2 — 0.4.0 imports cleanly on 4.28.1.
#   - flash-attn / apex       : skipped (build needs nvcc + a matching GPU).
set -eo pipefail

ENV_NAME="${ENV_NAME:-mobileviclip}"
L(){ echo "[$(date +%H:%M:%S)] $*"; }

L "=== Stage 1: create env $ENV_NAME (python 3.10) ==="
conda create -n "$ENV_NAME" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
L "python: $(which python) $(python --version 2>&1)"

L "=== Stage 2: PyTorch 1.13.1 + cu117 ==="
pip install --no-cache-dir \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

L "=== Stage 3: PyAV (binary wheel, bundled ffmpeg) ==="
pip install --no-cache-dir --only-binary=:all: "av==12.0.0"

L "=== Stage 4: core python deps ==="
pip install --no-cache-dir \
    transformers==4.28.1 tokenizers==0.13.3 timm==0.5.4 openai-clip==1.0.1 \
    einops==0.7.0 decord==0.6.0 opencv-python==4.8.0.76 \
    ftfy==6.2.3 regex==2024.7.24 pandas==2.0.3 scipy==1.13.0 scikit-learn==1.5.1 \
    numpy==1.26.4 pyyaml==6.0.1 easydict==1.13 yacs==0.1.8 tabulate==0.9.0 \
    tensorboard==2.19.0 wandb==0.16.1 webdataset==0.2.111 imageio==2.31.1 \
    accelerate==0.33.0 peft==0.4.0 safetensors==0.4.4 huggingface-hub==0.24.5 \
    fvcore==0.1.5.post20221221 iopath==0.1.10 termcolor==2.4.0 rich==13.4.2 \
    "setuptools==69.5.1" sentencepiece

L "=== Stage 5: spiking-neuron libs (MobileOne reparam block imports these) ==="
pip install --no-cache-dir "spikingjelly==0.0.0.0.14" "snntorch==0.9.4"

L "=== Stage 6: open_clip (MobileCLIP backbone). --no-deps is CRITICAL ==="
pip install --no-cache-dir --no-deps "open_clip_torch==2.20.0"

L "=== Stage 7: deepspeed (imported unguarded by the task entry scripts) ==="
DS_BUILD_OPS=0 pip install --no-cache-dir "deepspeed==0.10.1"

L "=== Stage 8: verify repo modules import ==="
export PYTHONPATH=".:$PYTHONPATH"
python - <<'PY'
import importlib
mods = ["open_clip","models","tasks_clip.retrieval_utils",
        "tasks_clip.retrieval","tasks_clip.pretrain",
        "dataset.video_utils","dataset.av_utils"]
ok = 0
for m in mods:
    try:
        importlib.import_module(m); print("OK  ", m); ok += 1
    except Exception as e:
        print("FAIL", m, "->", repr(e).splitlines()[0][:200])
print(f"{ok}/{len(mods)} modules import cleanly")
assert ok == len(mods), "some modules failed to import"
PY

L "=== DONE (env '$ENV_NAME' ready) ==="
L "Optional GPU extras (build on an Ampere+ node with nvcc):"
L "  pip install flash-attn==2.0.8 --no-build-isolation   # optional"
L "  apex from source                                       # optional"
