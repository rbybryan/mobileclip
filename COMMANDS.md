# Commands Cheat Sheet

All command lines needed to set up, download assets, replicate, train, and evaluate MobileViCLIP.
Run everything from the repo root unless noted. See [CHECKPOINTS.md](CHECKPOINTS.md) for asset details.

---

## 0. Clone / repo

```bash
# This repo (private)
git clone git@github.com:rbybryan/mobileclip.git
cd mobileclip

# Upstream reference (optional)
git clone https://github.com/MCG-NJU/MobileViCLIP.git
```

---

## 1. Environment

The pinned stack (from `mobileviclip.yml`): **Python 3.10**, **PyTorch 1.13.1 + CUDA 11.7**, plus compiled
extensions (flash-attn, apex, deepspeed). Note the yml's internal env name is `internvideo`.

### Option 0 — one shot (recommended, validated)

```bash
ENV_NAME=mobileviclip bash scripts/setup_env.sh
conda activate mobileviclip
```

`scripts/setup_env.sh` is the **validated** recipe — after it runs, all 7 repo entry modules
(including `tasks_clip/retrieval.py` and `tasks_clip/pretrain.py`) import cleanly on a CPU node.
It installs the exact working set (`requirements.lock.txt` is the resulting `pip freeze`).
It skips flash-attn/apex (optional; Ampere+ GPU only) and handles three pitfalls the yml doesn't:

- **`av`**: 11.0.0 has no Python-3.10 wheel → installs `av==12.0.0` (wheel bundles FFmpeg).
- **`open_clip_torch`**: required by the MobileCLIP backbone but missing from the yml. It **must**
  be installed with `--no-deps`, or pip silently upgrades torch to 2.x and breaks torchvision/ONNX.
- **`deepspeed`**: imported unguarded by the task scripts → installed with `DS_BUILD_OPS=0`.

Also needs `setuptools` (for `pkg_resources`), handled by the script.
(`spikingjelly`/`snntorch` are no longer needed — they were dead imports, removed.)

### Option A — create from the yml (fastest, may need a machine with CUDA 11.7 toolchain)

```bash
conda env create -f mobileviclip.yml
conda activate internvideo          # <- env name declared inside the yml
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### Option B — manual, step by step (more robust across machines)

```bash
# 1) base env
conda create -n mobileviclip python=3.10 -y
conda activate mobileviclip

# 2) PyTorch 1.13.1 + CUDA 11.7 (match the paper's stack)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 3) core python deps
pip install \
    transformers==4.28.1 tokenizers==0.13.3 timm==0.5.4 openai-clip==1.0.1 \
    einops==0.7.0 decord==0.6.0 av==11.0.0 opencv-python==4.8.0.76 \
    ftfy==6.2.3 regex==2024.7.24 pandas==2.0.3 scipy==1.13.0 scikit-learn==1.5.1 \
    numpy==1.26.4 pyyaml==6.0.1 easydict==1.13 yacs==0.1.8 tabulate==0.9.0 \
    tensorboard==2.19.0 wandb==0.16.1 webdataset==0.2.111 imageio==2.31.1 \
    accelerate==0.33.0 peft==0.12.0 safetensors==0.4.4 huggingface-hub==0.24.5

# 4) CUDA-compiled extensions (need nvcc + matching CUDA toolkit; slow to build)
pip install ninja packaging
pip install flash-attn==2.0.8 --no-build-isolation
pip install deepspeed==0.10.1
# apex (optional; only if training uses it):
# git clone https://github.com/NVIDIA/apex && cd apex && \
#   pip install -v --disable-pip-version-check --no-cache-dir \
#   --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Notes / gotchas
- `flash-attn`, `apex`, `deepspeed`, `dropout-layer-norm`, `fused-dense-lib` compile CUDA kernels — you need `nvcc`
  (a CUDA toolkit) and a GPU whose arch matches. Inference/eval usually works without apex.
- If flash-attn 2.0.8 won't build against your CUDA, try a newer wheel and test whether the model still loads.
- On a cluster, load the CUDA module first, e.g. `module load cuda/11.7`, before building extensions.
- Sanity check GPU visibility: `python -c "import torch; print(torch.cuda.device_count())"`.

### Export your working env (for reproducibility)
```bash
conda env export --no-builds > env.lock.yml
pip freeze > requirements.lock.txt
```

---

## 2. Download checkpoints & data

```bash
pip install gdown

# Pretrained weights (MobileViCLIP tiny/small + MobileCLIP s0/s2)
mkdir -p checkpoints
gdown --folder https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7 -O checkpoints

# Annotations (downstream + InternVid)
mkdir -p anno_downstream
gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream
```

Expected layout:
```
checkpoints/
├── mobileclip_s0.pt
├── mobileclip_s2.pt
├── mobileviclip_tiny.pt
└── mobileviclip_small.pt
anno_downstream/
├── InternVid/InternVid-10M-flt_*.json
├── anet_ret_val.json
└── ...
```

Videos (download from official sites) into `data/`:
```
data/
├── anet_1.3_video_val_resize/    # ActivityNet
├── kinetics_400_val_10s_320p/    # Kinetics-400
└── MSRVTT_Videos/                # MSR-VTT
```

---

## 3. Evaluation (zero-shot)

General form:
```bash
bash scripts/evaluation/clip/zero_shot/mobileviclip_<tiny|small>/eval_<dataset>.sh
```

Note: each script pins a GPU via `export CUDA_VISIBLE_DEVICES=<id>` — edit it to match your machine.
Datasets available: `anet`, `didemo`, `hmdb51`, `k400`, `msrvtt`, `ssv2_mc`, `ucf101`.

### MobileViCLIP-Tiny
```bash
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_anet.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_didemo.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_k400.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_hmdb51.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_ucf101.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_ssv2_mc.sh
```

### MobileViCLIP-Small
```bash
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_msrvtt.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_anet.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_didemo.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_k400.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_hmdb51.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_ucf101.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_ssv2_mc.sh
```

### Run a single eval manually (bypass the .sh)
```bash
export PYTHONPATH=${PYTHONPATH}:.
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port=$((10000 + RANDOM % 20000)) \
    tasks_clip/retrieval.py \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/config_msrvtt.py
```

---

## 4. Training (pretraining)

**Preflight — run this before every pretraining launch, not just the first:**
```bash
python experiments/check_readiness.py            # add --decode to also try decoding 1 file/corpus
```

Cheap, and it catches the failure that costs the most time: a config pointing at
videos that are not on disk. Note the pilot and the full corpus use **different**
`data_root`s (`data/internvid_zh_10k/videos` vs `data/video_sample/InternVId-FLT_<N>`),
so a green smoke test on `pilot_10k_r0.py` does *not* mean a full-corpus config
will find its data. Skipping this yields a multi-thousand-line flood of
`No such file or directory` ending in `RecursionError` — see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#training-dies-with-recursionerror-maximum-recursion-depth-exceeded-on-one-rank).

General form:
```bash
bash scripts/pretraining/clip/mobileviclip_<tiny|small>/run.sh
```

```bash
bash scripts/pretraining/clip/mobileviclip_tiny/run.sh
bash scripts/pretraining/clip/mobileviclip_small/run.sh
```

Default config uses `NNODE=2`, `NUM_GPUS=8` (16 GPUs). Edit `run.sh` (`NNODE`, `NUM_GPUS`, `CUDA_VISIBLE_DEVICES`) for your setup.

### Single-node manual launch
```bash
export PYTHONPATH=${PYTHONPATH}:.
torchrun --nproc_per_node=8 --master_port=$((11000 + RANDOM % 20000)) \
    tasks_clip/pretrain.py \
    scripts/pretraining/clip/mobileviclip_tiny/config.py
```

---

## 5. Git workflow (this repo)

```bash
git status
git add -A
git commit -m "message"
git push
```

---

## 6. Live handoff with Claude (server ⇄ Claude via git)

The server has no Claude Code, so you "talk" to Claude through git — a turn-based loop.
Full guide: [notes/git-handoff-workflow.md](notes/git-handoff-workflow.md).

```bash
# server → Claude: paste your output/error/question into notes/live/from-server.md, then:
git add notes/live/from-server.md
git commit -m "handoff: server update $(date +%F_%H%M)"
git push origin <branch>
# ...then ping Claude in chat: "拉一下 from-server.md"

# read Claude's reply and act:
git pull origin <branch>
cat notes/live/from-claude.md      # newest block at the top
```

---

## Reference results (from upstream, MSR-VTT retrieval, tiny)

```
             txt_r1  txt_r5  txt_r10  img_r1  img_r5  img_r10  r_mean
ret_test/     38.2    65.4    74.8     38.7    64.9    74.2     59.37
```
