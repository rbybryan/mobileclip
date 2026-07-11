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

```bash
conda env create -f mobileviclip.yml
conda activate mobileviclip
```

If `mobileviclip.yml` fails to solve, create manually and install PyTorch matching your CUDA:
```bash
conda create -n mobileviclip python=3.10 -y
conda activate mobileviclip
# example: CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
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

## Reference results (from upstream, MSR-VTT retrieval, tiny)

```
             txt_r1  txt_r5  txt_r10  img_r1  img_r5  img_r10  r_mean
ret_test/     38.2    65.4    74.8     38.7    64.9    74.2     59.37
```
