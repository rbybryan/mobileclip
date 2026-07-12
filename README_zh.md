# MobileViCLIP 运行指南（中文）

本文档介绍如何在**一台全新的 Linux 机器**上，从零开始一步步把本仓库（MobileViCLIP 复现项目）跑起来，
完成**零样本视频-文本检索 / 视频分类评测（zero-shot evaluation）**，并可选地进行训练。

> 英文说明请见 [`README.md`](README.md)；命令速查见 [`COMMANDS.md`](COMMANDS.md)；权重/数据下载见 [`CHECKPOINTS.md`](CHECKPOINTS.md)。

---

## 0. 环境要求（先读这里！）

本项目是一个**基于 GPU 的深度学习研究项目**，不是纯 CPU 程序。开始前请确认：

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04 / 22.04） |
| **GPU** | **必须有 NVIDIA GPU**（评测至少 1 张，显存建议 ≥ 16 GB） |
| NVIDIA 驱动 | 已安装，`nvidia-smi` 能正常输出 |
| CUDA | 环境按 **CUDA 11.7 / PyTorch 1.13.1 / Python 3.10** 固定 |
| 磁盘 | 建议 ≥ 100 GB（视频数据集很大） |
| 网络 | 需能访问 GitHub、Google Drive、以及各数据集官网 |

> ⚠️ **重要**：代码在导入模型时会**无条件 `import flash_attn`**
> （`models/backbones/internvideo2/` 里被导入），因此即使 MobileViCLIP-tiny/small
> 用的是 MobileCLIP 主干，也**必须安装 `flash-attn`** 才能跑起来。`flash-attn` 需要
> 编译 CUDA 扩展，是整个安装过程中最容易出问题的一步，请重点关注第 4 步。

先验证 GPU：

```bash
nvidia-smi
```

能看到显卡信息即可继续；如果提示找不到命令，请先安装 NVIDIA 驱动。

---

## 1. 安装系统级依赖

```bash
sudo apt-get update
sudo apt-get install -y git wget curl build-essential ffmpeg
```

- `git`：拉取代码
- `wget` / `curl`：下载安装脚本
- `build-essential`：编译 `flash-attn` 等 CUDA 扩展所需（提供 gcc/g++/make）
- `ffmpeg`：视频解码（`decord` / `av` 依赖）

---

## 2. 安装 Miniconda（如果还没有 conda）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda --version   # 验证安装成功
```

> 想让 conda 每次开终端自动可用，可运行 `conda init bash` 后重开终端。

---

## 3. 获取代码

```bash
git clone https://github.com/rbybryan/mobileclip.git
cd mobileclip
```

之后所有命令默认都在仓库根目录 `mobileclip/` 下执行。

---

## 4. 创建并激活 Python 环境

### 4.1 方式一：直接用 conda 配置文件（首选）

```bash
conda env create -f mobileviclip.yml
```

> 📌 **注意环境名字**：`mobileviclip.yml` 里的 `name:` 实际是 **`internvideo`**，
> 所以上面命令创建的环境叫 `internvideo`。请用：
>
> ```bash
> conda activate internvideo
> ```
>
> 如果你希望环境名就叫 `mobileviclip`，改用：
>
> ```bash
> conda env create -f mobileviclip.yml -n mobileviclip
> conda activate mobileviclip
> ```

### 4.2 方式二：手动创建（当 4.1 求解失败 / 显卡 CUDA 版本不同时）

`mobileviclip.yml` 把 CUDA、`flash-attn`、`apex` 等都锁死到特定版本，在新机器上可能求解失败。
这时手动装更稳妥：

```bash
# 1) 基础环境
conda create -n mobileviclip python=3.10 -y
conda activate mobileviclip

# 2) 安装与本机 CUDA 匹配的 PyTorch（这里以 CUDA 11.7 为例，对应仓库锁定版本）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu117

# 3) 安装常规 Python 依赖
pip install einops timm==0.5.4 transformers==4.28.1 tokenizers==0.13.3 \
    decord av opencv-python ftfy regex pandas numpy==1.26.4 scipy \
    tensorboard wandb pyyaml easydict deepspeed==0.10.1 open_clip_torch \
    openai-clip librosa
```

> 若某个包版本装不上，可放宽版本号；核心是 `torch / einops / timm / transformers / decord`。

### 4.3 安装 flash-attn（关键、也最容易失败的一步）

代码会无条件导入 `flash_attn`，因此**必须安装**。它需要编译 CUDA 扩展，耗时较长（可能 20–40 分钟）：

```bash
# 确保已激活环境，且 nvcc/CUDA 与 PyTorch 版本匹配
pip install ninja packaging
pip install flash-attn==2.0.8 --no-build-isolation
```

如果编译失败，常见原因与排查见文末[常见问题](#7-常见问题与排错)。

### 4.4 关于 apex（可选）

`apex` 只用于 `FusedAdam` 等融合优化器，代码里已做 `try/except` 兜底（`utils/optimizer.py`），
**不装也能跑**。默认优化器用普通 AdamW，无需 apex。如需安装可参考
[NVIDIA/apex](https://github.com/NVIDIA/apex)。

### 4.5 验证环境

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn ok')"
```

两条都不报错、且 `cuda` 为 `True`，说明环境基本就绪。

---

## 5. 下载预训练权重与标注文件

权重与标注托管在 Google Drive，用 `gdown` 下载：

```bash
pip install gdown

# 5.1 预训练权重 -> checkpoints/
mkdir -p checkpoints
gdown --folder https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7 -O checkpoints

# 5.2 标注文件 -> anno_downstream/
mkdir -p anno_downstream
gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream
```

下载完成后，`checkpoints/` 应包含：

```
checkpoints/
├── mobileclip_s0.pt       # MobileCLIP 主干（tiny 用）
├── mobileclip_s2.pt       # MobileCLIP 主干（small 用）
├── mobileviclip_tiny.pt   # MobileViCLIP 视频-文本权重
└── mobileviclip_small.pt
```

> 这些目录都在 `.gitignore` 里，不会被提交。

---

## 6. 下载视频数据集

视频文件很大，需要从各数据集**官网**自行下载，放到 `data/` 下。目录名要与
`configs/data.py` 里配置的一致：

```
data/
├── MSRVTT_Videos/                # MSR-VTT（检索）
├── anet_1.3_video_val_resize/    # ActivityNet（检索）
├── kinetics_400_val_10s_320p/    # Kinetics-400（分类）
├── didemo/processed_videos/test/ # DiDeMo（检索）
├── ucf101/videos/                # UCF101（分类）
├── hmdb51/videos_test/           # HMDB51（分类）
└── ssv2/20bn-something-something-v2/  # SSv2-MC
```

> 建议先只下载 **MSR-VTT**（体量最小），把评测流程完整跑通一次，再按需下载其它数据集。
> 每个数据集对应的标注 JSON 已在第 5 步随 `anno_downstream/` 一起下载好。

各数据集对应关系可在 `configs/data.py` 中查到（`data_root` 字段）。

---

## 7. 运行零样本评测

### 7.1 先改 GPU 编号（很重要）

每个评测脚本里都写死了 `export CUDA_VISIBLE_DEVICES=3`。如果你的机器不是 4 卡、
或想用别的卡，**必须改成本机存在的编号**（比如单卡机器改成 `0`）：

```bash
# 例：把 tiny 的 msrvtt 脚本里的 GPU 改成 0
sed -i 's/CUDA_VISIBLE_DEVICES=3/CUDA_VISIBLE_DEVICES=0/' \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh
```

（也可以直接用编辑器打开脚本手动改。）

### 7.2 运行（以 MobileViCLIP-Tiny + MSR-VTT 为例）

```bash
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh
```

脚本会自动设置 `PYTHONPATH`、用 `torchrun` 启动 `tasks_clip/retrieval.py`，
并加载同目录下的 `config_msrvtt.py`。

### 7.3 其它数据集 / 模型

通用形式：

```bash
bash scripts/evaluation/clip/zero_shot/mobileviclip_<tiny|small>/eval_<数据集>.sh
```

可用数据集：`msrvtt`、`anet`、`didemo`、`k400`、`hmdb51`、`ucf101`、`ssv2_mc`。例如：

```bash
# Tiny
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_anet.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_k400.sh

# Small
bash scripts/evaluation/clip/zero_shot/mobileviclip_small/eval_msrvtt.sh
```

### 7.4 不用 .sh，手动单卡启动

```bash
export PYTHONPATH=${PYTHONPATH}:.
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port=$((10000 + RANDOM % 20000)) \
    tasks_clip/retrieval.py \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/config_msrvtt.py
```

### 7.5 参考结果（MSR-VTT 检索，Tiny）

```
             txt_r1  txt_r5  txt_r10  img_r1  img_r5  img_r10  r_mean
ret_test/     38.2    65.4    74.8     38.7    64.9    74.2     59.37
```

数值接近上表即说明复现成功。

---

## 8.（可选）训练 / 预训练

训练需要 InternVid 视频数据（体量非常大）和多卡环境，请谨慎评估资源。

通用形式：

```bash
bash scripts/pretraining/clip/mobileviclip_<tiny|small>/run.sh
```

> 默认配置是 `NNODE=2`、`NUM_GPUS=8`（共 16 卡）。单机请编辑对应 `run.sh` 里的
> `NNODE`、`NUM_GPUS`、`CUDA_VISIBLE_DEVICES` 再运行。

单机手动启动示例：

```bash
export PYTHONPATH=${PYTHONPATH}:.
torchrun --nproc_per_node=8 --master_port=$((11000 + RANDOM % 20000)) \
    tasks_clip/pretrain.py \
    scripts/pretraining/clip/mobileviclip_tiny/config.py
```

---

## 9. 常见问题与排错

**Q1. `import flash_attn` 报错 / 评测一启动就 `ModuleNotFoundError: flash_attn`**
代码无条件依赖 `flash-attn`，必须装（见 4.3）。若编译失败：
- 确认 `nvcc --version` 的 CUDA 版本与 PyTorch 的 CUDA 版本一致；
- 先 `pip install ninja packaging`，再用 `--no-build-isolation` 安装；
- 显存/内存不足会导致编译被 kill，可减少并行：`MAX_JOBS=4 pip install flash-attn==2.0.8 --no-build-isolation`。

**Q2. `conda activate mobileviclip` 提示环境不存在**
`mobileviclip.yml` 里的环境名其实是 `internvideo`。请 `conda activate internvideo`，
或在创建时用 `-n mobileviclip` 指定名字（见 4.1）。

**Q3. `RuntimeError: CUDA error: invalid device ordinal` / 找不到 GPU**
脚本默认 `CUDA_VISIBLE_DEVICES=3`。单卡机器请改成 `0`（见 7.1）。

**Q4. `torch.cuda.is_available()` 返回 False**
说明 PyTorch 没装成 CUDA 版，或驱动/CUDA 不匹配。按 4.2 用对应 `+cuXXX` 的 wheel 重装。

**Q5. 读取视频报错 / `decord` 相关错误**
确认已 `apt-get install ffmpeg`，且视频确实放在 `configs/data.py` 里对应的 `data_root` 路径下。

**Q6. `gdown` 下载 Google Drive 文件夹失败或限速**
可多试几次，或升级 `pip install -U gdown`；大文件夹可分文件下载。

**Q7. 显存不够（OOM）**
调小对应 `config_*.py` 里的 `batch_size_test`（例如从 32 改到 8）。

---

## 10. 快速上手清单（TL;DR）

```bash
# 1. 系统依赖
sudo apt-get update && sudo apt-get install -y git wget build-essential ffmpeg

# 2. conda（若无）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3 && source $HOME/miniconda3/etc/profile.d/conda.sh

# 3. 代码
git clone https://github.com/rbybryan/mobileclip.git && cd mobileclip

# 4. 环境
conda env create -f mobileviclip.yml && conda activate internvideo
pip install ninja packaging && pip install flash-attn==2.0.8 --no-build-isolation

# 5. 权重 + 标注
pip install gdown
mkdir -p checkpoints anno_downstream
gdown --folder https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7 -O checkpoints
gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream

# 6. 下载 MSR-VTT 视频到 data/MSRVTT_Videos/（见第 6 步）

# 7. 改 GPU 编号后评测
sed -i 's/CUDA_VISIBLE_DEVICES=3/CUDA_VISIBLE_DEVICES=0/' \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh
```

---

## 参考

- 论文：[arXiv:2508.07312](https://arxiv.org/abs/2508.07312)（ICCV 2025）
- 上游代码：[MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)
- MobileCLIP 主干：[apple/ml-mobileclip](https://github.com/apple/ml-mobileclip)
