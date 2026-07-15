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
| **GPU** | **必须有 NVIDIA GPU**（评测 1 张即可，显存建议 ≥ 16 GB；已在 8×V100-32GB 上验证） |
| GPU 架构 | 任意 CUDA GPU 均可评测；`flash-attn` 仅 Ampere(A100/30/40 系)及以上支持，V100 / 2080 等非 Ampere 显卡**跳过它即可**（见下方说明与 4.3） |
| NVIDIA 驱动 | 已安装，`nvidia-smi` 能正常输出 |
| CUDA | 环境按 **CUDA 11.7 / PyTorch 1.13.1 / Python 3.10** 固定 |
| 磁盘 | 建议 ≥ 100 GB（视频数据集很大） |
| 网络 | 需能访问 GitHub、Google Drive、以及各数据集官网 |

> ℹ️ **关于 flash-attn（重要）**：MobileViCLIP tiny/small 用的是 MobileCLIP(FastViT)主干，
> 注意力为纯 PyTorch 实现，**运行时并不调用 `flash_attn`**。本仓库已把相关导入改为可选
> （`try/except` 兜底），因此**评测和训练都无需安装 flash-attn**。
> 只有当你要用 InternVideo2 主干、且显卡是 **Ampere(A100/30/40 系)及以上**时才需要它——
> V100 / Turing 等非 Ampere 显卡装不了、也用不到，直接跳过（见 4.3）。

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

### 4.0 （中国大陆用户先看）配置国内镜像

如果你在中国大陆，直连 conda 官方源 `repo.anaconda.com` 常常失败，典型报错长这样：

```
Retrying ... after connection broken by
'OSError('Tunnel connection failed: 302 Moved Temporarily')': /pkgs/main/terms.json
```

这是**网络问题**（conda 拉包 + 拉「服务条款 terms.json」文件失败），不是项目本身的问题。
把 conda / pip 指向**清华镜像**即可解决：

```bash
# 1) conda 走清华镜像
cat > ~/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
conda clean -i    # 清掉旧源缓存，强制用新镜像

# 2) 跳过 Anaconda 服务条款(ToS)交互检查（新版 conda 需要，否则会去拉 terms.json）
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
echo 'export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true' >> ~/.bashrc

# 3) pip 也走清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

配置完成后再执行下面的 4.1 / 4.2。

> ⚠️ 注意：`mobileviclip.yml` 把 build 号全锁死了，即便配好镜像也可能解不动，
> 这种情况**优先用 [4.2 手动方案](#42-方式二手动创建当-41-求解失败--显卡-cuda-版本不同时)** 更稳。
>
> 小贴士：Miniconda 安装脚本本身也可从清华镜像下载：
> `https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh`

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

### 4.3 flash-attn（可选 —— V100 / 非 Ampere 显卡请跳过）

**大多数情况下无需安装 flash-attn。** 本仓库已把 `flash_attn` 的导入改成可选（`try/except`
兜底），而 MobileViCLIP tiny/small 用 MobileCLIP 主干、运行时不调用它，所以**评测和训练在
没有 flash-attn 时也能跑通**。

- **V100 / T4 / 2080 等非 Ampere 显卡**：`flash-attn 2.x` 只支持 Ampere 及以上，在这些卡上
  **装不了也用不到，直接跳过本步**。
- **A100 / 30 / 40 / H100 等 Ampere+ 显卡，且要用 InternVideo2 主干**：才需要安装。它要编译
  CUDA 扩展、耗时较长（20–40 分钟），并需本机有匹配的 CUDA 工具链（nvcc）：

```bash
# 仅 Ampere+ 且确有需要时执行
pip install ninja packaging
pip install flash-attn==2.0.8 --no-build-isolation
```

如果编译失败，常见原因与排查见文末[常见问题](#10-常见问题与排错)。

### 4.4 关于 apex（可选）

`apex` 只用于 `FusedAdam` 等融合优化器，代码里已做 `try/except` 兜底（`utils/optimizer.py`），
**不装也能跑**。默认优化器用普通 AdamW，无需 apex。如需安装可参考
[NVIDIA/apex](https://github.com/NVIDIA/apex)。

### 4.5 验证环境

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

只要 `cuda` 为 `True` 即可。**不需要** `import flash_attn` 成功——本仓库已让它成为可选依赖，
V100 等非 Ampere 显卡不装 flash-attn 也能正常评测/训练。

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

> ⚠️ **标注文件夹（5.2）常常没有下载权限**：`gdown --folder` 会报
> `Cannot retrieve the folder information from the link` / `Permission denied`。
> 这个 `anno_downstream/` 是**数据**、不是仓库内容，作者的 Drive 需要审批，**很可能拿不到**。
> 好消息是**这些标注都可以自己生成**：
> - **MSR-VTT** → 见 [6.1](#61-msr-vtt检索最小推荐先跑) + `preprocess/build_msrvtt_1k_test.py`
> - **UCF101 / HMDB51 / Kinetics-400（分类）** → 见 [6.2–6.4](#62-64-分类数据集ucf101--hmdb51--kinetics-400组织数据并测试) + `preprocess/build_act_rec_anno.py`
>
> 即使权重那步（5.1）能下、标注这步（5.2）拿不到，也**不影响**——按第 6 步各数据集的「生成标注」小节自建即可。
> 若你确实有 Drive 权限，`gdown` 报错多为版本旧/限流，见 [常见问题 Q6.1](#10-常见问题与排错)。

---

## 6. 下载视频数据集

视频文件很大，需要从各数据集**官网**自行下载，放到 `data/` 下。目录名要与
`configs/data.py` 里配置的 `data_root` 一致：

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

> 建议先只下载 **MSR-VTT**（体量最小、且有直链），把评测流程完整跑通一次，再按需下载其它数据集。
>
> ⚠️ **标注 JSON 需自建**：第 5.2 步的 Drive `anno_downstream/` 往往**没有下载权限**（Permission denied）。
> 因此下面每个数据集除了下**视频**，还会给出**用公开划分文件自己生成标注 JSON** 的步骤
> （MSR-VTT 用 `preprocess/build_msrvtt_1k_test.py`，分类三件套用 `preprocess/build_act_rec_anno.py`）。
> 生成脚本会把标注**按你磁盘上实际存在的视频**过滤——缺失的视频会被丢弃，避免评测中途因找不到文件而崩溃。

> ⚠️ **诚实提醒**：除 MSR-VTT / UCF101 / HMDB51 有公开直链外，**ActivityNet、Kinetics-400、
> DiDeMo、SSv2 都需要申请表单、走 YouTube 抓取脚本，或注册后才能下载**，无法用一条 `wget` 搞定。
> 下面给出每个数据集的**官方入口 + 准确命令**；带 `⚠️` 的步骤表示需要人工申请/额外预处理。
> 仓库用的是**验证/测试集**（如 `*_val_*` / `*_test*`），一般无需下载庞大的训练集。

### 6.1 MSR-VTT（检索，最小，推荐先跑）

用 Frozen-in-Time 的公开镜像（约 6.7 GB，含全部 10000 个视频）：

```bash
mkdir -p data
cd data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip MSRVTT.zip
# 解压后视频在 MSRVTT/videos/all/videoXXXX.mp4，重命名/软链成仓库要求的目录：
ln -s "$(pwd)/MSRVTT/videos/all" "$(pwd)/MSRVTT_Videos"
cd ..
# 验证：应能看到 video0.mp4 ... video9999.mp4
ls data/MSRVTT_Videos/ | head
```

> 标注 `anno_downstream/msrvtt_1k_test.json` 用的是 1K-A 测试划分，视频文件名形如 `videoXXXX.mp4`，
> 与上面解压出来的命名一致。若 `ln -s` 不方便也可 `mv MSRVTT/videos/all MSRVTT_Videos`。

---

## 6.2–6.4 分类数据集（UCF101 / HMDB51 / Kinetics-400）：组织数据并测试

这三个是**零样本动作识别（分类）**任务，走 `is_act_rec=True` 流程，与检索略有不同。
下面先统一说明「数据怎么组织、怎么验证」，再分别给出 **下载 → 验证 → 运行 → 参考结果** 的完整步骤。

> 📌 **分类任务的数据模型（三者通用，先读）**
> - 标注 JSON 是一个列表，每条形如 `{"video": "<文件名>", "caption": "<类别名>"}`，其中
>   `caption` 就是**类别名**（如 `playing basketball`）。评测时会用 `kinetics_templates`
>   把每个类别名套成多条提示词（"a video of a person doing {}." 等），与视频特征逐一比对，
>   取最相似的类别作为预测——所以这是「视频 → 类别」的分类，而非「视频 ↔ 句子」的检索。
> - ⚠️ **标注需自建**：作者 Drive 里的 `anno_downstream/*.json` 往往**没权限**（见第 5.2 步）。
>   下面每个数据集都给出用 `preprocess/build_act_rec_anno.py` **从公开划分文件生成**标注的步骤——
>   和 6.1 里 `build_msrvtt_1k_test.py` 是一样的思路。
> - 视频按 `os.path.join(data_root, video)` 读取；**缺失文件会让评测直接崩溃**（不是跳过）。
>   所以生成脚本带 `--videos <data_root>` 时会**只保留磁盘上真实存在的视频**，缺的丢弃（K400 因 YouTube 失效抓不全时尤其有用）。
> - 三者的对应关系（`data_root` / 标注 / 脚本 / 脚本内写死的 GPU 编号）：
>
> | 数据集 | `data_root`（视频放这里） | 标注 JSON | 评测脚本 | 脚本默认 GPU |
> |---|---|---|---|---|
> | UCF101 | `./data/ucf101/videos` | `anno_downstream/ucf101_act_test.json` | `eval_ucf101.sh` | `=4` |
> | HMDB51 | `./data/hmdb51/videos_test` | `anno_downstream/hmdb51_act_test.json` | `eval_hmdb51.sh` | `=6` |
> | Kinetics-400 | `./data/kinetics_400_val_10s_320p` | `anno_downstream/kinetics400_validate.json` | `eval_k400.sh` | `=2` |
>
> **类别名文本的说明（可能带来与论文的小幅差异）**：脚本用标准可读类名作 `caption`——UCF101 把驼峰
> `ApplyEyeMakeup` 拆成 `apply eye makeup`，HMDB51 把 `brush_hair` 下划线换空格成 `brush hair`，
> K400 直接用官方 label。评测里 `pre_text` 会统一小写去标点，所以大小写无所谓；但若作者对某类的措辞不同，
> 那一类的准确率可能略有出入（与 6.1 里 MSR-VTT 的 caption-index 同理）。

> 💡 **通用验证脚本**：视频放好后，用下面这段确认「标注里的文件名能对应到真实文件」，
> 避免跑到一半才发现路径/命名不符（同样适用于第 7 步之前的自检）：
>
> ```bash
> python - <<'PY'
> import json, os
> checks = [("anno_downstream/ucf101_act_test.json",     "./data/ucf101/videos"),
>           ("anno_downstream/hmdb51_act_test.json",     "./data/hmdb51/videos_test"),
>           ("anno_downstream/kinetics400_validate.json","./data/kinetics_400_val_10s_320p")]
> for anno, root in checks:
>     if not os.path.exists(anno):
>         print(f"[skip] 标注未下载: {anno}"); continue
>     d = json.load(open(anno))
>     miss = [e["video"] for e in d[:300] if not os.path.exists(os.path.join(root, e["video"]))]
>     print(f"{anno}: 共 {len(d)} 条, 前300条缺失 {len(miss)} 个; 首条 -> {os.path.join(root, d[0]['video'])}")
>     if miss: print("   例如缺失:", miss[:3])
> PY
> ```
> 缺失数为 0 说明命名对得上，可以直接评测；若大量缺失，多半是**文件名/扩展名不一致**
> （如标注写 `.mp4` 而你的是 `.avi`，或标注含子目录前缀），按下面各数据集的说明调整平铺方式或补软链。

### 6.2 UCF101（分类，有直链）

**① 下载并把视频平铺到 `data/ucf101/videos/`**（官网 [CRCV/UCF101](https://www.crcv.ucf.edu/data/UCF101.php) 提供整包 `UCF101.rar`，约 6.5 GB）：

```bash
mkdir -p data/ucf101
cd data/ucf101
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
sudo apt-get install -y unrar   # 若无 unrar
unrar x UCF101.rar              # 解压出 UCF-101/<类别>/*.avi
# 仓库期望所有视频平铺在 data/ucf101/videos/ 下：
mkdir -p videos
find UCF-101 -name '*.avi' -exec mv {} videos/ \;
cd ../..
ls data/ucf101/videos/ | head   # 形如 v_ApplyEyeMakeup_g01_c01.avi
```

**② 生成标注 `ucf101_act_test.json`**（Drive 拿不到，用官方划分文件自建）。UCF101 的官方
train/test 划分是 `UCF101TrainTestSplits-RecognitionTask.zip`（CRCV 官网，很小）：

```bash
cd data/ucf101
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip -q UCF101TrainTestSplits-RecognitionTask.zip     # 解出 ucfTrainTestlist/
cd ../..
# 用 testlist01（split 1）生成标注，并按磁盘实际视频过滤：
python preprocess/build_act_rec_anno.py --dataset ucf101 \
    --src data/ucf101/ucfTrainTestlist --videos data/ucf101/videos \
    --out anno_downstream/ucf101_act_test.json
```

> 拿不到划分 zip 时的兜底：直接用**按类别分文件夹**的原始解压目录（`UCF-101/<类别>/*.avi`）生成——
> 但这会用**全部视频**（train+test），数量和参考值不同：
> `python preprocess/build_act_rec_anno.py --dataset ucf101 --from-dir data/ucf101/UCF-101 --videos data/ucf101/videos --out anno_downstream/ucf101_act_test.json`

**③ 验证**：跑上面的「通用验证脚本」，确认 `ucf101_act_test.json` 的文件名 0 缺失。
UCF101 标注里的 `video` 就是**平铺后的裸文件名**（不带类别子目录），与上面 `find ... -exec mv` 的结果一致。

**④ 运行评测**（脚本默认 `CUDA_VISIBLE_DEVICES=4`，先改成本机 GPU）：

```bash
sed -i 's/CUDA_VISIBLE_DEVICES=4/CUDA_VISIBLE_DEVICES=0/' \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_ucf101.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_ucf101.sh
```

**⑤ 参考结果（Tiny）**：`act_val/` 行的 **`img_r1` 即 Top-1 分类准确率**（视频→类别），
`img_r5` 即 Top-5：

```
              txt_r1  txt_r5  txt_r10   img_r1  img_r5  img_r10
act_val/       75.44   95.35    98.60    83.17   96.04    99.01
```

### 6.3 HMDB51（分类）

> ⚠️ **官方直链已失效**：Brown 大学的老地址
> `http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar`
> 现在会 301 跳转到 `https://serre.lab.brown.edu` 主页，`wget` 只会下到一个约 5 KB 的
> **HTML 页面**，`unrar` 报 `hmdb51_org.rar is not RAR archive`。整个 `serre-lab.clps.brown.edu`
> 域名已下线，重试无用。改用下面的 **HuggingFace 镜像**（内容一致：6766 个 `.avi`，约 2.12 GB）。

**① 下载并把视频平铺到 `data/hmdb51/videos_test/`（推荐用脚本，一步到位）**：

> `jili5044/hmdb51` 其实是**一个** Xet-backed LFS 文件 `hmdb51.zip`（约 2.12 GB）+ 一个
> README，**不是**散装 `.avi`，也不是按类别打包的 `.rar`。下面的脚本已封装「断点续传下载 →
> 校验字节数 → 解压 → 软链平铺 → 对标注 JSON 核对」全流程，并绕开了 hf-mirror 对 Xet 文件
> 的元数据坑（见下方 ⚠️）：

```bash
# 清掉可能下到的假文件（其实是 HTML）
rm -f data/hmdb51/hmdb51_org.rar

# 默认走 hf-mirror；官方源：HF_ENDPOINT=https://huggingface.co bash tools/download_hmdb51.sh
bash tools/download_hmdb51.sh

ls data/hmdb51/videos_test/ | head        # 形如 *.avi
ls data/hmdb51/videos_test/ | wc -l       # 目标 ~6766
```

> ⚠️ **`huggingface-cli download` 直接报错怎么办**：用镜像（`export HF_ENDPOINT=https://hf-mirror.com`）
> 跑 `huggingface-cli download jili5044/hmdb51 ...` 时，常见如下报错并**一个字节都下不下来**：
>
> ```
> huggingface_hub.errors.FileMetadataError: Distant resource does not seem to be on huggingface.co ...
> huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub ...
> ```
>
> 原因：`hmdb51.zip` 用 **Xet** 存储，hf-mirror 的 HEAD 响应缺 `x-linked-etag` 头，
> `huggingface_hub` 判定 `etag is None` 就在下载**之前**直接放弃（其实文件字节完全可达）。
> 三种解法（脚本用的是第 1 种，最稳）：
>
> 1. **直接 `wget` 拉那个 zip**（绕过 Python 客户端的元数据校验，支持断点续传）：
>    ```bash
>    export HF_ENDPOINT=https://hf-mirror.com
>    mkdir -p data/hmdb51_hf
>    wget -c "$HF_ENDPOINT/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip" \
>         -O data/hmdb51_hf/hmdb51.zip
>    unzip -n -q data/hmdb51_hf/hmdb51.zip -d data/hmdb51_hf
>    mkdir -p data/hmdb51/videos_test
>    find data/hmdb51_hf -name '*.avi' -exec ln -sf {} data/hmdb51/videos_test/ \;
>    ```
> 2. **关掉 Xet** 再用 CLI（新版 `huggingface_hub` 支持）：
>    ```bash
>    export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
>    huggingface-cli download jili5044/hmdb51 --repo-type dataset --local-dir data/hmdb51_hf
>    ```
> 3. **换官方源**（网络允许时最省事）：`unset HF_ENDPOINT` 后再跑 CLI。
>
> 上面 `tools/download_hmdb51.sh` 已内置解法 1（含字节数校验、解压、平铺、核对标注），无脑跑即可。

> HMDB51 的官方 test 划分由 `anno_downstream/hmdb51_act_test.json` 指定；上面把全部 `.avi`
> 平铺进 `videos_test/`，评测时只按标注 JSON 里的文件名读取（多出来的训练视频不会被用到）。
> 只要有 HMDB51 的 `.avi` 视频，**任何来源都可以**（Kaggle、其它 HF 镜像等），关键是文件名与标注对齐。

**② 生成标注 `hmdb51_act_test.json`**（Drive 拿不到，自建）。HMDB51 的官方划分文件
`test_train_splits.rar` 也托管在**已下线的 Brown 域名**上，多半下不到，所以**首选兜底方案**：
直接用上一步下下来的**按类别分文件夹**的视频目录生成（`build_act_rec_anno.py` 的 `--from-dir`）：

```bash
# 先确认 HF 下载下来的是按类别分的文件夹（<dir>/<class>/<clip>.avi）：
find data/hmdb51_hf -maxdepth 2 -type d | head
# 用类别文件夹名当类别、每个 .avi 生成一条；--videos 对齐 videos_test/ 里的实际文件
python preprocess/build_act_rec_anno.py --dataset hmdb51 \
    --from-dir data/hmdb51_hf --videos data/hmdb51/videos_test \
    --out anno_downstream/hmdb51_act_test.json
```

> ⚠️ `--from-dir` 用的是**全部视频**（train+test，约 6766 条），而论文参考行 `act_val/` 用的是
> **split-1 测试集**（约 1530 条）。因此 Top-1 会与下表有出入，但流程完全正确、可自洽验证。
> 若你能弄到 `testTrainMulti_7030_splits/`（如 Kaggle / 其它镜像），换成精确的 split-1 测试集：
> `python preprocess/build_act_rec_anno.py --dataset hmdb51 --src <testTrainMulti_7030_splits> --videos data/hmdb51/videos_test --out anno_downstream/hmdb51_act_test.json`

**③ 验证**：跑「通用验证脚本」（或下面单独针对 HMDB51 的一段）确认 `hmdb51_act_test.json` 0 缺失。
若大量缺失，多为**扩展名/命名不一致**（标注是裸文件名 `*.avi`，镜像可能带类别子目录）——用 `find ... -exec ln -sf`
把所有 `.avi` 平铺到 `videos_test/` 即可对齐：
>
> ```bash
> python - <<'PY'
> import json, os
> root = "./data/hmdb51/videos_test"
> d = json.load(open("anno_downstream/hmdb51_act_test.json"))
> miss = [e["video"] for e in d if not os.path.exists(os.path.join(root, e["video"]))]
> print(f"共 {len(d)} 条, 缺失 {len(miss)}; 首条 -> {os.path.join(root, d[0]['video'])}")
> if miss: print("例如缺失:", miss[:3])
> PY
> ```

**④ 运行评测**（脚本默认 `CUDA_VISIBLE_DEVICES=6`）：

```bash
sed -i 's/CUDA_VISIBLE_DEVICES=6/CUDA_VISIBLE_DEVICES=0/' \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_hmdb51.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_hmdb51.sh
```

**⑤ 参考结果（Tiny，split-1 测试集）**（`img_r1` = Top-1）：

```
              txt_r1  txt_r5  txt_r10   img_r1  img_r5  img_r10
act_val/       48.76   77.71    88.17    62.75   74.51    84.31
```

### 6.4 Kinetics-400（分类）⚠️ 走官方下载脚本

**① 下载验证集并转成 10s / 320p**。仓库用的是 **10 秒、320p 的验证集**
（`data/kinetics_400_val_10s_320p`）。用 CVDF 官方仓库脚本下载验证集，再用 ffmpeg 裁到 10s、缩到 320p：

```bash
# 1) 克隆官方下载器
git clone https://github.com/cvdfoundation/kinetics-dataset.git
cd kinetics-dataset

# 2) 只下 K400 验证集（脚本用 wget -c 支持断点续传）
bash ./k400_downloader.sh          # 下载 tar.gz（可只保留 val 部分）
bash ./k400_extractor.sh           # 解压出 k400/val/<类别>/*.mp4
cd ..

# 3) 缩放到 320p 短边并裁到 10s（示例，用 ffmpeg 批处理）
mkdir -p data/kinetics_400_val_10s_320p
for v in kinetics-dataset/k400/val/*/*.mp4; do
  ffmpeg -y -i "$v" -t 10 -vf "scale=-2:320" \
    "data/kinetics_400_val_10s_320p/$(basename "$v")"
done
ls data/kinetics_400_val_10s_320p/ | head
```

> ⚠️ Kinetics 视频来自 YouTube，部分链接会失效；CVDF 仓库提供了替换视频。
> 文件名（YouTube id）需与 `anno_downstream/kinetics400_validate.json` 对得上。

**② 生成标注 `kinetics400_validate.json`**（Drive 拿不到，自建）。K400 的官方验证集标注是一个 CSV
（列 `label,youtube_id,time_start,time_end`），CVDF 下载器里就带（`kinetics-dataset/k400/annotations/val.csv`），
也可从官方标注包获取。用它生成，并**按磁盘实际视频过滤**（失效抓不到的自动丢弃，避免评测崩溃）：

```bash
python preprocess/build_act_rec_anno.py --dataset k400 \
    --src kinetics-dataset/k400/annotations/val.csv \
    --videos data/kinetics_400_val_10s_320p \
    --out anno_downstream/kinetics400_validate.json
```

> 脚本按 CVDF 命名 `<ytid>_<start:06d>_<end:06d>.mp4` 匹配；若你的文件名只是 `<ytid>.mp4` 或别的形式，
> `--videos` 会**按 YouTube id 前缀**自动对齐，无需手动改名。CSV 路径按你的实际位置调整。

**③ 验证**：跑「通用验证脚本」看 `kinetics400_validate.json` 的缺失比例。因为第 ② 步已按磁盘过滤，
这里应为 0 缺失。K400 部分视频因 YouTube 失效抓不全属正常——**评测集会相应变小**，准确率因样本缺失与参考值有出入，但流程正确。

**④ 运行评测**（脚本默认 `CUDA_VISIBLE_DEVICES=2`）：

```bash
sed -i 's/CUDA_VISIBLE_DEVICES=2/CUDA_VISIBLE_DEVICES=0/' \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_k400.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_k400.sh
```

**⑤ 参考结果（Tiny，完整验证集）**（`img_r1` = Top-1）：

```
              txt_r1  txt_r5  txt_r10   img_r1  img_r5  img_r10
act_val/       56.25   81.89    88.75    80.75   95.00    98.00
```

> 💡 **三者小结**：分类任务看 `act_val/` 行的 **`img_r1`（Top-1）/ `img_r5`（Top-5）**。数值接近上表
> 即说明数据组织正确、复现成功；换 Small 模型把脚本路径里的 `mobileviclip_tiny` 换成 `mobileviclip_small` 即可。
> 若某数据集视频抓不全，Top-1 会相应偏低，属数据缺失而非流程错误。

### 6.5 DiDeMo（检索）⚠️ 从 YouTube 抓取

DiDeMo 不直接发布视频，需按 [LisaAnne/LocalizingMoments](https://github.com/LisaAnne/LocalizingMoments)
提供的清单从 YouTube 抓取，再做 30 秒截断（`trimmed30`）：

```bash
git clone https://github.com/LisaAnne/LocalizingMoments.git
cd LocalizingMoments
# 按其 README 用 yt-dlp 按 data/*_data.json 里的 YouTube id 抓取原视频，
# 然后转码/截断为 test 划分，放到仓库要求的路径：
#   data/didemo/processed_videos/test/<video>.mp4
cd ..
mkdir -p data/didemo/processed_videos/test
# （把处理好的 test 视频放入上面目录）
```

> `configs/data.py` 里 `didemo_ret_test` 设了 `trimmed30=True`，即只用每个视频前 30 秒；
> 文件名需与 `anno_downstream/didemo_ret_test.json` 对应。

### 6.6 ActivityNet（检索）⚠️ 需申请

ActivityNet 视频版权在 YouTube，需在 [activity-net.org/download.html](http://activity-net.org/download.html)
**填写申请表**获取下载脚本/直链。仓库用的是缩放后的验证集 `anet_1.3_video_val_resize`：

```bash
# 1) 按官网表单拿到 v1.3 验证集视频后，放到：
mkdir -p data/anet_1.3_video_val_resize
# 2) 缩放（官方评测常用短边 resize；文件名形如 v_XXXX.mp4 / .avi）
#    示例：短边缩到 256
# for v in <原始验证集>/*; do
#   ffmpeg -y -i "$v" -vf "scale=-2:256" "data/anet_1.3_video_val_resize/$(basename "$v")"
# done
```

> 文件名需与 `anno_downstream/anet_ret_val.json` 里的一致（`is_paragraph_retrieval=True`，段落级检索）。

### 6.7 SSv2-MC（Something-Something V2）⚠️ 需注册

SSv2 由 Qualcomm 托管，需在 [Qualcomm AI Datasets](https://www.qualcomm.com/developer/software/something-something-v2-dataset)
注册并同意研究许可后下载（分成 19 个分卷 + 标签文件）：

```bash
# 注册后从 Qualcomm 门户下载 20bn-something-something-v2-??.zip 与 labels，
# 合并解压到仓库要求的目录：
mkdir -p data/ssv2/20bn-something-something-v2
cat 20bn-something-something-v2-??.zip > ssv2.zip  # 分卷合并
unzip ssv2.zip -d data/ssv2/                        # 解出 *.webm
# 视频为 .webm，文件名是数字 id，与 anno_downstream/ssv2_mc_val.json 对应
ls data/ssv2/20bn-something-something-v2/ | head
```

各数据集与标注、`data_root` 的对应关系都可在 `configs/data.py` 中核对。

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

> 💡 **多卡 / 共享机器**：评测单张 V100-32GB 就够。先 `nvidia-smi` 看哪几张卡空闲（显存占用
> 接近 0），把 `CUDA_VISIBLE_DEVICES` 设成一张**空闲**的卡号，避免和别人抢卡。

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

## 8. 下载训练数据（InternVid-10M-FLT）

> ⚠️ **仅训练 / 预训练需要**；只做零样本评测可跳过本节。数据体量极大（10M 视频片段、数 TB 级），
> 且需从 YouTube 抓取，请先评估磁盘与时间成本。

预训练语料是 **InternVid-10M-FLT**，在 `configs/data.py`（`internvid_v1 … v11`）里被切成
**11 个分片**，最终合并为 `internvid_all`。每个分片包含：

| 组成 | 路径 | 来源 |
|------|------|------|
| 标注 JSON（11 份） | `anno_downstream/InternVid/InternVid-10M-flt_{1..11}.json` | 作者 Google Drive（**已在第 5 步下载好**） |
| 视频文件 | `data/video_sample/InternVId-FLT_{1..11}/` | 按 JSON 里的 YouTube id + 时间戳自行抓取 |

也就是说：**标注已经有了**（第 5 步的 `anno_downstream/`），本节只需把**视频**下到对应目录。

### 8.1 目标目录结构

```
data/video_sample/
├── InternVId-FLT_1/     # 对应 anno_downstream/InternVid/InternVid-10M-flt_1.json
├── InternVId-FLT_2/
├── ...
└── InternVId-FLT_11/
```

`dataset/pt_dataset.py` 按 `os.path.join(data_root, anno['video'])` 读取，`anno['video']` 是以
`.mp4` 结尾的相对文件名，所以视频要**按 JSON 里的 `video` 文件名**放进对应分片目录。

### 8.2（可选）获取官方元数据 / UMT 分数

作者的 11 份 JSON 已够训练用。如需官方完整元数据（含 UMT 相似度、YouTube 语言/类型信息），
可从 HuggingFace 拉取：

```bash
pip install -U "huggingface_hub[cli]"
# InternVid-10M-FLT 及带更多信息的 INFO 版元数据
huggingface-cli download OpenGVLab/InternVid --repo-type dataset --local-dir data/InternVid_meta
huggingface-cli download OpenGVLab/InternVid-10M-FLT-INFO --repo-type dataset --local-dir data/InternVid_meta_info
```

### 8.3 从 YouTube 抓取视频片段

InternVid **不直接托管视频**，需按元数据里的 `YoutubeID` + `Start/End_timestamp`
用 `yt-dlp` 下载原视频并按时间戳裁剪，切成 clip 放进分片目录、且**文件名与 JSON 里的
`video` 字段一致**。

> 📖 **完整、可复制的分步操作 + 现成脚本见：[`docs/internvid-download-zh.md`](docs/internvid-download-zh.md)。**
> 仓库已提供下载器 [`tools/download_internvid.py`](tools/download_internvid.py)，会读标注 JSON、
> 并发调用 `yt-dlp --download-sections` 只下所需时间段、按 `video` 字段命名写入分片目录、
> 自动跳过已存在文件并记录日志。

最快上手（先看命名，再冒烟抓 200 条）：

```bash
pip install -U yt-dlp && sudo apt-get install -y ffmpeg

# 1) 先 inspect：确认标注里的 video 命名能被解析
python tools/download_internvid.py inspect \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json

# 2) 冒烟测试：只抓前 200 条到分片 1 目录
python tools/download_internvid.py download \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
  --out  data/video_sample/InternVId-FLT_1 \
  --limit 200 --workers 4
```

去掉 `--limit` 即放量抓整个分片；11 个分片把 `_1` 换成 `_2 … _11`（详见上面的文档）。
手写单条 yt-dlp 的等价命令：

```bash
# 把 YouTube id=<VID> 的 [START,END] 秒片段下到分片 1 目录
yt-dlp -f mp4 --download-sections "*<START>-<END>" --force-keyframes-at-cuts \
  -o "data/video_sample/InternVId-FLT_1/<VID>_<START>_<END>.mp4" \
  "https://www.youtube.com/watch?v=<VID>"
```

> ⚠️ **注意事项**：
> - YouTube 会在下载约 3–5 万个片段后**封 IP**，全量 10M 需要代理池 / 多机分批 + 断点续传；
> - 部分视频已失效 / 下架，抓不全属正常 —— 配置里 `jump_filter=True` 会跳过缺失样本；
> - **只想先跑通训练流程**：只下 1 个分片（如 `InternVId-FLT_1/`）的一小部分视频，再把
>   `config.py` 里 `train_corpus` 从 `"internvid_all"` 改成 `"internvid_v1"` 即可。

---

## 9.（可选）训练 / 预训练（8×V100-32GB 单机）

脚本已按**单机 8×V100-32GB** 配好：`run.sh` 用 `NNODE=1`、`NUM_GPUS=8`；`config.py` 用
**fp16**（V100 无 bf16 张量核）、每卡 `batch_size=64`（全局 512，与论文一致）。训练与评测一样
**不依赖 flash-attn**（见第 0 节 / 4.3）。直接跑：

```bash
# Tiny
bash scripts/pretraining/clip/mobileviclip_tiny/run.sh
# Small
bash scripts/pretraining/clip/mobileviclip_small/run.sh
```

针对 8×V100 已改动的地方（**无需再手动改**）：

| 位置 | 原值 | 现值 | 原因 |
|------|------|------|------|
| `run.sh` `NNODE` / `NUM_GPUS` | `2`/`8`(tiny)、`2`/`4`(small)（共 16 / 8 卡，含 slurm 残留） | `1` / `8` | 单机 8 卡；并显式加 `--nnodes=1`、`CUDA_VISIBLE_DEVICES=0..7` |
| `config.py` `use_bf16` | `True` | `False` | V100 无 bf16 张量核，改用 fp16 |
| `config.py` `bf16`（新增） | 默认 `True` | `False` | `shared_utils.py` 用该键决定模型 dtype，默认 True，必须显式关掉，否则模型仍被转成 bf16 |
| tiny `config.py` `batch_size` | `32` | `64` | 8 卡 ×64 = 全局 512（原为 16 卡 ×32） |

> 显存吃紧时可把 `batch_size` 调回 `32`（全局 256），或减小 `num_frames`；`gradient_checkpointing`
> 已开启，DeepSpeed ZeRO-1 + fp16 动态 loss scaling 也已在 `config.py` 配好。

单机手动启动（等价于 run.sh）：

```bash
export PYTHONPATH=${PYTHONPATH}:.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=8 --master_port=$((11000 + RANDOM % 20000)) \
    tasks_clip/pretrain.py \
    scripts/pretraining/clip/mobileviclip_tiny/config.py
```

---

## 10. 常见问题与排错

**Q1. `ModuleNotFoundError: flash_attn` / 想装 flash-attn 却编译失败**
本仓库已把 `flash_attn` 导入改为可选，**评测/训练并不需要它**，一般可直接忽略该缺失。
只有 Ampere+ 显卡要用 InternVideo2 主干时才需要装；若要装却失败：
- 确认显卡是 Ampere 及以上（V100 / 2080 等**不支持**，别装）；
- 确认 `nvcc --version` 与 PyTorch 的 CUDA 版本一致；
- 先 `pip install ninja packaging`，再用 `--no-build-isolation`；内存不足加 `MAX_JOBS=4`。

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

**Q6.1 `gdown --folder` 报 `Cannot retrieve the folder information from the link`**
这条错误出在**列目录**阶段（还没开始下文件），几乎不是真的没权限（该文件夹是公开的），
而是**gdown 版本旧**或**访问过于频繁被限流**（错误里 “have had many accesses” 就是限流）。
按顺序排查：

1. **强制升级 gdown 再试**（旧版解析不了 Google 现在的目录响应）：
   ```bash
   pip install --upgrade --no-cache-dir gdown
   gdown --version        # 需 ≥ 5.2.0
   gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream
   ```
2. **被限流就等一会**：反复重试会被 Google 临时限流，**等 30–60 分钟**或换网络/VPN 后只跑一次。
3. **绕过 gdown（最稳）**：在**本地浏览器**打开
   [该文件夹](https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza)
   → 全选 → **下载**（Google 会打包成 zip）→ `scp` 到服务器再解压：
   ```bash
   scp anno_downstream.zip rby@<server>:~/mobileclip/
   cd ~/mobileclip && unzip anno_downstream.zip -d anno_downstream
   ```
4. **判断是版本问题还是单个文件夹问题**：权重文件夹
   `gdown --folder https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7 -O checkpoints`
   若也报同样错 → 是 gdown 版本/限流（做 1、2）；若只有标注文件夹报错 → 可能是**文件数 >50**
   触发 gdown 文件夹上限，改用第 3 步的浏览器打包下载。

**Q7. 显存不够（OOM）**
调小对应 `config_*.py` 里的 `batch_size_test`（例如从 32 改到 8）。

**Q8.（国内网络）`conda env create` 报 `Tunnel connection failed: 302` / 卡在 `pkgs/main/terms.json`**
conda 连不上官方源 `repo.anaconda.com`（拉包和服务条款文件都失败）。按 **[4.0](#40-中国大陆用户先看配置国内镜像)**
配置清华镜像 + `export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true` 即可；若仍解不动锁死的 yml，改用
**[4.2](#42-方式二手动创建当-41-求解失败--显卡-cuda-版本不同时)** 手动 pip 方案。
另外确认没有失效代理在捣乱：`env | grep -i proxy`、`grep -i proxy ~/.condarc`。

**Q9.（V100 / 非 Ampere 显卡）flash-attn 装不上 / 报 `sm_70 not supported`**
正常现象——`flash-attn 2.x` 只支持 Ampere 及以上。**无需安装**：本仓库已让它成为可选依赖，
MobileViCLIP tiny/small 运行时用纯 PyTorch 注意力，评测/训练都不调用 flash-attn。按 4.2 装好
torch + 依赖（跳过 4.3）即可运行。

---

## 11. 快速上手清单（TL;DR）

```bash
# 1. 系统依赖
sudo apt-get update && sudo apt-get install -y git wget build-essential ffmpeg

# 2. conda（若无）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3 && source $HOME/miniconda3/etc/profile.d/conda.sh

# 3. 代码
git clone https://github.com/rbybryan/mobileclip.git && cd mobileclip

# 4. 环境（V100 等非 Ampere 显卡：无需 flash-attn，用手动方案更稳）
conda create -n mobileviclip python=3.10 -y && conda activate mobileviclip
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --index-url https://download.pytorch.org/whl/cu117
pip install einops timm==0.5.4 transformers==4.28.1 tokenizers==0.13.3 decord av \
    opencv-python ftfy regex pandas numpy==1.26.4 scipy tensorboard wandb pyyaml \
    easydict deepspeed==0.10.1 openai-clip librosa
# flash-attn 不用装（仅 Ampere+ 且用 InternVideo2 主干时才需要）

# 5. 权重 + 标注
pip install gdown
mkdir -p checkpoints anno_downstream
gdown --folder https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7 -O checkpoints
gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream

# 6. 下载 MSR-VTT 视频（体量最小、有直链；其它数据集见第 6 步）
mkdir -p data && cd data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip MSRVTT.zip && ln -s "$(pwd)/MSRVTT/videos/all" "$(pwd)/MSRVTT_Videos"
cd ..

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
