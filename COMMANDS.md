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

> ⚠️ **That pin describes no machine this project actually uses (corrected 2026-07-30).** The
> 1.13.1+cu117 recipe is validated **only on a CPU node** (import check, below). **Both GPU boxes
> run PyTorch 2.5.1:**
>
> | machine | torch | why | source |
> |---|---|---|---|
> | **b100 / 8×H100 (sm_90)** — main training clock | **2.5.1+cu124** | system torch already has `sm_90`; repo pin "从没被装上" | `experiments/SETUP_B100.md` §2 (measured 2026-07-24) |
> | **101vector / 8×V100 (sm_70)** — eval + export | **2.5.1+cu118** | pinned *at 2.5.1* deliberately, for the sm_70 kernels | `experiments/ROADMAP.md:126`, `experiments/export_smoke_results.md` §env |
>
> Consequences, because several docs still reason from the 1.13.1 premise:
> - **`F.scaled_dot_product_attention` (torch ≥2.0) is available on both boxes, and has been
>   since at least 2026-07-24.** Backlog item **P1 is therefore not blocked** — see
>   `notes/architecture-round4.md` §8.1.
> - The DeepSpeed-compatibility risk that demoted P1 (`notes/paper-plan.md` §6.8 item 8) is
>   **already discharged**: the 1M-tier arms trained ~15 h under DeepSpeed on torch 2.5.1
>   (`ROADMAP.md` G9), and both R1 arms ran on `2.5.1+cu118` (`ROADMAP.md:304`).
> - `torch.compile`, `channels_last`, and CUDA graphs are likewise available — relevant to the
>   open **MFU 0.67%** question (`architecture-round4.md` §0.2).
>
> **Do not "upgrade" anything to act on this** — the upgrade already happened. What is missing is
> code that *uses* it, plus a `hasattr` guard so the CPU-node 1.13.1 recipe keeps importing.

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
python -m torch.distributed.run --nproc_per_node=1 --master_port=$((10000 + RANDOM % 20000)) \
    tasks_clip/retrieval.py \
    scripts/evaluation/clip/zero_shot/mobileviclip_tiny/config_msrvtt.py
```

### 3.1 GATE-0 — 时间维容量审计(零训练;做任何时序改动之前先跑)

回答"这个模型到底有没有编码时间方向,以及我们的基准能不能测出来"。
背景、判读表与后续决策见 [`notes/neural-inspired-round2.md`](notes/neural-inspired-round2.md) §2。

**(a) 静态审计 —— 只读 checkpoint,不用 GPU、不用数据、几秒钟出结果**

```bash
python experiments/temporal_capacity_audit.py --self-test        # 先自检(不需要权重)
python experiments/temporal_capacity_audit.py checkpoints/mobileviclip_small.pt
```

判读:时间卷积核**反对称分量占比**中位数 < 0.05 ⇒ 核近似对称 ⇒ 对时间反转基本不敏感;
**DC/AC** 中位数 > 0.95 ⇒ 近似纯低通 ⇒ 帧间残差被抹平。两者同时成立 ⇒ 模型几乎不编码
时间方向,预测 RTime-Binary ≈ 50%。若默认键名对不上,脚本会把找到的候选键打印出来,
按提示传 `--kernel-key`。

**(b) 动态审计 —— 评测期帧扰动(纯评测,不训练)**

由环境变量 `MVC_FRAME_PERTURB` 控制,**默认不设 = 原样基线路径(逐位一致)**:

| 取值 | 含义 | 测什么 |
|------|------|--------|
| `shuffle` | 随机置换 8 帧(按样本 index 确定性设种) | 模型总的时序敏感度 |
| `reverse` | 时间反转 | 专测"时间之箭" |
| `freeze` | 全部换成中间帧 | 多帧信息一共值多少 |

在 **MSR-VTT(外观向)与 SSv2-MC(时序向)两边都要跑**,和干净基线对比:

⚠️ 这是**评测**,按 README §3.0 的**双根规则**必须先 `unset MVC_DATA_DIR`(训练/试点 shell
才 export;混用会把仓库 `data/` 下已下好的评测集重定向到不存在的路径,详见
`notes/live/from-claude.md` (13))。走既有的 `experiments/run_eval.sh` 包装
——它会按基准自动选任务入口(**`ssv2_mc` 走 `retrieval_mc.py`,其余走 `retrieval.py`**),
并把结果写进各自的 `eval_res.json`:

```bash
unset MVC_DATA_DIR
python experiments/check_readiness.py            # msrvtt / ssv2_mc 应为 [OK ]

for DS in msrvtt ssv2_mc; do
  for P in none shuffle reverse freeze; do
    echo "===== $DS / perturb=$P ====="
    MVC_FRAME_PERTURB=$P CUDA_VISIBLE_DEVICES=0 \
      bash experiments/run_eval.sh small $DS \
        ./checkpoints/mobileviclip_small.pt \
        experiments/evals/gate0/${DS}_${P}
  done
done
```

汇总成一张表(主口径 = `ret_test_emb/` 原始余弦;`ret_test/` 是 DSL 口径):

```bash
python - <<'PY'
import json, glob, os
rows = {}
for p in sorted(glob.glob("experiments/evals/gate0/*/eval_res.json")):
    ds, perturb = os.path.basename(os.path.dirname(p)).rsplit("_", 1)
    rows.setdefault(ds, {})[perturb] = json.load(open(p))
for ds, d in rows.items():
    print(f"\n=== {ds} ===")
    for perturb in ("none", "shuffle", "reverse", "freeze"):
        if perturb in d:
            print(f"  {perturb:8s} {json.dumps(d[perturb], ensure_ascii=False)[:300]}")
PY
```

> ⚠️ `MVC_FRAME_PERTURB` **只在审计评测时设置,绝不要在训练时打开**。打开时日志会打印
> 一行 `帧顺序正在被人为扰动(GATE-0 审计模式)` 的 warning —— 训练日志里出现这行就是配置错了。

判读(`notes/neural-inspired-round2.md` §2c):

- MSR-VTT 与 SSv2-MC **都**几乎不变 ⇒ 模型时间盲**且**基准测不出来 ⇒ 只有 R2(可恢复顺序的
  读出头)值得投,且必须上 RTime 才能测;
- MSR-VTT 不变、**SSv2-MC 掉** ⇒ 模型有顺序信息,是**基准**的问题 ⇒ 推进 R2,补 RTime/Vinoground,
  按 Pareto 结果汇报。

无论哪个结果都是一张论文图(在自己的模型上测出的"可测性上界",与 ROADMAP 1.6 的多 T 退化
曲线同一格)。

---

### 3.2 R1 前置探针 — 假负例 / 近重复密度(零训练;`notes/r1-examination.md` §5)

在动手建 R1 文本负例库之前,先量化机会与风险。**A 部分现在就能跑**(不用 GPU / 权重 /
decord,只要 anno JSON);判读见 `notes/r1-examination.md` §5/§6。

**A · 精确重复密度(秒级,模型无关)** —— 用仓库自己的 `caption_to_idx` 统计字幕碰撞,
这正是损失多正例掩码 `get_mask` 用的键:

```bash
python experiments/r1_fn_density.py --self-test          # 先自检
python experiments/r1_fn_density.py \
    --anno anno_downstream/InternVid/InternVid-10M-flt_1.json
# 想看全量趋势可多给几片:--anno anno_downstream/InternVid/InternVid-10M-flt_*.json
# 语料很大时先抽样:--sample 200000
```

判读:`distinct_frac` 接近 1 ⇒ 65k 池里基本都是不同字幕(难负例供给充足);
「随机一对同字幕的概率」× 65k = 每个池子的期望假负例数 —— 这部分**只要负例库穿过
`get_mask`(examination §3)就被当正例处理掉**,`r1a_sigmoid.py` 已设 `caption_idx=True`。

**B · 嵌入近重复密度(现在一条命令,复用 eval 的模型构建路径)** —— idx 掩码抓不到、
但对比损失会当假负例的那部分。先 dump 文本特征,再算密度:

```bash
# ① 用冻结文本塔编码 InternVid 字幕(纯前向,单卡即可;自动去精确重复后编码)
python experiments/r1_textfeat_dump.py \
    --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
    --ckpt checkpoints/mobileviclip_small.pt --out /tmp/txtfeats.npy --sample 10000
# ② 算近重复密度
python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy --thresholds 0.9,0.95,0.99
```

判读:「有≥1近邻」占比高 ⇒ 大量语义近重复被当负例 ⇒ R1 需上基于余弦阈值的去重掩码
(`nR1_textbank` 的 `dedup_cos`);低 ⇒ 免了。
⚠️ `r1_textfeat_dump.py` 的**建模型/前向部分未在真权重上验证**(Claude 侧无权重);
若构建报错,把完整报错贴回 —— 字幕读取/去重/分块已自测,只差建模型那步。

> **在这两条之后**才开 R1 训练臂,且按 examination §6 重构为「SigLIP 制度」:
> sigmoid 损失(已实现,`r1a_sigmoid.py`)+ 温度 + 大负例池,一条臂。
> 训练臂里再加**批内饱和度**探针(**这才是 go/no-go**):每步记录 banked 负例占 softmax
> 质量的比例(<5% ⇒ 库惰性,砍)和正样本在 512 批内负例中的排名(多数已 rank-1 ⇒ 512 就已饱和)。
> **完整落地方案(触点/损失/fp16/开关/时序)见 [`notes/r1-implementation-plan.md`](notes/r1-implementation-plan.md)。**

---


### 3.3 R1 下一步 — 精确命令(三档,按信息价值排序)

全部从 repo 根目录跑。判读与方案见 [`notes/r1-implementation-plan.md`](notes/r1-implementation-plan.md)。

**① go/no-go 探针 —— 已跑过一轮(2026-07-27),但探针指标当时选错,已在本轮修正。**
结果 = **不建、暂不砍(inconclusive,偏向砍)**;完整判读与决策表见
[`notes/r1-implementation-plan.md`](notes/r1-implementation-plan.md) **§7**。要点:旧判据
「`top1_is_pos→1 ⇒ 砍`」是**错的科学**(最难负例随池 `~log(池大小)` 增长,512 全分开 ≠ 大池无益),
且 `top1/margin` 是**分离度代理,不是梯度量**。修正后探针改记两项**能直接判**的量:
`maxneg_by_pool`(最难负例随池大小是否仍在涨)与 `neg_grad_mass`(批内负例的有效梯度质量)。
`pilot_10k_r1.py` 是 sigmoid 试点;`MVC_LOG_SAT=<间隔>` 打开只读探针(默认不设 = 逐位不影响训练)。

```bash
cd "$(git rev-parse --show-toplevel)" && git pull origin main
export MVC_DATA_DIR=/home/vca1/datasets          # 试点视频根(训练/试点要 export)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7       # 按空卡改
# 重跑一轮读新指标(~180 步,10 epoch),每 20 步打一行 [SAT]:
MVC_LOG_SAT=20 bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
grep '\[SAT\]' experiments/runs/pilot_10k_r1/train.log | tail -20
```
判读(§7 决策表):`maxneg_by_pool` 到 512 **已平** ⇒ **砍**(没有更难的负例可加)。
仍在涨 + Part B 近重复密度**高** ⇒ **砍/先修标签**(「难负例」多是假负例)。
仍在涨 + 近重复密度**低** ⇒ **才建 R1**(三种里最不可能)。`neg_grad_mass≈0` ⇒ 批内负例已「死」。

**② Part B — 近重复密度(定 R1 要不要余弦去重掩码;顺便验证 dump 脚本在真权重上能跑)**

```bash
python experiments/r1_textfeat_dump.py \
    --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
    --ckpt checkpoints/mobileviclip_small.pt --out /tmp/txtfeats.npy --sample 10000
python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy --thresholds 0.9,0.95,0.99
```
判读:「有≥1近邻」占比高 ⇒ 需余弦去重掩码;低 ⇒ 免。
⚠️ dump 的建模型部分未在真权重上验证;报错就把完整 traceback 贴回。

**③ Part A 收紧到 55k(秒级,可选)**

```bash
python experiments/r1_fn_density.py --anno anno_downstream/InternVid/InternVid-10M-flt_*.json
```

> **注:** ① 的探针在 `models/criterions.py`(`saturation_stats` + `_maybe_log_saturation`,
> 默认关、只读、已单测:`maxneg_by_pool`/`neg_grad_mass` 数学对齐 numpy 参考)。它测的是
> **批内**结构(无需负例池);建池后的「banked 占 softmax 质量」探针在真正建 R1 池子时再加。
> 近重复去重掩码现在是**建池的前置必需**(不是可选)——见方案 §7.4。

---

### 3.3.1 R1 建池 A/B(2.4c)—— go/no-go 判 GO 之后建的臂

**方案 §7.6 判 GO,代码已落地**(`models/criterions.py` 池损失、`models/mobileviclip_small.py`
接线、`text_encoder.py` 暴露投影前特征、`experiments/r1_build_textbank.py` 缓存、
`experiments/configs/nR1_textbank_pilot.py` 臂配置)。**默认关 = 逐位不变**(不设 `criterion.text_bank`
的配置完全不受影响)。池的数学已 CPU 单测;端到端前向/缓存只能在服务器验。

> ⚠️ **建缓存的命令必须一行写完,别用反斜杠折行。** 2026-07-27 那次粘贴时 `\` 续行被拆断,
> `--ckpt/--out-feats/--out-idx` 全被丢掉,脚本用**默认**输出名把缓存建到了
> `experiments/cache/textbank_feats.fp16.npy`(而配置要的是 `..._pilot_...`),A/B 于是
> `FileNotFoundError`。**好消息:那份缓存本身是对的**——投影前特征只由 eval 配置里的冻结文本
> 底座(`mobileclip_s2.pt`)决定,与 `--ckpt` 无关,所以 `--ckpt` 丢了也没影响。**直接改名即可,不必重建:**
>
> **`cp` 的源+目标全路径太长会被终端折行,粘贴时折行变真换行 → `cp` 只拿到一个参数
> (`missing destination file operand`)。先 `cd` 进 cache 目录,用短文件名,每行都短到不折行:**
>
> ```bash
> cd "$(git rev-parse --show-toplevel)"/experiments/cache
> cp textbank_feats.fp16.npy textbank_pilot_feats.fp16.npy
> cp textbank_idx.npy textbank_pilot_idx.npy
> cd "$(git rev-parse --show-toplevel)"
> bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot.py 8
> ```
> 启动日志应出现:`[R1] text bank ON: pool=(8999, 512) ...`。

```bash
cd "$(git rev-parse --show-toplevel)" && git pull origin main      # 若报 "divergent branches",见下方 TROUBLESHOOTING
export MVC_DATA_DIR=/home/vca1/datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ① 建缓存(投影前文本特征 + caption_to_idx;单卡、纯前向、几分钟)。一行,勿折行。
#    建模型走的是 Part B 已验证的同一条路;若报错把完整 traceback 贴回。
python experiments/r1_build_textbank.py --anno anno_downstream/internvid_zh_10k_pilot.json --out-feats experiments/cache/textbank_pilot_feats.fp16.npy --out-idx experiments/cache/textbank_pilot_idx.npy

# ② A/B(等步数、同数据):对照 = pilot_10k_r1(无池),处理 = nR1_textbank_pilot(有池)。
#    对照通常已有(§3.3);只需再跑处理臂。启动日志应出现一行 `[R1] text bank ON: ...`。
bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot.py 8
# 若对照臂还没跑:
# bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
```
判读(看动力学,不看绝对精度):处理 vs 对照的 held-out R@1。**两个方向都报**——文本池硬化
**v2t(txt_r1)**,t2v(img_r1)只通过共享编码器间接受益(examination §1)。**kill 判据**:池 vs 无池
< 0.5 R-Sum ⇒ (d) 误判,停(方案 §5)。
> 池损失默认关的不变性:`vtc_loss_sigmoid` 里池块全在 `if text_bank is not None` 下;不设配置 ⇒ 逐位同旧。

### 3.3.2 R1 大数据 A/B —— 放到 1M 验证层跑(H100;真正发威档)

10k 试点判 GO 但幅度经 40-epoch 修正为 ΔR-Sum≈+3.9,且主要是「挡早塌」的小数据伪影(`r1-implementation-plan.md`
§9.4)。真正的检验在**批内负例成为瓶颈**的档:H100 现在训 1M 子集(`internvid_v1`),全局 batch 512 远低于
SigLIP 饱和点(~32k),bank 补 +16384 正好推进甜区,且 1M×3 epoch 不过拟合 ⇒ 领先=真硬负例。设计见
`r1-implementation-plan.md` §10、`notes/verification-tier-1m-zh.md`。**一对自配对 A/B,彼此相减(不减 r0)。**

```bash
cd "$(git rev-parse --show-toplevel)" && git fetch origin main && git reset --hard origin/main   # divergent 见 TROUBLESHOOTING
export MVC_DATA_DIR=/home/vca1/datasets

# ① 建 1M 文本负例库缓存(单卡、纯前向、~10–30 min、~1GB;一行,勿折行)。与试点缓存互不覆盖。
python experiments/r1_build_textbank.py --anno anno_downstream/InternVid/InternVid-10M-flt_1.json --out-feats experiments/cache/textbank_1m_feats.fp16.npy --out-idx experiments/cache/textbank_1m_idx.npy

# ② 跑两对(复用 1M 验证层 runner,串行;~15h/config → 4 条 ~2.5 天)。卡 = 0,1,2,3(维护者全部配额;
#    与全量 R0 时分复用,R0 空出后再跑)。处理臂启动日志应见 `[R1] text bank ON: pool=(~1e6, 512) ...`
bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_r1both_s42 sub1m_nR1_s42 sub1m_r1both_s43 sub1m_nR1_s43

# ③ 收结果(标准 MSR-VTT 1k,best epoch)
grep -hE "t2v_r1|v2t_r1|_match|R@" experiments/runs/sub1m_{r1both,nR1}_s4?.log | tail -40
```
判读:ΔR-Sum = `sub1m_nR1_sXX` − `sub1m_r1both_sXX`(**同 seed**),两 seed 同向且超 MSR-VTT 1k 的 ±SE 才算赢
(两方向都报;bank 硬化 v2t/`txt_r1`)。**两 seed 都清 ±SE ⇒ 真硬负例收益、带 bank 上全量 10M 终训;≈0/矛盾 ⇒
试点 +3.9 是 10k 专属、不迁移 ⇒ 撤 bank。** 全量 10M = 论文数字,1M 只是决策门。

---

### 3.4 选帧基线 — C2 必须打赢的那条平凡基线(零训练,V100 一晚)

**为什么现在做**:GATE-0 实测**帧的多样性值 ~13 个 R@1、帧的顺序值 ~0**
(`experiments/gate0_results.md`),而 1.6 的多 T 曲线显示 T=2 相对 T=8 掉 5.1 个
t2v R@1。两件事合起来说明:低帧预算下「挑哪几帧」可能很要紧,而**从来没量过**。
C2 的卖点是"弹性 T 在低 T 优于固定 T",审稿人必然会问有没有跟"零训练、只是把帧
挑得聪明一点"比过 —— 现在这张对照表不存在。跑完就存在了。

⚠️ 评测,按 README §3.0 双根规则必须先 `unset MVC_DATA_DIR`;脚本内部也会 unset,
并在检测到 `MVC_FRAME_PERTURB` 已设置时**拒绝执行**(两个效应混在一起读不出来)。

```bash
unset MVC_DATA_DIR
python experiments/check_readiness.py            # msrvtt 应为 [OK ]
python -m pytest tests/test_frame_select.py -v   # 34 项;先在本机确认钩子行为

# 9 个臂,单卡串行。已有 eval_res.json 的臂自动跳过 ⇒ 中断后重跑即可续跑
CUDA_VISIBLE_DEVICES=0 bash experiments/run_frame_select.sh \
    small msrvtt ./checkpoints/mobileviclip_small.pt \
    experiments/evals/frame_select

python experiments/collect_frame_select.py --root experiments/evals/frame_select
```

先 `FS_DRY_RUN=1 bash experiments/run_frame_select.sh` 看一眼要跑哪 9 个臂。

| 臂 | 帧从哪来 | 作用 |
|---|---|---|
| `pool16` | 整段 middle 采 16 帧,不选 | 复现 1.6 的 T=16 ⇒ 兼作"钩子未开=零操作"的验证 |
| `nativeT2` / `nativeT4` | 整段 middle 采 K 帧,不选 | 复现 1.6 的 T=2/T=4;与 `seluniform` 的差别只在**采样支撑** |
| `seluniform_K*` | 16 帧母序列里均匀取 K | **受控基线**,与 C2 训练切片口径逐位一致 |
| `seldivmax_K*` | 16 帧里贪心 k-center 选 K | **候选方案**,只用降采样像素打分、不过编码器 ⇒ 可部署 |
| `selrandom_K*` | 16 帧里确定性随机取 K | **地板**;uniform 若赢不过它,该 K 上选帧问题不存在 |

三条读法(汇总脚本会自动打出来):

1. **先看地板**。`uniform` 打不过 `random` ⇒ 在这个 K 上"挑哪几帧"不构成问题,
   `divmax` 的任何差值都别当真。
2. **判据钉在 t2v R@1、两口径同向、|Δ| ≥ 1.6**。1.6 是 1K 测试集的二项抽样误差
   (与 GATE-0 判 `reverse` 为噪声同一条带);`r_mean` 只当描述性数字,**不要**给它
   套 R@1 的阈值。
3. **功效上限**:1K 测试集只看得到 ≥1.6 R@1 的效应。若差值在 1 个点上下,本实验
   **判不了**,要换 MSR-VTT 全量 3k / DiDeMo / ANet ——「没测出来」不等于「没有效应」。

结果落档到 `experiments/frame_select_results.md`(模板已在,含判读栏)。

**成本诚实性(报结果时必须写)**:`divmax` 不过编码器,所以是同成本对照;用编码器
特征选帧要先把 16 帧全编码一遍,等于把想省的算力先花掉,**只能当上界报**。特征空间
上界若要做,走离线预计算下标的路子,别塞进数据集钩子。

---


### 3.5 选帧诊断 —— divmax 为什么失败(零 GPU,~10 min)

1.8 已定论:K=2 上**选帧要紧**(uniform 相对 random +3.40/+3.90)**但 divmax 是错的判据**
(相对 uniform −2.80/−2.40,且与纯随机无可测差别)。剩下的问题是**为什么**,
而两个预注册假说对**选帧下标**给出相反预测
([`experiments/frame_select_results.md`](experiments/frame_select_results.md) §4.1):

| 假说 | 端点命中率 | 平均配对间隔 \|i−j\| |
|---|---|---|
| (A) k-center 偏好离群帧 | 远高于随机基线(16 取 2 时 0.242) | 接近最大(15) |
| (B) 像素距离 ≠ 时间距离 | 接近随机基线 | 接近 (n+1)/3 ≈ 5.67 |

```bash
unset MVC_DATA_DIR                                  # 评测侧不 export(双根规则)
python experiments/frame_select_diag.py --limit 50  # 冒烟,几十秒
python experiments/frame_select_diag.py             # 全量 1000 条
```

不建模型、不过编码器、不需要 GPU;复用 `VidTxtRetEvalDataset` + 同一套变换
(`Resize(256)` + `/255`,**Normalize 在 `get_test_transform` 里是注释掉的**)
⇒ 打分用的像素与 1.8 那轮逐位相同。原始下标 dump 到
`experiments/evals/frame_select/diag_indices.json` 供复查。

脚本会拒绝在 `MVC_FRAME_SELECT` / `MVC_FRAME_PERTURB` 开着时运行 —— 那会改变
数据集吐出的帧,等于诊断了另一件事。

**产出怎么用**:若判 (A),论文里那句话是**「要有代表性的帧,不要极端帧」**,
与 GATE-0 相容(`freeze` 崩 13 分 ⇒ 需要一些分散;本轮 ⇒ 最大化分散是过头的)。
若判 (B),结论收窄为**「廉价像素代理不可用」**,不能推广到"多样性判据错"。

---

### 3.6 导出冒烟(ROADMAP 1.5 / 评审 U1)—— 零 GPU,零训练,随机权重

**产出决定 P9(分离注意力变体)是否提前触发**,进而决定弹性模型要不要直接用
可导出形态训练(免得第 8 周再单独微调一个部署变体)。**本表不含延迟** —— 真机
延迟档仍在第 8 周。

```bash
# 装多少跑多少;缺哪条依赖那一列就记 skip,不影响另外两条
pip install onnx onnxruntime onnxscript      # ONNX 路线(torch>=2.6 需要 onnxscript)
pip install litert-torch                     # TFLite 路线(ai-edge-torch 已改名,旧包是空壳)
pip install coremltools                      # Core ML 路线

python experiments/export_smoke.py --mini    # 先自检脚本本身(秒级)
python experiments/export_smoke.py           # 全尺寸,T∈{2,4,8,16}
python experiments/export_smoke.py --inference-mode   # 重参数化后的部署形态
```

导出的是**视觉塔**(`FastViT`,架构参数与 `build_vision_encoder` 逐项一致 ⇒ 含全部
时序算子);输入签名 `[B,T,C,H,W]` 与 `encode_vision` 对齐(脚本内做同一行 permute)。
文本塔不导 —— 端侧文本 embedding 通常离线算好。

产出:`experiments/exports/export_smoke.{json,md}`。按 U1 验收,把 md 那张表贴进
`notes/live/from-claude.md` 或 `TROUBLESHOOTING.md`。

**读表三条规则:**

- 三路全 **pass** ⇒ P9 **不触发**,弹性模型按现形态训练;
- **联合注意力导不出**(失败算子指向 attention / `temporal_embedding`)⇒ P9
  **提前进入 R3 架构决策视野**;
- **skip 不是结论**。脚本把「缺依赖」与「缺传递依赖」都归为 skip 而非 fail ——
  把环境问题记成 fail 会产生**假的 P9 触发**。别把 skip 读成 pass,也别读成 fail。

**已知(本机 CPU 预跑,torch 2.12)**:全尺寸 ONNX 路线 T=2 / T=4 **pass**
(导出 + `onnx.checker` + onnxruntime 加载推理全过)。TFLite / Core ML 两路本机
无依赖未测。

---

### 3.7 未见过的 T 泛化(ROADMAP 1.9)—— 零训练,答「C2 能不能无上限 T」

分析与判读见 [`notes/c2-arbitrary-t.md`](notes/c2-arbitrary-t.md)。官方权重上评**训练
从没见过的 T**:内插 T∈{3,5,6,12}(16 帧锚点内)、外推 T∈{24,32}(锚点外)。
单卡串行、可续跑;每个 T 就是一次普通 MSR-VTT 零样本评测。

```bash
cd "$(git rev-parse --show-toplevel)"
export CUDA_VISIBLE_DEVICES=0
for T in 3 5 6 12 24 32; do
  out=experiments/evals/c2_unseenT/msrvtt_T${T}
  [[ -f ${out}/eval_res.json ]] && { echo "skip T=${T} (done)"; continue; }
  bash experiments/run_eval.sh small msrvtt ./checkpoints/mobileviclip_small.pt \
       ${out} num_frames_test ${T}
done
```

**读法**(判据沿用 `experiments/README.md` §6.1):

- 主指标 **t2v R@1**,`ret_test/`(DSL)与 `ret_test_emb/`(raw)**两口径同向**才算数;
  阴性用 ±1.6 带(保守,照用),**阳性主张要求 |Δ| 明显大于 2.4**;`r_mean` 仅描述性。
- **参照行**:ROADMAP 1.6 已锁 T=2/4/8/16 的 raw r_mean = 55.4/60.0/61.6/62.3。
  新的 T 应当落在这条曲线上 —— **落在线上 = 泛化成立**;某个 T 明显掉出线 = 那里有问题。
- ⚠️ **T=24/32 可能 OOM**:stage-4 联合注意力是 O(T²)(N=64·T)。评测 batch 小,
  大概率放得下;**崩了也是结果** —— 那就是 `c2-arbitrary-t.md` 墙 1 的直接证据,照样记账,
  别去调 batch 硬扛。
### 3.7b 🔺 排 H100 新任务之前:先看 1M 层现状(30 秒,纯只读)

```bash
cd "$(git rev-parse --show-toplevel)"
python experiments/status_1m.py
```

只用标准库、不 import torch、不碰 GPU ⇒ **训练在跑时也能安全执行**。它回答三个
`nvidia-smi` 和 `ls` 都看不出来、但排队前必须知道的问题:

| 问题 | 为什么不能靠肉眼 |
|---|---|
| **还有臂在跑吗?** | `run_verify_1m.sh` 是**串行**的。再起一条会和在跑的抢同一组卡。脚本用 `pgrep` 直接查。 |
| **"跑完"的臂墙钟对吗?** | 参考 ~15 h/臂(`sub1m_r0_s42` 实测 15:11:58)。**短于一半 ⇒ 没跑满 epoch,结论无效**,不是"结果为负"。`sub1m_r1b_s42` 曾只跑 2:12:39 就进了"完成"清单。 |
| **哪些真有结果?** | **有 ckpt ≠ 有结论**。要有 `eval_res*.json` 才算测过;只有 ckpt 的是中断,`auto_resume` 可续。 |

输出的五种状态:`未排队` / `🟢 运行中` / `✅ 已完成` / `🔴 可疑`(有结果但墙钟过短)/
`⚠️ 中断`(有 ckpt 无 eval)。带 🔴 或 ⚠️ 的臂会附上**续跑命令**。

⚠️ **「运行中」要两个时钟同时新**:文件 mtime 会被 `git pull` / `cp` / `rsync` 无意刷新,
单看它会把三天前就死掉的日志报成在跑,于是**挡住本来可以排的队**;所以脚本同时读
日志里最后一行**自己的时间戳**,取两者中更旧的那个。

改参考墙钟:`--expect-hours 15.2`;只看某几条:`--pattern 'sub1m_r1*'`。

### 3.8 离线补评测 —— 训练跑完但评测被跳过时,救回已存的 ckpt(零重训)

用于 `Empty DataFrame` 那个坑(见 `TROUBLESHOOTING.md`「训练跑完但没有任何评测数字」)。
DeepSpeed 存的是**目录式** ckpt(`ckpt_02.pth/mp_rank_00_model_states.pt`),
`tasks_clip/shared_utils.py` 的**非 deepspeed 加载支路**现在会自动解析这种目录,
**不用重训**。(该支路 2026-07-29 之前只判 `osp.isfile`,目录恒为 False ⇒ 静默不加载,
见下面的自检。)

```bash
cd "$(git rev-parse --show-toplevel)"
source experiments/env_b100.sh         # ⚠️ b100 必做:设 PYTHONUSERBASE,否则 termcolor 等导不进
                                       #   (脚本现在也会自动补,但显式 source 更稳)
export CUDA_VISIBLE_DEVICES=3          # ⚠️ 只有 0-3 四张卡(#339,没有 4-7)
                                       #   训练在跑时可与之共卡:训练峰值 ~42G,80G 卡余量够
RUN=experiments/runs/sub1m_r0_s42      # 换成要补评的臂
bash experiments/run_eval.sh small msrvtt "${RUN}/ckpt_02.pth" \
     experiments/evals/sub1m/$(basename ${RUN}) num_frames_test 8
```

**⚠️ 跑完先做这个自检 —— 否则可能把「未加载权重」的数字当真**:

```bash
grep -E "Load pretrained model from|training from scratch|自动解析到" \
     experiments/evals/sub1m/$(basename ${RUN})/train.log
```
必须看到 **`Load pretrained model from .../ckpt_02.pth/mp_rank_00_model_states.pt`**。
看到 `training from scratch` 就是**权重没加载上,数字无效**(2026-07-29 踩过:DeepSpeed 存的是
**目录式** ckpt,而评测走 `deepspeed.enable=False` 那一支只判 `osp.isfile` ⇒ 目录恒为 False ⇒
静默跳过。现已改为自动解析目录 + 加载失败直接 raise)。

**读法**:`eval_res.json` 里 `ret_test/` = DSL 口径、`ret_test_emb/` = raw 主口径;
1M 层判胜负按 `notes/verification-tier-1m-zh.md` —— **同 seed 相减 + 两个 seed 同向**,
且差值要过 `experiments/README.md` §6.1 的带宽(阴性用 ±1.6,阳性要 |Δ|>2.4)。

⚠️ **三条容易错的**:
- ckpt 路径要指到**目录** `ckpt_02.pth`(不是里面的 `.pt` 文件);
- 各臂的 `output_dir` 是隔离的,别把 s42/s43 的结果写进同一个 `--out`;
- 补评的是**训练内 held-out 的替代**,和论文主表用的零样本评测是同一套脚本,口径一致。

#### 批量:补齐所有「有 ckpt 但没 `ranks_*.npz`」的臂

配对 bootstrap(§3.9)要 `ranks_*.npz`,而它 **2026-07-29 才开始落盘** ⇒ 之前评过的臂
都得再走一次前向。先看清单(零 GPU):

```bash
for d in experiments/runs/sub1m_*/; do
  n=$(basename "$d"); [ -e "$d/ckpt_02.pth" ] || continue
  printf "%-24s ranks=%s\n" "$n" "$(ls experiments/evals/sub1m/$n/ranks_*.npz 2>/dev/null | wc -l)"
done
```

`ranks=0` 的批量补(每条 MSR-VTT 1k 约 1 分钟,**串行**跑,别并发抢卡):

```bash
source experiments/env_b100.sh
export CUDA_VISIBLE_DEVICES=3
for d in experiments/runs/sub1m_*/; do
  n=$(basename "$d"); [ -e "$d/ckpt_02.pth" ] || continue
  ls experiments/evals/sub1m/$n/ranks_*.npz >/dev/null 2>&1 && { echo "skip $n"; continue; }
  echo "=== $n ==="
  bash experiments/run_eval.sh small msrvtt "$d/ckpt_02.pth" \
       "experiments/evals/sub1m/$n" num_frames_test 8 batch_size_test 32 || echo "FAILED $n"
done
```

跑完对**每条**臂做一次上面那个 `grep "Load pretrained model from"` 自检 —— 权重没加载
也会打印出一套很像样的数字(实测 r_mean 54.82),不查就会当真。

### 3.9 配对 bootstrap 比较两条臂(评审 U10-1 / ROADMAP 自查 G4)—— 零 GPU

**为什么不能直接拿 ±1.6 判 A/B**:那是**无配对**带(「一个绝对 R@1 的抽样误差」)。
两条臂评的是**同一批**查询,做差会抵消共享方差。实测差距极大 —— 同一对臂:

| R-Sum 的 95% CI 半宽 | |
|---|---|
| **配对** | **±1.5 量级** |
| 无配对 | ±17 量级 |

⇒ **约 11 倍分辨率**。这就是为什么 1M 层非它不可:`sub1m_r0_s42` 显示「10× 数据」
的效应只有 **+0.82 r_mean**,不配对根本分辨不出来(见 `notes/verification-tier-1m-zh.md` §6)。

```bash
cd "$(git rev-parse --show-toplevel)"
# raw 主口径
python experiments/collect_results.py --compare \
       experiments/evals/sub1m/sub1m_r1a_s42 \
       experiments/evals/sub1m/sub1m_r0_s42
# DSL 口径(两口径同向的纪律照旧)
python experiments/collect_results.py --compare \
       experiments/evals/sub1m/sub1m_r1a_s42 \
       experiments/evals/sub1m/sub1m_r0_s42 --prefix 'ret_test/'
```

**判读:以 95% CI 是否含 0 为准**,不要再套 ±1.6。可选 `--n-boot`(默认 10000)、
`--seed`(默认 0,可复现)、`--out <md>`。

⚠️ **需要 `ranks_*.npz`** —— 评测时自动落盘(每条臂 ~8KB),但**逐查询 rank 是
2026-07-29 才开始存的**。在那之前评过的臂(如首版 `sub1m_r0_s42`)没有,
**重跑一次评测即可**(纯前向,MSR-VTT 1k 约 1 分钟,命令见 §3.8)。

#### 自比对自检(拿到第一份 `ranks_*.npz` 后必做一次)

把**同一个目录**同时当臂和基线传进去。配对正确时两侧逐 bootstrap 抽到的是同一组
下标 ⇒ **每个指标 Δ 必须恰好 0.00,CI 必须是 `[0.00, 0.00]`**:

```bash
python experiments/collect_results.py \
    --compare experiments/evals/sub1m/sub1m_r0_s42 \
              experiments/evals/sub1m/sub1m_r0_s42 \
    --prefix ret_test_emb/
```

出现任何非零 Δ 或非零宽 CI ⇒ 配对下标没对齐(npz 里 i2t/t2i 顺序被打乱、或两侧
查询数不一致),**此时所有 A/B 结论作废**,先修再比。这一步零 GPU、约 10 秒,
`tests/test_paired_bootstrap.py` 里也钉了同一条不变式。
### 3.10 C1 教师嵌入离线提取(ROADMAP 2.2 / 评审 U7)—— 主线,GPU

`experiments/extract_teacher.py`。教师 = **InternVideo2-stage2**(U7-4 定案;换 Qwen 已否决,
换 IV3/IV2.5/Next 均无对比空间不可用)。学生训练时只读盘,不跑教师。

**先跑自检(零 GPU、零权重、零视频;接线前必做)**:
```bash
python experiments/extract_teacher.py self-test
python -m pytest tests/test_extract_teacher.py -q      # 44 项
```

**第一步永远是 100k 成本切片** —— U7-1 的语料范围(curated 2–3M vs 全量 10M)**至今未定**,
定它的方法就是先用切片实测吞吐再外推,别一上来就全量:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/extract_teacher.py run \
  --anno data/anno_pretrain/internvid_v1.jsonl \
  --out  experiments/teacher/iv2s2_100k \
  --teacher-ckpt checkpoints/InternVideo2-stage2_1b-224p-f4.pt \
  --limit 100000 --batch-size 8 --num-workers 8 --amp bf16
```
日志每块打 `clip/s`;拿它乘语料规模就是两档的 GPU 时数,决策留到数据出来那天。

**全语料按 shard 并行**(每卡一个 shard,彼此不通信,挂了各自续跑;块已存在即跳过):
```bash
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup python experiments/extract_teacher.py run \
    --anno data/anno_pretrain/internvid_v1.jsonl \
    --out  experiments/teacher/iv2s2_full \
    --teacher-ckpt checkpoints/InternVideo2-stage2_1b-224p-f4.pt \
    --shard-id $k --num-shards 4 --batch-size 8 --num-workers 8 \
    > experiments/teacher/extract_$k.log 2>&1 &
done
```

**逐帧 + 低 T 子集嵌入(U7-2 的「保险」)** —— 加 `--per-frame`。只多几次极廉价的池化,
**不是** T× 成本(实现见脚本 `ProjectorCapture`);存储按 IV2-1B 的 D=768 计
≈ `16×768×2B` ≈ **24 KB/片**(U7 里写的 ~45KB 是按更大的 D 估的):
```bash
python experiments/extract_teacher.py run ... --per-frame --subset-ts 2,4,8
```

**合并 + 体检**(体检不过就别往下接):
```bash
python experiments/extract_teacher.py merge  --out experiments/teacher/iv2s2_full
python experiments/extract_teacher.py verify --out experiments/teacher/iv2s2_full
```
`verify` 关注三格:`rows_unique`(行号唯一)、`fail_rate`(解码失败率)、
**`zero_emb_but_ok`**(标了成功却是零向量 ⇒ **教师前向出问题**,不是解码问题,必须查)。

⚠️ **三条容易踩的**:
1. **失败条保留占位行、不删** —— 删行会让 `rows` 与嵌入静默错位,下游只表现为
   「蒸馏没效果」。单块失败率超 `--fail-rate-abort`(默认 20%)直接抛错,不写半空的块。
2. **归一化用 ImageNet 那套**(0.485/0.456/0.406),不是 CLIP 那套 —— 依据是仓库自己的
   分派规则 `dataset/__init__.py:get_train_transform`。用错会整体偏移而 loss 照降。
3. **文本阶段(`--stage text`)依赖重** —— IV2-stage2 文本塔是 LLaMA 系,要 `peft`;
   仓库刻意不 eager import 它。视频阶段完全不碰这条链,先把视频跑完再说。

### 3.11 架构第四轮 · 零 GPU 闸门(`notes/architecture-round4.md` §3)

> ⚠️ **2026-07-29 首跑修订。** 本节最初给的是几段 heredoc,在服务器上**三条全废**,
> 其中一条是**静默失败**:`layer_scale_1d = 0` 匹配到 0 个键却照样存了 ckpt ⇒ `_ablate.pt`
> 与原文件逐位相同,评测会「通过」但什么都没消融。
> **根因(实测更正)**:~~发布权重是重参数化(推理态)~~ —— **错**。`inspect` 实测是
> **训练态**(40 个 `layer_scale_1d`、80 个 `layer_scale`、4 个 `temporal_embedding`、
> **0** 个 `reparam_conv*`)。真正的原因是 heredoc 写了 `sd.get("model", sd)`,而这份 ckpt
> 的顶层包装键**不叫 `model`**,于是回退到原始 dict、去遍历 `epoch`/`optimizer` 了。
> 现在全部走 `experiments/gate_checks.py`:**先探测形态,匹配到 0 个键立刻抛错、绝不落盘。**

```bash
# 0) 永远先跑这个 —— 看清 ckpt 是训练态还是推理态、各类键各有多少
python experiments/gate_checks.py inspect --ckpt checkpoints/mobileviclip_small.pt

# ⑤ INT8 会不会把融合核里的时序项抹掉(~10 s,零 GPU)
python experiments/gate_checks.py gate5 --ckpt checkpoints/mobileviclip_small.pt

# ③ 删时序分支 = 融合核置为单位核 δ、bias 置零(等价 layer_scale_1d=0,也是该支路的精确 INT8 模拟)
python experiments/gate_checks.py gate3 \
    --ckpt checkpoints/mobileviclip_small.pt --out checkpoints/_ablate_temporal.pt

# ④ 置零 temporal_embedding
python experiments/gate_checks.py gate4 \
    --ckpt checkpoints/mobileviclip_small.pt --out checkpoints/_ablate_tempemb.pt

# ① 空间参数是否与图像 ckpt 逐位相同(⇒ 骨干是否被冻 ⇒ GATE-0 腿②③ 是否混淆)
python experiments/gate_checks.py gate1 \
    --ckpt checkpoints/mobileviclip_small.pt --image-ckpt checkpoints/mobileclip_s2.pt

# 自检(零依赖、合成权重,改完脚本先跑这个)
python experiments/gate_checks.py self-test
```

③④ 产出的 ckpt 按下面的评测跑 MSR-VTT。**判据**:③ Δr_mean > 1.0(raw 与 DSL 同向)
才保留该支路;④ Δ > 0.5 才保留 embedding。

> 🔴 **每个臂必须给独立的 `output_dir`。** 首跑三个臂全写进同一个目录,
> `ranks_ret_test{,_emb}.npz`(配对 bootstrap 的输入)被后一个臂覆盖,只剩最后一个臂的秩,
> 配对 CI 因此做不了。下面的命令已带 `output_dir` 覆盖 ——
> ✅ **2026-07-30 已按此重跑,`eval_base/`、`eval_gate3/`、`eval_gate4/` 各自的两个
> `ranks_*.npz` 都已落地,配对 CI 现在可跑**(命令见本节末尾)。

```bash
export PYTHONPATH=${PYTHONPATH}:. ; export CUDA_VISIBLE_DEVICES=0
C=scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py
run () {  # run <臂名> <ckpt>
  python -m torch.distributed.run --nproc_per_node=1 --master_port=$((10000+RANDOM%20000)) \
      tasks_clip/retrieval.py "$C" pretrained_path "$2" output_dir "eval_$1"
}
run base    ./checkpoints/mobileviclip_small.pt
run gate3   ./checkpoints/_ablate_temporal.pt
run gate4   ./checkpoints/_ablate_tempemb.pt
```

**实测结果(2026-07-29 首跑;2026-07-30 分臂重跑**逐位复现**):**

> ⚠️ **2026-07-30 更正:本表首版把两个口径写反了。** `tasks_clip/retrieval_utils.py:154-160`
> 返回 `(dsl, dsl_T, raw, raw_T)`,`:84-85` 把第 1 位映射到 `ret_test/`、第 3 位映射到
> `ret_test_emb/` ⇒ **`ret_test/` 是 DSL、`ret_test_emb/` 是 raw,而 raw 才是论文主口径**
> (§3.7 / `experiments/README.md` §6.1 / `experiments/gate0_results.md` §1 一直是对的,
> 写反的只有本节和 `notes/architecture-round4.md` §3.0)。**判定不变**(两口径同向),
> 变的是**主口径的 Δ 反而更大**:③ −0.43、④ +0.70。

| 臂 | r_mean raw(`_emb/`,主口径) | r_mean DSL(`ret_test/`) | Δ raw(95% CI) | Δ DSL(95% CI) | 判定 |
|---|---|---|---|---|---|
| base | 61.55 | 62.28 | — | — | 与文档记载逐位一致 |
| ③ 时序支路置零 | 61.12 | 61.95 | **−0.43** [−0.82, −0.03] 显著 | −0.33 [−0.72, +0.05] n.s. | **CI 整段都在 1.0 以下**(两口径皆然)⇒ 支路上界 0.82,判据要 >1.0 ⇒ **不保留** |
| ④ `temporal_embedding` 置零 | **62.25** | **62.95** | **+0.70** [+0.20, +1.20] 显著 | **+0.67** [+0.17, +1.15] 显著 | 两口径**都显著** ⇒ 删掉**确实更好**(r_mean 口径)⇒ **不保留** |

> ✅ **配对 CI 已跑(2026-07-30,10000 次重采样,seed 0)**,上表 CI 即其输出。
> ③ 不是「没差别」而是**等价性结论**:支路有作用,但 95% CI 上界 0.82 < 判据 1.0。
> ④ 「删了更好」现在**站得住**(r_mean 两口径同时显著)。
>
> ⚠️ **但主指标 t2v R@1 没过两口径判据**:④ 的 `img_r1` raw **+1.70** [+0.40, +3.10] 显著,
> DSL 仅 +1.30 [−0.10, +2.70] **不显著**。可写的结论是「**r_mean 显著提升**」,
> **不能**写「t2v R@1 显著提升」。
>
> 📏 **本次实测配对分辨率**:R-Sum 95% CI 半宽 配对 **±2.30~3.00** vs 无配对 ±14.50~14.70
> ⇒ **约 4.9~6.3 倍**。G4 记录的「约 11 倍(±1.5 vs ±17)」来自另一对耦合更紧的臂 ——
> **把 5~11 倍当作可达区间,而不是把 11 倍当常数。**

**配对 bootstrap 命令(三个臂的 `ranks_*.npz` 都已就位):**

```bash
# 主口径(raw = ret_test_emb/,--prefix 的默认值)
python experiments/collect_results.py --compare eval_gate3 eval_base
python experiments/collect_results.py --compare eval_gate4 eval_base
# 第二口径(DSL)—— 两口径同向才算数
python experiments/collect_results.py --compare eval_gate3 eval_base --prefix 'ret_test/'
python experiments/collect_results.py --compare eval_gate4 eval_base --prefix 'ret_test/'
```
> ⚠️ MSR-VTT 的字幕以外观为主,是弱时序探针。**`ssv2_mc_val` 才是能证伪「删 `temporal_embedding`」
> 的基准**,批准前先跑它。

```bash
# ⑤-fused:重参数化之后量 |w_1d − δ| —— 这才是 P2 的 INT8 直接判据(λ 本身不是)
# ✅ 2026-07-30 已跑,结论:P2 机制被证伪(见下方红框)
python experiments/gate_checks.py gate5-fused \
    --ckpt checkpoints/mobileviclip_small.pt --config "$C"
```

> ⚠️ **`gate1` 可能给出「无法判定」** —— 若两份 ckpt 一份推理态一份训练态,键空间对不上。
> 脚本会**如实报告并退出码 2**,而不是打印 `0 / 0` 假装成结论(首跑就踩了这个)。
> 实测结果:80 个可比空间张量**逐位相同 0 个** ⇒ 空间塔确实训过 ⇒ GATE-0 腿②③ 前提成立。

> ✅ **λ 的「未核实」标记已撤回**:`gate5` 从发布权重里逐位复现出
> `layer_scale_1d` median **7.973e-04** / p95 **2.762e-03**,与 §2 P2 引用的 8e-4 / 2.8e-3 吻合。
> 但 **λ 本身不是 INT8 判据**(融合核是 `δ + λ·Δ`,要比的是 `λ·|Δ|`)—— 这个方法学判断被
> `gate5-fused` 证实了,而且差得很远。
>
> 🔴 **`gate5-fused` 结果(2026-07-30):P2 的 INT8 机制被证伪。** 融合后 `|w_1d − δ|`
> 逐通道最大值(n=9920):p05 4.695e-04 / p25 2.531e-03 / **p50 6.178e-03** / p75 1.252e-02 /
> p95 2.880e-02。INT8 抹零阈值 **3.937e-03**,**只有 36.2% 的通道被抹零,中位数比阈值高 1.57 倍**
> ⇒ **时序项在 INT8 下存活(约 64% 通道)**,不是「低于 LSB 被删掉」。
> **失效原因**:P2 用 `λ·|Δ|` 估算时假设 BN 折叠后的 depthwise 核是 O(0.1–1),
> 实测反推 `|Δ| ≈ 7.7`,**差了一到两个数量级**。λ 是对的,乘的那一项错了。
> **连带作废**:①「INT8 抹掉时序支路」机制;②「AC5/AC9 在 INT8 被兑现掉」的 pillar-2 反转;
> ③「`layer_scale_1d = 0` 就是精确的 INT8 模拟」(它比 INT8 更狠);
> ④ 口诀 `SNR ∝ 127·λ` 应为 **`SNR ∝ 127·λ·|Δ|`**。
> **仍成立**:该支路不值得保留 —— 但依据换成消融(上界 0.82 r_mean vs 27.8% ONNX 图),
> 不再是量化。详见 `notes/architecture-round4.md` §2 P2 的改写。
>
> ✅ **~~「对不上的数」~~ 已撤回(2026-07-30):6.1% 是对的,错的是我的 `gate5`。**
> 两个脚本的**分母**取法不同:
> - `temporal_capacity_audit.py:178` 只取 **`token_mixer.layer_scale`**(40 个)——
>   它缩放的是**空间 mixer** 支路,与 `layer_scale_1d` 缩放时序 mixer 支路**一一对应**。
>   中位 ≈ **1.3e-2**(正是 round4 §2 P2 独立引用的那个数)⇒ 7.973e-04 / 1.3e-2 = **6.1%** ✅
> - 我 07-29 版的 `gate5` 池化了**全部 80 个** `layer_scale`,把**块级**(缩放 ConvFFN
>   残差,另一个模块)也算了进去 ⇒ 分母被抬到 3.296e-02 ⇒ 假的 **2.42%**。
>
> **腿③ 原样成立。** `gate5` 已改:主报可比比值,池化比值只作**明确标注为不可比**的参考。
>
> ✅ **2026-07-30 服务器实测确认**:`temporal_capacity_audit.py` 输出
> **闸门比 = 0.0613**,时序中位 0.0008、空间中位 **0.0130**,且**两侧 n 都是 9920**
> (40 块 × 通道)⇒ 逐通道一一对应,这就是「可比」的定义。
> 该脚本 2026-07-24 的全部输出本次**逐位复现**,见 `experiments/gate0_results.md` §7。
>
> ✅ **2026-07-30 交叉验证:改好的 `gate5` 跑出同一个数。**
> `layer_scale_1d (时序 mixer) n=9920 median=7.973e-04` /
> `layer_scale (空间 mixer) n=9920 median=1.300e-02` ⇒ **0.0613** ——
> 与静态审计**由完全独立的代码路径**得到、四位小数一致;空间中位 1.300e-02 也正是
> §2 P2 一开始就引用的 1.3e-2。**三个来源、一个数。**
> 池化值仍会打印但标注为「**勿用于腿③** … 0.0242 … 混进了块级 ConvFFN 门」,
> ⇒ 这个错误现在被记录在**工具自己的输出里**,不只是散文里。

**② 训练到底卡在哪?(30 分钟,零 GPU 训练)** —— ✅ **已定案:不是 dataloader**(见下),
剩下的是 GPU 侧。**全部在 b100(H100)上跑** —— 101vector 是 8×V100 且没有训练数据,见下 ③:

> 🔴 **2026-07-30 修正:本节原先的两段 heredoc 都是坏的,一段都跑不了。**
> ① 解码那段读 `data/anno_pretrain/internvid_v1.jsonl` —— **该路径不存在**,真实标注是
> `anno_downstream/InternVid/InternVid-10M-flt_1.json`(**JSON 列表**,不是 JSONL),
> 且记录里的文件名键是 **`image`**(本仓库 video 也用 `image`,见
> `dataset/base_dataset.py:139-156`),还得和 `data_root` 拼起来才是完整路径。
> ② 吞吐那段调 `setup_main()` —— 它会**读 argv 里的 config** 并执行
> `init_distributed_mode()` + `dist.barrier()`,**裸 `python` / heredoc 根本起不来**。
> 现已合并为一个受版本管理的脚本 `experiments/probe_decode_bound.py`(A/B 两段都在里面,
> 路径全部从 config 自己解析,不硬编码)。

> ✅ **2026-07-30 定案(第二次实测,四段版):`dataloader 根本不是约束`。**
>
> | 段 | 实测(06:24) | 含义 |
> |---|---|---|
> | A 热解码 | ~20 ms/片 | 解码本身很快 |
> | A2 冷读 | I/O 占冷解码 **≈0%**,冷/热 1.2× | **挂载盘不慢** ⇒ H2 死 |
> | B loader-only(单卡 6 workers) | **132.7 样本/s** | 训练每卡只消费 **13.9** ⇒ **富余 9.5×** |
> | C 扫描 1→24 workers | 7.5 → 324.6 样本/s,**没饱和** | 机器 **192 核 / 可用 176** ⇒ H1 死 |
>
> ⇒ **`notes/architecture-round4.md` §0.2「训练是解码瓶颈」不成立。**
> MFU 0.67% 仍是真的,但原因在 **GPU 侧/训练循环**(depthwise 小核的 kernel-launch、
> 低占用率、DeepSpeed/通信),不在读数据。「6.6 天/轮」大概率仍可大幅压缩,
> 但**不是靠 `num_workers`、也不是靠预解码**。
>
> 🔴 **同一天里这个结论被推翻过两次,两次都是方法错,记在这里防第三次:**
> 1. **只比一个基准 ⇒ 判反。** 第一版把「单卡 6 workers 的 loader」直接和
>    「4 卡 24 workers 的训练」相比,算出「workers ×4 只换 +14%」,据此宣布
>    「共享资源饱和、加 workers 没用」。两边 workers 总数不同,**根本不可比**。
>    真扫描一跑,1→24 workers 一路线性到底,没有任何饱和。
> 2. **预热没排空 prefetch 队列 ⇒ 高档位虚高。** DataLoader 预取
>    `num_workers × prefetch_factor`(默认 2)个 batch;第一版预热 1 个 batch、
>    最多测 40 个,而 24 workers 的队列有 48 个 ⇒ 几乎全在吃存货,测出 **43× 超线性**、
>    每 worker 吞吐反随 workers 上升(物理上讲不通)。脚本现已预热
>    `2 × workers × prefetch_factor`,并对超线性直接打警告 —— 所以 324.6 那个数
>    **别引用**,只引用「还在爬、没饱和」这个方向。

`nproc` 不用单独跑 —— 脚本末尾会打印 `os.cpu_count()` 和**本进程实际可用核**
(容器里两者常常不一样,`nproc` 只告诉你其中一个;本机 192 / 176)。

```bash
# A 热解码 + A2 冷读 + B loader-only + C workers 扫描 —— 一个脚本跑完,自带交叉判读
export PYTHONPATH=${PYTHONPATH}:. ; export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.run --nproc_per_node=1 --master_port=$((10000+RANDOM%20000)) \
    experiments/probe_decode_bound.py \
    scripts/pretraining/clip/mobileviclip_small/config.py \
    output_dir probe_decode_bound
```

> 必须走 `torch.distributed.run`(理由见上 ②)。约 15 分钟。可选参数:
> `--batches N`(B 段迭代多少 batch,默认 50)、`--sweep 1,2,6,12,24`(C 段档位,
> 给 `none` 跳过)、`--cold-files N`(A2 每组取几个互不相同的文件,默认 25)。
> **任何一段失败都不挡后面的段。** 判读只有 **④** 定瓶颈:比较「同样 workers 下
> loader 供得起多少」与「训练每卡实际消费多少」,富余 ≥2× 就说明输入管线无罪。

**③ 证伪 / 证实「dataloader 无罪」—— 真训练(必须在 b100/H100 上跑)**

> ❌ **不能拿 101vector(V100)代跑**,两条独立的硬理由:
> 1. **那台机器没有训练数据。** `tools/data_audit/SETUP_V100.md` 开篇即写:本机只有
>    `/home/vca1/datasets/internvid_zh_10k` 这 1 万条抽样,**11 个 InternVid 训练分片一个都没有**
>    (7 个下游评测基准的视频也都没有,见 TROUBLESHOOTING「这台机器上根本没有下游评测数据」)。
>    `config.py` 的 `train_file` 指向 `data/video_sample/InternVId-FLT_*`,在 V100 上直接建不出 dataset。
> 2. **要检验的那个数本身是 H100 的数。** 基准 9.1821 s/it、MFU 0.67% 都是 4×H100 上测的
>    (峰值 ~990 TFLOP/s bf16);V100 是 sm70、无 bf16、fp16 峰值 ~125 TFLOP/s,差约 **8×**。
>    同样的 loader 吞吐在 V100 上算出来的 MFU 会高一个量级,**V100 的 s/it 既证实不了也证伪不了 H100 的结论**。
>
> V100 能做的是另一件事:用 `internvid_zh_10k_pilot` 语料给**那台机器自己**做一份
> 解码/IO 画像(`train_corpus internvid_zh_10k_pilot`,并先确认 `data/internvid_zh_10k/videos`
> 指向 `/home/vca1/datasets/internvid_zh_10k`)。那是「以后能不能挪到 V100 训」的问题,**不是闸②**。

> ⚠️ **2026-07-30 首次尝试直接崩了,先 `git pull` 再跑。** 基线 config 的
> `test_file` 键是 `act_val` 而 `test_types = ["ret_test"]`,2026-07-29 加的守卫
> (`pretrain.py:248`)会在第一个 step 之前 `raise ValueError` ⇒ **预训练当时谁都跑不起来**。
> 两个基线 config(`mobileviclip_small` / `_tiny`)已改成 `dict(ret_test=…)`,
> 见 TROUBLESHOOTING「训练跑完但没有任何评测数字」。

🔴 **别和历史基准比 —— b100 是共享机。** 8 张 H100 常年被别的租户占到 ~87%
(TROUBLESHOOTING「b100 is a shared, multi-tenant node」),而 `9.1821 s/it` 是**另一天**、
未知负载下测的。拿今天的 `num_workers=16` 去比它,等于把机器负载差异算到 workers 头上 ——
**这正是本节 ② 刚刚栽过的「只和一个陈旧基准比」那个错。**

所以**两档要背靠背在同一个会话里各跑一次**,顺序 6 → 16,中间不要隔夜:

```bash
# 先记录此刻的争抢情况(两档各跑前都记一次,便于事后判断负载是否变了)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv

export PYTHONPATH=${PYTHONPATH}:. ; export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ① 复现基准档(log_freq=1,每步打 s/it;看到十来步就 Ctrl-C)
python -m torch.distributed.run --nproc_per_node=4 --master_port=$((10000+RANDOM%20000)) \
    tasks_clip/pretrain.py scripts/pretraining/clip/mobileviclip_small/config.py \
    num_workers 6 output_dir probe_train_nw6

# ② 立刻再跑高 workers 档
python -m torch.distributed.run --nproc_per_node=4 --master_port=$((10000+RANDOM%20000)) \
    tasks_clip/pretrain.py scripts/pretraining/clip/mobileviclip_small/config.py \
    num_workers 16 output_dir probe_train_nw16
```

另开一个终端,在**任一档**跑的时候看 GPU(**两种读法相反,别只记一种**):

```bash
nvidia-smi dmon -s u -c 60
```

| sm 列 | 读法 |
|---|---|
| 持续 **<25%** | GPU 在等数据/等同步 ⇒ 还是喂不饱(输入或通信) |
| 持续 **>70% 而 MFU 仍 0.67%** | 核在跑但几乎不干活 ⇒ **kernel-launch / 低占用率**,这才是要修的 |

**判读**:比的是**今天这两档之间**的 s/it,不是和 9.1821 比。
两档基本持平 ⇒ **坐实 loader 无罪**,瓶颈在 GPU 侧;
16 档明显更快 ⇒ 单卡外推不成立(4 卡会互抢 CPU),回到调 workers 那条路。
历史基准 9.1821 s/it 只用来**旁证机器负载**:若今天的 6 档和它差很多,说明当时/现在的争抢不同,
更要以今天的两档对比为准。

**⑥⑦⑧** 仍见 `notes/architecture-round4.md` §3(⑥ 需让 `encode_vision` 别提前折叠 T,
打分路径 `get_sim` 已支持;⑦ 视觉塔分辨率无关,改 `img_size` 免插值免重训;
⑧ 随机权重即可导出计时,**延迟与权重无关**)。

> ⚠️ 实测结论,值得单独记一句:**时序分支 = 0.20% MACs 却占 27.8% 的 ONNX 图**、
> 18% 的 CPU mixer 墙钟(约 75% 是 permute 而非算术)。
> **这个仓库以后判效率,数的是全特征图 transpose 的个数,不是 FLOPs。**

### 3.12 SSv2-MC —— 唯一能证伪「删 `temporal_embedding`」的时序基准

> 🔴 **入口不是 `retrieval.py`,是 `tasks_clip/retrieval_mc.py`。**
> SSv2-MC 是 **174 选 1 的多选准确率**任务,不是检索。2026-07-30 我给过一条用
> `retrieval.py` 的命令,**那是错的** —— 即使数据齐了也跑不出 acc。
> 数据获取(标注 + 视频,视频有许可协议)见 [`CHECKPOINTS.md`](CHECKPOINTS.md) 的
> 「SSv2-MC」小节;`anno_downstream/ssv2_mc_val.json` 缺失时报
> `FileNotFoundError`,见 `TROUBLESHOOTING.md`。

**判据**:MSR-VTT 上「删 `temporal_embedding` 使 r_mean 显著提升」的结论,
在 SSv2-MC 上**至少不能显著变差**。这里若显著掉分,说明那个提升是 MSR-VTT
外观偏置的产物,round4 §3.0 闸④ 的结论必须收回。
上游参考:`mobileviclip_small` **acc 46.07**(随机基线 174 选 1 = 0.57%)。

```bash
export PYTHONPATH=${PYTHONPATH}:. ; export CUDA_VISIBLE_DEVICES=0
S=scripts/evaluation/clip/zero_shot/mobileviclip_small/config_ssv2_mc.py
runmc () {  # runmc <臂名> <ckpt>   —— 注意每个臂独立 output_dir(否则逐题结果被覆盖)
  python -m torch.distributed.run --nproc_per_node=1 --master_port=$((10000+RANDOM%20000)) \
      tasks_clip/retrieval_mc.py "$S" pretrained_path "$2" output_dir "ssv2_$1"
}
runmc base  ./checkpoints/mobileviclip_small.pt
runmc gate4 ./checkpoints/_ablate_tempemb.pt
runmc gate3 ./checkpoints/_ablate_temporal.pt   # 可选:时序支路在时序任务上值多少
```

**配对 CI(多选口径,2026-07-30 新增)** —— `retrieval_mc.py` 现在会把**逐题 0/1
正确性**存到 `<out_dir>/mc_correct.npz`(~2 KB),因此多选也能做配对比较:

```bash
python experiments/collect_results.py --compare-mc ssv2_gate4 ssv2_base
python experiments/collect_results.py --compare-mc ssv2_gate3 ssv2_base
```

> 为什么必须配对:两条臂答的是**同一批题**,裸 acc 相减会把共享方差算进去。
> 合成数据上实测配对半宽 ±0.72 vs 无配对 ±2.99(**4.2×**)。
> 单测见 `tests/test_paired_bootstrap.py`(多选部分 7 项,共 18 项)。

### 3.13 环境实况自检 + SDPA 可用性(零 GPU,~10 秒;round-5 §8.1)

**为什么要跑**:仓库多处文档按 `mobileviclip.yml` 的 **torch 1.13.1** 推理,断言
「SDPA 用不了、P1 被环境卡住」。**那个 pin 描述的机器一台都不存在** —— 两台 GPU 机
都是 **2.5.1**。下面这条命令是**一次性证实/证伪**,请在**两台机器上各跑一次**并把输出
贴回 `notes/live/from-server.md`。

```bash
# 两台机器各跑一次(b100 先 source experiments/env_b100.sh)
python - <<'PY'
import torch, torch.nn.functional as F
print("torch      :", torch.__version__)
print("arch_list  :", torch.cuda.get_arch_list() if torch.cuda.is_available() else "no cuda")
print("has SDPA   :", hasattr(F, "scaled_dot_product_attention"))
print("has compile:", hasattr(torch, "compile"))
PY
```

**判读**:`has SDPA = True` ⇒ **P1 不再有环境阻塞**,`architecture-round4.md` §5 里
「torch 1.13.1 没有 SDPA、只能手写分块」那句作废(已划掉),`paper-plan` §六.8 的
降级理由也只剩半条。`arch_list` 含 `sm_90` = H100 走完整 Flash 路径;只含 `sm_70`
= V100 走 memory-efficient 内核(仍然消掉 N×N 物化)。

**替换本身是数学恒等的**,本地 float64 已按本模型 stage-4 真实形状(N = T·H·W = 512,
20 头)验过,`max|朴素 − SDPA| = 2.08e-16`。复现:

```bash
python - <<'PY'
import torch, torch.nn.functional as F
torch.manual_seed(0)
B,T,H,W,C,hd = 2,8,8,8,640,32
N, nh, scale = T*H*W, C//32, 32**-0.5
x  = torch.randn(B,N,C, dtype=torch.float64)
qw = torch.randn(C*3,C, dtype=torch.float64)*0.02
q,k,v = (x@qw.T).reshape(B,N,3,nh,hd).permute(2,0,3,1,4).unbind(0)
ref = (((q*scale)@k.transpose(-2,-1)).softmax(-1)@v).transpose(1,2).reshape(B,N,C)
out = F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(B,N,C)
print("max|naive - SDPA| =", (ref-out).abs().max().item())
PY
```

⚠️ **落地时必须加版本护栏**:`mobileviclip.yml` 的 1.13.1 CPU 配方仍是 import 冒烟的
基准,所以走 `if hasattr(F, "scaled_dot_product_attention"): ... else: <现有朴素路径>`,
不要无条件调用。开关沿用仓库既有的 `use_flash_attn` 旗标(现在被接进来却**从不读**)。

### 3.14 P11 逐块深度敏感度扫描(零训练,~1 GPU-小时;round-5 §8.3 P11)

**目的**:`architecture-round4.md` §8.2 证明**每个 RepMixerBlock 的 MACs 完全相同
(157.29 M = 逐帧 2.04%)**,「stage 3 占一半」其实只是 `layers[2]=24`。⇒ 删**深度**是
第一次真正作用在 86.5% 主导项上的杠杆,而且**可以零训练地量出每一块值多少分**。

> ✅ **已实现(2026-07-31)。** `gate_checks.py gate6-dropblock` 现在存在,并复用了
> 「探模式 → 不匹配就抛错、绝不落盘」的骨架:它断言**恰好命中 3 个门**、断言**恰好 40 个块**、
> 拒绝在**重参数化态**权重上运行,`--block` 越界直接退出。

**要置零的张量(每块 3 个,少一个就不是恒等)**:

| # | 后缀 | 作用 | 数量(全模型) |
|---|---|---|---|
| 1 | `token_mixer.layer_scale_1d` | 时序 mixer 残差门 | 40 |
| 2 | `token_mixer.layer_scale` | **空间** mixer 残差门 | 40 |
| 3 | `layer_scale`(块级,**不带** `token_mixer.`) | ConvFFN 残差门 | 40 |

依据(已核源码):`RepMixer.forward:743-746` = `x + λ_1d·(mixer_1d−norm_1d)` 然后
`x + λ·(mixer−norm)`;`RepMixerBlock.forward:1075` = `token_mixer(x)` 然后
`x + drop_path(layer_scale·convffn(x))`。三者全零 ⇒ 该块**精确恒等**。

> 🔴 **必须写进断言的坑**:全模型 `layer_scale` 共 **80** 个 = 40 个
> `token_mixer.layer_scale`(空间 mixer)+ 40 个**块级** ConvFFN 门。**这正是 §0.1 里
> `gate5` 分母取错、凭空造出 2.42% 的同一个坑。** 子命令要 `assert` 每块恰好命中 3 个键,
> 并把命中的键名打出来。先看一眼真实键名:
>
> ```bash
> python - <<'PY'
> import torch, collections
> sd = torch.load("checkpoints/mobileviclip_small.pt", map_location="cpu")
> sd = sd.get("module", sd.get("model", sd))
> c = collections.Counter()
> for k in sd:
>     if k.endswith("layer_scale_1d"):            c["token_mixer.layer_scale_1d"] += 1
>     elif k.endswith("token_mixer.layer_scale"): c["token_mixer.layer_scale"] += 1
>     elif k.endswith("layer_scale"):             c["block-level layer_scale"] += 1
> print(c)   # 期望 40 / 40 / 40
> PY
> ```
>
> ⚠️ 顶层 wrapper 键**不是** `model`(§3 开头那次静默 no-op 就是踩了这个)——先按上面这样探。

**扫描循环(40 块;stage 4 的 4 个 `AttentionBlock` 门结构不同、`:1111-1188` **未核**,先别纳入)**:

**用脚本跑,不要手写 for 循环:**

```bash
bash experiments/run_dropblock_sweep.sh            # 全 40 块
bash experiments/run_dropblock_sweep.sh 0 9        # 只跑 0..9
```

终端**每臂只出一行**(块号 / 状态 / raw r_mean / DSL r_mean),完整输出落
`logs/dropblock/b<i>.log`;**第一个臂失败就停**;已完成的臂**自动跳过**,断了重跑同一条命令即可。

> 🔴 **2026-07-31 修正:本节初版给的循环有两个错,别再用。**
> ① `gate6-dropblock` 当时**还不存在** ⇒ 40 次全是 argparse 报错。
> ② 更隐蔽:**`eval_msrvtt.sh` 不接受任何位置参数** —— 它把 `output_dir` 写死在脚本里。
> 所以 `eval_msrvtt.sh <ckpt> <tag>` 会**忽略那个 ckpt**(评的其实是配置里的基线权重)
> 并让 **40 个臂全写进同一个目录**,`ranks_*.npz` 互相覆盖 ⇒ **配对 CI 直接做不了**
> (正是 §3.9 那个坑)。正确做法是覆盖 **`pretrained_path` + `output_dir`** 两个键,
> 与 §3.11 的 `run ()` 一致 —— 脚本里已经这么做了。

**判读**(⚠️ 上表的裸差**不是**判据):

```bash
# 先确保基线臂存在(§3.11 的 run base ...),再逐块配对
python experiments/collect_results.py --compare eval_drop_b7 eval_base
```

**两条强制纪律**:

1. **删块会改变下游每个 BN 的输入分布** ⇒ **评测前必须做一次前向 BN 重校准**
   (`experiments/per_t_bn.py` 已有这套机器,round4-P7 的分辨率扫描也要用同一套)。
   不做的话量到的是分布漂移,不是这块的价值。
2. **判读用配对 bootstrap(§3.9),不要看裸差**;两口径(raw `ret_test_emb/` +
   DSL `ret_test/`)同向才算(house rule)。

**判据(预先登记)**:若**没有任何单块**的代价 < +1.0 r_mean,或**最便宜的 6 块之和 > 2.0
r_mean** ⇒ 这份权重里深度不冗余,**关掉这条线**,预算全给 round4-P7(分辨率)。
反之:按 Δr_mean/Δms 贪心删,并与 round4-P7 组成 **(分辨率 × 深度) 二维等延迟前沿**
—— 两者都保算术强度(§8.2b),省下的延迟可以相乘。

**顺带产出**:这条曲线本身就是论文的 Fig-1 级分析(「一个蒸馏出来的移动端视频 CLIP
把算力放在哪、每一块买到了什么」),**零训练轮次**。

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
python -m torch.distributed.run --nproc_per_node=8 --master_port=$((11000 + RANDOM % 20000)) \
    tasks_clip/pretrain.py \
    scripts/pretraining/clip/mobileviclip_tiny/config.py
```

### 长跑运维:选空卡、后台挂、ETA、断点续训(b100 共用节点)

**① 选卡:看 `Memory-Used` 挑 0 MiB 的,不要看 `GPU-Util`。** 共用节点上其他租户
的进程在别的 PID namespace 里,`nvidia-smi` 进程表显示 `No running processes found`,
但他们的显存/算力占用是真的。`0% util ≠ 空闲`(util 只是采样瞬间有没有 kernel);
`0 MiB used` 才是真空。挑最空的 N 张:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=<挑出来的空卡,如 4,5,6,7>   # 传给下面的启动命令
```

**共用/争抢的判据(很重要):** 训练日志里的 `time:` 每步耗时,若对 4×H100 明显
偏慢(这个小模型正常应 ~1–3 s/step,出现 ~10 s/step 就是异常),同时 `data:` 接近 0
(排除数据瓶颈),就是那几张卡被**超额占用**了。两种来源,同一个「慢 + 显存对不上」
的征兆:

- **别的租户共用了同几张卡**,或
- **你自己把同一个 R0 启动了两遍**(最常见!),或上一次崩溃残留的 dataloader 孤儿进程。

对照证据:日志 `max mem` 是你**单个 job 单卡**的占用(如 ~22 GiB),若远小于
`nvidia-smi` 在那几张卡上显示的总显存(如 ~56 GiB ≈ 2×),差额就是「多出来的那份」。
日志里若同一 step 号出现**两行、eta/时间不同**,几乎坐实是两个 job 在写同一个日志。

**先数一下到底有几个 job 在跑**,按进程组 PGID 归并。注意 `torch.distributed.run`
把**每个 rank 放进各自的进程组**,每个 rank 带 6 个 dataloader worker = 一组 **7 个**;
所以一个 4 卡 job = **4 个「7」的 rank 组 + 1 个「1」的 launcher 组**:

```bash
ps -eo pgid,etime,cmd | grep '[t]asks_clip/pretrain.py' | grep -v grep \
  | awk '{print $1}' | sort | uniq -c | sort -rn
# 典型输出:8 个「7」+ 2 个「1」= 两个 4 卡 job(按 PID 段分两簇,如 21189xx / 21194xx)
```

**⚠️ 共用 root 账号 → `ps`/user 分不出「谁的」(全是 root)。** 动手前必须按「在干什么」
正面确认每一簇是你的 R0,而不是别人的进程:

```bash
ps -eo pgid,pid,lstart,cmd | grep '[t]asks_clip/pretrain.py' | grep -v grep | sort -k1
ls -l /proc/<某簇一个PID>/cwd                 # 应是 .../rby/mobileclip
ls -l /proc/<该PID>/fd | grep -E 'runs/|\.log' # 看它写的是不是你的 output_dir/日志
```
最硬的证据:两簇是否都在写**同一个你的** `r0_baseline.log`(日志里同一 step 出现两行、
eta 不同即是)——别人的 job 不会往你的 output 里写。**任一簇的 cwd/config 不是你的就别碰。**

**确认是你的之后,只停其中一个 job**(该 job 的全部 PGID 一起停;负号 = 整个进程组;
不用 `pkill`,也别碰没确认的簇):

```bash
for g in <launcherPGID> <rank1PGID> <rank2PGID> <rank3PGID> <rank4PGID>; do
    kill -TERM -$g            # 注意每个 PGID 前的负号
done
```

→ 停掉多余的一个后,存活 job 不再分卡,`time:` 通常大幅下降(实测两 job 挤 4 卡
~10 s/step;砍到一个约 ~5 s;换到确认 `0 MiB` 的空卡跑单个 job 更快)。总时长从
~15 天降到个位数天。**教训:R0 只需一个 job,启动后务必记下唯一 PID,别重复启动。**

**② 后台挂 + 记 PID(共用 root 账号,停止只能按 PID,严禁 `pkill`/`killall`):**

```bash
mkdir -p experiments/runs
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 4 \
    use_bf16 True bf16 True \
    > experiments/runs/r0_baseline.log 2>&1 &
echo "R0 PID = $!"          # 记牢这个 PID
```

**③ 看 loss / ETA(注意 `it/s` 是加载标注的假信号,真步速看带 `eta:` 的行):**

```bash
grep -E "eta:|loss:" experiments/runs/r0_baseline.log | tail -5
# 估算:time/step × (约 41591 步/epoch,batch64×4卡) × epochs
```
`lr: 0.000000` 开头几百步是正常的(warmup);`video-loss_vtc` 在 ~0.6–1.3 抖动、
整体趋降即健康。

**④ 断点续训:进度一直在存,但裸重跑不会续、会覆盖。** 每 `save_iter`(500)步存
`ckpt_iter<step>.pth`,每个 epoch 末存 `ckpt_<epoch>.pth`,都是含
`model/optimizer/scheduler/scaler/epoch/global_step` 的完整快照,落在 `output_dir`
(默认 `experiments/runs/r0_baseline`)。要**接着**练必须加 `auto_resume True`、且用
**同一个 output_dir**:

```bash
CUDA_VISIBLE_DEVICES=<空卡> nohup bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 4 \
    use_bf16 True bf16 True auto_resume True \
    > experiments/runs/r0_baseline.log 2>&1 &
```

它会自动扫 `output_dir` 挑步数最大的 ckpt,恢复优化器/scheduler/步数接着跑。

> **⚠️ 陈旧 ckpt 陷阱:** `auto_resume` 若在 `output_dir` 里扫到某个 `ckpt_<E>.pth`
> 且 `(E+1)` ≥ 总 epoch 数,会令 `start_epoch = E+1`,训练循环 `range(start,epochs)`
> 直接为空 → **误判「已训完」立刻退出、什么都不做**。所以**开新一轮 R0 前,若旧
> `output_dir` 里残留上一次跑的 `ckpt_00/01/02.pth`,要么先挪走,要么换一个干净的
> `output_dir`**(如 `output_dir experiments/runs/r0_baseline_v2`,可作为覆盖项传入)。

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
