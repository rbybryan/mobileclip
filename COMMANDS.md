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
cd ~/mobileclip && git pull origin main
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

```bash
cd ~/mobileclip && git pull origin main
export MVC_DATA_DIR=/home/vca1/datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ① 建缓存(投影前文本特征 + caption_to_idx;单卡、纯前向、几分钟)。
#    建模型走的是 Part B 已验证的同一条路;若报错把完整 traceback 贴回。
python experiments/r1_build_textbank.py \
    --anno anno_downstream/internvid_zh_10k_pilot.json \
    --ckpt checkpoints/mobileviclip_small.pt \
    --out-feats experiments/cache/textbank_pilot_feats.fp16.npy \
    --out-idx   experiments/cache/textbank_pilot_idx.npy

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
