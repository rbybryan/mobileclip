# b100 主机全量预训练配置与执行手册（InternVid-10M-FLT）

与 V100 试点（[`../recaption/docs/setup-v100.md`](../recaption/docs/setup-v100.md) 用于审查、
[`README.md` §2b](README.md) 用于 10k 试点训练）**并行**的另一条线：在 b100 主机上跑
**全量 10M** 预训练（`internvid_all` 语料，R0→R1 等步 A/B）。两条线互不共享状态、
互不阻塞。**试点数字一律不外推**（见 `README.md` §2b）；全量这条才是真实实验，
按 [`experiments/README.md`](README.md) 第三线 3.2 的门控推进（R0 → R1 → R2 → R3）。

> **两级分工**（见 [`README.md`「两级实验」](README.md)）：V100/10k 只做每个方案的
> 链路/稳定性 go-no-go，**排不了序**（试点评测饱和 + 噪声）；**方案之间真正排序（A/B）
> 和胜者的终训都在这台机器上**。所以下面 §6 会把 R0 与 r1 各臂都跑在 H100 上。

> **⚠️ 先认清这台机器（否则装错 torch、开错卡）：**
> - 主机昵称叫 **`b100`**（挂载都是 `/workspace/mount/b100_zaip_data*`），但**GPU 是
>   8×H100-80G（Hopper / sm_90）**，不是 Blackwell。别和
>   [`../recaption/docs/setup-v100.md`](../recaption/docs/setup-v100.md) 里的 V100 审计机搞混：
>   那台用于数据审查，这台用于训练。
> - **默认只用前 4 张卡**（`CUDA_VISIBLE_DEVICES=0,1,2,3`，`NUM_GPUS=4`）——后 4 张留给别的任务。
>   要用满 8 张必须有维护者的**明确指示**。
> - **⚠️ 这台机器是共用 root 账号，杀进程只准按 PID，禁止 `pkill -f` / `killall` 之类的
>   模式匹配。** 同机上还有别的团队线在跑**同一个** `tasks_clip/pretrain.py`；`pkill -f "tasks_clip/pretrain.py"` 会把**别人的训练一起杀掉**。
>   正确姿势：记下自己启动时打印的 PID（`nohup ... &` 后那行 `[1] <PID>`），
>   用 `ps -p <PID> -o pid,etime,cmd` 确认还活着、用 `ls -l /proc/<PID>/cwd` 确认是自己的
>   目录，再 `kill <PID>`。同理 `nvidia-smi` 里别人的进程一律不要碰。
> - **⚠️ 找数据集只准在 `/workspace/mount/b100_zaip_data/video_datasets/` 里找**（维护者
>   2026-07-24 明确要求）。**不要**对 `/workspace/mount/b100_zaip_data`、
>   `/workspace/mount/b100_zaip_data*` 这种整挂载点跑 `find` —— 那底下是别人的工作目录
>   （别人的工作目录 / model_zoo / …），既是在翻别人的东西，几百万文件也会把机器拖慢。
>   本文所有探盘命令都先 `VD=/workspace/mount/b100_zaip_data/video_datasets` 再限定
>   `find "$VD" -maxdepth N`。**唯一例外**是预训练数据 `b100_zaip_data_luan/InternVid-10M-FLT`
>   这个已知的确切路径（§4）——那是直接 `ls` 一个具体目录，不是全盘搜索。
> - 入口/联网：`root@<H100_HOST> -p <SSH_PORT>`。
>   全量视频在 `/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT/`；
>   工作区 `/workspace/mount/b100_zaip_data/rby/`。
> - **⚠️ rby/ 下有两个并排的项目目录，别搞混**（2026-07-23 起）：
>   - `rby/MobileViCLIP` —— **中文 NebulaCV 线**（上游 MCG-NJU 的 clone + 本地改动，
>     与中文线协作）。
>     **本手册不管它，别动。**
>   - `rby/mobileclip` —— **本 fork（英文原版模型改进线）**，R0–R3 全量预训练都在这里跑。
>     本手册下文的「项目根」都指它。

命令一律从项目根跑，除非另有说明。**逐节执行**：§2 的 arch 自检不过，后面全都白搭。

## ✅ 进度标记（Claude 依据 `from-server.md` 回报维护，服务器不用改这段）

截至 **2026-07-24**：

- [x] §0 隧道/代理 —— 已验证可用（clone / push 都走通过）
- [x] §1 项目 —— git 隔离（`.gitenv`）与 PAT 凭据已就绪；`rby/mobileclip` 已 clone
- [x] §2 环境 —— **改用系统 torch 2.5.1+cu124（已确认含 `sm_90`）+ 私有 user-site**；
      `mvc_env310` 提示符虽在，但 `python` 实际指向 `/usr/bin/python`，不再修它。
      余下只差补几个纯 python 小包（`source experiments/env_b100.sh` +
      `python experiments/check_imports.py`，已知缺 `termcolor`）
- [x] §3 权重 —— `mobileclip_s2.pt`(398MB) / `mobileviclip_small.pt`(74MB) /
      `mobileclip_s0.pt`(216MB) 均 `[OK ]`
- [x] §4 数据 —— **走了方案 C**（翻译线合并 jsonl 的 `abs_path`+`caption` 生成 11 片 anno，
      全量 10,647,458 条）+ 软链 `data/video_sample/InternVId-FLT_i`
- [x] §5 预检 —— 11 个 `internvid_v*` 全 `[OK ]`（抽查 20/20 在、真解码 OK）；权重齐
- [ ] **§6 冒烟（2 卡 debug）→ R0 真跑（4 卡）→ R1 三臂 ⟵ 从这里继续**
- [ ] §8 每完成一节回报一次（贴 `from-server.md`），Claude 据此更新本清单

> 评测基准 anno（msrvtt/didemo/anet/k400/ucf101/hmdb51/ssv2_mc）与 `internvid_zh_10k_pilot`
> 预检仍报缺——只影响 B0 评测 / 10k 试点，**不挡 R0–R3 预训练**。
> **但先别急着下**：本机 `/workspace/mount/b100_zaip_data/video_datasets/` 里已经存着大批
> 视频数据集，MSR-VTT 甚至自带现成的 `mobileviclip_annotations/msrvtt_1k_test.json`。
> 探盘与接线步骤见 **§4b**。

---

## 0. 联网（反向隧道 + 代理）

b100 主机本身不直连外网，靠一条**反向隧道**从能上网的操作机把 HK 代理带进来。
隧道只在**装环境/下权重**时需要；**训练读的是本地数据，不需要外网**——所以
多天的训练不能依赖这条隧道存活（见本节末尾的 offline 开关）。

```bash
# 在能上网的操作机上起隧道（保持这个 ssh 会话开着；掉了就重连）
ssh -R 3390:<HTTP_PROXY_HOST>:80 root@<H100_HOST> -p <SSH_PORT> -o ServerAliveInterval=240

# —— 进到 b100 后，这个 shell 里设代理（下载才走它）——
export http_proxy=http://127.0.0.1:3390
export https_proxy=http://127.0.0.1:3390
export HTTP_PROXY=http://127.0.0.1:3390
export HTTPS_PROXY=http://127.0.0.1:3390

# ⚠️ no_proxy 必须包含本机与内网段，否则 torchrun 的 rendezvous 和 NCCL
#    会被 HTTP 代理拦下、直接 hang 在第一个 barrier。
export no_proxy=127.0.0.1,localhost,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12
export NO_PROXY=$no_proxy
```

> **训练前把代理关掉、开 offline**，这样隧道掉线不会拖垮多天的 run：
> ```bash
> unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
> export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
> ```
> 只有在装包 / gdown 权重 / 首次让 HF 拉小模型时才需要代理开着。

---

## 1. 项目

账号是共用的，git 的一切配置/凭据都圈在 `rby/` 内——**每个用 git 的 shell 先
source `.gitenv`**（首次生成见 `TROUBLESHOOTING.md`「共用账号的隔离」一节；PAT 已随
首次 push 存进 `rby/.git-credentials`，之后 clone/pull/push 不再询问）：

```bash
source /workspace/mount/b100_zaip_data/rby/.gitenv
cd /workspace/mount/b100_zaip_data/rby
# 首次（走代理 clone；注意目标目录是 mobileclip，与中文线的 MobileViCLIP 并排）：
git clone https://github.com/rbybryan/mobileclip.git mobileclip
cd mobileclip
# 之后每次开工只需拉 main（服务器只从 main 取指令，见 CLAUDE.md）：
git pull origin main
```

---

## 2. 环境 —— 8×H100（Hopper / sm_90）的环境大坑

**仓库钉的栈是 torch 1.13.1 + cu117（`COMMANDS.md` §1）——那套构建带 sm_70（给
V100 用），没有 sm_90（Hopper）内核（sm_90 从 CUDA 11.8 才有）。直接拿到 H100 上，
每一次 CUDA 调用都会报
`CUDA error: no kernel image is available for execution on the device`。**

这跟 `TROUBLESHOOTING.md`「`CUDA error: no kernel image ...` on V100」一节是**同一类问题**：
torch 构建的 arch 列表里没有本机的 SM。**修法同一套**——装一个 arch 列表里含
`sm_90` 的 torch，再靠 `torch.cuda.get_arch_list()` 验证后才开跑。方向：H100 要
**更新**的构建，不是 cu117——但没 Blackwell 那么苛刻，任一 **cu118 及以上的 torch 2.x**
wheel 都带 sm_90（cu118 里就有；cu124 对 Hopper 更快，作稳妥起点）。

### ✅ 现行方案（2026-07-24 实测定案）：**沿用系统 torch + 私有 user-site**

三条实测结论，把这一节从"重建环境"改成"别重建"：

1. `/usr/bin/python` 的 torch 是 **2.5.1+cu124**，`get_arch_list()` **含 `sm_90`** ——
   上面那条 Hopper 顾虑在这台机器上**本来就不成立**（仓库钉的 1.13.1+cu117 从没被装上）。
2. deepspeed / transformers / decord / open_clip 这些重依赖系统里都有：2026-07-24 的冒烟
   日志里 deepspeed 已起、`models` 也 import 成功，peft 与 open_clip 两关都过了。
3. 但 **b100 是共用 root 账号**，`/usr/local/lib/python3.10/dist-packages` 是其他团队线
   共用的。往里 `pip install <包>==<版本>` 会把别人在用的 transformers / numpy / timm
   **降级**——事故级操作。**已经装进去的就到此为止，别再往系统里装。**

于是：**不重建 torch，缺的小包装进自己的 user-site**。每次开新 shell 先 source 一次：

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip
source experiments/env_b100.sh          # 只设 PYTHONUSERBASE / PYTHONPATH / MVC_*，不装任何东西
python experiments/check_imports.py     # 一次列全缺哪些包，并打印可直接复制的 pip 命令
```

`check_imports.py` 把整条 pretrain import 链上的第三方模块逐个试一遍，**一次报全**
（不用再一轮崩一个包）。按它末尾给出的命令补装，**只用这一种形式**：

```bash
python -m pip install --user --no-deps <包名...>
```

- `--user` + `PYTHONUSERBASE` → 落到 `rby/pyuser/`，不碰共用系统目录；user-site 在
  `sys.path` 里**优先于**系统 site-packages，所以补的包只对 source 过 `env_b100.sh`
  的 shell 生效，对别人无影响。
- `--no-deps` → 不让 pip 顺手升/降级 torch / transformers / numpy。
- 装完重跑 `check_imports.py`，直到打印"全部 [OK]"再进 §6。

> 已知第一批缺失：**`termcolor`**（2026-07-24 的 2 卡冒烟崩在 `utils/logger.py:14`）。
> 另注意 **`openai-clip` ≠ `open_clip_torch`**：前者模块名是 `clip`，后者才是仓库
> `import open_clip` 要的那个（见 `TROUBLESHOOTING.md` 同名节）。
>
> `PYTHONUSERBASE` 必须在**启动训练前**导出：`run_pretrain.sh` 用
> `python -m torch.distributed.run` 拉起子进程，子进程继承环境变量；脚本开头会把
> `python` 和 `userbase` 都打出来，看到 `userbase : <未设置>` 就是忘了 source。

### 备选：自建 conda 环境（**仅当**系统 torch 不可用时才走这条）

**⚠️ 别用 `python -m venv`（2026-07-23 实测翻车）**：miniforge 的 python 是 3.12，
venv 建出来**没有自带 pip**——shell 里的 `pip` 于是落到共用 conda base，torch 等全装进
`/root/miniforge3`（污染共用环境）；而且 py3.12 没有 tokenizers 0.13.3 的预编译轮子，
源码编译还要 Rust。完整症状/清理见 `TROUBLESHOOTING.md`「b100 §2：pip 装错解释器」。
**正确做法：conda 建 python=3.10 环境（目录仍在 rby/ 内），全程 `python -m pip`：**

```bash
# Python 3.10 + cu124 的 torch 2.x（自带 sm_90；cu118/cu121 也都行）
source /root/miniforge3/etc/profile.d/conda.sh
conda create -y -p /workspace/mount/b100_zaip_data/rby/mvc_env310 python=3.10
conda activate /workspace/mount/b100_zaip_data/rby/mvc_env310
which python && which pip   # 必须都在 .../rby/mvc_env310/bin/ 下；不是就停，别继续装
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 其余依赖：版本以 scripts/setup_env.sh / COMMANDS.md §1 为准，torch 已单独装好、
# 下面这些都别把它覆盖掉。关键几条：
python -m pip install "numpy<2"        # decord 0.6 的 wheel 在 numpy 2.x 下报 _ARRAY_API
python -m pip install open_clip_torch --no-deps  # 必须 --no-deps，否则悄悄把 torch 换成 PyPI 版
python -m pip install transformers==4.28.1 tokenizers==0.13.3 timm==0.5.4 openai-clip==1.0.1 \
    einops decord==0.6.0 av==12.0.0 opencv-python ftfy regex pandas scipy scikit-learn \
    pyyaml easydict yacs tabulate tensorboard wandb webdataset imageio \
    accelerate safetensors huggingface-hub setuptools
# ⚠️ transformers 钉 4.28.1，**别装 peft>=0.8**（会触发 `cannot import name 'Cache'`，
#    见 TROUBLESHOOTING.md 同名节）；预训练路径本就用不到 peft。
# torch 2.x 路径需要新版 deepspeed（0.10.1 在 torch 2.x 下 import 就崩，见
# TROUBLESHOOTING.md「import deepspeed 崩」一节）：
python -m pip install "deepspeed>=0.14" || true   # 装不上先跳过，§6 首光用 deepspeed.enable False 绕开
# flash-attn 可跳过（setup_env.sh 本就跳它）；H100 上 flash-attn 2.x 就支持（FA3 更快）。
# torch-2.x 下若仍有 transformers/import 报错，见 TROUBLESHOOTING.md 对应故障节。
```

**开跑前必做的 arch 自检**（这是权威判据，上面的 cu124 只是起点——以
`get_arch_list()` 里真有 `sm_90` 为准；没有就换个 cu11.8+/cu12x 的 torch 2.x wheel）：

```bash
python - <<'PY'
import torch
a = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| archs:", a, "| dev:", torch.cuda.get_device_name(0))
assert "sm_90" in a, \
    "torch 无 Hopper(sm_90) 内核 → 换 cu118+/cu12x 的 torch 2.x，见 TROUBLESHOOTING.md 的 arch 一节"
print(torch.zeros(1).cuda())   # 必须打印 tensor([0.], device='cuda:0')，不报 no kernel image
PY
```

> **H100（Hopper）有 bf16 tensor core** —— 基础配置把 `use_bf16`/`bf16` 写死成 `False`
> **只是因为 V100/Volta 没有 bf16**（`scripts/pretraining/clip/mobileviclip_small/config.py:88-96`）。
> H100 上应当在启动时用 `use_bf16 True bf16 True` 覆盖回来（§6；2026-07 起
> `shared_utils.py` 已统一只认 `use_bf16`、`bf16` 只是旧拼写回退，两个都传依旧无害）：这样一来
> fp16 那一整套动态 loss-scaling / NaN / `LayerNormFP32` 的故事（`TROUBLESHOOTING.md`
> 「反向出 NaN」一节）在 H100 上**根本不发生**，`deepspeed.initial_scale_power`
> 也变成无关项。**代价**：bf16 与 fp16 数值不完全一致，所以 A/B 两臂必须**都**用
> bf16（见 §6 的等步纪律）。

---

## 3. 权重（训练底座）

全量预训练把 `checkpoints/mobileclip_s2.pt` 同时载入视觉塔和文本塔
（`config.py:55,57`）。来源见 `CHECKPOINTS.md`（Apple MobileCLIP，作者 Google Drive 镜像）：

```bash
# 代理开着时下（gdown 走 https_proxy）
python -m pip install gdown
mkdir -p checkpoints
gdown --folder https://drive.google.com/drive/folders/1ROJYwQeO8lt4mEfNjEIAIDWj736WIQq7 -O checkpoints
ls -la checkpoints/mobileclip_s2.pt    # 必须在
```

---

## 4. 接线全量数据（11 个分片）

`configs/data.py:35-41` 里，第 i 个分片（i=1..11）要求两处，**大小写要一字不差**
（`flt` 小写 vs `InternVId-FLT` 大写，是上游的 typo，盘上必须照抄）：

| | 期望路径 |
|---|---|
| 视频目录 | `${MVC_DATA_DIR}/video_sample/InternVId-FLT_{i}` |
| anno json | `${MVC_ANNO_DIR}/InternVid/InternVid-10M-flt_{i}.json` |

`MVC_DATA_DIR`/`MVC_ANNO_DIR` 不设时默认 `data`/`anno_downstream`。

**先探盘**（别假设 `/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT/` 里面长什么样）：

```bash
FULL=/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT
ls -la "$FULL"
find "$FULL" -maxdepth 2 -type d | head -40
# 每个子目录里有多少 mp4：
for d in "$FULL"/*/; do printf '%6s  %s\n' "$(find "$d" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)" "$d"; done
# 训练 anno json 在不在盘上（可能要单独下）。
# ⚠️ 只在这两个**确切目录**里找，不要 find 整个 /workspace/mount/b100_zaip_data*（§0 规则）：
find "$FULL" /workspace/mount/b100_zaip_data/video_datasets/vd-foundation___InternVid-10M-FLT \
     -maxdepth 4 -name 'InternVid-10M-flt_*.json' 2>/dev/null | head
```

**再软链**成配置期望的布局（把盘上真实名字对到期望名字，避开大小写/层级问题）。
下面按探盘结果填 `SRC[i]` = 第 i 个分片视频目录的真实绝对路径：

```bash
WORK=/workspace/mount/b100_zaip_data/rby/mvc_stage
mkdir -p "$WORK/data/video_sample"

# —— 情况 A：$FULL 下就是 11 个分片目录，名字规整（如 InternVId-FLT_1 或 1）——
# 用探盘看到的真实名字改这一行的 glob / 映射即可：
for i in $(seq 1 11); do
  # 常见几种命名都试一下，命中第一个存在的：
  for cand in "$FULL/InternVId-FLT_$i" "$FULL/InternVid-FLT_$i" "$FULL/FLT_$i" "$FULL/$i"; do
    [ -d "$cand" ] && ln -sfn "$cand" "$WORK/data/video_sample/InternVId-FLT_$i" && break
  done
done
ls -l "$WORK/data/video_sample/"      # 应看到 11 条软链，各指向真实目录
export MVC_DATA_DIR="$WORK/data"

# —— anno ——：若探盘找到了 InternVid-10M-flt_*.json，软链进 anno_downstream/InternVid/：
mkdir -p anno_downstream/InternVid
# 把 <annos_dir> 换成探盘里 json 所在目录；找不到就走下面的 gdown：
# for i in $(seq 1 11); do ln -sfn <annos_dir>/InternVid-10M-flt_$i.json anno_downstream/InternVid/; done
# 盘上没有 anno 时（代理开着）从作者 anno 包下，落到默认 anno_downstream/：
# gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream
# MVC_ANNO_DIR 用默认 anno_downstream 即可，不用 export。
```

### ⚠️ anno 的 caption 格式必须先核对（§5 预检查不到这一层）

loader 对 `internvid_v*` 的读法（`dataset/pt_dataset.py`）有三条硬要求，盘上 anno
**必须**满足，否则不是崩就是训歪：

1. **是 JSON 数组，不是 jsonl**：loader 只 `json.load(f)`。若盘上是**逐行 jsonl**
   （中文线那套翻译产物就是 jsonl），`json.load` 直接解析失败 → 必须先转成数组。
2. **每条 `caption` 是纯字符串**：`internvid_v*` 配了 `jump_filter=True` 且**没有**
   `caption_augmentation`，取值路径就是 `anno[i]["caption"]` 再 `pre_text()`——要求 `str`。
   若 `caption` 是 **list**、或字段名叫 **`captions`**、或整条是别的嵌套结构，
   会在 `pre_caption` 抛 `NotImplementedError`（list 还会先 `.split()` 报错）。
3. **`video` 是相对 data_root 的 `.mp4` 路径**（拼到 `video_sample/InternVId-FLT_{i}/` 下）。

> **`check_readiness.py`（§5）只查 `video` 字段和文件在不在，从不看 `caption` 的类型。**
> 所以 11 个分片全 `[OK ]` ≠ caption 格式对——格式不对会等到 torchrun 起来才崩。
> **源 caption 格式与上面假设不一致时（已知会变），先在这里转好再进 §5。**

**先探 caption 格式**（把路径换成探盘找到的真实 anno）：

```bash
python - <<'PY'
import json
p="anno_downstream/InternVid/InternVid-10M-flt_1.json"
try:
    d=json.load(open(p))                       # 报错=不是 json 数组(多半是 jsonl)，需转
    assert isinstance(d,list), type(d)
    r=d[0]; print("keys:",list(r.keys()))
    print("video:",repr(r.get("video"))[:120])
    c=r.get("caption", r.get("captions"))
    print("caption type:",type(c).__name__,"| sample:",repr(c)[:200])
except Exception as e:
    print("NOT a json-array of {video,caption:str}:",repr(e))
PY
```

**若格式不符，转成 loader 要的 `{"video","caption":str}` JSON 数组**（按上一步 head
看到的真实结构改取值那两行）：

```bash
python - <<'PY'
import json
src="<盘上真实 anno>"; dst="anno_downstream/InternVid/InternVid-10M-flt_1.json"
def load(s):
    try: return json.load(open(s))             # json 数组
    except json.JSONDecodeError:               # jsonl 回退
        return [json.loads(l) for l in open(s) if l.strip()]
out=[]
for r in load(src):
    vid=r["video"]                             # 确认相对 InternVId-FLT_i/ 的 .mp4
    cap=r.get("caption", r.get("captions"))
    if isinstance(cap,list): cap=cap[0]        # 多 caption 取一条(或按需改)
    out.append({"video":vid,"caption":str(cap)})
json.dump(out, open(dst,"w"), ensure_ascii=False)
print(len(out),"records ->",dst)
PY
```

### 方案 C：用翻译线合并 jsonl 里的**源 caption** 直接生成 anno（免 gdown）

翻译线的产物**同时含原始英文源 caption**，可以据此直接生成 11 个训练 anno，
**省掉 §4 里 gdown 下作者 anno 包这一步**。文件（b100 绝对路径）：

```
/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
```

**已 probe 探明（2026-07-24）——两条都成立，可以替代下载：**
- **英文源 caption 字段 = `caption`**（样例 `'woman using a computer mouse and keyboard'`）；
  中文译文在 `raw_caption_zh_translated`。英文 R0–R3 用 **`caption`**。
- 每条还带绝对路径 **`abs_path`**（如
  `…/b100_zaip_data_luan/InternVId-10M-FLT/InternVId-FLT_4/shard_00006/InternVId-FLT_4/<file>.mp4`），
  以及一个**深度嵌套**的 `video` 字段（`InternVId-FLT_i/shard_xxxxx/InternVId-FLT_i/<file>.mp4`，
  不是简单的 `分片/文件名`）。
- ✅ **覆盖已确认为全量（2026-07-24）**：total **10,647,458** 条，`video` 前缀 **0** 不匹配；
  分片 FLT_1..10 **各 1,000,000**、FLT_11 **647,458**——即作者 InternVid-10M-FLT 全量，
  **非子集**，可直接替代 gdown。下面「① 覆盖核查」留作复核。

**为什么用 `abs_path` 而不是拼相对路径**：loader 是
`data_root_prefix + os.path.join(data_root, rec["video"])`，`data_root_prefix` 默认 `""`，
而 `os.path.join(root, 绝对路径)` 直接返回该绝对路径。所以**把 `abs_path` 塞进 `video`
字段**，路径就与 `MVC_DATA_DIR`/软链/那层 `shard_xxxxx` 嵌套**完全无关**，最稳。软链
（§4 情况 A）仍要建，只为让 §5 `check_readiness` 的「data_root 目录存在」这关过。

**① 覆盖核查（先补跑，被 `^S` 卡住的那段；若卡住按 `Ctrl-Q` 解冻）：**

```bash
SRC=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
python - <<PY
import json,re,collections
rx=re.compile(r"^(InternVId-FLT_(?:1[01]|[1-9]))/"); c=collections.Counter(); n=bad=nocap=0
for l in open("$SRC"):
    if not l.strip(): continue
    n+=1; r=json.loads(l); m=rx.match(r.get("video",""))
    c[m.group(1)]+=1 if m else 0; bad+=0 if m else 1
    cap=r.get("caption"); nocap+=0 if (isinstance(cap,str) and cap.strip()) else 1
print("total:",n,"| video 前缀不匹配:",bad,"| caption 空/非串:",nocap)
for i in range(1,12): print(f"  InternVId-FLT_{i}: {c[f'InternVId-FLT_{i}']}")
PY
```

**② 生成全部 11 片 anno（`video` 用 `abs_path`，`caption` 用 `caption`）：**

```bash
mkdir -p anno_downstream/InternVid
python - <<PY
import json,re,collections,os
src="$SRC"; rx=re.compile(r"^InternVId-FLT_(1[01]|[1-9])/")
buckets=collections.defaultdict(list); skip=0
for l in open(src):
    if not l.strip(): continue
    r=json.loads(l); m=rx.match(r.get("video",""))
    ap=r.get("abs_path"); cap=r.get("caption")
    if not (m and isinstance(ap,str) and ap and isinstance(cap,str) and cap.strip()):
        skip+=1; continue
    buckets[int(m.group(1))].append({"video":ap,"caption":cap.strip()})
os.makedirs("anno_downstream/InternVid",exist_ok=True)
for i in range(1,12):
    json.dump(buckets[i], open(f"anno_downstream/InternVid/InternVid-10M-flt_{i}.json","w"), ensure_ascii=False)
    print(f"InternVId-FLT_{i} -> {len(buckets[i])}")
print("skipped:",skip)
PY
```

**③ 建软链（只为过预检的「data_root 目录存在」这关）+ 预检：**

`video` 已是 `abs_path`（绝对路径），训练时**根本不经过 data_root**；但 `check_readiness`
仍会 `os.path.isdir(data_root)`，data_root 缺失就报 `[·· ]`。所以只需让默认
`data/video_sample/InternVId-FLT_i` 这 11 个目录**存在**即可——软链到真实分片目录最省事，
**不用 export `MVC_DATA_DIR`**（用默认 `data`）：

```bash
FULL=/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT   # abs_path 的根：InternVid(小写 vid)
mkdir -p data/video_sample
for i in $(seq 1 11); do ln -sfn "$FULL/InternVId-FLT_$i" "data/video_sample/InternVId-FLT_$i"; done
ls -ld data/video_sample/InternVId-FLT_*/    # 11 条软链，各解析到真实分片目录(InternVId-FLT_i 大写)
python experiments/check_readiness.py --decode   # 11 个 internvid_v* 全 [OK ](含真解码)才进 §6
```

> 若某片仍报 `data_root 缺失`，是 `$FULL/InternVId-FLT_i` 这个目录名的大小写没对上
> （根 `InternVid` vs 分片 `InternVId-FLT_i`，别写反）；报 `抽查…全部缺失` 才是 abs_path
> 指的 mp4 真不在盘上。

> 走了方案 C 就**不用** §4 上面那段 gdown（`1grbR39K...`）。`MVC_ANNO_DIR` 保持默认
> `anno_downstream`。生成的就是 `{"video"(=abs_path),"caption":str}` JSON 数组，天然满足
> 上面「caption 格式核对」三条。覆盖已确认为**作者 FLT 全量 10,647,458 条（非子集）**，
> 直接建 11 片、跑预检即可，无需再从 gdown 补齐。

> 若探盘发现 `$FULL` 布局本就是 `.../video_sample/InternVId-FLT_{i}`，那连软链都免了，
> 直接 `export MVC_DATA_DIR=<那个 video_sample 的父目录>`。软链方案只是为了对付命名不一致。

---

## 4b. 评测基准数据：**本机 `video_datasets/` 里已经有**（B0 评测用，不挡 R0–R3）

2026-07-24 探盘发现，b100 上 `/workspace/mount/b100_zaip_data/video_datasets/` 已经存着
大批视频数据集，**其中 MSR-VTT 甚至带了一个现成的 `mobileviclip_annotations/` 目录**：

```
video_datasets/MSR-VTT/
├── TrainValVideo/          # 训练+验证视频
├── TestVideo/              # 测试视频
├── annotation/             # 官方原始标注(test_videodatainfo.json 等)——不是仓库要的格式
└── mobileviclip_annotations/
    ├── msrvtt_1k_test.json      ← 仓库要的就是这个名字
    ├── mobileviclip_ann_file.json
    ├── README.md
    └── status.json
```

**所以评测 anno 大概率不用再从 gdown 下**——先探盘确认，别直接假设。

### 仓库对 7 个评测基准的期望路径（`configs/data.py:48-101`）

`${MVC_ANNO_DIR}` 默认 `anno_downstream`，`${MVC_DATA_DIR}` 默认 `data`。

| corpus | anno（`${MVC_ANNO_DIR}/…`） | 视频目录（`${MVC_DATA_DIR}/…`） | 盘上候选 |
|---|---|---|---|
| `msrvtt_1k_test` | `msrvtt_1k_test.json` | `MSRVTT_Videos` | ✅ `MSR-VTT/`（anno 已备好） |
| `didemo_ret_test` | `didemo_ret_test.json` | `didemo/processed_videos/test` | `DiDeMo/`、`OpenDataLab___DiDeMo/` |
| `anet_ret_val` | `anet_ret_val.json` | `anet_1.3_video_val_resize` | `ActivityNetDataset/` |
| `k400_act_val` | `kinetics400_validate.json` | `kinetics_400_val_10s_320p` | `OpenMMLab___Kinetics-400/` |
| `ucf101_act_test` | `ucf101_act_test.json` | `ucf101/videos` | ❓ 列表里没看到（`ucf_crimes` 是另一个数据集） |
| `hmdb51_act_test` | `hmdb51_act_test.json` | `hmdb51/videos_test` | ❓ 列表里没看到 |
| `ssv2_mc_val` | `ssv2_mc_val.json` | `ssv2/20bn-something-something-v2` | ❓ 列表里没看到 |

### 步骤 1 — 探一次（零成本，先搞清楚有什么）

> ⚠️ **搜索范围只限 `$VD`（= `video_datasets/`）**，这是 §0 的标准规则：不要对
> `/workspace/mount/b100_zaip_data` 整个挂载点跑 `find`，那底下是别人的工作目录。
> 下面每条命令都带 `"$VD"` 和 `-maxdepth`，照抄即可。

```bash
VD=/workspace/mount/b100_zaip_data/video_datasets

# ① 哪些数据集目录已经有人备好了 MobileViCLIP 格式的 anno（MSR-VTT 就是这么发现的）
for d in "$VD"/*/; do [ -d "$d/mobileviclip_annotations" ] && echo "$d"; done

# ② 仓库要的 7 个 anno 文件名，逐个在盘上找
for f in msrvtt_1k_test.json didemo_ret_test.json anet_ret_val.json \
         kinetics400_validate.json ucf101_act_test.json hmdb51_act_test.json ssv2_mc_val.json; do
  printf '== %s\n' "$f"; find "$VD" -maxdepth 4 -name "$f" 2>/dev/null | head -3
done

# ③ MSR-VTT 那个目录里的说明先读一眼（是谁、为什么、下到什么程度）
cat "$VD/MSR-VTT/mobileviclip_annotations/README.md"
cat "$VD/MSR-VTT/mobileviclip_annotations/status.json"
```

### 步骤 2 — MSR-VTT：先验格式，再定视频目录

要看的是两条：**JSON 数组**、有 **`video`** 字段。`caption` **字符串或字符串列表都行**
——评测的 retrieval dataset 两种都吃（`dataset/ret_dataset.py:323`，list 走多 caption 分支）。
`check_readiness.py` 查不到 caption 那一层，所以自己先看：

```bash
MSRVTT=/workspace/mount/b100_zaip_data/video_datasets/MSR-VTT
python - <<PY
import json, os
p = os.path.join("$MSRVTT", "mobileviclip_annotations", "msrvtt_1k_test.json")
d = json.load(open(p))
print("type:", type(d).__name__, "| n:", len(d))
print("keys:", sorted(d[0].keys()))
print("first:", json.dumps(d[0], ensure_ascii=False)[:300])
print("caption 类型:", type(d[0].get("caption")).__name__)
PY
```

`video` 字段写的是文件名还是相对路径，**决定步骤 3 怎么接**（别跳过这一条）。
2026-07-24 实测结果：`type: list | n: 1000`、`keys: ['caption','video']`、
`{"video": "TestVideo/video9770.mp4", "caption": ["a person is connecting …"]}`
—— **带子目录前缀**，`caption` 是列表。

**1k 测试集的视频可能横跨 `TestVideo/` 和 `TrainValVideo/` 两个目录**，所以先数覆盖率
（实测：`TestVideo` 2990 个文件覆盖 1000/1000，`TrainValVideo` 7010 个覆盖 0）：

```bash
python - <<PY
import json, os
M = "$MSRVTT"
d = json.load(open(os.path.join(M, "mobileviclip_annotations", "msrvtt_1k_test.json")))
names = {os.path.basename(r["video"]) for r in d}
print("anno 引用的唯一视频数:", len(names))
for sub in ("TestVideo", "TrainValVideo"):
    p = os.path.join(M, sub)
    have = set(os.listdir(p)) if os.path.isdir(p) else set()
    print(f"  {sub:14s} 目录内 {len(have):6d} 个文件，覆盖 {len(names & have)}/{len(names)}")
PY
```

### 步骤 3 — 接线（软链，不复制、不动源目录）

⚠️ **接线的唯一判据：`join(data_root, anno的video字段)` 必须落在真文件上。**
仓库两处都是这么拼的（`dataset/utils.py:121`、`check_readiness.py`），**没有任何地方
取 basename**。所以步骤 2 打出来的 `video` 字段长什么样，直接决定这一步怎么接：

| `video` 字段形如 | 接法 |
|---|---|
| `video9770.mp4`（纯文件名） | 把 mp4 **拍平**软链进 `data/MSRVTT_Videos/` |
| `TestVideo/video9770.mp4`（带子目录） | 把**子目录整个**软链进去，**不能拍平** |

**MSR-VTT 属于后者**（2026-07-24 服务器实测：`{"video": "TestVideo/video9770.mp4", …}`，
且 `TestVideo/` 已覆盖 1000/1000，`TrainValVideo/` 覆盖 0）：

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip
MSRVTT=/workspace/mount/b100_zaip_data/video_datasets/MSR-VTT

mkdir -p anno_downstream data/MSRVTT_Videos
ln -sfn "$MSRVTT/mobileviclip_annotations/msrvtt_1k_test.json" anno_downstream/msrvtt_1k_test.json

# 若之前按旧文档拍平软链过，先清掉：-type l 保证只删软链，真文件一个都碰不到
find data/MSRVTT_Videos -maxdepth 1 -type l -name '*.mp4' -delete

# 子目录整个软链（两个都接上，anno 换成别的划分也不用再动）
for sub in TestVideo TrainValVideo; do
  [ -d "$MSRVTT/$sub" ] && ln -sfn "$MSRVTT/$sub" "data/MSRVTT_Videos/$sub"
done

ls -l data/MSRVTT_Videos/                       # 只剩 TestVideo、TrainValVideo 两条软链
ls -l data/MSRVTT_Videos/TestVideo/video9770.mp4  # 必须能列出来
```

复验（只看 msrvtt 这一行）：

```bash
source experiments/env_b100.sh
python experiments/check_readiness.py --decode 2>&1 | grep -i msrvtt
```

预期 `[OK ] msrvtt   anno 1000 条;抽查 20 条,缺失 0;试解码 OK(N 帧)`。
若仍报"全部缺失"，新版脚本会把**它试过的完整路径**打出来——拿那条路径和
`ls` 出来的真实位置逐层比，差哪层补哪层。

> 其余 6 个基准照同样三步走：**探盘 → 验 anno 格式 → 软链成期望路径**。步骤 1 的
> ①② 先跑，结果贴回 `from-server.md`，缺哪几个再决定单独下哪几个。
> **每个基准都要单独跑一次步骤 2 的字段检查**——`video` 是纯文件名还是带子目录，各家
> anno 不一样，别照抄 MSR-VTT 的接法（这个坑记在 `TROUBLESHOOTING.md`
> 「预检报"抽查 N 条视频全部缺失"」一节）。
> **这一整节都不挡 R0–R3 预训练**——预训练只用 `internvid_all`（§4 已全绿）。

---

## 5. 预检（不过不许启动）

这一步就是 2026-07-22 那次 `RecursionError` 崩溃的教训：**数据没对上就先别 torchrun**。
（接线错了现在也会**快速失败**并指向本检查——见 `TROUBLESHOOTING.md`
「Training dies with `RecursionError`」——但预检更省事。）

```bash
python experiments/check_readiness.py           # 11 个 internvid_v* 都应亮 [OK ]
python experiments/check_readiness.py --decode  # 再抽一条真解码，确认 decord 能读
```

判读（`check_readiness.py`）：
- `[·· ] internvid_vN  ... data_root 缺失: ...` → `MVC_DATA_DIR`/软链没对上。
- `[·· ] internvid_vN  ... 抽查 N 条视频全部缺失` → 目录在、但 anno 里的 `video`
  字段拼出来的文件不在（层级/命名不符）。
- 全部 `[OK ]` 且 `checkpoints/mobileclip_s2.pt` 在，才算「R0–R3 全量预训练」解锁。
- ⚠️ **预检不查 caption**：它只验证 `video` 字段和文件在不在。若 anno 是 jsonl、
  或 `caption` 是 list/`captions` 键，11 个分片照样全 `[OK ]`，但 torchrun 一起就崩。
  这一层必须在 **§4 的「caption 格式核对」**里先转好——见上一节。
- ⚠️ **预检也不查 torch**：`check_readiness.py` 只 import `decord` 和 `configs.data`，
  **从不 import torch**。所以 11 个 `[OK ]` 只证明**数据**接对了，**不证明环境能训**
  （b100 2026-07-24 就是全绿之后冒烟才发现跑的是系统解释器）。torch/sm_90 请单独验：

  ```bash
  which python && python -c "import sys,torch; print(sys.executable, torch.__version__, torch.cuda.get_arch_list())"
  # python 必须在 .../rby/mvc_env310/bin/ 下；arch 列表必须含 sm_90
  ```

再跑一遍 §2 的 arch 自检确认这个 shell 的 torch 没被后续 pip 换掉。

---

## 6. 启动 R0 → R1（单机、**默认前 4 张 H100**，等步 A/B）

`experiments/run_pretrain.sh` 只支持**单机**（`--nnodes=1` 写死）；参数是
`<config> <num_gpus> [key value ...]`，多余的 `key value` 直接透传给
`tasks_clip/pretrain.py` 覆盖配置。基础配置：`internvid_all`、epochs=3、
batch_size=64/卡、lr=1e-5、deepspeed stage-1。
**⚠️ b100 上 `<num_gpus>` 绝不能省略**：脚本省参时默认 `NUM_GPUS=8`（那是 V100 机的
默认），在本机会直接占满 8 张卡、违反前-4-卡纪律——照下面的命令原样带上卡数。

> **本机默认只用前 4 张卡**（GPU 0–3），后 4 张留给别的任务。所以下面所有真跑都是
> `CUDA_VISIBLE_DEVICES=0,1,2,3` + `NUM_GPUS=4`。**要用满 8 张必须有维护者明确指示。**
>
> ⚠️ **4 卡 ⇒ 全局 batch = 64×4 = 256**（不是 README §4 里 8 卡的 512）。batch size 决定
> 批内负例数、直接改变对比任务本身（`README.md` §4），所以：**(a)** A/B 所有臂都用同一个
> 设置才是等步对照；**(b)** 别偷偷开满 8 卡——先问维护者。
> **07-23 评审建议（U5）：优先尝试 `batch_size 128`（4×128=512，对齐上游全局 batch，
> R0 复现少一个解释变量）**——H100-80G 显存大概率放得下（V100 峰值 23.2 GiB @bs=64
> ⇒ bs=128 估算 ~46–50 GiB），首光冒烟时先验一次显存/吞吐再定；不行则全臂统一 256，
> 并在论文实现细节里声明与上游 512 的差异。定了就全臂一致，禁混。
> **首日拿到单 epoch 墙钟后，按 评审 U6 的预算阶梯排队**
> （常态档：排序臂各 1 epoch 等步，R0 + 胜者跑满 3 epoch）。

> 🚫 **下面三步分成三个代码块是刻意的——不要一次性整段粘贴。**
> 2026-07-24 就出过事：整段贴进去后冒烟崩了，但**后面的 4 卡 R0 照样被拉起来**（`&`
> 后台任务不吃 Ctrl-C），等于在共用机器上跑了一个没验证过的作业。**每一步跑完看结果，
> 上一步不过就停在那里。**

**步骤 0 — 每个新 shell 先做（零 GPU）：**

```bash
source experiments/env_b100.sh
python experiments/check_imports.py     # 必须打印“全部 [OK]”，否则先按它给的命令补包
```

**步骤 1 — 首光冒烟**（DDP+AMP 绕开 deepspeed 构建，bf16 + debug，几十步自停，只验
arch/数据/前反向；2 卡更快）。

⚠️ **先选两张真空闲的卡,别硬编码 `0,1`。** 2026-07-27 就栽在这:`CUDA_VISIBLE_DEVICES=0,1`
撞上别人各占 ~70GB 的两张卡,整条链路全过、`Start training` 都到了,最后 `CUDA out of
memory`——**不是模型太大**(MobileViCLIP-small 才占 ~9GB),是卡被别人吃满了。判读见
`TROUBLESHOOTING.md`「冒烟 CUDA OOM——共用机上别人占了卡」。

先跑这一句挑空卡(`free>70000 MiB` 才算空闲),把冒烟命令里的两个卡号换成它标 ✅ 的:

```bash
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s  free=%6s MiB  used=%6s MiB  util=%s%%  %s\n",
               $1,$2,$3,$4,($2>70000?"空闲✅":"占用⛔")}'
```

```bash
G=0,1   # ← 换成上面标 ✅ 的两张卡号；不要盲抄 0,1
CUDA_VISIBLE_DEVICES=$G bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 2 \
    use_bf16 True bf16 True deepspeed.enable False debug True
```

> ⛔ **闸门：冒烟没跑到打印 loss,就不要执行步骤 2。** 崩了就贴日志回 `from-server.md`。
> 若崩在 `CUDA out of memory`,先按上面那句重挑空卡,不是改 batch。
>
> ℹ️ 冒烟日志里这两条**是正常的,不是错误**:
> - `_IncompatibleKeys(missing_keys=[... token_mixer.mixer_1d ... temporal_embedding ...])`
>   —— 这些是**视频专用**的时序参数,图像底座 `mobileclip_s2.pt` 里本就没有,按设计新初始化;
>   `unexpected_keys=[]` 才是关键(载入没有多余键)。
> - `No pretrained checkpoint provided, training from scratch` —— 指没有**整模型**续训
>   checkpoint,双塔已分别从 `mobileclip_s2.pt` 载入(日志上面有 `Load vision/text_encoder
>   checkpoint`),R0 本就该这样。

**步骤 2 — 真 R0**（全量基线；bf16；前 4 张卡；后台挂多天，掉隧道也不影响）。
**先用步骤 1 那句选卡工具确认要用的 4 张都 ✅ 空闲**——若前 4 张被别人占着,是需要和维护者/
中文线协调的事(等空 / 换 4 张空卡),别顺手开到后 4 张,也别去挤别人已占的卡:

```bash
mkdir -p experiments/runs   # nohup 重定向要先有这个目录
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 4 \
    use_bf16 True bf16 True > experiments/runs/r0_baseline.log 2>&1 &
echo "R0 PID = $!"            # 记下来：本机共用，之后要停只能按这个 PID 停
tail -f experiments/runs/r0_baseline.log     # 头几百步看 loss 是否正常下降
```

> ⚠️ `tail -f` 是前台的，Ctrl-C 只停 `tail`，**不会停训练**。真要停训练：先
> `ps -p <PID>` 确认还活着、`ls -l /proc/<PID>/cwd` 确认是 `rby/` 下的作业，再 `kill <PID>`。
> **禁止 `pkill -f pretrain.py` / `killall`**——本机 root 账号共用，别人的中文线跑的是同一个
> `tasks_clip/pretrain.py`（§0 标准规则）。

**步骤 3 — R1 三臂**（同数据、同步数、同 bf16、同 4 卡/256 batch —— 只换被测组件）。
**R0 至少确认 loss 在正常下降之后再排**：

```bash
for cfg in r1a_sigmoid r1b_attnpool r1_both; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/run_pretrain.sh \
      experiments/configs/$cfg.py 4 use_bf16 True bf16 True
done
```

> ⚠️ **等步纪律**：`batch_size`（4 卡 ⇒ 256）、卡数、`use_bf16/bf16`、deepspeed 开关、
> `initial_scale_power`（bf16 下无关）—— A/B 各臂必须逐项一致，否则批内负例数/数值路径
> 不同，就不再是等步对照。本机是共享的：起跑前 `nvidia-smi` 看清前 4 张卡有没有被别的
> 任务占着，别急着改配置去凑显存、更别顺手开到后 4 张。
>
> **首日先量一个 epoch 的墙钟**（`experiments/README.md` 3.2）：用它推算全量 3 个 epoch 的总时长
> 和是否需要缩 epoch，再决定 R1 三臂怎么排。

装上了 `deepspeed>=0.14` 之后，真跑可以去掉 `deepspeed.enable False`（基础配置
默认 stage-1 开）；但**一旦某一臂用了 deepspeed，所有臂都要用**，别混。

---

## 7. 两个从基础配置继承下来的坑（这台机器上照样有）

详见 `README.md` §4 的两条 caveat，这里只提示 H100 上同样继承：

1. **训练内评测被静默跳过**：基础配置 `test_file = dict(act_val=...)` 但
   `test_types = ["ret_test"]`，key 对不上 → 日志只打 `Skip eval act_val split`，
   全程不评。要在训练里看 held-out 检索，得把 test_file 的 **key 改成 `ret_test`**
   并指向一个真检索集。纯看 loss 曲线/吞吐的 R0 可以不管它。

2. **没有 `ckpt_best.pth`**：`stop_key = None` ⇒ best 分支永不触发。真实产出是每个
   epoch 一个 DeepSpeed 目录 `experiments/runs/<name>/ckpt_{epoch:02d}.pth/`，权重在
   里面的 `mp_rank_00_model_states.pt`。之后要评测/抽权重，对着这些目录，别找
   `ckpt_best.pth`。

---

## 8. 交回（服务器 → Claude）

把结果/报错写进 [`../notes/live/from-server.md`](../notes/live/from-server.md)（顶部新增
日期块），Claude 从那里读、据此改指令。至少回这几项，好让下一轮有据可依：
- §2 arch 自检的 `get_arch_list()` 实际输出、最终装成的 torch 版本；
- §4 探盘看到的 `$FULL` 真实布局（分片目录名、每片 mp4 数、anno json 在哪）；
- §5 `check_readiness.py` 的 11 行 internvid_v* 结果；
- §6 首日一个 epoch 的墙钟 + 头几百步 loss 走势（正常下降/发散/NaN）。
