# b100 主机全量预训练配置与执行手册（InternVid-10M-FLT）

与 V100 试点（[`../tools/data_audit/SETUP_V100.md`](../tools/data_audit/SETUP_V100.md) 用于审查、
[`README.md` §2b](README.md) 用于 10k 试点训练）**并行**的另一条线：在 b100 主机上跑
**全量 10M** 预训练（`internvid_all` 语料，R0→R1 等步 A/B）。两条线互不共享状态、
互不阻塞。**试点数字一律不外推**（见 `README.md` §2b）；全量这条才是真实实验，
按 [`ROADMAP.md`](ROADMAP.md) 第三线 3.2 的门控推进（R0 → R1 → R2 → R3）。

> **两级分工**（见 [`README.md`「两级实验」](README.md)）：V100/10k 只做每个方案的
> 链路/稳定性 go-no-go，**排不了序**（试点评测饱和 + 噪声）；**方案之间真正排序（A/B）
> 和胜者的终训都在这台机器上**。所以下面 §6 会把 R0 与 r1 各臂都跑在 H100 上。

> **⚠️ 先认清这台机器（否则装错 torch、开错卡）：**
> - 主机昵称叫 **`b100`**（挂载都是 `/workspace/mount/b100_zaip_data*`），但**GPU 是
>   8×H100-80G（Hopper / sm_90）**，不是 Blackwell。这跟
>   [`../tools/data_audit/SETUP_H100.md`](../tools/data_audit/SETUP_H100.md) 是**同一台物理机**
>   （只是那份用于数据审查、这份用于训练）。
> - **默认只用前 4 张卡**（`CUDA_VISIBLE_DEVICES=0,1,2,3`，`NUM_GPUS=4`）——后 4 张留给别的任务。
>   要用满 8 张必须有维护者的**明确指示**。
> - 入口/联网与 `SETUP_H100.md` §0 相同（`root@10.55.84.132 -p 31103`）。
>   全量视频在 `/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT/`；
>   工作区 `/workspace/mount/b100_zaip_data/rby/`；项目
>   `/workspace/mount/b100_zaip_data/rby/MobileViCLIP`。

命令一律从项目根跑，除非另有说明。**逐节执行**：§2 的 arch 自检不过，后面全都白搭。

---

## 0. 联网（反向隧道 + 代理）

b100 主机本身不直连外网，靠一条**反向隧道**从能上网的操作机把 HK 代理带进来。
隧道只在**装环境/下权重**时需要；**训练读的是本地数据，不需要外网**——所以
多天的训练不能依赖这条隧道存活（见本节末尾的 offline 开关）。

```bash
# 在能上网的操作机上起隧道（保持这个 ssh 会话开着；掉了就重连）
ssh -R 3390:proxyhk.zte.com.cn:80 root@10.55.84.132 -p 31103 -o ServerAliveInterval=240

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

```bash
cd /workspace/mount/b100_zaip_data/rby
# 首次（走代理 clone）：
git clone https://github.com/rbybryan/mobileclip.git MobileViCLIP
cd MobileViCLIP
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

```bash
# 建议起点：Python 3.10 venv + cu124 的 torch 2.x（自带 sm_90；cu118/cu121 也都行）
python -m venv ~/mvc_env && source ~/mvc_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 其余依赖：版本以 scripts/setup_env.sh / COMMANDS.md §1 为准，torch 已单独装好、
# 下面这些都别把它覆盖掉。关键几条：
pip install "numpy<2"          # decord 0.6 的 wheel 在 numpy 2.x 下报 _ARRAY_API
pip install open_clip_torch --no-deps   # 必须 --no-deps，否则悄悄把 torch 换成 PyPI 版
pip install transformers==4.28.1 tokenizers==0.13.3 timm==0.5.4 openai-clip==1.0.1 \
    einops decord==0.6.0 av==12.0.0 opencv-python ftfy regex pandas scipy scikit-learn \
    pyyaml easydict yacs tabulate tensorboard wandb webdataset imageio \
    accelerate safetensors huggingface-hub setuptools
# ⚠️ transformers 钉 4.28.1，**别装 peft>=0.8**（会触发 `cannot import name 'Cache'`，
#    见 TROUBLESHOOTING.md 同名节）；预训练路径本就用不到 peft。
# torch 2.x 路径需要新版 deepspeed（0.10.1 在 torch 2.x 下 import 就崩，见
# TROUBLESHOOTING.md「import deepspeed 崩」一节）：
pip install "deepspeed>=0.14" || true   # 装不上先跳过，§6 首光用 deepspeed.enable False 绕开
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
pip install gdown
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
# 训练 anno json 在不在盘上（可能要单独下）：
find "$FULL" /workspace/mount/b100_zaip_data* -maxdepth 4 -name 'InternVid-10M-flt_*.json' 2>/dev/null | head
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

> 若探盘发现 `$FULL` 布局本就是 `.../video_sample/InternVId-FLT_{i}`，那连软链都免了，
> 直接 `export MVC_DATA_DIR=<那个 video_sample 的父目录>`。软链方案只是为了对付命名不一致。

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

再跑一遍 §2 的 arch 自检确认这个 shell 的 torch 没被后续 pip 换掉。

---

## 6. 启动 R0 → R1（单机、**默认前 4 张 H100**，等步 A/B）

`experiments/run_pretrain.sh` 只支持**单机**（`--nnodes=1` 写死）；参数是
`<config> <num_gpus> [key value ...]`，多余的 `key value` 直接透传给
`tasks_clip/pretrain.py` 覆盖配置。基础配置：`internvid_all`、epochs=3、
batch_size=64/卡、lr=1e-5、deepspeed stage-1。

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
> **首日拿到单 epoch 墙钟后，按 `notes/plan-review-upgrades.md` U6 的预算阶梯排队**
> （常态档：排序臂各 1 epoch 等步，R0 + 胜者跑满 3 epoch）。

```bash
# 0) 首光冒烟：先用 DDP+AMP（绕开 deepspeed 构建）+ bf16 + debug，几十步自停，
#    只验证 arch/数据/前反向通了。冒烟用 2 卡更快（仍在前 4 张之内）：
CUDA_VISIBLE_DEVICES=0,1 bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 2 \
    use_bf16 True bf16 True deepspeed.enable False debug True

# 1) 真 R0（全量基线；bf16；前 4 张卡；后台挂多天，掉隧道也不影响）
mkdir -p experiments/runs   # nohup 重定向要先有这个目录
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 4 \
    use_bf16 True bf16 True > experiments/runs/r0_baseline.log 2>&1 &
tail -f experiments/runs/r0_baseline.log     # 头几百步看 loss 是否正常下降

# 2) R1 三臂（同数据、同步数、同 bf16、同 4 卡/256 batch —— 只换被测组件）
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
> **首日先量一个 epoch 的墙钟**（`ROADMAP.md` 3.2）：用它推算全量 3 个 epoch 的总时长
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
