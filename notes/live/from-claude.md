# from-claude → Server

Claude writes here in reply to [`from-server.md`](from-server.md). **Newest block at the TOP.**
Server reads this and acts. See [git-handoff-workflow.md](../git-handoff-workflow.md).

Reusable commands/fixes/plans also get logged into `COMMANDS.md` / `TROUBLESHOOTING.md` /
`notes/…` per [CLAUDE.md](../../CLAUDE.md) — this file is the live scratch channel.

<!-- Template — Claude copies this block to the top for each reply:

## YYYY-MM-DD HH:MM — <short topic>

**Diagnosis:** <what's going on>

**Do this next:**
```bash
<exact commands to run from repo root>
```

**Then report back** in from-server.md: <what output Claude needs to see next>

-->

---

## 2026-07-21 (4) — anno 成了;死在 DeepSpeed 现编 FusedAdam(nvcc 配不上 gcc 11)

**先说好消息:** anno 这步彻底通了 —— `可用 10000 / 缺失 0 / root 外 0`,
9k train + 1k val 都写出来了。冒烟也比上次走远得多:数据集建好(9000 条已序列化)、
`mobileclip_s2.pt` 双塔权重加载、优化器分组(458 + 875 个参数)全过,
说明 sm_70、路径、配置这三层都对了。

**Diagnosis:** 新报错在 `deepspeed.initialize` → `FusedAdam` 这一步。
DeepSpeed 默认的 Adam 是 **FusedAdam**,它要在首次运行时用**系统 `/usr/bin/nvcc`
现编一个 CUDA 扩展**(不是 torch wheel 里自带的那份 CUDA)。101vector 上系统
nvcc 比系统 g++ 老,老 nvcc 解析不了 gcc 11 的 `<functional>`,于是:

```
/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’
```

其余 7 张卡刷的 `fused_adam.so: cannot open shared object file` 是**下游噪音** ——
只有一张卡负责编译,其它卡在等那个从没被生成出来的 `.so`。编译修好就一起没了。
跟上次的 Volta 问题**无关**:这条 nvcc 命令行里 `-gencode=…sm_70` 是对的。

**修法(代码里已经改好,你只要 pull):** `utils/config_utils.py` 现在往生成的
`deepspeed_config.json` 里写 `"torch_adam": true`,DeepSpeed 就改用
`torch.optim.AdamW` —— 同一个算法,只是没有融合核,**什么都不用编译**。
(顺带修了两个坑:`adam_w_mode` 原来写在 `optimizer` 顶层,DeepSpeed 只从
`params` 里 pop,一直被静默忽略;还有 `bias_correction` 是 FusedAdam 专有参数,
原样传给 torch AdamW 会直接 `TypeError`,torch 路径下已去掉。)

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

# 上次失败留下的半成品编译缓存,清掉免得干扰
rm -rf ~/.cache/torch_extensions/py310_cu118/fused_adam

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

anno 那两个 json 已经生成好了,**不用再跑 `make_pilot_anno.py`**。

**顺手记一笔(重要,别忘):** 共享字幕 = **1 组 / 2 条 clip(0.0%)**。
意味着 r1 里的 **`caption_idx` 多正例在这批 10k 上是空操作** —— 批内基本撞不上
同字幕的两条 clip。所以试点 A/B 的差异只能归给 sigmoid 损失 + 注意力池化 +
t/b 大学习率这三项,**不能记到多正例头上**;多正例要等全量数据才谈得上验证。
这条已经写进 `experiments/README.md` §2b。

**Then report back** in from-server.md:冒烟的头几十行 —— 具体看
`Use deepspeed to initialize model!!!` 之后有没有顺利进到训练循环、
损失有没有正常打印。若还卡在编译上,把 `nvcc --version` / `g++ --version` /
`echo $CUDA_HOME` 三条的输出一起贴回来。

---

## 2026-07-21 (3) — `没有可用记录` 已修:片段在 InternVId-*/ 嵌套目录里

**Diagnosis:** 不是你命令打错,是 `make_pilot_anno.py` 的 bug。片段**不是平铺在
`videos/` 下**,而是在嵌套的 `InternVId-*/…` 目录里,jsonl 的 `video` 字段常常只是
裸文件名,真正定位文件的是记录里的 `abs_path`。审查工具一直有这套三候选解析
(`tools/data_audit/score_alignment.py: resolve_video_path()`),但
`make_pilot_anno.py` 只做了朴素的 `os.path.join(clips_root, video)` ——
于是 1 万条全部判为「文件缺失」。而且它在打印计数**之前**就 `sys.exit`,
所以那句话没法区分「字段名不对」还是「路径不对」。两个问题都已修并合进 `main`。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main     # 必须先拉,否则还是旧脚本

python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

ls -l anno_downstream/internvid_zh_10k_pilot*.json

# 成功后接 §2b 第 2–4 步
export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**若还是没记录**:新脚本会打印**分项计数** + 首条记录的**真实字段名** + 试过的路径。
把那一整段贴回来,一轮就能定位。对应处理见 `TROUBLESHOOTING.md`:
- 缺 video/caption 多 → 用新加的 `--video-key` / `--caption-key` 指定真实字段名;
- 文件缺失多 → `--clips-root` 指错了,或记录没有 `abs_path`;
- `root 外` 多 → 文件在 `--clips-root` 之外,anno 写不出相对路径。

⚠️ **不要**因为跑不通就改 `--caption-field zh`。文本塔是 CLIP 英文 BPE,
中文字幕喂进去是**另一个实验**,不是等价替代,会让 r0/r1 对照失效。
英文字幕若真的缺失,先报回来再一起定。

**Then report back** in from-server.md:脚本打印的那两行(可用条数 / 共享字幕比例),
或失败时的完整分项计数。

---

## 2026-07-21 (2) — sm_70 修好了;现在缺的是试点 anno(§2b 第 1 步没跑)

**Diagnosis:** 好消息:**torch 的问题解决了**。这次报错发生在
`setup_dataloaders` → `create_dataset`,说明 CUDA 初始化、模型构建、
`init_distributed_mode` 全都过了 —— 上一条的 sm_70 修复生效了。

现在只是 `anno_downstream/internvid_zh_10k_pilot.json` **还没生成**:它不在仓库里、
也不在审查抽样目录里,要用 `tools/make_pilot_anno.py` 从审查用的 10k jsonl 现做,
也就是 `experiments/README.md` §2b 的**第 1 步**。直接跳到训练命令就会缺这个文件。

**路径本身没问题** —— 已逐项对过 `tools/data_audit/`:

| 项 | 值 | 出处 |
|---|---|---|
| 审查 jsonl | `/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl` | `run_v100.sh` `$DATA` |
| 视频根目录 | `/home/vca1/datasets/internvid_zh_10k/videos` | `run_v100.sh` `$CLIPS` |
| `data_root` | `$MVC_DATA_DIR/internvid_zh_10k/videos` | `configs/data.py` |
| `anno_path` | `$MVC_ANNO_DIR/internvid_zh_10k_pilot.json` | 同上,`MVC_ANNO_DIR` 默认 `anno_downstream` |

只要 `MVC_DATA_DIR=/home/vca1/datasets`,`data_root` 就正好是审查工具用的那个
`videos/`,和脚本写出的相对 `video` 字段对得上(脚本会拿 `--clips-root` 逐条校验存在性)。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin handoff

# §2b 第 1 步(只需跑一次)
python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

ls -l anno_downstream/internvid_zh_10k_pilot*.json     # ~9k + 1k

# §2b 第 2–4 步
export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

⚠️ 看一眼脚本打印的两行再往下走:
- `文件缺失 M` 若很大 → jsonl 的 `video` 字段不是相对 `--clips-root` 的(可能带子目录
  或绝对路径),先看 schema:
  `head -1 /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl | python -m json.tool`
- `共享字幕: … 组 / 覆盖 … 条 clip (…%)` → 这就是 `caption_idx` 多正例修复在这批数据上的
  影响面上限,**记下来**,后面 r1 结论要用。

**Then report back** in from-server.md:`make_pilot_anno.py` 打印的那两行(可用条数 /
文件缺失 / 共享字幕比例),以及冒烟跑起来后损失有没有正常打印。

---

## 2026-07-21 — pilot_10k_r0 起不来:torch 没有 Volta 内核(**不是 deepspeed 的问题**)

**Diagnosis:** deepspeed 已经装好了(0.19.2,import 正常),这一步已经过去了。
新报错是**另一个更底层的问题**:`mobileviclip` 环境里的 **torch 2.11.0 是 CUDA 13
构建**(依赖里能看到 `cuda-toolkit 13.0.2` / `nvidia-cudnn-cu13` / `nvidia-nccl-cu13`)。
CUDA 13 已经**彻底删掉 Volta(sm_70)**,所以 warning 里列出的架构是
`7.5 / 8.0 / 8.6 / 9.0 / 10.0 / 12.0` —— **没有 7.0**。V100 正是 CC 7.0,
于是**任何一个 CUDA kernel 都跑不了**,第一个碰到的 `torch.distributed.barrier()`
就炸了(`utils/distributed.py:119`),8 张卡全一样。

所以 `deepspeed.enable False` 这条旁路救不了:崩溃发生在 `init_distributed_mode`,
比选不选 deepspeed 更早,跟框架无关。这正是 `tools/data_audit/SETUP_V100.md` §0
早就写过的 V100 三大坑之一 —— 当时只对**审查环境**做了处理,**训练环境
`mobileviclip` 漏了同样的处理**。

**Do this next:**

```bash
# 0) 先拉最新(本条修复 + §2b 的 sm_70 预检都在里面)
cd ~/mobileclip && git pull origin handoff

conda activate mobileviclip

# 1) 卸掉 CUDA-13 构建
pip uninstall -y torch torchvision torchaudio

# 2) 换成带 sm_70 的 cu118 构建(版本与审查环境一致、已在本机验证过;
#    torch>=2.5 才能配 deepspeed 0.19)
pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu118

# 3) decord 0.6 在 numpy 2 下会报 _ARRAY_API,数据加载器是 decord 的
pip install "numpy<2"

# 4) 自检:arch list 里必须出现 sm_70,且 .cuda() 不报错
python -c "import torch, decord; print(torch.__version__, torch.cuda.get_arch_list()); \
  print(torch.zeros(1).cuda(), torch.cuda.get_device_name(0))"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"

# 5) sm_70 出现后再重开冒烟
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**冒烟过了之后**:回 `experiments/README.md` §2b 按原顺序续跑,不用另记命令 ——
§2b 的 4 步已经是完整的(1 建 anno → 2 `export MVC_DATA_DIR` → 3 sm_70 预检 →
4 冒烟 + 等步数 A/B)。两个提醒:

- §2b 第 1 步 `tools/make_pilot_anno.py` **必须先跑过一次**,否则 r0 配置找不到
  `anno_downstream/internvid_zh_10k_pilot.json`;已经跑过就跳过。
- 第 4 步的等步数 A/B 是**成对**的,两条都要跑完才能比:
  `pilot_10k_r0.py 8` 与 `pilot_10k_r1.py 8`。

⚠️ 以后**不要再在这个环境里 `pip install -U torch/torchvision/torchaudio`,
也不要裸 `pip install torchvision`** —— PyPI 默认 wheel 是 CUDA 12.8/13 构建,
一升就把 Volta 支持删掉。101vector 上任何 torch 升级都必须走 cu118 index。

顺带:`FusedMLP of flash_attn is not installed!!!` 这类是**正常回退提示**不是错误
(sm_70 装不了 flash-attn,见 README_zh Q9);V100 无 bf16 已在配置里处理好
(`use_bf16=False` / `bf16=False`,fp16 AMP)。

**Then report back** in from-server.md:第 4 步自检的完整输出(尤其 arch list 里
有没有 `sm_70`),以及第 5 步冒烟跑起来后的头几十行(损失有没有正常打印)。

完整版已写进 `TROUBLESHOOTING.md` 末节;`experiments/README.md` §2b 也加了一条
开跑前的 sm_70 预检命令。
