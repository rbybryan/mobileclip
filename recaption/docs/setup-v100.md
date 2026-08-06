# 8×V100-32G 数据审查配置与执行手册（10k 抽样专用）

> 目标机器：`101vector`（10.235.31.101），8×V100-SXM2-32GB（Volta，CC 7.0）。
> **本机只有预抽取的 1 万条样本**（`/home/vca1/datasets/internvid_zh_10k`），
> 没有全量 10M 数据——所以本手册的全部测试都在这 1 万条上做，产出
> **10k 版诊断报告**；全量 10M 的审查在 H100 机上跑（`setup-h100.md`）。
> 本机双任务：① 10k 样本跑完整 S0–S3 流水线；② Gradio 人工查看（§6）。
> 只做审查，不改数据。
>
> **V100 三个硬性差异**（工具已内置处理，但装环境时必须知道）：
> 1. **新 torch wheel 不支持 Volta**：cu126/cu128 构建（torch≥2.7）已删掉
>    CC 7.0 内核，import 时警告、跑算子直接报 `no kernel image`。
>    必须装 **cu118/cu121 构建（torch 2.4/2.5 系）**。
> 2. **无 bf16** → S1/S2 自动落 fp16（脚本 `--dtype` 可强制）。
> 3. **vLLM V1 引擎不支持 Volta**（Qwen3-VL 又要求新版 vLLM）→ S2 不起
>    服务，用 transformers 后端 `vlm_audit_hf.py` 8 卡数据并行；
>    235B 裁判本地放不下（8×32GB < 权重 400GB+）→ 走云端 API。

> **并行提示**：标〖可并行〗的步骤互不依赖可同时开跑；完整调度表见
> `README.md`「并行调度总览」。严格顺序链只有冒烟 → S1 → s2-sample → S2 → S3。
> 10k 全流程很快（GPU 部分合计 ~3–4 小时，S2 全覆盖），与 H100 机的全量审查完全独立、
> 可同时进行。

## 0. 环境（一次性，~20 分钟）〖可并行：与 §2 S0、与 H100 机装环境同时进行；权重下载放后台〗

```bash
# 首选 venv（实测国内 conda 镜像不可靠：TUNA anaconda 403、aliyun anaconda 404，
# venv 完全不走 conda 频道；base 的 conda python ≥3.9 即可）
python -V
python -m venv ~/audit_env && source ~/audit_env/bin/activate
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# （之后每个新 shell：source ~/audit_env/bin/activate ——
#   提示符应显示 (audit_env)；若还是 (base) 就是没激活，跑 S1/S2 会报
#   No module named 'numpy'。run_v100.sh 已内置该预检并会给出同样提示。）
# 备选：一定要 conda 时换 BFSU 镜像：
#   conda create -n audit python=3.10 -y --override-channels \
#     -c https://mirrors.bfsu.edu.cn/anaconda/pkgs/main

# ⚠️ V100 关键一步：torch 必须装带 sm_70 内核的 cu118 构建，
#    不能用默认 PyPI 的新构建（cu126/cu128 无 Volta 支持）。
#    版本必须 ≥2.5：新版 transformers 无条件 import
#    torch.distributed.tensor.DTensor（2.5 起才是公开路径），
#    2.4.1 会报 "cannot import name 'DTensor'"
pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu118
# torchvision 必装（Qwen3-VL 的视频 processor 依赖）且必须与 torch 同源同版：
# 0.20.1+cu118 恰好锁定 torch 2.5.1。⚠️ 千万不要裸 `pip install torchvision`——
# 会从 PyPI 拉最新版并连带把 torch 换成 cu126/cu128 构建，Volta 直接废掉
# 若升 torch 后仍有 transformers 侧 import 报错，退而求其次锁旧版
# transformers（4.57 已支持 Qwen3-VL 与 SigLIP2，可配 torch 2.4/2.5）：
#   pip install "transformers==4.57.1"

pip install "numpy<2" "transformers>=4.57" accelerate sentence-transformers decord pillow requests
# numpy 必须 <2：decord 0.6 预编译 wheel 在 numpy 2.x 下报 _ARRAY_API 错误
# accelerate 必装：S2 用 device_map 直接把权重流式载入显存（8 进程并发加载
#   时不在 CPU 内存里展开 8×16GB），缺它报 "requires `accelerate`"
# 不要装 vllm —— V100 路径用不上
# ⚠️ 这一步不能跳过（只装 torch 会在 S1 冒烟时报 No module named 'numpy'）；
#    装完自检：
python -c "import numpy, torch, torchvision, transformers, accelerate, sentence_transformers, decord, PIL, requests; \
  print('deps OK', numpy.__version__)"

# 验证：必须打印 tensor(...) 和 V100 且无 "no kernel image" 报错
python -c "import torch; print(torch.zeros(1).cuda(), torch.__version__, \
  torch.cuda.get_device_name(0))"
```

```bash
# 权重镜像（写进 ~/.bashrc）
export HF_ENDPOINT=https://hf-mirror.com

# S2 模型预下载（32GB 卡用 8B；若是 16GB 卡改 Qwen3-VL-4B-Instruct）
pip install modelscope
mkdir -p ckpts
# 弱网加固版：--max-workers 1 串行下载（默认并发拉多个 ~5GB 分片，弱网上
# 互相抢带宽、单条连接存活更久更易被掐断），外层 until 循环断了自动重连。
# modelscope 支持断点续传（HTTP Range）：重跑同一条命令不会从头下——
# 已完成的文件直接跳过，下到一半的文件从断点继续，Ctrl-C / 断网都不丢进度。
until modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
    --local_dir ckpts/Qwen3-VL-8B-Instruct --max-workers 1; do
  echo "[$(date '+%F %T')] 下载中断，10s 后续传……"; sleep 10
done
# 反复报 IncompleteRead(... bytes read, ... more expected) / Max retries
# exceeded 就是链路断流，上面的循环会一直续传到下完，放后台挂着即可：
#   nohup bash -c 'until modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
#     --local_dir ckpts/Qwen3-VL-8B-Instruct --max-workers 1; do sleep 10; done' \
#     > ckpts/qwen3vl8b_download.log 2>&1 &
#   tail -f ckpts/qwen3vl8b_download.log   # 看进度
# 若速率长期只有几十 kB/s，多半是 ModelScope CDN 到本机的路由差：换
# hf-mirror 通道试试（不同 CDN，同样断点续传，两边下到一半的进度互不冲突，
# 哪个快用哪个；文件落同一目录，跑完任意一边即可）：
#   pip install -U "huggingface_hub[cli]"
#   export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
#   until hf download Qwen/Qwen3-VL-8B-Instruct \
#     --local-dir ckpts/Qwen3-VL-8B-Instruct; do sleep 10; done
#   （huggingface_hub 1.0 起 CLI 改名：huggingface-cli → hf，参数不变。
#    HF_HUB_DISABLE_XET=1 必加：hf-mirror 对 Xet 存储文件的元数据响应
#    不完整，否则报 "Distant resource does not seem to be on
#    huggingface.co"，详见 TROUBLESHOOTING.md。模型是公开的，无需 token。）
# SigLIP2 / LaBSE 体积小，S1 首次运行经 hf-mirror 自动下载即可
```

```bash
# ---- 路径变量 ----
# 本机 10k 抽样路径已是 run_v100.sh 的内置默认值（DATA/CLIPS/OUT 均可不
# export，直接 bash run_v100.sh <stage> 即可）。只需设模型路径并进仓库：
export S2_MODEL=ckpts/Qwen3-VL-8B-Instruct
cd "$(git rev-parse --show-toplevel)"                            # 仓库根目录

# 内置默认（想覆盖时才 export；注意变量要在同一 shell 里生效）：
#   DATA =/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl
#   CLIPS=/home/vca1/datasets/internvid_zh_10k/videos
#   OUT  =outputs/audit_10k
# 路径不对时脚本会立刻报 "错误：DATA 文件不存在：..." 并退出（快速失败），
# 不会再让后续步骤连锁崩溃。
```

### 0.5 权重下载期间可先做的测试〖可并行：五项互不依赖，1/2 现在就能跑〗

真正慢的下载只有 Qwen3-VL-8B（~17GB）；SigLIP2/LaBSE 只有几 GB 很快到位。
等待期间按价值排序可做：

**① 全量视频解码体检（零权重，~10–20 分钟，最值得先跑）**——提前清点
H 类工程问题（缺失/不可解码），并实测三级路径解析在全部 10k 上的命中率
（`ok` 应 ≥98%，过了后面 S1/S2 解码环节就不会有惊喜）：

```bash
source ~/audit_env/bin/activate && cd "$(git rev-parse --show-toplevel)"
python - <<'PY'
import json, sys, os, collections
sys.path.insert(0, "recaption/scripts")
from vlm_audit import resolve_video_path
import decord
DATA="/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl"
ROOT="/home/vca1/datasets/internvid_zh_10k/videos"
stats=collections.Counter(); bad=[]
for i,line in enumerate(open(DATA)):
    rec=json.loads(line)
    p=resolve_video_path(ROOT, rec, rec.get("video"))
    if not os.path.exists(p):
        stats["missing"]+=1; bad.append((rec.get("video"), "missing:"+p)); continue
    try:
        n=len(decord.VideoReader(p, num_threads=1))
        stats["ok" if n>0 else "empty"]+=1
        if n==0: bad.append((rec.get("video"),"empty"))
    except Exception as e:
        stats["decode_fail"]+=1; bad.append((rec.get("video"), type(e).__name__))
    if (i+1)%2000==0: print(i+1, dict(stats), flush=True)
print("FINAL", dict(stats))
[print("bad:", b) for b in bad[:10]]
PY
```

**② S0 命中样本人工过目（零权重，10 分钟）**——重点看 `zh_mostly_english`
那批：是"专有名词/歌名保留英文"（可接受）还是"整句没翻"（要重翻），
直接决定 RETRANSLATE 桶的真实规模：

```bash
# 一键版（自动按类别拆分到 outputs/audit_10k/s0_flag_split/ 并打印实例；
# 优先读全量 s0_flags.jsonl，没有则退回冒烟的 smoke_s0_flags.jsonl）：
SHOW_N=8 bash recaption/scripts/run_v100.sh s0-show
# 只看指定类别时直接调实现脚本：
python recaption/scripts/show_flags.py --flags outputs/audit_10k/s0_flags.jsonl \
  --out-dir outputs/audit_10k/s0_flag_split --per-flag 8 \
  --only zh_mostly_english,zh_low_cjk_ratio,instruction_leak,zh_very_short
```
各旗标的判定规则速查表见 §2。

**③ S1 + s2-sample 不用等 8B**——SigLIP2/LaBSE 一到就先把 S1 全量
（10–30 分钟）和 `s2-sample`（全覆盖）跑完；8B 落地后直接开 `s2`。

**④ 裁判链路冒烟 —— ⚠️ 本机做不了，已作废。**
~~从 101vector 直接打内网端点~~：**实测 101vector 连不上 `172.30.x.x`**
（2026-07，见 §4.3-端点），所以这一步在本机无法进行。
裁判链路的冒烟改在中转机上做，见
[`setup-judge-relay.md`](archive/setup-judge-relay.md) §5（那边 `--limit 5`，同样 5 条）。

> 仍想在本机验证 `vlm_audit.py` 的**非网络**部分（路径解析→抽帧→base64）时，
> 指一个本机能连的端点即可；纯粹验证抽帧的话直接看 §4.5 `s2-check` 更省事。

**⑤ Gradio 人工摸底（零权重、零 GPU）**——按 §6 优先翻
`quality_issue_candidates.jsonl`，频次表与 ② 的观察互相印证。

以下 §1–§5 全部封装在 `recaption/scripts/run_v100.sh`（内置 8 卡 fan-out、
断点续跑、验收检查），每步给出一键命令；想看展开的等价命令可
`cat recaption/scripts/run_v100.sh` 对应分支。

## 1. 冒烟（~10 分钟，必做）

```bash
bash recaption/scripts/run_v100.sh smoke
```
做三件事：S0 字段探测、S1 前 2000 条（fp16 推理 + 解码失败率，应 <2%）、
S2 在 GPU0 审 8 条（首次会下载模型）。

本机 10k jsonl 的实际 schema **已内置支持**，冒烟应打印：
`[keys] video=video zh=raw_caption_zh_translated en=caption`。
视频为 `videos/InternVId-FLT_N/shard_XXXXX/` 两级目录，工具按三级顺序自动
解析路径：`$CLIPS/<video>` → `$CLIPS/<abs_path 中 InternVId-FLT... 起的后缀>`
→ 记录里的 `abs_path` 原值，取第一个存在的。若冒烟解码失败率仍高，把
`head -1 /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl | python -m json.tool`
的 `video`/`abs_path` 两个值发出来排查。

## 2. S0 规则画像（1 万条，CPU 秒级）〖可并行：零依赖纯 CPU，t0 就能跑，先于装环境执行〗

```bash
bash recaption/scripts/run_v100.sh s0
python -m json.tool outputs/audit_10k/s0_summary.json | head -60
# （产物都在仓库根的 outputs/audit_10k/ 下；$OUT 只是 run_v100.sh 内部变量，
#   新开的 shell 里没有定义，别在命令里直接用）
```
S0 只用标准库——**任何现成 Python 都能跑**，不必等 torch 装好。
**决策门**（10k 上的比例可直接外推参考）：拒答+未翻译+指令泄漏合计 >5%
→ 翻译管线先回炉；重复 caption 率 >10% → 假负例问题记为报告重点。
可与目录里已有的 `QUALITY_ANALYSIS.md`、`quality_issue_candidates.jsonl`
对照，看规则命中是否覆盖了人工已发现的问题类型。

**看每类旗标的具体病例**（同样纯标准库，不用 venv）：

```bash
bash recaption/scripts/run_v100.sh s0-show          # 每类打印 5 条 zh/en 实例
SHOW_N=20 bash recaption/scripts/run_v100.sh s0-show   # 想多看几条
```

它同时把 `outputs/audit_10k/s0_flags.jsonl`（所有被标记记录的总表，含
原文和命中旗标）**按类别拆成单独文件** `outputs/audit_10k/s0_flag_split/<旗标名>.jsonl`
（如 `zh_mostly_english.jsonl`、`instruction_leak.jsonl`），方便逐类
`less` 细看或喂给 §6 的 Gradio。各旗标含义速查：

| 旗标 | 判定规则 | 典型病例 |
|---|---|---|
| `zh_no_chinese` | 中文栏 0 个汉字 | 整句英文原样残留（完全没翻） |
| `zh_mostly_english` | 英文字母占比 >50% | 大半句英文夹几个汉字 |
| `zh_low_cjk_ratio` | 有汉字但占比 <40% | 中英混杂（"一只猫在 sofa 上 sleeping"） |
| `zh_equals_en` | 去标点后 zh == en | 把英文原文当译文写回 |
| `instruction_leak` | 以"以下是/翻译：/Here is…"开头 | 模型指令语残留在译文里 |
| `refusal` | 含"抱歉/无法翻译/As an AI…" | 模型拒答文本进了数据 |
| `think_leak` | 含 `<think>`/"让我想想…" | 思维链泄漏 |
| `repetition_loop` | 单个 4-gram 覆盖 >50% 字符 | 复读机式循环输出 |
| `zh_very_short` | 汉字数 <6 | 信息量过低（"日落"） |
| `zh_too_short_vs_en` | 汉字数/英文词数 <0.5 | 疑似漏译了细节 |

另：`s0_summary.json` 里的时长统计从 `start_timestamp`/`end_timestamp`
字段直接计算（`duration_source.timestamps` 应 ≈10000、`none` 应 =0），
不再依赖文件名格式——旧版的 `bad_clip_names=10000` 属误报，已移除。

## 3. S1 嵌入打分（1 万条，8 卡 ~10–30 分钟，可断点续跑）〖可并行：权重下载可同时进行〗

```bash
bash recaption/scripts/run_v100.sh s1        # 8 卡后台启动
bash recaption/scripts/run_v100.sh status    # 随时看进度
```
中断后重跑 `s1` 自动续。
**验收点**：`status` 显示 100%（8 分片行数之和 = 10000）；`error` 率与冒烟一致。

（`run_v100.sh` 的 `S1_TOTAL_SHARDS/S1_SHARD_OFFSET` 双机分片参数在本机
**用不上**——那是两台机器共享全量数据盘时的加速项，本机没有全量数据。）

### 3.5 完成后体检（错误率 + 分数分布，CPU 秒级）

先确认每个分片正常收尾（结尾应是 `[done]` 而非 Traceback）：

```bash
wc -l outputs/audit_10k/s1_scores.*.jsonl        # 8 个分片行数之和 = 10000
tail -n1 outputs/audit_10k/s1_shard*.log         # 每个都应是 "[done] shard N: wrote ..."
```

再跑分布体检：

```bash
source ~/audit_env/bin/activate && cd "$(git rev-parse --show-toplevel)"
python - <<'PY'
import json, glob, collections
import numpy as np
rows = [json.loads(l) for f in glob.glob("outputs/audit_10k/s1_scores.*.jsonl") for l in open(f)]
print("总条数:", len(rows))
err = [r for r in rows if "error" in r]
print(f"error: {len(err)} ({len(err)/len(rows):.1%})", collections.Counter(r["error"] for r in err))
for k in ("sig_zh_mean", "sig_en_mean", "xling_cos"):
    v = np.array([r[k] for r in rows if k in r])
    print(f"{k}: n={len(v)} min={v.min():.3f} p5={np.percentile(v,5):.3f} "
          f"中位={np.median(v):.3f} p95={np.percentile(v,95):.3f} max={v.max():.3f}")
PY
```

判读标准：

- **error 率应 <2%** 且与冒烟一致（§0.5 解码体检 `ok≥98%` 过了的话这里
  不会有惊喜）。`decode:` 开头的错误是数据本身的工程问题（H 类桶），
  属于正常信号，不用修。
- **三列分数应有正常散布**：`n` 接近 10000（带 error 的记录没有分数）、
  无 NaN、不是所有值挤在同一个数上。SigLIP 余弦典型在 0~0.4 之间铺开；
  `xling_cos`（LaBSE 中英一致性）大部分应偏高、留一条低分尾巴——低尾
  就是疑似翻译坏掉的，交给 S3 分桶，这一步不用逐条解读。
- 若 `sig_zh_mean` 整体几乎为 0 或大量 NaN，是 fp16 推理出了问题
  （非数据问题）：按 §4 末尾的办法换 fp32 或排查。

体检过了先跑 §3.6 的联合分析，再进 §4.1 `s2-sample`——后者自带 S1 完成度
≥95% 的门槛，能正常产出 ~10000 条即说明 S1 产物结构与覆盖度均无问题。

**实测参考值（2026-07 本机 10k，v0.1）**——后续复跑/全量应与此量级相符：

| 列 | min | p5 | 中位 | p95 | max |
|---|---|---|---|---|---|
| `sig_zh_mean` | -0.018 | 0.079 | **0.128** | 0.177 | 0.247 |
| `sig_en_mean` | -0.002 | 0.091 | **0.152** | 0.216 | 0.322 |
| `xling_cos` | 0.013 | 0.611 | **0.767** | 0.877 | 1.000 |

error 0/10000（0.0%）。注意 0% 解码失败**不可外推到全量 10M**：这 1 万条是
预抽样，抽样时大概率已按「视频可读」条件筛过，H 类工程问题在本机天然为空。

### 3.6 S1 联合分析（秒级，进 S2 前必做）

```bash
bash recaption/scripts/run_v100.sh s1-report      # -> outputs/audit_10k/s1_analysis.md
```

§3.5 的边际分布只回答「S1 跑对没有」，**回答不了「数据有什么问题」**——
方案 §3 的解读矩阵每一条都是**联合条件**（en 高 + zh 低 + xling 低 → 翻译坏；
双低 → 源头错）。`analyze_s1.py` 补的就是这一层，并处理三个边际分布会骗人的地方：

1. **SigLIP 余弦不是质量分**。SigLIP 用 sigmoid 损失训 `t*cos+b`（t 大、b 很负），
   `0.13` 既不是「13% 对齐」，也不能跟 CLIP 余弦比。**只有排序有意义**——
   所以脚本一律输出分位数，正式阈值留给 S3 用 VLM 判定校准。
2. **zh 比 en 低 ≠ 翻译损伤**。InternVid-10M-**FLT** 本身就是「按英文 CLIP 分
   取前 30%」筛出来的，`sig_en` 打在**被选择的那个变量**上，天然上偏；
   加上 SigLIP2 文本塔未必把中英放在同一余弦尺度。
   **控制实验**：看 `xling` 最高十分位（忠实译文）内部残留的
   `sig_en − sig_zh`——那部分是编码器/选样基线偏移；只有低 xling 段的
   **超额落差**才是真实翻译损伤。脚本直接打印这两个数和差值。
3. **xling 高分可能是「没翻译」**。LaBSE 对相同字符串给 ~1.0，
   整句英文照抄反而拿满分（本机 `max=1.000` 即此嫌疑）。脚本用
   `s0_flags.jsonl` 的 `zh_no_chinese/zh_equals_en/zh_mostly_english`
   交叉核对，并给出**剔除照抄后**的 xling 中位数才是真实忠实度基线。

> 因此 §3.6 依赖 S0 产物：**先跑 `s0` 再跑 `s1-report`**（S0 秒级、纯标准库）。
> 没有 `s0_flags.jsonl` 也能跑，只是跳过第 3 项交叉核对。

报告最后一节「解读矩阵预览」按分位给出四角计数（RETRANSLATE / RECAPTION /
翻译可疑 / KEEP 候选），并预演 **S3 的 `--xling-threshold` 默认值会命中多少**。
**这不是结论，是 S2 的抽样预算表**：四个角都要有样本，S3 才能把分位映射成
问题概率。真正的问题率只有 §5 的 `report.md` 说了算。

> **本机 10k 的实测结论已归档到
> [`audit-findings.md`](archive/audit-findings.md)**
> （`outputs/` 被 gitignore，报告本身不入库）。要点：`xling_cos` 在本批是
> **弱工具**（ρ(sig_zh,xling)=0.04、动态范围窄），语言污染要看 S0 旗标而非
> xling；且 S3 默认 `--xling-threshold=0.7` 正落在本批 p25 附近，**跑 s3 前
> 先读那份笔记**。

## 4. S2 VLM 审核（**推荐 10k 全覆盖**，8 卡 ~2–3 小时）〖4.2 主力 / 4.3 裁判 / 4.4 人工抽检可并行：裁判走外部端点，不占本机 GPU〗

```bash
# 4.1 抽样——推荐全覆盖：默认参数（每分位桶 3000×10 桶）在 10k 上即选中
#     全部有分记录，直接跑，不加任何收缩参数：
bash recaption/scripts/run_v100.sh s2-sample
wc -l outputs/audit_10k/s2_sample.jsonl        # 应 ≈10000
#     （赶时间的快速版才用收缩参数，~1.4k 条 1 小时：
#      PER_BUCKET=100 FLAGGED=300 LOW_XLING=200 bash recaption/scripts/run_v100.sh s2-sample）

# 4.2 主力档：Qwen3-VL-8B，transformers 后端 8 卡数据并行
#     全覆盖 10k ≈ 每卡 1250 条 × 6–8s ≈ 2–3 小时；可断点续跑
bash recaption/scripts/run_v100.sh s2
bash recaption/scripts/run_v100.sh status

# 4.3 裁判档 / 4.4 人工抽检 —— 均已收官下线，见下方 §4.3

# 4.5 【S2 跑完必做】判定体检——8 个分片都说 "1250 verdicts" 不等于 1250 条判定可用
bash recaption/scripts/run_v100.sh s2-check     # 秒级，纯 CPU，产出 s2_health.md
```

### 4.2 与 4.3 分别在做什么（一句话：**4.2 给数据打分，4.3 给 4.2 打分**）

两步用的是**同一份 rubric、同一套 prompt、同样 8 帧输入**，区别只在
「谁来判」和「判多少」。方案 §2 的三档分工里，这是「**大模型定标准、
小模型跑量**」的落地。

**4.2 主力档 —— 产出审核结论本身**

- 对 `s2_sample.jsonl` 里**每一条**跑 Qwen3-VL-8B（本机 10k 即全覆盖），
  按 rubric 输出结构化 JSON：`en_score`/`zh_score` **独立打分**（分开「源头错」
  和「翻译搞坏」）、`trans_score`、问题标签、`granularity`、时序正确性、
  建议 `action`，以及 2–3 条 `better_zh` 候选中文短字幕。
- 8 卡数据并行：第 i 张卡认领 `行号 % 8 == i` 的记录（`vlm_audit_hf.py:124`），
  各写各的 `s2_verdicts.<i>.jsonl`，**可断点续跑**（重跑同一命令自动跳过已完成的）。
- **这是唯一产出全样本判定的一步，下游全部依赖它**：S3 把分位映射成问题概率、
  分桶、`xling` 阈值裁决（已随决策 1 下线）、以及 `better_zh` 直接进
  v0.2 的 `captions` 列表——没有它，后面每一步都无米下炊。

**4.3 裁判档 —— 审核「审核者」，不增加任何覆盖面**

- 取 `s2_sample.jsonl` 的**前 `JUDGE_N` 条**（默认 500），送云端 235B
  （任何 OpenAI 兼容端点）**再判一遍**，然后在重叠部分算 8B 与 235B 在
  `action` 上的一致率。
- 目的只有一个：回答「**8B 的判定能不能信**」。S3 是拿 8B 判定去校准阈值
  再外推到全量——判定不可信，后面所有百分比都是沙上建塔。所以这是决策门
  而不是可选项。
- 走云端 API，**不占本机 GPU**，只依赖 4.1 抽样完成，**可与 4.2 同时开跑**。

两个从代码里读出来、文档此前没写的细节：

1. **裁判子集是「数据文件顺序的前 N 条」，不是分层随机抽样**
   （`run_v100.sh` 的 `head -n "$JUDGE_N"`；而 `s2_sample.jsonl` 由
   `make_audit_sample.py` 第二趟顺着原始数据文件流式写出，所以顺序≈语料原序）。
   对「语料整体」的一致率估计没问题，但**稀有层**（S0 命中样本、低 `xling` 尾）
   在这 500 条里可能只有个位数，**那些层的一致率其实没被检验**。
   若一致率刚好卡在 75% 边缘，或某个层的判定特别关键，就对该层单独补测。
2. **4.3 可以在 4.2 还没跑完时就开跑**：8B 按 `行号 % 8` 分片，前 500 行均匀
   落在 8 张卡上、且都排在各分片靠前的位置，所以重叠很快就有、一致率能提前算出来。
   早发现 rubric 有问题 = 少浪费两三个小时的 8 卡时间。

**决策门**（10k 轮由已下线的 `judge-agree` 算出，结论见 [`data-issues.md`](data-issues.md)
§4 决策 2b）：五类 `action` 一致率 **≥75%**、二值
`{keep,fix_text}` vs 其余 **≥80%**、**各维度 Cohen's κ ≥0.6**（plan §4）。
**别只看一致率**——类别不平衡时双方一起说 `keep` 就能堆出高一致率，κ 才是
扣掉「碰巧一致」后的真实一致程度；混淆矩阵里的偏斜（如主力判 `recaption`、
裁判判 `retranslate`）说明 rubric 边界没写清，改 rubric 比换模型有用。
不达标先改 rubric 再重跑——
**在 10k 上把 rubric 调可信，正是全量（H100 机）开跑前最重要的准备**。
全覆盖的额外收益：10k 报告是全量实测（零抽样误差）；1 万对
（嵌入分, VLM 判定）配对点让 S3 的阈值校准曲线远比 1.4k 稳，直接
作为 H100 全量跑的先验；且全部 10k 都拿到 `better_zh` 候选修复字幕。
fp16 若出现乱码/NaN 输出（罕见）：`export S2_MODEL=Qwen/Qwen3-VL-4B-Instruct`
并在 `vlm_audit_hf.py` 加 `--dtype fp32` 重跑。

### 4.5 S2 完成后体检（秒级，**进 S3 之前必做**）

```bash
bash recaption/scripts/run_v100.sh s2-check      # -> outputs/audit_10k/s2_health.md
```

**为什么必须单独做这一步**：分片日志的 `[done] shard i: 1250 verdicts` 是**写出的行数**，
不是**可用判定数**。`vlm_audit_hf.py:160-167` 把每一个异常都兜住并照样写一行，
写成 `{"verdict": {"error": ...}}`；而下游两个消费者都会**静默丢弃**这些行：

| 位置 | 代码 | 行为 |
|---|---|---|
| `aggregate_report.py:72` | `if "error" in v and "action" not in v: continue` | 不计入任何比率 |

于是一个在 40% 的片子上失败的分片，在 `status` 里看起来是 100% 完成，
只是让 `report.md` 变得更安静一点。**`report.md` 里每一个百分比都是在幸存记录上算的，
而幸存记录不是随机子集**——解码失败集中在坏视频上，JSON 解析失败集中在长/怪 caption 上，
两者都与「质量差」正相关，丢掉它们会**系统性低估**问题率。

`s2-check` 按「能否让下一步失效」的顺序查五项：

1. **覆盖率** —— 判定行数 vs `s2_sample` 行数、重复 video、漏判 video。
   （断点续跑重复写入会让同一条被计两次，直接污染分桶比例。）
2. **失败率与失败原因** —— 拆成 `video_decode` / `json_parse` / `oom` / `other`，
   因为三者的处置完全不同：解码失败重跑无用（H 类工程问题），
   JSON 解析失败重跑或缩 prompt 有救，OOM 减 `--frames` 有救。默认门 **失败率 ≤5%**。
3. **字段完整性** —— 判定能解析、却仍缺 `trans_score` 的情况很常见；
   下游是 `.get()` 读的，**缺值会被当成「不坏」从而稀释差率**。
   `trans_score` 覆盖率 <90% 直接判不通过——§4 决策 1 的全部差率都建立在它上面。
4. **分布塌缩** —— 这是唯一能穿过前三项的失败：能解析、字段齐全、但 95% 都判 `keep`
   或所有分数都是 3。**S3 的分位校准靠的是判定与嵌入分之间的对比度**，
   判定没有方差，校准出来的阈值就是噪声。
5. **各层可用量** —— 抽样是分层的，稀有层（S0 命中、低 xling 尾）本来就小，
   被失败吃掉之后就报不出该层结论了——而那几层正是 T1/T3 要回答的问题所在。
6. **维度独立性 + action 自洽性** —— 前五项都过了，判定仍可能是**假的**：
   - rubric 要求 `trans_score` **只比中英两条文本、不看画面**，与 `en_score`
     （英文 vs 画面）量的是两回事。若两列**逐条同值率 >60%**，说明模型在照抄
     而非独立评估——**「翻译坏没坏」这件事根本没被测量过**，而
     §4 决策 1 的全部差率都建立在它上面（2026-07-22 的 10k 实测触发了这项怀疑：
     三列分数的逐档计数近乎重合，见 `data-issues.md` 问题 8）。
   - `action` 与自身分数是否矛盾：rubric 用分数模式定义每个 action
     （`retranslate` = 翻译坏了 = `trans_score` 低）。矛盾意味着
     **S3 按分数算阈值、`apply_buckets.py` 按 action 分流任务队列，
     两半流水线在同一批记录上各说各话**——报告里的问题率和实际派下去的
     任务描述的不是同一件事。10k 实测：`retranslate` 4120 条 vs
     `trans_score≤2` 仅 1958 条 → **≥52.5% 的 retranslate 自身 `trans_score≥3`**。

脚本**不通过时返回非零**（`run_v100.sh` 会拦住并提示），产出
`outputs/audit_10k/s2_health.md`。**体检结论要抄进
[`data-issues.md`](data-issues.md) §1 状态表**（`outputs/` 是 gitignore 的，
不落库就随会话丢失）。

### 4.6 文本-only 翻译重判 `s2-trans`（10k 实测触发，**决策 1 的前置**）

```bash
bash recaption/scripts/run_v100.sh s2-trans      # 8 卡；不解码视频，比 s2 快得多
bash recaption/scripts/run_v100.sh trans-check   # 重判体检（没跑完 <95% 会直接拒绝）
```

**为什么需要它**：2026-07-22 的 10k 全覆盖跑，机械体检 10000/10000 全过，
但 §6 查出 `en_score`/`zh_score`/`trans_score` **逐条同值率 92.5%／98.9%**——
模型只产生了**一个**整体质量分、复制进三个字段，`action` 再按固定阶梯
（`5→keep, 4→fix_text, 3→retranslate, ≤2→recaption/drop`）由它决定。
「翻译坏没坏」从未被测量过，而决策 1 的全部差率都建立在 `trans_score` 上。
完整证据与作废清单见 [`data-issues.md`](data-issues.md) 问题 8 详解。

**修法为什么是拆调用而不是改措辞**：翻译忠实度本就不需要看视频
（rubric 原话"仅比较中英两条文本"）。`vlm_audit.py:TRANS_PROMPT` 两处设计：

1. **不给帧** —— `--mode trans` 完全不打开视频文件，锚点从"不鼓励"变成"物理不可得"。
   附带好处：S2 的真正瓶颈是视频解码，这一趟没有解码，**快得多**。
2. **先证据后打分** —— 必须先逐条列出 `mismatches`，评分规则再锁死在该列表上
   （"mismatches 为空则必须给 5"），分数无法作为整体印象浮空输出。

产物 `s2_trans.*.jsonl` 会被 **`s3`（`aggregate_report.py`）自动优先采用**来算翻译差率，
找不到时退回主力档并打印警告；`trans-check` 在没跑完（<95%）时直接拒绝——
在半成品上算出来的差率没有意义。

**跑起来之后先别干等**（`s2-trans` 是后台 8 分片，`bash run_v100.sh status`
会显示进度）：

```bash
bash recaption/scripts/run_v100.sh trans-check    # 中途随时可跑
```

「修复跑了」不等于「修复生效了」。`check_trans.py` 从**部分产物**就能回答三件事，
好让失败的修复在几百条时就被发现，而不是等一整趟跑完：

1. **还锚不锚定**（最关键）——新 `trans_score` 与**旧 `en_score`** 的逐条同值率。
   旧 `trans` 本就≈旧 `en`，所以**新值若仍与旧 `en` >60% 同值，说明拿掉画面没解除锚定**，
   该停下来重想 prompt 而不是烧完剩下的机时。
2. **守不守自己的规则**——`TRANS_PROMPT` 写死「`mismatches` 为空 → 必须给 5」，
   这条的遵守率直接量化指令遵循度；低了说明分数仍是浮空的整体印象、证据列表只是摆设。
3. **改判幅度**——新差率（≤2）与分数分布。对照：主力档旧 `trans_score≤2` 是 19.6%，
   但那是整体质量分的复制品、**不是翻译差率**；这里才是第一个可信估计。
   S1 结论 3 预期它不会很高，**若显著偏高反而是新发现**，要回头查 S1 的假设。

分片按 `行号 % 8` 取样而非前 N 条，所以部分产物是均匀子样本，趋势可信。

> 同一个方法可推广：日后若要让 `en_score` / `zh_score` 真正分离，
> 也应拆成两次只给一侧文本的调用，而不是在同一个 prompt 里多写几句"请独立打分"。

---

### 4.3 交叉裁判档（已收官，2026-08-05 下线）

10k 轮的裁判一致性核验已经跑完并落账（决策 2b：500 条裁判证实问题 8，见
[`data-issues.md`](data-issues.md) §4）。`s2-judge` / `judge-agree` 两个子命令与
`judge_agreement.py` 随之下线；端点部署的三份完整记录（内网 Qwen3.6-27B、本机
Qwen3-VL-32B、InternVL3.5-38B）保留在
[`archive/setup-judge-api.md`](archive/setup-judge-api.md) 与
[`archive/setup-judge-relay.md`](archive/setup-judge-relay.md)。要在新一批数据上
重开这条线，从那两份文档恢复命令即可（`git log` 里有删除前的脚本）。

## 5. S3 校准 + 分桶 + 10k 诊断报告（CPU 秒级）〖不可并行：需 S0+S1+S2 全部产物，最后跑〗

```bash
bash recaption/scripts/run_v100.sh s3
```
产出 `outputs/audit_10k/report.md`（10k 问题率表 + sig_zh 分位校准表）和 `buckets.jsonl`。
**这份 10k 报告的两个用途**：① 第一份可汇报的问题率数字（抽样是分层随机的，
比例可外推到全量）；② 校准出的 `sig_zh` 阈值直接作为 H100 机全量 S3 的
`--sig-zh-threshold` 先验。审查到此为止；后续优化见
`recaption/docs/archive/audit-plan.md` §5、§7.6+。

## 6. 人工查看工具（Gradio，本机已就位）〖可并行：不依赖任何流水线步骤，随时可用，不占 GPU〗

抽样目录（`10.235.31.101:/home/vca1/datasets/internvid_zh_10k`）内容：

| 文件 | 说明 |
|---|---|
| `internvid_zh_10k.jsonl` | 1 万条抽样数据（即 §0 的 `DATA`） |
| `videos/` | 对应视频（即 §0 的 `CLIPS`） |
| `gradio_video_text_review.py` | Gradio 浏览工具 |
| `REVIEW_README.md` | 工具自带说明（以它为准） |
| `QUALITY_ANALYSIS.md` | 已有的一份初步质量分析，开工前先读 |
| `quality_issue_candidates.jsonl` | 已筛出的疑似问题样本——**Gradio 里优先翻这批**，摸底效率最高 |
| `sample_manifest.json` / `analyze_internvid_zh_10k.py` / `result/` | 抽样清单 / 分析脚本 / 分析产出 |

- 启动：
  ```bash
  ssh vca1@10.235.31.101        # 或本机直接 cd
  cd /home/vca1/datasets/internvid_zh_10k
  python gradio_video_text_review.py
  ```
- 访问：Gradio 默认端口 7860。本地浏览器打开需 SSH 端口转发：
  ```bash
  ssh -L 7860:localhost:7860 vca1@10.235.31.101   # 然后浏览器开 http://localhost:7860
  ```
  （若脚本以 `server_name="0.0.0.0"` 启动且办公网与 10.235.31.101 互通，
  可直接访问 `http://10.235.31.101:7860`。）

**定位：看数据，不是标数据。** 不要试图把 1 万条标完——量化结论一律以
S0–S3 的自动化产出为准，这个工具只承担三个定性任务：

1. **开工前摸底**（建议第一天就做，与装环境并行）：随机翻 100–200 条，
   对照方案 §1 的问题分类学 A–H 记一个简单频次表，每类问题记 3–5 个典型
   `video id` 留档。这份体感直接影响两件事：S0 决策门数字怎么解读、
   S2 rubric 措辞要不要针对高频病型加例子。
2. **S2 期间抽查 VLM 判定**：从 `s2_verdicts.*.jsonl` 里挑十几条
   `action=recaption/drop` 的样本对着画面看，快速人肉 sanity check
   （正式可信度以 §4 的 8B vs 裁判一致率决策门为准，这里只防系统性跑偏）。
3. **承接 §4 的 300 条人工抽检**：`human_check_300.jsonl` 的样本在此工具里
   过一遍即可；也不必逐条全标——与 VLM 判定一致的快速翻过，只在发现
   系统性分歧（某类问题 VLM 集体看错）时再扩量核对并记录。

产出物：一份简短的《人工摸底笔记》（问题类型频次 + 典型 video id 清单），
附到 S3 的 `report.md` 旁边，作为自动化外推数字的定性佐证。

## 6.5 一条命令打印全部状态（贴给协作方/AI 用）

```bash
bash recaption/scripts/run_v100.sh snapshot
```

秒级、纯 stdlib（不需要激活 venv），~60 行输出，一次给全：git 落后几个提交、
各阶段产物行数与完成度、**存活的后台进程**、粒度重判的 `n_info` 分布、
三张抽检表各填了多少行、以及从**不入库的** `outputs/*.md` 里回显的关键结论数字
（同值率 / 时序缺失率 / 翻译差率 / 分桶比例）。

> 需要别人（或下一次会话）判断「现在到哪了、下一步该干嘛」时，贴这一份就够，
> 不用逐个 `wc -l`。

---

## 7. 审查后的效度测评（已收官，2026-08-05 下线）

P0–P2 三档效度测评在 10k 轮已全部跑完，结论落在
[`audit-report.md`](audit-report.md) §6「效度与局限」与 [`data-issues.md`](data-issues.md)。
配套的 `review` / `review-score` / `review-bundle` / `s2-merge` / `s2-gran` /
`gran-check` 子命令与 `make_review_sheet.py`、`check_gran.py` 随之下线
（主计划 §6：❌ 任何有约束力的人力标注 D14、❌ better_zh 续产 D12）。

现在还在跑的目检走重述流水线自己的评审包：

```bash
N_SAMPLE=50 bash recaption/scripts/run_v100.sh pipeline-compare   # -> bundle_compare/index.html
bash recaption/scripts/run_v100.sh summary-table                  # -> table_compare/table.html
```

## 时间线与双机分工

| 步骤 | 本机（10k） | H100 机（全量 10M，见 setup-h100.md） |
|---|---|---|
| 环境 + 权重 | ~0.5–1 h | ~0.5–2 h（多下 235B） |
| S0 | 秒级 | ~20 min |
| S1 | ~10–30 min | ~1 天 |
| S2 主力 | ~2–3 h（**10k 全覆盖**，HF 后端） | 1–2 h（45k 条，vLLM dp=8） |
| S2 裁判 | ~0.5 h（500 条云端） | 1–2 h（2k 条本地 FP8 tp8） |
| S3 报告 | 秒级 | 分钟级 |
| **总计** | **~3–4 小时** | **~1.5 天** |

推荐节奏：**先在本机把 10k 全流程跑完**（半天内拿到 10k 报告 + 调好
rubric/阈值），再在 H100 机开全量——10k 校准出的 `sig_zh` 阈值和 rubric
修订直接复用，全量一次跑对，避免在 10M 规模上返工。
