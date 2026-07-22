# 8×V100-32G 数据审查配置与执行手册（10k 抽样专用）

> 目标机器：`101vector`（10.235.31.101），8×V100-SXM2-32GB（Volta，CC 7.0）。
> **本机只有预抽取的 1 万条样本**（`/home/vca1/datasets/internvid_zh_10k`），
> 没有全量 10M 数据——所以本手册的全部测试都在这 1 万条上做，产出
> **10k 版诊断报告**；全量 10M 的审查在 H100 机上跑（`SETUP_H100.md`）。
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
cd ~/mobileclip                            # 仓库根目录

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
source ~/audit_env/bin/activate && cd ~/mobileclip
python - <<'PY'
import json, sys, os, collections
sys.path.insert(0, "tools/data_audit")
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
SHOW_N=8 bash tools/data_audit/run_v100.sh s0-show
# 只看指定类别时直接调实现脚本：
python tools/data_audit/show_flags.py --flags outputs/audit_10k/s0_flags.jsonl \
  --out-dir outputs/audit_10k/s0_flag_split --per-flag 8 \
  --only zh_mostly_english,zh_low_cjk_ratio,instruction_leak,zh_very_short
```
各旗标的判定规则速查表见 §2。

**③ S1 + s2-sample 不用等 8B**——SigLIP2/LaBSE 一到就先把 S1 全量
（10–30 分钟）和 `s2-sample`（全覆盖）跑完；8B 落地后直接开 `s2`。

**④ 裁判链路冒烟（不依赖任何本地权重）**——用 5 条验证 `vlm_audit.py` 完整链路
（路径解析→抽帧→base64→API→JSON 解析），顺便提前看裁判的判定质量。
**默认走内网 `Qwen3.6-27B` 端点，不需要任何 API key**（前提是本机连得到
`172.30.x.x`，见 §4.3-端点）：

```bash
python - <<'PY'
import json
recs=[json.loads(l) for _,l in zip(range(5), open("/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl"))]
with open("outputs/audit_10k/api_smoke.jsonl","w") as f:
    for r in recs:
        f.write(json.dumps({"video":r["video"],"abs_path":r.get("abs_path"),
          "zh":r["raw_caption_zh_translated"],"en":r["caption"]},ensure_ascii=False)+"\n")
PY
python tools/data_audit/vlm_audit.py --input outputs/audit_10k/api_smoke.jsonl \
  --video-root /home/vca1/datasets/internvid_zh_10k/videos \
  --api-base http://172.30.85.198:8000/v1 --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 2 \
  --output outputs/audit_10k/api_smoke_verdicts.jsonl
cat outputs/audit_10k/api_smoke_verdicts.jsonl
# 用云端 key 时改回：--api-base https://dashscope.aliyuncs.com/compatible-mode/v1
#   --api-key $JUDGE_API_KEY --model qwen3-vl-235b-a22b-instruct
```

**⑤ Gradio 人工摸底（零权重、零 GPU）**——按 §6 优先翻
`quality_issue_candidates.jsonl`，频次表与 ② 的观察互相印证。

以下 §1–§5 全部封装在 `tools/data_audit/run_v100.sh`（内置 8 卡 fan-out、
断点续跑、验收检查），每步给出一键命令；想看展开的等价命令可
`cat tools/data_audit/run_v100.sh` 对应分支。

## 1. 冒烟（~10 分钟，必做）

```bash
bash tools/data_audit/run_v100.sh smoke
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
bash tools/data_audit/run_v100.sh s0
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
bash tools/data_audit/run_v100.sh s0-show          # 每类打印 5 条 zh/en 实例
SHOW_N=20 bash tools/data_audit/run_v100.sh s0-show   # 想多看几条
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
bash tools/data_audit/run_v100.sh s1        # 8 卡后台启动
bash tools/data_audit/run_v100.sh status    # 随时看进度
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
source ~/audit_env/bin/activate && cd ~/mobileclip
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
bash tools/data_audit/run_v100.sh s1-report      # -> outputs/audit_10k/s1_analysis.md
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
> [`notes/audit-10k-findings.md`](../../notes/audit-10k-findings.md)**
> （`outputs/` 被 gitignore，报告本身不入库）。要点：`xling_cos` 在本批是
> **弱工具**（ρ(sig_zh,xling)=0.04、动态范围窄），语言污染要看 S0 旗标而非
> xling；且 S3 默认 `--xling-threshold=0.7` 正落在本批 p25 附近，**跑 s3 前
> 先读那份笔记**。

## 4. S2 VLM 审核（**推荐 10k 全覆盖**，8 卡 ~2–3 小时）〖4.2 主力 / 4.3 裁判 / 4.4 人工抽检可并行：裁判走外部端点，不占本机 GPU〗

```bash
# 4.1 抽样——推荐全覆盖：默认参数（每分位桶 3000×10 桶）在 10k 上即选中
#     全部有分记录，直接跑，不加任何收缩参数：
bash tools/data_audit/run_v100.sh s2-sample
wc -l outputs/audit_10k/s2_sample.jsonl        # 应 ≈10000
#     （赶时间的快速版才用收缩参数，~1.4k 条 1 小时：
#      PER_BUCKET=100 FLAGGED=300 LOW_XLING=200 bash tools/data_audit/run_v100.sh s2-sample）

# 4.2 主力档：Qwen3-VL-8B，transformers 后端 8 卡数据并行
#     全覆盖 10k ≈ 每卡 1250 条 × 6–8s ≈ 2–3 小时；可断点续跑
bash tools/data_audit/run_v100.sh s2
bash tools/data_audit/run_v100.sh status

# 4.3 裁判档 —— 当前默认走【内网已有的 Qwen3.6-27B 端点】（无需 API key、
#     无需下权重、不占本机 GPU，可与 4.2 主力同时跑，只依赖 4.1 抽样完成）
#     ⚠️ 前提：本机能连到 172.30.x.x（不同网段，必须先实测，见下方 4.3-端点）
export JUDGE_API_BASE=http://172.30.85.198:8000/v1    # 优先 8 卡端点
export JUDGE_MODEL=/workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B
export JUDGE_API_KEY=EMPTY                            # vLLM 不校验，但脚本要求非空
JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge   # 跑完自动算一致率+κ
# 裁判在别的机器上跑完、只把 s2_verdicts_judge.jsonl 拷回来时，单独重算：
#   bash tools/data_audit/run_v100.sh judge-agree
# （其他裁判路线——H100 本地 235B-FP8 / 本机 32B / 无大裁判协议——
#   全部选型与命令见 SETUP_JUDGE_API.md §0 的路线表）

# 4.4 人工抽检 300 条（在 §6 的 Gradio 里过）
shuf -n 300 <(cat outputs/audit_10k/s2_verdicts.*.jsonl) > outputs/audit_10k/human_check_300.jsonl
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
  分桶、`xling` 阈值裁决（`calibrate_xling.py`）、以及 `better_zh` 直接进
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

**决策门**（由 `bash tools/data_audit/run_v100.sh judge-agree` 一次算全，
产出 `judge_agreement.md`）：五类 `action` 一致率 **≥75%**、二值
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

---

### 4.3-端点 裁判档首选：内网已有的 Qwen3.6-27B 端点（零部署，**先试这条**）

B100 集群上已经有别人起好的 vLLM OpenAI 兼容服务，**模型已确认支持视觉输入**
（2026-07 实测），可直接当裁判——不用下权重、不占本机 8 张卡、不花 API 费用。
**能连通的话，这条优于 4.3-本地的 32B**（那个要下 64GB 还要占满本机卡）。

| 端点 | 起法 | 建议 |
|---|---|---|
| `http://172.30.85.198:8000/v1` | 8 卡 | **首选**（并发余量大） |
| `http://172.30.62.177:8000/v1` | 8 卡 | 备选 |
| `http://172.30.86.57:5000/v1` | 4 卡 | 前两个不可用时再用 |

模型 ID 均为 `/workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B`
（起服务时没设 `--served-model-name`，**对外 ID 就是这串权重路径，原样传**；
不确定时先 `curl $JUDGE_API_BASE/models` 看一眼）。

**第一步永远是连通性实测**——`172.30.x.x` 与本机 `10.235.31.101` 不同网段：

```bash
for ep in 172.30.85.198:8000 172.30.62.177:8000 172.30.86.57:5000; do
  echo -n "$ep: "
  timeout 5 curl -s -o /dev/null -w "%{http_code}\n" "http://$ep/v1/models" || echo unreachable
done
```

- **连得上** → 用 §4 命令块里 4.3 的三个 `export` 直接跑 `s2-judge`；
  正式跑之前先用 §0.5 ④ 的 5 条冒烟验证整条链路。
- **连不上** → **实测结果就是连不上**（101vector 和 H100 都不行，2026-07）。
  **裁判改到 7 卡 V100 `root@10.55.87.68 -p 31179` 上跑** —— 该机已确认能访问
  `/workspace/mount/b100_zaip_data`（端点的模型权重和全量语料都在该挂载下），
  多半在 B100 网内。10k 记录保留了 `abs_path`（`resolve_video_path` 第三候选），
  **很可能连视频都不用拷，只需搬一个 500 行的 `s2_judge_sample.jsonl`**；
  真要拷也**只拷 `JUDGE_N` 条，不是全部 1 万条**。
  **逐条命令见 [`SETUP_JUDGE_RELAY.md`](SETUP_JUDGE_RELAY.md)**
  （三项前提实测 → 轻量环境 → 送脚本和样本 → 视频解析预检 → 冒烟 →
  正式跑 500 → 回传算分 → 常见问题表）；选型理由见
  [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0 路线 A′。

  > ⛔ **别在中转机上重新抽样**：裁判必须评的是主力判过的**同一批记录**，
  > 从全量 10M 另抽一批与本机 10k 交集可能为空，`judge-agree` 会直接报
  > 「重叠不足」。搬 `s2_judge_sample.jsonl`，不要重抽。

**参数已按内网既有调用代码对齐完毕**（2026-07，逐项比对表见
`SETUP_JUDGE_API.md` §0 路线 A′）。三个结论：

- 🔴 **`vlm_audit.py` 现在默认发送 `chat_template_kwargs.enable_thinking=False`**。
  Qwen3.x 的 thinking 默认开启，不关的话模型会把 `max_tokens` 全花在 `<think>` 里、
  没轮到输出 JSON 就被截断，**每一条裁判都会失败**。这项不是优化，是必需。
- 超时默认 180→**600 秒**（house 值），重试补上了**指数退避**（原本是立刻重打）。
- 并发：house 脚本用 `MAX_WORKERS=32`，我们的 `--concurrency 8` **远低于它，安全**
  （此前担心"打满别人服务"的顾虑可以取消）。

> ⚠️ **证据强度打折**：27B 裁 8B 只有约 3 倍参数差距，且同厂同族——
> 方案 §2 原本要的是 235B **加换一家**（InternVL）互为第二意见，正是为了避开
> 共同盲区。**同族模型的一致率会被共同盲区抬高，一致不等于正确。**
> 所以跑完 `judge-agree` **要看 κ 而不只是一致率**，并按
> `SETUP_JUDGE_API.md` §0 路线 D 的 D2/D3 提高 4.4 人工抽检的权重。

### 4.3-本地 裁判档备选：本机部署 Qwen3-VL-32B（不用 H100、不用云端 API）

H100 机用不上、又不想调商业 API 时走这条：**32B 是 8×V100-32G 能装下的
最强裁判**（235B 在 Volta 上不可能——无 FP8 硬件支持，BF16 原版 ~470GB
也放不下 256GB 显存）。32B 仍是主力 8B 的 4 倍档，一致率有意义。

选型与其他备选的权衡见 [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) 的
「裁判选型」一节；
**连 32B 都跑不动就走那里的路线 D**（换族 8B + 分歧定向人工抽检）。

> ⏱ **排期提醒**：与云端裁判不同，本地裁判**要占本机 GPU**，不能和 4.2 主力
> 抢卡。两种排法：① 等 4.2 跑完再跑（简单，推荐）；② 对半分——主力
> `NGPU=4` 用 GPU0-3，裁判用 GPU4-7（见下方"两进程并行"，把
> `CUDA_VISIBLE_DEVICES` 改成 `4,5,6,7` 单进程即可）。

#### 选哪个模型：Qwen3-VL-32B（求稳）vs InternVL3.5-38B-HF（求证据强度）

下面步骤 0–5 对两者通用，**只有 `--model` 不同**。取舍：

| | `Qwen3-VL-32B-Instruct`（~64GB） | `OpenGVLab/InternVL3_5-38B-HF`（~76GB） |
|---|---|---|
| 与主力 8B 的关系 | **同族**（同为 Qwen3-VL） | **半跨族**：视觉塔跨族、语言塔同族 ⚠️ |
| 一致率的含义 | 共有盲区会让两边一起错，偏乐观 | **画面对齐**维度误差较独立；**翻译忠实度**维度仍共享 Qwen3 盲区 |
| 集成风险 | 几乎为零（与已跑通的 8B 同架构） | **低**（已实测：原生 HF 格式，无 `trust_remote_code`，不用动环境） |

> 📌 **2026-07-21 实测更正——「跨族」只对了一半。** 上面那条探针命令跑通了，
> 仓库存在，`config.json` 同时也把独立性的底牌翻了出来：
>
> ```json
> "model_type": "internvl",
> "architectures": ["InternVLForConditionalGeneration"],
> "vision_config": { "model_type": "internvl_vision", "hidden_size": 3200, "num_hidden_layers": 45 },
> "text_config":   { "model_type": "qwen3", "_name_or_path": ".../Qwen3-32B", "num_hidden_layers": 64 }
> ```
>
> **InternVL3.5-38B ＝ InternViT（~6B）＋ Qwen3-32B。** 视觉塔确实换了族，
> 但**语言塔还是 Qwen3** —— 与主力 `Qwen3-VL-8B`、A′ 裁判 `Qwen3.6-27B` 同族。
>
> 这件事直接影响该信谁：裁判要打的几个分里，**`trans_score`（翻译忠实度）
> 几乎是纯语言任务**，走的正是共享的 Qwen3 语言塔；而 §3 的 T4 明说
> **「决策 1 的全部差率都建立在 `trans_bad` 那一行上」**。也就是说
> **最需要独立第二意见的那一维，恰恰是这个模型最不独立的一维。**
> 反过来，`en_score`/`zh_score` 这类**画面对齐**判断吃视觉塔，换族收益是真的。
>
> **怎么用**：它仍比 Qwen3-VL-32B 独立（后者两塔同族），值得选；但
> **`trans_score` 上一致率高不要读成「判定可信」**，那更可能是同族语言塔的
> 共有盲区。这一维要么挂人工抽检（§3 T4 的路线 D），要么另找语言塔也跨族的
> 模型（如 Gemma-3-27B-IT，Gemma 语言塔 + SigLIP 视觉塔，真跨族；代价是
> 中文与 rubric 遵循度都没在本项目验证过，且视觉侧与 S1 用的 SigLIP2 同源）。
> **别为此加跑一档**——先按现有计划走，把这条写进报告的方法学局限即可。
>
> 另两条已确认：**无 `auto_map`** → 原生 transformers 实现，`trust_remote_code`
> 不需要，§0 的 `audit_env`（transformers ≥4.57 ≥ 所需的 4.52.1）直接能加载，
> **不用另建 `judge_env`**；`torch_dtype` 为 **`bfloat16`** → V100 无 bf16，
> 必须按下面的冒烟检查 ① 用 fp16 载入并验中文是否乱码。

> ⚠️ **`-HF` 后缀是硬性要求，下错等于白下。** InternVL 官方发两种格式：
> **GitHub 格式**（`OpenGVLab/InternVL3_5-38B`，无后缀）要 `trust_remote_code`，
> 调用方式是自定义的 `model.chat()` + 自建 `pixel_values`，与
> `vlm_audit_hf.py` 用的 `AutoModelForImageTextToText`/`AutoProcessor`
> **完全不兼容**，等于重写审核脚本；
> **`-HF` 格式**是原生 transformers 实现，支持 `eager`/`sdpa`
> （Volta 没有 flash-attn，这点必须满足），现有脚本 `--model` 一换就能跑。
> 需 `transformers>=4.52.1`（§0 装的 ≥4.57 已满足，**不用动环境**）。
>
> 开下之前先花 10 秒确认仓库存在：
>
> ```bash
> modelscope download --model OpenGVLab/InternVL3_5-38B-HF \
>   --local_dir /tmp/ivl_probe config.json
> python -m json.tool /tmp/ivl_probe/config.json | head -20
> ```
>
> 拿不到就试 `OpenGVLab/InternVL3_5-38B-Instruct-HF`；两个都没有说明这一档
> 只有 GitHub 格式 → **放弃跨族、留在 Qwen3-VL-32B**（重写脚本的代价
> 超过独立性收益）。
>
> **换族模型下完必须先跑 2 条冒烟**，两个检查点：① fp16 下中文是否正常
> （InternVL3.5 是 bf16 训练，38B 比 32B 更易溢出；真乱码就 `--dtype fp32`，
> 152GB 八卡放得下、速度减半）；② 输出能否解析出 `action` 字段——
> **rubric 是照 Qwen 的行为写的**，换族模型可能不守格式，那就要调 prompt
> 而不是硬跑 500 条。

#### 步骤 0：预检（1 分钟）

**不需要新建环境——直接用 §0 的 `audit_env`，零新增依赖。** 32B 与 8B 主力
走的是**同一个脚本、同一套依赖**（`vlm_audit_hf.py` + torch 2.5.1+cu118 +
transformers≥4.57 + accelerate），区别只在 `--model` 和 `--device-map`。
§0 里 `accelerate` 已经装了（8B 用 `device_map` 流式载入就依赖它），
所以这里**什么都不用装**。

```bash
source ~/audit_env/bin/activate && cd ~/mobileclip
df -h .                     # 验收点：剩余 ≥80GB（权重 ~64GB + 余量；8B 的 17GB 已占）
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
                            # 验收点：8 张 32GB 卡，且 4.2 主力已结束（used 接近 0）
python -c "import torch, transformers, accelerate; print(torch.__version__, \
  transformers.__version__, accelerate.__version__)"
                            # 验收点：torch 2.5.1+cu118 / transformers ≥4.57 / accelerate 有版本号
                            # 三者齐全即可开跑；缺 accelerate 才需 pip install accelerate
```

> ⚠️ **审核期间不要动这个环境的版本**。两条红线：
> ① **`pip install -U torch/transformers` 属于高危操作**——不只是可能打断
> 正在跑的 4.2，更要命的是**换了库版本，8B 的输出也会变**，先后跑出来的
> 分片判定不再可比，一致率和 S3 校准都失去意义。整个审核周期内把
> `audit_env` 当成冻结环境，要升级就等 S3 出完报告。
> ② **绝对不要往 `audit_env` 里装 vllm**——它会带上自己的 cu12x torch，
> 直接废掉 Volta（同 `CLAUDE.md` 里 open_clip_torch 那个坑）。V100 路径全程用不到 vllm。
>
> **什么时候才真的需要另开环境**：换用**非 Qwen 系**裁判且它要求不同的
> transformers 版本时（例如某些 InternVL 版本需 `trust_remote_code` 配特定
> 版本）。这时新建 `~/judge_env` 装它那套，让 `audit_env` 保持冻结——
> 隔离的目的是**保住主力判定的可复现性**，不是怕装不上。

#### 步骤 1：下载权重（~64GB）

**先测速再选通道——不要盲目挂着等。** 64GB 在不同速率下差别是几十倍：

```bash
D=ckpts/Qwen3-VL-32B-Instruct
a=$(du -sm $D 2>/dev/null | cut -f1 || echo 0); sleep 60; b=$(du -sm $D | cut -f1)
echo "速率 $(( (b-a) )) MB/s ；已下 ${b} MB / 65536 MB ；按此速率还需 $(( (65536-b)/((b-a>0?b-a:1)*60) )) 小时"
```

| 实测速率 | 64GB 预计耗时 | 该怎么办 |
|---|---|---|
| **< 2 MB/s** | > 9 小时 | **必须换通道**，按下面 A→D 逐级试 |
| 2–10 MB/s | 2–9 小时 | 先试 A（多连接），不行就后台挂着过夜 |
| > 10 MB/s | < 2 小时 | 就这样，别折腾 |

> **注意区分两种"慢"**（§0 和 `TROUBLESHOOTING.md` 讲的是前者）：
> **链路不稳**（反复 `IncompleteRead`）→ 降并发 `--max-workers 1` + until 循环；
> **链路稳但慢**（速率恒定在几百 kB/s，不报错）→ **降并发是反效果**，
> 要的是**多连接并行**（下面方案 A/C）。

**A/B/C 谁更快？取决于卡在哪一环——按下面的顺序试，别猜：**

| 方案 | CDN | 并行方式 | 何时它最快 |
|---|---|---|---|
| A `aria2c -x 16` | hf-mirror | 单文件 16 连接 | 服务端**按连接限速**时 |
| B `hf_transfer` | hf-mirror | 单文件多线程（连接少、不可调） | 同 A 但幅度小，胜在零配置 |
| C `modelscope --max-workers 4` | **ModelScope（阿里）** | 4 文件并行、每文件 1 连接 | **路由本身是瓶颈**时（Qwen 系国内节点通常最优） |

**关键判断：A 和 B 走同一个 hf-mirror**，只是连接数不同。所以
**B 已经慢 → A 的上限也有限**（说明是这条链路整体拥堵，不是单连接被限速），
这时应该换 CDN（C）而不是加连接（A）。

试探顺序按「收益 ÷ 折腾成本」：**B（若已在跑，零成本，先测）→ C（modelscope
早已装好，只改一个参数，且换的是最可能更优的国内路由）→ A（要装 aria2c，
成本最高但天花板也最高）**。注意 §0 给 8B 用的是 `--max-workers 1`，那是
**链路不稳**的处方；稳但慢时改 4，若瓶颈是单连接限速，这一步直接就是 4 倍。

换通道最多损失**一个正在下的分片**（~4GB），已下完的分片各家都会跳过，
所以试错很便宜——但**务必先 `pkill` 干净再换**，别两个通道同时跑。

> 💡 在 A/B/C 之间纠结之前，**先花两分钟确认方案 D**：A/B/C 是在"几小时"里
> 挑，D（内网 rsync，64GB 约 10 分钟）是直接把问题消掉。

**方案 A：aria2c 多连接（单连接被限速时提升最明显，通常 3–10×）**

```bash
# aria2c：优先包管理器；无 sudo 时用 conda 或静态二进制
sudo apt install -y aria2 || conda install -y -c conda-forge aria2
aria2c --version    # 验收点：打印版本号

wget https://hf-mirror.com/hfd/hfd.sh && chmod +x hfd.sh
./hfd.sh -h         # 先看本版支持哪些参数（不同版本 --local-dir 名称有差异）
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
nohup ./hfd.sh Qwen/Qwen3-VL-32B-Instruct --tool aria2c -x 16 \
  --local-dir ckpts/Qwen3-VL-32B-Instruct > download_32b.log 2>&1 &
tail -f download_32b.log
```

`-x 16` = 每个文件 16 条连接。**若速率没变化**，说明瓶颈是总带宽而非
单连接限速，直接跳到方案 D。

**方案 B：hf_transfer（Rust 多线程后端，装起来最省事）**

```bash
pip install -U "huggingface_hub[cli]" hf_transfer
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
nohup bash -c 'until hf download Qwen/Qwen3-VL-32B-Instruct \
  --local-dir ckpts/Qwen3-VL-32B-Instruct; do sleep 10; done' \
  > download_32b.log 2>&1 &
```

代价：hf_transfer 没有细粒度进度条、报错信息也更含糊；出问题就
`unset HF_HUB_ENABLE_HF_TRANSFER` 退回普通模式定位。

> ⛔ **101vector 实测：方案 B 在本机走不通，别浪费时间。** `audit_env` 里的
> `huggingface_hub` 是新版，**已弃用 `hf_transfer`、改用 Xet**（会打
> FutureWarning 让你改用 `HF_XET_HIGH_PERFORMANCE`），而 hf-mirror 给不出
> Xet 文件的完整元数据 → 本地没有的文件一律 `Error: Local entry not found`
> 致命退出，`until` 循环每 10 秒空转一次、一个字节都下不来。
> `HF_HUB_DISABLE_XET=1` 在这个版本救不回来。**直接走方案 C**（ModelScope，
> 另一套服务、不涉及 Xet）。诊断细节见 `TROUBLESHOOTING.md` 同名条目。
>
> 日志里 `Couldn't access the Hub ... Defaulting to existing file` 是**良性**的
> （小文件本地已有、跳过校验）；`Error: Local entry not found` 才是致命的那条。
> 也**不要为此降级 `huggingface_hub`**——它是 `transformers` 的依赖，
> 审核期间 `audit_env` 必须冻结（步骤 0 的红线）。

**方案 C：ModelScope 提高并发**（Qwen 系在国内通常 ModelScope 路由最好）

```bash
nohup bash -c 'until modelscope download --model Qwen/Qwen3-VL-32B-Instruct \
  --local_dir ckpts/Qwen3-VL-32B-Instruct --max-workers 4; do sleep 10; done' \
  > download_32b.log 2>&1 &
```

注意这里是 `--max-workers 4` 而非 §0 的 `1`——**稳但慢**用 4，**不稳**才用 1。

**方案 D：内网直接拷（有第二台机器时最快，且不吃外网）**

H100 机（或任何已有该权重的机器）到本机通常是千兆内网 ≈ 110 MB/s，
**64GB 约 10 分钟**，比任何外网通道都快：

```bash
# 先按 SETUP_JUDGE_API.md §1.2 的检索脚本确认对方机器上有没有现成权重
rsync -avP --partial <user>@<H100_IP>:~/mobileclip/ckpts/Qwen3-VL-32B-Instruct/ \
  ckpts/Qwen3-VL-32B-Instruct/
```

`-P` 断点续传，中断了重跑同一条命令即可。若两机挂同一存储，直接 `ln -s`
省掉拷贝。

**方案 E：换更小的裁判**——上面都试过仍要 5 小时以上时，权衡一下：
`InternVL3_5-8B` 只有 ~17GB（1/4 体量），走
[`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) 的 **「路线 D：完全没有大裁判时的
替代校准协议」**（换族 8B + 分歧定向人工抽检）。判定强度弱于 32B，但**今晚就能跑完**，而不是明天还在下载。

> ⚠️ **不要在多个工具之间反复横跳**：已完整下好的分片各家都会跳过，但
> **下到一半的文件**在换工具后多半要从头来（临时文件命名和续传约定不同）。
> 换通道请在"当前文件下完"的间隙换，且换完就别再换回去。

> ⚠️ **同一时刻只跑一个下载进程**。手滑把 `nohup ... &` 敲了两遍时：
> ① 两个进程都用 `>` 打开**同一个日志文件**，后者截断前者、两边各按自己的
> 偏移量写——**日志直接变成乱码，进度和报错都看不到了**，而这恰恰是你现在
> 最需要的信息；② `hf_transfer` 本身已是多线程，再开一个只是抢带宽，
> 限速链路上反而更慢。（写坏文件的风险倒不大：`huggingface_hub` 对每个文件
> 加锁 `.cache/huggingface/download/*.lock`，两边会按文件串行。）
>
> ```bash
> pgrep -af "hf download|modelscope download"   # 看清楚在跑几个
> kill %3            # 多开的那个 job（%N 见 jobs 输出）
> pkill -f "hf download"   # until 循环已停，收掉残留子进程不会自动重启
> ```
>
> 杀掉不丢进度（`.incomplete` + 元数据续传），重启一个即可。
>
> **`HF_HUB_ENABLE_HF_TRANSFER=1` 的坑**：若 `hf_transfer` 包没装，
> `hf download` 会**秒退**并报 "Fast download using 'hf_transfer' is enabled
> but 'hf_transfer' package is not available"——外层 `until` 循环会每 10 秒
> 重试一次、**永远下不动**。所以启动后**第一件事是看日志**：
>
> ```bash
> tail -20 download_32b.log
> python -c "import hf_transfer; print('OK', hf_transfer.__version__)"  # 没装就 pip install hf_transfer
> ```

下完必做完整性校验（步骤 2）——半个分片会在加载时才报错，很浪费时间。

#### 步骤 2：验完整性（必做——断过的下载会缺分片，加载时才报错很浪费时间）

```bash
du -sh ckpts/Qwen3-VL-32B-Instruct       # 验收点：≈64G
python - ckpts/Qwen3-VL-32B-Instruct <<'PY'
import json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
print("model_type:", cfg.get("model_type"))          # 应含 qwen3_vl
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
have = [f for f in need if os.path.exists(os.path.join(d, f))]
print(f"分片: {len(have)}/{len(need)} {'齐全' if len(have)==len(need) else '不完整!'}")
tot = idx["metadata"]["total_size"]
size = sum(os.path.getsize(os.path.join(d, f)) for f in have)
print(f"字节: {size:,} / {tot:,} {'OK' if size >= tot else '下载不完整!'}")
PY
```

#### 步骤 3：冒烟 5 条（~10 分钟，含首次加载）

32B fp16 权重 64GB > 单卡 32GB，**必须模型并行**：`--device-map auto`
让 accelerate 把层切到多张卡，同时 `--num-shards 1`（模型已占多卡，
再起多进程必 OOM）。

```bash
OUT=outputs/audit_10k
head -n 5 $OUT/s2_sample.jsonl > $OUT/s2_judge_smoke.jsonl
python tools/data_audit/vlm_audit_hf.py \
  --input $OUT/s2_judge_smoke.jsonl \
  --video-root /home/vca1/datasets/internvid_zh_10k/videos \
  --model ckpts/Qwen3-VL-32B-Instruct --dtype fp16 \
  --device-map auto --num-shards 1 --shard-id 0 \
  --output "$OUT/s2_verdicts_judge_smoke.jsonl"
python -m json.tool <<< "$(head -1 $OUT/s2_verdicts_judge_smoke.jsonl)"
```

**验收点**：① 5 行输出，`verdict` 里 `en_score`/`zh_score`/`action` 齐全、
中文可读（不是乱码）；② 加载时没有 `Some parameters are on the meta device`
或 CPU/disk offload 提示。确认切分方式：

```bash
python - <<'PY'
import torch
from transformers import AutoModelForImageTextToText
m = AutoModelForImageTextToText.from_pretrained(
    "ckpts/Qwen3-VL-32B-Instruct", torch_dtype=torch.float16,
    attn_implementation="sdpa", device_map="auto")
devs = set(str(v) for v in m.hf_device_map.values())
print("使用设备:", sorted(devs))     # 验收点：全是 0..7，出现 'cpu'/'disk' 即降级了
PY
```

出现 `cpu`/`disk` 说明显存判定不足（多半有残留进程占卡）：
`nvidia-smi` 杀掉残留后重试——**CPU offload 会让速度慢 10 倍以上，
宁可不跑也别带着 offload 跑完 500 条**。

#### 步骤 4：正式跑 500 条

`device_map auto` 是**流水线并行**（同一时刻只有一张卡在算），单进程吞吐
约 30–60 秒/条 → 500 条约 **4–8 小时**。8 张卡只用了 ~64/256GB 显存，
所以**推荐两进程各占 4 卡做数据并行，墙钟减半（2–4 小时）**：

```bash
OUT=outputs/audit_10k
head -n 500 $OUT/s2_sample.jsonl > $OUT/s2_judge_sample.jsonl

for i in 0 1; do
  gpus=$([ $i -eq 0 ] && echo 0,1,2,3 || echo 4,5,6,7)
  CUDA_VISIBLE_DEVICES=$gpus nohup python tools/data_audit/vlm_audit_hf.py \
    --input $OUT/s2_judge_sample.jsonl \
    --video-root /home/vca1/datasets/internvid_zh_10k/videos \
    --model ckpts/Qwen3-VL-32B-Instruct --dtype fp16 \
    --device-map auto --num-shards 2 --shard-id $i \
    --output "$OUT/s2_verdicts_judge.{shard}.jsonl" \
    > $OUT/judge_shard$i.log 2>&1 &
done

# 进度（每片应到 250）：
wc -l $OUT/s2_verdicts_judge.*.jsonl; pgrep -fc vlm_audit_hf
```

子集口径与云端裁判一致（`head -n 500` = 语料原序前 500 条，非分层抽样），
稀有层可能只占个位数——理由与补测建议见上一节「4.2 与 4.3 分别在做什么」细节 1。

4 卡放 64GB 权重余量充足（128GB 总量）；**不要再切成 4 进程 × 2 卡**——
64GB 权重正好塞满 2×32GB，没有激活/KV 余量，必 OOM。中断了重跑同一条
命令即可断点续跑（按 `video` 去重）。

#### 步骤 5：算一致率（决策门）

```bash
OUT=outputs/audit_10k python - <<'PY'
import glob, json, os
out = os.environ["OUT"]
NEED = {"retranslate", "recaption", "drop"}      # 需干预 vs 文本侧可用
judge = {}
for p in glob.glob(f"{out}/s2_verdicts_judge.*.jsonl"):
    for l in open(p):
        r = json.loads(l); judge[r["video"]] = r.get("verdict", {})
tot = a5 = a2 = 0
for p in glob.glob(f"{out}/s2_verdicts.*.jsonl"):
    for l in open(p):
        r = json.loads(l); v = r.get("verdict", {}); j = judge.get(r["video"])
        if not j or "action" not in v or "action" not in j:
            continue
        tot += 1
        a5 += v["action"] == j["action"]
        a2 += (v["action"] in NEED) == (j["action"] in NEED)
print(f"配对 {tot} 条")
print(f"五类 action 一致率: {a5/max(1,tot):.1%}   <- 决策门 ≥75%")
print(f"二值一致率(是否需干预): {a2/max(1,tot):.1%}   <- 更贴近 S3 分桶，参考 ≥80%")
PY
```

不达标先改 `vlm_audit.py` 的 rubric 再重跑 S2，**不要拿不可信判定去校准
S3 阈值**。二值口径的理由见 `SETUP_JUDGE_API.md`「路线 D」的 D3
（`recaption` vs `retranslate` 的边界连人都会犹豫，五类精确匹配偏苛刻）。

#### 常见问题

| 现象 | 原因与处理 |
|---|---|
| `CUDA out of memory` | 每进程给的卡不足 4 张，或有残留进程占卡（`nvidia-smi` 后 `pkill -f vlm_audit_hf`）；确认没同时在跑 4.2 主力 |
| 加载后 `hf_device_map` 里出现 `cpu`/`disk` | 可用显存不够 → 释放残留进程；跑起来也是 10 倍慢，务必先解决 |
| 输出乱码 / NaN / 全是空判定 | fp16 溢出（32B 比 8B 更易触发，因为原始权重是 bf16）→ 改 `--dtype fp32`（128GB，8 卡单进程可放下，速度约减半），或退回 `SETUP_JUDGE_API.md`「路线 D」|
| 比预估慢很多（>2 分钟/条） | 多半是 offload（见上）或视频解码慢（`--frames 6` 降一档）；流水线并行本身单卡串行，属预期 |
| `KeyError: 'qwen3_vl'` | `transformers < 4.57` → `pip install -U "transformers>=4.57"` |

---

### 4.3-半跨族 备选裁判：InternVL3.5-38B-HF 本机部署（B0–B9 完整流程）

与上面 32B 那条并列的**另一个兜底**。它 ＝ **InternViT ＋ Qwen3-32B**，
所以是**半跨族**（详见上面选型表下的「2026-07-21 实测更正」块）：

| 维度 | 独立性 | 后果 |
|---|---|---|
| `en_score` / `zh_score` / 对齐类 | ✅ **真独立**（视觉塔换了族） | 一致率有信息量，跨族收益成立 |
| `trans_score`（纯文本判断） | ❌ 几乎没有（共用 Qwen3 语言先验） | **这一维仍受同族盲区污染** |

**为什么这仍然值得做**：相比 C 的 `Qwen3-VL-32B`（视觉+语言**双同源**），
它至少在吃视觉的那几维买到了真独立，外加 4× 规模差。
**但要记住它买不到什么**：决策 1（xling 阈值）的全部差率押在 `trans_score`
上，而那恰是它最不独立的一维——**那条结论只能靠人工抽检兜底，
不要因为裁判点头就当它被验证过了。**

代价是集成风险更高（下面 B1/B5/B6 三处验收点就是为它设的）。
**本节自包含**，照 B0→B9 顺序执行即可，不用回看 32B 那节。

#### B0. 先确认该不该走这条

- 路线 A′（中转机 `10.55.87.68` + 27B 端点，零下载）**还没排除**就别开始——
  见 §1「下一步做什么」与 [`SETUP_JUDGE_RELAY.md`](SETUP_JUDGE_RELAY.md)。
- 磁盘要再腾 **~76GB**。已下的 32B 若不再需要，可 `rm -rf ckpts/Qwen3-VL-32B-Instruct` 换空间。
- **只做审查、不动环境**：本模型要求 `transformers>=4.52.1`，§0 装的 ≥4.57 已满足，
  **不需要 pip 安装任何东西**（`audit_env` 审核期间冻结，见步骤 0 的红线）。

#### B1. 探针：确认 `-HF` 仓库存在（10 秒，**下之前必做**）

**必须是 `-HF` 后缀。** 无后缀的 GitHub 格式要 `trust_remote_code`、调用方式是
自定义 `model.chat()` + 自建 `pixel_values`，与 `vlm_audit_hf.py` 用的
`AutoModelForImageTextToText`/`AutoProcessor` **完全不兼容**（等于重写脚本）；
`-HF` 是原生 transformers 实现，支持 `eager`/`sdpa`——Volta 没有 flash-attn，
这一条是硬性前提。

```bash
source ~/audit_env/bin/activate && cd ~/mobileclip
modelscope download --model OpenGVLab/InternVL3_5-38B-HF \
  --local_dir /tmp/ivl_probe config.json
python -m json.tool /tmp/ivl_probe/config.json | head -20
```

**验收点**：能取到 `config.json`，且 `architectures` 里是
`InternVLForConditionalGeneration` 一类的原生类名（**不是**指向
`modeling_internvl_chat.py` 的 `auto_map`）。

取不到就试 `OpenGVLab/InternVL3_5-38B-Instruct-HF`；**两个都没有 →
放弃跨族，回到上面的 Qwen3-VL-32B**（重写审核脚本的代价超过独立性收益）。

#### B2. 预检（1 分钟）

```bash
df -h .        # 验收点：剩余 ≥90GB（权重 76GB + 余量）
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
               # 验收点：8×32GB，且 4.2 主力已结束（本地裁判要占卡，不能与主力抢）
python -c "import transformers, accelerate, torch; print(transformers.__version__, \
  accelerate.__version__, torch.__version__)"   # 验收点：transformers ≥4.52.1
```

#### B3. 下载（~76GB，后台）

> 🔍 **先花 30 秒查 b100 存储——很可能根本不用下。** 中转机
> `10.55.87.68` 能访问 `/workspace/mount/b100_zaip_data`，端点的权重就放在
> 该挂载的 `model_zoo/` 下，InternVL 很可能已经在那儿：
>
> ```bash
> ssh -p 31179 root@10.55.87.68 \
>   'ls /workspace/mount/b100_zaip_data/model_zoo/ | head -50
>    find /workspace/mount/b100_zaip_data -maxdepth 4 -iname "*InternVL*" 2>/dev/null | head'
> ```
>
> 找到且 101vector 连得到该机 → `rsync -avP` 走内网约 100 MB/s，
> **76GB 十几分钟**，比任何外网通道快一到两个数量级。
>
> ⏱ **本网络实测（2026-07-21）**：ModelScope `--max-workers 4` 时
> **每连接稳定在 230–250 kB/s、总吞吐约 1 MB/s → 76GB 要 ~21 小时**。
> 四条连接速率几乎一致，是**按连接限速**的特征而非总带宽跑满，
> 因此**加并发通常近似线性提速**：改 `--max-workers 16` 后测 60 秒，
> 涨到 3–4 MB/s 即证实（~6 小时），仍是 1 MB/s 则说明总带宽就这么多、
> 换通道无用，只能走上面的 b100 存储或放弃这条路线。
>
> ⚠️ **别让它变成关键路径。** 这是路线 C 的备选，主线是 A′（中转机 + 27B
> 端点，几分钟预检）。让它后台慢慢下（断了自动续传、不丢进度），
> 人去推进 A′ 和 `xling-check`——A′ 一通，这 76GB 大概率用不上。

hf-mirror 通道在本机**已实测不可用**（新版 `huggingface_hub` 改走 Xet →
`Local entry not found`，详见 `TROUBLESHOOTING.md`），直接用 ModelScope：

```bash
mkdir -p ckpts
nohup bash -c 'until modelscope download --model OpenGVLab/InternVL3_5-38B-HF \
  --local_dir ckpts/InternVL3_5-38B-HF --max-workers 4; do sleep 10; done' \
  > download_ivl38b.log 2>&1 &
```

测速（<2 MB/s 就换通道，方案 A/D 见上面步骤 1 的表）：

```bash
D=ckpts/InternVL3_5-38B-HF
a=$(du -sm $D 2>/dev/null|cut -f1); a=${a:-0}; sleep 60; b=$(du -sm $D|cut -f1)
echo "$(( b - a )) MB/s ；已下 ${b}/77824 MB"
```

#### B4. 完整性校验（必做）

```bash
du -sh ckpts/InternVL3_5-38B-HF        # 验收点：≈76G
python - ckpts/InternVL3_5-38B-HF <<'PY'
import json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
print("architectures:", cfg.get("architectures"), "| auto_map:", bool(cfg.get("auto_map")))
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
have = [f for f in need if os.path.exists(os.path.join(d, f))]
print(f"分片: {len(have)}/{len(need)} {'齐全' if len(have)==len(need) else '不完整!'}")
tot = idx["metadata"]["total_size"]
size = sum(os.path.getsize(os.path.join(d, f)) for f in have)
print(f"字节: {size:,} / {tot:,} {'OK' if size >= tot else '下载不完整!'}")
PY
```

`auto_map: True` 说明拿到的仍是 GitHub 格式（走远程代码）——**回到 B1 重选**。

#### B5. 加载冒烟 2 条（~15 分钟，含首次加载）

76GB fp16 单卡放不下，必须 `--device-map auto` 跨卡切分 + `--num-shards 1`：

```bash
OUT=outputs/audit_10k
head -n 2 $OUT/s2_sample.jsonl > $OUT/ivl_smoke.jsonl
python tools/data_audit/vlm_audit_hf.py --input $OUT/ivl_smoke.jsonl \
  --video-root /home/vca1/datasets/internvid_zh_10k/videos \
  --model ckpts/InternVL3_5-38B-HF --dtype fp16 \
  --device-map auto --num-shards 1 --shard-id 0 \
  --output "$OUT/ivl_smoke_verdict.jsonl"
python -m json.tool <<< "$(head -1 $OUT/ivl_smoke_verdict.jsonl)"
```

**三个验收点，缺一不可**：

1. **没有 CPU offload**——加载日志不出现 `meta device`/offload 提示。
   不确定就单独查：`hf_device_map` 的值应全是 `0..7`，出现 `cpu`/`disk`
   说明有残留进程占卡（`nvidia-smi` → `pkill -f vlm_audit_hf`），
   带着 offload 跑会慢 10 倍以上。
2. **中文正常**（不是乱码/空串）。InternVL3.5 是 bf16 训练的，V100 只能 fp16，
   38B 比 32B 更容易溢出 → 真乱码就改 `--dtype fp32`（152GB，8 卡放得下，速度减半）。
3. **JSON 能解析出 `action`**——见 B6。

#### B6. 换族特有：rubric 兼容性判读 ⚠️

`vlm_audit.py` 的 rubric 与输出格式是**照 Qwen 的行为调出来的**，
换族模型未必守同样的格式。**这一步不过就别跑 500 条**，否则可能整批空判定。

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open("outputs/audit_10k/ivl_smoke_verdict.jsonl")]
ok = [r for r in rows if isinstance(r.get("verdict"), dict) and "action" in r["verdict"]]
print(f"可解析 {len(ok)}/{len(rows)}")
for r in rows:
    v = r.get("verdict", {})
    print(" ", r["video"], "|", {k: v.get(k) for k in ("action", "en_score", "zh_score", "trans_score")},
          "| error:", v.get("error", "-"))
PY
```

- **2/2 可解析** → 进 B7。
- **解析失败**：先看原始输出是不是被 markdown 包了（```json ... ```）或带前言。
  `extract_json` 会剥 `<think>` 并扫首个合法 JSON 对象，多数情况能兜住；
  兜不住时**改 `vlm_audit.py` 的 `PROMPT` 末尾加一句「只输出 JSON，不要用
  代码块包裹」再重跑冒烟**——不要改解析器去迁就单个模型。
- **字段在但分数明显异常**（例如全 0 或全 5）：说明 rubric 表述对该模型歧义，
  记进 `DATA_ISSUES.md` 并考虑放弃跨族——**一个不守 rubric 的裁判比没有裁判更糟**。

#### B7. 正式跑 500 条

`device_map auto` 是流水线并行（同一时刻只有一张卡在算）。76GB 权重最少要 3 张卡，
给 4 张余量合适，**两进程各占 4 卡做数据并行，墙钟减半**：

```bash
OUT=outputs/audit_10k
head -n 500 $OUT/s2_sample.jsonl > $OUT/s2_judge_sample.jsonl

for i in 0 1; do
  gpus=$([ $i -eq 0 ] && echo 0,1,2,3 || echo 4,5,6,7)
  CUDA_VISIBLE_DEVICES=$gpus nohup python tools/data_audit/vlm_audit_hf.py \
    --input $OUT/s2_judge_sample.jsonl \
    --video-root /home/vca1/datasets/internvid_zh_10k/videos \
    --model ckpts/InternVL3_5-38B-HF --dtype fp16 \
    --device-map auto --num-shards 2 --shard-id $i \
    --output "$OUT/s2_verdicts_judge_ivl.{shard}.jsonl" \
    > $OUT/judge_ivl_shard$i.log 2>&1 &
done

wc -l $OUT/s2_verdicts_judge_ivl.*.jsonl; pgrep -fc vlm_audit_hf   # 每片应到 250
```

**不要切成 4 进程 × 2 卡**：76GB 权重超过 2×32GB，必 OOM。中断重跑同一命令
即可断点续跑（按 `video` 去重）。

> 💡 **InternVL 比 Qwen 更吃显存/更慢的原因**：它用**动态分块**（一张图按
> 分辨率切成多个 tile 再送进视觉塔），8 帧可能膨胀成几十个图块 → token 数
> 远超 Qwen 的同帧数。若 OOM 或慢得离谱，**先降 `--frames 8 → 4`，
> 再降 `--max-side 448 → 336`**（`--max-side` 在送进 processor 前缩图，
> 图小 → 切出的 tile 少）。改了要在 `DATA_ISSUES.md` 记一笔：
> 裁判与主力的输入帧数不同，一致率的可比性会打折扣。

#### B8. 算一致率（决策门）

`judge-agree` 读单个 `s2_verdicts_judge.jsonl`，先把两片合并：

```bash
OUT=outputs/audit_10k
cat $OUT/s2_verdicts_judge_ivl.*.jsonl > $OUT/s2_verdicts_judge.jsonl
wc -l $OUT/s2_verdicts_judge.jsonl            # 应 ≈500
bash tools/data_audit/run_v100.sh judge-agree # -> outputs/audit_10k/judge_agreement.md
```

输出含**五类一致率 + 二值一致率 + 两个 Cohen's κ + 混淆矩阵**。判读口径：
**二值一致率（`{keep,fix_text}` vs 其余）才是 S3 分桶真正依赖的那个**，
五类精确匹配偏苛刻，只作参考；κ<0.6 的维度按计划 §4 回炉改 rubric。
把裁判是谁（InternVL3.5-38B-HF，跨族）写进 `DATA_ISSUES.md`——
**跨族裁判通过的决策门，证据强度高于同族**，这一点值得记下来。

#### B9. 常见问题（InternVL 特有）

| 现象 | 原因与处理 |
|---|---|
| `auto_map` 存在 / 报 `trust_remote_code` | 下成了 GitHub 格式 → 回 B1 换 `-HF` 仓库 |
| `flash_attn` 相关 ImportError | Volta 编不出 flash-attn。`-HF` 变体默认走 sdpa，脚本已显式指定；若仍触发说明拿的不是 `-HF` 格式 |
| OOM | 每进程给的卡 <4 张；或动态分块把 token 撑爆 → 降 `--frames`/`--max-side`（见 B7 提示） |
| 输出乱码 / NaN | fp16 溢出（38B 更易触发）→ `--dtype fp32`（152GB，8 卡单进程，速度减半） |
| 判定能出但 `action` 字段缺失 | rubric 格式不兼容 → 见 B6，改 PROMPT 而不是改解析器 |
| 比 32B 慢很多 | 属预期（动态分块 token 更多）；先确认不是 CPU offload（B5 验收点 1） |

## 5. S3 校准 + 分桶 + 10k 诊断报告（CPU 秒级）〖不可并行：需 S0+S1+S2 全部产物，最后跑〗

```bash
bash tools/data_audit/run_v100.sh s3
```
产出 `outputs/audit_10k/report.md`（10k 问题率表 + sig_zh 分位校准表）和 `buckets.jsonl`。
**这份 10k 报告的两个用途**：① 第一份可汇报的问题率数字（抽样是分层随机的，
比例可外推到全量）；② 校准出的 `sig_zh` 阈值直接作为 H100 机全量 S3 的
`--sig-zh-threshold` 先验。审查到此为止；后续优化见
`notes/internvid-zh-data-audit-plan.md` §5、§7.6+。

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

## 时间线与双机分工

| 步骤 | 本机（10k） | H100 机（全量 10M，见 SETUP_H100.md） |
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
