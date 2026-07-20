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
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu118
# 若升 torch 后仍有 transformers 侧 import 报错，退而求其次锁旧版
# transformers（4.57 已支持 Qwen3-VL 与 SigLIP2，可配 torch 2.4/2.5）：
#   pip install "transformers==4.57.1"

pip install "numpy<2" "transformers>=4.57" sentence-transformers decord pillow requests
# numpy 必须 <2：decord 0.6 预编译 wheel 在 numpy 2.x 下报 _ARRAY_API 错误
# 不要装 vllm —— V100 路径用不上
# ⚠️ 这一步不能跳过（只装 torch 会在 S1 冒烟时报 No module named 'numpy'）；
#    装完自检：
python -c "import numpy, torch, transformers, sentence_transformers, decord, PIL, requests; \
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
modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ckpts/Qwen3-VL-8B-Instruct
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

**④ 云端裁判链路冒烟（不依赖任何本地权重）**——若 API key 已就绪，
用 5 条验证 `vlm_audit.py` 完整链路（路径解析→抽帧→base64→API→JSON 解析），
顺便提前看 235B 的判定质量：

```bash
python - <<'PY'
import json
recs=[json.loads(l) for _,l in zip(range(5), open("/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl"))]
with open("outputs/audit_10k/api_smoke.jsonl","w") as f:
    for r in recs:
        f.write(json.dumps({"video":r["video"],"abs_path":r.get("abs_path"),
          "zh":r["raw_caption_zh_translated"],"en":r["caption"]},ensure_ascii=False)+"\n")
PY
export JUDGE_API_KEY=sk-...
python tools/data_audit/vlm_audit.py --input outputs/audit_10k/api_smoke.jsonl \
  --video-root /home/vca1/datasets/internvid_zh_10k/videos \
  --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 --api-key $JUDGE_API_KEY \
  --model qwen3-vl-235b-a22b-instruct --frames 8 --concurrency 2 \
  --output outputs/audit_10k/api_smoke_verdicts.jsonl
cat outputs/audit_10k/api_smoke_verdicts.jsonl
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

## 4. S2 VLM 审核（**推荐 10k 全覆盖**，8 卡 ~2–3 小时）〖4.2 主力 / 4.3 裁判 / 4.4 人工抽检可并行：裁判走云端 API 不占本机 GPU〗

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

# 4.3 裁判档：云端 235B（任何 OpenAI 兼容端点；500 条约十几元）
#     —— 不占本机 GPU，可与 4.2 主力同时跑（只依赖 4.1 抽样完成）
export JUDGE_API_KEY=sk-...          # DashScope / OpenRouter 等
JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge   # 跑完自动打印一致率
# （替代方案：H100 机在线时，把裁判放它的本地 235B-FP8——H100 起
#   --host 0.0.0.0 服务、本机把 JUDGE_API_BASE 指到其内网地址即可，
#   零 API 费用；分步部署手册见 SETUP_JUDGE_API.md）

# 4.4 人工抽检 300 条（在 §6 的 Gradio 里过）
shuf -n 300 <(cat outputs/audit_10k/s2_verdicts.*.jsonl) > outputs/audit_10k/human_check_300.jsonl
```
**决策门**：8B vs 裁判 action 一致率 ≥75%，不达标先改 rubric 再重跑——
**在 10k 上把 rubric 调可信，正是全量（H100 机）开跑前最重要的准备**。
全覆盖的额外收益：10k 报告是全量实测（零抽样误差）；1 万对
（嵌入分, VLM 判定）配对点让 S3 的阈值校准曲线远比 1.4k 稳，直接
作为 H100 全量跑的先验；且全部 10k 都拿到 `better_zh` 候选修复字幕。
fp16 若出现乱码/NaN 输出（罕见）：`export S2_MODEL=Qwen/Qwen3-VL-4B-Instruct`
并在 `vlm_audit_hf.py` 加 `--dtype fp32` 重跑。

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
