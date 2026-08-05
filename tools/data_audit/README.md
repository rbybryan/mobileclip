# InternVid-zh 数据审核工具集（配套 `notes/internvid-zh-data-audit-plan.md`）

对 `aligned_internvid_zh_translated_retry_merged.jsonl`（10M 条，字段自动探测，
兼容 `video/caption/caption_zh` 等常见命名）做四阶段质量审核：

| 阶段 | 脚本 | 覆盖 | 算力 |
|---|---|---|---|
| S0 规则画像 | `profile_jsonl.py` | 全量 10M | CPU，~20 分钟 |
| S1 嵌入打分 | `score_alignment.py` | 全量 10M | SigLIP2，~100–200 GPU·h |
| S2 VLM 审核 | `make_audit_sample.py` + `vlm_audit.py`（vLLM 服务，Ampere+）或 `vlm_audit_hf.py`（transformers 后端，V100 可用） | 分层抽样 ~4–5 万 | Qwen3-VL-8B，~15–30 GPU·h |
| S3 校准分桶 | `aggregate_report.py`（前置：`calibrate_xling.py` 裁决 xling 阈值） | 全量外推 | CPU |
| S4 落地 v0.2 | `apply_buckets.py` | 全量 | CPU |

> **跑到哪了 / 发现了什么问题** → **[`DATA_ISSUES.md`](DATA_ISSUES.md)**（活账本：
> 流水线状态 + 问题登记 + 待决策）。统计细节见
> [`notes/audit-10k-findings.md`](../../notes/audit-10k-findings.md)。

> **想「用」这 1 万条训练、而不是「审」它？走另一条路。** 本工具集只做质量审核
> （字幕↔视频对齐打分/分桶）。要拿 101vector 上的
> `/home/vca1/datasets/internvid_zh_10k` **跑 10k 试点训练**（`pilot_10k_r0/r1`），
> 见 [`experiments/README.md`](../../experiments/README.md) §2b。三步：
> ① 用 [`tools/make_pilot_anno.py`](../make_pilot_anno.py) 从 `internvid_zh_10k.jsonl`
> 建出 `anno_downstream/internvid_zh_10k_pilot{,_val}.json`（约 9k 训 + 1k 评）；
> ② `export MVC_DATA_DIR=/home/vca1/datasets`，让 corpus 的 `data_root`
> （`configs/data.py` 里的 `internvid_zh_10k/videos`）直接对上那个 `videos/` 目录，
> 免软链；③ `python experiments/check_readiness.py` 确认 pilot 亮灯后启动。
> ⚠️ **字幕默认取英文**（`make_pilot_anno.py --caption-field en`）：文本塔是
> CLIP 英文 BPE；切 `zh` 是另一个实验，别为了「能跑」就改。数据缺失时训练怎么
> 报错、怎么排查见
> [`TROUBLESHOOTING.md`](../../TROUBLESHOOTING.md)「Training dies with `RecursionError`」一节。

## 按机器选手册（两台服务器分工不同）

| 机器 | 手册 | 数据范围 | S2 路径 | 耗时 |
|---|---|---|---|---|
| 8×H100-80G | **[`SETUP_H100.md`](SETUP_H100.md)** | **全量 10M** | vLLM 服务（8B dp=8 + 235B-FP8 裁判本地 tp8） | ~1.5 天 |
| 7×V100（`10.55.87.68`，中转机） | **[`SETUP_JUDGE_RELAY.md`](SETUP_JUDGE_RELAY.md)** | 只跑 **4.3 裁判档** | 内网 `Qwen3.6-27B` 端点（101vector/H100 都连不上，只有它连得到） | ~1 小时 |
| 8×V100-32G（101vector） | **[`SETUP_V100.md`](SETUP_V100.md)** | **仅 1 万条预抽样**（`/home/vca1/datasets/internvid_zh_10k`，本机没有全量数据） | transformers 后端 `run_v100.sh`（无 bf16/vLLM；torch 须用 cu118 构建；裁判走云端 API；S2 推荐 10k 全覆盖） | ~3–4 小时 |

推荐节奏：先在 V100 机把 10k 全流程跑完（半天内出 10k 报告 + 调好 rubric
和 `sig_zh` 阈值），再在 H100 机开全量，校准结果直接复用、避免 10M 规模返工。

人工查看：V100 机（10.235.31.101）已备 1 万条预抽样 + Gradio 浏览工具
（`gradio_video_text_review.py`），定位是**看数据做定性校验、不是标注任务**，
用法与建议流程见 `SETUP_V100.md` §6。

## 并行调度总览（严格依赖链之外的都可同时跑）

严格依赖链只有一条：**冒烟 → S1 → s2-sample → S2 → S3**。其余步骤全部
可以塞进空档，按时间窗分四组：

| 时间窗 | 可同时进行（互不依赖） |
|---|---|
| **t0 开工即并行** | ① 两台机器各自装环境（H100/V100 手册 §0）；② **S0 全量画像**（纯 CPU、零依赖，不用等环境）；③ 大模型权重 ModelScope 下载（后台挂着）；④ **Gradio 人工摸底**（V100 机 1 万条预抽样，翻 100–200 条记问题频次，见 SETUP_V100.md §6） |
| **冒烟通过后** | ① **S1**（单机 8 卡分片并行；**双机对半**：`S1_TOTAL_SHARDS=16`，V100 跑分片 0–7、H100 跑 8–15，前提是两机挂同一数据盘，S1 时长近乎减半）；② 权重下载没完可继续；③ S0 若还没跑完不影响 S1 |
| **S1 完成后** | s2-sample（分钟级）→ 然后 ① **S2 主力**、② **S2 裁判**（只依赖 s2-sample，**不依赖主力结果**）、③ 人工抽检（依赖主力产出，可边出边标）三线并行。V100 裁判走云端 API 天然不占卡；H100 可 GPU 对半（8B dp=4 + 235B tp=4，见 SETUP_H100.md §4）；双机则主力/裁判各占一台 |
| **收尾** | S2 主力一跑完先 `s2-check` 体检（秒级、纯 CPU，见 SETUP_V100.md §4.5）；S3 需要 S0+S1+S2 全部产物，最后跑（分钟级）；4.4 一致率检查需主力+裁判都完成 |

单机版关键并行点：S0 与装环境同时开（当天先出翻译残留率数字）；
权重下载与 S1 重叠；裁判与人工抽检和主力审核重叠。

## 8×V100 快速上手（一键脚本，详细版见 SETUP_V100.md）

V100（Volta）无 bf16、且新版 vLLM（Qwen3-VL 所需）不支持 Volta，
所以 S1/S2 自动用 fp16、S2 用 transformers 后端 `vlm_audit_hf.py`
8 卡数据并行（输出格式与 `vlm_audit.py` 相同）：

```bash
# DATA/CLIPS/OUT 默认即本机（101vector）的 1 万条抽样，无需任何 export；
# 逐条执行——s1/s2 是后台任务，用 status 等到 100% 再进下一步
bash tools/data_audit/run_v100.sh smoke      # 冒烟：字段探测/解码/模型下载
bash tools/data_audit/run_v100.sh s0         # 规则画像（10k 秒级）
bash tools/data_audit/run_v100.sh s1         # 嵌入打分（10k 8 卡 ~10–30 分钟）
bash tools/data_audit/run_v100.sh status     # 等 S1 到 10000/10000
bash tools/data_audit/run_v100.sh s1-report  # S1 联合分析 s1_analysis.md（秒级，进 S2 前必做）
bash tools/data_audit/run_v100.sh s2-sample  # 默认参数在 10k 上即全覆盖
bash tools/data_audit/run_v100.sh s2         # 8 卡 VLM 审核全部 10k（HF 后端，~2–3 小时）
export JUDGE_API_KEY=sk-...                  # 可选：235B 裁判（不想用商业 API →
                                             # 自托管同一份开源权重，见 SETUP_JUDGE_API.md §0）
JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge
bash tools/data_audit/run_v100.sh xling-check # 裁决 xling 阈值（s3 前必做，见 DATA_ISSUES.md §4）
bash tools/data_audit/run_v100.sh s3         # 诊断报告 report.md + buckets.jsonl
```

16GB 的 V100 需 `export S2_MODEL=Qwen/Qwen3-VL-4B-Instruct`。
若 fp16 输出乱码/NaN（罕见的 bf16→fp16 溢出），改用 4B 模型加 `--dtype fp32`。

## 重述抽样检验：50 片全流程 → TSV 目检（主计划 Stage 0，V100 单机可完成）

在花大算力之前，用 RECAPTION 桶头部 50 片把「重述 → 诊断 → A′ 句切视图」整条链跑通，
产出一张 **源caption(en) / 翻译源caption(zh) / 5 条重述 / A′ 视图** 并排的 TSV 供目检：

```bash
source ~/audit_env/bin/activate
# ① 抽样队列（前置: s3+s4 已产 recaption_tasks.jsonl；50 片 × 5 重述 ≈ 8 卡几分钟）
head -n 50 outputs/audit_10k/v0_2/recaption_tasks.jsonl > outputs/audit_10k/v0_2/recap_sample.jsonl
RECAP_INPUT=outputs/audit_10k/v0_2/recap_sample.jsonl bash tools/data_audit/run_v100.sh recap
bash tools/data_audit/run_v100.sh status        # 等 recap 行到 50/50（或看 recap_shard*.log）
# ② 句切 + 诊断 + TSV（CPU 秒级）
bash tools/data_audit/run_v100.sh recap-tsv
# ③ scp outputs/audit_10k/recap_inspect.tsv 回本地，Excel/WPS 直接打开（utf-8-sig）
```

- 输出进 `$OUT/recap5.*.jsonl`，与全量跑同名——**续跑按 video 去重**，抽样的 50 片不会重算；
  之后直接 `bash run_v100.sh recap`（不带 RECAP_INPUT）即无缝转全量。
- TSV 列：`video | 分桶理由 | en_source | zh_translated | recap_1..5 | A′视图 |
  jaccard_med | ident_pairs | COLLAPSE 旗`。目检三问：五条重述有差异吗（COLLAPSE 列）？
  重述比旧翻译明显更全吗？A′ 视图成句、无半截话吗？
- 工具：`split_views.py`（A′ 规则，MiniRBT tokenizer 计 32 token，离线自动落 30 字符预算）·
  `recap_diag.py`（坍缩红旗 + `--compare` 侧批对照）· `make_recap_tsv.py`（汇总表）。

### 优化流水线小样本端到端（`pipeline-sample`）

**2026-08-05 起默认精简为 A/B 方案真正用到的两步**（维护者指令）：① 独立重述 ×N →
④b2 直融（长稿→短稿），**两步都 8 卡分片**；外加 CPU 免费的多样性诊断和 ⑥⑦ 产物。
派生 / SigLIP 打分 / A′ 句切 / 投票浓缩 / GDINO+OCR 验真全部保留但**默认关**，
各由一个 `DO_*` 开关控制（`PIPE_FULL=1` 一键恢复全链）。开关全表 + 实测预期见
[`docs/recaption-ab-pipeline_zh.md`](../../docs/recaption-ab-pipeline_zh.md) §6。

```bash
source ~/audit_env/bin/activate
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
# 产物: outputs/audit_10k/recap_pipeline.tsv + bundle_pipeline/

# 要把这批落成训练数据前，把画面验真加回来（唯一外部幻觉门）
pip install rapidocr-onnxruntime onnxruntime   # OCR 验真档；装不上会自动降级
DO_VERIFY=1 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
```

- 默认口径下 50 片只剩两个 LLM 步骤，都是 8 卡并行；`DO_VERIFY=1` 首次会经 hf-mirror
  下载 GroundingDINO-base（~700MB, fp32）。`PIPE_FULL=1` 的全链仍是单卡串行 ~1 小时。
- **① 的选帧（2026-08-05）**：分段网格 + 相位抖动不变（N 遍独立性的来源），只在段内
  从 3 个候选里挑清晰 / 非纯黑 / 不与上一帧重复的一帧（`frame_pick.py`；
  `FRAME_SELECT=uniform` 回退旧行为，单测 `tests/test_recap_frame_pick.py` 钉死等价）。
- **ASID 三修复（§5.8，维护者批准 2026-07-27）**：① 两版融合稿的 ✗ 句不再直接删——
  `--mode refine` 发一次"只修问题处"的最小修改调用并复验，终稿 `*_fused_*_final.jsonl`
  （修复 prompt 变更后重跑先 `rm pipe*_fused_*_r.*.jsonl`）；② struct 重述 JSON 失败
  三级兜底（重试→逐条抢救→dense 段落兜底），保证不再丢片，`n_dense_fallback` 可查；
  ③ TSV 旗标新增 `MISS_D/MISS_V`：融合产出对 ≥2 票单元 gloss 覆盖 <70% 即标漏说。
- **⏱ 分段计时与全量外推**：每段墙钟自动落 `pipe[_st]_timing.tsv`，结尾打印
  「秒/片 → 全桶小时 → 每 10 万片小时」外推表。**测真实成本必须用冷样本**
  （缓存续跑的段显示 ~0s，外推无效）：`SAMPLE_OFFSET=200 N_SAMPLE=50 ... pipeline-sample`
  从队列第 201 片起取一批没跑过的。**行名带「默认关」的段显示 0s 是被开关关掉，不是快**。
  ④b2 直融已按 `FUSE_GPUS`（默认 8）分片，外推不必再 ÷8；`DO_*` 打开的单卡段仍要。
- 每片一行：`en_source | zh_translated | recap_1..5(⟨SigLIP 分⟩) | A−保留(top-3 重述号) |
  A′句切视图 | ✓过验真视图[gloss] | ·未验(保守留) | ✗被滤视图[gdino/ocr] |
  投票单元(票数) | 淘汰单元 | ✓/·/✗浓缩 | 坍缩旗标`——
  这就是主计划 G2 各臂（A_llm/A−/C_v/**C_fuse**）的样本级预览。
- **固定槽位重述 = 方案B（§5.6b，pipeline-compare 的 B 臂）**：`RECAP_STYLE=slot`——
  `--mode recap-slot` 每遍输出四栏行式（主体≤2/动作≤2/场景/文字，行尾 `[gloss]`，
  **空栏写「无」不许编**），正则解析零 JSON、解码 token 比 struct 降 ~40-50%，投票带
  同槽约束；解析空→0.3 温重试→dense 兜底（`n_dense_fallback`）。产物 `recap_slot.*` /
  `pipe_sl_*`，与 dense/struct 完全隔离。
- **结构化重述 = 旧 B 臂备选（§5.6；手动 `RECAP_STYLE=struct` 可用）**：
  `RECAP_STYLE=struct` 启用——`--mode recap-struct` 每遍一次解码产 5-8 条带属性
  （subject/action/scene/text）≤28 字短句 + 内联 en_gloss，**独立跑 N=3 遍**
  （temp0.9+帧抖动）保住跨遍投票；② 派生步自动跳过。**两臂产物完全隔离**：B 臂重述写
  `recap_struct.*.jsonl`、下游全部带 `_st` 后缀（`pipe_st_*`、`recap_pipeline_st.tsv`、
  `bundle_pipeline_st/`），与 A 臂互不覆盖，无需任何 rm 即可并存。
- **A/B 对比（`bash tools/data_audit/run_v100.sh pipeline-compare`）**：同一批
  N_SAMPLE 片先后跑 A(dense)、B(struct)（各自按 video 续跑，已缓存部分秒过），再由
  `make_style_compare.py` 合成 `review_compare.tsv` + `bundle_compare/`——每片并排
  en源/原zh/A重述/**A直融/A投票**/B重述/**B直融/B投票** + 四选一「最佳浓缩」。
- **直接 LLM 融合「直融」（§5.7，与投票浓缩双版并出）**：`--mode fuse-direct` 每片一次
  文本调用读 N 遍独立重述，**语义投票（≥2 段支持才写）+ 压缩**一步到位，输出 2-5 条
  ≤28 字短句 + gloss，并**自报 dropped**（被排除的单源细节，落在 `verdict.dropped`）。
  输出 `*_fused_d.*`，与下条投票浓缩（`*_fused_v.*`）**每次同时产出、各自验真**，
  单臂评审包并排两列+版本对比单选。**换 prompt 重跑前先
  `rm outputs/audit_10k/pipe*_fused*.jsonl`**。
- **C_fuse 投票浓缩（§5.5，与直融并出的第二版）**：`fuse_condense.py`
  把 A_llm 短句跨 5 条重述聚类计票（纯规则），support≥2 的单元（默认；50 片实测 k=3
  回退率 34/50，对多样性健康的重述太严）交 `--mode condense` 打包成 2–5 条 ≤28 字
  **高浓度短句**，再照走验真。**覆盖优先、不过度浓缩**：全淘时保底留前 3 个单元
  （原为 1），浓缩 prompt 明令"宁多一条、不丢单元信息"；浓缩短句（去 ✗）即 RECAPTION
  桶的训练 `captions`。`FUSE_MIN_SUPPORT` 可改门槛（**重跑前先
  `rm outputs/audit_10k/pipe_fused*.jsonl`**——浓缩按 video 续跑，旧门槛/旧 prompt 的
  结果不会自动重算）；本段任何失败只会让对应列为空，不阻塞 TSV/评审包。评审包按指令
  只显示 en源/原zh/重述/直融/投票浓缩，全景各阶段列仍在 `recap_pipeline.tsv`。
- **模型依赖（V100 盒子现状即可跑，无需任何新 LLM）**：全链**唯一的 LLM = Qwen3-VL-8B**
  （`$S2_MODEL`，本地 `ckpts/Qwen3-VL-8B-Instruct` 优先）——重述用 temp0.9 采样制造投票方差、
  派生用 temp0 贪心保证闭集确定性，两处用法刻意不同。其余全是非 LLM 小模型：SigLIP2 打分
  （s1 轮已缓存）、GroundingDINO-base（~700MB，经 hf-mirror）、RapidOCR（pip 走已配的
  aliyun 镜像）。
- 验真档 v0 = GroundingDINO(按 en_gloss) + RapidOCR(引号文字模糊匹配 ≥0.7)；
  **InternVL 二审档属 Stage 2-6，后补**——当前不可判的视图标 `unverified` 并保守保留。
  **纪律红线：绝不能把 Qwen3-VL-8B 当验真器用**（Qwen 验 Qwen 抓不住系统性幻觉，
  fusion 方案 §3.5 / 审计结论 8'）；验真器装不上就让视图保持 `unverified`，不要找 LLM 替补。
- 复跑安全：重述/验真都按 video 续跑去重；打分文件每次重建。

## 快速上手（A100/H100 等 Ampere+ 机器，vLLM 服务方式）

```bash
DATA=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
CLIPS=/path/to/internvid/clips          # 视频根目录（video 字段相对于它）
OUT=outputs/audit_v0_1
mkdir -p $OUT

# ---- S0：先跑纯文本画像（无 GPU，先看翻译痕迹/未翻译/重复率）----
python tools/data_audit/profile_jsonl.py \
  --input $DATA --out-summary $OUT/s0_summary.json --out-flags $OUT/s0_flags.jsonl

# ---- S1：全量 SigLIP2 打分（8 卡并行示例；可断点续跑）----
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python tools/data_audit/score_alignment.py \
    --input $DATA --video-root $CLIPS \
    --output "$OUT/s1_scores.{shard}.jsonl" \
    --num-shards 8 --shard-id $i --frames 4 --batch-clips 16 &
done; wait

# ---- S2：分层抽样 + VLM 审核 ----
python tools/data_audit/make_audit_sample.py \
  --input $DATA --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --per-bucket 3000 --flagged 10000 --low-xling 5000 \
  --output $OUT/s2_sample.jsonl

vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 \
  --limit-mm-per-prompt image=8 --max-model-len 16384 &
python tools/data_audit/vlm_audit.py \
  --input $OUT/s2_sample.jsonl --video-root $CLIPS \
  --api-base http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct \
  --frames 8 --concurrency 32 --output $OUT/s2_verdicts.jsonl

# ---- S3：校准阈值 + 全量分桶 + 报告 ----
python tools/data_audit/aggregate_report.py \
  --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --verdicts $OUT/s2_verdicts.jsonl \
  --out-buckets $OUT/buckets.jsonl --out-report $OUT/report.md

# ---- S4：分桶落地成 v0.2 数据 + 重翻/重写任务队列 ----
python tools/data_audit/apply_buckets.py --input $DATA \
  --buckets $OUT/buckets.jsonl --verdicts $OUT/s2_verdicts.jsonl \
  --out-data $OUT/v0_2/internvid_zh_v0_2.jsonl \
  --out-retranslate $OUT/v0_2/retranslate_tasks.jsonl \
  --out-recaption $OUT/v0_2/recaption_tasks.jsonl \
  --out-drop $OUT/v0_2/dropped.jsonl
```

先用 `--limit 20000`（S0/S1）和 `--limit 200`（S2）各跑一次冒烟，确认字段探测
（脚本会打印 `[keys] video=... zh=... en=...`）和视频路径拼接正确，再放全量。

## 网络受限机器（国内集群常见：PyPI/conda 被 reset、HF 不通）

1. **S0 零依赖**：`profile_jsonl.py` 只用标准库，任何现成 Python 直接跑，
   不用等环境装好。
2. **跳过 conda，复用训练环境的 torch/decord**（省 700MB+ 下载，且 conda 源
   往往同样不通）：
   ```bash
   conda activate mobileviclip
   python -m venv --system-site-packages ~/audit_env && source ~/audit_env/bin/activate
   pip install -i https://mirrors.aliyun.com/pypi/simple --timeout 120 \
     "transformers>=4.57" sentence-transformers requests
   ```
   TUNA 超时就换 aliyun/ustc/sjtu 源；全不通说明机器要代理
   （`echo $https_proxy`，找管理员）。V100 路径不需要 vllm，别装。
   conda 的国内 anaconda 镜像已普遍不可用（实测 TUNA 403、aliyun 404）：
   **首选按上面方案用 venv 跳过 conda**；一定要 conda 时试 BFSU 镜像
   `conda create -n audit python=3.10 -y --override-channels -c
   https://mirrors.bfsu.edu.cn/anaconda/pkgs/main`；torch 按驱动选构建
   （`nvidia-smi` Driver ≥535 用默认 cu12 轮子，更老的 V100 集群用
   `--index-url https://download.pytorch.org/whl/cu118`）。
3. **完全离线**：在有网的同架构机器 `pip download -d audit_wheels
   --platform manylinux2014_x86_64 --python-version 310 --only-binary=:all:
   "transformers>=4.57" sentence-transformers requests`，拷过去
   `pip install --no-index --find-links audit_wheels ...`。
4. **模型权重**：`export HF_ENDPOINT=https://hf-mirror.com`（写进 `~/.bashrc`），
   或用 ModelScope 预下载后传本地路径：
   ```bash
   pip install modelscope
   modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ckpts/Qwen3-VL-8B-Instruct
   export S2_MODEL=ckpts/Qwen3-VL-8B-Instruct   # run_v100.sh 会直接用本地路径
   ```
   SigLIP2 / LaBSE 同理（`--model`/`--xlingual-model` 均接受本地目录）。
5. 训练环境 torch 偏老时 S1（SigLIP2）通常仍可跑；仅当 S2 加载 Qwen3-VL
   报 torch 版本错误时，再经镜像/离线 wheel 升 `torch>=2.4`（V100 选 cu118）。

## 依赖

在仓库现有环境（含 `decord`、`torch`）之上：`transformers>=4.57`
（SigLIP2 需 ≥4.49，Qwen3-VL 需 ≥4.57）、`sentence-transformers`（LaBSE，
可用 `--xlingual-model ''` 关闭）、`requests`；仅 Ampere+ 走 vLLM 服务方式时
需要 `vllm>=0.11`（建议单独 conda 环境，与训练环境隔离；V100 不需要装）。

## 输出

- `s0_summary.json` / `s0_flags.jsonl`：翻译痕迹、未翻译、重复 caption 等规则命中
- `s1_scores.*.jsonl`：每条 `sig_zh_mean/sig_en_mean/xling_cos`（+解码错误）
- `s2_verdicts.jsonl`：VLM 结构化判定（对齐/翻译/粒度/时序 + 建议动作 + 候选新中文短 caption）
- `s2_health.md`：**S2 判定体检**（`run_v100.sh s2-check`，S2 跑完必做）——
  可用/失败判定数、失败原因拆分、字段完整性、判定塌缩检查、各层可用量。
  判定失败也会照写一行且被下游静默丢弃，不查会系统性低估问题率，见 `SETUP_V100.md` §4.5
- `buckets.jsonl`：全量每条 → `KEEP / FIX_TEXT / RETRANSLATE / RECAPTION / DROP`
- `report.md`：问题率外推表 + 阈值校准表，直接可贴进周报
- `v0_2/internvid_zh_v0_2.jsonl`：训练可用的 v0.2 数据
  （`{"video","caption","captions":[...]}`，loader 直接兼容）
  + 重翻/重写任务队列 + 剔除清单

完整分步操作手册（含环境准备、冒烟验证、每步验收点/决策门、v0.2→v0.3 节奏）
见 `notes/internvid-zh-data-audit-plan.md` §5–§7。
A/B 重述管线（dense×5 vs slot×3）的分步图解、模型清单、prompt 全文与分段计时读法
见 [`docs/recaption-ab-pipeline_zh.md`](../../docs/recaption-ab-pipeline_zh.md)。
