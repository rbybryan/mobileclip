# InternVid-zh v0.1（10M 翻译版）质量审核与 v0.2 优化方案

> 审核对象：`.../caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl`
> （InternVid-10M-FLT 英文 caption → Qwen3 系列 LLM 翻译为中文，含 retry 合并）。
>
> 回答三个问题：① 这版数据可能有哪些问题、各自来自哪个环节；② 用哪些开源
> SOTA VLM（含视频模型）、以什么流程审核 10M 规模的数据；③ 审核结果如何落成
> 一版更好的 v0.2 数据。配套可执行工具在 `tools/data_audit/`（审核四阶段
> S0–S3 + 落地脚本 `apply_buckets.py` 一一对应），完整命令见 §7 操作手册。
>
> 与 `notes/data-quality-caprl-scheme.md`（CapRL 数据强化方案）的关系：本方案是
> 它的**前置诊断层 + 中文化落地**——先量化 v0.1 的病灶分布，再决定重写预算投给谁；
> v0.2 的重写形态直接复用该方案 §4.4 的「K 条 ≤32-token 短字幕」结论。

---

## 1. v0.1 预期问题分类学（按引入环节归因）

这批数据经过了「上游自动字幕 → CLIP 过滤 → 翻译」三级流水线，每级都会引入
特定形态的错误。**关键认知：翻译只是最后一环；相当一部分病灶在英文源头，
只审翻译审不出来。**

| # | 问题 | 引入环节 | 典型形态 | 预估手段 |
|---|---|---|---|---|
| A | **视频-文本不对齐** | InternVid 上游（图像 captioner 逐帧生成 + LLM 汇总，FLT 只是 CLIP 分数前 30%） | caption 描述了不存在的主体/动作；只对某一帧成立、对整段视频不成立 | S1 双语对齐分 + S2 VLM |
| B | **源 caption 事实错误** | 上游 captioner 幻觉 | 数错人数、认错物体、读了水印/字幕文字当内容 | S2 VLM（`en_score`/`en_issues`） |
| C | **翻译错误** | 本次翻译环节 | 误译（多义词按错义翻）、漏译、增译、生硬翻译腔 | S1 `xling_cos` + S2 `trans_score` |
| D | **翻译工程残留** | 翻译环节 + retry 合并 | 指令泄漏（"以下是翻译："）、拒答（"抱歉…"）、`<think>` 泄漏、未翻译（原文照抄）、中英混杂、包裹引号、重复循环、截断 | S0 规则即可抓 |
| E | **粒度粗 / 信息量低** | 上游 caption 天生短平 | "a man is talking" 类笼统句大量重复；实体/属性/场景细节缺失 | S0 重复统计 + S2 `granularity` |
| F | **时序信息缺失或错误** | 上游 caption 无时序意识 | 多事件片段只描述一个事件；先后顺序写反 | S2 `temporal_order_*` |
| G | **分布性问题** | 全链路 | 同一 caption 对应数千个视频（对比学习中互为假负例）；域分布失衡 | S0 去重统计 + S1 嵌入聚类（可选） |
| H | **工程完整性** | 数据管理 | 视频文件缺失/不可解码、clip 命名不合规、JSON 解析失败、重复条目 | S0 + S1 解码错误 |

对训练的伤害排序（对 MobileViCLIP 的 InfoNCE 而言）：A/B（错对齐 = 毒样本）>
D 中的拒答/未翻译（文本塔学到英文或废话）> C > G（假负例）> E/F（不是毒、
是营养不良——上限问题）。

## 2. 审核用开源 VLM 选型（截至 2026-07 的开源 SOTA，均支持视频/多帧）

分三档使用，不是选一个，而是**大模型定标准、小模型跑量、嵌入模型全覆盖**：

| 档位 | 模型 | 用途 | 备注 |
|---|---|---|---|
| **裁判档**（几千条校准集） | Qwen3-VL-235B-A22B-Instruct；InternVL3.5-241B-A28B | 产出"金标"判定，校准 8B 模型与阈值；争议样本仲裁 | MoE，激活 22B/28B，2–4 卡可服务；两家互为第二意见 |
| **主力档**（4–5 万抽样审核，可扩到全量） | **Qwen3-VL-8B-Instruct**（首选）；InternVL3.5-8B；MiniCPM-V-4.5；Keye-VL-1.5-8B | 结构化质检判定（对齐/翻译/粒度/时序 + 建议动作 + 候选新 caption） | 中文原生强、视频多帧输入、vLLM 吞吐高；单卡即可 |
| **嵌入档**（全量 10M） | **SigLIP2-so400m**（多语言，可直接算中文-帧对齐）；LaBSE / BGE-M3（中英文本一致性）；InternVideo2-CLIP（视频原生第二意见，可选） | 每条打 `sig_zh / sig_en / xling` 三个分，撑起全量分桶 | 这是把 10M 审完的关键——VLM 只审抽样，嵌入分负责外推 |

翻译质量专项还可加 **CometKiwi-XL**（reference-free 翻译 QE）做第二意见，
但 `xling_cos` + VLM `trans_score` 一般已够用。

> 为什么主力选 Qwen3-VL-8B：中文能力在 8B 档最强（判定里要写中文短 caption）、
> 原生多图/视频输入、Apache-2.0、vLLM 支持成熟。MiniCPM-V-4.5 / Keye-VL 作为
> 交叉验证：同一批校准集跑两个模型，判定一致率低的维度说明 rubric 要改。

## 3. 四阶段审核流水线（工具已就位，`tools/data_audit/`）

设计原则：**便宜的信号全覆盖，昂贵的信号管校准**。10M 条不可能全上大 VLM，
但可以让全量的嵌入分经过抽样 VLM 判定的校准后具备"问题概率"语义。

```
S0 规则画像(全量, CPU, ~20min)          profile_jsonl.py
   └─ 抓 D/G/H：翻译残留、未翻译、重复、schema
S1 嵌入打分(全量, ~100–200 GPU·h)       score_alignment.py
   └─ sig_zh(中文↔帧) / sig_en(英文↔帧) / xling(中↔英)，三分定位病灶环节：
      en高zh低xling低→翻译坏了；en低→源头就错；双低→重写；解码失败→工程桶
S2 VLM 审核(分层抽样 4–5 万, ~15–30 GPU·h)  make_audit_sample.py + vlm_audit.py
   └─ Qwen3-VL-8B 按 rubric 输出结构化 JSON：en/zh/翻译三个 0-5 分、问题标签、
      粒度、时序、建议动作(keep/fix_text/retranslate/recaption/drop)、
      2-3 条候选中文短 caption（审核顺带产修复素材）
   └─ 其中 ~2k 条再过 235B 裁判 + ~500 条人工抽检 → 验证 8B 判定可信度
S3 校准与外推(CPU)                      aggregate_report.py
   └─ 按 sig_zh 分位统计 VLM 差率曲线 → 选阈值（目标精度可调，默认 0.8）
   └─ 全量分桶 KEEP/FIX_TEXT/RETRANSLATE/RECAPTION/DROP + 问题率外推报告
```

抽样设计（S2）：sig_zh 十分位 × 每桶 3k（覆盖整条分数曲线，校准才有效）
+ S0 命中样本 1 万（过采样疑难）+ 低 xling 5 千 ≈ **4.5 万条**。

算力预算（A100 当量，B100 更快）：S1 全量 4 帧 SigLIP2 约 100–200 卡时
（瓶颈在视频解码，多开 CPU 线程/分片并行）；S2 4.5 万条 × 8 帧 × Qwen3-VL-8B
在 vLLM 下约 15–30 卡时；裁判档 2k 条忽略不计。**总计 ≈ 1–2 天 8 卡**，
即可拿到全量分桶和一份带外推问题率的报告。

（可选加强：预算充裕时用 Qwen3-VL-4B/8B 对全量 10M 做一遍二值门控审核，
约 1.4k 卡时 / 8 卡一周——建议先看 S3 报告，仅当嵌入分与 VLM 判定的
相关性不足以支撑外推时才升级。）

## 4. 判定标准（rubric 摘要，完整 prompt 在 `vlm_audit.py`）

- `en_score` / `zh_score`：caption↔画面符合度 0–5，≤2 判"差"。两个分**独立打**，
  用于区分"源头错"与"翻译搞坏"。
- `trans_score`：只比中英文本的忠实+流畅，≤2 判差；标签含 mistranslation /
  untranslated_fragment / artifact_text / omission / addition / awkward_chinese。
- `granularity`：coarse（"一个人在说话"级）/ adequate / dense。
- `temporal_order_relevant/correct`：多事件片段的顺序是否写对（对齐
  caprl-scheme 的时序硬负方向，wrong 样本天然是硬负素材）。
- `action`：keep / fix_text（仅规则清洗）/ retranslate（英文没错翻译错）/
  recaption（英文源头就错，需按画面重写）/ drop（视频不可用）。
- `better_zh`：2–3 条 ≤30 字、互补侧面的中文短 caption——**满足训练侧
  `max_txt_l=32` 约束**（`scripts/pretraining/clip/mobileviclip_small/config.py:18`），
  直接可进 `captions` 列表（loader 已支持 uniform 采样，
  `dataset/pt_dataset.py:176-189`）。

抗偏差护栏：判定用 temperature=0；en/zh 独立打分防"翻译顺就给高分"；
`<think>` 剥离后再解析 JSON；8B 判定与 235B 裁判在校准集上求一致率
（Cohen's κ），κ<0.6 的维度回炉改 rubric；人工 500 条抽检幻觉率。

## 5. v0.2 数据优化方案（按桶施策）

审核不是目的，分桶后每桶有明确动作（占比以 S3 报告实测为准）：

| 桶 | 动作 | 工具/方法 |
|---|---|---|
| KEEP | 直接保留，附带 `sig_zh` 分入库（后续 curriculum/过滤可用） | `apply_buckets.py` |
| FIX_TEXT | 规则清洗：去包裹引号/前缀/换行/HTML 转义，**零模型成本** | `apply_buckets.py` 已内置（`clean_zh`） |
| RETRANSLATE | 英文没错、翻译坏了 → 用更强配置重翻：Qwen3 大杯 + 关 thinking 输出、严格输出格式、按 S0 学到的失败模式改 prompt；重翻后必须复算 `xling_cos` 回归验证 | `apply_buckets.py` 导出任务队列 → 现有翻译管线 v0_2 |
| RECAPTION | **英文源头就错 → 不再走"英文→翻译"级联，直接用 Qwen3-VL 看视频写中文**：每片 K=2–3 条 ≤30 字短 caption，一步同时解决 A/B/C/E 四类问题 | `apply_buckets.py` 导出任务队列；S2 的 `better_zh` 已是同款产物；放量即 caprl-scheme §4.2 路径 B0 的中文版 |
| DROP | 剔除（不可解码/拒答/无法配文），并记录原因供补抓 | `apply_buckets.py` 导出 `dropped.jsonl` |

三个结构性升级（随 v0.2 一起做，成本低收益高）：

1. **多字幕 schema**：v0.2 输出
   `{"video", "caption": <清洗后译文>, "captions": [<译文>, <recaption 短句>...]}`
   ——向后兼容现有 loader，uniform 采样直接吃多视图；原译文保留为视图之一
   （防训练/测试长度分布漂移，同 caprl-scheme §8 护栏）。
2. **假负例治理**：S0 的重复 caption 统计直接给出"同 caption 多视频"簇，
   要么改写去同质化（recaption 时加"避免笼统句"约束），要么训练侧同簇不互为负例。
3. **时序硬负素材**：S2 标出的 `temporal_order_relevant=true` 子集是
   caprl-scheme Pillar C（事件顺序反事实硬负）的天然种子集，留好标记。

**建议节奏**：v0.2 先只做 FIX_TEXT + RETRANSLATE + DROP（纯文本侧，几天内可出）
→ 小预算重训验证不掉点、翻译类指标改善；RECAPTION 桶（要过视频，成本最高）
作为 v0.3 分批放量，优先 `sig_en` 最低的片段（劣质源头收益最大），预算参照
caprl-scheme §7：3M 片 × K=3 约 1.5k–3k A100·时。

## 6. 验收

- **数据面**：v0.2 复跑 S0/S1，要求 D 类残留率 < 0.1%、`xling_cos` 中位数上移、
  RECAPTION 桶复审 `zh_score` 均值显著提升；重复 caption 率下降。
- **训练面**：等 token 预算小规模重训（对齐 MobileCLIP2 的 12.8M 消融惯例），
  中文检索评测（VATEX-zh / 翻译版 MSRVTT-1k-zh）R@1 对比 v0.1；英文基准
  （MSR-VTT-1k、K400 零样本）不回退作为护栏。
- **人工面**：v0.1 vs v0.2 各抽 300 条盲评（对齐/流畅双维度），胜率 > 65%。

## 7. 操作手册（在数据机上按序执行）

以下命令假设在本仓库 clone 的根目录执行；每步末尾给出**验收点**和
**决策门**（不达标就停下修，不要带病往下跑）。
**审查阶段只需 7.0–7.5（产出诊断报告即止）；7.6 起是数据优化阶段，
待报告评审后再启动。**

> **8×V100 一键通道**：全部审查步骤已封装为
> `bash tools/data_audit/run_v100.sh {smoke|s0|s1|status|s2-sample|s2|s2-judge|s3}`，
> 只需先 `export CLIPS=视频根目录`。V100（Volta）与下文 A100 命令的三个差异
> 已在脚本内处理：
> ① 无 bf16 → S1/S2 自动落 fp16（`--dtype` 可强制）；
> ② vLLM V1 引擎不支持 Volta、而 Qwen3-VL 需要新版 vLLM → S2 不起服务，
>   改用 transformers 后端 `vlm_audit_hf.py` 做 8 卡数据并行（同一 rubric/
>   输出格式，S3 无感）；
> ③ 235B 裁判 8×32GB 放不下（权重就 >400GB）→ `s2-judge` 走 OpenAI 兼容
>   云端 API（DashScope/OpenRouter 的 qwen3-vl-235b，2k 条约几十元），
>   或降级为本地换一个模型家族（InternVL3.5-8B）交叉验证。
> 显存注意：32GB 卡用 Qwen3-VL-8B（fp16 约 17GB）；**16GB 卡请
> `export S2_MODEL=Qwen/Qwen3-VL-4B-Instruct`**。
> 时间预估（8×V100，解码为主瓶颈）：S1 全量 ~1–2 天；S2 4.5 万条 ~半天
> （每卡 ~5.6k 条 × 5–8 s）；嫌慢可 `export PER_BUCKET=2000` 收缩抽样。

### 7.0 环境准备（一次性，~15 分钟）

```bash
# 首选（网络通畅、独立环境）——注意：网络受限/国内集群请改用
# tools/data_audit/README.md「网络受限机器」一节的 venv 复用方案，
# 不要硬拉 conda + torch；V100 机器不需要装 vllm。
conda create -n audit python=3.10 -y && conda activate audit
pip install "torch>=2.4" "transformers>=4.57" sentence-transformers \
            decord pillow numpy requests
pip install "vllm>=0.11"        # 仅 Ampere+ 走 vLLM 服务方式才需要

export DATA=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
export CLIPS=/path/to/internvid/clips   # TODO: 改成视频根目录（video 字段相对它拼路径）
export OUT=outputs/audit_v0_1 && mkdir -p $OUT
```

### 7.1 冒烟（~10 分钟，必做）

```bash
# S0 前 2 万条：确认字段探测（看 stderr 的 "[keys] video=... zh=... en=..."）
python tools/data_audit/profile_jsonl.py --input $DATA --limit 20000 \
  --out-summary $OUT/smoke_s0.json --out-flags $OUT/smoke_s0_flags.jsonl

# S1 前 2 千条：确认视频路径拼接与解码
python tools/data_audit/score_alignment.py --input $DATA --video-root $CLIPS \
  --output "$OUT/smoke_s1.{shard}.jsonl" --limit 2000 --frames 4 --batch-clips 16

# 解码失败率检查（应 <2%；偏高先查 $CLIPS 路径或视频完整性，再放全量）
python -c "
import json;rows=[json.loads(l) for l in open('$OUT/smoke_s1.0.jsonl')]
err=sum('error' in r for r in rows);print(f'decode error: {err}/{len(rows)}')"
```
**验收点**：`[keys]` 探测的三个字段正确（不对则后续所有命令加
`--video-key/--zh-key/--en-key`）；解码失败率 <2%。

### 7.2 S0 规则画像（全量，CPU ~20 分钟）

```bash
python tools/data_audit/profile_jsonl.py --input $DATA \
  --out-summary $OUT/s0_summary.json --out-flags $OUT/s0_flags.jsonl
python -m json.tool $OUT/s0_summary.json | head -60   # 看 flag_rates 和重复率
```
**决策门**：拒答+未翻译+指令泄漏合计 >5% → 翻译管线本身要回炉，先修管线
再谈数据修复；重复 caption 率 >10% → §5 假负例治理升为必做项。

### 7.3 S1 嵌入打分（全量，8 卡 ~1 天，可断点续跑）

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i nohup python tools/data_audit/score_alignment.py \
    --input $DATA --video-root $CLIPS --output "$OUT/s1_scores.{shard}.jsonl" \
    --num-shards 8 --shard-id $i --frames 4 --batch-clips 16 \
    > $OUT/s1_shard$i.log 2>&1 &
done
watch -n 60 'wc -l outputs/audit_v0_1/s1_scores.*.jsonl'   # 总数应逼近 10M/8×8
```
中断后原样重跑同一命令即可续（按输出行数跳过已完成）。
**验收点**：8 个分片行数之和 = 输入行数；`error` 记录占比与冒烟一致。

### 7.4 S2 分层抽样 + VLM 审核（~4.5 万条，8B 单卡数小时）

```bash
python tools/data_audit/make_audit_sample.py --input $DATA \
  --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --per-bucket 3000 --flagged 10000 --low-xling 5000 \
  --output $OUT/s2_sample.jsonl

# 主力档：Qwen3-VL-8B（单卡起服务；并发按显存调）
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 \
  --limit-mm-per-prompt image=8 --max-model-len 16384 &
python tools/data_audit/vlm_audit.py --input $OUT/s2_sample.jsonl \
  --video-root $CLIPS --api-base http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct --frames 8 --concurrency 32 \
  --output $OUT/s2_verdicts.jsonl

# 裁判档：同一批样本头 2000 条再过 235B（4 卡 tensor parallel），校验 8B 可信度
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct --port 8001 -tp 4 \
  --limit-mm-per-prompt image=8 &
python tools/data_audit/vlm_audit.py --input $OUT/s2_sample.jsonl \
  --video-root $CLIPS --api-base http://localhost:8001/v1 \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct --frames 8 --concurrency 16 \
  --limit 2000 --output $OUT/s2_verdicts_judge.jsonl

# 人工抽检 500 条（盲评表：视频名 + 中文 caption + 8B 判定）
shuf -n 500 $OUT/s2_verdicts.jsonl > $OUT/human_check_500.jsonl
```
（**V100 环境**：上面两段 vLLM 服务命令分别替换为
`bash tools/data_audit/run_v100.sh s2`（8 卡 transformers 后端）和
`export JUDGE_API_KEY=...; bash tools/data_audit/run_v100.sh s2-judge`
（云端裁判 + 自动输出一致率）；产物文件名为分片形式
`s2_verdicts.*.jsonl`，S3 命令传 glob 即可。）
**决策门**：8B 与 235B 在 `action` 上的一致率 <75%（或 zh_score ≤2 判定的
κ<0.6）→ 改 rubric/换主力模型重跑 S2，不要用不可信判定去校准阈值。

### 7.5 S3 校准 + 全量分桶 + 报告（CPU 分钟级）

```bash
python tools/data_audit/aggregate_report.py \
  --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --verdicts $OUT/s2_verdicts.jsonl --target-precision 0.8 \
  --out-buckets $OUT/buckets.jsonl --out-report $OUT/report.md
```
**验收点**：`report.md` 的 sig_zh 分位表差率应随分位单调下降、最低分位
zh 差率 ≥80%；不单调说明嵌入分外推不可靠 → 提高 S2 抽样量或对可疑区间
全量过一遍 4B/8B VLM 门控再分桶。`report.md` 即周报素材。

### 7.6 S4 落地 v0.2（文本侧，CPU 小时级）

```bash
python tools/data_audit/apply_buckets.py --input $DATA \
  --buckets $OUT/buckets.jsonl --verdicts $OUT/s2_verdicts.jsonl \
  --out-data $OUT/v0_2/internvid_zh_v0_2.jsonl \
  --out-retranslate $OUT/v0_2/retranslate_tasks.jsonl \
  --out-recaption $OUT/v0_2/recaption_tasks.jsonl \
  --out-drop $OUT/v0_2/dropped.jsonl
```
然后：① `retranslate_tasks.jsonl` 喂给翻译管线 v0_2（prompt 按 S0 学到的
失败模式加固：禁止前缀/引号/解释、关 thinking 输出、限长），重翻结果并回
`internvid_zh_v0_2.jsonl`；② 对 v0.2 **复跑 7.2 和 7.3 做回归**——
**验收门**：D 类残留率 <0.1%、`xling_cos` 中位数较 v0.1 上移、重复率下降。

### 7.7 小预算重训验证

v0.2 接入 `configs/data.py`（新增 `internvid_zh_v0_2` 语料项，多字幕已兼容
`caption_augmentation.caption_sample_type='uniform'`），等 token 预算重训
（12.8M 样本消融惯例），中文检索（VATEX-zh / MSRVTT-1k-zh）R@1 对比 v0.1，
英文基准不回退（§6 验收标准）。

### 7.8 v0.3 RECAPTION 放量（衔接 caprl-scheme P0）

`recaption_tasks.jsonl` 按 `sig_en` 从低到高排序分批：Qwen3-VL 直接看视频写
K=2–3 条 ≤30 字中文短 caption（prompt 复用 `vlm_audit.py` 的 `better_zh`
定义），产出并回 `captions` 列表；预算参照 caprl-scheme §7
（3M 片 × K=3 ≈ 1.5k–3k A100·时，优先最差的 1M 片）。
