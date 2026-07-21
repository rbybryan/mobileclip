# 数据问题登记表与流水线状态（InternVid-zh v0.1 / 10k 抽样）

> **这是审查过程的活账本**：跑到哪了、发现了什么问题、每个问题什么状态、
> 哪些决策还悬着。每完成一个阶段就回来更新，**不要只写在聊天里**（CLAUDE.md §0）。
>
> 分工：
> - **本文件** = 状态 + 问题登记 + 待决策（一眼看全局）
> - [`notes/audit-10k-findings.md`](../../notes/audit-10k-findings.md) = 各阶段的统计细节与推理过程
> - `outputs/audit_10k/report.md` = S3 自动生成的正式诊断报告（**不入库**，gitignore）
>
> 问题类别 A–H 沿用
> [`notes/internvid-zh-data-audit-plan.md`](../../notes/internvid-zh-data-audit-plan.md) §1 的分类学。

---

## 1. 流水线状态

机器：`101vector` 8×V100-32G ｜ 数据：`/home/vca1/datasets/internvid_zh_10k`（10000 条）

| 阶段 | 状态 | 产物 | 一句话结论 |
|---|---|---|---|
| S0 规则画像 | ✅ 完成 | `s0_flags.jsonl` | 语言污染 327 条（3.3%）；其余旗标分布**待补录**（见下） |
| S1 嵌入打分 | ✅ 完成 | `s1_scores.*.jsonl` | error 0%，SigLIP 信号健康；**`xling_cos` 是弱工具** |
| S1 联合分析 | ✅ 完成 | `s1_analysis.md` | 翻译损伤真实但小、只在极低尾；S3 阈值有雷 |
| **S2 VLM 审核** | 🔄 **运行中** | `s2_verdicts.*.jsonl` | 10k 全覆盖，8 卡 HF 后端，预计 ~2–3 h |
| S2 裁判（云端 235B） | ⬜ 未开始 | `s2_verdicts_judge.jsonl` | 需 `JUDGE_API_KEY`；**可与主力并行，不占本机 GPU** |
| S2 人工抽检 300 | ⬜ 未开始 | `human_check_300.jsonl` | 在 §6 Gradio 里过 |
| **xling 阈值裁决** | ⬜ 未开始（工具已就位） | `xling_calibration.md` | `run_v100.sh xling-check`，**S2 一完成就跑，裁决 §4 决策 1** |
| S3 校准 + 分桶 | ⬜ 未开始 | `report.md` / `buckets.jsonl` | **开跑前先解决 §4 的两个待决策** |

刷新状态：

```bash
bash tools/data_audit/run_v100.sh status        # S1/S2 行数 + 存活进程 + 日志尾行
```

> S2 刚启动时行数为 0 是正常的（各分片要先加载权重）；
> **进程数 0 且未到 100% = 分片已退出**，查对应 `s2_shard*.log` 的 Traceback，
> 修好后重跑 `s2` 即可断点续跑。

---

## 2. 已确认的数据问题

状态含义：`已确认` = 有量化证据 ｜ `待裁决` = 等 S2/S3 ｜ `已处置` = v0.2 已修

| # | 类别 | 问题 | 量化证据（10k） | 发现于 | 状态 |
|---|---|---|---|---|---|
| 1 | D | **语言污染**：部分未翻 / 中英混杂（**非**逐字照抄） | 327 条（3.3%）命中 `zh_no_chinese`+`zh_equals_en`+`zh_mostly_english` | S0 | 已确认 |
| 2 | D | 逐字照抄英文原文（xling≈1.0 的极端形态） | 仅 12 条（0.12%），量级可忽略 | S1 联合分析 | 已确认 |
| 3 | C | 翻译损伤（译文比原文更偏离画面） | 超额落差仅 **+0.0106**（基线 +0.0171），且**只集中在 xling 极低尾** | S1 联合分析 | 已确认（量级小） |
| 4 | A/B | 源 caption 与画面不对齐 | RECAPTION 候选 1195 条（11.9%，en 低 + zh 低） | S1 分位预览 | 待裁决（S2） |
| 5 | — | **`xling_cos` 判别力不足**（工具问题，非数据问题，但会污染分桶） | ρ(sig_zh,xling)=0.039、ρ(sig_en,xling)=-0.018、p25–p75 仅 0.711–0.813 | S1 联合分析 | 已确认 → 见 §4 决策 1 |
| 6 | H | 视频缺失 / 不可解码 | 本机 **0/10000**——但**不可外推**：10k 是预抽样，大概率已按「可读」筛过 | S1 | 已确认（本机） |

**关键交叉结论**（问题 1 + 2 + 5）：语言污染的主流形态是**部分未翻**，
LaBSE 对这类只给中间分——S0 判为污染的 327 条里，**315 条（96%）xling 检不出来**。
→ **语言污染一律以 S0 旗标为判据，不要用 `xling_cos`。**

### 待补录

S0 的完整旗标分布还没记进来（只提取了语言污染这一类）。补一条命令即可：

```bash
python -m json.tool outputs/audit_10k/s0_summary.json | head -60
```

重点看两个决策门（SETUP_V100.md §2）：**拒答 + 未翻译 + 指令泄漏合计 >5%**
→ 翻译管线要回炉；**重复 caption 率 >10%** → 假负例治理升为必做项。
把数字填进上表并在此说明是否触发。

---

## 3. 待 S2 裁决

| # | 问题 | 规模 | S2 要回答什么 |
|---|---|---|---|
| T1 | 「**翻译可疑**」格：en 高 + zh 高 + xling 低 | **1522 条（15.2%）** | 中英两侧画面对齐都正常，只有 LaBSE 说不忠实。按问题 5，**这格很可能主要是 xling 噪声**。看 VLM `trans_score≤2` 的真实差率——若很低，说明该格是假警报 |
| T2 | RECAPTION 候选（en 低 + zh 低） | 1195 条（11.9%） | `en_score≤2` 的比例，确认是源头错而非评分尺度问题 |
| T3 | RETRANSLATE 候选（en 高 + zh 低 + xling 低） | 387 条（3.9%） | `trans_score` 是否显著低于 T1 那格——这是 xling 有无判别力的直接对照 |
| T4 | 8B 判定本身可不可信 | 500 条 | 8B vs 云端 235B 裁判的 `action` 一致率 **≥75%**（§4 决策门）。不达标先改 rubric 重跑 S2，**不要拿不可信判定去校准阈值** |

---

## 4. 待决策（阻塞 S3）

### 决策 1：`--xling-threshold` 要不要保留 ⚠️ **跑 s3 前必须处理**

`aggregate_report.py` 的分桶规则里，**低 xling 单独就能触发 RETRANSLATE**：

```python
elif (fl & RETRANSLATE_FLAGS) or (rec["xling_cos"] < args.xling_threshold):
    bucket = "RETRANSLATE"
```

默认阈值 **0.7**，而本批 xling 的 **p25 = 0.711** —— 阈值正落在分布最稠密处，
会把约 **23% 的语料**仅凭一个 ρ≈0 的弱信号扫进重翻队列。这是「用未校准的
绝对阈值去切未校准的分数」。

**工具已就位**——S2 一跑完就执行，它会直接给出建议：

```bash
bash tools/data_audit/run_v100.sh xling-check     # -> outputs/audit_10k/xling_calibration.md
```

`calibrate_xling.py` 产出的正是 `report.md` **不产出**的那份证据（后者的十分位
是按 `sig_zh` 切的，不是按 `xling`）：`trans_score≤2` 差率 × xling 十分位、
当前 0.7 阈值的精度/召回、阈值扫描，以及**改用 S0 旗标做检测器**的对照。

三种可能的结论，脚本会自己判并给出改法：

- 曲线**平坦** → 让 xling 退出分桶规则（`--xling-threshold 0`）或压到极低尾；
- 曲线**单调下降且能达标** → 直接给出校准后的阈值；
- 曲线**有判别力但达不到目标精度** → 注意这可能是**算术上不可达**而非 xling
  的错（精度 ≤ 基准率 × lift；基准率若只有 ~9%，要 80% 精度需要整个被选集合上
  维持 ~9× lift）。此时三选一：降低 `--target-precision`、改成「xling 低 **且**
  有 S0 旗标」的**合取**条件（把 `aggregate_report.py:179` 的 `or` 改成 `and`）、
  或只保留极低尾。

> 现在**不要**改死默认值——没有 S2 判定就改，只是把一个未经验证的猜测
> 换成另一个。

### 决策 2：全量（H100）跑之前要不要改 rubric

10k 的价值就是**在小规模上把 rubric 和阈值调可信**再上全量（SETUP_V100.md §4）。
若 T4 的一致率 <75%，或某个维度 κ<0.6，先改 `vlm_audit.py` 的 rubric 再重跑，
**不要带病上 10M**。

---

## 5. 更新约定

每完成一个阶段：① 更新 §1 状态表；② 新问题登记进 §2（带量化证据和发现阶段）；
③ 已裁决的从 §3/§4 移走并写明结论；④ commit 推送，服务器 `git pull` 即可看到。
统计细节和推理过程写进 [`notes/audit-10k-findings.md`](../../notes/audit-10k-findings.md)，
本文件只保留结论和状态。
