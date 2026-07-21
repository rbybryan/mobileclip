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
| S2 裁判 | 🟡 改在中转机跑 | `s2_verdicts_judge.jsonl` | 走内网 `Qwen3.6-27B`（**✅ 已确认多模态**）。**101vector 和 H100 都连不上端点**，改到 7 卡 V100 `10.55.87.68` 上跑（该机可访问 b100 存储）——三项前提待测，见 §4 决策 2b。主力/裁判分工见 [`SETUP_V100.md`](SETUP_V100.md) §4.2–4.3 |
| S2 人工抽检 300 | ⬜ 未开始 | `human_check_300.jsonl` | 在 §6 Gradio 里过 |
| **xling 阈值裁决** | ⬜ 未开始（工具已就位） | `xling_calibration.md` | `run_v100.sh xling-check`，**S2 一完成就跑，裁决 §4 决策 1** |
| S3 校准 + 分桶 | ⬜ 未开始 | `report.md` / `buckets.jsonl` | **开跑前先解决 §4 的两个待决策** |

刷新状态：

```bash
bash tools/data_audit/run_v100.sh status        # S1/S2 行数 + 存活进程 + 日志尾行
```

### 下一步做什么（按顺序，2026-07-21）

散在几个手册里的步骤在这里排个先后——**站在终端前照这个顺序做**：

| # | 动作 | 为什么是这个顺序 |
|---|---|---|
| 1 | `run_v100.sh status` 看 S2 主力到没到 10000 | 后面每一步都吃它的产物 |
| 2 | **裁判走 H100 中转（路线 A′）**：先做那 500 条视频在 H100 上的解析预检 | 探针二已实测不通（§4 决策 2b），H100 中转是当前首选；**零下载、零占卡** |
| 3 | S2 一到 100% 立刻 `run_v100.sh xling-check` | 裁决 §4 决策 1，**阻塞 S3**；不依赖裁判，可与第 2 步并行 |
| 4 | Gradio 人工抽检 300（§6） | 不占 GPU、不依赖裁判，随时可插空做；裁判若最终跑不成，它就是唯一的独立效度锚点 |
| 5 | 两个决策都裁完 → `run_v100.sh s3` | 需要 S0+S1+S2 全部产物 |

> ⚠️ **本地 32B 下载（路线 C）是兜底，不是当前主线——A′ 没排除之前别耗在它上面。**
> 那是 64GB / 数小时的投入，而第 2 步的预检只要几分钟。只有当 H100 中转也
> 走不通时才回来下它；真要下，**hf-mirror 通道已实测不可用**（新版
> `huggingface_hub` 弃用 `hf_transfer` 改走 Xet，镜像给不出元数据 →
> `Local entry not found`），改用 ModelScope：
>
> ```bash
> pkill -f "hf download"; unset HF_HUB_ENABLE_HF_TRANSFER HF_HUB_DISABLE_XET
> nohup bash -c 'until modelscope download --model Qwen/Qwen3-VL-32B-Instruct \
>   --local_dir ckpts/Qwen3-VL-32B-Instruct --max-workers 4; do sleep 10; done' \
>   > download_32b_ms.log 2>&1 &
> ```
>
> 完整流程见 [`SETUP_V100.md`](SETUP_V100.md) §4.3-本地，诊断详情见
> `TROUBLESHOOTING.md`。

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

> ⚠️ **「重复 caption 率」这个门,只能用全量判,抽样判不了。** 10k 是 10M 的
> 0.1% 抽样,一组 k=2 的重复要两条都被抽中才可见(概率 1e-6);全量哪怕 5% 的
> clip 处在重复对里,10k 里的期望命中也只有约 0.5 组。`make_pilot_anno.py`
> 在 10k 上报的 0.0% 因此**对全量毫无约束力**,`outputs/audit_10k/s0_summary.json`
> 里的 `duplicate_zh_caption_rate` 同理。推导与副作用见
> [`notes/audit-10k-findings.md` 结论 5](../../notes/audit-10k-findings.md)。
> 另注意口径:该字段数的是 **zh** 侧重复,而试点训练喂的是 **en** 侧字幕。

---

## 3. 待 S2 裁决

| # | 问题 | 规模 | S2 要回答什么 |
|---|---|---|---|
| T1 | 「**翻译可疑**」格：en 高 + zh 高 + xling 低 | **1522 条（15.2%）** | 中英两侧画面对齐都正常，只有 LaBSE 说不忠实。按问题 5，**这格很可能主要是 xling 噪声**。看 VLM `trans_score≤2` 的真实差率——若很低，说明该格是假警报 |
| T2 | RECAPTION 候选（en 低 + zh 低） | 1195 条（11.9%） | `en_score≤2` 的比例，确认是源头错而非评分尺度问题 |
| T3 | RETRANSLATE 候选（en 高 + zh 低 + xling 低） | 387 条（3.9%） | `trans_score` 是否显著低于 T1 那格——这是 xling 有无判别力的直接对照 |
| T4 | 8B 判定本身可不可信 | 500 条 | 跑 `bash tools/data_audit/run_v100.sh judge-agree`（裁判在别处跑完、把 `s2_verdicts_judge.jsonl` 拷回来也能独立重算）。三个门：五类 `action` 一致率 **≥75%**、二值 `{keep,fix_text}` vs 其余 **≥80%**、**各维度 Cohen's κ ≥0.6**（plan §4 要求，此前无实现）。<br>**看 κ 不要只看一致率**——类别不平衡下双方一起说 `keep` 就能堆出高一致率。**`trans_bad` 那一行尤其关键**：决策 1 的全部差率都建立在它上面。<br>不达标先改 rubric 重跑 S2，**不要拿不可信判定去校准阈值**；裁判完全用不上时走 `SETUP_JUDGE_API.md` §0 路线 D（换族 8B 找分歧 → 分歧定向人工抽检 → 决策门改挂人工） |

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

### 决策 2b：裁判用谁（无云端 API key）

暂时没有 DashScope/OpenRouter key。内网候选：B100 集群上 3 个已起好的
vLLM 端点，模型均为 `Qwen3.6-27B`（`172.30.85.198:8000`、`172.30.62.177:8000`
为 8 卡，`172.30.86.57:5000` 为 4 卡，`max_model_len` 262144）。

这是 [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0 的**路线 A′**（复用已起好的
端点，零部署、不占本机卡，可用时优先于路线 C 的本地 32B）。

- ✅ **探针一（多模态）已通过**——`Qwen3.6-27B` 确认支持视觉输入，
  `vlm_audit.py` 的 `image_url` 链路可用。既有调用参考实现见 §0 路线 A′
  引用的 `request_multilabel_v3_resume.py`（接入前按该节表格核对并发/超时/
  图像尺寸等参数）。
- ❌ **101vector 连不上端点**（2026-07 实测）——`172.30.x.x` 与 `10.235.31.101` 不同网段。
- ❌ **H100 也连不上端点**（2026-07 实测）。
- 🔄 **当前方案：在 7 卡 V100 `root@10.55.87.68 -p 31179` 上跑裁判**。
  已确认该机**能访问 `/workspace/mount/b100_zaip_data` 存储**，而端点的模型权重
  与全量语料都在该挂载下，所以它多半在 B100 网内。附带好处：10k 记录保留
  `abs_path`（`resolve_video_path` 第三候选），**很可能不用拷视频，
  只需搬一个 500 行的 `s2_judge_sample.jsonl`**。环境也轻——`vlm_audit.py`
  只要 `requests/numpy/pillow/decord`，**不需要 torch、不占 GPU**。
  **逐条命令见 [`SETUP_JUDGE_RELAY.md`](SETUP_JUDGE_RELAY.md)**（选型理由见
  [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0 路线 A′）。

  **三个仍待实测的前提**（按顺序）：① 该机到 `172.30.x.x` 的连通性；
  ② 101vector → 该机的文件通道；③ 那 500 条视频在该机上能否解析。

> ⛔ **别在中转机上重新抽样**：裁判必须评主力判过的**同一批记录**，
> 从全量 10M 另抽一批与本机 10k 交集可能为空，`judge-agree` 会报「重叠不足」。

这台也不行时：退回路线 C（本地 32B）/ 路线 D（无大裁判协议）。
**另一条不冲突的路**：H100 虽连不上端点，但可在本机自托管 235B-FP8（路线 A）——
证据强度远高于 27B 且无同族偏差，代价 ~235GB 下载。可以先用 27B 拿初步一致率、
再用 235B 复核。

> ⚠️ **即便可用，也要记明证据强度打折**：27B 裁 8B 只有约 3 倍参数差距，
> 且同厂同族——方案 §2 原本要的是 235B + 换一家（InternVL）互为第二意见，
> 就是为了避开共同盲区。同族模型的一致率会被抬高，**一致不等于正确**。
> 若决策门靠它通过，须提高人工抽检（300 条）的权重作为唯一独立信号。

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
