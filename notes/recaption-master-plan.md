# 重字幕主计划 v1（Recaption Master Plan）— 两份方案的裁决与唯一执行线

> **日期**：2026-07-24。**角色**：本文是重字幕工作的**唯一执行层**——
> [`caption-fusion-scheme.md`](caption-fusion-scheme.md)（五样本重述+融合）与
> [`data-quality-caprl-scheme.md`](data-quality-caprl-scheme.md)（CapRL 数据强化）两份设计
> 各自的分阶段清单（其 §6.2 / §7）**均被本文取代**；两份原文保留为设计依据与方法论论证，
> 不再照其执行。审计事实底座见 [`audit-10k-findings.md`](audit-10k-findings.md)。
>
> **一句话**：**S0 五样本重述照跑（已实现）；训练验证前置、人评标定与融合后置**——先用
> "不融合的多视图富集"过训练门 G1（它不需要一分钱人力标注），融合（CFU）作为门控增强
> 只有在 G1 通过、且免费诊断显示投票信号有信息之后才花人力与算力。

---

## 0. 对两份方案的裁决（评估结论）

### 0.1 两份方案共同的、经审计确认仍成立的判断（保留）

1. **重字幕是主力手段**（审计结论 10/12）：粒度粗 93.6%、整体质量差 19.6%、翻译差仅 3.6%
   → 「直接看视频写中文」一步同时解决对齐/粒度/翻译三类问题。RECAPTION 桶 25.3%。
2. **32-token 硬约束**（caprl §4.4）：任何稠密产物必须降维成 ≤32-token 短视图才进训练；
   时间戳不进对比文本，时序只用「先/然后/最后」措辞。
3. **S0 前端三决定**（fusion §1.1，已实现于 `--mode recap`）：五次**独立**调用（非一次吐五条）、
   temp 0.8–1.0 制造真方差、每样本帧相位抖动。这三条的论证扎实，不动。
4. **信任门精确优先**（fusion §0.3）：冻结文本塔下假单元=烧录的假对齐，失败代价不对称。
5. **一致性抓不住系统性幻觉**（fusion §3.5）：验真器必须异质（GroundingDINO/OCR/二号 VLM），
   同族自查无效——这条自我批判是两份文档里最有价值的方法论资产，全文继承。
6. **可证伪消融纪律**（fusion §6.4 / caprl P0）：融合必须在等预算下同时 ≥ uniform 与 ≥ best-single
   才算成立，不预设结论。

### 0.2 对 `data-quality-caprl-scheme.md` 的修正

| # | 问题 | 裁决 |
|---|---|---|
| C1 | **两条训练线混写**：方案按英文线（MobileCLIP-S2 / MSR-VTT / CapRL-Video-4B 英文字幕器）立论，但重字幕的当下主战场是中文线（NebulaCV-24M / MiniRBT 冻结文本塔 / InternVid-zh） | 两线**分开钉**（§2 D1）：zh=Qwen3-VL-8B 直出中文（已实现、已论证）；EN=CapRL-Video-4B（paper 线 C1，归 `paper-plan.md` U7 所有） |
| C2 | **P0 一步到 3M×K≈9M 次生成**，缺中间决策门 | 改为 300k 子集 → G1 训练门 → 全量（§3 Stage 3/5）；全量前必须有子集级证据 |
| C3 | Pillar C（tags/硬负）、Pillar D（教师嵌入/KD）的所有权已被 `paper-plan.md` §7 / U7 / U9 收编 | 本计划**不重复规划**，只保留接口（§6 砍掉清单、§2 D9） |
| C4 | §8 「原始字幕保留为采样视图之一」被当成普适护栏 | 只对 **KEEP 桶**成立；**RECAPTION 桶成员的原字幕是实测差的字幕，不得回流为训练视图**（§4） |

### 0.3 对 `caption-fusion-scheme.md` 的修正

| # | 问题 | 裁决 |
|---|---|---|
| F1 | **顺序倒置**：150–300 片逐单元人评标定（人·日级，全计划最贵的人力项）排在任何训练证据之前（其 F0 在训练验证前就要标定 k\*） | **训练门 G1 先行**（零人力），标定后置到 G1 通过 + 免费诊断确认投票信号有信息之后（§3 Stage 4） |
| F2 | §6.4 消融的 **arm A 未定义 32-token 处理**：五条稠密重述「直接 uniform」会被截成半句，消融混淆「融合价值」与「长度处理」 | arm A 定稿为**句级切分短视图**（确定性、零额外生成，§4.2）；三臂全部服从同一 ≤32-token 规则 |
| F3 | S4 用 GroundingDINO 验物体存在性，但它是**英文词表**检测器，而单元是中文——方案未给桥 | S1 分解时每单元**顺带产 `en_gloss`**（同一次调用多一个字段，近零成本），F1 验真用之（§2 D8） |
| F4 | §4 schema 示例里 RECAPTION 片仍保留原字幕作视图 | 同 C4，删（§4） |
| F5 | §3.4 要求 onscreen_text「必须过 OCR 才进融合稿」，但 OCR 验真器在 F1 才建——**F0 期该门无着落** | F0 期守恒策略（§2 D7）：`onscreen_text` 一律不进融合稿、`count` 一律降级模糊量词；F1 用 OCR **找回**文字召回。消融解读时预登记：若 C 臂恰在含屏幕文字的片上输给 A，归因于此，是 F1 的启动理由而非融合的失败 |
| F6 | 全量成本按 V100-HF 后端不现实（10M 桶级 ≈ 数十天） | 算力分档（§5）：V100-HF 只跑 10k 级试点；300k 级起用 H100+vLLM；Stage 0 顺带实测吞吐，填进成本表再排全量 |

### 0.4 一条本计划新增的、两份原文都没有的保险

**G1 失败 ≠ 数据方向死刑。** fusion §7 已诚实指出：冻结 32-token 文本塔可能把「更好字幕」的
收益在编码端抹平。若 G1 富集臂不涨点，在放弃前**必须**先跑一次便宜的归因实验：同一 arm-A 数据
+ 文本塔 LoRA（或解冻末层）短训。数据好而塔瓶颈 → 病灶在塔，方向转向「文本侧微调」；
LoRA 也不涨 → 才是数据/任务本身无头寸。不做这步就下结论，会把「塔的问题」误判成「数据的问题」。
且这一风险可以**提前预警**：D11 探针 (iii)（视图可分性）若显示冻结塔对候选短视图已饱和
（片内视图两两余弦过高），则 G1 从一开始就带 LoRA 臂，而不是等失败后再补。

### 0.5 外部评审补强（2026-07-24，两份独立评审的载荷结论）

本计划定稿前经两路独立评审（VLM 方法论 + 统计设计），采纳如下（引用见 §9）：

1. **融合的先验要诚实地写低**。2024–26 没有任何发表证据表明「N 样本融合」在对比训练上优于
   「K 条候选逐步采样」：MobileCLIP 存 5 条 CoCa 字幕但增益集中在前 1–2 条（2311.17049）、
   MobileCLIP2 在 2 条合成字幕即饱和（2508.20691）、CLIPS 甚至证明**只采稠密字幕的子片段**
   好于用整条（2411.16828，丢信息反而涨点）。机制上，**逐步 uniform 采样在期望里就是软性
   support 加权**——一个 support=1 的幻觉单元只在极小步数占比里进梯度；融合是把这个软投票
   硬阈值化，同时牺牲被验证过的多样性杠杆。→ 预期 **C ≈ A**；融合保留为门控假设（§3 Stage 4），
   不预支任何工程。
2. **但「G1 零效 ⇒ 融合无用」不成立**（统计评审）：融合的机制是移除 A 臂里被烧录的幻觉——
   若幻觉损失恰好抵消富集收益，「C>A≈基线」正是融合预测的图景。→ **G1 的 NO-GO 只裁「重述
   通道」，不裁融合**；融合生死只由 G2 的直接对比裁（§3 G1/G2 判据即按此写）。
3. **更便宜的共识先上阶梯**：best-of-N 选择在视频规模有已发表实证（Panda-70M 以检索分选多
   教师最佳字幕，预训练 +7.0 MSR-VTT R@1，2402.19479）。「保 top-3 重述 + 弃底」以 ~5% 的
   工程成本捕获融合可及价值的大部分 → 设为 **A− 选择臂**（§3 Stage 4），C 必须同时赢 A 与 A−。
4. **冻结文本塔是已知弱配置**：LiT 原文自己就报「锁文本塔效果不佳」（2111.07991），而全部
   合成字幕收益都是在**可训文本塔**下测得的；CLIP 类文本塔有效长度 <20 token（Long-CLIP
   2403.15378）进一步压密度收益的天花板。→ D11 新增探针 (iii)，饱和即触发 0.4 的 LoRA 臂前置。
5. **字幕器与采样数**：H100+vLLM 上 Qwen3-VL-30B-A3B（激活 ~3B）单条成本 ≤ 8B dense 且质量
   更高 → Stage 3 预飞 bake-off 后定（V100 装不下 30B 总参，Stage 0 维持 8B 不变）。评审建议
   N 降到 2–3（多样性在 2 条即饱和）；**本计划维持 N=5**，分歧记录在案——理由：同一批产物要
   喂 A/A−/C 三臂 + 规模化投票诊断，N=3 只省约一天 H100 空档却废掉 G2 的干净测试与 5 档
   support 分辨率。若 G2 裁掉融合，Stage 5 全量生产即降为 N=2–3（届时只剩 A/A− 消费）。
6. **勘误（fusion §7）**：「缓存视觉塔前向、解码 5 次摊薄成本」在**帧相位抖动下不成立**——
   每样本帧不同，prefill 无法复用，5× 生成成本是真成本。成本表按真 5× 排（§5.1）。
7. **标定预算大幅收缩**（统计评审）：按**单元分层**（support 层 × 常见类别）标 ~600 个判定
   （≈1 人·日）即可支撑 k\* 四选一；整片全标（1,500–6,000 判定）是 5–10× 浪费。高风险类别
   （onscreen_text/count）**零决策标注**、预注册 always-verify——证明「可信任」所需的标注量
   （~60 个零错高支持单元 ≈ 标 300+ 片）远贵于直接验真。协议细节见 §5.2。

---

## 1. 事实底座（一屏读完）

| 项 | 事实 | 出处 |
|---|---|---|
| 两条训练线 | **zh 线**：NebulaCV-CLIP-24M（FastViT-MCI0 + MiniRBT-H288 中文文本塔，冻结、仅 MLP 投影可训，max_txt 32），InternVid-zh 11 分片，b100 与 wky/zyc 协作；**EN 线**：MobileViCLIP 复现+论文（CVPR'27，R0–R3，4×H100） | `b100-local-changes.md`、`paper-plan.md` |
| 审计定论 | 粒度粗 93.6% / 质量差 19.6% / 翻译差 3.6% / RECAPTION 桶 25.3% / RETRANSLATE 并入 RECAPTION（结论 13）| `audit-10k-findings.md` |
| 已实现 | S0 五样本重述全链：`RECAP_PROMPT` + `vlm_audit_hf.py --mode recap`（N=5、temp 0.9、帧抖动）+ `run_v100.sh recap`（8 卡分发、可续跑）；输入队列 `$OUT/v0_2/recaption_tasks.jsonl`（`apply_buckets --out-recaption` 产物） | #215/#216 |
| 未实现 | `fuse_captions.py`（S1–S7）、`fuse_verify.py`（S4/S6 异质验真）、句切视图工具、recap 诊断 | fusion §6.1 |
| 消费端 | `pt_dataset.py` uniform 模式**只读 `captions` 列表**；原字幕想当视图必须显式放进列表。⚠️ `min_caption_length` 按 `split()` 数词——**中文无空格恒为 1**，zh 配置必须 ≤1，否则静默全滤 | `pt_dataset.py:173-201` |
| 算力 | 101vector 8×V100（HF 后端、审计/试点）；b100 8×H100（论文线默认前 4 卡；vLLM 可用；**优先级=EN 论文线**，zh 大批量用空档并先打招呼） | `SETUP_B100.md`、`plan-review-upgrades.md` U6 |
| EN 线接口 | C1 需每片 ≥2 条合成短字幕（CapRL-Video-4B，10M ≈ 3–6 天 @4×H100，vLLM） | `plan-review-upgrades.md` §3.6、U7-3 |

---

## 2. 定论表（本计划做出的选择，每条一行理由）

| # | 决策 | 理由 |
|---|---|---|
| D1 | **前端分线钉死**：zh = Qwen3-VL-8B `RECAP_PROMPT`、N=5、temp 0.9、帧抖动（不动）；EN = CapRL-Video-4B（形态经 1k 试点在「prompt 约束 K 条短句」vs「稠密+句切」二选一，归 paper-plan 执行） | zh 直出避开翻译级联且已实现；EN 器为论文故事与吞吐最优 |
| D2 | **融合不上任何关键路径**：EN 论文线 C1 不引入融合；zh 线融合只在 G1+诊断双绿灯后启动 | deadline 红线；未经验证的机器不背关键路径 |
| D3 | **G1 = 300k 均匀子集「富集」试验**，不是「只修 25% 桶」试验 | 粒度粗 93.6% 是全语料病，富集测的才是主效应；均匀抽样免除桶置信度混淆；12.8M 样本/臂对齐仓库消融惯例 |
| D4 | **N=5 一次产齐**（300k×5，不为 G1 单独产 N=2）；字幕器经 Stage 3 预飞 bake-off（8B vs 30B-A3B @H100+vLLM）后定 | 同一批产物喂 G1（arm A）+ G2（arm A−/C）+ 规模化 support 诊断；重跑比一次产齐更贵；评审分歧见 0.5-5 |
| D5 | **视图阶梯定稿**：**A（主臂）**= 对每条重述各做一次 S7 式短视图派生（文本 LLM、K=2–3 条 ≤32 tok），全部并集；**A−（选择臂）**= A 的视图但只保片-帧 CLIP 分（NebulaCV-24M 冻结初始权重打分）top-3 重述；**A′（稳健性）**= 句级切分（`。；！？`切、超长按`，`再切、MiniRBT 实测 ≤32 tok、仍超长丢弃；零生成成本）；**B（best-single）**= 降为可选 | A 与 C 用同一派生提示 → 消融不混淆「共识」与「格式」（0.3-F2 + 评审 3 番）；A− 是 Panda-70M 式选择（0.5-3）；A′ 有 DreamLIP/CLIPS 先例，兼作 A 工具链拖延时的 G1 预登记降级 |
| D6 | **原字幕去留按桶**：KEEP=保留为唯一/其一视图；RECAPTION 桶=不回流 | 见 0.2-C4 |
| D7 | **F0 高风险类别守恒策略**：`onscreen_text` 不进融合稿；`count` 一律模糊量词 | 见 0.3-F5；无验真器期的精确优先 |
| D8 | **S1 单元带 `en_gloss`**，F1 的 GroundingDINO 用它查询 | 见 0.3-F3 |
| D9 | RETRANSLATE 并入 RECAPTION（全量不再维护独立通道）；FIX_TEXT/DROP 照旧文本侧；tags/硬负/教师嵌入归 paper-plan U7/U9，本计划只保证 schema 兼容 | 审计结论 13；所有权单一 |
| D10 | **zh 评测集钉死**：门控指标 = 2k held-out **KEEP 桶**片内检索（字幕经审计、干净）；报告指标 = MSR-VTT-1k 查询一次性机翻中文 + 人抽 50 条校对 | 用 v0.1 噪声字幕当验证集会系统性低估富集收益；KEEP 桶是现成的干净集 |
| D11 | **三个 10 分钟级探针（b100 跑，权重在那）**：(i) MiniRBT 对重述样本的实测 token/字比（钉视图字预算）；(ii) 冻结文本塔顺序敏感探针（1k 对顺序互换句余弦）——zh(MiniRBT) 与 EN(MobileCLIP-S2) 各一次；(iii) **视图可分性探针**：冻结 MiniRBT 编码 ~1k 片的候选短视图，报片内视图两两余弦与「视图↔原字幕」余弦分布 | (i) 决定 D5 阈值；(ii) 决定 temporal_order/硬负线是否需 LoRA（caprl §6.3 提出后一直没人跑）；(iii) 预登记：片内两两余弦中位 >0.9 = 塔已饱和 → G1 直接带 LoRA 臂（0.4/0.5-4） |
| D12 | **better_zh 存档**：不再生产分面候选；10k 轮 s4-merged 落地产物仅供回归对照 | fusion §0.1 论证其投票机制自相矛盾 |
| D13 | **统计判定权属预登记**：G1/G2 用三区规则 + 配对 bootstrap + 种子数（§5.3），冻结后不临场改 | 审计三次代理指标教训的流程化（audit 结论 15）；评审 0.5-2 |

---

## 3. 执行阶梯（唯一时序；门未过不得进下一段花钱步骤）

```
Stage 0  (V100, 现在)  10k 试点重述 + 免费诊断 + 两探针        → 吞吐实测、温度坍缩检查
Stage 1  (b100, 在途)  zh v0.1 基线数据 + 基线训练             → G0: 基线可复现
Stage 2  (Claude)      工具落地: split_views / recap_diag / fuse_captions(F0)
Stage 3  (H100 空档)   300k×5 重述 → arm-A 数据 → 12.8M 样本×2 臂 → G1: 富集 vs 基线
Stage 4  (G1 过后)     k* 标定(~600 判定, §5.2) + F0 融合 → A/A−/C 消融 → G2: 融合是否成立
Stage 5  (G2 裁决后)   全量生产: 胜者配方, 桶优先级 rollout (RECAPTION→KEEP 低分序) → G3 验收
Stage 6  (paper 线)    CapRL-Video-4B 英文字幕 pass — 归 paper-plan Wk2-3, 本计划仅供 schema
```

### Stage 0 — 10k 试点重述 + 诊断（V100，本周可跑，零人力）

前置：10k 轮 `s3` + `s4`（apply_buckets）已产 `$OUT/v0_2/recaption_tasks.jsonl`（2,532 片，
RETRANSLATE 已并入与否以 apply_buckets 当前行为为准，量级不变）。

```bash
# ① 五样本重述（2,532 片 × 5 ≈ 12.7k 次生成 ≈ 1.3× 一遍 s2 的生成量, 8 卡当天级）
bash tools/data_audit/run_v100.sh recap
# ② 跑完记录吞吐（填 §5 成本表；这是全量排期的唯一可信输入）:
#    grep -c . $OUT/recap5.*.jsonl ; 用日志时间戳算 片·样本/秒/卡

# ③ 两个侧批诊断（各 ~300 片, 主批跑完后再跑或占 2 张空卡; 服务于 §7 的相关性诊断）:
head -n 300 "$OUT/v0_2/recaption_tasks.jsonl" > "$OUT/v0_2/recap_diag300.jsonl"
#   (a) 固定帧对照 —— 量化「一致性是否由画面驱动」(support 直方图应与主批显著不同):
CUDA_VISIBLE_DEVICES=0 python tools/data_audit/vlm_audit_hf.py \
  --input "$OUT/v0_2/recap_diag300.jsonl" --video-root "$CLIPS" \
  --model Qwen/Qwen3-VL-8B-Instruct --mode recap --n-captions 5 --temperature 0.9 \
  --no-frame-jitter --num-shards 1 --shard-id 0 --output "$OUT/recap5_nojit.{shard}.jsonl"
#   (b) 低温对照 —— 量化「先验驱动核」(两温下都 5/5 一致的单元 = 模型先验, 非画面证据):
CUDA_VISIBLE_DEVICES=1 python tools/data_audit/vlm_audit_hf.py \
  --input "$OUT/v0_2/recap_diag300.jsonl" --video-root "$CLIPS" \
  --model Qwen/Qwen3-VL-8B-Instruct --mode recap --n-captions 5 --temperature 0.3 \
  --num-shards 1 --shard-id 0 --output "$OUT/recap5_t03.{shard}.jsonl"
```

**产出与验收**：
- `recap5.{0..7}.jsonl` 覆盖 ≥95% 队列 + 两个侧批产物就位；
- **温度坍缩检查**（recap_diag，Stage 2 工具就绪前先人工抽 20 片）：五条重述需有实质差异；
  预登记红旗 = 成对字符 3-gram Jaccard 中位数 > 0.8（≈五条近乎雷同）→ 升 temp / 查抖动；
- 三探针（D11，b100 侧）结果记入本文件附录；
- 30 片 `make_review_bundle` 快速目检（不打分、只看重述是否像样）。

### Stage 1 — zh v0.1 基线（b100，已在途）→ **G0**

按 `notes/live/from-claude.md` 的 §2 返工指引继续：conda py3.10 环境 → 11 分片数据装配 →
NebulaCV 基线训练。**G0 = 基线训练收敛且在 D10 两个评测集上出数**。没有 G0，后面一切比较无锚。
（检查项：zh 配置 `min_caption_length ≤ 1`，见 §1 消费端陷阱。）

### Stage 2 — 工具落地（Claude 侧，与 Stage 0/1 并行）

1. `tools/data_audit/split_views.py`：稠密重述 → 句级短视图（A′ 规则，MiniRBT tokenizer 实测截断）；
   输出直接是 `captions` 列表补丁。含单测（超长句、混杂标点、空重述）。
2. `tools/fuse_captions.py` 的 `--mode per-recap-views`：对**单条**重述跑 S7 式派生（K=2–3 条
   ≤32 tok 互补短句）——这是 **arm A 的生产工具**（D5），与融合共用同一派生提示。
   附带片-帧 CLIP 打分列（NebulaCV-24M 冻结初始权重）供 A− 选择与 B 回退。
3. `tools/data_audit/recap_diag.py`：读 `recap5*.jsonl`（主批+侧批），报：每片五条长度分布、
   成对 3-gram Jaccard / LaBSE 余弦、语料级坍缩红旗；**主批 vs 固定帧侧批的 support 直方图
   TV 距离**（红旗 <0.05：一致性非画面驱动）；**两温交集 5/5 率**（先验核占比，分类别）；
   **类别 5/5 率排序**（红旗：onscreen_text/count ≥ subject——最弱类别最自信 = §3.5 signature）。
4. `tools/fuse_captions.py`（F0 主流程）：S1 分解（含 `en_gloss`）→ S2 对齐计票 → S3 分级
   （k\* 可配，默认待标定）→ D7 类别守恒策略 → S5 闭集整合 → S6 自验收（反向分解 precision +
   recall + >10% 回退最佳单条）→ S7 短视图。**不含** S4 异质验真（那是 `fuse_verify.py`，Stage 5 后）。
   合成用本地 Qwen3 文本模型或裁判中转（`SETUP_JUDGE_RELAY.md`），实现时定。
5. 就绪后在 Stage 0 的 recap5-10k 上端到端冒烟（S1–S2 至少覆盖 **≥500 片**），产出第一份
   **support 分布报告**。**「投票有信息」判据（预登记，Stage 4 的入场券）**：按常见类别
   P(5/5) ≤ 60% 且 P(support∈{2,3,4}) ≥ 15%；若 P(support=1) ≥ 80% → 先修 S2 对齐阈值 τ
   （过紧）再判，不直接判死。

### Stage 3 — 300k 富集试验 → **G1（本计划的主门）**

1. **子集**：全语料均匀随机 300k（排除 DROP/不可读）；对子集跑 S1 嵌入分 + 既定阈值分桶
   （只为 D6 的「原字幕去留」和后续优先级，不改抽样本身）。
2. **预飞（H100，~2h）**：vLLM 起服 + 1k 片 bake-off：Qwen3-VL-8B vs **Qwen3-VL-30B-A3B**
   （同 `RECAP_PROMPT`、同 N、同 temp）。按人工抽检质量 + 实测吞吐定 Stage 3 字幕器（0.5-5；
   30B-A3B 激活 ~3B，vLLM 下单条成本预计 ≤8B dense；V100 后备档只能 8B）。
3. **重述**：300k × 5 = 1.5M 次生成。H100+vLLM ≈ **2–4 天 @4 卡**（按预飞实测修正）；
   V100-HF 后备档按 Stage 0 实测吞吐折算（预计 ~7 天 @8 卡，仅当 H100 无空档）。
   与 EN 论文线的卡期冲突时**论文线优先**（§1 算力红线）。
4. **两臂训练**（等步数、等 batch、12.8M 样本/臂，NebulaCV 同起点）：
   - `R_zh0`：v0.1 原字幕（子集内）；
   - `R_zhA`：富集——KEEP 片 `captions=[原字幕]+派生视图(A)`，RECAPTION 片 `captions=派生视图(A)`
     （无原字幕，D6）。A 的形态见 D5；若 per-recap 派生工具未就绪，预登记降级用 A′（句切）。
   - （若 D11-iii 探针显示塔饱和：加 `R_zhA+LoRA` 第三臂，文本塔 LoRA rank 4–8。）
5. **G1 判据（预登记三区，配对 bootstrap，门控+报告两集同向；单种子 SE≈1.4pp、MDE≈3pp——
   审计预测的是多 pp 级效应，够用）**：
   - **GO**：`R_zhA − R_zh0` ≥ **+1.5pp** 主 R@1（软 ROI 门）→ Stage 4；
   - **中间区** (0, +1.5pp)：A 臂补第二种子再判，不即时裁决；
   - **NO-GO**：≤ 0 → 跑 0.4 的 LoRA 归因实验后裁方向（塔瓶颈 vs 数据无头寸）。
     **NO-GO 只裁重述通道，不裁融合**（0.5-2）；若维护者仍要一锤定音，可在 10k 融合冒烟
     数据上做一次 C-vs-A 小样本对比再收摊。

### Stage 4 — 标定 + 融合消融 → **G2**

进入条件 = G1 过 **且** Stage 2-5 的「投票有信息」判据过（含 τ 修正回路）。

1. **k\* 标定（人力预算见 §5.2：~600 单元判定 ≈ 1 人·日，按单元分层，不整片全标）**：
   复用 merge-review 交互包（`make_review_bundle.py` 家族）标 present/absent（对照真实视频），
   对 k∈{2..5} 画 support→precision 曲线（isotonic 平滑），按精确优先取 k\*；
   `onscreen_text`/`count` **零决策标注**，各标 ~30 条仅作审计，守恒策略默认不放松（0.5-7）。
   标定顺带产出：**高 support 单元的实测错误率**（=系统性幻觉过门率，§7 风险表的直接测量）。
2. **消融臂**（同 300k 子集、同 12.8M 样本预算、只换 captions 来源；种子见 §5.3）：
   - **A** = 全 5 重述的派生视图并集（=R_zhA，复用 G1 已跑种子）；
   - **A−** = 片-帧 CLIP 分 top-3 重述的派生视图（选择臂，0.5-3）；
   - **C** = F0 融合短视图；
   - **A′** = 句切视图（1 种子，量化「格式效应」）；**B** = best-single（可选，预算许可才跑）。
3. **G2 判据（预登记三区，D13）**：记 `Δ = C − max(A, A−)`（逐 query 配对）：
   - **ADOPT 融合**：Δ 点估计 ≥ **+1.0pp** 主 R@1 **且** 配对 95% CI 下界 > −0.3pp（U10 无害门）
     **且** 3/3 种子同向 **且** 门控/报告两集同向；
   - **KILL**：Δ ≤ 0 或 CI 上界 < +0.5pp → 全量配方 = A 或 A−（择优），融合归档为
     「已证伪于本设置」，S0 产物照用；
   - **中间区**：加 2 种子或合并两评测集（~3k query，SE 降至 ~0.7pp）再判一次，仍中间 → KILL
     （融合的复杂度要求它明确赢，不确定性归零假设）。
   - ADOPT 后才启动 `fuse_verify.py`（F1：PaddleOCR-zh + GroundingDINO(`en_gloss`) + 二号 VLM
     限流）找回 D7 牺牲的召回；同时按 §7 触发条件跑二号 VLM 交叉抽检。

### Stage 5 — 全量生产 → **G3**

胜者配方按**桶优先级 rollout**：RECAPTION 桶 → KEEP 按 `sig_zh` 升序（分数现成，caprl §7 的
「劣质先行」原则），预算走到哪算哪；每 50 万片出一次增量数据版本。**采样数在此降档**：若 G2
杀掉融合，生产 N 降为 2–3（届时只有 A/A− 消费，多样性两条即饱和，0.5-5）；融合被采纳才维持
N=5。**G3 验收**沿用 `internvid-zh-data-audit-plan.md` §5：复跑 S0/S1 回归 + 300 条盲评
胜率 >65% + 训练侧 held-out 不退。

### Stage 6 — EN 论文线接口（所有权在 paper-plan，此处只钉两点）

1. 产物 schema 与 §4 相同（`captions` 列表、≤32 token、无时间戳、原 FLT 字幕保留为视图——
   EN 线无逐片审计标签，MobileCLIP2 同款混合先例）；
2. 字幕形态二选一（「prompt 约束 K=2–3 条短句」vs「稠密一条+句切」）在教师 pass 的 100k 试点上
   顺带裁决：按 32-token 截断存活率 + 500 条抽检幻觉率（<5%，caprl §8 既定门）。
   **zh 线的 G2 若证明融合成立，EN 线也只在 camera-ready 阶段考虑移植**（D2 红线）。

---

## 4. 数据 schema 定稿（v0.2 记录形态）

```jsonc
// 训练 JSON 保持精瘦(10M 级全量会整体载入内存): 稠密重述/融合稿一律留在旁路文件, 不进训练 anno
{
  "video": "clip.mp4",
  "caption": "<主字幕: KEEP=原zh; RECAPTION=首条短视图>",     // 兼容非 uniform 读者
  "captions": ["...全部训练视图, 见下"],                      // uniform 采样唯一消费的字段
  "meta": {"bucket": "KEEP|RECAPTION|FIX_TEXT", "src": "v0_2A|v0_2C", "n_views": 3}
}
// KEEP:      captions = [原zh] (+富集后: 句切视图×K)
// RECAPTION: captions = 句切视图×K (arm A) 或 融合短视图×K (arm C); 不含原zh (D6)
// 旁路留痕:  recap5.*.jsonl (五条稠密), fused.jsonl (融合稿+fuse_meta precision/recall/fallback)
```

zh 陷阱清单（装配时逐条核）：`min_caption_length ≤ 1`；视图 token 数按 **MiniRBT tokenizer**
实测（D11 探针定字预算，不按字数估）；`。`后无空格的句切正则；引号「」“”不被 pre_text 剥掉
（onscreen_text 视图依赖它）。

---

## 5. 成本与统计口径

### 5.1 生成成本（吞吐以 Stage 0 实测为准，下表为排期用估计）

| 批次 | 生成量 | V100-HF (8卡) | H100-vLLM (4卡) |
|---|---|---|---|
| Stage 0：10k 桶试点 | 12.7k | **当天级**（≈1.3× 一遍 s2） | — |
| Stage 3：300k×5 | 1.5M | ~7 天（后备档） | **2–4 天** |
| Stage 5：RECAPTION 桶全量（~2.5M×5） | 12.5M | 不做（F6） | ~3 周量级 → 必须分版本滚动 |
| 融合 S1/S2/S5 文本调用 | 每片 ~3 次短调用 | 文本模型另计，量级远小于 S0 | 同左 |

### 5.2 k\* 标定协议（Stage 4 才花；统计评审定稿）

目标是**在 4 个候选阈值里选一个**，不是精确估计每层精度——按**单元**分层抽样，不整片全标：

- **分层**：support 层 s∈{2,3,4,5} 各 ~120 个单元（常见类别合并抽），每片每层 ≤2 个单元
  （抽自 ~120–150 片的池，控制设计效应 <1.2）；另抽 ~80 个 support=1 单元（不为选 k\*，
  只为量化 S4 验真的负载与单例错误率）。合计 **~600 判定 ≈ 1 人·日**。
- **分波停止**：每层按 40 一波；当恰有一个 k 的 Wilson 下界 ≥ 目标精度且 k−1 的上界 < 目标
  精度即停；相邻层精度差 <3pp = 选择不敏感，按 0.3 精确优先原则**向上取整**。
  对 support→precision 做 isotonic 回归平滑后再读阈值。
- **高风险类别**：`onscreen_text`/`count` 各 ~30 条**审计标注**（不参与 k\* 决策），
  always-verify 预注册不动——「证明可信任」需 ~60 个零错高支持单元（三分律 <5% 错误率），
  其稀有度意味着要标 300+ 片，而 OCR/检测验真近乎免费，标注的期望价值为负（0.5-7）。

### 5.3 训练消融口径（G1/G2 通用，预登记，D13；统计评审定稿）

- 等步数、等 batch、同起点、同评测、同 max_txt_l=32；**逐 query 配对分析**（同一 query 在两臂
  的对错配对，SE(Δ)≈1.0–1.2pp，优于独立差的 1.3√2）；评测报 query-level bootstrap 95% CI。
- **种子**：G1 每臂 1 种子（MDE≈3pp，audit 预测多 pp 效应，够）；中间区补种子（Stage 3-5）。
  G2 的 A/A−/C 各 **3 种子**（A 复用 G1 的跑），A′ 1 种子，B 可选；先用基线重复跑实测
  σ_seed——若 >1pp，全部口径按实测重标。
- **三区判定**：G1 见 Stage 3-5；G2 见 Stage 4-3。不再使用「+0.5pp 点估计即过」之类
  低于噪声分辨率的判据（1k query 下那是掷硬币）。
- **算力注记**：12.8M 样本/臂的短训 ≈ 半天–1 天 @4×H100 量级（首跑实测后修正）；G2 全包
  （3+3+3+1 臂·种子）≈ 1 周 4×H100 空档，与论文线协调后排期。
- 分短/长查询子集各报一行（fusion §8 长度失配护栏）；两评测集（门控 KEEP-2k + MSR-VTT-zh-1k）
  可合并成 ~3k query 提分辨率（SE→~0.7pp），中间区专用。

---

## 6. 明确不做（守边界）

- ❌ CapRL B1（GRPO 自训字幕器）——维持 caprl §4.2 与 U 评审的「除非 B0 显域 gap」；
- ❌ fusion F3（DPO/SynPO 偏好训字幕器）——G2 都没过之前不谈；
- ❌ N∈{3,8} 采样数消融、温度消融——G2 后可选项，不占主线；
- ❌ RETRANSLATE 独立通道 @全量、`xling_cos` 分桶（决策 1）、better_zh 续产（D12）；
- ❌ CapRL-MCQ 当事实门（fusion §3.5 已限定其为完整性排序器）；
- ❌ zh 线新增任何 KD/教师嵌入工作——归 paper-plan U7/U9，勿双头。

## 7. 风险登记（预登记的失败模式与信号）

| 风险 | 信号（预登记） | 预案 |
|---|---|---|
| 温度坍缩→投票失效 | Stage 0 红旗：3-gram Jaccard 中位 >0.8；或成对嵌入余弦均值 ≥0.95 | 升 temp 至 1.0 / 查帧抖动是否生效，重产侧批复检 |
| 一致性非画面驱动 | 固定帧侧批 vs 主批的 support 直方图 TV 距离 <0.05 | 投票权重下调预期；加速转向异质验真 |
| 先验驱动核过大 | 两温（0.3/0.9）交集 5/5 单元占比高，且集中在某类别 | 该类别并入 always-verify；k\* 上调 |
| 最弱类别最自信（§3.5 signature） | P(5/5 \| onscreen_text 或 count) ≥ P(5/5 \| subject) | 守恒策略绝不放松；F1 提前 |
| 系统性幻觉过投票门 | Stage 4 标定：高 support 单元实测错误率偏高 | k\* 上调；**触发二号 VLM 交叉抽检**（~200 片、5 样本：某类别 ≥20% 的 Qwen-5/5 单元得 0/5 跨模型支持 → 该类别 always-verify） |
| 冻结塔抹平收益 | D11-iii 探针饱和（片内视图余弦中位 >0.9）；或 G1 NO-GO | 探针饱和 → G1 直接带 LoRA 臂；G1 后 → 0.4 归因实验，先归因再裁方向 |
| 评测集偏置 | 门控（KEEP 检索）与报告（MSR-VTT-zh）不同向 | 以两者同向为「过」，分歧=停下查数据 |
| 卡期冲突 | EN 论文线 R0–R3 占满 H100 | zh Stage 3 降 V100 后备档或顺延；论文优先（§1） |
| 8B 幻觉底噪偏高 | Stage 0 目检 / Stage 4 标定的单例（support=1）错误率 | k\* 上调；Stage 3 预飞若 30B-A3B 质量明显更好则换器重估 |

## 8. 服务器下一步（copy-paste；也会同步到 notes/live/from-claude.md）

```bash
# 101vector (V100 审计机):
bash tools/data_audit/run_v100.sh recap        # Stage 0 ①; 前置=s3+s4 已产 recaption_tasks.jsonl
# 跑完把 (总生成数, 起止时间, 卡数) 记回 from-server.md → 填 §5.1 吞吐

# b100: 继续 from-claude.md 2026-07-23(6) 的 §2 返工三步(conda py3.10 重建), 直到 G0 基线出数
```

## 9. 锚点

审计：[`audit-10k-findings.md`](audit-10k-findings.md) ·
[`../tools/data_audit/DATA_ISSUES.md`](../tools/data_audit/DATA_ISSUES.md)。
设计依据（执行层已由本文取代）：[`caption-fusion-scheme.md`](caption-fusion-scheme.md) ·
[`data-quality-caprl-scheme.md`](data-quality-caprl-scheme.md)。
论文线：[`paper-plan.md`](paper-plan.md)（§7）· [`plan-review-upgrades.md`](plan-review-upgrades.md)（U6/U7/U9）。
代码：`tools/data_audit/vlm_audit_hf.py`（`--mode recap`）· `tools/data_audit/run_v100.sh`（`recap`）·
`tools/data_audit/apply_buckets.py` · `dataset/pt_dataset.py:173-201`。

**评审补强引用（§0.5；两份原文 §2/§9 之外新增，2026-07-24 评审网扫）**：
MobileCLIP（arXiv 2311.17049，5 条字幕、增益集中前 1–2 条）· CLIPS（arXiv 2411.16828，
稠密字幕**子采样**优于整条）· Panda-70M（arXiv 2402.19479，best-of-N 教师选择 +7.0 MSR-VTT R@1）·
LiT（arXiv 2111.07991，锁文本塔是弱配置）· Long-CLIP（arXiv 2403.15378，CLIP 有效文本长度
<20 token）· DreamLIP（arXiv 2403.17007，长字幕拆子视图）· n=5 一次调用整合先例
（arXiv 2510.27164，仅字幕质量、非对比训练）· Qwen3-VL-30B-A3B（HF 卡，MoE 激活 ~3B）。
