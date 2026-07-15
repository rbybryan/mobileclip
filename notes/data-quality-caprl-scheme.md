# 训练数据质量升级方案：CapRL++ 引导的「强化视频-文本数据」配方 for MobileViCLIP

> 目标：在**不改动模型结构、不引入任何推理期开销**的前提下，用「更好的数据」把
> MobileViCLIP 的零样本视频检索/分类和**时序理解**能力顶上去。方案对齐 2025–2026
> 主流「蒸馏即数据 / 强化数据」路线（MobileCLIP2 DFNDR、HQ-CLIP、CapRL++、ReSpec），
> 并针对本仓库「视频 + 端侧 + 冻结骨干」的特点做了原生改造。
>
> 本文档是设计方案（notes/），不改动训练代码；落地时按 §7 的分阶段清单执行。

---

## 0. 现状与问题（扎根本仓库代码）

| 维度 | 本仓库当前实现 | 位置 |
|---|---|---|
| 训练语料 | InternVid-10M-FLT 视频-文本对，纯 JSON `{"video","caption"}` | `configs/data.py:6-88`（`internvid_v1…v11` → `internvid_all`） |
| 字幕来源 | **上游 OpenGVLab 的单条 LLM 字幕**，直接读取，无重写 | `dataset/pt_dataset.py:261-272` `get_anno` |
| 字幕钩子 | 已支持 `captions` 列表 + `caption_augmentation.caption_sample_type='uniform'` 随机取一条、`min_caption_length`、`jump_filter` | `dataset/pt_dataset.py:165-189, 92-120` |
| 训练损失 | **纯对比 InfoNCE**（`loss_weight=dict(vtc=1.0)`） | `scripts/pretraining/clip/mobileviclip_small/config.py:60-64`；`criterions.py:93-131` |
| 蒸馏损失 | `UTA_Loss` / `new_UTA_Loss` **存在但未接入**任何 config | `models/criterions.py:409, 522` |
| 骨干 | 加载 MobileCLIP-S2 冻结骨干（`freeze_text=True`），仅时序分支训练 | `models/mobileviclip_small.py:203-262` |

**三个可量化的数据侧短板：**

1. **字幕质量参差**：InternVid-FLT 的 "FLT" 只是按 CLIP-score 取了上游 30% 子集，字幕仍
   是单条、偏短、时序信息稀疏；MobileViCLIP 的对比学习**只吃这一条**。
2. **无字幕多样性/集成**：MobileCLIP2 的核心增益来自「每样本多条合成字幕 + 教师嵌入强化」，
   本仓库完全缺席（`caption_augmentation` 钩子空转）。
3. **时序盲**：CLIP 类视频模型的公认弱点（TVBench/TemporalBench 上顺序不敏感），而现有字幕
   与对比目标都不提供时序判别信号。

---

## 1. 我们借鉴的四条主流方法（各取其精华）

| 方法 | 一句话核心 | 我们要移植的点 |
|---|---|---|
| **CapRL / CapRL++**（ICLR'26 / arXiv 2606.09393） | 字幕质量 = **效用**：好字幕能让一个「看不到画面的 LLM」正确回答关于该画面的选择题；用该 MCQ 准确率作**可验证奖励**、GRPO 训练字幕器 | 用 4B 级紧凑字幕器产出**密集、时序对齐、可验证**的视频字幕；奖励含时序定位 MCQ | 
| **MobileCLIP2 / DFNDR**（TMLR'25） | 「蒸馏即数据」：每样本存**多条合成字幕 + 教师集成嵌入**，训练走 KD（λ=1）；数据效率 3.3–5× | 多字幕 + 教师嵌入「强化记录」；把休眠的蒸馏损失接进来 |
| **HQ-CLIP**（ICCV'25，VLM-150M） | 用 LVLM 产出**四种粒度**监督：长正描述 / 长负描述 / 短正标签 / 短负标签；150M 打败 DFN-2B(10×) | 多粒度标签 + **硬负字幕**作为 InfoNCE 之外的额外监督 |
| **ReSpec**（本仓库 notes 已点名） | 相关性(relevance)+ 特异性(specificity) 在线过滤，27.5% 数据 ≈ 全量检索 | 重写前先过滤，省重写算力、去泛化字幕 |

> 关键判断：这四者都指向同一件事——**把「原始 web 字幕 + 裸 InfoNCE」升级为「效用最优化
> 的多粒度字幕 + 教师强化 + 硬负监督」**。本方案是它们在**视频 + 端侧**场景下的统一落地。

---

## 2. 方案总览：InternVid-DR（Data-Reinforced）

四根支柱，均可独立开关、分阶段上线：

```
                原始 internvid_all (10M)
                        │
  ┌─────────────────────┴─────────────────────┐
  │ Pillar A  质量过滤 (ReSpec 风格 + SemDeDup) │  → 高价值子集 ~2–3M
  └─────────────────────┬─────────────────────┘
                        │
  ┌─────────────────────┴─────────────────────┐
  │ Pillar B  CapRL++ 效用强化重写 (核心创新)   │  → 每片 K 条密集/时序对齐字幕
  └─────────────────────┬─────────────────────┘
                        │
  ┌─────────────────────┴─────────────────────┐
  │ Pillar C  多粒度 + 时序硬负标注 (HQ-CLIP)   │  → tags + 反事实硬负字幕
  └─────────────────────┬─────────────────────┘
                        │
  ┌─────────────────────┴─────────────────────┐
  │ Pillar D  强化记录 + 蒸馏损失 (DFNDR 移植)  │  → 教师嵌入；vtc + KD + 硬负 margin
  └────────────────────────────────────────────┘
```

---

## 3. Pillar A — 质量过滤（先剪枝，再花重写算力）

在昂贵的 LVLM 重写之前先过滤，是 ReSpec/CapRL 共识（CapRL 用 SemDeDup 去重）。三个信号：

1. **相关性 (relevance)**：用**已在仓库里的**冻结 MobileCLIP-S2 图像塔（或更强教师
   InternVideo2 / SigLIP2）对每片抽 1–3 关键帧算 image-text 余弦，取分布上分位。
2. **特异性 (specificity)**：把字幕嵌入与一个「泛化字幕根」质心（如 "a video of ..." 类
   模板集的均值）比较，惩罚泛泛而谈的字幕（ReSpec 报告 27.5% 数据 ≈ 49 avg 检索）。
3. **语义去重 (SemDeDup)**：在视频嵌入空间聚类去近重复，防高频场景主导 batch。

**产出**：`InternVid-FLT-curated`（~2–3M 片）+ 每片的 `clip_score` / `specificity` 分数。
**落地**：新增 `preprocess/curate_internvid.py`，读 `configs/data.py` 的 shard，写回带分数
的 JSON；训练侧靠 `dataset/pt_dataset.py` 已有的 `min_caption_length` / `jump_filter` 阈值控制。

> 为什么先做：无训练、无推理期成本、直接服务「复现 + 改进」目标，且把后续 LVLM 重写的
> token 预算集中在真正有价值的片段上（重写 3M 而非 10M，成本降 ~3×）。

---

## 4. Pillar B — CapRL++ 效用强化重写（核心创新，视频原生）

这是方案的「主发动机」，把 CapRL++ 的**可验证效用奖励**用到 InternVid 的视频重写上。

### 4.1 CapRL++ 配方（我们照搬的部分）
- **解耦两阶段**：LVLM 生成候选字幕 → 一个**无视觉 LLM**（Qwen2.5-3B-Instruct 级）仅凭
  字幕回答该片的 MCQ；奖励 = MCQ 准确率。天然抗 reward-hacking（决策解耦、选项随机打乱、
  N 次采样平均）——对比 LLM-as-judge 会退化成空字幕或冗长跑题。
- **视频 MCQ 两类**：
  (i) **时序定位题**（事件边界、时间顺序、状态转移，带时间戳）；
  (ii) **通用理解题**（主体/动作/场景）。
  由强 LVLM 生成，并过滤到「有画面能答、无画面答不出」：`M_Vf(q,V)=a AND M_Vf(q)≠a`。
- **视频奖励**：`R = R_acc + α·R_format + β·R_len`
  - `R_acc`：无视觉 LLM 的 MCQ 准确率（二值、精确匹配）；
  - `R_format`：时间戳合法 + **时序单调**（起始时间非降）——这是我们最看重的时序信号；
  - `R_len`：反冗长长度惩罚。
- **GRPO** 训练；紧凑字幕器（Qwen3-VL-4B 级）即可匹敌 72B/235B → 标注可规模化。

### 4.2 我们的两条落地路径（成本分层）

- **路径 B0（推荐先做，零 RL 成本）**：直接调用**已开源的 CapRL++ 视频字幕器**
  （InternLM/CapRL，Qwen3-VL-4B 级）对 §3 的 curated 子集批量产出 **K=2–4 条**密集字幕。
  MobileCLIP2 证明**前 1–2 条字幕贡献主要增益**，故 K 不必大。无需自训字幕器即可拿到
  80% 收益。
- **路径 B1（可选，域自适应）**：若通用字幕器在 InternVid（YouTube 短片）域上有差距，
  用 InternVid 自身抽取的 MCQ 跑一轮 GRPO 做**域适配**（复用 CapRL++ 奖励代码）。
  仅当 B0 的消融显示域 gap 明显时才做。

### 4.3 为什么优于「直接 GPT-4o 重写」(ShareGPT4Video 路线)
- 效用奖励**可验证**，规避「冗长但跑题 / 幻觉细节」两种失败模式；
- 4B 模型达 72B/235B 质量 → 标注 3M 片在**单机多卡**可行，成本可控；
- **时序定位 MCQ 直接对症 MobileViCLIP 的顺序盲**——这是本方案最独特的一环（见 §5）。

---

## 5. Pillar C — 多粒度 + 时序硬负标注（HQ-CLIP × 本仓库时序方向的结合）

在 K 条正字幕之外，为每片再产出（用同一字幕器/或规则扰动，几乎零额外成本）：

- **短正标签 `tags`**：实体/动作短语（HQ-CLIP 的 short positive tags），供额外对齐监督；
- **时序硬负字幕 `hard_negs`**：对真实字幕做**反事实扰动**——
  颠倒事件顺序（"先坐下再起身" ↔ "先起身再坐下")、替换关键实体/动作。
  这是 HQ-CLIP「负描述」与本仓库 `notes/whitespace-directions.md#2「时序硬负微调」`的融合。

> **创新点闭环**：CapRL++ 的时序定位 MCQ 迫使字幕器写出**时序对齐**的字幕 → 我们把这些
> 时序信息**扰动成对比硬负** → 在损失里显式惩罚「顺序错误」的字幕。把「字幕生成」与
> 「模型已知弱点(时序盲)」串成一条闭环，而现有 MobileCLIP2/HQ-CLIP 都未针对视频时序。

---

## 6. Pillar D — 强化记录 + 蒸馏损失（DFNDR/DataCompDR 移植到视频）

### 6.1 强化样本 schema（扩展 InternVid JSON，向后兼容）
```jsonc
{
  "video": "clip.mp4",
  "captions": ["dense caption 1", "dense caption 2", "..."],  // Pillar B（已被 loader 支持）
  "tags":     ["person running", "sunset beach"],              // Pillar C
  "hard_negs":["先起身再坐下 ...", "..."],                      // Pillar C
  "clip_score": 0.31, "specificity": 0.72,                      // Pillar A（jump_filter 阈值用）
  // 可选（Pillar D-强化，仅在 curated 子集上预算内烘焙）：
  "teacher_img_emb": "embs/clip.npy",     // 视频教师(InternVideo2)集成嵌入
  "teacher_txt_emb": ["...","..."]        // 每条 caption 的文本教师嵌入
}
```
`dataset/pt_dataset.py:176-189` 已支持 `captions` 列表 + `uniform` 采样；只需**新增一种
`caption_sample_type`**（如 `dr_multi`）同时返回 `tags`/`hard_negs`，并按需 lazy-load 教师嵌入。

> **存储可行性**：MobileCLIP2 的 DFNDR 全量 ~140TB（20 亿样本 ×30 增强 bf16）。我们**只在
> curated ~2–3M 子集上烘焙教师嵌入**，量级降 ~1000×，单盘可存 → 让「蒸馏即数据」在本仓库
> 变得现实。这是端侧小学生 + 冻结骨干场景下每 FLOP 收益最高的一招（MobileCLIP2 数据效率 3.3–5×）。

### 6.2 损失（终于把休眠的蒸馏损失接进来）
`loss_weight=dict(vtc=1.0, vtc_kd=λ_kd, vtc_hn=λ_hn)`：
- `vtc`：对采样到的那条 caption 走原 InfoNCE（不变）；
- `vtc_kd`：用视频-文本教师（InternVideo2 / 复用 `criterions.py:new_UTA_Loss`）的**相似度矩阵
  软标签**做 KD（DFNDR 的 `L_Distill`，KL 散度、温度可调）；
- `vtc_hn`：把 `hard_negs` 作为 batch 内额外负样本，加 **margin 项**惩罚「时序错误字幕」得高分。

---

## 7. 分阶段落地清单（ROI 从高到低，均可提前止损）

| 阶段 | 内容 | 训练？ | 新增文件/改动 | 验收 |
|---|---|---|---|---|
| **P0** | Pillar A 过滤 + Pillar B0 重写(用开源 CapRL++ 器)，多字幕 `uniform` 采样 | 只重训对比 | `preprocess/curate_internvid.py`；`tools/recaption_caprl.py`；新 JSON | R@1 vs 基线（等算力/等数据预算） |
| **P1** | Pillar C 多粒度 + 时序硬负 + `vtc_hn` 损失 | 同上 | `criterions.py` 加硬负项；`caption_sample_type='dr_multi'` | TVBench/TemporalBench ↑，标准检索不掉 |
| **P2** | Pillar D 教师嵌入强化记录 + `vtc_kd` KD | 同上 | 接入 `new_UTA_Loss`；烘焙脚本 | 数据效率（达标所需样本数）↓ |
| **P3** | Pillar B1 GRPO 域自适配字幕器（仅当 B0 有域 gap） | 训字幕器 | 复用 CapRL++ GRPO 代码 | 字幕 Prism 分 / 下游再 ↑ |

**先做 P0**：无 RL、无推理开销、直接服务「复现+改进」，且在模型完全复现前就有用。

---

## 8. 评测与护栏

- **主指标**：零样本视频检索 MSR-VTT-1k（`configs/data.py` 已配 `msrvtt_1k_test`）、
  视频分类 K400；**时序指标**：TVBench / TemporalBench（需在 `configs/data.py` 加 benchmark）
  以验证「时序对齐字幕」主张。
- **消融**：字幕条数 K；过滤比例；有无 `hard_negs`；有无 KD——逐一验证每根支柱的边际贡献。
- **护栏（防分布漂移/幻觉）**：
  - 把**原始上游字幕保留为 K 条之一**，避免完全脱离真实文本分布；
  - CapRL 抗 hacking 三件套：奖励解耦、MCQ 选项随机打乱、N 次采样平均；
  - 每阶段监控 held-out 检索，硬负 margin 设小值起步防训练崩塌。

---

## 9. 相对既有工作的创新性小结

1. **时序-效用闭环**（本方案独有）：用 CapRL++ 的时序定位 MCQ 作判别性奖励通道 → 生成时序
   对齐字幕 → 反事实扰动成对比硬负 → 损失显式惩罚顺序错误。把字幕生成与 MobileViCLIP 的
   已知弱点(时序盲)串成闭环；CapRL++ 只做字幕、HQ-CLIP 只做图像，均未打通到视频对比时序。
2. **端侧可行的「蒸馏即数据」**：只在 curated 子集烘焙教师嵌入，把 DFNDR 的 140TB 降到单盘
   量级，让冻结骨干 + 小学生场景吃到 3.3–5× 数据效率。
3. **零 RL 起步路径 (B0)**：直接用开源 CapRL++ 字幕器即拿主要收益，把「自训字幕器」降级为
   可选的域适配步骤，大幅降低复现门槛。

---

## 参考

- CapRL++（视频，arXiv 2606.09393）· CapRL（ICLR'26，arXiv 2509.22647，代码 InternLM/CapRL）
- MobileCLIP2 / DFNDR（TMLR'25，arXiv 2508.20691）
- HQ-CLIP / VLM-150M（ICCV'25，arXiv 2507.22431）
- ShareGPT4Video（NeurIPS'24，arXiv 2406.04325）
- ReSpec / 本仓库既有方向：`notes/whitespace-directions.md#1,#2`、`notes/replication-plan.md`
- 本仓库现状锚点：`configs/data.py`、`dataset/pt_dataset.py:165-272`、
  `models/criterions.py:409,522`、`scripts/pretraining/clip/mobileviclip_small/config.py:60-64`
