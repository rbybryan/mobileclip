# 训练数据质量升级方案：CapRL++ 引导的「强化视频-文本数据」配方 for MobileViCLIP

> ⚠️ **重字幕（Pillar B / §4 / §7-P0）的执行层已被
> [`master-plan.md`](master-plan.md) 取代（2026-07-24）**：两条训练线的
> 字幕器分开钉（zh=Qwen3-VL 直出、EN=CapRL-Video-4B 归 paper-plan U7）、P0 改为
> 300k 子集→G1 门→全量的阶梯。Pillar A（过滤）与 Pillar C/D（tags/硬负/教师嵌入/KD）的
> 所有权在 `paper-plan.md` §7 / `plan-review-upgrades.md` U7/U9。本文的 §4.4（32-token
> 降维）、§6.3（冻结塔梯度流分析）、§8（护栏）仍是设计依据，继续有效。

> 目标：在**不改动模型结构、不引入任何推理期开销**的前提下，用「更好的数据」把
> MobileViCLIP 的零样本视频检索/分类和**时序理解**能力顶上去。方案对齐 2025–2026
> 主流「蒸馏即数据 / 强化数据」路线（MobileCLIP2 DFNDR、HQ-CLIP、CapRL++、ReSpec），
> 并针对本仓库「视频 + 端侧 + 冻结骨干」的特点做了原生改造。
>
> 本文档是设计方案（notes/），不改动训练代码；落地时按 §7 的分阶段清单执行。
>
> **v2 修订**（自审后）：新增 §4.4「32-token 文本上下文」一级约束及应对；修正 §6
> 对 `new_UTA_Loss` 的定性（它是**视觉特征对齐**损失，不是相似度矩阵 KD——DFNDR 式
> KD 需新写）；新增 §6.3 冻结塔梯度流分析（文本教师嵌入在 `freeze_text=True` 下收益
> 近零）；修正 ReSpec 描述并补正式引用（CVPR'25, arXiv 2504.14875）；硬负构造改为
> 「事件顺序扰动」而非绝对时间戳（模型只见均匀 8 帧，感知不到时间戳）；新增 §8 的
> 训练/测试长度失配护栏与 §7 的算力预算。

---

## 0. 现状与问题（扎根本仓库代码）

| 维度 | 本仓库当前实现 | 位置 |
|---|---|---|
| 训练语料 | InternVid-10M-FLT 视频-文本对，纯 JSON `{"video","caption"}` | `configs/data.py:6-88`（`internvid_v1…v11` → `internvid_all`） |
| 字幕来源 | **上游 OpenGVLab 的单条 LLM 字幕**，直接读取，无重写 | `dataset/pt_dataset.py:261-272` `get_anno` |
| 字幕钩子 | 已支持 `captions` 列表 + `caption_augmentation.caption_sample_type='uniform'` 随机取一条、`min_caption_length`、`jump_filter` | `dataset/pt_dataset.py:165-189, 92-120` |
| **文本长度** | **`max_txt_l = 32` token**（训练时超长即截断） | `scripts/pretraining/clip/mobileviclip_small/config.py:18,29` |
| 训练损失 | **纯对比 InfoNCE**（`loss_weight=dict(vtc=1.0)`） | `config.py:60-64`；`criterions.py:93-131` |
| 蒸馏损失 | `UTA_Loss` / `new_UTA_Loss` 存在但未接入；**它们是视觉特征对齐损失**（student/teacher 特征余弦，middle+final，`[K,B,N,C]`），不是相似度矩阵 KD | `models/criterions.py:409, 522` |
| 骨干 | 加载 MobileCLIP-S2 冻结骨干（**`freeze_text=True`**），仅时序分支 + vision CLIP projector 训练 | `models/mobileviclip_small.py:203-262`；`config.py:51-52` |

> **勘误（2026-07-23，见 `plan-review-upgrades.md` §2.6-2）：** 上表「视觉塔仅时序分支
> + projector 可训」有误——规范预训练配置是 **`freeze_vision=False`**（config.py:50），
> **视觉塔全量可训**；「仅时序分支」只适用于 `freeze_vision=True` 的微调模式
> （whitespace §2）。影响：§6.3 对视频侧 KD 的定性偏保守——KD/`uta` 的梯度受体是
> **整个视觉塔**，预期效力更强；文本侧（frozen、仅 projection）的结论不变。
| 帧采样 | 每片**均匀采 8 帧**（`num_frames=8`），无时间戳输入 | `config.py`、`dataset/video_utils.py` |

**三个可量化的数据侧短板：**

1. **字幕质量参差**：InternVid-FLT 的 "FLT" 只是按 CLIP-score 取了上游 30% 子集，字幕仍
   是单条、偏短、时序信息稀疏；MobileViCLIP 的对比学习**只吃这一条**。
2. **无字幕多样性/集成**：MobileCLIP2 的核心增益来自「每样本多条合成字幕 + 教师嵌入强化」，
   本仓库完全缺席（`caption_augmentation` 钩子空转）。
3. **时序盲**：CLIP 类视频模型的公认弱点（TVBench/TemporalBench 上顺序不敏感），而现有字幕
   与对比目标都不提供时序判别信号。

**两个方案必须尊重的硬约束（v2 新增，详见 §4.4 / §6.3）：**

- **32-token 文本窗**：任何「密集长字幕」若直接送入训练会被截断到 ~25 个词，信息全丢。
- **文本塔全冻结**：文本侧的任何监督（长文本适配、文本教师 KD）梯度无处可去，除非
  显式加 LoRA / 解冻。

---

## 1. 我们借鉴的四条主流方法（各取其精华）

| 方法 | 一句话核心 | 我们要移植的点 |
|---|---|---|
| **CapRL / CapRL++**（ICLR'26 arXiv 2509.22647 / arXiv 2606.09393） | 字幕质量 = **效用**：好字幕能让一个「看不到画面的 LLM」正确回答关于该画面的选择题；用该 MCQ 准确率作**可验证奖励**、GRPO 训练字幕器 | 用 4B 级紧凑字幕器产出**密集、时序对齐、可验证**的视频字幕；奖励含时序定位 MCQ |
| **MobileCLIP2 / DFNDR**（TMLR'25, arXiv 2508.20691） | 「蒸馏即数据」：每样本存**多条合成字幕 + 教师集成嵌入**，训练走 KD（λ=1）；数据效率 3.3–5× | 多字幕 + 教师嵌入「强化记录」；KD 损失（需新写，见 §6.2） |
| **HQ-CLIP**（ICCV'25, arXiv 2507.22431，VLM-150M） | 用 LVLM 产出**四种粒度**监督：长正描述 / 长负描述 / 短正标签 / 短负标签；150M 打败 DFN-2B(10×) | 多粒度标签 + **硬负字幕**作为 InfoNCE 之外的额外监督 |
| **ReSpec**（CVPR'25, arXiv 2504.14875） | 在线过滤四准则：模态对齐（干净）、任务相关性（与下游任务分布的概率对齐，密度估计）、**特异性**（与「空文本」root embedding 的距离，作信息量代理）、低延迟 | 重写前先过滤，省重写算力、去泛化字幕 |

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
  │ Pillar B  CapRL++ 效用强化重写 (核心创新)   │  → 每片 K 条短字幕(≤32 tok)
  └─────────────────────┬─────────────────────┘     + 密集字幕句级切分视图
                        │
  ┌─────────────────────┴─────────────────────┐
  │ Pillar C  多粒度 + 时序硬负标注 (HQ-CLIP)   │  → tags + 事件顺序反事实硬负
  └─────────────────────┬─────────────────────┘
                        │
  ┌─────────────────────┴─────────────────────┐
  │ Pillar D  强化记录 + 蒸馏损失 (DFNDR 移植)  │  → 视频教师特征/相似度 KD
  └────────────────────────────────────────────┘     (作用面：时序分支+projector)
```

---

## 3. Pillar A — 质量过滤（先剪枝，再花重写算力）

在昂贵的 LVLM 重写之前先过滤，是 ReSpec/CapRL 共识（CapRL 用 SemDeDup 去重）。三个信号：

1. **模态对齐 (alignment)**：用**已在仓库里的**冻结 MobileCLIP-S2 图像塔（或更强教师
   InternVideo2 / SigLIP2）对每片抽 1–3 关键帧算 image-text 余弦，取分布上分位。
2. **特异性 (specificity)**：按 ReSpec 定义，度量字幕嵌入与**空文本 `""` 的 root
   embedding**（最不具体的锚点）的距离，距离越近说明字幕越「泛」，予以惩罚。
   （若后续要对齐特定下游任务，还可加 ReSpec 的 relevance 密度估计项，初版可省。）
3. **语义去重 (SemDeDup)**：在视频嵌入空间聚类去近重复，防高频场景主导 batch。

**产出**：`InternVid-FLT-curated`（~2–3M 片）+ 每片的 `clip_score` / `specificity` 分数。
**落地**：新增 `preprocess/curate_internvid.py`，读 `configs/data.py` 的 shard，写回带分数
的 JSON；训练侧靠 `dataset/pt_dataset.py` 已有的 `min_caption_length` / `jump_filter` 阈值控制。

> 为什么先做：无训练、无推理期成本、直接服务「复现 + 改进」目标，且把后续 LVLM 重写的
> token 预算集中在真正有价值的片段上（重写 3M 而非 10M，成本降 ~3×）。

> **外部 SOTA 复核附注（2026-07-30）：** 一次外部交叉核查提出「DFN/MetaCLIP2 式语料过滤」
> 疑似空白，核查后发现**不成立** —— 上表三个信号（alignment/specificity/SemDeDup）已经是
> 一个可比的三信号过滤栈,而非欠规格。与 DFN 谱系相比,唯一确实缺席的是 **DFN 本体**
> （Fang 等,*NeurIPS* 2023 ——训练一个专用**过滤网络**,而不是用现成的 CLIP-score 阈值 /
> ReSpec 启发式打分）或 MetaCLIP2 式**元数据平衡重采样**。两者都是真实存在但**价值不确定**
> 的第四信号：DFN 在图文数据上验证过明显优于纯 CLIP-score 过滤,但从未在视频-字幕数据上
> 验证过,且训练一个过滤网络是不小的额外投入。**结论：不作为新增项排队** —— 只有当上述
> 三信号栈的过滤质量被证明不够(如 Pillar A 产出的子集在 B0/R0 上明显劣于预期)时才值得
> 追加这第四个信号。

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
  （InternLM/CapRL，Qwen3-VL-4B 级）对 §3 的 curated 子集批量产出字幕（形态见 §4.4）。
  MobileCLIP2 证明**前 1–2 条字幕贡献主要增益**，故 K 不必大（K=2–4）。无需自训字幕器
  即可拿到 80% 收益。
- **路径 B1（可选，域自适应）**：若通用字幕器在 InternVid（YouTube 短片）域上有差距，
  用 InternVid 自身抽取的 MCQ 跑一轮 GRPO 做**域适配**（复用 CapRL++ 奖励代码）。
  仅当 B0 的消融显示域 gap 明显时才做。

### 4.3 为什么优于「直接 GPT-4o 重写」(ShareGPT4Video 路线)
- 效用奖励**可验证**，规避「冗长但跑题 / 幻觉细节」两种失败模式；
- 4B 模型达 72B/235B 质量 → 标注 3M 片在**单机多卡**可行，成本可控；
- **时序定位 MCQ 直接对症 MobileViCLIP 的顺序盲**——这是本方案最独特的一环（见 §5）。

> **重述-融合（补充件）**：对 RECAPTION 桶的低质片，[`fusion-scheme.md`](fusion-scheme.md)
> 用**自己的前端**——同一提示对同片**独立采样 5 条稠密重述**（非分面候选）——再做「并集-验证-整合」融合成
> 一条更全、且验收过幻觉的字幕，按 §4.4 降维成 K 条训练视图。**关键点**：跨样本一致性（support 投票）要成立，
> 多条字幕必须是**独立冗余观测**（自一致性，SelfCheckGPT 式），故前端刻意**不分面**——分面会摧毁投票信号
> （见该文 §0.1 对旧 better_zh 方案自相矛盾的修正）。算子基于 SelfCheckGPT/FActScore/OwlCap/Scene-Graph-Consolidation；
> 信任门阈值 k\* 经视频接地人评**标定**（偏精确），系统性幻觉靠**异质验真器**（GroundingDINO/OCR/二号 VLM）兜。
> **是否比直接 uniform-K 更好，是其 §6.4 的可证伪消融**，不预设结论。

### 4.4 一级约束：`max_txt_l = 32`，密集字幕必须「降维」进训练（v2 新增）

CapRL++ 的原生输出是**带时间戳的密集长段落**，而本仓库文本塔训练窗口只有
**32 个 token（约 25 个英文词）**，直接投喂会被截断成「开头半句」。三层应对，全部
在数据侧完成、不动冻结文本塔：

1. **产出形态改造（首选）**：用 prompt + `R_len` 把字幕器约束为
   「**每片 K 条、每条 ≤ 30 token、彼此覆盖不同事实**」的**检索风格短字幕**
   （CapRL 的效用奖励天然兼容：K 条合起来答 MCQ）。相当于把「一段密集描述」
   重排成「K 个短训练视图」。
2. **句级切分兜底**：若直接用现成密集字幕（省一次生成），做**句子级切分**，
   每句独立成一条 caption 进 `captions` 列表，超 32 token 的句子丢弃或压缩。
   这与 uniform 采样配合 ≈ DreamLIP 式「长字幕拆子视图」。
3. **时间戳不进对比文本**：模型只均匀采 8 帧、无时间戳输入，绝对时间戳
   （"at 00:07…"）对文本塔是纯噪声 token。**strip 掉时间戳**，时序信息以
   「事件顺序措辞」（before/after/then）保留——这也是 §5 硬负扰动的作用面。
4. （可选低成本档）`max_txt_l` 提到 77——MobileCLIP 文本塔原生上下文即 77，
   零改动即可容纳更长句；再往上（如 255）需要位置嵌入插值 + 解冻微调，且
   MobileCLIP2 消融显示 77 vs 255 差异在 1 个标准差内，**不做**。

---

## 5. Pillar C — 多粒度 + 时序硬负标注（HQ-CLIP × 本仓库时序方向的结合）

在 K 条正字幕之外，为每片再产出（用同一字幕器/或规则扰动，几乎零额外成本）：

- **短正标签 `tags`**：实体/动作短语（HQ-CLIP 的 short positive tags），供额外对齐监督，
  同时天然满足 32-token 约束、贴近 MSR-VTT 短查询分布（见 §8 护栏）；
- **时序硬负字幕 `hard_negs`**：对真实字幕做**事件顺序反事实扰动**——交换事件先后
  （"a man sits down **then** stands up" ↔ "a man stands up **then** sits down"）、替换关键
  动作/实体。**只扰动顺序措辞，不用绝对时间戳**（模型感知不到时间戳，见 §4.4-3）。
  这是 HQ-CLIP「负描述」与本仓库 `notes/whitespace-directions.md#2「时序硬负微调」`的融合。

> **创新点闭环**：CapRL++ 的时序定位 MCQ 迫使字幕器写出**事件顺序正确**的字幕 → 我们把
> 顺序**扰动成对比硬负** → 在损失里显式惩罚「顺序错误」的字幕。把「字幕生成」与「模型
> 已知弱点(时序盲)」串成一条闭环，而现有 MobileCLIP2/HQ-CLIP 都未针对视频时序。

---

## 6. Pillar D — 强化记录 + 蒸馏损失（DFNDR 移植到视频，v2 大幅修正）

### 6.1 强化样本 schema（扩展 InternVid JSON，向后兼容）
```jsonc
{
  "video": "clip.mp4",
  "caption": "<原始上游字幕，保留为视图之一>",                 // §8 护栏
  "captions": ["short caption 1", "short caption 2", "..."],  // Pillar B（loader 已支持）
  "tags":     ["person running", "sunset beach"],              // Pillar C
  "hard_negs":["a man stands up then sits down", "..."],       // Pillar C（顺序反事实）
  "clip_score": 0.31, "specificity": 0.72,                      // Pillar A（jump_filter 阈值用）
  // 可选（Pillar D，仅在 curated 子集上预算内烘焙）：
  "teacher_vid_emb": "embs/clip.npy"      // 视频教师(InternVideo2)最终嵌入(+可选中间特征)
  // 注意：不烘焙文本教师嵌入 —— 见 §6.3，freeze_text 下无梯度受体
}
```
`dataset/pt_dataset.py:176-189` 已支持 `captions` 列表 + `uniform` 采样；只需**新增一种
`caption_sample_type`**（如 `dr_multi`）同时返回 `tags`/`hard_negs`，并按需 lazy-load 教师嵌入。

> **存储可行性**：MobileCLIP2 的 DFNDR 全量 ~140TB（20 亿样本 ×30 增强 bf16）。我们**只在
> curated ~2–3M 子集上烘焙视频教师嵌入**（单嵌入 512-d bf16 ≈ 1KB/片，3M 片 ≈ 3GB；即使
> 加中间特征也在百 GB 量级），单盘可存 → 让「蒸馏即数据」在本仓库变得现实。

### 6.2 损失（v2 修正：分清两种蒸馏，DFNDR 式 KD 需新写）

`loss_weight=dict(vtc=1.0, vtc_kd=λ_kd, vtc_hn=λ_hn, uta=λ_uta)`：

- `vtc`：对采样到的那条 caption 走原 InfoNCE（不变）；
- `vtc_kd`（**新写**，~30 行）：DFNDR 式**相似度矩阵 KD**——teacher 的 batch 内
  video-text 相似度矩阵做软标签，对 student 相似度矩阵取行/列双向 KL（温度可调）。
  仓库里**没有**现成实现，此前误以为可复用 `new_UTA_Loss`，实际后者是特征对齐损失；
- `uta`（**复用现成**）：`criterions.py:new_UTA_Loss` 本来就是「student 视觉特征 ↔
  teacher 视觉特征」的余弦对齐（middle+final）。用 InternVideo2 视频教师蒸馏
  **时序分支**的特征——与「只有时序分支在训练」天然匹配，是本仓库独有的低成本选项；
- `vtc_hn`：把 `hard_negs` 编码后作为该样本的**额外文本负例**拼进相似度矩阵，
  加 margin 项惩罚「顺序错误字幕」得高分。

### 6.3 冻结塔梯度流分析（v2 新增——决定 Pillar D 的作用边界）

`freeze_text=True` 且视觉塔仅时序分支 + CLIP projector 可训（`config.py:51-52`，
`mobileviclip_small.py`），意味着：

- **文本侧 KD 无梯度受体**：烘焙每条 caption 的文本教师嵌入毫无用处——文本塔参数
  一个都不会动。schema 里**砍掉** `teacher_txt_emb`（除非走下面的可选项）；
- KD/对比的全部塑形能力集中在**时序分支 + vision projector**——恰好是本仓库的训练
  自由度所在，所以 `uta` 视频特征蒸馏是四个损失里与训练设置最匹配的一项；
- **可选扩展**：给冻结文本塔加 **LoRA**（rank 4–8，几 MB），让 §4.4 的新字幕分布和
  `vtc_hn` 的顺序判别信号能微调文本表征。这是唯一能让文本塔「学会区分事件顺序」的
  通路——纯冻结下，若 MobileCLIP-S2 的文本嵌入本身分不开 "A then B" 与 "B then A"，
  `vtc_hn` 的收益上限就是零。**落地时先做一个 10 分钟的探针实验**：用冻结文本塔编码
  1k 对顺序互换句，看嵌入余弦是否 ≈1；若是，则 `vtc_hn` 必须搭配文本 LoRA 才有意义。

---

## 7. 分阶段落地清单（ROI 从高到低，均可提前止损）

| 阶段 | 内容 | 训练？ | 新增文件/改动 | 验收 |
|---|---|---|---|---|
| **P0** | Pillar A 过滤 + Pillar B0 重写(开源 CapRL++ 器，§4.4 短字幕形态) + 多字幕 `uniform` 采样 | 只重训对比 | `preprocess/curate_internvid.py`；`tools/recaption_caprl.py`；新 JSON | R@1 vs 基线（等算力/等数据预算） |
| **P1** | Pillar C 多粒度 + 时序硬负 + `vtc_hn`（先跑 §6.3 文本塔顺序探针，决定是否配 LoRA） | 同上(+可选 LoRA) | `criterions.py` 加硬负项；`caption_sample_type='dr_multi'` | TVBench/TemporalBench ↑，标准检索不掉 |
| **P2** | Pillar D：烘焙视频教师嵌入 + 接 `uta`(现成) + 新写 `vtc_kd` | 同上 | 烘焙脚本；`vtc_kd` 实现 | 数据效率（达标所需样本数）↓ |
| **P3** | Pillar B1 GRPO 域自适配字幕器（仅当 B0 有域 gap） | 训字幕器 | 复用 CapRL++ GRPO 代码 | 字幕 Prism 分 / 下游再 ↑ |

**先做 P0**：无 RL、无推理开销、直接服务「复现+改进」，且在模型完全复现前就有用。

**算力预算粗估（v2 新增，P0 的重写步）**：3M 片 × K=3 条 ≈ 9M 次生成。Qwen3-VL-4B +
vLLM 批推理、8 帧输入，按 1–2 段/秒/卡估 ≈ **1.5k–3k A100·小时**（16 卡 4–8 天）。
Pillar A 过滤（单塔嵌入）与 Pillar D 烘焙（教师前向一遍）各 ≈ 数百卡时，量级更小。
若预算紧张：先只重写 curated 集中 specificity 最低的 1M 片（劣质字幕收益最大），
其余保留原字幕——过滤分数天然给出重写优先级。

---

## 8. 评测与护栏

- **主指标**：零样本视频检索 MSR-VTT-1k（`configs/data.py` 已配 `msrvtt_1k_test`）、
  视频分类 K400；**时序指标**：TVBench / TemporalBench（需在 `configs/data.py` 加 benchmark）
  以验证「时序对齐字幕」主张。
- **消融**：字幕条数 K；过滤比例；短字幕形态 vs 句级切分（§4.4-1 vs 4.4-2）；
  有无 `hard_negs`（± 文本 LoRA）；`uta` vs `vtc_kd`——逐一验证每根支柱的边际贡献。
- **护栏（防分布漂移/失配）**：
  - **训练/测试长度失配**（v2 新增）：MSR-VTT 查询是 ~9 词短句，若训练全是长句会伤
    短查询检索。三重对冲：§4.4 本就产短字幕；`tags` 提供超短视图；**原始上游字幕
    保留为采样视图之一**（uniform 采样里给它固定权重，如 1/K）；
  - CapRL 抗 hacking 三件套照搬：奖励解耦、MCQ 选项随机打乱、N 次采样平均；
  - 幻觉抽检：对重写字幕抽样 500 条人工/强模型核对，幻觉率 >5% 则收紧 `R_acc` 阈值；
  - 每阶段监控 held-out 检索；`vtc_hn` margin 从小值(0.1)起步防训练崩塌；
  - 每个 loss 权重改动跑 12M-样本小预算先验证（对齐 MobileCLIP2 的 12.8M 消融惯例）。

---

## 9. 相对既有工作的创新性小结

1. **时序-效用闭环**（本方案独有）：用 CapRL++ 的时序定位 MCQ 作判别性奖励通道 → 生成
   事件顺序正确的字幕 → 顺序反事实扰动成对比硬负 → 损失显式惩罚顺序错误（必要时经
   文本 LoRA 打通梯度）。把字幕生成与 MobileViCLIP 的已知弱点(时序盲)串成闭环；
   CapRL++ 只做字幕、HQ-CLIP 只做图像，均未打通到视频对比时序。
2. **端侧可行的「蒸馏即数据」**：只在 curated 子集烘焙**视频侧**教师嵌入（§6.3 论证了
   文本侧在冻结下无意义），把 DFNDR 的 140TB 降到 GB 量级；且识别出仓库休眠的
   `new_UTA_Loss` 恰好匹配「只训时序分支」的特征蒸馏形态。
3. **零 RL 起步路径 (B0)** + **32-token 适配**：直接用开源 CapRL++ 字幕器、以「K 条短
   检索风格字幕」形态产出，绕开自训 RL 与长上下文两大门槛，把创新落在本仓库真实约束内。

---

## 参考

- CapRL++（视频，arXiv 2606.09393）· CapRL（ICLR'26，arXiv 2509.22647，代码 InternLM/CapRL）
- MobileCLIP2 / DFNDR（TMLR'25，arXiv 2508.20691）
- HQ-CLIP / VLM-150M（ICCV'25，arXiv 2507.22431）
- ReSpec（CVPR'25，arXiv 2504.14875）
- ShareGPT4Video（NeurIPS'24，arXiv 2406.04325）
- 本仓库既有方向：`notes/whitespace-directions.md#1,#2`、`notes/replication-plan.md`
- 本仓库现状锚点：`configs/data.py`、`dataset/pt_dataset.py:165-272`、
  `models/criterions.py:409,522`、`scripts/pretraining/clip/mobileviclip_small/config.py:18,51-52,60-64`
