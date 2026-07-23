# 候选字幕融合方案：把 `better_zh` 的 2–3 条互补候选合并为「更全 + 不幻觉」的优化字幕集

> **范围与边界**：本文档只解决一件事——**如何把 `better_zh` 里每片 2–3 条互补的短候选字幕，
> 融合/整合（fusion / consolidation）为一条更完整、且不引入幻觉的字幕**，并把它「降维」成
> 符合 32-token 训练窗的检索视图。它是 [`notes/data-quality-caprl-scheme.md`](data-quality-caprl-scheme.md)
> 的**补充件**，不是替代。
>
> **与 caprl-scheme.md 的分工**：那份方案把 K 条候选当作 **K 个独立训练视图**、交给
> `uniform` 采样消费（其 §4.4 / §5），它**不做候选间的融合**。本方案专门补上中间那一步——
> 在候选变成训练视图**之前**，先做一次「并集-验证-整合」，再按 caprl-scheme.md §4.4 的方式切成短视图。
> 即：本方案是插在 caprl-scheme.md **Pillar B（重写产出 K 条候选）与训练视图消费之间的一个新算子**。
>
> 本文档是设计方案（notes/），不改训练代码；落地按 §6 分阶段清单执行。所有引用的 arXiv id
> 均已联网核对（见 §8「参考」），无法证实的条目已显式标注。

---

## 0. 动因：P0 门通过（人工复核）

用户已**人工复核** `better_zh`（对低质片段，Qwen3-VL-8B 看 8 帧生成的每片 2–3 条 ≤30 字检索风格中文候选）：

- **P0 结论（通过）**：新候选**比原上游字幕更准确、也更详细**——RECAPTION 方向经人评证实。
  （回顾：4 阶段审计已定位问题不在翻译（错误率 3.6%），而在**英文源字幕**——~16% 与视频不符、
  ~99% 无时序信息、常过泛或为乱码 OCR。）
- **关键观察（本方案的直接动因）**：**3 条候选各自携带不同细节**——A 讲主体+动作，B 讲画面内文字，
  C 讲场景/环境。**把它们并集整合，理论上能得到一条比任何单条都更完整的字幕**。

> 一句话问题定义：候选是**互补而非冗余**的。既然互补，融合就有信息增益；但融合=**取并集**，
> 天然会把任何一条候选的幻觉也一并吸收进来。**本方案的中心张力**就是 OwlCap/CCCaption 点明的
> **完整性 ↔ 幻觉**权衡：既要吃到 A/B/C 的全部真实细节，又不能把某条候选的臆造细节转正。

---

## 1. 借鉴的方法（均已核对 arXiv id）

| 方法 | 一句话核心 | 我们移植的点 |
|---|---|---|
| **OwlCap / CSER**（AAAI'26，arXiv 2508.18634） | 视频字幕的「运动-细节」失衡问题；提出 **Caption Set Equivalence Reward**：把字幕拆成**原子单元(unit)**，做 **unit-to-set 匹配 + 双向验证**（precision=不臆造 / recall=不漏），在 GRPO 里同时压幻觉、促完整 | **原子单元分解 + 双向 precision/recall** 是我们「融合后验收」的核心度量 |
| **CCCaption**（CVPR'26，arXiv 2602.21655） | 双奖励 RL 求「**完整且正确**」：完整性=用多个 LVLM 把图拆成一组**视觉 query**、奖励能答更多 query 的字幕；正确性=把字幕拆成 **sub-caption query 逐条验真**、罚幻觉 | 「**拆成 query 逐条验真**」正是我们**验证单例细节(singleton)**的手段 |
| **CapRL / CapRL++**（ICLR'26 arXiv 2509.22647 / arXiv 2606.09393） | 字幕质量=**效用**：好字幕能让无视觉 LLM 答对该片 MCQ；MCQ 准确率作可验证奖励（解耦、抗 hacking） | 用 **MCQ 效用**作**融合选择器/验收器**：融合稿 vs 最佳单条，谁让 LLM 答对更多 MCQ 谁胜 |
| **Scene-Graph Consolidation**（arXiv 2502.16427） | 把多段字幕各自解析成场景图，用 **Hungarian 匹配**对齐同一实体，**合并=取属性并集**，迭代合并成一张图再解码成字幕；**无需额外微调** | 提供**「并集-融合」的算法骨架**：实体对齐 + 属性并集 + 迭代整合，天然去重、保细节 |
| **SynPO**（arXiv 2506.00835） | 视频细节字幕的偏好优化：构造偏好对、防负偏好主导、保语言能力、免参考模型 | 备选（F2）：若要把「融合稿 > 单条」变成可训练偏好信号时的 DPO 侧参考 |

> 判断：**OwlCap+CCCaption 给「怎么验收（拆单元、双向验真）」，Scene-Graph Consolidation 给
> 「怎么并集融合」，CapRL 给「用什么客观信号裁决」**。三者拼起来正好是一条
> 「**union → verify → consolidate**」的融合流水线，且全部落在**数据侧**，不动冻结文本塔。

---

## 2. 方案总览：CFU（Candidate-Fusion & Utility-verify）

对每片的候选集 `C = {c1, c2, c3}`（≤30 字/条），产出一条「验收合格的融合稿 `f`」+ 由 `f` 派生的
**K 条 ≤32-token 训练短视图**。核心是**先分解到细节单元、按一致性分级、验证单例、再整合**：

```
better_zh 候选  C = {c1, c2, c3}   (每条 ≤30 字, 各带不同细节)
        │
 ┌──────┴───────────────────────────────────────────────┐
 │ S1 分解  每条 → 原子细节单元 u=(实体/属性/动作/文字/场景/顺序) │  (OwlCap unit / SG node)
 └──────┬───────────────────────────────────────────────┘
        │
 ┌──────┴───────────────────────────────────────────────┐
 │ S2 对齐+计票  跨候选合并同义单元, 记 support = 出现候选数    │  (Hungarian/嵌入匹配, 取并集)
 └──────┬───────────────────────────────────────────────┘
        │
 ┌──────┴───────────────────────────────────────────────┐
 │ S3 分级                                                 │
 │   trusted   : support ≥ 2   → 直接采信 (自一致性)         │
 │   singleton : support = 1   → 必须回看视频验真           │  (CCCaption 逐 query 验真)
 └──────┬───────────────────────────────────────────────┘
        │
 ┌──────┴───────────────────────────────────────────────┐
 │ S4 单例验证  Qwen3-VL-8B 对 8 帧做 yes/no; 或 CapRL-MCQ  │  → 通过者升为 verified
 └──────┬───────────────────────────────────────────────┘
        │
 ┌──────┴───────────────────────────────────────────────┐
 │ S5 整合  LLM 把 {trusted ∪ verified} 融成 1 条密集稿 f,   │  (event-order 措辞, 见 §5)
 │          禁止新增未列出的细节 (closed-set 约束)           │
 └──────┬───────────────────────────────────────────────┘
        │
 ┌──────┴───────────────────────────────────────────────┐
 │ S6 验收  双向 precision/recall + 幻觉率闸门 (>10% 回退)   │  (OwlCap CSER 风格)
 └──────┬───────────────────────────────────────────────┘
        │
 ┌──────┴───────────────────────────────────────────────┐
 │ S7 降维  由 f 派生 K 条 ≤32-token 检索短视图 (§4.4 对齐)  │  → 进 captions 列表训练
 └────────────────────────────────────────────────────────┘
```

**为什么是「先分解再融合」而不是「直接让 LLM 把三条揉一起」**：直接揉合无法区分「A 独有的真细节」
与「B 独有的臆造细节」——两者在 LLM 眼里都是「某候选提到过」，会被无差别转正。分解到单元后，
**一致性(support)成为可计的信任先验**，单例被强制送去验真，这才是压幻觉的着力点。

---

## 3. 融合与幻觉控制（核心）

### 3.1 分解到原子单元（S1）
用一次 LLM 调用把每条候选拆成结构化单元。单元类型对齐 P0 观察到的互补维度：

```jsonc
// c1="一名男子在厨房切菜" 的分解示例
[
  {"type":"subject",  "text":"一名男子"},
  {"type":"action",   "text":"切菜"},
  {"type":"setting",  "text":"厨房"}
]
```
类型集：`subject / attribute / action / onscreen_text / setting / temporal_order`。
`onscreen_text`（画面内文字/OCR）**单列**——它是 P0 里候选 B 的典型独有细节，也是最易幻觉的一类，需单独闸门（§3.4）。

### 3.2 跨候选对齐 + 计票（S2）
把所有候选的单元放进一个池，做**同义合并**：文本嵌入余弦 >τ（初值 0.85）或同 `type` 下 LLM 判同义 →
合并为一个单元，`support = 覆盖到它的候选数`。合并时取**属性并集**（Scene-Graph Consolidation 的做法：
"红色卡车" + "卡车在左侧" → "左侧的红色卡车"）。产出去重后的单元集，每个单元带 `support ∈ {1,2,3}`。

### 3.3 自一致性分级 + 单例验真（S3–S4）——**信任规则**
- **trusted（support ≥ 2）**：≥2 条独立候选都提到 → 采信，不再验证。这是**跨候选自一致性**：
  两次独立 VLM 生成同时臆造同一细节的概率远低于单次，故一致即高置信。
- **singleton（support = 1）**：只有一条提到 → **默认不可信**，必须回看视频验真：
  - **验真-A（默认，便宜）**：把单元改写成一个封闭 yes/no 问题（"画面里有厨房吗？"），
    用 **Qwen3-VL-8B 看同样的 8 帧**回答；N=3 次采样多数通过才升 `verified`（CCCaption 逐 query 验真的轻量版）。
  - **验真-B（可选，更硬）**：用 **CapRL/CapRL++ 的 MCQ 效用**——把含该单元 vs 不含该单元的两版短稿分别喂
    无视觉 LLM 答该片 MCQ，含之者答对率显著更高才保留。成本更高，仅对 `onscreen_text` 等高风险单元开。
- 验真未过的单例**丢弃**（宁漏勿错；漏掉的细节 recall 损失由 §3.5 的 recall 项监控）。

### 3.4 高风险单元的额外闸门
- `onscreen_text`：VLM 转写文字幻觉率高。规则：**必须 support≥2 或 验真-B 通过**才进融合稿；否则整条丢弃，不做「猜一半」。
- `count`/数量词（"三个人"）：VLM 计数公认弱。数量词一律降级为**非数值措辞**（"几个人"/"a group"）或走验真-B，绝不直接采信单例数字。

### 3.5 整合 + 验收（S5–S6）——**闭集融合 + 双向验收**
- **整合（S5）**：给 LLM **只有 `{trusted ∪ verified}` 单元列表**，指令：「**仅用下列事实**写一条通顺中文字幕，
  **不得新增任何未列出的细节**（closed-set / 抽取式，非生成式补全），保留事件先后措辞」。closed-set 约束是防
  LLM 在「润色」时再造幻觉的关键。
- **验收（S6，OwlCap CSER 风格双向验证）**：
  - **precision（压幻觉）**：把融合稿 `f` **反向再分解**成单元，逐个检查是否 ∈ 输入单元集；
    出现输入里没有的单元 = 融合期新造幻觉，记为 `hallu_unit`。`precision = 1 − |hallu_unit| / |f 的单元|`。
  - **recall（保完整）**：`recall = |f 覆盖的 trusted∪verified 单元| / |trusted∪verified|`；漏采信细节则 recall 掉。
  - **幻觉率闸门**：与审计一致，**片级幻觉率 = 1 − precision**；若 `>10%` → 该片**回退用「最佳单条」**（见 §3.6），
    不用融合稿。批级统计若整体 `1−precision > 10%` → 收紧 τ / 关闭验真-A 改验真-B / 回退整批。

### 3.6 回退策略（融合不是免费的）
每片都保留一条**安全底**：若融合稿未过验收，回退为**最佳单条候选**——用 §3.4 的 MCQ 效用或简单的
片-帧 CLIP-score（复用仓库冻结 MobileCLIP-S2 图像塔）在 `C` 内选分最高的一条。**保证融合永不使某片变差**。

---

## 4. 适配 32-token 训练窗（与 caprl-scheme.md §4.4 对齐）

融合稿 `f` 通常比单条长、可能超 32 token。**不把 `f` 直接投喂对比模型**（会被截断成半句、且训练/测试长度失配，见 §7）。
`f` 的定位是**验证/记录产物**（人审、留痕、算 recall），训练视图由它**派生**：

1. **主路：重提示派生 K 条短视图**。给 LLM `f` + 其单元集，指令「产出 K 条 ≤30 字检索风格短句，**每条覆盖不同的
   trusted/verified 单元子集**（一条主体+动作、一条含画面文字、一条场景），彼此互补」。这等于把「一条密集稿」
   重排成「K 个短训练视图」，与 caprl-scheme.md §4.4-1 完全一致，且 K 条的**并集≈f 的信息**，符合 CapRL 效用观。
2. **兜底：句级切分**。省一次生成时，直接对 `f` 按句切分，逐句成一条 caption，超 32 token 的句子压缩或丢弃
   （对应 caprl-scheme.md §4.4-2 的 DreamLIP 式子视图）。
3. **短标签副产物**：把 `subject/action/setting` 单元直接吐成 `tags`（超短视图），喂 caprl-scheme.md Pillar C，
   天然满足 32-token 且贴近 MSR-VTT 短查询分布（§7 护栏）。

> **写回训练 JSON（向后兼容 caprl-scheme.md §6.1 schema）**：
> ```jsonc
> {
>   "video": "clip.mp4",
>   "caption": "<原上游字幕, 保留为采样视图之一>",       // §7 长度护栏
>   "captions": ["短视图1(主体+动作)", "短视图2(含画面文字)", "短视图3(场景)"], // ← 本方案 S7 产出
>   "fused_dense": "<融合稿 f, 仅记录/验证, 不进对比训练>",  // 留痕, 不采样
>   "fuse_meta": {"precision":0.94, "recall":0.88, "fallback":false, "K_from_fusion":3}
> }
> ```
> loader 侧无需改动：`captions` 列表已被 `dataset/pt_dataset.py:176-189` 的 `uniform` 采样支持；
> `fused_dense`/`fuse_meta` 是新字段，训练时忽略即可。

---

## 5. 保留时序角度（与 caprl-scheme.md 的时序闭环一致）

融合是「取并集」，若不约束会把多条候选压成一句无序静态描述，抹掉事件顺序。三条护栏：

1. **`temporal_order` 作为一类单元**：S1 分解时显式抽取「先…然后…最后」措辞；只要任一候选含顺序信息，
   它就进单元集，S5 整合时**必须保留**。
2. **整合提示强制事件顺序措辞**：S5 指令含「若含多动作，用 before/then/after（之前/然后/之后）串联」——
   与 caprl-scheme.md §5 的「事件顺序措辞而非绝对时间戳」一致（模型只见均匀 8 帧、无时间戳）。
3. **不引入时间戳**：融合稿一律 **strip 绝对时间戳**（"00:07"），顺序只用相对措辞承载。
4. **喂给硬负**：派生的含顺序短视图可直接作 caprl-scheme.md §5 `hard_negs` 的**顺序反事实扰动源**
   （"坐下然后起立"↔"起立然后坐下"），把融合与既有时序闭环接上。

---

## 6. 落地流水线 + 分阶段清单

### 6.1 新增文件
```
tools/fuse_captions.py     # S1–S7 主流程, 输入 better_zh JSON, 输出带 captions/fused_dense 的训练 JSON
tools/fuse_verify.py       # S4 单例验真 (Qwen3-VL-8B yes/no) + S6 双向验收, 可独立跑做审计
```
`fuse_captions.py` 建议 CLI：
```bash
python tools/fuse_captions.py \
  --in  better_zh.jsonl \
  --out internvid_fused.jsonl \
  --k 3 --tau 0.85 --n-verify 3 \
  --hallu-gate 0.10 --singleton-verify vqa \   # vqa=验真-A(默认) | mcq=验真-B
  --fallback best-single \
  --frames-root data/frames               # 复用已抽的 8 帧, 不重抽
```

### 6.2 分阶段（ROI 从高到低，均可提前止损；接在 caprl-scheme.md P0 之后）

| 阶段 | 内容 | 训练？ | 新增/改动 | 验收门 |
|---|---|---|---|---|
| **F0** | S1–S3 分解+计票+分级；**先只用 trusted(support≥2) 融合**，singleton 一律丢 | 否(先出数据) | `tools/fuse_captions.py`(无验真分支) | 抽 300 片人审：融合稿幻觉率 <10% 且比最佳单条更全 |
| **F1** | 接 S4 验真-A(Qwen3-VL yes/no) + S6 双向验收 + 回退；产 `captions` 短视图 | **只重训对比** | `tools/fuse_verify.py`；写 `captions` | 见 §6.3 消融：F1 数据 R@1 ≥ 最佳单条 且 ≥ uniform-K |
| **F2** | 高风险单元走验真-B(CapRL MCQ)；`temporal_order` 短视图接 §5 硬负 | 同上(+可选硬负) | MCQ 生成/验证脚本 | TVBench/TemporalBench ↑，标准检索不掉 |
| **F3**(可选) | 把「融合稿 ≻ 单条」做成 DPO/SynPO 偏好信号训字幕器 | 训字幕器 | 复用 SynPO/CapRL 代码 | 仅当 F1/F2 消融证明融合确有增益才做 |

**先做 F0**：零训练、零推理开销，且只用 trusted 单元时**幻觉风险最低**（全是≥2票细节），
能最快用人审确认「融合是否真比单条更全」这个前提。

### 6.3 关键消融（让「融合有用」成为**可证伪假设**，而非默认结论）
在同一 curated 子集、**等数据预算 + 等算力**下，只换字幕来源，比 MSR-VTT-1k / K400 检索：

| 臂 | 每片训练视图来源 |
|---|---|
| **A. uniform-K（对照，=caprl-scheme.md 现状）** | 原 2–3 条 better_zh 候选，直接 uniform 采样 |
| **B. best-single** | 只保留 §3.6 选出的最佳单条候选 |
| **C. fusion（本方案）** | S7 派生的 K 条融合短视图 |
| **C′. fusion+tags** | C 再加 `tags` 超短视图 |

**判据**：C 需**同时** ≥ A 与 ≥ B 才算「融合成立」；若 C ≈ A，则融合的额外算力不值，回退 uniform-K。
额外看**长度失配敏感度**：分「短查询/长查询」子集报 R@1（§7）。

---

## 7. 成本与护栏

**算力粗估（对 better_zh 覆盖的低质片段，设 N 片）**：
- S1 分解 + S2 对齐 + S5 整合：均为**纯文本 LLM**调用，每片 ~3 次短调用，Qwen 级 vLLM 批推 ≈ **数十片/秒/卡**，
  N=1M 片约 **十几卡时**，可忽略。
- S4 验真-A：**VLM 看 8 帧**，是主成本；每片 singleton 数 s×N 次 VQA。设 s≈2、N=1M → 2M 次 VQA，
  Qwen3-VL-8B + vLLM ≈ 1–2 片/秒/卡 → **300–600 卡时**（16 卡 1–2 天）。**复用已抽的 8 帧，不重抽**。
- S4 验真-B（MCQ）：仅对高风险单元开，量级更小但每次两遍 LLM，按需限流。
- 总体比 caprl-scheme.md §7 的重写步（9M 次生成、1.5k–3k A100·时）**小一个量级**——融合是重写的**后处理**，不是重复重写。

**护栏（honest limits）**：
- **训练/测试长度失配**：MSR-VTT 查询是 ~9 词短句；融合稿更长更全，若派生视图偏长会伤短查询检索。
  三重对冲同 caprl-scheme.md §8：S7 本就产 ≤30 字短视图、`tags` 提供超短视图、原上游字幕保留为采样视图之一。
  消融 §6.3 必须分短/长查询报点。
- **幻觉闸门 >10%**：与审计既有闸门一致（§3.5）；片级过闸即回退最佳单条，批级过闸即收紧参数或回退整批。
- **验真器同源风险**：S4 用 Qwen3-VL-8B 验真，而候选也是 Qwen3-VL-8B 生成——**同模型的系统性幻觉不会被自查抓到**。
  缓解：验真时**换提示/换温度/换帧采样相位**制造独立性，高风险单元（`onscreen_text`）尽量走验真-B（换成无视觉 LLM 的 MCQ 链路，异质）。
- **冻结文本塔的收益上限（诚实存疑）**：文本塔 `freeze_text=True`。**「更完整的字幕」能否被一个冻结的
  32-token 文本编码器转化为更好的检索表征，本身不确定**——若 MobileCLIP-S2 文本嵌入对「多细节 vs 少细节」
  短句本就近似饱和，融合的信息增益可能在编码端被抹平。这正是 §6.3 消融要回答的核心问题，**不预设结论**。
- **support≥2 的前提**：本方案的信任先验建立在「≥2 条独立候选一致即可信」。若 better_zh 只有 2 条候选，
  则 support 上限=2、trusted 判据退化为「两条都提」，可信细节会偏少、更依赖 S4 验真。**候选数 K=2 vs 3 的影响
  应作为 §6.3 的附加消融**。

---

## 8. 参考（arXiv id 均已联网核对，2026-07 检索）

- **OwlCap: Harmonizing Motion-Detail for Video Captioning via HMD-270K and Caption Set Equivalence Reward**
  — AAAI'26，arXiv **2508.18634**。用户点名；CSER=单元分解+unit-to-set 匹配+双向 precision/recall，
  在 GRPO 里平衡完整性与幻觉。**已核对。**
- **CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning**
  — CVPR'26，arXiv **2602.21655**。完整性=拆视觉 query 奖答更多；正确性=拆 sub-caption query 逐条验真罚幻觉。
  **已核对**（我们 S4 单例验真直接借此）。
- **CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning**
  — ICLR'26，arXiv **2509.22647**，代码 InternLM/CapRL。字幕效用=无视觉 LLM 答 MCQ 准确率，作可验证奖励。**已核对。**
- **CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning**
  — arXiv **2606.09393**。CapRL 的图+视频统一版，覆盖时空理解。用户点名；**arXiv 页面已核对存在**
  （日期 2606=2026-06，属很新预印本，正式发表状态未定，视为「已核对 id、结论以原文为准」）。
- **Fine-Grained Captioning of Long Videos through Scene Graph Consolidation**
  — arXiv **2502.16427**。分段字幕→场景图→Hungarian 对齐+属性并集+迭代合并→图到文解码；无需额外微调。
  **已核对**（我们 S2 并集融合的算法骨架）。
- **SynPO: Synergizing Descriptiveness and Preference Optimization for Video Detailed Captioning**
  — arXiv **2506.00835**，代码 github.com/longmalongma/SynPO。防负偏好主导、保语言能力、免参考模型的偏好优化。
  **已核对**（F3 可选偏好训练的参考）。

**相关但本方案未深用（检索到、供延伸）**：AVC-DPO（arXiv 2507.01492，DPO 对齐视频字幕）；
ShareGPT4Video（NeurIPS'24，arXiv 2406.04325，已在 caprl-scheme.md 引）。

**无法证实/存疑**：暂无——本方案所有点名方法（OwlCap、CapRL、CapRL++）与全部引用 id 均已在
arXiv 检索到对应页面。唯一保留意见是 **CapRL++（2606.09393）为 2026-06 的极新预印本**，
其「视频时序 MCQ」细节以原文为准，勿在细节上过度外推。

**本仓库锚点**：`dataset/pt_dataset.py:165-189`（`captions` 列表 + `uniform` 采样钩子）、
[`notes/data-quality-caprl-scheme.md`](data-quality-caprl-scheme.md)（§4.4 短视图降维、§5 时序硬负、§6.1 JSON schema、§8 护栏）。
