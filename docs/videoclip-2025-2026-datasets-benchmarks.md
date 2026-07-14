# 2025–2026 VideoCLIP 训练集与 Benchmark 调研报告

> 调研范围：2025–2026 年 VideoCLIP（视频-文本对比学习 / video-language pretraining）研究中主流、常用、有影响力的**训练集**与**评测基准**。
>
> 方法：6 个搜索角度 → 26 个来源 → 提取 109 条声明 → 对抗性核实 25 条（3 票制，需 2/3 通过；23 confirmed / 2 refuted）。关键数字均经多源核实，并标注置信度与需注意的坑。

## 总览

当前研究已固化为 **「大规模网络视频-文本预训练 + 多任务下游评测」** 范式。开源预训练基座以 **InternVid** 和 **Panda-70M** 为主，视频基础模型的数据规模持续放大到亿级。下游评测在检索 / 动作识别上沿用经典「六件套 + 标准动作集」，而 2025–2026 的新增量几乎全部集中在**长视频、时序逻辑、跨视频推理**这几个诊断性方向。

---

## 一、大规模视频-文本预训练数据集（训练集）

| 数据集 | 规模 | 来源 / 特点 | 代表模型 / 出处 | 置信度 |
|---|---|---|---|---|
| **InternVid** | >7M 视频、~760K 小时、**234M 片段**、4.1B 词 | OpenGVLab；LLM 多尺度生成字幕；含 InternVid-10M/50M/200M 子集与 **InternVid-10M-FLT**（顶部 30% 高质量过滤子集） | ViCLIP（ViT-L）；ICLR 2024 spotlight；2024.6 发布 230M 全量版 | 高 |
| **Panda-70M** | **70.7M**（70,723,513）高分辨率视频-字幕对，源自 ~3.78M 视频、167K 小时 | 视频取自 **HD-VILA-100M**（长 YouTube 切片）；字幕由 **8 个跨模态 teacher 自动生成**，再用检索模型筛选 | CVPR 2024（arXiv 2402.19479） | 高 |
| **DataCompDR-1B** | ~1B 网络图文对 | MobileCLIP 引入的 DataComp-1B **强化版**（加合成字幕），用于图文预训练阶段 | MobileViCLIP | 高 |

**预训练规模已进入亿级**（经核实）：

- **InternVideo2**：**412M** 条多模态数据（2M 视频 + 50M 视频-文本对 + 60M 视频-音频-语音-文本对 + 300M 图文对），其中视频-文本语料约 100M。
- **VideoPrism**：对比训练约 **600M** 视频-文本对（36M 高质量 + 582M 噪声文本）。

> ⚠️ **注意坑**：一条常见误传「InternVideo2 使用 WebVid10M/WebVid2M」经 0-3 全票驳回，**不成立**。WebVid 因版权已下架，在 2025–2026 基本被 InternVid / Panda-70M 取代（其确切现状仍是开放问题）。

---

## 二、主流下游评测 Benchmark（经典、仍是标准）

**文本-视频检索「六件套」**（截至 2025 年 9 月仍为主流，新数据集只是增量补充）：

`MSR-VTT · MSVD · DiDeMo · ActivityNet(-Captions) · VATEX · LSMDC`

— InternVideo2 等模型均在这六个基准上报告结果。

**零样本动作识别标准集**：

| Benchmark | 规模 |
|---|---|
| Kinetics-400 | 400 类 |
| UCF-101 | 13,320 片段 / 101 类 |
| HMDB-51 | 6,766 片段 / 51 类 |
| Something-Something V2 | 220,847 视频（常改写为 173 路多选 SSV2-MC） |

**长视频 QA**：**EgoSchema**（InternVideo2 在此取得零样本 SOTA，+9 分）。

---

## 三、2025–2026 新出现 / 成为主流的 Benchmark

### 长视频理解

> ⚠️ **命名撞车**：存在**两个不同的 LVBench**，引用时务必核对统计数字。

- **LvBench**（IJCV 2025, arXiv 2312.04817）：100 部跨题材**电影**、**20,061 QA**、视频时长 70s–4h，覆盖单 / 多 / 全场景。
- **LVBench**（ICCV 2025, zai-org, arXiv 2406.08035）：**极长视频**基准，103 个 YouTube 视频、最长约 2 小时（平均约现有最长数据集的 5 倍），评测 6 项核心能力、26 种细粒度问题类型。

### 诊断型 / 时序逻辑 / 跨视频推理（2025–2026 新趋势）

| Benchmark | 规模 | 任务类型 | 出处 |
|---|---|---|---|
| **TimeBlind** | 600 实例 / **2,400 视频-问题对** | 时空组合性；三层时序理解（Events / Event Attributes / Structural Event Logic） | arXiv 2602.00288 |
| **CrossVid** | **5,331 视频 / 9,015 QA**、4 维度 10 任务 | **跨视频推理（CVR）**，要求跨多视频聚合比较，填补单视频基准空白 | AAAI 2026（arXiv 2511.12263） |
| **Video-MME-Logical** | 503,750 程序化合成视频（500K 训练 + 3,750 测试） | 受控**时序逻辑**诊断；25 细粒度类别 / 5 逻辑操作组 | arXiv 2606.27828 |

### 代表性模型时间线

- **MobileViCLIP**（ICCV 2025, arXiv 2508.07312）：**移动端 / 高效方向**，两阶段训练——图文用 DataCompDR-1B，视频-文本用 InternVid-10M-FLT；检索零样本评测于 MSR-VTT / DiDeMo / ActivityNet-Captions。**与本仓库最相关。**
- **InternVideo2.5**（2025.1, OpenGVLab, arXiv 2501.12386）：聚焦「长且丰富上下文建模（LRC）」，可记忆比原版长 ≥6 倍的视频输入。

---

## 四、需注意的坑

1. **两个 LVBench 命名撞车**：LvBench（IJCV 2025，电影 QA）vs LVBench（ICCV 2025，极长 YouTube 视频）。
2. **时间敏感性**：TimeBlind（2602.00288）、Video-MME-Logical（2606.27828）等 arXiv 编号对应 2026 年 2 月及以后，属**极新预印本**，尚未充分同行评审，最终数字可能变动。
3. **MobileViCLIP 检索描述**：其中「ActivityNet 5–10 分钟 / 200 类」不准确（实际平均 ~2–3 分钟；200 类是动作识别的 ActivityNet-200 划分，而非 Captions 检索）。
4. **两条被驳回声明（勿采信）**：「InternVideo2 使用 WebVid10M/WebVid2M」（0-3 驳回）、「VidVec / OriginAI SOTA」（1-2 驳回）。
5. **DataCompDR-1B** 严格说是加入合成字幕 / 强化的 DataComp-1B，非纯原始网络图文对。

---

## 五、未覆盖 / 待补充调研的开放问题

- 主流 **VideoQA / 时序 grounding** 基准（MVBench、Video-MME、NExT-QA、TempCompass、MLVU）的相对地位与规模。
- 除 InternVid / Panda-70M 外，是否有 2025–2026 更大规模 / 更高质量开源语料（如 LLaVA-Video-178K、InternVid2）成为新基座。
- WebVid（WebVid-2M / 10M）的确切退役现状。
- TimeBlind、CrossVid、Video-MME-Logical 等诊断型新基准的实际采纳度与主流模型横向比较数据。

---

## 主要来源

| 类别 | 来源 |
|---|---|
| InternVid | arXiv 2307.06942；github.com/OpenGVLab/InternVideo |
| Panda-70M | arXiv 2402.19479；github.com/snap-research/Panda-70M |
| InternVideo2 | arXiv 2403.15377 |
| InternVideo2.5 | arXiv 2501.12386 |
| MobileViCLIP | arXiv 2508.07312 |
| VideoPrism | arXiv 2402.13217 |
| 检索基准标准集 | arXiv 2509.16560；arXiv 2505.23952 |
| LvBench（电影 QA） | link.springer.com/article/10.1007/s11263-025-02555-8；arXiv 2312.04817 |
| LVBench（极长视频） | github.com/zai-org/LVBench |
| TimeBlind | arXiv 2602.00288 |
| CrossVid | arXiv 2511.12263 |
| Video-MME-Logical | arXiv 2606.27828 |
