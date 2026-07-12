# 2025 年图像 · 视频 CLIP 研究进展报告

> 从端侧高效模型(MobileCLIP2)到前沿通用骨干(SigLIP 2 / Perception Encoder)—— 对比语言-图像/视频模型的架构、数据、训练目标与基准全景。视角锚定本仓库(MobileViCLIP 复现,基于 Apple MobileCLIP)。
>
> **交互版(带图表,可分享 Artifact):** https://claude.ai/code/artifact/4e34bc69-c8bf-46a6-ade1-0f11b23c8288
>
> 数据截至 2026-07 · 由 5 个并行研究智能体基于一手来源(arXiv / 实验室博客 / Hugging Face)检索并交叉验证。

**读数须知:** 下文绝大多数指标为**论文 / 官方模型卡自报数**(author-reported),几乎没有独立第三方复现;跨模型比较需谨慎(评测协议、prompt、分辨率、数据划分常不一致)。已知易误引之处见第 9 节。

---

## 1 · 执行摘要:五条主线

2025 年,CLIP 家族已从"单纯的对比学习编码器"演变为**多目标训练的通用视觉骨干**。五条趋势同时推进:

1. **编码器沦为 VLM 骨干** —— 研究重心从"训练编码器"转向"被大模型冻结复用"。零样本 ImageNet 退居次位,下游 VQA/OCR 才是主战场。**SigLIP 2 已是 2025 事实上的默认多模态骨干**(Qwen3-VL、Gemma 3)。
2. **零样本 ImageNet 饱和** —— 开源前沿卡在 **~85%**;CoCa 于 2022 创下的 **86.3%**(闭源)至今未被零样本超越。年增益从 +10 分掉到不足 1 分。
3. **世界级数据,无"英语税"** —— MetaCLIP 2 证明:从零在 300+ 语言全球数据上训练,反而**提升**英语零样本(81.3% > 80.5%),打破"多语言诅咒"。
4. **训练目标"大杂烩化"** —— 对比损失叠加字幕生成、自蒸馏、掩码预测;更激进者(AIMv2、OpenVision 2)**彻底弃对比,只用自回归/字幕生成**。
5. **蒸馏即数据 · 端侧爆发** —— Apple 把教师嵌入+合成字幕**离线烘焙进数据集**;DeepMind 证明"主动筛选=隐式蒸馏"。~30–300M 小模型已能在手机上跑到 77–82% 零样本 ImageNet。

---

## 2 · 核心论点:端侧正在逼近前沿

在一部 2020 年的手机(iPhone 12 Pro Max)上,MobileCLIP2 用 ≤27 ms 的图+文编码延迟,把零样本 ImageNet 推到 **81.9%** —— 距离需要 1–2B/18B 参数或服务器 GPU 才能达到的 ~85% 开源前沿,仅差约 3 分。

**精度–延迟帕累托(端侧图+文编码延迟,iPhone 12 Pro Max):**

| 变体 | MobileCLIP v1 精度 @ 延迟 | MobileCLIP2 精度 @ 延迟 |
|---|---|---|
| S0 | 67.8% @ 3.1 ms | **71.5%** @ 4.8 ms |
| S2 | 74.4% @ 6.9 ms | **77.2%** @ 6.9 ms |
| B  | 76.8% @ 13.7 ms | **79.4%** @ 13.7 ms |
| S3 | 78.3% @ 14.6 ms | **80.7%** @ 14.6 ms |
| S4 | 79.4% @ 26.2 ms | **81.9%** @ 26.2 ms |

前沿天花板参照(服务器/GPU,无端侧延迟):CoCa 2022 = **86.3%**(闭源,天花板);2025 开源前沿 ≈ **85%**(Perception Encoder-G / SigLIP 2-g)。

> **一句话:**「多模态强化训练」把一个大 CLIP 集成的知识离线蒸馏进小学生模型,让精度/延迟帕累托曲线大幅平移 —— 这正是本仓库(MobileCLIP / MobileViCLIP)路线在 2025 年被验证成功的核心。

---

## 3 · 前沿通用图文 CLIP

2025 前沿由四家定义:**SigLIP 2**(多语言 sigmoid + NaFlex)、**Perception Encoder**(中间层特征最优)、**MetaCLIP 2**(全球无过滤)、**AIMv2**(自回归)。

**2025 零样本 ImageNet-1k 排行**(自报;各模型都在 79–86% 窄带内):

| 模型 | 机构 | 时间 | 零样本 IN-1k | 关键思想 |
|---|---|---|---:|---|
| CoCa *(2022 闭源,参照)* | Google | 2022 | 86.3 | 对比+字幕,JFT-3B,至今未被零样本超越 |
| Perception Encoder-G | Meta FAIR | 2025.04 | **85.4** | 「最好的特征不在输出层」,中间层 + 语言/空间对齐 |
| SigLIP 2-g | Google DeepMind | 2025.02 | 84.5 | Sigmoid + 字幕 + 自蒸馏 + 在线筛选;多语言;NaFlex |
| DFN5B-CLIP-H/14 | Apple | 2023→24 | 84.4 | 小型数据过滤网络筛选数十亿对(MobileCLIP2 数据引擎) |
| EVA-CLIP-18B | BAAI | 2024.02 | 83.8 | 最大开源 CLIP,纯公开 2B 数据暴力扩参 |
| SigLIP 2-So400m | Google DeepMind | 2025.02 | 83.2 | — |
| **MobileCLIP2-S4** | **Apple** | **2025.08** | **81.9** | **端侧**逼近前沿的高效学生模型(见 §4) |
| MetaCLIP 2-H/14 | Meta FAIR | 2025.07 | 81.3 | 全球 300+ 语言「无过滤」,破多语言诅咒 |
| **MobileCLIP2-B** | **Apple** | **2025.08** | **79.4** | **端侧** |
| SigLIP 2-B | Google DeepMind | 2025.02 | 79.1 | 轻量多语言 |

其他要角:**AIMv2-3B**(Apple,2024.11,自回归;89.5% 是**冻结探针**非零样本,零样本 LiT ~77%);**OpenVision / v2**(UCSC-VLAA,全开源,v2 弃文本塔纯字幕生成);**TULIP**(UC Berkeley,补细粒度短板);**LLM2CLIP**(微软,用 LLM 当文本教师)。

---

## 4 · 移动端 / 高效 CLIP(与本仓库最相关)

**MobileCLIP2**(Apple,TMLR 2025.08,arXiv 2508.20691)是本仓库骨干的**直接继任者**。架构家族沿用 MobileCLIP 的 MCi 混合卷积-Transformer 可重参编码器;升级集中在**数据侧**。

| 变体 | 零样本 IN-1k | 延迟(图+文) | 参数(图+文) | 说明 |
|---|---:|---:|---:|---|
| MobileCLIP2-S0 | 71.5 | ~4.8 ms | 11.4+63.4 M | 比 v1-S0(67.8)+3.7 分 |
| MobileCLIP2-S2 | 77.2 | 6.9 ms | 35.7+63.4 M | — |
| MobileCLIP2-B  | 79.4 | 13.7 ms | 86.3+63.4 M | 比 v1-B +2.2 分 |
| MobileCLIP2-S3 | 80.7 | 14.6 ms | 125+124 M | — |
| MobileCLIP2-S4 | 81.9 | 26.2 ms | 322+124 M | ≈ SigLIP-SO400M 精度,参数少 ~2×;比 DFN ViT-L/14 延迟低 ~2.5× |

### 深挖 A · MobileCLIP2 的 DFNDR 数据强化配方

MobileCLIP2 的精度红利几乎全部来自**数据侧的离线蒸馏**,而非架构 —— 核心是把 v1 的 **DataCompDR** 换成更强的 **DFNDR**(DFN 数据强化)。所谓"强化",是把教师模型的嵌入与合成字幕**离线预算、烘焙进数据集**,训练时当作廉价的知识蒸馏回放。

| 维度 | DataCompDR · MobileCLIP v1 | DFNDR · MobileCLIP2 |
|---|---|---|
| 基底数据 | DataComp | DFN-2B(更干净的过滤数据) |
| 合成字幕器 | CoCa ViT-L/14 | CoCa ViT-L/14(DFN-2B 预训练 + MSCOCO-38k 微调) |
| 每样本合成字幕 | 5 条 | 5 条 |
| 教师集成 | ViT-L/14(DataComp-13B)+ OpenAI ViT-L/14 | DFN2B-CLIP-ViT-L/14-s39b(logit 70.0)+ DFN2B-CLIP-ViT-L/14(logit 60.0) |
| 图像增强 | 30×(1B)/ 10×(12M) | 30× |
| 存储量级 | DataCompDR-1B ~140 TB(嵌入 bf16) | DFN-2B 规模(嵌入 bf16 存储) |

论文点名的三个关键杠杆:

- **教师集成配对**(logit scale 70.0 + 60.0)带来 **+2.8% ImageNet**(相对最佳单教师);
- **对比蒸馏中的温度调优**被强调为核心可调项;
- **字幕器微调**(MSCOCO-38k 默认,DOCCI 略优 +0.8%)提升字幕多样性,且**多模型字幕组合是可加的**。

净效果:MobileCLIP2-B 在架构与延迟不变下,比 v1-B **+2.2 分**(76.8 → 79.4)。

### 深挖 B · ACED ⟷ MobileCLIP:两种蒸馏哲学

同样是"把大 CLIP 塞进小模型",Apple 与 Google DeepMind 走了**相反**的路 —— 一个**离线显式**存嵌入,一个**在线隐式**靠数据筛选。

| 维度 | MobileCLIP · 离线 / 显式 | ACED · 在线 / 隐式 |
|---|---|---|
| 机制 | 预算教师嵌入 + 合成字幕,**烘焙进数据集**,训练时廉价回放 | 按"可学习性"对**整批**在线选样,对齐大型**冻结参考模型** |
| 核心主张 | 存一次,训练极省;可复现 | "**主动筛选 = 隐式蒸馏**",无需额外损失项即胜过显式 KD |
| 存储 | 巨大(~140 TB / 1B 对) | 无需存嵌入 |
| 训练额外开销 | 低(读预算嵌入) | 参考模型在线推理 |
| 代表结果 | MobileCLIP2-S4 **81.9%**(端侧) | ACED-F0 **68.5%** @3.3 GFLOPs;F1 74.9;F2 76.9 |

> **裁决?尚无定论。** ACED(DeepMind,CVPR 2025,arXiv 2411.18674)自报在 **27 任务均值**上以更低 FLOP 超过 MobileCLIP-S0/S1/S2(**注意:是 27 任务均值,非 ImageNet 单项**)。两条路线也并非对立 —— ACED 本身 = 主动筛选(ACID)+ **轻量**显式 KD,JEST 是其前身。选择取决于瓶颈:**存储/可复现优先 → 离线(MobileCLIP 路线)**;**存储受限、愿付在线算力 → 在线(ACED 路线)**。对本仓库(MobileViCLIP 复现)而言,MobileCLIP 的离线强化数据是现成、可复现的起点。

旁支:**FastVLM / FastViTHD**(Apple,CVPR 2025)是同谱系(FastViT/MobileOne 可重参)的端侧 VLM 编码器,高分辨率却输出更少视觉 token,0.5B 版 TTFT 比 LLaVA-OneVision-0.5B 快 ~85×。

---

## 5 · 深度对比:SigLIP 2 ⟷ MobileCLIP2,该怎么选?

两者常被放在一起比较,但**定位正交**:MobileCLIP2 是端侧优先的高效学生模型,SigLIP 2 是服务器级的多语言通用骨干。

**关键发现:** 在 **86M 视觉参数**这一档,MobileCLIP2-B(79.4)与 SigLIP 2-B(79.1)基本打平 —— 但 MobileCLIP2 用的是可重参混合架构,端侧延迟远低于纯 ViT;高参数端两者收敛(SigLIP 2-g 天花板 84.5 更高)。

| 维度 | MobileCLIP2 · Apple | SigLIP 2 · Google DeepMind |
|---|---|---|
| 发布 | 2025.08 · TMLR(2508.20691) | 2025.02 · arXiv 2502.14786 |
| 定位 | 端侧优先,高效图文检索 / 零样本 | 服务器级,多语言通用骨干 / VLM 视觉塔 |
| 图像编码器 | MCi 混合卷积-Transformer(FastViT 谱系,结构重参数化) | 标准 ViT(+ NaFlex 变分辨率) |
| 训练范式 | 多模态强化训练:离线蒸馏(教师集成嵌入 + 合成字幕烘焙进 DFNDR) | 统一配方:sigmoid 对比 + 字幕解码 + 自蒸馏 + 掩码预测 + 在线筛选 |
| 多语言 | 以英语为主 | ✓ 原生 109 语言 |
| 可变分辨率 | 固定(多为 224/256) | ✓ NaFlex 原生长宽比 |
| 稠密 / 空间特征 | 检索/分类导向,一般 | ✓ 自蒸馏带来更强稠密/定位特征 |
| 规模档位 | S0 / S2 / B / S3 / S4(端侧 → 445M) | B 86M / L 303M / So400m 400M / g 1B |
| 零样本 IN-1k | 71.5 → 81.9 | 79.1 → 84.5 |
| 端侧延迟 | ✓ 4.8–26 ms(iPhone 官方实测) | ✗ 无官方端侧延迟(面向 GPU) |
| 部署 / 生态 | Core ML / 手机 / 边缘 | GPU 服务器;Qwen3-VL、Gemma 3 视觉塔 |

**选型建议:**

| 需求 | 推荐 |
|---|---|
| 📱 端侧 / 手机 / 隐私 / 低延迟 | **MobileCLIP2**(轻量 S0/S2,兼顾精度 B/S4) |
| 🌏 多语言 / 非英语 | **SigLIP 2** 或 MetaCLIP 2 |
| 🧠 作为 VLM / 大模型视觉塔 | **SigLIP 2**(生态最成熟),或 AIMv2 / Perception Encoder |
| 📄 文档 / OCR / 高分辨率 | **SigLIP 2 NaFlex** |
| 🎬 端侧视频检索 | **MobileViCLIP**(直接基于 MobileCLIP,见 §6) |
| ⚖️ 精度天花板优先 | **SigLIP 2-g / Perception Encoder-G**(~85%) |

> **结论:** 不是"谁更强",而是"部署在哪"。设备上 → MobileCLIP2;服务器 / 多语言 / VLM 骨干 → SigLIP 2。在 86M 视觉参数一档两者精度打平,MobileCLIP2 的护城河是**端侧延迟**。

---

## 6 · 端侧视频 CLIP 深挖:MobileViCLIP(即本仓库)

**MobileViCLIP**(ICCV 2025,arXiv 2508.07312,南京大学 / 蚂蚁 / 上海 AI Lab)是 2025 年——大概也是唯一——真正可在手机上运行的视频-CLIP,而且它**直接以 Apple MobileCLIP 为骨干**。

**视频-文本检索 MSR-VTT 零样本 Text→Video R@1:**

| 模型(端侧标注) | 机构 | 时间 | MSR-VTT 零样本 R@1 | K400 零样本 |
|---|---|---|---:|---:|
| InternVideo2-6B | OpenGVLab | 2024.03 | 55.9 | 72.7 |
| VideoPrism-g | Google | 2024 / 检索权重 25.07 放出 | 52.7 | 76.4 |
| Perception Encoder-G | Meta | 2025.04 | 51.2 | 76.9 |
| VideoCLIP-XL | 华南理工 / 阿里 | 2024.10 | 50.1 | — |
| **MobileViCLIP-Small** 🔹端侧 | 南大 / 蚂蚁 / 上海AI | 2025.08 | 42.5 | 63.1 |
| **MobileViCLIP-Tiny** 🔹端侧 | 南大 / 蚂蚁 / 上海AI | 2025.08 | 38.7 | 56.2 |

**关键设计:**

- **时间维结构重参数化** —— 训练时建立时序结构,推理时**折叠掉**(沿用 MobileCLIP 的 RepMixer 思路),几乎零额外推理开销;在 InternVid-10M-FLT 上训练。
- **骨干 = MobileCLIP** —— Tiny 基于 MobileCLIP-S0,Small 基于 MobileCLIP-S2。
- **端侧规格** —— Tiny 53.9M / ~12.4 ms;Small 99.4M / ~37.8 ms(iPad Air 2020)。Small **比 InternVideo2-L14 快约 55×** 而检索质量相近,比 InternVideo2-S14 高 +6.9% 且快 6.7×。

> **对本仓库的启示:** MobileViCLIP 提供了把 MobileCLIP-S0/S2 低成本"视频化"用于端侧视频检索的现成范式 —— 也正是本仓库复现与改进的目标。InternVideo2-6B(55.9 / 微调 62.8)是精度天花板参照。

---

## 7 · 视频全景 · 数据与训练方法

**视频侧:** InternVideo2-6B 仍是开源检索天花板;**InternVideo2.5**(2025.01)是视频 **MLLM**,用长富上下文(LRC)把处理帧数扩到 ~10k 帧。两种时序哲学:显式时间注意力 vs Perception Encoder 的"数据胜过架构"(帧平均池化 + 对齐良好的合成字幕即可媲美)。

**数据 / 训练方法:**

- **数据筛选:** DFN、MetaCLIP(质胜于量)、JEST(算力省 ~10×)、ACID/ACED(筛选=蒸馏)。
- **合成字幕:** CLIPS 的"反效应"—— **短**字幕优于完整长字幕;Recap-DataComp 用 LLaMA-3 重写 13 亿图;LaCLIP / VeCLIP / DreamLIP。
- **训练目标:** sigmoid 损失(省显存,batch 解耦)、纯生成式(OpenVision 2)、掩码(FLIP,>3× 提速)、CLIPA(反尺度律,$10k 训到 81.1%)、NeuCLIP(小 batch 归一化优化)。

---

## 8 · 2025 关键发布时间线

| 时间 | 事件 |
|---|---|
| 2025.01 | **InternVideo2.5** —— 视频 MLLM,长富上下文(~10k 帧)· OpenGVLab |
| 2025.02 | **SigLIP 2** —— 多语言 sigmoid + NaFlex,成 2025 默认 VLM 骨干 · Google DeepMind |
| 2025.03 | **TULIP** —— 对比 + 重建 + 生成增强,补细粒度短板 · UC Berkeley |
| 2025.04 | **Perception Encoder** —— 中间层特征最优,图/视频/稠密 SOTA · Meta FAIR |
| 2025.05 | **OpenVision** —— 全开源编码器族(5.9M–632M)· UCSC-VLAA |
| 2025.07 | **MetaCLIP 2** 全球无过滤 · **VideoPrism** 检索权重放出 · Meta / Google |
| 2025.08 | **MobileCLIP2**(TMLR)· **MobileViCLIP**(ICCV)—— 端侧图文与视频双双落地 · Apple / 南大 |
| 2025.09 | **OpenVision 2** —— 弃文本塔与对比损失,纯字幕生成 · UCSC-VLAA |
| 2025.11 | **NeuCLIP** 小 batch 归一化优化 · **D4C** 数据无关量化 |

---

## 9 · 重要提醒:读数与纠正

- **EVA-CLIP-18B = 83.8%**:真实零样本 IN-1k 是 83.8%;常被误引的 80.7% 实为 27 数据集平均值。
- **AIMv2 的 89.5% ≠ 零样本**:那是冻结骨干 + attentive probe;其零样本(LiT)约 77%。
- **视频数字不可直接比**:MSR-VTT(1K-A vs full)、Kinetics 的 prompt/协议跨论文不一致,仅近似。例如"InternVideo2(72.7)< ViCLIP(75.7)"实为评测协议差异所致。
- **SigLIP 2 检索 R@1**:自动抽取曾出现错位,检索数应视为近似;其 ImageNet 数可靠。
- **DatologyAI 省算力宣称**:公司博客、无论文无权重,属营销口径,本报告未采信。
- 全部头部数字均为**自报**,未见独立第三方审计;OpenCLIP / CLIP_benchmark 是常用交叉校验,量级排序与上表大体吻合。CoCa 的 86.3% 基于闭源 JFT-3B,未被复现,仅作天花板参照。

---

## 10 · 参考文献(arXiv 编号)

**前沿图文编码器**
SigLIP 2 `2502.14786` · MetaCLIP 2 `2507.22062` · Perception Encoder `2504.13181` · AIMv2 `2411.14402` · EVA-CLIP-18B `2402.04252` · OpenVision `2505.04601` / v2 `2509.01644` · TULIP `2503.15485` · LLM2CLIP `2411.04997`

**移动端 / 高效**
MobileCLIP2 `2508.20691` · MobileCLIP `2311.17049` · ACID/ACED `2411.18674` · TinyCLIP `2309.12314` · FastVLM `2412.13303` · D4C 量化 `2511.15411`

**视频**
InternVideo2 `2403.15377` · InternVideo2.5 `2501.12386` · **MobileViCLIP `2508.07312`** · VideoPrism `2402.13217` · VideoCLIP-XL `2410.00741`

**数据 / 训练**
DFN `2309.17425` · JEST `2406.17711` · CLIPS `2411.16828` · Recap-DataComp `2406.08478` · SigLIP `2303.15343` · NeuCLIP `2511.08417`

**综述**
VLM 2.6 万篇分析 `2510.09586` · 高效 VLM 综述 `2504.09724`

---

*本报告由 5 个并行研究智能体基于一手来源检索与交叉验证生成。数值均为作者自报,非独立复现;跨模型对比仅供参考。生成日期 2026-07-12。*
