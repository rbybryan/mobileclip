# 计划评审与升级提案(2026-07-23)

**问题:** 全面评审当前模型改进计划——C1–C4 主线、卫生闸门、条件项(R4/P9)、
数据方案(`data-quality-caprl-scheme.md`)、以及 #197 新增的神经启发实现计划
(`neural-inspired-implementation.md`)——并回答「有没有更好且可行的改法」。
**一句话结论:** 计划的**方向、门控序列与实验纪律都站得住,不需要推倒重来**;
真正的缺口在 (a) **三处风险的时间排布**(P9 导出、BN×弹性 T、预算阶梯),
(b) **一条缺失的新颖性防线**(弹性推理 prior-art 轴),(c) 若干文档间不一致。
本文给出升级提案 **U1–U10**,全部落进现有 R 序列与 ROADMAP 三线,**不新开战线**。
**分工:** 「能不能发」见 [`publishability-assessment.md`](publishability-assessment.md);
「现在轮到哪步」见 [`../experiments/ROADMAP.md`](../experiments/ROADMAP.md);
本文回答「计划本身还有什么问题、怎么改」。

---

## 0. 评审范围(都核了什么)

| 对象 | 内容 |
|---|---|
| 计划文档 | `paper-plan(-zh).md`(含 §六 9 项红队修正)、`ROADMAP.md`、`README.md` 两级实验/§2b/§4、`SETUP_B100.md`、`data-quality-caprl-scheme.md`、`neural-inspired-implementation.md`(#197)、`whitespace-directions.md`、`model-architecture-improvements(-zh).md`、`pilot_10k_results.md`、`publishability-assessment.md` |
| 代码抽查 | `models/mobileviclip_small.py`(norm_1d=MobileOneBlock_1d 内含 BN、FastViT `norm_layer=nn.BatchNorm2d`、`temporal_embedding (1,C,8)` 零初始化、reparameterize 折叠路径)、`scripts/pretraining/clip/mobileviclip_small/config.py`(`freeze_vision=False / freeze_text=True`)、`tasks_clip/pretrain.py`(无 EMA) |
| 外部文献 | 2024–2026 弹性推理 / 帧数弹性 / 高效视频-文本 网扫(§3) |

## 1. 计划里做对了的(保持,不动)

1. **两级验证**(V100/10k 只筛链路、H100/全量才排序)——职责划分正确且已写透。
2. **门控序列 R0→R1→R2→R3(→R3b)** + 每级 go/no-go 判据 + 失败后备(效率故事)。
3. **组件开关纪律**:默认关闭、off-path 逐位等价、零初始化 step-0==基线——#197 也继承了。
4. **§六 9 项红队修正**质量高:caption_idx 空转、t/b 学习率、raw 主口径、R3b 对照、
   判据加严、R2 三个设计细节,都是真问题真修法。
5. **砍掉清单**明确(P6/P8/量化/流式),防散焦。
6. **诚实的自评**(`publishability-assessment.md`):零主实验结果 ⇒ 一切待验,不自嗨。

⇒ 下面所有升级都是**在这套骨架上打补丁**,没有一条要求改变主线故事或门控结构。

## 2. 发现的问题(按严重度)

### 2.1 高危 · P9 导出验证排在第 8 周,但它根本不需要训练好的权重

`whitespace-directions.md §0` 已核实:仓库唯一导出脚本(`models/convert_model.py`)
**静默跳过全部时序算子**,联合 T·H·W 注意力 + 硬编码 reshape + 时序 Conv1d 都是疑似
导出阻塞点;而 `paper-plan-zh.md` 把导出决策点放在**第 8 周**。问题在于:**算子覆盖率
/可转换性测试只需要架构 + 随机权重**,不需要任何训练产物。按现排期,若第 8 周才发现
联合注意力导不出,C4 延迟表(发表必要条件之一,见 `publishability-assessment.md` §2)
和 P9 变体的微调都被压进截稿前最后几周。**这是纯时间排布错误,零成本可修。**→ U1。

### 2.2 高危 · C2 有一个代码级坐实的隐藏失败模式:BN 统计随 T 漂移

代码事实链(2026-07-23 核对):

- 时序分支遍布**所有** RepMixer 块:`x + layer_scale_1d*(mixer_1d(x) − norm_1d(x))`,
  其中 `mixer_1d/norm_1d` 都是 `MobileOneBlock_1d`——**内含 BatchNorm**(重参数化时折叠);
- FastViT 主干 `norm_layer` 默认 `nn.BatchNorm2d`,且预训练配置 `freeze_vision=False`
  ⇒ 全塔可训,**BN running stats 一定随训练更新**;
- 时序 conv(k=3)在 T=2 时 100% 位置是 padding 边界、T=16 时只有 ~12%,且时序混合
  发生在每个 block ⇒ **下游所有 BN 看到的特征分布随 T 变**。

随机 T 训练下,BN running stats 收敛为「跨 T 的混合物」;推理固定某个 T 时统计失配。
这正是 Frame Flexible Network(CVPR 2023)命名的 **temporal frequency deviation**——
他们的结论是:不处理规范化统计,「一模型多帧数」在低 T 端**必然掉点**;slimmable
networks(ICLR 2019)在宽度弹性上给出同款修法(switchable BN)。**C2 的头条主张
(单 ckpt 每个 T 点 ≥ 固定 T 基线)直接暴露在这个失败模式下,而全部计划文档无一提及。**
额外一环:`reparameterize()` 用 running stats 折叠权重 ⇒ per-T 统计自然给出 per-T
折叠权重,与「按 T 导出静态图」的部署形态天然兼容,不损害「单一训练权重」主张
(slimmable 先例)。→ U3。

### 2.3 中高危 · 新颖性防线缺「弹性推理」这条 prior-art 轴

`paper-plan-zh.md` §二的 novelty defense 只预判了「这不就是视频版 MobileCLIP2」;
§六.9 的投稿前文献复扫也只覆盖「首个视频 DR」。但 C2 的另一半身位落在**弹性推理**
文献里:Frame Flexible Network(CVPR'23,监督动作识别的一模型多帧数)、slimmable/
US-Nets(宽度弹性 + switchable BN + sandwich rule)、FlexiViT(patch 弹性)、
Matryoshka 系(MRL/MatFormer/M3)、LiteFrame 2026(video-LLM 帧数缩放,
`model-architecture-improvements.md` P2 行已引)。审稿人从这条轴发难的概率不低于
「视频版 MobileCLIP2」。且网扫证实这条轴上**确有一篇强邻**(FluxViT,ICCV 2025:
弹性采样 × InternVideo2-1B 蒸馏 × 多预算零样本检索)——C2 的可声称空间因此收窄为
**帧数旋钮 × 移动端卷积混合双编码器 × 固定 T 锚点蒸馏 × 端侧评测**的组合格。
详细核对与四条差异化轴见 §3。→ U2(动机图实证)+ 写作层补丁(§6)。

### 2.4 中危 · 运行矩阵在膨胀,而 4×H100 的时钟没有量化预案

现在排进全量 A/B 的臂:R0、R1a/R1b/R1、R2(+#197 B1 的 cos/rkd/geo 三 KD 子臂)、
R3、R3b,再加 #197 Tier-A 的 A1/A2 ——以单臂 3 epoch 计,**总量 ≈ 33 个「单 epoch
墙钟 E」**(不含教师推理与评测)。E 在首日实测前未知:E≈12h 则 ~17 天可承受;
E≈24h 则 ~33 天,叠加数据审查/教师 pass/写作,对 11 月截稿已经危险;E≈48h 直接
不可行。现有预案只有「裁臂顺序 R0>R2>R3>R3b>R1」,**缺「臂内缩预算」这个更细的旋钮**
(排序臂降为 1 epoch、只有 R0 与胜者终训跑满 3 epoch ⇒ 总量近乎减半,而配方排序在
同架构/同数据下通常保序)。→ U6。

### 2.5 中危 · R0 复现保真度:默认 4 卡 ⇒ 全局 batch 256 ≠ 上游 512

批内负例数 = 对比任务本身。`SETUP_B100.md` §6 已把「4×128=512」列为备选;考虑到
H100-80G 显存充裕(V100 峰值 23.2 GiB @bs=64 ⇒ bs=128 估算 ~46-50 GiB,bf16 同量级),
**建议把 4×128=512 升为 R0 的默认推荐**(首光实测显存/吞吐后定),否则 R0 与上游
论文数字之间永远隔着一个「batch 不同」的解释变量。无论选哪个,全臂一致纪律不变。→ U5。

### 2.6 低危 · 文档间不一致(5 条,读者会踩)

| # | 位置 | 问题 | 处置 |
|---|---|---|---|
| 1 | `paper-plan-zh.md` §一/§四 | 算力写「8×V100」,实际主战场已是 4×H100(`SETUP_B100.md`) | §七补丁声明以 SETUP_B100 为准 |
| 2 | `data-quality-caprl-scheme.md` §0/§6.3 | 称「视觉塔仅时序分支+projector 可训」;实际预训练配置 `freeze_vision=False` ⇒ **视觉塔全量可训**(该描述只适用于 `freeze_vision=True` 的微调模式)。影响:视频侧 KD 的梯度受体是**整塔**,`uta`/KD 的预期效力比 §6.3 写的更强(文本侧结论不变) | 原文加勘误注 |
| 3 | `model-architecture-improvements*.md` P2 | T 集合写 {4,8,12,16},计划为 {2,4,8,16} | 以 paper-plan 为准,不改旧笔记 |
| 4 | C1 教师语料 | `paper-plan` 写全量 10M,`data-quality` §6 只在 curated 2–3M 上烘焙 | 真决策,见 U7 |
| 5 | 相似度矩阵 KD 双所有权 | `data-quality` §6.2 的 `vtc_kd`(DFNDR 式 KL)与 #197 B1 的 `kd_geo`(Tung-Mori Gram MSE)是**近亲**,分属两文档、两命名 | 合并进同一消融表,见 U9 |

### 2.7 低危 · 头条主张缺不确定度口径

§六.7 的判据(多基准 R-Sum 同向 + 临界补种子)管**组件采纳**;但 C2 的头条是
「**每个 T 点** ≥ 固定 T 基线」——逐点主张对噪声更敏感,而 MSR-VTT 1k 上
1000 查询的二项波动 ~1.3pp ⚠️ **数值更正(2026-07-29):应为 ≈1.6pp**。R@1≈45% 时
`sqrt(0.45·0.55/1000) = 1.57pp`;1.3pp 对应的是 p≈0.2,不是我们的工作点。
`neural-inspired-round2.md` §1.1 用的 ≈1.6 是对的,以它为准。**这不是笔误级小事**:
U10-3 预注册的非劣边际 δ=0.3 与之相差约 8×(ROADMAP **G2**,🔴),而 1M 层实测
更进一步显示「10× 数据」这种巨大处理效应也只值 +0.82 r_mean、落在带内 ⇒ **单纯放大 n
救不了,必须走配对 bootstrap**(U10-1,ROADMAP **G4** 已升 🔴)。
缺:评测期 bootstrap CI、per-T 非劣性边际的预注册。→ U10、§6.1。

## 3. 文献核对(2026-07-23 网扫,~15 组检索词覆盖 2024–2026H1)

### 3.1 C2 kill-check:没被做掉,但新颖性边距只剩「一篇论文宽」——FluxViT

**FluxViT** — *Make Your Training Flexible: Towards Deployment-Efficient Video Models*,
OpenGVLab/南大,**ICCV 2025**,[arXiv:2503.14237](https://arxiv.org/abs/2503.14237),
代码已开源。已核实:训练用**弹性采样网格**(帧数 4–24、分辨率 168–280)+ **token
选择器**,多 token 预算(2048/1024/512)混训;**从 InternVideo2-1B 蒸馏**(=我们
C1 的同一个教师);含 CLIP 式视频-文本对比阶段,**单 checkpoint 在多个 token 预算下
报零样本 T2V 检索**(MSR-VTT/DiDeMo/ANet/LSMDC;FluxViT-B @2048 tokens MSR-VTT
R@1=49.9);FluxViT-S=24M / B=97M;只报 GFLOPs,**无端侧延迟**;论文明确批评 FFN
低帧数掉点严重。

⇒ 「弹性推理预算 × 大教师蒸馏 × 零样本检索」**这个组合已存在**。paper-plan §一
「没有任何已有工作把 DR 式离线强化与帧数弹性结合起来」的说法**已过时,必须改写**。
C2 存活靠四条(都以 FluxViT 已核实细节为锚):

1. **旋钮语义**:FluxViT 的预算旋钮是 *token 数*——端侧仍要**解码稠密帧网格**再过
   选择器;C2 的旋钮是*帧数本身*:少解码、少编码、无选择器,才是端侧诚实旋钮。
2. **骨干类别**:token 选择预设 ViT token 流;MobileCLIP-S2 的 FastViT 卷积混合
   前段**没有可丢的 token 流**,该机制在移动端骨干类上不可用——帧数弹性是这一类
   骨干的自然机制。FluxViT 无移动端故事;我们的全部前提就是端侧。
3. **蒸馏几何**:Flux-UMT 的教师目标随采样网格移动;C2 把**所有 T∈{2,4,8,16} 对齐
   同一份冻结 T=16 锚点**——跨 T 一致性目标 + 外部冻结基础教师 + embedding 空间,
   这个具体组合未见发表。
4. **C1 离线强化**:DataCompDR 式存储教师嵌入 + 合成重字幕 vs FluxViT 的在线蒸馏。

**代价:** FluxViT-S(24M)必须进 C4 基准表(GFLOPs 对齐行,注明其无延迟数据);
理想情况在端侧指标上正面赢它。**若论文说不清对 FluxViT-S 的胜点,C2 框架塌方。**

其他近邻(均非 kill):StretchySnake(arXiv:2510.16209,SSM 多时空尺度,监督动作
识别、无语言无教师);Qwen2-VL 系动态帧数(生成式 VLM);M3/MQT-LLaVA(token 弹性,
生成式);帧选择线(FFS/GenS/Q-Frame 等——选喂哪些帧给固定模型,与「模型本身帧数
弹性」正交)。Semantic Scholar 上 MobileViCLIP 尚无相关引用工作、MCG-NJU 无后继。

### 3.2 相邻弹性谱系(必须引用并逐一区分)

| 工作 | 一句话 | C2 差异 |
|---|---|---|
| FFN(CVPR'23,2303.14817) | 命名 temporal frequency deviation;监督动作识别一模型多帧数,跨帧数特征对齐 + **per-频率专用组件** | 对比式视频-文本、零样本迁移、外部冻结教师固定 T=16 锚点、无 per-T 专用参数 |
| Slimmable/US-Nets(1812.08928/1903.05134) | 宽度弹性;switchable BN、sandwich rule、inplace 蒸馏 | 弹性轴=时间输入长度;教师=外部固定模型非自身最大配置(U3/U4 反向借它的技术) |
| Once-for-All(1908.09791) | 深/宽/核/分辨率渐进收缩 + 自蒸馏 | 同上;无时间轴无文本 |
| FlexiViT(CVPR'23,2212.08013) | patch 尺寸弹性,**且已用固定强教师蒸馏增强** | 空间 patch 轴 → 我们是时间帧轴 + 视频基础教师 + 零样本检索 |
| MRL/MatFormer/M3/MQT-LLaVA | 嵌入维/FFN 宽/视觉 token 数嵌套 | 输入/算力弹性 vs 输出/结构嵌套;P10 已把 MRL 作彩蛋收编 |
| ElasticTok(ICLR'25,2410.08368) | 自编码器逐帧自适应 token 数 | 重建式 tokenizer vs 判别式双编码器 |
| **FluxViT(ICCV'25,2503.14237)** | 见 §3.1 | 见 §3.1 四条 |

**诚实点(写作红线):** FlexiViT 与 FluxViT **已经**做了「弹性 × 固定教师蒸馏」——
C2 的新颖性只能落在**组合格**上(帧数轴 × 移动端卷积混合双编码器 × 固定 T 锚点 ×
端侧评测),**永远不要声称「弹性蒸馏」本身是新的**。

### 3.3 动机图证据(U2 的文献地基)

- FFN 的 TFD 是规范引用(监督模型换帧数即崩);FluxViT 论文可引其「FFN 两帧即大幅
  退化」的复述;CLIP4Clip Fig 3(d) 有帧数消融(但那是「训测同帧数」);
- **「固定 T=16 训练 → T=2/4/8 零样本评测」的失配退化曲线,在视频-文本模型上
  未见发表** ⇒ U2 的曲线本身就是一个小贡献;
- **必须预防的反向审稿意见**:*Revealing Single Frame Bias*(ACL'23,2206.03428)——
  MSR-VTT/DiDeMo/ANet 静态偏置重,单帧模型都能打;应对:(a) 这恰说明弹性模型在
  这些集上低 T 端有高天花板可够,(b) **SSv2-MC(时序重)必须在评测集里**展示
  T 真正重要的地方——现计划已含 SSv2-MC,保住别裁。

### 3.4 MobileCLIP2-S2 初始化:确认可行(U8 GO)

TMLR 2025(2508.20691),HF 已放 **apple/MobileCLIP2-S0/S2/B/S3/L-14**。S2↔S2 架构
一致的三重证据:参数量逐塔相同(35.7M+63.4M)、HF 卡用法直接实例化 `MobileCLIP-S2`
OpenCLIP 配置、论文只把 S3/S4/L-14 列为新架构。源头收益 IN-1k +2.8(77.2 vs 74.4)。
**⇒ 学生初始化直接换血安全**;截至 2026-07 无更新的移动端图像 CLIP 取代它。

### 3.5 TempMe 口径:微调协议,不能进零样本表(C4 表设计修正)

TempMe(ICLR'25)是**逐数据集微调**方法(PEFT 0.50M 或全微调,CLIP4Clip 式协议),
论文**无零样本评测**。⇒ C4 里它只能进 (a) 单独的微调对比表(若我们也做一次
per-dataset 微调)或 (b) 延迟/GFLOPs-only 行;混进零样本表是 desk-reject 级错误。
2026 的 token 压缩后继都转向 video-LLM(InfoMerge/TTF/CVPR'26 统一时空压缩),
双编码器检索位上 TempMe 仍是正确基线。

### 3.6 CapRL 确认(C1 字幕通道可行性 GO)

CapRL(ICLR'26,2509.22647)**仅图像**;其视频后继 **CapRL++**(2606.09393,
2026-06,同组)已放出 **CapRL-Video-4B**——`data-quality-caprl-scheme.md` §4.2
路径 B0 引用的正是它,方案成立。备选:Tarsier2-Recap-7B(百万级重字幕的成熟马)、
Qwen2.5-VL-7B。吞吐粗估(vLLM,3–4B,8 帧/片,~100–150 输出 token):
5–10 clips/s/GPU ⇒ **10M 片 ≈ 3–6 天 @4×H100**(7B 翻倍);重写一次、离线存储,
成本摊到所有臂。

### 3.7 竞争格局与 C1 定位修正

- 截至 2026-07 **无移动端视频-文本后继者**发表,赛道仍空,但 CVPR 2027 窗口真实;
- InternVideo 谱系(2.5 / Next / 3)都不是移动端双编码器,引用作背景即可;
- **C1 定位红线**:OpenGVLab 仓库**已发布** InternVideo2-S/B/L(从 1B 蒸馏)以及
  MobileCLIP 骨干的小型 VideoCLIP(仓库放品、无专文)——C1 必须引用该系列,并把
  新颖性钉在**离线强化数据配方**(存储多目标嵌入 + 相似度矩阵蒸馏 + RL 字幕器
  重字幕 = DataCompDR 移植到视频),**而不是**「把 InternVideo2 蒸进 MobileCLIP」
  ——后者 OpenGVLab 事实上已做。

### 3.8 由 §3 直接触发的计划修改(已落 §6)

1. paper-plan novelty defense 增补 FluxViT 四条差异化 + 「弹性×蒸馏已有」红线
   + C1 对 OpenGVLab 蒸馏系列的定位声明;
2. C4 表:加 FluxViT-S 行(GFLOPs 对齐),TempMe 移入微调/延迟-only 协议区;
3. `publishability-assessment.md` §1 C2 行由「真·新」改为「组合新(边距=FluxViT
   一篇宽)」,§2 必要条件加「对 FluxViT-S 有明确胜点」。

## 4. 升级提案 U1–U10(按 收益/成本 排序;全部落进现有结构)

| # | 提案 | 成本 | 落点 |
|---|---|---|---|
| U1 | 导出冒烟**立即做**(随机权重即可) | 零 GPU,1–2 天 | ROADMAP 第一线 1.5 |
| U2 | B0 官方权重的**多 T 退化曲线**(动机图) | 纯评测,V100 可跑 | ROADMAP 第一线 1.6(依赖 2.1) |
| U3 | C2 加 **per-T BN 校准**(默认)/私有 BN(触发) | 校准=纯前向;诊断 CPU 可做 | 2.1 验收 + README §2b 矩阵 |
| U4 | R3 的 **T 采样课程 fallback**(分层采样→sandwich) | 配置级;sandwich ≈1.6×步预算 | R3 变体,先于 R4 触发 |
| U5 | R0 **全局 batch 对齐上游 512**(4×128) | 首光验一次显存 | SETUP_B100 §6 |
| U6 | **预算阶梯**:排序臂 1 epoch、R0+胜者 3 epoch | 决策规则,零成本 | paper-plan §七 + SETUP_B100 |
| U7 | C1 细化:教师语料决策 + per-frame 教师嵌入保险 + ≥2 合成字幕 | 存储 ~45GB/1M 片 | ROADMAP 2.2 设计项 |
| U8 | 便宜标准项:EMA、训练内 held-out 评测修 key、MobileCLIP2-S2 初始化(§3.4 已确认可行) | 各≤半天 | R1+ 配置 |
| U9 | 相对 KD **单一所有权**:`vtc_kd` 与 `kd_geo` 并入同一消融表 ⚠️ **前提有坑:`criterion["loss_weight"]` 是纯日志开关**(round2 §5 P2)——`tasks_clip/pretrain.py:114` 是**无权重求和**,`:49` 只建 meter。合并消融表前必须先把权重落到 `forward()` 里,否则两项都以 1.0 悄悄入损失,消融表读出来的是假的 | 写作/配置整理 | R2 消融设计 |
| U10 | 统计口径:bootstrap CI 进 `collect_results.py`、胜者双种子、per-T 非劣边际预注册 | 评测器改动,半天 | 判据节 |

### U1 · 导出冒烟立即做(P9 决策点从第 8 周拆前一半到现在)

- **做什么:** 用**随机权重**实例化带全部时序算子的 `MobileViCLIP_Small`,分别走
  ONNX(→onnxruntime 加载)/ TFLite / coremltools **转换**,按 T∈{2,4,8,16} 各出一张
  静态图;记录每条路线 pass/fail 与失败算子名。第二档(真机延迟)仍留在第 8 周。
- **为什么:** §2.1;转换性是 P9 触发条件的全部信息量,今天就能拿到。若联合注意力
  确认导不出,P9 分离注意力变体**提前进入 R3 的架构决策视野**(至少要决定:弹性模型
  是否直接用可导出形态训练,避免第 8 周再微调一个部署变体)。
- **验收:** 三条路线各有明确 pass/fail + 失败算子清单,写进
  `notes/live/from-claude.md` 或 TROUBLESHOOTING;触发 P9 与否有了事实依据。
- **哪台机器:** 101vector(CPU 即可转换;`pip install onnx onnxruntime coremltools` 级依赖)。

### U2 · B0 官方权重的多 T 退化曲线(论文 Fig-1 + C2 难度标定)

- **做什么:** 2.1 的 T 无关化代码就绪后,把**官方 `mobileviclip_small.pt`**(固定
  T=8 训练)直接在 T∈{2,4,8,16} 下评 MSR-VTT(`temporal_embedding` 插值 + BN 原样)。
  纯评测,V100 可跑。
- **为什么:** 这就是 FFN 式「固定帧数模型换帧数即崩」的实证——一张图同时完成
  (a) 论文动机(Fig-1),(b) 标定 C2 要跨越的 gap 有多大(若 T=4 只掉 1 点,C2 的
  头条空间就小,早知道早调整),(c) §2.3 差异化的第一份自有证据。**零训练成本**。
- **验收:** 曲线 + 数字进 `experiments/`(独立小节),并回填 paper-plan 动机段。

### U3 · per-T BN 处理(保住 C2 头条主张的关键补丁)

- **默认档(必做):** 训练照旧(随机 T);训练结束后对每个 T∈{2,4,8,16} 用几百个
  batch **只前向刷新 BN running stats**(momentum 复位),得到 4 套 BN buffer;评测
  与导出按 T 取对应 buffer(重参数化折叠自然得到 per-T 折叠权重)。存储开销:每套
  buffer 仅 BN 均值/方差,MB 级。
- **触发档(诊断说话):** 若诊断显示漂移大(见下),升级为 FFN 式 per-T 私有 BN
  affine(训练期即分 T),参数量可忽略。
- **诊断(V100/10k 级,加进 README §2b 矩阵):** 同一 ckpt 分别以纯 T=2 与纯 T=16
  前向若干 batch,量化各层 running_mean/var 的偏移;偏移显著即坐实风险。CPU 可做。
- **判据联动:** R3 的「每 T 点 ≥ 固定 T 基线」改为**校准后口径**;R3b 对照组不校准
  (它本来就是单 T 训练)。写作层把「单 checkpoint」表述为「单一训练权重 + per-T
  BN 校准 buffer」(slimmable 先例,诚实且无损故事)。

### U4 · R3 的 T 采样课程 fallback(先于 R4,便宜一档)

若 R3 低 T 端掉点,现计划直接跳 R4(非对称帧编码,一周级)。中间还有两级更便宜的:
1. **分层 T 采样 + per-T 损失权重**(§六.9 的「低 T KD 加权」已在计划内,落实即可);
2. **sandwich 采样**(US-Nets):每步训 T=2 + T=16 + 一个随机中间 T,锚死 Pareto
   两端。代价 ≈1.6× 步预算(帧数加权),仍远低于 R4。
触发顺序:R3 原版 → (1) → (2) → R4。每级都有明确判据(低 T 端相对 R3b 的差距)。

### U5 · R0 全局 batch 对齐上游 512

见 §2.5。首光冒烟时顺手验 bs=128/卡 的显存与吞吐;可行即全臂 4×128,不可行则
全臂 256 并在论文实现细节里声明与上游的差异。**禁止臂间混用**(既有纪律)。

### U6 · 预算阶梯(把「裁臂」细化为「臂内缩预算」)

首日拿到单 epoch 墙钟 E 后按下表执行,写死不再临场发挥:

| 总预算允许 | R0 | 排序臂(R1×3、R2+KD 子臂、R3、R3b、A1/A2 ⚠️) | 胜者终训 |
|---|---|---|---|
| 宽裕(33E 可承受) | 3 ep | 各 3 ep | —(R3 即终训) |
| 常态 | 3 ep | **各 1 ep(等步严格一致)** | 胜者补满 3 ep |
| 紧张 | 3 ep | 砍臂:A1/A2 与 rkd 子臂降为 10k 链路记录;R1 降短训 | 胜者 3 ep |

依据:同架构、同数据、同超参族内的**配方排序**在缩预算下通常保序(DR/蒸馏文献惯例);
风险(排序翻转)由「胜者终训后与 R0 的最终差值」兜底——终训数字才进论文主表。
~~另:若排队仍超时,向维护者申请后 4 张卡(`SETUP_B100.md` 红线:需明确指示),
8 卡 = 两臂并行(各 4 卡、同全局 batch),不改任何对照条件。~~

> ### ⚠️ 本节三处已被实测推翻(2026-07-29 补,详见 ROADMAP §方案完整性自查)
>
> **① 「宽裕档」不存在。** 实测 **9.1821 s/it @ 全局 batch 512 ⇒ 55.8 samples/s**;
> 全量 10.65M = 20,796 it/epoch ≈ **53.0 h/epoch**,R0 三个 epoch ≈ **159 h ≈ 6.6 天(4 卡)**。
> 「33E」≈ **73 天,直接不可行**。**按常态档执行**(≈26 天),不要再把宽裕档当选项。
>
> **② 「申请后 4 张卡」这条路不存在。** #339 已确认:GPU 0–3 就是维护者在这台 H100 的
> **全部配额,没有 4–7**。⇒ **6.6 天/R0 是硬约束**,只能在**臂数/epoch 数**上裁;
> 且全量 R0 与 1M 验证层**共用这 4 张卡,必须串行排队**。
>
> **③ 排序臂里的 A1 已被降级。** round2 §6.1:A1 惊奇池化只能利用**外观显著性**,
> 无法触及时间顺序(F1 证明池化头无条件顺序盲),第一轮「它是预测编码故事的聚合器半边」
> 的说法已撤回;`neural-inspired-implementation.md §A1` 另把它钉死为**非贡献项**。
> A2 则因 GATE-0 结案(顺序值≈0)+ SSv2-MC 暂缓而**失去了它的头条证据集**。
> ⇒ 这两臂即便在常态档也应**优先裁掉**,而不是等到紧张档。
>
> **④ 更根本的:排序臂现在排不出序。** 1M 层首个结果显示 10× 数据只买到 +0.82 raw r_mean、
> 落在 ±1.6 带内 ⇒ 在补上**配对 query-level bootstrap**(U10-1 / ROADMAP **G4** 🔴)之前,
> 任何「各 1 ep 排序」的读数都不可信。**G4 是本节能否执行的前提。**

### U7 · C1 细化三则(进 ROADMAP 2.2 的设计决策清单)

1. **教师语料决策(§2.6-4 的正解):** 建议 **curated 2–3M 先行**(与 CapRL 重写
   预算共享同一子集,教师 pass 成本 ~1/3),R2 与 R0/R1 保持**等步数**对照(DR 的
   标准比较法:同样本预算、更好的数据)。全量 10M 教师 pass 仅当 R2 显示数据饥饿
   再补——10k 成本试点(2.2 已排)同时给出两档的外推数字,决策留在数据出来那天。
2. **per-frame 教师嵌入保险:** 默认存 pooled 视频嵌入(~1KB/片);**另对 ~1M 子集
   加存 per-frame CLS**(16×d fp16 ≈ 45KB/片 ⇒ ~45GB)。这解锁三件事而不必重跑
   教师:低 T 学生对齐「同帧子集的教师池化」vs「全 16 帧池化」的可达性消融
   (§六.9 的设计决策 3)、注意力池化的时序显著性蒸馏信号、R4 自蒸馏复用。
3. **≥2 条合成字幕/片:** MobileCLIP2 的最大单项杠杆;`captions` 列表 + uniform
   采样钩子已在,只需 CapRL 产出侧保证 K≥2(`data-quality` §4.4 形态)。
4. **(2026-07-27 增补) 教师身份定案 + Qwen 搭车否决:** 教师维持 **InternVideo2-stage2**
   (对比式共享空间,vtc_kd 可算)。「重述 pass 顺手抽 Qwen3-VL 特征作主教师」经带检索
   评审**否决**:Qwen 全模型无对比对齐空间(LLM2CLIP 2411.04997 反证)、覆盖仅
   RECAPTION 桶 25%(偏置 + 成对信号 ~6%)、生成式→对比式特征蒸馏零发表先例、且省下
   的 ~17 H100·h 与 IV2-1B 全语料 pass(~21 H100·h 计算,4×H100 实际 1–2 天含解码)
   同量级。完整数字与架构事实见 `notes/caption-kd-review-2026-07.md`。
   **换代际同判**(2026-07-27 晚核验):InternVideo3(2606.12195,agentic MLLM)、
   InternVideo2.5(MLLM)、InternVideo-Next(2512.01342,无文本监督)均无对比空间,
   不可作 vtc_kd 教师;**更强教师的正道 = IV2-stage2-6B 或 1B+6B 集成**,
   按 U7-1 的 100k 切片决策法先试再定。

### U8 · 便宜标准项(半天级,各自独立)

- **EMA 学生权重**:`tasks_clip/pretrain.py` 现无任何权重平均;蒸馏/CLIP 训练加
  EMA 是近零成本的标准稳定器。R1 起默认开,消融一行。
- **训练内 held-out 评测修 key**(`README §4` 已记载的坑):R1 起必开,提前看到
  发散/过拟合,不再瞎跑到 epoch 末。
- **MobileCLIP2-S2 初始化**(§3.4 已确认架构一致、HF 可下):R0 仍用上游
  MobileCLIP-S2 初始化**锚定复现**;改进臂统一换 MobileCLIP2 初始化,并加一条
  降预算 R0'(VTC-only + 新初始化)隔离「初始化收益」vs「方法收益」——归因干净,
  免费涨点不背锅。`CHECKPOINTS.md` 补 `apple/MobileCLIP2-S2` 下载条目。

### U9 · 相对 KD 单一所有权(R2 消融表的整并)

`vtc_kd`(DFNDR 式相似度矩阵 KL,`data-quality` §6.2)与 `kd_geo`(Tung-Mori Gram
MSE,#197 B1)都是 O(B²) 关系型蒸馏,信号高度重叠。整并为 R2 内**一张消融表、
一个所有者**:`arm-cos`(仅 uta 余弦)→ `+sim-KL` → `+geo` →(预算内)`+两者`;
rkd 降为附录/缩预算档。两文档各加一行交叉引用,避免两处「novel loss」互相稀释。

**(2026-07-27) 表加一行可选 `+geo(Qwen)`**——生成式教师几何是否在对比式几何之外带
增量(文献空白,便宜可测):重述 sample-1 **prefill** 单层 hook(LLM 层 30/36 +
视觉塔 mean-pool 1152-d 对照),视觉 token 位 masked-mean → 逐维 z-score(必须,杀
massive-activation 维)→ L2 → PCA-1024(2 KiB/片,@6M 12 GiB),损失用 Tung-Mori
归一化 Gram Frobenius(**不用 KL**——任意空间教师无校准温度)。**CKA 前置门**
(~1 GPU·h,2.5k 试点):CKA(Qwen,IV2)>0.8 → 冗余撤行;∈[0.4,0.7] 且
CKA(Qwen,字幕文本)≥CKA(IV2,字幕文本) → 建行;层扫 {18,24,30,36}+塔出口。
**硬约束**:抓取不得迫使重述任务离开 vLLM(退 HF generate = 5–20× 慢);否则改
单独 prefill-only pass。设计全文见 `notes/caption-kd-review-2026-07.md` §一。

### U10 · 统计口径三件套

1. `collect_results.py` 输出 **query-level bootstrap 95% CI**(纯评测器改动);
2. **胜者双种子**(只胜者,不是全臂——预算可控);
3. **per-T 非劣性预注册**:R3 头条「每 T 点 ≥ R3b/固定基线」预先写死边际
   (建议 δ=0.3 R@1,CI 下界不低于 −δ 记为「非劣」,正差且 CI 不含 0 记为「胜」)。

## 5. 明确不做的(守边界,防散焦)

- 不新增基准承诺(MSVD/LSMDC 维持移除;不碰 temporal grounding——`whitespace`
  已标记为拥挤区)。
- P6(token 合并)维持「引用+对比」不复活;量化维持 kill-criterion 门控;
  N5/N7/N8 维持 skip(#197 的理由成立)。
- **不自训 GRPO 字幕器**(CapRL 路径 B1)除非 B0 消融显示明确域 gap——
  `data-quality` §4.2 的排序已正确,重申以防手痒。
- 不为 U1–U10 新开任何「研究线」:它们全部是现有 R 序列的加固件。

## 6. 落地状态

本次随本文一并修改(同一 PR):

- [x] `experiments/ROADMAP.md` —— 第一线加 1.5(U1 导出冒烟)、1.6(U2 多 T 曲线);
      2.1 验收补 per-T BN 诊断(U3);状态表加行 + 链接本文
- [x] `experiments/README.md` §2b —— 矩阵加「per-T BN 漂移诊断」行(U3)
- [x] `notes/paper-plan-zh.md` —— 新增 §七(2026-07-23 评审补丁:U1/U3/U4/U5/U6/
      U10 的计划层变更 + §2.6 勘误声明 + 投稿前文献复扫范围扩到弹性推理轴);
      `paper-plan.md` 英文版加对应 §7 摘要
- [x] `notes/data-quality-caprl-scheme.md` —— §0 加 `freeze_vision=False` 勘误注(§2.6-2)
- [x] `experiments/SETUP_B100.md` §6 —— batch 512 建议升级(U5)+ 预算阶梯指针(U6)
- [x] `notes/publishability-assessment.md` —— C2 行改「组合新」+ 必要条件加
      「对 FluxViT-S 有明确胜点」(§3.8-3)
- [x] `CHECKPOINTS.md` —— 补 `apple/MobileCLIP2-S2` 条目(U8)

待服务器/后续会话:U1 导出冒烟执行、U2 依赖 2.1 代码、U3 诊断脚本、
U7 随 2.2 落地、U8/U9/U10 随对应配置与评测器改动落地。

### 6.1 逐条落地复核(2026-07-28;**本节为准**,上面那份是 07-23 提案当天的状态)

复核方法:每条按**代码/文档实际存在性**核,不按"计划里写了"算。

| # | 落地状态 | 证据 / 缺口 |
|---|---|---|
| **U1** 导出冒烟 | 🟡 **两路已结论,一路待补** | `experiments/export_smoke.py` + `tests/test_export_smoke.py`(22 项)+ `COMMANDS.md` §3.6 + `export_smoke_results.md`。101vector 2026-07-27 实测:**ONNX / Core ML 在 T∈{2,4,8,16} 全 pass ⇒ P9 暂不触发**;TFLite 卡 `litert-torch` 需 torch≥2.6(本机 2.5.1 锁给 V100 sm_70)⇒ **独立 venv 补测**。⚠️ 该工作 07-27 就完工,却**滞留 `claude/export-smoke` 未合入 main 至 07-28**,期间 ROADMAP 一直写着"脚本待写" |
| **U2** 多 T 退化曲线 | ✅ **闭环** | ROADMAP 1.6:emb r_mean 55.4/60.0/61.6/62.3 @ T=2/4/8/16;T=8 DSL t2v=42.5 复现 B0 锚点 |
| **U3** per-T BN | 🟡 **工具齐,真 ckpt 未诊断** | `experiments/per_t_bn.py`(snapshot / recalibrate / drift report / per-T bank)+ 单测 + `README.md` §2b 矩阵行。**缺**:对真 ckpt 跑一次漂移诊断(判据联动"校准后口径"还没有数据支撑) |
| **U4** T 采样课程 fallback | ⬜ **计划级(按设计如此)** | 只在 `paper-plan-zh.md` §七.4;`utils/elastic_t.py` 无 sandwich / 分层采样。**这是触发型 fallback,R3 未跑就不该落代码** —— 不算缺口 |
| **U5** 全局 batch 512 | 🟡 **文档已改,未验** | `SETUP_B100.md` §6 建议 `batch_size 128`(4×128=512);**首光显存验证要等 R0 起跑** |
| **U6** 预算阶梯 | ✅ **规则已写死** | `SETUP_B100.md` §6 指针 + 本文 §U6 表;按首日单 epoch 墙钟触发,R0 未跑 |
| **U7** C1 三则 | ⬜ **随 2.2 未动** | ROADMAP 2.2 仍未勾;教师提取脚本未写。**已补**:U7-4 教师身份定案(IV2-stage2;Qwen 搭车否决,`caption-kd-review-2026-07.md`) |
| **U8** 便宜标准项 | 🟡 **三分之二** | ✅ `CHECKPOINTS.md` 已补 `apple/MobileCLIP2-S2`;✅ 训练内 held-out 评测可用(R1 A/B 逐 epoch `ret_test/` 出数即证);❌ **EMA 至今不在 `tasks_clip/pretrain.py`**(全仓 grep 无 `model_ema`/`ExponentialMovingAverage`)—— **U8 里唯一真缺口,R1 起本应默认开** |
| **U9** 相对 KD 单一所有权 | ⬜ **设计级** | `kd_geo` 仅存在于 `neural-inspired-implementation.md`,`vtc_kd` 仅存在于 `data-quality-caprl-scheme.md`,**两边都无代码、也没互加交叉引用行**。R2 未开工,按序不算迟到,但那行交叉引用**是零成本的,应当补** |
| **U10** 统计口径三件套 | 🟡 **一实践、一缺、一冲突** | ❌ (1) **bootstrap CI 不在 `collect_results.py`**(grep 无);🟡 (2) 双种子**已实践**(R1 A/B 跑了 seed 42/43),但未写成"胜者必须双种子"的硬规则;⚠️ (3) **δ=0.3 与实测噪声带冲突** —— 见下 |

**⚠️ 必须重定的一条:U10-3 的 δ=0.3 R@1 非劣边际已被后来的实测推翻。**
本文 07-23 提议 R3 头条「每 T 点非劣」预注册 **δ=0.3 R@1**。但 2026-07-27 的 1.8 选帧基线
测出:MSR-VTT 1K 上两条**采样时刻只差 3.1%** 的臂能拉开 **raw 2.40 R@1**,据此
`experiments/README.md` §6.1 把仓库级规则定为「±1.6 是**下界**;阳性主张要求 |Δ| 明显
大于 **2.4**」。**δ=0.3 比该噪声地板低约 8 倍 ⇒ 按它预注册的"非劣"在 1K 测试集上根本
不可测**。三条出路(R3 头条主张成形前必须选一条并写死):
① 把非劣边际抬到噪声带以上(≥2.4 量级,但那样"非劣"就很宽松);
② 改用**多基准聚合 R-Sum + 两口径同向**(README §6 既有纪律)而不是单点 R@1;
③ 换更大的评测集把 SE 压下去(成本最高)。
**在选定之前,`paper-plan-zh.md` §七.249 里的 δ=0.3 应视为待重定,不得直接引用。**

**本次(2026-07-28)据此复核所做的修改**:
- [x] 合入滞留的 `claude/export-smoke`(U1 完工件),ROADMAP 1.5 由「脚本待写」改为
      「两路已结论 / P9 暂不触发 / TFLite 独立 venv 补」,状态表与命令索引同步
- [x] 本节 §6.1 逐条审计表 + U10-3 冲突登记
- [ ] **待办**:U8 的 EMA 接线(R1 起默认开)、U10-1 的 bootstrap CI 进 `collect_results.py`、
      U9 的两文档交叉引用行、U3 真 ckpt 漂移诊断、U1 的 TFLite 补测 + `dynamic_axes` 一图多 T 验证

---

*代码行号与配置以 2026-07-23 仓库状态为准;文献结论以 §3 的网扫日期为准,
投稿前按 §六.9 + 本文 §2.3 的扩大范围复扫一次。*
