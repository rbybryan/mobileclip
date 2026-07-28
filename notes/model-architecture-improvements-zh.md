# MobileViCLIP 模型结构改进报告(逐点验证版 · 中文)

**日期:** 2026-07-15
**性质:** 本报告是 `notes/model-architecture-improvements.md`(英文提案 P1–P10)的
**逐点验证 + 中文正式版**。每一条提案都经过两轮核查:
(a) **代码核查** —— 在本仓库源码中逐行确认所有事实性断言;
(b) **文献核查** —— 用 2024–2026 SOTA 论文核对外部证据。
验证中发现的错误已修正,并在下文明确标注(共 1 处实质修正、1 处措辞收紧、1 处结论加强)。

与既有笔记的分工:
- `notes/whitespace-directions.md` —— 数据管线、时序难负例、量化、帧选择(模型*外围*);
- `notes/data-quality-caprl-scheme.md` —— 字幕/数据质量;
- **本报告 —— 模型本体**:架构、token 混合、池化、训练目标(P1–P10);
- `notes/model-architecture-improvements-round3-zh.md` —— **第三轮新增架构方向 · 最终稿**
  (C1–C9,逐点源码核查版):改正了 3 处过时前提(T=8 已解硬编码、冻结 bug 已修、
  temporal_pool 已实现)+ 2 处机制错误(C1 联合注意力无法塞进时序卷积、C2 与全局
  set_num_frames 冲突)+ 1 处表述不准(C6 温度已被 R1a 覆盖);含明确死路清单。本报告是其前置背景。

---

## 一、模型结构(源码级摘要,已核实)

规范模型为 `MobileViCLIP_Small`(`models/mobileviclip_small.py`),Tiny 为同一套代码的
小宽度/浅深度版本。已核实:Small 基于 **MobileCLIP-S2**(视觉塔 MCi2 ≈35.7M 参数,
文本塔 12 层 base ≈63.4M 参数,合计 99.1M);Tiny 基于 **MobileCLIP-S0**
(`scripts/pretraining/clip/mobileviclip_tiny/config.py:38`,文本塔为 4 层 "mct")。

**视觉塔** = MobileCLIP 的 FastViT/MCi 图像骨干**逐帧**运行,外加三处视频化改动:

1. **时序 RepMixer 分支**:每个 `RepMixer` 并联一条时间轴 depthwise
   `Conv1d(k=3)`(`mixer_1d / norm_1d / layer_scale_1d`),推理时结构重参数化为单个
   `Conv1d`(即论文的"时间维结构重参数化")。
2. **第 4 阶段 T·H·W 联合全局注意力**:`MHSA` 把全部 `8×8×8 = 512` 个 token 拉平做
   完整自注意力,并加 `temporal_embedding`(形状 `(1, C, 8)`)。
   空间分辨率已核实:stem 两次 stride-2(/4)+ 三次 PatchEmbed 下采样(共 /32),
   256/32 = 8×8。
3. **输出头**:`conv_exp`(MobileOne+SE)→ `GlobalPool2D` 逐帧均值池化 + 线性投影 →
   对 8 帧 embedding **直接取平均**(`FastViT.forward`)。

**文本塔** 冻结,仅开放投影层(`freeze_text=True, open_text_projection=True`)。

**训练目标** 仅 VTC(InfoNCE,`VTC_VTM_Loss.vtc_loss`),温度**固定** 1/100
(`temp == temp_min == 0.01`,clamp 使其恒定),全局 batch 512(8×64,V100),
InternVid-10M-FLT 上 3 个 epoch,LiT 式(视觉开放、文本锁定)。

**代码中的隐性约束与"沉睡代码"(全部核实):**

- `T = 8` **硬编码**在所有时序 reshape 中(`MHSA.forward`、`RepMixer.forward`、
  `AttentionBlock.forward` 的 `x.reshape(8, ...)`)及 `temporal_embedding` 形状里。
- 注意力为**朴素实现**(显式 `softmax(q@kᵀ)@v`);`use_flash_attn` 等旗标一路传入
  `basic_blocks(...)` 后**被忽略**(grep 证实 forward 中无任何引用)。
- 已实现但**从未被调用**的代码:完整的 SigLIP 式损失
  `VTC_VTM_Loss.vtc_loss_sigmoid`(`criterions.py:133`,含可学习 `t`/`b` 入参;
  全仓库 grep 无调用点)、`get_sim_sigmoid`(`criterions.py:62`)、
  规范模型中未实例化的 `AttentionPoolingBlock`(注意:InternVideo2 骨干用
  **自己的同名副本**作 `clip_projector` 头,`internvideo2.py:484` —— 这恰是该模式
  在本仓库内可行的直接证据)、`open_text_lora` 配置旗标。
- **冻结逻辑 bug**:`hasattr(name, '_1d')` 对字符串恒为 `False`
  (`mobileviclip_small.py:56` / `mobileviclip_tiny.py:55`),故
  `open_vision_clip_projector` 的该分支永远不触发,仅
  `"mixer_1d" in name ...` 分支有效。任何基于冻结的实验(如仅训时序分支)须先修此处。

**算力去向(解析估算,非实测):** 以 Small(256²、T=8)按 MACs 粗算,FastViT 设计使
各 block 代价近似恒定(≈157M MACs/块/帧),阶段 1–4 的逐帧卷积/FFN 合计
≈6.9G MACs/帧 × 8 帧 ≈ 55G;第 4 阶段联合注意力(qkv/proj + 512² 矩阵乘)
≈4.7G/视频,**仅占总量约 8%**。结论:**逐帧卷积的 ×T 代价主导总 FLOPs**,
而联合注意力主导**峰值激活内存**且是**最难导出**的算子
(参见 `whitespace-directions.md §0`:现有唯一 Core ML 导出脚本静默跳过全部时序算子)。
两者是不同的问题,须分别对待。

---

## 二、逐点验证结果

| # | 提案 | 代码核查 | 文献核查 | 结论 |
|---|------|---------|---------|------|
| P1 | SDPA/高效注意力 | ✅ 朴素注意力、旗标被忽略属实 | ⚠️ **修正** | **成立,但须先升级环境**:锁定环境为 PyTorch **1.13.1**(`mobileviclip.yml`),`F.scaled_dot_product_attention` 是 2.0 才有的 API;且 Flash 内核要求 sm80+,**V100(sm70)只能走 memory-efficient 内核**。收益仍真实(消除 N×N 显存物化 + 中等提速),但不是 Flash 级加速。可选路线:升 torch ≥2.0(需验证 DeepSpeed 兼容)或在 1.13 上用 `xformers.ops.memory_efficient_attention`。 |
| P2 | 解除 T=8 硬编码、弹性帧数训练 | ✅ 三处 reshape + embedding 形状均核实;`interpolate_pos_embed` 已存在可复用 | ✅ FlexiViT / LiteFrame(2026) | 成立。另注:TimeSformer 报告联合注意力在 32 帧即 OOM——硬编码之外,**联合注意力本身也会限制帧数扩展**,与 P9 联动。 |
| P3 | 帧维注意力池化替代均值 | ✅ `mean(0)` 属实;⚠️ 措辞收紧:`AttentionPoolingBlock` 在规范模型中未用,但 InternVideo2 骨干以同名副本作头部在用(仓库内证据);`Attention_global_residual` 的 proj 零初始化可用于"零回退"启动 | ✅ InternVideo2 / Perception Encoder 均用注意力池化头 | 成立,且比原稿更强:仓库内已有同模式的工作实现可参照。开销 ≈ 8 个 512 维 token 上的一次 cross-attention,总 FLOPs 占比 ~1e-5 量级。 |
| P4 | SigLIP sigmoid 损失 + 可学习温度/偏置 | ✅ `vtc_loss_sigmoid` 完整存在、零调用;`temp==temp_min` 两个配置均属实;⚠️ 该实现用 `torch.eye` 忽略 `idx`;另查明 `idx` 即数据集样本下标(`pt_dataset.py __getitem__` 返回 `index`),故跨样本重复字幕在现有 softmax 版里同样是假负例 | ✅ SigLIP 原文:小 batch 下 sigmoid 优于 softmax;MobileCLIP2:温度调优值最高 ~2.8% | 成立,且工程量比预想更小(接线改动而非新码)。须把 `get_mask`(`criterions.py:270`)移植进 sigmoid 目标,SigLIP 初始化 t=log 10、b=−10。 |
| P5 | 视频版数据强化蒸馏(InternVideo2 教师) | ✅ `caption_augmentation` 钩子存在(`pt_dataset.py:40`);存储估算 512 维 fp16 ≈1KB/条 → 10M 条 ≈10–20GB 无误 | ✅ MobileCLIP(CVPR'24)/ MobileCLIP2(TMLR'25)DR 配方;InternVideo2-S/B 本身即从 1B 蒸馏 | 成立。**预期收益最大**的一项;图像侧同配方 +2–5 点。教师推理成本(InternVideo2-L 过 10M×8 帧,约 GPU-周量级)先在切片上实测。 |
| P6 | 第 4 阶段时序 token 合并(TempMe 式) | ✅ 512 token / 8×8 空间尺寸核实;解析估算显示注意力仅占总 MACs ~8% → 原稿"收益有界"的诚实定位被证实 | ✅ TempMe(ICLR'25):−95% token、−51% GFLOPs、R-Sum +4.4%(参数高效)/+7.9%(全微调) | 成立,定位为**峰值内存 / 端侧导出收益**,GPU 总吞吐收益有限。依赖 P2。 |
| P7 | 非对称帧编码(关键帧全路径 + 廉价路径) | ✅ ×T 逐帧代价主导总 FLOPs 由解析估算证实;时序 RepMixer 可作跨帧注入点属实 | ✅ DeltaCNN / Skip-Conv / SparseTem / LiteFrame / MeToM(CVPR'26) | 成立,**研究型赌注**(动作类基准可能掉点)。必须击败"均匀 T=4 采样"这一平凡基线才值得保留。依赖 P2,建议配合 P5 的自蒸馏损失做微调而非重训。 |
| P8 | 文本塔瘦身(mct + 蒸馏) | ✅ S2 文本塔 63.4M > 视觉塔 35.7M(官方数字核实);mct 实现与分发均在仓库内(`modules/text/repmixer.py`;`text_encoder.py:116` `variant == "mct"`) | ✅ MobileCLIP-S0 证明 DR 训练下小文本塔掉点 ≤1–2;TinyCLIP 权重继承 | 成立。端侧**模型体积**最大杠杆(≈−45%),优于对视觉塔做 INT8。 |
| P9 | 分离式时空注意力(可导出的第 4 阶段) | ✅ 联合注意力为导出阻塞点(引 whitespace 笔记 §0 的已核实结论) | ✅ **结论加强**:TimeSformer 消融显示分离式**精度更高且更省**(K400/SSv2 最优;联合式 32 帧 OOM) | 成立且证据比原稿更强:这不只是"省算力换掉点"的权衡,文献支持"可能不掉点甚至更好"。逐帧 64-token 注意力静态形状、可直接导出。 |
| P10 | 套娃(Matryoshka)embedding 头 | ✅ 仅损失层改动,与 P4/P5 兼容 | ✅ MRL(NeurIPS'22)标准结论 | 成立,零推理开销,端侧 ANN 索引缩小 4–8×。 |

**验证过程中另发现并核实的事实**(不改变结论,写入备查):
`tasks_clip/shared_utils.py:186` 存在模型级 `reparameterize_model` 工具(默认关闭),
GPU 侧评测应打开以对齐论文的推理形态。

---

## 三、改进建议(经验证后的最终版)

### 优先级总表

| # | 建议 | 类型 | 推理开销 | 预期效果 | 风险 |
|---|------|------|---------|---------|------|
| P1 | 高效注意力内核(先升 torch 或用 xformers) | 工程修复 | ↓(GPU) | 训练/评测提速省显存,数学等价 | ~0 |
| P2 | 解除 T=8 硬编码;帧数弹性训练(T∈{4,8,12,16}) | 工程+训练 | = | 帧数成为部署旋钮;P6/P7 前置 | 低 |
| P3 | 帧维注意力池化替代均值(复用仓库内模式,零初始化启动) | 架构(头) | +ε | 每 FLOP 精度收益最高,+0.5~1.5 R@1 量级 | 低 |
| P4 | 接通 `vtc_loss_sigmoid` + 可学习温度/偏置(移植 `get_mask`) | 目标函数 | = | batch=512 下优于 softmax;抗字幕噪声 | 低 |
| P5 | **视频版数据强化蒸馏**:离线存 InternVideo2-L embedding,VTC + embedding-KD + 相似度矩阵 KD | 目标函数 | = | **预期精度收益最大**(图像侧同配方 +2~5 点);兼具论文贡献形态 | 中(教师推理与存储) |
| P6 | 第 4 阶段时序 token 合并 | 架构 | ↓ | 峰值内存与端侧延迟收益;GPU 总量收益有限(~8% MACs) | 中 |
| P7 | 非对称帧编码(K=2–4 关键帧全路径,余帧浅路径经时序 RepMixer 注入) | 架构 | ↓↓ | 直击 ×T 主导成本,视频编码 FLOPs 近半 | 高(研究赌注) |
| P8 | 文本塔换 mct + 蒸馏(或 TinyCLIP 权重继承) | 架构(文本) | ↓体积 | 总参数 ≈−45%,掉点预期 ≤1 R@1 | 中 |
| P9 | 分离式时空注意力(逐帧空间注意力 + 已有 1D 时序卷积) | 架构 | =/↓ | 解锁端侧导出;文献支持不掉点甚至更好 | 中 |
| P10 | 套娃 embedding(64/128/256/512 嵌套) | 目标函数 | = | 端侧 ANN 索引 4–8× 缩小 | 低 |

### 建议节奏

```
阶段 0(随复现进行):  P1(注意力内核)+ 修复冻结 bug            [天级]
阶段 1(首批改进):    P4(sigmoid+温度)→ P3(注意力池化)        [各一次 A/B,1–2 周]
阶段 2(主攻):        P5(视频 DR 蒸馏),顺带折入 P10
阶段 3(效率线):      P2(弹性 T)→ P9(可导出)→ P6(合并)→ P7(赌注)
体积需求出现时:       P8
```

P3+P4+P5 相互兼容、共用同一评测框架;效率线(P2→P9→P6→P7)独立推进,
以 `whitespace-directions.md §0` 记录的端侧导出前置条件为闸门。
所有 A/B 均在等步数、等数据预算下对比,评测集:MSR-VTT / DiDeMo / ANet 零样本检索
+ K400 零样本分类;时序敏感项(P7/P9)加测 SSv2-MC。

---

## 四、参考文献

- MobileViCLIP, ICCV 2025 — arXiv:2508.07312(本仓库基线)
- MobileCLIP, CVPR 2024(arXiv:2311.17049)/ MobileCLIP2, TMLR 2025
  (arXiv:2508.20691)— 多模态强化训练 / DR 数据集;S2 参数拆分 35.7M+63.4M
- SigLIP, ICCV 2023(arXiv:2303.15343);SigLIP 2(arXiv:2502.14786)
- TempMe, ICLR 2025(arXiv:2409.01156);ToMe(ICLR 2023);vid-TLDR(CVPR 2024)
- InternVideo2(arXiv:2403.15377)— 教师模型;S/B 从 1B 蒸馏;注意力池化头
- TimeSformer, ICML 2021(arXiv:2102.05095)— 分离式 vs 联合时空注意力消融
- FlexiViT, CVPR 2023 — 训练期尺寸弹性(本文用于时间维)
- DeltaCNN(CVPR 2022);Skip-Convolutions(ICCV 2021);SparseTem
  (arXiv:2410.20790);LiteFrame(arXiv:2605.17260);MeToM(CVPR 2026)
- TinyCLIP, ICCV 2023 — 文本塔权重继承压缩
- Matryoshka Representation Learning, NeurIPS 2022
- Perception Encoder(arXiv:2504.13181)

*代码行号以 2026-07-15 的仓库状态为准;文件漂移后请按符号名重新定位。*
