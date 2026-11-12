# 6 个模型更改 · 逐个说清

> **Release interpretation correction:** historical “非劣” / “non-inferior” labels below describe exploratory screening, not a confirmatory non-inferiority test. A CI containing zero is insufficient; the observed seed-noise threshold was retrospective. See [results and limitations](../../docs/RESULTS.md). Measurements are retained as historical records.


> 精简结果报告设计的 6 个方案,每个一节:**改了什么(代码钩子)/ config 一处改动 /
> 为什么这么设计 / 结果与为什么 / 怎么读**。数字见
> [`retrospective-brief.md`](retrospective-brief.md);怎么跑见 [`run.md`](run.md)。

## 0. 三条共同约定(读每一节前先记住)

1. **一处改动**:每个方案都 `from ...config import *` 继承基线,只覆盖**一行**。所以任何差异
   都能归因到这一行,不掺别的。
2. **零初始化 ⇒ step 0 逐位等价基线**:新池化头(attn/stats/surprise)的输出门全零初始化,
   删分支把时序权重置零。训练第 0 步这些方案与均值基线**逐位相同**——「有没有用」只由训练回答,
   排除了「新参数一上来就扰动」这个混淆。
3. **veto-only(只否决,不确立)**:1M 层样本小,**只能证伪、不能证明等价**。所以加法方案要
   「显著赢」才算数;减法方案(删分支)只要「不显著输」就放行到全量。判胜负一律**减同 seed 的
   `r0`**,并要**两个 seed 同向 + 配对 bootstrap 95% CI 不含 0**。

---

## 1. 基线(baseline)—— `sub1m_r0`

**改的是什么**:什么都不改。8 帧各自过视觉塔得到帧嵌入,再**按帧取算术平均**成 clip 嵌入
(`models/mobileviclip_small.py:1625` `temporal_pooling=="mean"` ⇒ `temporal_pool=None`;
前向在 `:1678` 做 `x.mean(0)`),对比损失用 softmax InfoNCE(`models/criterions.py:172` `vtc_loss`)。

**config**:无覆盖。`train_corpus="internvid_v1"` / `batch_size=128` / 3 epoch / 测 MSR-VTT 1k。

**角色**:所有 A/B 臂的参照系。它自己也回答了一个校准问题——「10× 数据(1M→全量的比例)只值
约 +0.82 r_mean」,这个尺度小到**无配对**的 ±1.6 带根本分不出来,正是 1M 层必须上**配对**
bootstrap 的原因(见 [`verification-tier-1m-zh.md`](../../notes/verification-tier-1m-zh.md) §6)。

---

## 2. 删时序分支(P1-trained)—— `sub1m_P1trained` · ✅ 全轮唯一通过

**改的是什么**:把视觉塔 stage 1–3 里那条**跨帧 1D 时序卷积分支**整条删掉。机制:`RepMixer`
的前向是 `x = x + λ_1d·(mixer_1d(x) − norm_1d(x))`(`models/mobileviclip_small.py:959`),
`apply_arch_flags`(`:198`)把 40 个 `layer_scale_1d` 门 `λ_1d` **置零且冻结** ⇒ 该支路恒等 ⇒
等价于它不存在。断言恰好命中 40 个(stage 1–3 是 RepMixer,stage 4 是注意力块,数不对就是选错张量)。

**config**(`sub1m_P1trained_s42.py`):
```python
model["vision_encoder"]["disable_temporal_branch"] = True
```

**为什么这么设计**:GATE-0 的时序容量审计给了强信号——**打乱帧序几乎不掉点、把 8 帧换成
同一帧重复才 −13**,说明模型用到的时序结构很弱;再加上核反对称性 0.5011≈随机、门比 0.0613。
既然这条分支没在学有用的东西,就**减掉它换效率**:实测导出算子 −27.8%(440 个 ONNX 节点)、
CPU mixer 路径 −18%,且 T 从 44 块里的 40 块消失 ⇒ 导出从「1 权重→N 图」塌成一张动态 batch
的图像图。**它是减法,零新增参数**,判读方向和别的臂相反:**不需要赢基线,只要不显著输**。

**结果与为什么**:raw R_sum 365.5 / DSL 368.6(基线 363.9 / 367.7),配对 CI
`删 − 基线 = +1.40 [−1.10, +3.90]`,六指标 CI 全含 0 ⇒ **非劣通过**。为什么删了不掉点:上面
GATE-0 已答——这条分支本就没承载可用的时序信息,置零它约等于什么都没动。

**怎么读**:veto 层「非劣」**不等于**「等价」,只是「1M 这个样本量没测出差」。它的正当用途是
**保险丝**——若显著输基线就说明「训练时也不能删」,在上 10M 前拦下;现在没输,放行到全量再确证。

---

## 3. 注意力池化(attn)—— `sub1m_r1b` · ❌ 不复现

**改的是什么**:把「按帧均值」换成**可学习的逐帧注意力加权**。一个单头交叉注意力块
(`TemporalAttentionPool` `models/mobileviclip_small.py:1682`):query = 帧均值,对 8 帧的
key/value 做注意力,输出 = `mean + proj(attn_out)`,**`proj` 零初始化** ⇒ step 0 输出 ≡ mean。

**config**(`sub1m_r1b_s42.py`):
```python
model["vision_encoder"]["temporal_pooling"] = "attn"
```

**为什么这么设计**:均值把 8 帧同权相加,可能被无信息帧稀释;让模型**学一个软权重**、多看
判别性强的帧,是最标准的池化升级。它也是**惊奇池化必须跑赢的工程孪生基线**——若脑启发版
赢不了这个朴素可学习版,就说明「脑启发」没带来额外东西。

**结果与为什么**:seed 42 看着涨(raw 366.7 vs 基线 363.9,+2.8),**但 seed 43 归零**
(raw 364.0,配对 `attn − 基线 = +0.10 [−3.10, +3.40]`,干净的零)。⇒ seed 42 的 +2.8 是
**种子地板噪声**,不是真效应。为什么:注意力是个**一阶凸组合**(权重和为 1 的加权平均),
表达力只比均值多一点点;在 1M 数据 + 冻结文本塔下,这点自由度学到的东西被 seed 抖动淹没。

**怎么读**:**单 seed 的正向差不能信**——这一臂就是「为什么 1M 层强制两个 seed」的活教材。

---

## 4. 二阶统计池化(stats / P9)—— `sub1m_P9` · ❌ 对基线显著负

**改的是什么**:在均值之外**加一个帧间标准差(二阶)项**:`out = mean + W2·LN(std)`,
`W2` 零初始化(`TemporalStatsPool` `models/mobileviclip_small.py:1716`)。`std` 沿帧维算
(`x.std(dim=1)`),T=1 时退化回 `mean`(避免 NaN)。

**config**(`sub1m_P9_s42.py`):
```python
model["vision_encoder"]["temporal_pooling"] = "stats"
optimizer["different_lr"] = dict(enable=True, module_names=["temporal_pool"], lr=1e-4)
```

**为什么这么设计**:GATE-0 说有用的时序结构是**置换不变且高阶**的(打乱≈0 ⇒ 置换不变;
重复帧 −13 ⇒ 不是一阶)。而 mean 是 rank-1 **一阶**统计、attn 是**一阶**凸组合,都表达不了。
**帧间标准差是最小的置换不变二阶统计量**——这是不加辅助损失(⇒ 没有可刷分的面)的前提下,
把检索损失在「帧残差子空间」上变成非平的唯一做法。**一处刻意偏离提案**:不对 mean 支路加 LN
(LN 会改向量方向、而下游只做 L2 归一消不掉),否则 step 0 就不等于基线、破坏零初始化纪律。

**结果与为什么**:raw R_sum 359.6 / DSL 363.4,配对 `P9 − 基线 = −4.30 [−8.30,−0.30]`(raw)/
`−4.30 [−8.00,−0.50]`(DSL),**两口径 CI 都不含 0 ⇒ 显著负**。为什么会**倒退**:二阶项没帮上,
反而当了噪声通道——`W2` 学出了非零权重却指向对检索无用(甚至有害)的方向,把干净的均值嵌入
搅浑了。这是 veto 层最干脆的一类结果:**单 seed 显著负即一枪毙命**,不必再跑 seed 43。

**怎么读**:**收尾必须连 `‖W2‖` 一起看**。若 `W2≈0` 则退化成 mean、打平基线不代表方向错;
本臂是 `W2` 真的学动了、且学坏了,所以「显著负」是对**这个二阶项**的否决,证据确凿。

---

## 5. 惊奇池化(surprise / nA1)—— `sub1m_nA1` · ❌ 干净的零(退化成均值)

**改的是什么**:脑启发(预测编码,Rao & Ballard 1999)的池化。均值当「预测/主旨」,每帧
偏离均值的**预测误差**当「惊奇度」,按惊奇度 softmax 加权池化,再用标量门 `alpha` 控制修正量:
`out = mean + alpha·(surprise_pooled − mean)`,`alpha` 零初始化
(`TemporalSurprisePool` `neural_inspired/temporal_surprise_pool.py`)。
🔑 `gamma`(惊奇锐度)**必须非零初始化**:若 `gamma` 也从 0 起,softmax 均匀 ⇒ `pooled==mean` ⇒
`(pooled−mean)==0` ⇒ **所有参数梯度恒为 0**、门永远开不了(零梯度不动点,autograd 验证过)。

**config**(`sub1m_nA1_s42.py`):
```python
model["vision_encoder"]["temporal_pooling"] = "surprise"
optimizer["different_lr"] = dict(enable=True, module_names=["temporal_pool"], lr=1e-4)  # 否则门在主 lr 下几乎不动
```

**为什么这么设计**:比「学任意权重」(attn)更强的先验——**信息量∝偏离主旨的程度**,直接把
「哪帧值得多看」写死成预测误差,不靠数据从零学注意力。判胜负基准是 **attn(r1b)**,不只是基线:
surprise 必须**同时赢 mean 和 attn** 才算脑启发带来了 attn 给不了的东西。

**结果与为什么**:**没有数值**——因为门 `alpha` 训练结束仍 ≲0.01,`out≈mean`,这一臂**退化成了
均值基线**。为什么退化:1e-4 的单独 lr 仍不足以让 `alpha` 离开 0 邻域(检索损失对这个标量门的
梯度太小),模型「懒得」用惊奇加权。所以它既没赢也没输,是**干净的零**。

**怎么读**:这**不代表方向错,只代表没学动**。判读铁律(见 `neural_inspired/README.md`):
**收尾必须连最终 `alpha` 一起报**;`alpha≲0.01` ⇒ 无法判读方向,只能说这套 lr/预算下门没打开。
要真正检验脑启发,得先让门学得动(更大 lr / 更长训练 / 换门的参数化),再谈胜负。

---

## 6. sigmoid 损失(SigLIP)—— `sub1m_r1a` · ❌ DSL 显著负

**改的是什么**:把对比损失从 **softmax InfoNCE** 换成 **SigLIP 的逐对 sigmoid**。不再在一整
行/列上做 softmax 归一,而是每个「视频-文本」对独立过 `logsigmoid`(正对 +、负对 −),配可学习
的 `logit_scale`/`logit_bias`(`get_sim_sigmoid` `models/criterions.py:141`;
`vtc_loss_sigmoid:225`)。开关是 `model.use_sigmoid_loss`(`mobileviclip_small.py:166`,
`:419` 处切到 sigmoid 路径),损失轴与池化轴正交。

**config**(`sub1m_r1a_s42.py`):
```python
model["use_sigmoid_loss"] = True
available_corpus["internvid_v1"]["caption_idx"] = True   # 共享字幕按多正例掩码,别误当负例
optimizer["different_lr"] = dict(enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3)  # 否则离不开 SigLIP 初值
```

**为什么这么设计**:SigLIP(2303.15343)论文主张 sigmoid 损失**不需要大 batch 的全局归一**、
小 batch 更稳、且天然处理多正例(同一视频多条字幕不必互相当负例)。在 batch 128 的 1M 层试它,
是想看「换个损失几何」能否白捡一点。多正例掩码 + logit_scale/bias 大 lr 是它的**标配**,少一样
就是没给它公平机会。

**结果与为什么**:raw R_sum 360.1 / DSL 362.0(基线 363.9 / 367.7)。配对:raw
`−3.80 [−8.40, +0.80]`(卡边界、含 0),**DSL `−0.95 [−1.68,−0.20]` r_mean / `−5.70
[−10.10,−1.20]` R_sum ⇒ 显著负**。为什么输、且**DSL 口径尤其输**:sigmoid 把每对独立打分、
不做行列归一,产出的相似度矩阵**列间尺度更散**;而 DSL(dual-softmax)靠「一个查询在所有候选
上的分布」二次加权,对这种尺度分散**更敏感**,于是把 sigmoid 的弱点放大成显著负。softmax 基线
的行列归一恰好喂饱了 DSL,这也是为什么**两口径必须都看**。

**怎么读**:raw 卡边界、DSL 显著负——**任一主口径显著负即否决**。不是 sigmoid 本身不行,而是
在「冻结文本塔 + batch 128 + 这套数据」的组合里它没占到便宜,且与下游 DSL 评测口径相性差。
