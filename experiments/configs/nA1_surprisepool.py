"""nA1 — 脑启发组件 A/B:预测编码「惊奇池化」(neural_inspired/README.md N2 池化半).

相对 R0 的唯一变化:model.vision_encoder.temporal_pooling="surprise"。
- 视频 embedding 从「8 帧简单平均」换成 neural_inspired.TemporalSurprisePool:
  帧均值为预测,按每帧预测误差能量加权;`alpha` 零初始化 ⇒ step 0 与 mean 逐位等价,
  严格不退化起步;
- 它是 r1b_attnpool(工程基线)的**脑启发孪生**。

这是可选 add-on,可整包移除(见 neural_inspired/README.md)。

淘汰标准:等步数下**必须跑赢 `temporal_pooling="attn"`(r1b)**才保留 —— 不是只赢 R0。
(该排序在 H100 全量做;V100/10k 只做链路冒烟,见 pilot_10k_surprise.py。)

**收尾时必须把最终 `alpha` 和 retrieval 一起报**(读法见 neural_inspired/README.md):
V100 两次试点实测 alpha 只走到 Adam 天花板的 3~6%(梯度符号在 batch 间翻转)。
按该位移率外推 1-epoch:有信号 → alpha~0.2(门真开,A/B 有效);仍是噪声 →
alpha~0.01,`output ≈ mean`,这一臂会退化成 mean-vs-attn。**两种情形 retrieval 都
可能"输给 r1b",含义却相反,只能靠 alpha 区分。** 故 alpha 仍 ≲0.01 时**不得**按
#197 T1 裁掉本方向,应先打开下面那行 different_lr 重跑一臂再下结论。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

model["vision_encoder"]["temporal_pooling"] = "surprise"

# 可选:给新增池化模块单独学习率(默认关闭,先跑同 lr 的干净 A/B)
# optimizer["different_lr"] = dict(enable=True, module_names=["temporal_pool"], lr=1e-4)

output_dir = "experiments/runs/nA1_surprisepool"
