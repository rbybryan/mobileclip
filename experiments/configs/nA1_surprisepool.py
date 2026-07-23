"""nA1 — 脑启发组件 A/B:预测编码「惊奇池化」(notes/neural-inspired-directions.md N2 池化半).

相对 R0 的唯一变化:model.vision_encoder.temporal_pooling="surprise"。
- 视频 embedding 从「8 帧简单平均」换成 neural_inspired.TemporalSurprisePool:
  帧均值为预测,按每帧预测误差能量加权;`alpha` 零初始化 ⇒ step 0 与 mean 逐位等价,
  严格不退化起步;
- 它是 r1b_attnpool(工程基线)的**脑启发孪生**。

这是可选 add-on,可整包移除(见 neural_inspired/README.md)。

淘汰标准:等步数下**必须跑赢 `temporal_pooling="attn"`(r1b)**才保留 —— 不是只赢 R0。
(该排序在 H100 全量做;V100/10k 只做链路冒烟,见 pilot_10k_surprise.py。)
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

model["vision_encoder"]["temporal_pooling"] = "surprise"

# 可选:给新增池化模块单独学习率(默认关闭,先跑同 lr 的干净 A/B)
# optimizer["different_lr"] = dict(enable=True, module_names=["temporal_pool"], lr=1e-4)

output_dir = "experiments/runs/nA1_surprisepool"
