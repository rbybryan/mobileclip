"""R1b — 组件 A/B:时间维注意力池化(paper-plan C3 / 旧提案 P3)。

相对 R0 的唯一变化:model.vision_encoder.temporal_pooling="attn"。
- 视频 embedding 从"8 帧简单平均"换成一个零初始化的 cross-attention 池化
  (query = 帧均值),step 0 与 mean 池化逐位等价,严格不退化起步;
- 新模块无预训练权重;如需给它单独更大学习率,打开下面的 different_lr。

淘汰标准:等步数下相对 R0 ≥ +0.3 R-Sum 才保留。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

model["vision_encoder"]["temporal_pooling"] = "attn"

# 可选:给新增池化模块单独的学习率(默认关闭,先跑同 lr 的干净 A/B)
# optimizer["different_lr"] = dict(enable=True, module_names=["temporal_pool"], lr=1e-4)

output_dir = "experiments/runs/r1b_attnpool"
