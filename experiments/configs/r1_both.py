"""R1 — 组件叠加:sigmoid 损失 + 时间维注意力池化(R1a + R1b)。

只有当 R1a、R1b 至少一个单独通过淘汰标准后才值得跑;
若叠加低于单项最优,取单项最优作为 R1 进入后续 DR 实验(R2/R3)。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

model["use_sigmoid_loss"] = True
model["vision_encoder"]["temporal_pooling"] = "attn"

output_dir = "experiments/runs/r1_both"
