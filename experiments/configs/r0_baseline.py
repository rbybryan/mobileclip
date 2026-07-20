"""R0 — 复现基线:与原 repo 预训练配方完全一致(唯一差异是 output_dir)。

所有 A/B 实验的对照组。任何优化组件都不开启:
softmax InfoNCE + 固定温度 1/100 + 逐帧 mean 池化。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

output_dir = "experiments/runs/r0_baseline"
