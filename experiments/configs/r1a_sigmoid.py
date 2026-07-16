"""R1a — 组件 A/B:SigLIP 式 sigmoid 损失(paper-plan C3 / 旧提案 P4)。

相对 R0 的唯一变化:model.use_sigmoid_loss=True。
- 损失换成 `VTC_VTM_Loss.vtc_loss_sigmoid`(可学习 t/b,SigLIP 初始化
  t'=log 10, b=-10),softmax 的固定温度 temp 不再参与;
- 共享字幕(同一视频多个 clip 同 idx)按多正例处理,而非假负例。

淘汰标准(见 notes/paper-plan-zh.md):等步数下相对 R0 ≥ +0.3 R-Sum 才保留。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

model["use_sigmoid_loss"] = True

output_dir = "experiments/runs/r1a_sigmoid"
