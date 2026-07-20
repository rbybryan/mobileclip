"""R1a — 组件 A/B:SigLIP 式 sigmoid 损失(paper-plan C3 / 旧提案 P4)。

相对 R0 的唯一变化:model.use_sigmoid_loss=True。
- 损失换成 `VTC_VTM_Loss.vtc_loss_sigmoid`(可学习 t/b,SigLIP 初始化
  t'=log 10, b=-10),softmax 的固定温度 temp 不再参与;
- 共享字幕(同一视频多个 clip 同 idx)按多正例处理,而非假负例。

淘汰标准(见 notes/paper-plan-zh.md):等步数下相对 R0 ≥ +0.3 R-Sum 才保留。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

model["use_sigmoid_loss"] = True

# 数据侧前置(审核修正 1):共享字幕的 clip 返回相同 idx(caption 哈希),
# 多正例掩码才真正生效——否则 idx 逐样本唯一,get_mask 退化为单位阵。
for __c in available_corpus["internvid_all"]:
    __c["caption_idx"] = True

# 审核修正 3:logit_scale/logit_bias 必须用大学习率才能离开 SigLIP 初始值
# (主 lr 1e-5 下 b=-10 几乎不动,等于训固定 t/b)。
optimizer["different_lr"] = dict(
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)

output_dir = "experiments/runs/r1a_sigmoid"
