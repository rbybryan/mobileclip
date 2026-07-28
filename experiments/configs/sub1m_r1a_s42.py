"""1M 验证层 —— R1a:SigLIP 式 sigmoid 损失(seed 42)。见 notes/verification-tier-1m-zh.md。

相对 sub1m_r0 的唯一变化:use_sigmoid_loss=True(+ 多正例掩码 + logit_scale/bias 大 lr)。
损失轴,与池化轴(r1b/nA1)正交;判胜负减 sub1m_r0(同 seed)。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

seed = 42

model["use_sigmoid_loss"] = True                          # noqa: F405
available_corpus["internvid_v1"]["caption_idx"] = True    # noqa: F405  共享字幕按多正例
optimizer["different_lr"] = dict(                         # noqa: F405  logit_scale/bias 必须大 lr 才离开 SigLIP 初值
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)

output_dir = "experiments/runs/sub1m_r1a_s42"
