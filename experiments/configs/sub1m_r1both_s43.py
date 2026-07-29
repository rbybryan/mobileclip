"""1M 验证层 —— R1 文本负例库 A/B 的对照臂(bank OFF,seed 43)。种子一致性确认。
与 sub1m_r1both_s42 逐字一致,只改 seed 与 output_dir。见 notes/r1-implementation-plan.md §10。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

seed = 43

model["use_sigmoid_loss"] = True                          # noqa: F405
model["vision_encoder"]["temporal_pooling"] = "attn"      # noqa: F405
available_corpus["internvid_v1"]["caption_idx"] = True    # noqa: F405
optimizer["different_lr"] = dict(                         # noqa: F405
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)

output_dir = "experiments/runs/sub1m_r1both_s43"
