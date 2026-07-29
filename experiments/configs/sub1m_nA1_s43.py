"""1M 验证层 —— nA1:惊奇池化(seed 43,42 的孪生)。见 notes/verification-tier-1m-zh.md。
胜负基准是 r1b(attn),不只 R0;收尾连 alpha 一起报(判退化)。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405  环内评 MSR-VTT 1k
test_types = ["ret_test"]

seed = 43

model["vision_encoder"]["temporal_pooling"] = "surprise"  # noqa: F405
optimizer["different_lr"] = dict(                         # noqa: F405
    enable=True, module_names=["temporal_pool"], lr=1e-4
)

output_dir = "experiments/runs/sub1m_nA1_s43"
