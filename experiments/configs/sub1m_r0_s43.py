"""1M 验证层 —— R0 基线(seed 43,42 的孪生)。见 notes/verification-tier-1m-zh.md。"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

seed = 43

output_dir = "experiments/runs/sub1m_r0_s43"
