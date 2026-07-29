"""1M 验证层 —— R0 基线(seed 43,42 的孪生)。见 notes/verification-tier-1m-zh.md。"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405  环内评 MSR-VTT 1k
test_types = ["ret_test"]

seed = 43

output_dir = "experiments/runs/sub1m_r0_s43"
