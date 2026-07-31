"""1M 验证层 —— **P9 统计池化**(seed 42)。见 notes/verification-tier-1m-zh.md §9。

相对 sub1m_r0 的唯一变化:temporal_pooling="stats"(W2 零初始化,step 0 与 mean **逐位**等价)。
判胜负减 sub1m_r0(同 seed);**同时必须跑赢 r1b(attn)** —— 二阶统计要证明自己比
一阶凸组合多出东西,不然这一臂没有立论。

⚠️ **收尾必须连 `temporal_pool.w2` 的范数一起报**(理由见 pilot_10k_P9.py)。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

seed = 42

model["vision_encoder"]["temporal_pooling"] = "stats"   # noqa: F405
optimizer["different_lr"] = dict(                       # noqa: F405  同 nA1,避免退化成 mean
    enable=True, module_names=["temporal_pool"], lr=1e-4
)

output_dir = "experiments/runs/sub1m_P9_s42"
