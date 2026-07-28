"""1M 验证层 —— nA1:脑启发「惊奇池化」(seed 42)。见 notes/verification-tier-1m-zh.md。

相对 sub1m_r0 的唯一变化:temporal_pooling="surprise"(r1b 的脑启发孪生)。
**门 alpha 零初始化,主 lr 下几乎不动 → 会退化成 mean**;按 nA1_surprisepool.py 自身要求
打开 temporal_pool 的 different_lr(lr=1e-4)给它公平机会。

判读特殊:
- 胜负基准是 **r1b(attn)**,不只是 R0——surprise 必须同时赢 mean 和 attn 才算贡献。
- **收尾必须连最终 alpha 一起报**:alpha 仍 ≲0.01 ⇒ output≈mean ⇒ 这一臂退化,retrieval
  输给 r1b 不代表方向错(读法见 neural_inspired/README.md)。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

seed = 42

model["vision_encoder"]["temporal_pooling"] = "surprise"  # noqa: F405
optimizer["different_lr"] = dict(                         # noqa: F405  给惊奇门单独 lr,避免退化成 mean
    enable=True, module_names=["temporal_pool"], lr=1e-4
)

output_dir = "experiments/runs/sub1m_nA1_s42"
