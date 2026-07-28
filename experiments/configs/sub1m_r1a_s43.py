"""1M 验证层 —— R1a:SigLIP sigmoid 损失(seed 43,42 的孪生)。见 notes/verification-tier-1m-zh.md。"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

seed = 43

model["use_sigmoid_loss"] = True                          # noqa: F405
available_corpus["internvid_v1"]["caption_idx"] = True    # noqa: F405
optimizer["different_lr"] = dict(                         # noqa: F405
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)

output_dir = "experiments/runs/sub1m_r1a_s43"
