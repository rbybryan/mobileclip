"""Seed-43 twin of `pilot_10k_r1.py` (control, no bank) — confirmation that the
R1 A/B gap isn't a single-seed / nondeterminism artifact. Identical to the seed-42
control except `seed=43` and a separate `output_dir`. Pair with
`nR1_textbank_pilot_s43.py`. See notes/r1-implementation-plan.md §9.
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_zh_10k_pilot"
test_file = dict(ret_test=available_corpus["internvid_zh_10k_pilot_val"])  # noqa: F405
test_types = ["ret_test"]
stop_key = None

model["use_sigmoid_loss"] = True  # noqa: F405
model["vision_encoder"]["temporal_pooling"] = "attn"  # noqa: F405
available_corpus["internvid_zh_10k_pilot"]["caption_idx"] = True  # noqa: F405
optimizer["different_lr"] = dict(  # noqa: F405
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)

scheduler["epochs"] = 10  # noqa: F405
save_iter = 0
seed = 43

output_dir = "experiments/runs/pilot_10k_r1_s43"
