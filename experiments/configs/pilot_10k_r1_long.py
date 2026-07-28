"""Longer-schedule twin of `pilot_10k_r1.py` (control, no bank) — §9.3. Pairs with
`nR1_textbank_pilot_long.py` to answer "does no-pool catch up with more steps?"
Identical to the seed-42 control except a 40-epoch cosine schedule (vs 10) and a
separate output_dir; seed stays 42 (base default). See notes/r1-implementation-plan.md §9.3.
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
scheduler["epochs"] = 40  # noqa: F405
save_iter = 0

output_dir = "experiments/runs/pilot_10k_r1_long"
