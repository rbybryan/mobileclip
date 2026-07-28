"""Longer-schedule twin of `nR1_textbank_pilot.py` (treatment, R1 text bank ON) — §9.3.
Answers the one open caveat from the A/B verdict (§9.1/§9.2): "does no-pool catch up
with more steps?" Identical to the seed-42 treatment except a 40-epoch cosine schedule
(vs 10) and a separate output_dir; seed stays 42 (base default). Control twin =
`pilot_10k_r1_long.py`. Read: does the bank's lead persist / widen with 4x the steps,
or does the control close it? See notes/r1-implementation-plan.md §9.3.
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
scheduler["epochs"] = 40  # noqa: F405  (4x the pilot; cosine stretches over 40)
save_iter = 0

# the R1 text negative pool (same pilot cache as the 10-epoch runs)
criterion["text_bank"] = dict(  # noqa: F405
    enable=True,
    path="experiments/cache/textbank_pilot_feats.fp16.npy",
    idx_path="experiments/cache/textbank_pilot_idx.npy",
    K=16384,
    dedup_cos=0.90,
)

output_dir = "experiments/runs/nR1_textbank_pilot_long"
