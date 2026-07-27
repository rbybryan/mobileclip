"""nR1 — R1 SigLIP-regime text negative pool, pilot A/B arm (notes/r1-implementation-plan.md
§2/§7.6). Treatment for the go/no-go A/B; the **control is `pilot_10k_r1.py`** (identical setup,
no pool). Matched steps, same 10k data — the only difference is `criterion.text_bank`.

Prereq: build the cache first (COMMANDS §3.4):
    python experiments/r1_build_textbank.py \
        --anno anno_downstream/internvid_zh_10k_pilot.json \
        --out-feats experiments/cache/textbank_pilot_feats.fp16.npy \
        --out-idx   experiments/cache/textbank_pilot_idx.npy

Read (dynamics, not absolute accuracy): does the pool move held-out R@1 vs `pilot_10k_r1`?
Report BOTH directions — the text bank hardens v2t (txt_r1); t2v (img_r1) benefits only
indirectly through the shared encoders (examination §1). Kill criterion: pool vs no-pool
< 0.5 R-Sum ⇒ (d) misdiagnosed, stop.
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

# --- identical to pilot_10k_r1 (the control) ---
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

# --- the only difference: the R1 text negative pool (read by MobileViCLIP_Small._init_text_bank) ---
criterion["text_bank"] = dict(  # noqa: F405
    enable=True,
    path="experiments/cache/textbank_pilot_feats.fp16.npy",   # pre-projection feats [K, C_pre]
    idx_path="experiments/cache/textbank_pilot_idx.npy",       # caption_to_idx [K]
    K=16384,                 # min(K, pool); the ~9k-caption pilot pool -> uses all
    dedup_cos=0.90,          # from Part B (§7.6): sparse near-dup mask, ~free
)

output_dir = "experiments/runs/nR1_textbank_pilot"
