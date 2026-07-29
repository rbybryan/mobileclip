"""1M 验证层 —— R1 文本负例库 A/B 的处理臂(bank ON,seed 43)。种子一致性确认。
与 sub1m_nR1_s42 逐字一致,只改 seed 与 output_dir。复用同一 1M 缓存(textbank_1m_*)。
见 notes/r1-implementation-plan.md §10。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

seed = 43

# ⚠️ 训练内评测的 key 必须落在 test_types 里。基线 config 是
# `test_file = dict(act_val=...)` + `test_types = ["ret_test"]` —— 键对不上,
# 唯一的测试集会被整个跳过,跑完只得到 `Empty DataFrame`(#334 已为其余臂修过,
# 本臂 #335 新增时漏了)。见 TROUBLESHOOTING「训练跑完但没有任何评测数字」。
test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

model["use_sigmoid_loss"] = True                          # noqa: F405
model["vision_encoder"]["temporal_pooling"] = "attn"      # noqa: F405
available_corpus["internvid_v1"]["caption_idx"] = True    # noqa: F405
optimizer["different_lr"] = dict(                         # noqa: F405
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)

criterion["text_bank"] = dict(                            # noqa: F405
    enable=True,
    path="experiments/cache/textbank_1m_feats.fp16.npy",
    idx_path="experiments/cache/textbank_1m_idx.npy",
    K=16384,
    dedup_cos=0.90,
)

output_dir = "experiments/runs/sub1m_nR1_s43"
