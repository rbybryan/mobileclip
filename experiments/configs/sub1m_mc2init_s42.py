"""1M 验证层 —— **MC2-init**:视觉+文本双塔换 MobileCLIP2-S2 初始化(seed 42)。
见 `notes/verification-tier-1m-zh.md` / `verification-tier-1m-zh.md` §11.6-3。

相对 sub1m_r0 的唯一变化:`vision_ckpt_path` / `text_ckpt_path` →
`checkpoints/mobileclip2_s2.pt`(架构与 v1 完全一致,CHECKPOINTS.md 已核)。

为什么**双塔一起换**:文本塔冻结 ⇒ 它定义输出空间;只换视觉塔会让 step-0 的
图文对齐断裂(MC2 视觉 × MC1 文本),把「更好初始化」的收益先烧掉一截。
双塔同换 ⇒ step-0 对齐是 MC2 自己的,与 r0(MC1 对)结构对称,A/B 干净。

⚠️ 训前必须过 CHECKPOINTS.md 的**键集一致性检查**(load_state_dict 是
strict=False,键对不上会**静默空载** —— gate5 同款坑)。

判读:同 seed 减 r0_s42,配对 bootstrap,两口径。
显著正 ⇒ 改进臂统一换 MC2 init(R0 仍锚 v1 复现,见 plan-review U8);
NS ⇒ 记 null(初始化优势没穿过视频微调);显著负 ⇒ 先查加载,再谈结论。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

seed = 42

model["vision_ckpt_path"] = "checkpoints/mobileclip2_s2.pt"  # noqa: F405
model["text_ckpt_path"] = "checkpoints/mobileclip2_s2.pt"    # noqa: F405

output_dir = "experiments/runs/sub1m_mc2init_s42"
