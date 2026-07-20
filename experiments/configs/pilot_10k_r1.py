"""P1 试点 — 10k 抽样上的全组件微型 A/B(**非论文数字**)。

与 pilot_10k_r0 唯一差异 = 三个优化开关全开(sigmoid + caption_idx 多正例 +
时间维注意力池化 + t/b 大学习率),等步数对照。

看什么(动力学,不看绝对精度):
- 损失正常下降、不发散;
- 训完从 ckpt 读 logit_scale / logit_bias:必须离开 SigLIP 初始值
  (exp(t') != 10, b != -10),否则 different_lr 没生效;
- 注意力池化的 proj 权重非零(已离开 mean 等价点);
- held-out 1k 自检索 R@1 的相对方向仅作参考。
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

output_dir = "experiments/runs/pilot_10k_r1"
