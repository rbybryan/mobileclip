"""1M 验证层 —— R1 文本负例库 A/B 的**对照臂**(bank OFF,seed 42)。

= SigLIP 制度(sigmoid 损失 + 时间维注意力池化 + caption_idx 多正例 + t/b 大 lr),**不含**文本负例库。
既是 `sub1m_nR1_s42`(处理臂)的**匹配对照**,也顺带补上试点缺的「r1a+r1b 合并」臂。
这对臂**彼此相减**判胜负(不是减 sub1m_r0):ΔR-Sum = sub1m_nR1_s42 − 本臂(同 seed),
两个 seed(42/43)方向一致且超 MSR-VTT 1k 的 ±SE 才算数。

是 10k 试点 `pilot_10k_r1.py`(对照)的 1M 直放大;唯一放大项 = 数据 10k→1M、epoch 10→3、fp16→bf16。
设计与判读见 notes/verification-tier-1m-zh.md、notes/r1-implementation-plan.md §10。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"        # 单个 1M 分片(全量 ~10%)
batch_size = 128
batch_size_test = 128

seed = 42

model["use_sigmoid_loss"] = True                          # noqa: F405
model["vision_encoder"]["temporal_pooling"] = "attn"      # noqa: F405  与试点同:注意力池化
available_corpus["internvid_v1"]["caption_idx"] = True    # noqa: F405  共享字幕按多正例
optimizer["different_lr"] = dict(                         # noqa: F405  logit_scale/bias 必须大 lr 才离开 SigLIP 初值
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)
# epochs 继承 base=3(与 1M 验证层同配方);test_file 继承 base = 标准 MSR-VTT 1k(不缩)。

output_dir = "experiments/runs/sub1m_r1both_s42"
