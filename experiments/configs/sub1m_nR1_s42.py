"""1M 验证层 —— R1 文本负例库 A/B 的**处理臂**(bank ON,seed 42)。

在 `sub1m_r1both_s42`(对照)基础上,**唯一新增** = `criterion.text_bank`(1M 缓存)。
判胜负:本臂 − sub1m_r1both_s42(同 seed),两个 seed 方向一致。

**为什么 1M 才是 R1 真正的检验档(试点测不到的那个)**:R1 的价值假设是「批内负例成为信息瓶颈时,
额外硬负例才发力」。全局 batch 512 远低于 SigLIP 饱和点(~32k),bank 每步再补 K=16384 → 有效负例 ~17k,
正好推向 SigLIP 甜区;且 1M / 3 epoch(~5.9k step)**不会过拟合**,所以这里若领先,是**真·硬负例**收益,
不是 10k 试点那种「挡早塌」的小数据伪影(见 §9.4 诚实登记)。

**先决条件:先建 1M 文本负例库缓存**(训练环境,单卡,forward-only,一行不要断行;见 COMMANDS §3.5):
    python experiments/r1_build_textbank.py --anno anno_downstream/InternVid/InternVid-10M-flt_1.json --out-feats experiments/cache/textbank_1m_feats.fp16.npy --out-idx experiments/cache/textbank_1m_idx.npy

设计与判读见 notes/verification-tier-1m-zh.md、notes/r1-implementation-plan.md §10。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

seed = 42

model["use_sigmoid_loss"] = True                          # noqa: F405
model["vision_encoder"]["temporal_pooling"] = "attn"      # noqa: F405
available_corpus["internvid_v1"]["caption_idx"] = True    # noqa: F405
optimizer["different_lr"] = dict(                         # noqa: F405
    enable=True, module_names=["logit_scale", "logit_bias"], lr=1e-3
)

# 唯一差异:R1 文本负例库(1M 池;每步 randperm 随机抽 K=16384 条,用当前投影层重投影当 sigmoid 负列)
criterion["text_bank"] = dict(                            # noqa: F405
    enable=True,
    path="experiments/cache/textbank_1m_feats.fp16.npy",   # [K, C_pre] 预投影文本特征(~1M×512 fp16 ≈ 1GB)
    idx_path="experiments/cache/textbank_1m_idx.npy",       # [K] caption_to_idx(共享 idx 的 banked 文本当正例)
    K=16384,                 # 每步子采样;1M 池 → randperm 抽 16384,全池随训练轮换
    dedup_cos=0.90,          # 近重复稀疏掩码(试点 Part B 判据),近乎零成本
)

output_dir = "experiments/runs/sub1m_nR1_s42"
