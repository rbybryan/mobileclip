"""1M 验证层 —— R0 基线(seed 42)。判读/设计见 notes/verification-tier-1m-zh.md。

介于 `pilot_10k_*`(链路冒烟,非数字)与全量 `r0_baseline`(10M,论文数字)之间的
**中间验证层**:单个 1M 分片(全量 ~10%)、batch 128(与在跑的全量 R0 一致)、
3 epoch(同配方)、**标准 MSR-VTT 1k 测试(测试集不缩)**。

所有 A/B 臂减去**本臂(同 seed)**判胜负;须两个 seed(42/43)方向一致、且差值
超过 MSR-VTT 1k 的 ±SE 才算数(小/矛盾差 = 不确定,只能上全量定夺)。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"        # 单个 1M 分片(全量 ~10%);train_file 由模板惰性解析
batch_size = 128                     # 与在跑的全量 R0 一致(经 "${batch_size}" 传到 inputs.batch_size.video)
batch_size_test = 128

# 环内评测标准 MSR-VTT 1k;必须把 test_file 的键改成 "ret_test" 以匹配 test_types,
# 否则基线默认的 dict(act_val=...) 会被跳过(见 log "Skip eval act_val split")。
test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

seed = 42

output_dir = "experiments/runs/sub1m_r0_s42"
