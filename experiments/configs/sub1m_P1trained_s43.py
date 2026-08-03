"""1M 验证层 —— **P1-trained** 的 seed-43 孪生:删掉整条时序分支后重训。见 §11.1。

相对 sub1m_P1trained_s42 的唯一变化:seed=43(以及 output_dir)。
为什么要它:§10 收官后 P1trained 是全轮唯一的正面结论(非劣通过),但那是单 seed;
r1b 的教训(§10.3(b))是单 seed 的 +2.8 可以整个是种子地板。头条结论必须双 seed。

读法(§11.1):跨 seed 对 r0_s42 配对 bootstrap(与 §10.3(b) 同惯例),两口径,
判据 = CI 下界仍在实测种子地板 δ=2.5(保守 5.5)之内。r0_s43 只在结果落在模糊带时才补。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

seed = 43

model["vision_encoder"]["disable_temporal_branch"] = True   # noqa: F405

output_dir = "experiments/runs/sub1m_P1trained_s43"
