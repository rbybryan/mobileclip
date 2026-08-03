"""1M 验证层 —— **P1-trained**:删掉整条时序分支后重训(seed 43,42 的孪生)。见 §10.6。

`sub1m_P1trained_s42` 的 seed-43 复现。§10 全轮收官后,P1trained 是**唯一非劣通过**的臂,
但结论目前只有一个 seed —— 而它是**真要出货的模型**,house rule 要两个 seed 方向一致才算数
(§10.2 实测种子地板 δ=2.5 R_sum)。本臂补齐第二 seed,把「训练时也能删」坐实。

相对 sub1m_r0 的唯一变化仍是:disable_temporal_branch=True(减法,零新增参数)。
判读方向和别的臂相反:本臂**不需要赢** r0,只需**不显著输**(NS ≠ 等价,只是没测出损失)。
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
