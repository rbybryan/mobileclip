"""1M 验证层 —— **lean**:P1trained + 删 `temporal_embedding`(两个减法合体,seed 42)。
见 `notes/next-training-round-zh.md` / `verification-tier-1m-zh.md` §11.3、§11.6-2。

相对 sub1m_r0 的变化(两项都是**减法**,零新增参数):
  1. `disable_temporal_branch=True`   —— P1trained,1M 已非劣通过(§10.3(c))
  2. `disable_temporal_embedding=True` —— 评测版 eval_gate4 两口径显著为正
     (+0.70 raw / +0.67 DSL,v100_sweep_results §1)

这是**出货配置的保险丝臂**:通过后塔里不再有任何 T 形状参数 ⇒
C2 的「未见过的 T」从插值技巧变成结构性事实。

判读(两组比较都要做,配对 bootstrap,两口径):
  - lean − P1trained_s42:隔离第 2 项的**训练期**效应(eval 版是 +0.7,训练版待验)
  - lean − r0_s42:整体非劣,判据同 §10.3(c),δ=2.5(保守 5.5)
显著负超 δ ⇒ 保险丝熔断,出货配置退回只删时序分支。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

seed = 42

model["vision_encoder"]["disable_temporal_branch"] = True     # noqa: F405
model["vision_encoder"]["disable_temporal_embedding"] = True  # noqa: F405

output_dir = "experiments/runs/sub1m_lean_s42"
