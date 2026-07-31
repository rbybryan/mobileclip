"""10k 链路 go/no-go —— **P1-trained**:整条时序分支删掉后重训(round-5 §8 P1)。

P1 的评测版(`eval_gate3`)是在**带着该分支训出来的**权重上做消融
(Δ −0.43 raw / −0.33 DSL,预登记判据「>1.0 才杀」⇒ 通过)。
**「从一开始就没有它,训出来会怎样」是另一个问题**,这条臂回答它。

本臂是**减法**:不加任何参数,10k 档看的就是「loss 照常降」。
🔺 启动日志里必须出现
`[P1] 时序分支已删除:40 个 layer_scale_1d 置零并冻结`
—— **没有这一行就说明 flag 没生效,立刻停,别浪费 H100。**
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_zh_10k_pilot"
test_file = dict(ret_test=available_corpus["internvid_zh_10k_pilot_val"])  # noqa: F405
test_types = ["ret_test"]
stop_key = None

seed = 42
model["vision_encoder"]["disable_temporal_branch"] = True  # noqa: F405

scheduler["epochs"] = 10  # noqa: F405
save_iter = 0

output_dir = "experiments/runs/pilot_10k_P1trained"
