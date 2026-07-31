"""10k 链路 go/no-go —— **P1-trained**:整条时序分支删掉后重训(round-5 §8 P1)。

P1 的评测版(`eval_gate3`)是在**带着该分支训出来的**权重上做消融(Δ −0.43/−0.33,
预登记判据「>1.0 才杀」⇒ 通过)。**「从一开始就没有它,训出来会怎样」是另一个问题**,
这条臂回答它。

本臂是**减法**:不加任何参数。10k 档看的就是「loss 照常降」。
启动日志里应当出现 `[P1] 时序分支已删除:40 个 layer_scale_1d 置零并冻结`——
**没看到这行就说明 flag 没生效,别继续跑。**
"""
from experiments.configs.pilot_10k_r0 import *  # noqa: F401,F403

seed = 42
model["vision_encoder"]["disable_temporal_branch"] = True   # noqa: F405
output_dir = "experiments/runs/pilot_10k_P1trained"
