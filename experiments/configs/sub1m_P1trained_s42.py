"""1M 验证层 —— **P1-trained**:删掉整条时序分支后重训(seed 42)。见 §9。

相对 sub1m_r0 的唯一变化:disable_temporal_branch=True(减法,零新增参数)。
**收益**(已实测,§8.3 P1):440 个 ONNX 节点(27.8%)、80 次全张量转置、
CPU mixer 路径 −18%;且 T 从 44 块里的 40 块消失 ⇒ 导出从「1 权重 → N 图」
塌成**一张动态 batch 的图像图** + 每 T 一个小的 stage-4 图。

⚠️ **判读方向和别的臂相反**:这一臂**不需要赢** r0,只需要**不显著输**。
但 1M 层**只能否决、不能确立等价**(§6)—— NS ≠ 等价,只是没测出来。
所以本臂的正当用途是**保险**:若它显著输 r0,说明「训练时也能删」不成立,
在上 10M 之前就拦下来。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

seed = 42

model["vision_encoder"]["disable_temporal_branch"] = True   # noqa: F405

output_dir = "experiments/runs/sub1m_P1trained_s42"
