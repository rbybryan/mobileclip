"""1M 验证层 —— **C1 教师蒸馏臂**(seed 42)。§11.2 阶梯的第 ⑤ 级。

相对 sub1m_r0 的**唯一变化**:加 `vtc_kd`(嵌入蒸馏,`loss_kd_feat`,λ=1.0 预登记)。
教师 = InternVideo2-stage2 1B 的离线视频嵌入(`experiments/teacher/iv2s2_1m`,
按 anno 行号对齐;抽取与 merge 见 COMMANDS.md §3.10)。

判胜负减 `sub1m_r0_s42`(同 seed,配对 bootstrap 两口径,ranks 在
`experiments/evals/sub1m/sub1m_r0_s42`)。负 ≠ 方向死(λ/τ 是超参),但 1M 层不给
刷超参的面 —— **最多一次重试,重试参数已预登记为 λ_feat=0.25,其余不动**
(notes/c1-kd-training-manual-zh.md §3)。

⚠️ 开跑前的三道闸门(手册 §0–§2)必须全绿:anno 对齐 align、库体检 verify、
b100 冒烟(KD loss 有限 + kd_hit=1.0000)。跳过任何一道,15 h 就可能白烧。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_v1"        # 单个 1M 分片(与 r0 同语料;jump_filter=True ⇒ idx=anno 行号)
batch_size = 128
batch_size_test = 128

test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])  # noqa: F405
test_types = ["ret_test"]

seed = 42

# —— C1 KD(默认关;这里显式打开)——
# λ 预登记:kd_feat=1.0(首跑)/ 0.25(唯一重试位);kd_rel 首跑为 0(它是 U9 说的
# kd_geo 近亲,进同一消融表,但**不与 feat 同臂首跑** —— 单变量纪律)。
criterion["loss_weight"]["kd_feat"] = 1.0   # noqa: F405
criterion["vtc_kd"] = dict(                 # noqa: F405
    enable=True,
    teacher_dir="experiments/teacher/iv2s2_1m",
    # IV2-1B 投影后视频嵌入维度。以抽取 manifest 的 embed_dim 为准 ——
    # 手册 §0 的 verify 会打印它,不一致就改这里(跑错首个 forward 也会炸,不会静默)。
    teacher_dim=768,
    tau=0.07,                               # 只被 kd_rel 用;feat 蒸馏与它无关
)

output_dir = "experiments/runs/sub1m_C1kd_s42"
