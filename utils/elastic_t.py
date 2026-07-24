"""C2 · 弹性 T 训练的采样工具(ROADMAP 2.1)。

两个纯函数,配合 tasks_clip/pretrain.py 里的 elastic_t 开关使用:

- ``choose_step_t(choices, seed, step)``:为第 ``step`` 个优化步选一个 T。
  只依赖 (seed, step) ⇒ **所有 DDP rank 不通信即选到同一个 T**(模块级
  帧数上下文要求每 step 单一 T,见 models/mobileviclip_small.py 顶部注释)。
- ``subsample_frame_indices(total, t)``:从解码好的 ``total`` 帧母序列中均匀
  取 ``t`` 帧的下标(取每个等分段的中点,``floor((i+0.5)*total/t)``)。

约定(写死,教师/学生/评测共用同一条,审核修正 9 的「帧索引可复现」要求):

    total=16:t=16 -> [0..15](恒等)     t=8 -> [1,3,5,7,9,11,13,15]
             t=4  -> [2,6,10,14]         t=2 -> [4,12]

训练用法(每 step):16 帧母序列按当步 T 切片后喂模型;T==total 时不切片,
路径与固定 T 训练逐位一致。
"""
import random
from typing import List, Sequence


def subsample_frame_indices(total: int, t: int) -> List[int]:
    """均匀子采样下标:每个等分段取中点。严格递增、范围 [0, total)。"""
    if not 1 <= t <= total:
        raise ValueError(
            f"t={t} 不合法:需要 1 <= t <= total={total}"
            "(elastic_t.choices 里不能有超过 num_frames 的值)"
        )
    return [int((i + 0.5) * total / t) for i in range(t)]


def choose_step_t(choices: Sequence[int], seed: int, step: int) -> int:
    """按 (seed, step) 决定性地从 choices 里选当步的 T(跨 rank 一致)。"""
    if len(choices) == 0:
        raise ValueError("elastic_t.choices 为空")
    rng = random.Random(1000003 * int(seed) + int(step))
    return int(rng.choice(list(choices)))
