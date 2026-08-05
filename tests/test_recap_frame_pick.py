"""内容感知选帧（重述步 ①）的纯 CPU 单测——不碰视频、不需要 torch/decord。

两条不变量最关键，写错了在真实数据上**看不出来**：
  1. `n_cand=1` 必须逐帧等价于旧的均匀网格，否则「换选帧」会和「换抽帧位置」
     混在一起，A/B 对照失去意义；
  2. 段边界（抖动网格）不许被内容打分挪动，否则 N 遍之间的独立性被破坏，
     跨遍投票/直融的「多遍一致=可信」前提就垮了。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "data_audit"))
from frame_pick import (candidate_indices, frame_stats,  # noqa: E402
                        select_frames)


# ---------------------------------------------------------------- 候选网格
def test_single_candidate_matches_legacy_uniform_grid():
    """n_cand=1 = 旧实现的 np.clip(((arange+phase)/num_frames*n).astype(int))。"""
    for n in (7, 16, 90, 301):
        for num_frames in (4, 8):
            for phase in (0.1, 0.3, 0.5, 0.9):
                legacy = np.clip(
                    ((np.arange(num_frames) + phase) / num_frames * n)
                    .astype(int), 0, n - 1).tolist()
                got = candidate_indices(n, num_frames, phase, n_cand=1)
                assert [g[0] for g in got] == legacy
                assert all(len(g) == 1 for g in got)


def test_candidates_stay_inside_their_segment():
    """window=0.12 段宽，候选不许越过相邻段中心——越界=段序被打乱。"""
    n, num_frames, phase = 240, 8, 0.5
    seg = n / num_frames
    groups = candidate_indices(n, num_frames, phase, n_cand=3, window=0.12)
    assert len(groups) == num_frames
    for i, group in enumerate(groups):
        centre = (i + phase) * seg
        for idx in group:
            assert abs(idx - centre) <= 0.12 * seg + 1


def test_candidates_are_clipped_and_deduped_on_short_clips():
    """帧数少于段数时候选会撞在一起：必须去重且全部落在 [0, n-1]。"""
    groups = candidate_indices(3, 8, 0.5, n_cand=3, window=0.12)
    assert len(groups) == 8
    for group in groups:
        assert group == sorted(set(group), key=group.index)  # 去重、保序
        assert all(0 <= i <= 2 for i in group)


def test_groups_are_temporally_ordered():
    groups = candidate_indices(120, 8, 0.5, n_cand=3)
    firsts = [g[0] for g in groups]
    assert firsts == sorted(firsts)


# ---------------------------------------------------------------- 帧统计
def _frame(fill=None, noise_seed=None, size=64):
    if noise_seed is not None:
        rng = np.random.RandomState(noise_seed)
        return rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return np.full((size, size, 3), fill, dtype=np.uint8)


def test_black_frame_has_no_contrast_and_no_sharpness():
    sharp, contrast, _ = frame_stats(_frame(fill=0))
    assert sharp == 0.0 and contrast == 0.0


def test_noisy_frame_is_sharper_than_flat_frame():
    assert frame_stats(_frame(noise_seed=0))[0] > frame_stats(_frame(fill=128))[0]


# ---------------------------------------------------------------- 段内选帧
def _stats(spec):
    """spec: {下标: (清晰度, 对比度, 缩略图常数值)} -> select_frames 的 stats。"""
    return {i: (sharp, contrast, np.full(16, val, dtype=np.float32))
            for i, (sharp, contrast, val) in spec.items()}


def test_picks_the_sharper_candidate():
    stats = _stats({0: (1.0, 50, 10), 1: (9.0, 50, 10), 2: (2.0, 50, 10)})
    assert select_frames([[0, 1, 2]], stats) == [1]


def test_rejects_black_frame_even_when_it_scores_sharpest():
    """对比度低于阈值 = 转场黑帧：哪怕清晰度最高也不选。"""
    stats = _stats({0: (9.0, 1, 0), 1: (3.0, 50, 10), 2: (2.0, 50, 10)})
    assert select_frames([[0, 1, 2]], stats) == [1]


def test_falls_back_to_black_frame_only_if_every_candidate_is_black():
    """全黑段不能返回空——照旧给一帧，让下游 decode/描述自己处理。"""
    stats = _stats({0: (0.5, 1, 0), 1: (2.0, 1, 0)})
    assert select_frames([[0, 1]], stats) == [1]


def test_prefers_a_novel_frame_over_a_repeat_of_the_previous_pick():
    """第二段：清晰度略低但画面不同 > 与上一帧几乎相同（静止镜头去重）。"""
    stats = _stats({0: (5.0, 50, 10),        # 第一段选中
                    1: (5.0, 50, 10),        # 与上一帧同画面
                    2: (4.2, 50, 200)})      # 略钝但是新画面
    assert select_frames([[0], [1, 2]], stats) == [0, 2]


def test_segment_boundaries_are_never_crossed():
    """内容打分只在段内挑：每段的选中帧必须来自该段自己的候选列表。"""
    groups = candidate_indices(240, 8, 0.5, n_cand=3)
    rng = np.random.RandomState(7)
    stats = {i: (float(rng.rand()), 50.0,
                 rng.rand(16).astype(np.float32) * 255)
             for g in groups for i in g}
    picked = select_frames(groups, stats)
    assert len(picked) == len(groups)
    for idx, group in zip(picked, groups):
        assert idx in group


def test_same_frame_is_not_shown_twice_when_avoidable():
    """短视频里相邻段候选重叠：能错开就错开，不要把同一帧喂两次。"""
    stats = _stats({0: (5.0, 50, 10), 1: (4.9, 50, 200)})
    assert select_frames([[0, 1], [0, 1]], stats) == [0, 1]
