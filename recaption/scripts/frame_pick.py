#!/usr/bin/env python3
"""内容感知选帧（重述步 ① 的唯一可调项）——纯 CPU、只依赖 numpy + PIL。

维护者 2026-08-05 的问题是「这 8 帧怎么挑？」。答案分两半，都在本文件里：

  1. **哪些帧可以选**（`candidate_indices`）：仍是 TSN 均匀分段 + 每遍相位抖动。
     这一层绝不能动——N 遍之间的独立性全靠它，抖动没了跨遍投票就失去意义
     （多遍融合设计，见 recaption/docs/master-plan.md）。
  2. **段内选哪一帧**（`frame_stats` + `select_frames`）：在每段的候选里挑
     **清晰、非纯色、且与上一帧不重复**的那一帧。8 帧就是模型看到的全部证据，
     纯均匀抽样会把名额浪费在运动模糊、转场黑帧和静止重复画面上——每浪费一格，
     可用证据就少一格，模型就更可能去"脑补"。

拆成独立模块是为了能在没有 torch/decord 的机器上跑单测
（`tests/test_recap_frame_pick.py`）；vlm_audit_hf.sample_frames_pil 只负责
解码，选帧逻辑全在这里。
"""

import numpy as np
from PIL import Image


def candidate_indices(n, num_frames, phase=0.5, n_cand=1, window=0.12):
    """TSN 分段中心 → 每段的候选帧下标（按时间先后返回）。

    `phase` = 段内偏移（0.5 = 段中心），重述每遍扫一次以制造帧抖动。
    `n_cand` > 1 时，在抖动后的中心两侧 ±`window` 段宽内均匀给出候选。
    `n_cand` = 1 时逐帧等价于旧的纯均匀网格（回归测试钉死这一点）。
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if n <= 0:
        raise ValueError("empty video")
    seg = n / float(num_frames)
    groups = []
    for i in range(num_frames):
        base = (i + phase) * seg
        if n_cand <= 1:
            offsets = [0.0]
        else:
            offsets = [(-window + 2.0 * window * j / (n_cand - 1)) * seg
                       for j in range(n_cand)]
        group = []
        for off in offsets:
            idx = int(min(max(int(base + off), 0), n - 1))
            if idx not in group:
                group.append(idx)
        groups.append(group)
    return groups


def frame_stats(arr, size=32):
    """单帧 (清晰度, 对比度, 缩略图向量)，HxWx3 uint8 输入，~1ms。

    清晰度 = 32x32 灰度缩略图上的平均绝对梯度：运动模糊、叠化转场得分低，
    画面锐利得分高。对比度 = 标准差，纯黑/纯色转场帧≈0（InternVid 切片
    首尾常见）。缩略图向量用于判断与上一帧是否近乎重复。
    """
    img = Image.fromarray(arr).convert("L").resize((size, size))
    g = np.asarray(img, dtype=np.float32)
    sharp = float(np.abs(np.diff(g, axis=1)).mean()
                  + np.abs(np.diff(g, axis=0)).mean())
    return sharp, float(g.std()), g.reshape(-1)


def select_frames(groups, stats, min_contrast=8.0, w_div=0.5):
    """每段选一帧：清晰、非纯色、且不与上一段已选帧重复。

    只在**段内**移动，段边界（=抖动网格）原样保留，所以 N 遍的独立性不受影响。
    `stats` 是 {下标: frame_stats(...)}，须覆盖 groups 里所有下标。
    """
    picked, prev = [], None
    for group in groups:
        smax = max(stats[i][0] for i in group)
        smax = smax if smax > 0 else 1.0
        best, best_score = group[0], None
        for idx in group:
            sharp, contrast, vec = stats[idx]
            score = sharp / smax
            if prev is not None:
                # 与上一保留帧的新鲜度，归一到 0..1（差 32 个灰阶即封顶）
                score += w_div * min(1.0,
                                     float(np.abs(vec - prev).mean()) / 32.0)
            if contrast < min_contrast:
                score -= 10.0        # 纯黑/纯色转场帧，一票否决
            if idx in picked:
                score -= 1.0         # 同一帧不重复展示
            if best_score is None or score > best_score:
                best, best_score = idx, score
        picked.append(best)
        prev = stats[best][2]
    return picked
