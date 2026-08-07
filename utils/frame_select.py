"""训练无关的选帧基线 —— C2 必须打赢的那条平凡基线。

## 为什么要有这个文件

GATE-0(`experiments/gate0_results.md`)在官方权重上实测到两件事:

- 打乱 8 帧顺序:t2v R@1 42.50 -> 42.6,**帧的顺序值 ~0**;
- 8 帧全换成中间帧:42.50 -> 29.5,**帧的多样性值 ~13 个 R@1**。

而 B0 多 T 曲线(B0 多 T 曲线)显示 T=2 相对 T=8 掉 6.1 r_mean。两件事放一起
读:**在低帧预算下"挑哪几帧"应该很要紧,而这一点从来没量过**。

C2(弹性 T)的卖点是"单一 checkpoint 在任意 T 下 Pareto 优于固定 T 基线"。
审稿人必然会问:有没有跟"什么都不训练、只是把帧挑得聪明一点"比过?本模块
提供的就是那条对照。两种结果都有用 —— 选帧若补回大半个缺口,C2 的表述要改;
若补不回来,C2 的故事更硬,且手里多了一张必需的对照表。

## 成本诚实性(这是本模块最容易被读错的地方)

所有模式都在**已解码的 T 帧母序列**上做选择,输出 K 帧,**模型真正只跑 K 帧**
—— 与 `num_frames_test=K` 成本一致,可以直接同台比较。

打分只用**降采样后的像素**,不过编码器。这是刻意的:用编码器特征选帧要先把
16 帧全编码一遍,等于把想省的算力先花掉,那种做法只能当**上界**报,不是可
部署方案(见 `notes/verification-tier-1m-zh.md` R4 一节同样的论证)。特征空间上界若要做,
走离线预计算下标的路子,不要放进这里。

## 与 elastic_t 的关系

`uniform` 模式直接复用 `elastic_t.subsample_frame_indices`,所以它与 C2 训练
时的切片口径**逐位一致** —— 这正是我们要对照的那条基线,而不是另起一套。
"""
from typing import List, Optional, Sequence

from .elastic_t import subsample_frame_indices

MODES = ("uniform", "divmax", "random")


def pixel_features(frames, grid: int = 16):
    """把 [T, C, H, W] 压成 [T, D] 的廉价内容特征(不过编码器)。

    做法:自适应平均池化到 grid x grid,展平,再减去整段的均值帧。

    减均值的作用是把"整段视频共有的部分"去掉,让距离反映**帧之间的差异**
    而不是视频的全局亮度/色调。注意 frames 已经过 transform 的 Normalize,
    那是每通道的仿射变换,只会给各通道的距离加一个固定权重,不影响选择的
    定性行为。
    """
    import torch.nn.functional as F

    feats = F.adaptive_avg_pool2d(frames, grid).flatten(1)  # [T, C*grid*grid]
    return feats - feats.mean(0, keepdim=True)


def greedy_kcenter(feats, k: int) -> List[int]:
    """贪心 k-center:选 k 个互相尽量远的帧。返回**升序**下标。

    种子帧取离整段均值最近的那一帧(K=1 时退化为"最有代表性的一帧",而不是
    某个离群帧);之后每步取"到已选集合的最小距离"最大的那一帧。

    确定性:torch 的 argmin/argmax 在并列时返回最小下标,全程无随机数。

    已知偏好(报结果时要一起写):k-center 天然偏向离群帧,所以糊帧、闪光帧、
    转场帧有被选中的倾向 —— 这既可能是增益(多样性)也可能是损失(低质量帧)。
    这正是本实验要测的东西,不要预设结论。
    """
    import torch

    T = feats.shape[0]
    if not 1 <= k <= T:
        raise ValueError(f"k={k} 不合法:需要 1 <= k <= T={T}")

    center = feats.mean(0, keepdim=True)
    first = int(((feats - center) ** 2).sum(-1).argmin())
    selected = [first]

    while len(selected) < k:
        # 到已选集合的最小距离
        d = torch.stack(
            [((feats - feats[j]) ** 2).sum(-1) for j in selected], dim=0
        ).min(0).values
        d[torch.tensor(selected, device=d.device)] = -1.0  # 排除已选
        selected.append(int(d.argmax()))

    return sorted(selected)


def random_indices(total: int, k: int, seed: int, index: int) -> List[int]:
    """按 (seed, index) 决定性地随机取 k 个下标 —— 选帧实验的**地板**。

    存在的意义:如果 divmax 打不过 random,说明"多样性"这个信号在这个语料上
    没用;如果 uniform 也打不过 random,说明整个选帧问题在 T 这个尺度上不存在。
    没有这条地板,uniform vs divmax 的差值无法定标。
    """
    import random as _random

    if not 1 <= k <= total:
        raise ValueError(f"k={k} 不合法:需要 1 <= k <= total={total}")
    rng = _random.Random(1000003 * int(seed) + int(index))
    return sorted(rng.sample(range(total), k))


def select_indices(
    mode: str,
    total: int,
    k: int,
    frames=None,
    index: int = 0,
    seed: int = 0,
) -> Optional[List[int]]:
    """统一入口。返回升序下标;``k >= total`` 时返回 None 表示"不必切片"。

    返回 None 而不是 ``list(range(total))``,是为了让调用方能走"原样返回同一
    对象"的零操作路径 —— 与 GATE-0 扰动钩子同一个约定。
    """
    if mode not in MODES:
        raise ValueError(f"未知的选帧模式 {mode!r};可选 {'/'.join(MODES)}")
    if k >= total:
        return None
    if mode == "uniform":
        return subsample_frame_indices(total, k)
    if mode == "random":
        return random_indices(total, k, seed=seed, index=index)
    if frames is None:
        raise ValueError("divmax 模式需要 frames 才能打分")
    return greedy_kcenter(pixel_features(frames), k)
