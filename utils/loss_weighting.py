"""U9 修复:`criterion.loss_weight` 从「纯日志开关」变成真权重。

历史坑(round2 §5 P2 / 评审 U9):`tasks_clip/pretrain.py` 原来是
`loss = sum(loss_dict.values())` 的**无权重求和**,`loss_weight` 只在建 meter 时被读。
于是任何新损失都以隐式 1.0 入总损失,靠改权重做的消融全是假的。

这里的规矩:
  - `loss_dict` 的键形如 `loss_<name>`,必须能在 `loss_weight[<name>]` 找到对应项,
    找不到**立刻抛错**(否则又回到隐式 1.0);
  - 权重恰为 1.0 时**跳过乘法**:`x * 1.0` 数值上逐位等于 `x`,但会给计算图加一个
    mul 节点;跳过它,基线臂(`vtc=1.0`)的图与修复前**完全相同** ⇒ 逐位等价免证。

纯 Python、无 torch 依赖(对张量只用 `*` 和 `+`)⇒ 单测在无 GPU 机器上可跑。
"""
from __future__ import annotations

PREFIX = "loss_"


def weighted_loss_sum(loss_dict, loss_weight):
    """按 `loss_weight` 加权求和 `loss_dict`(键 `loss_x` ↔ 权重键 `x`)。

    Raises:
        KeyError: loss_dict 有键在 loss_weight 里找不到对应权重。
        ValueError: loss_dict 为空,或某个对应权重恰为 0(权重 0 的损失就不该被算出来
            —— 模型侧按 loss_weight 决定算哪些,算了又乘 0 是白烧算力,多半是接线错)。
    """
    if not loss_dict:
        raise ValueError("loss_dict 为空 —— 模型 forward 没有返回任何损失")
    missing = [k for k in loss_dict
               if not k.startswith(PREFIX) or k[len(PREFIX):] not in loss_weight]
    if missing:
        raise KeyError(
            f"这些损失在 criterion.loss_weight 里没有对应权重:{missing}。"
            f"已有权重键:{sorted(loss_weight)}。补上显式权重(哪怕就是 1.0),"
            f"否则它会以隐式 1.0 入总损失 —— 正是 U9 要堵的洞。")
    total = None
    for k, v in loss_dict.items():
        w = loss_weight[k[len(PREFIX):]]
        if w == 0:
            raise ValueError(
                f"{k} 被算出来了但权重是 0 —— 模型侧应按 loss_weight 跳过它;"
                f"要临时关某损失,把它从模型输出里去掉,而不是乘 0。")
        term = v if w == 1.0 else v * w
        total = term if total is None else total + term
    return total
