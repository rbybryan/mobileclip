# 模型改动说明

这里记录 1M 子集实验中的五种改动和一个基线。数字来自[结果汇总](retrospective-brief.md)，运行方法见[运行指南](run.md)。

## 比较方式

各配置继承同一个训练配置，再覆盖模型开关及需要单独设置的学习率。它们并非都只改一行；例如 sigmoid 实验还改了多正例掩码和 logit 参数的学习率，因此结论对应的是整套配置。

新增池化头使用零初始化的输出投影或门，以保留初始的均值输出。删时序分支是另一种干预：把已有分支的门置零并冻结，不能据此声称它与未修改的预训练模型逐位相同。

检索结果按同 seed 配对比较。区间包含零表示当前比较不能确定差异方向，不能直接写成“非劣”或“等价”。

## 1. 基线

`sub1m_r0` 使用均值池化和 softmax 对比损失。视觉塔与训练代码来自 MobileViCLIP；这轮实验沿用它们，只在指定位置比较改动。

相关实现：[视觉模型](../../models/mobileviclip_small.py)、[对比损失](../../models/criterions.py)。训练配置为 batch 128、3 epoch，评测用 MSR-VTT 1k。

## 2. 删时序分支

`sub1m_P1trained` 设置 `disable_temporal_branch=True`。`apply_arch_flags` 将 stage 1–3 中 40 个 `layer_scale_1d` 门置零并冻结，从而移除该分支对输出的贡献。另一个 `skip_temporal_branch` 开关负责跳过计算；仅将门置零不会省掉算子。

我选择测试这一点，是因为[帧序诊断](../../experiments/gate0_results.md)中打乱帧序的影响较小。这个观察提出了删分支的可能性，但不足以证明模型完全不使用时序信息。

配对 raw R-sum 差值为 +1.40 [−1.10, +3.90]。这仍允许小幅损失，因此我把它保留为后续效率实验的候选。导出算子约减少 27.8%、CPU mixer 测量约降低 18% 是组件级结果，还需要完整模型的设备测量。

[旁路测试](../../tests/test_skip_temporal.py)检查门为零时跳过计算的输出，以及开关关闭时的原有行为。

## 3. 注意力池化

`sub1m_r1b` 将 `temporal_pooling` 设为 `"attn"`。`TemporalAttentionPool` 用帧均值作 query，对逐帧特征做注意力，再把投影后的修正加到均值上。输出投影零初始化，使初始输出等于均值池化。

这个改动用于检验可学习的帧权重是否比等权平均有效。seed 42 的 raw R-sum 比基线高 2.8，但 seed 43 的配对差值为 +0.10 [−3.10, +3.40]。收益未跨 seed 复现，现有实验也不能确定波动的唯一原因。

实现见 `TemporalAttentionPool`，初始化检查在[组件测试](../../tests/test_optimization_components.py)。

## 4. 二阶统计池化

`sub1m_P9` 使用 `TemporalStatsPool`：`mean + W2 · LN(std)`。帧间标准差提供均值之外的变化信息，`W2` 零初始化；均值支路不加额外归一化，避免改变初始输出。

配置同时给 `temporal_pool` 设置了 `1e-4` 的学习率。配对 raw R-sum 差值为 −4.30 [−8.30, −0.30]，dual-softmax 为 −4.30 [−8.00, −0.50]。在这次实验中表现更差。判断原因还要结合权重范数和训练过程，不能仅从检索下降断言二阶信息无用。

## 5. 惊奇池化

`sub1m_nA1` 使用 [TemporalSurprisePool](../../neural_inspired/temporal_surprise_pool.py)，按帧偏离均值的程度加权，再通过门加入修正：`mean + alpha · (surprise_pooled − mean)`。

`alpha` 从零开始，锐度参数 `gamma` 保持非零初值，避免加权结果也恰好等于均值、使门没有梯度。配置给池化头单独设置 `1e-4` 的学习率。

记录中的最终 `alpha` 仍接近零（约不超过 0.01），输出因而接近均值。这说明本次训练没有充分使用该路径；它并不能单独检验一个充分训练后的惊奇加权方案。相关检查在[模块测试](../../tests/test_neural_inspired.py)。

## 6. sigmoid 损失

`sub1m_r1a` 开启 `use_sigmoid_loss`，用逐对 sigmoid 替代 softmax 对比损失。实现是 [vtc_loss_sigmoid](../../models/criterions.py)，并接入了共享字幕的多正例掩码。`logit_scale` 和 `logit_bias` 使用单独的 `1e-3` 学习率。

这用于比较不同对比损失在当前数据与训练预算下的表现。raw R-sum 配对差值为 −3.80 [−8.40, +0.80]，dual-softmax 为 −5.70 [−10.10, −1.20]。后者区间完全低于零。当前结果只支持这套配置在该评测下表现更差，尚未分离出损失、参数学习率和评分方式各自的影响。

[组件测试](../../tests/test_optimization_components.py)检查了旧公式兼容性和重复字幕的多正例处理。
