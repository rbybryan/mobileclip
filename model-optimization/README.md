# 1M 子集上的模型改造

这轮实验在 InternVid 的 1M 子集上比较五种改动：删时序分支、注意力池化、二阶统计池化、惊奇池化和 sigmoid 损失。均值池化加 softmax 是基线，评测用 MSR-VTT 1k。

我最想继续测试的是删时序分支。它的 raw R-sum 配对差值为 +1.40，95% 区间为 [−1.10, +3.90]，仍可能有小幅损失。此前写成“非劣通过”不够严谨。注意力池化的收益没有跨 seed 复现，另外几种改动也没有给出稳定提升。

[结果总结](../docs/RESULTS.md)说明了统计口径和限制；[原始汇总表](docs/retrospective-brief.md)保留各项检索指标。

## 改了哪些地方

| 改动 | 主要配置 | 对应文件 |
|---|---|---|
| 删时序分支 | `disable_temporal_branch=True` | `sub1m_P1trained_s{42,43}.py` |
| 注意力池化 | `temporal_pooling="attn"` | `sub1m_r1b_s{42,43}.py` |
| 二阶统计池化 | `temporal_pooling="stats"`，池化头单独学习率 | `sub1m_P9_s42.py` |
| 惊奇池化 | `temporal_pooling="surprise"`，池化头单独学习率 | `sub1m_nA1_s{42,43}.py` |
| sigmoid 损失 | `use_sigmoid_loss=True`，多正例掩码和 logit 参数学习率 | `sub1m_r1a_s{42,43}.py` |

配置在 [experiments/configs](../experiments/configs/)，基线是 `sub1m_r0_s{42,43}.py`。这些是配置文件清单，不代表每个 seed 都已有完成的实验结果。

[实现说明](docs/model-changes.md)记录了每种改动的动机、代码位置和结果。模型实现仍在 [models/mobileviclip_small.py](../models/mobileviclip_small.py)、[models/criterions.py](../models/criterions.py) 和 [neural_inspired](../neural_inspired/)，本目录只放本轮实验的说明与运行脚本。

## 运行

先按 [运行指南](docs/run.md)准备环境、数据和权重。从仓库根目录执行：

```bash
# 训练、离线评测、配对比较；会启动较长的 GPU 实验
bash model-optimization/scripts/run_model_opt.sh 0,1,2,3

# 已有训练权重时，只做评测和比较
PHASE=eval,compare bash model-optimization/scripts/run_model_opt.sh 3
```

脚本支持按阶段或指定配置运行，完整参数见运行指南。日志、权重和逐查询 rank 文件在服务器上生成，不随仓库分发。

## 更多记录

- [详细复盘](docs/retrospective.md)：包含文本负例库等其他实验，以及当时的分析。
- [1M 实验设计](../notes/verification-tier-1m-zh.md)：数据规模、配对评测和候选筛选的依据。
- [精简结果](docs/retrospective-brief.md)：完整的 raw / dual-softmax 指标与区间。

本目录于 2026-08-05 从 `notes/` 整理而来。旧的 `1m-round-retrospective.md` 和 `1m-round-retrospective-brief.md` 分别对应这里的 `docs/retrospective.md` 和 `docs/retrospective-brief.md`。
