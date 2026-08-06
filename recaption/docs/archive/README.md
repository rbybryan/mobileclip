# archive/ —— 留档，不再随代码更新

这里的文档**平时不用读**：要么该阶段已经做完、要么执行层已被
[`master-plan.md`](../master-plan.md) 取代、要么要等后续阶段才启用。
内容原样保留（历史结论仍被现行文档引用，链接有效），但**不再维护**——
与现行文档冲突时，一律以 `master-plan.md` 和 `ab-pipeline_zh.md` 为准。

| 文件 | 归档理由 |
|---|---|
| [`audit-plan.md`](audit-plan.md) | 10k 数据审计的方法论（分层抽样、S0–S3 阶段定义）。审计已完成，结论见 `../audit-report.md` |
| [`audit-findings.md`](audit-findings.md) | 那次审计的原始统计与推理明细。成稿是 `../audit-report.md`，问题清单是 `../data-issues.md` |
| [`fusion-scheme.md`](fusion-scheme.md) | 多遍融合的最初设计（为什么要独立 N 遍）。**文首自述执行层已被 master-plan 取代**；仍是「一致性抓不住系统性幻觉」等方法论论证的出处 |
| [`caprl-scheme.md`](caprl-scheme.md) | CapRL 数据强化路线。同样已被 master-plan 取代执行层；两条训练线的分工裁决见 master-plan §0.2 |
| [`kd-review.md`](kd-review.md) | Qwen 特征搭车蒸馏的外部证据评审。裁决已落进 master-plan §0.7，zh 线不再做 KD |
| [`setup-h100.md`](setup-h100.md) | H100 + vLLM 的环境配置。**要等 Stage 3（300k 级）才启用**，当前全部工作在 V100 上 |
| [`setup-judge-api.md`](setup-judge-api.md) | 云端裁判端点配置。**对应的 `s2-judge` / `judge-agree` 子命令已于 2026-08-05 删除**（决策 2b 已落地）——要重开这条线，先从 `git log` 取回删除前的 `run_v100.sh` 与 `judge_agreement.py` |
| [`setup-judge-relay.md`](setup-judge-relay.md) | 自托管裁判中转的同款配置，同上（子命令同样已删除）|

要重新启用其中某条线（比如 Stage 3 上 H100），把对应文件移回 `../` 并在
`../../README.md` 的现行文档表里加一行即可。
