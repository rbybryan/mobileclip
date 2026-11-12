# model-optimization/ —— 1M 验证层的模型改造(6 个方案)

> **Release interpretation correction:** historical “非劣” / “non-inferior” labels below describe exploratory screening, not a confirmatory non-inferiority test. A CI containing zero is insufficient; the observed seed-noise threshold was retrospective. See [results and limitations](../docs/RESULTS.md). Measurements are retained as historical records.


精简结果报告设计的那 6 个模型更改都汇总在这个目录里:**每个方案改了什么、为什么、
怎么一条脚本复现、结论如何**。2026-08-05 从 `notes/` 整理过来(映射表见文末)。

```
model-optimization/
  docs/       模型改动逐条说明 / 运行说明 / 结果报告(全部 markdown)
  scripts/    一体化入口 run_model_opt.sh(训练 → 补评 → 配对比较)
```

> 模型**代码**(池化头、损失、时序分支)嵌在核心文件里、**config**属于共享的 `sub1m_*`
> 家族、通用**跑批/评测**器被多条实验线共用——这些是横切设施,**不搬进本目录**,而是在
> §2/§3 指清位置。本目录搬进来的是本轮**自己的**东西:结果报告 + 改动说明 + 一体化入口。

---

## 0. 三十秒版

在 **1M 验证层**(InternVid 单个 1M 子分片、batch 128、3 epoch、测 MSR-VTT 1k)上,
拿基线 `sub1m_r0` 做**同 seed 相减**的参照,试了三条**加法**(注意力/二阶统计/惊奇三种池化)、
一条**换损失**(SigLIP sigmoid)、一条**减法**(删掉整条跨帧时序 1D 卷积分支)。每个方案都是
相对基线的**一处 config 改动**、step 0 与基线**逐位等价**(新头零初始化 / 删分支置零)。
**结果:唯一通过的是减法「删时序分支」(非劣);三条池化 + sigmoid 全部未通过。**

```bash
# 一体化:训练 → 离线补评(落 ranks)→ 配对 bootstrap 比较。挂后台记 PID
# ⚠️ 共用 root:停止只能按 PID / 其 PGID,严禁 pkill
nohup bash model-optimization/scripts/run_model_opt.sh 0,1,2,3 > experiments/runs/model_opt.log 2>&1 &
echo "model-opt PID = $!"

# 已训练完、只要 CI:
PHASE=eval,compare bash model-optimization/scripts/run_model_opt.sh 3
```

命令一律**从仓库根目录**执行。分段跑法、判读规则见 [`docs/run.md`](docs/run.md)。

---

## 1. 该看哪个文件

**先看这两个:**

| 文件 | 是什么 |
|---|---|
| [`docs/model-changes.md`](docs/model-changes.md) | **逐条机制**:6 个改动各改了什么(代码钩子 file:line)、为什么这么设计、为什么通过/失败、怎么读结果(门的范数坑) |
| [`docs/retrospective-brief.md`](docs/retrospective-brief.md) | **精简结果报告**:R_sum + 逐指标 R@1/5/10(raw/DSL)+ 关键配对 CI + 讨论。要数字看这份 |

**运行与深挖:**

| 文件 | 是什么 |
|---|---|
| [`docs/run.md`](docs/run.md) | 训练 / 补评 / 比较的**可复制粘贴配方**,配 `scripts/run_model_opt.sh`;含判读规则与常见坑 |
| [`docs/retrospective.md`](docs/retrospective.md) | **完整复盘报告**:全轮叙事 + 机制分析 + 其余两臂(文本负例库、SigLIP 对照)+ 附录(负例、raw/DSL、逐指标配对 CI) |
| 仓库根 [`../notes/verification-tier-1m-zh.md`](../notes/verification-tier-1m-zh.md) | **判据设计权威**:为什么是 1M 层、为什么配对 bootstrap、各臂判决表(横切设施,留在 notes/) |

---

## 2. 脚本与 config 一览

**一体化入口**是 [`scripts/run_model_opt.sh`](scripts/run_model_opt.sh)——一条脚本三段
`train → eval → compare`(不带参数即打印用法),覆盖全部 6 个方案;各段做什么、落点、分段与
手动跑法见 [`docs/run.md`](docs/run.md)。

**config 仍在 `experiments/configs/`**(与其余 `sub1m_*` 实验同族,不拆出来)。每个方案 =
相对基线的一处覆盖:

| 方案 | 一处 config 改动 | config(× seed) | 判定 |
|---|---|---|---|
| 基线 | —(`temporal_pooling="mean"` + softmax) | `sub1m_r0_s{42,43}` | 参照 |
| **删时序分支** | `disable_temporal_branch=True` | `sub1m_P1trained_s{42,43}` | ✅ 非劣通过 |
| 注意力池化 | `temporal_pooling="attn"` | `sub1m_r1b_s{42,43}` | ❌ 不复现(seed 地板) |
| 二阶统计池化 | `temporal_pooling="stats"` + `temporal_pool` 单独 lr | `sub1m_P9_s42`(单 seed) | ❌ 对基线显著负 |
| 惊奇池化 | `temporal_pooling="surprise"` + `temporal_pool` 单独 lr | `sub1m_nA1_s{42,43}` | ❌ 干净的零(退化成均值) |
| sigmoid 损失 | `use_sigmoid_loss=True` + 多正例掩码 + logit 大 lr | `sub1m_r1a_s{42,43}` | ❌ DSL 显著负 |

---

## 3. 与仓库其他部分的关系(横切设施在哪)

模型优化是**横切**的:改动嵌在核心模型文件、跑批器被多条线共用,所以这几样**不在本目录**,
按下表到 canonical 位置找。机制与判读见 [`docs/model-changes.md`](docs/model-changes.md)。

| 东西 | 位置(留在原处的原因) |
|---|---|
| **模型改动代码** | 三种池化头 + softmax/sigmoid 损失 + 删分支开关都在 `models/mobileviclip_small.py` / `models/criterions.py` / `neural_inspired/`——嵌在核心文件里,不外挪。**类名与 file:line 见 [`docs/model-changes.md`](docs/model-changes.md) 各节**(单一出处,免两处维护) |
| **config** | `experiments/configs/sub1m_*.py`(与 nR1/r1both/C1kd/lean 等**非本报告**的 sub1m 臂同族,拆开反而割裂) |
| **共享跑批/评测器** | `experiments/`:`run_pretrain.sh` / `run_eval.sh` / `collect_results.py` / `paired_bootstrap.py`(gate / v100 / frame_agg / KD 各线都在用,搬走会全断) |
| **判据设计 + 状态工具** | `notes/verification-tier-1m-zh.md`(13 个 config 都引它)、`experiments/status_1m.py` / `run_verify_1m.sh`(通用 1M 层设施,非本报告 6 臂专属) |

产物(`experiments/runs/`、`experiments/evals/`)与权重、标注**不进 git**,在服务器上就地生成。

---

## 4. 旧路径 → 新路径

2026-08-05 的一次性搬迁。历史 handoff 笔记里的旧路径按此表换算:

| 旧 | 新 |
|---|---|
| `notes/1m-round-retrospective.md` | `model-optimization/docs/retrospective.md` |
| `notes/1m-round-retrospective-brief.md` | `model-optimization/docs/retrospective-brief.md` |

(其余三份 `docs/model-changes.md`、`docs/run.md`、`scripts/run_model_opt.sh` 是本轮新增,见 §1 与顶部目录树。)

**没搬的**(横切设施,路径不变):`experiments/configs/sub1m_*.py`、`models/`、
`neural_inspired/`、`experiments/{run_pretrain,run_eval}.sh`、`collect_results.py`、
`notes/verification-tier-1m-zh.md`、`experiments/{status_1m.py,run_verify_1m.sh}`——见 §3。
