# recaption/ —— InternVid-zh 字幕重述与数据审计

整条「用 VLM 重写中文字幕」的链路都在这个目录里：**设计文档、运行说明、全部脚本**。
2026-08-05 从 `docs/` `notes/` `tools/data_audit/` 三处合并过来（映射表见文末）。

```
recaption/
  docs/       设计 / 计划 / 报告 / 环境搭建 / 已知问题（全部 markdown）
  scripts/    全部代码，统一入口 run_v100.sh
```

---

## 0. 三十秒版

同一段视频**独立看 N 遍**、各写一版中文描述（A 臂整段稠密写 5 遍；B 臂按固定五栏写 3 遍），
再让同一个 LLM **只读文字**把 N 版融合成「长稿（保细节）+ 短稿（≤30 字训练字幕）」。
**训练用的就是那条短稿。**

```bash
source ~/audit_env/bin/activate            # 审计机专用 venv，不是 conda

# A/B 双臂对比 + 并排评审包（默认只跑 ①重述 + ④b2直融，两步都 8 卡分片）
SAMPLE_OFFSET=300 N_SAMPLE=50 bash recaption/scripts/run_v100.sh pipeline-compare

# 要把某一批落成训练数据前：把画面验真加回来（唯一外部幻觉门）
DO_VERIFY=1 N_SAMPLE=50 bash recaption/scripts/run_v100.sh pipeline-sample
```

命令一律**从仓库根目录**执行。完整开关表见 [`docs/ab-pipeline_zh.md`](docs/ab-pipeline_zh.md) §6，
也抄了一份在仓库根的 [`COMMANDS.md`](../COMMANDS.md) §4.5。

---

## 1. 该看哪个文件

**先看这两个：**

| 文件 | 是什么 |
|---|---|
| [`docs/ab-pipeline_zh.md`](docs/ab-pipeline_zh.md) | **一页速览**：A/B 两臂分步图、每步默认开/关、选帧说明、prompt 全文、开关表、实测耗时 |
| [`docs/master-plan.md`](docs/master-plan.md) | **设计权威**：整条方案的定版计划、各阶段拍板记录、风险登记。与其他文档冲突以此为准 |

**运行与排错：**

| 文件 | 是什么 |
|---|---|
| [`docs/audit-workflow.md`](docs/audit-workflow.md) | 审计流水线 S0–S4 逐步操作说明（原 `tools/data_audit/README.md`） |
| [`docs/setup-v100.md`](docs/setup-v100.md) | V100 审计机环境搭建（venv、权重预下载、常见坑） |
| [`docs/setup-h100.md`](docs/setup-h100.md) | H100 机上的对应配置 |
| [`docs/setup-judge-api.md`](docs/setup-judge-api.md) · [`docs/setup-judge-relay.md`](docs/setup-judge-relay.md) | 云端/自托管裁判端点（可选步骤） |
| [`docs/data-issues.md`](docs/data-issues.md) | 数据本身的已知问题清单（编号 1–15，审计结论的落点） |
| 仓库根 [`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md) | 报错 → 原因 → 修复（跨全项目，含重述流水线） |

**设计依据与结论（读的时候按这个顺序）：**

| 文件 | 是什么 |
|---|---|
| [`docs/audit-plan.md`](docs/audit-plan.md) | 数据审计怎么做的方法论（分层抽样、S0–S3 各阶段定义） |
| [`docs/audit-findings.md`](docs/audit-findings.md) | 1 万条抽样的审计发现（15 条结论，重述方案的出发点） |
| [`docs/audit-report.md`](docs/audit-report.md) | 审计报告成稿（按问题归属重组的对外版本） |
| [`docs/fusion-scheme.md`](docs/fusion-scheme.md) | 多遍融合的设计依据（为什么要独立 N 遍、投票怎么算） |
| [`docs/caprl-scheme.md`](docs/caprl-scheme.md) | 数据质量 / CapRL 路线的方案与边界 |
| [`docs/kd-review.md`](docs/kd-review.md) | 字幕蒸馏方向的评审记录 |

> 执行层已由 `master-plan.md` 取代 `fusion-scheme.md` / `caprl-scheme.md`——后两者读作
> **设计依据**，具体跑什么以 master-plan 和 ab-pipeline 为准。

---

## 2. 脚本一览（`scripts/`）

统一入口是 **`run_v100.sh`**，子命令自带中文说明（`bash recaption/scripts/run_v100.sh` 不带参数即打印用法）。

| 脚本 | 角色 |
|---|---|
| `run_v100.sh` | **唯一入口**：smoke / s0–s4 审计 / recap / pipeline-sample / pipeline-compare / summary-table / snapshot |
| `vlm_audit.py` · `vlm_audit_hf.py` | VLM 判定与重述的两个后端（API 端点 / 本地 transformers 8 卡分片）；**所有 prompt 都在 `vlm_audit.py`** |
| `frame_pick.py` | ① 重述的内容感知选帧（分段网格+抖动不变，段内挑清晰/非黑/不重复的一帧） |
| `fuse_condense.py` | ④b1 跨遍规则计票（对照诊断版融合） |
| `fuse_verify.py` | ⑤ Panel-V 画面验真（GroundingDINO + RapidOCR，缺件自动降级为 unverified） |
| `split_views.py` · `recap_diag.py` | A′ 句切 · 多样性/坍缩诊断（独立采样前提的哨兵） |
| `score_alignment.py` | SigLIP 图文对齐打分 |
| `make_*.py` | 各类产物：全景 TSV、评审表、离线评审包、A/B 并排对比、表格总结页 |
| `apply_buckets.py` · `analyze_s1.py` · `check_*.py` · `aggregate_report.py` 等 | 审计各阶段的分桶、体检与报告 |

单测在仓库根的 `tests/`（不随脚本搬走，保持 `pytest tests/` 从根目录一把跑）：
`test_recap_frame_pick.py` · `test_vlm_prompts.py` · `test_slot_parse.py` ·
`test_fuse_condense.py` · `test_split_views.py` · `test_fuse_verify_helpers.py` ·
`test_summary_table.py`。

---

## 3. 与仓库其他部分的关系

- **服务器⇄Claude 的交接通道仍在** [`notes/live/`](../notes/live/)（`from-server.md` / `from-claude.md`），
  因为它不只谈重述，还带训练/评测。约定见 [`notes/git-handoff-workflow.md`](../notes/git-handoff-workflow.md)。
- 产物（`outputs/`）、视频与标注**不进 git**，按 `docs/setup-v100.md` 在服务器上就地生成。
- 训练/评测线的文档仍在 `notes/` 与 `docs/`，与本目录不重叠。

---

## 4. 旧路径 → 新路径

2026-08-05 的一次性搬迁。历史 handoff 笔记里的旧路径按此表换算：

| 旧 | 新 |
|---|---|
| `tools/data_audit/*.py` `*.sh` | `recaption/scripts/` |
| `tools/fuse_verify.py` | `recaption/scripts/fuse_verify.py` |
| `tools/data_audit/README.md` | `recaption/docs/audit-workflow.md` |
| `tools/data_audit/DATA_ISSUES.md` | `recaption/docs/data-issues.md` |
| `tools/data_audit/SETUP_V100.md` / `SETUP_H100.md` | `recaption/docs/setup-v100.md` / `setup-h100.md` |
| `tools/data_audit/SETUP_JUDGE_API.md` / `SETUP_JUDGE_RELAY.md` | `recaption/docs/setup-judge-api.md` / `setup-judge-relay.md` |
| `docs/recaption-ab-pipeline_zh.md` | `recaption/docs/ab-pipeline_zh.md` |
| `docs/internvid-zh-audit-report.md` | `recaption/docs/audit-report.md` |
| `notes/recaption-master-plan.md` | `recaption/docs/master-plan.md` |
| `notes/caption-fusion-scheme.md` | `recaption/docs/fusion-scheme.md` |
| `notes/data-quality-caprl-scheme.md` | `recaption/docs/caprl-scheme.md` |
| `notes/caption-kd-review-2026-07.md` | `recaption/docs/kd-review.md` |
| `notes/internvid-zh-data-audit-plan.md` | `recaption/docs/audit-plan.md` |
| `notes/audit-10k-findings.md` | `recaption/docs/audit-findings.md` |

**服务器上要改的只有命令前缀**：`tools/data_audit/run_v100.sh` → `recaption/scripts/run_v100.sh`。
输出目录（`outputs/audit_10k/…`）、环境变量、子命令名一个都没变，已有缓存和续跑不受影响。
