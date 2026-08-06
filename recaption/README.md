# recaption/ —— InternVid-zh 字幕重述与数据审计

整条「用 VLM 重写中文字幕」的链路：文档在 `docs/`，代码在 `scripts/`（入口 `run_v100.sh`）。

## 三十秒版

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

命令一律**从仓库根目录**执行。开关全表见 [`docs/ab-pipeline_zh.md`](docs/ab-pipeline_zh.md) §6，
也抄了一份在仓库根的 [`COMMANDS.md`](../COMMANDS.md) §4.5。

## 现行文档（`docs/`，六份，只有这些跟着代码走）

| 文件 | 什么时候看 |
|---|---|
| [`ab-pipeline_zh.md`](docs/ab-pipeline_zh.md) | **最常看**：A/B 两臂分步图、每步默认开/关、选帧说明、prompt 全文、开关表、耗时 |
| [`master-plan.md`](docs/master-plan.md) | **设计权威**：定版计划、各阶段拍板记录、风险登记。与其他文档冲突以此为准 |
| [`audit-workflow.md`](docs/audit-workflow.md) | 审计流水线 S0–S4 的逐步操作 |
| [`setup-v100.md`](docs/setup-v100.md) | V100 审计机环境搭建（venv、权重预下载、常见坑） |
| [`data-issues.md`](docs/data-issues.md) | 数据本身的已知问题清单（编号 1–15），审计结论的落点 |
| [`audit-report.md`](docs/audit-report.md) | 10k 抽样审查报告成稿——「这批数据到底什么毛病」的答案 |

报错先查仓库根的 [`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md)。

## 留档（`docs/archive/`，八份，平时不用读）

已完成、已被 `master-plan.md` 取代、或要等后续阶段才启用的文档。链接仍然有效，但
**不再随代码更新**——清单与归档理由见 [`docs/archive/README.md`](docs/archive/README.md)。

## 脚本（`scripts/`）

唯一入口是 **`run_v100.sh`**，不带参数即打印全部子命令。

| 脚本 | 角色 |
|---|---|
| `run_v100.sh` | 入口：smoke / s0–s4 审计 / recap / pipeline-sample / pipeline-compare / summary-table / snapshot |
| `vlm_audit.py` · `vlm_audit_hf.py` | VLM 判定与重述的两个后端（API 端点 / 本地 transformers 8 卡分片）；**所有 prompt 都在 `vlm_audit.py`** |
| `frame_pick.py` | ① 重述的内容感知选帧（分段网格+抖动不变，段内挑清晰/非黑/不重复的一帧） |
| `fuse_condense.py` · `fuse_verify.py` | ④b1 跨遍规则计票 · ⑤ 画面验真（GroundingDINO+RapidOCR，缺件降级为 unverified） |
| `split_views.py` · `recap_diag.py` · `score_alignment.py` | A′ 句切 · 坍缩诊断（独立采样的哨兵）· SigLIP 对齐打分 |
| `make_*.py` · `extract_examples.py` | 产物：全景 TSV、离线评审包、A/B 并排对比、表格总结页、报告例子 |
| `apply_buckets.py` · `analyze_s1.py` · `check_s2.py` · `check_trans.py` · `aggregate_report.py` · `profile_jsonl.py` | 审计各阶段的分桶、体检与报告 |

单测在仓库根的 `tests/`（不随脚本搬走，`pytest tests/` 从根目录一把跑）：
`test_recap_frame_pick` · `test_vlm_prompts` · `test_prompt_docs_in_sync` ·
`test_slot_parse` · `test_fuse_condense` · `test_split_views` ·
`test_fuse_verify_helpers` · `test_summary_table`。

> `test_prompt_docs_in_sync` 把 `docs/ab-pipeline_zh.md` §5 的 prompt 全文与
> `scripts/vlm_audit.py` 逐字钉死——改了代码里的 prompt 就必须同步文档，否则单测红。

## 与仓库其他部分的关系

- 服务器⇄Claude 的交接通道在 [`notes/live/`](../notes/live/)（不只谈重述，故未并入本目录），
  约定见 [`notes/git-handoff-workflow.md`](../notes/git-handoff-workflow.md)。
- 产物（`outputs/`）、视频与标注**不进 git**，按 `docs/setup-v100.md` 在服务器上就地生成。
- 审计报告的对外精简版是 [`docs/final_report.md`](../docs/final_report.md)（在仓库根的 `docs/`，不在本目录）。

> 本目录 2026-08-05 由 `docs/` `notes/` `tools/data_audit/` 合并而来：旧路径一律
> `tools/data_audit/xxx` → `recaption/scripts/xxx`，文档另有改名。旧名对照用
> `git log --diff-filter=R --summary -1 <合并提交>` 可查。
