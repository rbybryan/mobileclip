# 8×V100-32G 数据审查完整配置与执行手册

> 目标机器：8×V100-SXM2-32GB（Volta，CC 7.0）。与 H100 手册
> （`SETUP_H100.md`）互为姊妹篇——两台机器都可独立跑完全流程；
> H100 空闲时优先用 H100（S2 快一个量级）。
> 本手册只覆盖审查 S0–S3（产出诊断报告），不改数据。
>
> **V100 三个硬性差异**（工具已内置处理，但装环境时必须知道）：
> 1. **新 torch wheel 不支持 Volta**：cu126/cu128 构建（torch≥2.7）已删掉
>    CC 7.0 内核，import 时警告、跑算子直接报 `no kernel image`。
>    必须装 **cu118/cu121 构建（torch 2.4/2.5 系）**。
> 2. **无 bf16** → S1/S2 自动落 fp16（脚本 `--dtype` 可强制）。
> 3. **vLLM V1 引擎不支持 Volta**（Qwen3-VL 又要求新版 vLLM）→ S2 不起
>    服务，用 transformers 后端 `vlm_audit_hf.py` 8 卡数据并行；
>    235B 裁判本地放不下（8×32GB < 权重 400GB+）→ 走云端 API。

## 0. 环境（一次性，~20 分钟）

```bash
# conda 源若报 TUNA 403，用 --override-channels 绕开
conda create -n audit python=3.10 -y --override-channels \
  -c https://mirrors.aliyun.com/anaconda/pkgs/main
conda activate audit
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# ⚠️ V100 关键一步：torch 必须装带 sm_70 内核的 cu118 构建，
#    不能用默认 PyPI 的新构建（cu126/cu128 无 Volta 支持）
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu118

pip install "transformers>=4.57" sentence-transformers decord pillow numpy requests
# 不要装 vllm —— V100 路径用不上

# 验证：必须打印 tensor(...) 和 V100 且无 "no kernel image" 报错
python -c "import torch; print(torch.zeros(1).cuda(), torch.__version__, \
  torch.cuda.get_device_name(0))"
```

```bash
# 权重镜像（写进 ~/.bashrc）
export HF_ENDPOINT=https://hf-mirror.com

# S2 模型预下载（32GB 卡用 8B；若是 16GB 卡改 Qwen3-VL-4B-Instruct）
pip install modelscope
mkdir -p ckpts
modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ckpts/Qwen3-VL-8B-Instruct
# SigLIP2 / LaBSE 体积小，S1 首次运行经 hf-mirror 自动下载即可
```

```bash
# 路径变量（每个新 shell 都要；或写进 ~/.bashrc）
export DATA=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
export CLIPS=/path/to/internvid/clips      # TODO: 视频根目录（video 字段相对它）
export OUT=outputs/audit_v0_1
export S2_MODEL=ckpts/Qwen3-VL-8B-Instruct
cd /path/to/mobileclip仓库根目录
```

以下 §1–§5 全部封装在 `tools/data_audit/run_v100.sh`（内置 8 卡 fan-out、
断点续跑、验收检查），每步给出一键命令；想看展开的等价命令可
`cat tools/data_audit/run_v100.sh` 对应分支。

## 1. 冒烟（~10 分钟，必做）

```bash
bash tools/data_audit/run_v100.sh smoke
```
做三件事：S0 前 2 万条（看 `[keys]` 字段探测）、S1 前 2000 条
（fp16 推理 + 解码失败率，应 <2%）、S2 在 GPU0 审 8 条（首次会下载模型）。
字段探测不对 → 各命令补 `--video-key/--zh-key/--en-key`。

## 2. S0 规则画像（全量，CPU ~20 分钟，可先于装环境执行）

```bash
bash tools/data_audit/run_v100.sh s0
python -m json.tool $OUT/s0_summary.json | head -60
```
S0 只用标准库——**任何现成 Python 都能跑**，不必等 torch 装好。
**决策门**：拒答+未翻译+指令泄漏合计 >5% → 翻译管线先回炉；
重复 caption 率 >10% → 假负例问题记为报告重点。

## 3. S1 嵌入打分（全量 10M，8 卡 ~1.5–2 天，可断点续跑）

```bash
bash tools/data_audit/run_v100.sh s1        # 8 卡后台启动
bash tools/data_audit/run_v100.sh status    # 随时看进度
```
瓶颈在 CPU 视频解码；中断后重跑 `s1` 自动续。
**验收点**：`status` 显示 100%；`error` 率与冒烟一致。

## 4. S2 分层抽样 + VLM 审核

```bash
# 4.1 抽样（S1 ≥95% 后执行；V100 慢可 export PER_BUCKET=2000 收缩）
bash tools/data_audit/run_v100.sh s2-sample

# 4.2 主力档：Qwen3-VL-8B，transformers 后端 8 卡数据并行（~半天）
bash tools/data_audit/run_v100.sh s2
bash tools/data_audit/run_v100.sh status

# 4.3 裁判档：云端 235B（任何 OpenAI 兼容端点；2k 条约几十元）
export JUDGE_API_KEY=sk-...          # DashScope / OpenRouter 等
bash tools/data_audit/run_v100.sh s2-judge   # 跑完自动打印一致率
# （替代方案：两台机器都在时，把裁判放 H100 本地 235B-FP8，
#   见 SETUP_H100.md §4.3，然后把 s2_verdicts_judge.jsonl 拷回来）

# 4.4 人工抽检 500 条
shuf -n 500 <(cat $OUT/s2_verdicts.*.jsonl) > $OUT/human_check_500.jsonl
```
**决策门**：8B vs 裁判 action 一致率 ≥75%，不达标先改 rubric 再重跑。
fp16 若出现乱码/NaN 输出（罕见）：`export S2_MODEL=Qwen/Qwen3-VL-4B-Instruct`
并在 `vlm_audit_hf.py` 加 `--dtype fp32` 重跑。

## 5. S3 校准 + 全量分桶 + 诊断报告（CPU 分钟级）

```bash
bash tools/data_audit/run_v100.sh s3
```
产出 `$OUT/report.md`（问题率外推表 + 校准表）和 `buckets.jsonl`。
**审查到此为止**；后续优化见 `notes/internvid-zh-data-audit-plan.md` §5、§7.6+。

## 时间线汇总（8×V100 vs 8×H100）

| 步骤 | 8×V100 | 8×H100（见 SETUP_H100.md） |
|---|---|---|
| 环境 + 权重 | ~0.5–1 h | ~0.5–2 h（多下 235B） |
| S0 全量 | ~20 min（CPU，两边一样） | ~20 min |
| S1 全量 10M | ~1.5–2 天 | ~1 天 |
| S2 主力 45k | ~半天（HF 后端） | 1–2 h（vLLM dp=8） |
| S2 裁判 2k | ~1 h（云端 API） | 1–2 h（本地 FP8 tp8） |
| S3 报告 | 分钟级 | 分钟级 |
| **总计** | **~2–2.5 天** | **~1.5 天** |

两台都可用时的推荐分工：S0 哪台闲跑哪台；S1 放 V100（解码瓶颈，卡型差距
小）；S2 主力 + 裁判放 H100（差距最大）；S3 在有全部产物的一台上跑。
