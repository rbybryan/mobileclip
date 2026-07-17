# InternVid-zh 数据审核工具集（配套 `notes/internvid-zh-data-audit-plan.md`）

对 `aligned_internvid_zh_translated_retry_merged.jsonl`（10M 条，字段自动探测，
兼容 `video/caption/caption_zh` 等常见命名）做四阶段质量审核：

| 阶段 | 脚本 | 覆盖 | 算力 |
|---|---|---|---|
| S0 规则画像 | `profile_jsonl.py` | 全量 10M | CPU，~20 分钟 |
| S1 嵌入打分 | `score_alignment.py` | 全量 10M | SigLIP2，~100–200 GPU·h |
| S2 VLM 审核 | `make_audit_sample.py` + `vlm_audit.py`（vLLM 服务，Ampere+）或 `vlm_audit_hf.py`（transformers 后端，V100 可用） | 分层抽样 ~4–5 万 | Qwen3-VL-8B，~15–30 GPU·h |
| S3 校准分桶 | `aggregate_report.py` | 全量外推 | CPU |
| S4 落地 v0.2 | `apply_buckets.py` | 全量 | CPU |

## 按机器选手册（两台服务器各一份，均可独立跑完全流程）

| 机器 | 手册 | S2 路径 | 全流程耗时 |
|---|---|---|---|
| 8×H100-80G | **[`SETUP_H100.md`](SETUP_H100.md)** | vLLM 服务（8B dp=8 + 235B-FP8 裁判本地 tp8） | ~1.5 天 |
| 8×V100-32G | **[`SETUP_V100.md`](SETUP_V100.md)** | transformers 后端 `run_v100.sh`（无 bf16/vLLM；torch 须用 cu118 构建；裁判走云端 API） | ~2–2.5 天 |

两台都可用时的分工建议见 `SETUP_V100.md` 末节：S1（解码瓶颈）放 V100，
S2 主力+裁判（卡型差距最大）放 H100。

## 8×V100 快速上手（一键脚本，详细版见 SETUP_V100.md）

V100（Volta）无 bf16、且新版 vLLM（Qwen3-VL 所需）不支持 Volta，
所以 S1/S2 自动用 fp16、S2 用 transformers 后端 `vlm_audit_hf.py`
8 卡数据并行（输出格式与 `vlm_audit.py` 相同）：

```bash
export CLIPS=/path/to/internvid/clips        # 唯一必改项；DATA/OUT 有默认值
bash tools/data_audit/run_v100.sh smoke      # 冒烟：字段探测/解码/模型下载
bash tools/data_audit/run_v100.sh s0         # 全量规则画像（CPU）
bash tools/data_audit/run_v100.sh s1         # 全量嵌入打分（8 卡后台，可续跑）
bash tools/data_audit/run_v100.sh status     # 进度
bash tools/data_audit/run_v100.sh s2-sample  # 分层抽样
bash tools/data_audit/run_v100.sh s2         # 8 卡 VLM 审核（HF 后端）
export JUDGE_API_KEY=sk-...                  # 可选：云端 235B 裁判 2k 条
bash tools/data_audit/run_v100.sh s2-judge
bash tools/data_audit/run_v100.sh s3         # 诊断报告 report.md + buckets.jsonl
```

16GB 的 V100 需 `export S2_MODEL=Qwen/Qwen3-VL-4B-Instruct`。
若 fp16 输出乱码/NaN（罕见的 bf16→fp16 溢出），改用 4B 模型加 `--dtype fp32`。

## 快速上手（A100/H100 等 Ampere+ 机器，vLLM 服务方式）

```bash
DATA=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
CLIPS=/path/to/internvid/clips          # 视频根目录（video 字段相对于它）
OUT=outputs/audit_v0_1
mkdir -p $OUT

# ---- S0：先跑纯文本画像（无 GPU，先看翻译痕迹/未翻译/重复率）----
python tools/data_audit/profile_jsonl.py \
  --input $DATA --out-summary $OUT/s0_summary.json --out-flags $OUT/s0_flags.jsonl

# ---- S1：全量 SigLIP2 打分（8 卡并行示例；可断点续跑）----
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python tools/data_audit/score_alignment.py \
    --input $DATA --video-root $CLIPS \
    --output "$OUT/s1_scores.{shard}.jsonl" \
    --num-shards 8 --shard-id $i --frames 4 --batch-clips 16 &
done; wait

# ---- S2：分层抽样 + VLM 审核 ----
python tools/data_audit/make_audit_sample.py \
  --input $DATA --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --per-bucket 3000 --flagged 10000 --low-xling 5000 \
  --output $OUT/s2_sample.jsonl

vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 \
  --limit-mm-per-prompt image=8 --max-model-len 16384 &
python tools/data_audit/vlm_audit.py \
  --input $OUT/s2_sample.jsonl --video-root $CLIPS \
  --api-base http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct \
  --frames 8 --concurrency 32 --output $OUT/s2_verdicts.jsonl

# ---- S3：校准阈值 + 全量分桶 + 报告 ----
python tools/data_audit/aggregate_report.py \
  --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --verdicts $OUT/s2_verdicts.jsonl \
  --out-buckets $OUT/buckets.jsonl --out-report $OUT/report.md

# ---- S4：分桶落地成 v0.2 数据 + 重翻/重写任务队列 ----
python tools/data_audit/apply_buckets.py --input $DATA \
  --buckets $OUT/buckets.jsonl --verdicts $OUT/s2_verdicts.jsonl \
  --out-data $OUT/v0_2/internvid_zh_v0_2.jsonl \
  --out-retranslate $OUT/v0_2/retranslate_tasks.jsonl \
  --out-recaption $OUT/v0_2/recaption_tasks.jsonl \
  --out-drop $OUT/v0_2/dropped.jsonl
```

先用 `--limit 20000`（S0/S1）和 `--limit 200`（S2）各跑一次冒烟，确认字段探测
（脚本会打印 `[keys] video=... zh=... en=...`）和视频路径拼接正确，再放全量。

## 网络受限机器（国内集群常见：PyPI/conda 被 reset、HF 不通）

1. **S0 零依赖**：`profile_jsonl.py` 只用标准库，任何现成 Python 直接跑，
   不用等环境装好。
2. **跳过 conda，复用训练环境的 torch/decord**（省 700MB+ 下载，且 conda 源
   往往同样不通）：
   ```bash
   conda activate mobileviclip
   python -m venv --system-site-packages ~/audit_env && source ~/audit_env/bin/activate
   pip install -i https://mirrors.aliyun.com/pypi/simple --timeout 120 \
     "transformers>=4.57" sentence-transformers requests
   ```
   TUNA 超时就换 aliyun/ustc/sjtu 源；全不通说明机器要代理
   （`echo $https_proxy`，找管理员）。V100 路径不需要 vllm，别装。
   conda 报 `HTTP 403 FORBIDDEN ... mirrors.tuna...anaconda/pkgs/main`
   是 condarc 里的 TUNA anaconda 源被服务端拒绝：要么按上面方案跳过 conda，
   要么 `conda create -n audit python=3.10 -y --override-channels -c
   https://mirrors.aliyun.com/anaconda/pkgs/main` 绕开；torch 按驱动选构建
   （`nvidia-smi` Driver ≥535 用默认 cu12 轮子，更老的 V100 集群用
   `--index-url https://download.pytorch.org/whl/cu118`）。
3. **完全离线**：在有网的同架构机器 `pip download -d audit_wheels
   --platform manylinux2014_x86_64 --python-version 310 --only-binary=:all:
   "transformers>=4.57" sentence-transformers requests`，拷过去
   `pip install --no-index --find-links audit_wheels ...`。
4. **模型权重**：`export HF_ENDPOINT=https://hf-mirror.com`（写进 `~/.bashrc`），
   或用 ModelScope 预下载后传本地路径：
   ```bash
   pip install modelscope
   modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir ckpts/Qwen3-VL-8B-Instruct
   export S2_MODEL=ckpts/Qwen3-VL-8B-Instruct   # run_v100.sh 会直接用本地路径
   ```
   SigLIP2 / LaBSE 同理（`--model`/`--xlingual-model` 均接受本地目录）。
5. 训练环境 torch 偏老时 S1（SigLIP2）通常仍可跑；仅当 S2 加载 Qwen3-VL
   报 torch 版本错误时，再经镜像/离线 wheel 升 `torch>=2.4`（V100 选 cu118）。

## 依赖

在仓库现有环境（含 `decord`、`torch`）之上：`transformers>=4.57`
（SigLIP2 需 ≥4.49，Qwen3-VL 需 ≥4.57）、`sentence-transformers`（LaBSE，
可用 `--xlingual-model ''` 关闭）、`requests`；仅 Ampere+ 走 vLLM 服务方式时
需要 `vllm>=0.11`（建议单独 conda 环境，与训练环境隔离；V100 不需要装）。

## 输出

- `s0_summary.json` / `s0_flags.jsonl`：翻译痕迹、未翻译、重复 caption 等规则命中
- `s1_scores.*.jsonl`：每条 `sig_zh_mean/sig_en_mean/xling_cos`（+解码错误）
- `s2_verdicts.jsonl`：VLM 结构化判定（对齐/翻译/粒度/时序 + 建议动作 + 候选新中文短 caption）
- `buckets.jsonl`：全量每条 → `KEEP / FIX_TEXT / RETRANSLATE / RECAPTION / DROP`
- `report.md`：问题率外推表 + 阈值校准表，直接可贴进周报
- `v0_2/internvid_zh_v0_2.jsonl`：训练可用的 v0.2 数据
  （`{"video","caption","captions":[...]}`，loader 直接兼容）
  + 重翻/重写任务队列 + 剔除清单

完整分步操作手册（含环境准备、冒烟验证、每步验收点/决策门、v0.2→v0.3 节奏）
见 `notes/internvid-zh-data-audit-plan.md` §5–§7。
