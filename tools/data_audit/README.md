# InternVid-zh 数据审核工具集（配套 `notes/internvid-zh-data-audit-plan.md`）

对 `aligned_internvid_zh_translated_retry_merged.jsonl`（10M 条，字段自动探测，
兼容 `video/caption/caption_zh` 等常见命名）做四阶段质量审核：

| 阶段 | 脚本 | 覆盖 | 算力 |
|---|---|---|---|
| S0 规则画像 | `profile_jsonl.py` | 全量 10M | CPU，~20 分钟 |
| S1 嵌入打分 | `score_alignment.py` | 全量 10M | SigLIP2，~100–200 GPU·h |
| S2 VLM 审核 | `make_audit_sample.py` + `vlm_audit.py` | 分层抽样 ~4–5 万 | Qwen3-VL-8B，~15–30 GPU·h |
| S3 校准分桶 | `aggregate_report.py` | 全量外推 | CPU |

## 快速上手（在有数据盘和 GPU 的机器上）

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
```

先用 `--limit 20000`（S0/S1）和 `--limit 200`（S2）各跑一次冒烟，确认字段探测
（脚本会打印 `[keys] video=... zh=... en=...`）和视频路径拼接正确，再放全量。

## 依赖

在仓库现有环境（含 `decord`、`torch`）之上：`transformers>=4.49`（SigLIP2）、
`sentence-transformers`（LaBSE，可用 `--xlingual-model ''` 关闭）、`requests`；
S2 服务端需要 `vllm>=0.11`（建议单独 conda 环境，与训练环境隔离）。

## 输出

- `s0_summary.json` / `s0_flags.jsonl`：翻译痕迹、未翻译、重复 caption 等规则命中
- `s1_scores.*.jsonl`：每条 `sig_zh_mean/sig_en_mean/xling_cos`（+解码错误）
- `s2_verdicts.jsonl`：VLM 结构化判定（对齐/翻译/粒度/时序 + 建议动作 + 候选新中文短 caption）
- `buckets.jsonl`：全量每条 → `KEEP / FIX_TEXT / RETRANSLATE / RECAPTION / DROP`
- `report.md`：问题率外推表 + 阈值校准表，直接可贴进周报

分桶之后的 v0.2 数据生产（清洗、重翻、按画面重写中文短 caption）见
`notes/internvid-zh-data-audit-plan.md` §5。
