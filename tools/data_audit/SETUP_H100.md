# 8×H100-80G 数据审查完整配置与执行手册

> 姊妹篇：V100 机器用 [`SETUP_V100.md`](SETUP_V100.md)（差异：torch 须
> cu118 构建、fp16、transformers 后端、裁判走云端）。
>
> 目标机器：8×H100 80GB，Driver 580.105.08（CUDA 13.0，向下兼容全部 cu12x
> wheel）；可联网但 PyPI/HF 直连不稳 → 全程走 aliyun pip 源 + hf-mirror/ModelScope。
> H100 结论：**用 vLLM 服务路径**（比 V100 的 HF 后端快一个量级），
> 235B 裁判用 FP8 版**本地 8 卡即可起**，不需要云端 API。
> 本手册只覆盖审查 S0–S3（产出诊断报告），不改数据。

## 0. 环境（一次性，~20 分钟）

```bash
# conda 源若报 TUNA 403，用 --override-channels 绕开
conda create -n audit python=3.10 -y --override-channels \
  -c https://mirrors.aliyun.com/anaconda/pkgs/main
conda activate audit
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 先装 vllm（会带上匹配的 torch cu12x），再补其余依赖
pip install "vllm>=0.11"
pip install "transformers>=4.57" sentence-transformers decord pillow numpy requests

# 验证：应打印 True 和 H100
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
# 权重镜像（写进 ~/.bashrc）：SigLIP2/LaBSE 体积小，首次运行自动经镜像下载
export HF_ENDPOINT=https://hf-mirror.com

# 两个大模型建议用 ModelScope 预下载（国内直连、可断点）
pip install modelscope
mkdir -p ckpts
modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
  --local_dir ckpts/Qwen3-VL-8B-Instruct
modelscope download --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --local_dir ckpts/Qwen3-VL-235B-A22B-FP8        # 裁判档，~235GB，磁盘留够
```

```bash
# 路径变量（每个新 shell 都要；或写进 ~/.bashrc）
export DATA=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
export CLIPS=/path/to/internvid/clips      # TODO: 视频根目录（video 字段相对它）
export OUT=outputs/audit_v0_1 && mkdir -p $OUT
cd /path/to/mobileclip仓库根目录
```

## 1. 冒烟（~10 分钟，必做）

```bash
# S0 前 2 万条：确认 stderr 打印的 "[keys] video=... zh=... en=..." 字段正确
python tools/data_audit/profile_jsonl.py --input $DATA --limit 20000 \
  --out-summary $OUT/smoke_s0.json --out-flags $OUT/smoke_s0_flags.jsonl

# S1 前 2000 条：确认视频路径与解码（H100 自动用 bf16）
CUDA_VISIBLE_DEVICES=0 python tools/data_audit/score_alignment.py \
  --input $DATA --video-root $CLIPS --output "$OUT/smoke_s1.{shard}.jsonl" \
  --limit 2000 --frames 4 --batch-clips 32
python -c "
import json;rows=[json.loads(l) for l in open('$OUT/smoke_s1.0.jsonl')]
err=sum('error' in r for r in rows);print(f'decode error: {err}/{len(rows)} (应<2%)')"
```
字段探测不对 → 后续所有命令加 `--video-key/--zh-key/--en-key`；
解码失败率高 → 先查 `$CLIPS` 路径拼接。

## 2. S0 规则画像（全量，CPU ~20 分钟）

```bash
python tools/data_audit/profile_jsonl.py --input $DATA \
  --out-summary $OUT/s0_summary.json --out-flags $OUT/s0_flags.jsonl
python -m json.tool $OUT/s0_summary.json | head -60
```
**决策门**：拒答+未翻译+指令泄漏合计 >5% → 翻译管线要整体回炉；
重复 caption 率 >10% → 假负例问题记为报告重点。

## 3. S1 嵌入打分（全量 10M，8 卡 ~1 天，可断点续跑）

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i nohup python tools/data_audit/score_alignment.py \
    --input $DATA --video-root $CLIPS --output "$OUT/s1_scores.{shard}.jsonl" \
    --num-shards 8 --shard-id $i --frames 4 --batch-clips 32 \
    > $OUT/s1_shard$i.log 2>&1 &
done
# 进度（总数应逼近 10M）：
watch -n 60 'cat outputs/audit_v0_1/s1_scores.*.jsonl | wc -l'
```
瓶颈在 CPU 视频解码而非 H100 算力；中断后重跑同一命令自动续。
**验收点**：8 分片行数之和 = 输入行数；`error` 率与冒烟一致。

## 4. S2 分层抽样 + VLM 审核（~4.5 万条）

```bash
# 4.1 抽样（S1 完成 ≥95% 后执行）
python tools/data_audit/make_audit_sample.py --input $DATA \
  --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --per-bucket 3000 --flagged 10000 --low-xling 5000 \
  --output $OUT/s2_sample.jsonl
wc -l $OUT/s2_sample.jsonl

# 4.2 主力档：Qwen3-VL-8B，8 卡数据并行服务（45k 条约 1–2 小时）
vllm serve ckpts/Qwen3-VL-8B-Instruct --port 8000 \
  --data-parallel-size 8 --limit-mm-per-prompt image=8 --max-model-len 16384 &
# （若你的 vllm 版本不支持 --data-parallel-size，就去掉该参数单卡服务，
#   CUDA_VISIBLE_DEVICES=0，约 8–13 小时，同样能跑完）
python tools/data_audit/vlm_audit.py --input $OUT/s2_sample.jsonl \
  --video-root $CLIPS --api-base http://localhost:8000/v1 \
  --model ckpts/Qwen3-VL-8B-Instruct --frames 8 --concurrency 64 \
  --output $OUT/s2_verdicts.jsonl
# 跑完关掉服务释放显存：
pkill -f "vllm serve ckpts/Qwen3-VL-8B-Instruct"
```

```bash
# 4.3 裁判档：235B-A22B FP8 本地 8 卡（2k 条约 1–2 小时）
vllm serve ckpts/Qwen3-VL-235B-A22B-FP8 --port 8001 \
  --tensor-parallel-size 8 --limit-mm-per-prompt image=8 --max-model-len 32768 &
head -n 2000 $OUT/s2_sample.jsonl > $OUT/s2_judge_sample.jsonl
python tools/data_audit/vlm_audit.py --input $OUT/s2_judge_sample.jsonl \
  --video-root $CLIPS --api-base http://localhost:8001/v1 \
  --model ckpts/Qwen3-VL-235B-A22B-FP8 --frames 8 --concurrency 16 \
  --output $OUT/s2_verdicts_judge.jsonl
pkill -f "vllm serve ckpts/Qwen3-VL-235B-A22B-FP8"

# 4.4 一致率检查（决策门 ≥75%，不达标先改 rubric 再重跑 S2）
python - <<'PY'
import json, os
out = os.environ["OUT"]
judge = {json.loads(l)["video"]: json.loads(l).get("verdict", {})
         for l in open(f"{out}/s2_verdicts_judge.jsonl")}
agree = tot = 0
for l in open(f"{out}/s2_verdicts.jsonl"):
    r = json.loads(l); v = r.get("verdict", {}); j = judge.get(r["video"])
    if j and "action" in v and "action" in j:
        tot += 1; agree += (v["action"] == j["action"])
print(f"8B vs 235B action 一致率: {agree}/{tot} = {agree/max(1,tot):.1%}")
PY

# 4.5 人工抽检 500 条
shuf -n 500 $OUT/s2_verdicts.jsonl > $OUT/human_check_500.jsonl
```

## 5. S3 校准 + 全量分桶 + 诊断报告（CPU 分钟级）

```bash
python tools/data_audit/aggregate_report.py \
  --scores "$OUT/s1_scores.*.jsonl" --flags $OUT/s0_flags.jsonl \
  --verdicts "$OUT/s2_verdicts.jsonl" --target-precision 0.8 \
  --out-buckets $OUT/buckets.jsonl --out-report $OUT/report.md
```
产出 `report.md`（各类问题率外推表 + sig_zh 分位校准表）和
`buckets.jsonl`（全量每条的问题归类）。**审查到此为止**；
后续数据优化（v0.2）见 `notes/internvid-zh-data-audit-plan.md` §5、§7.6+。

## 时间线汇总（8×H100）

| 步骤 | 耗时 | 卡数 |
|---|---|---|
| 环境 + 权重下载 | ~0.5–2 h（235B 下载看带宽） | — |
| 冒烟 | ~10 min | 1 |
| S0 全量 | ~20 min | CPU |
| S1 全量 10M | ~1 天（解码瓶颈） | 8 |
| S2 主力 45k | 1–2 h（dp=8）/ 8–13 h（单卡） | 8 / 1 |
| S2 裁判 2k | 1–2 h | 8 |
| S3 报告 | 分钟级 | CPU |

总计 **~1.5 天** 拿到全量诊断报告。备选：不想起 vLLM 服务时，
`run_v100.sh` 的 transformers 后端在 H100 上同样能跑（自动 bf16），只是慢。
