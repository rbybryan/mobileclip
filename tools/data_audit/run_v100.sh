#!/usr/bin/env bash
# InternVid-zh v0.1 数据审查一键脚本 —— 8x V100 版（只做审查 S0–S3，不改数据）
#
# 用法（DATA/CLIPS/OUT 默认即本机 101vector 的 1 万条抽样，无需 export；
#       逐条执行，s1/s2 是后台任务，用 status 等到 100% 再进下一步）：
#   bash tools/data_audit/run_v100.sh smoke      # ① 冒烟（必须先跑，~10min）
#   bash tools/data_audit/run_v100.sh s0         # ② 规则画像（10k 秒级）
#   bash tools/data_audit/run_v100.sh s0-show    # ②b 按旗标类别拆分+打印实例（SHOW_N=5 可调）
#   bash tools/data_audit/run_v100.sh s1         # ③ 嵌入打分（10k 8 卡 ~10–30min，可续跑）
#   bash tools/data_audit/run_v100.sh status     # 随时看 S1/S2 进度
#   bash tools/data_audit/run_v100.sh s2-sample  # ④ 抽样（默认参数在 10k 上=全覆盖）
#   bash tools/data_audit/run_v100.sh s2         # ⑤ 8 卡 VLM 审核全部 10k（HF 后端，~2–3h）
#   JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge  # ⑥ 云端裁判（可选，需 API key）
#   bash tools/data_audit/run_v100.sh s3         # ⑦ 校准+分桶+诊断报告 report.md
# 跑其他数据（如全量挂载后）：export DATA/CLIPS/OUT 覆盖默认值即可。
#
# V100 三个硬件差异已在脚本/工具内处理：
#   1. Volta 无 bf16      -> score_alignment.py / vlm_audit_hf.py 自动落到 fp16
#   2. vLLM V1 不支持 Volta -> S2 用 transformers 后端 (vlm_audit_hf.py) 8 卡数据并行
#   3. 235B 裁判本地放不下(8x32GB=256GB < 权重) -> s2-judge 走 OpenAI 兼容云端 API
set -euo pipefail
cd "$(dirname "$0")/../.."   # 仓库根目录

# ---------------- 配置（可用环境变量覆盖） ----------------
# 默认即本机（101vector）的 1 万条抽样——不 export 任何变量也能直接跑
DATA=${DATA:-/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl}
CLIPS=${CLIPS:-/home/vca1/datasets/internvid_zh_10k/videos}   # 视频根目录（video 字段相对它）
OUT=${OUT:-outputs/audit_10k}
NGPU=${NGPU:-8}
S1_FRAMES=${S1_FRAMES:-4}
S2_FRAMES=${S2_FRAMES:-8}
S2_MODEL=${S2_MODEL:-Qwen/Qwen3-VL-8B-Instruct}   # 16GB 卡改 Qwen/Qwen3-VL-4B-Instruct
# 抽样规模（V100 吞吐低于 A100，默认略收缩；嫌慢再调小 PER_BUCKET）
PER_BUCKET=${PER_BUCKET:-3000}; FLAGGED=${FLAGGED:-10000}; LOW_XLING=${LOW_XLING:-5000}
# 云端裁判（示例为 DashScope 兼容模式；OpenRouter 等任何 OpenAI 兼容端点均可）
JUDGE_API_BASE=${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}
JUDGE_API_KEY=${JUDGE_API_KEY:-}
JUDGE_MODEL=${JUDGE_MODEL:-qwen3-vl-235b-a22b-instruct}
JUDGE_N=${JUDGE_N:-2000}

mkdir -p "$OUT"
# 快速失败守卫：变量拼错（如 $SAMPLE 未 export 导致 DATA=/xxx.jsonl）时
# 立即报清晰错误，而不是让后续步骤连锁崩溃
need_data() { [ -f "$DATA" ] || { echo "错误：DATA 文件不存在：'$DATA'
  （若用了 \$SAMPLE 等变量，确认已在当前 shell export；本机默认无需 export）"; exit 1; }; }
need_clips() { [ -d "$CLIPS" ] || { echo "错误：CLIPS 目录不存在：'$CLIPS'"; exit 1; }; }
# 依赖预检：最常见的失败是跑在 conda (base) 而不是 venv 里（提示符应为 (audit_env)）
need_py() {
  missing=$(python -c 'import importlib.util as u, sys
print(" ".join(m for m in sys.argv[1:] if u.find_spec(m) is None))' \
    numpy torch transformers sentence_transformers decord PIL requests)
  if [ -n "$missing" ]; then
    echo "错误：当前 Python（$(command -v python)）缺依赖：$missing"
    echo "  多半是没激活虚拟环境（提示符还是 (base)）。先执行："
    echo "    source ~/audit_env/bin/activate"
    echo "  若激活后仍缺，按 SETUP_V100.md §0 安装："
    echo "    pip install \"numpy<2\" \"transformers>=4.57\" sentence-transformers decord pillow requests"
    exit 1
  fi
}
expected_total() { wc -l < "$DATA"; }
s1_lines() { cat "$OUT"/s1_scores.*.jsonl 2>/dev/null | wc -l; }

case "${1:-}" in
# ---------------------------------------------------------------- ① 冒烟
smoke)
  need_data; need_clips; need_py
  echo "== S0 冒烟：前 2 万条，确认字段探测 =="
  python tools/data_audit/profile_jsonl.py --input "$DATA" --limit 20000 \
    --out-summary "$OUT/smoke_s0.json" --out-flags "$OUT/smoke_s0_flags.jsonl"
  echo "== S1 冒烟：前 2000 条，确认视频路径与 fp16 推理 =="
  rm -f "$OUT/smoke_s1.0.jsonl"
  CUDA_VISIBLE_DEVICES=0 python tools/data_audit/score_alignment.py \
    --input "$DATA" --video-root "$CLIPS" --output "$OUT/smoke_s1.{shard}.jsonl" \
    --limit 2000 --frames "$S1_FRAMES" --batch-clips 16
  python - "$OUT/smoke_s1.0.jsonl" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
err=sum('error' in r for r in rows)
ok=[r['sig_zh_mean'] for r in rows if r.get('sig_zh_mean') is not None]
print(f"decode error: {err}/{len(rows)} ({err/max(1,len(rows)):.1%})  <- 应 <2%")
print(f"sig_zh_mean 非零样本: {len(ok)}, 中位数≈{sorted(ok)[len(ok)//2] if ok else 'N/A'}")
PY
  echo "== S2 冒烟：GPU0 审 8 条（首次会下载 $S2_MODEL） =="
  [ -s "$OUT/smoke_s1.0.jsonl" ] || { echo "S1 冒烟无产物，先解决上一步"; exit 1; }
  python tools/data_audit/make_audit_sample.py --input "$DATA" \
    --scores "$OUT/smoke_s1.*.jsonl" --flags "$OUT/smoke_s0_flags.jsonl" \
    --per-bucket 2 --buckets 4 --flagged 4 --low-xling 4 \
    --output "$OUT/smoke_s2_sample.jsonl"
  rm -f "$OUT/smoke_s2_verdicts.0.jsonl"
  CUDA_VISIBLE_DEVICES=0 python tools/data_audit/vlm_audit_hf.py \
    --input "$OUT/smoke_s2_sample.jsonl" --video-root "$CLIPS" \
    --model "$S2_MODEL" --frames "$S2_FRAMES" --limit 8 \
    --output "$OUT/smoke_s2_verdicts.{shard}.jsonl"
  head -2 "$OUT/smoke_s2_verdicts.0.jsonl"
  echo "== 冒烟通过。检查上面 [keys] 探测的字段名；不对则在本脚本各命令上补 --video-key/--zh-key/--en-key =="
  ;;
# ---------------------------------------------------------------- ② S0 全量
s0)
  need_data
  python tools/data_audit/profile_jsonl.py --input "$DATA" \
    --out-summary "$OUT/s0_summary.json" --out-flags "$OUT/s0_flags.jsonl"
  echo "== 决策门：拒答+未翻译+指令泄漏合计 >5% 先修翻译管线；重复率 >10% 记入报告 =="
  echo "== 查看: python -m json.tool $OUT/s0_summary.json | head -60 =="
  echo "== 看各类命中的具体例子: bash $0 s0-show =="
  ;;
# ------------------------------------------------- ②b 旗标实例展示（stdlib，无需 venv）
s0-show)
  FLAGS_FILE="$OUT/s0_flags.jsonl"
  [ -f "$FLAGS_FILE" ] || FLAGS_FILE="$OUT/smoke_s0_flags.jsonl"
  [ -f "$FLAGS_FILE" ] || { echo "未找到 $OUT/s0_flags.jsonl（先跑: bash $0 s0）"; exit 1; }
  python tools/data_audit/show_flags.py --flags "$FLAGS_FILE" \
    --out-dir "$OUT/s0_flag_split" --per-flag "${SHOW_N:-${2:-5}}" --width 120
  echo "每类完整清单在 $OUT/s0_flag_split/<flag>.jsonl，可直接喂给 Gradio 或 less 细看；"
  echo "多看几条: SHOW_N=20 bash $0 s0-show；只看部分类别加 --only 直接调 show_flags.py"
  ;;
# ---------------------------------------------------------------- ③ S1 全量 8 卡
s1)
  need_data; need_clips; need_py
  # 双机对半加速：V100 机 S1_TOTAL_SHARDS=16 S1_SHARD_OFFSET=0 bash $0 s1
  #              H100 机 S1_TOTAL_SHARDS=16 S1_SHARD_OFFSET=8 …（或手动循环）
  TOT=${S1_TOTAL_SHARDS:-$NGPU}; OFF=${S1_SHARD_OFFSET:-0}
  for i in $(seq 0 $((NGPU-1))); do
    sid=$((i+OFF))
    CUDA_VISIBLE_DEVICES=$i nohup python tools/data_audit/score_alignment.py \
      --input "$DATA" --video-root "$CLIPS" --output "$OUT/s1_scores.{shard}.jsonl" \
      --num-shards "$TOT" --shard-id "$sid" --frames "$S1_FRAMES" --batch-clips 16 \
      > "$OUT/s1_shard$sid.log" 2>&1 &
    echo "shard $sid/$TOT -> GPU$i (log: $OUT/s1_shard$sid.log)"
  done
  echo "已后台启动。中断后重跑本命令即可续跑；进度: bash $0 status"
  if [ "$TOT" -gt "$NGPU" ]; then
    echo "注意：双机分片下本机 status 只会到 ~$((NGPU*100/TOT))%，需两机产物汇总"
  fi
  ;;
# ---------------------------------------------------------------- 进度
status)
  need_data
  total=$(expected_total)
  s1=$(s1_lines)
  echo "S1: $s1 / $total ($(python3 -c "print(f'{$s1/$total:.1%}')"))"
  s2=$(cat "$OUT"/s2_verdicts.*.jsonl 2>/dev/null | wc -l)
  s2t=$( [ -f "$OUT/s2_sample.jsonl" ] && wc -l < "$OUT/s2_sample.jsonl" || echo 0)
  echo "S2: $s2 / $s2t"
  tail -n 2 "$OUT"/s1_shard*.log 2>/dev/null | grep -a progress | tail -4 || true
  ;;
# ---------------------------------------------------------------- ④ 抽样
s2-sample)
  need_data
  [ "$(s1_lines)" -ge "$(( $(expected_total) * 95 / 100 ))" ] || \
    { echo "S1 完成度不足 95%，先等 S1（bash $0 status）"; exit 1; }
  python tools/data_audit/make_audit_sample.py --input "$DATA" \
    --scores "$OUT/s1_scores.*.jsonl" --flags "$OUT/s0_flags.jsonl" \
    --per-bucket "$PER_BUCKET" --flagged "$FLAGGED" --low-xling "$LOW_XLING" \
    --output "$OUT/s2_sample.jsonl"
  wc -l "$OUT/s2_sample.jsonl"
  ;;
# ---------------------------------------------------------------- ⑤ S2 8 卡 HF 审核
s2)
  need_clips; need_py
  [ -f "$OUT/s2_sample.jsonl" ] || { echo "先跑 s2-sample"; exit 1; }
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i nohup python tools/data_audit/vlm_audit_hf.py \
      --input "$OUT/s2_sample.jsonl" --video-root "$CLIPS" --model "$S2_MODEL" \
      --frames "$S2_FRAMES" --num-shards "$NGPU" --shard-id "$i" \
      --output "$OUT/s2_verdicts.{shard}.jsonl" \
      > "$OUT/s2_shard$i.log" 2>&1 &
    echo "shard $i -> GPU$i (log: $OUT/s2_shard$i.log)"
  done
  echo "已后台启动（可续跑）。进度: bash $0 status"
  ;;
# ---------------------------------------------------------------- ⑥ 云端裁判（可选）
s2-judge)
  [ -n "$JUDGE_API_KEY" ] || { echo "export JUDGE_API_KEY=... （DashScope/OpenRouter 等）"; exit 1; }
  need_clips; need_py
  head -n "$JUDGE_N" "$OUT/s2_sample.jsonl" > "$OUT/s2_judge_sample.jsonl"
  python tools/data_audit/vlm_audit.py --input "$OUT/s2_judge_sample.jsonl" \
    --video-root "$CLIPS" --api-base "$JUDGE_API_BASE" --api-key "$JUDGE_API_KEY" \
    --model "$JUDGE_MODEL" --frames "$S2_FRAMES" --concurrency 8 \
    --output "$OUT/s2_verdicts_judge.jsonl"
  python - "$OUT" <<'PY'
import glob, json, sys
out = sys.argv[1]
judge = {json.loads(l)["video"]: json.loads(l).get("verdict", {})
         for l in open(f"{out}/s2_verdicts_judge.jsonl")}
agree = tot = 0
for p in glob.glob(f"{out}/s2_verdicts.*.jsonl"):
    for l in open(p):
        r = json.loads(l); v = r.get("verdict", {})
        j = judge.get(r["video"])
        if j and "action" in v and "action" in j:
            tot += 1; agree += (v["action"] == j["action"])
print(f"8B vs 裁判 action 一致率: {agree}/{tot} = {agree/max(1,tot):.1%}  <- 决策门 ≥75%")
PY
  ;;
# ---------------------------------------------------------------- ⑦ 报告
s3)
  python tools/data_audit/aggregate_report.py \
    --scores "$OUT/s1_scores.*.jsonl" --flags "$OUT/s0_flags.jsonl" \
    --verdicts "$OUT/s2_verdicts.*.jsonl" --target-precision 0.8 \
    --out-buckets "$OUT/buckets.jsonl" --out-report "$OUT/report.md"
  echo "== 诊断报告: $OUT/report.md（含问题率外推表；本轮到此为止，不改数据） =="
  ;;
*)
  sed -n '2,/^set -euo/p' "$0" | grep -E '^#( |$)' | sed 's/^# \?//'
  ;;
esac
