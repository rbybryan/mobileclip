#!/usr/bin/env bash
# InternVid-zh v0.1 数据审查一键脚本 —— 8x V100 版（只做审查 S0–S3，不改数据）
#
# 用法（DATA/CLIPS/OUT 默认即本机 101vector 的 1 万条抽样，无需 export；
#       逐条执行，s1/s2 是后台任务，用 status 等到 100% 再进下一步）：
#   bash tools/data_audit/run_v100.sh smoke      # ① 冒烟（必须先跑，~10min）
#   bash tools/data_audit/run_v100.sh s0         # ② 规则画像（10k 秒级）
#   bash tools/data_audit/run_v100.sh s0-show    # ②b 按旗标类别拆分+打印实例（SHOW_N=5 可调）
#   bash tools/data_audit/run_v100.sh s1         # ③ 嵌入打分（10k 8 卡 ~10–30min，可续跑）
#   bash tools/data_audit/run_v100.sh s1-report  # ③b S1 联合分析 -> s1_analysis.md（秒级）
#   bash tools/data_audit/run_v100.sh status     # 随时看 S1/S2 进度
#   bash tools/data_audit/run_v100.sh s2-sample  # ④ 抽样（默认参数在 10k 上=全覆盖）
#   bash tools/data_audit/run_v100.sh s2         # ⑤ 8 卡 VLM 审核全部 10k（HF 后端，~2–3h）
#   bash tools/data_audit/run_v100.sh s2-check   # ⑤b S2 判定体检 -> s2_health.md（秒级，S2 跑完必做）
#   bash tools/data_audit/run_v100.sh s2-trans   # ⑤c 文本-only 翻译重判（修 halo；不解码视频，比 s2 快）
#   bash tools/data_audit/run_v100.sh trans-check # ⑤d 重判体检（可中途跑：早判修复有没有生效）
#   bash tools/data_audit/run_v100.sh review       # ⑧ 生成三张人工抽检表（TSV，填完用 review-score 统计）
#   bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/review_better_zh.tsv
#   bash tools/data_audit/run_v100.sh s2-gran      # ⑨ 粒度计数式重判（验证 dense=0 是不是标签塌缩）
#   JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge  # ⑥ 云端裁判（可选，需 API key）
#   bash tools/data_audit/run_v100.sh judge-agree # ⑥b 一致率+Cohen's κ（可对外来裁判文件独立重算）
#   bash tools/data_audit/run_v100.sh xling-check # ⑥c 裁决 xling 阈值（s3 前必做）
#   bash tools/data_audit/run_v100.sh s3         # ⑦ 校准+分桶+诊断报告 report.md
#   bash tools/data_audit/run_v100.sh snapshot   # ⑩ 一条命令打印全部状态（贴给 Claude 用）
# 跑其他数据（如全量挂载后）：export DATA/CLIPS/OUT 覆盖默认值即可。
#
# V100 三个硬件差异已在脚本/工具内处理：
#   1. Volta 无 bf16      -> score_alignment.py / vlm_audit_hf.py 自动落到 fp16
#   2. vLLM V1 不支持 Volta -> S2 用 transformers 后端 (vlm_audit_hf.py) 8 卡数据并行
#   3. 235B 裁判本地放不下(8x32GB=256GB < 权重, 且 Volta 无 FP8) -> s2-judge 走
#      OpenAI 兼容端点。不想用云端商业 API 时改指向 H100 机自托管的同一份开源权重
#      (JUDGE_API_BASE=http://<H100_IP>:8001/v1)，判定完全一致；选型见
#      SETUP_JUDGE_API.md §0
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
# 权重镜像默认走 hf-mirror（直连 huggingface.co 在本网络会无限卡 0%）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# S2 模型自动探测：优先用 §0 预下载的本地权重（零联网）；没有才回退在线仓库名。
# 注意在线回退时 8 个分片会并发下载同一份 ~17GB 权重（共享缓存带文件锁，
# 实际近似串行），弱网上表现为全部卡在 "Fetching ... 0%"——强烈建议先预下载。
if [ -z "${S2_MODEL:-}" ]; then
  if ls ckpts/Qwen3-VL-8B-Instruct/*.safetensors >/dev/null 2>&1; then
    S2_MODEL=ckpts/Qwen3-VL-8B-Instruct
  else
    S2_MODEL=Qwen/Qwen3-VL-8B-Instruct   # 16GB 卡改 Qwen/Qwen3-VL-4B-Instruct
  fi
fi
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
    numpy torch torchvision transformers accelerate sentence_transformers decord PIL requests)
  if [ -n "$missing" ]; then
    echo "错误：当前 Python（$(command -v python)）缺依赖：$missing"
    echo "  多半是没激活虚拟环境（提示符还是 (base)）。先执行："
    echo "    source ~/audit_env/bin/activate"
    echo "  若激活后仍缺，按 SETUP_V100.md §0 安装："
    echo "    pip install \"numpy<2\" \"transformers>=4.57\" accelerate sentence-transformers decord pillow requests"
    exit 1
  fi
}
expected_total() { wc -l < "$DATA"; }
# 注意 { cat ... || true; }：无匹配文件时 cat 失败，在 set -o pipefail 下会把
# 整条管道判 1，再被 set -e 杀掉整个脚本（症状：status 一行都不输出就退出）
s1_lines() { { cat "$OUT"/s1_scores.*.jsonl 2>/dev/null || true; } | wc -l; }
s2_lines() { { cat "$OUT"/s2_verdicts.*.jsonl 2>/dev/null || true; } | wc -l; }
trans_lines() { { cat "$OUT"/s2_trans.*.jsonl 2>/dev/null || true; } | wc -l; }

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
# ---------------------------------------------------------------- ③.5 S1 联合分析
s1-report)
  # 只需 numpy（纯 CPU 读 jsonl），不要求 torch/decord 那套 GPU 依赖
  python -c 'import numpy' 2>/dev/null || {
    echo "错误：当前 Python（$(command -v python)）缺 numpy。"
    echo "  多半是没激活虚拟环境：source ~/audit_env/bin/activate"; exit 1; }
  ls "$OUT"/s1_scores.*.jsonl >/dev/null 2>&1 || \
    { echo "未找到 $OUT/s1_scores.*.jsonl（先跑: bash $0 s1）"; exit 1; }
  FLAGS_ARG=""
  if [ -f "$OUT/s0_flags.jsonl" ]; then
    FLAGS_ARG="--flags $OUT/s0_flags.jsonl"
  else
    echo "提示：没有 $OUT/s0_flags.jsonl，将跳过「高 xling = 未翻译」交叉核对"
    echo "      （补跑 bash $0 s0 后重跑本命令即可，秒级）"
  fi
  python tools/data_audit/analyze_s1.py --scores "$OUT/s1_scores.*.jsonl" \
    $FLAGS_ARG --out-md "$OUT/s1_analysis.md"
  ;;
# ---------------------------------------------------------------- 进度
status)
  need_data
  total=$(expected_total)
  s1=$(s1_lines)
  s1_alive=$(pgrep -fc score_alignment.py || true)
  s2_alive=$(pgrep -fc vlm_audit_hf.py || true)
  echo "S1: $s1 / $total ($(python3 -c "print(f'{$s1/$total:.1%}')"))  存活进程: $s1_alive"
  s2=$(s2_lines)
  s2t=$( [ -f "$OUT/s2_sample.jsonl" ] && wc -l < "$OUT/s2_sample.jsonl" || echo 0)
  echo "S2: $s2 / $s2t  存活进程: $s2_alive"
  echo "（刚启动时行数为 0 是正常的：各分片要先下载/加载模型权重，看下方日志尾行确认在动；"
  echo "  进程数 0 且未到 100% = 分片已退出，查对应日志的 Traceback 后重跑即可断点续跑）"
  if ls "$OUT"/s2_verdicts.*.jsonl >/dev/null 2>&1; then
    echo "---- S2 各分片已写行数（每审完 1 条实时 +1；V100 单条约 20–60s）----"
    for f in "$OUT"/s2_verdicts.*.jsonl; do echo "  $(wc -l < "$f")  $f"; done
  fi
  if ls "$OUT"/s2_trans.*.jsonl >/dev/null 2>&1; then
    tr_alive=$(pgrep -fc "vlm_audit_hf.py.*--mode trans" || true)
    echo "---- S2-trans（文本-only 翻译重判）: $(trans_lines) / $s2t  存活进程: $tr_alive ----"
    echo "  体检（可中途跑，早发现修复没生效就不用等）: bash $0 trans-check"
  fi
  echo "---- 各分片日志尾行 ----"
  tail -n 1 "$OUT"/s1_shard*.log "$OUT"/s2_shard*.log 2>/dev/null || true
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
  echo "[model] S2_MODEL=$S2_MODEL"
  if [ ! -d "$S2_MODEL" ]; then
    echo "警告：用的是在线仓库名（本地未发现 ckpts/Qwen3-VL-8B-Instruct 的 safetensors）。"
    echo "  8 个分片将并发下载 ~17GB 权重，弱网上会长时间卡 'Fetching 0%'；"
    echo "  建议 Ctrl-C 后先按 SETUP_V100.md §0 预下载到 ckpts/ 再跑本步。"
  fi
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i nohup python tools/data_audit/vlm_audit_hf.py \
      --input "$OUT/s2_sample.jsonl" --video-root "$CLIPS" --model "$S2_MODEL" \
      --frames "$S2_FRAMES" --num-shards "$NGPU" --shard-id "$i" \
      --output "$OUT/s2_verdicts.{shard}.jsonl" \
      > "$OUT/s2_shard$i.log" 2>&1 &
    echo "shard $i -> GPU$i (log: $OUT/s2_shard$i.log)"
  done
  echo "已后台启动（可续跑）。进度: bash $0 status"
  echo "跑完后先做体检再进下一步: bash $0 s2-check"
  ;;
# --------------------------------- ⑤c 文本-only 翻译重判（修 halo，见 DATA_ISSUES 问题 8）
s2-trans)
  need_py
  [ -f "$OUT/s2_sample.jsonl" ] || { echo "先跑 s2-sample"; exit 1; }
  echo "[model] S2_MODEL=$S2_MODEL （文本模式：不读视频、不解码、不看帧）"
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i nohup python tools/data_audit/vlm_audit_hf.py \
      --input "$OUT/s2_sample.jsonl" --video-root "$CLIPS" --model "$S2_MODEL" \
      --mode trans --num-shards "$NGPU" --shard-id "$i" \
      --max-new-tokens "${TRANS_MAX_TOKENS:-512}" \
      --output "$OUT/s2_trans.{shard}.jsonl" \
      > "$OUT/s2_trans_shard$i.log" 2>&1 &
    echo "shard $i -> GPU$i (log: $OUT/s2_trans_shard$i.log)"
  done
  echo "已后台启动（可续跑）。比 s2 快得多：S2 的瓶颈是视频解码，这一趟完全不解码。"
  echo "进度: for f in $OUT/s2_trans.*.jsonl; do wc -l \$f; done"
  echo "跑完后: bash $0 xling-check   # 会自动优先用 s2_trans.*.jsonl"
  echo "中途随时: bash $0 trans-check # 早判修复有没有生效，没生效就别等它跑完"
  ;;
# ------------------------------- ⑤d 重判体检（可中途跑：修复没生效就不必等跑完）
trans-check)
  ls "$OUT"/s2_trans.*.jsonl >/dev/null 2>&1 || \
    { echo "未找到 $OUT/s2_trans.*.jsonl（先跑: bash $0 s2-trans）"; exit 1; }
  # 用数组传参：`--main $OUT/s2_verdicts.*.jsonl` 不加引号会被 shell 展开成 8 个
  # 文件名，argparse 只收第一个、其余全变成 unrecognized arguments。
  TC_ARGS=(--trans "$OUT/s2_trans.*.jsonl")
  ls "$OUT"/s2_verdicts.*.jsonl >/dev/null 2>&1 && \
    TC_ARGS+=(--main "$OUT/s2_verdicts.*.jsonl")
  [ -f "$OUT/s2_sample.jsonl" ] && TC_ARGS+=(--sample "$OUT/s2_sample.jsonl")
  python3 tools/data_audit/check_trans.py "${TC_ARGS[@]}" \
    --out-md "$OUT/trans_health.md" || {
      echo; echo "== 重判体检未通过：先看上方结论，别把它喂给 xling-check =="; exit 1; }
  ;;
# ------------------------------------------------- ⑤b S2 判定体检（S2 跑完必做）
s2-check)
  # 纯 stdlib，不需要 numpy/torch——没激活虚拟环境也能跑
  ls "$OUT"/s2_verdicts.*.jsonl >/dev/null 2>&1 || \
    { echo "未找到 $OUT/s2_verdicts.*.jsonl（先跑: bash $0 s2）"; exit 1; }
  SC_ARGS=(--verdicts "$OUT/s2_verdicts.*.jsonl")
  [ -f "$OUT/s2_sample.jsonl" ] && SC_ARGS+=(--sample "$OUT/s2_sample.jsonl")
  python3 tools/data_audit/check_s2.py "${SC_ARGS[@]}" \
    --out-md "$OUT/s2_health.md" || {
      echo
      echo "== 体检未通过（详见上方「结论」与 $OUT/s2_health.md）：先处理再跑 xling-check/s3 =="
      exit 1; }
  echo
  echo "== 体检通过。下一步: bash $0 xling-check（裁决决策 1，阻塞 s3） =="
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
  exec "$0" judge-agree
  ;;
# ------------------------------------------- ⑥b 一致率 + Cohen's κ（可独立重算）
judge-agree)
  ls "$OUT"/s2_verdicts.*.jsonl >/dev/null 2>&1 || \
    { echo "未找到 $OUT/s2_verdicts.*.jsonl（先跑: bash $0 s2）"; exit 1; }
  [ -f "$OUT/s2_verdicts_judge.jsonl" ] || \
    { echo "未找到 $OUT/s2_verdicts_judge.jsonl"
      echo "  —— 跑 bash $0 s2-judge；或裁判在别的机器上跑完后把该文件拷回来"
      echo "     （见 SETUP_JUDGE_API.md §0 路线 A′「探针二不通时」）"; exit 1; }
  python tools/data_audit/judge_agreement.py \
    --main "$OUT/s2_verdicts.*.jsonl" --judge "$OUT/s2_verdicts_judge.jsonl" \
    --out-md "$OUT/judge_agreement.md"
  ;;
# ------------------------------------------------- ⑥.5 xling 阈值裁决（跑 s3 前）
xling-check)
  python -c 'import numpy' 2>/dev/null || {
    echo "错误：当前 Python（$(command -v python)）缺 numpy。"
    echo "  多半是没激活虚拟环境：source ~/audit_env/bin/activate"; exit 1; }
  # 优先用文本-only 重判的 trans_score：主力档的 trans_score 与 en_score 92.5%
  # 逐条同值（halo，DATA_ISSUES.md 问题 8），拿它算差率是自证循环。
  if ls "$OUT"/s2_trans.*.jsonl >/dev/null 2>&1; then
    XV="$OUT/s2_trans.*.jsonl"
    echo "[xling-check] 用文本-only 重判的 trans_score（s2_trans.*.jsonl）—— 独立测量，可信"
    # 别在半成品上校准：s2-trans 是后台 8 分片，刚起步时只有几十条
    if [ -f "$OUT/s2_sample.jsonl" ]; then
      have=$(trans_lines); want=$(wc -l < "$OUT/s2_sample.jsonl")
      alive=$(pgrep -fc "vlm_audit_hf.py.*--mode trans" || true)
      if [ "$have" -lt "$(( want * 95 / 100 ))" ]; then
        echo "⚠️ s2-trans 才写了 $have / $want 条（$(python3 -c "print(f'{$have/$want:.1%}')")）——**还没跑完**。"
        echo "   在半成品上校准出来的阈值没有意义。存活进程: $alive"
        [ "$alive" -gt 0 ] && echo "   还在跑，等着就行： bash $0 status" \
                           || echo "   ⛔ 进程已退出且未跑完：查 $OUT/s2_trans_shard*.log 的 Traceback，重跑 s2-trans 可续跑"
        echo "   想先看修复有没有生效： bash $0 trans-check"
        exit 1
      fi
    fi
  else
    XV="$OUT/s2_verdicts.*.jsonl"
    echo "⚠️ [xling-check] 未找到 $OUT/s2_trans.*.jsonl，退回主力档 s2_verdicts.*.jsonl。"
    echo "   主力档的 trans_score 与 en_score 92.5% 逐条同值（halo，见 DATA_ISSUES.md 问题 8），"
    echo "   **本次输出只能当参考，不足以裁决决策 1** —— 先跑: bash $0 s2-trans"
  fi
  ls $XV >/dev/null 2>&1 || \
    { echo "未找到 $OUT/s2_verdicts.*.jsonl（先跑完: bash $0 s2）"; exit 1; }
  python tools/data_audit/calibrate_xling.py \
    --verdicts "$XV" --scores "$OUT/s1_scores.*.jsonl" \
    --target-precision "${TARGET_PRECISION:-0.8}" \
    --out-md "$OUT/xling_calibration.md"
  echo
  echo "== 把结论记进 tools/data_audit/DATA_ISSUES.md §4 决策 1，再跑 s3 =="
  ;;
# ------------------------------------------------- ⑧ 人工抽检表（P0/P1 效度测评）
review)
  [ -f "$OUT/s2_sample.jsonl" ] || { echo "缺 $OUT/s2_sample.jsonl"; exit 1; }
  R=tools/data_audit/make_review_sheet.py
  # ① better_zh 质量（P0，落地 v0.2 前必做）—— 需要 buckets，先跑过 s3
  if [ -f "$OUT/buckets.jsonl" ]; then
    python3 $R --kind better-zh --sample "$OUT/s2_sample.jsonl" \
      --verdicts "$OUT/s2_verdicts.*.jsonl" --buckets "$OUT/buckets.jsonl" \
      -n "${REVIEW_N:-50}" --out "$OUT/review_better_zh.tsv"
  else
    echo "跳过 better-zh 表：缺 $OUT/buckets.jsonl（先跑 bash $0 s3）"
  fi
  # ② S0 旗标 vs 实测翻译分 的冲突记录（P1）
  if ls "$OUT"/s2_trans.*.jsonl >/dev/null 2>&1; then
    python3 $R --kind conflicts --sample "$OUT/s2_sample.jsonl" \
      --trans "$OUT/s2_trans.*.jsonl" -n "${CONFLICT_N:-15}" \
      --out "$OUT/review_conflicts.tsv"
  else
    echo "跳过 conflicts 表：缺 $OUT/s2_trans.*.jsonl（先跑 bash $0 s2-trans）"
  fi
  # ③ 随机抽检，验证单一质量分（P1）
  python3 $R --kind random --sample "$OUT/s2_sample.jsonl" \
    --verdicts "$OUT/s2_verdicts.*.jsonl" -n "${RANDOM_N:-300}" \
    --out "$OUT/review_random.tsv"
  echo
  echo "== 三张表已生成在 $OUT/ ；在 Excel/VSCode 里填最后几列，配合 §6 的 Gradio 看画面 =="
  echo "== 填完统计： bash $0 review-score $OUT/review_better_zh.tsv =="
  ;;
review-score)
  [ -n "${2:-}" ] || { echo "用法: bash $0 review-score <填好的.tsv>"; exit 1; }
  python3 tools/data_audit/make_review_sheet.py --score "$2"
  ;;
# ------------------------------- ⑨ 粒度计数式重判（查 dense=0 是不是标签塌缩）
s2-gran)
  need_py
  [ -f "$OUT/s2_sample.jsonl" ] || { echo "先跑 s2-sample"; exit 1; }
  echo "[model] S2_MODEL=$S2_MODEL （文本模式：只看中文字幕，数信息点，不看画面）"
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i nohup python tools/data_audit/vlm_audit_hf.py \
      --input "$OUT/s2_sample.jsonl" --video-root "$CLIPS" --model "$S2_MODEL" \
      --mode gran --num-shards "$NGPU" --shard-id "$i" \
      --max-new-tokens "${GRAN_MAX_TOKENS:-512}" \
      --output "$OUT/s2_gran.{shard}.jsonl" \
      > "$OUT/s2_gran_shard$i.log" 2>&1 &
    echo "shard $i -> GPU$i (log: $OUT/s2_gran_shard$i.log)"
  done
  echo "已后台启动（可续跑）。同样不解码视频，比 s2 快得多。"
  echo "跑完后统计 n_info 分布："
  echo "  python3 -c \"import json,glob;from collections import Counter;c=Counter(json.loads(l).get('verdict',{}).get('n_info') for f in glob.glob('$OUT/s2_gran.*.jsonl') for l in open(f));print(sorted(c.items(),key=lambda x:(x[0] is None,x[0])))\""
  ;;
# ------------------------------------------------ ⑩ 快照：一条命令打印全部状态
snapshot)
  # 纯 stdlib，不需要 venv；任何文件缺失都不报错，只标 "-"
  python3 - "$OUT" "$DATA" <<'PYSNAP'
import glob, json, os, subprocess, sys
from collections import Counter
OUT, DATA = sys.argv[1], sys.argv[2]

def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""

def lines(pat):
    n, files = 0, sorted(glob.glob(os.path.join(OUT, pat)))
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                n += sum(1 for _ in fh)
        except OSError:
            pass
    return n, len(files)

print("# 审查流水线快照\n")
print("## git")
print(f"- HEAD: `{sh('git log --oneline -1')}`")
beh = sh("git rev-list --count HEAD..origin/main 2>/dev/null")
print(f"- 落后 origin/main: **{beh or '?'}** 个提交"
      + ("  ← **先 git pull**" if beh not in ("0", "") else ""))
print(f"- 工作区改动: {len(sh('git status --porcelain').splitlines())} 个文件\n")

print("## 各阶段产物")
tot = 0
try:
    tot = sum(1 for _ in open(DATA, encoding="utf-8", errors="ignore"))
except OSError:
    pass
# whole=每条记录一行（可以算完成度）；subset=按设计只覆盖一部分，算百分比会误导
rows = [("S0 旗标", "s0_flags.jsonl", "subset:仅命中旗标的记录"),
        ("S1 打分", "s1_scores.*.jsonl", "whole"),
        ("S2 抽样", "s2_sample.jsonl", "whole"),
        ("S2 判定", "s2_verdicts.*.jsonl", "whole"),
        ("S2 翻译重判", "s2_trans.*.jsonl", "whole"),
        ("S2 粒度重判", "s2_gran.*.jsonl", "whole"),
        ("裁判档", "s2_verdicts_judge.jsonl", "subset:按设计只判 JUDGE_N 条"),
        ("S3 分桶", "buckets.jsonl", "whole")]
print("| 阶段 | 产物 | 行数 | 进度 | 分片 |")
print("|---|---|---|---|---|")
for name, pat, kind in rows:
    n, nf = lines(pat)
    if not n:
        prog = "未生成"
    elif kind == "whole":
        prog = f"{n/tot:.0%}" if tot else "-"
        if tot and n < tot:
            prog += " ⏳"
    else:
        prog = kind.split(":", 1)[1]
    print(f"| {name} | `{pat}` | {n:,} | {prog} | {nf or '-'} |")
print(f"\n（源数据 `{os.path.basename(DATA)}` 共 {tot:,} 行）\n")

print("## 存活进程")
# 不用 pgrep：-f 会连带匹配到承载本命令的 shell（其命令行里含有模式字面量），
# 括号技巧也挡不住嵌套调用。改为显式读 ps 并把"看起来像本快照自己"的行剔掉。
_ps = [l.strip() for l in sh("ps -eo args=").splitlines() if l.strip()]
_self = ("pgrep", "run_v100.sh", "snapshot", "ps -eo")
cmds = [l for l in _ps if not any(t in l for t in _self)]
def alive(sub):
    return sum(1 for l in cmds if sub in l)
n_all = alive("vlm_audit_hf.py")
procs = [("S1 打分", alive("score_alignment.py")),
         ("S2/重判（全部）", n_all),
         ("　└ 翻译重判", alive("--mode trans")),
         ("　└ 粒度重判", alive("--mode gran")),
         ("裁判档", sum(1 for l in cmds if "vlm_audit.py" in l))]
hit = False
for label, c in procs:
    if c:
        print(f"- {label}: **{c}** 个")
        hit = True
if not hit:
    print("- （无后台任务在跑）")
print()

# 粒度重判结果
gn, _ = lines("s2_gran.*.jsonl")
if gn:
    c, err = Counter(), 0
    for f in glob.glob(os.path.join(OUT, "s2_gran.*.jsonl")):
        for line in open(f, encoding="utf-8", errors="ignore"):
            try:
                v = (json.loads(line).get("verdict") or {})
            except json.JSONDecodeError:
                continue
            if v.get("n_info") is None:
                err += 1
            else:
                c[v["n_info"]] += 1
    print(f"## 粒度重判 n_info 分布（{sum(c.values()):,} 条可用，失败 {err}）\n")
    if c:
        tt = sum(c.values())
        ks = sorted(c)
        print("| n_info | " + " | ".join(str(k) for k in ks) + " |")
        print("|---" * (len(ks) + 1) + "|")
        print("| 条数 | " + " | ".join(f"{c[k]:,}" for k in ks) + " |")
        acc, med = 0, None
        for k in ks:
            acc += c[k]
            if med is None and acc >= tt / 2:
                med = k
        print(f"\n- **中位数 {med}**，均值 {sum(k*v for k,v in c.items())/tt:.2f}"
              "　（「一个男人在说话」= 主体1+动作1 = 2）")
        coarse = sum(v for k, v in c.items() if k <= 2)
        adeq = sum(v for k, v in c.items() if k in (3, 4))
        dense = sum(v for k, v in c.items() if k >= 5)
        print(f"- 分档：**≤2 粗 {coarse/tt:.1%}｜3–4 中 {adeq/tt:.1%}｜≥5 密 {dense/tt:.1%}**"
              "　（原三档标签为 coarse 93.6% / adequate 6.4% / dense 0）")
        z = c.get(0, 0)
        if z / tt > 0.05:
            # 零信息点的字幕本不该有这么多——先看它们长什么样，再决定是数据问题还是 rubric 问题
            zh = {}
            sp = os.path.join(OUT, "s2_sample.jsonl")
            if os.path.exists(sp):
                for line in open(sp, encoding="utf-8", errors="ignore"):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    zh[r.get("video")] = r.get("zh")
            ex = []
            for f in glob.glob(os.path.join(OUT, "s2_gran.*.jsonl")):
                for line in open(f, encoding="utf-8", errors="ignore"):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (r.get("verdict") or {}).get("n_info") == 0:
                        t = (zh.get(r.get("video")) or "").strip()
                        ex.append(t if t else "(原字幕为空)")
                    if len(ex) >= 6:
                        break
                if len(ex) >= 6:
                    break
            print(f"\n- ⚠️ **`n_info=0` 占 {z/tt:.1%}** —— 字幕零信息点本不该这么多，"
                  "要么原字幕真的是空/无意义，要么是模型在这批上没按 rubric 走。样例：")
            for t in ex:
                print(f"  - `{t[:60]}`")
    print()

print("## 人工抽检表")
print("| 表 | 总行 | 已填 | 状态 |")
print("|---|---|---|---|")
import csv
for f in ("review_better_zh.tsv", "review_conflicts.tsv", "review_random.tsv"):
    p = os.path.join(OUT, f)
    if not os.path.exists(p):
        print(f"| `{f}` | - | - | 未生成 |")
        continue
    try:
        rr = list(csv.DictReader(open(p, encoding="utf-8-sig"), delimiter="\t"))
    except Exception:
        print(f"| `{f}` | ? | ? | 读取失败 |")
        continue
    cols = [c for c in (rr[0] if rr else {}) if c and "[" in c]
    filled = sum(1 for r in rr if any((r.get(c) or "").strip() for c in cols))
    st = "✅ 可统计" if filled else "⬜ 待填"
    print(f"| `{f}` | {len(rr)} | **{filled}** | {st} |")
print()

print("## 关键结论回显（来自不入库的 outputs/）")
for f, pats in [("s2_health.md", ["逐条完全同值", "视频含多个先后事件", "字幕根本没描述顺序"]),
                ("trans_health.md", ["差率（≤2）", "锚定已解除"]),
                ("xling_calibration.md", ["基准差率", "lift ="]),
                ("report.md", ["| KEEP |", "| RECAPTION |", "| RETRANSLATE |"])]:
    p = os.path.join(OUT, f)
    if not os.path.exists(p):
        print(f"- `{f}`：未生成")
        continue
    hits = []
    for line in open(p, encoding="utf-8", errors="ignore"):
        if any(x in line for x in pats):
            hits.append(line.strip()[:110])
    print(f"- **`{f}`**")
    for h in hits[:6]:
        print(f"  - {h}")
print()

print("## 未完成任务的日志尾行")
any_log = False
for pat in ("s2_gran_shard*.log", "s2_trans_shard*.log", "s2_shard*.log"):
    fs = sorted(glob.glob(os.path.join(OUT, pat)))
    if not fs:
        continue
    tails = {sh(f"tail -n1 {f}") for f in fs}
    print(f"- `{pat}`（{len(fs)} 个）：")
    for t in list(tails)[:4]:
        print(f"  - {t[:120]}")
    any_log = True
if not any_log:
    print("- （无）")
PYSNAP
  ;;
# ---------------------------------------------------------------- ⑦ 报告
s3)
  # 决策 1 已裁决（2026-07-22）：xling_cos 退出分桶规则。实测 ρ=-0.031、
  # lift 1.34×，默认 0.7 的精度 4.8%（基准 3.6%）= 95.2% 误伤，且会抢在
  # sig_zh 的 RECAPTION 判断之前把 21.1% 的语料钉死。详见 DATA_ISSUES.md 决策 1。
  # 想复现旧行为：XLING_THRESHOLD=0.7 bash run_v100.sh s3
  S3_ARGS=(--scores "$OUT/s1_scores.*.jsonl" --flags "$OUT/s0_flags.jsonl"
           --verdicts "$OUT/s2_verdicts.*.jsonl"
           --target-precision "${TARGET_PRECISION:-0.8}"
           --xling-threshold "${XLING_THRESHOLD:-0}")
  # 有文本-only 重判就用它算「翻译差率」——合并 rubric 那一列实为整体质量分的
  # 复制品（DATA_ISSUES.md 问题 8），会把 3.6% 的真实差率报成 19.6%。
  ls "$OUT"/s2_trans.*.jsonl >/dev/null 2>&1 && \
    S3_ARGS+=(--trans-verdicts "$OUT/s2_trans.*.jsonl")
  [ -n "${SIG_ZH_THRESHOLD:-}" ] && S3_ARGS+=(--sig-zh-threshold "$SIG_ZH_THRESHOLD")
  python tools/data_audit/aggregate_report.py "${S3_ARGS[@]}" \
    --out-buckets "$OUT/buckets.jsonl" --out-report "$OUT/report.md"
  echo "== 诊断报告: $OUT/report.md（含问题率外推表；本轮到此为止，不改数据） =="
  ;;
*)
  sed -n '2,/^set -euo/p' "$0" | grep -E '^#( |$)' | sed 's/^# \?//'
  ;;
esac
