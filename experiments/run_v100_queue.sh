#!/usr/bin/env bash
# V100 排队总入口(2026-07-31)—— **现在就能跑、不需要我再写任何代码**的全部项目。
#
#   bash experiments/run_v100_queue.sh            # 全跑(约 2–3 小时)
#   bash experiments/run_v100_queue.sh 1 2 3      # 只跑指定阶段
#
# 阶段:
#   0  就绪自检(1 min)         —— 决定阶段 3 做不做
#   1  gate3/gate4 配对 CI(4 min)—— **把 P1 / P5 从裸差变成判据**
#   2  P7 分辨率前沿(5 min)     —— 唯一存活效率线的第一份实测
#   3  多基准锚点 k400/ucf101/hmdb51(30 min)—— 让 1M 层重新有分辨力(§6 判定为「必须」)
#   4  分辨率 × 多基准 的 2-D 前沿(60 min)—— 只有阶段 2、3 都通了才有意义
#
# ⚠️ 全部零训练。所有 BN 相关的正式数仍欠一个重校准 CLI(`per_t_bn.py` 只有纯函数)。
set -uo pipefail

CKPT=${CKPT:-./checkpoints/mobileviclip_small.pt}
CFGDIR=${CFGDIR:-scripts/evaluation/clip/zero_shot/mobileviclip_small}
LOGDIR=${LOGDIR:-logs/v100_queue}
STAGES=${*:-0 1 2 3 4}

export PYTHONPATH="${PYTHONPATH:-}:."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$LOGDIR"

has(){ for s in $STAGES; do [ "$s" = "$1" ] && return 0; done; return 1; }

# 与 run_dropblock_sweep.sh 完全同一套调用规矩:eval_*.sh **不接受位置参数**,
# 必须覆盖 pretrained_path + output_dir,否则评的是基线且各臂互相覆盖 rank 文件。
run_eval(){  # run_eval <config.py> <output_dir> [extra k v ...]
  local cfg="$1" out="$2"; shift 2
  if [ -n "$(find "$out" -name 'eval_res*.json' 2>/dev/null | head -1)" ]; then
    echo "   跳过 $out(已有结果)"; return 0
  fi
  python -m torch.distributed.run --nproc_per_node=1 \
      --master_port=$((10000 + RANDOM % 20000)) \
      tasks_clip/retrieval.py "$cfg" \
      pretrained_path "$CKPT" output_dir "$out" "$@" \
      >>"$LOGDIR/$(basename "$out").log" 2>&1 \
    && echo "   ✅ $out" \
    || { echo "   🔴 $out 失败 → $LOGDIR/$(basename "$out").log"; tail -n 15 "$LOGDIR/$(basename "$out").log"; return 1; }
}

echo "=============================================================="
echo "V100 队列 · 权重 $CKPT · 阶段 [$STAGES]"
echo "脚本 commit $(git rev-parse --short HEAD 2>/dev/null || echo '非 git 工作树')"
echo "=============================================================="

# ---------------------------------------------------------------- 0
if has 0; then
  echo; echo "### 阶段 0 · 就绪自检 ——「k400/ucf101/hmdb51」三行决定阶段 3/4 做不做"
  python experiments/check_readiness.py 2>&1 | tee "$LOGDIR/readiness.log" | tail -30
fi

# ---------------------------------------------------------------- 1
if has 1; then
  echo; echo "### 阶段 1 · gate3(P1 删时序分支)/ gate4(P5 删 temporal_embedding)的配对 CI"
  echo "    裸差已知:gate3 −0.43/−0.33 · gate4 +0.70/+0.67(raw/DSL)。"
  echo "    P1 的预登记判据是「Δ>1.0 才杀」⇒ 看起来通过;**但判胜负要 CI 不含 0**。"
  for arm in eval_gate3 eval_gate4; do
    if [ ! -d "$arm" ]; then echo "   ⚠️ 缺 $arm/,跳过(先按 COMMANDS.md §3.14 造消融权重并评测)"; continue; fi
    for pref in "ret_test_emb/" "ret_test/"; do
      echo "--- $arm vs eval_base · 口径 $pref"
      python experiments/collect_results.py --compare "$arm" eval_base --prefix "$pref" \
        2>&1 | tee -a "$LOGDIR/${arm}_compare.log" | grep -E "^\| (r_mean|R_sum)|配对:|无配对:"
    done
  done
fi

# ---------------------------------------------------------------- 2
if has 2; then
  echo; echo "### 阶段 2 · round4-P7 分辨率前沿(P11 关线后唯一存活的效率线)"
  bash experiments/run_resolution_sweep.sh
fi

# ---------------------------------------------------------------- 3
if has 3; then
  echo; echo "### 阶段 3 · 多基准锚点(官方权重)—— §6 把多基准 R-Sum 从纪律升级为**必须**"
  echo "    MSR-VTT 1k 在 1M 尺度上 10× 数据只涨 +0.82(落在 ±1.6 带内)⇒ 单基准不够用。"
  echo "    DiDeMo/ANet 的 anno 拿不到,但这三个的 anno 一直是齐的。"
  for b in k400 ucf101 hmdb51; do
    echo "--- $b"
    run_eval "$CFGDIR/config_${b}.py" "eval_${b}_base" || true
  done
  echo "    参考(eval_k400.sh 内嵌的官方数):k400 act_val/ r_mean ≈ 87.88"
fi

# ---------------------------------------------------------------- 4
if has 4; then
  echo; echo "### 阶段 4 · 分辨率 × 多基准 的 2-D 前沿(只在阶段 2、3 都通了之后才有意义)"
  echo "    为什么值:P7 单靠 MSR-VTT 判不了 —— §6 的分辨力问题在这里同样成立。"
  for b in k400 ucf101 hmdb51; do
    for R in 160 181 224; do
      echo "--- $b @ ${R}²"
      run_eval "$CFGDIR/config_${b}.py" "eval_${b}_r${R}" \
               inputs.image_res "$R" model.vision_encoder.img_size "$R" || true
    done
  done
fi

echo; echo "=============================================================="
echo "汇总:"
echo "    python experiments/collect_results.py --dirs . --out experiments/v100_queue_all.md"
echo
echo "⚠️ 未做 BN 重校准 ⇒ 阶段 2/4 的分辨率数只能当趋势筛。换分辨率会改**每一层**的"
echo "   激活统计,漂移远大于 P11 删块;正式数要等 BN 重校准 CLI(per_t_bn.py 目前无 CLI),"
echo "   且按 round-5 §11.3 那一遍必须 with_cp=False。"
echo "=============================================================="
