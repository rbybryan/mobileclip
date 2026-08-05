#!/usr/bin/env bash
# 1M 验证层 · 模型优化一体化跑法 —— 精简结果报告(6 个方案)的**唯一复现入口**。
# 设计与结论:model-optimization/(README.md, docs/model-changes.md, docs/run.md)。
# 结果数字:model-optimization/docs/retrospective-brief.md。
#
# 一条脚本三段:①训练 → ②离线补评(落 ranks)→ ③配对 bootstrap 比较。
# run_verify_1m.sh 只管①、且默认清单里没有 P9 / P1trained;本脚本把 6 个方案
# 补齐、并把②③接上,一次跑完就有可判读的 CI。
#
# 用法(挂后台记 PID —— 共用 root,停止只能按 PID/PGID,严禁 pkill):
#   nohup bash model-optimization/scripts/run_model_opt.sh 0,1,2,3 > experiments/runs/model_opt.log 2>&1 &
#   echo "model-opt PID = $!"
#
# 只跑某几段(逗号分隔,默认 train,eval,compare 全跑):
#   PHASE=eval,compare bash model-optimization/scripts/run_model_opt.sh 3
#   PHASE=compare      bash model-optimization/scripts/run_model_opt.sh 3        # 纯 CPU,GPU 参数被忽略
#
# 只跑指定几个方案(短名不带 .py;省略 = 下面的默认 6 方案清单):
#   bash model-optimization/scripts/run_model_opt.sh 0,1,2,3 sub1m_r0_s42 sub1m_P1trained_s42
#
# 数据路径:export MVC_DATA_DIR=... MVC_ANNO_DIR=...(见 experiments/README.md)。
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

GPUS=${1:?usage: run_model_opt.sh <CUDA_VISIBLE_DEVICES> [config_short_name ...]}
shift || true
NG=$(printf '%s' "$GPUS" | tr ',' '\n' | grep -c .)
PHASE=${PHASE:-train,eval,compare}

# 精简报告的 6 个方案(× seed 42/43;P9 二阶统计单 seed 即否决,见 docs/model-changes.md §4)。
# 基线 r0 在最前 —— ②③都要拿它当同 seed 参照。
if [[ $# -gt 0 ]]; then
    ARMS=("$@")
else
    ARMS=(
        sub1m_r0_s42        sub1m_r0_s43         # 基线(均值池化 + softmax)
        sub1m_P1trained_s42 sub1m_P1trained_s43  # 删时序分支(减法,零新增参数)
        sub1m_r1b_s42       sub1m_r1b_s43         # 注意力池化
        sub1m_P9_s42                              # 二阶统计池化(单 seed)
        sub1m_nA1_s42       sub1m_nA1_s43         # 惊奇池化
        sub1m_r1a_s42       sub1m_r1a_s43         # sigmoid 损失
    )
fi

_has() { printf '%s' ",$PHASE," | grep -q ",$1,"; }
EVALS=experiments/evals/sub1m
mkdir -p experiments/runs "$EVALS"
echo "model-opt: GPUS=$GPUS NG=$NG PHASE=$PHASE arms=${ARMS[*]}"

# ── ① 训练 ────────────────────────────────────────────────────────────────
# 沿用 run_verify_1m.sh 踩过的两条铁律:日志一律**追加**(>>,否则 auto_resume
# 续跑会抹掉前一腿墙钟,status_1m.py 误判 🔴);auto_resume True 让被打断的臂从
# ckpt_latest 续、没 ckpt 的从头起(找不到 ckpt = 全新,安全)。
if _has train; then
  for c in "${ARMS[@]}"; do
    cfg="experiments/configs/${c}.py"
    [[ -f "$cfg" ]] || { echo "!! 缺 $cfg —— 跳过"; continue; }
    echo "════════ $(date '+%F %T')  TRAIN ${c} ════════"
    echo "════════ $(date '+%F %T')  (re)start ${c} — runner append ════════" \
        >> "experiments/runs/${c}.log"
    CUDA_VISIBLE_DEVICES="$GPUS" bash experiments/run_pretrain.sh \
        "$cfg" "$NG" use_bf16 True bf16 True auto_resume True \
        >> "experiments/runs/${c}.log" 2>&1
    echo "════════ $(date '+%F %T')  DONE  ${c} ════════"
  done
fi

# ── ② 离线补评(落 ranks_*.npz,配对 bootstrap 的输入)─────────────────────
# ranks 2026-07-29 才开始落盘,补评是纯前向(MSR-VTT 1k ~1 分钟/臂,串行别抢卡)。
# ckpt 指到**目录** ckpt_02.pth(不是里面的 .pt);已有 ranks 的臂跳过。
# 跑完 grep 自检:没看到 "Load pretrained model from ..." 就是权重没加载、数字无效
#（未加载也会打印一套很像样的 r_mean，不查就当真）。
if _has eval; then
  export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"   # 补评单卡即可,取第一张
  [[ -f experiments/env_b100.sh ]] && source experiments/env_b100.sh >/dev/null 2>&1 || true
  for c in "${ARMS[@]}"; do
    run="experiments/runs/${c}"
    [[ -e "$run/ckpt_02.pth" ]] || { echo "!! ${c}: 无 ckpt_02.pth,先跑训练 —— 跳过"; continue; }
    if ls "$EVALS/$c"/ranks_*.npz "$run"/ranks_*.npz >/dev/null 2>&1; then
        echo "skip eval ${c}(已有 ranks)"; continue
    fi
    echo "════════ $(date '+%F %T')  EVAL ${c} ════════"
    bash experiments/run_eval.sh small msrvtt "$run/ckpt_02.pth" \
         "$EVALS/$c" num_frames_test 8 batch_size_test 32 || { echo "FAILED eval $c"; continue; }
    if ! grep -q "Load pretrained model from" "$EVALS/$c/train.log" 2>/dev/null; then
        echo "🔴 ${c}: 未见 'Load pretrained model from' —— 权重可能没加载,数字勿信(见 COMMANDS §3.8)"
    fi
  done
fi

# ── ③ 配对 bootstrap 比较(纯 CPU,零 GPU)──────────────────────────────────
# 每条非基线臂减**同 seed 的 r0**,raw(ret_test_emb/)与 DSL(ret_test/)两口径都出;
# 判读以 95% CI 是否含 0 为准(别再套 ±1.6)。ranks 可能在 runs/ 或 evals/,自动指对。
if _has compare; then
  _ranksdir() {  # 回显装着 ranks_*.npz 的目录(evals 优先,再 runs);都没有回显空
    ls "$EVALS/$1"/ranks_*.npz >/dev/null 2>&1 && { echo "$EVALS/$1"; return; }
    ls "experiments/runs/$1"/ranks_*.npz >/dev/null 2>&1 && { echo "experiments/runs/$1"; return; }
    echo ""
  }
  for c in "${ARMS[@]}"; do
    [[ "$c" == sub1m_r0_* ]] && continue                 # 基线不与自己比
    seed="${c##*_}"                                      # s42 / s43
    base="sub1m_r0_${seed}"
    ad=$(_ranksdir "$c"); bd=$(_ranksdir "$base")
    [[ -n "$ad" && -n "$bd" ]] || { echo "!! ${c} vs ${base}: 缺 ranks(arm='$ad' base='$bd')—— 先跑 eval,跳过"; continue; }
    echo "════════ ${c} − ${base} ════════"
    echo "--- raw(ret_test_emb/)---"
    python experiments/collect_results.py --compare "$ad" "$bd" || true
    echo "--- DSL(ret_test/)---"
    python experiments/collect_results.py --compare "$ad" "$bd" --prefix 'ret_test/' || true
  done
fi

echo "model-opt: ALL DONE (PHASE=$PHASE)"
