#!/usr/bin/env bash
# 1M 验证层 A/B —— 在指定的一组卡上**串行**跑 R0 / r1a / r1b / nA1 × seed{42,43}。
# 设计与判读见 notes/verification-tier-1m-zh.md。
#
# 用法(挂后台并记 PID —— 共用 root,停止只能按这个 PID / 其 PGID,严禁 pkill):
#   nohup bash experiments/run_verify_1m.sh 4,5,6,7 > experiments/runs/verify_1m.log 2>&1 &
#   echo "verify-1m PID = $!"
#
# 只跑指定几个(短名,不带 .py),例如先出一个 seed 的四臂:
#   bash experiments/run_verify_1m.sh 4,5,6,7 sub1m_r0_s42 sub1m_r1a_s42 sub1m_r1b_s42 sub1m_nA1_s42
#
# batch/输出目录/种子都写在各 config 里;这里不再传 batch。断点续训:给某个 config
# 追加 `auto_resume True`(单独重跑那一条即可,output_dir 已隔离,不会互相覆盖)。
set -euo pipefail

GPUS=${1:?usage: run_verify_1m.sh <CUDA_VISIBLE_DEVICES> [config_short_name ...]}
shift
NG=$(printf '%s' "$GPUS" | tr ',' '\n' | grep -c .)

if [[ $# -gt 0 ]]; then
    CONFIGS=("$@")
else
    # seed 42 四臂在前(~2.5 天出第一个 seed 的完整 A/B),seed 43 在后做一致性确认
    CONFIGS=(
        sub1m_r0_s42 sub1m_r1a_s42 sub1m_r1b_s42 sub1m_nA1_s42
        sub1m_r0_s43 sub1m_r1a_s43 sub1m_r1b_s43 sub1m_nA1_s43
    )
fi

mkdir -p experiments/runs
echo "verify-1m: GPUS=$GPUS  NG=$NG  configs=${CONFIGS[*]}"

for c in "${CONFIGS[@]}"; do
    cfg="experiments/configs/${c}.py"
    [[ -f "$cfg" ]] || { echo "!! missing $cfg — skip"; continue; }
    echo "════════ $(date '+%F %T')  START ${c} ════════"
    CUDA_VISIBLE_DEVICES="$GPUS" bash experiments/run_pretrain.sh \
        "$cfg" "$NG" \
        use_bf16 True bf16 True \
        > "experiments/runs/${c}.log" 2>&1
    echo "════════ $(date '+%F %T')  DONE  ${c}  (log: experiments/runs/${c}.log) ════════"
done

echo "verify-1m: ALL DONE"
