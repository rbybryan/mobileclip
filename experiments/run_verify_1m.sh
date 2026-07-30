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
# ⚠️ 下面的默认清单是 **8 臂**(r0/r1a/r1b/nA1 × seed{42,43}),按 r0 实测 15:11:58/臂
# 约 **5 天**。#335 新增的 4 个臂 **不在默认清单里**(避免把在跑的排期悄悄拉长到 ~7.6 天),
# 要跑请显式指定:
#   bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_nR1_s42 sub1m_r1both_s42 sub1m_nR1_s43 sub1m_r1both_s43
#
# batch/输出目录/种子都写在各 config 里;这里不再传 batch。
# **断点续训已内置**:下面对 run_pretrain 传了 `auto_resume True` ⇒ 某臂被信号打断
# (前台 runner 遇 tunnel 掉线 / Ctrl-C ⇒ SignalException ⇒ set -e 整条 runner 退出)后,
# 只需**用剩余的 config 名重跑本脚本**:已有 ckpt 的臂(如被打断的那条)会从 ckpt_latest
# 续训,没 ckpt 的臂正常从头开始(auto_resume 找不到 ckpt = 全新起步,安全)。
# ⚠️ **务必挂 nohup 后台**(见上面用法),别再前台跑 —— 前台是这次被打断的根因。
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
        use_bf16 True bf16 True auto_resume True \
        > "experiments/runs/${c}.log" 2>&1
    echo "════════ $(date '+%F %T')  DONE  ${c}  (log: experiments/runs/${c}.log) ════════"
done

echo "verify-1m: ALL DONE"
