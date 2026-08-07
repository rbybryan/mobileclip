#!/usr/bin/env bash
# 选帧基线(情形 B)—— 零训练,在官方权重上跑完整对照阵列。
#
# 问题:C2 的卖点是"弹性 T 在低帧预算下比固定 T 好"。但从没人测过
#       "什么都不训练、只是把帧挑得聪明一点"能补回多少。GATE-0 实测帧的
#       多样性值 ~13 个 R@1、顺序值 ~0,所以这条平凡基线可能真的有威胁。
#
# 用法:
#   bash experiments/run_frame_select.sh [small|tiny] [benchmark] [ckpt] [outroot]
#   默认:small msrvtt ./checkpoints/mobileviclip_small.pt experiments/evals/frame_select
#
# 环境变量:
#   FS_KS="2 4"        要测的帧预算(默认 2 4)
#   FS_MODES="..."     选帧模式(默认 uniform divmax random)
#   FS_POOL=16         母序列帧数(默认 16)
#   FS_DRY_RUN=1       只打印要跑的命令,不真跑
#   CUDA_VISIBLE_DEVICES=0
#
# 已存在 eval_res.json 的臂会被跳过 ⇒ 中断后重跑本脚本即可续跑。
set -euo pipefail

MODEL=${1:-small}
BENCH=${2:-msrvtt}
CKPT=${3:-./checkpoints/mobileviclip_small.pt}
OUTROOT=${4:-experiments/evals/frame_select}

POOL=${FS_POOL:-16}
KS=${FS_KS:-"2 4"}
MODES=${FS_MODES:-"uniform divmax random"}
DRY=${FS_DRY_RUN:-0}

# 双根规则:这是评测,必须 unset(README §3.0;混用会把仓库 data/ 下的评测集
# 重定向到不存在的路径)。训练/试点 shell 才 export MVC_DATA_DIR。
unset MVC_DATA_DIR || true

# 选帧开关绝不能泄漏到参照臂上
unset MVC_FRAME_SELECT MVC_FRAME_SELECT_K || true
# 与 GATE-0 扰动开关互斥:同时开会把两个效应混在一起,读不出来
if [[ -n "${MVC_FRAME_PERTURB:-}" && "${MVC_FRAME_PERTURB}" != "none" ]]; then
  echo "拒绝执行:MVC_FRAME_PERTURB=${MVC_FRAME_PERTURB} 已设置。" >&2
  echo "选帧实验必须在未扰动的帧上跑,先 unset MVC_FRAME_PERTURB。" >&2
  exit 1
fi

run_arm () {  # run_arm <arm_name> <num_frames_test> [env assignments...]
  local arm=$1; shift
  local nft=$1; shift
  local out="${OUTROOT}/${BENCH}_${arm}"

  if [[ -f "${out}/eval_res.json" ]]; then
    echo "[skip] ${arm} —— 已有结果 ${out}/eval_res.json"
    return 0
  fi

  echo "===== [run ] ${arm}  (num_frames_test=${nft}${*:+, $*}) ====="
  if [[ "${DRY}" == "1" ]]; then
    echo "  env $* bash experiments/run_eval.sh ${MODEL} ${BENCH} ${CKPT} ${out} num_frames_test ${nft}"
    return 0
  fi
  env "$@" bash experiments/run_eval.sh "${MODEL}" "${BENCH}" "${CKPT}" "${out}" \
      num_frames_test "${nft}"
}

# --- 参照臂:不选帧 ------------------------------------------------------
# pool<POOL> 复现 B0 多 T 曲线 的 T=16 读数 ⇒ 兼作"钩子未开启时逐位等价"的验证
run_arm "pool${POOL}" "${POOL}"
# native_T<K>:整段视频上直接 middle 采 K 帧。与 uniform 臂的差别只在**采样支撑**
# (整段 vs 16 帧母序列),用来判断"池子本身"有没有影响 —— 别把两者混为一谈
for K in ${KS}; do
  run_arm "nativeT${K}" "${K}"
done

# --- 候选臂:从 POOL 帧母序列里选 K 帧,模型真正只跑 K 帧(成本与 nativeT<K> 一致)
for K in ${KS}; do
  for M in ${MODES}; do
    run_arm "sel${M}_K${K}" "${POOL}" \
      "MVC_FRAME_SELECT=${M}" "MVC_FRAME_SELECT_K=${K}" "MVC_FRAME_SELECT_SEED=0"
  done
done

echo
echo "全部臂已完成。汇总:"
echo "  python experiments/collect_frame_select.py --root ${OUTROOT}"
