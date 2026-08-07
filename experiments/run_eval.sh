#!/usr/bin/env bash
# 用指定 checkpoint 跑一个零样本评测基准,结果写到指定目录。
#
# 用法:
#   bash experiments/run_eval.sh <small|tiny> <benchmark> <ckpt_path> <out_dir> [key value ...]
#   benchmark ∈ {msrvtt, didemo, anet, k400, ucf101, hmdb51, ssv2_mc}
#
# 例(锁定官方权重的基线数字):
#   bash experiments/run_eval.sh small msrvtt ./checkpoints/mobileviclip_small.pt \
#        experiments/evals/baseline_official/msrvtt
#
# 例(评测某个实验产出的 ckpt):
#   bash experiments/run_eval.sh small msrvtt \
#        experiments/runs/r1a_sigmoid/ckpt_best.pth \
#        experiments/evals/r1a_sigmoid/msrvtt
#
# 结果落盘 <out_dir>/eval_res.json,主键 `ret_test/` 为 DSL(dual-softmax)
# 口径、`ret_test_emb/` 为原始余弦口径 —— 对比时两个口径都要看
# (见 tasks_clip/retrieval_utils.py 的 raw/DSL 两口径)。
set -euo pipefail

MODEL=${1:?usage: run_eval.sh <small|tiny> <benchmark> <ckpt> <out_dir> [key value ...]}
BENCH=${2:?missing benchmark}
CKPT=${3:?missing checkpoint path}
OUT=${4:?missing output dir}
shift 4

CONFIG="scripts/evaluation/clip/zero_shot/mobileviclip_${MODEL}/config_${BENCH}.py"
[[ -f "${CONFIG}" ]] || { echo "no such eval config: ${CONFIG}"; exit 1; }

# ssv2_mc 走多选任务入口,其余走检索入口(与原 repo 的 eval_*.sh 一致)
TASK="tasks_clip/retrieval.py"
[[ "${BENCH}" == "ssv2_mc" ]] && TASK="tasks_clip/retrieval_mc.py"

export OMP_NUM_THREADS=1
export PYTHONPATH=${PYTHONPATH:-}:.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MASTER_PORT=$((12000 + RANDOM % 20000))

mkdir -p "${OUT}"

# --- b100 环境自愈 + 依赖预检 ---------------------------------------------
# b100 是**共用 root**:`python` 本来就该是 /usr/bin/python(系统 torch 2.5.1+cu124,
# 含 sm_90),缺的小包装在**私有 user-site**,靠 experiments/env_b100.sh 设
# PYTHONUSERBASE。新开一个 shell 忘了 source,就会在 torchrun 三层栈之后炸
# `ModuleNotFoundError: termcolor`(2026-07-29 补评连崩两次,SETUP_B100.md §2 早有记载)。
# env_b100.sh 只设环境变量、重复 source 无副作用 ⇒ 这里检测到就自动补上。
if [[ -z "${PYTHONUSERBASE:-}" && -f experiments/env_b100.sh ]]; then
    _ub=$(bash -c 'source experiments/env_b100.sh >/dev/null 2>&1; printf "%s" "${PYTHONUSERBASE:-}"' || true)
    if [[ -n "${_ub}" && -d "${_ub}" ]]; then
        echo ">> PYTHONUSERBASE 未设,且检测到 user-site ${_ub} —— 自动 source experiments/env_b100.sh"
        # shellcheck disable=SC1091
        source experiments/env_b100.sh > /dev/null
    fi
fi
echo "python    : $(command -v python)"
echo "userbase  : ${PYTHONUSERBASE:-<未设置>}"

# 预检:用**即将启动的那个解释器**试导一次,别等 torchrun 起完三层栈才崩。
if ! python -c "import termcolor, torch" > /dev/null 2>&1; then
    echo ""
    echo "!! 预检失败:当前 python($(command -v python))导不进 termcolor / torch。"
    echo "   b100 修法(按顺序试):"
    echo "     source experiments/env_b100.sh                      # 设 PYTHONUSERBASE,多数情况这一步就够"
    echo "     python -m pip install --user --no-deps termcolor     # 仍缺才装,落到自己的 user-site"
    echo "   ⚠️ 禁止裸 pip install —— b100 是共用 root,会改到 wky/zyc 的环境(SETUP_B100.md §2)。"
    exit 1
fi

# 用 `python -m torch.distributed.run` 而不是裸 `torchrun`:裸 torchrun 走 PATH,
# 在装了系统级 torch 的机器上会解析到 /usr/local/bin/torchrun -> /usr/bin/python3,
# 于是整个任务在**系统解释器**里跑,绕开已激活的 conda/venv(表现为 venv 里明明装了的
# 包报 ModuleNotFoundError,例如 termcolor)。b100 训练 2026-07-24、评测 2026-07-29
# 各崩过一次。`python -m` 保证用的就是当前 `python`。
echo "python    : $(command -v python)"   # 对不上你激活的环境就是这个坑
python -m torch.distributed.run --nproc_per_node=1 \
         --master_port="${MASTER_PORT}" \
         "${TASK}" \
         "${CONFIG}" \
         pretrained_path "${CKPT}" \
         output_dir "${OUT}" \
         "$@"
