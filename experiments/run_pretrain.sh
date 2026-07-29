#!/usr/bin/env bash
# 训练一个实验配置(单机多卡)。
#
# 用法:
#   bash experiments/run_pretrain.sh <exp_config.py> [NUM_GPUS] [key value ...]
#
# 例:
#   bash experiments/run_pretrain.sh experiments/configs/r0_baseline.py 8
#   # 冒烟测试(几十步即停,数据/模型/损失全链路检查):
#   bash experiments/run_pretrain.sh experiments/configs/r1a_sigmoid.py 1 debug True
#
# 数据路径通过环境变量注入(见 experiments/README.md):
#   export MVC_DATA_DIR=/path/to/data  MVC_ANNO_DIR=/path/to/anno_downstream
set -euo pipefail

CONFIG=${1:?usage: run_pretrain.sh <exp_config.py> [NUM_GPUS] [key value ...]}
shift
NUM_GPUS=${1:-8}
if [[ $# -gt 0 ]]; then shift; fi

export OMP_NUM_THREADS=1
export PYTHONPATH=${PYTHONPATH:-}:.
MASTER_PORT=$((11000 + RANDOM % 20000))

echo "config    : ${CONFIG}"
echo "gpus      : ${NUM_GPUS}"
echo "overrides : $*"
echo "MVC_DATA_DIR=${MVC_DATA_DIR:-data} MVC_ANNO_DIR=${MVC_ANNO_DIR:-anno_downstream}"

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

# 用 `python -m torch.distributed.run` 而不是裸 `torchrun`：裸 torchrun 走 PATH，
# 在装了系统级 torch 的机器上会解析到 /usr/local/bin/torchrun -> /usr/bin/python3，
# 于是整个训练在**系统解释器**里跑，绕开已激活的 conda/venv 环境（b100 2026-07-24
# 就是这么崩的）。`python -m` 保证用的就是当前 `python`。
python -m torch.distributed.run \
         --nnodes=1 \
         --nproc_per_node="${NUM_GPUS}" \
         --master_port="${MASTER_PORT}" \
         tasks_clip/pretrain.py \
         "${CONFIG}" "$@"
