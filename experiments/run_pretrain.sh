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

# 用 `python -m torch.distributed.run` 而不是裸 `torchrun`：裸 torchrun 走 PATH，
# 在装了系统级 torch 的机器上会解析到 /usr/local/bin/torchrun -> /usr/bin/python3，
# 于是整个训练在**系统解释器**里跑，绕开已激活的 conda/venv 环境（b100 2026-07-24
# 就是这么崩的）。`python -m` 保证用的就是当前 `python`。
echo "python    : $(command -v python)"
echo "userbase  : ${PYTHONUSERBASE:-<未设置>}"   # b100 需先 source experiments/env_b100.sh
python -m torch.distributed.run \
         --nnodes=1 \
         --nproc_per_node="${NUM_GPUS}" \
         --master_port="${MASTER_PORT}" \
         tasks_clip/pretrain.py \
         "${CONFIG}" "$@"
