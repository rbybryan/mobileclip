export MASTER_PORT=$((13000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='zs_didemo'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=16

export CUDA_VISIBLE_DEVICES=4
MASTER_PORT=$((17660 + $RANDOM % 10))
# 用 `python -m torch.distributed.run` 而不是裸 `torchrun`:裸 torchrun 走 PATH,
# 在装了系统级 torch 的机器上会解析到 /usr/local/bin/torchrun -> /usr/bin/python3,
# 于是整个任务在**系统解释器**里跑,绕开已激活的 conda/venv(表现为 venv 里明明装了的
# 包报 ModuleNotFoundError,例如 termcolor)。b100 训练 2026-07-24、评测 2026-07-29
# 各崩过一次。`python -m` 保证用的就是当前 `python`。
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} \
         --master_port=${MASTER_PORT} \
         tasks_clip/retrieval.py \
         $(dirname $0)/config_didemo.py \

#                txt_r1  txt_r5  txt_r10  txt_r_mean  img_r1  img_r5  img_r10  img_r_mean  r_mean
# ret_test/       41.59   67.60    77.13       62.11   41.59   68.72    78.59       62.97   62.54
# ret_test_emb/   40.25   67.49    76.23       61.32   39.24   67.15    77.91       61.43   61.38