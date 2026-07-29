export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='zs_ucf101'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=16

export CUDA_VISIBLE_DEVICES=4
MASTER_PORT=$((50660 + $RANDOM % 10))
# 用 `python -m torch.distributed.run` 而不是裸 `torchrun`:裸 torchrun 走 PATH,
# 在装了系统级 torch 的机器上会解析到 /usr/local/bin/torchrun -> /usr/bin/python3,
# 于是整个任务在**系统解释器**里跑,绕开已激活的 conda/venv(表现为 venv 里明明装了的
# 包报 ModuleNotFoundError,例如 termcolor)。b100 训练 2026-07-24、评测 2026-07-29
# 各崩过一次。`python -m` 保证用的就是当前 `python`。
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} \
         --master_port=${MASTER_PORT} \
         tasks_clip/retrieval.py \
         $(dirname $0)/config_ucf101.py \

#               txt_r1  txt_r5  txt_r10  txt_r_mean  img_r1  img_r5  img_r10  img_r_mean  r_mean
# act_val/       75.44   95.35    98.60       89.80   83.17   96.04    99.01       92.74   91.27
# act_val_emb/   75.55   95.90    98.81       90.09   80.20   95.05    99.01       91.42   90.75