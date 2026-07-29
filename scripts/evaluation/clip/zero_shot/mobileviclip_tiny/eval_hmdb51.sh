export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='zs_hmdb51'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PARTITION='video'
NNODE=1
NUM_GPUS=1
NUM_CPU=16


export CUDA_VISIBLE_DEVICES=6
MASTER_PORT=$((35660 + $RANDOM % 10))
# 用 `python -m torch.distributed.run` 而不是裸 `torchrun`:裸 torchrun 走 PATH,
# 在装了系统级 torch 的机器上会解析到 /usr/local/bin/torchrun -> /usr/bin/python3,
# 于是整个任务在**系统解释器**里跑,绕开已激活的 conda/venv(表现为 venv 里明明装了的
# 包报 ModuleNotFoundError,例如 termcolor)。b100 训练 2026-07-24、评测 2026-07-29
# 各崩过一次。`python -m` 保证用的就是当前 `python`。
python -m torch.distributed.run --nproc_per_node=${NUM_GPUS} \
         --master_port=${MASTER_PORT} \
         tasks_clip/retrieval.py \
         $(dirname $0)/config_hmdb51.py \

#               txt_r1  txt_r5  txt_r10  txt_r_mean  img_r1  img_r5  img_r10  img_r_mean  r_mean
# act_val/       48.76   77.71    88.17       71.55   62.75   74.51    84.31       73.86   72.70
# act_val_emb/   48.82   77.39    88.82       71.68   60.78   72.55    84.31       72.55   72.11
