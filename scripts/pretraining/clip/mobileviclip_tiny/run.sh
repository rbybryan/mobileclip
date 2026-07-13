export OMP_NUM_THREADS=1

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='mobileviclip_tiny'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"

# ============ single node, 8x V100-32GB ============
# global batch = NUM_GPUS * batch_size = 8 * 64 = 512 (same as the paper's 16x32)
NNODE=1
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_PORT=$((11000 + $RANDOM % 20000))

torchrun --nnodes=${NNODE} \
         --nproc_per_node=${NUM_GPUS} \
         --master_port=${MASTER_PORT} \
         tasks_clip/pretrain.py \
         $(dirname $0)/config.py
