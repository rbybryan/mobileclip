#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-11670}
ANNO_DIR=${ANNO_DIR:-../video-text-dataset/internvid-cap-zh}
DATA_ROOT=${DATA_ROOT:-/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT}
HF_MODEL_PATH=${HF_MODEL_PATH:-hfl/minirbt-h288}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HF_HOME=${HF_HOME:-/workspace/mount/b100_zaip_data/wky_b100/cache/hfhub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export ANNO_DIR DATA_ROOT
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    tasks_clip/pretrain.py \
    scripts/pretraining/clip/mobileviclip_nebulacv/config.py \
    model.text_encoder.hf_model_name "${HF_MODEL_PATH}" \
    model.text_encoder.hf_tokenizer_name "${HF_MODEL_PATH}" \
    "$@"
