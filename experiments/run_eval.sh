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
# (见 notes/script-level-optimizations-zh.md H3)。
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
