#!/usr/bin/env bash
# P11 逐块深度敏感度扫描(round-5 §8.3 P11 / COMMANDS.md §3.14)
#
#   bash experiments/run_dropblock_sweep.sh                  # 全 40 块
#   bash experiments/run_dropblock_sweep.sh 0 9              # 只跑 0..9
#   KEEP_CKPT=1 bash experiments/run_dropblock_sweep.sh      # 保留 40 份消融权重(约 40×ckpt 大小)
#
# 为什么要有这个脚本,而不是在命令行写 for 循环:
#   ① **日志**:一个裸循环把 40 次 torch.distributed.run 的完整输出全冲到终端,
#      几十万行,出了错也翻不到。这里每臂单独落 logs/dropblock/b<i>.log,
#      终端只出**一行**。
#   ② **正确性**:`eval_msrvtt.sh` **不接受任何位置参数** —— 它把 output_dir 写死。
#      直接 `eval_msrvtt.sh <ckpt> <tag>` 会:评的是**基线**权重(ckpt 参数被忽略)、
#      40 个臂**全写进同一个目录**(rank 文件互相覆盖,配对 CI 直接做不了,§3.9 的坑)。
#      正确做法是覆盖 `pretrained_path` 与 `output_dir` 两个键(同 §3.11)。
#   ③ **快速失败**:第一个臂就错的话立刻停,而不是把同一个错误重复 40 遍。
#   ④ **可续跑**:已完成的臂自动跳过,断了直接重跑同一条命令。
set -uo pipefail

CKPT=${CKPT:-./checkpoints/mobileviclip_small.pt}
CFG=${CFG:-scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py}
LOGDIR=${LOGDIR:-logs/dropblock}
TMPDIR_CKPT=${TMPDIR_CKPT:-./checkpoints/_dropblock}
LO=${1:-0}; HI=${2:-39}

export PYTHONPATH="${PYTHONPATH:-}:."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$LOGDIR" "$TMPDIR_CKPT"

[ -f "$CKPT" ] || { echo "找不到权重:$CKPT(用 CKPT=... 覆盖)"; exit 1; }
[ -f "$CFG"  ] || { echo "找不到配置:$CFG"; exit 1; }

echo "块 $LO..$HI · 权重 $CKPT"
echo "每臂日志 → $LOGDIR/b<i>.log ;终端只出一行/臂"
echo "--------------------------------------------------------------"
printf '%-6s %-10s %-9s %-9s %s\n' 块 状态 raw_rmean DSL_rmean 备注

for i in $(seq "$LO" "$HI"); do
  OUT="eval_drop_b${i}"
  LOG="$LOGDIR/b${i}.log"
  ABL="$TMPDIR_CKPT/_drop_b${i}.pt"

  # ---- 可续跑:两个 ranks 都在就算完成 ----
  if [ -f "$OUT/ranks_ret_test.npz" ] && [ -f "$OUT/ranks_ret_test_emb.npz" ]; then
    printf '%-6s %-10s %-9s %-9s %s\n' "$i" "跳过" "-" "-" "已有结果"
    continue
  fi

  # ---- ① 造消融权重(内部断言恰好命中 3 个门)----
  if ! python experiments/gate_checks.py gate6-dropblock \
        --ckpt "$CKPT" --block "$i" --out "$ABL" >>"$LOG" 2>&1; then
    printf '%-6s %-10s %-9s %-9s %s\n' "$i" "失败" "-" "-" "gate6 出错 → $LOG"
    echo; echo "🔴 第一个失败的臂就停了(见 $LOG 末尾):"; tail -n 15 "$LOG"; exit 1
  fi

  # ---- ② 评测:必须覆盖 pretrained_path + output_dir(见文件头 ②)----
  if ! python -m torch.distributed.run --nproc_per_node=1 \
        --master_port=$((10000 + RANDOM % 20000)) \
        tasks_clip/retrieval.py "$CFG" \
        pretrained_path "$ABL" output_dir "$OUT" >>"$LOG" 2>&1; then
    printf '%-6s %-10s %-9s %-9s %s\n' "$i" "失败" "-" "-" "评测出错 → $LOG"
    echo; echo "🔴 第一个失败的臂就停了(见 $LOG 末尾):"; tail -n 25 "$LOG"; exit 1
  fi

  # ---- ③ 只把两个 r_mean 摘出来上屏 ----
  read -r RAW DSL < <(python - "$OUT" <<'PY'
import sys, os, json
d = sys.argv[1]; vals = {}
for tag, fn in (("raw", "ret_test_emb"), ("dsl", "ret_test")):
    p = os.path.join(d, f"{fn}.json")
    try:
        with open(p) as f:
            vals[tag] = f"{json.load(f).get('r_mean', float('nan')):.2f}"
    except Exception:
        vals[tag] = "?"
print(vals.get("raw", "?"), vals.get("dsl", "?"))
PY
)
  printf '%-6s %-10s %-9s %-9s %s\n' "$i" "完成" "$RAW" "$DSL" "$OUT"
  [ "${KEEP_CKPT:-0}" = "1" ] || rm -f "$ABL"
done

cat <<'EOF'
--------------------------------------------------------------
基线锚点:raw(ret_test_emb/,**主口径**) r_mean 61.55 · DSL(ret_test/) 62.28

⚠️ 上表的裸差**不是**判据。按 house rule 走配对 bootstrap,两口径同向且 CI 不含 0:
    python experiments/collect_results.py --compare eval_drop_b7 eval_base
  (先确保 eval_base/ 存在:COMMANDS.md §3.11 的 `run base ...`)

⚠️ 删块会移动下游每个 BN 的输入分布 ⇒ **评测前应做前向 BN 重校准**
  (experiments/per_t_bn.py)。且按 round-5 §11.3,重校准那一遍**必须 with_cp=False**,
  否则 use_reentrant=False 的检查点会把 BN running stats 更新两次。
  本脚本**没有**替你做重校准 —— 未校准的结果只能当粗筛,不能进论文。

判据(预先登记,§8.3 P11):没有任何单块代价 < +1.0 r_mean,或最便宜 6 块之和 > 2.0
⇒ 这份权重里深度不冗余,关掉这条线,预算全给 round4-P7(分辨率)。
EOF
