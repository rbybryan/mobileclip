#!/usr/bin/env bash
# round4-P7 等像素分辨率前沿(P11 关线后**唯一存活的效率线**)
#
#   bash experiments/run_resolution_sweep.sh                  # 128 160 181 224 256
#   bash experiments/run_resolution_sweep.sh 224 256          # 只跑这两个
#
# 为什么这条线能零训练跑:视觉塔**与分辨率无关**已核实 —— 无绝对位置嵌入、
# `RepCPE` 是卷积核不是网格、head 是 `GlobalPool` mean(见 v100-10k-advance-tests.md A7)。
# ⇒ 改 `img_size` **不需要插值、不需要重训**,直接换权重前向即可。
#
# 两个键**必须同时改**(只改一个会静默不一致):
#   inputs.image_res              —— 数据侧 resize/crop 的目标边长
#   model.vision_encoder.img_size —— 模型侧构图用的边长
# 嵌套键用点号,`utils/config.py:190` 的 `full_k.split(".")` 支持。
#
# 🔴 **未做 BN 重校准的分辨率数只能当趋势,不能进论文** —— 而且比 P11 删块更不能:
#    换分辨率会改**每一层**的激活统计(空间尺寸变了),BN 漂移远大于删一个块。
#    P11 的教训是「未校准值是代价上界」;这里连方向都可能被 BN 主导。
#    正式数要等 BN 重校准 CLI(`experiments/per_t_bn.py` 目前只有纯函数、无 CLI)。
set -uo pipefail

CKPT=${CKPT:-./checkpoints/mobileviclip_small.pt}
CFG=${CFG:-scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py}
LOGDIR=${LOGDIR:-logs/resolution}
RES_LIST=${*:-128 160 181 224 256}

export PYTHONPATH="${PYTHONPATH:-}:."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$LOGDIR"

[ -f "$CKPT" ] || { echo "找不到权重:$CKPT(用 CKPT=... 覆盖)"; exit 1; }
[ -f "$CFG"  ] || { echo "找不到配置:$CFG"; exit 1; }

echo "分辨率 $RES_LIST · 权重 $CKPT"
echo "每档日志 → $LOGDIR/r<res>.log ;终端只出一行/档"
echo "解析器 v3 · 脚本 commit $(git rev-parse --short HEAD 2>/dev/null || echo '非 git 工作树')"
echo "--------------------------------------------------------------"
printf '%-6s %-10s %-9s %-9s %s\n' 边长 状态 raw_rmean DSL_rmean 备注

for R in $RES_LIST; do
  OUT="eval_res${R}"
  LOG="$LOGDIR/r${R}.log"

  if [ -n "$(find "$OUT" -name 'eval_res*.json' 2>/dev/null | head -1)" ]; then
    printf '%-6s %-10s %-9s %-9s %s\n' "$R" "跳过" "-" "-" "已有结果"
    continue
  fi

  if ! python -m torch.distributed.run --nproc_per_node=1 \
        --master_port=$((10000 + RANDOM % 20000)) \
        tasks_clip/retrieval.py "$CFG" \
        inputs.image_res "$R" model.vision_encoder.img_size "$R" \
        pretrained_path "$CKPT" output_dir "$OUT" >>"$LOG" 2>&1; then
    printf '%-6s %-10s %-9s %-9s %s\n' "$R" "失败" "-" "-" "评测出错 → $LOG"
    echo; echo "🔴 第一个失败的档就停了(见 $LOG 末尾):"; tail -n 25 "$LOG"; exit 1
  fi

  read -r RAW DSL < <(python - "$OUT" <<'PY'
import sys, os, json
raw = dsl = "?"
names = ("eval_res.json", "eval_res_best.json", "eval_res_latest.json")
for dirpath, _d, files in os.walk(sys.argv[1]):
    hit = next((f for f in names if f in files), None)
    if not hit:
        continue
    try:
        with open(os.path.join(dirpath, hit)) as f:
            blob = json.load(f)
    except Exception:
        continue
    for proto, m in blob.items():
        if not isinstance(m, dict) or "r_mean" not in m:
            continue
        p = proto.strip("/").replace("/", "_")
        if p == "ret_test_emb":
            raw = f"{m['r_mean']:.2f}"
        elif p == "ret_test":
            dsl = f"{m['r_mean']:.2f}"
print(raw, dsl)
PY
)
  NOTE="$OUT"
  if [ "$RAW" = "?" ] && [ "$DSL" = "?" ]; then
    FOUND=$(find "$OUT" -name 'eval_res*.json' 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
      NOTE="$OUT  🔴 有 $FOUND 却没解出 r_mean —— 顶层键:$(python -c \
        "import json,sys;print(list(json.load(open(sys.argv[1])).keys()))" "$FOUND" 2>/dev/null)"
    else
      NOTE="$OUT  🔴 目录下没有 eval_res*.json;实有文件:$(ls "$OUT" 2>/dev/null | tr '\n' ' ' | cut -c1-120)"
    fi
  fi
  if [ -z "$(find "$OUT" -name 'ranks_*.npz' 2>/dev/null | head -1)" ]; then
    NOTE="$NOTE  ⚠️无 ranks_*.npz(配对 CI 做不了)"
  fi
  printf '%-6s %-10s %-9s %-9s %s\n' "$R" "完成" "$RAW" "$DSL" "$NOTE"
done

cat <<'EOF'
--------------------------------------------------------------
基线锚点(256²):raw(ret_test_emb/,**主口径**) r_mean 61.55 · DSL(ret_test/) 62.28

⚠️ **这一轮是趋势筛,不是判据。** 未做 BN 重校准 ⇒ 换分辨率改的是**每一层**的激活统计,
  漂移量远大于 P11 的删块。正式数必须等 BN 重校准 CLI,且按 §11.3 那一遍 `with_cp=False`。

配对 CI(两口径同向 + CI 不含 0):
    python experiments/collect_results.py --compare eval_res224 eval_res256
成本轴(免费、零数据,和精度轴配成前沿):v100-10k-advance-tests.md **A7**。
EOF
