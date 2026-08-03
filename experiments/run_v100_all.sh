#!/usr/bin/env bash
# =============================================================================
# V100 总队列(2026-07-31)—— **一个脚本跑完当前所有能在 V100 上做的测试**
#
#   bash experiments/run_v100_all.sh              # 全部阶段(约 3–5 小时)
#   bash experiments/run_v100_all.sh 0 1          # 只跑预检 + 冒烟(约 10 分钟)
#   bash experiments/run_v100_all.sh 6            # 只跑 10k 门(约 1–2 小时)
#
# 阶段:
#   0  预检           git 状态 / 就绪自检 / 配置能否被解析        ~2 min
#   1  🔺 冒烟        新代码 P9 + P1-trained 各几十步就停         ~10 min
#   2  gate3/gate4    P1 与 P5 的配对 CI(把裸差变成判据)        ~4 min
#   3  🔺 P7 前沿     分辨率 **× 帧数 T** —— 等像素前沿           ~20 min
#   4  多基准锚点     k400 / ucf101 / hmdb51                      ~30 min
#   5  2-D 前沿       分辨率 × 基准                               ~60 min
#   6  🔺 10k 门      P9 / P1-trained 的链路 go/no-go + 强制检查   ~2 h
#   7  微检查         A2/A7/A9/A10/A11/A12/A13/B1(全零数据)       ~20 min
#   9  A5 空间mixer   INT8 的 rho 直方图 —— 单独决定 AC9 生死       ~10 min
#  10  C-2 + C-4      聚合扫描 + 方差-vs-覆盖判别(一次前向多种打分) ~40 min
#   8  汇总           一张表 + PASS/FAIL 清单(永远放最后)         ~1 min
#
# 设计原则(踩过的坑都在这里):
#   ① **阶段之间不 fail-fast**:一个阶段炸掉不该让后面 60 小时的排队全废。
#      但**阶段 1 例外** —— 冒烟不过就不该进阶段 6,脚本会自己拦。
#   ② **可续跑**:已有 eval_res*.json 的臂跳过;10k 门已有 ckpt 的跳过。
#   ③ **每项一份日志**,终端只出结论行。`?` 打不出来时当场打诊断(见 #397)。
#   ④ **两条强制检查自动做**:P1 的 flag 是否真的生效、P9 的 W2 是否真的学动了。
#      这两条不自动化,就会重演 nA1「alpha ≲0.01 ⇒ 结果不可判读」那一次。
#
# ⚠️ 未做 BN 重校准 ⇒ 阶段 3/5 的分辨率数只能当**趋势筛**,不能进论文。
#    换分辨率会改**每一层**的激活统计,漂移远大于 P11 删块(见 experiments/dropblock_results.md §4)。
# =============================================================================
set -uo pipefail

CKPT=${CKPT:-./checkpoints/mobileviclip_small.pt}
CFGDIR=${CFGDIR:-scripts/evaluation/clip/zero_shot/mobileviclip_small}
LOGDIR=${LOGDIR:-logs/v100_all}
NGPU=${NGPU:-1}
STAGES=${*:-0 1 2 3 4 5 6 7 9 10 8}

export PYTHONPATH="${PYTHONPATH:-}:."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$LOGDIR"

RESULTS=()          # "PASS|FAIL|SKIP<TAB>名称<TAB>说明"
SMOKE_OK=1

has(){ for s in $STAGES; do [ "$s" = "$1" ] && return 0; done; return 1; }
note(){ RESULTS+=("$1"$'\t'"$2"$'\t'"$3"); printf '   %-6s %-26s %s\n' "$1" "$2" "$3"; }
hdr(){ echo; echo "=============================================================="; echo "### $*"; echo "=============================================================="; }

# eval_*.sh **不接受位置参数**(它把 output_dir 写死)⇒ 必须覆盖两个键,
# 否则评的是基线权重、且所有臂写进同一目录互相覆盖 ranks_*.npz(#392 的坑)。
run_eval(){  # run_eval <tag> <config.py> <output_dir> [extra k v ...]
  local tag="$1" cfg="$2" out="$3"; shift 3
  if [ -n "$(find "$out" -name 'eval_res*.json' 2>/dev/null | head -1)" ]; then
    note SKIP "$tag" "已有结果 $out"; return 0
  fi
  if python -m torch.distributed.run --nproc_per_node=1 \
        --master_port=$((10000 + RANDOM % 20000)) \
        tasks_clip/retrieval.py "$cfg" \
        pretrained_path "$CKPT" output_dir "$out" "$@" \
        >>"$LOGDIR/${tag}.log" 2>&1; then
    note PASS "$tag" "$(rmean "$out")"
  else
    note FAIL "$tag" "见 $LOGDIR/${tag}.log:$(tail -n 3 "$LOGDIR/${tag}.log" | tr '\n' ' ' | cut -c1-90)"
  fi
}

# 只用标准库读 eval_res*.json(结构 {protocol: {metric: value}});
# 解不出来时**当场**打诊断,不再打一个哑 `?`(#397)。
rmean(){
  python - "$1" <<'PY'
import sys, os, json
raw = dsl = None
for dirpath, _d, files in os.walk(sys.argv[1]):
    hit = next((f for f in ("eval_res.json","eval_res_best.json","eval_res_latest.json") if f in files), None)
    if not hit: continue
    try: blob = json.load(open(os.path.join(dirpath, hit)))
    except Exception: continue
    for proto, m in blob.items():
        if not isinstance(m, dict): continue
        v = m.get("r_mean") or m.get("acc")
        if v is None: continue
        p = proto.strip("/").replace("/", "_")
        if p.endswith("_emb"): raw = v
        else: dsl = v
if raw is None and dsl is None:
    f = [os.path.join(r,x) for r,_,fs in os.walk(sys.argv[1]) for x in fs if x.startswith("eval_res")]
    print("🔴 解不出 r_mean;目录内 eval_res*: " + (str(f[:2]) if f else "无") +
          " · 实有文件: " + " ".join(sorted(os.listdir(sys.argv[1]))[:6] if os.path.isdir(sys.argv[1]) else ["<目录不存在>"]))
else:
    print(f"raw {raw if raw is not None else '-'} · DSL {dsl if dsl is not None else '-'}")
PY
}

echo "=============================================================="
echo "V100 总队列 · 阶段 [$STAGES]"
echo "权重 $CKPT · 日志 $LOGDIR/ · commit $(git rev-parse --short HEAD 2>/dev/null || echo '非 git 工作树')"
echo "=============================================================="

# ------------------------------------------------------------------ 0 预检
if has 0; then
  hdr "阶段 0 · 预检"
  BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
  # 本仓反复栽在「服务器停在滞留分支上 ⇒ pull 了但没生效」(TROUBLESHOOTING 三次)
  if [ "$BR" = "main" ]; then note PASS "git分支" "main($(git rev-parse --short HEAD))"
  else note FAIL "git分支" "在 $BR,不是 main ⇒ 先 git checkout -B main origin/main"; fi

  [ -f "$CKPT" ] && note PASS "权重" "$CKPT" || note FAIL "权重" "缺 $CKPT"

  # configs/data.py:20-21 —— 这两个环境变量是训练/评测**唯一的隐式依赖**,
  # 没设就默认 data/ 与 anno_downstream/,在 101vector 上对不上。
  # 🔴 2026-07-31:上一版只把这两个变量**打印**出来就记 PASS,于是
  #    「MVC_DATA_DIR=<未设→data>」这行明明白白摆在屏幕上,却没人当回事 ——
  #    阶段 1 随后连崩两次,真因就是它(解码器去读不存在的 data/internvid_zh_10k/videos)。
  #    **能自动判定的事,不许只打印。** 现在:能自动补就补,补不上就 FAIL。
  if [ -z "${MVC_DATA_DIR:-}" ] && [ -d /home/vca1/datasets/internvid_zh_10k/videos ]; then
    export MVC_DATA_DIR=/home/vca1/datasets      # configs/data.py:104 记的就是这个路径
    note PASS "数据环境变量" "MVC_DATA_DIR 未设 → 自动补为 $MVC_DATA_DIR"
  else
    note PASS "数据环境变量" "MVC_DATA_DIR=${MVC_DATA_DIR:-<未设→data>} MVC_ANNO_DIR=${MVC_ANNO_DIR:-<未设→anno_downstream>}"
  fi
  VIDROOT="${MVC_DATA_DIR:-data}/internvid_zh_10k/videos"
  if [ -d "$VIDROOT" ] && [ -n "$(find "$VIDROOT" -name '*.mp4' 2>/dev/null | head -1)" ]; then
    note PASS "10k 视频根" "$VIDROOT"
  else
    note FAIL "10k 视频根" "🔴 $VIDROOT 不存在或没有 mp4 ⇒ 阶段 1/6 必炸。修法:export MVC_DATA_DIR=/home/vca1/datasets"
  fi
  for f in "${MVC_ANNO_DIR:-anno_downstream}/internvid_zh_10k_pilot.json" \
           "${MVC_ANNO_DIR:-anno_downstream}/internvid_zh_10k_pilot_val.json"; do
    [ -s "$f" ] && note PASS "10k anno" "$(basename "$f")" \
                || note FAIL "10k anno" "缺 $f ⇒ 阶段 1/6 必炸(tools/make_pilot_anno.py 生成)"
  done

  python experiments/check_readiness.py >"$LOGDIR/readiness.log" 2>&1
  note PASS "就绪自检" "见 $LOGDIR/readiness.log"
  for b in msrvtt k400 ucf101 hmdb51; do
    if grep -qiE "^\s*\[OK\s*\].*$b|$b.*\[OK\s*\]" "$LOGDIR/readiness.log" 2>/dev/null; then
      note PASS "数据:$b" "视频就位"
    else
      note SKIP "数据:$b" "未见 [OK ] ⇒ 相关阶段会失败,属预期"
    fi
  done

  # 配置能否被仓库自己的加载器解析(比等到训练第 1 步再炸便宜得多)
  for c in pilot_10k_P9 pilot_10k_P1trained sub1m_P9_s42 sub1m_P1trained_s42; do
    if python -c "
import sys; sys.path.insert(0,'.')
from utils.config import Config
c = Config.from_file('experiments/configs/$c.py')
ve = c.model.vision_encoder
print('$c', ve.get('temporal_pooling','mean'), ve.get('disable_temporal_branch', False))
" >>"$LOGDIR/configs.log" 2>&1; then
      note PASS "配置:$c" "$(tail -1 "$LOGDIR/configs.log")"
    else
      note FAIL "配置:$c" "解析失败 → $LOGDIR/configs.log"
    fi
  done
fi

# ------------------------------------------------------------------ 1 冒烟
if has 1; then
  hdr "阶段 1 · 🔺 新代码冒烟(几十步即停)—— 不过就别进阶段 6"
  for c in pilot_10k_P1trained pilot_10k_P9; do
    L="$LOGDIR/smoke_${c}.log"
    bash experiments/run_pretrain.sh "experiments/configs/${c}.py" "$NGPU" debug True \
      >"$L" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      note FAIL "冒烟:$c" "退出码 $rc → $L"; SMOKE_OK=0
      echo "   ---- $L 末 25 行 ----"
      sed -n 's/^/   | /p' <(tail -n 25 "$L")
      # 最常见的三类原因,直接在这里点名,省一轮往返
      grep -qiE "MVC_DATA_DIR|No such file|FileNotFoundError" "$L" && \
        echo "   💡 像是数据/路径:确认 export MVC_DATA_DIR=/home/vca1/datasets"
      grep -qiE "ModuleNotFoundError|ImportError" "$L" && \
        echo "   💡 像是缺包:看上一行是哪个模块"
      grep -qiE "AssertionError.*layer_scale_1d" "$L" && \
        echo "   💡 P1 的 40 断言没过:说明 layer_scale_1d 数目不是 40,先跑 gate 普查"
      echo "   --------------------------------"
      continue
    fi

    if [ "$c" = "pilot_10k_P1trained" ]; then
      # P1 的 flag 若没生效,臂**看起来照跑**,只是时序分支还活着 —— 静默失效,必须显式验
      if grep -q "时序分支已删除:40 个" "$L"; then
        note PASS "冒烟:$c" "flag 生效,40 个 layer_scale_1d 已冻结"
      else
        note FAIL "冒烟:$c" "🔴 日志里没有 '[P1] 时序分支已删除:40 个' ⇒ flag 没生效"; SMOKE_OK=0
      fi
    else
      note PASS "冒烟:$c" "跑通(W2 是否学动要看阶段 6)"
    fi
  done
  [ $SMOKE_OK -eq 1 ] && echo "   ⇒ 冒烟全过,阶段 6 可以跑" || echo "   ⇒ 🔴 冒烟未过,阶段 6 将被跳过"
fi

# ------------------------------------------------------------------ 2 gate CI
if has 2; then
  hdr "阶段 2 · gate3(P1 删时序分支)/ gate4(P5 删 temporal_embedding)的配对 CI"
  echo "裸差已知:gate3 −0.43/−0.33 · gate4 +0.70/+0.67(raw/DSL)。判胜负要 CI 不含 0。"
  for arm in eval_gate3 eval_gate4; do
    if [ ! -d "$arm" ] || [ ! -d eval_base ]; then
      note SKIP "配对CI:$arm" "缺 $arm/ 或 eval_base/"; continue
    fi
    okc=1
    for pref in "ret_test_emb/" "ret_test/"; do
      python experiments/collect_results.py --compare "$arm" eval_base --prefix "$pref" \
        >>"$LOGDIR/compare_${arm}.log" 2>&1 || okc=0
    done
    if [ $okc -eq 1 ]; then
      note PASS "配对CI:$arm" "$(grep -E '^\| (r_mean|R_sum)' "$LOGDIR/compare_${arm}.log" | tail -2 | tr '\n' ' ' | cut -c1-100)"
    else
      note FAIL "配对CI:$arm" "见 $LOGDIR/compare_${arm}.log(多半缺 ranks_*.npz)"
    fi
  done
fi

# ------------------------------------------------------------------ 3 分辨率
if has 3; then
  hdr "阶段 3 · 🔺 round4-P7 前沿:分辨率 **× 帧数 T**(P11 关线后唯一存活的效率线)"
  echo "为什么必须带 T:P7 的主张是**等像素**前沿 —— (256²,T=8) ≡ (181²,T=16) ≡ (128²,T=32)。"
  echo "只扫分辨率、把 T 钉在 8,拿到的是一条 1-D 切片,**回答不了 P7 问的那个 2-D 问题**。"
  # ① 分辨率轴(T=8,与既有基线锚点同口径)
  for R in 128 160 181 224 256; do
    run_eval "res${R}_T8" "$CFGDIR/config_msrvtt.py" "eval_res${R}_T8" \
             inputs.image_res "$R" model.vision_encoder.img_size "$R"
  done
  # ② 等像素网格:像素预算 ∝ T·R² ⇒ 沿对角线走才是「同成本换配比」
  for R in 128 181 256; do
    for T in 4 16; do
      run_eval "res${R}_T${T}" "$CFGDIR/config_msrvtt.py" "eval_res${R}_T${T}" \
               inputs.image_res "$R" model.vision_encoder.img_size "$R" \
               num_frames_test "$T"
    done
  done
  echo "   读法:比较**同一像素预算 T·R²** 的格子。已知 T-curve r_mean ≈ 63.0 − 15.2/T(GATE-0),"
  echo "   所以「降分辨率换更多帧」有没有净赚,是这张表唯一要回答的问题。"
fi

# ------------------------------------------------------------------ 4 多基准
if has 4; then
  hdr "阶段 4 · 多基准锚点 —— §6 已把多基准 R-Sum 从纪律升级为**必须**"
  echo "MSR-VTT 1k 在 1M 尺度上 10× 数据只涨 +0.82(落在 ±1.6 带内)⇒ 单基准不够用。"
  for b in k400 ucf101 hmdb51; do
    run_eval "${b}_base" "$CFGDIR/config_${b}.py" "eval_${b}_base"
  done
  echo "   参考(eval_k400.sh 内嵌官方数):k400 act_val/ r_mean ≈ 87.88"
fi

# ------------------------------------------------------------------ 5 2-D 前沿
if has 5; then
  hdr "阶段 5 · 2-D 前沿(分辨率 × 基准)—— P7 单靠 MSR-VTT 判不了,同样的分辨力问题"
  for b in k400 ucf101 hmdb51; do
    for R in 160 181 224; do
      run_eval "${b}_r${R}" "$CFGDIR/config_${b}.py" "eval_${b}_r${R}" \
               inputs.image_res "$R" model.vision_encoder.img_size "$R"
    done
  done
fi

# ------------------------------------------------------------------ 6 10k 门
if has 6; then
  hdr "阶段 6 · 🔺 10k 链路 go/no-go —— 决定 H100 那两条 15 小时的臂能不能上"
  if has 1 && [ $SMOKE_OK -ne 1 ]; then
    note SKIP "10k门" "阶段 1 冒烟未过,不浪费 2 小时"
  else
    for c in pilot_10k_P1trained pilot_10k_P9; do
      OUT="experiments/runs/${c}"
      if [ -n "$(find "$OUT" -name 'ckpt_*' 2>/dev/null | head -1)" ]; then
        note SKIP "10k:$c" "已有 ckpt"
      else
        if bash experiments/run_pretrain.sh "experiments/configs/${c}.py" "$NGPU" \
             >"$LOGDIR/${c}.log" 2>&1; then
          note PASS "10k:$c" "训练完成"
        else
          note FAIL "10k:$c" "见 $LOGDIR/${c}.log"
          echo "   ---- $LOGDIR/${c}.log 末 20 行 ----"
          sed -n 's/^/   | /p' <(tail -n 20 "$LOGDIR/${c}.log")
        fi
      fi
    done

    # 🔺 强制检查:P9 的 W2 到底学动了没有。
    # 仍≈0 ⇒ 退化成 mean ⇒ **该臂的任何 retrieval 数都不可判读**,更不该上 H100。
    # 这正是 nA1「alpha 仍 ≲0.01」那次踩的坑(verification-tier-1m-zh.md §怎么读结果 2)。
    W2=$(python - "experiments/runs/pilot_10k_P9" <<'PY' 2>/dev/null
import sys, os, glob, torch
best = None
for p in glob.glob(os.path.join(sys.argv[1], "**", "*.pt*"), recursive=True):
    try: sd = torch.load(p, map_location="cpu")
    except Exception: continue
    for key in ("module", "model", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict): sd = sd[key]
    if not isinstance(sd, dict): continue
    for k, v in sd.items():
        if "temporal_pool" in k and "w2" in k and hasattr(v, "norm"):
            n = float(v.float().norm())
            best = n if best is None else max(best, n)
print("" if best is None else f"{best:.6f}")
PY
)
    if [ -z "$W2" ]; then
      note SKIP "P9:‖W2‖" "没找到 temporal_pool.w2(ckpt 还没落盘?)"
    elif python -c "import sys; sys.exit(0 if float('$W2')>1e-4 else 1)"; then
      note PASS "P9:‖W2‖" "$W2 > 1e-4 ⇒ 学动了,可以上 H100"
    else
      note FAIL "P9:‖W2‖" "🔴 $W2 ≈ 0 ⇒ 退化成 mean,**别上 H100**,先调 different_lr"
    fi
  fi
fi

# ------------------------------------------------------------------ 7 微检查
if has 7; then
  hdr "阶段 7 · 微检查(A2 SDPA恒等 / A7 分辨率MACs / A11 P15等价 / A12 BN双更新 / B1 词表)"
  echo "A7 是**阶段 3 的成本轴** —— 没有它,阶段 3 的精度轴无法解读成前沿。"
  python experiments/v100_micro_checks.py all >"$LOGDIR/micro_checks.log" 2>&1
  mrc=$?
  while IFS= read -r line; do
    case "$line" in
      "[PASS]"*) note PASS "${line#\[PASS\] }" "" ;;
      "[FAIL]"*) note FAIL "${line#\[FAIL\] }" "见 $LOGDIR/micro_checks.log" ;;
    esac
  done < <(grep -E '^\[(PASS|FAIL)\]' "$LOGDIR/micro_checks.log" 2>/dev/null)
  grep -E '^    ' "$LOGDIR/micro_checks.log" 2>/dev/null | head -20
  [ $mrc -eq 0 ] || echo "   ⇒ 有微检查未过,细节见 $LOGDIR/micro_checks.log"
fi

# ------------------------------------------------------------------ 9 A5
if has 9; then
  hdr "阶段 9 · A5 空间 mixer 的 INT8 rho 直方图 —— 单独决定 AC9 是重新立论还是该埋"
  echo "§8 把这一项明确标注为「未验证」。gate5-fused 只量过**时序**核;"
  echo "配套那句「空间 mixer 只跑在 1–3 LSB」至今没人量过,而时序那次估计错了 1–2 个数量级。"
  if python experiments/gate_checks.py gate5-spatial \
       --ckpt "$CKPT" --config "$CFGDIR/config_msrvtt.py" \
       >"$LOGDIR/gate5_spatial.log" 2>&1; then
    note PASS "A5 空间mixer" "$(grep -E '头条|⇒' "$LOGDIR/gate5_spatial.log" | head -2 | tr '\n' ' ' | cut -c1-110)"
    grep -E '^    (分位|p[0-9])' "$LOGDIR/gate5_spatial.log" | head -8
  else
    note FAIL "A5 空间mixer" "见 $LOGDIR/gate5_spatial.log"
    sed -n 's/^/   | /p' <(tail -n 15 "$LOGDIR/gate5_spatial.log")
  fi
fi

# ------------------------------------------------------------------ 10 C-2/C-4
if has 10; then
  hdr "阶段 10 · C-2 聚合扫描 + C-4 方差-vs-覆盖判别(§4 那个给出**相反处方**的判别实验)"
  echo "一次前向拿逐帧嵌入,再用 {mean,max,LSE,top-2,单帧} 各打一次分 —— 不必重跑评测。"
  echo "读法:oracle(max) − mean **>1.0** ⇒ 覆盖说(处方=late-interaction);"
  echo "      ≈0 ⇒ 方差缩减说(处方=继续一阶池化,P9 那类二阶统计要另找立论)。"
  if python experiments/frame_agg_probe.py --config "$CFGDIR/config_msrvtt.py" \
       --ckpt "$CKPT" --ts 8,16 >"$LOGDIR/frame_agg.log" 2>&1; then
    note PASS "C-2/C-4 聚合判别" "$(grep -E '^\*\*oracle|⇒' "$LOGDIR/frame_agg.log" | head -2 | tr '\n' ' ' | cut -c1-110)"
    grep -E '^T=' "$LOGDIR/frame_agg.log" | head -12
  else
    note FAIL "C-2/C-4 聚合判别" "见 $LOGDIR/frame_agg.log"
    sed -n 's/^/   | /p' <(tail -n 20 "$LOGDIR/frame_agg.log")
  fi
fi

# ------------------------------------------------------------------ 8 汇总
if has 8; then
  hdr "阶段 8 · 汇总"
  python experiments/collect_results.py --dirs . --out experiments/v100_all_results.md >/dev/null 2>&1 \
    && echo "全量表 → experiments/v100_all_results.md" \
    || echo "⚠️ collect_results 失败"
fi

hdr "PASS / FAIL 清单"
p=0; f=0; s=0
for r in "${RESULTS[@]:-}"; do
  [ -z "$r" ] && continue
  case "${r%%$'\t'*}" in PASS) p=$((p+1));; FAIL) f=$((f+1));; SKIP) s=$((s+1));; esac
  printf '%s\n' "$r" | awk -F'\t' '{printf "  %-6s %-26s %s\n", $1, $2, $3}'
done
echo "--------------------------------------------------------------"
echo "PASS $p · FAIL $f · SKIP $s"
cat <<'EOF'

下一步:
  · 阶段 2 的 CI 不含 0 ⇒ P1 / P5 可写进论文
  · 阶段 6 全 PASS 且 ‖W2‖ > 1e-4 ⇒ H100 可排:
        bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_P1trained_s42 sub1m_P9_s42

⚠️ 阶段 3/5 未做 BN 重校准 ⇒ 只能当趋势筛。换分辨率改的是**每一层**的激活统计,
   漂移远大于 P11 删块;正式数要等 BN 重校准 CLI(per_t_bn.py 目前只有纯函数),
   且按 round-5 §11.3 那一遍必须 with_cp=False。
EOF
[ $f -eq 0 ]
