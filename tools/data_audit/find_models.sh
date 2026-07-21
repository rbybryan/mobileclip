#!/usr/bin/env bash
# find_models.sh —— 探测「这台机器上到底有哪些模型能用」
#
# 场景：手册默认裁判是 Qwen3-VL-235B-A22B-Instruct-FP8，但服务器上不一定有
# （没下过 / 下了一半 / 放在别的挂载点 / 磁盘不够再下一份）。先跑这个脚本，
# 再决定是直接用、是软链进 ckpts/、还是退到 32B / 跨族 / 路线 D。
#
# 用法（仓库根目录，秒级；不写任何文件、不动权重）：
#   bash tools/data_audit/find_models.sh              # 本地扫描 + 在跑的服务 + 选型建议
#   bash tools/data_audit/find_models.sh --deep       # 加：按 safetensors 体积反查（慢，跨全盘）
#   bash tools/data_audit/find_models.sh --remote     # 加：探测 ModelScope/HF 镜像还能下什么
#   bash tools/data_audit/find_models.sh --verify DIR # 单个目录完整性核验（分片齐不齐）
#   ROOTS="/nas/models /workspace" bash tools/data_audit/find_models.sh   # 自定义搜索根
#
# 相关：SETUP_H100.md §0.1（本脚本的使用位置）、§4.3（裁判选型表）、
#       SETUP_JUDGE_API.md §1.2（手工检索版）/ §0 路线 D（完全没有大裁判时）。

set -u

DEEP=0
REMOTE=0
VERIFY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --deep)   DEEP=1 ;;
    --remote) REMOTE=1 ;;
    --verify) VERIFY="${2:-}"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（-h 看用法）" >&2; exit 1 ;;
  esac
  shift
done

# 每个搜索根的超时（NFS 挂载点可能很慢）；没有 timeout 命令时自动降级为不限时
FIND_TIMEOUT=${FIND_TIMEOUT:-60}
run_find() {  # run_find <dir> <find-args...>
  if command -v timeout >/dev/null 2>&1; then
    timeout "$FIND_TIMEOUT" find "$@" 2>/dev/null
  else
    find "$@" 2>/dev/null
  fi
}

# ---------------------------------------------------------------- 单目录核验
if [ -n "$VERIFY" ]; then
  python3 - "$VERIFY" <<'PY'
import json, os, sys
d = sys.argv[1]
if not os.path.isdir(d):
    sys.exit(f"不是目录: {d}")
cfg_path = os.path.join(d, "config.json")
if not os.path.exists(cfg_path):
    sys.exit(f"没有 config.json，不是权重目录: {d}")
cfg = json.load(open(cfg_path))
print("路径      :", d)
print("model_type:", cfg.get("model_type"), "  (Qwen3-VL 期望含 qwen3_vl)")
qc = cfg.get("quantization_config") or {}
print("量化      :", qc.get("quant_method", "无（BF16/FP16 原版）"))
idx_path = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx_path):
    idx = json.load(open(idx_path))
    need = sorted(set(idx["weight_map"].values()))
    have = [f for f in need if os.path.exists(os.path.join(d, f))]
    tot = idx.get("metadata", {}).get("total_size", 0)
    size = sum(os.path.getsize(os.path.join(d, f)) for f in have)
    print(f"分片      : {len(have)}/{len(need)} {'齐全' if len(have)==len(need) else '不完整!'}")
    print(f"字节      : {size:,} / {tot:,} {'OK' if tot and size >= tot else '下载不完整!'}")
    ok = len(have) == len(need) and tot and size >= tot
else:
    shards = [f for f in os.listdir(d) if f.endswith((".safetensors", ".bin"))]
    size = sum(os.path.getsize(os.path.join(d, f)) for f in shards)
    print(f"单体权重  : {len(shards)} 个文件，{size/2**30:.1f} GiB（无 index.json，无法校验总量）")
    ok = len(shards) > 0
for f in ("preprocessor_config.json", "tokenizer_config.json", "chat_template.json"):
    print(f"{f:26s}:", "有" if os.path.exists(os.path.join(d, f)) else "缺（vLLM 可能报错）")
print()
print("结论:", "可以直接 vllm serve 这个路径" if ok else "不可用 —— 重跑 modelscope download 断点续传补齐")
PY
  exit $?
fi

echo "=============================================================="
echo " 1. 机器算力 / 磁盘"
echo "=============================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used \
    --format=csv,noheader 2>/dev/null
  GPU_N=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  GPU_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
           | head -1 | awk '{printf "%d", $1/1024}')
else
  echo "（没有 nvidia-smi）"; GPU_N=0; GPU_GB=0
fi
echo
echo "当前目录可用磁盘："
df -h . | tail -1
DISK_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
DISK_GB=${DISK_GB:-0}

echo
echo "=============================================================="
echo " 2. 已经在跑的推理服务（权重可能已经起好了，不用重下）"
echo "=============================================================="
ps -eo pid,etime,cmd 2>/dev/null | grep -E "vllm|sglang|lmdeploy" | grep -v grep || echo "（无）"
echo
for p in 8000 8001 8002 8080 30000; do
  if command -v curl >/dev/null 2>&1; then
    r=$(curl -s -m 2 "http://127.0.0.1:$p/v1/models" 2>/dev/null)
    [ -n "$r" ] && echo "端口 $p 有 OpenAI 兼容服务： $(echo "$r" | tr -d '\n' | cut -c1-300)"
  fi
done

echo
echo "=============================================================="
echo " 3. 本机权重目录（含 config.json 的目录 = 一份模型）"
echo "=============================================================="
DEFAULT_ROOTS="./ckpts $HOME/ckpts $HOME/.cache/huggingface/hub $HOME/.cache/modelscope/hub
  $HOME/.cache/modelscope /root /workspace /data /mnt /models /opt/models /share $HOME"
CAND=$(mktemp)
SUMMARY=$(mktemp)
export MODELS_SUMMARY="$SUMMARY"
for d in ${ROOTS:-$DEFAULT_ROOTS}; do
  [ -d "$d" ] || continue
  run_find "$d" -maxdepth 6 -name config.json -type f -printf '%h\n' >> "$CAND"
done
sort -u "$CAND" -o "$CAND"

python3 - "$CAND" <<'PY'
import json, os, sys
rows = []
for d in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    try:
        cfg = json.load(open(os.path.join(d, "config.json")))
    except Exception:
        continue
    mt = cfg.get("model_type") or "?"
    idx = os.path.join(d, "model.safetensors.index.json")
    files = os.listdir(d)
    weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
    if not weights and not os.path.exists(idx):
        continue                       # 只有 config.json，不是可用权重
    quant = (cfg.get("quantization_config") or {}).get("quant_method", "-")
    if os.path.exists(idx):
        try:
            j = json.load(open(idx))
            need = sorted(set(j["weight_map"].values()))
            have = sum(os.path.exists(os.path.join(d, f)) for f in need)
            tot = j.get("metadata", {}).get("total_size", 0)
            gb = tot / 2**30
            state = "齐全" if have == len(need) else f"缺{len(need)-have}片!"
            shard = f"{have}/{len(need)}"
        except Exception:
            gb, state, shard = 0.0, "index损坏", "?"
    else:
        gb = sum(os.path.getsize(os.path.join(d, f)) for f in weights) / 2**30
        state, shard = "单体", str(len(weights))
    rows.append((gb, d, mt, quant, shard, state))

if not rows:
    print("（没扫到任何权重目录 —— 换 ROOTS= 指定挂载点，或加 --deep 按体积反查）")
else:
    rows.sort(reverse=True)
    print(f"{'大小GiB':>8}  {'model_type':<18} {'量化':<8} {'分片':<9} {'状态':<8} 路径")
    for gb, d, mt, q, shard, state in rows:
        print(f"{gb:8.1f}  {mt[:18]:<18} {q[:8]:<8} {shard:<9} {state:<8} {d}")
    print()
    print("说明：HF 缓存的可用路径是 models--*/snapshots/<hash>/ 那一层，可直接 vllm serve。")
    print("      '缺N片!' = 下载断过，重跑同一条 modelscope download 即断点续传。")

# 供第 4 节选型判断用（字段：大小GiB \t model_type \t ok/partial \t 路径）
with open(os.environ.get("MODELS_SUMMARY", "/dev/null"), "w") as fh:
    for gb, d, mt, q, shard, state in rows:
        fh.write(f"{gb:.1f}\t{mt}\t{'ok' if state in ('齐全','单体') else 'partial'}\t{d}\n")
PY

if [ "$DEEP" = "1" ]; then
  echo
  echo "=============================================================="
  echo " 3b. 按体积反查（>3GB 的 safetensors 所在目录；不跨挂载点）"
  echo "=============================================================="
  run_find / -xdev -type f -name "*.safetensors" -size +3G -printf '%h\n' \
    | sort | uniq -c | sort -rn | head -20
  echo "（NFS/独立挂载点要单独跑：ROOTS=/your/mount bash $0 --deep）"
fi

if [ "$REMOTE" = "1" ]; then
  echo
  echo "=============================================================="
  echo " 3c. 还能下什么（探测镜像连通性 + 候选仓库是否存在）"
  echo "=============================================================="
  for url in "https://modelscope.cn" "${HF_ENDPOINT:-https://hf-mirror.com}"; do
    code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$url" 2>/dev/null)
    echo "$url -> HTTP ${code:-超时}   (000 = 连不上，先配代理)"
  done
  echo "--- 候选仓库在 ModelScope 上是否存在（200 = 可下）---"
  for m in Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 Qwen/Qwen3-VL-32B-Instruct \
           Qwen/Qwen3-VL-30B-A3B-Instruct Qwen/Qwen3-VL-8B-Instruct \
           OpenGVLab/InternVL3_5-38B; do
    code=$(curl -s -o /dev/null -m 25 -w '%{http_code}' \
      "https://modelscope.cn/api/v1/models/$m" 2>/dev/null)
    echo "  $m -> ${code:-超时} $([ "$code" = "200" ] && echo '(可下)')"
  done
  echo "--- 按关键词搜镜像上还有哪些同族仓库（hf-mirror 代理了 HF 的 /api）---"
  for q in "${SEARCH:-Qwen3-VL InternVL3_5}"; do
    for kw in $q; do
      echo "  [$kw]"
      curl -sL -m 25 "${HF_ENDPOINT:-https://hf-mirror.com}/api/models?search=$kw&limit=15" \
        2>/dev/null | python3 -c "
import json,sys
try: ms = json.load(sys.stdin)
except Exception: sys.exit('    (搜索接口不通，改用 modelscope.cn 网页搜)')
for m in ms: print('   ', m.get('id'))
"
    done
  done
  echo "（自定义关键词：SEARCH='Qwen3-VL GLM-4.5V' bash $0 --remote）"
fi

echo
echo "=============================================================="
echo " 4. 选型建议（对照 SETUP_H100.md §4.3 的裁判选型表）"
echo "=============================================================="
# $SUMMARY 每行：大小GiB \t model_type \t ok|partial \t 路径（下面按路径名匹配型号）
have()    { awk -F'\t' -v p="$1" 'tolower($4) ~ p'                "$SUMMARY" | grep -q .; }
have_ok() { awk -F'\t' -v p="$1" 'tolower($4) ~ p && $3=="ok"'    "$SUMMARY" | grep -q .; }
first_ok() { awk -F'\t' -v p="$1" 'tolower($4) ~ p && $3=="ok" {print $4; exit}' "$SUMMARY"; }

echo "GPU: ${GPU_N} 张 / 单卡 ~${GPU_GB}GB    当前盘可用: ${DISK_GB}GB"
echo
if have_ok "235b"; then
  P=$(first_ok "235b")
  echo "✅ 本机已有 235B 裁判权重：$P"
  echo "   接入（之后手册命令原样可用）："
  echo "     ln -s \"$P\" ckpts/Qwen3-VL-235B-A22B-FP8"
  echo "   先核验完整性：bash $0 --verify \"$P\""
elif have "235b"; then
  echo "⚠️  有 235B 目录但分片不全 → 重跑同一条 modelscope download 断点续传："
  echo "     modelscope download --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \\"
  echo "       --local_dir ckpts/Qwen3-VL-235B-A22B-FP8 --max-workers 1"
elif have_ok "internvl"; then
  P=$(first_ok "internvl")
  echo "✅ 没有 235B，但有 InternVL 权重：$P"
  echo "   这是**跨族**裁判（比同族 235B 更有信息量，见 §4.3 同族警告），直接用："
  echo "     vllm serve \"$P\" --port 8001 --tensor-parallel-size 2 \\"
  echo "       --limit-mm-per-prompt image=8 --max-model-len 16384"
elif have_ok "32b|30b"; then
  P=$(first_ok "32b|30b")
  echo "✅ 没有 235B，但有 32B/30B 档：$P → 按 §4.3「磁盘/时间紧」行用它当裁判"
  echo "     vllm serve \"$P\" --port 8001 --tensor-parallel-size 1 \\"
  echo "       --limit-mm-per-prompt image=8 --max-model-len 16384"
elif [ "${DISK_GB:-0}" -ge 300 ]; then
  echo "❌ 本机没有大裁判权重，但磁盘够（${DISK_GB}GB ≥ 300GB）→ 后台下 235B-FP8："
  echo "     nohup modelscope download --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \\"
  echo "       --local_dir ckpts/Qwen3-VL-235B-A22B-FP8 --max-workers 1 \\"
  echo "       > download_235b.log 2>&1 &"
elif [ "${DISK_GB:-0}" -ge 90 ]; then
  echo "❌ 没有大裁判且磁盘只剩 ${DISK_GB}GB（<300GB）→ 退到 32B（~64GB，单卡 tp1）："
  echo "     modelscope download --model Qwen/Qwen3-VL-32B-Instruct \\"
  echo "       --local_dir ckpts/Qwen3-VL-32B-Instruct"
else
  echo "❌ 没有大裁判且磁盘只剩 ${DISK_GB}GB → 别再下大模型了，二选一："
  echo "   ① 云端裁判 API：SETUP_JUDGE_API.md（V100 侧 s2-judge 同款）"
  echo "   ② 走 SETUP_JUDGE_API.md §0「路线 D」：换族 8B 找分歧 + 分歧定向人工抽检"
fi
echo
echo "（裁判模型的硬性要求与同族偏差警告见 SETUP_H100.md §4.3；主力档仍是 8B）"
rm -f "$CAND" "$SUMMARY"
