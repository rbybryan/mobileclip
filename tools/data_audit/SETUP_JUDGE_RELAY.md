# 中转机裁判执行手册 —— `10.55.87.68`（7×V100），逐条命令

> **这台机器的角色**：只做 **S2 的 4.3 裁判档**。它既连得到内网
> `Qwen3.6-27B` 端点（待实测），又挂载了 `/workspace/mount/b100_zaip_data`——
> 而 101vector 和 H100 **都连不上端点**（2026-07 实测），所以裁判只能落在这里。
>
> **不做什么**：不跑 S0/S1/S2 主力、不训练、不改数据。主力判定仍在 101vector。
>
> **不占 GPU**：推理在端点那边，本机只做「抽帧 → base64 → 发请求 → 解析 JSON」。
> 所以**不需要 torch，也不需要那 7 张卡**。
>
> 裁判在整条流水线里的位置与意义见
> [`SETUP_V100.md`](SETUP_V100.md) §4.2–4.3（一句话：**4.2 给数据打分，4.3 给 4.2 打分**）；
> 裁判选型（为什么是 27B、有什么代价）见 [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0。

**约定**：下面每条命令都标了在**哪台机器**上执行——
`[101]` = 101vector（10.235.31.101）、`[中转]` = 10.55.87.68。
每步末尾的**验收点**不过就停下，不要带病往下跑。

---

## 1. 三项前提实测（~5 分钟，任一失败即停）

### 1.1 `[中转]` 端点连通性 —— 最关键

```bash
ssh -p 31179 root@10.55.87.68
for ep in 172.30.85.198:8000 172.30.62.177:8000 172.30.86.57:5000; do
  echo -n "$ep: "
  timeout 5 curl -s -o /dev/null -w "%{http_code}\n" "http://$ep/v1/models" || echo unreachable
done
```

**验收点**：至少一个返回 `200`。**全是 `unreachable` → 这台也用不了**，
退回 [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0 的路线 C（本地 32B）或
路线 D（无大裁判协议）。

### 1.2 `[中转]` 确认对外模型 ID

```bash
export JUDGE_EP=http://172.30.85.198:8000/v1      # 用 1.1 里通的那个，优先 8 卡端点
curl -s $JUDGE_EP/models | python3 -m json.tool | head -20
```

**验收点**：`data[0].id` 应是
`/workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B`。
**以这里打印的字符串为准**（起服务时没设 `--served-model-name`，
对外 ID 就是权重路径），后面 `--model` 必须原样传。

### 1.3 `[中转]` 共享存储可见

```bash
ls -d /workspace/mount/b100_zaip_data && echo "存储 OK"
```

**验收点**：能列出。这是第 4 步「不用拷视频」的前提。

---

## 2. `[中转]` 环境（~3 分钟，无 GPU、无 torch）

`vlm_audit.py` 只用 `requests / numpy / pillow / decord`，是自包含脚本。

```bash
mkdir -p ~/judge && cd ~/judge
python3 -m venv ~/judge_env && source ~/judge_env/bin/activate
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip install "numpy<2" pillow decord requests
```

> `numpy<2` 是硬约束：decord 0.6 的预编译 wheel 在 numpy 2.x 下报 `_ARRAY_API` 错误。

```bash
python -c "import numpy, requests, decord, PIL; print('deps OK', numpy.__version__)"
```

**验收点**：打印 `deps OK 1.x`。

---

## 3. 把脚本和样本送过来（~2 分钟）

`vlm_audit.py` 不依赖仓库里任何其他文件，**只拷这一个脚本即可**，
不必在中转机上 clone 私有仓库。

```bash
# [101] 生成裁判样本（s2_sample.jsonl 的前 500 行）
cd ~/mobileclip
head -n 500 outputs/audit_10k/s2_sample.jsonl > outputs/audit_10k/s2_judge_sample.jsonl
wc -l outputs/audit_10k/s2_judge_sample.jsonl        # 应是 500

# [101] 送到中转机
scp -P 31179 tools/data_audit/vlm_audit.py \
             outputs/audit_10k/s2_judge_sample.jsonl \
             root@10.55.87.68:~/judge/
```

> ⛔ **不要在中转机上重新抽样。** 裁判必须评**主力判过的同一批记录**；
> 从全量 10M 另抽一批与 101vector 那 10k 的交集可能为空，
> 最后 `judge-agree` 只会告诉你「重叠不足 30 条」，几百条推理全白跑。
>
> **两机直连不通时**走两跳：先 `scp` 到你本地，再 `scp` 上去。文件只有几百 KB。

**验收点**：`[中转] ls -l ~/judge/` 能看到 `vlm_audit.py` 和
`s2_judge_sample.jsonl`（500 行）。

---

## 4. `[中转]` 视频解析预检 —— 决定要不要拷视频（~1 分钟）

10k 记录保留了 `abs_path`，而 `resolve_video_path` 的第三候选就是它的原值。
若指向共享存储上的原始位置，本机直接就能打开，**一个视频都不用拷**。

```bash
cd ~/judge && source ~/judge_env/bin/activate
python - s2_judge_sample.jsonl "${CLIPS:-/nonexistent}" <<'PY'
import json, os, sys, collections
sys.path.insert(0, ".")
from vlm_audit import resolve_video_path      # 与审核器同一套解析逻辑，不要手写
sample, root = sys.argv[1], sys.argv[2]
stats, miss = collections.Counter(), []
for line in open(sample, encoding="utf-8"):
    rec = json.loads(line)
    p = resolve_video_path(root, rec, rec.get("video"))
    if os.path.exists(p):
        stats["ok"] += 1
    else:
        stats["missing"] += 1
        if len(miss) < 5: miss.append((rec.get("video"), p))
print(dict(stats))
for v, p in miss: print("MISSING", v, "->", p)
PY
```

**验收点与分支**：

- `ok` = 500 → **不用拷视频**，直接进第 5 步（`--video-root` 传什么都行，
  实际命中的是 `abs_path` 那条候选）。
- `ok` 是 0 或很少 → 看打印的 `MISSING` 路径差在哪，按序尝试：
  1. 共享存储里若有语料视频根目录，`export CLIPS=<该目录>` 后重跑本预检；
  2. 仍不行 → 按
     [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0「若预检没通过：把那 500 条视频也拷过去」
     生成清单并 rsync（**只拷 500 条，不是全部 1 万条**）。

---

## 5. `[中转]` 冒烟 5 条（~2 分钟，必做）

```bash
cd ~/judge && source ~/judge_env/bin/activate
python vlm_audit.py \
  --input s2_judge_sample.jsonl \
  --video-root "${CLIPS:-/nonexistent}" \
  --api-base "$JUDGE_EP" --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 2 --limit 5 \
  --output s2_verdicts_judge.jsonl

head -1 s2_verdicts_judge.jsonl | python -m json.tool
```

**验收点**：每条都有完整的 `verdict` 对象，含
`en_score / zh_score / trans_score / action / better_zh`。

**冒烟输出直接写进正式产物文件是有意的**：`vlm_audit.py` 支持断点续跑
（按 `video` 去重跳过），所以这 5 条会计入第 6 步的 500 条，不会重复推理。

**若看到 `{"error": "no valid verdict JSON in response"}`**：几乎一定是
**Qwen3.x 的 thinking 模式没关掉** —— 模型把 `max_tokens` 全花在 `<think>` 里、
还没输出裁判 JSON 就被截断，而被截断的块没有闭合 `</think>`，剥离正则也救不回来。
本仓库已默认发送 `chat_template_kwargs.enable_thinking=False`，
**确认你拷过去的 `vlm_audit.py` 是最新版**（`grep -c enable_thinking vlm_audit.py` 应 ≥1）。
反之，若端点**拒收**该字段（报 400），才加 `--allow-thinking`。

---

## 6. `[中转]` 正式跑 500 条（~20–40 分钟）

```bash
cd ~/judge && source ~/judge_env/bin/activate
nohup python vlm_audit.py \
  --input s2_judge_sample.jsonl \
  --video-root "${CLIPS:-/nonexistent}" \
  --api-base "$JUDGE_EP" --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 8 \
  --output s2_verdicts_judge.jsonl > judge.log 2>&1 &

watch -n 30 'wc -l ~/judge/s2_verdicts_judge.jsonl; tail -3 ~/judge/judge.log'
```

**参数依据**（对齐内网既有调用代码 `request_multilabel_v3_resume.py`）：
`--concurrency 8` 远低于其 `MAX_WORKERS=32`，**对共享端点是安全的**；
超时默认已是 600 秒；重试带指数退避。跑得慢可以提到 16，仍在 house 值以内。

**中断了直接重跑同一条命令**即可续（按 `video` 跳过已完成的）。

**验收点**：`wc -l` 到 500；错误率检查——

```bash
python - <<'PY'
import json
rows = [json.loads(l) for l in open("s2_verdicts_judge.jsonl")]
err = [r for r in rows if "error" in (r.get("verdict") or {})]
print(f"总 {len(rows)}，失败 {len(err)} ({len(err)/max(len(rows),1):.1%})")
for r in err[:3]: print(" ", r["video"], r["verdict"]["error"][:90])
PY
```

失败率应 <5%。若大面积失败，回看第 5 步的 thinking 说明。

---

## 7. 回传并算一致率（~2 分钟）

```bash
# [101] 取回判定
cd ~/mobileclip
scp -P 31179 root@10.55.87.68:~/judge/s2_verdicts_judge.jsonl outputs/audit_10k/

# [101] 一致率 + Cohen's κ
bash tools/data_audit/run_v100.sh judge-agree
```

产出 `outputs/audit_10k/judge_agreement.md`。**三个门**：五类 `action` 一致率
≥75%、二值 `{keep,fix_text}` vs 其余 ≥80%、**各维度 Cohen's κ ≥0.6**。

**读法要点**：

- **别只看一致率**。类别不平衡下，双方一起说 `keep` 就能堆出高一致率；
  κ 才是扣掉「碰巧一致」后的真实一致程度。
- **`trans_bad` 那一行最关键** —— 决策 1（`--xling-threshold`）的全部差率都
  建立在它上面，它的 κ 不过关，`xling-check` 的裁决就不可信。
- **混淆矩阵看偏斜**：若「主力判 `recaption` / 裁判判 `retranslate`」这类格子
  特别大，说明 rubric 里这两类的边界没写清，**改 rubric 比换模型有用**。

**验收点**：三个门的结论记进
[`DATA_ISSUES.md`](DATA_ISSUES.md) §3 的 T4，**并注明裁判是 `Qwen3.6-27B`**。

> ⚠️ **证据强度要打折**：27B 裁 8B 只有约 3 倍参数差距，且**同厂同族**——
> 方案 §2 原本要的是 235B **加换一家**（InternVL）互为第二意见，正是为了避开
> 共同盲区。**同族模型的一致率会被共同盲区抬高，一致不等于正确。**
> 所以这一轮通过≠万事大吉：按 [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0
> 路线 D 的 D2/D3 提高人工抽检（300 条）的权重，那是唯一的独立信号。
> 若 H100 有空，可另起 235B-FP8（路线 A）复核同一批，无同族偏差。

---

## 8. 跑完之后

裁判只解掉 T4 一个门。回到 101vector 继续：

```bash
# [101] 决策 1：xling 阈值裁决（S2 主力跑完后）
bash tools/data_audit/run_v100.sh xling-check

# [101] 两个决策都落地后才跑 S3
bash tools/data_audit/run_v100.sh s3
```

---

## 9. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 全部 `no valid verdict JSON in response` | Qwen3.x thinking 未关，输出被截断在 `<think>` 里 | 用最新 `vlm_audit.py`（默认发 `enable_thinking=False`）；端点拒收该字段才加 `--allow-thinking` |
| 400 且报错提到 `chat_template_kwargs` | 端点不认该字段 | 加 `--allow-thinking`，并把 `--max-tokens` 提到 2048 留出 think 的余量 |
| 404 / `model not found` | 模型 ID 不对 | 以 §1.2 `curl $JUDGE_EP/models` 打印的字符串为准，原样传 |
| 大量 `decode:` 错误 | 视频没解析到 | 回到 §4 预检，先把 `ok` 修到 500 |
| 大量超时 / 429 | 端点被占满或本机并发过高 | 降 `--concurrency` 到 4；脚本已带指数退避，不必手动重试 |
| `judge-agree` 报「重叠不足 30 条」 | 裁判评的不是主力那批记录 | 十有八九是在中转机上重新抽了样；改用 101vector 的 `s2_judge_sample.jsonl` |
