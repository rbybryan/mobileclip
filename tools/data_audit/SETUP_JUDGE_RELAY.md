# 中转机裁判执行手册 —— `10.55.87.68`（7×V100），逐条命令

> **这台机器的角色**：只做 **S2 的 4.3 裁判档**。101vector 和 H100
> **都连不上端点**（2026-07 实测），而这台**连得上**——所以裁判只能落在这里。
>
> **§1 三项前提已全部实测通过（2026-07-21）**：
> `172.30.85.198:8000` 和 `172.30.62.177:8000` 返回 **200**
> （4 卡的 `172.30.86.57:5000` 返回 `000`，不可用，别再试）；
> 模型 ID 确认为 `/workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B`
> （`max_model_len` 262144）；共享存储可见。**从第 2 步接着做即可。**
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

> ✅ **实测（2026-07-21）**：`172.30.85.198:8000` → `200`、
> `172.30.62.177:8000` → `200`、`172.30.86.57:5000` → `000`（不通）。
> **用前两个 8 卡端点**；4 卡那个已确认不可用。

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

## 2. `[中转]` 环境（~2 分钟，无 GPU、无 torch）

> ⚠️ **别建空 venv。** 实测这台机器 **pip 出不去**（`mirrors.aliyun.com` 报
> `SSL: UNEXPECTED_EOF_WHILE_READING`，TLS 被中间设备掐断），空 venv 里什么都
> 装不上。但它本来就带 conda `base`（`/root/miniforge3`）且在做 grounding-dino
> 标注工作，**所需的包大概率已经齐了**——直接用 base，不要隔离。

`vlm_audit.py` 需要：`numpy` + `pillow` + `requests` + **任意一个视频后端**
（`decord` / `opencv-python` / `av`，三选一即可）。先查 base 里有什么：

```bash
deactivate 2>/dev/null; rm -rf ~/judge_env      # 清掉上一步可能建的空 venv
mkdir -p ~/judge && cd ~/judge
python3 - <<'PY'
mods = {"numpy": "numpy", "PIL": "pillow", "requests": "requests",
        "decord": "decord", "cv2": "opencv-python", "av": "av"}
have, miss = [], []
for m, pkg in mods.items():
    try:
        __import__(m); have.append(m)
    except ImportError:
        miss.append(f"{m}({pkg})")
core_ok = all(m in have for m in ("numpy", "PIL", "requests"))
backend = [m for m in ("decord", "cv2", "av") if m in have]
print("已有:", ", ".join(have) or "无")
print("缺失:", ", ".join(miss) or "无")
print("核心三件套:", "OK" if core_ok else "缺！")
print("可用视频后端:", backend or "无 —— 这是唯一的硬缺口")
PY
```

**验收点**：`核心三件套 OK` 且 `可用视频后端` 非空 → **环境就绪，直接进第 3 步**。

> ✅ **实测（2026-07-21）**：base 里已有 `numpy, PIL, requests, cv2`，
> 缺 `decord` 和 `av`。**核心三件套 OK，视频后端 = `cv2`，零安装即可开跑。**
> 下面第 5/6 步的命令已显式带上 `--frame-backend opencv`。

**若缺视频后端**（三个都没有），按成功率排序试：

```bash
# ① conda 装（走 conda 通道，不经 PyPI，这台机器最可能通）
conda install -y -c conda-forge opencv          # 或 av
# ② 换 pip 源 + 放宽 TLS（TLS 被拦时 http 源有时能过）
pip install opencv-python-headless \
  -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
# ③ 离线 wheel：在能上网的机器下好再 scp 过来
#    pip download opencv-python-headless -d /tmp/w   （能上网的机器）
#    pip install --no-index --find-links=/tmp/w opencv-python-headless
```

> **不必执着于 decord。** `vlm_audit.py` 支持 `--frame-backend
> {auto,decord,opencv,pyav}`，三种后端用**同一套 TSN 分段中心索引公式**，
> 已实测在同一文件上取到的帧**逐像素完全一致**。`auto`（默认）按
> decord → opencv → pyav 顺序挑第一个可用的，并在 stderr 打印
> `[frames] backend=xxx`。
>
> 之所以优先 decord：主力 8B 那边（`vlm_audit_hf.py`）用的就是它，同一个解码器
> 才能保证判的是同一帧。**换了后端就把它记到 T4 结论旁边**——不同解码器对同一
> 文件可能报出不同的总帧数，那会让索引落在稍不同的位置，是一致率的一个小混淆项。

---

## 3. 把脚本和样本送过来（~2 分钟）

`vlm_audit.py` 不依赖仓库里任何其他文件，**只拷这一个脚本即可**，
不必在中转机上 clone 私有仓库。

> 🔴 **拷之前必须 `git pull`。** 有**两个**关键修复只存在于新版
> `vlm_audit.py` 里，拷了旧版会在第 5 步以很有迷惑性的方式失败：
> ① `chat_template_kwargs.enable_thinking=False`（不发的话 Qwen3.x 的 thinking
> 会吃光 `max_tokens`，**每一条都返回 `no valid verdict JSON`**）；
> ② `--frame-backend`（旧版硬依赖 `decord`，而中转机装不上）。

```bash
# [101] 先更新到最新代码
cd ~/mobileclip && git pull origin main

# [101] 生成裁判样本（s2_sample.jsonl 的前 500 行）
head -n 500 outputs/audit_10k/s2_sample.jsonl > outputs/audit_10k/s2_judge_sample.jsonl
wc -l outputs/audit_10k/s2_judge_sample.jsonl        # 应是 500

# [101] 送到中转机
scp -P 31179 tools/data_audit/vlm_audit.py \
             outputs/audit_10k/s2_judge_sample.jsonl \
             root@10.55.87.68:~/judge/
```

```bash
# [中转] 确认拷过去的是新版（两个都必须 ≥1，任一为 0 就是拷了旧版）
cd ~/judge
echo -n "enable_thinking: "; grep -c enable_thinking vlm_audit.py
echo -n "frame-backend:   "; grep -c "frame-backend" vlm_audit.py
```

> ⛔ **不要在中转机上重新抽样。** 裁判必须评**主力判过的同一批记录**；
> 从全量 10M 另抽一批与 101vector 那 10k 的交集可能为空，
> 最后 `judge-agree` 只会告诉你「重叠不足 30 条」，几百条推理全白跑。
>
> **两机直连不通时**走两跳：先 `scp` 到你本地，再 `scp` 上去。文件只有几百 KB。

**验收点**：`[中转] ls -l ~/judge/` 能看到 `vlm_audit.py` 和
`s2_judge_sample.jsonl`（500 行）；上面两个 `grep -c` 分别是 **2** 和 **1**
（只要都 ≥1 即可）。

---

## 4. `[中转]` 视频解析预检 —— 决定要不要拷视频（~1 分钟）

10k 记录保留了 `abs_path`，而 `resolve_video_path` 的第三候选就是它的原值。
若指向共享存储上的原始位置，本机直接就能打开，**一个视频都不用拷**。

```bash
cd ~/judge          # 用 conda base，见 §2；不要激活空 venv
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
- `ok` 是 0 或很少 → 按下面的 §4.1 定位，别急着拷视频。

> 📌 **读懂这里的输出**：`resolve_video_path` 全部落空时返回的是**候选 1**
> （`video_root/<video>`），所以打印出来的那条路径**不代表**它试过的全部。
> 候选 2/3 只在记录里**有 `abs_path` 字段**时才存在。所以第一件事是确认该字段
> 在不在——这决定了是「路径根不对」还是「压根没有备用候选」。

> 📍 **`CLIPS` 就设这个（2026-07-22 确认）**——裁判机上有**同一份 10k 子集**：
>
> ```bash
> export CLIPS=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/internvid_zh_10k
> ```
>
> **走过的弯路，别再走**：全量语料在
> `/workspace/mount/b100_zaip_data_**luan**/InternVid-10M-FLT/`
> （`InternVId-FLT_1…11`），但**那个 `luan` 挂载在裁判机上不可达**——
> 它只在另一台（`…cka…`）上能看到。裁判机可达的是 `b100_zaip_data`，
> 而 10k 子集正好在它下面，**用子集就够**（裁判只看这 500 条）。
>
> 顺带解释为什么之前怎么设都不对：样本的 `video` 字段形如
> `videos/InternVId-FLT_10/shard_00044/InternVId-FLT_10/xxx.mp4`，那个
> `videos/` 前缀是各机器本地的。`resolve_video_path` 现在**额外尝试 `video`
> 字段里从 `InternVId` 起的后缀**（2026-07-22 新增，已用真实路径形状测过），
> 所以 `CLIPS` 指到**含 `videos/` 的那一层**（候选 1 命中）或**再往下一层的
> `videos/` 本身**（候选 2 命中）**都能解析**，不必纠结。

### 4.1 `[中转]` 全 missing 时的定位（~3 分钟）

```bash
# ① 记录里到底有哪些路径字段？
cd ~/judge
python3 - <<'PY'
import json
rec = json.loads(open("s2_judge_sample.jsonl").readline())
print("字段:", sorted(rec.keys()))
print("video    =", rec.get("video"))
print("abs_path =", rec.get("abs_path", "<<< 没有这个字段 >>>"))
PY
```

- **有 `abs_path`** → 它指向的多半是 101vector 上的本地路径，本机自然没有；
  但候选 2（`CLIPS` + `abs_path` 里从 `InternVId` 起的后缀）只要 `CLIPS` 设对就能命中。
- **没有 `abs_path`** → 只剩候选 1，**必须把 `CLIPS` 设成正确的根**才行。

```bash
# ② 确认 10k 子集在本机可见，并核对目录层级
export CLIPS=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/internvid_zh_10k
ls "$CLIPS" | head                       # 应能看到 videos/ 等
ls "$CLIPS/videos" | head                # 应是 InternVId-FLT_* 若干目录

# 直接找样本第一条那个 clip
FN=$(python3 -c "import json;print(json.loads(open('s2_judge_sample.jsonl').readline())['video'].split('/')[-1])")
echo "找: $FN"
find "$CLIPS/videos" -name "$FN" 2>/dev/null | head -3
```

> 这里**看不到**（路径变了/权限不够）→ 才走本节末尾的「拷 500 条视频」。

```bash
# ③ 重新拉最新 vlm_audit.py（候选 2 是 2026-07-22 才加的），再重跑 §4 预检
#    [101] 上执行：
#      cd ~/mobileclip && git pull origin main
#      scp -P 31179 tools/data_audit/vlm_audit.py root@10.55.87.68:~/judge/
#    [中转] 上确认 CLIPS 已 export（见 ②），然后重跑 §4 那段预检，ok 应变成 500
```

**换个根时的反推规则**：把 ② 找到的**完整路径**与记录的 `video` 字段右对齐，
**公共后缀之前的那一截就是 `CLIPS`**。例如

```
实际      <CLIPS>/videos/InternVId-FLT_10/shard_00044/InternVId-FLT_10/xxx.mp4
video 字段         videos/InternVId-FLT_10/shard_00044/InternVId-FLT_10/xxx.mp4
=> CLIPS = 上面那截 <CLIPS>
```

- **共享存储里根本没有这批视频** → 才走拷贝路线：按
  [`SETUP_JUDGE_API.md`](SETUP_JUDGE_API.md) §0「若预检没通过：把那 500 条视频也拷过去」
  生成清单并 rsync（**只拷 500 条，不是全部 1 万条**，几百 MB 量级）。

---

## 5. `[中转]` 冒烟 5 条（~2 分钟，必做）

```bash
cd ~/judge          # 用 conda base，见 §2；不要激活空 venv
python vlm_audit.py \
  --input s2_judge_sample.jsonl \
  --video-root "${CLIPS:-/nonexistent}" \
  --api-base "$JUDGE_EP" --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 2 --limit 5 \
  --frame-backend opencv \
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
cd ~/judge          # 用 conda base，见 §2；不要激活空 venv
nohup python vlm_audit.py \
  --input s2_judge_sample.jsonl \
  --video-root "${CLIPS:-/nonexistent}" \
  --api-base "$JUDGE_EP" --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 8 \
  --frame-backend opencv \
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
