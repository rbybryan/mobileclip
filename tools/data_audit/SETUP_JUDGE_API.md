# 跨机裁判部署手册：H100 起 235B 服务，V100 走 API 调用

> 目标：在 8×H100-80G 机上用 vLLM 起 **Qwen3-VL-235B-A22B-Instruct-FP8**
> 的 OpenAI 兼容服务，让 V100 机（101vector，10.235.31.101）的
> `s2-judge` 裁判步骤通过内网 API 调用它——**零云端费用**，且两台机器的
> 裁判判定出自同一个模型，校准结果可互相复用。
>
> 前置条件只有一个：**两台机器内网互通**（§3 有验证方法）。
> 配套手册：[`SETUP_H100.md`](SETUP_H100.md)（H100 全量审查）、
> [`SETUP_V100.md`](SETUP_V100.md)（V100 10k 审查）。

## 总览

```
V100 机 (101vector)                     H100 机
┌─────────────────────┐    内网 HTTP    ┌──────────────────────────┐
│ run_v100.sh s2-judge │ ─────────────> │ vllm serve 235B-FP8      │
│ (vlm_audit.py 抽帧、 │  8帧JPEG/条    │ --port 8001 --host       │
│  base64、发请求)     │ <───────────── │ 0.0.0.0（tp8 或 tp4）    │
└─────────────────────┘   JSON 判定     └──────────────────────────┘
```

每条请求只发 8 帧压缩 JPEG（约 1–2MB），抽帧解码都在 V100 本地做，
内网千兆带宽绰绰有余。

---

## 0. 裁判选型：不用商业模型会不会掉质量？

**先澄清两件事：**

1. **S3（`aggregate_report.py`）不含任何模型**——纯 Python 标准库、CPU，
   只做「按 S1 分位统计 S2 差率 → 选阈值 → 全量分桶 → 出报告」。
   那里没有任何商业依赖可去掉。
2. **整条流水线唯一的商业触点是 `s2-judge`，而它调的也不是商业模型**：
   `run_v100.sh` 默认 `JUDGE_MODEL=qwen3-vl-235b-a22b-instruct`，
   模型本身是 **Apache-2.0 开源权重**，DashScope 只是它的一个托管端点。
   所以「不用商业模型」= **换成自托管同一份权重**，判定完全一致、零质量损失。

按优先级选：

| 路线 | 裁判模型 | 需要什么 | 相对 235B 的质量 |
|---|---|---|---|
| **A（首选）** | Qwen3-VL-235B-A22B-Instruct-**FP8**，H100 机自托管 | 8×H100-80G + 两机内网互通 | **完全一致**（同一份权重）；见下文 §1–§5 |
| B | InternVL3.5-241B-A28B（同为开源） | 同上 | 同档；作为 A 的第二意见（plan §2 的交叉验证档） |
| **A′（有就先用）** | 网内**已经起好**的 vLLM 端点：B100 集群 `Qwen3.6-27B`，**✅ 已确认多模态** | 零部署、零下载、不占本机 GPU；**101vector/H100 连不上，须在中转机 `10.55.87.68` 上跑** | 27B ≈ 主力 8B 的 3 倍；**能连通就优于 C**（C 要下 64GB 且占满本机 8 卡） |
| C | Qwen3-VL-32B-Instruct，V100 本地 | 仅 8×V100-32G | 有损但可用（≈主力 8B 的 4 倍） |
| **D（H100 完全用不上时）** | 不用大裁判：换族 8B 交叉验证 + **分歧定向**人工抽检 | 零额外资源，仅 V100 本机 | 决策门改挂人工；见下方 D1–D4 完整协议 |

### 为什么 V100 上不了 235B（不是许可问题，是硬件）

Volta **没有 FP8 支持**，FP8 版权重跑不了；BF16 原版 ~470GB，
8×32GB=256GB 也放不下。所以 V100 本地裁判的天花板就是 ~32B 档。

### 路线 A′：复用网内**已经起好**的 vLLM 端点（零部署，优先于 C）

若网内已有别人起好的 OpenAI 兼容端点，就省掉了下载权重和起服务的全部工作，
也不占本机 8 张卡——**可用的话应排在路线 C 前面**。
已知候选（B100 集群，模型均为 `Qwen3.6-27B`，`max_model_len` 262144）：

| 端点 | 起法 |
|---|---|
| `http://172.30.85.198:8000/v1` | 8 卡 |
| `http://172.30.62.177:8000/v1` | 8 卡 |
| `http://172.30.86.57:5000/v1` | 4 卡 |

**探针一：是不是多模态 —— ✅ 已确认通过（2026-07，维护者实测）。**
`Qwen3.6-27B` **支持视觉输入**，`vlm_audit.py` 的 `image_url` 内容块可用，
路线 A′ 成立。网段连通性也已定案：**只有中转机 `10.55.87.68` 连得上**（见下）。

#### 已按内网参考实现对齐（2026-07，**已核对完毕**）

house convention 取自调用这批端点的既有代码
`/workspace/mount/b100_zaip_data/wky_b100/project/Qwen3_5_vllm_dev/request_multilabel_v3_resume.py`
（用 `openai` SDK + `ThreadPoolExecutor`）。逐项比对结果：

| 项 | 参考实现 | 我们的现状 | 结论 |
|---|---|---|---|
| **thinking 开关** | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | **原本没发** → 已改为默认发送 | ⚠️ **最关键的一项，见下** |
| 单请求超时 | `timeout=600` | 原 180 → **已改默认 600** | 已对齐 |
| 重试 | 3 次，`sleep(2 ** attempt)` 退避 | 原本重试**无退避**（立刻重打） → 已加同款指数退避 | 已修 |
| 并发 | `MAX_WORKERS = 32` | `s2-judge` 用 8 | ✅ **8 远低于 house 值，安全**（不必再担心打满） |
| 鉴权 | `api_key="EMPTY"` | `--api-key EMPTY` | 一致 |
| 模型 ID | 权重路径原样 | `--model` 原样传 | 一致 |
| 图像编码 | `data:image/{mime};base64,...`，**原图不缩放** | 同格式，但缩到 `--max-side 448` JPEG q80 | 我们更省，安全（对方全分辨率都跑得动） |
| 消息结构 | `system` + `user`(image→text) | 全部放 `user`(image→text) | 顺序一致，可用 |
| `temperature` | `0.0` | `0.0` | 一致 |
| `max_tokens` | 512（其任务） | 1024（我们要出 `better_zh`，需更多） | 合理 |

> 🔴 **`enable_thinking: False` 是必须的，不是可选优化。**
> Qwen3.x 的 thinking 模式**默认开启**。不关的话，模型会把整个 `max_tokens`
> 预算花在 `<think>` 里，还没轮到输出裁判 JSON 就被截断；而**被截断的 think 块
> 没有闭合的 `</think>`**，`extract_json` 的剥离正则也救不回来——结果是
> **每一条都返回 `{"error": "no valid verdict JSON in response"}`**，
> 整轮裁判白跑且报错信息还很有迷惑性。
> `vlm_audit.py` 现在**默认发送**该字段；只有当端点拒收这个字段时（某些云端
> 网关）才加 `--allow-thinking` 关掉。

（探针一的复测命令保留在此，换端点/换模型时可复用：）

`vlm_audit.py` 发的是 `image_url` 内容块（8 帧 base64 JPEG，见其 `audit_one`）——
**端点必须支持视觉输入**。模型名里没有 `VL`/`Vision` 字样时**不能想当然**。
在**能连到该端点的机器**上跑：

```bash
python - <<'PY'
import base64, io, requests
from PIL import Image
EP    = "http://172.30.86.57:5000/v1"                            # 改成要测的端点
MODEL = "/workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B"  # 见 /v1/models
buf = io.BytesIO(); Image.new("RGB", (64, 64), (220, 30, 30)).save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
body = {"model": MODEL, "max_tokens": 16, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": "这张图是什么颜色？只回答颜色。"}]}]}
r = requests.post(EP + "/chat/completions", json=body, timeout=60)
print("HTTP", r.status_code); print(r.text[:500])
PY
```

> 用 64×64 而不是 1×1：部分视觉塔会因图小于 patch size 直接报错，
> 那会把**多模态模型误判成纯文本**。

- 正常回答颜色（红/red）→ **多模态**（`Qwen3.6-27B` 已确认属于此类）；
- HTTP 400 且报错含 `multimodal`/`image`/`content` 不支持 → **纯文本**，见文末。

**探针二：哪台机器连得上 —— ✅ 已定案（2026-07），不必重测。**
`172.30.x.x` 与 `10.235.31.101` 是不同网段。实测：**101vector 连不上、H100 也连不上、
7 卡 V100 `10.55.87.68` 连得上**（前两个 8 卡端点 200，4 卡的不通）。
换端点/换机器时才需要复测：

```bash
# 在 101vector 上跑
for ep in 172.30.85.198:8000 172.30.62.177:8000 172.30.86.57:5000; do
  echo -n "$ep: "
  timeout 5 curl -s -o /dev/null -w "%{http_code}\n" "http://$ep/v1/models" || echo unreachable
done
```

**探针二不通时：借一台中转机跑裁判。**

> ⛔ **先说一个会让整件事白做的坑：不要在中转机上重新抽样。**
> 裁判的意义是**和主力判过的同一批记录**逐条比对；在中转机上从全量 10M 里
> 另抽一批，与 101vector 那 10k 的交集可能是空的，`judge-agree` 会直接告诉你
> 「重叠不足 30 条」。**必须把 101vector 生成的 `s2_judge_sample.jsonl` 原样
> 搬过去**（它就是 `s2_sample.jsonl` 的前 `JUDGE_N` 行）。

#### 指定中转机：`root@10.55.87.68 -p 31179`（7×V100）

**101vector 和 H100 都连不上端点**（2026-07 实测）。目前唯一可行的落点是
这台 7 卡 V100——**已确认它能访问 `/workspace/mount/b100_zaip_data` 存储**，
而端点的模型权重（`.../model_zoo/Qwen3.6-27B`）就在该挂载下，所以它也在 B100 网内。
**注意：全量语料并不在这个挂载上**——它在 `b100_zaip_data_luan/InternVid-10M-FLT`，
中转机不可达；裁判用 `b100_zaip_data` 下的 **10k 子集**即可（只读 500 条）。

两个附带好处：

1. **环境很轻**——`vlm_audit.py` 只用 `requests / numpy / pillow` ＋ 任一视频后端
   （`decord`/`opencv`/`av` 三选一），
   **不需要 torch，也不占 GPU**（推理在端点那边），不必照搬 101vector 那套
   cu118 audit_env；而且它是自包含脚本，**只拷这一个文件**即可，
   不必在中转机上 clone 私有仓库。
2. **不用拷视频**——同一份 10k 子集就在 `b100_zaip_data` 挂载下，`CLIPS` 指到它
   即可，**只需搬一个 500 行的 jsonl**。（注意：最初设想的「靠记录里的 `abs_path`
   直接打开」**实测不成立**，`abs_path` 指向 101vector 本地路径。）

> 📖 **完整逐条命令**（三项前提实测 → 轻量环境 → 送脚本和样本 → 视频解析预检
> → 冒烟 5 条 → 正式跑 500 → 回传算分 → 常见问题表）见
> **[`SETUP_JUDGE_RELAY.md`](SETUP_JUDGE_RELAY.md)**。本节只讲选型理由。

> ⛔ **不要在中转机上重新抽样。** 裁判必须评**主力判过的同一批记录**；
> 从全量 10M 另抽一批与 101vector 那 10k 的交集可能为空，`judge-agree`
> 只会报「重叠不足 30 条」，几百条推理全白跑。搬 `s2_judge_sample.jsonl`。

> 💡 **H100 虽然连不上端点，但它仍可以走路线 A**：在本机自托管 235B-FP8，
> 与 8B 的能力差距远大于 27B，且没有同族偏差问题（见本节末尾的警告）。
> 代价是 ~235GB 下载。**赶时间用这台中转机 + 27B 端点，要证据强度就在 H100 上
> 起 235B。**两条路互不冲突，可以先跑 27B 拿到初步一致率，再用 235B 复核。

#### 若视频解析预检没通过：把那 500 条视频也拷过去

即 [`SETUP_JUDGE_RELAY.md`](SETUP_JUDGE_RELAY.md) §4 的失败分支——中转机解析不到
视频时才需要这一步，**仍然不用拷全部 1 万条**，裁判只看 `JUDGE_N` 条。
先在 101vector 上挑出那批视频的清单：

```bash
source ~/audit_env/bin/activate && cd "$(git rev-parse --show-toplevel)"
OUT=outputs/audit_10k
CLIPS=/home/vca1/datasets/internvid_zh_10k/videos
head -n "${JUDGE_N:-500}" $OUT/s2_sample.jsonl > $OUT/s2_judge_sample.jsonl

python - "$OUT/s2_judge_sample.jsonl" "$CLIPS" > $OUT/judge_video_list.txt <<'PY'
import json, os, sys
sys.path.insert(0, "tools/data_audit")
from vlm_audit import resolve_video_path          # 复用同一套路径解析，避免手工拼错
sample, root = sys.argv[1], os.path.abspath(sys.argv[2])
missing = outside = 0
for line in open(sample, encoding="utf-8"):
    rec = json.loads(line)
    p = os.path.abspath(resolve_video_path(root, rec, rec.get("video")))
    if not os.path.exists(p):
        missing += 1; continue
    if p.startswith(root + os.sep):
        print(os.path.relpath(p, root))           # rsync --files-from 要相对路径
    else:
        outside += 1
        print("OUTSIDE_ROOT:", p, file=sys.stderr)
print(f"[missing] {missing}  [outside_root] {outside}", file=sys.stderr)
PY

wc -l $OUT/judge_video_list.txt        # 应 ≈ JUDGE_N；差太多先看上面的 stderr
```

拷过去（保持相对目录结构，中转机上的 `--video-root` 才能原样解析）：

```bash
RELAY="-p 31179 root@10.55.87.68"      # 按实际改；rsync 用 -e 传端口
rsync -av -e "ssh -p 31179" --files-from=$OUT/judge_video_list.txt \
  $CLIPS/ root@10.55.87.68:/data/judge_videos/
scp -P 31179 $OUT/s2_judge_sample.jsonl root@10.55.87.68:/data/
```

在中转机上跑裁判（只需 `requests`/`pillow`/`decord`，**不需要 GPU**——
推理在端点那边）：

```bash
python tools/data_audit/vlm_audit.py --input /data/s2_judge_sample.jsonl \
  --video-root /data/judge_videos \
  --api-base http://172.30.85.198:8000/v1 --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 8 --output /data/s2_verdicts_judge.jsonl
```

拷回来后在 101vector 上算一致率（**不必重跑裁判**）：

```bash
scp -P 31179 root@10.55.87.68:/data/s2_verdicts_judge.jsonl outputs/audit_10k/
bash tools/data_audit/run_v100.sh judge-agree
```

都不可行时退回路线 C（本地 32B）/ 路线 D（无大裁判协议）。

**两个探针都过 → 接进 `s2-judge`：**

```bash
# 在 101vector 上
export JUDGE_API_BASE=http://172.30.85.198:8000/v1     # 优先 8 卡端点（并发余量大）
export JUDGE_MODEL=/workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B
export JUDGE_API_KEY=EMPTY        # vLLM 默认不校验，但脚本要求非空
JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge
```

⚠️ 起服务时没设 `--served-model-name` 的话，vLLM 对外的模型 ID **就是那串
权重路径**，必须原样传给 `JUDGE_MODEL`——先 `curl $JUDGE_API_BASE/models` 确认。

**若探针一判定为纯文本**：`en_score`/`zh_score`/`action` 都要看画面，纯文本
模型给不了，**T4 的 action 一致率决策门无法用它复现**（那就走路线 D 的 D1–D4）。
但 rubric 里 `trans_score` 是**明确定义为纯文本**的（"仅比较中英两条文本：
翻译忠实+流畅度"，见 `vlm_audit.py` 的 `PROMPT`），纯文本模型可以独立复核它——
而这恰好是 [`DATA_ISSUES.md`](DATA_ISSUES.md) §4 **决策 1（xling 阈值）**唯一依赖的分
（`calibrate_xling.py` 的全部差率都来自 8B 的 `trans_score`）。
所以纯文本端点仍能解掉决策 1 的可信度问题，只是解不掉决策 2。
需要给 `vlm_audit.py` 加一个"不发帧"的文本模式（**尚未实现**，要用再提）。

> ⚠️ **同族偏差同样适用**：27B 裁 8B 只有约 3 倍参数差距，且同厂同族——
> 方案 §2 原本要的是 235B **+ 换一家**（InternVL）互为第二意见，正是为了避开
> 共同盲区。同族模型的一致率会被抬高，**一致不等于正确**。若决策门靠它通过，
> 须按路线 D 的 D2/D3 提高人工抽检权重作为唯一独立信号。

### 路线 C：V100 本地 32B 裁判（A/B/A′ 都不可用时）

32B fp16 约 64GB，单卡放不下 → 必须**模型并行**（`--device-map auto`），
放弃 8 卡数据并行。裁判只跑 500–2000 条，慢一点可以接受：

```bash
source ~/audit_env/bin/activate && cd "$(git rev-parse --show-toplevel)"
OUT=outputs/audit_10k
head -n 500 $OUT/s2_sample.jsonl > $OUT/s2_judge_sample.jsonl

python tools/data_audit/vlm_audit_hf.py \
  --input $OUT/s2_judge_sample.jsonl --video-root /home/vca1/datasets/internvid_zh_10k/videos \
  --model Qwen/Qwen3-VL-32B-Instruct --dtype fp16 \
  --device-map auto --num-shards 1 --shard-id 0 \
  --output "$OUT/s2_verdicts_judge.jsonl"
```

输出格式与 `vlm_audit.py` 相同。**注意**：`--device-map auto` 时模型已跨多卡，
不能再按卡起 8 个进程（必 OOM）——单进程用 `--num-shards 1`。

> 📖 **完整实操**（下载 / 校验 / 冒烟 / 两进程 ×4 卡把墙钟减半 / 一致率脚本 /
> 常见问题）见 **[`SETUP_V100.md`](SETUP_V100.md) §4.3-本地**。上面这条命令
> 只是最小可跑版本。

### 路线 D：**完全没有大裁判**时的替代校准协议（H100 用不上时走这条）

先认清裁判到底在干什么：**它不是真理，它只是把人工判定放大到几千条。**
效度的最终锚点从来是人，大裁判只是省人力。所以没有大裁判 ≠ 不能校准，
只是要**把校准集缩小、把人放回环上**。四步，全部只用 V100 本机资源：

**D1. 换族同档模型做第二意见**（~1 小时，8 卡数据并行，与主力同一套脚本）

```bash
source ~/audit_env/bin/activate && cd "$(git rev-parse --show-toplevel)"
OUT=outputs/audit_10k
head -n 1000 $OUT/s2_sample.jsonl > $OUT/s2_judge_sample.jsonl
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python tools/data_audit/vlm_audit_hf.py \
    --input $OUT/s2_judge_sample.jsonl \
    --video-root /home/vca1/datasets/internvid_zh_10k/videos \
    --model OpenGVLab/InternVL3_5-8B --dtype fp16 \
    --num-shards 8 --shard-id $i \
    --output "$OUT/s2_verdicts_x8b.{shard}.jsonl" &
done; wait
```

同档模型**证明不了**主力 8B 是对的（两边一样弱），但它能干一件同样有用的事：
**定位分歧**。两个不同家族的模型看同一条片子给出不同 `action`，说明这条
要么本身模糊、要么 rubric 没说清——这正是下一步该给人看的样本。

**D2. 分歧定向的人工抽检**（替代随机 300，人力效率高得多）

随机抽 300 条里可能 250 条是双方都判 `keep` 的显然样本，白看。改成：
**全部分歧样本（预计 10–20%，约 100–200 条）+ 随机对照 100 条**。
前者暴露判定边界，后者防止「只看难例」导致的乐观偏差。
在 `SETUP_V100.md` §6 的 Gradio 里过，逐条记 `video` + 人工 `action`
存成 `human_check.jsonl`，字段与 `s2_verdicts` 的 `verdict.action` 同名即可复用 §4 的统计脚本。

**D3. 重挂决策门**（T4 的 ≥75% 原本挂在 235B 上，现在挂人）

- **对照对象**：8B 主力 vs **人工**（不是 vs 同档模型）。
- **样本量**：随机对照那 100 条即可给出 ±~9% 的 95% CI；想收紧到 ±5%
  需要 ~300 条随机样本。**决定阈值时用随机对照子集算，别把分歧样本混进去**
  （那是过采样的难例，一致率天然偏低）。
- **指标**：5 类 `action` 精确匹配是很苛刻的口径，`recaption` vs
  `retranslate` 的边界连人都会犹豫。建议**先二值化再算**：
  `{keep, fix_text}`（文本侧可用）vs `{retranslate, recaption, drop}`（需干预）
  ——这才是 S3 分桶真正依赖的判别。二值一致率 ≥80% 即可放行去校准阈值；
  五类一致率单独记录作参考，不做门。

**D4. 自洽性检查**（零成本，测的是稳定性而非正确性）

同一批 200 条用主力 8B 跑两遍（`--frames 8` vs `--frames 6`，或
temperature>0 各跑一次），统计每个 rubric 维度的自身翻转率。
**翻转率高的维度 = 该维度的判定是噪声**，在 S3 里就别拿它当强证据
（例如 `temporal_order_correct` 若翻转率 >30%，时序结论只能定性用）。
这一步不需要任何新权重，S2 跑完随手就能做。

> **如果只是 235GB 下载/磁盘的问题、H100 本身可用**：别走 D，走 §0 表格里的
> 「磁盘/时间紧」一行——`Qwen3-VL-32B-Instruct` BF16 只有 ~64GB，
> 单张 H100-80G 就能 tp1 服务，仍是主力 8B 的 4 倍档。
> **如果手上有任何一张 Ampere 及以上的卡**（A100/A6000/4090 等），
> 优先把 32B 放那张卡上用 vLLM 服务，比 V100 的路线 C 快一个量级。

---

## 1.（H100 机）环境与权重

已按 `SETUP_H100.md` §0 装好环境的直接跳到 §2。

### 1.1 Python 环境（~10 分钟）

```bash
# venv 优先（国内 conda 镜像不可靠）；base 的 python ≥3.9 即可
python -m venv ~/audit_env && source ~/audit_env/bin/activate
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

pip install "vllm>=0.11"        # 会自动带上匹配的 torch cu12x
pip install "transformers>=4.57" decord pillow requests
# 若 decord 报 numpy _ARRAY_API 错误：pip install "numpy<2"

# 验收点：应打印 True 和 H100
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

网络不通（pip/HF 被 reset）时，先按 `SETUP_H100.md` 开头的"联网"段配好
代理，或参考 `README.md`「网络受限机器」一节换源/离线安装。

### 1.2 先检索机器上是否已有权重（可省 235GB 下载）

```bash
# ① 是否已有服务在跑（权重可能已经起好了）
ps aux | grep -i vllm | grep -v grep
ss -tlnp | grep -E ':(8000|8001)'

# ② 按名字搜常见位置（秒级）
for d in ~ /root /workspace /data /mnt /models /opt \
         ~/.cache/huggingface/hub ~/.cache/modelscope/hub; do
  [ -d "$d" ] && find "$d" -maxdepth 6 \( -type d -o -type l \) \
    \( -iname "*235b*" -o -iname "*qwen3-vl*" \) 2>/dev/null
done
# 装了 mlocate 更快：locate -i 235b | head
# HF 缓存的可用路径在 models--Qwen--*/snapshots/<hash>/ 那一层；
# ModelScope 缓存在 ~/.cache/modelscope/hub/models/Qwen/ 下

# ③ 名字搜不到按体积反查（235B 分片都是数 GB safetensors）
find / -xdev -type f -name "*.safetensors" -size +3G -printf '%h\n' 2>/dev/null \
  | sort | uniq -c | sort -rn | head
# -xdev 不跨挂载；/workspace/mount 等 NFS 挂载点单独跑（慢，放后台）
```

找到后**必须验证完整性和版本**（断过的下载会缺分片）：

```bash
DIR=/找到的/路径
du -sh "$DIR"    # FP8 版 ≈ 235G；BF16 原版 ≈ 470G
python - "$DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
print("model_type:", cfg.get("model_type"))          # 应含 qwen3_vl
qc = cfg.get("quantization_config", {})
print("量化:", qc.get("quant_method", "无（BF16 原版）"))  # 期望 fp8
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
have = [f for f in need if os.path.exists(os.path.join(d, f))]
print(f"分片: {len(have)}/{len(need)} {'齐全' if len(have)==len(need) else '不完整!'}")
tot = idx["metadata"]["total_size"]
size = sum(os.path.getsize(os.path.join(d, f)) for f in have)
print(f"字节: {size:,} / {tot:,} {'OK' if size >= tot else '下载不完整!'}")
PY
```

验证通过后二选一接入，之后手册命令原样可用：

```bash
ln -s "$DIR" ckpts/Qwen3-VL-235B-A22B-FP8          # 方式 A：软链进 ckpts/
# 方式 B：直接 serve 该路径，并用 --served-model-name 固定对外模型名：
#   vllm serve "$DIR" --served-model-name ckpts/Qwen3-VL-235B-A22B-FP8 ...
#   （V100 侧 JUDGE_MODEL 不用改）
```

若找到的是 **BF16 原版**（~470GB）：tp8 也能起（每卡 ~59GB 权重）但 KV
很紧——`--max-model-len` 压到 16384、必要时 `--gpu-memory-utilization 0.95`，
且无法再与 8B 主力对半分卡；裁判档对量化不敏感，长期用建议仍补 FP8 版。

### 1.3 下载 235B-FP8 权重（§1.2 没找到才需要；~235GB，放后台，看带宽 0.5–2h+）

```bash
pip install modelscope
mkdir -p ckpts
df -h .    # 验收点：磁盘剩余 ≥300GB 再开下
nohup modelscope download --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --local_dir ckpts/Qwen3-VL-235B-A22B-FP8 > download_235b.log 2>&1 &
tail -f download_235b.log   # 断了重跑同一命令即可断点续传
```

**验收点**：`du -sh ckpts/Qwen3-VL-235B-A22B-FP8` 约 235G，
且目录里有 `config.json` 和一批 `*.safetensors`。

---

## 2.（H100 机）起服务

两种方案二选一。关键参数是 `--host 0.0.0.0`（监听所有网卡，内网才能
访问）；`--port 8001` 与主力档的 8000 错开。

### 方案 A：8 卡独占 tp8（H100 空闲时，吞吐最高）

```bash
source ~/audit_env/bin/activate
nohup vllm serve ckpts/Qwen3-VL-235B-A22B-FP8 --port 8001 --host 0.0.0.0 \
  --tensor-parallel-size 8 --limit-mm-per-prompt image=8 --max-model-len 32768 \
  > judge_serve.log 2>&1 &
```

### 方案 B：GPU 对半（H100 同时在跑自己的 S2 主力时）

8B 主力 dp=4 占 GPU0-3，235B tp=4 占 GPU4-7。tp4 时每卡 ~59GB 权重，
KV 余量紧，`--max-model-len` 不要超 16384：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve ckpts/Qwen3-VL-8B-Instruct \
  --port 8000 --data-parallel-size 4 --limit-mm-per-prompt image=8 \
  --max-model-len 16384 > main_serve.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup vllm serve ckpts/Qwen3-VL-235B-A22B-FP8 \
  --port 8001 --host 0.0.0.0 --tensor-parallel-size 4 \
  --limit-mm-per-prompt image=8 --max-model-len 16384 > judge_serve.log 2>&1 &
```

### 2.1 等服务就绪（235B 加载要几分钟）

```bash
tail -f judge_serve.log        # 出现 "Application startup complete" 即就绪
curl -s http://localhost:8001/v1/models | python -m json.tool
```

**验收点**：curl 返回 JSON，`"id"` 字段为 `ckpts/Qwen3-VL-235B-A22B-FP8`
——这个字符串就是 V100 侧要填的 `JUDGE_MODEL`（vLLM 用启动路径作模型名）。

### 2.2 本机冒烟一条请求（可选但推荐）

```bash
curl -s http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "ckpts/Qwen3-VL-235B-A22B-FP8",
  "messages": [{"role": "user", "content": "用一句话自我介绍"}],
  "max_tokens": 50}' | python -m json.tool
```

**验收点**：返回里有正常中文回复（纯文本请求即可，图片链路由 V100 侧
`s2-judge` 冒烟验证）。

---

## 3.（V100 机）连通性验证

```bash
# <H100_IP> 换成 H100 机的内网地址
curl -s --connect-timeout 5 http://<H100_IP>:8001/v1/models | python -m json.tool
```

**验收点**：返回与 H100 本机 curl 相同的 JSON。失败排查顺序：

| 现象 | 原因与处理 |
|---|---|
| `Connection timed out` | 防火墙挡了 8001：H100 机 `iptables -I INPUT -p tcp --dport 8001 -j ACCEPT`（或找管理员开安全组）|
| `Connection refused` | 服务没起/没监听外网：确认进程活着、启动参数带 `--host 0.0.0.0`、`ss -tlnp \| grep 8001` 显示 `0.0.0.0:8001` 而非 `127.0.0.1:8001` |
| 能 ping 通但 curl 挂起 | V100 机 shell 里有 `http_proxy` 残留把内网请求带偏：`unset http_proxy https_proxy` 或 `export no_proxy=<H100_IP>` |
| 返回 404/模型名不对 | `JUDGE_MODEL` 要与 `/v1/models` 的 `"id"` 完全一致（含 `ckpts/` 前缀）|

---

## 4.（V100 机）跑裁判

```bash
source ~/audit_env/bin/activate && cd "$(git rev-parse --show-toplevel)"
export JUDGE_API_BASE=http://<H100_IP>:8001/v1
export JUDGE_MODEL=ckpts/Qwen3-VL-235B-A22B-FP8   # 以 /v1/models 返回为准
export JUDGE_API_KEY=EMPTY   # vLLM 未设 --api-key 时任意非空值；run_v100.sh 要求非空

# 先 5 条冒烟（验证抽帧→base64→API→JSON 解析全链路）：
JUDGE_N=5 bash tools/data_audit/run_v100.sh s2-judge
cat outputs/audit_10k/s2_verdicts_judge.jsonl | head -2

# 冒烟 OK 后正式跑（tp8 用 500–2000 条；对半 tp4 时建议先 500）：
JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge
```

跑完自动打印 **8B vs 裁判 action 一致率**（决策门 ≥75%，不达标先改
rubric 再重跑 S2，见 `SETUP_V100.md` §4）。

**并发说明**：`run_v100.sh` 里裁判并发是 8。H100 侧 tp8 独占时可以更高，
想加速可手动跑 `vlm_audit.py --concurrency 16`（对半 tp4 时保持 8）。

**前提**：`s2-judge` 需要 `outputs/audit_10k/s2_sample.jsonl` 已存在
（即 S1 + `s2-sample` 已完成）；裁判不依赖 S2 主力的结果，可与主力并行跑。

---

## 5. 收尾

```bash
# H100 机：裁判跑完释放显存（下次要用再起）
pkill -f "vllm serve ckpts/Qwen3-VL-235B-A22B-FP8"
```

H100 机自己的全量审查后续也要跑裁判（§4.3），服务可以留着不杀，
两边共用；只是注意别和全量 S2 主力抢卡（用方案 B 的对半布局）。

## 常见问题

- **OOM / 起不来**：确认没有别的进程占卡（`nvidia-smi`）；tp4 方案把
  `--max-model-len` 降到 16384、必要时加 `--gpu-memory-utilization 0.92`。
- **加载极慢**：FP8 权重 235GB 首次从磁盘读入需要几分钟属正常；放在
  NVMe 而非 NFS 挂载上会快很多。
- **一致率异常低（<50%）**：先抽几条 `s2_verdicts_judge.jsonl` 人工看
  ——大概率是 `JUDGE_MODEL` 填错导致 404 后大量空判定，或帧没发出去
  （检查 `s2_judge_sample.jsonl` 的 video 路径能否在 V100 本地解码）。
- **安全**：服务无鉴权暴露在内网。介意的话起服务加 `--api-key <随机串>`，
  V100 侧 `JUDGE_API_KEY` 填同一个值。
