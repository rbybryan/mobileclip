# 训练数据下载 / 抓取：InternVid-10M-FLT（具体操作）

本文档手把手说明如何为 **MobileViCLIP 预训练**准备训练视频数据。零样本评测**不需要**训练数据；
只有你要跑 `scripts/pretraining/...` 时才需要本文。

> 配套脚本：[`tools/download_internvid.py`](../tools/download_internvid.py)。
> 简版说明见 [`README_zh.md` 第 8 节](../README_zh.md)。

---

## 0. 先搞清楚：你需要的到底是什么

`configs/data.py` 把预训练语料 **InternVid-10M-FLT** 切成 **11 个分片**
（`internvid_v1 … v11`，合并为 `internvid_all`）。每个分片 = **标注 + 视频**两部分：

| 组成 | 路径（第 N 个分片） | 状态 |
|------|--------------------|------|
| 标注 JSON | `anno_downstream/InternVid/InternVid-10M-flt_<N>.json` | ✅ **已在 README 第 5 步从 Google Drive 下好** |
| 视频文件 | `data/video_sample/InternVId-FLT_<N>/` | ❌ **需要你自己抓** —— 就是本文要做的事 |

标注 JSON 是一个列表，每条形如：

```json
{"video": "<某个文件名>.mp4", "caption": "a man is riding a horse ..."}
```

训练时 `dataset/pt_dataset.py` 用
`os.path.join(data_root, anno["video"])` 读取视频，即
`data/video_sample/InternVId-FLT_<N>/<video 字段>`。
**所以你的任务：把每条 `video` 指向的那个 `.mp4` 文件，放到对应分片目录下、且文件名一模一样。**

### 数据从哪来？为什么要“抓”？

InternVid **不直接托管视频**（版权在 YouTube）。它只发布**元数据**：每个片段对应一个
YouTube 视频 id + 起止时间戳。所以流程是：

```
YoutubeID + [Start, End]  --yt-dlp-->  下载该时间段  --ffmpeg-->  切成 clip.mp4  -->  放进分片目录
```

官方元数据（`InternVid-10M-FLT-INFO.jsonl`，约 2.81 GB）列名为：
`YoutubeID`、`Start_timestamp`、`End_timestamp`、`Caption`、`Aesthetic_Score`、`UMT_Score`。

> 💡 **拿不到作者那 11 份 JSON？**（`anno_downstream/InternVid/` 来自作者的 Google Drive，
> 若该文件夹**私有 / 需申请访问**）——预训练标注可以**绕开作者文件夹**，直接从官方 InternVid
> 元数据自建，见 [附录 B](#附录-b拿不到作者-json-时用官方元数据自建)。注意：**下游评测**标注
> （msrvtt/anet/k400…）没有这条捷径，仍需作者文件夹。

---

## 1. 环境准备

```bash
# 抓取 + 切分工具
pip install -U yt-dlp
sudo apt-get install -y ffmpeg      # 若未装

# 校验
yt-dlp --version
ffmpeg -version | head -1
```

> ⚠️ **磁盘**：单个分片就有近百万 clip，全量 10M 是 **数 TB** 级。请先用 `--limit` 做小规模验证，
> 确认磁盘够再放量。

---

## 2. 第一步永远是：先看清标注里的 `video` 字段

不同来源的标注，`video` 命名可能不同（有的直接是 `<id>_<start>_<end>.mp4`，有的是相对路径或纯 id）。
**动手抓之前先 inspect 一个分片**，脚本会告诉你能否自动解析：

```bash
python tools/download_internvid.py inspect \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
  --samples 10
```

输出示例：

```
# entries: 998572
# keys in first entry: ['video', 'caption']
{"video": "dQw4w9WgXcQ_12.0_22.5.mp4", "caption": "...", "parsed": ["dQw4w9WgXcQ", 12.0, 22.5]}
...
# regex matched 10/10 sampled names
```

- **matched N/N** → 文件名已编码 `id_起_止`，可直接进第 3 步。
- **matched 0/N** → 命名不同，看下面 [附录 A](#附录-a当文件名无法直接解析时) 用官方元数据来对齐。

---

## 3. 抓取一个分片（先小样本冒烟，再放量）

### 3.1 冒烟测试（强烈建议先做）

只抓前 200 条，确认整条链路通：

```bash
python tools/download_internvid.py download \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
  --out  data/video_sample/InternVId-FLT_1 \
  --limit 200 --workers 4
```

脚本会：并发调用 `yt-dlp --download-sections "*start-end"` 只下所需时间段、切成 mp4、按 `video`
字段命名写入 `--out`；已存在的文件自动跳过；每次结果追加到 `data/video_sample/InternVId-FLT_1/_download_log.tsv`。

结束会打印：`DONE: ok=… skip=… fail=…`。有一定比例 `fail` 是**正常的**（见第 5 节）。

### 3.2 放量抓整个分片

去掉 `--limit`、按机器/网络调大并发：

```bash
python tools/download_internvid.py download \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
  --out  data/video_sample/InternVId-FLT_1 \
  --workers 8
```

11 个分片依次替换 `_1` → `_2 … _11`（可写个循环）：

```bash
for N in $(seq 1 11); do
  python tools/download_internvid.py download \
    --anno anno_downstream/InternVid/InternVid-10M-flt_${N}.json \
    --out  data/video_sample/InternVId-FLT_${N} \
    --workers 8
done
```

> 断点续跑：脚本对已存在文件返回 `skip`，直接重跑同一条命令即可继续。

---

## 4. 只想先跑通训练：单分片即可

不必等全量 10M。抓 **1 个分片的一小部分**，把训练语料从 `internvid_all` 改成单分片即可启动：

```bash
# 抓少量
python tools/download_internvid.py download \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
  --out  data/video_sample/InternVId-FLT_1 \
  --limit 2000 --workers 8
```

编辑 `scripts/pretraining/clip/mobileviclip_tiny/config.py`：

```python
train_corpus = "internvid_v1"     # 原为 "internvid_all"
```

然后按 `README_zh.md` 第 9 节启动 8×V100 训练。跑通后再逐步补数据、换回 `internvid_all`。

---

## 5. 会遇到的坑（都在预期内）

| 现象 | 原因 / 处理 |
|------|-------------|
| 部分 `fail`，抓不全 | 视频被删/设私有/地区限制。**正常**：config 里 `jump_filter=True`，且 `pt_dataset.py` 加载失败会随机重采样，缺样本不影响训练启动。 |
| 抓了几万个后大量 `fail: HTTP 429 / Sign in to confirm` | YouTube 封 IP（约 3–5 万片段）。用**代理池**、多机分批、降 `--workers`、隔段时间再续跑。 |
| `Sign in to confirm you're not a bot` / 年龄限制 | 传 cookies：`--cookies cookies.txt`（浏览器导出，或 `yt-dlp --cookies-from-browser chrome` 生成）。 |
| 下载很慢 | 提高 `--workers`；`yt-dlp` 可加 `--concurrent-fragments`；换更快出口。 |
| 磁盘爆了 | clip 很多，先 `--limit` 估算单条平均大小 × 条数；分盘存放，软链到分片目录。 |

> **合规提醒**：请遵守 YouTube 服务条款与各视频版权，仅将数据用于研究用途。

---

## 6. 验证

```bash
# 每个分片抓了多少
for N in $(seq 1 11); do
  d=data/video_sample/InternVId-FLT_${N}
  [ -d "$d" ] && printf "%s: %s mp4\n" "$d" "$(find "$d" -name '*.mp4' | wc -l)"
done

# 抽查一个能被 decord 正常打开（训练用的解码器）
python - <<'PY'
import glob, decord
f = glob.glob("data/video_sample/InternVId-FLT_1/*.mp4")[0]
vr = decord.VideoReader(f)
print(f, "frames:", len(vr))
PY
```

能数出文件、且 decord 能读，就说明数据可用于训练。

---

## 附录 A：当文件名无法直接解析时（用官方元数据对齐）

如果第 2 步 `inspect` 显示 `matched 0/N`，说明 `video` 字段没有直接编码 `id_起_止`。
这时用官方 **FLT-INFO** 元数据来补上 YouTube id + 时间戳。

1. 该数据集是 **gated**（需登录并同意条款）：先在
   [huggingface.co/datasets/OpenGVLab/InternVid-10M-FLT-INFO](https://huggingface.co/datasets/OpenGVLab/InternVid-10M-FLT-INFO)
   页面点同意，再登录后下载：

   ```bash
   huggingface-cli login          # 贴上你的 HF token
   huggingface-cli download OpenGVLab/InternVid-10M-FLT-INFO \
     --repo-type dataset --local-dir data/InternVid_info
   ```

2. 把 info jsonl 传给脚本，让它按名字对齐（脚本默认按
   `<YoutubeID>_<起秒>_<止秒>.mp4` 组键，见 `load_info_lookup`）：

   ```bash
   python tools/download_internvid.py download \
     --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
     --out  data/video_sample/InternVId-FLT_1 \
     --info-jsonl data/InternVid_info/InternVid-10M-FLT-INFO.jsonl \
     --limit 200 --workers 4
   ```

3. 若组键规则和你的标注仍对不上，改一处即可：
   - `tools/download_internvid.py` 里的 `parse_video_name()`（正则）——文件名自带 id/时间戳时；
   - 或 `load_info_lookup()` 里 `key = f"{yid}_{start}_{end}.mp4"` 的拼法——用 info 对齐时。

> 官方项目与脚本参考：
> [OpenGVLab/InternVideo · Data/InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)、
> 社区批量下载器 [mindsailor/YouTube-Clip-Downloader-W-InternVid](https://github.com/mindsailor/YouTube-Clip-Downloader-W-InternVid)。

---

## 附录 B：拿不到作者 JSON 时，用官方元数据自建

作者的 11 份 `InternVid-10M-flt_<N>.json` 在他们的 Google Drive 上。如果那个文件夹**私有 /
需申请访问**，预训练可以**完全绕开它**：InternVid 的**元数据是公开可直接下载的**，
自己转成本仓库要的标注格式即可。（⚠️ **视频本身仍需 YouTube 抓取**，元数据只给 id + 时间戳 +
字幕；下游**评测**标注没有这条路。）

### B.1 直接下载官方元数据（二选一）

**OpenDataLab（推荐，通常不需要 gated 审批）：**

```bash
pip install opendatalab
odl login                       # 注册后登录
odl get vd-foundation/InternVid-10M-FLT -d data/InternVid_info
# 详见 https://opendatalab.com/vd-foundation/InternVid-10M-FLT/download
```

**HuggingFace（需在网页点同意 + 登录）：**

```bash
huggingface-cli login
huggingface-cli download OpenGVLab/InternVid-10M-FLT-INFO \
  --repo-type dataset --local-dir data/InternVid_info
```

拿到的是一个 jsonl，列名 `YoutubeID / Start_timestamp / End_timestamp / Caption / ...`。

### B.2 转成本仓库的标注格式

本仓库训练读的是一个 JSON 列表，每条 `{"video": "<name>.mp4", "caption": "..."}`
（`data_root` = `data/video_sample/InternVId-FLT_<N>`）。下面脚本把官方 jsonl 转成第 1 个分片
的标注，命名规则与 `tools/download_internvid.py` 默认解析（`<id>_<起秒>_<止秒>.mp4`）**一致**，
直接写到 config 已指向的路径，无需改 `configs/data.py`：

```bash
mkdir -p anno_downstream/InternVid
python - <<'PY'
import json, os
SRC = "data/InternVid_info/InternVid-10M-FLT-INFO.jsonl"   # 改成你下载到的实际路径
DST = "anno_downstream/InternVid/InternVid-10M-flt_1.json"
LIMIT = 5000   # 先做子集；要全量把这行设为 None

def sec(ts):
    ts = str(ts)
    if ":" in ts:
        h, m, s = ts.split(":"); return int(h)*3600 + int(m)*60 + float(s)
    return float(ts)

out = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if LIMIT and len(out) >= LIMIT: break
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        yid, st, en = r["YoutubeID"], sec(r["Start_timestamp"]), sec(r["End_timestamp"])
        cap = r.get("Caption") or r.get("caption") or ""
        out.append({"video": f"{yid}_{st}_{en}.mp4", "caption": cap})
os.makedirs(os.path.dirname(DST), exist_ok=True)
json.dump(out, open(DST, "w"), ensure_ascii=False)
print(f"wrote {len(out)} entries -> {DST}")
PY
```

### B.3 抓视频 + 启动训练

标注就绪后，和正文完全一样：

```bash
# 抓第 1 分片对应的视频（文件名已与标注一致，脚本 inspect 会 matched N/N）
python tools/download_internvid.py download \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
  --out  data/video_sample/InternVId-FLT_1 --limit 200 --workers 4

# config 用单分片即可： train_corpus = "internvid_v1"
bash scripts/pretraining/clip/mobileviclip_tiny/run.sh
```

> 要多分片就把上面 `_1` 换成 `_2 … _11`，转换脚本按行区间切分 jsonl（如每片取不同的
> `LIMIT` 段），分别写到对应 `InternVid-10M-flt_<N>.json` 即可。

---

## 一页速查

```bash
# 0) 工具
pip install -U yt-dlp && sudo apt-get install -y ffmpeg

# 1) 先看标注命名
python tools/download_internvid.py inspect \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json

# 2) 冒烟（200 条）
python tools/download_internvid.py download \
  --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
  --out  data/video_sample/InternVId-FLT_1 --limit 200 --workers 4

# 3) 只跑通训练：改 config 用单分片
#    train_corpus = "internvid_v1"
#    bash scripts/pretraining/clip/mobileviclip_tiny/run.sh
```
