# Troubleshooting

## `sqlite3.OperationalError: attempt to write a readonly database`（Gradio 审阅工具）

### Symptom

以非 `vca1` 账号（如 `rby`）启动 `/home/vca1/datasets/internvid_zh_10k/gradio_video_text_review.py`：

```
File ".../gradio_video_text_review.py", line 95, in _initialize
    conn.execute("PRAGMA journal_mode=WAL")
sqlite3.OperationalError: attempt to write a readonly database
```

### Cause

**与 gradio 无关，是文件权限。** 该工具把标注库和导出文件写在**它自己所在的目录**下：

```python
db_path     = result_dir / "review_annotations.sqlite3"
export_path = result_dir / "review_results.jsonl"
```

`result_dir` 位于 `/home/vca1/...`，其他账号只有读权限；SQLite 打开 WAL 需要在库文件
**所在目录**里创建 `-wal` / `-shm`，因此即便库文件本身可读也会失败。
（这也解释了为什么报错发生在 `PRAGMA journal_mode=WAL` 这一行而不是 `connect()`。）

### Fix

按代价从低到高，三选一：

**① 用离线评审包（推荐，且本来就是为抽检设计的）** —— 不需要 gradio、不写 `vca1` 目录、
不需要服务器 GUI，而且页面**自带抽检表的填写字段**（Gradio 那个是浏览数据集用的，没有）：

```bash
bash tools/data_audit/run_v100.sh review-bundle
```

详见 [`tools/data_audit/SETUP_V100.md`](tools/data_audit/SETUP_V100.md) §7.0b。

**② 把结果目录指到自己家目录**（若脚本支持该参数）：

```bash
python gradio_video_text_review.py --help     # 先看有没有 --result-dir / --out-dir 之类
python gradio_video_text_review.py --result-dir ~/review_out
```

**③ 脚本拷到自己目录再跑**（`result_dir` 多半是相对脚本/工作目录推导的）：

```bash
mkdir -p ~/review_gui && cp /home/vca1/datasets/internvid_zh_10k/gradio_video_text_review.py ~/review_gui/
cd ~/review_gui
python gradio_video_text_review.py \
  --data /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
  --video-root /home/vca1/datasets/internvid_zh_10k/videos
```

### Notes

- **别为这个去 `chmod` 别人的目录**，也别用 `sudo` 跑——标注库写进他人家目录后续更难维护。
- 装 gradio 时**务必用独立 venv**（`~/gradio_env`），不要装进 `audit_env`：
  它会拉进 fastapi/pandas/pydantic 一堆依赖，有污染这套已验证冻结环境的风险。

---

## `libtorchaudio.so` / `libtorch_cuda_cpp.so: cannot open shared object file`

### Symptom

Running an evaluation (e.g. `scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_msrvtt.sh`)
fails at import time with:

```
OSError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory
...
OSError: Could not load this library: .../torchaudio/lib/libtorchaudio.so
```

### Cause

`torchaudio` was built against a **different torch version** than the one installed.
The env is pinned to `torch 1.13.1+cu117`, whose runtime uses `libtorch_cuda.so`; the
split-out `libtorch_cuda_cpp.so` only existed in torch ≤ 1.12. When a later
`pip install` pulls in a mismatched `torchaudio` build, its shared libs can no longer
load, and the failure happens at *import* time — taking down even video/image-only
tasks (like MSRVTT retrieval) that never touch audio.

### Fix — reinstall torch / torchvision / torchaudio so they match

```bash
conda activate mobileviclip
pip install --no-cache-dir --force-reinstall --no-deps \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

Verify all three load and report matching versions:

```bash
python -c "import torch, torchaudio, torchvision; print(torch.__version__, torchaudio.__version__, torchvision.__version__)"
# expected: 1.13.1+cu117 0.13.1+cu117 0.14.1+cu117
```

### Notes

- `scripts/setup_env.sh` pins `torchaudio==0.13.1+cu117` (matching
  `requirements.lock.txt`) to keep the torch/torchaudio builds in sync and avoid
  this drift on fresh setups.
- `dataset/utils.py` imports `torchaudio` lazily (inside `load_audio_from_path`),
  so a broken/missing audio stack only affects audio tasks — video/image evals
  import cleanly regardless.

---

## `ImportError: cannot import name 'Cache' from 'transformers'`

### Symptom

An evaluation fails while importing the model backbone:

```
File ".../models/backbones/internvideo2/internvideo2_clip_text.py", line 5, in <module>
    from peft import get_peft_model, LoraConfig, TaskType
...
File ".../peft/peft_model.py", line 38, in <module>
    from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel
ImportError: cannot import name 'Cache' from 'transformers'
```

### Cause

`peft >= 0.8` imports the `Cache` class from `transformers`, which only exists in
`transformers >= 4.36`. This repo pins `transformers==4.28.1` (alongside
`tokenizers==0.13.3` and the InternVideo2 text code), so a newer peft (the lock file
previously pinned `peft==0.12.0`) can never import.

The repo only uses `get_peft_model`, `LoraConfig`, and `TaskType` — all stable since
peft 0.2 — so the fix is to downgrade peft rather than bump transformers.

### Fix — downgrade peft to a version compatible with transformers 4.28.1

Install with `--no-deps` (see the warning below):

```bash
conda activate mobileviclip
pip install --no-cache-dir --no-deps "peft==0.4.0"
```

Verify it imports:

```bash
python -c "from peft import get_peft_model, LoraConfig, TaskType; print('peft OK')"
```

`scripts/setup_env.sh` and `requirements.lock.txt` now pin `peft==0.4.0` so fresh
setups avoid this mismatch.

> **Warning — always use `--no-deps` for peft (and open_clip).** Installing
> `peft` without `--no-deps` lets pip's resolver UPGRADE torch (e.g. to a 2.x
> wheel) to "satisfy" peft's unpinned `torch` requirement. That uninstalls the
> `torch 1.13.1+cu117` build and breaks `torchvision` / `torchaudio` (which are
> built for 1.13.1) — see the next section. peft's real dependencies
> (`transformers`, `accelerate`, `safetensors`, `huggingface-hub`, etc.) are
> already present in this env, so `--no-deps` is safe.

---

## `ImportError: cannot import name '_cast_Long' from 'torch.onnx.symbolic_opset9'`

### Symptom

```
UserWarning: Failed to load image Python extension: Could not load this library:
  .../torchvision/image.so
...
ImportError: cannot import name '_cast_Long' from 'torch.onnx.symbolic_opset9'
```

Often preceded by a pip warning like:

```
torchvision 0.14.1+cu117 requires torch==1.13.1, but you have torch 2.x which is incompatible.
torchaudio 0.13.1+cu117 requires torch==1.13.1, but you have torch 2.x which is incompatible.
```

### Cause

`torch` was upgraded to a 2.x wheel (usually by a `pip install <pkg>` that ran
**without** `--no-deps` — commonly `peft` or `open_clip`), which uninstalled the
`torch 1.13.1+cu117` build. `torchvision 0.14.1` / `torchaudio 0.13.1` are compiled
against torch 1.13.1 and can no longer load their C++ extensions against torch 2.x.

### Fix — restore the matching torch stack, then reinstall the offending pkg with `--no-deps`

```bash
conda activate mobileviclip

# Restore the cu117 trio
pip install --no-cache-dir --force-reinstall --no-deps \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Reinstall whatever pulled torch up, this time isolated from its deps
pip install --no-cache-dir --no-deps "peft==0.4.0" -i https://pypi.org/simple

# Verify all three agree and load
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)"
# expected: 1.13.1+cu117 0.14.1+cu117 0.13.1+cu117
```

---

## Missing `anno_downstream/*.json` (e.g. `msrvtt_1k_test.json`)

### Symptom

An eval fails because its annotation file does not exist, e.g.:

```
FileNotFoundError: anno_downstream/msrvtt_1k_test.json
```

(or the config points at `anno_downstream/<dataset>.json` and that path is absent).

### Cause

The downstream annotation JSONs are **data**, not part of the repo. `README_zh.md`
§5.2 fetches them from the author's Google Drive folder:

```bash
gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream
```

That folder requires access approval, so it may not be reachable.

### Fix A — build the MSR-VTT annotation from public sources (no Drive access)

`msrvtt_1k_test.json` is the standard MSR-VTT 1K-A / JSFusion test split, identical
across InternVideo2 / UMT / VINDLU / CLIP4Clip. Reconstruct it with the helper
script from the public Frozen-in-Time zip (the same zip used for the videos):

```bash
cd ~/mobileclip
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip -q MSRVTT.zip                                  # -> ./MSRVTT/
python preprocess/build_msrvtt_1k_test.py --src ./MSRVTT
ls -la anno_downstream/msrvtt_1k_test.json           # 1000 entries
```

If the script prints `MSR_VTT.json -> None`, your `MSRVTT/` has no annotation
folder (videos-only package). Locate the source files with:

```bash
find ./MSRVTT \( -name 'MSR_VTT.json' -o -iname '*jsfusion*' \) 2>/dev/null
```

and pass the directory that contains `annotation/` as `--src`, or use Fix B.

### Fix B — convert CLIP4Clip's `MSRVTT_JSFUSION_test.csv`

The widely-mirrored CLIP4Clip CSV (columns `key,vid_key,video_id,sentence`)
directly holds the 1000 video→caption pairs. The same script converts it:

```bash
python preprocess/build_msrvtt_1k_test.py --src /path/to/MSRVTT_JSFUSION_test.csv
```

### Fix C — request the Drive folder / ask upstream

For the other downstream sets (`didemo_ret_test`, `anet_ret_val`,
`kinetics400_validate`, `ucf101_act_test`, `hmdb51_act_test`, `ssv2_mc_val`, …),
request access to the Drive folder above, or open an issue asking the author to
mirror `anno_downstream/` on a public host (e.g. a Hugging Face dataset).

> **Expected schema.** The retrieval loader expects a JSON list of objects with a
> `video` (or `image`) filename and a `caption` string, e.g.
> `[{"video": "video7010.mp4", "caption": "a man is spreading butter on bread"}, ...]`.
> Paths are resolved relative to each dataset's `data_root` (for MSR-VTT,
> `./data/MSRVTT_Videos`).

> **Exact-reproducibility note.** The 1K-A protocol pins one specific caption per
> video via a `*jsfusion*caption_idx*.pkl`. `build_msrvtt_1k_test.py` uses it
> automatically when present; if it is absent the script falls back to the first
> caption — the eval still runs, but R@k may differ slightly from the paper.

---

## HMDB51 download: `FileMetadataError: Distant resource does not seem to be on huggingface.co`

### Symptom

Downloading the HMDB51 videos from the HF mirror dies immediately — **zero bytes
fetched** — with:

```
huggingface_hub.errors.FileMetadataError: Distant resource does not seem to be on
huggingface.co. It is possible that a configuration issue prevents you from
downloading resources from https://huggingface.co. ...
...
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to
locate the file on the Hub and we cannot find the requested files in the local cache.
```

after running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download jili5044/hmdb51 --repo-type dataset --local-dir data/hmdb51_hf
```

### Cause

The dataset `jili5044/hmdb51` is a **single Xet-backed LFS file** — `hmdb51.zip`
(~2.12 GB) plus a `README.md`. For Xet-stored files, `hf-mirror.com`'s HEAD
response does not carry the `x-linked-etag` header. `huggingface_hub` treats a
missing etag as "the file isn't really on the Hub" and raises `FileMetadataError`
**before downloading anything** — hence `LocalEntryNotFoundError` too.

The bytes themselves are fine: `…/resolve/main/hmdb51.zip` just 302-redirects to a
CDN that serves plain HTTP range requests (`accept-ranges: bytes`). The failure is
purely the client's pre-download metadata check against the mirror.

> Note: the official `huggingface.co` endpoint returns the `x-linked-etag`
> correctly, so this error is specific to the mirror. It is **not** a firewall or
> SSL problem despite what the message suggests.

### Fix — bypass the metadata check (any one of these)

**A. One-shot script (recommended).** Downloads, verifies the byte count, unzips,
flattens every `.avi` into `data/hmdb51/videos_test/`, and checks the annotation:

```bash
bash tools/download_hmdb51.sh
# official endpoint instead of the mirror:
#   HF_ENDPOINT=https://huggingface.co bash tools/download_hmdb51.sh
```

**B. Direct `wget`** of the one zip (resumable, skips the Python client entirely):

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p data/hmdb51_hf
wget -c "$HF_ENDPOINT/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip" \
     -O data/hmdb51_hf/hmdb51.zip
unzip -n -q data/hmdb51_hf/hmdb51.zip -d data/hmdb51_hf
mkdir -p data/hmdb51/videos_test
find data/hmdb51_hf -name '*.avi' -exec ln -sf {} data/hmdb51/videos_test/ \;
```

**C. Disable Xet** and keep using the CLI (recent `huggingface_hub`):

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
huggingface-cli download jili5044/hmdb51 --repo-type dataset --local-dir data/hmdb51_hf
```

Then verify `data/hmdb51/videos_test/` holds ~6766 `.avi` and that
`anno_downstream/hmdb51_act_test.json` reports 0 missing (see README_zh §6.3).

---

## `modelscope download` keeps failing: `IncompleteRead(... bytes read, ... more expected)` / `Max retries exceeded`

Seen while pre-downloading Qwen3-VL weights for the data audit
(`tools/data_audit/SETUP_V100.md` §0 / `SETUP_H100.md` §0):

```
WARNING | modelscope_hub.download | Download failed for model-00002-of-00004.safetensors:
('Connection broken: IncompleteRead(59314357 bytes read, 4510618059 more expected)', ...), will retry
ERROR   | modelscope_hub.download | Failed to download model-00004-of-00004.safetensors:
None: Max retries exceeded with url: ...
```

**What it means.** The TCP connection to the ModelScope CDN is being cut
mid-transfer. On a slow/flaky link (tens of kB/s) each ~5 GB safetensors shard
needs a connection that survives for hours, and the default snapshot download
pulls several shards concurrently, which makes every connection slower and
longer-lived — i.e. more likely to die. After the built-in retries are
exhausted the file errors out and the snapshot ends incomplete.

**Nothing is lost.** `modelscope` resumes partial files with HTTP `Range`
requests: re-running the *same* command skips completed files and continues
half-downloaded ones from where they stopped. So the fix is simply to retry in
a loop, with concurrency turned down to one file at a time:

```bash
until modelscope download --model Qwen/Qwen3-VL-8B-Instruct \
    --local_dir ckpts/Qwen3-VL-8B-Instruct --max-workers 1; do
  echo "download interrupted, resuming in 10s"; sleep 10
done
```

Wrap it in `nohup ... &` to keep it running unattended (full recipe in
`SETUP_V100.md` §0).

**If throughput stays pathological** (tens of kB/s for hours), the route to
ModelScope's CDN is the bottleneck — try the hf-mirror channel instead, which
uses a different CDN and also resumes; whichever is faster wins, and both write
into the same `--local-dir`:

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
until hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir ckpts/Qwen3-VL-8B-Instruct; do
  sleep 10
done
```

(`huggingface_hub` ≥1.0 renamed the CLI: `huggingface-cli` prints a
deprecation notice and exits — use `hf` with the same arguments. On older
hub versions `huggingface-cli download` still works.)

`HF_HUB_DISABLE_XET=1` is required on the mirror channel: hf-mirror's
metadata responses for Xet-stored files are incomplete, producing
`Distant resource does not seem to be on huggingface.co` — the same
failure mode as the HMDB51 entry above. A related benign variant:
`Couldn't access the Hub to check for update but local file already
exists. Defaulting to existing file.` just means an already-complete
local file couldn't be re-validated against the Hub and is used as-is —
harmless for finished files. Note the model is public: no HF token is
needed, so an auth error message never means you should paste a token
somewhere.

Sanity check when a channel reports success — all four shards present and full
size (the 8B model totals ~17 GB; a partially written shard fails loudly at
load time):

```bash
ls -l ckpts/Qwen3-VL-8B-Instruct/model-0000*-of-00004.safetensors
```

---

## hf-mirror + new `huggingface_hub`: `Error: Local entry not found` on every retry

Symptom — an `until hf download ...` loop that never downloads anything, each
iteration failing in seconds:

```
FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated
  as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead
Fetching 26 files:   4%|▍  | 1/26
Couldn't access the Hub to check for update but local file already exists. Defaulting to existing file.
Error: Local entry not found. Distant resource does not seem to be on huggingface.co.
```

**Cause.** Recent `huggingface_hub` (the version in `audit_env` on 101vector)
**dropped `hf_transfer` in favour of Xet** as its transfer backend — the
FutureWarning says so outright. hf-mirror cannot serve complete metadata for
Xet-backed files, so any file **not already present locally** fails hard with
`Local entry not found`. `HF_HUB_DISABLE_XET=1` was the cure on older hub
versions and no longer rescues this one.

Read the two messages correctly, they mean opposite things:

- `Couldn't access the Hub ... Defaulting to existing file` — **benign**: a small
  file is already local and revalidation was skipped.
- `Error: Local entry not found` — **fatal**: a file is missing locally *and*
  unfetchable from the mirror. This is the one that ends the run.

**Fix — switch service, not settings.** Use ModelScope: a different service
with its own client, no Xet involved, and the best domestic route for Qwen
models anyway.

```bash
pkill -f "hf download"; sleep 2; pgrep -af "hf download"    # stop the spinning loop
unset HF_HUB_ENABLE_HF_TRANSFER HF_HUB_DISABLE_XET          # both now useless here
nohup bash -c 'until modelscope download --model Qwen/Qwen3-VL-32B-Instruct \
  --local_dir ckpts/Qwen3-VL-32B-Instruct --max-workers 4; do sleep 10; done' \
  > download_32b_ms.log 2>&1 &
```

**Do not downgrade `huggingface_hub` to get the old behaviour back.**
`transformers` depends on it, and during an audit `audit_env` is a frozen
environment — changing library versions mid-run can change model outputs and
break comparability between shards already computed and shards computed later
(`tools/data_audit/SETUP_V100.md` §4.3-本地 步骤 0). ModelScope sidesteps the
whole issue at zero cost.

If ModelScope is also slow, `hfd.sh --tool aria2c` is still worth a try: it hits
hf-mirror's plain `resolve` URLs with curl/aria2c and never touches the Python
client's Xet path, so it can succeed where the CLI fails.

---

## Weight download is *slow but stable* (constant few hundred kB/s, no errors)

Different failure from the entry above, and the fix is the **opposite**. That
one is a *flaky* link — connections die, `IncompleteRead` is logged, and the
cure is fewer concurrent workers plus a retry loop. This one never errors: the
transfer just crawls. Turning concurrency **down** here makes it worse.

Seen while pre-downloading Qwen3-VL-32B (~64 GB) for the local judge
(`tools/data_audit/SETUP_V100.md` §4.3-本地).

**Measure before choosing** — at 64 GB the difference between channels is hours:

```bash
D=ckpts/Qwen3-VL-32B-Instruct
a=$(du -sm $D 2>/dev/null | cut -f1 || echo 0); sleep 60; b=$(du -sm $D | cut -f1)
echo "$(( b - a )) MB/s"
```

Under ~2 MB/s a 64 GB download needs 9+ hours — switch channels rather than
wait. The ladder, cheapest first:

1. **aria2c multi-connection** (`hfd.sh --tool aria2c -x 16`). Biggest win when
   the CDN throttles *per connection* rather than per client — commonly 3–10×.
2. **`hf_transfer`** (`HF_HUB_ENABLE_HF_TRANSFER=1`), a Rust multi-threaded
   backend for the `hf` CLI. Easiest to try; costs you progress detail and
   legible error messages, so unset it when debugging.
3. **More ModelScope workers** (`--max-workers 4`). Note this contradicts the
   flaky-link recipe above on purpose: *stable but slow* wants more parallelism,
   *unstable* wants less.
4. **Copy over the LAN from a machine that already has the weights.** Gigabit
   intranet ≈ 110 MB/s puts 64 GB at ~10 minutes, beating any public channel:
   `rsync -avP --partial user@host:.../Qwen3-VL-32B-Instruct/ ckpts/Qwen3-VL-32B-Instruct/`
   (search the other machine first with the recipe in
   `tools/data_audit/SETUP_JUDGE_API.md` §1.2).
5. **Downsize the model.** If every channel still says 5+ hours, a 17 GB
   alternative that runs tonight beats a 64 GB one that lands tomorrow.

**Don't ping-pong between tools.** Every downloader skips files that are
already complete, but a *half-written* file generally restarts from zero under
a different tool, since temp-file naming and resume conventions differ. Switch
channels between files, and once switched, stay switched.

Whichever channel finishes, verify shard count and total bytes against
`model.safetensors.index.json` before loading — a truncated shard only
announces itself minutes into model loading.

---

## Running the next retrieval dataset (DiDeMo, ActivityNet)

Once MSR-VTT works end-to-end, the other zero-shot **retrieval** benchmarks run
through the exact same three-step pattern — only the dataset names change:

1. an **annotation JSON** under `anno_downstream/`,
2. the **videos** on disk at the `data_root` the config expects, and
3. run the matching `eval_*.sh`.

### The retrieval configs at a glance

| Dataset | Script | `anno_path` | `data_root` (videos) | `caption` field |
|---|---|---|---|---|
| MSR-VTT 1K-A | `eval_msrvtt.sh` | `anno_downstream/msrvtt_1k_test.json` | `./data/MSRVTT_Videos` | single string |
| DiDeMo | `eval_didemo.sh` | `anno_downstream/didemo_ret_test.json` | `./data/didemo/processed_videos/test` | list of sentences |
| ActivityNet | `eval_anet.sh` | `anno_downstream/anet_ret_val.json` | `./data/anet_1.3_video_val_resize` | list of sentences |

> **Paragraph retrieval.** DiDeMo and ActivityNet are *paragraph* retrieval
> (`is_paragraph_retrieval=True`): each video is matched against **all its
> sentences joined into one paragraph**. Their annotation `caption` is therefore a
> **list of strings**, not a single string:
> ```json
> [{"video": "xxxxx.mp4", "caption": ["a woman walks in.", "she sits down.", "..."]}, ...]
> ```
> The loader joins the list with spaces (`" ".join(caption)`, see
> `preprocess_para_retrieval_data` in `dataset/ret_dataset.py`). MSR-VTT's
> `caption` is a plain string. DiDeMo also sets `trimmed30=True` — only the first
> 30 s of each video is read.

### Step 1 — annotation JSON

Like `msrvtt_1k_test.json`, `didemo_ret_test.json` and `anet_ret_val.json` come
from the author's Google Drive `anno_downstream/` folder (access-gated). With
Drive access:

```bash
gdown --folder https://drive.google.com/drive/folders/1grbR39KnMLpeHPNpX4MBYArhr6wASJza -O anno_downstream
```

Without it: the DiDeMo and ActivityNet paragraph-retrieval splits are the
standard ones shared across CLIP4Clip / Frozen-in-Time / collaborative-experts,
so rebuild them from those public releases into the schema above (a list of
`{"video", "caption": [...]}`). These annotation files are small text — the
videos are the large download.

### Step 2 — videos + path (same symlink trick as MSR-VTT)

The config points at a specific `data_root`; your videos just need to be
reachable there. After downloading each dataset's videos, symlink your real
folder to the path the config expects:

```bash
cd ~/mobileclip

# DiDeMo  -> config wants ./data/didemo/processed_videos/test
mkdir -p data/didemo/processed_videos
ln -s "$(pwd)/<your_didemo_test_videos>" "$(pwd)/data/didemo/processed_videos/test"

# ActivityNet  -> config wants ./data/anet_1.3_video_val_resize
ln -s "$(pwd)/<your_anet_val_videos>" "$(pwd)/data/anet_1.3_video_val_resize"
```

Confirm the `video` filenames in the annotation resolve to real files before
launching (catches path/extension mismatches early):

```bash
python - <<'PY'
import json, os
checks = [("anno_downstream/didemo_ret_test.json", "./data/didemo/processed_videos/test"),
          ("anno_downstream/anet_ret_val.json",    "./data/anet_1.3_video_val_resize")]
for anno, root in checks:
    if not os.path.exists(anno):
        print(f"[skip] {anno} not built yet"); continue
    v = json.load(open(anno))[0]["video"]
    p = os.path.join(root, v)
    print(f"{anno}: first video -> {p}  exists={os.path.exists(p)}")
PY
```

### Step 3 — run

```bash
conda activate mobileviclip
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_didemo.sh
bash scripts/evaluation/clip/zero_shot/mobileviclip_tiny/eval_anet.sh
```

> **GPU index gotcha.** These scripts hard-code a GPU that may not exist on your
> box: `eval_didemo.sh` sets `CUDA_VISIBLE_DEVICES=1` and `eval_anet.sh` sets
> `CUDA_VISIBLE_DEVICES=7` (`eval_msrvtt.sh` ships with `=3`). On a single-GPU
> machine, change it to `0` (as you did for MSR-VTT) or the run fails with an
> invalid-device error.

### Expected numbers (MobileViCLIP-Tiny, from each script's header comment)

| Dataset | txt R@1 | txt R@5 | txt R@10 | vid R@1 | vid R@5 | vid R@10 |
|---|---|---|---|---|---|---|
| MSR-VTT | 39.0 | 65.5 | 75.1 | 38.3 | 66.4 | 74.9 |
| DiDeMo | 37.11 | 64.91 | 73.32 | 36.66 | 64.69 | 74.89 |
| ActivityNet | 28.21 | 54.75 | 68.29 | 29.29 | 58.41 | 71.77 |

Landing within ~1 point of these confirms the model, video decoding, and the
annotation split are all wired up correctly for that dataset.

## `import deepspeed` crashes: pydantic `root_validator` error, or `cannot import name 'log' from torch.distributed.elastic`

### Symptom

Any `tasks_clip/*` entry point dies at startup, once per rank, with a traceback
ending inside `import deepspeed` — one of two variants:

```
pydantic.errors.PydanticUserError: If you use `@root_validator` with pre=False
(the default) you MUST specify `skip_on_failure=True`.
```

```
ImportError: cannot import name 'log' from
'torch.distributed.elastic.agent.server.api'
```

Both hit on 101vector (2026-07-21) in a conda env with torch 2.x.

### Cause

An **old DeepSpeed build is incompatible with the modern env on two separate
axes**:

1. old DeepSpeed uses pydantic-v1 style validators → crashes under
   **pydantic ≥ 2** (first variant);
2. old DeepSpeed's `elasticity/elastic_agent.py` imports `log`, which was
   removed from torch's elastic API in **torch ≥ 2.0** (second variant).

`pip install "pydantic<2"` only fixes axis 1 — in a torch 2.x env the import
then fails on axis 2. The old DeepSpeed simply cannot run with torch 2.x.

### Fix

```bash
# 0) FIRST: git pull. Since PR #84 the repo guards the import
#    (tasks_clip/shared_utils.py) so a broken deepspeed no longer kills
#    runs that don't use it. A traceback pointing at a bare
#    `import deepspeed` on shared_utils.py line 8 means you are on old code.

# 1) Real fix in a torch 2.x env: upgrade deepspeed (needs pydantic >= 2 —
#    if you downgraded pydantic for axis 1, upgrade it back):
pip install -U pydantic
DS_BUILD_OPS=0 pip install -U "deepspeed>=0.14"
python -c "import deepspeed; print(deepspeed.__version__)"   # must not raise

# 2) Or bypass for pilots/smoke runs (DDP+AMP path, no deepspeed at all):
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 deepspeed.enable False
```

Note: the pinned-original env (`mobileviclip.yml`, torch 1.13.1) does not hit
either axis — this only bites self-built torch 2.x envs. Official R0–R3
replication runs must use the DeepSpeed path (original recipe is DeepSpeed
stage-1 + fp16); the bypass is for pilots and plumbing checks only.

## `CUDA error: no kernel image is available for execution on the device` on V100

### Symptom

Every rank dies at the **first CUDA call** — `torch.distributed.barrier()` inside
`utils/distributed.py:119`, long before any model or data is touched:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelImageForDevice' ...
```

Preceded, once per GPU, by the real clue:

```
UserWarning: Found GPU0 Tesla V100-SXM2-32GB which is of compute capability (CC) 7.0.
The following list shows the CCs this version of PyTorch was built for ...
- 7.5 ... - 8.0 ... - 8.6 ... - 9.0 ... - 10.0 ... - 12.0
```

Hit on 101vector (2026-07-21) in the `mobileviclip` conda env, running
`experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8`.

### Cause

**The installed PyTorch has no Volta (`sm_70`) kernels.** The env had grown to
`torch 2.11.0` built against **CUDA 13** (`cuda-toolkit 13.0.2`,
`nvidia-cudnn-cu13`, `nvidia-nccl-cu13` in the dependency list). CUDA 13 dropped
offline compilation for Maxwell/Pascal/**Volta**, so those wheels ship
`sm_75/80/86/90/100/120` only — note **7.0 is absent from the list above**.
A V100 is CC 7.0, so *every* GPU kernel fails, NCCL's barrier being merely the
first one attempted.

This is **not** a DeepSpeed problem, and `deepspeed.enable False` cannot help:
the crash is upstream of any framework choice, in `init_distributed_mode`.
This is exactly the hazard `tools/data_audit/SETUP_V100.md` §0 warns about for
the audit env — the *training* env just never got the same treatment.

### Fix

Reinstall PyTorch from the **cu118** index (the cu118/cu121 builds still carry
`sm_70`), matching the versions already validated on this machine for the audit
env. `--index-url` replaces the aliyun mirror for this one command:

```bash
conda activate mobileviclip

# 1) drop the CUDA-13 build (leaves the stray nvidia-*-cu13 wheels harmless)
pip uninstall -y torch torchvision torchaudio

# 2) reinstall a Volta-capable build (torch >= 2.5 keeps deepspeed 0.19 happy)
pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu118

# 3) decord 0.6 wheels break under numpy 2 (_ARRAY_API not found) — the dataset
#    loader is decord-based, so pin numpy back down
pip install "numpy<2"

# 4) verify: arch list MUST contain sm_70, and the .cuda() must not raise
python -c "import torch, decord; print(torch.__version__, torch.cuda.get_arch_list()); \
  print(torch.zeros(1).cuda(), torch.cuda.get_device_name(0))"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
```

Only after `sm_70` appears in `get_arch_list()` is it worth relaunching:

```bash
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

> **Never `pip install -U torch/torchvision/torchaudio` (or bare
> `pip install torchvision`) in this env again** — the default PyPI wheels are
> CUDA 12.8/13 builds and will silently un-Volta the machine. Any torch upgrade
> on 101vector must go through the cu118 index. The same rule, and why, is in
> `tools/data_audit/SETUP_V100.md` §0.

The other Volta consequence is already handled in the configs: no bf16 tensor
cores, so `use_bf16=False` / `bf16=False` / fp16 AMP
(`scripts/pretraining/clip/mobileviclip_small/config.py`). `flash_attn` is also
unavailable on `sm_70`; the `FusedMLP of flash_attn is not installed!!!` lines
are harmless fallback notices, not errors (see README_zh Q9).

## `FileNotFoundError: anno_downstream/internvid_zh_10k_pilot.json` (10k pilot)

### Symptom

The V100 torch fix above worked — the run gets past CUDA init, model build and
`init_distributed_mode`, then dies in dataset construction:

```
File "/home/vca1/rby/mobileclip/dataset/pt_dataset.py", line 88, in __init__
    with open(self.label_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'anno_downstream/internvid_zh_10k_pilot.json'
```

### Cause

Nothing is mis-pathed — **the annotation file simply has not been generated yet.**
It is not shipped with the repo and is not part of the audit sample; it is built
from the 10k audit jsonl by `tools/make_pilot_anno.py`, which is
`experiments/README.md` §2b **step 1**. Jumping straight to the training command
skips it.

The paths involved all agree already (verified against `tools/data_audit/`):

| Thing | Value on 101vector | Where it comes from |
|---|---|---|
| audit jsonl | `/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl` | `tools/data_audit/run_v100.sh` `$DATA`, `SETUP_V100.md` §0 |
| clips root | `/home/vca1/datasets/internvid_zh_10k/videos` | `run_v100.sh` `$CLIPS` |
| `data_root` | `$MVC_DATA_DIR/internvid_zh_10k/videos` | `configs/data.py` `internvid_zh_10k_pilot` |
| `anno_path` | `$MVC_ANNO_DIR/internvid_zh_10k_pilot.json` | same entry; `MVC_ANNO_DIR` defaults to `anno_downstream` |

So with `MVC_DATA_DIR=/home/vca1/datasets` the `data_root` resolves to the same
`videos/` directory the audit tooling uses, and the `video` field written by
`make_pilot_anno.py` is relative to that root — consistent by construction,
because the script validates every record against `--clips-root` and drops
records whose file is missing.

### Fix

Run §2b step 1 once, from the repo root:

```bash
cd ~/mobileclip
python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

ls -l anno_downstream/internvid_zh_10k_pilot*.json    # ~9k train + 1k eval
```

Read the two lines it prints before moving on:

- `记录: 可用 N  解析失败 …  文件缺失 M` — **M is the count of jsonl records whose
  video is missing on disk.** A large M means the `video` field is not relative to
  `--clips-root` (e.g. it carries a subdirectory or an absolute path); inspect the
  schema with
  `head -1 /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl | python -m json.tool`
  and adjust `--clips-root` accordingly. If *every* record is dropped the script
  exits with `没有可用记录——检查 --input 字段名或 --clips-root`.
- `共享字幕: … 组 / 覆盖 … 条 clip (…%)` — the share of clips with duplicate
  captions, i.e. the real blast radius of the `caption_idx` multi-positive fix.
  Worth recording; it bounds how much that component can possibly matter here.

Then continue with §2b steps 2–4 (`export MVC_DATA_DIR`, the sm_70 pre-flight,
smoke, and the equal-step r0/r1 A/B pair).

### Variant: `make_pilot_anno.py` exits with `没有可用记录`

If step 1 above prints only

```
没有可用记录——检查 --input 字段名或 --clips-root
```

you are on the pre-fix script: it dropped **every** record and exited *before*
printing its own counters, so the message cannot distinguish a field-name miss
from a file-location miss. `git pull` first — the script now prints a per-reason
breakdown (parse / missing `video` / missing `caption` / file not found /
resolved outside `--clips-root`), the first record's actual key names, and the
exact paths it tried.

**The usual cause is clip layout.** The clips are **not** flat under `videos/`;
they sit in a nested `InternVId-*/…` tree, and the jsonl's `video` field is
often just the bare clip name. The audit tooling has always handled this in
`resolve_video_path()` (`tools/data_audit/score_alignment.py`), which tries, in
order: `clips_root/<video>` → `clips_root/<the InternVId-* suffix of the
record's abs_path>` → the raw `abs_path`. `make_pilot_anno.py` originally did a
naive `os.path.join(clips_root, video)` and so dropped all 10k records; it now
shares that same resolution order and writes the **`clips_root`-relative** path
into the anno (dataset loading joins it onto `data_root`, so an absolute path
would break).

Re-run step 1 after pulling. Read the breakdown it prints:

- **`没有 video 字段` / `没有 caption 字段` large** → the key names are not in the
  guess table. The printed `首条记录的字段` shows the real ones; pass them with
  the new `--video-key` / `--caption-key` flags.
- **`文件缺失` large** → the tried paths are printed; check `--clips-root` really
  is the directory holding the clips, and that records carry `abs_path` if the
  clips are nested.
- **`root 外` large** → files resolve, but only via an absolute path outside
  `--clips-root`, which cannot be written as a relative anno path. Raise
  `--clips-root` to a common parent and point `data_root` (`configs/data.py`) at
  the same place.

> Note on captions: `--caption-field` defaults to `en` because the text tower is
> CLIP's English BPE. This corpus is the zh-translated InternVid, so if the
> English side turns out to be absent, do **not** silently switch to `zh` — a
> Chinese caption through an English BPE tokenizer is a different experiment,
> not a drop-in substitute, and would invalidate comparison with the baseline.
> Report it instead.

---

## DeepSpeed startup dies in `FusedAdamBuilder().load()`: `parameter packs not expanded with ‘...’`

**Symptom.** The 8-GPU pilot gets all the way through dataset construction, model
build and optimizer grouping, then every rank dies inside
`deepspeed.initialize` → `_configure_basic_optimizer` → `FusedAdam(...)`:

```
RuntimeError: Error building extension 'fused_adam': ... ninja ...
/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’:
  435 |         function(_Functor&& __f)
```

Ranks that lose the race to build it report the follow-on error instead:

```
ImportError: ~/.cache/torch_extensions/py310_cu118/fused_adam/fused_adam.so:
cannot open shared object file: No such file or directory
```

**Cause.** DeepSpeed's default Adam is `FusedAdam`, a CUDA extension that is
**JIT-compiled on first use** with the *system* `nvcc` (`/usr/bin/nvcc`) — not
with the CUDA that ships inside the torch wheel. On 101vector the system nvcc
predates the system GCC, and old nvcc cannot parse GCC 11's `<functional>`
header; `parameter packs not expanded with ‘...’` in `std_function.h` is the
signature of exactly that mismatch. Note the second `.so` error is *downstream
noise*: only one rank compiles, the other seven wait for the artefact that the
failed build never produced. Fix the compile and both go away.

This is independent of the Volta/`sm_70` issue above — the nvcc command line
here already carries `-gencode=arch=compute_70,code=sm_70` correctly.

**Fix (already in the repo — just pull).** `utils/config_utils.py` now writes
`"torch_adam": true` into the generated `deepspeed_config.json`, so DeepSpeed
builds `torch.optim.AdamW` instead of `FusedAdam`. Same algorithm, no fused
kernel, and **nothing to compile**. Nothing to run — pull and re-launch.

Two details worth knowing if you touch that code:

- `torch_adam` and `adam_w_mode` must sit inside `optimizer.params`; DeepSpeed
  pops them from there and passes every *remaining* key as a kwarg to the
  optimizer constructor. (The old config put `adam_w_mode` one level up, where
  it was silently ignored.)
- Because the rest of `params` is forwarded verbatim, `bias_correction` — a
  `FusedAdam`-only argument — must be dropped in the torch path or
  `torch.optim.AdamW` raises `TypeError`. The code only adds it on the fused
  branch. torch's AdamW does bias correction unconditionally, so the behaviour
  is unchanged.

**If you ever want the fused kernel back** (marginal gain at this model size;
the optimizer is not the bottleneck here), re-enable it *and* give nvcc a host
compiler it can handle:

```bash
# option A: keep system nvcc, point it at an older host compiler
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-10'      # only if g++-10 exists

# option B: use an nvcc matching the cu118 torch build (CUDA 11.8 supports GCC 11)
conda install -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8
export CUDA_HOME="$CONDA_PREFIX"

rm -rf ~/.cache/torch_extensions/py310_cu118/fused_adam   # stale failed build
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 \
    deepspeed.torch_adam False debug True
```

Useful for the record when diagnosing: `nvcc --version`, `g++ --version`,
`echo $CUDA_HOME`.

---

## Training dies on the first forward: `Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same`

**Symptom.** DeepSpeed initialises fine, the dataloader is built
(`MetaLoader has 1 dataloaders, 17 batches in total`), and then every rank dies
in the very first `model(image, text_input, idx=idx)`:

```
File "models/backbones/internvideo2/mobileclip/modules/common/mobileone.py", line 222, in forward
    scale_out = self.rbr_scale(x)
  ...
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same
```

Just above it, one rank prints the tell-tale warning:

```
[WARNING] [torch_autocast.py:122:autocast_if_enabled] torch.autocast is enabled
outside DeepSpeed but disabled within the DeepSpeed engine.
```

**Cause.** Two halves of the mixed-precision setup stopped meeting.
`tasks_clip/shared_utils.py` casts the **whole model** to fp16/bf16
(`model.half()`), and the batch stays fp32 — the `torch.cuda.amp.autocast(...)`
around the forward in `tasks_clip/pretrain.py` was what reconciled them.

Recent DeepSpeed **explicitly disables any outer autocast inside
`engine.forward`**: `deepspeed/runtime/torch_autocast.py:autocast_if_enabled`
wraps the module call in `torch.autocast(..., enabled=False)` unless a
`torch_autocast` section is present in the DeepSpeed config. So the autocast in
our train loop is a no-op on the DeepSpeed path, leaving fp32 activations
meeting fp16 weights at the first conv. This is a DeepSpeed-version regression,
not a config error on the machine — nothing about Volta/`sm_70` is involved.

**Fix (already in the repo — just pull).** `tasks_clip/pretrain.py` now casts
the batch to the model's own parameter dtype before the forward:

```python
param_dtype = next(model_without_ddp.parameters()).dtype   # once, before the loop
...
image = image.to(device, dtype=param_dtype, non_blocking=True)
```

Deliberately **`param_dtype`, not `data_type`**: the precision switch is read in
three places with different defaults, so `data_type` is not reliably the dtype
the weights actually have —

| file | key | default | decides |
|------|-----|---------|---------|
| `tasks_clip/shared_utils.py` | `use_bf16` (legacy `bf16` as fallback) | `True` | model → bf16 or fp16 |
| `tasks_clip/pretrain.py` | `use_bf16` | `True` | `data_type` for autocast |
| `utils/config_utils.py:130` | `use_bf16` | `False` | `bf16` vs `fp16` section in `deepspeed_config.json` |

(`shared_utils.py` originally read a **different key**, `bf16` — unified to
`use_bf16` in 2026-07, ported from the b100 NebulaCV line, keeping `bf16` as a
fallback for old configs. The defaults still differ, so:) they only agree when
a config sets `use_bf16` explicitly (the pilot configs set both spellings). Asking the model for its dtype sidesteps all three; when
`use_half_precision` is false it is `float32` and the cast is a no-op.

**Why not `fp16: {auto_cast: true}` instead?** That DeepSpeed option ("*auto_cast
automatically casts inputs to fp16*") would also fix it, but only for fp16 — the
`bf16` section has no equivalent, so an H100 bf16 run would hit the same wall.
The cast in the train loop covers both and does not depend on the DeepSpeed
version.

**Eval is not affected.** Epoch-end retrieval runs
`evaluation_wrapper(model_without_ddp, ...)` — the bare module, not the DeepSpeed
engine — so its surrounding autocast is still live and casts the inputs.

**What is normal after this fix.** With the outer autocast neutralised the
forward is pure fp16, and DeepSpeed's dynamic loss scaling (`initial_scale_power:
16`) does the compensating. A handful of `OVERFLOW! ... Skipping step` lines
while the scale settles is expected. What is *not* normal: a loss that is `nan`
from step 0, or the loss scale collapsing to `min_loss_scale` and staying there.

---

## Same run, one layer deeper: `RuntimeError: expected scalar type Float but found Half` in `LayerNormFP32`

**Symptom.** With the batch-dtype fix above in place, the vision tower now runs, and
all ranks die slightly later — in the **text** tower's first LayerNorm:

```
File "models/backbones/internvideo2/mobileclip/transformer.py", line 46, in forward
    return super().forward(x.to(torch.float32)).to(inp_dtype)
  ...
RuntimeError: expected scalar type Float but found Half
```

**Cause.** Same root cause (DeepSpeed disables the outer autocast), different
victim. `LayerNormFP32` casts only its **input** to fp32; `self.weight` and
`self.bias` were made fp16 by `model.half()`, and `F.layer_norm` requires input
and parameters to share a dtype.

This used to work by accident: `layer_norm` is on autocast's **fp32 cast-policy
list**, and that policy casts *every* input — activations *and* `weight`/`bias` —
to fp32. The class was never self-consistent; autocast was silently completing it.
With autocast disabled inside the engine, the class has to do its own job.

**Fix (already in the repo — just pull).** Both copies of the class —
`mobileclip/transformer.py` and `mobileclip/modules/common/transformer.py`
(near-duplicate files; **keep them in sync**) — now cast the parameters too:

```python
out = F.layer_norm(
    x.to(torch.float32),
    self.normalized_shape,
    self.weight.to(torch.float32) if self.weight is not None else None,
    self.bias.to(torch.float32) if self.bias is not None else None,
    self.eps,
)
return out.to(inp_dtype)
```

Verified on CPU: the old form raises the same error class against a `.half()`
module, the new form is **bit-identical** to an all-fp32 LayerNorm cast back to
fp16, and the `elementwise_affine=False` branch (params are `None`) still works.

### The general shape of this bug

Anything that was quietly leaning on autocast is now exposed. Two directions:

1. **Crashes** — an explicit `.float()` on an activation that then meets an fp16
   *parameter*. That is exactly the two bugs above. A sweep of the whole r0
   forward path (vision tower, text tower, VTC loss, all-gather) found **no
   remaining instances**; the text tower's attention softmax
   (`transformer.py:295`) is already correct because it casts back before the
   next matmul, and softmax has no parameters.
2. **Silent precision changes** — ops on autocast's fp32 list that now run in
   fp16. The one that matters is the VTC loss: `log_softmax` and `sum` were
   fp32 under autocast and had become fp16. `models/criterions.py` now casts the
   similarity matrices back to fp32 before `log_softmax`.

   **How much did that actually change?** Measured at `B=512`, `temp=0.01`:
   loss `13.929688` (fp16) vs `13.929939` (fp32) — a difference of **2.5e-4**.
   PyTorch's `log_softmax` already accumulates in fp32 internally, so there is no
   "stepwise" loss. The cast is there to restore the repo's original semantics
   exactly, not to rescue the numbers.

   The same measurement argued *against* a change: casting the vision tower's
   attention softmax to fp32 (mirroring the text tower) turned out to be
   **bit-identical** to the plain fp16 softmax — max elementwise difference
   `0.0`, identical row sums — because that kernel also accumulates in fp32.
   It was reverted rather than left in as diff noise.

---

## Pilot dies with `torch.OutOfMemoryError` — but the card is shared, not too small

**Symptom.** All dtype problems are gone; the run reaches the training loop and does
real work in the vision tower before every rank dies:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 480.00 MiB.
GPU 4 has a total capacity of 31.73 GiB of which 404.50 MiB is free.
Process 2621830 has 16.93 GiB memory in use.
Including non-PyTorch memory, this process has 14.39 GiB memory in use.
```

**Read the second and third lines before touching any config.** That "Process
…" line is *someone else's* job. Do the arithmetic:

| | |
|---|---|
| card total | 31.73 GiB |
| **co-tenant process** | **~16.9–17.0 GiB** |
| our training process | 14.36–14.52 GiB (12.46 GiB allocated by PyTorch) |
| free when it died | 268–436 MiB (needed 480 MiB) |

17.0 + 14.5 ≈ 31.5 — the card is saturated by **two** jobs. On an empty card the
pilot needs ~14.5 GiB at `batch_size=64`, i.e. it fits with ~17 GiB to spare.
**Nothing about the training config is wrong.**

**Identifying the co-tenant.** The PIDs were `2621826`–`2621833` — eight
consecutive PIDs, one per GPU. That is the shape of an 8-card data-parallel job,
and on this machine it is the **S2 VLM audit**: `run_v100.sh s2` launches one
`vlm_audit_hf.py` per card (`CUDA_VISIBLE_DEVICES=$i`), and Qwen3-VL-8B in fp16
is ~16 GB of weights. `DATA_ISSUES.md` lists S2 as running. Check with:

```bash
nvidia-smi                                   # who holds the memory
bash tools/data_audit/run_v100.sh status     # S2 progress + live processes
```

GPU 0 and GPU 6 additionally showed ~11 stale ~404 MiB processes. Those are
leftovers worth reaping (`nvidia-smi`, then kill your own dead PIDs) but they are
not what caused this.

**What to do.**

- **Preferred: let S2 finish first.** It is the blocking step for the whole audit
  pipeline (everything downstream eats its output) and it is ~2–3 h. Then run the
  pilot on free cards at the documented `batch_size=64`.
- **To prove the training loop right now**, run the smoke with a small batch — it
  fits in the ~14.7 GiB S2 leaves behind:

  ```bash
  export MVC_DATA_DIR=/home/vca1/datasets
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 654 MiB was reserved-but-unallocated
  bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 \
      batch_size 16 debug True
  ```

  Overriding `batch_size` on the command line **does** reach the dataloader:
  `utils/config.py:get_config` applies `merge_list(cfg, args.opts)` *before*
  `eval_dict_leaf(cfg)`, so the `"${batch_size}"` interpolation in
  `inputs.batch_size.{image,video}` is resolved from the overridden value.

> ⚠️ **Batch size is not a free knob for the A/B.** It sets the number of
> in-batch negatives, which changes the contrastive task itself — a bs=16 run and
> a bs=64 run are not comparable. Use a small batch **only** for the smoke test.
> The r0/r1 pair must use the **same** batch size as each other; prefer the
> documented 64 per GPU (global 512) on free cards.

---

## `RuntimeError: Function 'DivBackward0' returned nan values in its 1th output.`

**Symptom.** GPUs are free, `batch_size=64` fits, the forward completes — and the
*backward* dies on every rank:

```
File "models/criterions.py", line 52, in get_sim
    sim_v2t = vision_proj @ text_proj.T / temp
RuntimeError: Function 'DivBackward0' returned nan values in its 1th output.
```

**Read "1th output" carefully.** `DivBackward0` has two outputs: index 0 is the
gradient w.r.t. the numerator, index **1** is the gradient w.r.t. the
**denominator** — here `self.temp`, a learnable `nn.Parameter` initialised to
0.01. So the NaN is specifically in the temperature gradient.

### Two separate things are going on

**1. Why it *crashes* (the immediate blocker): `debug True` turns an expected
event into a fatal one.**

`tasks_clip/pretrain.py` enabled `torch.autograd.set_detect_anomaly(True)`
whenever `debug` was set. That is fundamentally incompatible with fp16 **dynamic
loss scaling**, whose whole method is to start from a deliberately over-large
scale, let the first backward passes overflow, detect the inf/NaN, *skip* the
step and halve the scale — the `OVERFLOW! ... Skipping step` lines. Anomaly
detection raises the instant a NaN appears in any backward output, i.e. *before*
DeepSpeed inspects the gradients, converting a self-correcting event into a hard
crash.

Fixed in the repo: anomaly detection is now skipped (with a log line saying so)
when DeepSpeed + fp16 are both active. It still turns on for bf16 / fp32 /
non-DeepSpeed runs, where it is safe and useful.

**2. Why a NaN appears at all: the learnable temperature caps the usable loss
scale at ~54 — but only from a *random* init. See the correction below.**

`sim = (v @ tᵀ) / temp` with `temp = 0.01` gives the temperature gradient a
`1/temp² = 1e4` amplification. Measured on CPU in fp32 at `B=512` (the global
batch — `vtc_loss` all-gathers), with random normalised embeddings:

| | |
|---|---|
| `\|dL/dtemp\|` | **1.2e3** |
| fp16 max | 65504 |
| **max loss scale before `temp` overflows** | **≈ 54** |
| max loss scale the tower embeddings could take (`\|dL/dvision_proj\|` max = 5.7e-2) | ≈ 1.15e6 |

DeepSpeed's default `initial_scale_power: 16` (= 65536) therefore needs **11
consecutive halvings** before a single step can land — and `debug` mode stops
after **5** steps (`global_step % (2*log_freq + 3) == 0`), so the smoke run would
spend its entire life backing off and show nothing even without the crash.

`initial_scale_power` is now read from the config
(`deepspeed.initial_scale_power`, default lowered to **8** = 256) and can be
overridden per-run:

```bash
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 \
    deepspeed.initial_scale_power 5 debug True
```

Loss scaling is pure numerics — gradients are unscaled before the optimizer step
— so changing it does **not** change the experiment, only how many opening steps
are wasted on back-off.

### Correction after the run (2026-07-22) — the ≈54 bound is a random-init worst case

The table above assumed **random normalised embeddings**, and on that assumption
predicted an opening loss around 13.9. The actual pilot run opened at
**`loss_vtc = 1.1257`**. The prediction was wrong, and the reason matters:

Both towers are initialised from the pretrained `checkpoints/mobileclip_s2.pt`,
so they are *already aligned* before a single step — the epoch-0 held-out
retrieval is 79.6 / 83.0 R@1 (see `experiments/README.md` §2b). With an aligned
model the positive pair already dominates its row, so the softmax is far from
uniform, the residual `(p − y)` is small, and `|dL/dtemp|` is correspondingly far
below the random-init figure. The `1/temp² = 1e4` amplification is real; the
`1.2e3` gradient it was multiplied against is not what this model starts from.

Measured consequence: at `deepspeed.initial_scale_power 5` (= 32) the run
produced **no `OVERFLOW!` lines at all** and completed the smoke cleanly. Treat
≈54 as a lower bound on the safe scale, not an estimate of it.

### ~~`temp` is already pinned at its floor~~ — WRONG, corrected 2026-07-22 (full run)

I previously wrote that `clip_contrastive_temperature()`'s `temp.clamp_(min=temp_min)`
pins `temp` at the 0.01 floor, inferring from a 50-step `debug` run that logged
`temperature: 0.0100` throughout. **That inference was wrong.** In the full 170-step
r0 run `temp` moves **up**: `0.0100 → 0.0101` (step ~11) `→ 0.0102 → 0.0103`, then
plateaus as the cosine schedule decays the lr. The clamp is a floor, and the
gradient never pushes into it — this model wants a *softer* temperature, not a
sharper one. `temp` is a live parameter, so **freezing it would change what r0 is**;
the "close to a no-op" claim is withdrawn.

Why the smoke run looked frozen: 50 steps at lr ≤ 1e-5 move `temp` by ~5e-5, which
rounds away at the logged 4 decimals.

**Separately, in r1 `temp` really is inert** — but for a different reason: the
sigmoid branch calls `vtc_loss_sigmoid(..., temp=1.0, t=logit_scale.exp(), b=logit_bias)`
and never passes `self.temp`, so it receives **no gradient at all**. The r1 log
correspondingly reads `temperature: 0.0100` for all 170 steps. An unused parameter
still sitting in the optimizer — harmless, but don't read it as "temperature is
stable", it simply isn't connected.

## `no_weight_decay()` is dead code — wd 0.2 hits `temp` / `logit_scale` / `logit_bias`

**Symptom.** `MobileViCLIP_Small.no_weight_decay()` declares
`{"temp", "logit_scale", "logit_bias"}`, yet the run log shows all three in the
decayed group:

```
param temp: wd: 0.2, lr: 1e-05
param logit_scale: wd: 0.2, lr: 0.001
param logit_bias: wd: 0.2, lr: 0.001
```

**Cause — two independent misses.** In `utils/optimizer.py:99-107` the calls are
**commented out**, leaving a bare `pass`, so `no_decay = {}` always:

```python
    no_decay = {}
    if hasattr(model, 'no_weight_decay'):
        pass
        # no_decay = model.no_weight_decay()
```

And the usual fallback doesn't catch them either: `add_weight_decay()` zeroes wd
when `len(param.shape) == 1 or name.endswith(".bias")`, but these three are
**0-dim** scalars (`torch.ones([])` → `torch.Size([])`, length **0**), so they miss
the bias/BN filter as well.

**Size of the effect.** AdamW's decoupled decay multiplies by `(1 - lr·wd)` per
step. Over the 170-step pilot:

| param | lr | `(1 - lr·wd)^170` | drift |
|---|---|---|---|
| `logit_scale`, `logit_bias` | 1e-3 | 0.967 | **−3.3 % toward 0** |
| `temp` | 1e-5 | 0.99966 | negligible |

So it is a real but second-order pull on exactly the two parameters r1 is trying
to learn fast. Not diagnosed as the cause of anything yet — logged because
"declared no-decay is silently ignored" is the kind of defect that quietly biases
an ablation. Upstream code; **not changed here.**

## §3 B0 五个评测全部失败 —— 本机没有下游评测数据

### Symptom

`experiments/README.md` §3 的循环跑完,5 个基准**一个数字都没出**,但报错分两种。

**A 类:anno 文件缺失**(`didemo` / `anet` / `ssv2_mc`)—— 秒崩在建 dataset 时:

```
File "dataset/utils.py", line 106, in load_anno
    cur_ann = json.load(open(fp, "r"))
FileNotFoundError: [Errno 2] No such file or directory: 'anno_downstream/didemo_ret_test.json'
```

**B 类:anno 有、视频没有**(`msrvtt` / `k400`)—— 走得更远:配置打印正常、
标注读入正常(`Loading anno_downstream/msrvtt_1k_test.json: 1000/1000`)、
模型载入正常、连文本特征都算完了(`Finished computing text features`),
才在第一批视频上崩:

```
[...] /github/workspace/src/video/video_reader.cc:83: ERROR opening:
      /home/vca1/datasets/MSRVTT_Videos/video7020.mp4, No such file or directory
...
RuntimeError: Error reading /home/vca1/datasets/MSRVTT_Videos/video7020.mp4...
```

### Cause

**这台机器上根本没有下游评测数据。** `tools/data_audit/SETUP_V100.md` 第一段就写着:

> 本机只有预抽取的 1 万条样本(`/home/vca1/datasets/internvid_zh_10k`),
> 没有全量 10M 数据

`MVC_DATA_DIR=/home/vca1/datasets` 下只有 `internvid_zh_10k/`。**7 个评测基准的视频
一个都不在**,11 个 InternVid 训练分片的视频也一个都不在;其中 DiDeMo / ANet /
SSv2 连 anno 都没有(在受限的作者 Google Drive 里,见本文件
「Missing `anno_downstream/*.json`」一节),另外 4 个(msrvtt / k400 / ucf101 /
hmdb51)anno 是齐的 —— 这 4 个只差视频,是 B0 的可行范围。

B 类之所以显得像"配置对了",是因为 `data_root` 只在**真正读视频那一刻**才被触碰
—— 在此之前 anno、权重、文本塔全都能正常跑通,很容易误判成偶发的解码错误。

### Fix

**先跑预检,不要靠崩溃来发现缺数据**(仓库里早就有这个脚本,10 秒):

```bash
cd ~/mobileclip
export MVC_DATA_DIR=/home/vca1/datasets
python experiments/check_readiness.py
```

它对每个数据集分别报 anno / data_root / 抽查视频三项,正好把 A 类和 B 类分开,
末尾还列出"已解锁的 runbook 步骤"。

补数据的路径:

**四个基准的 anno 已经在盘上,只差视频**(2026-07-22 预检实测),其中三个的视频
是公开源、无需授权:

- **HMDB51** — anno 1511 条;视频 ~2.1 GB,`bash tools/download_hmdb51.sh`
  一条命令搞定(脚本已内置 hf-mirror 那个坑的解法,幂等)。**最便宜,先做它。**
- **MSR-VTT** — anno 1000 条;视频 ~6.7 GB 公开 wget。**主指标**,也是唯一
  有外部数字可对照的(t2v R@1≈42.5)。见 `README_zh.md` §6.1。
- **UCF101** — anno 3783 条;视频 ~6.5 GB,CRCV 官网整包,见 `README_zh.md` §6.2。
- **K400** — anno 10117 条;视频体量大且源不稳。可选,不作为阻塞项。

⇒ **B0 = `msrvtt` + `hmdb51` + `ucf101`**(合计约 15 GB),检索 + 分类两类任务都有。
剩下三个近期不做:

- **DiDeMo / ActivityNet** — anno + 视频都缺,重建方法见本文件
  「Running the next retrieval dataset (DiDeMo, ActivityNet)」。
- **SSv2-MC** — anno + 视频都缺,视频需在 20BN/Qualcomm 注册后下载。

### 连带影响:§4 的 InternVid 训练视频**一个都不在**,而且不是"换台机器"能解决的

`configs/data.py` 里 `internvid_v1..v11` 的 `data_root` 是
`$MVC_DATA_DIR/video_sample/InternVId-FLT_{i}`,2026-07-22 的 `check_readiness.py`
显示 11 个分片**全部** `data_root 缺失`。

> **更正(2026-07-22,本节初稿写错过一次)。** 初稿写的是"§4 需要先解决 InternVid
> **全量数据**的落盘位置(H100 机或换机)",这个说法有两处不准:
>
> 1. **盘上那 11 个 anno 不是全量**,而是本仓库自己生成的子集 —— 每片恰好 5000 条、
>    合计 55000 条。生成脚本就在 `docs/internvid-download-zh.md`:
>    `LIMIT = 5000  # 先做子集;要全量把这行设为 None` 与
>    `PER_SHARD = 5000  # 每片条数;设 None = 把全部行平均切到 11 片`。
>    所以"全量 10M"从来没有在这份 anno 里出现过。
> 2. **视频不存在"别处有一份、搬过来就行"**。InternVid 只发布 URL 列表,视频要
>    **逐条从 YouTube 抓**(`tools/download_internvid.py`,文档同上:
>    "❌ 需要你自己抓 —— 就是本文要做的事"),且**必然有永久损耗** ——
>    `Private video` / `Video unavailable` 的条目是真没了,抓不回来
>    (config 里 `jump_filter=True` 且加载失败会重采样,所以缺样本不影响训练启动)。
>
> 正确的说法是:**§4 是一个数据获取任务,不是机器搬迁任务**。换到 H100 也一样要
> 从头抓视频;换机只在带宽/并发/磁盘上有优势。

所以在 101vector 上,**§2b 之后立刻能做的只有**:

1. `pilot_10k_r1a` / `pilot_10k_r1b` 归因拆包(用的就是那 1 万条,数据已在);
2. 补完视频后的 B0(`msrvtt` + `hmdb51` + `ucf101`,约 15 GB,全部公开源)。

§4 要开跑,得先决定:抓多少条(现有 anno 定义了 55k;要更多得重跑转换脚本)、
在哪台机器上抓、能接受多大的 YouTube 损耗率。这是独立的一步,别混进 §4 的
训练排期里。

### 顺带:`BASELINE.md` 在评测全崩时依然会被写出来,而且内容是试点数字

`collect_results.py` 默认扫 `experiments/runs` **和** `experiments/evals`。
那次 5 个评测全崩、`experiments/evals/baseline_official/` 下一个
`eval_res.json` 都没有,它照样从 `experiments/runs/` 捞到了 10k 试点的
`pilot_10k_r0` / `pilot_10k_r1`,写出一个名叫 **BASELINE.md**、里面却是
**试点自检索** r_mean 93.37 / 86.83 的文件,退出码 0、零警告。

那些数字和 MSR-VTT 基线(t2v R@1≈42.5)不是一回事,放在这个文件名下极易被后来
的人当成基线引用。两条防线:

```bash
# 1) 只扫 evals/
python experiments/collect_results.py --dirs experiments/evals \
       --baseline baseline_official --out experiments/BASELINE.md
```

2) 脚本现在会在生成文件顶部写明**扫了哪些目录、含哪些 run**,并在
   `--baseline` 指定的 run 找不到时打印显式警告。读 BASELINE.md 前先看头两行。

## 下过的数据集被预检报成缺失(`data_root 缺失`)—— 先别重下

### Symptom

`experiments/check_readiness.py` 报某个评测集 `data_root 缺失`,但你记得下过。
2026-07-22 的 HMDB51 就是这样。

### Cause 1(最常见):`MVC_DATA_DIR` 让预检看错了根

**各数据集的下载脚本/文档写死的是仓库相对路径 `data/…`**,例如
`tools/download_hmdb51.sh`:

```bash
RAW_DIR="data/hmdb51_hf"
DEST_DIR="data/hmdb51/videos_test"
```

而 `configs/data.py` 的 `data_root` 是 `$MVC_DATA_DIR/hmdb51/videos_test`。
一旦 `export MVC_DATA_DIR=/home/vca1/datasets`(10k 试点需要它),预检就跑去
`/home/vca1/datasets/hmdb51/videos_test` 找 —— 那里从来没放过评测数据。

**这台机器上两批数据不在一个根下**:评测集在仓库 `data/`,试点语料在
`/home/vca1/datasets`。所以预检要**跑两遍**,见 `experiments/README.md` §3.0。

```bash
cd ~/mobileclip
unset MVC_DATA_DIR
python experiments/check_readiness.py          # 评测集应在这一遍亮灯
```

### Cause 2:软链悬空(`ls` 看得见,`os.path.exists` 说没有)

`videos_test/` 里是**软链**,指向 `data/hmdb51_hf/…`。若为了回收那 2.1 GB 把
`data/hmdb51_hf/` 删了(或用相对路径建过软链),软链就全部悬空:
`ls` 照样列出 6766 个名字,但跟随软链的检查全失败。

```bash
find data/hmdb51/videos_test -maxdepth 1 -xtype l | wc -l   # >0 就是悬空
```

修法**不需要重下**(只要 `data/hmdb51_hf/` 里的 `.avi` 还在):

```bash
find data/hmdb51/videos_test -maxdepth 1 -xtype l -delete
find "$(cd data/hmdb51_hf && pwd)" -name '*.avi' -exec ln -sf {} data/hmdb51/videos_test/ \;
```

注意 `find` 的根必须是**绝对路径**,否则软链目标会相对于**软链所在目录**解析,
再次全部悬空(`tools/download_hmdb51.sh` 里对此有长注释)。

### 通用定位手法:拿 anno 里的真实文件名去搜

anno 记的是**裸文件名**,所以可以做外科式搜索,比乱翻目录快得多:

```bash
cd ~/mobileclip
ANNO=anno_downstream/hmdb51_act_test.json       # 换成任何数据集的 anno 即可
NAME=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))[0]['video'])" "$ANNO")
echo "找这个文件:$NAME"
find ~ /home/vca1 -name "$NAME" 2>/dev/null
```

找到后,`data_root` 就是**该文件所在目录**(分类数据集是平铺的,检索数据集同理);
把它对上 `configs/data.py` 里那一行,或用 `MVC_DATA_DIR` / 软链接过去即可。

找不到单个文件时,按目录聚合统计**视频文件都堆在哪**,一眼看出数据集位置:

```bash
for d in ~ /home/vca1 /data /mnt; do
  [ -d "$d" ] && find "$d" \( -name '*.avi' -o -name '*.mp4' \) -printf '%h\n' 2>/dev/null
done | sort | uniq -c | sort -rn | head -20
```

顺带找找原始压缩包(还在就不用重下,`download_hmdb51.sh` 会校验字节数后跳过下载):

```bash
find ~ /home/vca1 -name 'hmdb51*.zip' -size +1G -printf '%10s  %p\n' 2>/dev/null
# 期望 2117581749 字节;放回 data/hmdb51_hf/hmdb51.zip 再跑脚本即可(幂等)
```

### 一条可用于任何数据集的判据

`anno` 若是用 `preprocess/build_act_rec_anno.py --videos <data_root>` 生成的,
它**只保留磁盘上真实存在的视频**(见该文件 `--videos` 的实现:
"Keep only entries whose video exists in videos_dir")。所以:

> **anno 的条数 < 官方划分条数 ⇒ 生成它的那一刻,视频就在盘上。**

HMDB51 实例:split-1 测试集 51 类 × 30 = **1530**,盘上 anno 是 **1511**,
少 19 条 —— 那 19 条是当时磁盘上就缺的。**1511 这个数字本身就是"视频曾在盘上"
的证据**,遇到"是不是根本没下过"的疑问时可以用它反推。

### 已确证的实例(2026-07-22,101vector)

按上面的手法搜完,结论是**四个基准的视频全都在,一个都没丢**,全部位于
`~/mobileclip/data/`(即 `/home/vca1/rby/mobileclip/data/`):

| 基准 | 盘上目录 | 数量 | 与 `configs/data.py` |
|---|---|---|---|
| `hmdb51` | `data/hmdb51/videos_test` | 6754 | ✅ 一致 |
| `ucf101` | `data/ucf101/videos` | 13320 | ✅ 一致 |
| `k400` | `data/kinetics_400_val_10s_320p` | 10119 | ✅ 一致 |
| `msrvtt` | `data/MSRVTT/videos/all` | 10004 | ⚠️ 需软链成 `data/MSRVTT_Videos` |

`data/hmdb51_hf/hmdb51.zip` 也在,字节数 **2117581749**,与
`tools/download_hmdb51.sh` 的 `EXPECTED_SIZE` 完全一致。

**唯一的真实问题是 `MVC_DATA_DIR` 指错了根**,加上 MSR-VTT 的目录名和 config
不一致(`MSRVTT/videos/all` vs `MSRVTT_Videos`)。修法就是 `unset MVC_DATA_DIR`
加一条软链,零下载。

> ⚠️ **`find ~ /home/vca1 ...` 会重复计数。** 该机 `~` = `/home/vca1/rby`,
> 正是 `/home/vca1` 的**子目录**,所以同一棵树被遍历两遍:命中行成对重复、
> 目录计数**翻倍**。上表已折半。写这类搜索时,要么只给互不包含的根,
> 要么最后 `| sort -u`。

> 另注:`/home/vca1/datasets`(10k 试点数据)**没有**出现在按目录聚合的结果里,
> 尽管预检确认那 1 万条视频存在。最可能的原因是它是个**软链**,而 `find`
> 默认不跟随软链(要跟随得用 `find -L`)。排查数据位置时记住这一条,
> 否则会误判成"那批数据也没了"。

### §4 线索:`InternVId-FLT_1` 的目录已存在但抓取没完成

2026-07-22 的预检里,11 个 InternVid 分片**报错并不一致**,这条信息容易被略过:

```
internvid_v1   anno 5000 条,抽查 20 条视频全部缺失(检查 MVC_DATA_DIR / 目录层级)
internvid_v2   anno 5000 条,但 data_root 缺失: data/video_sample/InternVId-FLT_2
```

差别来自 `check_corpus()` 的两级判断:先 `os.path.isdir(root)`,过了才抽查视频。
所以:

- **v2–v11** —— 目录**不存在**,连抓取都没开始过。
- **v1** —— 目录**存在**,但 anno 要的 20 条视频一条都不在。

⇒ 有人开过 shard 1 的抓取(`tools/download_internvid.py` 会 `mkdir -p` 出 `--out`),
但没抓成、或抓下来的文件名与 anno 对不上。排查:

```bash
cd ~/mobileclip
ls data/video_sample/InternVId-FLT_1 | head
ls data/video_sample/InternVId-FLT_1 | wc -l
cat data/video_sample/InternVId-FLT_1/_download_log.tsv 2>/dev/null | tail -20
# 和 anno 要的名字对一下(前 3 条):
python -c "import json;print([e['video'] for e in json.load(open('anno_downstream/InternVid/InternVid-10M-flt_1.json'))[:3]])"
```

若 `_download_log.tsv` 里大量 `fail`,那是 YouTube 侧的永久损耗(见
「§3 B0 五个评测全部失败」一节的 §4 更正);若文件在、只是命名不同,
那就是 `--out` 的字段命名和 anno 的 `video` 字段没对齐,重建软链/改名即可,
**不必重抓**。

---

## Training dies with `RecursionError: maximum recursion depth exceeded` on one rank

**Symptom** — thousands of `No such file or directory` lines, then:

```
maximum recursion depth exceeded in comparison
maximum recursion depth exceeded while calling a Python object
maximum recursion depth exceeded in __instancecheck__
```

```
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
  rank : 5 (local_rank: 5)  exitcode : 1
```

**Read this failure backwards.** Two things mislead:

- **The `RecursionError` is not the bug.** It is what the retry path does when
  *every* sample fails. The real message is the flood above it, which names a
  file that is not on disk.
- **Rank 5 is not special.** It is just the first stack to blow; the others were
  killed by torchrun (`Sending process ... closing signal SIGTERM`). Nothing about
  rank 5, GPU 5, or its data shard is implicated.

### Cause: a corpus whose videos are not on disk

If the failing paths look like `data/video_sample/InternVId-FLT_<N>/`, that is the
**full 11-shard** `internvid_v*` / `internvid_all` corpus. Why those shards are
missing on this machine, and what to check, is already covered above in
「§3 B0 五个评测全部失败」一节的「连带影响:§4 的 InternVid 训练视频**一个都不在**」
与「§4 线索:`InternVId-FLT_1` 的目录已存在但抓取没完成」两节。

The trap worth stating separately: **the pilot and the full corpus use different
`data_root`s**, so a green smoke test on the pilot says nothing about a
full-corpus config finding its data.

| corpus | `data_root` |
|---|---|
| `internvid_v{1..11}`, `internvid_all` | `data/video_sample/InternVId-FLT_<N>` |
| `internvid_zh_10k_pilot`, `..._val` | `data/internvid_zh_10k/videos` |

The full-corpus configs are `experiments/configs/r1_both.py` and
`experiments/configs/r1a_sigmoid.py` (both iterate `available_corpus["internvid_all"]`);
the pilot ones are `pilot_10k_r0.py` / `pilot_10k_r1.py`. Preflight either way:

```bash
python experiments/check_readiness.py      # --decode also tries decoding 1 file/corpus
```

### Why it surfaced as `RecursionError` instead of a clear error

`VidTxtPtTrainDataset.__getitem__` handled a bad sample by picking a random new
index and **recursing into itself**, with no bound. `num_tries` was stored on the
dataset but only ever checked by `AudioVidTxtPtTrainDataset`; the plain
image / video / audio datasets ignored it. Fine for a handful of corrupt files;
with 100 % missing it recursed until the interpreter's stack limit.

Fixed — all four pretraining datasets now go through `retry_load_item()` in
`dataset/pt_dataset.py`: a **bounded loop** that then raises a message naming the
file and pointing at `check_readiness.py`.

The bound is `max(num_tries, MIN_LOAD_RETRIES)` with `MIN_LOAD_RETRIES = 50`, and
the floor is the point. InternVid has permanent YouTube attrition, so a *healthy*
corpus still has some fraction `p` of dead links, and the chance one sample
exhausts n retries is ~`p^(n+1)`. Honouring the config default of `num_tries = 3`
would mean 1e-4 (p=0.1) to 8e-3 (p=0.3) per sample — **5 to 445 spurious crashes
per 55k-sample epoch**, which is worse than the recursion it replaced. At n=50 the
same figures are 1e-51 and 2e-27: random attrition never trips it, while a
fully-missing corpus still fails after 51 quick attempts. Covered by
`tests/test_dataset_retry.py` (including a 30 %-attrition case).

On old code you get `RecursionError`; after `git pull` the same data problem names
the missing path directly.

> Two smaller things went with it: the handlers logged `{ann}`, which is assigned
> *inside* the `try` — had `get_anno()` itself thrown, the handler would have died
> with `UnboundLocalError` while reporting the original error (it never fired,
> since `get_anno()` is pure string joining). And the stray `print(e)` alongside
> the logger, which is what interleaved bare `Error reading ...mp4` lines into the
> flood, is gone.

---

## b100 上一切 git 命令都报 `fatal: not a git repository (or any parent up to mount point /workspace/mount)`

**Symptom** — 在 `/workspace/mount/b100_zaip_data/rby/MobileViCLIP` 里跑任何 git 命令
（`git status`、`git diff`、`git stash create`…）都得到：

```
fatal: not a git repository (or any parent up to mount point /workspace/mount)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
```

`GIT_DISCOVERY_ACROSS_FILESYSTEM` 那行是干扰项——问题不是"跨文件系统找不到上层
仓库"，而是这个目录**根本没有 `.git`**。

**Cause** — 项目当年不是按 `experiments/SETUP_B100.md` §1 `git clone` 下来的，而是
把文件直接拷贝到盘上（rsync/cp 不带 `.git`）。目录里只有工作区文件，没有版本库，
git 自然一路向上找到挂载点也找不到仓库。

**Fix** — 用 `git clone --no-checkout` 只下载版本库，把 `.git` 嫁接进现有目录；现有
文件一个字节都不动，之后它就是一个"基于 `origin/main` 的脏工作区"，本地改动全部
可见，`git pull` 也恢复可用：

```bash
cd /workspace/mount/b100_zaip_data/rby
git clone --no-checkout https://github.com/rbybryan/mobileclip.git MobileViCLIP_gitref
mv MobileViCLIP_gitref/.git MobileViCLIP/.git
rm -rf MobileViCLIP_gitref
cd MobileViCLIP
git config core.fileMode false   # 挂载盘权限位噪声不算改动
git reset -q origin/main         # 索引对齐 main；不碰工作区
git status -sb                   # 至此本地改动现形
```

**共用账号的隔离** — b100 的 root 账号多人共用，git 的"全局"配置不要写进
`~/.gitconfig`。每个用 git 的 shell 先 source 这份环境（文件生成一次即可）：

```bash
cat > /workspace/mount/b100_zaip_data/rby/.gitenv <<'EOF'
export GIT_CONFIG_GLOBAL=/workspace/mount/b100_zaip_data/rby/.gitconfig
export GIT_CONFIG_SYSTEM=/dev/null
EOF
source /workspace/mount/b100_zaip_data/rby/.gitenv
```

之后 `git config --global …`（`safe.directory`、`user.name/email`、代理等）都只写进
`rby/.gitconfig`，其他使用者完全看不到。

**凭据（每次 `git pull` 都要密码）** — 不要用默认的 `credential.helper store`：它会把
token 写进 `$HOME/.git-credentials`，正好落在共用的 root 家目录里。改成把 helper 和
token 都锁进 `rby/`，并用 `includeIf gitdir:` 门控，使它**只对 `rby/` 下的仓库生效**：

```bash
RBY=$(cd /workspace/mount/b100_zaip_data/rby && pwd -P)   # 取真实路径，软链已解析

# ① 身份 + 凭据 helper：单独一个文件，只放 rby/ 内
cat > "$RBY/.gitconfig-cred" <<EOF
[user]
    name = rbybryan
    email = br2566@columbia.edu
[credential]
    helper = store --file $RBY/.git-credentials
EOF

# ② source 过 .gitenv 的 shell 走这条
git config --file "$RBY/.gitconfig" "includeIf.gitdir:$RBY/.path" "$RBY/.gitconfig-cred"

# ③ 兜底：忘了 source .gitenv 时走共用 $HOME（只写路径，不含 token；可跳过）
git config --file "$HOME/.gitconfig" "includeIf.gitdir:$RBY/.path" "$RBY/.gitconfig"

# ④ 写 token（read -s，不进 shell history）
read -rsp "GitHub PAT: " TOK; echo
umask 077
printf 'https://rbybryan:%s@github.com\n' "$TOK" > "$RBY/.git-credentials"
unset TOK
chmod 600 "$RBY/.git-credentials"
```

②③ 用 `git config --file`（同键覆盖），重复跑不会写重。自检（**新开 shell、不 source**）：
`git config --show-origin --get credential.helper` 应指向 `rby/.gitconfig-cred`，
随后 `git pull origin main` 不再问密码。

**做了 ③ 之后不需要每次登录再 `source .gitenv`** —— 身份和凭据由 `$HOME/.gitconfig` 里的
`includeIf` 自动加载。**唯一例外是 `safe.directory`**：git 检查「可疑的仓库所有权」发生在
*early / protected config* 阶段，那时仓库尚未认出，`gitdir:` 条件无从匹配，因此写在
conditional include 里的 `safe.directory` **不被采纳**（实测：`git config --get-all
safe.directory` 看得见该值，同一仓库仍报 `detected dubious ownership`）。若确实撞上这个报错，
把它直接写进共用 `$HOME/.gitconfig`（只有路径、无 token），此后永久免 source：

```bash
git config --global --add safe.directory /workspace/mount/b100_zaip_data/rby/MobileViCLIP
```

`.gitenv` 里的 `GIT_CONFIG_SYSTEM=/dev/null`（屏蔽 `/etc/gitconfig`）同样只在 source 后生效。

四点提醒：GitHub 已取消密码认证，"密码"处填 **PAT**；共用 root 下 `chmod 600` 给不了任何
隔离（大家都是 root），所以请用 **只授权 `rbybryan/mobileclip` 的 fine-grained PAT**；
`gitdir:` 末尾的 `/` 不能省（等价于 `rby/**`）；`credential.helper` 是**叠加**的，若共用
`$HOME/.gitconfig` 里已有别人的全局 helper，会先轮到它 ——
`git config --get-all --show-origin credential.helper` 查一下。

另一条路（单仓库够用时）：把 token 直接写进该仓库 `.git/config` 的 remote URL —— 本就在
`rby/` 内，天然隔离，但只管一个仓库，不像上面那样覆盖 `rby/` 下所有仓库。

---

## b100 §2：pip 装错解释器——venv 提示符亮着，包却全进了共用 conda base；tokenizers 还编不出来

**Symptom** — 按旧版 §2 `python -m venv` + `source activate` 建环境（提示符变成
`(mvc_env) (base)` 两个叠着），随后：

- 每条 `pip install` 的日志里路径全是 `/root/miniforge3/lib/python3.12/site-packages`
  （共用 conda base），不是 venv；
- `pip install transformers==4.28.1 tokenizers==0.13.3 ...` 在 tokenizers 处失败：
  `error: can't find Rust compiler`，整条命令一个包都没装上；
- arch 自检 `python - <<PY import torch` 报 `ModuleNotFoundError: No module named 'torch'`。

**Cause** — 两层：

1. miniforge 的 python 是 **3.12**，它建的 venv **没有自带 `pip`**。于是 shell 里
   `python` 指向 venv（空的），`pip` 却回落到 PATH 里 conda base 的 pip——torch 等
   768 MB 全装进了共用的 `/root/miniforge3`，venv 一个包都没有。提示符
   `(mvc_env) (base)` 叠着就是警报。
2. 即使解释器对了，**py3.12 也没有 tokenizers 0.13.3 的预编译轮子**（cp312 无 wheel，
   源码编译需要 Rust）。仓库钉的是 **Python 3.10**——cp310 有现成轮子。

**Fix** — 三步（详见 `experiments/SETUP_B100.md` §2 新配方）：

```bash
# ① 退出坏 venv 并删除
deactivate 2>/dev/null || true
rm -rf /workspace/mount/b100_zaip_data/rby/mvc_env

# ② 清理误装进共用 base 的包（这些在装前都不存在，卸载即还原原状）
/root/miniforge3/bin/python -m pip uninstall -y \
  torch torchvision torchaudio triton sympy mpmath numpy networkx jinja2 MarkupSafe \
  fsspec pillow open_clip_torch deepspeed py-cpuinfo hjson msgpack ninja einops \
  psutil pydantic pydantic-core annotated-types typing-inspection \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12

# ③ conda 建 python=3.10 环境（目录在 rby/ 内），此后全程 `python -m pip`
source /root/miniforge3/etc/profile.d/conda.sh
conda create -y -p /workspace/mount/b100_zaip_data/rby/mvc_env310 python=3.10
conda activate /workspace/mount/b100_zaip_data/rby/mvc_env310
which python && which pip    # 都必须在 .../rby/mvc_env310/bin/ 下，否则停
```

通用教训：**装第一个包之前先 `which python && which pip`**；两者不在同一个环境
就别装。`(env) (base)` 提示符叠加 = python/pip 可能已分家。

---

## 冒烟 import 崩在 peft / `cannot import name 'Cache' from 'transformers'`（且路径是 `/usr/local/lib`）

**症状**（b100 2026-07-24 首光冒烟）：conda 环境 `mvc_env310` 明明是激活的，
`bash experiments/run_pretrain.sh ... 2 ... debug True` 却报：

```
File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 37, in <module>
    from transformers import Cache, DynamicCache, EncoderDecoderCache, PreTrainedModel
ImportError: cannot import name 'Cache' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
...
failed (exitcode: 1) local_rank: 0 ... of binary: /usr/bin/python3
```

**这里其实是两个独立故障叠在一起，别只治后半截：**

**(a) 跑错解释器（主因）。** 注意路径是 `/usr/local/lib/python3.10/dist-packages`、
binary 是 `/usr/bin/python3`、torchrun 是 `/usr/local/bin/torchrun`——**整个训练根本没
在你激活的 conda 环境里跑**。原因：`run_pretrain.sh` 从前调用**裸 `torchrun`**，它按
`PATH` 解析；机器上装了系统级 torch 时会命中 `/usr/local/bin/torchrun`，其 shebang 指向
`/usr/bin/python3`，于是绕开 conda 环境。
**Fix：**已把启动器改成 `python -m torch.distributed.run`（保证用当前 `python`），并会
打印 `python : <路径>`。**起跑前自己也核一遍：**

```bash
which python && python -c "import sys,torch; print(sys.executable, torch.__version__, torch.cuda.get_arch_list())"
# python 必须在 .../rby/mvc_env310/bin/ 下；arch 列表必须含 sm_90（H100）
```

**(b) `peft` 被无谓地 import 进来。** 调用链是
`models/__init__.py` → `mobileviclip_tiny` → `backbones/internvideo2/__init__.py`
**第 4 行 `from .internvideo2_clip_text import LLaMA, Tokenizer`** → `import peft`。
但 MobileViCLIP 两个塔**从不使用 `LLaMA`/`Tokenizer`**，全仓库也只有
`internvideo2_clip_text.py` 依赖 `peft`——纯属被 `__init__.py` 的**急切导入**捎带进来。
装了 `peft>=0.8` 而 `transformers` 钉在 4.28.1（无 `Cache`）就炸；环境里干脆没装 peft
则报 `ModuleNotFoundError: No module named 'peft'`。
**Fix：**`backbones/internvideo2/__init__.py` 已改为**惰性导入**（PEP 562 模块级
`__getattr__`）：`InternVideo2`/`TextTransformer`/`ClipTokenizer` 照常拿到，
`LLaMA`/`Tokenizer` 只在**真被访问时**才导入 `internvideo2_clip_text`。于是训练路径
**完全不碰 peft**，`peft` 装没装、装的什么版本都不再影响预训练。

> ⚠️ **`check_readiness.py` 全绿不代表环境能训**：它只 import `decord` 和
> `configs.data`，**从不 import torch**。§5 十一个 `[OK ]` 只证明数据接对了；
> torch/sm_90 要靠 §2 的 arch 自检或 §6 首光冒烟才算验过。

---

## `ModuleNotFoundError: No module named 'open_clip'`——装成了 `openai-clip`

**症状**：`import models` 崩在
`models/backbones/internvideo2/mobileclip/tokenizer.py:7: import open_clip`，
而你确信"CLIP 已经装了"（`pip install openai-clip==1.0.1`）。

**Cause**：**这是两个完全不相干的项目，模块名都不一样**：

| 包名 | 项目 | 提供的模块 |
|------|------|-----------|
| `openai-clip` | OpenAI CLIP | **`clip`** |
| **`open_clip_torch`** | LAION / mlfoundations OpenCLIP | **`open_clip`** ← 仓库要的 |

装 `openai-clip` 再多次也补不上 `import open_clip`。

**Fix**（`scripts/setup_env.sh` Stage 6 的规范装法）：

```bash
python -m pip install --no-cache-dir --no-deps "open_clip_torch==2.20.0"
```

**`--no-deps` 不是可选项**：`open_clip_torch` 的 metadata 里 torch 没有上界，不加
`--no-deps` 时 pip 会为"满足依赖"把 torch 升级掉（1.13.1 栈上会连带把
torchvision/torchaudio 弄坏，报 `cannot import name '_cast_Long'`）。

> 装之前先 `python -c "import sys;print(sys.executable)"` 确认装到的是**自己的**环境——
> b100 是共用 root 账号，装进 `/usr/local/lib/python3.10/dist-packages` 会影响别人。

## `criterion.loss_weight` 是**日志开关**,不是损失权重 —— 改它不影响优化

**Symptom.** 在实验配置里写 `criterion["loss_weight"].update(dict(kd_geo=0.5))`,
期望新增的辅助损失按 0.5 加权。实际:权重取任何非零值,**优化行为完全相同**;
只有 `0` 与"非 0"有区别(而且区别只体现在日志里多/少一行 meter)。

**Cause.** 两处配合导致权重被丢弃。`tasks_clip/pretrain.py:48` 只用
`loss_weight` 的**键**来建 metric meter:

```python
loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]
```

而真正的总损失是 `model.forward` 返回字典的**无权求和**(`:94-95`):

```python
loss_dict = model(image, text_input, idx=idx)
loss = sum(loss_dict.values())          # ← 权重从未参与
```

`MobileViCLIP_Small.forward`(`models/mobileviclip_small.py:153-155`)返回
`dict(loss_vtc=loss_vtc)`,自身也没有乘任何权重。所以 `vtc=1.0` 恰好"正确"
只是因为 1.0 是恒等元。

**Fix.** 新增辅助损失时,**在 `forward` 内部自己乘权重**,例如:

```python
w = self.config.criterion.loss_weight
out = dict(loss_vtc=loss_vtc)
if w.get("kd_geo", 0):
    out["loss_kd_geo"] = w["kd_geo"] * self.geo_loss(...)   # ← 权重在这里生效
return out
```

同时保持"权重为 0 时**提前返回**、完全不执行该分支"——不要写
`loss + 0.0 * aux`:乘 0 仍会执行算子,可能改变 fp16 归约顺序(破坏与基线的逐位
一致),辅助分支里若有 dropout 还会消耗 RNG 状态。

**影响范围.** `notes/neural-inspired-implementation.md` 的 B1 geometry-KD 方案里
`criterion["loss_weight"].update(dict(kd_cos=1.0, kd_geo=1.0))` 就踩了这个坑
(1.0 恰好无害,但一旦想调权重就会静默失效)。已在
`notes/neural-inspired-round2.md` §5 P2 记录。**上游代码,此处未改动。**

## 大负例池 + `temp=1/100` 在 fp16 下会溢出(设计期预警,尚未实测)

**Symptom(预期).** 给对比损失加上几万条缓存负例后,DeepSpeed 频繁 skip step、
loss scale 一路退避,或直接出 `inf`/`nan`。

**Cause.** `temp = 1/100` 意味着 logit 尺度是 **100**。`[512, 65536]` 的 logit
矩阵在 fp16 下,`exp(logit)` 在池子还没用满时就会越过 fp16 上限(65504)。

**Fix.** logit 矩阵与 log-softmax 一律在 **fp32** 中计算并显式减最大值;把
matmul 沿负例维**分块**;每步记录 `max|logit|` 与 DeepSpeed 的 skipped-step 计数。
详见 `notes/neural-inspired-round2.md` §3 R1。
---

## b100 共用系统 site-packages：缺包别再往系统里装，用私有 user-site

**症状**：在 b100 上跑 §6 冒烟，一轮崩一个包——先是 peft 的
`cannot import name 'Cache'`，再是 `No module named 'open_clip'`，接着
`utils/logger.py:14 → No module named 'termcolor'`……而且 `run_pretrain.sh` 打出的
`python : /usr/bin/python`、traceback 里全是
`/usr/local/lib/python3.10/dist-packages/...`，尽管提示符上亮着 `(mvc_env310)`。

**Cause**：两件事叠在一起。

1. **激活的环境没生效**：提示符只是 `PS1` 文本，`command -v python` 仍是
   `/usr/bin/python`。凡是 `python -m pip install` 装的东西，全落进**系统**
   site-packages。
2. **b100 的 root 账号是多人共用的**，`/usr/local/lib/python3.10/dist-packages`
   同时被 wky（中文 NebulaCV 线）和 zyc（翻译流水线）使用。往里装包 = 改别人的环境；
   带版本号装（`pip install transformers==4.28.1 numpy==1.26.4 timm==0.5.4`，也就是
   `scripts/setup_env.sh` Stage 4 那串）会把他们在用的版本**降级**，属于事故级操作。

**注意 `scripts/setup_env.sh` 是给 torch 1.13.1+cu117 的 V100 环境写的，不能照抄到
b100**：b100 的系统 torch 是 2.5.1+cu124（`get_arch_list()` 含 `sm_90`，Hopper 本来就
支持），照抄 Stage 4 会把整台机器的公共依赖拽回旧版本。

**Fix**：沿用系统 torch，缺的小包装进**自己的 user-site**（详见
`experiments/SETUP_B100.md` §2「现行方案」）。

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip
source experiments/env_b100.sh          # 只 export PYTHONUSERBASE / PYTHONPATH / MVC_*
python experiments/check_imports.py     # 一次列全缺哪些包 + 打印可复制的 pip 命令
python -m pip install --user --no-deps <上一步列出的包>
python experiments/check_imports.py     # 直到打印“全部 [OK]”
```

为什么这样安全：

- `--user` 配合 `PYTHONUSERBASE=/workspace/mount/b100_zaip_data/rby/pyuser`，包落到
  rby 自己的目录，**系统 site-packages 一个字节都不动**；user-site 在 `sys.path` 里
  优先于系统目录，所以补的包（乃至更高版本）只对 source 过 `env_b100.sh` 的 shell 生效。
- `--no-deps` 挡住 pip 顺手升/降级 torch、transformers、numpy 这些共用重依赖。
- 用 `python -m pip`（而不是裸 `pip`）保证装到的就是当前解释器；因为跑的是
  `/usr/bin/python` 本身（不在 venv 里），`--user` 不会被 pip 拒绝。

**别再一轮试一个包**：`tasks_clip/pretrain.py` 的 import 链很长，
`experiments/check_imports.py` 会把链上所有第三方模块逐个 try 一遍，一次报全，并顺带
打印 `sys.executable` / `torch.__version__` / `get_arch_list()`，用来确认解释器和
Hopper 支持。

> `PYTHONUSERBASE` 要在**启动训练前**导出：`run_pretrain.sh` 用
> `python -m torch.distributed.run` 拉子进程，子进程继承环境变量。脚本开头会打印
> `python` 与 `userbase`，看到 `userbase : <未设置>` 就是忘了 `source`。

## 评测时打印 `sim_v2t has nan` / `sim_t2v has nan`(fp16,偶发)

**症状**:训练日志的**评测窗口**里穿插一两行 `sim_v2t has nan`(无 logger 前缀的
裸 print),训练 loss 本身正常,当轮检索指标也能正常算出。07-24 的 V100 r3e 冒烟
在 epoch3 出现过一次,epoch4–9 消失(from-claude (17))。

**原因**:打印来自 `models/criterions.py:55` 的 NaN 检查,调用方是
`tasks_clip/retrieval_utils.py:139`(`evaluation_dot`)。`use_half_precision=True`
时模型整体转 fp16,评测前向偶发激活上溢(>65504)→ 个别特征向量出 inf/NaN →
`F.normalize` 后污染相似度矩阵整行。每个 rank 独立提取全量特征、先算完的 rank
先打分,所以这行会穿插在 rank0 的进度条中间——打印位置不代表发生在训练里。

**处理**:
- **单次/偶发**:不用管。训练路径无 NaN(loss 有限)即无碍,下一轮评测消失即自愈。
- **持续出现**:评测切 fp32——跑该次评测时置 `use_half_precision=False`(只影响
  这次评测,训练配置不动);b100 上用 bf16(动态范围大,天然免疫)。
- 快速定位坏在哪一侧:在 `tasks_clip/retrieval_utils.py:139` 前临时加
  `print(torch.isnan(image_feats).any(), torch.isnan(text_feats).any())`。
