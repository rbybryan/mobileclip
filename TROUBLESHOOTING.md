# Troubleshooting

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
three places under two different key names with two different defaults, so
`data_type` is not reliably the dtype the weights actually have —

| file | key | default | decides |
|------|-----|---------|---------|
| `tasks_clip/shared_utils.py:57` | **`bf16`** | `True` | model → bf16 or fp16 |
| `tasks_clip/pretrain.py` | `use_bf16` | `True` | `data_type` for autocast |
| `utils/config_utils.py:130` | `use_bf16` | `False` | `bf16` vs `fp16` section in `deepspeed_config.json` |

They only agree when a config sets **both** `bf16` and `use_bf16` explicitly
(the pilot configs do). Asking the model for its dtype sidesteps all three; when
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
