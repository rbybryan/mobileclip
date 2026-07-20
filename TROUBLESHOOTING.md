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
until HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  Qwen/Qwen3-VL-8B-Instruct --local-dir ckpts/Qwen3-VL-8B-Instruct; do
  sleep 10
done
```

Sanity check when a channel reports success — all four shards present and full
size (the 8B model totals ~17 GB; a partially written shard fails loudly at
load time):

```bash
ls -l ckpts/Qwen3-VL-8B-Instruct/model-0000*-of-00004.safetensors
```

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
