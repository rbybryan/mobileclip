#!/usr/bin/env python3
"""Stage-2 VLM audit, transformers backend — for GPUs vLLM won't serve on.

Same input/output contract as vlm_audit.py (aggregate_report.py consumes both),
but runs Qwen3-VL / Qwen2.5-VL in-process via HuggingFace transformers instead
of an OpenAI-compatible endpoint. Written for 8x V100:

  * Volta has no bf16 and no FlashAttention-2, and vLLM's V1 engine does not
    support it -- this script uses fp16 + SDPA attention instead.
  * data-parallel: one process per GPU via --num-shards/--shard-id
    (CUDA_VISIBLE_DEVICES pins the card), resumable per shard.

Example (single shard; run_v100.sh fans out all 8):
    CUDA_VISIBLE_DEVICES=0 python tools/data_audit/vlm_audit_hf.py \
        --input outputs/audit_v0_1/s2_sample.jsonl --video-root $CLIPS \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --num-shards 8 --shard-id 0 \
        --output "outputs/audit_v0_1/s2_verdicts.{shard}.jsonl"

Model picks for V100: 32GB card -> Qwen3-VL-8B-Instruct (fp16 ~17GB);
16GB card -> Qwen3-VL-4B-Instruct. If fp16 ever yields garbage/NaN output
(rare bf16->fp16 overflow), drop to the 4B model with --dtype fp32.

Deps: torch, transformers>=4.57 (Qwen3-VL), decord, pillow, numpy.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_audit import (DERIVE_KEYS, DERIVE_PROMPT, GRAN_KEYS,  # noqa: E402
                       GRAN_PROMPT, MERGE_KEYS, MERGE_PROMPT, PROMPT,
                       RECAP_PROMPT, TRANS_KEYS, TRANS_PROMPT, VERDICT_KEYS,
                       extract_json, resolve_video_path)


def sample_frames_pil(path, num_frames, max_side, phase=0.5):
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    n = len(vr)
    if n == 0:
        raise RuntimeError("empty video")
    # TSN-style segment centers: InternVid clips are shot-cut, so the first/last
    # frames are often transitions/black frames — never sample the boundaries.
    # `phase` is the within-segment offset (0.5 = centre). --mode recap sweeps it
    # per sample (frame jitter) so each of the N independent captions sees a
    # slightly different 8-frame view: a unit surviving across DIFFERENT visual
    # evidence is more robustly grounded than one that merely survives decoding
    # noise on a fixed frame set (notes/caption-fusion-scheme.md §1.1-3).
    idx = np.clip(((np.arange(num_frames) + phase) / num_frames * n).astype(int),
                  0, n - 1)
    frames = vr.get_batch(idx).asnumpy()
    out = []
    for fr in frames:
        img = Image.fromarray(fr)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        out.append(img)
    return out


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_RECAP_STRIP = "\"'“”「」《》 \t\r\n"


def clean_caption(text):
    """A recap generation is a plain caption, not JSON: drop any <think> block
    (Qwen3.x thinking can leak through the HF chat template) and surrounding
    quotes/whitespace, and collapse newlines so one caption is one line."""
    t = _THINK_RE.sub("", text).strip(_RECAP_STRIP)
    t = re.sub(r"\s*\n+\s*", "，", t)
    # a line already ending in punctuation + the newline-comma would double up
    t = re.sub(r"[，,。.；;]*，[，,]*", "，", t)
    return t.strip(_RECAP_STRIP)


@torch.no_grad()
def generate(model, processor, frames, prompt, max_new_tokens,
             temperature=0.0, top_p=None):
    # frames may be empty (--mode trans): the message is then pure text, which
    # the same chat template handles -- no vision tower call at all.
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, temperature=temperature or None,
        top_p=top_p if temperature > 0 else None)
    new_tokens = gen[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="sample jsonl from make_audit_sample.py")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--output", required=True,
                    help="output jsonl; '{shard}' is replaced by shard id")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--device-map", default="cuda",
                    help="'cuda' = whole model on one card (default, pairs with "
                         "--num-shards data parallelism); 'auto' = shard the model "
                         "across all visible GPUs, for models too big for one card "
                         "(e.g. a 32B fp16 judge on 8xV100-32G; use --num-shards 1)")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="max records for this shard")
    ap.add_argument("--n-captions", type=int, default=5,
                    help="[--mode recap] independent captions per clip (default 5; "
                         ">=5 keeps the support vote meaningful, see fusion scheme "
                         "§1.1/§6.3). Don't go below 5.")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="[--mode recap] sampling temperature (default 0.9). Must be "
                         "high enough to produce real variance: at low temp the N "
                         "captions collapse to near-identical text and the agreement "
                         "signal becomes vacuous (fusion scheme §1.1-2).")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="[--mode recap] nucleus sampling top_p (default 0.9)")
    ap.add_argument("--no-frame-jitter", action="store_true",
                    help="[--mode recap] reuse the SAME 8 frames for every sample "
                         "(centre phase); by default each sample jitters the "
                         "within-segment phase so variance also comes from seeing "
                         "different frames. --no-frame-jitter isolates decoding "
                         "variance for the temperature ablation (§6.4).")
    ap.add_argument("--mode", default="full",
                    choices=["full", "trans", "gran", "merge", "recap", "derive"],
                    help="'full' = combined video rubric (default). 'recap' = write "
                         "N independent Chinese captions FROM the frames (RECAPTION "
                         "bucket front-end for the fusion scheme; RECAP_PROMPT, "
                         "temperature + frame jitter, no en/zh shown). 'derive' = "
                         "text-only per-recap view derivation (A_llm, master plan "
                         "D5): input = recap5 jsonl, one call per recap, outputs "
                         "verdict.views = [{text, gloss, src_recap}]. 'trans' = "
                         "text-only translation re-judgement: no video is read "
                         "and no frames are shown, so trans_score cannot anchor "
                         "on the video-alignment judgement (see vlm_audit.py "
                         "TRANS_PROMPT and DATA_ISSUES.md problem 8). 'gran' = "
                         "text-only granularity re-judgement that asks for an "
                         "information-point COUNT instead of a 3-way band, so "
                         "the label space cannot collapse (problem 14)")
    args = ap.parse_args()

    if args.dtype == "auto":
        # capability >= 8.0 (Ampere+), NOT is_bf16_supported(): that reports
        # True on Volta where bf16 is only software-emulated and crawls
        dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16) if torch.cuda.is_available() else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[args.dtype]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] {args.model} dtype={dtype} (Volta: fp16+sdpa)", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
        device_map=args.device_map if torch.cuda.is_available() else "cpu").eval()
    processor = AutoProcessor.from_pretrained(args.model)

    out_path = args.output.replace("{shard}", str(args.shard_id))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[resume] shard {args.shard_id}: {len(done)} verdicts present",
              file=sys.stderr)

    todo = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % args.num_shards != args.shard_id:
                continue
            rec = json.loads(line)
            if rec["video"] not in done:
                todo.append(rec)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[audit] shard {args.shard_id}: {len(todo)} clips", file=sys.stderr)

    n_done = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for rec in todo:
            out = {"video": rec["video"], "stratum": rec.get("stratum"),
                   "s1": rec.get("s1"), "s0_flags": rec.get("s0_flags")}
            try:
                if args.mode == "recap":
                    # N independent captions FROM the frames. Each sample jitters
                    # the frame phase so variance comes from BOTH decoding and
                    # different views; no en/zh is shown (RECAP_PROMPT has no
                    # placeholders). Plain-text out, so no extract_json below.
                    path = resolve_video_path(args.video_root, rec, rec["video"])
                    recaps = []
                    for s in range(args.n_captions):
                        phase = (0.5 if args.no_frame_jitter
                                 else (s + 0.5) / args.n_captions)
                        frames = sample_frames_pil(path, args.frames,
                                                   args.max_side, phase=phase)
                        try:
                            text = generate(model, processor, frames, RECAP_PROMPT,
                                            args.max_new_tokens,
                                            temperature=args.temperature,
                                            top_p=args.top_p)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            text = generate(model, processor, frames, RECAP_PROMPT,
                                            args.max_new_tokens,
                                            temperature=args.temperature,
                                            top_p=args.top_p)
                        cap = clean_caption(text)
                        if cap:
                            recaps.append(cap)
                    out["verdict"] = {"recaps": recaps, "n_samples": len(recaps)}
                elif args.mode == "derive":
                    # A_llm per-recap view derivation (master plan D5/D8): one
                    # text-only call per recap; the dense recap is the sole
                    # evidence (closed set -- frames withheld on purpose so the
                    # deriver cannot add new details). Reuses the cached VLM as
                    # a text LLM: no extra weights needed on the box.
                    recaps = [c for c in ((rec.get("verdict") or {}).get("recaps")
                                          or []) if isinstance(c, str) and c.strip()]
                    views = []
                    for i, cap in enumerate(recaps):
                        prompt = DERIVE_PROMPT.format(dense=cap)
                        obj = None
                        try:
                            text = generate(model, processor, [], prompt,
                                            args.max_new_tokens)
                            obj = extract_json(text, DERIVE_KEYS)
                        except (ValueError, torch.cuda.OutOfMemoryError) as ex:
                            if isinstance(ex, torch.cuda.OutOfMemoryError):
                                torch.cuda.empty_cache()
                            try:  # one retry with mild sampling
                                text = generate(model, processor, [], prompt,
                                                args.max_new_tokens,
                                                temperature=0.3)
                                obj = extract_json(text, DERIVE_KEYS)
                            except Exception:  # noqa: BLE001
                                obj = None
                        for v in (obj or {}).get("views") or []:
                            if isinstance(v, dict) and isinstance(v.get("text"),
                                                                  str) \
                                    and v["text"].strip():
                                gloss = [g.strip().lower()
                                         for g in (v.get("gloss") or [])
                                         if isinstance(g, str) and g.strip()]
                                views.append({"text": v["text"].strip(),
                                              "gloss": gloss, "src_recap": i})
                    seen, uniq = set(), []
                    for v in views:
                        if v["text"] not in seen:
                            seen.add(v["text"])
                            uniq.append(v)
                    out["verdict"] = {"views": uniq, "n_views": len(uniq),
                                      "n_recaps": len(recaps)}
                elif args.mode == "merge":
                    # 合并 better_zh 的多条候选（在 s2_verdicts 里）为一条。纯文本。
                    cands = [c for c in ((rec.get("verdict") or {}).get("better_zh")
                                         or rec.get("better_zh") or [])
                             if isinstance(c, str) and c.strip()]
                    if len(cands) < 2:
                        # 0-1 条无从合并：原样带出（1 条就是它自己），不调模型
                        out["verdict"] = {"merged_zh": cands[0] if cands else None,
                                          "n_cands": len(cands)}
                        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                        fout.flush(); n_done += 1
                        continue
                    frames = []
                    keys = MERGE_KEYS
                    prompt = MERGE_PROMPT.format(
                        cands="\n".join(f"{i+1}. {c}" for i, c in enumerate(cands)))
                elif args.mode == "gran":
                    frames = []
                    keys = GRAN_KEYS
                    prompt = GRAN_PROMPT.format(zh=(rec.get("zh") or "(空)"))
                elif args.mode == "trans":
                    # No video is opened at all: translation fidelity is a
                    # text-vs-text property, and withholding the frames is the
                    # whole point of this mode.
                    frames = []
                    keys = TRANS_KEYS
                    prompt = TRANS_PROMPT.format(en=(rec.get("en") or "(无)"),
                                                 zh=(rec.get("zh") or "(空)"))
                else:
                    keys = VERDICT_KEYS
                    path = resolve_video_path(args.video_root, rec, rec["video"])
                    frames = sample_frames_pil(path, args.frames, args.max_side)
                    prompt = PROMPT.format(n_frames=args.frames,
                                           en=(rec.get("en") or "(无)"),
                                           zh=(rec.get("zh") or "(空)"))
                if args.mode not in ("recap", "derive"):  # these built their verdict above
                    try:
                        text = generate(model, processor, frames, prompt,
                                        args.max_new_tokens)
                        out["verdict"] = extract_json(text, keys)
                    except (ValueError, torch.cuda.OutOfMemoryError) as ex:
                        if isinstance(ex, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        # one retry with mild sampling shakes off malformed JSON
                        text = generate(model, processor, frames, prompt,
                                        args.max_new_tokens, temperature=0.3)
                        out["verdict"] = extract_json(text, keys)
            except Exception as ex:  # noqa: BLE001 - keep the shard running
                msg = f"{type(ex).__name__}: {str(ex)[:120]}"
                if "decode" in msg.lower() or "Video" in type(ex).__name__ \
                        or isinstance(ex, (RuntimeError, OSError)):
                    out["verdict"] = {"video_readable": False, "action": "drop",
                                      "error": msg}
                else:
                    out["verdict"] = {"error": msg}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            # first verdict logged immediately so a fresh launch is visibly alive;
            # V100 throughput is ~1-3 clips/min per shard, so every-10 ≈ every few min
            if n_done == 1 or n_done % 10 == 0:
                print(f"[progress] shard {args.shard_id}: {n_done}/{len(todo)}",
                      file=sys.stderr, flush=True)
    print(f"[done] shard {args.shard_id}: {n_done} verdicts -> {out_path}")


if __name__ == "__main__":
    main()
