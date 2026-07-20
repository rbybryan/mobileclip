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
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_audit import (PROMPT, extract_json,  # noqa: E402 - shared rubric/parser
                       resolve_video_path)


def sample_frames_pil(path, num_frames, max_side):
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    n = len(vr)
    if n == 0:
        raise RuntimeError("empty video")
    # TSN-style segment centers: InternVid clips are shot-cut, so the first/last
    # frames are often transitions/black frames — never sample the boundaries.
    idx = ((np.arange(num_frames) + 0.5) / num_frames * n).astype(int)
    frames = vr.get_batch(idx).asnumpy()
    out = []
    for fr in frames:
        img = Image.fromarray(fr)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        out.append(img)
    return out


@torch.no_grad()
def generate(model, processor, frames, prompt, max_new_tokens, temperature=0.0):
    content = [{"type": "image", "image": im} for im in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=temperature > 0, temperature=temperature or None)
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
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="max records for this shard")
    args = ap.parse_args()

    if args.dtype == "auto":
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16) if torch.cuda.is_available() else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[args.dtype]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"[load] {args.model} dtype={dtype} (Volta: fp16+sdpa)", file=sys.stderr)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
        device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
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
            path = resolve_video_path(args.video_root, rec, rec["video"])
            try:
                frames = sample_frames_pil(path, args.frames, args.max_side)
                prompt = PROMPT.format(n_frames=args.frames,
                                       en=(rec.get("en") or "(无)"),
                                       zh=(rec.get("zh") or "(空)"))
                try:
                    text = generate(model, processor, frames, prompt,
                                    args.max_new_tokens)
                    out["verdict"] = extract_json(text)
                except (ValueError, torch.cuda.OutOfMemoryError) as ex:
                    if isinstance(ex, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    # one retry with mild sampling shakes off malformed JSON
                    text = generate(model, processor, frames, prompt,
                                    args.max_new_tokens, temperature=0.3)
                    out["verdict"] = extract_json(text)
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
            if n_done % 100 == 0:
                print(f"[progress] shard {args.shard_id}: {n_done}/{len(todo)}",
                      file=sys.stderr)
    print(f"[done] shard {args.shard_id}: {n_done} verdicts -> {out_path}")


if __name__ == "__main__":
    main()
