#!/usr/bin/env python3
"""Stage-1 cheap full-corpus scorer: multilingual SigLIP2 frame-text alignment.

For every record it samples N frames and computes:
  * sig_zh_mean / sig_zh_max : cosine(frames, zh caption)  -- zh<->video alignment
  * sig_en_mean / sig_en_max : cosine(frames, en caption)  -- source-caption alignment
  * xling_cos                : cosine(zh, en) in a multilingual *text* embedding
                               space (LaBSE by default) -- translation faithfulness
                               proxy, independent of the video

Interpretation matrix (thresholds calibrated in Stage-3 against VLM verdicts):
  sig_en high, sig_zh low, xling low  -> translation broke the caption (RETRANSLATE)
  sig_en low,  sig_zh low             -> source caption misaligned      (RECAPTION)
  sig_en high, sig_zh high            -> keep
  video decode error                  -> engineering bucket             (DROP/refetch)

Runs sharded and resumable:
    python tools/data_audit/score_alignment.py \
        --input .../aligned_internvid_zh_translated_retry_merged.jsonl \
        --video-root /path/to/clips --output outputs/audit/s1_scores.{shard}.jsonl \
        --num-shards 8 --shard-id $i --frames 4 --batch-clips 16

Deps: torch, transformers>=4.49 (siglip2), decord, pillow;
      sentence-transformers for --xlingual-model (optional, pass '' to disable).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

VIDEO_KEYS = ["video", "video_path", "video_id", "clip", "clip_name", "filename"]
ZH_KEYS = ["caption_zh", "zh_caption", "caption_cn", "cn_caption", "translated_caption",
           "translation", "zh", "chinese_caption"]
EN_KEYS = ["caption", "caption_en", "en_caption", "source_caption", "src_caption", "en"]


def detect_keys(rec, args):
    v = args.video_key or next((k for k in VIDEO_KEYS if k in rec), None)
    z = args.zh_key or next((k for k in ZH_KEYS if k in rec), None)
    e = args.en_key or next((k for k in EN_KEYS if k in rec and k != z), None)
    if not (v and z):
        sys.exit(f"cannot detect keys in {sorted(rec.keys())}; pass --video-key/--zh-key")
    return v, z, e


def sample_frames(path, num_frames):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path, num_threads=1)
    n = len(vr)
    if n == 0:
        raise RuntimeError("empty video")
    idx = np.linspace(0, n - 1, num_frames).round().astype(int)
    batch = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(fr) for fr in batch]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--output", required=True,
                    help="output jsonl; '{shard}' is replaced by shard id")
    ap.add_argument("--model", default="google/siglip2-so400m-patch16-384")
    ap.add_argument("--xlingual-model", default="sentence-transformers/LaBSE",
                    help="multilingual text embedder for zh<->en consistency; '' disables")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--batch-clips", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-key", default=None)
    ap.add_argument("--zh-key", default=None)
    ap.add_argument("--en-key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    xling = None
    if args.xlingual_model:
        from sentence_transformers import SentenceTransformer
        xling = SentenceTransformer(args.xlingual_model, device=args.device)

    out_path = args.output.replace("{shard}", str(args.shard_id))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    done = 0
    if os.path.exists(out_path):  # resume: skip already-scored records of this shard
        done = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"[resume] shard {args.shard_id}: {done} records already scored",
              file=sys.stderr)
    fout = open(out_path, "a", encoding="utf-8")

    keys = None
    batch, seen_in_shard, written = [], 0, 0

    def flush(batch):
        nonlocal written
        if not batch:
            return
        # --- video decode (CPU) ---
        frames_per_clip, ok = [], []
        for rec in batch:
            try:
                frames_per_clip.append(sample_frames(rec["path"], args.frames))
                ok.append(True)
            except Exception as ex:  # noqa: BLE001 - any decode failure is data signal
                frames_per_clip.append(None)
                ok.append(False)
                rec["error"] = f"decode: {type(ex).__name__}"
        flat = [fr for fl in frames_per_clip if fl for fr in fl]
        img_emb = None
        if flat:
            inputs = processor(images=flat, return_tensors="pt").to(args.device)
            img_emb = model.get_image_features(**{k: v.to(dtype) if v.is_floating_point()
                                                  else v for k, v in inputs.items()})
            img_emb = torch.nn.functional.normalize(img_emb, dim=-1).float()
        texts, owners = [], []
        for i, rec in enumerate(batch):
            if not ok[i]:
                continue
            texts.append(rec["zh"]); owners.append((i, "zh"))
            if rec.get("en"):
                texts.append(rec["en"]); owners.append((i, "en"))
        txt_emb = None
        if texts:
            tin = processor(text=texts, return_tensors="pt",
                            padding="max_length", truncation=True).to(args.device)
            txt_emb = model.get_text_features(**tin)
            txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1).float()
        xl = None
        if xling is not None:
            pairs = [(rec["zh"], rec["en"]) for rec in batch if rec.get("en")]
            if pairs:
                embs = xling.encode([t for p in pairs for t in p],
                                    convert_to_numpy=True, normalize_embeddings=True,
                                    batch_size=128, show_progress_bar=False)
                xl = {}
                pi = 0
                for i, rec in enumerate(batch):
                    if rec.get("en"):
                        a, b = embs[2 * pi], embs[2 * pi + 1]
                        xl[i] = float(np.dot(a, b))
                        pi += 1
        # --- assemble scores ---
        cursor = 0
        sims = {}
        for i, fl in enumerate(frames_per_clip):
            if fl is None:
                continue
            fe = img_emb[cursor:cursor + len(fl)]
            cursor += len(fl)
            sims[i] = fe
        for (i, lang), te in zip(owners, txt_emb if txt_emb is not None else []):
            s = (sims[i] @ te).cpu().numpy()
            batch[i][f"sig_{lang}_mean"] = round(float(s.mean()), 4)
            batch[i][f"sig_{lang}_max"] = round(float(s.max()), 4)
        for i, rec in enumerate(batch):
            out = {"video": rec["video"]}
            for k in ("sig_zh_mean", "sig_zh_max", "sig_en_mean", "sig_en_max", "error"):
                if k in rec:
                    out[k] = rec[k]
            if xl and i in xl:
                out["xling_cos"] = round(xl[i], 4)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
        fout.flush()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            if args.limit and line_no >= args.limit:
                break
            if line_no % args.num_shards != args.shard_id:
                continue
            seen_in_shard += 1
            if seen_in_shard <= done:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(json.dumps({"video": None, "error": "parse"},
                                      ensure_ascii=False) + "\n")
                written += 1
                continue
            if keys is None:
                keys = detect_keys(rec, args)
                print(f"[keys] video={keys[0]} zh={keys[1]} en={keys[2]}", file=sys.stderr)
            vk, zk, ek = keys
            batch.append({
                "video": rec.get(vk),
                "path": os.path.join(args.video_root, str(rec.get(vk, ""))),
                "zh": str(rec.get(zk, "") or "")[:512],
                "en": (str(rec.get(ek, "") or "")[:512] if ek else ""),
            })
            if len(batch) >= args.batch_clips:
                flush(batch)
                batch = []
                if written % 2000 < args.batch_clips:
                    print(f"[progress] shard {args.shard_id}: {done + written} scored",
                          file=sys.stderr)
    flush(batch)
    fout.close()
    print(f"[done] shard {args.shard_id}: wrote {written} new records to {out_path}")


if __name__ == "__main__":
    main()
