#!/usr/bin/env python3
"""Panel-V view-level verification, v0 (master plan §5.4 / Stage 2-6).

Filters derived views (vlm_audit_hf --mode derive output) with heterogeneous,
non-Qwen evidence:

  * gdino : GroundingDINO open-vocab detection (transformers-native class, fp32
            -- deformable attention is fp16-fragile; SM70-safe, no CUDA ext).
            A view with `gloss` passes if any gloss phrase is detected in any
            sampled frame. ~0.65 phrase precision -> ONE vote, never sole proof
            of absence for gloss-free views.
  * ocr   : RapidOCR (onnxruntime; PP-OCRv4 weights) for views quoting on-screen
            text “...” -- every quoted span must fuzzy-match the OCR concat
            (>= --ocr-min-ratio) or the view fails.

Views with no checkable evidence (no gloss, no quotes) or with unavailable
verifiers are marked "unverified" and KEPT (master plan: 不确定→保守留).
C_v = pass + unverified; "fail" views are the filtered set.

Both verifier stacks degrade gracefully: if a model/import is unavailable the
affected checks record status "unverified" and the pipeline continues.

Example (sample scale, single GPU):
    python tools/fuse_verify.py \
        --views 'outputs/audit_10k/pipe_views.*.jsonl' \
        --video-root /home/vca1/datasets/internvid_zh_10k/videos \
        --output outputs/audit_10k/pipe_verified.jsonl --verifiers gdino,ocr
"""

import argparse
import glob
import json
import os
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_audit"))

QUOTE_RE = re.compile(r"“([^”]{1,40})”")
WS = re.compile(r"\s+")


# ---------------------------------------------------------------- pure helpers
def quoted_spans(text):
    """On-screen-text claims inside a view (RECAP_PROMPT rule 4 quoting style)."""
    return [s.strip() for s in QUOTE_RE.findall(text or "") if s.strip()]


def fuzzy_contains(span, blob):
    """Best sliding-window similarity of `span` against `blob` (both normalized).
    Cheap O(len(blob)*len(span)) — fine at sample scale."""
    span = WS.sub("", span or "")
    blob = WS.sub("", blob or "")
    if not span:
        return 1.0
    if not blob:
        return 0.0
    if span in blob:
        return 1.0
    w = len(span)
    best = 0.0
    for i in range(0, max(1, len(blob) - w + 1)):
        r = SequenceMatcher(None, span, blob[i:i + w]).ratio()
        if r > best:
            best = r
            if best >= 0.999:
                break
    return best


def iglob_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


# ---------------------------------------------------------------- verifiers
def sample_frames(path, num_frames, max_side=640):
    from decord import VideoReader  # noqa: deferred heavy import
    from PIL import Image
    vr = VideoReader(path, num_threads=2)
    n = len(vr)
    idx = [min(n - 1, int((i + 0.5) * n / num_frames)) for i in range(num_frames)]
    arr = vr.get_batch(idx).asnumpy()
    ims = []
    for a in arr:
        im = Image.fromarray(a)
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        ims.append(im)
    return ims


class GDino:
    """transformers-native GroundingDINO; fp32; one query string per view."""

    def __init__(self, model_id, device, box_threshold, text_threshold):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)
        self.torch = torch
        self.device = device
        self.box_t, self.text_t = box_threshold, text_threshold
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(device).eval()

    def check(self, frames, gloss):
        query = ". ".join(g.lower().strip(". ") for g in gloss) + "."
        inputs = self.proc(images=frames, text=[query] * len(frames),
                           return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        sizes = [f.size[::-1] for f in frames]
        try:
            res = self.proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=self.box_t,
                text_threshold=self.text_t, target_sizes=sizes)
        except TypeError:  # older transformers signature
            res = self.proc.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=self.box_t,
                text_threshold=self.text_t, target_sizes=sizes)
        max_score, frames_hit = 0.0, 0
        for r in res:
            scores = [float(s) for s in r.get("scores", [])]
            if scores:
                frames_hit += 1
                max_score = max(max_score, max(scores))
        return {"max_score": round(max_score, 3), "frames_hit": frames_hit,
                "n_frames": len(frames)}


class Ocr:
    """RapidOCR (onnxruntime backend, PP-OCRv4-zh weights)."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()

    def concat(self, frames):
        import numpy as np
        texts = []
        for im in frames:
            result, _ = self.engine(np.array(im))
            for item in result or []:
                if len(item) >= 2 and isinstance(item[1], str):
                    texts.append(item[1])
        return "".join(texts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", required=True,
                    help="derive-output glob (verdict.views records)")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--verifiers", default="gdino,ocr",
                    help="comma list of: gdino, ocr")
    ap.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--ocr-min-ratio", type=float, default=0.70)
    ap.add_argument("--device", default=None, help="cuda/cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from vlm_audit import resolve_video_path  # noqa: E402 light import

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    want = {v.strip() for v in args.verifiers.split(",") if v.strip()}
    gdino = ocr = None
    if "gdino" in want:
        try:
            gdino = GDino(args.gdino_model, device, args.box_threshold,
                          args.text_threshold)
            print(f"[gdino] {args.gdino_model} on {device} (fp32)", file=sys.stderr)
        except Exception as ex:  # noqa: BLE001
            print(f"[gdino] unavailable ({type(ex).__name__}: {str(ex)[:120]}) "
                  f"-> gloss checks will be 'unverified'", file=sys.stderr)
    if "ocr" in want:
        try:
            ocr = Ocr()
            print("[ocr] RapidOCR ready", file=sys.stderr)
        except Exception as ex:  # noqa: BLE001
            print(f"[ocr] unavailable ({type(ex).__name__}: {str(ex)[:120]}) "
                  f"-> quoted-text checks will be 'unverified'", file=sys.stderr)

    done = set()
    if os.path.exists(args.output):
        for line in open(args.output, encoding="utf-8"):
            try:
                done.add(json.loads(line)["video"])
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"[resume] {len(done)} clips already verified", file=sys.stderr)

    n_clips = n_pass = n_fail = n_unv = 0
    with open(args.output, "a", encoding="utf-8") as fout:
        for line in iglob_lines(args.views):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = rec.get("video")
            if vid in done:
                continue
            views = (rec.get("verdict") or {}).get("views") or []
            out = {"video": vid, "views": [], "error": None}
            frames = ocr_blob = None
            try:
                path = resolve_video_path(args.video_root, rec, vid)
                frames = sample_frames(path, args.frames, args.max_side)
            except Exception as ex:  # noqa: BLE001 decode failure -> unverified
                out["error"] = f"decode: {type(ex).__name__}: {str(ex)[:80]}"
            for v in views:
                checks, failed, ran = {}, False, False
                text, gloss = v.get("text", ""), v.get("gloss") or []
                if frames is not None and gdino is not None and gloss:
                    try:
                        g = gdino.check(frames, gloss)
                        checks["gdino"] = g
                        ran = True
                        if g["frames_hit"] == 0:
                            failed = True
                    except Exception as ex:  # noqa: BLE001
                        checks["gdino"] = {"error": str(ex)[:80]}
                spans = quoted_spans(text)
                if frames is not None and ocr is not None and spans:
                    try:
                        if ocr_blob is None:
                            ocr_blob = ocr.concat(frames)
                        ratios = [round(fuzzy_contains(s, ocr_blob), 3)
                                  for s in spans]
                        checks["ocr"] = {"spans": len(spans),
                                         "min_ratio": min(ratios)}
                        ran = True
                        if min(ratios) < args.ocr_min_ratio:
                            failed = True
                    except Exception as ex:  # noqa: BLE001
                        checks["ocr"] = {"error": str(ex)[:80]}
                status = "fail" if failed else ("pass" if ran else "unverified")
                out["views"].append({**v, "status": status, "checks": checks})
                n_pass += status == "pass"
                n_fail += status == "fail"
                n_unv += status == "unverified"
            out["n_pass"] = sum(1 for v in out["views"] if v["status"] == "pass")
            out["n_fail"] = sum(1 for v in out["views"] if v["status"] == "fail")
            out["n_unverified"] = sum(1 for v in out["views"]
                                      if v["status"] == "unverified")
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n_clips += 1
            if args.limit and n_clips >= args.limit:
                break
            if n_clips == 1 or n_clips % 10 == 0:
                print(f"[progress] {n_clips} clips", file=sys.stderr, flush=True)
    tot = max(1, n_pass + n_fail + n_unv)
    print(f"[done] {n_clips} clips: views pass={n_pass} fail={n_fail} "
          f"unverified={n_unv} (C_v 保留 pass+unverified = "
          f"{(n_pass + n_unv) / tot:.0%}) -> {args.output}")


if __name__ == "__main__":
    main()
