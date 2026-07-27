#!/usr/bin/env python3
"""A/B style comparison sheet: 方案A (dense recaption) vs 方案B (OwlCap struct).

Joins the two pipeline-sample runs of the SAME sample (dense outputs `pipe_*`,
struct outputs `pipe_st_*`) into one review sheet for make_review_bundle.py:
per clip -- en source | old zh | A recaps | A condensed | B recaps | B condensed
+ one forced-choice question. Nothing is recomputed here; missing inputs give
empty cells, never a crash (either arm may not have run yet).

Example (what run_v100.sh pipeline-compare calls):
    python tools/data_audit/make_style_compare.py \
        --tasks outputs/audit_10k/v0_2/pipeline_sample.jsonl \
        --video-root data/InternVid-zh/videos \
        --recaps-a outputs/audit_10k/pipe_recaps.jsonl \
        --fused-a outputs/audit_10k/pipe_fused_verified.jsonl \
        --fused-a-raw 'outputs/audit_10k/pipe_fused.*.jsonl' \
        --recaps-b outputs/audit_10k/pipe_st_recaps.jsonl \
        --fused-b outputs/audit_10k/pipe_st_fused_verified.jsonl \
        --fused-b-raw 'outputs/audit_10k/pipe_st_fused.*.jsonl' \
        --output outputs/audit_10k/review_compare.tsv
"""

import argparse
import glob
import json
import os
import re
import sys

SEP = " ‖ "
WS = re.compile(r"\s+")


def flat(x):
    return WS.sub(" ", str(x)).strip() if x is not None else ""


def load_jsonl(pattern):
    recs = {}
    if not pattern:
        return recs
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = r.get("video")
                if v:
                    recs[v] = r
    return recs


def recap_cell(rec):
    caps = ((rec or {}).get("verdict") or {}).get("recaps") or []
    return SEP.join(flat(c) for c in caps if isinstance(c, str) and c.strip())


def condensed_cell(verified_rec, raw_rec):
    """Final condensed views: verified file preferred (drop status=fail);
    raw condense output as fallback (no verification ran -> keep all)."""
    views = (verified_rec or {}).get("views")
    if views is None:
        views = ((raw_rec or {}).get("verdict") or {}).get("views") or []
    return SEP.join(flat(v.get("text")) for v in views
                    if isinstance(v, dict) and v.get("status") != "fail"
                    and flat(v.get("text")))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="sampled task queue jsonl")
    ap.add_argument("--video-root", default="")
    ap.add_argument("--recaps-a", default=None)
    ap.add_argument("--fused-a", default=None, help="A vote-condensed, verified")
    ap.add_argument("--fused-a-raw", default=None)
    ap.add_argument("--dfused-a", default=None, help="A direct-fused, verified")
    ap.add_argument("--dfused-a-raw", default=None)
    ap.add_argument("--recaps-b", default=None)
    ap.add_argument("--fused-b", default=None, help="B vote-condensed, verified")
    ap.add_argument("--fused-b-raw", default=None)
    ap.add_argument("--dfused-b", default=None, help="B direct-fused, verified")
    ap.add_argument("--dfused-b-raw", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vlm_audit import resolve_video_path  # noqa: E402

    ra, fa, fra = (load_jsonl(args.recaps_a), load_jsonl(args.fused_a),
                   load_jsonl(args.fused_a_raw))
    da, dra = load_jsonl(args.dfused_a), load_jsonl(args.dfused_a_raw)
    rb, fb, frb = (load_jsonl(args.recaps_b), load_jsonl(args.fused_b),
                   load_jsonl(args.fused_b_raw))
    db, drb = load_jsonl(args.dfused_b), load_jsonl(args.dfused_b_raw)

    hdr = ["video", "video_path", "英文en", "原zh",
           "A重述", "A直融", "A投票", "B重述", "B直融", "B投票",
           "最佳浓缩[A直融/A投票/B直融/B投票/都不行]", "note"]
    n = 0
    with open(args.output, "w", encoding="utf-8-sig", newline="") as fout:
        fout.write("\t".join(hdr) + "\n")
        with open(args.tasks, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = task.get("video")
                if not vid:
                    continue
                fout.write("\t".join([
                    flat(vid),
                    resolve_video_path(args.video_root, task, vid),
                    flat(task.get("en")),
                    flat(task.get("zh_old") or task.get("zh")),
                    recap_cell(ra.get(vid)),
                    condensed_cell(da.get(vid), dra.get(vid)),
                    condensed_cell(fa.get(vid), fra.get(vid)),
                    recap_cell(rb.get(vid)),
                    condensed_cell(db.get(vid), drb.get(vid)),
                    condensed_cell(fb.get(vid), frb.get(vid)),
                    "", ""]) + "\n")
                n += 1
    both = sum(1 for v in ra if v in rb)
    print(f"[done] {n} rows -> {args.output} "
          f"(A covered {len(ra)}, B covered {len(rb)}, both {both})")


if __name__ == "__main__":
    main()
