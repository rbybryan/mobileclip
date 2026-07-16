#!/usr/bin/env python3
"""Build the stratified sample that Stage-2 (VLM audit) will judge.

Strata:
  * K quantile buckets over the Stage-1 `sig_zh_mean` score (uniform coverage of
    the alignment spectrum, so Stage-3 can calibrate thresholds on the full curve)
  * an oversampled slice of Stage-0 flagged records (artifact/refusal/ratio flags)
  * a slice of low `xling_cos` records (suspected mistranslation)

Output jsonl carries video/en/zh plus the stage-1 scores and stage-0 flags so the
audit results can be joined and analysed without re-reading the 10M file.

Example:
    python tools/data_audit/make_audit_sample.py \
        --input .../aligned_internvid_zh_translated_retry_merged.jsonl \
        --scores 'outputs/audit/s1_scores.*.jsonl' \
        --flags outputs/audit/s0_flags.jsonl \
        --per-bucket 3000 --buckets 10 --flagged 10000 --low-xling 5000 \
        --output outputs/audit/s2_sample.jsonl
"""

import argparse
import glob
import json
import random
import sys

VIDEO_KEYS = ["video", "video_path", "video_id", "clip", "clip_name", "filename"]
ZH_KEYS = ["caption_zh", "zh_caption", "caption_cn", "cn_caption", "translated_caption",
           "translation", "zh", "chinese_caption"]
EN_KEYS = ["caption", "caption_en", "en_caption", "source_caption", "src_caption", "en"]


def iter_jsonl(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="original 10M jsonl")
    ap.add_argument("--scores", required=True, help="stage-1 output glob")
    ap.add_argument("--flags", default=None, help="stage-0 flags jsonl (optional)")
    ap.add_argument("--buckets", type=int, default=10)
    ap.add_argument("--per-bucket", type=int, default=3000)
    ap.add_argument("--flagged", type=int, default=10000)
    ap.add_argument("--low-xling", type=int, default=5000)
    ap.add_argument("--xling-threshold", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    score_files = sorted(glob.glob(args.scores))
    if not score_files:
        sys.exit(f"no score files match {args.scores}")

    # pass 1: reservoir-sample per stratum over the score stream
    scores = {}          # video -> score record (only for selected videos)
    all_sig = []
    rows = []
    for rec in iter_jsonl(score_files):
        v = rec.get("video")
        if not v:
            continue
        rows.append(rec)
        if rec.get("sig_zh_mean") is not None:
            all_sig.append(rec["sig_zh_mean"])
    if not all_sig:
        sys.exit("no sig_zh_mean scores found in stage-1 outputs")
    all_sig.sort()
    edges = [all_sig[int(q * len(all_sig) / args.buckets)]
             for q in range(1, args.buckets)]
    print(f"[buckets] sig_zh_mean edges: {edges}", file=sys.stderr)

    def bucket_of(s):
        for bi, e in enumerate(edges):
            if s < e:
                return bi
        return len(edges)

    reservoirs = [[] for _ in range(args.buckets)]
    seen = [0] * args.buckets
    low_xling_res, low_xling_seen = [], 0
    for rec in rows:
        s = rec.get("sig_zh_mean")
        if s is not None:
            bi = bucket_of(s)
            seen[bi] += 1
            if len(reservoirs[bi]) < args.per_bucket:
                reservoirs[bi].append(rec)
            else:
                j = rng.randrange(seen[bi])
                if j < args.per_bucket:
                    reservoirs[bi][j] = rec
        x = rec.get("xling_cos")
        if x is not None and x < args.xling_threshold:
            low_xling_seen += 1
            if len(low_xling_res) < args.low_xling:
                low_xling_res.append(rec)
            else:
                j = rng.randrange(low_xling_seen)
                if j < args.low_xling:
                    low_xling_res[j] = rec

    for bi, res in enumerate(reservoirs):
        for rec in res:
            scores.setdefault(rec["video"], rec)["_stratum"] = f"sig_q{bi}"
    for rec in low_xling_res:
        scores.setdefault(rec["video"], rec).setdefault("_stratum", "low_xling")

    flags = {}
    if args.flags:
        flagged_pool, fseen = [], 0
        for rec in iter_jsonl([args.flags]):
            fseen += 1
            if len(flagged_pool) < args.flagged:
                flagged_pool.append(rec)
            else:
                j = rng.randrange(fseen)
                if j < args.flagged:
                    flagged_pool[j] = rec
        for rec in flagged_pool:
            flags[rec["video"]] = rec["flags"]
            if rec["video"] not in scores:
                scores[rec["video"]] = {"video": rec["video"], "_stratum": "s0_flagged"}

    wanted = set(scores.keys())
    print(f"[sample] selecting {len(wanted)} videos", file=sys.stderr)

    # pass 2: stream the original file once, emit selected records
    keys = None
    n_out = 0
    with open(args.output, "w", encoding="utf-8") as fout, \
         open(args.input, encoding="utf-8", errors="replace") as fin:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if keys is None:
                v = next((k for k in VIDEO_KEYS if k in rec), None)
                z = next((k for k in ZH_KEYS if k in rec), None)
                e = next((k for k in EN_KEYS if k in rec and k != z), None)
                keys = (v, z, e)
                print(f"[keys] video={v} zh={z} en={e}", file=sys.stderr)
            v, z, e = keys
            vid = rec.get(v)
            if vid not in wanted:
                continue
            sc = scores.get(vid, {})
            out = {
                "video": vid,
                "zh": rec.get(z),
                "en": rec.get(e) if e else None,
                "stratum": sc.get("_stratum"),
                "s1": {k: sc[k] for k in
                       ("sig_zh_mean", "sig_zh_max", "sig_en_mean", "sig_en_max",
                        "xling_cos", "error") if k in sc},
                "s0_flags": flags.get(vid, []),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
            wanted.discard(vid)
            if not wanted:
                break
    print(f"[done] wrote {n_out} sample records to {args.output}")


if __name__ == "__main__":
    main()
