#!/usr/bin/env python3
"""Free diagnostics over recap5 output (master plan Stage 0 / Stage 2-2).

Per clip: recap count, length stats, pairwise char-3gram Jaccard (median/max),
identical-pair count. Corpus level: the pre-registered temperature-collapse
red flag (median of per-clip median Jaccard > --collapse-threshold), share of
clips with an identical pair, length percentiles.

Optionally --compare a second batch (e.g. the fixed-frame or low-temp side
batch) to print both summaries side by side.

Example:
    python tools/data_audit/recap_diag.py \
        --recaps 'outputs/audit_10k/recap5.*.jsonl' \
        --out-perclip outputs/audit_10k/recap_diag.jsonl
"""

import argparse
import glob
import json
import re
import sys
from itertools import combinations

WS = re.compile(r"\s+")


def iglob_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def ngrams(text, n=3):
    t = WS.sub("", text or "")
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else {t} if t else set()


def jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def pctl(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def scan(pattern, out_perclip=None):
    per_clip_med, per_clip_max, lens, n_ident_clips, clips = [], [], [], 0, 0
    fout = open(out_perclip, "w", encoding="utf-8") if out_perclip else None
    for line in iglob_lines(pattern):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        recaps = [c for c in ((rec.get("verdict") or {}).get("recaps") or [])
                  if isinstance(c, str) and c.strip()]
        clips += 1
        lens.extend(len(WS.sub("", c)) for c in recaps)
        grams = [ngrams(c) for c in recaps]
        pairs = [jaccard(a, b) for a, b in combinations(grams, 2)]
        ident = sum(1 for a, b in combinations(recaps, 2) if a.strip() == b.strip())
        med = pctl(sorted(pairs), 0.5) if pairs else None
        mx = max(pairs) if pairs else None
        if med is not None:
            per_clip_med.append(med)
            per_clip_max.append(mx)
        if ident:
            n_ident_clips += 1
        if fout:
            fout.write(json.dumps({
                "video": rec.get("video"), "n_recaps": len(recaps),
                "len_min": min(lens[-len(recaps):]) if recaps else 0,
                "len_max": max(lens[-len(recaps):]) if recaps else 0,
                "jaccard_med": round(med, 4) if med is not None else None,
                "jaccard_max": round(mx, 4) if mx is not None else None,
                "n_identical_pairs": ident,
            }, ensure_ascii=False) + "\n")
    if fout:
        fout.close()
    return dict(clips=clips, per_clip_med=sorted(per_clip_med),
                per_clip_max=sorted(per_clip_max), lens=sorted(lens),
                n_ident_clips=n_ident_clips)


def report(name, s, collapse_threshold):
    print(f"== {name}: {s['clips']} clips ==")
    if not s["per_clip_med"]:
        print("  (no multi-recap clips found)")
        return
    med_of_med = pctl(s["per_clip_med"], 0.5)
    p90 = pctl(s["per_clip_med"], 0.9)
    frac_hi = sum(1 for x in s["per_clip_med"] if x > collapse_threshold) / len(s["per_clip_med"])
    print(f"  recap 长度(字符, 去空白): p10={pctl(s['lens'], 0.1)} 中位={pctl(s['lens'], 0.5)} "
          f"p90={pctl(s['lens'], 0.9)}")
    print(f"  片内成对 3-gram Jaccard: 中位数的中位数={med_of_med:.3f}  p90={p90:.3f}")
    print(f"  片内中位 Jaccard > {collapse_threshold}: {frac_hi:.1%} 的片   "
          f"完全相同对出现于: {s['n_ident_clips']}/{s['clips']} 片")
    flag = med_of_med is not None and med_of_med > collapse_threshold
    print(f"  温度坍缩红旗(预登记: 中位数>{collapse_threshold}): "
          f"{'🔴 触发 —— 升 temp / 查帧抖动 (master plan §7)' if flag else '✅ 未触发'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recaps", required=True, help="recap5 shard glob")
    ap.add_argument("--compare", default=None,
                    help="optional second glob (side batch) for side-by-side summary")
    ap.add_argument("--out-perclip", default=None, help="per-clip metrics jsonl")
    ap.add_argument("--collapse-threshold", type=float, default=0.8)
    args = ap.parse_args()

    main_stats = scan(args.recaps, args.out_perclip)
    report("主批", main_stats, args.collapse_threshold)
    if args.compare:
        cmp_stats = scan(args.compare)
        report("对照批", cmp_stats, args.collapse_threshold)
    if main_stats["clips"] == 0:
        sys.exit(f"no records matched {args.recaps}")


if __name__ == "__main__":
    main()
