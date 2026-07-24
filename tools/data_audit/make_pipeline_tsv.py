#!/usr/bin/env python3
"""Full caption-optimization pipeline TSV: one clip per row, every stage visible.

Joins (by video):
  --tasks    recaption_tasks.jsonl        (en source / zh_old translation / reasons)
  --recaps   recap5 shard glob            (S0: 5 dense recaptions)
  --scores   score_alignment w/ tag glob  (optional; tag=recap_i SigLIP scores
                                           -> A- top-k recap selection)
  --aprime   split_views.py output        (optional; A' sentence-split views)
  --verified fuse_verify.py output        (optional; A_llm views + panel status)
  --derived  derive-output glob           (fallback when --verified absent:
                                           shows A_llm views as unverified)
  --diag     recap_diag.py per-clip jsonl (optional; collapse metrics)

Row = video | reasons | en_source | zh_translated | recap_1..5 (⟨sig⟩ appended
when scores present) | A_minus_keep | views_A_prime | views_pass ✓ |
views_unverified · | views_fail ✗[verifier] | jaccard_med | ident_pairs | flags.

utf-8-sig + tabs -> opens directly in Excel/WPS.
"""

import argparse
import glob
import json
import re
import sys

FLAT = re.compile(r"[\t\r\n]+")
SEP = " ‖ "


def iglob_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def load_jsonl(path_or_glob, key="video"):
    out = {}
    if not path_or_glob:
        return out
    for line in iglob_lines(path_or_glob):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = rec.get(key)
        if v is not None and v not in out:
            out[v] = rec
    return out


def load_scores(path_or_glob):
    """{(video, tag) -> sig_zh_mean}"""
    out = {}
    if not path_or_glob:
        return out
    for line in iglob_lines(path_or_glob):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("tag") is not None and rec.get("sig_zh_mean") is not None:
            out[(rec.get("video"), rec["tag"])] = rec["sig_zh_mean"]
    return out


def flat(x):
    return FLAT.sub(" ", str(x if x is not None else "")).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--recaps", required=True)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--aprime", default=None)
    ap.add_argument("--verified", default=None)
    ap.add_argument("--derived", default=None)
    ap.add_argument("--diag", default=None)
    ap.add_argument("--top-k", type=int, default=3, help="A- keeps top-k recaps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    recaps = load_jsonl(args.recaps)
    if not recaps:
        sys.exit(f"no recap records matched {args.recaps}")
    scores = load_scores(args.scores)
    aprime = load_jsonl(args.aprime)
    verified = load_jsonl(args.verified)
    derived = load_jsonl(args.derived)
    diag = load_jsonl(args.diag)

    header = ["video", "bucket_reasons", "en_source", "zh_translated",
              *[f"recap_{i + 1}" for i in range(5)], "A_minus_keep",
              "views_A_prime", "views_pass", "views_unverified", "views_fail",
              "jaccard_med", "ident_pairs", "flags"]

    n_rows = 0
    with open(args.output, "w", encoding="utf-8-sig", newline="") as fout:
        fout.write("\t".join(header) + "\n")
        for line in iglob_lines(args.tasks):
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = task.get("video")
            r = recaps.get(vid)
            if r is None:
                continue
            caps = [c for c in ((r.get("verdict") or {}).get("recaps") or [])
                    if isinstance(c, str)]
            # recap columns, with SigLIP score when available
            cap_cols = []
            cap_scores = []
            for i in range(5):
                if i < len(caps):
                    s = scores.get((vid, f"recap_{i + 1}"))
                    cap_scores.append((s if s is not None else float("-inf"), i))
                    cap_cols.append(flat(caps[i]) +
                                    (f" ⟨{s:.3f}⟩" if s is not None else ""))
                else:
                    cap_cols.append("")
            keep = ""
            if any(s != float("-inf") for s, _ in cap_scores):
                top = sorted(cap_scores, reverse=True)[:args.top_k]
                keep = ",".join(str(i + 1) for _, i in sorted(top, key=lambda x: x[1]))
            # A_llm views: prefer verified, fall back to raw derived
            vrec = verified.get(vid) or {}
            vviews = vrec.get("views")
            if vviews is None:
                vviews = [{**v, "status": "unverified", "checks": {}}
                          for v in ((derived.get(vid, {}).get("verdict") or {})
                                    .get("views") or [])]
            def fmt(v):
                g = ",".join(v.get("gloss") or [])
                return flat(v.get("text", "")) + (f" [{g}]" if g else "")
            passes = SEP.join(fmt(v) for v in vviews if v.get("status") == "pass")
            unv = SEP.join(fmt(v) for v in vviews if v.get("status") == "unverified")
            fails = SEP.join(
                fmt(v) + "[" + "+".join(sorted(v.get("checks") or {})) + "]"
                for v in vviews if v.get("status") == "fail")
            drec = diag.get(vid, {})
            jmed = drec.get("jaccard_med")
            flags = []
            if isinstance(jmed, (int, float)) and jmed > 0.8:
                flags.append("COLLAPSE")
            if (drec.get("n_identical_pairs") or 0) >= 3:
                flags.append("IDENT")
            if vrec.get("error"):
                flags.append("DECODE_ERR")
            row = [flat(vid), flat("; ".join(task.get("reasons") or [])),
                   flat(task.get("en")), flat(task.get("zh_old") or task.get("zh")),
                   *cap_cols, keep,
                   SEP.join((aprime.get(vid, {}).get("views")) or []),
                   passes, unv, fails,
                   f"{jmed:.3f}" if isinstance(jmed, (int, float)) else "",
                   str(drec.get("n_identical_pairs", "")),
                   "|".join(flags)]
            fout.write("\t".join(row) + "\n")
            n_rows += 1
            if args.limit and n_rows >= args.limit:
                break
    print(f"[done] {n_rows} rows -> {args.output}")
    print("列: video|理由|en源|zh译|5重述(⟨SigLIP分⟩)|A−保留|A′句切|"
          "✓过验真|·未验|✗被滤[验真器]|多样性|旗标")


if __name__ == "__main__":
    main()
