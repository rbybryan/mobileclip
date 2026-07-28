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
  --units    fuse_condense.py output      (optional; C_fuse voted units+dropped)
  --fused    fuse_verify.py over condense (optional; C_fuse views + status)
  --fused-raw condense-output glob        (fallback when --fused absent)
  --diag     recap_diag.py per-clip jsonl (optional; collapse metrics)

Row = video | reasons | en_source | zh_translated | recap_1..5 (⟨sig⟩ appended
when scores present) | A_minus_keep | views_A_prime | views_pass ✓ |
views_unverified · | views_fail ✗[verifier] | jaccard_med | ident_pairs | flags.

utf-8-sig + tabs -> opens directly in Excel/WPS.
"""

import argparse
import glob
import json
import os
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
    ap.add_argument("--units", default=None,
                    help="fuse_condense.py output (voted units + dropped)")
    ap.add_argument("--fused", default=None,
                    help="fuse_verify.py output over the VOTE-condensed views (§5.5)")
    ap.add_argument("--fused-raw", default=None,
                    help="raw condense output (fallback when --fused absent: "
                         "shows vote-condensed views as unverified)")
    ap.add_argument("--dfused", default=None,
                    help="fuse_verify.py output over the DIRECT-fused views (§5.7)")
    ap.add_argument("--dfused-raw", default=None,
                    help="raw fuse-direct output (fallback + source of the "
                         "LLM-self-reported dropped list)")
    ap.add_argument("--diag", default=None)
    ap.add_argument("--top-k", type=int, default=3, help="A- keeps top-k recaps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sheet-output", default=None,
                    help="also write a review sheet for make_review_bundle.py "
                         "(better_zh-style index.html bundle); needs --video-root")
    ap.add_argument("--video-root", default=None,
                    help="resolve video_path for the review sheet")
    args = ap.parse_args()

    resolve = None
    if args.sheet_output:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vlm_audit import resolve_video_path  # noqa: E402
        resolve = resolve_video_path

    recaps = load_jsonl(args.recaps)
    if not recaps:
        sys.exit(f"no recap records matched {args.recaps}")
    scores = load_scores(args.scores)
    aprime = load_jsonl(args.aprime)
    verified = load_jsonl(args.verified)
    derived = load_jsonl(args.derived)
    units = load_jsonl(args.units)
    fusedv = load_jsonl(args.fused)
    fused_raw = load_jsonl(args.fused_raw)
    dfusedv = load_jsonl(args.dfused)
    dfused_raw = load_jsonl(args.dfused_raw)
    diag = load_jsonl(args.diag)

    header = ["video", "bucket_reasons", "en_source", "zh_translated",
              *[f"recap_{i + 1}" for i in range(5)], "A_minus_keep",
              "views_A_prime", "views_pass", "views_unverified", "views_fail",
              "fuse_units", "fuse_dropped", "fuse_pass", "fuse_unverified",
              "fuse_fail", "direct_long", "direct_pass", "direct_unverified",
              "direct_fail", "direct_dropped", "jaccard_med", "ident_pairs",
              "flags"]

    # review sheet: maintainer directive 2026-07-27 -- show ONLY the decision
    # chain (en source / old zh / 5 recaps / final condensed views); the full
    # per-stage panorama stays in --output for analysis
    SHEET_Q = ["重述质量[重述更好/差不多/旧译更好]",
               "直融质量[可直接用/信息不足/含幻觉]", "note"]
    sheet_hdr = (["video", "video_path", "英文en", "原zh"]
                 + [f"重述{i + 1}" for i in range(5)]
                 + ["直融长稿", "直融"] + SHEET_Q)
    sheet_rows = []

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
            # C_fuse: voted units (with support counts) + condensed views
            urec = units.get(vid) or {}
            fb = " ⚠保底" if urec.get("fallback") else ""
            u_kept = SEP.join(f"{flat(u.get('text'))}({u.get('support')})"
                              for u in (urec.get("units") or [])) + fb
            u_drop = SEP.join(f"{flat(u.get('text'))}({u.get('support')})"
                              for u in (urec.get("dropped") or []))
            frec = fusedv.get(vid) or {}
            fviews = frec.get("views")
            if fviews is None:
                fviews = [{**v, "status": "unverified", "checks": {}}
                          for v in ((fused_raw.get(vid, {}).get("verdict") or {})
                                    .get("views") or [])]
            f_pass = SEP.join(fmt(v) for v in fviews if v.get("status") == "pass")
            f_unv = SEP.join(fmt(v) for v in fviews
                             if v.get("status") == "unverified")
            f_fail = SEP.join(
                fmt(v) + "[" + "+".join(sorted(v.get("checks") or {})) + "]"
                for v in fviews if v.get("status") == "fail")
            # direct fusion (§5.7): verified views + LLM-self-reported dropped
            dfrec = dfusedv.get(vid) or {}
            dviews = dfrec.get("views")
            if dviews is None:
                dviews = [{**v, "status": "unverified", "checks": {}}
                          for v in ((dfused_raw.get(vid, {}).get("verdict") or {})
                                    .get("views") or [])]
            d_pass = SEP.join(fmt(v) for v in dviews if v.get("status") == "pass")
            d_unv = SEP.join(fmt(v) for v in dviews
                             if v.get("status") == "unverified")
            d_fail = SEP.join(
                fmt(v) + "[" + "+".join(sorted(v.get("checks") or {})) + "]"
                for v in dviews if v.get("status") == "fail")
            d_drop = SEP.join(flat(t) for t in ((dfused_raw.get(vid, {})
                              .get("verdict") or {}).get("dropped") or []))
            d_long = flat(((dfused_raw.get(vid, {}) or {})
                           .get("verdict") or {}).get("long") or "")
            drec = diag.get(vid, {})
            jmed = drec.get("jaccard_med")
            flags = []
            if isinstance(jmed, (int, float)) and jmed > 0.8:
                flags.append("COLLAPSE")
            if (drec.get("n_identical_pairs") or 0) >= 3:
                flags.append("IDENT")
            if vrec.get("error"):
                flags.append("DECODE_ERR")
            # §5.8-3 Missing-direction sentinel: fused output must cover the
            # majority-voted (support>=2) unit glosses; coverage <70% -> flag
            want = {g for u in (urec.get("units") or [])
                    if (u.get("support") or 0) >= 2
                    for g in (u.get("gloss") or [])}
            if want:
                for mtag, mviews in (("MISS_D", dviews), ("MISS_V", fviews)):
                    have = {g for v in mviews if v.get("status") != "fail"
                            for g in (v.get("gloss") or [])}
                    cov = len(want & have) / len(want)
                    if cov < 0.7:
                        flags.append(f"{mtag}{int(cov * 100)}")
            row = [flat(vid), flat("; ".join(task.get("reasons") or [])),
                   flat(task.get("en")), flat(task.get("zh_old") or task.get("zh")),
                   *cap_cols, keep,
                   SEP.join((aprime.get(vid, {}).get("views")) or []),
                   passes, unv, fails,
                   u_kept, u_drop, f_pass, f_unv, f_fail,
                   d_long, d_pass, d_unv, d_fail, d_drop,
                   f"{jmed:.3f}" if isinstance(jmed, (int, float)) else "",
                   str(drec.get("n_identical_pairs", "")),
                   "|".join(flags)]
            fout.write("\t".join(row) + "\n")
            if args.sheet_output:
                kept = {k for k in keep.split(",") if k}
                recap_cells = []
                for k in range(5):
                    c = cap_cols[k]
                    if c and str(k + 1) in kept:
                        c += " ★A−"
                    recap_cells.append(c)
                vote_c = SEP.join(flat(v.get("text", "")) for v in fviews
                                  if v.get("status") != "fail")
                if urec.get("fallback"):
                    vote_c += " ⚠保底"
                dir_c = SEP.join(flat(v.get("text", "")) for v in dviews
                                 if v.get("status") != "fail")
                sheet_rows.append(
                    [flat(vid),
                     resolve(args.video_root or "", task, vid) if resolve else "",
                     flat(task.get("en")),
                     flat(task.get("zh_old") or task.get("zh")),
                     *recap_cells, d_long, dir_c]
                    + [""] * len(SHEET_Q))
            n_rows += 1
            if args.limit and n_rows >= args.limit:
                break
    if args.sheet_output and sheet_rows:
        with open(args.sheet_output, "w", encoding="utf-8-sig", newline="") as f:
            f.write("\t".join(sheet_hdr) + "\n")
            for r in sheet_rows:
                f.write("\t".join(r) + "\n")
        print(f"[sheet] {len(sheet_rows)} rows -> {args.sheet_output} "
              f"(feed to make_review_bundle.py)")
    print(f"[done] {n_rows} rows -> {args.output}")
    print("列: video|理由|en源|zh译|5重述(⟨SigLIP分⟩)|A−保留|A′句切|"
          "✓过验真|·未验|✗被滤[验真器]|投票单元(票)|淘汰单元|投票浓缩✓·✗|"
          "直融✓·✗|直融dropped|多样性|旗标")


if __name__ == "__main__":
    main()
