#!/usr/bin/env python3
"""Inspection TSV for the recap sample run: source EN caption vs zh translation
vs the 5 recaptions vs the split A' views, one clip per row.

Joins (by video):
  --tasks   recaption_tasks.jsonl  (apply_buckets output: video/en/zh_old/reasons)
  --recaps  recap5 shard glob      (verdict.recaps = N dense zh recaptions)
  --views   split_views.py output  (optional; pooled A' views)
  --diag    recap_diag.py per-clip (optional; jaccard/collapse metrics)

Rows are emitted only for clips that HAVE recaps (i.e. the sampled subset).
Output is UTF-8 with BOM (opens correctly in Excel/WPS), tab-separated;
tabs/newlines inside text fields are flattened to spaces.

Example:
    python recaption/scripts/make_recap_tsv.py \
        --tasks  outputs/audit_10k/v0_2/recaption_tasks.jsonl \
        --recaps 'outputs/audit_10k/recap5.*.jsonl' \
        --views  outputs/audit_10k/recap_views.jsonl \
        --diag   outputs/audit_10k/recap_diag.jsonl \
        --output outputs/audit_10k/recap_inspect.tsv
"""

import argparse
import glob
import json
import re
import sys

FLAT = re.compile(r"[\t\r\n]+")


def iglob_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def load_jsonl(path_or_glob, key="video"):
    out = {}
    for line in iglob_lines(path_or_glob):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = rec.get(key)
        if v is not None and v not in out:
            out[v] = rec
    return out


def flat(x, maxlen=0):
    s = FLAT.sub(" ", str(x if x is not None else "")).strip()
    return (s[:maxlen] + "…") if maxlen and len(s) > maxlen else s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True,
                    help="recaption_tasks.jsonl (has video/en/zh_old/reasons)")
    ap.add_argument("--recaps", required=True, help="recap5 shard glob")
    ap.add_argument("--views", default=None, help="split_views.py output jsonl")
    ap.add_argument("--diag", default=None, help="recap_diag.py --out-perclip jsonl")
    ap.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    ap.add_argument("--n-recap-cols", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    recaps = load_jsonl(args.recaps)
    views = load_jsonl(args.views) if args.views else {}
    diag = load_jsonl(args.diag) if args.diag else {}
    if not recaps:
        sys.exit(f"no recap records matched {args.recaps} — run `run_v100.sh recap` first")

    K = args.n_recap_cols
    header = (["video", "bucket_reasons", "en_source", "zh_translated", "len_zh"]
              + [f"recap_{i + 1}" for i in range(K)]
              + ["n_views", "views_A_prime", "jaccard_med", "ident_pairs", "flag_collapse"])

    n_rows = n_tasks_seen = 0
    with open(args.output, "w", encoding="utf-8-sig", newline="") as fout:
        fout.write("\t".join(header) + "\n")
        for line in iglob_lines(args.tasks):
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_tasks_seen += 1
            vid = task.get("video")
            r = recaps.get(vid)
            if r is None:
                continue  # not in the sampled/finished subset
            caps = [c for c in ((r.get("verdict") or {}).get("recaps") or [])
                    if isinstance(c, str)]
            vrec = views.get(vid, {})
            drec = diag.get(vid, {})
            jmed = drec.get("jaccard_med")
            zh_old = task.get("zh_old") or task.get("zh") or ""
            row = ([flat(vid),
                    flat("; ".join(task.get("reasons") or [])),
                    flat(task.get("en")),
                    flat(zh_old),
                    str(len(zh_old))]
                   + [flat(caps[i]) if i < len(caps) else "" for i in range(K)]
                   + [str(vrec.get("n_views", "")),
                      flat(" ‖ ".join(vrec.get("views") or [])),
                      f"{jmed:.3f}" if isinstance(jmed, (int, float)) else "",
                      str(drec.get("n_identical_pairs", "")),
                      "COLLAPSE" if (isinstance(jmed, (int, float)) and jmed > 0.8)
                      or (drec.get("n_identical_pairs") or 0) >= 3 else ""])
            fout.write("\t".join(row) + "\n")
            n_rows += 1
            if args.limit and n_rows >= args.limit:
                break
    print(f"[done] {n_rows} rows (from {len(recaps)} clips with recaps; "
          f"{n_tasks_seen} tasks scanned) -> {args.output}")
    print("列: video | 分桶理由 | 源caption(en) | 翻译源caption(zh) | 5 条重述 | "
          "A′句切视图 | 多样性指标")


if __name__ == "__main__":
    main()
