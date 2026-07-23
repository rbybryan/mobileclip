#!/usr/bin/env python3
"""Stage-4 materializer: turn audit buckets into the v0.2 dataset + repair queues.

Reads the original 10M jsonl and the bucket assignment from
aggregate_report.py, then writes:

  --out-data        v0.2 training jsonl (KEEP as-is, FIX_TEXT cleaned by rule,
                    DROP removed). Schema is loader-compatible:
                      {"video", "caption": <zh>, "captions": [<zh>, <better_zh>...]}
                    `captions` extras come from --verdicts (the VLM's better_zh
                    repair captions), only for videos it actually judged.
  --out-retranslate task queue jsonl {"video","en","zh_old","reasons"} for the
                    RETRANSLATE bucket (feed to the translation pipeline v0_2)
  --out-recaption   task queue jsonl for the RECAPTION bucket (feed to the
                    Qwen3-VL direct-zh captioning run)
  --out-drop        dropped records with reasons (for re-fetch / bookkeeping)

Rule cleaning applied to FIX_TEXT (and harmlessly to KEEP): strip wrapping
quotes, leading instruction prefixes ("以下是翻译：..."), trailing meta notes,
HTML escapes, collapse whitespace/newlines.

Example:
    python tools/data_audit/apply_buckets.py \
        --input $DATA --buckets $OUT/buckets.jsonl --verdicts $OUT/s2_verdicts.jsonl \
        --out-data $OUT/v0_2/internvid_zh_v0_2.jsonl \
        --out-retranslate $OUT/v0_2/retranslate_tasks.jsonl \
        --out-recaption $OUT/v0_2/recaption_tasks.jsonl \
        --out-drop $OUT/v0_2/dropped.jsonl
"""

import argparse
import html
import json
import os
import re
import sys
from collections import Counter

VIDEO_KEYS = ["video", "video_path", "video_id", "clip", "clip_name", "filename"]
ZH_KEYS = ["caption_zh", "zh_caption", "caption_cn", "cn_caption", "translated_caption",
           "translation", "zh", "chinese_caption", "raw_caption_zh_translated",
           "caption_zh_translated"]
EN_KEYS = ["caption", "caption_en", "en_caption", "source_caption", "src_caption", "en"]

PREFIX_RE = re.compile(
    r"^\s*(以下是|这是|好的[，,]?|翻译[：:]|译文[：:]|中文翻译[：:]|"
    r"Translation[:：]|Here is the translation[:：]?)\s*[：:]?\s*")
META_RE = re.compile(r"[（(](注|注意|说明|备注)[：:][^）)]*[）)]\s*$")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
WRAPPING_QUOTES = [("“", "”"), ("\"", "\""), ("'", "'"), ("「", "」"), ("《", "》")]


def iglob_lines(pattern):
    import glob as _g
    for fp in sorted(_g.glob(pattern)):
        with open(fp, encoding="utf-8", errors="replace") as fh:
            yield from fh


def clean_zh(text):
    t = html.unescape(text).strip()
    t = THINK_RE.sub("", t)
    for _ in range(3):  # prefixes/quotes can nest ("好的，以下是翻译：“...”")
        t = PREFIX_RE.sub("", t).strip()
        for lq, rq in WRAPPING_QUOTES:
            if t.startswith(lq) and t.endswith(rq) and len(t) > 2:
                t = t[1:-1].strip()
    t = META_RE.sub("", t)
    t = re.sub(r"\s*\n+\s*", "，", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="original v0.1 jsonl")
    ap.add_argument("--buckets", required=True, help="buckets.jsonl from aggregate_report.py")
    ap.add_argument("--verdicts", default=None,
                    help="s2_verdicts.jsonl; better_zh becomes extra captions")
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-retranslate", required=True)
    ap.add_argument("--out-recaption", required=True)
    ap.add_argument("--out-drop", required=True)
    ap.add_argument("--min-better-zh-len", type=int, default=4,
                    help="discard better_zh candidates shorter than this many chars")
    ap.add_argument("--merged", default=None,
                    help="s2_merged.*.jsonl（--mode merge 产出）；有则用合并后的密字幕做主"
                         " caption，三条候选仍作为额外正样本挂在 captions 里")
    args = ap.parse_args()

    buckets = {}
    for line in open(args.buckets, encoding="utf-8"):
        try:
            r = json.loads(line)
            buckets[r["video"]] = (r["bucket"], r.get("reasons", []))
        except (json.JSONDecodeError, KeyError):
            continue
    print(f"[buckets] loaded {len(buckets):,} assignments", file=sys.stderr)

    better = {}
    if args.verdicts:
        for line in open(args.verdicts, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cands = [c.strip() for c in (r.get("verdict") or {}).get("better_zh") or []
                     if isinstance(c, str) and len(c.strip()) >= args.min_better_zh_len]
            if cands:
                better[r["video"]] = cands[:3]
        print(f"[verdicts] better_zh available for {len(better):,} videos", file=sys.stderr)

    merged = {}
    if args.merged:
        for line in iglob_lines(args.merged):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = (r.get("verdict") or {}).get("merged_zh")
            if isinstance(m, str) and len(m.strip()) >= args.min_better_zh_len:
                merged[r["video"]] = m.strip()
        print(f"[merged] merged_zh available for {len(merged):,} videos", file=sys.stderr)

    for p in (args.out_data, args.out_retranslate, args.out_recaption, args.out_drop):
        os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    f_data = open(args.out_data, "w", encoding="utf-8")
    f_ret = open(args.out_retranslate, "w", encoding="utf-8")
    f_rec = open(args.out_recaption, "w", encoding="utf-8")
    f_drop = open(args.out_drop, "w", encoding="utf-8")

    keys = None
    stats = Counter()
    with open(args.input, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_error"] += 1
                continue
            if keys is None:
                v = next((k for k in VIDEO_KEYS if k in rec), None)
                z = next((k for k in ZH_KEYS if k in rec), None)
                e = next((k for k in EN_KEYS if k in rec and k != z), None)
                if not (v and z):
                    sys.exit(f"cannot detect keys in {sorted(rec.keys())}")
                keys = (v, z, e)
                print(f"[keys] video={v} zh={z} en={e}", file=sys.stderr)
            vk, zk, ek = keys
            vid = rec.get(vk)
            zh = str(rec.get(zk, "") or "")
            en = str(rec.get(ek, "") or "") if ek else ""
            bucket, reasons = buckets.get(vid, ("KEEP", ["unscored"]))
            stats[bucket] += 1

            if bucket == "DROP":
                f_drop.write(json.dumps({"video": vid, "reasons": reasons},
                                        ensure_ascii=False) + "\n")
                continue
            if bucket == "RETRANSLATE":
                f_ret.write(json.dumps({"video": vid, "en": en, "zh_old": zh,
                                        "reasons": reasons}, ensure_ascii=False) + "\n")
                continue
            if bucket == "RECAPTION":
                f_rec.write(json.dumps({"video": vid, "en": en, "zh_old": zh,
                                        "reasons": reasons}, ensure_ascii=False) + "\n")
                continue
            # KEEP / FIX_TEXT -> v0.2 data
            zh_clean = clean_zh(zh)
            if not zh_clean:
                stats["cleaned_to_empty"] += 1
                f_drop.write(json.dumps({"video": vid, "reasons": ["cleaned_to_empty"]},
                                        ensure_ascii=False) + "\n")
                continue
            if bucket == "FIX_TEXT" and zh_clean != zh.strip():
                stats["fixed_by_rule"] += 1
            extras = better.get(vid, [])
            primary = merged.get(vid) or zh_clean         # 合并字幕当主，没有则退回原 zh
            if merged.get(vid):
                stats["merged_primary"] += 1
            # 主 caption + 去重后的候选（原 zh 也进正样本池）
            captions = [primary] + [c for c in [zh_clean, *extras] if c and c != primary]
            out = {"video": vid, "caption": primary}
            if len(captions) > 1:
                out["captions"] = captions
                stats["with_extra_captions"] += 1
            f_data.write(json.dumps(out, ensure_ascii=False) + "\n")

    for fh in (f_data, f_ret, f_rec, f_drop):
        fh.close()
    print(json.dumps(dict(stats.most_common()), ensure_ascii=False, indent=2))
    print(f"[done] v0.2 data -> {args.out_data}", file=sys.stderr)


if __name__ == "__main__":
    main()
