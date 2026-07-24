#!/usr/bin/env python3
"""A' sentence-split views: dense zh recaptions -> <=32-token training views.

Implements the deterministic A' rule from notes/recaption-master-plan.md D5:
split on sentence-final punctuation, re-split overlong sentences on commas
(greedy packing), measure length with the actual training tokenizer
(hfl/minirbt-h288, counted WITH special tokens against max_txt_l=32), drop
fragments that still exceed the budget, drop empty/punct-only/too-short views.

Input  : recap5 shard glob (records {"video":..., "verdict":{"recaps":[...]}})
Output : jsonl {"video", "views", "n_recaps", "n_views", "n_dropped_overlong"}

If the tokenizer cannot be loaded (offline box), falls back to a conservative
character budget (--char-budget, default 30 chars) with a stderr warning.

Example:
    python tools/data_audit/split_views.py \
        --recaps 'outputs/audit_10k/recap5.*.jsonl' \
        --output outputs/audit_10k/recap_views.jsonl
"""

import argparse
import glob
import json
import re
import sys

PRIMARY_SPLIT = re.compile(r"(?<=[。；！？!?;\n])")
SECONDARY_SPLIT = re.compile(r"(?<=[，、,])")
# a view must contain at least one CJK char or letter (not punctuation only)
HAS_CONTENT = re.compile(r"[一-鿿A-Za-z0-9]")


def iglob_lines(pattern):
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def make_fits(tokenizer_name, max_tokens, char_budget):
    """Return fits(text)->bool. Tokenizer counted WITH special tokens ([CLS]/[SEP])
    to match the training-side max_txt_l truncation."""
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # noqa: deferred heavy import
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            print(f"[tokenizer] {tokenizer_name} loaded; budget = {max_tokens} tokens "
                  f"(incl. special tokens)", file=sys.stderr)
            return lambda t: len(tok(t)["input_ids"]) <= max_tokens
        except Exception as e:  # offline / missing transformers -> char fallback
            print(f"[tokenizer] load failed ({e}); falling back to char budget "
                  f"{char_budget}", file=sys.stderr)
    else:
        print(f"[tokenizer] disabled; char budget {char_budget}", file=sys.stderr)
    return lambda t: len(t) <= char_budget


def split_dense(text, fits, min_chars=4):
    """One dense caption -> list of fitting views. Deterministic, no generation.

    Returns (views, n_dropped_overlong)."""
    views, dropped = [], 0
    if not isinstance(text, str) or not text.strip():
        return views, dropped
    for sent in PRIMARY_SPLIT.split(text):
        sent = sent.strip()
        if not sent or not HAS_CONTENT.search(sent):
            continue
        if fits(sent):
            views.append(sent)
            continue
        # overlong sentence: greedy-pack comma clauses into fitting chunks
        chunk = ""
        for clause in SECONDARY_SPLIT.split(sent):
            cand = chunk + clause
            if fits(cand):
                chunk = cand
            else:
                if chunk and HAS_CONTENT.search(chunk):
                    views.append(chunk.strip())
                if fits(clause):
                    chunk = clause
                else:  # single clause exceeds budget -> drop (D5)
                    dropped += 1
                    chunk = ""
        if chunk and HAS_CONTENT.search(chunk):
            views.append(chunk.strip())
    # final filter: too-short fragments; order-preserving dedupe
    seen, out = set(), []
    for v in views:
        v = v.strip("，、,；;")
        if len(v) < min_chars or not HAS_CONTENT.search(v):
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recaps", required=True,
                    help="recap5 shard glob, e.g. 'outputs/audit_10k/recap5.*.jsonl'")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", default="hfl/minirbt-h288",
                    help="HF tokenizer for the <=32-token budget (training tokenizer); "
                         "'' disables and uses --char-budget")
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="token budget incl. special tokens (= training max_txt_l)")
    ap.add_argument("--char-budget", type=int, default=30,
                    help="fallback character budget when tokenizer unavailable")
    ap.add_argument("--min-chars", type=int, default=4,
                    help="drop views shorter than this many chars (matches "
                         "apply_buckets --min-better-zh-len)")
    args = ap.parse_args()

    fits = make_fits(args.tokenizer or None, args.max_tokens, args.char_budget)

    n_clips = n_views_total = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for line in iglob_lines(args.recaps):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            recaps = (rec.get("verdict") or {}).get("recaps") or []
            pooled, dropped = [], 0
            for cap in recaps:
                vs, dr = split_dense(cap, fits, args.min_chars)
                pooled.extend(vs)
                dropped += dr
            seen, views = set(), []
            for v in pooled:
                if v not in seen:
                    seen.add(v)
                    views.append(v)
            fout.write(json.dumps({
                "video": rec.get("video"), "views": views,
                "n_recaps": len(recaps), "n_views": len(views),
                "n_dropped_overlong": dropped,
            }, ensure_ascii=False) + "\n")
            n_clips += 1
            n_views_total += len(views)
    print(f"[done] {n_clips} clips -> {n_views_total} views "
          f"({n_views_total / max(1, n_clips):.1f}/clip) -> {args.output}")


if __name__ == "__main__":
    main()
