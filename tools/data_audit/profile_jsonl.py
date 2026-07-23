#!/usr/bin/env python3
"""Stage-0 heuristic profiler for the aligned InternVid zh-translation JSONL.

Streams the full 10M-line file (no GPU, no model, stdlib only) and flags
per-record problems that need no vision:

  * schema / parse errors, missing or empty fields
  * zh field not actually Chinese (untranslated / language mix / mojibake)
  * translation artifacts: instruction leakage ("以下是翻译…"), model refusals,
    leaked <think> blocks, wrapping quotes, "Translation:" prefixes
  * length-ratio outliers between en source and zh translation
    (truncation on one side, hallucinated expansion on the other)
  * intra-caption n-gram repetition loops
  * exact-duplicate zh captions across clips (generic caption detector)
  * clip duration stats, from start/end timestamp fields when present
    (falls back to parsing `<yid>_<start>_<end>.mp4` clip names)

Outputs a summary JSON (global stats + flag rates + top duplicated captions)
and a flags JSONL with one line per *flagged* record for downstream joins.

Example:
    python tools/data_audit/profile_jsonl.py \
        --input .../aligned_internvid_zh_translated_retry_merged.jsonl \
        --out-summary outputs/audit/s0_summary.json \
        --out-flags outputs/audit/s0_flags.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

# ---------------------------------------------------------------- key guessing

VIDEO_KEYS = ["video", "video_path", "video_id", "clip", "clip_name", "filename"]
ZH_KEYS = ["caption_zh", "zh_caption", "caption_cn", "cn_caption", "translated_caption",
           "translation", "zh", "chinese_caption", "raw_caption_zh_translated",
           "caption_zh_translated"]
EN_KEYS = ["caption", "caption_en", "en_caption", "source_caption", "src_caption", "en"]

CLIP_NAME_RE = re.compile(
    r"^(?P<yid>[A-Za-z0-9_-]{11})_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)\.mp4$"
)

START_TS_KEYS = ["start_timestamp", "start_time", "start_sec", "start"]
END_TS_KEYS = ["end_timestamp", "end_time", "end_sec", "end"]
TS_HMS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:[.,]\d+)?)$")


def parse_ts(v):
    """Timestamp -> seconds; accepts float/int, '12.5', 'MM:SS(.ms)', 'HH:MM:SS(.ms)'."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    m = TS_HMS_RE.match(s)
    if m:
        h = int(m.group(1) or 0)
        return h * 3600 + int(m.group(2)) * 60 + float(m.group(3).replace(",", "."))
    try:
        return float(s)
    except ValueError:
        return None

# ------------------------------------------------------------- text heuristics

ARTIFACT_PATTERNS = [
    # Anchored at caption start; bare "这是/以下是/Here is" openers are normal
    # captions, so the Chinese ones must mention 翻译/译文 nearby to count.
    ("instruction_leak", re.compile(
        r"^\s*((以下|这里?)是.{0,10}(翻译|译文)|好的[，,].{0,8}(翻译|译文|以下)|"
        r"翻译[：:如]|译文[：:]|中文翻译[：:]|将.{0,12}翻译|"
        r"Translation[:：]|Here('s| is) (the |a |your )?translat|The translation)", re.I)),
    # A refusal needs the refusal construction, not just an apology word:
    # captions legitimately transcribe on-screen "抱歉/sorry" (e.g. show titles).
    ("refusal", re.compile(
        r"(抱歉|对不起|很遗憾)[，,]?\s*(我|这边)?(无法|不能|没法|做不到)|"
        r"无法(为您?)?(翻译|提供|处理)|不能翻译|作为(一个)?(AI|人工智能|语言模型)|"
        r"I('m| am) sorry[^.!?]{0,20}(cannot|can't|unable)|"
        r"I can(no|')t (translate|assist|help|fulfill)|"
        r"unable to (translate|assist)|As an AI", re.I)),
    # CoT leaks sit at the caption start; "首先，我们…" is normal tutorial
    # narration, so only first-person singular counts.
    ("think_leak", re.compile(
        r"</?think>|^\s*让我(先|来)?(想|分析|看看)|^\s*首先[，,]?我(?!们)", re.I)),
    ("meta_note", re.compile(r"[（(]?(注|注意|说明|备注)[：:）)]")),
    ("html_or_escape", re.compile(r"&\w{2,6};|\\u[0-9a-fA-F]{4}|<[a-zA-Z/][^>]*>")),
]

# Pipeline artifacts are *additions* relative to the en source. If the source
# itself contains the trigger's counterpart, the zh hit is just a faithful
# translation of on-screen text / speech — skip the flag.
ARTIFACT_EN_EXEMPT = {
    "refusal": re.compile(r"sorry|pardon|apolog|excuse|can(no|')?t|unable", re.I),
    "think_leak": re.compile(r"\bfirst\b|let me|let's|\bthink", re.I),
    "meta_note": re.compile(r"\bnote|caution|warning|attention|notice|remark", re.I),
    "instruction_leak": re.compile(
        r"^\s*(here('s| is)|this is|the following|translation\b|sure\b|okay?\b)", re.I),
}

WRAPPING_QUOTES = [("“", "”"), ("\"", "\""), ("'", "'"), ("「", "」"), ("《", "》")]


def cjk_count(text):
    return sum(1 for ch in text if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿")


def ascii_alpha_count(text):
    return sum(1 for ch in text if ch.isascii() and ch.isalpha())


def max_ngram_repeat_ratio(text, n=4):
    """Fraction of the string covered by the single most repeated char n-gram."""
    if len(text) < 2 * n:
        return 0.0
    grams = Counter(text[i:i + n] for i in range(len(text) - n + 1))
    gram, cnt = grams.most_common(1)[0]
    if cnt < 3:
        return 0.0
    return min(1.0, cnt * n / len(text))


def normalize_for_dup(text):
    return re.sub(r"[\s，。,.！!？?、：:；;\"'“”]+", "", text).lower()


LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")


def strip_shared_latin(zh, en):
    """Drop Latin/digit runs that appear verbatim in the English source.

    Keeping a brand or model number in Latin script is *correct* translation
    behaviour -- "asus p8z77-v bm motherboard" -> "华硕 P8Z77-V BM 主板" even
    translates the brand (ASUS -> 华硕) -- but such a caption has a mechanically
    low CJK ratio, so the ratio flags below fire on it. Measuring the ratio on
    the remainder separates "kept a proper noun" from "left it untranslated".

    Evidence: the 15-record conflict sheet (DATA_ISSUES.md problem 12) was
    15/15 correct translations, every one flagged only because the caption was
    proper-noun dominated (YouTube / CNBC / Windows 10 / TEDx / P8Z77-V ...).
    This function clears 14 of those 15. The holdout is "CNBC 采访 Chamath
    Palathia" whose English source misspells the name as "chamat", so the match
    is not verbatim. Matching is kept exact on purpose: a Latin run absent from
    the source really is a zh/en discrepancy worth surfacing, even when -- as
    here -- the zh side is the correct one.

    Separator-insensitive match: a version string is written differently on the
    two sides -- zh "v3.1.3.1" vs source "v3 1 3 1" -- so a verbatim substring
    test misses it and the caption still trips the ratio flags. We therefore
    also test the run against `en` with separators ([.\\-_ ] and whitespace)
    removed, requiring >=2 alnum chars so single digits do not match by
    accident. This is purely additive (a run is stripped if EITHER test passes),
    so it can only lower the flag count, never raise it -- the two-sided
    acceptance below still holds.

    KNOWN LIMIT: this cannot tell a retained proper noun from a retained common
    word, since both appear in `en`. A caption that leaves "man is talking"
    untranslated is no longer flagged by the ratio rules. `zh_no_chinese` and
    `zh_equals_en` are deliberately still measured on the raw string, so wholly
    untranslated records are still caught; only partial mixing can slip. The
    acceptance check for this function is therefore two-sided -- see
    DATA_ISSUES.md problem 12.
    """
    if not en:
        return zh
    en_low = en.lower()
    en_dense = re.sub(r"[\s._\-]+", "", en_low)
    out, pos = [], 0
    for m in LATIN_RUN.finditer(zh):
        run = m.group(0).lower()
        run_dense = re.sub(r"[\s._\-]+", "", run)
        if run in en_low or (len(run_dense) >= 2 and run_dense in en_dense):
            out.append(zh[pos:m.start()])
            pos = m.end()
    out.append(zh[pos:])
    return "".join(out)


def check_record(zh, en, flags):
    zh_s, en_s = zh.strip(), (en or "").strip()
    if not zh_s:
        flags.append("zh_empty")
        return
    n_cjk = cjk_count(zh_s)
    if n_cjk == 0:
        flags.append("zh_no_chinese")          # untranslated or garbage
    # The two ratio flags are measured on the remainder after removing Latin
    # runs the English source also contains, so that correctly-retained proper
    # nouns do not read as contamination (strip_shared_latin).
    core = strip_shared_latin(zh_s, en_s)
    n_cjk_core = cjk_count(core)
    n_alpha_core = ascii_alpha_count(core)
    total_core = max(1, len(core.replace(" ", "")))
    if n_cjk and n_cjk_core / total_core < 0.4:
        flags.append("zh_low_cjk_ratio")       # heavy language mixing
    if n_alpha_core > 0 and n_alpha_core / total_core > 0.5:
        flags.append("zh_mostly_english")
    if en_s and normalize_for_dup(zh_s) == normalize_for_dup(en_s):
        flags.append("zh_equals_en")

    for name, pat in ARTIFACT_PATTERNS:
        if not pat.search(zh_s):
            continue
        exempt = ARTIFACT_EN_EXEMPT.get(name)
        if exempt is not None and en_s and exempt.search(en_s):
            continue
        flags.append(name)
    for lq, rq in WRAPPING_QUOTES:
        if zh_s.startswith(lq) and zh_s.endswith(rq) and len(zh_s) > 2:
            flags.append("wrapping_quotes")
            break
    if "\n" in zh_s:
        flags.append("multiline")

    if max_ngram_repeat_ratio(zh_s) > 0.5:
        flags.append("repetition_loop")

    if en_s:
        en_words = max(1, len(en_s.split()))
        ratio = n_cjk / en_words if n_cjk else 0.0
        # a normal en->zh translation lands around 1.2-2.5 zh chars per en word
        if n_cjk and ratio < 0.5:
            flags.append("zh_too_short_vs_en")   # likely truncated translation
        elif ratio > 6.0:
            flags.append("zh_too_long_vs_en")    # likely hallucinated expansion
    if n_cjk and n_cjk < 6:
        flags.append("zh_very_short")            # coarse / low-information caption


def pctl(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(p / 100.0 * len(sorted_vals)))
    return sorted_vals[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="input jsonl")
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-flags", required=True,
                    help="jsonl of flagged records: {video, flags, zh, en}")
    ap.add_argument("--video-key", default=None)
    ap.add_argument("--zh-key", default=None)
    ap.add_argument("--en-key", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only scan first N lines (smoke test)")
    ap.add_argument("--dup-top-k", type=int, default=50)
    args = ap.parse_args()

    vkey, zkey, ekey = args.video_key, args.zh_key, args.en_key
    n_total = n_parse_err = 0
    flag_counter = Counter()
    zh_lens, en_lens, durations = [], [], []
    dup_hash = Counter()
    dup_example = {}
    video_seen = Counter()
    dur_from_ts = dur_from_name = no_duration_info = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out_flags)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_summary)) or ".", exist_ok=True)
    fout = open(args.out_flags, "w", encoding="utf-8")

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if args.limit and n_total >= args.limit:
                break
            n_total += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_parse_err += 1
                flag_counter["parse_error"] += 1
                continue

            if vkey is None:
                vkey = next((k for k in VIDEO_KEYS if k in rec), None)
                zkey = zkey or next((k for k in ZH_KEYS if k in rec), None)
                ekey = ekey or next((k for k in EN_KEYS if k in rec and k != zkey), None)
                print(f"[keys] video={vkey} zh={zkey} en={ekey} "
                      f"(record keys: {sorted(rec.keys())})", file=sys.stderr)
                if not (vkey and zkey):
                    sys.exit("cannot auto-detect video/zh keys; pass --video-key/--zh-key")

            video = str(rec.get(vkey, ""))
            zh = str(rec.get(zkey, "") or "")
            en = str(rec.get(ekey, "") or "") if ekey else ""

            flags = []
            if not video:
                flags.append("video_missing")
            else:
                video_seen[video] += 1
                st = et = None
                for k in START_TS_KEYS:
                    if k in rec:
                        st = parse_ts(rec[k]); break
                for k in END_TS_KEYS:
                    if k in rec:
                        et = parse_ts(rec[k]); break
                if st is not None and et is not None and et > st:
                    durations.append(et - st)
                    dur_from_ts += 1
                else:
                    m = CLIP_NAME_RE.match(os.path.basename(video))
                    if m:
                        durations.append(float(m.group("end")) - float(m.group("start")))
                        dur_from_name += 1
                    else:
                        no_duration_info += 1

            check_record(zh, en, flags)

            zh_lens.append(cjk_count(zh))
            if en:
                en_lens.append(len(en.split()))
            if zh.strip():
                h = hashlib.md5(normalize_for_dup(zh).encode()).digest()[:8]
                dup_hash[h] += 1
                if dup_hash[h] <= 1:
                    dup_example[h] = zh.strip()[:80]

            if flags:
                for fl in flags:
                    flag_counter[fl] += 1
                fout.write(json.dumps(
                    {"video": video, "flags": flags, "zh": zh, "en": en},
                    ensure_ascii=False) + "\n")

            if n_total % 1_000_000 == 0:
                print(f"[progress] {n_total:,} lines, {sum(flag_counter.values()):,} flags",
                      file=sys.stderr)
    fout.close()

    zh_lens.sort(); en_lens.sort(); durations.sort()
    dup_videos = sum(c - 1 for c in video_seen.values() if c > 1)
    top_dups = [
        {"caption": dup_example.get(h, "?"), "count": c}
        for h, c in dup_hash.most_common(args.dup_top_k) if c > 1
    ]
    n_dup_captions = sum(c - 1 for c in dup_hash.values() if c > 1)

    summary = {
        "input": args.input,
        "total_lines": n_total,
        "parse_errors": n_parse_err,
        "keys": {"video": vkey, "zh": zkey, "en": ekey},
        "flag_rates": {k: {"count": v, "rate": round(v / max(1, n_total), 6)}
                       for k, v in flag_counter.most_common()},
        "flagged_records": sum(1 for _ in open(args.out_flags, encoding="utf-8")),
        "zh_cjk_len_pctl": {p: pctl(zh_lens, p) for p in (1, 5, 25, 50, 75, 95, 99)},
        "en_word_len_pctl": {p: pctl(en_lens, p) for p in (1, 5, 25, 50, 75, 95, 99)},
        "clip_duration_pctl": {p: pctl(durations, p) for p in (1, 5, 25, 50, 75, 95, 99)},
        "duration_source": {"timestamps": dur_from_ts, "clip_name": dur_from_name,
                            "none": no_duration_info},
        "duplicate_video_entries": dup_videos,
        "duplicate_zh_captions": n_dup_captions,
        "duplicate_zh_caption_rate": round(n_dup_captions / max(1, n_total), 6),
        "top_duplicated_captions": top_dups,
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("top_duplicated_captions",)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
