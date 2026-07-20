#!/usr/bin/env python3
"""Stage-3 aggregation: calibrate cheap scores against VLM verdicts, then
bucket the FULL corpus into repair actions and emit a markdown report.

Two things happen here:

1. Calibration (needs s2 verdicts): for each Stage-1 score decile, report the
   VLM-measured bad-rate (zh_score<=2 / trans bad / en bad), and pick the
   sig_zh_mean threshold that hits --target-precision for "misaligned".
   These per-decile rates, reweighted by decile population, also give the
   *extrapolated* problem rates for the whole 10M corpus.

2. Full-corpus bucketing (streams s1 scores + s0 flags over all 10M):
       DROP        video decode error / refusal / empty zh
       FIX_TEXT    only cleanable artifacts (quotes, prefixes, multiline)
       RETRANSLATE zh broken but en fine: xling_cos low or zh-artifact flags,
                   while sig_en_mean is above the misalignment threshold
       RECAPTION   sig_en_mean below threshold (source caption misaligned)
                   or sig_zh_mean below threshold without a translation cause
       KEEP        everything else

Example:
    python tools/data_audit/aggregate_report.py \
        --scores 'outputs/audit/s1_scores.*.jsonl' \
        --flags outputs/audit/s0_flags.jsonl \
        --verdicts outputs/audit/s2_verdicts.jsonl \
        --out-buckets outputs/audit/buckets.jsonl \
        --out-report outputs/audit/report.md
"""

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict

FIX_ONLY_FLAGS = {"wrapping_quotes", "multiline", "instruction_leak", "meta_note",
                  "html_or_escape"}
DROP_FLAGS = {"zh_empty", "refusal", "zh_no_chinese", "repetition_loop"}
RETRANSLATE_FLAGS = {"zh_mostly_english", "zh_equals_en", "zh_low_cjk_ratio",
                     "think_leak", "zh_too_short_vs_en", "zh_too_long_vs_en"}


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


def calibrate(verdict_path, target_precision):
    """Per-sig-decile VLM bad rates + threshold selection."""
    rows = []
    action_counter, issue_counter = Counter(), Counter()
    gran_counter = Counter()
    verdict_files = sorted(glob.glob(verdict_path))
    if not verdict_files:
        print(f"[warn] no verdict files match {verdict_path}; "
              "skipping calibration (bucketing on flags/thresholds only)",
              file=sys.stderr)
        return None
    for rec in iter_jsonl(verdict_files):
        v = rec.get("verdict") or {}
        if "error" in v and "action" not in v:
            continue
        s1 = rec.get("s1") or {}
        rows.append({
            "sig_zh": s1.get("sig_zh_mean"),
            "sig_en": s1.get("sig_en_mean"),
            "xling": s1.get("xling_cos"),
            "zh_bad": (v.get("zh_score") is not None and v["zh_score"] <= 2),
            "en_bad": (v.get("en_score") is not None and v["en_score"] <= 2),
            "trans_bad": (v.get("trans_score") is not None and v["trans_score"] <= 2),
            "coarse": v.get("granularity") == "coarse",
            "temporal_wrong": v.get("temporal_order_correct") is False,
        })
        action_counter[v.get("action", "?")] += 1
        gran_counter[v.get("granularity", "?")] += 1
        for k in ("en_issues", "trans_issues"):
            for it in v.get(k) or []:
                issue_counter[f"{k}:{it}"] += 1
    if not rows:
        return None

    scored = sorted((r for r in rows if r["sig_zh"] is not None),
                    key=lambda r: r["sig_zh"])
    deciles = []
    n = len(scored)
    for d in range(10):
        chunk = scored[d * n // 10:(d + 1) * n // 10]
        if not chunk:
            continue
        deciles.append({
            "decile": d,
            "sig_zh_range": [chunk[0]["sig_zh"], chunk[-1]["sig_zh"]],
            "n": len(chunk),
            "zh_bad_rate": round(sum(r["zh_bad"] for r in chunk) / len(chunk), 3),
            "en_bad_rate": round(sum(r["en_bad"] for r in chunk) / len(chunk), 3),
            "trans_bad_rate": round(sum(r["trans_bad"] for r in chunk) / len(chunk), 3),
            "coarse_rate": round(sum(r["coarse"] for r in chunk) / len(chunk), 3),
        })
    # pick largest threshold t such that precision(zh_bad | sig_zh < t) >= target
    threshold = None
    bad_below, total_below = 0, 0
    for r in scored:
        total_below += 1
        bad_below += r["zh_bad"]
        if total_below >= 50 and bad_below / total_below >= target_precision:
            threshold = r["sig_zh"]
    return {
        "n_verdicts": len(rows),
        "action_dist": dict(action_counter.most_common()),
        "granularity_dist": dict(gran_counter.most_common()),
        "top_issues": dict(issue_counter.most_common(20)),
        "temporal_wrong_rate": round(
            sum(r["temporal_wrong"] for r in rows) / len(rows), 4),
        "sig_zh_deciles": deciles,
        "suggested_sig_zh_threshold": threshold,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="stage-1 output glob")
    ap.add_argument("--flags", default=None, help="stage-0 flags jsonl")
    ap.add_argument("--verdicts", default=None,
                    help="stage-2 verdicts jsonl (glob ok, e.g. 's2_verdicts.*.jsonl')")
    ap.add_argument("--target-precision", type=float, default=0.8)
    ap.add_argument("--sig-zh-threshold", type=float, default=None,
                    help="override calibrated misalignment threshold")
    ap.add_argument("--sig-en-threshold", type=float, default=None,
                    help="defaults to the zh threshold")
    ap.add_argument("--xling-threshold", type=float, default=0.7)
    ap.add_argument("--out-buckets", required=True)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()

    score_files = sorted(glob.glob(args.scores))
    if not score_files:
        sys.exit(f"no stage-1 score files match {args.scores} — run S1 first")

    calib = calibrate(args.verdicts, args.target_precision) if args.verdicts else None
    t_zh = args.sig_zh_threshold or (calib or {}).get("suggested_sig_zh_threshold")
    if t_zh is None:
        print("[warn] no calibrated threshold; bucketing on flags only "
              "(pass --sig-zh-threshold to enable score buckets)", file=sys.stderr)
    t_en = args.sig_en_threshold or t_zh

    flags_by_video = defaultdict(list)
    if args.flags:
        for rec in iter_jsonl([args.flags]):
            flags_by_video[rec["video"]] = rec["flags"]

    bucket_counter = Counter()
    n = 0
    with open(args.out_buckets, "w", encoding="utf-8") as fout:
        for rec in iter_jsonl(score_files):
            v = rec.get("video")
            if not v:
                bucket_counter["DROP"] += 1
                continue
            n += 1
            fl = set(flags_by_video.get(v, []))
            reasons = []
            if rec.get("error") or fl & DROP_FLAGS:
                bucket = "DROP"
                reasons = ([rec["error"]] if rec.get("error") else []) + \
                          sorted(fl & DROP_FLAGS)
            elif t_en is not None and rec.get("sig_en_mean") is not None \
                    and rec["sig_en_mean"] < t_en:
                bucket = "RECAPTION"
                reasons = [f"sig_en_mean={rec['sig_en_mean']}<{t_en}"]
            elif (fl & RETRANSLATE_FLAGS) or \
                    (rec.get("xling_cos") is not None
                     and rec["xling_cos"] < args.xling_threshold):
                bucket = "RETRANSLATE"
                reasons = sorted(fl & RETRANSLATE_FLAGS)
                if rec.get("xling_cos") is not None \
                        and rec["xling_cos"] < args.xling_threshold:
                    reasons.append(f"xling_cos={rec['xling_cos']}")
            elif t_zh is not None and rec.get("sig_zh_mean") is not None \
                    and rec["sig_zh_mean"] < t_zh:
                bucket = "RECAPTION"
                reasons = [f"sig_zh_mean={rec['sig_zh_mean']}<{t_zh}"]
            elif fl & FIX_ONLY_FLAGS:
                bucket = "FIX_TEXT"
                reasons = sorted(fl & FIX_ONLY_FLAGS)
            else:
                bucket = "KEEP"
            bucket_counter[bucket] += 1
            fout.write(json.dumps({"video": v, "bucket": bucket, "reasons": reasons},
                                  ensure_ascii=False) + "\n")

    lines = ["# InternVid-zh v0.1 数据审核报告", ""]
    lines.append(f"- 打分记录数: **{n:,}**")
    lines.append(f"- 阈值: sig_zh<{t_zh} / sig_en<{t_en} → RECAPTION; "
                 f"xling_cos<{args.xling_threshold} → RETRANSLATE\n")
    lines.append("## 全量分桶（修复动作）\n")
    lines.append("| bucket | count | rate |")
    lines.append("|---|---|---|")
    for b, c in bucket_counter.most_common():
        lines.append(f"| {b} | {c:,} | {c / max(1, n):.2%} |")
    if calib:
        lines.append("\n## VLM 审核样本（Stage-2）校准\n")
        lines.append(f"- 有效判定数: {calib['n_verdicts']:,}")
        lines.append(f"- 建议 sig_zh 阈值 (precision≥{args.target_precision}): "
                     f"{calib['suggested_sig_zh_threshold']}")
        lines.append(f"- 时序顺序错误率: {calib['temporal_wrong_rate']:.2%}")
        lines.append(f"- VLM 建议动作分布: `{json.dumps(calib['action_dist'], ensure_ascii=False)}`")
        lines.append(f"- 粒度分布: `{json.dumps(calib['granularity_dist'], ensure_ascii=False)}`")
        lines.append(f"- 高频问题: `{json.dumps(calib['top_issues'], ensure_ascii=False)}`\n")
        lines.append("| sig_zh 分位 | 区间 | n | zh差率 | en差率 | 翻译差率 | 粒度粗率 |")
        lines.append("|---|---|---|---|---|---|---|")
        for d in calib["sig_zh_deciles"]:
            lines.append(f"| q{d['decile']} | [{d['sig_zh_range'][0]:.3f},"
                         f" {d['sig_zh_range'][1]:.3f}] | {d['n']} | {d['zh_bad_rate']:.1%}"
                         f" | {d['en_bad_rate']:.1%} | {d['trans_bad_rate']:.1%}"
                         f" | {d['coarse_rate']:.1%} |")
        lines.append("\n> 按各分位人群占比（全量各 10%）加权，即得全量问题率外推。")
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
