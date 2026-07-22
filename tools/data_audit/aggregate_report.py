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
RECOMMENDED_MIN_LIFT = 2.0

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


def load_trans_scores(trans_path):
    """video -> trans_score from the text-only re-judgement pass (--mode trans).

    The combined S2 rubric does not actually measure translation quality (one
    quality score gets copied into all three fields; DATA_ISSUES.md problem 8),
    so `trans_bad` taken from s2_verdicts is really "overall quality bad" and
    reads ~19.6% instead of the true 3.6%. When s2_trans.*.jsonl exists, use it.
    """
    out = {}
    for rec in iter_jsonl(sorted(glob.glob(trans_path))):
        v = rec.get("verdict") or {}
        if v.get("trans_score") is not None:
            out[rec.get("video")] = v["trans_score"]
    return out


def calibrate(verdict_path, target_precision, trans_scores=None):
    """Per-sig-decile VLM bad rates + threshold selection."""
    trans_scores = trans_scores or {}
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
        ts = trans_scores.get(rec.get("video"), v.get("trans_score"))
        rows.append({
            "sig_zh": s1.get("sig_zh_mean"),
            "sig_en": s1.get("sig_en_mean"),
            "xling": s1.get("xling_cos"),
            "zh_bad": (v.get("zh_score") is not None and v["zh_score"] <= 2),
            "en_bad": (v.get("en_score") is not None and v["en_score"] <= 2),
            "trans_bad": (ts is not None and ts <= 2),
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
    n_bad = sum(r["zh_bad"] for r in scored)
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
    # Sweep sig_zh ascending. `threshold` is the target-meeting pick; the sweep
    # also records what IS achievable, because "no threshold met the target" is
    # usually a statement about arithmetic rather than about sig_zh:
    #
    #     precision(selected) <= base_rate * lift(selected)
    #
    # With a zh_bad base rate of ~20% and a worst-decile lift of ~1.7x, no cut
    # can reach 0.8 -- the target is unreachable no matter how it is tuned, and
    # silently returning None makes RECAPTION structurally empty while the
    # report still prints an authoritative-looking "96% KEEP". So report the
    # ceiling and a coverage/precision curve, and let the caller pick a target
    # that exists. (Same trap as decision 1 / xling; see DATA_ISSUES.md.)
    threshold = None
    best_precision, best_at = 0.0, None
    curve, checkpoints = [], [0.05, 0.10, 0.20, 0.30, 0.50]
    ci = 0
    bad_below, total_below = 0, 0
    for r in scored:
        total_below += 1
        bad_below += r["zh_bad"]
        prec = bad_below / total_below
        if total_below >= 50:
            if prec >= target_precision:
                threshold = r["sig_zh"]
            if prec > best_precision:
                best_precision, best_at = prec, r["sig_zh"]
        while ci < len(checkpoints) and total_below >= checkpoints[ci] * n:
            curve.append({"coverage": checkpoints[ci], "n": total_below,
                          "threshold": r["sig_zh"], "precision": round(prec, 3),
                          "recall": round(bad_below / max(1, n_bad), 3)})
            ci += 1
    return {
        "n_verdicts": len(rows),
        "action_dist": dict(action_counter.most_common()),
        "granularity_dist": dict(gran_counter.most_common()),
        "top_issues": dict(issue_counter.most_common(20)),
        "temporal_wrong_rate": round(
            sum(r["temporal_wrong"] for r in rows) / len(rows), 4),
        "sig_zh_deciles": deciles,
        "suggested_sig_zh_threshold": threshold,
        "zh_bad_base_rate": round(n_bad / n, 4) if n else None,
        "best_achievable_precision": round(best_precision, 3),
        "best_precision_threshold": best_at,
        "precision_curve": curve,
        "target_precision": target_precision,
    }


def validate_proxy(flags_by_video, trans_scores, proxy_flags):
    """Measure the S0-flag proxy against the text-only labels, on their overlap.

    Lesson 3 in DATA_ISSUES.md §0 has now been learned three times -- xling
    (decision 1), flags-as-translation-judge (problem 11, precision 6.6%),
    flags-as-contamination-judge (problem 12, 15/15 false positives). Each time
    a proxy was allowed to assign buckets without anyone checking it against
    ground truth first. This makes that check a gate instead of a lesson.

    Returns None when there are no labels to validate against.
    """
    if not trans_scores:
        return None
    bad = {v for v, s in trans_scores.items() if s <= 2}
    hit = {v for v in trans_scores
           if set(flags_by_video.get(v, [])) & proxy_flags}
    base = len(bad) / len(trans_scores)
    tp = len(hit & bad)
    return {
        "n_labelled": len(trans_scores),
        "n_hit": len(hit),
        "base_rate": base,
        "precision": tp / len(hit) if hit else 0.0,
        "recall": tp / len(bad) if bad else 0.0,
        "lift": (tp / len(hit)) / base if hit and base else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="stage-1 output glob")
    ap.add_argument("--flags", default=None, help="stage-0 flags jsonl")
    ap.add_argument("--verdicts", default=None,
                    help="stage-2 verdicts jsonl (glob ok, e.g. 's2_verdicts.*.jsonl')")
    ap.add_argument("--trans-verdicts", default=None,
                    help="s2_trans.*.jsonl from the text-only re-judgement "
                         "(--mode trans); when given, the 翻译差率 column comes "
                         "from it instead of the combined rubric, which does not "
                         "actually measure translation (DATA_ISSUES.md problem 8)")
    ap.add_argument("--target-precision", type=float, default=0.8)
    ap.add_argument("--sig-zh-threshold", type=float, default=None,
                    help="override calibrated misalignment threshold")
    ap.add_argument("--sig-en-threshold", type=float, default=None,
                    help="defaults to the zh threshold")
    ap.add_argument("--xling-threshold", type=float, default=0.7)
    ap.add_argument("--proxy-min-lift", type=float, default=RECOMMENDED_MIN_LIFT,
                    help="S0 旗标要参与分桶，其对真标签的 lift 至少要到这个值"
                         "（lift = 精度/基准率）。达不到就退出分桶规则，只对"
                         "没有真标签的记录生效。设 0 可强行启用（不建议）")
    ap.add_argument("--out-buckets", required=True)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()

    score_files = sorted(glob.glob(args.scores))
    if not score_files:
        sys.exit(f"no stage-1 score files match {args.scores} — run S1 first")

    trans_scores = load_trans_scores(args.trans_verdicts) if args.trans_verdicts else {}
    if args.trans_verdicts:
        print(f"[info] 翻译差率 sourced from {len(trans_scores):,} text-only "
              f"re-judgements ({args.trans_verdicts})", file=sys.stderr)
    calib = calibrate(args.verdicts, args.target_precision,
                      trans_scores) if args.verdicts else None
    t_zh = args.sig_zh_threshold or (calib or {}).get("suggested_sig_zh_threshold")
    if t_zh is None:
        print("[warn] 没有校准出阈值 → RECAPTION 桶将为空、几乎所有记录落进 KEEP。"
              "\n        这几乎总是「目标精度算术上不可达」而非数据没问题——"
              "\n        看 report.md 的「⚠️ 阈值没校准出来」一节，那里给了可达上限"
              "\n        与覆盖率/精度曲线；据此设 --target-precision 或直接给"
              "\n        --sig-zh-threshold。", file=sys.stderr)
    t_en = args.sig_en_threshold or t_zh

    flags_by_video = defaultdict(list)
    if args.flags:
        for rec in iter_jsonl([args.flags]):
            flags_by_video[rec["video"]] = rec["flags"]

    # Gate: never let the flag proxy assign a bucket before it has been checked
    # against ground truth (see validate_proxy).
    proxy = validate_proxy(flags_by_video, trans_scores, RETRANSLATE_FLAGS)
    proxy_ok = True
    if proxy is None:
        if args.flags:
            print("[proxy] ⚠️ 没有 --trans-verdicts，S0 旗标无法验证就在分桶——"
                  "\n        这正是问题 11/12 的成因。先跑 s2-trans 再来。",
                  file=sys.stderr)
    else:
        proxy_ok = proxy["lift"] >= args.proxy_min_lift
        print(f"[proxy] S0 旗标 vs 真标签（{proxy['n_labelled']:,} 条有判定）："
              f"基准 {proxy['base_rate']:.1%} → 精度 {proxy['precision']:.1%}，"
              f"召回 {proxy['recall']:.1%}，lift {proxy['lift']:.2f}×",
              file=sys.stderr)
        if not proxy_ok:
            print(f"[proxy] ❌ lift < {args.proxy_min_lift} → **旗标退出分桶规则**；"
                  f"无真标签的记录改由 sig_* 决定。"
                  f"\n        要强行启用：--proxy-min-lift 0（不建议）",
                  file=sys.stderr)

    bucket_counter = Counter()
    n = 0
    n_trans_labeled = n_trans_proxy = n_flag_vs_label = 0
    with open(args.out_buckets, "w", encoding="utf-8") as fout:
        for rec in iter_jsonl(score_files):
            v = rec.get("video")
            if not v:
                bucket_counter["DROP"] += 1
                continue
            n += 1
            fl = set(flags_by_video.get(v, []))
            reasons = []
            # 真标签说"翻译没问题"、S0 旗标却说"该重翻"——这批正是
            # untranslated 0.3% vs S0 语言污染 3.3% 那个 10x 缺口的人群
            if trans_scores.get(v) is not None and trans_scores[v] > 2 \
                    and (fl & RETRANSLATE_FLAGS):
                n_flag_vs_label += 1
            if rec.get("error") or fl & DROP_FLAGS:
                bucket = "DROP"
                reasons = ([rec["error"]] if rec.get("error") else []) + \
                          sorted(fl & DROP_FLAGS)
            elif t_en is not None and rec.get("sig_en_mean") is not None \
                    and rec["sig_en_mean"] < t_en:
                bucket = "RECAPTION"
                reasons = [f"sig_en_mean={rec['sig_en_mean']}<{t_en}"]
            elif trans_scores.get(v) is not None and trans_scores[v] <= 2:
                # Ground truth beats proxy: for records the text-only pass
                # judged, route on the measured translation score instead of
                # the S0 flags. Those flags were measured at 6.6% precision /
                # 7.0% recall against exactly this label (xling-check), i.e.
                # the flag-built bucket is ~93% false positives AND misses ~93%
                # of real translation errors -- it is close to uncorrelated
                # with the thing it names.
                bucket = "RETRANSLATE"
                reasons = [f"trans_score={trans_scores[v]}(text-only)"]
                n_trans_labeled += 1
            elif proxy_ok and v not in trans_scores and \
                    ((fl & RETRANSLATE_FLAGS)
                     or (rec.get("xling_cos") is not None
                         and rec["xling_cos"] < args.xling_threshold)):
                # Only records with NO measured label fall back to the proxy.
                # A record the text-only pass scored >=2 has been *measured* to
                # translate fine; letting a 6.6%-precision flag overrule that
                # would be preferring the proxy to the ground truth it was
                # validated against. (This is what inflated RETRANSLATE from
                # 333 to 628 on the first labelled run.)
                if trans_scores:
                    n_trans_proxy += 1
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
    if proxy is not None:
        if not proxy_ok:
            verdict = f"❌ 未通过（lift < {args.proxy_min_lift}）→ 已退出分桶规则"
        elif proxy["lift"] < RECOMMENDED_MIN_LIFT:
            # Passing only because the caller lowered the bar -- do not let that
            # render as an endorsement of the proxy.
            verdict = (f"⚠️ 门槛被手动调低到 {args.proxy_min_lift}（建议 "
                       f"{RECOMMENDED_MIN_LIFT}）→ 强行启用。lift "
                       f"{proxy['lift']:.2f}× 意味着这个旗标几乎不带信息")
        else:
            verdict = "✅ 通过，允许参与分桶"
        lines.append("## 代理指标验证门（S0 旗标 vs 真标签）\n")
        lines.append(f"在 **{proxy['n_labelled']:,}** 条有文本-only 真标签的记录上实测：\n")
        lines.append(f"| 基准率 | 精度 | 召回 | lift |\n|---|---|---|---|")
        lines.append(f"| {proxy['base_rate']:.1%} | **{proxy['precision']:.1%}** | "
                     f"{proxy['recall']:.1%} | **{proxy['lift']:.2f}×** |\n")
        lines.append(f"**判定**：{verdict}\n")
        if not proxy_ok:
            lines.append("旗标只对**没有真标签**的记录起兜底作用，现已停用——"
                         "那些记录改由 `sig_*` 规则决定。**本报告的 RETRANSLATE 桶"
                         "因此只含真标签判进来的记录。**\n")
        lines.append("> 这道门是 DATA_ISSUES.md §0 教训 3 的代码化：代理指标"
                     "（`xling`、S0 旗标）先后三次被发现「不测量它命名的东西」，"
                     "所以它现在必须先对真标签验一次精度才能参与分桶。\n")
    if t_zh is None and calib:
        base = calib.get("zh_bad_base_rate") or 0
        best = calib.get("best_achievable_precision") or 0
        lines.append("## ⚠️ 阈值没校准出来 —— 下面的分桶表不能当结论\n")
        lines.append(f"没有任何 `sig_zh` 阈值达到 `--target-precision "
                     f"{calib.get('target_precision')}`，于是 **RECAPTION 桶为空**、"
                     "几乎所有记录落进 KEEP。**「96% KEEP」是阈值缺失的产物，不是数据结论。**\n")
        lines.append(f"- `zh_bad` 基准率 = **{base:.1%}**；"
                     f"实际可达的最高精度 = **{best:.1%}**"
                     f"（在 sig_zh < {calib.get('best_precision_threshold')}）\n")
        lines.append("**精度 ≤ 基准率 × lift**。基准率 "
                     f"{base:.1%} 时要 {calib.get('target_precision'):.0%} 精度，"
                     f"需要全程维持 ≈{(calib.get('target_precision') or 0) / max(base, 1e-9):.1f}× lift；"
                     "而 sig_zh 十分位表上最差/最好之比通常只有 2–3×。"
                     "**这是算术不可达，不是调阈值能救的**——与 `xling` 决策 1 同一个陷阱。\n")
        if calib.get("precision_curve"):
            lines.append("按覆盖率取阈值时实际能拿到什么（据此挑一个存在的目标）：\n")
            lines.append("| 覆盖率 | n | sig_zh 阈值 | 精度 | 召回 |")
            lines.append("|---|---|---|---|---|")
            for c in calib["precision_curve"]:
                lines.append(f"| {c['coverage']:.0%} | {c['n']:,} | {c['threshold']:.4f} "
                             f"| {c['precision']:.1%} | {c['recall']:.1%} |")
            lines.append("")
        lines.append("> **RECAPTION 的误判代价是不对称的**：把一条尚可的 caption 送去重写，"
                     "损失只是算力，拿回来的往往还更好；漏掉一条坏 caption 才是真损失。"
                     "所以这里**不该追求高精度**——按覆盖率（例如最差 20–30%）取阈值，"
                     "或直接 `--target-precision 0.4` 之类可达的值，比空桶有用得多。\n")
    lines.append("## 全量分桶（修复动作）\n")
    lines.append("| bucket | count | rate |")
    lines.append("|---|---|---|")
    for b, c in bucket_counter.most_common():
        lines.append(f"| {b} | {c:,} | {c / max(1, n):.2%} |")
    if t_zh is None:
        lines.append("\n> ⚠️ 见上节：`RECAPTION` 缺席是阈值没校准出来所致，**不代表源 caption 没问题**。")
    if bucket_counter.get("RETRANSLATE"):
        nr = bucket_counter["RETRANSLATE"]
        if n_trans_labeled:
            lines.append(f"\n> ✅ `RETRANSLATE` 里有 **{n_trans_labeled:,}** 条是按"
                         "**文本-only 实测 `trans_score≤2`** 判进来的（真标签）。")
        if n_trans_proxy or not n_trans_labeled:
            lines.append(f"\n> ⚠️ `RETRANSLATE` 里有 **{n_trans_proxy or nr:,}** 条只有 "
                         "**S0 旗标/xling 代理**、没有实测标签。该代理实测**精度 6.6%、"
                         "召回 7.0%**（`xling_calibration.md` §F）——**约 93% 是误伤，"
                         "且漏掉约 93% 的真实翻译错误**。这部分**不能当作翻译问题率**，"
                         "全量跑想要真标签就得把 `--mode trans` 跑到全量"
                         "（不解码视频，比主力档便宜得多）。")
    if n_flag_vs_label:
        lines.append(f"\n> 🔎 另有 **{n_flag_vs_label:,}** 条：**文本-only 实测说翻译没问题"
                     f"（`trans_score≥3`），但命中了 S0 的 RETRANSLATE 旗标**。"
                     "已按真标签处理（**不**进 RETRANSLATE）。这批人就是"
                     "「`untranslated` 仅 0.3% vs S0 语言污染 3.3%」那个 **10× 缺口**——"
                     "**已结案（问题 12）：缺口是 S0 误报，0.3% 才是真的**。"
                     "15 条抽检 15/15 判「译文没问题」，机制是比例旗标在惩罚"
                     "**正确保留的专有名词**（`Windows 10 开始菜单` 这类）。"
                     "`profile_jsonl.py` 已加 `strip_shared_latin()` 修正；"
                     "**若本行数字仍然很大，说明跑的是修复前的 `s0_flags.jsonl`，"
                     "重跑 `run_v100.sh s0` 即可**。"
                     "见 `notes/audit-10k-findings.md` 结论 14。")
    if calib:
        lines.append("\n## VLM 审核样本（Stage-2）校准\n")
        lines.append(f"- 有效判定数: {calib['n_verdicts']:,}")
        lines.append(f"- 建议 sig_zh 阈值 (precision≥{args.target_precision}): "
                     f"{calib['suggested_sig_zh_threshold']}")
        lines.append(f"- 时序顺序错误率: {calib['temporal_wrong_rate']:.2%}")
        lines.append(f"- VLM 建议动作分布: `{json.dumps(calib['action_dist'], ensure_ascii=False)}`")
        lines.append(f"- 粒度分布: `{json.dumps(calib['granularity_dist'], ensure_ascii=False)}`")
        lines.append(f"- 高频问题: `{json.dumps(calib['top_issues'], ensure_ascii=False)}`\n")
        lines.append(
            "> ⚠️ **上面两行（动作分布、高频问题）来自合并 rubric，读法受限**："
            "10k 实测显示模型只产生一个整体质量分、复制进三个字段，`action` 是该分数上的"
            "固定阶梯（`5→keep, 4→fix_text, 3→retranslate, ≤2→recaption/drop`）。"
            "所以 **`retranslate` 的占比不是翻译问题率**，`trans_issues:*` 的计数同理。"
            "详见 `tools/data_audit/DATA_ISSUES.md` 问题 8。\n")
        if trans_scores:
            lines.append(f"> ✅ 下表「翻译差率」一列已改用**文本-only 重判**"
                         f"（{len(trans_scores):,} 条，不看画面、先证据后打分），"
                         "是独立测量，可以直接引用。\n")
        else:
            lines.append("> ⚠️ 下表「翻译差率」一列仍来自合并 rubric，**实为整体质量分的复制品**。"
                         "跑 `run_v100.sh s2-trans` 后重跑本步即可换成独立测量。\n")
        lines.append("| sig_zh 分位 | 区间 | n | zh差率 | en差率 | 翻译差率 | 粒度粗率 |")
        lines.append("|---|---|---|---|---|---|---|")
        for d in calib["sig_zh_deciles"]:
            lines.append(f"| q{d['decile']} | [{d['sig_zh_range'][0]:.3f},"
                         f" {d['sig_zh_range'][1]:.3f}] | {d['n']} | {d['zh_bad_rate']:.1%}"
                         f" | {d['en_bad_rate']:.1%} | {d['trans_bad_rate']:.1%}"
                         f" | {d['coarse_rate']:.1%} |")
        lines.append("\n> 按各分位人群占比（全量各 10%）加权，即得全量问题率外推。")
        same = sum(1 for d in calib["sig_zh_deciles"]
                   if abs(d["zh_bad_rate"] - d["en_bad_rate"]) <= 0.02)
        if same >= 8:
            lines.append(
                "\n> ⚠️ **`zh差率` 与 `en差率` 两列几乎逐格相同——它们其实是同一列。**"
                "合并 rubric 下 `en_score` 与 `zh_score` 逐条同值 93.4%（问题 8），"
                "所以「中文与画面不符」和「英文与画面不符」在本轮**没有被区分开**。"
                "两列一致**不是**交叉验证，别当成两条独立证据。")
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
