#!/usr/bin/env python3
"""Decide whether `xling_cos` deserves a place in S3's bucketing rule.

Run this the moment S2 finishes, BEFORE `aggregate_report.py`.

Why this exists
---------------
`aggregate_report.py` routes a record to RETRANSLATE on low xling ALONE:

    elif (fl & RETRANSLATE_FLAGS) or (rec["xling_cos"] < args.xling_threshold):
        bucket = "RETRANSLATE"

`t_zh` / `t_en` in that script are *calibrated* against VLM verdicts, but
`--xling-threshold` is a hardcoded 0.7 that never is.  On the 10k sample that
default sits at p25 (xling p25 = 0.711), so it would sweep ~23% of the corpus
into the retranslate queue on a signal measured at rho ~= 0.04 against visual
alignment.  Worse, that branch is checked *before* the `sig_zh` RECAPTION
branch, so an over-broad threshold does not merely inflate RETRANSLATE -- it
steals records from RECAPTION, which is the more expensive, genuinely needed
repair.

What it computes
----------------
The evidence `report.md` does not produce (its deciles are cut on `sig_zh`,
not on `xling`):

  * VLM-measured `trans_score<=2` bad-rate per **xling decile** -> is the
    curve monotone, or flat?
  * precision / recall / F1 of the current 0.7 cutoff, against the base rate
  * a threshold sweep, picking the largest threshold meeting --target-precision
  * the same numbers for **S0 flags** as an alternative detector, since S1
    already showed S0 catches language contamination that xling misses (96%)

then prints a recommendation.

Usage:
    python tools/data_audit/calibrate_xling.py \
        --verdicts 'outputs/audit_10k/s2_verdicts.*.jsonl' \
        --out-md outputs/audit_10k/xling_calibration.md

Verdict records already carry `s1.xling_cos` and `s0_flags` (put there by
make_audit_sample.py), so no join is needed.  `--scores` is only a fallback
for verdicts written without the `s1` passthrough.

Deps: numpy only.
"""

import argparse
import glob
import json
import sys

import numpy as np

# same set aggregate_report.py uses for the RETRANSLATE flag branch
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


def rankdata(a):
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3 or np.all(y == y[0]) or np.all(x == x[0]):
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def prf(pred, truth):
    """precision / recall / F1 of a boolean detector."""
    tp = int((pred & truth).sum())
    npred, ntrue = int(pred.sum()), int(truth.sum())
    p = tp / npred if npred else float("nan")
    r = tp / ntrue if ntrue else float("nan")
    f = 2 * p * r / (p + r) if npred and ntrue and (p + r) > 0 else float("nan")
    return p, r, f, tp, npred


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", required=True, help="s2_verdicts glob (quote it)")
    ap.add_argument("--scores", default="",
                    help="s1_scores glob; only needed if verdicts lack the s1 passthrough")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--target-precision", type=float, default=0.8,
                    help="same default as aggregate_report.py --target-precision")
    ap.add_argument("--current-threshold", type=float, default=0.7,
                    help="aggregate_report.py's --xling-threshold default")
    ap.add_argument("--bad-score", type=int, default=2,
                    help="trans_score <= this counts as bad (matches aggregate_report)")
    args = ap.parse_args()

    vpaths = sorted(glob.glob(args.verdicts))
    if not vpaths:
        sys.exit(f"no verdict files match {args.verdicts!r} (quote the glob)")

    fallback = {}
    if args.scores:
        for r in iter_jsonl(sorted(glob.glob(args.scores))):
            if r.get("video") is not None and "xling_cos" in r:
                fallback[r["video"]] = r["xling_cos"]

    xl, bad, flagged = [], [], []
    n_raw = n_noverdict = n_noxling = n_notrans = 0
    for rec in iter_jsonl(vpaths):
        n_raw += 1
        v = rec.get("verdict") or {}
        if "error" in v and "action" not in v:
            n_noverdict += 1
            continue
        x = (rec.get("s1") or {}).get("xling_cos")
        if x is None:
            x = fallback.get(rec.get("video"))
        if x is None:
            n_noxling += 1
            continue
        ts = v.get("trans_score")
        if ts is None:
            n_notrans += 1
            continue
        xl.append(float(x))
        bad.append(ts <= args.bad_score)
        flagged.append(bool(set(rec.get("s0_flags") or []) & RETRANSLATE_FLAGS))

    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit("# `xling_cos` 是否该留在 S3 分桶规则里 —— 校准结论\n")
    emit("## A. 样本量")
    emit(f"- 读入判定 **{n_raw}** 条（{len(vpaths)} 个分片）")
    emit(f"- 可用（有 xling + trans_score）：**{len(xl)}**")
    if n_noverdict or n_noxling or n_notrans:
        emit(f"- 排除：判定失败 {n_noverdict}／缺 xling {n_noxling}／缺 trans_score {n_notrans}")
    if len(xl) < 100:
        emit("\n> ⚠️ 可用样本 <100，结论不可靠。等 S2 跑完再来。")
        _write(args, out)
        return

    xl = np.array(xl)
    bad = np.array(bad, dtype=bool)
    flagged = np.array(flagged, dtype=bool)
    base = float(bad.mean())
    emit(f"- **基准差率**（全样本 trans_score≤{args.bad_score}）= **{base:.1%}**\n")

    if bad.sum() == 0:
        emit("> VLM 没判出任何翻译差的样本——要么翻译确实没问题，"
             "要么 rubric 太松。**先查 §4 的 8B vs 裁判一致率**，不要据此调阈值。")
        _write(args, out)
        return

    # ---------------- B. deciles ----------------
    emit(f"## B. trans_score≤{args.bad_score} 差率 × xling 十分位")
    emit("| 十分位 | xling 区间 | n | 差率 | vs 基准 |")
    emit("|---|---|---|---|---|")
    order = np.argsort(xl)
    xs, bs = xl[order], bad[order]
    n = len(xs)
    rates = []
    for d in range(10):
        lo, hi = d * n // 10, (d + 1) * n // 10
        if hi <= lo:
            continue
        chunk = bs[lo:hi]
        rate = float(chunk.mean())
        rates.append(rate)
        emit(f"| D{d} | {xs[lo]:.3f}–{xs[hi - 1]:.3f} | {hi - lo} | "
             f"{rate:.1%} | {rate / base:.2f}× |")
    emit()

    # ---------------- C. monotonicity ----------------
    rho = spearman(xl, bad.astype(float))
    lift = rates[0] / rates[-1] if rates and rates[-1] > 0 else float("inf")
    emit("## C. 单调性")
    emit(f"- ρ(xling, trans_bad) = **{rho:.3f}**（期望**显著为负**）")
    emit(f"- 最低十分位差率 {rates[0]:.1%} vs 最高十分位 {rates[-1]:.1%}"
         f"　→ **lift = {lift:.2f}×**")
    flat = abs(rho) < 0.15 and lift < 1.5
    if flat:
        emit("- → **曲线基本平坦**：xling 分位几乎不预测翻译差率。")
    else:
        emit("- → 曲线有判别力，可以选阈值。")
    emit()

    # ---------------- D. current default ----------------
    t = args.current_threshold
    p, r, f, tp, npred = prf(xl < t, bad)
    emit(f"## D. 当前默认阈值 {t} 的实际表现")
    emit(f"- 命中 **{npred}** 条（占 {npred / n:.1%} 语料），其中真差 **{tp}** 条")
    emit(f"- **精度 {p:.1%}**（基准 {base:.1%}，即 lift {p / base:.2f}×）"
         f"　**召回 {r:.1%}**　F1 {f:.2f}")
    if p < args.target_precision:
        emit(f"- ⚠️ 精度 {p:.1%} < 目标 {args.target_precision:.0%}："
             f"这批被扫进 RETRANSLATE 的记录里有 **{1 - p:.1%}** 是误伤——"
             "而且它们会**抢在 `sig_zh` 的 RECAPTION 判断之前**被钉死。")
    emit()

    # ---------------- E. threshold sweep ----------------
    emit(f"## E. 阈值扫描（目标精度 {args.target_precision:.0%}）")
    emit("| 阈值 | 命中数 | 占语料 | 精度 | 召回 | F1 |")
    emit("|---|---|---|---|---|---|")
    cands = sorted({round(float(np.percentile(xl, q)), 3)
                    for q in (1, 2, 5, 10, 15, 20, 25)} | {0.55, 0.6, 0.65, 0.7, 0.75})
    for c in cands:
        p_, r_, f_, tp_, np_ = prf(xl < c, bad)
        if np_ == 0:
            continue
        emit(f"| {c:.3f} | {np_} | {np_ / n:.1%} | {p_:.1%} | {r_:.1%} | {f_:.2f} |")
    # largest threshold still meeting target precision (mirrors calibrate())
    best = None
    cum_bad = 0
    for i in range(n):
        cum_bad += int(bs[i])
        if i + 1 >= 50 and cum_bad / (i + 1) >= args.target_precision:
            best = float(xs[i])
    emit()
    if best is None:
        emit(f"- **没有任何阈值能达到 {args.target_precision:.0%} 精度**"
             "（且命中数 ≥50）。")
    else:
        p_, r_, f_, tp_, np_ = prf(xl < best, bad)
        emit(f"- 满足目标精度的最大阈值 = **{best:.3f}**"
             f"（命中 {np_} 条 = {np_ / n:.1%}，精度 {p_:.1%}，召回 {r_:.1%}）")
    emit()

    # ---------------- F. S0 flags as the alternative ----------------
    emit("## F. 对照：改用 S0 旗标做检测器")
    fp, fr, ff, ftp, fnp = prf(flagged, bad)
    emit(f"- `RETRANSLATE_FLAGS` 命中 **{fnp}** 条（{fnp / n:.1%}），"
         f"其中真差 {ftp} 条")
    emit(f"- **精度 {fp:.1%}**（lift {fp / base:.2f}×）　**召回 {fr:.1%}**　F1 {ff:.2f}")
    both_p, both_r, both_f, _, both_np = prf(flagged | (xl < t), bad)
    emit(f"- 现行规则（旗标 **或** xling<{t}）合并效果："
         f"命中 {both_np}（{both_np / n:.1%}），精度 {both_p:.1%}，召回 {both_r:.1%}")
    emit()

    # ---------------- G. recommendation ----------------
    emit("## G. 建议")
    if flat:
        emit(f"**让 `xling_cos` 退出分桶规则**，或把 `--xling-threshold` 压到极低尾"
             f"（≈p5 = {np.percentile(xl, 5):.3f}，只保留最确凿的一小撮）。")
        emit("理由：B/C 两节显示差率随 xling 分位基本不变，"
             f"当前默认 {t} 的精度 {p:.1%} 相对基准 {base:.1%} 几乎没有提升，"
             "却要付出扫掉大批语料、并从 RECAPTION 抢走记录的代价。")
        if not np.isnan(fp):
            emit(f"S0 旗标以 {fp:.1%} 的精度做同一件事，"
                 f"{'更可靠' if fp > p else '至少不更差'}——语言污染判据交给它。")
        emit("\n改法（二选一）：")
        emit("```bash")
        emit(f"# ① 压到极低尾")
        emit(f"python tools/data_audit/aggregate_report.py ... "
             f"--xling-threshold {np.percentile(xl, 5):.3f}")
        emit("# ② 完全退出：传一个不可能命中的值，让分桶只由 S0 旗标 + sig_* 决定")
        emit("python tools/data_audit/aggregate_report.py ... --xling-threshold 0")
        emit("```")
    elif best is not None:
        emit(f"**把 `--xling-threshold` 设为 {best:.3f}**"
             f"（当前默认 {t} 精度只有 {p:.1%}，低于目标 {args.target_precision:.0%}）。")
        emit("```bash")
        emit(f"python tools/data_audit/aggregate_report.py ... --xling-threshold {best:.3f}")
        emit("```")
    else:
        # 低基准率下，target-precision 可能在算术上就不可达：precision <= base * lift
        need_lift = args.target_precision / base if base > 0 else float("inf")
        achievable = [(prf(xl < c, bad)[0], c) for c in cands
                      if prf(xl < c, bad)[4] >= 50]
        achievable = [(p_, c) for p_, c in achievable if not np.isnan(p_)]
        best_p, best_c = max(achievable) if achievable else (float("nan"), float("nan"))
        emit(f"曲线**有判别力**（lift {lift:.1f}×），但没有阈值达到 "
             f"{args.target_precision:.0%} 精度。")
        emit(f"- 注意这可能是**算术上不可达**，而不是 xling 的错："
             f"精度 ≤ 基准率 × lift，基准率只有 {base:.1%}，"
             f"要达到 {args.target_precision:.0%} 需要**整个被选集合**上维持 "
             f"{need_lift:.1f}× 的 lift（不只是极尾）。")
        emit(f"- 实际能拿到的最高精度 ≈ **{best_p:.1%}**（阈值 {best_c:.3f}）。")
        emit("\n三选一：")
        emit(f"1. **降低目标精度**到现实水平（如 `--target-precision "
             f"{max(0.3, round(best_p - 0.05, 1)):.1f}`），明确接受误伤率；")
        emit("2. **改成合取条件**「xling 低 **且** 有 S0 旗标」——F 节若显示旗标精度"
             "远高于 xling，合取能同时保住精度和一部分召回（需改 "
             "`aggregate_report.py` 那行 `or` 为 `and`）；")
        emit(f"3. **只保留极低尾**（`--xling-threshold {np.percentile(xl, 2):.3f}`），"
             "当作高置信度小样本处理，其余交给 S0 旗标。")
    emit(f"\n> 结论记进 `tools/data_audit/DATA_ISSUES.md` §4 决策 1，并标为已裁决。")

    _write(args, out)


def _write(args, out):
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        print(f"\n[written] {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
