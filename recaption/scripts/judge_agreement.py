#!/usr/bin/env python3
"""Score main-model verdicts against judge verdicts: agreement + Cohen's kappa.

Two reasons this is a separate script rather than the snippet inside
`run_v100.sh s2-judge`:

1. **Cohen's kappa was specified but never implemented.** The plan gates on it
   twice -- `recaption/docs/archive/audit-plan.md` §4 ("8B 判定与 235B 裁判
   ... 求一致率(Cohen's κ)，κ<0.6 的维度回炉改 rubric") and §7.4 -- but the
   only thing computed anywhere was 5-class exact-match agreement. Raw
   agreement is inflated by class imbalance: if 80% of records are `keep`,
   two models that both always say `keep` "agree" 80% of the time while
   carrying zero information. Kappa corrects for that chance agreement.

2. **The judge may not run on this machine.** If 101vector cannot reach the
   endpoint and the judge runs on a relay host, the verdicts come back as a
   file -- with the logic welded into the s2-judge branch there was no way to
   score it without re-running the judge.

Reports, for `action`:
  * 5-class exact agreement + kappa (the harsh view)
  * binary agreement + kappa, collapsing to {keep, fix_text} vs
    {retranslate, recaption, drop} -- this is the distinction S3 bucketing
    actually rests on, and the one recaption/docs/archive/setup-judge-api.md route D3 recommends
    gating on
  * a confusion matrix, so a systematic skew (e.g. the main model calling
    `recaption` where the judge says `retranslate`) is visible rather than
    buried in one number

and per-dimension kappa for the binary rubric axes (zh/en/trans bad,
coarse, temporal), so a single bad axis can be sent back to the rubric
instead of condemning the whole run.

Usage:
    python recaption/scripts/judge_agreement.py \
        --main 'outputs/audit_10k/s2_verdicts.*.jsonl' \
        --judge outputs/audit_10k/s2_verdicts_judge.jsonl \
        --out-md outputs/audit_10k/judge_agreement.md

Deps: stdlib only.
"""

import argparse
import glob
import json
import sys
from collections import Counter

TEXT_SIDE = {"keep", "fix_text"}          # 文本侧可用，无需重做内容
ACTIONS = ["keep", "fix_text", "retranslate", "recaption", "drop"]


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


def load(pattern, label):
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no {label} files match {pattern!r} (quote the glob)")
    out = {}
    for rec in iter_jsonl(paths):
        v = rec.get("verdict") or {}
        if "action" not in v:
            continue
        vid = rec.get("video")
        if vid is not None:
            out[vid] = v
    return out, paths


def kappa(pairs):
    """Cohen's kappa over (a, b) label pairs."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe >= 1.0:
        return float("nan")          # 完全退化（双方都只出一个类）
    return (po - pe) / (1 - pe)


def rate(pairs):
    return sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else float("nan")


def verdict_line(name, k, gate):
    if k != k:  # NaN
        return f"（{name} κ 无法计算：某一方只输出了单一类别，说明该维度判定退化）"
    return f"{'✅ 通过' if k >= gate else '❌ 未达标'}（κ={k:.3f}，门槛 {gate}）"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True, help="s2_verdicts glob (quote it)")
    ap.add_argument("--judge", required=True, help="judge verdicts glob (quote it)")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--agree-gate", type=float, default=0.75,
                    help="5-class raw agreement gate (recaption/docs/setup-v100 §4)")
    ap.add_argument("--binary-gate", type=float, default=0.80,
                    help="binary raw agreement gate (recaption/docs/setup-judge-api §0 route D3)")
    ap.add_argument("--kappa-gate", type=float, default=0.6,
                    help="per-dimension kappa gate (plan §4)")
    args = ap.parse_args()

    main_v, main_paths = load(args.main, "main")
    judge_v, judge_paths = load(args.judge, "judge")
    shared = sorted(set(main_v) & set(judge_v))

    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit("# 主力 vs 裁判：一致率与 Cohen's κ\n")
    emit(f"- 主力判定 **{len(main_v)}** 条（{len(main_paths)} 个文件）")
    emit(f"- 裁判判定 **{len(judge_v)}** 条（{len(judge_paths)} 个文件）")
    emit(f"- **重叠 {len(shared)}** 条（以下全部基于重叠部分）\n")
    if len(shared) < 30:
        emit("> ⚠️ 重叠不足 30 条，统计不可靠。检查两边是否跑的同一批样本"
             "（裁判取的是 `s2_sample.jsonl` 的前 N 条）。")
        _write(args, out)
        return

    pairs = [(main_v[v].get("action"), judge_v[v].get("action")) for v in shared]
    pairs = [(a, b) for a, b in pairs if a and b]

    # ---- 5 类 ----
    a5, k5 = rate(pairs), kappa(pairs)
    emit("## 1. `action` 五类精确匹配")
    emit(f"- 原始一致率 **{a5:.1%}**（门槛 {args.agree_gate:.0%}）"
         f"　{'✅ 通过' if a5 >= args.agree_gate else '❌ 未达标'}")
    emit(f"- **Cohen's κ = {k5:.3f}**")
    emit(f"- κ 与原始一致率的差距说明**类别不平衡**："
         f"若大量样本都是 `keep`，双方一起说 `keep` 也能堆出高一致率，κ 才是去掉"
         f"「碰巧一致」之后的真实一致程度。**以 κ 为准。**\n")

    # ---- 二值 ----
    bpairs = [(a in TEXT_SIDE, b in TEXT_SIDE) for a, b in pairs]
    ab, kb = rate(bpairs), kappa(bpairs)
    emit("## 2. 二值折叠：`{keep, fix_text}` vs `{retranslate, recaption, drop}`")
    emit("这是 **S3 分桶真正依赖的判别**（要不要动这条数据），也是 "
         "`recaption/docs/archive/setup-judge-api.md` §0 路线 D3 建议改挂的口径——"
         "`recaption` vs `retranslate` 的边界连人都会犹豫，不该拿来当门。")
    emit(f"- 原始一致率 **{ab:.1%}**（门槛 {args.binary_gate:.0%}）"
         f"　{'✅ 通过' if ab >= args.binary_gate else '❌ 未达标'}")
    emit(f"- **Cohen's κ = {kb:.3f}**\n")

    # ---- 混淆矩阵 ----
    emit("## 3. 混淆矩阵（行=主力 8B，列=裁判）")
    cm = Counter(pairs)
    present = [a for a in ACTIONS if any(x == a for x, _ in pairs) or
               any(y == a for _, y in pairs)]
    extra = sorted({x for p in pairs for x in p} - set(ACTIONS))
    cols = present + extra
    emit("| 主力＼裁判 | " + " | ".join(f"`{c}`" for c in cols) + " | 合计 |")
    emit("|---" * (len(cols) + 2) + "|")
    for r in cols:
        row = [cm.get((r, c), 0) for c in cols]
        emit(f"| **`{r}`** | " + " | ".join(str(x) for x in row) +
             f" | {sum(row)} |")
    emit("| **合计** | " +
         " | ".join(str(sum(cm.get((r, c), 0) for r in cols)) for c in cols) +
         f" | {len(pairs)} |")
    emit("\n> 看**非对角线的偏斜**：若某个格子特别大（例如主力判 `recaption`、"
         "裁判判 `retranslate`），那是 rubric 里这两类的边界没说清，"
         "改 rubric 比换模型有用。\n")

    # ---- 各维度 κ ----
    emit(f"## 4. 各 rubric 维度的 κ（plan §4 门槛：κ ≥ {args.kappa_gate}）")
    emit("κ 低的维度**单独回炉改 rubric**即可，不必否定整轮判定。\n")
    dims = [
        ("zh_bad（zh_score≤2）", lambda v: (v.get("zh_score") is not None
                                            and v["zh_score"] <= 2)),
        ("en_bad（en_score≤2）", lambda v: (v.get("en_score") is not None
                                            and v["en_score"] <= 2)),
        ("trans_bad（trans_score≤2）", lambda v: (v.get("trans_score") is not None
                                                  and v["trans_score"] <= 2)),
        ("granularity=coarse", lambda v: v.get("granularity") == "coarse"),
        ("temporal_order_correct=False", lambda v: v.get("temporal_order_correct")
         is False),
    ]
    emit("| 维度 | 一致率 | 主力判 True | 裁判判 True | κ | 判定 |")
    emit("|---|---|---|---|---|---|")
    degenerate = []
    for name, fn in dims:
        dp = [(fn(main_v[v]), fn(judge_v[v])) for v in shared]
        k = kappa(dp)
        na, nb, n = sum(a for a, _ in dp), sum(b for _, b in dp), len(dp)
        # κ 无法区分「两边真的不一致」与「有一方根本没有方差」。后者下 pe→po，
        # κ→0，哪怕一致率 90%+ 也一样——这不是判定不可信，而是 κ 这个统计量
        # 在该维度上无定义。必须分开报，否则会把「裁判说这类问题不存在」
        # 误读成「裁判在这维度不可靠」。
        deg = na in (0, n) or nb in (0, n)
        if k != k:
            verdict = "— κ 无定义"
        elif deg:
            verdict = "⚠️ **退化，κ 不适用**"
            degenerate.append((name, na, nb, n, rate(dp)))
        else:
            verdict = "✅" if k >= args.kappa_gate else "❌ 回炉"
        emit(f"| {name} | {rate(dp):.1%} | {na}/{n} | {nb}/{n} | "
             f"{'n/a' if k != k else f'{k:.3f}'} | {verdict} |")
    emit()
    if degenerate:
        emit("> ⚠️ **标「退化」的行不要当成「该维度不可信」。** κ 衡量的是"
             "「扣掉碰巧一致之后还剩多少一致」；当**一方几乎全判同一类**时，"
             "「碰巧一致」的期望值 `pe` 会逼近实际一致率 `po`，于是 κ→0 —— "
             "**哪怕原始一致率 90%+ 也一样**。这时 κ=0 表达的是"
             "「这个统计量在此处无定义」，而不是「两边判得不一致」。")
        for name, na, nb, n, r in degenerate:
            # 退化可能出在任一方，且「全 False」与「全 True」含义相反，
            # 必须分开措辞，否则会把「普遍存在」写成「不存在」。
            parts = []
            for who, cnt in (("主力", na), ("裁判", nb)):
                if cnt == 0:
                    parts.append(f"**{who}认为这类问题不存在**（0/{n}）")
                elif cnt == n:
                    parts.append(f"**{who}认为这类问题普遍存在**（{n}/{n}）")
            emit(f">   - `{name}`：一致率 {r:.1%}，" + "；".join(parts) +
                 "。**该维度的结论要据此下**，不是看 κ。")
        emit()
    emit("> `trans_bad` 这一行是 `calibrate_xling.py` 全部差率的基础。"
         "**它若因退化而 κ=0，决策 1 不是「不可信」，而是「问题稀有到 xling "
         "根本没东西可检测」——那恰恰支持把 xling 移出分桶规则。**"
         "只有在两边都有足够 True、κ 仍低时，才说明判定不可靠。\n")

    # ---- 结论 ----
    emit("## 5. 结论")
    emit(f"- 五类门（≥{args.agree_gate:.0%}）："
         f"{'✅ 通过' if a5 >= args.agree_gate else '❌ 未达标'}　"
         f"原始 {a5:.1%} / κ {k5:.3f}")
    emit(f"- 二值门（≥{args.binary_gate:.0%}）："
         f"{'✅ 通过' if ab >= args.binary_gate else '❌ 未达标'}　"
         f"原始 {ab:.1%} / κ {kb:.3f}")
    emit("\n把结论记进 `recaption/docs/data-issues.md` §3 的 T4，"
         "**并注明裁判是哪个模型**——同族/小模型裁判的通过要打折"
         "（见 `recaption/docs/archive/setup-judge-api.md` §0 路线 A′ 末尾的同族偏差警告）。")

    _write(args, out)


def _write(args, out):
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        print(f"\n[written] {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
