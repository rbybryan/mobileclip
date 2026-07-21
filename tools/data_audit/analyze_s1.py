#!/usr/bin/env python3
"""Stage-1 deep-read: the JOINT structure of sig_zh / sig_en / xling_cos.

The marginal min/median/p95 printed by SETUP_V100.md §3.5 only answers "did S1
run correctly". It cannot answer "what is wrong with the data", because every
diagnosis in the plan's interpretation matrix is a *joint* condition:

    sig_en high + sig_zh low + xling low -> translation broke it (RETRANSLATE)
    sig_en low  + sig_zh low             -> source caption wrong (RECAPTION)
    sig_en high + sig_zh high            -> KEEP

This script reads the same s1_scores.*.jsonl and reports the joint structure,
plus three checks that the marginals actively hide:

  1. CALIBRATION.  Raw SigLIP cosines are not quality scores.  SigLIP is
     trained with a sigmoid loss on `t * cos + b` with a large learned t and a
     very negative b, so "0.13" is not "13% aligned" and is not comparable to
     a CLIP cosine.  Only the *ranking* is meaningful -- which is why S3
     calibrates deciles against VLM verdicts instead of using a fixed cutoff.
     Everything below is therefore reported in quantiles, not absolute values.

  2. THE zh<en GAP IS CONFOUNDED.  sig_en is scored on the very quantity
     InternVid-10M-FLT selected on (top-30% by English CLIP score), so it is
     upward-biased by construction, and SigLIP2's text tower need not place zh
     and en on the same cosine scale.  A raw `mean(sig_en) > mean(sig_zh)` is
     therefore NOT evidence that translation hurt alignment.  The control:
     measure the gap *within the most faithful translations* (top xling
     decile).  Whatever gap survives there is encoder/selection artifact, not
     translation damage; only the excess gap in the low-xling tail is real.

  3. HIGH xling CAN MEAN UNTRANSLATED.  LaBSE maps identical strings to
     identical vectors, so a caption left in English scores xling_cos ~= 1.0 --
     the maximum -- while being a D-class failure.  xling is only trustworthy
     as "faithful" once the copied-English records are excluded, which is what
     --flags (s0_flags.jsonl) is used for here.

Usage:
    python tools/data_audit/analyze_s1.py \
        --scores 'outputs/audit_10k/s1_scores.*.jsonl' \
        --flags  outputs/audit_10k/s0_flags.jsonl \
        --out-md outputs/audit_10k/s1_analysis.md

--flags is optional but strongly recommended (it powers check 3).
Deps: numpy only.
"""

import argparse
import glob
import json
import sys
from collections import Counter

import numpy as np

# S0 flags that mean "the zh field is really English", i.e. xling_cos is high
# for the wrong reason.
COPIED_EN_FLAGS = {"zh_no_chinese", "zh_equals_en", "zh_mostly_english"}


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
    """Average ranks with ties averaged (avoids a scipy dependency)."""
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
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pct(n, d):
    return f"{n} ({n / d:.1%})" if d else f"{n} (n/a)"


def describe(v):
    return (f"n={v.size} min={v.min():.3f} p5={np.percentile(v, 5):.3f} "
            f"p25={np.percentile(v, 25):.3f} median={np.median(v):.3f} "
            f"p75={np.percentile(v, 75):.3f} p95={np.percentile(v, 95):.3f} "
            f"max={v.max():.3f} mean={v.mean():.3f} sd={v.std():.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True,
                    help="glob for s1_scores.*.jsonl (quote it)")
    ap.add_argument("--flags", default="",
                    help="s0_flags.jsonl; enables the untranslated-vs-faithful check")
    ap.add_argument("--out-md", default="",
                    help="also write the report to this markdown file")
    ap.add_argument("--low-q", type=float, default=0.25,
                    help="quantile defining 'low' for the interpretation matrix")
    ap.add_argument("--copied-xling", type=float, default=0.95,
                    help="xling_cos at/above this is suspected copied English")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.scores))
    if not paths:
        sys.exit(f"no files matched {args.scores!r} (quote the glob)")
    rows = list(iter_jsonl(paths))
    if not rows:
        sys.exit("no records read")

    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    # ---------------- A. completeness ----------------
    n = len(rows)
    err = [r for r in rows if "error" in r]
    emit(f"# S1 联合分析（{len(paths)} 个分片，{n} 条）\n")
    emit("## A. 完整性")
    emit(f"- 总条数 **{n}**，error **{pct(len(err), n)}**（门槛 <2%）")
    if err:
        for k, c in Counter(r["error"] for r in err).most_common():
            emit(f"  - `{k}`: {c}")
    scored = [r for r in rows if "error" not in r]
    have_zh = np.array([r["sig_zh_mean"] for r in scored if "sig_zh_mean" in r])
    have_en = np.array([r["sig_en_mean"] for r in scored if "sig_en_mean" in r])
    have_xl = np.array([r["xling_cos"] for r in scored if "xling_cos" in r])
    emit(f"- 有 sig_zh **{pct(have_zh.size, n)}** / sig_en **{pct(have_en.size, n)}** "
         f"/ xling **{pct(have_xl.size, n)}**")
    n_nan = sum(int(np.isnan(v).sum()) for v in (have_zh, have_en, have_xl))
    emit(f"- NaN 合计 **{n_nan}**（>0 说明 fp16 推理有问题，非数据问题）")
    if have_zh.size:
        emit(f"- sig_zh 唯一值数 **{np.unique(have_zh).size}**（过小=分数塌缩）")
    emit()

    # 只保留三个分数齐全的记录做联合分析
    j = [r for r in scored
         if "sig_zh_mean" in r and "sig_en_mean" in r and "xling_cos" in r]
    if len(j) < 10:
        emit("> 三分齐全的记录不足 10 条，无法做联合分析。")
        _write(args, out)
        return
    zh = np.array([r["sig_zh_mean"] for r in j])
    en = np.array([r["sig_en_mean"] for r in j])
    xl = np.array([r["xling_cos"] for r in j])
    emit(f"三分齐全用于以下联合分析：**{len(j)}** 条\n")

    # ---------------- B. marginals ----------------
    emit("## B. 边际分布（绝对值不可解读，只看形状）")
    emit("```")
    emit(f"sig_zh : {describe(zh)}")
    emit(f"sig_en : {describe(en)}")
    emit(f"xling  : {describe(xl)}")
    emit("```")
    emit("> SigLIP 余弦经 sigmoid 损失 `t*cos+b` 训练，t 大、b 负，"
         "余弦 0.13 不等于「13% 对齐」，也不能与 CLIP 余弦比较。"
         "**只有排序有意义**——所以下面全部用分位数。\n")

    # ---------------- C. joint structure ----------------
    emit("## C. 联合结构（相关性）")
    r_zh_en = spearman(zh, en)
    emit(f"- ρ(sig_zh, sig_en) = **{r_zh_en:.3f}**"
         "　→ 高=两者共享「视频好不好拍/好不好描述」的公共难度分量；"
         "接近 0 说明两个塔的打分几乎无关，外推要谨慎")
    emit(f"- ρ(sig_zh, xling) = **{spearman(zh, xl):.3f}**")
    emit(f"- ρ(sig_en, xling) = **{spearman(en, xl):.3f}**")
    emit()

    # ---------------- D. the confounded gap ----------------
    d = en - zh
    emit("## D. zh<en 的落差：是翻译损伤还是编码器/选择偏差？")
    emit("```")
    emit(f"d = sig_en - sig_zh : {describe(d)}")
    emit("```")
    r_d_xl = spearman(xl, d)
    emit(f"- ρ(xling, d) = **{r_d_xl:.3f}**（期望**显著为负**："
         "翻译越不忠实、落差越大）")
    hi = xl >= np.percentile(xl, 90)
    lo = xl <= np.percentile(xl, 10)
    emit(f"- **控制组**：xling 最高十分位（忠实译文）内 d 中位数 = "
         f"**{np.median(d[hi]):+.4f}**")
    emit(f"- 对照：xling 最低十分位内 d 中位数 = **{np.median(d[lo]):+.4f}**")
    excess = float(np.median(d[lo]) - np.median(d[hi]))
    emit(f"- **超额落差（低 xling − 高 xling）= {excess:+.4f}**")
    emit(f"  → 高 xling 组里残留的 {np.median(d[hi]):+.4f} 是"
         "**编码器语言尺度 + FLT 按英文 CLIP 分选样**造成的基线偏移，"
         "不是翻译的锅；只有这里的**超额部分**才是真实翻译损伤。")
    if r_d_xl > -0.05:
        emit("  ⚠️ ρ(xling,d) 几乎为 0：xling 与画面落差彼此无关，"
             "说明二者至少有一个没测到该测的东西——S2 判定回来后务必复核。")
    emit()

    # ---------------- E. the high-xling trap ----------------
    emit("## E. xling 高分陷阱（未翻译=满分）")
    copied = xl >= args.copied_xling
    emit(f"- xling ≥ {args.copied_xling} 的记录：**{pct(int(copied.sum()), len(j))}**"
         "　← 疑似原文照抄（LaBSE 对相同字符串给 ~1.0）")
    if args.flags:
        vid2flags = {}
        for r in iter_jsonl(sorted(glob.glob(args.flags))):
            v = r.get("video")
            fl = r.get("flags") or r.get("flag") or []
            if isinstance(fl, str):
                fl = [fl]
            if v is not None:
                vid2flags.setdefault(v, set()).update(fl)
        if not vid2flags:
            emit("- ⚠️ `--flags` 没读到任何记录，跳过交叉核对")
        else:
            vids = [r.get("video") for r in j]
            is_copied_flag = np.array(
                [bool(vid2flags.get(v, set()) & COPIED_EN_FLAGS) for v in vids])
            emit(f"- S0 判为「中文栏其实是英文」的记录："
                 f"**{pct(int(is_copied_flag.sum()), len(j))}**")
            both = int((copied & is_copied_flag).sum())
            emit(f"  - 其中 xling ≥ {args.copied_xling} 的：**{both}**"
                 f"（占高 xling 组 {both / max(int(copied.sum()), 1):.1%}）"
                 "　→ 比例高即证实「高 xling = 没翻译」这条陷阱在本批数据里成立")
            clean = ~is_copied_flag
            if clean.sum() > 10:
                emit(f"  - **剔除照抄后**的 xling 中位数 = "
                     f"**{np.median(xl[clean]):.3f}**"
                     f"（原 {np.median(xl):.3f}）——这才是真实翻译忠实度基线")
    else:
        emit("- 未传 `--flags`，无法区分「忠实翻译」与「原文照抄」"
             "；建议补跑 s0 后重跑本脚本")
    emit()

    # ---------------- F. interpretation matrix ----------------
    q = args.low_q
    zt, et, xt = (np.quantile(zh, q), np.quantile(en, q), np.quantile(xl, q))
    emit(f"## F. 解读矩阵预览（低 = 各自最低 {q:.0%} 分位）")
    emit(f"阈值：sig_zh<{zt:.4f}　sig_en<{et:.4f}　xling<{xt:.4f}"
         "（**临时分位阈值，正式阈值由 S3 用 VLM 判定校准**）\n")
    zl, el, xll = zh < zt, en < et, xl < xt
    cells = [
        ("RETRANSLATE 候选", "en 高 + zh 低 + xling 低", (~el) & zl & xll),
        ("RECAPTION 候选", "en 低 + zh 低", el & zl),
        ("翻译可疑（画面未受损）", "en 高 + zh 高 + xling 低", (~el) & (~zl) & xll),
        ("KEEP 候选", "en 高 + zh 高 + xling 高", (~el) & (~zl) & (~xll)),
    ]
    emit("| 桶 | 条件 | 条数 | 占比 |")
    emit("|---|---|---|---|")
    for name, cond, mask in cells:
        c = int(mask.sum())
        emit(f"| {name} | {cond} | {c} | {c / len(j):.1%} |")
    covered = np.zeros(len(j), dtype=bool)
    for _, _, m in cells:
        covered |= m
    emit(f"| （其余混合情形） | — | {int((~covered).sum())} | "
         f"{(~covered).sum() / len(j):.1%} |")
    emit("\n> 这张表**不是结论**，只是 S2 该往哪抽样的预算表："
         "四个角的样本都要有，S3 才能把分位映射成问题概率。\n")

    # ---------------- G. next ----------------
    emit("## G. 下一步")
    emit("1. 若尚未跑 S0：`bash tools/data_audit/run_v100.sh s0`"
         "（秒级，且 `s2-sample` 需要 `s0_flags.jsonl`）")
    emit("2. `bash tools/data_audit/run_v100.sh s2-sample`（10k 全覆盖）")
    emit("3. `bash tools/data_audit/run_v100.sh s2`（8 卡 ~2–3 小时）")
    emit("4. `bash tools/data_audit/run_v100.sh s3` → `report.md`（阈值在此才算数）")

    _write(args, out)


def _write(args, out):
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        print(f"\n[written] {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
