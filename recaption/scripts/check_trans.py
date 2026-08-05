#!/usr/bin/env python3
"""Did the text-only translation re-judgement actually break the halo?

`s2-trans` exists because the combined S2 rubric never measured translation
quality: en_score/zh_score/trans_score came back per-record identical on
92.5%/98.9% of the 10k, and `action` was a fixed ladder on that one number
(recaption/docs/data-issues.md problem 8). The fix withholds the video and forces evidence
before scoring -- but "the fix ran" is not "the fix worked", and the run takes
a while. This script answers the question from a PARTIAL output, so a failed
fix can be caught after a few hundred records instead of after the whole pass.

Three questions, in order of what would make you stop the run:

  1. IS IT STILL ANCHORED?  Per-record agreement between the NEW trans_score
     and the OLD en_score. The old trans_score was ~the old en_score, so if the
     new one still matches old en_score on >60% of records, withholding the
     frames did not help and the prompt needs rethinking -- stop and rethink
     rather than burn the rest of the pass.
  2. DID THE MODEL FOLLOW ITS OWN SCORING RULE?  TRANS_PROMPT says "mismatches
     empty -> must give 5". The rate at which that holds measures instruction
     adherence directly; if it is low, the score is again a free-floating
     impression and the evidence list is decoration.
  3. WHAT CHANGED?  Old vs new bad-rate (<=2) and score distribution, so the
     size of the correction is on the record before it propagates into
     xling-check and S3.

Usage:
    python recaption/scripts/check_trans.py \
        --trans   'outputs/audit_10k/s2_trans.*.jsonl' \
        --main    'outputs/audit_10k/s2_verdicts.*.jsonl' \
        --sample  outputs/audit_10k/s2_sample.jsonl \
        --out-md  outputs/audit_10k/trans_health.md

--main and --sample are optional (they power checks 1/3 and the progress line).
Safe to run mid-pass. Deps: stdlib only.
"""

import argparse
import glob
import json
import sys
from collections import Counter

HALO_GATE = 0.6      # new trans vs old en per-record identity
RULE_GATE = 0.8      # "mismatches empty -> 5" adherence


def iter_jsonl(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def pct(n, d):
    return f"{n / d:.1%}" if d else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trans", required=True, help="s2_trans glob (QUOTE it)")
    ap.add_argument("--main", default=None, help="s2_verdicts glob (QUOTE it)")
    ap.add_argument("--sample", default=None, help="s2_sample.jsonl, for progress")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    tpaths = sorted(glob.glob(args.trans))
    if not tpaths:
        sys.exit(f"no files match {args.trans!r} (quote the glob; "
                 "and the pass may not have written anything yet)")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    new = {}
    failed = 0
    score_dist = Counter()
    mism_empty_scores = Counter()
    mism_nonempty_scores = Counter()
    untrans = 0
    for rec in iter_jsonl(tpaths):
        v = rec.get("verdict") or {}
        if "error" in v and "trans_score" not in v:
            failed += 1
            continue
        ts = v.get("trans_score")
        if ts is None:
            failed += 1
            continue
        new[rec.get("video")] = ts
        score_dist[ts] += 1
        mm = v.get("mismatches")
        if isinstance(mm, list):
            (mism_empty_scores if not mm else mism_nonempty_scores)[ts] += 1
        if v.get("untranslated") is True:
            untrans += 1

    n_new = len(new)
    emit("# 文本-only 翻译重判体检（check_trans.py）")
    emit()
    total_expected = None
    if args.sample:
        total_expected = sum(1 for _ in open(args.sample, encoding="utf-8"))
    # 失败的记录也已经"跑过"了，只是没产出可用分数——拿 n_new 单独跟抽样数比，
    # 会把"跑完了但有 1 条失败"误报成"还没跑完"。
    processed = n_new + failed
    emit(f"- 已完成 **{processed}** 条" +
         (f" / 共 {total_expected}（**{pct(processed, total_expected)}**）"
          if total_expected else "") +
         f"；其中可用 {n_new}、判定失败 {failed}")
    if total_expected and processed < total_expected:
        emit("- ⏳ **这一趟还没跑完**——下面的比率是部分样本上的，"
             "趋势可信（分片按 `行号 % 8` 均匀取样，不是前 N 条），但最终数字以跑完为准。")
    emit()

    # ------------------------------------------------------- 1 还锚不锚定
    emit("## 1. 还锚定在画面对齐分上吗（**最关键**）")
    emit()
    problems, notes = [], []
    if not args.main:
        emit("- （未传 `--main`，跳过——这一项是本脚本的主要目的，建议补上）")
    else:
        mpaths = sorted(glob.glob(args.main))
        old_en, old_trans = {}, {}
        for rec in iter_jsonl(mpaths):
            v = rec.get("verdict") or {}
            vid = rec.get("video")
            if v.get("en_score") is not None:
                old_en[vid] = v["en_score"]
            if v.get("trans_score") is not None:
                old_trans[vid] = v["trans_score"]
        ov = [k for k in new if k in old_en and k in old_trans]
        if not ov:
            emit("- ⚠️ 与主力档没有重叠记录，无法比较（`--main` 路径对吗？）")
        else:
            same_en = sum(new[k] == old_en[k] for k in ov)
            same_tr = sum(new[k] == old_trans[k] for k in ov)
            mad_en = sum(abs(new[k] - old_en[k]) for k in ov) / len(ov)
            mad_tr = sum(abs(new[k] - old_trans[k]) for k in ov) / len(ov)
            emit(f"重叠 **{len(ov)}** 条：")
            emit()
            emit("| 比较 | 逐条同值 | 占比 | 平均绝对差 | 读法 |")
            emit("|---|---|---|---|---|")
            emit(f"| 新 `trans` vs **旧 `en`** | {same_en} | **{pct(same_en, len(ov))}** | "
                 f"{mad_en:.2f} | **>60% = 仍被画面对齐分锚定，修复失败** |")
            emit(f"| 新 `trans` vs 旧 `trans` | {same_tr} | {pct(same_tr, len(ov))} | "
                 f"{mad_tr:.2f} | 越低说明改判越多（旧值≈旧 en，本就不可信） |")
            emit()
            if same_en / len(ov) > HALO_GATE:
                problems.append(
                    f"新 `trans_score` 与**旧 `en_score`** 仍有 {pct(same_en, len(ov))} "
                    "逐条同值——拿掉画面**没能**解除锚定。别再等这一趟跑完："
                    "先在几十条上人工看新 prompt 的原始输出，"
                    "多半是模型仍在用「整体印象」而非 `mismatches` 打分")
            else:
                emit(f"✅ 同值率 {pct(same_en, len(ov))} < {HALO_GATE:.0%}，"
                     "**锚定已解除**——新 `trans_score` 是独立测量，可用于决策 1。")
                emit()

    # ------------------------------------------------------- 2 守不守规则
    emit("## 2. 模型守自己的评分规则吗（`mismatches` 为空 → 必须给 5）")
    emit()
    n_empty = sum(mism_empty_scores.values())
    n_nonempty = sum(mism_nonempty_scores.values())
    if not (n_empty or n_nonempty):
        emit("- 判定里没有 `mismatches` 字段，无法核对（prompt 被改过？）")
    else:
        ok5 = mism_empty_scores.get(5, 0)
        emit(f"- `mismatches` 为空：**{n_empty}** 条，其中给 5 分的 "
             f"**{ok5}（{pct(ok5, n_empty)}）**")
        emit(f"- `mismatches` 非空：**{n_nonempty}** 条，"
             f"平均分 {sum(k * v for k, v in mism_nonempty_scores.items()) / n_nonempty:.2f}"
             if n_nonempty else "- `mismatches` 非空：0 条")
        emit()
        emit(f"- 空 mismatches 的分数分布：{dict(sorted(mism_empty_scores.items()))}")
        emit(f"- 非空 mismatches 的分数分布：{dict(sorted(mism_nonempty_scores.items()))}")
        emit()
        if n_empty and ok5 / n_empty < RULE_GATE:
            problems.append(
                f"`mismatches` 为空却没给 5 的占 {pct(n_empty - ok5, n_empty)}"
                "——模型没守自己的评分规则，分数仍是浮空的整体印象，证据列表成了摆设")
        if n_empty and n_nonempty:
            gap = (sum(k * v for k, v in mism_nonempty_scores.items()) / n_nonempty)
            if gap >= 4.0:
                notes.append(f"列了 `mismatches` 的记录平均仍有 {gap:.2f} 分，"
                             "偏高——可能在把无关紧要的措辞差异也列成 mismatch")

    # ------------------------------------------------------- 3 改判幅度
    emit("## 3. 新的翻译差率")
    emit()
    bad = sum(v for k, v in score_dist.items() if k <= 2)
    emit(f"- `trans_score` 分布：{dict(sorted(score_dist.items()))}")
    emit(f"- **差率（≤2）= {bad}／{n_new} = {pct(bad, n_new)}**")
    emit(f"- `untranslated=true`：{untrans}（{pct(untrans, n_new)}）"
         "——与 S0 的语言污染旗标（327 条 / 3.3%）对照着看")
    emit()
    emit("> 对照：主力档旧 `trans_score≤2` = 19.6%，但那是「整体质量分」的复制品，"
         "**不是翻译差率**。这里的数字才是第一个可信估计。"
         "S1 结论 3 预期它不会很高（翻译超额落差仅 +0.0106，且只在极低尾）——"
         "若这里显著偏高，那才是新发现，要回头查 S1 的假设。")
    emit()

    # ------------------------------------------------------- 结论
    emit("## 结论")
    emit()
    if notes:
        emit("📌 记一笔：")
        for p in notes:
            emit(f"- {p}")
        emit()
    if problems:
        emit("⚠️ **修复没生效，别把结果喂给 xling-check**：")
        for p in problems:
            emit(f"- {p}")
    elif total_expected and processed < total_expected:
        emit(f"✅ 目前为止一切正常（{pct(processed, total_expected)}）。跑完后：")
        emit("1. 重跑本脚本确认最终数字")
        emit("2. `bash recaption/scripts/run_v100.sh xling-check`")
    else:
        emit("✅ 重判可用。下一步：")
        emit("1. `bash recaption/scripts/run_v100.sh xling-check` —— 裁决决策 1")
        emit("2. 结论抄进 `recaption/docs/data-issues.md` §4 决策 1 与 `recaption/docs/audit-findings.md`")
    emit()

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[written] {args.out_md}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
