#!/usr/bin/env python3
"""Stage-2 post-run health check: is `s2_verdicts.*.jsonl` fit to feed S3?

The shard logs say "1250 verdicts" whether or not the model actually produced a
parseable verdict: `vlm_audit_hf.py` catches every failure and still writes a
line, as `{"verdict": {"error": ...}}`.  Both downstream consumers then drop
those lines silently --

    aggregate_report.py:72   if "error" in v and "action" not in v: continue
    calibrate_xling.py:136   same test, counted as n_noverdict

-- so a shard that failed on 40% of its clips looks complete in `status` and
merely produces a quieter report.  Every percentage in report.md is computed
over the *surviving* records, and the survivors are not a random subset (decode
failures concentrate in broken videos, JSON failures in long/odd captions).
This script makes that attrition visible before it is baked into a threshold.

Five checks, in the order they can invalidate the next step:

  1. COVERAGE.  verdict lines vs s2_sample lines, duplicates, missing videos.
  2. ATTRITION.  usable vs failed, split by cause (video decode / JSON parse /
     other), because the two have different fixes and different biases.
  3. FIELD COMPLETENESS.  A record can parse and still lack `trans_score` or
     `action`; those fields are read with .get() downstream and a null quietly
     becomes "not bad".  Reported per field, since decision 1 rests entirely on
     trans_score coverage.
  4. DISTRIBUTION SANITY.  Mode collapse (95% `keep`, or every score == 3) is
     the failure that survives every other check: it parses, it is complete, and
     it makes S3's decile calibration meaningless because there is no contrast
     left to calibrate against.
  5. PER-STRATUM USABLE COUNTS.  The sample is stratified; a stratum wiped out
     by attrition cannot be reported on, and the rare strata (S0-flagged, low
     xling) are exactly the small ones.

Usage:
    python tools/data_audit/check_s2.py \
        --verdicts 'outputs/audit_10k/s2_verdicts.*.jsonl' \
        --sample   outputs/audit_10k/s2_sample.jsonl \
        --out-md   outputs/audit_10k/s2_health.md

--sample is optional (it powers check 1).  Deps: stdlib only.
"""

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict

# Verdict fields S3 / calibrate_xling actually read. Anything missing here is a
# silent null downstream, not an error.
REQUIRED_FIELDS = ["action", "en_score", "zh_score", "trans_score",
                   "granularity", "video_readable"]
SCORE_FIELDS = ["en_score", "zh_score", "trans_score"]
ACTIONS = ["keep", "fix_text", "retranslate", "recaption", "drop"]

# Attrition causes worth separating: they have different fixes.
#   decode  -> video/codec problem, re-run will not help
#   parse   -> model emitted no valid JSON; retry or shorter prompt helps
#   oom     -> batch/frame-count problem, re-run with fewer frames helps
DECODE_HINTS = ("decode", "video", "codec", "reader", "frame")
PARSE_HINTS = ("no valid verdict json", "json", "expecting value")
OOM_HINTS = ("outofmemory", "cuda out of memory")


def iter_jsonl(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield p, json.loads(line)
                except json.JSONDecodeError:
                    continue


def classify_error(msg):
    low = str(msg).lower()
    if any(h in low for h in OOM_HINTS):
        return "oom"
    if any(h in low for h in PARSE_HINTS):
        return "json_parse"
    if any(h in low for h in DECODE_HINTS):
        return "video_decode"
    return "other"


def pct(n, d):
    return f"{n / d:.1%}" if d else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", required=True,
                    help="s2_verdicts glob (QUOTE it, e.g. '.../s2_verdicts.*.jsonl')")
    ap.add_argument("--sample", default=None,
                    help="s2_sample.jsonl; enables the coverage check")
    ap.add_argument("--out-md", default=None, help="write the report here too")
    ap.add_argument("--max-fail-rate", type=float, default=0.05,
                    help="gate: fail if attrition exceeds this (default 0.05)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.verdicts))
    if not paths:
        sys.exit(f"no verdict files match {args.verdicts!r} (quote the glob)")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    per_shard = defaultdict(lambda: {"n": 0, "ok": 0})
    seen = Counter()
    usable, failed = 0, 0
    err_kinds = Counter()
    field_present = Counter()
    action_dist = Counter()
    gran_dist = Counter()
    score_dist = {k: Counter() for k in SCORE_FIELDS}
    better_zh_counts = Counter()
    stratum_total, stratum_ok = Counter(), Counter()
    unreadable_model, unreadable_decode = 0, 0
    # check 6: are the three "independent" scores actually independent, and does
    # the chosen `action` agree with the scores the same record was given?
    n_scored = 0
    eq = Counter()          # how often two scores land on the exact same value
    absdiff = Counter()     # summed |a-b|, divided by n_scored at print time
    per_action = defaultdict(lambda: {"n": 0, "en": 0, "zh": 0, "tr": 0,
                                      "en_bad": 0, "zh_bad": 0, "tr_bad": 0})
    conflicts = Counter()
    action_n = Counter()

    for path, rec in iter_jsonl(paths):
        v = rec.get("verdict") or {}
        vid = rec.get("video")
        stratum = rec.get("stratum") or "?"
        seen[vid] += 1
        per_shard[path]["n"] += 1
        stratum_total[stratum] += 1
        if "error" in v and "action" not in v:
            failed += 1
            err_kinds[classify_error(v["error"])] += 1
            continue
        usable += 1
        per_shard[path]["ok"] += 1
        stratum_ok[stratum] += 1
        for k in REQUIRED_FIELDS:
            if v.get(k) is not None:
                field_present[k] += 1
        action_dist[v.get("action", "(missing)")] += 1
        gran_dist[v.get("granularity", "(missing)")] += 1
        for k in SCORE_FIELDS:
            score_dist[k][v.get(k)] += 1
        bz = v.get("better_zh") or []
        better_zh_counts[len(bz) if isinstance(bz, list) else "(not a list)"] += 1
        if v.get("video_readable") is False:
            # vlm_audit_hf.py writes {video_readable: false, action: "drop",
            # error: ...} for decode failures -- those are usable records (they
            # carry an action) but they are an engineering fault, not a model
            # judgement that the frames are blank. Keep the two apart.
            if "error" in v:
                unreadable_decode += 1
            else:
                unreadable_model += 1

        en, zh, tr = (v.get("en_score"), v.get("zh_score"), v.get("trans_score"))
        act = v.get("action")
        if None not in (en, zh, tr):
            n_scored += 1
            eq["en=zh"] += (en == zh)
            eq["en=trans"] += (en == tr)
            eq["zh=trans"] += (zh == tr)
            eq["all three"] += (en == zh == tr)
            absdiff["en-zh"] += abs(en - zh)
            absdiff["en-trans"] += abs(en - tr)
            absdiff["zh-trans"] += abs(zh - tr)
            d = per_action[act or "(missing)"]
            d["n"] += 1
            d["en"] += en
            d["zh"] += zh
            d["tr"] += tr
            d["en_bad"] += en <= 2
            d["zh_bad"] += zh <= 2
            d["tr_bad"] += tr <= 2
            # Rubric (vlm_audit.py PROMPT) defines each action by a score
            # pattern. Where the action contradicts the scores, S3 and
            # apply_buckets.py disagree about the same record: calibrate()
            # thresholds on scores, apply_buckets routes on action.
            if act == "retranslate" and tr >= 3:
                conflicts["action=retranslate 但 trans_score≥3（翻译并不坏）"] += 1
            if act == "keep" and zh <= 2:
                conflicts["action=keep 但 zh_score≤2（中文与画面不符）"] += 1
            if act == "recaption" and en >= 4:
                conflicts["action=recaption 但 en_score≥4（英文源头没问题）"] += 1
            if act == "drop" and v.get("video_readable") is not False and zh >= 3:
                conflicts["action=drop 但画面可辨且 zh_score≥3"] += 1
        if act is not None:
            action_n[act] += 1

    total = usable + failed
    emit("# S2 判定体检（check_s2.py）")
    emit()
    emit(f"读入 {len(paths)} 个分片文件，共 **{total}** 行。")
    emit()

    # ---------------------------------------------------------------- 1 覆盖
    emit("## 1. 覆盖率")
    emit()
    dupes = sum(c - 1 for c in seen.values() if c > 1)
    emit(f"- 唯一 video：{len(seen)}；**重复行 {dupes}**"
         "（断点续跑重复写入会造成重复计数，>0 需去重后再跑 S3）")
    if args.sample:
        sample_ids, n_sample = set(), 0
        with open(args.sample, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_sample += 1
                try:
                    sample_ids.add(json.loads(line).get("video"))
                except json.JSONDecodeError:
                    continue
        missing = sample_ids - set(seen)
        emit(f"- 抽样 {n_sample} 条 → 已判定 {len(sample_ids & set(seen))}；"
             f"**未覆盖 {len(missing)}**（>0 说明有分片没跑完，重跑 `s2` 会断点续上）")
    else:
        emit("- （未传 `--sample`，跳过「抽样条目是否全部判到」的核对）")
    emit()
    emit("| 分片 | 行数 | 可用 | 可用率 |")
    emit("|---|---|---|---|")
    for p in paths:
        d = per_shard[p]
        emit(f"| `{p.rsplit('/', 1)[-1]}` | {d['n']} | {d['ok']} | {pct(d['ok'], d['n'])} |")
    emit()

    # ---------------------------------------------------------------- 2 失败
    emit("## 2. 判定失败率（下游会静默丢弃这些行）")
    emit()
    emit(f"- 可用判定 **{usable}**（{pct(usable, total)}）")
    emit(f"- 判定失败 **{failed}**（{pct(failed, total)}）")
    if err_kinds:
        emit()
        emit("| 失败原因 | 条数 | 说明 |")
        emit("|---|---|---|")
        why = {
            "video_decode": "视频损坏/解码失败——重跑无用，属 H 类工程问题",
            "json_parse": "模型没吐出合法 JSON——重跑或缩短 prompt 有救",
            "oom": "显存不足——减少 `--frames` 或 `--max-side` 后重跑",
            "other": "其他异常——查分片日志 Traceback",
        }
        for k, n in err_kinds.most_common():
            emit(f"| {k} | {n} | {why.get(k, '')} |")
    emit()
    emit(f"- 另有 **{unreadable_decode}** 条解码失败但仍带 `action=drop`——"
         "**它们算进了「可用」，实际是 H 类工程问题**，本机 S1 是 0/10000，"
         "这里若非 0 说明 S2 的解码路径比 S1 严（帧数/后端不同）")
    emit(f"- 以及 **{unreadable_model}** 条模型判 `video_readable=false`"
         "（画面黑屏/花屏/纯色，是真实数据问题）")
    emit()

    # ---------------------------------------------------------------- 3 字段
    emit("## 3. 字段完整性（在可用判定内）")
    emit()
    emit("| 字段 | 有值 | 覆盖率 |")
    emit("|---|---|---|")
    for k in REQUIRED_FIELDS:
        emit(f"| `{k}` | {field_present[k]} | {pct(field_present[k], usable)} |")
    emit()
    emit("> `trans_score` 覆盖率尤其要看：`DATA_ISSUES.md` §4 决策 1（xling 阈值去留）"
         "的全部差率都建立在它上面，缺值会被下游当成「不坏」而稀释差率。")
    emit()
    bz_ok = sum(n for k, n in better_zh_counts.items()
                if isinstance(k, int) and k >= 1)
    emit(f"- `better_zh` 至少 1 条：{bz_ok}（{pct(bz_ok, usable)}）；"
         f"分布 {dict(sorted((k, v) for k, v in better_zh_counts.items() if isinstance(k, int)))}")
    emit("  （这批就是 v0.2 RECAPTION 桶的现成产物，空得太多说明画面普遍不可辨或 prompt 被截断）")
    emit()

    # ---------------------------------------------------------------- 4 分布
    emit("## 4. 分布合理性（防「判定塌缩」）")
    emit()
    emit("| action | 条数 | 占比 |")
    emit("|---|---|---|")
    for a in ACTIONS + [k for k in action_dist if k not in ACTIONS]:
        if action_dist.get(a):
            emit(f"| {a} | {action_dist[a]} | {pct(action_dist[a], usable)} |")
    emit()
    emit("| 分数 | " + " | ".join(str(s) for s in range(6)) + " | null |")
    emit("|---|" + "---|" * 7)
    for k in SCORE_FIELDS:
        row = [str(score_dist[k].get(s, 0)) for s in range(6)]
        row.append(str(score_dist[k].get(None, 0)))
        emit(f"| `{k}` | " + " | ".join(row) + " |")
    emit()
    emit(f"- `granularity`：{dict(gran_dist.most_common())}")
    emit()
    top_action, top_n = (action_dist.most_common(1) or [("", 0)])[0]
    emit(f"- 最大 action 占比：**{top_action} {pct(top_n, usable)}**")
    emit()

    # ---------------------------------------------------------------- 5 分层
    emit("## 5. 各层可用量（稀有层被失败吃掉就报不出该层结论）")
    emit()
    emit("| stratum | 总数 | 可用 | 可用率 |")
    emit("|---|---|---|---|")
    for s, n in stratum_total.most_common():
        emit(f"| {s} | {n} | {stratum_ok[s]} | {pct(stratum_ok[s], n)} |")
    emit()

    # ------------------------------------------------ 6 维度独立性 + 自洽性
    emit("## 6. 三个分数真的独立吗 / action 与分数自洽吗")
    emit()
    emit("rubric 要求 `en_score`（英文 vs 画面）、`zh_score`（中文 vs 画面）、"
         "`trans_score`（**只比中英两条文本**，不看画面）**独立打分**。"
         "前两个天然相关（同一段视频、互为译文），但 `trans_score` 量的是"
         "**另一回事**——它若与 `en_score` 高度同值，说明模型是照抄而非独立评估，"
         "那么「翻译坏没坏」这件事其实**没有被测量过**。")
    emit()
    if n_scored:
        emit("| 两两关系 | 完全同值 | 占比 | 平均绝对差 |")
        emit("|---|---|---|---|")
        for a, b in (("en=zh", "en-zh"), ("en=trans", "en-trans"),
                     ("zh=trans", "zh-trans")):
            emit(f"| `{a}` | {eq[a]} | {pct(eq[a], n_scored)} | "
                 f"{absdiff[b] / n_scored:.2f} |")
        emit(f"| **三者全同** | {eq['all three']} | {pct(eq['all three'], n_scored)} | — |")
        emit()
        emit("| action | 条数 | 均 en | 均 zh | 均 trans | en≤2 | zh≤2 | trans≤2 |")
        emit("|---|---|---|---|---|---|---|---|")
        for a in ACTIONS + [k for k in per_action if k not in ACTIONS]:
            d = per_action.get(a)
            if not d or not d["n"]:
                continue
            n = d["n"]
            emit(f"| {a} | {n} | {d['en'] / n:.2f} | {d['zh'] / n:.2f} | "
                 f"{d['tr'] / n:.2f} | {pct(d['en_bad'], n)} | "
                 f"{pct(d['zh_bad'], n)} | {pct(d['tr_bad'], n)} |")
        emit()
        emit("> 这张表是**读 rubric 边界**用的：`retranslate` 那行的 `trans≤2` 该明显"
             "高于 `keep` 行，`recaption` 那行的 `en≤2` 该明显高于其他行。"
             "若各行数字差不多，说明模型是按别的东西在选 action，"
             "**`action_dist` 里那个最大占比就不能当结论用**。")
        emit()
        if conflicts:
            emit("**action 与自身分数矛盾的记录**（同一条记录，S3 按分数算阈值、"
                 "`apply_buckets.py` 按 action 分流，矛盾即两半流水线各说各话）：")
            emit()
            emit("| 矛盾 | 条数 | 占该 action |")
            emit("|---|---|---|")
            for k, n in conflicts.most_common():
                act = k.split("=")[1].split(" ")[0]
                emit(f"| {k} | {n} | {pct(n, action_n.get(act, 0))} |")
        else:
            emit("- 未发现 action 与分数矛盾的记录。")
    else:
        emit("- （没有三项分数齐全的记录，跳过）")
    emit()

    # ---------------------------------------------------------------- 判定门
    emit("## 结论")
    emit()
    problems = []   # 拦住流程
    notes = []      # 不拦，但必须读到（多半是真实数据问题而非流程问题）
    fail_rate = failed / total if total else 1.0
    if fail_rate > args.max_fail_rate:
        problems.append(f"判定失败率 {fail_rate:.1%} > {args.max_fail_rate:.0%}"
                        "——先看第 2 节的失败原因表，能救的救（json_parse / oom）再重跑 `s2`")
    if dupes:
        problems.append(f"有 {dupes} 行重复 video，先去重再跑 S3")
    if usable and field_present["trans_score"] / usable < 0.9:
        problems.append("`trans_score` 覆盖率 <90%，决策 1 的差率不可信")
    if usable and top_n / usable > 0.9:
        problems.append(f"`{top_action}` 占了 {pct(top_n, usable)}，判定接近塌缩，"
                        "S3 的分位校准会失去对比度——先人工看十几条确认不是真的这么均质")
    for k in SCORE_FIELDS:
        nz = sum(v for kk, v in score_dist[k].items() if kk is not None)
        if nz:
            top = max((v for kk, v in score_dist[k].items() if kk is not None))
            if top / nz > 0.9:
                problems.append(f"`{k}` 有 >90% 落在同一个分值上，该维度无区分度")
    gz = sum(gran_dist.values())
    if gz and max(gran_dist.values()) / gz > 0.9:
        top_g = gran_dist.most_common(1)[0]
        notes.append(f"`granularity` 有 {pct(top_g[1], gz)} 判为 `{top_g[0]}`。"
                     "E 类（粒度粗）在 InternVid 上本就是主症状、且已被 S1 独立佐证"
                     "（`xling` 动态范围窄的同一病因），**大概率是真的**——"
                     "但该维度因此没有区分度，别再拿它做分桶特征")
    if n_scored:
        if eq["en=trans"] / n_scored > 0.6:
            problems.append(
                f"`en_score` 与 `trans_score` 有 {pct(eq['en=trans'], n_scored)} "
                "完全同值——两者量的是不同的东西（画面对齐 vs 文本忠实），"
                "如此同步说明是 halo/照抄而非独立评估。**决策 1（xling 阈值）的差率"
                "全部建立在 `trans_score` 上，这一项不成立则那个决策无从裁起**；"
                "先在 §6 表里看 `retranslate` 行的 `trans≤2` 是否真的高于 `keep` 行，"
                "再靠裁判档/人工抽检定夺")
        for k, n in conflicts.items():
            act = k.split("=")[1].split(" ")[0]
            base = action_n.get(act, 0)
            if base and n / base > 0.3:
                problems.append(f"{pct(n, base)} 的 `{act}` 记录与自身分数矛盾（{k}）"
                                "——S3 按分数、`apply_buckets.py` 按 action，两半会打架")
    if notes:
        emit("📌 **不拦流程，但结论里必须写进去**：")
        for p in notes:
            emit(f"- {p}")
        emit()
    if problems:
        emit("⚠️ **先处理这些再进下一步**：")
        for p in problems:
            emit(f"- {p}")
    else:
        emit("✅ 判定可用。下一步（`DATA_ISSUES.md` §1「下一步做什么」）：")
        emit("1. `bash tools/data_audit/run_v100.sh xling-check` —— 裁决决策 1，阻塞 S3")
        emit("2. 裁判档 500 条（`SETUP_JUDGE_RELAY.md`）+ Gradio 人工抽检 300，两者可并行")
        emit("3. 两个决策都裁完 → `bash tools/data_audit/run_v100.sh s3`")
    emit()

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[written] {args.out_md}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
