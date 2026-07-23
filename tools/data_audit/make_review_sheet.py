#!/usr/bin/env python3
"""Build (and later score) human-review sheets for the audit's open questions.

The 10k audit left three things measured by nobody but the 8B itself. Each needs
human eyes on a small sample, and each needs a *different* sheet:

  --kind better-zh   Are the generated replacement captions actually better?
                     This is the highest-stakes gap: better_zh goes straight
                     into v0.2 `captions` (apply_buckets.py:98) having never
                     been evaluated, and ~70% of the RECAPTION bucket is
                     records whose original caption was fine. If the rewrites
                     hallucinate, v0.2 is a net regression.
                     Stratified: half RECAPTION (will be rewritten), half KEEP
                     (control -- if rewrites beat originals here too, the
                     bucket boundary is too conservative).

  --kind conflicts   The 351 records where the text-only pass measured the
                     translation as fine (trans_score>=3) but S0 flags said
                     "retranslate". This is the population behind the 10x gap
                     between S0 language-contamination (3.3%) and measured
                     `untranslated` (0.3%). Decides which detector to believe.

  --kind random      Plain random sample for the 300-record spot check: does
                     the single quality score mean anything? (What does a
                     score-3 clip actually look like?)

Sheets are TSV -- open in Excel/VSCode, fill the blank columns, save. Then:

    python tools/data_audit/make_review_sheet.py --score <filled.tsv>

prints the rates. Blank rows are ignored, so partial reviews are fine.

Deps: stdlib only. Videos are found by `video` / `abs_path` in the Gradio tool.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_audit import resolve_video_path  # noqa: E402 —— 审查时真正解码成功的那个解析器

# Columns the human fills in. Kept to closed vocabularies so scoring is
# unambiguous -- free-text goes in `note`.
BLANKS = {
    "better-zh": [
        ("准确性", "better/same/worse"),      # vs the original zh, against the video
        ("具体性", "better/same/worse"),      # information content
        ("幻觉", "y/n"),                      # says something not in the video
        ("note", ""),
    ],
    "conflicts": [
        ("谁对", "s0/vlm/unclear"),           # s0=really contaminated, vlm=translation is fine
        ("note", ""),
    ],
    "random": [
        ("同意打分", "y/n"),                  # does the quality score match your read
        ("note", ""),
    ],
    "merge": [
        ("幻觉", "y/n"),                      # fused caption states something no candidate did
        ("更全", "y/n"),                      # fused is more complete than the best single candidate
        ("note", ""),
    ],
}

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


def load(pattern):
    if not pattern:
        return []
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no files match {pattern!r} (quote the glob)")
    return list(iter_jsonl(paths))


def build(args):
    sample = {r["video"]: r for r in load(args.sample)}

    n_missing = [0]

    def vpath(vid):
        """能被播放器/Gradio 直接打开的路径。复用 vlm_audit.resolve_video_path ——
        它会在 video_root/<video>、剥前缀、abs_path 等候选里返回**第一个真实存在**的，
        正是审查时成功解码这些视频用的同一套逻辑（简化版会盲拼一个多半不存在的路径）。"""
        rec = sample.get(vid) or {}
        pth = resolve_video_path(args.video_root, rec, vid)
        if not (pth and os.path.exists(pth)):
            n_missing[0] += 1
        return pth
    if not sample:
        sys.exit("--sample is required to build a sheet (it carries zh/en)")
    verdicts = {r["video"]: (r.get("verdict") or {}) for r in load(args.verdicts)} \
        if args.verdicts else {}
    trans = {r["video"]: (r.get("verdict") or {}) for r in load(args.trans)} \
        if args.trans else {}
    merged = {r["video"]: (r.get("verdict") or {}) for r in load(args.merged)} \
        if args.merged else {}
    buckets = {r["video"]: r.get("bucket") for r in load(args.buckets)} \
        if args.buckets else {}

    rng = random.Random(args.seed)
    rows, cols = [], []

    if args.kind == "better-zh":
        if not verdicts:
            sys.exit("--verdicts required for --kind better-zh (better_zh lives there)")
        if not buckets:
            sys.exit("--buckets required for --kind better-zh (for the stratification)")
        pools = {"RECAPTION": [], "KEEP": []}
        for vid, b in buckets.items():
            if b in pools and (verdicts.get(vid, {}).get("better_zh")):
                pools[b].append(vid)
        half = max(1, args.n // 2)
        picked = []
        for b, pool in pools.items():
            rng.shuffle(pool)
            take = pool[:half]
            picked += [(v, b) for v in take]
            if len(take) < half:
                print(f"[warn] {b} 只有 {len(take)} 条可选（想要 {half}）", file=sys.stderr)
        rng.shuffle(picked)          # 混排，避免评审者按桶产生预期
        cols = ["video", "video_path", "bucket", "sig_zh", "原zh", "候选1", "候选2", "候选3"]
        for vid, b in picked:
            s = sample.get(vid, {})
            cands = [c for c in (verdicts[vid].get("better_zh") or []) if isinstance(c, str)]
            cands += [""] * (3 - len(cands))
            rows.append([vid, vpath(vid), b,
                         f'{(s.get("s1") or {}).get("sig_zh_mean", "")}',
                         s.get("zh") or "", *cands[:3]])

    elif args.kind == "conflicts":
        if not trans:
            sys.exit("--trans required for --kind conflicts (s2_trans.*.jsonl)")
        pool = []
        for vid, s in sample.items():
            ts = trans.get(vid, {}).get("trans_score")
            if ts is not None and ts > 2 and (set(s.get("s0_flags") or []) & RETRANSLATE_FLAGS):
                pool.append(vid)
        print(f"[info] 冲突记录总数 {len(pool)}", file=sys.stderr)
        rng.shuffle(pool)
        cols = ["video", "video_path", "S0旗标", "trans_score", "原zh", "英文en"]
        for vid in pool[:args.n]:
            s = sample[vid]
            rows.append([vid, vpath(vid),
                         ",".join(sorted(set(s.get("s0_flags") or []) & RETRANSLATE_FLAGS)),
                         trans[vid].get("trans_score"),
                         s.get("zh") or "", s.get("en") or ""])

    elif args.kind == "random":
        if not verdicts:
            sys.exit("--verdicts required for --kind random")
        pool = [v for v in sample if v in verdicts]
        rng.shuffle(pool)
        cols = ["video", "video_path", "zh分", "action", "粒度", "原zh", "英文en"]
        for vid in pool[:args.n]:
            s, v = sample[vid], verdicts[vid]
            rows.append([vid, vpath(vid), v.get("zh_score"), v.get("action"),
                         v.get("granularity"), s.get("zh") or "", s.get("en") or ""])

    elif args.kind == "merge":
        if not (verdicts and merged):
            sys.exit("--verdicts（候选 better_zh）与 --merged（s2_merged.*.jsonl 的 merged_zh）都要给")
        pool = []
        for vid, v in verdicts.items():
            cs = [c for c in (v.get("better_zh") or []) if isinstance(c, str) and c.strip()]
            m = (merged.get(vid) or {}).get("merged_zh")
            if len(cs) >= 2 and isinstance(m, str) and m.strip():
                pool.append(vid)
        print(f"[info] 可评融合记录（≥2候选且有融合稿）总数 {len(pool)}", file=sys.stderr)
        rng.shuffle(pool)
        cols = ["video", "video_path", "候选1", "候选2", "候选3", "融合稿"]
        for vid in pool[:args.n]:
            cs = [c for c in (verdicts[vid].get("better_zh") or []) if isinstance(c, str)]
            cs += [""] * (3 - len(cs))
            rows.append([vid, vpath(vid), *cs[:3],
                         (merged[vid].get("merged_zh") or "")])

    blanks = BLANKS[args.kind]
    header = cols + [f"{c}[{h}]" if h else c for c, h in blanks]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(list(r) + [""] * len(blanks))
    # 配套 jsonl：Gradio 按 jsonl 加载子集（§6 的 quality_issue_candidates.jsonl 即此用法）。
    # 只有 TSV 的话，评审者能看到文字却调不出画面，而这三张表全都要对着画面判。
    jl = os.path.splitext(args.out)[0] + ".jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for r in rows:
            rec = dict(zip(cols, r))
            src = sample.get(rec["video"]) or {}
            f.write(json.dumps({**{k: v for k, v in src.items() if k != "s1"}, **rec},
                               ensure_ascii=False) + "\n")
    found = len(rows) - n_missing[0]
    print(f"[written] {args.out} —— {len(rows)} 行待评")
    print(f"[written] {jl} —— 同一批，供 Gradio 加载看画面")
    print(f"[videos]  可定位 {found}/{len(rows)}", file=sys.stderr)
    if found == 0:
        print("  ⚠️ 一个视频都定位不到！检查 --video-root 是否指向真实的视频目录，"
              "以及 s2_sample.jsonl 里的 video/abs_path 字段。"
              "\n     诊断： python3 -c \"import json;r=json.loads(open('%s').readline());"
              "print(r.get('video'),'|',r.get('abs_path'))\"" % args.sample, file=sys.stderr)
    print(f"  评审列：{'  '.join(c + ('(' + h + ')' if h else '') for c, h in blanks)}")
    print(f"  填完后： python {sys.argv[0]} --score {args.out}")


def score(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit("表是空的")
    # 评审列 = 表头里带 [ ] 的，或名为 note 的
    review_cols = [c for c in rows[0] if c and ("[" in c or c == "note")]
    filled = [r for r in rows if any((r.get(c) or "").strip()
                                     for c in review_cols if c != "note")]
    print(f"# 抽检结果：{path}")
    print(f"\n- 总行数 {len(rows)}，**已填 {len(filled)}**"
          f"（{len(filled)/len(rows):.0%}）")
    if not filled:
        sys.exit("\n还没有任何一行被填写。")
    for c in review_cols:
        if c == "note":
            continue
        vals = Counter((r.get(c) or "").strip().lower() for r in filled
                       if (r.get(c) or "").strip())
        if not vals:
            continue
        n = sum(vals.values())
        print(f"\n## {c}（n={n}）")
        for k, v in vals.most_common():
            print(f"- `{k}`：{v}（{v/n:.1%}）")
        # 两个决策门，直接在这里判
        if c.startswith("幻觉"):
            bad = vals.get("y", 0) + vals.get("yes", 0)
            gate = bad / n
            print(f"\n> **幻觉率 {gate:.1%}** —— "
                  + ("⚠️ **>10%，`better_zh` 不能直接入库**，落地前须先过滤或换模型重生成。"
                     if gate > 0.10 else
                     "✅ ≤10%，可以入库（仍建议在 `apply_buckets.py` 后抽查一次成品）。"))
        if c.startswith("准确性"):
            worse = vals.get("worse", 0) / n
            better = vals.get("better", 0) / n
            print(f"\n> 更准确 {better:.1%} vs 更差 {worse:.1%} —— "
                  + ("⚠️ **变差的比变好的多，重写是净损失**，RECAPTION 方案要重新考虑。"
                     if worse > better else
                     "✅ 重写方向正确。"))
        if c.startswith("更全"):
            better = (vals.get("y", 0) + vals.get("yes", 0)) / n
            print(f"\n> **更全率 {better:.1%}** —— "
                  + ("✅ >50%，融合确比最佳单条更全（信息增益成立，值得进 §6.3 消融证实）。"
                     if better > 0.5 else
                     "⚠️ ≤50%，融合相对单条的**增益有限**——先看幻觉率是否达标，"
                     "再据 §6.3 消融决定是否值得这份算力。"))
        if c.startswith("谁对"):
            s0 = vals.get("s0", 0) / n
            print(f"\n> 判 S0 对的占 {s0:.1%} —— "
                  + ("S0 旗标是对的，**文本-only rubric 对残留英文不够敏感**，"
                     "「语言污染 3.3%」成立，需给翻译重判补一条残留英文检查。"
                     if s0 > 0.5 else
                     "实测判定是对的，**S0 把「中英混杂但主体已翻」也算成了污染**，"
                     "「语言污染 3.3%」应下修，S0 旗标定义需收紧。"))
    notes = [(r.get("video"), r.get("note")) for r in filled if (r.get("note") or "").strip()]
    if notes:
        print(f"\n## 备注（{len(notes)} 条）")
        for v, t in notes[:20]:
            print(f"- `{v}`：{t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=sorted(BLANKS))
    ap.add_argument("--sample", default=None, help="s2_sample.jsonl（提供 zh/en/s0_flags）")
    ap.add_argument("--verdicts", default=None, help="s2_verdicts glob（提供 better_zh/分数）")
    ap.add_argument("--trans", default=None, help="s2_trans glob（提供实测 trans_score）")
    ap.add_argument("--merged", default=None,
                    help="s2_merged glob（--kind merge 用：提供 merged_zh 融合稿）")
    ap.add_argument("--buckets", default=None, help="buckets.jsonl（提供分桶，用于分层）")
    ap.add_argument("-n", type=int, default=50, help="抽多少条（better-zh 会平分到两个桶）")
    ap.add_argument("--seed", type=int, default=0, help="固定随机种子，保证可复现")
    ap.add_argument("--out", default=None, help="输出 TSV（同时写同名 .jsonl 供 Gradio）")
    ap.add_argument("--video-root", default="", help="视频根目录，用于拼 video_path")
    ap.add_argument("--score", default=None, help="改为读取已填好的 TSV 并统计")
    args = ap.parse_args()

    if args.score:
        return score(args.score)
    if not (args.kind and args.out):
        ap.error("建表需要 --kind 与 --out（或用 --score 统计已填好的表）")
    build(args)


if __name__ == "__main__":
    main()
