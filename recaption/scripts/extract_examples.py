#!/usr/bin/env python3
"""Pull REAL examples out of the audit outputs, one block per finding.

docs/caption-quality-report.md states rates; a reader cannot check a rate. This emits the
actual records behind each claim so the findings can be spot-checked -- and so
nobody is tempted to illustrate a data-quality report with invented captions.

Every example is verbatim from the audit outputs. Nothing here is generated or
paraphrased; the only editorial act is which records get picked, and that rule
is printed next to each block.

Selection is deterministic (no RNG): each block takes the top-N by a stated
ordering, so re-running on the same inputs reproduces the same appendix.

Usage (on the data machine, from repo root):
    python recaption/scripts/extract_examples.py            # writes the appendix
    python recaption/scripts/extract_examples.py -n 5       # more per block

Reads $OUT (default outputs/audit_10k) + $DATA. Deps: stdlib only.
"""

import argparse
import glob
import json
import os
import sys

OUT = os.environ.get("OUT", "outputs/audit_10k")
DATA = os.environ.get(
    "DATA", "/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl")

# S0 flags whose *nominal* meaning is language contamination. Kept here so the
# "false positive" block can show what the pre-fix rule was firing on.
RATIO_FLAGS = {"zh_low_cjk_ratio", "zh_mostly_english"}


def iter_jsonl(paths):
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            continue


def load():
    src = {}
    for rec in iter_jsonl([DATA]):
        v = rec.get("video")
        if v:
            src[v] = (rec.get("raw_caption_zh_translated") or "",
                      rec.get("caption") or "")
    verd, trans, s1 = {}, {}, {}
    for rec in iter_jsonl(sorted(glob.glob(f"{OUT}/s2_verdicts.*.jsonl"))):
        if rec.get("video"):
            verd[rec["video"]] = rec.get("verdict") or {}
            if rec.get("s1"):
                s1[rec["video"]] = rec["s1"]
    for rec in iter_jsonl(sorted(glob.glob(f"{OUT}/s2_trans.*.jsonl"))):
        if rec.get("video"):
            trans[rec["video"]] = rec.get("verdict") or {}
    for rec in iter_jsonl(sorted(glob.glob(f"{OUT}/s1_scores.*.jsonl"))):
        if rec.get("video"):
            s1.setdefault(rec["video"], {}).update(rec)
    flags = {}
    for rec in iter_jsonl([f"{OUT}/s0_flags.jsonl"]):
        if rec.get("video"):
            flags[rec["video"]] = rec.get("flags") or []
    if not src:
        sys.exit(f"没读到源数据：{DATA}\n（在数据机上跑，或 export DATA=...）")
    if not verd and not trans:
        sys.exit(f"没读到判定：{OUT}/s2_verdicts.*.jsonl —— 先跑 S2")
    return src, verd, trans, s1, flags


def short(t, n=None):
    """Collapse whitespace. By default returns the FULL caption (no truncation)
    so the appendix shows verbatim text a reader can check against the video --
    a data-quality report should not hide the very strings it is judging. Pass n
    to cap length only where a runaway cell would break table layout."""
    t = " ".join((t or "").split())
    if n is not None and len(t) > n:
        return t[:n] + "…"
    return t


def clip(v):
    """Last path segment, which carries the youtube id + timestamps."""
    return (v or "").rsplit("/", 1)[-1]


def block(f, title, rule, rows, cols):
    f.write(f"\n### {title}\n\n**选取规则**：{rule}\n\n")
    if not rows:
        f.write("> ⚠️ 本批数据中没有符合条件的记录。\n")
        return
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * len(cols) + "\n")
    for r in rows:
        f.write("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--per-block", type=int, default=3)
    ap.add_argument("--out", default="docs/caption-quality-examples.md")
    args = ap.parse_args()
    N = args.per_block
    src, verd, trans, s1, flags = load()

    def sig(v):
        s = s1.get(v) or {}
        return s.get("sig_zh_mean")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 附录：真实例子\n\n")
        f.write("> **本文件由 `recaption/scripts/extract_examples.py` 生成，"
                "请勿手改**（重跑会覆盖）。\n>\n"
                "> 每条都是审查产物里的**原始记录**，逐字摘录、未经改写。"
                "唯一的编辑行为是「挑哪几条」，而挑选规则印在每个小节标题下，"
                "且是确定性的（按给定顺序取前 N 条，无随机）。\n>\n"
                f"> 配套结论见 [`caption-quality-report.md`](caption-quality-report.md)。"
                f"样本 = {len(src):,} 条 ｜ 每节 {N} 例。\n")

        f.write("\n---\n\n## 一、英文源 caption 的问题\n")

        # 1. source misalignment -- lowest sig_zh among VLM-confirmed bad
        rows = []
        cand = [(v, x) for v, x in verd.items()
                if x.get("en_score") is not None and x["en_score"] <= 2
                and sig(v) is not None]
        for v, x in sorted(cand, key=lambda kv: sig(kv[0]))[:N]:
            rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[1]),
                         short(src.get(v, ("", ""))[0]),
                         f"{sig(v):.3f}", x.get("en_score"),
                         ", ".join(x.get("en_issues") or []) or "—"])
        block(f, "1. 与画面不符（源头错）",
              "`en_score≤2` 且 `sig_zh` 最低的前 N 条 —— 即"
              "「VLM 与嵌入模型都认为最差」的记录，两个独立信号交叉确认",
              rows, ["clip", "英文源 caption", "中文译文", "sig_zh", "en分", "VLM 标注的问题"])

        # 2. temporal: video has order, caption says nothing
        rows = []
        cand = [(v, x) for v, x in verd.items()
                if x.get("temporal_order_relevant") is True
                and x.get("temporal_order_correct") is None]
        for v, x in sorted(cand, key=lambda kv: -(len(src.get(kv[0], ("", ""))[1])))[:N]:
            rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[1]),
                         short(src.get(v, ("", ""))[0])])
        block(f, "2. 视频有先后事件，字幕只字未提（时序缺失）",
              "`temporal_order_relevant=true` 且 `temporal_order_correct=null`，"
              "取英文最长的前 N 条（长句都不写顺序，最能说明问题）",
              rows, ["clip", "英文源 caption", "中文译文"])

        # 2b. temporal stated but wrong -- the whole population is tiny
        rows = []
        for v, x in list((v, x) for v, x in verd.items()
                         if x.get("temporal_order_correct") is False)[:N]:
            rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[1]),
                         short(src.get(v, ("", ""))[0])])
        block(f, "2b. 字幕写了顺序、但写反了",
              "`temporal_order_correct=false` —— **全库仅 8 条**，此处列出前 N 条。"
              "n 太小不可外推，仅作定性参考",
              rows, ["clip", "英文源 caption", "中文译文"])

        # 3. coarse: shortest captions
        rows = []
        cand = [(v, x) for v, x in verd.items() if src.get(v)]
        for v, x in sorted(cand, key=lambda kv: len(src[kv[0]][1]))[:N]:
            rows.append([f"`{clip(v)}`", short(src[v][1]), short(src[v][0]),
                         len(src[v][1].split()), x.get("granularity") or "—"])
        block(f, "3. 描述笼统（粒度粗）",
              "英文 caption **最短**的前 N 条 —— 用**长度**这个不依赖模型的客观量挑选，"
              "因为 `granularity` 那一维已作废（被质量分闸住，见 caption-quality-report §1.3）",
              rows, ["clip", "英文源 caption", "中文译文", "英文词数", "VLM 粒度判定"])

        # 4. error types
        for sub, (issue, label) in zip("abcd", (
                ("wrong_subject", "认错主体"),
                ("wrong_action", "认错动作"),
                ("hallucinated_detail", "编造细节"),
                ("ocr_or_watermark_read", "把水印/字幕当画面内容"))):
            rows = []
            cand = [(v, x) for v, x in verd.items()
                    if issue in (x.get("en_issues") or []) and sig(v) is not None]
            for v, x in sorted(cand, key=lambda kv: sig(kv[0]))[:N]:
                rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[1]),
                             f"{sig(v):.3f}", x.get("en_score")])
            block(f, f"4{sub}. 错误类型：{label}（`{issue}`）",
                  f"`en_issues` 含 `{issue}`，按 `sig_zh` 从低到高取前 N 条。"
                  "⚠️ 该标签出自已塌缩的合并 rubric，**看形态、不看比例**",
                  rows, ["clip", "英文源 caption", "sig_zh", "en分"])

        f.write("\n---\n\n## 二、中文译文 caption 的问题\n")

        # 5. real translation errors -- from the dedicated text-only pass
        rows = []
        cand = [(v, x) for v, x in trans.items()
                if x.get("trans_score") is not None and x["trans_score"] <= 2]
        for v, x in sorted(cand, key=lambda kv: kv[1].get("trans_score", 9))[:N]:
            rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[1]),
                         short(src.get(v, ("", ""))[0]),
                         x.get("trans_score"),
                         ", ".join(x.get("trans_issues") or []) or "—"])
        block(f, "5. 真实翻译错误（3.6% 的那批）",
              "**文本-only 重判**的 `trans_score≤2`，按分数升序取前 N 条。"
              "这一版判定不看画面、只比中英两条文本，是对翻译环节的专项测量",
              rows, ["clip", "英文源", "中文译文", "翻译分", "问题标签"])

        # 6. genuinely untranslated
        rows = []
        cand = [(v, x) for v, x in trans.items() if x.get("untranslated") is True]
        for v, x in cand[:N]:
            rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[1]),
                         short(src.get(v, ("", ""))[0])])
        block(f, "6. 真·未翻译（0.3%）",
              "文本-only 重判 `untranslated=true` —— **全库仅 33 条**",
              rows, ["clip", "英文源", "中文栏内容"])

        # 7. THE false positives -- fixed in problem 12
        rows = []
        cand = [v for v, fl in flags.items() if RATIO_FLAGS & set(fl)]
        for v in sorted(cand, key=lambda v: len(src.get(v, ("", ""))[0]))[:N]:
            zh, en = src.get(v, ("", ""))
            rows.append([f"`{clip(v)}`", short(zh), short(en),
                         ", ".join(sorted(RATIO_FLAGS & set(flags[v])))])
        block(f, "7. 语言污染的**误报**（3.3% → ~0.3% 的由来）",
              "**修复后**仍命中比例旗标的记录。修复前有 327 条，"
              "其中 15 条人工抽检 15/15 是正确译文（保留了品牌名/型号）",
              rows, ["clip", "中文译文", "英文源", "命中旗标"])

        f.write("\n> 📌 **修复前被误报的典型形态**（人工抽检的 15 条，"
                "逐字摘自 `review_conflicts.tsv`）：\n>\n"
                "> | 中文译文 | 英文源 | 为什么被误报 |\n> |---|---|---|\n"
                "> | `华硕 P8Z77-V BM 主板` | `asus p8z77-v bm motherboard` | "
                "CJK 占比 4/15=0.27 < 0.4 —— 而它连 ASUS 都译成了「华硕」 |\n"
                "> | `Windows 10 开始菜单` | `windows 10 start menu` | "
                "型号名不该翻，却拉低了 CJK 比 |\n"
                "> | `Sun Cool 官方网站` | `sun cool official site` | 同上 |\n")

        f.write("\n---\n\n## 三、工程类问题\n")

        rows = []
        cand = [(v, x) for v, x in verd.items() if x.get("video_readable") is False]
        for v, x in cand[:N]:
            rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[1]),
                         "空" if not (x.get("better_zh") or []) else "非空",
                         x.get("action") or "—"])
        block(f, "8. 画面本身不可用（黑屏/花屏/纯色，0.49%）",
              "`video_readable=false` —— 全库 49 条。"
              "注意 `better_zh` 应为空（模型看不到画面就写不出字幕），可作自洽性检查",
              rows, ["clip", "英文源 caption", "better_zh", "VLM 建议动作"])

        f.write("\n---\n\n## 四、v0.2 修复素材：`better_zh` 长什么样\n")
        rows = []
        cand = [(v, x) for v, x in verd.items()
                if (x.get("better_zh") or []) and x.get("zh_score") is not None
                and x["zh_score"] <= 2]
        for v, x in sorted(cand, key=lambda kv: sig(kv[0]) if sig(kv[0]) is not None
                           else 9)[:N]:
            cands = [c for c in x["better_zh"] if isinstance(c, str)]
            rows.append([f"`{clip(v)}`", short(src.get(v, ("", ""))[0]),
                         " ／ ".join(short(c) for c in cands[:3])])
        block(f, "9. RECAPTION 桶的候选字幕（原文 vs 重写）",
              "`zh_score≤2` 且 `sig_zh` 最低的前 N 条 —— 即最该被重写的那批。"
              "⚠️ **这些候选的质量从未被评估**（caption-quality-report §5 的 P0 硬门），"
              "此处仅供**肉眼预览形态**，不构成质量证据",
              rows, ["clip", "原中文字幕", "better_zh 候选（≤30 字）"])

        f.write("\n---\n\n"
                "> **怎么核对**：`clip` 列是 `youtube_id_起始时间_结束时间.mp4`，"
                "在 §6 的 Gradio 工具里搜这个名字即可对着画面看。\n")

    print(f"[written] {args.out}")
    print(f"  源数据 {len(src):,} 条 ｜ 判定 {len(verd):,} ｜ 翻译重判 {len(trans):,} "
          f"｜ S0 旗标 {len(flags):,}")


if __name__ == "__main__":
    main()
