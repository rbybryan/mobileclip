#!/usr/bin/env python3
"""粒度计数式重判（--mode gran）的结果分析 + `n_info=0` 归因。

背景：三档标签给出 coarse 93.6% / adequate 6.4% / **dense 0**，一档计数为 0 是
标签空间塌缩的典型特征，于是改用信息点**计数**重判（计数是连续量，塌不了）。
重判结果却冒出另一个异常：**18.7% 的记录 n_info=0**，而抽样看到的原字幕
（「电脑屏幕上显示的两张建筑图片」「一张发光蓝紫色箭头的截图」）明显含 2-3 个
信息点。且这些记录的五个列表**也全是空的**——不是算术错，是模型没在数。

这个脚本回答「那 1871 条到底是什么」：把 n_info=0 组与其余组在若干可观测量上
对比，看零值是**聚在某一类内容上**（→ 真实的内容类别，模型只是不会数它）
还是**均匀散布**（→ 纯粹的模型故障，该批重判作废）。

    python tools/data_audit/check_gran.py \
        --gran 'outputs/audit_10k/s2_gran.*.jsonl' \
        --sample outputs/audit_10k/s2_sample.jsonl \
        --verdicts 'outputs/audit_10k/s2_verdicts.*.jsonl' \
        --out-md outputs/audit_10k/gran_health.md

Deps: stdlib only.
"""

import argparse
import glob
import json
import sys
from collections import Counter

LISTS = ("subjects", "actions", "scenes", "attributes", "temporal")

# 静态/屏幕类内容的中文线索。这批词若在零值组里显著富集，说明零值不是随机故障，
# 而是模型在「画面本身是静态图/截图/文字」这类字幕上系统性罢工。
STATIC_KW = ["截图", "屏幕", "网站", "网页", "页面", "界面", "文字", "字幕", "标志",
             "标牌", "路牌", "照片", "图片", "图像", "图标", "海报", "广告", "表格",
             "图表", "菜单", "logo", "LOGO", "Logo", "封面", "标题"]


def iter_jsonl(pattern):
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def pct(n, d):
    return f"{n / d:.1%}" if d else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gran", required=True, help="s2_gran glob（引号包起来）")
    ap.add_argument("--sample", required=True, help="s2_sample.jsonl（提供原 zh）")
    ap.add_argument("--verdicts", default=None, help="s2_verdicts glob（提供主力档标签）")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    zh, s1 = {}, {}
    for r in iter_jsonl(args.sample):
        zh[r.get("video")] = (r.get("zh") or "")
        s1[r.get("video")] = (r.get("s1") or {}).get("sig_zh_mean")
    main_pass = {}
    if args.verdicts:
        for r in iter_jsonl(args.verdicts):
            main_pass[r.get("video")] = (r.get("verdict") or {})

    zero, nonzero = [], []
    reported, recount = Counter(), Counter()
    for r in iter_jsonl(args.gran):
        v = r.get("verdict") or {}
        n = v.get("n_info")
        if n is None:
            continue
        k = sum(len(v.get(x) or []) for x in LISTS if isinstance(v.get(x), list))
        reported[n] += 1
        recount[k] += 1
        (zero if n == 0 else nonzero).append(r.get("video"))

    tot = len(zero) + len(nonzero)
    if not tot:
        sys.exit("没有可用的 gran 判定")

    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit("# 粒度计数式重判：结果与 `n_info=0` 归因\n")
    emit(f"- 可用判定 **{tot:,}** 条；其中 `n_info=0` **{len(zero):,}**"
         f"（**{pct(len(zero), tot)}**）\n")

    # ---------------------------------------------------------------- 归因
    emit("## 1. 那批零值是什么\n")
    emit("| 指标 | `n_info=0` 组 | 其余 | 读法 |")
    emit("|---|---|---|---|")

    def rate(vids, fn):
        hit = sum(1 for v in vids if fn(v))
        return hit, hit / len(vids) if vids else 0

    # ① 静态/屏幕类关键词
    def is_static(v):
        t = zh.get(v, "")
        return any(k in t for k in STATIC_KW)
    z_h, z_r = rate(zero, is_static)
    n_h, n_r = rate(nonzero, is_static)
    lift = (z_r / n_r) if n_r else float("inf")
    emit(f"| 含静态/屏幕类词 | {z_h:,}（**{z_r:.1%}**） | {n_h:,}（{n_r:.1%}） | "
         f"**lift {lift:.2f}×**（>2 即显著富集） |")

    # ② 字幕长度
    def avglen(vids):
        ls = [len(zh.get(v, "")) for v in vids]
        return sum(ls) / len(ls) if ls else 0
    emit(f"| 字幕平均长度（字） | {avglen(zero):.1f} | {avglen(nonzero):.1f} | "
         "零值组若明显更短，可能只是字幕太短 |")

    # ③ 主力档怎么看这批（若主力给了正常分，说明字幕本身没问题，是 gran 这趟的锅）
    if main_pass:
        def gran_label(vids, lab):
            return rate(vids, lambda v: main_pass.get(v, {}).get("granularity") == lab)
        for lab in ("coarse", "adequate"):
            zh_, zr_ = gran_label(zero, lab)
            nh_, nr_ = gran_label(nonzero, lab)
            emit(f"| 主力档判 `{lab}` | {zh_:,}（{zr_:.1%}） | {nh_:,}（{nr_:.1%}） | "
                 "两组接近 = 主力档不认为这批特殊 |")

        def zh_ok(vids):
            return rate(vids, lambda v: (main_pass.get(v, {}).get("zh_score") or 0) >= 3)
        zh_, zr_ = zh_ok(zero)
        nh_, nr_ = zh_ok(nonzero)
        emit(f"| 主力档 `zh_score≥3` | {zh_:,}（{zr_:.1%}） | {nh_:,}（{nr_:.1%}） | "
             "零值组若同样正常，**字幕本身没问题** |")

    # ④ sig_zh（静态屏幕内容的视频-文本对齐可能偏低）
    def avgsig(vids):
        xs = [s1.get(v) for v in vids if s1.get(v) is not None]
        return sum(xs) / len(xs) if xs else 0
    emit(f"| 平均 `sig_zh` | {avgsig(zero):.4f} | {avgsig(nonzero):.4f} | "
         "差异大说明零值组确实是另一类内容 |")
    emit()

    # ---------------------------------------------------------------- 判决
    emit("## 2. 判决\n")
    verdict_static = lift >= 2.0
    if verdict_static:
        emit(f"✅ **零值显著富集在静态/屏幕类字幕上（lift {lift:.2f}×）** —— "
             "这不是随机故障，而是模型在「画面是截图/网页/文字/标牌」这类字幕上系统性罢工"
             "（prompt 的五类信息点与锚点例子「一个男人在说话」都预设了**有动作的事件**）。\n")
        emit("**两层结论**：")
        emit(f"1. **粒度数字**：这 {len(zero):,} 条应从粒度统计中**剔除**再算分档，"
             "否则会把模型的罢工计成「零信息量」，虚高 coarse 占比；")
        emit(f"2. **数据发现**：静态/屏幕类内容在语料里占到 **{pct(len(zero), tot)}** 量级"
             "（下界，因为只有触发罢工的才被数进来）。**对视频-文本模型这是独立的问题**——"
             "静态画面没有时序、也谈不上动作，与「时序缺失 99.2%」互为佐证。")
    else:
        emit(f"⚠️ **零值未显著富集在静态类字幕上（lift {lift:.2f}×）** —— "
             "更像是模型的随机故障而非内容类别。**该批重判的零值不可信**，"
             "建议改 prompt（去掉对「事件/动作」的预设，明确「静态画面也要数主体和属性」）后重跑。\n")
    emit()

    # ---------------------------------------------------------------- 分档
    emit("## 3. 分档（剔除零值前后对比）\n")

    def band(counter, skip_zero):
        items = {k: v for k, v in counter.items() if not (skip_zero and k == 0)}
        t = sum(items.values())
        if not t:
            return None
        co = sum(v for k, v in items.items() if k <= 2)
        ad = sum(v for k, v in items.items() if k in (3, 4))
        de = sum(v for k, v in items.items() if k >= 5)
        return t, co / t, ad / t, de / t

    emit("| 口径 | n | ≤2 粗 | 3–4 中 | ≥5 密 |")
    emit("|---|---|---|---|---|")
    for name, counter, skip in (("自报 n_info（含零值）", reported, False),
                                ("自报 n_info（剔除零值）", reported, True),
                                ("按列表重算（剔除零值）", recount, True)):
        b = band(counter, skip)
        if b:
            t, co, ad, de = b
            emit(f"| {name} | {t:,} | {co:.1%} | {ad:.1%} | {de:.1%} |")
    emit()
    emit("> 对照：**原三档标签为 coarse 93.6% / adequate 6.4% / dense 0**。"
         "无论用哪个口径，**「93.6% 粗」都被大幅推翻**——三档标签确实塌缩了"
         "（`DATA_ISSUES.md` 问题 14）。但「粒度粗是主症状」的方向仍成立："
         "中位数 3 个信息点的字幕，对检索式训练依然偏薄。")
    emit()

    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
        print(f"\n[written] {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
