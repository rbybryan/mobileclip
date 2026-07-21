#!/usr/bin/env python3
"""把 internvid_zh 10k 抽样 jsonl 转成训练用 anno 格式(10k 试点专用)。

输入:数据审查用的 jsonl(每行一条,字段名自动猜测,与
tools/data_audit/make_audit_sample.py 同一套 key 列表);
输出:repo 训练/评测代码期望的 anno json —— `[{"video": ..., "caption": ...}]`,
拆成 train 与 held-out eval 两份(eval 用于试点期的自检索评测)。

默认取英文字幕(文本塔是 CLIP 英文 BPE);--caption-field zh 取中文翻译。
同时打印共享字幕统计——量化 caption_idx 多正例修复在这批数据上的影响面。

用法(101vector 上):
    python tools/make_pilot_anno.py \
        --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
        --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
        --out-train anno_downstream/internvid_zh_10k_pilot.json \
        --out-eval anno_downstream/internvid_zh_10k_pilot_val.json
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

# 与 tools/data_audit/make_audit_sample.py 保持一致的字段名猜测表
VIDEO_KEYS = ["video", "video_path", "video_id", "clip", "clip_name", "filename"]
ZH_KEYS = ["caption_zh", "zh_caption", "caption_cn", "cn_caption", "translated_caption",
           "translation", "zh", "chinese_caption", "raw_caption_zh_translated",
           "caption_zh_translated"]
EN_KEYS = ["caption", "caption_en", "en_caption", "source_caption", "src_caption", "en"]

# 片段实际存放在 InternVId-*/… 的嵌套目录里,而 `video` 字段常常只是裸文件名,
# 真正定位文件的是记录里的 `abs_path`。与
# tools/data_audit/score_alignment.py: resolve_video_path() 保持同一套解析顺序。
CLIP_SUBPATH_RE = re.compile(r"InternVId[^/]*/.*")


def pick(rec, keys):
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def resolve_video(clips_root, rec, video):
    """定位这条记录的视频文件。返回 (命中的绝对/相对路径 or None, 试过的候选列表)。

    候选顺序与审查工具一致:clips_root/<video> → clips_root/<abs_path 里的
    InternVId-* 后缀> → 原始 abs_path。
    """
    cands = []
    if video:
        cands.append(os.path.join(clips_root, str(video)))
    ap = rec.get("abs_path")
    if ap:
        m = CLIP_SUBPATH_RE.search(str(ap))
        if m:
            cands.append(os.path.join(clips_root, m.group(0)))
        cands.append(str(ap))
    for c in cands:
        if os.path.isfile(c):
            return c, cands
    return None, cands


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="10k 抽样 jsonl")
    ap.add_argument("--clips-root", default=None,
                    help="视频根目录;给出时会剔除文件缺失的记录")
    ap.add_argument("--caption-field", choices=["en", "zh"], default="en")
    ap.add_argument("--video-key", default=None,
                    help="显式指定 video 字段名(默认按 VIDEO_KEYS 猜)")
    ap.add_argument("--caption-key", default=None,
                    help="显式指定 caption 字段名(默认按 --caption-field 猜)")
    ap.add_argument("--eval-split", type=int, default=1000,
                    help="held-out 评测集条数(0 = 不拆)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", default=None,
                    help="eval anno 输出路径(eval-split>0 时必填)")
    args = ap.parse_args()
    if args.eval_split > 0 and not args.out_eval:
        sys.exit("--eval-split > 0 时必须给 --out-eval")

    cap_keys = ([args.caption_key] if args.caption_key
                else (EN_KEYS if args.caption_field == "en" else ZH_KEYS))
    vid_keys = [args.video_key] if args.video_key else VIDEO_KEYS

    records = []
    n_parse = n_no_video = n_no_caption = n_missing = n_outside = 0
    first_keys, tried_samples = None, []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_parse += 1
                continue
            if first_keys is None:
                first_keys = sorted(rec.keys())
            video, caption = pick(rec, vid_keys), pick(rec, cap_keys)
            if not video:
                n_no_video += 1
                continue
            if not caption:
                n_no_caption += 1
                continue
            if args.clips_root:
                hit, cands = resolve_video(args.clips_root, rec, video)
                if hit is None:
                    n_missing += 1
                    if len(tried_samples) < 3:
                        tried_samples.append(cands)
                    continue
                # anno 里必须写**相对 clips_root** 的路径:训练时 dataset 会把它
                # 拼到 configs/data.py 的 data_root 上。
                rel = os.path.relpath(hit, args.clips_root)
                if rel.startswith(os.pardir):
                    n_outside += 1
                    if len(tried_samples) < 3:
                        tried_samples.append(cands)
                    continue
                video = rel
            records.append({"video": video, "caption": caption})

    if not records:
        print("没有可用记录。逐项计数:", file=sys.stderr)
        print(f"  JSON 解析失败      : {n_parse}", file=sys.stderr)
        print(f"  没有 video 字段    : {n_no_video}", file=sys.stderr)
        print(f"  没有 caption 字段  : {n_no_caption}", file=sys.stderr)
        print(f"  视频文件找不到     : {n_missing}", file=sys.stderr)
        print(f"  文件在 clips-root 外: {n_outside}", file=sys.stderr)
        print(f"\n首条记录的字段: {first_keys}", file=sys.stderr)
        print(f"video 键猜测表  : {vid_keys}", file=sys.stderr)
        print(f"caption 键猜测表: {cap_keys}", file=sys.stderr)
        for cands in tried_samples:
            print(f"  试过的路径: {cands}", file=sys.stderr)
        if n_no_video or n_no_caption:
            print("\n→ 字段名不在猜测表里:用 --video-key / --caption-key 显式指定"
                  "(字段名见上面「首条记录的字段」)。", file=sys.stderr)
        if n_missing:
            print("\n→ 文件定位失败:确认 --clips-root 指向真正装片段的目录;"
                  "片段若在 InternVId-*/ 子目录里,记录需要带 abs_path 字段。", file=sys.stderr)
        if n_outside:
            print("\n→ 文件找得到但不在 --clips-root 底下:训练 anno 只能写相对路径,"
                  "把 --clips-root 提到能覆盖这些文件的公共上层目录"
                  "(并让 configs/data.py 的 data_root 与之对应)。", file=sys.stderr)
        sys.exit(1)

    # 共享字幕统计(caption_idx 多正例修复的影响面)
    counts = Counter(r["caption"] for r in records)
    dup_groups = sum(1 for c in counts.values() if c > 1)
    dup_clips = sum(c for c in counts.values() if c > 1)
    print(f"记录: 可用 {len(records)}  解析失败 {n_parse}  缺 video {n_no_video}  "
          f"缺 caption {n_no_caption}  文件缺失 {n_missing}  root 外 {n_outside}")
    print(f"共享字幕: {dup_groups} 组 / 覆盖 {dup_clips} 条 clip "
          f"({dup_clips / len(records):.1%}) —— caption_idx 多正例的实际影响面")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    eval_recs = []
    if args.eval_split > 0:
        # 评测集内字幕必须唯一,否则自检索的 txt->video 映射不适定
        seen = set()
        rest = []
        for r in records:
            if len(eval_recs) < args.eval_split and r["caption"] not in seen:
                eval_recs.append(r)
                seen.add(r["caption"])
            else:
                rest.append(r)
        records = rest

    for path, data in [(args.out_train, records), (args.out_eval, eval_recs)]:
        if not data:
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"写出 {path}: {len(data)} 条")


if __name__ == "__main__":
    main()
