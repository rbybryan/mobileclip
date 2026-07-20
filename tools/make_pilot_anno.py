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
import sys
from collections import Counter

# 与 tools/data_audit/make_audit_sample.py 保持一致的字段名猜测表
VIDEO_KEYS = ["video", "video_path", "video_id", "clip", "clip_name", "filename"]
ZH_KEYS = ["caption_zh", "zh_caption", "caption_cn", "cn_caption", "translated_caption",
           "translation", "zh", "chinese_caption", "raw_caption_zh_translated",
           "caption_zh_translated"]
EN_KEYS = ["caption", "caption_en", "en_caption", "source_caption", "src_caption", "en"]


def pick(rec, keys):
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="10k 抽样 jsonl")
    ap.add_argument("--clips-root", default=None,
                    help="视频根目录;给出时会剔除文件缺失的记录")
    ap.add_argument("--caption-field", choices=["en", "zh"], default="en")
    ap.add_argument("--eval-split", type=int, default=1000,
                    help="held-out 评测集条数(0 = 不拆)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", default=None,
                    help="eval anno 输出路径(eval-split>0 时必填)")
    args = ap.parse_args()
    if args.eval_split > 0 and not args.out_eval:
        sys.exit("--eval-split > 0 时必须给 --out-eval")

    cap_keys = EN_KEYS if args.caption_field == "en" else ZH_KEYS
    records, n_bad, n_missing = [], 0, 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            video, caption = pick(rec, VIDEO_KEYS), pick(rec, cap_keys)
            if not video or not caption:
                n_bad += 1
                continue
            if args.clips_root and not os.path.isfile(os.path.join(args.clips_root, video)):
                n_missing += 1
                continue
            records.append({"video": video, "caption": caption})

    if not records:
        sys.exit("没有可用记录——检查 --input 字段名或 --clips-root")

    # 共享字幕统计(caption_idx 多正例修复的影响面)
    counts = Counter(r["caption"] for r in records)
    dup_groups = sum(1 for c in counts.values() if c > 1)
    dup_clips = sum(c for c in counts.values() if c > 1)
    print(f"记录: 可用 {len(records)}  解析失败 {n_bad}  文件缺失 {n_missing}")
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
