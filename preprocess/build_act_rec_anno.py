#!/usr/bin/env python
"""Build the zero-shot action-recognition annotation JSONs from public split files.

The downstream annotation JSONs normally come from the author's Google Drive
`anno_downstream/` folder, which is access-gated (permission denied). This script
reconstructs the three *classification* ones from the datasets' own public split /
label files, in the exact schema this repo's act-recognition loader expects:

    a JSON list of {"video": "<filename on disk>", "caption": "<class name>"}

`caption` is the human-readable class name; `dataset/ret_dataset.py:build_data_act_rec`
templates it with `kinetics_templates` ("A video of a person doing {}." etc.) and the
set of unique captions defines the class list. `video` is joined onto the dataset's
`data_root` and must match a real file — a missing file raises and stops the eval,
so pass `--videos <data_root>` to keep only entries whose video is present on disk.

Usage (see README_zh 6.2-6.4 for the full download steps):

  # UCF101  (needs the official UCF101TrainTestSplits-RecognitionTask.zip -> ucfTrainTestlist/)
  python preprocess/build_act_rec_anno.py --dataset ucf101 \
      --src ./ucfTrainTestlist --videos ./data/ucf101/videos \
      --out anno_downstream/ucf101_act_test.json

  # HMDB51  (needs test_train_splits -> testTrainMulti_7030_splits/)
  python preprocess/build_act_rec_anno.py --dataset hmdb51 \
      --src ./testTrainMulti_7030_splits --videos ./data/hmdb51/videos_test \
      --out anno_downstream/hmdb51_act_test.json

  # Kinetics-400  (needs the official validate CSV: label,youtube_id,time_start,time_end)
  python preprocess/build_act_rec_anno.py --dataset k400 \
      --src ./kinetics400_val.csv --videos ./data/kinetics_400_val_10s_320p \
      --out anno_downstream/kinetics400_validate.json

Notes on class-name text (may cause small deviations from the paper's numbers):
  - UCF101 class names are CamelCase ("ApplyEyeMakeup") and are split into words
    ("apply eye makeup").
  - HMDB51 class names use underscores ("brush_hair") -> spaces ("brush hair").
  - Kinetics-400 labels already read naturally and are used verbatim.
The eval lowercases/strips captions via `pre_text`, so exact casing does not matter,
but if the author phrased a class differently, that class's accuracy may differ.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict


def find_one(root, *patterns):
    if os.path.isfile(root):
        return root
    for pat in patterns:
        hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None


def find_many(root, pattern):
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def camel_to_words(name):
    """'ApplyEyeMakeup' -> 'apply eye makeup', 'TaiChi' -> 'tai chi', 'YoYo' -> 'yo yo'."""
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    return " ".join(parts).lower().strip()


def normalize_class(name):
    """Class folder name -> readable caption ('brush_hair'->'brush hair', 'TaiChi'->'tai chi')."""
    if "_" in name:
        return name.replace("_", " ").lower().strip()
    if re.search(r"[a-z][A-Z]", name):  # CamelCase, e.g. UCF101 folders
        return camel_to_words(name)
    return name.lower().strip()


VIDEO_EXTS = (".avi", ".mp4", ".webm", ".mkv", ".mov")


def build_from_class_dir(root):
    """Fallback: build from a class-organized video tree (<root>/<class>/<clip>.<ext>).

    Uses EVERY video under each class folder (train+test), so the sample set is the
    whole dataset, not the split-1 test set — numbers will differ from the paper's
    `act_val/` reference rows. Use the split-file mode when you can get the splits.
    `video` is the basename (matches the flattened data_root in README_zh 6.2/6.3).
    """
    out = []
    for dirpath, _, files in os.walk(root):
        cls = os.path.basename(dirpath.rstrip("/"))
        vids = [fn for fn in files if fn.lower().endswith(VIDEO_EXTS)]
        if not vids:
            continue
        caption = normalize_class(cls)
        for fn in vids:
            out.append({"video": fn, "caption": caption})
    if not out:
        raise SystemExit(f"No videos found under {root!r} (expected <root>/<class>/<clip>.avi).")
    print(f"[info] built from class-organized dir {root} (ALL videos, not just test split)")
    return out


def build_ucf101(src, split):
    """UCF101 test split: ucfTrainTestlist/testlist{split:02d}.txt

    Each line is 'ClassName/v_ClassName_gNN_cNN.avi'. The class is the path's first
    component; video is the basename (matches the flattened data/ucf101/videos/).
    """
    testlist = find_one(src, f"testlist{split:02d}.txt", "testlist*.txt")
    if not testlist:
        raise SystemExit(
            f"Could not find testlist{split:02d}.txt under {src!r}.\n"
            "Get it from the official UCF101TrainTestSplits-RecognitionTask.zip "
            "(https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)."
        )
    out = []
    for ln in open(testlist):
        ln = ln.strip().split()[0] if ln.strip() else ""
        if not ln:
            continue
        cls = ln.split("/")[0]
        fname = os.path.basename(ln)
        out.append({"video": fname, "caption": camel_to_words(cls)})
    print(f"[info] UCF101 testlist: {testlist}")
    return out


def build_hmdb51(src, split):
    """HMDB51 test split: testTrainMulti_7030_splits/<class>_test_split{split}.txt

    Each per-class file lists '<filename>.avi <tag>' where tag 2 == test split.
    Class name comes from the split filename (underscores -> spaces).
    """
    files = find_many(src, f"*_test_split{split}.txt")
    if not files:
        raise SystemExit(
            f"Could not find *_test_split{split}.txt under {src!r}.\n"
            "Get them from the official test_train_splits "
            "(http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/"
            "test_train_splits.rar) or any mirror; unrar -> testTrainMulti_7030_splits/."
        )
    out = []
    for f in files:
        cls = re.sub(rf"_test_split{split}$", "", os.path.splitext(os.path.basename(f))[0])
        caption = cls.replace("_", " ").strip()
        for ln in open(f):
            parts = ln.strip().split()
            if len(parts) >= 2 and parts[1] == "2":  # 2 == test
                out.append({"video": parts[0], "caption": caption})
    print(f"[info] HMDB51 split files: {len(files)} classes under {src}")
    return out


def _k400_video_name(ytid, ts, te):
    return f"{ytid}_{int(ts):06d}_{int(te):06d}.mp4"


def build_k400(src):
    """Kinetics-400 validation CSV -> {video, caption}.

    Accepts the official columns label,youtube_id,time_start,time_end[,split].
    The CVDF extractor names files '<ytid>_<start:06d>_<end:06d>.mp4'; `--videos`
    resolves whatever naming is actually on disk (see resolve_against_disk).
    """
    csv_path = find_one(src, "*val*.csv", "*validate*.csv", "*.csv")
    if not csv_path:
        raise SystemExit(f"Could not find a Kinetics validation CSV under {src!r}.")
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        need = ("label", "youtube_id", "time_start", "time_end")
        if not all(k in cols for k in need):
            raise SystemExit(f"Unexpected Kinetics CSV columns: {reader.fieldnames} "
                             f"(need {need}).")
        for row in reader:
            ytid = row[cols["youtube_id"]].strip()
            ts, te = row[cols["time_start"]], row[cols["time_end"]]
            label = row[cols["label"]].strip().strip('"')
            out.append({"video": _k400_video_name(ytid, ts, te), "caption": label,
                        "_ytid": ytid})
    print(f"[info] Kinetics-400 CSV: {csv_path}")
    return out


def resolve_against_disk(entries, videos_dir, dataset):
    """Keep only entries whose video exists in videos_dir; fix K400 names by ytid.

    Returns (kept, dropped). A missing file crashes the eval, so this guarantees the
    generated JSON only references videos that are actually present.
    """
    on_disk = set(os.listdir(videos_dir)) if os.path.isdir(videos_dir) else set()
    if not on_disk:
        print(f"[warn] {videos_dir!r} is empty or missing; skipping disk filtering. "
              "Run this again with the videos in place before evaluating.")
        for e in entries:
            e.pop("_ytid", None)
        return entries, []

    # For K400, index disk files by youtube id prefix to tolerate naming differences.
    by_ytid = defaultdict(list)
    if dataset == "k400":
        for fn in on_disk:
            m = re.match(r"([A-Za-z0-9_\-]{11})", fn)
            if m:
                by_ytid[m.group(1)].append(fn)

    kept, dropped = [], []
    for e in entries:
        v = e["video"]
        if v in on_disk:
            e.pop("_ytid", None)
            kept.append(e)
            continue
        if dataset == "k400":
            ytid = e.pop("_ytid", None)
            cands = by_ytid.get(ytid or "", [])
            if cands:
                e["video"] = sorted(cands, key=len)[0]
                kept.append(e)
                continue
        e.pop("_ytid", None)
        dropped.append(v)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ucf101", "hmdb51", "k400"])
    ap.add_argument("--src", default=None,
                    help="UCF101: ucfTrainTestlist/ dir; HMDB51: testTrainMulti_7030_splits/ "
                         "dir; K400: the validation CSV (or a dir containing it). "
                         "Omit and use --from-dir for the class-organized fallback (ucf101/hmdb51).")
    ap.add_argument("--from-dir", dest="from_dir", default=None,
                    help="Fallback for ucf101/hmdb51 when you can't get the split files: a "
                         "class-organized video tree (<dir>/<class>/<clip>.avi). Uses ALL videos "
                         "(train+test), so numbers differ from the split-1 reference.")
    ap.add_argument("--videos", default=None,
                    help="data_root with the videos; if given, entries whose video is "
                         "not on disk are dropped (a missing file otherwise crashes the eval).")
    ap.add_argument("--split", type=int, default=1, help="UCF101/HMDB51 split id (default 1).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.from_dir:
        if args.dataset == "k400":
            raise SystemExit("--from-dir is only for ucf101/hmdb51; use --src <val CSV> for k400.")
        entries = build_from_class_dir(args.from_dir)
    elif not args.src:
        raise SystemExit("Pass --src (split files / CSV) or --from-dir (class-organized videos).")
    elif args.dataset == "ucf101":
        entries = build_ucf101(args.src, args.split)
    elif args.dataset == "hmdb51":
        entries = build_hmdb51(args.src, args.split)
    else:
        entries = build_k400(args.src)

    n_classes = len({e["caption"] for e in entries})
    print(f"[info] parsed {len(entries)} entries across {n_classes} classes")

    if args.videos:
        entries, dropped = resolve_against_disk(entries, args.videos, args.dataset)
        if dropped:
            print(f"[warn] dropped {len(dropped)} entries with no matching video on disk "
                  f"(e.g. {dropped[:3]}). For K400 this is normal (dead YouTube links).")
        print(f"[info] {len(entries)} entries remain after disk filtering "
              f"({len({e['caption'] for e in entries})} classes present)")
    else:
        for e in entries:
            e.pop("_ytid", None)
        print("[warn] --videos not given: JSON not filtered against disk. If any listed "
              "video is missing, the eval will crash — re-run with --videos before evaluating.")

    if not entries:
        raise SystemExit("No entries to write. Check --src / --videos paths.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(entries, open(args.out, "w"))
    print(f"[ok] wrote {len(entries)} entries -> {args.out}")
    print(f"[ok] sample: {entries[0]}")


if __name__ == "__main__":
    main()
