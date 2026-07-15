#!/usr/bin/env python
"""Build anno_downstream/msrvtt_1k_test.json (MSR-VTT 1K-A / JSFusion test split).

The downstream annotation JSONs normally come from the author's Google Drive
folder, which needs access approval. This script reconstructs the MSR-VTT one
from public sources instead, in the exact schema this repo's retrieval loader
expects: a JSON list of {"video": "videoXXXX.mp4", "caption": "..."} (1000 rows).

Two input modes (auto-detected, or force with --mode):

  1) Frozen-in-Time zip  (public, no login):
        wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
        unzip MSRVTT.zip            # creates ./MSRVTT/
        python preprocess/build_msrvtt_1k_test.py --src ./MSRVTT

  2) CLIP4Clip CSV  (MSRVTT_JSFUSION_test.csv, columns: key,vid_key,video_id,sentence):
        python preprocess/build_msrvtt_1k_test.py --src /path/to/MSRVTT_JSFUSION_test.csv

Output (default): anno_downstream/msrvtt_1k_test.json
"""
import argparse
import csv
import glob
import json
import os
import pickle
from collections import defaultdict


def find_one(root, *patterns):
    for pat in patterns:
        hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if hits:
            return sorted(hits, key=len)[0]
    return None


def from_frozen_in_time(root):
    ann_path = find_one(root, "MSR_VTT.json", "*_annotation.json")
    val_list = find_one(root, "*val_list_jsfusion*.txt", "*test_list_jsfusion*.txt")
    if not ann_path or not val_list:
        raise SystemExit(
            f"Could not locate annotation/split under {root!r}.\n"
            f"  MSR_VTT.json      -> {ann_path}\n"
            f"  val_list_jsfusion -> {val_list}\n"
            f"Run:  find {root} -name 'MSR_VTT.json' -o -name '*jsfusion*'  and share the output."
        )
    cap_idx_path = find_one(root, "*jsfusion*caption_idx*.pkl", "*jsfusion*cap*idx*.pkl",
                            "*caption_idx*.pkl")

    data = json.load(open(ann_path))
    caps = defaultdict(list)
    # MSR_VTT.json ships in two shapes; support both. Order captions per video
    # deterministically so the jsfusion caption index lines up.
    if isinstance(data, dict) and data.get("sentences"):
        # {"sentences": [{"video_id", "sen_id", "caption"}, ...]}
        for s in sorted(data["sentences"], key=lambda x: x.get("sen_id", 0)):
            caps[s["video_id"]].append(s["caption"])
    elif isinstance(data, dict) and data.get("annotations"):
        # COCO-style (Frozen-in-Time): {"annotations": [{"image_id", "id", "caption"}, ...]}
        for a in sorted(data["annotations"], key=lambda x: x.get("id", 0)):
            caps[a["image_id"]].append(a["caption"])
    else:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        raise SystemExit(f"Unrecognized annotation schema in {ann_path} (keys={keys}). "
                         "Expected a 'sentences' or 'annotations' list.")

    val_ids = [ln.strip() for ln in open(val_list) if ln.strip()]

    cap_idx = {}
    if cap_idx_path:
        cap_idx = pickle.load(open(cap_idx_path, "rb"))
        print(f"[info] using caption index: {cap_idx_path}")
    else:
        print("[warn] no jsfusion caption-index pkl found; using the FIRST caption per "
              "video. This runs fine but R@k may differ slightly from the paper's split.")

    out = []
    for vid in val_ids:
        idx = int(cap_idx.get(vid, 0)) if cap_idx else 0
        idx = min(idx, len(caps[vid]) - 1)
        out.append({"video": f"{vid}.mp4", "caption": caps[vid][idx]})
    return out


def from_csv(path):
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id") or row.get("vid_key") or row.get("key")
            cap = row.get("sentence") or row.get("caption")
            if not vid or cap is None:
                raise SystemExit(f"Unexpected CSV columns: {list(row.keys())}")
            vid = vid if vid.endswith(".mp4") else f"{vid}.mp4"
            out.append({"video": vid, "caption": cap})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Frozen-in-Time MSRVTT/ dir, or a CLIP4Clip MSRVTT_JSFUSION_test.csv")
    ap.add_argument("--out", default="anno_downstream/msrvtt_1k_test.json")
    ap.add_argument("--mode", choices=["auto", "fit", "csv"], default="auto")
    args = ap.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "csv" if args.src.lower().endswith(".csv") else "fit"

    out = from_csv(args.src) if mode == "csv" else from_frozen_in_time(args.src)

    if len(out) != 1000:
        print(f"[warn] expected 1000 rows for MSR-VTT 1K-A, got {len(out)}. "
              "Double-check the split file.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"[ok] wrote {len(out)} entries -> {args.out}")
    print(f"[ok] sample: {out[0]}")


if __name__ == "__main__":
    main()
