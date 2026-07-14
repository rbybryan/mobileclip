#!/usr/bin/env python3
"""Download / scrape InternVid-10M-FLT clips for MobileViCLIP pretraining.

The training corpus is defined in ``configs/data.py`` as 11 shards
(``internvid_v1 … v11``). Each shard is a pair of:

  * annotation JSON  ->  ``anno_downstream/InternVid/InternVid-10M-flt_<N>.json``
      a JSON list of ``{"video": "<name>.mp4", "caption": "..."}`` entries
      (see dataset/pt_dataset.py::VidTxtPtTrainDataset.get_anno, which reads
      ``os.path.join(data_root, anno["video"])``).
  * video directory  ->  ``data/video_sample/InternVId-FLT_<N>/``
      the actual clip files, named exactly as the ``video`` field above.

InternVid does NOT host the videos. Each clip is a time range of a public
YouTube video, so we fetch the source with yt-dlp and cut [start, end] with
ffmpeg, writing the result under the shard directory using the same file name
the annotation expects.

USAGE
-----
1) Inspect one shard's JSON first, to see the real ``video`` naming:

     python tools/download_internvid.py inspect \
       --anno anno_downstream/InternVid/InternVid-10M-flt_1.json

2) Download a shard (start small with --limit for a smoke test):

     python tools/download_internvid.py download \
       --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
       --out  data/video_sample/InternVId-FLT_1 \
       --limit 200 --workers 4

IMPORTANT: verify the clip-name regex in ``parse_video_name`` matches YOUR JSON
(run ``inspect`` first). If your ``video`` field does not encode the YouTube id
and timestamps, use --info-jsonl to join against the official metadata
(OpenGVLab/InternVid-10M-FLT-INFO.jsonl); see the guide in
docs/internvid-download-zh.md.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- The one place you may need to ADAPT to your annotation files. ----------
# Default assumes clip names like "<YoutubeID>_<start>_<end>.mp4" where start/end
# are seconds (int or float), e.g. "dQw4w9WgXcQ_12.0_22.5.mp4".
_CLIP_RE = re.compile(
    r"^(?P<yid>[A-Za-z0-9_-]{11})_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)\.mp4$"
)


def parse_video_name(name):
    """Return (youtube_id, start_seconds, end_seconds) or None if it doesn't match."""
    base = os.path.basename(name)
    m = _CLIP_RE.match(base)
    if not m:
        return None
    return m.group("yid"), float(m.group("start")), float(m.group("end"))


def load_info_lookup(info_jsonl):
    """Build {video_name -> (yid, start, end)} from the official FLT-INFO jsonl.

    The official metadata columns are YoutubeID, Start_timestamp, End_timestamp,
    Caption, Aesthetic_Score, UMT_Score. Timestamps look like "HH:MM:SS.mmm".
    Adjust the composed key if your annotation's ``video`` field differs.
    """
    def to_sec(ts):
        ts = str(ts)
        if ":" in ts:
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        return float(ts)

    lookup = {}
    with open(info_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yid = r["YoutubeID"]
            start = to_sec(r["Start_timestamp"])
            end = to_sec(r["End_timestamp"])
            key = f"{yid}_{start}_{end}.mp4"
            lookup[key] = (yid, start, end)
    return lookup


def have(cmd):
    return subprocess.run(["which", cmd], capture_output=True).returncode == 0


def download_clip(yid, start, end, out_path, cookies=None, retries=2):
    """Download the source video's [start, end] range straight into out_path.

    Uses yt-dlp's --download-sections so only the needed range is fetched, then
    re-encodes to a keyframe-accurate mp4.
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skip"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    url = f"https://www.youtube.com/watch?v={yid}"
    cmd = [
        "yt-dlp",
        "-q", "--no-warnings",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio/best",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o", out_path,
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd.append(url)
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return "ok"
        last_err = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or [""]
    return f"fail: {last_err[0][:200]}"


def cmd_inspect(args):
    with open(args.anno) as f:
        annos = json.load(f)
    print(f"# entries: {len(annos)}")
    print(f"# keys in first entry: {list(annos[0].keys())}")
    n_match = 0
    for a in annos[: args.samples]:
        v = a.get("video")
        parsed = parse_video_name(v) if v else None
        n_match += parsed is not None
        print(json.dumps({"video": v, "caption": (a.get('caption') or '')[:60],
                          "parsed": parsed}, ensure_ascii=False))
    print(f"\n# regex matched {n_match}/{min(args.samples, len(annos))} sampled names")
    if n_match == 0:
        print("!! The default clip-name regex did NOT match. Edit parse_video_name(), "
              "or use --info-jsonl with the official FLT-INFO metadata. "
              "See docs/internvid-download-zh.md")


def cmd_download(args):
    if not have("yt-dlp"):
        sys.exit("yt-dlp not found. Install: pip install -U yt-dlp")
    if not have("ffmpeg"):
        sys.exit("ffmpeg not found. Install: apt-get install -y ffmpeg")

    with open(args.anno) as f:
        annos = json.load(f)
    if args.limit:
        annos = annos[: args.limit]

    lookup = load_info_lookup(args.info_jsonl) if args.info_jsonl else None

    jobs = []
    unresolved = 0
    for a in annos:
        name = a.get("video")
        if not name:
            continue
        resolved = None
        if lookup is not None:
            resolved = lookup.get(os.path.basename(name))
        if resolved is None:
            resolved = parse_video_name(name)
        if resolved is None:
            unresolved += 1
            continue
        yid, start, end = resolved
        jobs.append((yid, start, end, os.path.join(args.out, os.path.basename(name))))

    print(f"resolved {len(jobs)}/{len(annos)} entries "
          f"({unresolved} unresolved) -> {args.out}")
    if not jobs:
        sys.exit("Nothing resolved. Run `inspect` and fix parse_video_name / pass --info-jsonl.")

    os.makedirs(args.out, exist_ok=True)
    manifest = open(os.path.join(args.out, "_download_log.tsv"), "a")
    counts = {"ok": 0, "skip": 0, "fail": 0}

    def run_one(job):
        yid, start, end, out_path = job
        status = download_clip(yid, start, end, out_path, cookies=args.cookies)
        return out_path, status

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            out_path, status = fut.result()
            key = "ok" if status == "ok" else "skip" if status == "skip" else "fail"
            counts[key] += 1
            manifest.write(f"{status}\t{out_path}\n")
            if i % 50 == 0 or key == "fail":
                print(f"[{i}/{len(jobs)}] ok={counts['ok']} skip={counts['skip']} "
                      f"fail={counts['fail']} :: {status}\t{os.path.basename(out_path)}")
    manifest.close()
    print(f"DONE: ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")
    print("Missing clips are fine: configs set jump_filter=True and the dataset "
          "resamples on load errors.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="print sample entries + check name parsing")
    pi.add_argument("--anno", required=True)
    pi.add_argument("--samples", type=int, default=10)
    pi.set_defaults(func=cmd_inspect)

    pd = sub.add_parser("download", help="fetch clips for one shard")
    pd.add_argument("--anno", required=True, help="InternVid-10M-flt_<N>.json")
    pd.add_argument("--out", required=True, help="data/video_sample/InternVId-FLT_<N>")
    pd.add_argument("--limit", type=int, default=0, help="only first N entries (smoke test)")
    pd.add_argument("--workers", type=int, default=4)
    pd.add_argument("--cookies", default=None, help="cookies.txt for age-restricted videos")
    pd.add_argument("--info-jsonl", default=None,
                    help="official FLT-INFO jsonl, to resolve names if regex fails")
    pd.set_defaults(func=cmd_download)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
