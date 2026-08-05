#!/usr/bin/env python3
"""Split a Stage-0 flags JSONL by flag type and print a short per-flag preview.

Each flag gets its own <out-dir>/<flag>.jsonl with ALL its records (for full
inspection with less/Gradio); stdout only previews --per-flag examples per
flag so the terminal stays readable. Flags are printed most-frequent first.

Example (what `run_v100.sh s0-show` runs):
    python recaption/scripts/show_flags.py \
        --flags outputs/audit_10k/s0_flags.jsonl \
        --out-dir outputs/audit_10k/s0_flag_split --per-flag 5
"""

import argparse
import json
import os
import signal
from collections import defaultdict

if hasattr(signal, "SIGPIPE"):  # die quietly when piped into head/less
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flags", required=True, help="s0_flags.jsonl from profile_jsonl.py")
    ap.add_argument("--out-dir", required=True, help="per-flag jsonl files land here")
    ap.add_argument("--per-flag", type=int, default=5,
                    help="examples printed per flag; the split files always get all")
    ap.add_argument("--only", default="",
                    help="comma-separated flag names to restrict to; empty = all")
    ap.add_argument("--width", type=int, default=70, help="preview truncation (chars)")
    args = ap.parse_args()

    only = {f for f in args.only.split(",") if f}
    by_flag = defaultdict(list)
    with open(args.flags, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for fl in rec.get("flags", []):
                if not only or fl in only:
                    by_flag[fl].append(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    for fl in sorted(by_flag, key=lambda k: -len(by_flag[k])):
        recs = by_flag[fl]
        path = os.path.join(args.out_dir, f"{fl}.jsonl")
        with open(path, "w", encoding="utf-8") as fout:
            for r in recs:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"===== {fl}  共 {len(recs)} 条  -> {path} =====")
        for r in recs[:args.per_flag]:
            print(f"  [zh] {r.get('zh', '')[:args.width]}")
            if r.get("en"):
                print(f"  [en] {r['en'][:args.width]}")
            print(f"  [video] {r.get('video', '')}")
            print("  ---")
        if len(recs) > args.per_flag:
            print(f"  ... 其余 {len(recs) - args.per_flag} 条见 {path}")
        print()


if __name__ == "__main__":
    main()
