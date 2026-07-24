#!/usr/bin/env python3
"""Single-file HTML visualization of recap_pipeline.tsv (clickable index + cards).

Reads the panorama TSV produced by make_pipeline_tsv.py (single source of truth
-- no re-joining) and renders one self-contained HTML: a summary header, a
clickable index table (video / flags / view counts), and one card per clip
showing en source vs zh translation vs the 5 recaptions (with SigLIP scores)
vs A' split views vs the verified/unverified/filtered LLM views.

Optional --copy-videos: also copy each clip's mp4 next to the HTML (bundle
mode, plays inline via <video> tags) -- needs --video-root.

Example:
    python tools/data_audit/make_pipeline_html.py \
        --tsv outputs/audit_10k/recap_pipeline.tsv \
        --output outputs/audit_10k/recap_pipeline.html
"""

import argparse
import csv
import html
import os
import shutil
import sys

SEP = " ‖ "

CSS = """
body{font-family:system-ui,'PingFang SC','Microsoft YaHei',sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px}
table.idx{border-collapse:collapse;width:100%;background:#fff;font-size:13px}
table.idx th,table.idx td{border:1px solid #ddd;padding:4px 8px;text-align:left}
table.idx tr:hover{background:#eef}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}
.card h3{margin:0 0 8px;font-size:14px;word-break:break-all}
.row{margin:6px 0}
.lab{display:inline-block;min-width:88px;color:#666;font-size:12px;vertical-align:top}
.txt{display:inline-block;max-width:920px;vertical-align:top}
.recap{margin:2px 0;padding:4px 8px;background:#f2f6ff;border-radius:4px;font-size:13px}
.view{display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border-radius:10px;font-size:12px}
.pass{background:#e2f5e5;border:1px solid #9fd3a8}
.unv{background:#eee;border:1px solid #ccc;color:#555}
.fail{background:#fde3e3;border:1px solid #e6a0a0;text-decoration:line-through}
.flag{background:#c0392b;color:#fff;padding:1px 6px;border-radius:4px;font-size:11px;margin-left:6px}
.keep{background:#2e7d32;color:#fff;padding:1px 6px;border-radius:4px;font-size:11px}
.old{background:#fff7e0;padding:4px 8px;border-radius:4px;font-size:13px}
.en{color:#555;font-size:12px}
video{max-width:480px;display:block;margin:6px 0}
a.top{font-size:11px;float:right}
"""


def views_of(cell):
    return [v for v in (cell or "").split(SEP) if v.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--copy-videos", action="store_true",
                    help="copy each clip's mp4 into <output_dir>/videos/ and "
                         "embed <video> players (bundle mode)")
    ap.add_argument("--video-root", default=None, help="needed with --copy-videos")
    args = ap.parse_args()

    with open(args.tsv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit(f"no rows in {args.tsv}")

    vids_dir = None
    resolve = None
    if args.copy_videos:
        if not args.video_root:
            sys.exit("--copy-videos needs --video-root")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vlm_audit import resolve_video_path  # noqa: E402
        resolve = resolve_video_path
        vids_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "videos")
        os.makedirs(vids_dir, exist_ok=True)

    e = html.escape
    n_fail = sum(len(views_of(r.get("views_fail"))) for r in rows)
    n_pass = sum(len(views_of(r.get("views_pass"))) for r in rows)
    n_unv = sum(len(views_of(r.get("views_unverified"))) for r in rows)
    n_flag = sum(1 for r in rows if (r.get("flags") or "").strip())

    out = ["<meta charset='utf-8'><title>recap pipeline 目检</title>",
           f"<style>{CSS}</style><div class='wrap'>",
           "<h1>重述流水线目检（每片一卡）</h1>",
           f"<p>{len(rows)} 片 · LLM 视图 ✓{n_pass} / ·{n_unv} / ✗{n_fail} · "
           f"旗标片 {n_flag}。列语义：<b>✓</b>=过验真 <b>·</b>=未验(保守留) "
           f"<b>✗</b>=被验真器过滤（删除线）。索引点 video 跳卡片。</p>",
           "<table class='idx'><tr><th>#</th><th>video</th><th>旗标</th>"
           "<th>A−保留</th><th>✓/·/✗</th><th>理由</th></tr>"]
    for i, r in enumerate(rows):
        out.append(
            f"<tr><td>{i + 1}</td><td><a href='#c{i}'>{e(r['video'])}</a></td>"
            f"<td>{e(r.get('flags') or '')}</td>"
            f"<td>{e(r.get('A_minus_keep') or '—')}</td>"
            f"<td>{len(views_of(r.get('views_pass')))}/"
            f"{len(views_of(r.get('views_unverified')))}/"
            f"{len(views_of(r.get('views_fail')))}</td>"
            f"<td>{e(r.get('bucket_reasons') or '')}</td></tr>")
    out.append("</table>")

    for i, r in enumerate(rows):
        out.append(f"<div class='card' id='c{i}'><a class='top' href='#'>↑顶部</a>"
                   f"<h3>{i + 1}. {e(r['video'])}"
                   + "".join(f"<span class='flag'>{e(fl)}</span>"
                             for fl in (r.get("flags") or "").split("|") if fl)
                   + "</h3>")
        if vids_dir is not None:
            src = resolve(args.video_root, {}, r["video"])
            base = os.path.basename(r["video"])
            dst = os.path.join(vids_dir, base)
            if os.path.exists(src):
                if not os.path.exists(dst):
                    shutil.copyfile(src, dst)
                out.append(f"<video controls preload='none' "
                           f"src='videos/{e(base)}'></video>")
        out.append(f"<div class='row'><span class='lab'>en 源</span>"
                   f"<span class='txt en'>{e(r.get('en_source') or '')}</span></div>")
        out.append(f"<div class='row'><span class='lab'>zh 旧译</span>"
                   f"<span class='txt old'>{e(r.get('zh_translated') or '')}</span></div>")
        keep = {k for k in (r.get("A_minus_keep") or "").split(",") if k}
        for k in range(1, 6):
            cap = r.get(f"recap_{k}") or ""
            if not cap:
                continue
            badge = f" <span class='keep'>A−</span>" if str(k) in keep else ""
            out.append(f"<div class='row'><span class='lab'>重述 {k}</span>"
                       f"<span class='txt recap'>{e(cap)}{badge}</span></div>")
        segs = [("A′ 句切", "views_A_prime", "unv"),
                ("✓ 过验真", "views_pass", "pass"),
                ("· 未验", "views_unverified", "unv"),
                ("✗ 被滤", "views_fail", "fail")]
        for lab, key, cls in segs:
            vs = views_of(r.get(key))
            if not vs:
                continue
            out.append(f"<div class='row'><span class='lab'>{lab} ({len(vs)})</span>"
                       f"<span class='txt'>"
                       + "".join(f"<span class='view {cls}'>{e(v)}</span>" for v in vs)
                       + "</span></div>")
        jm = r.get("jaccard_med") or ""
        out.append(f"<div class='row'><span class='lab'>多样性</span>"
                   f"<span class='txt en'>成对 Jaccard 中位 {e(jm) or '—'} · "
                   f"相同对 {e(r.get('ident_pairs') or '0')}</span></div></div>")
    out.append("</div>")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[done] {len(rows)} cards -> {args.output}"
          + (f" (+videos in {vids_dir})" if vids_dir else ""))


if __name__ == "__main__":
    main()
