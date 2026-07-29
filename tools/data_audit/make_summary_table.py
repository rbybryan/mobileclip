#!/usr/bin/env python3
"""Summary TABLE of the A/B recaption pipeline (维护者指令 2026-07-28):
one row per clip — video as a looping GIF | 源caption(en) | 源翻译(zh) |
方案A 重述/直融长稿/直融短稿 | 方案B 同三列(属性徽标).

Nothing is recomputed: the input is the SAME review_compare.tsv that
make_style_compare.py already writes (pipeline-compare step ⑥), so this is a
pure re-presentation. GIFs are decoded with the pipeline's own backend ladder
(decord > opencv > pyav, TSN segment centres) and cached — an existing
gifs/<video>.gif is never re-encoded, so re-runs are cheap.

    python tools/data_audit/make_summary_table.py \
        --sheet outputs/audit_10k/review_compare.tsv \
        --out-dir outputs/audit_10k/table_compare

Output: <out-dir>/table.html (static, no JS needed to read) + gifs/.
打包: tar czf table.tgz -C outputs/audit_10k table_compare  → 本地解包直接开.

Deps: pillow + one video backend (decord | opencv-python | av) — the audit
env has all of these; --no-gif builds a text-only table with zero deps.
"""

import argparse
import csv
import html
import os
import re
import sys

BUILD_TAG = "T20260728a"   # 页面右上角版本角标, 与卡片包 BUILD_TAG 同用途
BADGE = re.compile(r"【(人物|物体|主体|动作|场景|文字)】")
SEP = re.compile(r"\s*‖\s*")

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>A/B 重述总结表</title>
<style>
 body{font:14px/1.65 -apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;
      margin:0;background:#f4f5f7;color:#1b1e22}
 .wrap{padding:14px 18px}
 h1{font-size:17px;margin:0 0 2px} .sub{color:#666;font-size:12px;margin:0 0 12px}
 table{border-collapse:collapse;background:#fff;width:100%}
 th,td{border:1px solid #d6d9de;padding:8px 10px;vertical-align:top;text-align:left}
 thead th{position:sticky;top:0;background:#eef1f5;z-index:2;font-size:13px}
 thead tr:nth-child(2) th{top:37px;background:#f5f7fa;font-weight:500}
 tbody tr:nth-child(even){background:#fafbfc}
 td.gif{width:310px;min-width:220px}
 td.gif img{max-width:300px;width:100%;border-radius:6px;display:block;background:#000}
 td.gif .vid{color:#888;font-size:11px;word-break:break-all;margin-top:4px}
 .nogif{width:280px;height:120px;display:flex;align-items:center;justify-content:center;
        background:#f0f0f2;border:1px dashed #c6c9ce;border-radius:6px;color:#a12;font-size:12px}
 td.src{min-width:180px;max-width:260px}
 td.cap{min-width:210px;max-width:330px}
 ol{margin:0;padding-left:20px} ol li{margin:0 0 4px}
 .long{color:#333} .short{font-weight:600;color:#0a5c2e;background:#f0fff4;
        padding:4px 6px;border-radius:5px;display:inline-block}
 .slot{display:inline-block;font-size:11px;line-height:1.6;padding:0 5px;margin-right:3px;
       border-radius:4px;background:#e8f0fe;color:#1a56a8;border:1px solid #c3d7f7}
 .zh{background:#fffdf2}
 .tag{float:right;color:#999;font-size:12px}
</style>
<div class="wrap">
<h1>A/B 重述总结表 <span class="tag">v__BTAG__</span></h1>
<p class="sub">__SUB__</p>
<table>
<thead>
<tr><th rowspan="2">视频</th><th rowspan="2">源caption(en)</th><th rowspan="2">源翻译(zh)</th>
    <th colspan="3">方案A（稠密×5）</th><th colspan="3">方案B（固定槽位×3）</th></tr>
<tr><th>重述</th><th>直融长稿</th><th>直融短稿</th><th>重述</th><th>直融长稿</th><th>直融短稿</th></tr>
</thead>
<tbody>
__ROWS__
</tbody>
</table>
</div>
"""


def deco(text):
    """escape 后把【属性】渲染成蓝徽标 (与卡片包 deco 同规则)."""
    return BADGE.sub(r'<span class="slot">\1</span>', html.escape(text))


def cell_caps(text):
    """‖ 分隔的多条 caption -> 有序编号列表; 单条原样 (与卡片包同约定)."""
    parts = [p.strip() for p in SEP.split(text or "") if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return deco(parts[0])
    return "<ol>" + "".join(f"<li>{deco(p)}</li>" for p in parts) + "</ol>"


def make_gif(video_path, out_path, frames, max_side, ms):
    from PIL import Image
    from vlm_audit import _BACKENDS, _resolve_backend
    arr = _BACKENDS[_resolve_backend("auto")](video_path, frames)
    imgs = []
    for fr in arr:
        im = Image.fromarray(fr)
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        imgs.append(im)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=ms, loop=0, optimize=True)


def safe_name(video):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(video)) + ".gif"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True,
                    help="review_compare.tsv (make_style_compare.py 产出)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gif-frames", type=int, default=16)
    ap.add_argument("--gif-max-side", type=int, default=300)
    ap.add_argument("--gif-ms", type=int, default=150,
                    help="每帧毫秒 (16帧×150ms ≈ 2.4s 循环)")
    ap.add_argument("--no-gif", action="store_true",
                    help="只出文本表格 (无视频依赖, 秒级)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    with open(args.sheet, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit(f"{args.sheet} 是空的")

    gif_dir = os.path.join(args.out_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    body, n_gif, n_fail = [], 0, 0
    for r in rows:
        vid = r.get("video") or ""
        gif_rel = f"gifs/{safe_name(vid)}"
        gif_abs = os.path.join(args.out_dir, gif_rel)
        vcell = ""
        if not args.no_gif:
            if not os.path.exists(gif_abs):
                vp = r.get("video_path") or ""
                try:
                    if not (vp and os.path.exists(vp)):
                        raise FileNotFoundError(vp or "(no video_path)")
                    make_gif(vp, gif_abs, args.gif_frames,
                             args.gif_max_side, args.gif_ms)
                except Exception as ex:  # noqa: BLE001 - 单片失败绝不拖死整表
                    n_fail += 1
                    print(f"[gif-fail] {vid}: {type(ex).__name__}: {ex}",
                          file=sys.stderr)
        if os.path.exists(gif_abs):
            n_gif += 1
            vcell = f'<img src="{html.escape(gif_rel)}" loading="lazy" alt="">'
        else:
            vcell = '<div class="nogif">GIF 缺失（解码失败或视频不在）</div>'
        vcell += f'<div class="vid">{html.escape(vid)}</div>'
        body.append(
            "<tr>"
            f'<td class="gif">{vcell}</td>'
            f'<td class="src">{html.escape(r.get("英文en") or "")}</td>'
            f'<td class="src zh">{html.escape(r.get("原zh") or "")}</td>'
            f'<td class="cap">{cell_caps(r.get("A重述"))}</td>'
            f'<td class="cap long">{deco((r.get("A长稿") or "").strip())}</td>'
            f'<td class="cap"><span class="short">{deco((r.get("A直融") or "").strip())}</span></td>'
            f'<td class="cap">{cell_caps(r.get("B重述"))}</td>'
            f'<td class="cap long">{deco((r.get("B长稿") or "").strip())}</td>'
            f'<td class="cap"><span class="short">{deco((r.get("B直融") or "").strip())}</span></td>'
            "</tr>")

    sub = (f"{len(rows)} 片 · GIF {n_gif}/{len(rows)}"
           + (f"（{n_fail} 片解码失败）" if n_fail else "")
           + " · 直融短稿=最终训练字幕(≤30字) · 长稿=细节参考 · B重述带属性徽标")
    page = (PAGE.replace("__ROWS__", "\n".join(body))
                .replace("__SUB__", sub).replace("__BTAG__", BUILD_TAG))
    out_html = os.path.join(args.out_dir, "table.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[done] {len(rows)} rows -> {out_html} (GIF {n_gif}, fail {n_fail})")


if __name__ == "__main__":
    main()
