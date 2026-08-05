#!/usr/bin/env python3
"""Summary TABLE of the A/B recaption pipeline (维护者指令 2026-07-28):
one row per clip — video as a looping GIF | 源caption(en) | 源翻译(zh) |
方案A 重述/直融长稿/直融短稿 | 方案B 同三列(属性徽标).

Nothing is recomputed: the input is the SAME review_compare.tsv that
make_style_compare.py already writes (pipeline-compare step ⑥), so this is a
pure re-presentation. GIFs are decoded with the pipeline's own backend ladder
(decord > opencv > pyav, TSN segment centres) and cached — an existing
gifs/<video>.gif is never re-encoded, so re-runs are cheap.

    python recaption/scripts/make_summary_table.py \
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

BUILD_TAG = "T20260729a"   # 页面右上角版本角标, 与卡片包 BUILD_TAG 同用途
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
 td.why{min-width:120px;max-width:170px;color:#a12;font-size:12px}
 .tag{float:right;color:#999;font-size:12px}
</style>
<div class="wrap">
<h1>A/B 重述总结表 <span class="tag">v__BTAG__</span></h1>
<p class="sub">__SUB__</p>
<table>
<thead>
<tr><th rowspan="2">视频</th><th rowspan="2">源caption(en)</th><th rowspan="2">源翻译(zh)</th><th rowspan="2">判错原因</th>
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


def cell_caps_text(text):
    """csv/xlsx 版 cell_caps: 纯文本, 一条 caption 一行编号 (单条原样)."""
    parts = [p.strip() for p in SEP.split(text or "") if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(parts))


EXPORT_COLS = ["序号", "video", "英文en", "原zh", "判错原因", "A重述", "A长稿", "A直融",
               "B重述", "B长稿", "B直融", "video_path"]


def write_csv(path, recs):
    """utf-8-sig + 逗号分隔: Excel/WPS 双击直开中文不乱码; 多条重述在单元格内
    以换行分隔 (Excel 开自动换行即逐行显示)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(EXPORT_COLS)
        for r in recs:
            w.writerow([r.get(c, "") for c in EXPORT_COLS])


def write_xlsx(path, recs, thumb_dir):
    """openpyxl 可用时出 xlsx: 首列 GIF 首帧静态缩略图 (Excel 不播动图, 动图
    看 table.html), 文本列自动换行+列宽. 返回 None=成功 / 字符串=跳过原因."""
    try:
        from openpyxl import Workbook
        from openpyxl.drawing.image import Image as XImage
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        return "openpyxl 未安装 (audit_env 里 `pip install openpyxl` 后重跑本命令即可补出)"
    from PIL import Image
    wb = Workbook()
    ws = wb.active
    ws.title = "AB对比"
    hdr = ["缩略图"] + EXPORT_COLS
    ws.append(hdr)
    widths = {"缩略图": 22, "序号": 7, "video": 18, "英文en": 30, "原zh": 26, "判错原因": 18, "A重述": 42,
              "A长稿": 42, "A直融": 26, "B重述": 42, "B长稿": 42, "B直融": 26,
              "video_path": 24}
    for i, h in enumerate(hdr, 1):
        ws.cell(row=1, column=i).font = Font(bold=True)
        ws.column_dimensions[get_column_letter(i)].width = widths.get(h, 20)
    ws.freeze_panes = "A2"
    wrap = Alignment(wrap_text=True, vertical="top")
    os.makedirs(thumb_dir, exist_ok=True)
    for ri, r in enumerate(recs, 2):
        ws.append([""] + [r.get(c, "") for c in EXPORT_COLS])
        for ci in range(2, len(hdr) + 1):
            ws.cell(row=ri, column=ci).alignment = wrap
        thumb, g = None, r.get("gif_abs")
        if g and os.path.exists(g):
            thumb = os.path.join(thumb_dir,
                                 os.path.basename(g)[:-4] + ".png")
            if not os.path.exists(thumb):
                try:
                    im = Image.open(g)
                    im.seek(0)
                    im = im.convert("RGB")
                    im.thumbnail((150, 150))
                    im.save(thumb)
                except Exception:  # noqa: BLE001 - 缩略图失败只缺图不缺行
                    thumb = None
        if thumb and os.path.exists(thumb):
            img = XImage(thumb)
            ws.add_image(img, f"A{ri}")
            ws.row_dimensions[ri].height = max(84.0, img.height * 0.75)
        else:
            ws.row_dimensions[ri].height = 84.0
    wb.save(path)
    return None


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


def gif_name(idx, video):
    """gifs/ 文件按表格行序排列 (维护者 2026-07-29): 001_ 序号前缀,
    文件夹里按名排序=表格顺序, 按行号即可定位."""
    return f"{idx:03d}_{safe_name(video)}"


def adopt_legacy_gif(gif_dir, gif_abs, video):
    """无前缀旧名或旧序号前缀的既有 GIF -> 改名复用, 不重新编码."""
    if os.path.exists(gif_abs):
        return
    import glob as _glob
    cands = [os.path.join(gif_dir, safe_name(video))]
    cands += _glob.glob(os.path.join(gif_dir,
                                     "[0-9][0-9][0-9]_" + safe_name(video)))
    for c in cands:
        if os.path.exists(c) and c != gif_abs:
            os.rename(c, gif_abs)
            return


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

    body, recs, n_gif, n_fail = [], [], 0, 0
    for idx, r in enumerate(rows, 1):
        vid = r.get("video") or ""
        gif_rel = f"gifs/{gif_name(idx, vid)}"
        gif_abs = os.path.join(args.out_dir, gif_rel)
        adopt_legacy_gif(gif_dir, gif_abs, vid)
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
        vcell += f'<div class="vid">#{idx:03d} · {html.escape(vid)}</div>'
        body.append(
            "<tr>"
            f'<td class="gif">{vcell}</td>'
            f'<td class="src">{html.escape(r.get("英文en") or "")}</td>'
            f'<td class="src zh">{html.escape(r.get("原zh") or "")}</td>'
            f'<td class="why">{html.escape(r.get("判错原因") or "")}</td>'
            f'<td class="cap">{cell_caps(r.get("A重述"))}</td>'
            f'<td class="cap long">{deco((r.get("A长稿") or "").strip())}</td>'
            f'<td class="cap"><span class="short">{deco((r.get("A直融") or "").strip())}</span></td>'
            f'<td class="cap">{cell_caps(r.get("B重述"))}</td>'
            f'<td class="cap long">{deco((r.get("B长稿") or "").strip())}</td>'
            f'<td class="cap"><span class="short">{deco((r.get("B直融") or "").strip())}</span></td>'
            "</tr>")
        recs.append({
            "序号": f"{idx:03d}",
            "video": vid, "video_path": r.get("video_path") or "",
            "英文en": r.get("英文en") or "", "原zh": r.get("原zh") or "",
            "判错原因": r.get("判错原因") or "",
            "A重述": cell_caps_text(r.get("A重述")),
            "A长稿": (r.get("A长稿") or "").strip(),
            "A直融": (r.get("A直融") or "").strip(),
            "B重述": cell_caps_text(r.get("B重述")),
            "B长稿": (r.get("B长稿") or "").strip(),
            "B直融": (r.get("B直融") or "").strip(),
            "gif_abs": gif_abs})

    sub = (f"{len(rows)} 片 · GIF {n_gif}/{len(rows)}"
           + (f"（{n_fail} 片解码失败）" if n_fail else "")
           + " · 直融短稿=最终训练字幕(≤30字) · 长稿=细节参考 · B重述带属性徽标"
           " · gifs/ 文件名带 #序号 前缀=表格行序")
    page = (PAGE.replace("__ROWS__", "\n".join(body))
                .replace("__SUB__", sub).replace("__BTAG__", BUILD_TAG))
    out_html = os.path.join(args.out_dir, "table.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    out_csv = os.path.join(args.out_dir, "table.csv")
    write_csv(out_csv, recs)
    out_xlsx = os.path.join(args.out_dir, "table.xlsx")
    skip = write_xlsx(out_xlsx, recs, os.path.join(args.out_dir, "thumbs"))
    print(f"[done] {len(rows)} rows -> {out_html} (GIF {n_gif}, fail {n_fail})")
    print(f"[csv]  {out_csv} (utf-8-sig, Excel/WPS 直开)")
    if skip:
        print(f"[xlsx] 跳过: {skip}")
    else:
        print(f"[xlsx] {out_xlsx} (首列=GIF首帧静态缩略图; 动图看 table.html)")


if __name__ == "__main__":
    main()
