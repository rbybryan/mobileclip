#!/usr/bin/env python3
"""把一张抽检表打包成可离线评审的目录：视频 + 一个自带填写界面的 HTML。

为什么不用 §6 的 Gradio：
  1. 它需要 `gradio`，装进验证过的 `audit_env` 会拉进 fastapi/pandas/pydantic
     一堆依赖，有污染冻结环境的风险（这个仓库已经踩过 torch 被静默升级的坑）；
  2. 更根本的是**它是浏览数据集用的，没有抽检表的填写字段** —— 即使装上，
     也得一边看 Gradio 一边在 Excel 里按 video id 对着填，两个窗口来回切；
  3. 服务器无 GUI，看画面还得端口转发。

这个脚本把「看画面」和「填表」合成一件事：产出的目录 scp 到本地，
浏览器打开 `index.html` 即可 —— 视频、原字幕、候选字幕、单选按钮在同一屏，
进度存 localStorage（关掉再开不丢），填完点一个按钮导出 TSV，传回服务器统计。
全程离线、无外部依赖、不需要在服务器上装任何东西。

    python tools/data_audit/make_review_bundle.py \
        --sheet outputs/audit_10k/review_better_zh.tsv \
        --out-dir outputs/audit_10k/bundle_better_zh

然后：
    tar czf bundle.tgz -C outputs/audit_10k bundle_better_zh
    # 本地： scp ... && tar xzf bundle.tgz\n    #        然后双击 index.html（macOS: open / Linux: xdg-open）——见包内 README.txt

Deps: stdlib only（视频用 copy/硬链接，不转码）。
"""

import argparse
import csv
import html
import json
import os
import shutil
import sys

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>抽检 · __TITLE__</title>
<style>
 :root{--bd:#d6d9de;--mut:#666;--hi:#0b64c8}
 *{box-sizing:border-box}
 body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK SC",sans-serif;
      margin:0;background:#f4f5f7;color:#1b1e22}
 header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--bd);
        padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 header b{font-size:16px} .sp{flex:1}
 button{font:inherit;padding:7px 14px;border:1px solid var(--bd);border-radius:6px;
        background:#fff;cursor:pointer}
 button.p{background:var(--hi);color:#fff;border-color:var(--hi)}
 #prog{color:var(--mut)}
 .card{background:#fff;margin:14px auto;max-width:1080px;border:1px solid var(--bd);
       border-radius:10px;padding:14px 16px}
 .card.done{opacity:.55}
 .top{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
 video{width:420px;max-width:100%;height:auto;background:#000;border-radius:6px;
       position:sticky;top:58px}
 .nov{width:420px;max-width:100%;min-height:150px;display:flex;flex-direction:column;
      align-items:center;justify-content:center;text-align:center;color:#a12;
      background:#fff5f5;border:1px dashed #e0a0a0;border-radius:6px;padding:12px;word-break:break-all}
 .meta{flex:1;min-width:280px}
 .idx{color:var(--mut);font-size:13px}
 .cap{margin:8px 0;padding:8px 10px;border-radius:6px;background:#f7f8fa;border:1px solid #eceef1}
 .cap .lbl{font-size:12px;color:var(--mut);display:block;margin-bottom:2px}
 .orig{background:#fffdf2;border-color:#f0e6c0}
 .fused{background:#f0fff4;border-color:#a9dcb9}
 .fused .lbl{color:#1a7f37;font-weight:600}
 .filt{background:#fdf0f0;border-color:#e6b0b0}
 .filt .lbl{color:#a12;font-weight:600}
 .q{margin-top:10px;padding-top:10px;border-top:1px dashed var(--bd)}
 .q .name{font-weight:600;margin-right:8px}
 label.opt{display:inline-block;margin:3px 6px 3px 0;padding:4px 12px;border:1px solid var(--bd);
           border-radius:20px;cursor:pointer;user-select:none}
 label.opt:has(input:checked){background:var(--hi);color:#fff;border-color:var(--hi)}
 label.opt input{display:none}
 .note{width:100%;margin-top:6px;padding:6px 8px;border:1px solid var(--bd);border-radius:6px;font:inherit}
 .hint{max-width:1080px;margin:14px auto;color:var(--mut);font-size:14px;
       background:#fff;border:1px solid var(--bd);border-left:3px solid var(--hi);
       border-radius:6px;padding:10px 14px}
</style>
<header>
  <b>抽检 · __TITLE__</b>
  <span id="prog"></span><span id="vstat"></span><span class="sp"></span>
  <button onclick="jumpNext()">跳到下一条未填 (n)</button>
  <button class="p" onclick="dump()">导出 TSV</button>
</header>
<div class="hint">__HINT__<br>
进度自动存在浏览器本地，关掉再开不会丢。填完点<b>导出 TSV</b>，把下载到的文件传回服务器跑
<code>run_v100.sh review-score</code>。</div>
<div id="app"></div>
<script>
const ROWS=__ROWS__, COLS=__COLS__, QS=__QS__, KEY="rev_"+__KEY__;
const st=JSON.parse(localStorage.getItem(KEY)||"{}");
function save(){localStorage.setItem(KEY,JSON.stringify(st));prog()}
function prog(){
  const d=ROWS.filter((_,i)=>QS.some(q=>q.opts.length&&(st[i]||{})[q.name])).length;
  document.getElementById("prog").textContent=`已填 ${d} / ${ROWS.length}`;
  ROWS.forEach((_,i)=>document.getElementById("c"+i)?.classList.toggle("done",
    QS.filter(q=>q.opts.length).every(q=>(st[i]||{})[q.name])));
}
function set(i,k,v){(st[i]=st[i]||{})[k]=v;save()}
function jumpNext(){
  const i=ROWS.findIndex((_,i)=>!QS.filter(q=>q.opts.length).every(q=>(st[i]||{})[q.name]));
  if(i>=0)document.getElementById("c"+i).scrollIntoView({behavior:"smooth",block:"center"});
}
addEventListener("keydown",e=>{if(e.key==="n"&&!/INPUT|TEXTAREA/.test(e.target.tagName))jumpNext()});
const app=document.getElementById("app");
ROWS.forEach((r,i)=>{
  const d=document.createElement("div");d.className="card";d.id="c"+i;
  const CAPRE=/^(原zh|候选\\d|英文en|融合稿|重述\\d|A′视图|✓视图|·视图|✗视图|投票单元|淘汰单元|✓浓缩|·浓缩|✗浓缩|浓缩|直融长稿|直融|投票浓缩|A重述|B重述|A长稿|B长稿|A浓缩|B浓缩|A直融|A投票|B直融|B投票)$/;
  const caps=COLS.filter(c=>CAPRE.test(c)&&r[c])
    .map(c=>{const parts=String(r[c]).split(" ‖ ");
      const body=parts.length>1?parts.map((p,j)=>`${j+1}. ${p}`).join("<br>"):r[c];
      return `<div class="cap ${c==="原zh"?"orig":(c==="融合稿"||c[0]==="✓")?"fused":(c[0]==="✗"||c==="淘汰单元")?"filt":""}"><span class="lbl">${c}</span>${body}</div>`;}).join("");
  const other=COLS.filter(c=>!CAPRE.test(c)&&!/^(video|video_path)$/.test(c)&&r[c])
    .map(c=>`${c}=<b>${r[c]}</b>`).join(" · ");
  const qs=QS.map(q=>q.opts.length
    ? `<div class="q"><span class="name">${q.name}</span>`+q.opts.map(o=>
        `<label class="opt"><input type="radio" name="q${i}_${q.name}" value="${o}"
         ${(st[i]||{})[q.name]===o?"checked":""}
         onchange="set(${i},'${q.name}',this.value)">${o}</label>`).join("")+`</div>`
    : `<div class="q"><input class="note" placeholder="备注（可留空）" value="${(st[i]||{})[q.name]||""}"
        oninput="set(${i},'${q.name}',this.value)"></div>`).join("");
  const vhtml=r.__has
    ? `<video src="videos/${r.__f}" controls preload="metadata"
        onerror="this.outerHTML='<div class=nov>⚠️ 视频文件在，但浏览器放不了<br><small>多半是编码问题，见 README.txt</small></div>'"></video>`
    : `<div class="nov">⚠️ 此条视频未打包<br><small>只能看文本判；video_path=<br>${r.video_path||r.video}</small></div>`;
  d.innerHTML=`<div class="top">${vhtml}
    <div class="meta"><div class="idx">#${i+1} / ${ROWS.length} · ${r.video}</div>
    <div style="margin:4px 0 8px;color:#666;font-size:13px">${other}</div>${caps}</div></div>${qs}`;
  app.appendChild(d);
});
prog();
(function(){
  const has=ROWS.filter(r=>r.__has).length, el=document.getElementById("vstat");
  el.textContent=` · 视频 ${has}/${ROWS.length}`;
  if(has===0){el.style.color="#a12";el.textContent+=" ⚠️ 一个都没打包！见 README.txt";}
  else if(has<ROWS.length){el.style.color="#a60";}
})();
function dump(){
  const hdr=COLS.concat(QS.map(q=>q.col));
  const lines=[hdr.join("\\t")].concat(ROWS.map((r,i)=>
    COLS.map(c=>r[c]??"").concat(QS.map(q=>((st[i]||{})[q.name]||""))).join("\\t")));
  const b=new Blob([lines.join("\\n")+"\\n"],{type:"text/tab-separated-values"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);
  a.download=__FNAME__;a.click();
}
</script>
"""

HINTS = {
    "better_zh": "对着画面比较<b>原 zh</b> 与 3 条候选：候选整体上更准确吗？信息更具体吗？"
                 "<b>有没有说画面里没有的东西（幻觉）</b>——这一列最要紧。",
    "conflicts": "S0 规则说这条该重翻，而文本-only 实测说翻译没问题。"
                 "看画面和中英文本，判断<b>谁对</b>：<code>s0</code>=中文确实有未翻译/污染；"
                 "<code>vlm</code>=翻译其实没问题。",
    "random": "看画面判断这条字幕的质量，与模型给的 <code>zh分</code> 是否一致。",
    "merge": "对着画面看 <b>3 条候选</b> 和绿色的 <b>融合稿</b>："
             "<b>幻觉</b>=融合稿说了任何一条候选都没提、且画面里也没有的东西（y=有，最要紧）；"
             "<b>更全</b>=融合稿比其中写得最好的那一条候选含更多<b>真实</b>细节（y=更全）。",
    "compare": "A/B×两种融合的四版对比（同一批片、同一 prompt/门槛）：<b>A</b>=稠密重述×5，"
               "<b>B</b>=固定槽位×3（§5.6b：主体/动作/场景/文字 四栏、空栏写「无」）；"
               "<b>直融</b>=LLM 直接读 N 遍重述、语义数票+压缩一步"
               "（ASID 式，自报淘汰）；<b>投票</b>=规则跨遍计票后 LLM 只打包过票单元。"
               "对着画面做一道单选：<b>A直融/A投票/B直融/B投票</b> 哪版最适合当这段视频的"
               "训练字幕（信息覆盖、真实性、句子质量一起权衡）；全都差选「都不行」并在 "
               "note 写差在哪。",
    "pipeline_st": "（方案B 结构化臂）对着画面依次读：英文en → 黄色原zh（旧译）→ 重述"
                   "（每遍 5-8 条带属性短句，⟨⟩为 SigLIP 分）→ <b>直融</b>（LLM 一步语义"
                   "投票+压缩）与<b>投票浓缩</b>（规则计票后打包）。两道单选：重述质量、"
                   "版本对比（直融 vs 投票哪版更好）。",
    "pipeline": "对着画面依次读：<b>英文en</b>（源字幕）→ 黄色<b>原zh</b>（旧译，将被"
                "替换）→ <b>5 条重述</b>（⟨⟩内是 SigLIP 图文分，★A− 为分数前三）→ 两版"
                "最终字幕候选：<b>直融</b>（LLM 直接读全部重述、语义数票+压缩一步，ASID 式）"
                "与<b>投票浓缩</b>（规则跨遍计票后 LLM 只打包过票单元；⚠保底=没单元过门、"
                "按排序留前 3）。两版都已剔除验真失败句。两道单选：<b>重述质量</b>=重述整体"
                "是否比旧译更全且真实；<b>版本对比</b>=直融 vs 投票哪版更适合当训练字幕。",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True, help="review_*.tsv（make_review_sheet.py 产出）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--link", action="store_true",
                    help="用硬链接代替复制（同盘时省空间；打包 tar 时会自动展开）")
    args = ap.parse_args()

    with open(args.sheet, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit(f"{args.sheet} 是空的")
    hdr = [c for c in rows[0] if c]
    qcols = [c for c in hdr if "[" in c or c == "note"]
    cols = [c for c in hdr if c not in qcols]
    qs = [{"name": c.split("[")[0], "col": c,
           "opts": (c.split("[")[1].rstrip("]").split("/") if "[" in c else [])}
          for c in qcols]

    vdir = os.path.join(args.out_dir, "videos")
    os.makedirs(vdir, exist_ok=True)
    ok, miss = 0, []
    for r in rows:
        src = (r.get("video_path") or "").strip()
        fn = os.path.basename(src) or (r.get("video") or "").replace("/", "_")
        r["__f"] = fn
        dst = os.path.join(vdir, fn)
        if os.path.exists(dst):
            r["__has"] = True; ok += 1; continue
        if src and os.path.exists(src):
            try:
                (os.link if args.link else shutil.copy2)(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            r["__has"] = True; ok += 1
        else:
            r["__has"] = False; miss.append(r.get("video"))

    base = os.path.basename(args.sheet).replace("review_", "").replace(".tsv", "")
    page = (PAGE.replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
                .replace("__COLS__", json.dumps(cols, ensure_ascii=False))
                .replace("__QS__", json.dumps(qs, ensure_ascii=False))
                .replace("__KEY__", json.dumps(base))
                .replace("__FNAME__", json.dumps(os.path.basename(args.sheet)))
                .replace("__TITLE__", html.escape(base))
                .replace("__HINT__", HINTS.get(base, "按表头提示填写。")))
    idx = os.path.join(args.out_dir, "index.html")
    with open(idx, "w", encoding="utf-8") as f:
        f.write(page)

    with open(os.path.join(args.out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "抽检评审包 —— 本地打开方法\n"
            "================================\n\n"
            "1. 整个文件夹一起传/解压，index.html 和 videos/ 必须在一起。\n"
            "   只拷 index.html 会导致视频全部加载不了。\n\n"
            "2. 打开 index.html：\n"
            "   - Windows：直接双击\n"
            "   - macOS：  open index.html\n"
            "   - Linux：  xdg-open index.html   （`open` 是 macOS 命令，Linux 上没有）\n"
            "   或在浏览器里 文件→打开 选中它。\n\n"
            "3. 视频放不了时先看页头「视频 X/N」：\n"
            "   - N 分之 0：打包时服务器没找到视频（$CLIPS 路径不对），\n"
            "     需重打包，不是本地的问题。\n"
            "   - 个别放不了、显示「浏览器放不了」：多半是 H.265/HEVC 编码，\n"
            "     Chrome 放不了，换 Firefox 或用系统播放器打开对应 video_path。\n\n"
            "4. 填完点页面右上「导出 TSV」，把下载到的 TSV 传回服务器：\n"
            "   bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/<那张表>.tsv\n")

    size = sum(os.path.getsize(os.path.join(vdir, x)) for x in os.listdir(vdir))
    print(f"[bundle] {args.out_dir}")
    print(f"  视频 {ok}/{len(rows)} 份，合计 {size/1e6:.0f} MB")
    if miss:
        print(f"  ⚠️ {len(miss)} 条找不到视频（页面仍可填文本）：{miss[:3]}...")
    print(f"  打包： tar czf {base}.tgz -C {os.path.dirname(args.out_dir) or '.'} "
          f"{os.path.basename(args.out_dir)}")
    print(f"  本地： scp <server>:<path>/{base}.tgz . && tar xzf {base}.tgz")
    print(f"        再双击 {os.path.basename(args.out_dir)}/index.html"
          "（macOS: open · Linux: xdg-open · 见包内 README.txt）")


if __name__ == "__main__":
    main()
