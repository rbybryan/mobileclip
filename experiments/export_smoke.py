#!/usr/bin/env python3
"""导出冒烟(ROADMAP 1.5 / 评审 U1)——零训练、零 GPU、随机权重。

## 任务与验收(`notes/plan-review-upgrades.md` U1,原文照抄要点)

- 用**随机权重**实例化带全部时序算子的视觉塔,分别走
  **ONNX(→ onnxruntime 加载)/ TFLite / coremltools 转换**,
  按 **T∈{2,4,8,16}** 各出一张静态图;
- 记录每条路线 **pass/fail 与失败算子名**;
- 第二档(真机延迟)仍留在第 8 周,本脚本**不测延迟**。

**为什么现在做**:转换性是 P9(分离注意力变体)触发条件的全部信息量,今天就能拿到。
若联合注意力确认导不出,P9 需**提前进入 R3 的架构决策视野** —— 也就是要决定弹性
模型是否直接用可导出形态训练,避免第 8 周再微调一个部署变体。

## 导出的是什么

视觉塔(`FastViT`),按 `models/mobileviclip_small.py:build_vision_encoder` 的
**同一套架构参数**构造,因此包含全部时序算子(RepMixer 1d 分支、AttentionBlock
的 `temporal_embedding`、时间维池化)。文本塔不导出 —— 端侧部署里文本 embedding
通常离线算好,延迟档只看视觉侧。

输入签名与 `encode_vision` 对齐:**[B, T, C, H, W]**(脚本内部做 permute,
与模型代码里那一行一致),输出 [B, 512]。

## 三条路线各自独立

任何一条缺依赖 ⇒ 记 `skip(缺 xxx)` 而**不是**让整个脚本崩。所以在只装了
onnx 的机器上也能拿到 ONNX 那一列的结论。

用法:

    python experiments/export_smoke.py --mini          # 自检脚本本身(秒级)
    python experiments/export_smoke.py                 # 全尺寸,T∈{2,4,8,16}
    python experiments/export_smoke.py --inference-mode   # 重参数化后的部署形态

依赖(装多少跑多少):

    pip install onnx onnxruntime          # ONNX 路线
    pip install ai-edge-torch             # TFLite 路线(Google 官方 torch->tflite)
    pip install coremltools               # Core ML 路线
"""
import argparse
import json
import os
import re
import sys
import traceback
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TS = (2, 4, 8, 16)

# 从异常文本里抠算子名。不同后端措辞不同,全都收集,原始报错另外保留。
_OP_PATTERNS = (
    r"aten::([A-Za-z0-9_.]+)",                          # torch/onnx 通用
    r"torch\.ops\.aten\.([A-Za-z0-9_.]+)",
    r"[Ee]xporting the operator '?([A-Za-z0-9_:.]+)'?",  # torch.onnx
    r"[Uu]nsupported:? ONNX export of (?:operator )?'?([A-Za-z0-9_:.]+)'?",
    r'Op "([A-Za-z0-9_.]+)" .*not implemented',          # coremltools
    r"[Cc]onversion for torch operator '?([A-Za-z0-9_.]+)'? not implemented",
    r"PyTorch convert function for op '([A-Za-z0-9_.]+)' not implemented",
)


def extract_ops(text: str):
    """从报错文本里抽出疑似「不支持的算子」名字,去重保序。"""
    found, seen = [], set()
    for pat in _OP_PATTERNS:
        for m in re.findall(pat, text or ""):
            if m not in seen:
                seen.add(m)
                found.append(m)
    return found


# ---------------------------------------------------------------- 模型
def build_vision_tower(mini: bool = False, inference_mode: bool = False):
    """随机权重的视觉塔。

    非 mini 分支的架构参数与 `models/mobileviclip_small.py:build_vision_encoder`
    **逐项一致**(layers / embed_dims / mlp_ratios / downsamples / se_downsamples /
    pos_embs / token_mixers / projection_dim),否则导出结论对不上真模型。
    mini 分支照抄 `tests/test_t_agnostic.py:_mini_fastvit`,只用来自检脚本本身。
    """
    import torch
    from models.mobileviclip_small import FastViT, RepCPE

    torch.manual_seed(0)
    common = dict(
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
        downsamples=[True, True, True, True],
        se_downsamples=[False, False, True, True],
        pos_embs=[None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
        temporal_pooling="mean",
        with_cp=False,                       # 导出不要梯度检查点
        inference_mode=inference_mode,
    )
    if mini:
        model = FastViT(layers=[1, 1, 1, 1], embed_dims=[16, 32, 64, 640],
                        mlp_ratios=[1, 1, 1, 1], **common)
    else:
        model = FastViT(layers=[4, 12, 24, 4], embed_dims=[80, 160, 320, 640],
                        mlp_ratios=[3, 3, 3, 3], clip_embed_dim=512,
                        attn_pool_num_heads=16, head_drop_path_rate=0.0,
                        projection_dim=512, **common)
    return model.eval()


def apply_skip_temporal(tower) -> int:
    """round-5 §7 P1 结构性旁路:让每个 RepMixer 直接跳过时序支路。

    效率量测不需要零门(延迟/图规模不依赖权重值,§3 检查 8),所以这里**不**要求
    λ_1d 已置零 —— 与 `MobileViCLIP_Small` 的 skip_temporal_branch flag(精度路径,
    强制先置零)刻意不同。返回命中的 RepMixer 数(全尺寸=40,mini=3)。
    """
    from models.mobileviclip_small import RepMixer

    n = 0
    for m in tower.modules():
        if isinstance(m, RepMixer):
            m.skip_temporal = True
            n += 1
    assert n > 0, "没有找到 RepMixer —— --skip-temporal 无处生效"
    return n


def _wrap(tower):
    """把 [B,T,C,H,W] 的部署输入签名包上去。

    `MobileViCLIP_Small.encode_vision` 里就是这一行 permute
    (`image.permute(0, 2, 1, 3, 4)`),导出时必须带上,否则导出的图输入布局
    与端侧喂进来的不一致。
    """
    import torch.nn as nn

    class _Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, video):                     # [B, T, C, H, W]
            return self.inner(video.permute(0, 2, 1, 3, 4))   # -> [B, C, T, H, W]

    return _Wrapped(tower).eval()


# ---------------------------------------------------------------- 三条路线
def route_onnx(model, example, out_path, opset: int):
    """导出 + **用 onnxruntime 真加载一次**(U1 明确要求后半步)。"""
    import torch
    import onnx                                        # noqa: F401 - 缺就 skip
    import onnxruntime as ort

    torch.onnx.export(
        model, (example,), out_path,
        input_names=["video"], output_names=["embed"],
        opset_version=opset, dynamic_axes=None,        # U1 要静态图
    )
    g = onnx.load(out_path)
    onnx.checker.check_model(g)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"video": example.numpy()})[0]
    # 图规模统计:P1 效率主张(基线 1581 节点 / 340 layout op vs 旁路 1141)的判据。
    # layout op 口径 = 纯数据搬运/变形,不算任何 FLOP。
    hist = {}
    for node in g.graph.node:
        hist[node.op_type] = hist.get(node.op_type, 0) + 1
    layout_kinds = ("Transpose", "Reshape", "Unsqueeze", "Squeeze", "Flatten", "Expand")
    return {"loaded_ok": True, "out_shape": list(out.shape),
            "bytes": os.path.getsize(out_path),
            "nodes": len(g.graph.node),
            "layout_ops": sum(hist.get(k, 0) for k in layout_kinds),
            "op_hist_top": sorted(hist.items(), key=lambda kv: -kv[1])[:8]}


class BackendUnavailable(Exception):
    """后端装了但入口对不上(改名/换 API)—— 是环境问题,**不是导不出**。

    必须与真正的导出失败区分开:两者都会让那一列变红,但只有后者是 P9 的证据。
    """


def route_tflite(model, example, out_path, opset: int):
    """torch -> tflite。

    ⚠️ 包名迁移过:`ai-edge-torch` 已废弃并**改名为 `litert-torch`**;pip 装
    `ai-edge-torch` 只会拿到一个 1.8 kB 的空壳,`ai_edge_torch.convert` 不存在
    (实测 2026-07-27 在 101vector 上就是这么炸的)。所以两个名字都试,
    都找不到入口就抛 BackendUnavailable ⇒ 记 skip 而不是 fail。
    """
    mod = None
    for name in ("litert_torch", "ai_edge_torch"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        if hasattr(mod, "convert"):
            break
        mod = None
    if mod is None:
        raise BackendUnavailable(
            "litert_torch / ai_edge_torch 都没有 convert 入口。"
            "`ai-edge-torch` 已改名 `litert-torch`,试 `pip install litert-torch`。"
        )
    mod.convert(model, (example,)).export(out_path)
    return {"backend": mod.__name__, "bytes": os.path.getsize(out_path)}


def route_coreml(model, example, out_path, opset: int):
    """coremltools 要 TorchScript,先 trace。"""
    import coremltools as ct
    import torch

    traced = torch.jit.trace(model, example, strict=False)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="video", shape=tuple(example.shape))],
        convert_to="mlprogram",
    )
    mlmodel.save(out_path)
    return {"saved": os.path.basename(out_path)}


ROUTES = (
    # onnxscript:torch>=2.6 的 dynamo ONNX 导出器需要它,缺了会在导出中途抛
    # ModuleNotFoundError —— 声明在这里,以免变成一条看起来像"导不出"的 fail。
    ("onnx", route_onnx, "onnx", ("onnx", "onnxruntime", "onnxscript")),
    # 两个名字任一即可(ai-edge-torch 已改名 litert-torch);具体入口由 route 再验。
    ("tflite", route_tflite, "tflite", ("litert_torch|ai_edge_torch",)),
    ("coreml", route_coreml, "mlpackage", ("coremltools",)),
)


def missing_deps(mods):
    """哪些声明依赖没装。条目里的 `a|b` 表示两者装其一即可(包改名的情况)。"""
    import importlib.util

    def present(spec):
        return any(importlib.util.find_spec(m) is not None for m in spec.split("|"))

    return [m for m in mods if not present(m)]


def run_one(name, fn, ext, deps, model, example, outdir, t, opset):
    """跑一条路线。**任何失败都不得抛出** —— 三条路线必须互不影响。

    ⚠️ `ModuleNotFoundError` 必须归为 **skip 而非 fail**。这条不是洁癖:本表是
    P9 的触发判据,把"环境缺件"记成"导不出"会产生**假的 P9 触发**,进而让 R3
    的架构决策建立在不存在的证据上。实测踩过 —— torch 2.12 的 ONNX 导出器需要
    `onnxscript`,缺它时报的是 ModuleNotFoundError,第一版把它记成了 fail。
    """
    miss = missing_deps(deps)
    if miss:
        return {"status": "skip", "reason": f"缺依赖 {', '.join(miss)}", "ops": []}
    path = os.path.join(outdir, f"vision_T{t}.{ext}")
    try:
        extra = fn(model, example, path, opset)
        return {"status": "pass", "path": path, **extra}
    except ModuleNotFoundError as exc:                        # 传递依赖缺失 != 导不出
        return {"status": "skip",
                "reason": f"缺传递依赖 {exc.name}(装上再跑,别读成 fail)",
                "ops": []}
    except BackendUnavailable as exc:                         # 后端入口对不上 != 导不出
        return {"status": "skip", "reason": f"后端 API 不匹配:{exc}", "ops": []}
    except Exception as exc:                                  # noqa: BLE001 - 就是要全兜住
        # `module 'X' has no attribute 'Y'` 只可能是**库版本/API 不匹配**,
        # 不可能是「这个算子导不出」——后者一律走 NotImplementedError /
        # torch.onnx.errors.* / 后端自己的 "not implemented" 文案。
        # 实测两次都栽在这:`ai_edge_torch.convert` 缺失(包改名)、
        # `torch.int1` 缺失(litert-torch 要 torch>=2.6,本机锁 2.5.1 给 V100 的 sm_70)。
        # 归 fail 就是**假 P9 触发**,所以这里必须归 skip。
        if isinstance(exc, AttributeError) and \
                re.search(r"module '[^']+' has no attribute", str(exc)):
            return {"status": "skip",
                    "reason": f"库版本/API 不匹配:{exc}(不是导出失败)", "ops": []}
        tb = traceback.format_exc()
        return {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}".strip()[:400],
            "ops": extract_ops(tb),
            "traceback_tail": tb.strip().splitlines()[-3:],
        }


# ---------------------------------------------------------------- 报告
def render(results, ts, meta):
    L = [f"# 导出冒烟结果(ROADMAP 1.5 / U1)\n",
         f"- 模型:{meta['model']}(随机权重,`inference_mode={meta['inference_mode']}`,"
         f"`skip_temporal={meta.get('skip_temporal', False)}`)",
         f"- 输入:`[{meta['batch']}, T, 3, {meta['img']}, {meta['img']}]`,ONNX opset {meta['opset']}",
         f"- torch {meta['torch']} · 本表**不含延迟**(第二档仍在第 8 周)\n",
         "| T | " + " | ".join(n for n, *_ in ROUTES) + " |",
         "|---|" + "---|" * len(ROUTES)]
    mark = {"pass": "✅ pass", "fail": "❌ fail", "skip": "⏭️ skip"}
    for t in ts:
        row = [f"| {t} "]
        for name, *_ in ROUTES:
            r = results[str(t)][name]
            cell = mark[r["status"]]
            if r["status"] == "skip":
                cell += f"({r['reason']})"
            elif r["status"] == "fail" and r["ops"]:
                cell += f" — `{'`, `'.join(r['ops'][:3])}`"
            elif r["status"] == "pass" and "nodes" in r:
                cell += f"({r['nodes']}n/{r['layout_ops']}l)"
            row.append(f"| {cell} ")
        L.append("".join(row) + "|")

    # 失败算子汇总:P9 决策要的就是这张清单
    ops = {}
    for t in ts:
        for name, *_ in ROUTES:
            r = results[str(t)][name]
            if r["status"] == "fail":
                ops.setdefault(name, set()).update(r["ops"])
    L.append("\n## 失败算子清单(P9 触发判据的全部依据)\n")
    if not ops:
        L.append("无失败路线,或失败信息里未出现可识别的算子名(看 JSON 里的 traceback)。")
    for name, s in ops.items():
        L.append(f"- **{name}**:" + (", ".join(f"`{o}`" for o in sorted(s)) or "未能从报错中抽出算子名"))

    L.append("\n## 怎么用这张表\n")
    L.append("- **三条路线都 pass** ⇒ P9(分离注意力变体)**不触发**,弹性模型按现形态训练;")
    L.append("- **联合注意力导不出**(失败算子指向 attention/`temporal_embedding` 相关)")
    L.append("  ⇒ P9 **提前进入 R3 架构决策**:要决定弹性模型是否直接用可导出形态训练,")
    L.append("  避免第 8 周再单独微调一个部署变体(U1 的原话)。")
    L.append("- **只是缺依赖(skip)** ⇒ 不是结论,装上再跑;别把 skip 读成 pass。")
    L.append("\n验收去处:把本表贴进 `notes/live/from-claude.md` 或 `TROUBLESHOOTING.md`(U1)。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=",".join(map(str, DEFAULT_TS)))
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--mini", action="store_true", help="缩小版,只用来自检脚本本身")
    ap.add_argument("--inference-mode", action="store_true",
                    help="重参数化后的部署形态(RepMixer 合并分支)")
    ap.add_argument("--skip-temporal", action="store_true",
                    help="P1 结构性旁路:RepMixer 跳过整条时序支路(round-5 §7)。"
                         "与基线各跑一次、diff ONNX 节点数即为 P1 的效率判据;"
                         "⚠️ 用不同 --outdir,否则文件互相覆盖")
    ap.add_argument("--outdir", default="experiments/exports")
    ap.add_argument("--json", default=None, help="默认写到 <outdir>/export_smoke.json")
    args = ap.parse_args()

    import torch

    ts = [int(x) for x in args.ts.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    img = 64 if args.mini else args.img

    print(f"构造视觉塔({'mini' if args.mini else '全尺寸'},"
          f"inference_mode={args.inference_mode})…", flush=True)
    inner = build_vision_tower(args.mini, args.inference_mode)
    if args.skip_temporal:
        n = apply_skip_temporal(inner)
        print(f"[P1-旁路] 已在 {n} 个 RepMixer 上开启 skip_temporal", flush=True)
    tower = _wrap(inner)

    results = {}
    for t in ts:
        example = torch.randn(args.batch, t, 3, img, img)
        with torch.no_grad():                       # 先确认 eager 能跑,免得把模型 bug 当导出失败
            try:
                shape = list(tower(example).shape)
                eager = f"eager OK {shape}"
            except Exception as exc:                # noqa: BLE001
                eager = f"eager FAILED: {type(exc).__name__}: {exc}"
        print(f"\n== T={t} == {eager}", flush=True)
        results[str(t)] = {}
        for name, fn, ext, deps in ROUTES:
            r = run_one(name, fn, ext, deps, tower, example, args.outdir, t, args.opset)
            results[str(t)][name] = r
            detail = r.get("reason") or r.get("error") or r.get("path", "")
            if "nodes" in r:
                detail += f" · {r['nodes']} 节点 / {r['layout_ops']} layout"
            print(f"   {name:<7} {r['status']:<5} {detail}", flush=True)
            if r.get("ops"):
                print(f"           疑似算子: {', '.join(r['ops'][:6])}", flush=True)

    meta = {"model": "mini FastViT" if args.mini else "FastViT(= build_vision_encoder 同参)",
            "inference_mode": args.inference_mode, "skip_temporal": args.skip_temporal,
            "batch": args.batch, "img": img, "opset": args.opset, "torch": torch.__version__}
    report = render(results, ts, meta)
    out_json = args.json or os.path.join(args.outdir, "export_smoke.json")
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)
    out_md = os.path.splitext(out_json)[0] + ".md"
    with open(out_md, "w") as f:
        f.write(report + "\n")
    print(f"\n{report}\n\nJSON: {out_json}\nMD:   {out_md}")


if __name__ == "__main__":
    main()
