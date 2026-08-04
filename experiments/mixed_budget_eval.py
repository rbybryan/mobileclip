#!/usr/bin/env python3
"""混合预算(T × R)离线组合评测 —— C2-2D 的闸门 1(§11.8-N1 / publishability §1.1)。

## 它回答什么

同样的算力预算,`8帧@181²`、`4帧@224²`、混合 `2帧@256² + 6帧@160²` 谁赢?
这是 C2 → C2-2D 升级的第一道骨牌(publishability-assessment.md §1.1 条件 1),
**零训练**:原料是标准 eval 在若干 (T, R) 单点上落盘的嵌入,组合在这里用 numpy 做。

## 为什么离线线性组合是**精确**的(不是近似)

前提 = 「逐帧独立域」三开关全开(评测时用环境变量,见下):

    MVC_DISABLE_TEMPORAL_BRANCH=1     # P1:时序分支恒等(1M 已证非劣)
    MVC_DISABLE_TEMPORAL_EMBEDDING=1  # P5:唯一 T 形状参数置零(eval 版显著为正)
    MVC_PER_FRAME_ATTENTION=1         # 新:stage-4 注意力帧内块对角

此时整塔逐帧严格独立,clip 嵌入 = 帧嵌入的**无权平均**(head 是 GlobalPool mean),
且落盘的是**未归一化**嵌入(归一化在 get_sim 里)⇒

    E(t1@R1 + t2@R2) = (t1·E(t1@R1) + t2·E(t2@R2)) / (t1+t2)

逐位成立。顺带:单独比较「三开关 vs 基线」就是 §11.7-2 全减法闸门本身。

## 用法(从仓库根)

    # 1) 打印需要的单点评测命令(不跑):
    python experiments/mixed_budget_eval.py --print-unit-commands \
        --ckpt checkpoints/mobileviclip_small.pt --root experiments/evals/mb

    # 2) 照打印的命令逐条跑完单点(每条 ~1 分钟)之后,组合出结果表:
    python experiments/mixed_budget_eval.py --root experiments/evals/mb \
        --out experiments/mixed_budget_results.md

    # 自定义分配(格式 "t@R" 或 "t1@R1+t2@R2"):
    python experiments/mixed_budget_eval.py --root ... --alloc "8@256" "2@256+6@160"

判读:表按 cost 排;同 cost 档里混合分配是否赢均匀分配。配对显著性用现成的
    python experiments/collect_results.py --compare <alloc目录A> <alloc目录B>
(本脚本给每个分配存了与标准评测同款的 ranks npz。)

⚠️ cost 列用 t·(R/256)² 的解析代理(卷积塔 MACs ≈ ∝ R²);要换成 A7 的实测
逐分辨率 MACs,改 `_cost_of` 一处即可。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

# 预注册分配(改动要先记 notes;单点集合由这里自动推出)
DEFAULT_ALLOCS = [
    # 均匀(1D 基线)
    "8@256", "8@224", "8@181", "8@160", "8@128",
    "4@256", "4@224", "2@256",
    # 混合(2D 候选)
    "2@256+6@160", "1@256+7@160", "2@256+6@128",
    "2@224+6@160", "1@256+4@128", "2@181+4@128",
]

ENV_FLAGS = ("MVC_DISABLE_TEMPORAL_BRANCH=1 MVC_DISABLE_TEMPORAL_EMBEDDING=1 "
             "MVC_PER_FRAME_ATTENTION=1")


def parse_alloc(spec: str):
    """'2@256+6@160' -> [(2, 256), (6, 160)]"""
    groups = []
    for part in spec.split("+"):
        t, r = part.strip().split("@")
        groups.append((int(t), int(r)))
    assert groups, f"空分配:{spec}"
    return groups


def _cost_of(groups) -> float:
    """成本代理:Σ t·(R/256)²(∝ MACs;要换 A7 实测数就改这里)。"""
    return sum(t * (r / 256.0) ** 2 for t, r in groups)


def unit_dir(root: str, t: int, r: int) -> str:
    return os.path.join(root, f"t{t}_r{r}")


def load_unit(root: str, t: int, r: int):
    path = os.path.join(unit_dir(root, t, r), "emb_dump.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"缺单点嵌入:{path}\n先用 --print-unit-commands 打印命令并跑完对应单点。")
    z = np.load(path)
    return z["video_emb"].astype(np.float32), z["text_emb"].astype(np.float32)


def load_id_maps(root: str, units):
    """任一单点目录里的 id_maps.json;都没有则退化为恒等映射(MSR-VTT 1k 即如此)。"""
    for t, r in units:
        p = os.path.join(unit_dir(root, t, r), "id_maps.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f)
            txt2img = m.get("txt2img")
            img2txt = m.get("img2txt")
            if isinstance(txt2img, dict):
                txt2img = [txt2img[str(i)] for i in range(len(txt2img))]
            if isinstance(img2txt, dict):
                img2txt = [img2txt[str(i)] for i in range(len(img2txt))]
            return txt2img, img2txt
    return None, None


def metrics_for(sim_i2t: np.ndarray, txt2img, img2txt):
    """复用仓库自己的 itm_eval(口径逐位一致),两口径都算。"""
    from tasks_clip.retrieval_utils import itm_eval

    n = sim_i2t.shape[0]
    if txt2img is None:
        txt2img = list(range(n))
        img2txt = [[i] for i in range(n)]

    def _softmax0(x):
        e = np.exp(x - x.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    out = {}
    for name, i2t in (
        ("ret_test_emb/", sim_i2t),                       # raw
        ("ret_test/", sim_i2t * _softmax0(sim_i2t)),      # DSL(retrieval_utils:154 同款)
    ):
        t2i = i2t.T if name.endswith("_emb/") else sim_i2t.T * _softmax0(sim_i2t.T)
        ranks_out = {}
        out[name] = (itm_eval(i2t, t2i, txt2img, img2txt, ranks_out=ranks_out), ranks_out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="单点嵌入根目录(t{T}_r{R}/ 子目录)")
    ap.add_argument("--alloc", nargs="*", default=None, help='如 "8@256" "2@256+6@160"')
    ap.add_argument("--out", default=None, help="结果 markdown 输出路径")
    ap.add_argument("--print-unit-commands", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints/mobileviclip_small.pt",
                    help="--print-unit-commands 用")
    ap.add_argument("--bench", default="msrvtt", help="--print-unit-commands 用")
    args = ap.parse_args()

    specs = args.alloc if args.alloc else DEFAULT_ALLOCS
    allocs = [(s, parse_alloc(s)) for s in specs]
    units = sorted({g for _, groups in allocs for g in groups})

    if args.print_unit_commands:
        print(f"# 需要 {len(units)} 个单点评测(每条 ~1 分钟;从仓库根跑):")
        for t, r in units:
            d = unit_dir(args.root, t, r)
            print(
                f"MVC_DUMP_EMB_DIR={d} {ENV_FLAGS} "
                f"bash experiments/run_eval.sh small {args.bench} {args.ckpt} {d} "
                f"num_frames_test {t} inputs.image_res {r} model.vision_encoder.img_size {r}"
            )
        print("# 跑完后:python experiments/mixed_budget_eval.py --root", args.root,
              "--out experiments/mixed_budget_results.md")
        return

    emb = {}
    text_ref = None
    for t, r in units:
        v, txt = load_unit(args.root, t, r)
        emb[(t, r)] = v
        if text_ref is None:
            text_ref = txt
        else:
            # 文本塔冻结 ⇒ 各单点的文本嵌入必须一致(fp16 容差);不一致说明单点跑错了
            assert np.abs(text_ref - txt).max() < 1e-2, f"t{t}_r{r} 的 text_emb 与首个单点不一致"

    from experiments.paired_bootstrap import save_ranks

    rows = []
    for spec, groups in allocs:
        total_t = sum(t for t, _ in groups)
        combined = sum(t * emb[(t, r)] for t, r in groups) / float(total_t)
        vn = combined / np.linalg.norm(combined, axis=-1, keepdims=True)
        tn = text_ref / np.linalg.norm(text_ref, axis=-1, keepdims=True)
        sim_i2t = vn @ tn.T

        txt2img, img2txt = load_id_maps(args.root, units)
        res = metrics_for(sim_i2t, txt2img, img2txt)

        adir = os.path.join(args.root, "alloc_" + spec.replace("@", "r").replace("+", "_"))
        for name, (m, ranks_out) in res.items():
            save_ranks(adir, name, ranks_out["i2t"], ranks_out["t2i"])
        raw, dsl = res["ret_test_emb/"][0], res["ret_test/"][0]
        rows.append((spec, _cost_of(groups), raw, dsl, adir))

    rows.sort(key=lambda r: r[1])
    lines = [
        "# 混合预算 (T×R) 组合评测 —— C2-2D 闸门 1",
        "",
        "口径:raw = `ret_test_emb/`(主)· DSL = `ret_test/`;cost = Σ t·(R/256)²。",
        "逐帧独立域三开关全开(P1 + P5 + per-frame-attn)。配对显著性:",
        "`collect_results.py --compare <alloc目录A> <alloc目录B>`。",
        "",
        "| 分配 | cost | raw r_mean | raw R_sum | DSL r_mean | DSL R_sum | ranks 目录 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    def _rsum(m):
        return sum(m[k] for k in ("txt_r1", "txt_r5", "txt_r10", "img_r1", "img_r5", "img_r10"))

    for spec, cost, raw, dsl, adir in rows:
        lines.append(
            f"| `{spec}` | {cost:.2f} | {raw['r_mean']:.2f} | {_rsum(raw):.1f} "
            f"| {dsl['r_mean']:.2f} | {_rsum(dsl):.1f} | {adir} |"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
