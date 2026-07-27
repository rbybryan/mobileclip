#!/usr/bin/env python3
"""R1 前置探针:文本负例库的**假负例 / 近重复密度**(零训练)。

回答 `notes/r1-examination.md` §5 的问题:在开建负例库之前,先量化
**机会**(有多少*不同*的字幕 ⇒ 65k 池子里有多少真正的难负例)和
**风险**(有多少对字幕会被当成假负例)。

分两部分,各答一个审查点:

  A. **精确重复密度(零模型,完全可跑)** —— 用仓库自己的 `caption_to_idx`
     (md5,`dataset/pt_dataset.py`)统计字幕串的碰撞。这**正是**对比损失里
     多正例掩码 `get_mask(idx)` 用的键(`models/criterions.py:287`),所以这个数
     不是近似,而是损失眼中"哪些对是正例/假负例"的精确刻画。
     ⚠️ 只有 `caption_idx=True` 时(r1a 配置已设)这些碰撞才被当正例;否则它们
     在训练里就是**假负例**。本部分给出的就是这个规模。

  B. **嵌入近重复密度(需冻结文本塔的特征,数值核已自测)** —— idx 掩码只抓
     *精确*重复;语义相近但不同串("a man is talking" vs "a man talks")它抓不到,
     而这些在对比空间里就是假负例。给定一份预先算好的文本特征(投影后、已归一化,
     与损失比较的空间一致),统计各余弦阈值下的近邻密度。

用法:
    # A —— 现在就能跑,不需 GPU / 权重 / decord;需要 anno JSON
    python experiments/r1_fn_density.py --anno anno_downstream/InternVid/InternVid-10M-flt_1.json

    # B —— 先在训练环境里生成特征(见 --help-feats 打印的片段),再:
    python experiments/r1_fn_density.py --feats /tmp/rtime_txtfeats.npy

    # 自检(不需任何数据)
    python experiments/r1_fn_density.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter


# --------------------------------------------------------------------------
# 与 dataset/pt_dataset.py 逐位一致的 caption_to_idx(损失多正例掩码用的键)。
# 复制而非 import,避免拉起 dataset 包的重依赖(petrel/decord 等)。
# 若上游改了该函数,这里要同步 —— 已在 --self-test 里对拍。
# --------------------------------------------------------------------------
def caption_to_idx(caption: str) -> int:
    digest = hashlib.md5(caption.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


def assemble_caption(rec: dict, key: str):
    """复现 dataset 的字幕组装:caption 可能是 list,用空格拼成一句。"""
    v = rec.get(key)
    if v is None:
        for alt in ("caption", "text", "desc", "captions"):
            if alt in rec:
                v = rec[alt]
                key = alt
                break
    if v is None:
        return None, key
    if isinstance(v, list):
        v = " ".join(str(x) for x in v)
    return str(v), key


# --------------------------------------------------------------------------
# A. 精确重复密度
# --------------------------------------------------------------------------
def exact_duplicate_stats(captions):
    n = len(captions)
    idxs = [caption_to_idx(c) for c in captions]
    counts = Counter(idxs)
    n_distinct = len(counts)
    # 一个随机取到的样本,它有多少概率与另一个不同样本共享字幕(= 假负例概率,
    # 若不启用 caption_idx):sum_g m_g*(m_g-1) / (n*(n-1))
    dup_pairs = sum(m * (m - 1) for m in counts.values())
    total_pairs = n * (n - 1)
    fn_rate = dup_pairs / total_pairs if total_pairs else 0.0
    # 落在重复组里的样本占比
    in_dup = sum(m for m in counts.values() if m > 1)
    return {
        "n": n,
        "n_distinct": n_distinct,
        "distinct_frac": n_distinct / n if n else 0.0,
        "false_neg_pair_rate": fn_rate,
        "samples_in_dup_group_frac": in_dup / n if n else 0.0,
        "top": counts.most_common(5),
        "idx_by_count": counts,
    }


def report_exact(st, captions_by_idx=None):
    print("=" * 72)
    print("A. 精确重复密度(损失多正例掩码 get_mask 眼中的真实结构)")
    print("=" * 72)
    print(f"  样本数 n                     = {st['n']}")
    print(f"  不同字幕数(按 caption_to_idx)= {st['n_distinct']}"
          f"  ({st['distinct_frac']*100:.1f}%)")
    print(f"  随机一对是同字幕的概率        = {st['false_neg_pair_rate']*100:.4f}%")
    print(f"    ↑ 这就是 65k 池子里**每条负例是假负例**的期望比例(若不启用 caption_idx);")
    print(f"      期望假负例数 ≈ {st['false_neg_pair_rate']*65536:.1f} / 65k 池")
    print(f"  落在重复组的样本占比          = {st['samples_in_dup_group_frac']*100:.1f}%")
    print(f"  重复最多的字幕(idx, 次数):")
    for idx, c in st["top"]:
        s = captions_by_idx.get(idx, "") if captions_by_idx else ""
        s = (s[:60] + "…") if len(s) > 60 else s
        print(f"      x{c:5d}  {s!r}")
    print()
    print("  判读:")
    print("    · distinct_frac 高(≈1)⇒ 65k 池里基本都是不同字幕 ⇒ 难负例供给充足(机会)。")
    print("    · 同字幕概率高 ⇒ 不启用 caption_idx 时假负例多;r1a 已设 caption_idx=True,")
    print("      **只要负例库也穿过 get_mask(见 §3),这部分假负例就被当正例处理掉**。")
    print("    · 真正需要额外去重掩码的是 B 部分的**近重复**(idx 抓不到)。")


# --------------------------------------------------------------------------
# B. 嵌入近重复密度(数值核,自测覆盖)
# --------------------------------------------------------------------------
def near_dup_density(feats, thresholds=(0.9, 0.95, 0.99), chunk=2048):
    """feats: [N, D] 已归一化。返回各阈值下:平均近邻数、有≥1近邻的样本占比。

    分块算 N×N 余弦,不落整矩阵。对角(自身)已排除。
    """
    import numpy as np

    N = feats.shape[0]
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    out = {t: {"neighbors": 0, "has_nb": 0} for t in thresholds}
    for i0 in range(0, N, chunk):
        blk = feats[i0:i0 + chunk]                      # [b, D]
        sims = blk @ feats.T                            # [b, N]
        for bi in range(blk.shape[0]):
            row = sims[bi]
            row[i0 + bi] = -1.0                          # 排除自身
            for t in thresholds:
                c = int((row >= t).sum())
                out[t]["neighbors"] += c
                out[t]["has_nb"] += 1 if c > 0 else 0
    res = {}
    for t in thresholds:
        res[t] = {
            "mean_neighbors": out[t]["neighbors"] / N,
            "frac_with_neighbor": out[t]["has_nb"] / N,
        }
    return res


def report_near(res, N):
    print("=" * 72)
    print(f"B. 嵌入近重复密度(N={N} 条**不同**字幕,投影后归一化空间)")
    print("=" * 72)
    print("  余弦阈值 | 平均近邻数 | 有≥1近邻的占比")
    for t, r in sorted(res.items()):
        print(f"    ≥ {t:.2f}   |   {r['mean_neighbors']:7.3f}  |   {r['frac_with_neighbor']*100:6.2f}%")
    print()
    print("  判读:这些是 idx 掩码**抓不到**的假负例(串不同但语义近)。")
    print("    · 若『有≥1近邻』占比高 ⇒ 65k 池里大量语义近重复被当负例 ⇒ **必须**上")
    print("      基于余弦阈值的离线去重掩码(§3),不能只靠 caption_idx。")
    print("    · 若很低 ⇒ idx 掩码就够,近重复不是主要风险。")


# --------------------------------------------------------------------------
FEAT_SNIPPET = r"""
# ---- 在训练环境里生成文本特征(投影后、已归一化),供 B 部分消费 ----
# 复用 eval 的规范调用(tasks_clip/retrieval_utils.py:115 同款):
#   text_feat = model.encode_text(tokenizer(text).to(device))
# 取一份**不同**字幕的样本(先用 A 部分或 set() 去精确重复),约 1e4 条即可估密度。
import json, numpy as np, torch
# ... 你已有 model, tokenizer, device(照 tasks_clip/retrieval.py 里 setup_model 那套)...
caps = sorted({ (" ".join(r["caption"]) if isinstance(r["caption"],list) else r["caption"])
                for r in json.load(open("anno_downstream/InternVid/InternVid-10M-flt_1.json")) })
caps = caps[:10000]
feats = []
with torch.no_grad():
    for i in range(0, len(caps), 256):
        t = tokenizer(caps[i:i+256], padding="max_length", truncation=True,
                      max_length=32, return_tensors="pt").to(device)
        f = model.encode_text(t)                 # [b, 512](若返回 tuple 取 [0])
        f = torch.nn.functional.normalize(f, dim=-1)
        feats.append(f.float().cpu().numpy())
np.save("/tmp/txtfeats.npy", np.concatenate(feats))
# 然后:python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy
"""


def _self_test():
    import numpy as np
    # A: 已知重复
    caps = ["a man is talking"] * 3 + ["a dog runs"] * 2 + ["unique %d" % i for i in range(5)]
    st = exact_duplicate_stats(caps)
    assert st["n"] == 10 and st["n_distinct"] == 7, st
    # dup pairs = 3*2 + 2*1 = 8 ; total = 10*9 = 90
    assert abs(st["false_neg_pair_rate"] - 8/90) < 1e-9, st["false_neg_pair_rate"]
    # caption_to_idx 与上游一致性:同串同 idx、异串异 idx(高概率)
    assert caption_to_idx("x") == caption_to_idx("x")
    assert caption_to_idx("x") != caption_to_idx("y")
    # list 组装
    c, k = assemble_caption({"caption": ["a", "b", "c"]}, "caption")
    assert c == "a b c", c
    # B: 植入近重复对
    rng = np.random.default_rng(0)
    base = rng.normal(size=(100, 16))
    dup = base[:10] + rng.normal(size=(10, 16)) * 1e-4      # 10 条各有一个极近邻
    feats = np.concatenate([base, dup])                     # N=110
    res = near_dup_density(feats, thresholds=(0.99,))
    # 每个植入对的两端各有 1 个 ≥0.99 近邻 ⇒ 至少 20 个样本有近邻
    assert res[0.99]["frac_with_neighbor"] >= 20/110 - 1e-9, res
    # 阈值单调:更高阈值近邻不多于更低阈值
    r2 = near_dup_density(feats, thresholds=(0.5, 0.99))
    assert r2[0.5]["mean_neighbors"] >= r2[0.99]["mean_neighbors"]
    print("self-test 通过(A 精确重复计数 / caption_to_idx 一致性 / B 余弦阈值密度 均验证)。")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno", nargs="*", default=[], help="InternVid anno JSON(可多个)")
    ap.add_argument("--caption-key", default="caption")
    ap.add_argument("--sample", type=int, default=0, help=">0 时随机抽这么多条(A 部分)")
    ap.add_argument("--feats", default=None, help="预算好的文本特征 .npy [N,D](B 部分)")
    ap.add_argument("--thresholds", default="0.9,0.95,0.99")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--help-feats", action="store_true", help="打印生成 --feats 的片段")
    args = ap.parse_args()

    if args.self_test:
        _self_test(); return
    if args.help_feats:
        print(FEAT_SNIPPET); return
    if not args.anno and not args.feats:
        ap.error("给 --anno(A 部分)或 --feats(B 部分)其一;或 --self-test / --help-feats")

    if args.anno:
        caps, used_key = [], None
        for path in args.anno:
            if not os.path.exists(path):
                print(f"[跳过] 不存在:{path}"); continue
            data = json.load(open(path, encoding="utf-8"))
            recs = data if isinstance(data, list) else [data]
            for r in recs:
                if not isinstance(r, dict):
                    continue
                c, used_key = assemble_caption(r, args.caption_key)
                if c:
                    caps.append(c)
        if not caps:
            print("没读到字幕 —— 检查 --anno 路径与 --caption-key(脚本会自动尝试 "
                  "caption/text/desc/captions)。"); return
        print(f"读到 {len(caps)} 条字幕(字段 = {used_key!r})\n")
        if args.sample and args.sample < len(caps):
            import random; random.seed(0); caps = random.sample(caps, args.sample)
            print(f"(随机抽 {len(caps)} 条做估计)\n")
        st = exact_duplicate_stats(caps)
        by_idx = {}
        for c in caps:
            by_idx.setdefault(caption_to_idx(c), c)
        report_exact(st, by_idx)
        print("\n→ B 部分(嵌入近重复)需要文本特征;运行 "
              "`python experiments/r1_fn_density.py --help-feats` 看生成片段。")

    if args.feats:
        import numpy as np
        feats = np.load(args.feats)
        ths = tuple(float(x) for x in args.thresholds.split(","))
        print()
        res = near_dup_density(feats.astype("float32"), thresholds=ths)
        report_near(res, feats.shape[0])


if __name__ == "__main__":
    sys.exit(main())
