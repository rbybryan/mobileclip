#!/usr/bin/env python3
"""汇总选帧基线(情形 B)的结果,输出 markdown 表 + 判读。

只依赖标准库。读 experiments/evals/frame_select/<bench>_<arm>/eval_res.json。

两个口径都报(仓库规矩:结论必须两口径同向才可信):
  ret_test/      DSL(dual-softmax,测试集级先验)
  ret_test_emb/  原始余弦 —— **论文主口径**

列名映射(见 experiments/README.md §3.4,别读反):
  img_r1 = t2v R@1     txt_r1 = v2t R@1

## 判据为什么钉在 R@1 而不是 r_mean

本实验里模型权重和测试集都固定,唯一变的是喂进去的帧 ⇒ 复跑逐位可复现,
不存在训练种子噪声。真正的不确定性来自**测试集本身只有 1000 条查询**的二项
抽样误差:p≈0.42 时 SE = sqrt(p(1-p)/1000) ≈ 1.6 个点。这与 GATE-0 判 reverse
的 −1.2 为噪声用的是同一条带(`experiments/gate0_results.md` §3.3)。

`r_mean` 是 6 个召回的平均,它们在同一批查询上高度相关,SE 介于 1.6 与
1.6/sqrt(6) 之间 —— **没有干净的闭式**。与其编一个阈值,这里只把 r_mean 当
描述性数字报,判据一律走 R@1 + 两口径同向。

⚠️ 功效上限:1K 测试集只能可靠地看到 >= 1.6 R@1 的效应。若 divmax 相对
uniform 只差 1 个点左右,本实验**判不了**,要换更大的测试集(MSR-VTT 全量
3k / DiDeMo / ANet)才能定论 —— 别把"没测出来"写成"没有效应"。
"""
import argparse
import glob
import json
import os
import re

# MSR-VTT 1K 上 R@1 的二项抽样误差(见上文推导)
NOISE_R1 = 1.6

CALIBERS = [("ret_test_emb", "raw(主口径)"), ("ret_test", "DSL")]
MODE_ORDER = {"uniform": 0, "divmax": 1, "random": 2}


def load_arms(root):
    arms = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "eval_res.json"))):
        name = os.path.basename(os.path.dirname(path))
        bench, _, arm = name.partition("_")
        with open(path) as f:
            arms.setdefault(bench, {})[arm] = json.load(f)
    return arms


def get(res, caliber, metric):
    try:
        return float(res[caliber][metric])
    except (KeyError, TypeError, ValueError):
        return None


def fmt(v, width=6):
    return f"{v:>{width}.2f}" if isinstance(v, float) else " " * (width - 1) + "-"


def delta_r1(v, base):
    """R@1 差值 + 噪声标注。只对 R@1 用,不要拿去标 r_mean。"""
    if not isinstance(v, float) or not isinstance(base, float):
        return "-"
    d = v - base
    return f"{d:+.2f}" + ("" if abs(d) >= NOISE_R1 else " (噪声内)")


def arm_sort_key(arm):
    if m := re.match(r"sel(\w+)_K(\d+)$", arm):
        return (2, int(m.group(2)), MODE_ORDER.get(m.group(1), 9), m.group(1))
    if m := re.match(r"nativeT(\d+)$", arm):
        return (1, int(m.group(1)), 0, "")
    return (0, 0, 0, arm)


def both_calibers_agree(arms, a, b, metric="img_r1"):
    """两口径上 a-b 的差值是否同号且都超噪声。返回 (是否成立, raw 差值)。"""
    ds = []
    for caliber, _ in CALIBERS:
        va, vb = get(arms.get(a, {}), caliber, metric), get(arms.get(b, {}), caliber, metric)
        if not isinstance(va, float) or not isinstance(vb, float):
            return None, None
        ds.append(va - vb)
    raw = ds[0]
    ok = all(abs(d) >= NOISE_R1 for d in ds) and (ds[0] > 0) == (ds[1] > 0)
    return ok, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/evals/frame_select")
    args = ap.parse_args()

    arms_by_bench = load_arms(args.root)
    if not arms_by_bench:
        print(f"没找到任何结果:{args.root}/*/eval_res.json")
        print("先跑 bash experiments/run_frame_select.sh")
        return

    for bench, arms in arms_by_bench.items():
        print(f"\n## {bench}\n")
        for caliber, label in CALIBERS:
            if not any(get(r, caliber, "img_r1") is not None for r in arms.values()):
                continue
            print(f"### {label} (`{caliber}/`)\n")
            print("| 臂 | t2v R@1 | v2t R@1 | r_mean(仅描述) | t2v R@1 vs 同 K native |")
            print("|---|---|---|---|---|")
            for arm in sorted(arms, key=arm_sort_key):
                res = arms[arm]
                m = re.match(r"sel\w+_K(\d+)$", arm)
                ref = arms.get(f"nativeT{m.group(1)}") if m else None
                print(
                    f"| `{arm}` "
                    f"| {fmt(get(res, caliber, 'img_r1'))} "
                    f"| {fmt(get(res, caliber, 'txt_r1'))} "
                    f"| {fmt(get(res, caliber, 'r_mean'))} "
                    f"| {delta_r1(get(res, caliber, 'img_r1'), get(ref, caliber, 'img_r1') if ref else None)} |"
                )
            print()

        # ---- 判读:t2v R@1,两口径同向,±%.1f 以内不算数 ----
        print(f"**判读(判据 = t2v R@1,两口径同向,|Δ| < {NOISE_R1} 一律不算效应)**\n")
        ks = sorted({int(m.group(1)) for a in arms
                     if (m := re.match(r"sel\w+_K(\d+)$", a))})
        for K in ks:
            out = [f"- **K={K}**:"]

            ok_floor, d_floor = both_calibers_agree(arms, f"seluniform_K{K}", f"selrandom_K{K}")
            if ok_floor is False:
                out.append(
                    f"⚠️ **地板先看**:uniform 相对 random 只有 {d_floor:+.2f},没过噪声带 ⇒ "
                    f"在 K={K} 上「挑哪几帧」可能根本不构成问题,下面 divmax 的任何差值都别当真。"
                )

            ok, d = both_calibers_agree(arms, f"seldivmax_K{K}", f"seluniform_K{K}")
            if ok is None:
                out.append("臂不齐,无法判读。")
            elif not ok:
                out.append(
                    f"divmax 相对 uniform {d:+.2f} —— **未过判据**(噪声内或两口径不同向)⇒ "
                    f"廉价多样性选帧在 K={K} 上**无可证增益**,C2 在这一点上不受威胁。"
                )
            elif d > 0:
                out.append(f"divmax 相对 uniform **{d:+.2f}**,两口径同向 ⇒ **选帧确有增益**。")
                base = get(arms.get(f"nativeT{K}", {}), "ret_test_emb", "img_r1")
                pool = get(arms.get("pool16", {}), "ret_test_emb", "img_r1")
                if isinstance(base, float) and isinstance(pool, float) and pool - base > 0:
                    out.append(
                        f"低 T 缺口(pool16 − nativeT{K})= {pool - base:.2f},零训练选帧补回 "
                        f"**{100 * d / (pool - base):.0f}%** —— **C2 的净增益必须扣掉这一块再报**。"
                    )
            else:
                out.append(
                    f"divmax 相对 uniform **{d:+.2f}(更差)**,两口径同向 ⇒ 多样性选帧在本语料上**有害**;"
                    f"最可能的机制是 k-center 偏好离群帧,选中了糊帧/闪光帧/转场帧(见 utils/frame_select.py)。"
                )
            print(" ".join(out))

        print()
        print("**兼容性核对(必看)**:`pool16` 应复现 ROADMAP 1.6 的 T=16 读数,")
        print("`nativeT2/T4` 应复现 1.6 的 T=2/T=4 读数。对不上说明评测链路被改动了,")
        print("先查这个再读上面的结论。")
        print()
        print("**报结果时必须一并写明**:divmax 只用降采样像素打分、不过编码器,是")
        print("**可部署**方案;用编码器特征选帧要先把母序列全编码一遍,那只能当上界报,")
        print("不是同成本对照(见 utils/frame_select.py 顶部「成本诚实性」)。")


if __name__ == "__main__":
    main()
