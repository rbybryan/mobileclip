#!/usr/bin/env python3
"""C-2(round4-P6 聚合扫描)+ C-4(§4 方差-vs-覆盖判别)—— **一次前向,多种打分**。

    python experiments/frame_agg_probe.py \
        --config scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py \
        --ckpt ./checkpoints/mobileviclip_small.pt --ts 8,16

**为什么合成一个脚本**:两件事要的输入是同一个 —— **逐帧嵌入**。
拿到 (Nv,T,D) 之后,P6 的 {mean,max,LSE,top-k} 和 §4 的 {单随机帧, oracle 最佳帧,
mean-T} 都只是同一张 (Nv,T,Nt) 相似度张量上的不同归约,**不需要再跑一次前向**。

**为什么不改评测路径**:`encode_vision` 先把 T 塌掉,要保住逐帧就得动
`retrieval_utils` 那条所有结论都依赖的链路。这里改用**加法**:模型上一个默认关闭的
`return_frames` 开关(现有路径逐位不变),再由本脚本自己算分。

v2(2026-08-04,首轮结果逼出来的三件事):
  1. **τ 敏感度**:首轮 LSE(τ=0.05,v1 写死的默认)+1.15/+1.56 r_mean 高于 mean,
     但只测了一个 τ。现在 `--taus` 整条扫出来。**头条仍报 τ=0.05**(它是在看到任何
     结果之前定下的,等效预登记);扫描只当敏感度曲线,**不许**事后挑最好的 τ 当头条
     —— 那是在测试集上选超参。
  2. **配对 CI**:裸差不是判据(house rule)。每个聚合器 vs mean 做逐查询配对
     bootstrap(与 §3.9 同一套机器),CI 不含 0 才算数。
  3. **两口径**:raw 之外补 DSL(与 `evaluation_video_clip` 完全同式:
     `S·softmax(S,dim=0)`),并把逐查询 rank 落盘成 `ranks_ret_test{,_emb}.npz`,
     后续任何两臂都能用 `collect_results.py --compare` 重比。

判读(§4 预登记):
  · **oracle 最佳帧 ≫ mean(>1.0)** ⇒ T 的价值是**覆盖**(某些帧才有答案)
    ⇒ 处方是 late-interaction / max 类聚合器;
  · **oracle ≈ mean(或更差)而 mean 随 T 稳步上升** ⇒ 价值是**方差缩减**
    ⇒ 处方是继续用一阶池化,P9 那类二阶统计才有立足点。
  **这两者给出相反的处方**,所以这一项值得单独跑。

⚠️ 口径提示:本脚本自建前向,绝对值与 `retrieval.py` 管线略有偏移
(首轮:probe mean-8 raw 60.77 vs 管线 61.55)。**脚本内部的方法间差值可判,
跨管线的绝对值不可比。**
"""
import argparse
import os
import sys

import numpy as np
import torch


def aggregate(per_frame, method, tau=0.05, frame_idx=None):
    """per_frame: (Nv, T, Nt) 逐帧相似度 → (Nv, Nt)。"""
    if method == "mean":
        return per_frame.mean(1)
    if method in ("max", "oracle_frame"):        # oracle 最佳帧 == 逐查询取 max
        return per_frame.max(1).values
    if method == "lse":
        return tau * torch.logsumexp(per_frame / tau, dim=1)
    if method in ("top2", "top4"):
        k = min(int(method[3:]), per_frame.shape[1])
        return per_frame.topk(k, dim=1).values.mean(1)
    if method == "single_frame":
        return per_frame[:, frame_idx if frame_idx is not None else 0]
    raise ValueError(method)


def ranks_both_dirs(S):
    """S: (Nv, Nt) 聚合后相似度,对角线是正例。

    返回 {"i2t": ranks, "t2i": ranks}(命名与 `paired_bootstrap.RECALL_SPECS` 对齐:
    i2t = 视频查文本 ⇒ txt_r*;t2i = 文本查视频 ⇒ img_r*)。
    rank = 严格大于正例分数的竞争者个数(连续分数下与 `itm_eval` 的 argsort 位置一致)。
    """
    out = {}
    for key, s in (("i2t", S), ("t2i", S.t())):
        gt = s.diagonal().unsqueeze(1)
        out[key] = (s > gt).sum(1).cpu().numpy().astype(np.int32)
    return out


def dsl(S):
    """与 `evaluation_video_clip` 完全同式的 dual-softmax 口径(retrieval_utils.py:154-155)。

    返回 (i2t_dsl, t2i_dsl) 两张已重加权的矩阵 —— 注意 DSL 的 i2t 与 t2i **不再互为转置**,
    所以不能像 raw 那样只传一张。
    """
    return S * S.softmax(dim=0), S.t() * S.t().softmax(dim=0)


def ranks_dsl(S):
    i2t_m, t2i_m = dsl(S)
    out = {}
    for key, s in (("i2t", i2t_m), ("t2i", t2i_m)):
        gt = s.diagonal().unsqueeze(1)
        out[key] = (s > gt).sum(1).cpu().numpy().astype(np.int32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ts", default="8,16", help="逐个 T 各跑一次(mean-8 vs mean-16 要这个)")
    ap.add_argument("--taus", default="0.02,0.05,0.1,0.2,0.5,1.0",
                    help="LSE 的 τ 敏感度扫描;头条口径固定 0.05(预登记),别删它")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="配对 bootstrap 重采样次数(house 标准 10000;全套约 10–20 min CPU)")
    ap.add_argument("--dump-dir", default="experiments/frame_agg",
                    help="逐查询 rank 落盘根目录(每方法一个子目录,可喂 collect_results --compare)")
    ap.add_argument("--out", default="experiments/frame_agg_results.md")
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                    help="配置覆盖(可重复)。示例:LSE×gate4 可加性 = "
                         "--set model.vision_encoder.disable_temporal_embedding true "
                         "--dump-dir experiments/frame_agg_gate4(⚠️ 换 dump-dir,别覆盖基线 rank)")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from utils.config import Config, eval_dict_leaf
    from dataset import create_dataset, create_loader
    from models.mobileviclip_small import MobileViCLIP_Small, set_num_frames
    from experiments.paired_bootstrap import save_ranks, paired_bootstrap, recalls_from_ranks
    from experiments.bn_recalibrate import set_dotted, parse_value

    if args.set and args.dump_dir == "experiments/frame_agg":
        print("⚠️ 开了 --set 但 --dump-dir 还是默认值 —— 会覆盖基线 rank,建议换目录", flush=True)

    taus = [float(x) for x in args.taus.split(",")]
    # (标签, method, tau)。mean 是配对基线;single_frame 只进 C-4 读数,不进 CI。
    METHODS = ([("mean", "mean", None), ("max", "max", None),
                ("top2", "top2", None), ("top4", "top4", None),
                ("single_frame", "single_frame", None)]
               + [(f"lse@{t:g}", "lse", t) for t in taus])

    all_ranks = {}      # {(T, 标签, 口径): {"i2t":…, "t2i":…}}
    rows = []           # (T, 标签, 口径, recalls dict) 打表用
    for T in [int(x) for x in args.ts.split(",")]:
        # 🔴 `Config.from_file` **不做** ${...} 插值 —— 插值在 `get_config` 里,
        #    而那条路要解析 argv。不自己补 eval_dict_leaf,train_file 会保持字面量
        #    `${available_corpus[${train_corpus}]}`(2026-07-31 就是这样炸的)。
        cfg = eval_dict_leaf(Config.from_file(args.config))
        cfg.num_frames_test = T
        # 插值已经把它固化成 8 了,所以**必须**再显式覆盖一次
        cfg.inputs.video_input.num_frames_test = T
        cfg.distributed = False
        for k, v in args.set:               # --set 覆盖(插值之后,建模之前)
            set_dotted(cfg, k, parse_value(v))
            print(f"[set] {k} = {parse_value(v)}", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MobileViCLIP_Small(config=cfg, tokenizer=None, is_pretrain=False)
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        for k in ("model", "module", "state_dict"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
        missing, _ = model.load_state_dict(sd, strict=False)
        vis_missing = [k for k in missing if not k.startswith("text_encoder.")]
        if len(vis_missing) > 20:
            raise SystemExit(f"[FAIL] 视觉塔没对上;前 5: {vis_missing[:5]}")
        # 🔴 结构 flag(--set disable_temporal_* 等)是**置零参数**,刚被上面的 load 冲掉
        #    ⇒ 载后必须重放,否则 flag 静默失效(gate 评测当年为此被迫用消融 ckpt 绕路)
        if hasattr(model, "apply_arch_flags"):
            model.apply_arch_flags()
        model = model.to(device).eval()
        model.vision_encoder.return_frames = True      # ← 唯一的开关(默认 False)
        set_num_frames(T)

        # **只建评测集** —— setup_dataloaders 会连训练集一起建,而 eval config 的
        # train_file 本来就不是给它用的(同上,那正是炸点)。
        test_datasets, _names = create_dataset("ret_eval", cfg)
        loader = create_loader(
            test_datasets, [None] * len(test_datasets),
            batch_size=[cfg.inputs.batch_size_test[d.media_type] for d in test_datasets],
            num_workers=[cfg.num_workers] * len(test_datasets),
            is_trains=[False] * len(test_datasets),
            collate_fns=[None] * len(test_datasets),
        )[0]

        # 🔴 照 `evaluation_video_clip`(retrieval_utils.py:107-150)抄 —— 那才是本模型
        #   真正走的评测路径。两处必须照搬,否则要么报错、要么**静默算出对不上的数**:
        #   ① 文本用 `model.encode_text(tokenizer(text).to(device))`。**不要**用
        #      retrieval_utils.py:19 那个 InternVideo2 的 HF-tokenizer 取特征函数 ——
        #      它会去取 `text_input.attention_mask`,而 ClipTokenizer 返回**裸 Tensor**,
        #      于是 `'Tensor' object has no attribute 'attention_mask'`。
        #   ② **两侧都不再单独乘投影头** —— 投影已经在塔内(`GlobalPool2D(out_dim=512)`),
        #      `encode_vision` / `encode_text` 的输出就是最终嵌入。多乘一次不会报错,
        #      只会得到和既有口径对不上的数,**那比报错更糟**。
        texts = loader.dataset.text
        with torch.no_grad():
            tfs = []
            for i0 in range(0, len(texts), 256):
                tf = model.encode_text(model.tokenizer(texts[i0:i0 + 256]).to(device))
                tfs.append(tf.float().cpu())
            Tx = torch.nn.functional.normalize(torch.cat(tfs), dim=-1)

            vfeats = []
            for image, _ in loader:                    # 载入器产出 (image, idx)
                image = image.to(device, non_blocking=True)
                f = model.encode_vision(image, test=True)         # (B,T,D),return_frames 已开
                vfeats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu())
        V = torch.cat(vfeats)                          # (Nv,T,D)
        if V.ndim != 3:
            raise SystemExit(f"[FAIL] 期望逐帧 (Nv,T,D),拿到 {tuple(V.shape)} "
                             "⇒ return_frames 没生效")
        if V.shape[0] != Tx.shape[0]:
            raise SystemExit(f"[FAIL] 本脚本按对角线正例打分,要求 Nv == Nt;"
                             f"拿到 Nv={V.shape[0]} Nt={Tx.shape[0]}(多字幕基准要先改 GT 映射)")
        per_frame = torch.einsum("vtd,nd->vtn", V, Tx)

        for label, m, tau in METHODS:
            S = aggregate(per_frame, m, tau if tau is not None else 0.05)
            pair = {"ret_test_emb": ranks_both_dirs(S), "ret_test": ranks_dsl(S)}
            line = f"T={T:<3} {label:<13}"
            for cal in ("ret_test_emb", "ret_test"):
                all_ranks[(T, label, cal)] = pair[cal]
                rec = recalls_from_ranks({k: v.astype(np.int64) for k, v in pair[cal].items()})
                rows.append((T, label, cal, rec))
                tag = "raw" if cal == "ret_test_emb" else "dsl"
                line += (f"  {tag} r_mean={rec['r_mean']:.2f}"
                         f" t2v_r1={rec['img_r1']:.2f} v2t_r1={rec['txt_r1']:.2f}")
                d = os.path.join(args.dump_dir, f"T{T}_{label.replace('@', '_tau')}")
                save_ranks(d, cal + "/", pair[cal]["i2t"], pair[cal]["t2i"])
            print(line, flush=True)

    # ---------------- C-4 预登记判读(两个 T 都报,别只报第一个) ----------------
    Ts = sorted({t for t, _, _, _ in rows})
    verdicts = []
    gaps = {}
    for T in Ts:
        r = {lab: rec["r_mean"] for t, lab, cal, rec in rows if t == T and cal == "ret_test_emb"}
        gaps[T] = r["max"] - r["mean"]
    gap_txt = " · ".join(f"T={T} {gaps[T]:+.2f}" for T in Ts)
    coverage = any(g > 1.0 for g in gaps.values())
    verdicts.append(
        f"[verdict] C-4 oracle(max)−mean(raw):{gap_txt}(判据 >1.0)⇒ "
        + ("**覆盖说**成立:处方 = late-interaction / max 类聚合器。" if coverage else
           "**方差缩减说**成立(hard-max 甚至更差):一阶池化没错;"
           "P6 的**训练**族按 kill 表维持死亡,P9 二阶统计不复活。"))
    raw_rm = {(t, lab): rec["r_mean"] for t, lab, cal, rec in rows if cal == "ret_test_emb"}
    if all((T, "lse@0.05") in raw_rm for T in Ts):      # --taus 里删了 0.05 就只报 C-4
        verdicts.append(
            "[verdict] C-2 LSE(τ=0.05,预登记)− mean(raw r_mean):"
            + " · ".join(f"T={T} {raw_rm[(T, 'lse@0.05')] - raw_rm[(T, 'mean')]:+.2f}" for T in Ts)
            + " ⇒ 是否算数看下面 [CI] 行(两口径同向且不含 0)。")
    for v in verdicts:
        print("\n" + v, flush=True)

    # ---------------- 配对 CI:每个聚合器 vs mean(两口径) ----------------
    ci_rows = []
    ci_methods = [lab for lab, m, _ in METHODS if lab not in ("mean", "single_frame")]
    total = len(Ts) * len(ci_methods) * 2
    done = 0
    for T in Ts:
        for cal in ("ret_test_emb", "ret_test"):
            base = {k: v.astype(np.int64) for k, v in all_ranks[(T, "mean", cal)].items()}
            for lab in ci_methods:
                arm = {k: v.astype(np.int64) for k, v in all_ranks[(T, lab, cal)].items()}
                res = paired_bootstrap(arm, base, n_boot=args.n_boot, seed=0)
                ci_rows.append((T, lab, cal, res))
                done += 1
                rm, rs = res["r_mean"], res["R_sum"]
                mark = lambda x: "*" if x["significant"] else " "
                print(f"[CI] T={T:<3} {lab:<9} {'raw' if cal.endswith('emb') else 'dsl'} vs mean  "
                      f"Δr_mean={rm['delta']:+.2f} [{rm['lo']:+.2f},{rm['hi']:+.2f}]{mark(rm)} "
                      f"ΔR_sum={rs['delta']:+.2f} [{rs['lo']:+.2f},{rs['hi']:+.2f}]{mark(rs)} "
                      f"({done}/{total})", flush=True)

    # ---------------- markdown ----------------
    md = ["# C-2 聚合扫描 + C-4 方差-vs-覆盖判别(v2:τ 扫描 + 配对 CI + 两口径)", "",
          "口径:本脚本自建前向,**方法间差值可判,跨管线绝对值不可比**"
          "(probe mean-8 raw ≈ 60.8 vs 管线 61.55)。",
          "逐查询 rank 已落盘 `experiments/frame_agg/T*_*/ranks_ret_test{,_emb}.npz`,"
          "可用 `collect_results.py --compare` 任意两臂重比。", "",
          "| T | 聚合 | raw r_mean | raw t2v R@1 | raw v2t R@1 | DSL r_mean |",
          "|---|---|---|---|---|---|"]
    by_key = {(t, lab, cal): rec for t, lab, cal, rec in rows}
    for T in Ts:
        for label, _m, _tau in METHODS:
            raw = by_key[(T, label, "ret_test_emb")]
            d_ = by_key[(T, label, "ret_test")]
            md.append(f"| {T} | {label} | {raw['r_mean']:.2f} | {raw['img_r1']:.2f} "
                      f"| {raw['txt_r1']:.2f} | {d_['r_mean']:.2f} |")
    md += ["", "## 配对 CI(vs mean,同脚本内同一前向,10000 次重采样,seed=0)", "",
           "| T | 聚合 | 口径 | Δr_mean | 95% CI | ΔR_sum | 95% CI | 判定 |",
           "|---|---|---|---|---|---|---|---|"]
    for T, lab, cal, res in ci_rows:
        rm, rs = res["r_mean"], res["R_sum"]
        sig = "**显著**" if rm["significant"] else "不显著"
        md.append(f"| {T} | {lab} | {'raw' if cal.endswith('emb') else 'DSL'} "
                  f"| {rm['delta']:+.2f} | [{rm['lo']:+.2f}, {rm['hi']:+.2f}] "
                  f"| {rs['delta']:+.2f} | [{rs['lo']:+.2f}, {rs['hi']:+.2f}] | {sig} |")
    md += [""] + [v.replace("[verdict] ", "**") + "**" for v in verdicts] + ["",
           "⚠️ 纪律:头条只报 **τ=0.05**(v1 写死的默认,看到结果前已定 = 预登记);"
           "τ 扫描只当敏感度曲线。若别的 τ 更好,要先在**别的基准/验证集**上复核才允许换。", "",
           "⚠️ 部署语义:相似度级聚合 ⇒ 检索索引要存 **T 份逐帧嵌入**(×T 存储 / ×T 点积),"
           "不是 gate4 那种改完就免费的减法 —— 报收益时把这条成本一起报。"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(md) + "\n")
    print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
