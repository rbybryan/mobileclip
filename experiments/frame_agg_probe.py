#!/usr/bin/env python3
"""C-2(round4-P6 聚合扫描)+ C-4(§4 方差-vs-覆盖判别)—— **一次前向,多种打分**。

    python experiments/frame_agg_probe.py \
        --config scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py \
        --ckpt ./checkpoints/mobileviclip_small.pt --ts 8,16

**为什么合成一个脚本**:两件事要的输入是同一个 —— **逐帧嵌入**。
拿到 (Nv,T,D) 之后,P6 的 {mean,max,LSE,top-2} 和 §4 的 {单随机帧, oracle 最佳帧,
mean-T} 都只是同一张 (Nv,T,Nt) 相似度张量上的不同归约,**不需要再跑一次前向**。

**为什么不改评测路径**:`encode_vision` 先把 T 塌掉,要保住逐帧就得动
`retrieval_utils` 那条所有结论都依赖的链路。这里改用**加法**:模型上一个默认关闭的
`return_frames` 开关(现有路径逐位不变),再由本脚本自己算分。

判读(§4 预登记):
  · **oracle 最佳帧 ≫ mean** ⇒ T 的价值是**覆盖**(某些帧才有答案)
    ⇒ 处方是 late-interaction / max 类聚合器;
  · **oracle ≈ mean 而 mean 随 T 稳步上升** ⇒ 价值是**方差缩减**
    ⇒ 处方是继续用一阶池化,P9 那类二阶统计才有立足点。
  **这两者给出相反的处方**,所以这一项值得单独跑。
"""
import argparse
import sys

import torch


def _r_metrics(sim):
    """sim: (Nv, Nt);对角线是正例。返回 t2v / v2t 的 R@1/5/10 与 r_mean。"""
    n = sim.shape[0]
    gt = torch.arange(n, device=sim.device)
    out = {}
    for name, s in (("v2t", sim), ("t2v", sim.t())):
        rank = (s > s.gather(1, gt.view(-1, 1))).sum(1)
        for k in (1, 5, 10):
            out[f"{name}_r{k}"] = (rank < k).float().mean().item() * 100
    out["r_mean"] = sum(out[f"{n_}_r{k}"] for n_ in ("v2t", "t2v") for k in (1, 5, 10)) / 6
    return out


def aggregate(per_frame, method, tau=0.05, frame_idx=None):
    """per_frame: (Nv, T, Nt) 逐帧相似度 → (Nv, Nt)。"""
    if method == "mean":
        return per_frame.mean(1)
    if method in ("max", "oracle_frame"):        # oracle 最佳帧 == 逐查询取 max
        return per_frame.max(1).values
    if method == "lse":
        return tau * torch.logsumexp(per_frame / tau, dim=1)
    if method == "top2":
        k = min(2, per_frame.shape[1])
        return per_frame.topk(k, dim=1).values.mean(1)
    if method == "single_frame":
        return per_frame[:, frame_idx if frame_idx is not None else 0]
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ts", default="8,16", help="逐个 T 各跑一次(mean-8 vs mean-16 要这个)")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--out", default="experiments/frame_agg_results.md")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from utils.config import Config, eval_dict_leaf
    from dataset import create_dataset, create_loader
    from models.mobileviclip_small import MobileViCLIP_Small, set_num_frames
    from tasks_clip.retrieval_utils import extract_text_feats

    rows = []
    for T in [int(x) for x in args.ts.split(",")]:
        # 🔴 `Config.from_file` **不做** ${...} 插值 —— 插值在 `get_config` 里,
        #    而那条路要解析 argv。不自己补 eval_dict_leaf,train_file 会保持字面量
        #    `${available_corpus[${train_corpus}]}`(2026-07-31 就是这样炸的)。
        cfg = eval_dict_leaf(Config.from_file(args.config))
        cfg.num_frames_test = T
        # 插值已经把它固化成 8 了,所以**必须**再显式覆盖一次
        cfg.inputs.video_input.num_frames_test = T
        cfg.distributed = False

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

        # 文本:完全照 retrieval_utils.evaluation 的算法,别自己造
        max_txt_l = cfg.inputs.max_txt_l
        if not isinstance(max_txt_l, int):
            max_txt_l = max_txt_l[loader.dataset.media_type]
        with torch.no_grad():
            text_feats, _atts = extract_text_feats(
                loader.dataset.text, max_txt_l, model.tokenizer, model, device)
            Tx = torch.nn.functional.normalize(
                model.text_proj(text_feats[:, 0]), dim=-1).float().cpu()

            vfeats = []
            for image, _ in loader:                     # 载入器产出 (image, idx)
                image = image.to(device, non_blocking=True)
                f = model.vision_proj(model.encode_vision(image, test=True))  # (B,T,D)
                vfeats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu())
        V = torch.cat(vfeats)                           # (Nv,T,D)
        if V.ndim != 3:
            raise SystemExit(f"[FAIL] 期望逐帧 (Nv,T,D),拿到 {tuple(V.shape)} "
                             "⇒ return_frames 没生效")
        per_frame = torch.einsum("vtd,nd->vtn", V, Tx)

        for m in ("mean", "max", "lse", "top2", "single_frame"):
            r = _r_metrics(aggregate(per_frame, m, args.tau))
            rows.append((T, m, r))
            print(f"T={T:<3} {m:<13} r_mean={r['r_mean']:.2f}  "
                  f"t2v_r1={r['t2v_r1']:.2f}  v2t_r1={r['v2t_r1']:.2f}", flush=True)

    md = ["# C-2 聚合扫描 + C-4 方差-vs-覆盖判别", "",
          "| T | 聚合 | r_mean | t2v R@1 | v2t R@1 |", "|---|---|---|---|---|"]
    for T, m, r in rows:
        md.append(f"| {T} | {m} | {r['r_mean']:.2f} | {r['t2v_r1']:.2f} | {r['v2t_r1']:.2f} |")
    T0 = rows[0][0]
    best = {m: r["r_mean"] for t, m, r in rows if t == T0}
    if "max" in best and "mean" in best:
        d = best["max"] - best["mean"]
        md += ["", f"**oracle(max) − mean = {d:+.2f} r_mean**(T={T0})", "",
               ("⇒ **覆盖**说成立:某些帧才有答案,处方是 late-interaction / max 类聚合器。"
                if d > 1.0 else
                "⇒ **方差缩减**说成立:oracle 帧没比 mean 好多少,一阶池化没错;"
                "P9 那类二阶统计要另找立论。")]
    open(args.out, "w").write("\n".join(md) + "\n")
    print(f"\n[written] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
