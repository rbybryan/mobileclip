#!/usr/bin/env python3
"""BN 重校准 CLI —— 阶段 3/5 分辨率数从「趋势筛」升级为「正式数」的前提。

    python experiments/bn_recalibrate.py \
        --config scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py \
        --ckpt ./checkpoints/mobileviclip_small.pt \
        --res 224 --t 8 --out experiments/bn_ckpts/recal_res224_T8.pt

产物是一个**打过 BN 补丁的完整 ckpt**(`{"model": state_dict}`),直接喂给现有评测
(`run_eval.sh` / `run_v100_all.sh` 的 `pretrained_path`)—— 评测链路零改动。

为什么必须有它(纪律出处,别绕开):
  · 换分辨率/删块会移动**下游每一个 BN** 的输入分布;不重校准,量到的是分布漂移,
    不是改动本身的价值(`experiments/dropblock_results.md` §4;v100 清单 §3 强制纪律 1)。
  · 🔴 **梯度检查点会让 BN 双更新**(round-5 §11.3 / A12 已两轮实证:
    `num_batches_tracked` 一步变 2)⇒ 本脚本强制 `gradient_checkpointing=False`,
    并在校准后断言 `num_batches_tracked == 实际 batch 数`,不对就拒绝落盘。
  · 校准语料默认 **10k pilot val**(训练域,零测试集泄漏)。把 `--calib-corpus`
    换成评测基准自身(如 msrvtt_1k_test)= 在测试分布上校准,论文里必须单独声明,
    默认不要这么做。

工具底座:`experiments/per_t_bn.py`(纯函数,tests/test_t_agnostic.py 有单测);
本脚本只做接线:配置覆盖 → 建模型/载权重 → 校准前向 → 漂移报告 → 落盘。
"""
import argparse
import json
import os
import sys

import torch


def set_dotted(cfg, key: str, value):
    """按点分路径写配置(dict-like 优先,退而 setattr)。"""
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


def parse_value(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def calib_batches(loader, n_target: int):
    """从 loader 循环取满 n_target 个 batch(语料比 batch 数少时绕圈)。"""
    n = 0
    while n < n_target:
        got_any = False
        for image, _ in loader:
            got_any = True
            yield image
            n += 1
            if n >= n_target:
                return
        if not got_any:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="评测 config(定模型形状与预处理)")
    ap.add_argument("--ckpt", required=True, help="要校准的权重")
    ap.add_argument("--res", type=int, required=True, help="目标分辨率(两处键都会改)")
    ap.add_argument("--t", type=int, default=8, help="目标帧数 T")
    ap.add_argument("--calib-corpus", default="internvid_zh_10k_pilot_val",
                    help="configs/data.py available_corpus 的键;默认 10k pilot val(训练域)")
    ap.add_argument("--batches", type=int, default=100,
                    help="校准前向的 batch 数(momentum=None 累积均值,几百即稳)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True,
                    help="打过 BN 补丁的 ckpt 输出路径(喂给评测的 pretrained_path)")
    ap.add_argument("--save-bank", default=None,
                    help="可选:把 {T: BN snapshot} 存成 per-T bank(per_t_bn.save_bn_bank)")
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                    help="额外配置覆盖(如 model.vision_encoder.disable_temporal_branch true)")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from utils.config import Config, eval_dict_leaf
    from dataset import create_dataset, create_loader
    from models.mobileviclip_small import MobileViCLIP_Small, set_num_frames
    from experiments.per_t_bn import (bn_modules, snapshot_bn_stats, reset_bn_stats,
                                      recalibrate_bn, bn_drift_report, save_bn_bank)
    from configs.data import available_corpus

    # 🔴 插值陷阱(TROUBLESHOOTING 末节):Config.from_file 不做 ${...} 插值,必须自己
    #    eval_dict_leaf;插值把 num_frames_test 固化,所以下面两处都要显式覆盖。
    cfg = eval_dict_leaf(Config.from_file(args.config))
    cfg.inputs.image_res = args.res                       # 分辨率两键都改,
    cfg.model.vision_encoder.img_size = args.res          # 只改一个会静默不一致
    cfg.num_frames_test = args.t
    cfg.inputs.video_input.num_frames_test = args.t
    cfg.distributed = False
    # 🔴 A12/round-5 §11.3:检查点会让 BN 双更新 ⇒ 校准一律关掉
    cfg.gradient_checkpointing = False
    if args.calib_corpus not in available_corpus:
        raise SystemExit(f"[FAIL] 未知校准语料 {args.calib_corpus};"
                         f"可选:{sorted(k for k in available_corpus if 'test' in k or 'val' in k)}")
    cfg.test_file = dict(ret_test=available_corpus[args.calib_corpus])
    cfg.test_types = ["ret_test"]
    cfg.inputs.batch_size_test = dict(video=args.batch_size, image=args.batch_size)
    for k, v in args.set:
        set_dotted(cfg, k, parse_value(v))
        print(f"[set] {k} = {parse_value(v)}")

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
    # 结构 flag(--set disable_temporal_* / skip_temporal_branch)会被 load 冲掉 ⇒ 载后重放
    if hasattr(model, "apply_arch_flags"):
        model.apply_arch_flags()
    model = model.to(device).eval()
    set_num_frames(args.t)

    # 只建评测集(setup_dataloaders 会连训练集一起建,eval config 的 train_file 不是给它用的)
    test_datasets, _names = create_dataset("ret_eval", cfg)
    loader = create_loader(
        test_datasets, [None] * len(test_datasets),
        batch_size=[cfg.inputs.batch_size_test[d.media_type] for d in test_datasets],
        num_workers=[cfg.num_workers] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[None] * len(test_datasets),
    )[0]

    # 🔴 只动视觉塔的 BN:reset 全模型会把(将来可能出现的)文本侧 BN 清成初始值
    #    却没人给它前向,那是静默腐蚀。快照/清零/校准都以 vision_encoder 为界。
    ve = model.vision_encoder
    n_bn = sum(1 for _ in bn_modules(ve))
    before = snapshot_bn_stats(ve)
    reset_bn_stats(ve)
    used = recalibrate_bn(
        ve, calib_batches(loader, args.batches),
        forward_fn=lambda _ve, batch: model.encode_vision(
            batch.to(device, non_blocking=True), test=True),
    )
    after = snapshot_bn_stats(ve)
    if used == 0:
        raise SystemExit("[FAIL] 校准语料一个 batch 都没产出 —— 检查 anno/视频根")

    # A12 判据:num_batches_tracked 必须 == 实际 batch 数(双更新会是 2×)
    for name, m in bn_modules(ve):
        nbt = int(m.num_batches_tracked)
        if nbt != used:
            raise SystemExit(f"[FAIL] {name}.num_batches_tracked={nbt} ≠ 前向数 {used}"
                             " ⇒ BN 双更新仍在发生(检查 gradient_checkpointing)—— 拒绝落盘")
        break

    rep = bn_drift_report(after, before)
    print(f"[calib] {n_bn} 个 BN · {used} 个 batch({args.calib_corpus},"
          f"res={args.res}, T={args.t})")
    print(f"[drift] 全局 mean_z={rep['global_']['mean_z']:.4f} "
          f"logvar_shift={rep['global_']['logvar_shift']:.4f}(相对校准前)")
    for name, z in rep["top"][:5]:
        print(f"    top: {name}  mean_z={z:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, args.out)
    meta = dict(src_ckpt=args.ckpt, config=args.config, res=args.res, t=args.t,
                calib_corpus=args.calib_corpus, batches=used,
                batch_size=args.batch_size, overrides=args.set,
                drift_global=rep["global_"],
                drift_top=[(n, round(z, 4)) for n, z in rep["top"]])
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[written] {args.out}(+ .json 侧车记录校准口径)")
    if args.save_bank:
        save_bn_bank(args.save_bank, {args.t: after})
        print(f"[written] {args.save_bank}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
