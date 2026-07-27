#!/usr/bin/env python3
"""R1 Part B 前置:把 InternVid 字幕用**冻结文本塔**编码成投影后归一化特征,存 .npy。

配合 `experiments/r1_fn_density.py --feats` 量化**近重复密度**(idx 掩码抓不到、
但对比损失会当假负例的那部分)。见 `notes/r1-examination.md` §5/§7。

**为什么这样构建模型(不重造轮子):** 直接 `Config.from_file(<eval 配置>)` +
`MobileViCLIP_Small(config, is_pretrain=False)` —— 这正是 GATE-0 评测在服务器上跑通的同一条
构建路径,模型**自己的** `load_checkpoint` 负责底座权重的键重映射(不需要我猜 mobileclip_s2.pt
的内部键名)。再用 `strict=False` 覆盖一份 `mobileviclip_small.pt`,取到**训练后的**文本投影
(LiT:文本塔冻结,只有 projection_layer 训练并存进 small.pt)。跳过 setup_model 里
deepspeed/DDP/optimizer/auto_resume 那些推理用不到的部分。

⚠️ 本脚本的**模型加载/前向部分没有在真权重上跑过**(Claude 侧无权重/训练环境)。字幕读取、
去重、分块、保存这些非模型逻辑已本地自测。若模型构建报错,请把**完整报错**贴回;
最可能的点是 `Config.from_file` 的字段或 `encode_text` 的返回形状,按报错微调即可。

用法(训练环境,单卡足够;纯前向、无需 decord/视频):
    python experiments/r1_textfeat_dump.py \
        --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
        --ckpt checkpoints/mobileviclip_small.pt \
        --out /tmp/txtfeats.npy --sample 10000
    # 然后:
    python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy --thresholds 0.9,0.95,0.99

自检(不建模型、不需数据):
    python experiments/r1_textfeat_dump.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 复用 r1_fn_density 里已自测的字幕组装 / caption_to_idx(单一来源,避免漂移)。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "r1fd", os.path.join(os.path.dirname(os.path.abspath(__file__)), "r1_fn_density.py"))
_r1fd = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_r1fd)
assemble_caption = _r1fd.assemble_caption
caption_to_idx = _r1fd.caption_to_idx

DEFAULT_CONFIG = "scripts/evaluation/clip/zero_shot/mobileviclip_small/config_msrvtt.py"


def load_distinct_captions(anno_paths, caption_key, sample, seed=0):
    """读所有 anno,组装字幕,按 caption_to_idx 去精确重复(Part B 只看不同字幕)。"""
    seen, caps = set(), []
    used_key = caption_key
    for path in anno_paths:
        if not os.path.exists(path):
            print(f"[跳过] 不存在:{path}"); continue
        data = json.load(open(path, encoding="utf-8"))
        for r in (data if isinstance(data, list) else [data]):
            if not isinstance(r, dict):
                continue
            c, used_key = assemble_caption(r, caption_key)
            if not c:
                continue
            k = caption_to_idx(c)
            if k not in seen:
                seen.add(k); caps.append(c)
    if sample and sample < len(caps):
        import random
        random.seed(seed); caps = random.sample(caps, sample)
    return caps, used_key


def build_text_model(config_path, ckpt_path):
    """构建 MobileViCLIP_Small 并覆盖训练后权重。返回 (model, tokenizer, device)。"""
    import torch
    from utils.config import Config
    from models.mobileviclip_small import MobileViCLIP_Small

    config = Config.from_file(config_path)
    # 只做推理:关掉分布式 / deepspeed / 半精度,免得触发那些路径。
    config.distributed = False
    if hasattr(config, "deepspeed"):
        config.deepspeed.enable = False
    config.use_half_precision = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    # 模型自己的 load_checkpoint 会从 config.model.*_ckpt_path(mobileclip_s2.pt)
    # 载入冻结底座并做键重映射。
    model = MobileViCLIP_Small(config=config, is_pretrain=False)

    # 覆盖训练后的权重(视觉 + 训练过的文本 projection_layer)。strict=False:
    # 冻结文本塔的键在 small.pt 里缺失是预期的(它们来自底座),忽略即可。
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("model", sd.get("module", sd)) if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        tp = [k for k in missing if "text_encoder.projection_layer" in k]
        if tp:
            print("⚠️ 注意:small.pt 里没有 text_encoder.projection_layer ——"
                  "用的是底座的初始投影(近重复结构受 512→512 线性映射影响很小,估计仍近似有效)。")
    else:
        print(f"⚠️ 未找到 {ckpt_path};用底座权重(投影未训练)。")

    model = model.to(device).eval()
    return model, model.tokenizer, device


def dump_feats(model, tokenizer, device, caps, out, max_txt_l=32, bs=256):
    import numpy as np, torch
    feats = []
    with torch.no_grad():
        for i in range(0, len(caps), bs):
            chunk = caps[i:i + bs]
            # ClipTokenizer.forward(sentences, context_length) -> [b, ctx] 索引;
            # eval 路径(retrieval_utils.py:115)用的就是 model.encode_text(tokenizer(text)).
            ids = tokenizer(chunk).to(device)
            f = model.encode_text(ids)
            if isinstance(f, (tuple, list)):
                f = f[0]
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
            if i % (bs * 20) == 0:
                print(f"  {i+len(chunk)}/{len(caps)}")
    arr = np.concatenate(feats)
    np.save(out, arr)
    print(f"已存 {arr.shape} → {out}")
    print(f"下一步:python experiments/r1_fn_density.py --feats {out} --thresholds 0.9,0.95,0.99")


def _self_test():
    # 只测非模型逻辑:去重 + 组装。
    import tempfile
    recs = ([{"caption": ["a", "man"]}] * 3 + [{"caption": "a man"}]      # 同字幕不同表示?
            + [{"caption": f"u{i}"} for i in range(5)])
    p = os.path.join(tempfile.gettempdir(), "_r1dump_selftest.json")
    json.dump(recs, open(p, "w"))
    caps, key = load_distinct_captions([p], "caption", sample=0)
    # "a man"(str) 与 ["a","man"]→"a man" 组装后同串 ⇒ 同 idx ⇒ 去重后应为 1 + 5 = 6
    assert key == "caption"
    assert len(caps) == 6, (len(caps), caps)
    assert len(set(caption_to_idx(c) for c in caps)) == 6
    print("self-test 通过(字幕组装 + 精确去重;模型部分需在训练环境验证)。")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno", nargs="*", default=[])
    ap.add_argument("--ckpt", default="checkpoints/mobileviclip_small.pt")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="仅用于构建模型(复用 eval 配置的构建路径)")
    ap.add_argument("--caption-key", default="caption")
    ap.add_argument("--sample", type=int, default=10000)
    ap.add_argument("--max-txt-l", type=int, default=32)
    ap.add_argument("--out", default="/tmp/txtfeats.npy")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test(); return
    if not args.anno:
        ap.error("需要 --anno(一个或多个 InternVid anno JSON);或 --self-test")

    caps, key = load_distinct_captions(args.anno, args.caption_key, args.sample)
    if not caps:
        print("没读到字幕 —— 检查 --anno / --caption-key。"); return
    print(f"不同字幕 {len(caps)} 条(字段 {key!r});开始用冻结文本塔编码 …")
    try:
        model, tok, device = build_text_model(args.config, args.ckpt)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("\n模型构建失败 —— 把上面完整报错贴回。字幕已读好,只差建模型这步。")
        return
    dump_feats(model, tok, device, caps, args.out, max_txt_l=args.max_txt_l)


if __name__ == "__main__":
    sys.exit(main())
