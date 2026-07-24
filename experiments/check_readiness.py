#!/usr/bin/env python3
"""检查数据集/权重就绪状态:哪些测试集已到位、哪些实验步骤已解锁。

只用标准库,读 configs/data.py 的 corpus 定义(MVC_DATA_DIR / MVC_ANNO_DIR
环境变量生效),逐个检查:
  - anno 文件存在?能解析?多少条记录?
  - data_root 目录存在?抽查前 N 条记录的视频文件是否真的在盘上?
再检查 checkpoints/ 下的权重,最后汇总"当前已解锁的 runbook 步骤"。

用法(在仓库根目录、设好环境变量后):
    python experiments/check_readiness.py            # 全量检查
    python experiments/check_readiness.py --sample 50 # 每个数据集抽查 50 条
    python experiments/check_readiness.py --decode    # 额外用 decord 试解码 1 条(需训练环境)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.data import available_corpus  # noqa: E402

# 各评测基准对应的 corpus 名(与 run_eval.sh 的 benchmark 名对齐)
BENCHMARKS = {
    "msrvtt": "msrvtt_1k_test",
    "didemo": "didemo_ret_test",
    "anet": "anet_ret_val",
    "k400": "k400_act_val",
    "ucf101": "ucf101_act_test",
    "hmdb51": "hmdb51_act_test",
    "ssv2_mc": "ssv2_mc_val",
}
CKPTS = {
    "mobileclip_s2.pt": "训练底座(R0–R3 预训练必需)",
    "mobileviclip_small.pt": "官方权重(B0 锁基线必需)",
    "mobileclip_s0.pt": "Tiny 底座(可选)",
}


def load_anno(path):
    """return (n_records, records or None). 支持 json 数组与 jsonl。"""
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[":
                recs = json.load(f)
                return len(recs), recs
            recs = [json.loads(l) for l in f if l.strip()]
            return len(recs), recs
    except Exception as e:
        return -1, str(e)


def media_path(rec, data_root):
    for k in ("video", "image", "audio"):
        if k in rec:
            return os.path.join(data_root, rec[k])
    return None


def check_corpus(name, entry, sample_n, decode):
    """return dict(name, ready, detail)"""
    anno, root = entry["anno_path"], entry["data_root"]
    if not os.path.isfile(anno):
        return dict(name=name, ready=False, detail=f"anno 缺失: {anno}")
    n, recs = load_anno(anno)
    if n < 0:
        return dict(name=name, ready=False, detail=f"anno 解析失败: {recs}")
    if not os.path.isdir(root):
        return dict(name=name, ready=False, detail=f"anno {n} 条,但 data_root 缺失: {root}")
    checked = missing = 0
    first_ok = first_missing = None
    for rec in recs[:sample_n]:
        p = media_path(rec, root)
        if p is None:
            continue
        checked += 1
        if os.path.isfile(p):
            first_ok = first_ok or p
        else:
            missing += 1
            first_missing = first_missing or p
    if checked == 0:
        return dict(name=name, ready=False, detail=f"anno {n} 条,但记录里没有 video/image 字段")
    if missing == checked:
        # 一定要把**试过的完整路径**打出来:全缺最常见的死因是 anno 的 video 字段带
        # 子目录前缀(如 "TestVideo/xxx.mp4"),而视频被拍平软链到了 data_root 根下,
        # 于是 join() 出来的路径多了一层。只说"检查目录层级"看不出是哪一层错了。
        return dict(name=name, ready=False,
                    detail=f"anno {n} 条,抽查 {checked} 条视频全部缺失;"
                           f"试过的路径例如 {first_missing} —— 把它和盘上实际位置比一下,"
                           "差在哪一层就补/去哪一层(anno 的 video 字段是相对 data_root 的)")
    detail = f"anno {n} 条;抽查 {checked} 条,缺失 {missing}"
    if decode and first_ok:
        try:
            import decord
            vr = decord.VideoReader(first_ok, num_threads=1)
            detail += f";试解码 OK({len(vr)} 帧)"
        except Exception as e:
            return dict(name=name, ready=False, detail=detail + f";试解码失败: {e}")
    return dict(name=name, ready=(missing == 0), detail=detail)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=20, help="每个数据集抽查的记录数")
    ap.add_argument("--decode", action="store_true", help="用 decord 试解码 1 条视频")
    args = ap.parse_args()

    print(f"MVC_DATA_DIR={os.environ.get('MVC_DATA_DIR', '(默认 data)')}  "
          f"MVC_ANNO_DIR={os.environ.get('MVC_ANNO_DIR', '(默认 anno_downstream)')}\n")

    results = {}
    print("== 评测基准 ==")
    for bench, corpus in BENCHMARKS.items():
        r = check_corpus(corpus, available_corpus[corpus], args.sample, args.decode)
        results[bench] = r["ready"]
        print(f"  [{'OK ' if r['ready'] else '·· '}] {bench:8s} {r['detail']}")

    print("\n== 训练语料 ==")
    train_ready = {}
    for name in [f"internvid_v{i}" for i in range(1, 12)] + \
                ["internvid_zh_10k_pilot", "internvid_zh_10k_pilot_val"]:
        r = check_corpus(name, available_corpus[name], args.sample, args.decode)
        train_ready[name] = r["ready"]
        print(f"  [{'OK ' if r['ready'] else '·· '}] {name:28s} {r['detail']}")

    print("\n== 权重 checkpoints/ ==")
    ckpt_ready = {}
    for f, why in CKPTS.items():
        p = os.path.join("checkpoints", f)
        ok = os.path.isfile(p)
        ckpt_ready[f] = ok
        size = f"{os.path.getsize(p) / 1e6:.0f}MB" if ok else "缺失"
        print(f"  [{'OK ' if ok else '·· '}] {f:24s} {size:8s} {why}")

    full = all(train_ready[f"internvid_v{i}"] for i in range(1, 12))
    pilot = train_ready["internvid_zh_10k_pilot"] and train_ready["internvid_zh_10k_pilot_val"]
    # 这份结论只对**当前这个 MVC_DATA_DIR** 成立。该机上评测数据在仓库 data/、
    # 试点语料在 /home/vca1/datasets,不在一个根下,所以单跑一次必然有一半误报。
    root_desc = os.environ.get("MVC_DATA_DIR", "data(默认)")
    print(f"\n== 已解锁的步骤(experiments/ROADMAP.md)== [仅基于 MVC_DATA_DIR={root_desc}]")
    steps = [
        ("10k 试点(§2b:pilot_10k_r0/r1)", pilot and ckpt_ready["mobileclip_s2.pt"]),
        ("B0 锁基线(§3,至少 MSR-VTT)", results["msrvtt"] and ckpt_ready["mobileviclip_small.pt"]),
        ("B0 全量基准", all(results.values()) and ckpt_ready["mobileviclip_small.pt"]),
        ("R0–R3 全量预训练(§4)", full and ckpt_ready["mobileclip_s2.pt"]),
    ]
    for desc, ok in steps:
        print(f"  [{'可跑' if ok else '待条件'}] {desc}")

    # 提示要对准真实死因:anno 缺失和 data_root 缺失的修法完全不同,
    # 之前无论哪种都劝人去重跑 make_pilot_anno.py,会让人白重建一份好好的 anno。
    if not pilot:
        pilot_anno = available_corpus["internvid_zh_10k_pilot"]["anno_path"]
        if not os.path.isfile(pilot_anno):
            print(f"\n提示:pilot anno 不存在({pilot_anno}),"
                  "先跑 tools/make_pilot_anno.py(README §2b)。")
        else:
            print(f"\n提示:pilot anno 是好的,缺的是**视频目录**。"
                  f"\n      该机试点语料在 /home/vca1/datasets,评测数据在仓库 data/ —— "
                  "两批不在一个根下。"
                  f"\n      换个根再跑一遍:"
                  "MVC_DATA_DIR=/home/vca1/datasets python experiments/check_readiness.py")


if __name__ == "__main__":
    main()
