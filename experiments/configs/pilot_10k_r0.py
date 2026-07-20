"""P0 试点 — 10k 抽样上的链路冒烟 + 方向性基线(**非论文数字**)。

用途(见 experiments/README.md §2b):
- 全链路验证:decord 解码这批视频、损失、DeepSpeed、ckpt、epoch 末评测;
- 微型基线:与 pilot_10k_r1 等步数对照,只看训练动力学与相对方向。

与真 R0 的差异:训练集只有 ~9k 条(全量的千分之一)、评测是 held-out 1k
自检索(不是 MSR-VTT)。任何数字不外推、不进结论。
10 个 epoch ≈ 9k/512 × 10 ≈ 180 步,足够看到损失曲线形态。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_zh_10k_pilot"
test_file = dict(ret_test=available_corpus["internvid_zh_10k_pilot_val"])  # noqa: F405
test_types = ["ret_test"]
stop_key = None

scheduler["epochs"] = 10  # noqa: F405
save_iter = 0  # 步数太少,只留 epoch 末 ckpt

output_dir = "experiments/runs/pilot_10k_r0"
