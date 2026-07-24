"""r3e 试点 — C2 弹性 T 的链路冒烟(V100/10k;**非论文数字**)。

只验证 README §2b 矩阵「C2 T 无关化」行的机械判据:
- num_frames=16 解码母序列 + 每 step 随机 T∈{2,4,8,16} 子采样(utils/elastic_t.py,
  跨 rank 由 (seed, global_step) 决定性同步)后,前后向 / 损失 / ckpt 全链路跑通;
- 不看精度、不排序(10k 禁令);batch_size 因 T=16 峰值显存约 2× 而减半——
  链路冒烟无对照要求,真 A/B 的等步纪律见 SETUP_B100.md §6。

先跑单测,再冒烟:
    python -m pytest tests/test_t_agnostic.py -v
    bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r3e.py 8 debug True
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_zh_10k_pilot"
test_file = dict(ret_test=available_corpus["internvid_zh_10k_pilot_val"])  # noqa: F405
test_types = ["ret_test"]
stop_key = None

num_frames = 16       # 弹性子采样的解码母序列(评测仍按 num_frames_test=8)
batch_size = 32       # T=16 峰值显存 ~2×(V100-32G 红线,README §2b)
scheduler["epochs"] = 10  # noqa: F405
save_iter = 0

# C2 弹性 T 开关(tasks_clip/pretrain.py;默认关,此处显式开)
elastic_t = dict(enable=True, choices=[2, 4, 8, 16])

output_dir = "experiments/runs/pilot_10k_r3e"
