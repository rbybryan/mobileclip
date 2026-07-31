"""10k 链路 go/no-go —— **P9 统计池化**(round-5 §8 P9)。

这一档的职责**只有三条**(ROADMAP 验证分三档 ①):跑得通、loss 正常降、
**新参数收到梯度**。**它排不了方案序**,别拿它的 retrieval 数去和 r1b 比。

🔺 **必须检查的那一条**:训完从 ckpt 读 `temporal_pool.w2` 的范数。
仍≈0 ⇒ 这一臂**生来就是死的**(退化成 mean),此时任何 retrieval 数都不可判读 ——
这正是 nA1 那条「alpha 仍 ≲0.01」的坑。给 temporal_pool 单独 lr 就是为了避开它。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_zh_10k_pilot"
test_file = dict(ret_test=available_corpus["internvid_zh_10k_pilot_val"])  # noqa: F405
test_types = ["ret_test"]
stop_key = None

seed = 42
model["vision_encoder"]["temporal_pooling"] = "stats"  # noqa: F405
optimizer["different_lr"] = dict(  # noqa: F405  同 nA1:不给单独 lr 会退化成 mean
    enable=True, module_names=["temporal_pool"], lr=1e-4
)

scheduler["epochs"] = 10  # noqa: F405
save_iter = 0

output_dir = "experiments/runs/pilot_10k_P9"
