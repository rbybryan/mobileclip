"""P-surprise 试点 — 10k 上给「脑启发·惊奇池化」做链路冒烟(**非论文数字**)。

相对 pilot_10k_r0 的唯一变化:
    model.vision_encoder.temporal_pooling = "surprise"
即把 8 帧简单平均换成预测编码式的「惊奇加权池化」(neural_inspired.TemporalSurprisePool):
帧均值当预测,按每帧预测误差能量加权;`alpha` 零初始化 ⇒ step-0 与 mean 逐位等价。
其余全部与 R0 一致(softmax VTC、fixed T=8),以**隔离池化这一处改动**。

这是可选的脑启发 add-on(见 neural_inspired/README.md),最终若增益微小可整包移除。

V100/10k 只看动力学(判据,不看绝对精度):
- 损失正常下降、不发散;
- 训完从 ckpt 读 `vision_encoder.temporal_pool` 的参数:`alpha` 必须离开 0
  (`gamma`/`key` 也应离开 init),否则惊奇池化没在参与计算 —— 说明还是退化成 mean。
「更好与否」的排序是 H100 全量的事(淘汰标准:等步数 ≥ attn 池化),10k 不下结论。
"""
from scripts.pretraining.clip.mobileviclip_small.config import *  # noqa: F401,F403

train_corpus = "internvid_zh_10k_pilot"
test_file = dict(ret_test=available_corpus["internvid_zh_10k_pilot_val"])  # noqa: F405
test_types = ["ret_test"]
stop_key = None

model["vision_encoder"]["temporal_pooling"] = "surprise"  # noqa: F405

scheduler["epochs"] = 10  # noqa: F405
save_iter = 0  # 步数太少,只留 epoch 末 ckpt

output_dir = "experiments/runs/pilot_10k_surprise"
