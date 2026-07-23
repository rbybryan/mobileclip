# 10k 试点结果 · go/no-go 记录（ROADMAP 1.3）

V100 服务器（`rby@101vector`，8×V100）上 `pilot_10k_r0` / `pilot_10k_r1` 两条
链路的**正跑**结果与判据核对。运行日期 2026-07-23。

> 判据是**只看动力学**（ROADMAP 1.3）：损失正常下降、r1 的
> `logit_scale/logit_bias` 离开初始值、`temporal_pool.proj` 非零。
> **不看** held-out 检索分数——10k 的评测在训练前就近饱和，无法给方案排序
> （见 README §2b 与顶部两级实验说明）。

## 配置对照

| run | temporal pooling | 损失 | `different_lr` | 备注 |
|---|---|---|---|---|
| `pilot_10k_r0` | mean（无参数） | softmax VTC | 关 | 链路基线 |
| `pilot_10k_r1` | **attn**（`temporal_pool`，新初始化） | **sigmoid / SigLIP** | 开：`logit_scale`,`logit_bias` @1e-3 | 一次性同时打开两处改动的**冒烟**，非 A/B |

两处改动（池化 + 损失）在 r1 里是**捆绑**的，因此 r1 不能用来判断"哪个更好"；
逐项拆开的 A/B 在 H100 用 `r1a_sigmoid` / `r1b_attnpool` / `r1_both` 单独跑。

## 动力学（判据一：损失正常下降 ✅）

两条都单调下降，全程无 NaN / 无跳变（fp16 动态损失缩放下 anomaly detection 已关，见 8690b5d）：

| | ep0 | ep9 | 走向 |
|---|---|---|---|
| r0 avg `loss_vtc`（softmax 口径） | 1.01 | **0.27** | 单调↓ |
| r1 avg `loss_vtc`（sigmoid 口径，数值不可与 r0 比） | 7.07 | **4.18** | 单调↓，ep9 仍在降 |

10 epoch × 17 batch ≈ **170 步**（9000/512），r1 的新组件几乎没练开。

## held-out 检索（仅供体检，不作判据）

`ret_test` r_mean（DSL 口径）：

| ep | r0 | r1 |
|---|---|---|
| 0 | 92.03 | 89.95 |
| 5 | 93.27 | 86.67 |
| 9 | **93.20** | **86.97** |

**r1 低于 r0、且低于自己的 ep0——这是预期信号，不是回退：**
1. `temporal_pool`（attn）新初始化，起点比无参数 mean pooling 差 → ep0 就低于 r0；
2. sigmoid + 新 `logit_scale/logit_bias` 重标定相似度尺度，短暂把近饱和的 CLIP 初始表征拉偏；
3. 只有 170 步，损失还在降，新组件没恢复。
img 侧（≈88.6）比 txt 侧（≈85.3）掉得少，符合"重标定 + 新池化"的特征。
按两级设计，10k **无法**据此给 r0/r1 排序（评测饱和 + BN running-stats 混淆 + ~1.3pp 二项 SE）。

## 判据二 / 三：待服务器一条命令核实 ⏳

r0 是纯 mean-pool + softmax，没有 sigmoid 参数、没有 `temporal_pool.proj`，判据二三不适用——**r0 直接通过**。
r1 判据一已过，判据二三需要读 ckpt。**Claude 端无 GPU、无 ckpt，只能由服务器跑**。
init 值：`logit_scale = log(10)`（即 `t=exp(logit_scale)=10`）、`logit_bias = -10`
（`models/mobileviclip_small.py:102-103`）。在服务器上：

```bash
# 仓库根目录，conda activate mobileviclip
python - <<'PY'
import torch
sd = torch.load("experiments/runs/pilot_10k_r1/ckpt_09.pth", map_location="cpu")
sd = sd.get("model", sd.get("module", sd))   # 兼容 {'model':...} / {'module':...} / 裸 state_dict
ls, lb = sd.get("logit_scale"), sd.get("logit_bias")
w = sd.get("vision_encoder.temporal_pool.proj.weight")
print("exp(logit_scale) =", None if ls is None else round(ls.float().exp().item(), 4), "  (init 10 → 判据: ≠10)")
print("logit_bias       =", None if lb is None else round(lb.float().item(), 4),        "  (init -10 → 判据: ≠-10)")
print("temporal_pool.proj.weight  L2 =", None if w is None else round(w.float().norm().item(), 4),
      " mean|w| =", None if w is None else round(w.float().abs().mean().item(), 6), "  (判据: 非零)")
PY
```

三行都满足（`exp(logit_scale)≠10`、`logit_bias≠-10`、proj 范数非零）→ **1.3 全过**。
若 `logit_scale/logit_bias` 仍卡在 10 / -10，说明 `different_lr` 没接上（检查
`utils/optimizer.py` 的 `module_names` 匹配日志：本次日志已显示
`param logit_scale use different_lr: 0.001`，故大概率已在动）。

同时按 ROADMAP 1.3 补跑 `pytest tests/test_optimization_components.py`。

## 给 H100 阶段的设计提示（不改 V100 判据）

r1 的 ep0 回退暴露了一个**可提前消除的**问题，H100 正式跑前值得处理：

- **attn 池化的初始化**：让 `temporal_pool` 在 init 时≈mean pooling（例如输出
  投影 zero-init + 残差，或 v≈identity/proj≈0，使输出退化为帧均值），避免一开始
  丢掉强初始表征、给 A/B 一个干净起点。
- **`logit_scale/logit_bias` 的 warmup**：这两个 @1e-3 的高 lr 参数在头几步会剧烈
  重标定；适当 warmup 可减小对 CLIP 初始对齐的冲击。
- H100 的 r1a/r1b **必须分开跑**（配置已就绪），别再把两处改动捆在一起，否则
  仍是混淆比较。

相关：`experiments/README.md` §2b（试点 runbook）、§4（R0→R1 全量 A/B）、
`experiments/ROADMAP.md` 1.3 / 3.2、`experiments/SETUP_B100.md`（H100 runbook）。
