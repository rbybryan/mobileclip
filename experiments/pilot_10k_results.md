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
1. **主因是 sigmoid 损失切换**:新 `logit_scale/logit_bias` 重标定相似度尺度,短暂把近饱和的
   CLIP 初始表征拉偏(实测头 10 步就把 held-out 从 90 拉到 85)。**不是池化的锅** ——
   `temporal_pool` 零初始化 + 残差(`mean + proj(out)`),**init 时严格等于 r0 的 mean-pool**,
   只随训练慢慢偏离(见判据三:proj 从 0 练到 L2=0.18)。
2. 只有 170 步,损失还在降,新组件没恢复。
img 侧（≈88.6）比 txt 侧（≈85.3）掉得少,符合"尺度重标定"的特征。
按两级设计,10k **无法**据此给 r0/r1 排序（评测饱和 + BN running-stats 混淆 + ~1.3pp 二项 SE）。
这也细化了 H100 分工:`r1b`(attn 池化)从 mean 起步、只增不减,**安全**;`r1a`(sigmoid)才是
扰动初始对齐的那一个,重点观察。

## 判据二 / 三：已核实通过 ✅（2026-07-23）

r0 是纯 mean-pool + softmax，没有 sigmoid 参数、没有 `temporal_pool.proj`，判据二三不适用——**r0 直接通过**。
r1 判据一已过，判据二三需要读 ckpt。**Claude 端无 GPU、无 ckpt，只能由服务器跑**。
init 值：`logit_scale = log(10)`（即 `t=exp(logit_scale)=10`）、`logit_bias = -10`
（`models/mobileviclip_small.py:102-103`）。

**注意 ckpt 是 DeepSpeed 目录不是单文件**：ZeRO stage-1 把 `ckpt_09.pth/` 存成**目录**,
权重在 `ckpt_09.pth/mp_rank_00_model_states.pt` 的 `module` 键下(各 rank 的
`zero_pp_rank_*_optim_states.pt` 只是优化器分片,不含可读权重)。直接
`torch.load("…/ckpt_09.pth")` 会报 `IsADirectoryError`。在服务器上:

```bash
# 仓库根目录，conda activate mobileviclip
python - <<'PY'
import torch
f = "experiments/runs/pilot_10k_r1/ckpt_09.pth/mp_rank_00_model_states.pt"  # DeepSpeed 权重
obj = torch.load(f, map_location="cpu")
sd = obj.get("module", obj)          # 模型 state_dict 在 'module' 键下
ls, lb = sd.get("logit_scale"), sd.get("logit_bias")
w  = sd.get("vision_encoder.temporal_pool.proj.weight")
print("present:", ("logit_scale" in sd), ("logit_bias" in sd),
      ("vision_encoder.temporal_pool.proj.weight" in sd))
print("exp(logit_scale) =", None if ls is None else round(ls.float().exp().item(),4), " (init 10 → 判据 ≠10)")
print("logit_bias       =", None if lb is None else round(lb.float().item(),4),       " (init -10 → 判据 ≠-10)")
print("temporal_pool.proj.weight L2 =", None if w is None else round(w.float().norm().item(),4),
      " mean|w| =", None if w is None else round(w.float().abs().mean().item(),6), " (判据 非零)")
PY
```

若 `module` 键取不到,用目录里的 `zero_to_fp32.py` 先合并出 fp32 全量再读:
`python experiments/runs/pilot_10k_r1/ckpt_09.pth/zero_to_fp32.py experiments/runs/pilot_10k_r1/ckpt_09.pth /tmp/r1_fp32.pt`,
再把上面的 `f` 换成 `/tmp/r1_fp32.pt`(此时 `sd = obj`,无 `module` 包裹)。

**实测(2026-07-23,`ckpt_09/mp_rank_00_model_states.pt`,`present: True True True`)——三条全过 ✅**:

| 参数 | init | ckpt_09 实测 | 判定 |
|---|---|---|---|
| `exp(logit_scale)` | 10 | **10.4815** | ≠10 ✅ |
| `logit_bias` | −10 | **−9.7578** | ≠−10 ✅ |
| `temporal_pool.proj.weight` L2 | **0**(见下) | **0.1775**(mean\|w\|=2.9e-4) | 非零 ✅ |

`logit_scale`/`logit_bias` 双双离开 init → `different_lr @1e-3` 的接线**确实在更新它们**,
这正是判据二三要证的关键。

**更正一处早先说法**:`temporal_pool.proj` **不是**"init 就非零"。读代码
(`models/mobileviclip_small.py:1350-1365`)它是**零初始化 + 残差**——
`nn.init.zeros_(self.proj.weight/bias)`、前向 `return mean + self.proj(out)`,即
**init 时 `proj≡0`、池化输出严格等于帧均值(退化为 mean-pool)**。所以实测 L2=0.1775
是它**从零练起来**的证据,判据三比"随机 init 本就非零"强得多。

`pytest tests/test_optimization_components.py` 亦按 ROADMAP 1.3 补跑(链路层面已通过)。

## 给 H100 阶段的设计提示（不改 V100 判据）

- ~~**attn 池化的初始化**~~ —— **已无需处理**(先前建议作废):读代码确认
  `temporal_pool` 本就零初始化输出投影 + 残差(`mean + proj(out)`,
  `models/mobileviclip_small.py:1350-1365`),init 即退化为 mean-pool,起点已经干净。
- **`logit_scale/logit_bias` 的 warmup**:这两个 @1e-3 的高 lr 参数在头几步剧烈重标定
  (实测 10 步内把 r1 的 held-out 从 90 拉到 85),是 r1 ep0 回退的**主因**;H100 上适当
  warmup / 降这两个参数的 lr 可减小对 CLIP 初始对齐的冲击。这是 sigmoid 臂(`r1a`)的事。
- H100 的 r1a/r1b **必须分开跑**（配置已就绪），别再把两处改动捆在一起,否则仍是混淆比较。
  已知:`r1b`(attn 池化)从 mean 起步、安全;`r1a`(sigmoid)是扰动源,是重点观察对象。

相关：`experiments/README.md` §2b（试点 runbook）、§4（R0→R1 全量 A/B）、
`experiments/ROADMAP.md` 1.3 / 3.2、`experiments/SETUP_B100.md`（H100 runbook）。
