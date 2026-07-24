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

---

# 脑启发 add-on 试点 · `TemporalSurprisePool`（2026-07-24）

`pilot_10k_surprise`(只把 r0 的 `temporal_pooling="mean"` 换成 `"surprise"`,
损失/lr/数据全同 r0:softmax VTC、`use_sigmoid_loss=False`、`different_lr` 关、全 1e-5)
在 8×V100 跑完冒烟(`debug True`,4 iter/epoch)+ 正跑(10 epoch × 17 batch)。

## 现象:动力学与 r0 **逐点重合**

| | ep0 avg `loss_vtc` | ep9 avg | ep9 `ret_test` r_mean(DSL) |
|---|---|---|---|
| r0(mean 池化) | 1.01 | 0.27 | 93.20 |
| surprise 池化 | 1.03 | 0.28 | 93.50 |

链路层面**通过**:8 卡 DeepSpeed 正常初始化、`temporal_pool.{alpha,gamma,key}` 出现在
优化器参数组(lr 1e-5)、损失单调下降、无 NaN、无跳变。检索 93.5 vs 93.2 在 ~1.3pp
二项 SE + BN running-stats 混淆之内,**不可据此排序**(两级实验:10k 只做链路冒烟)。

## 但这条重合是**报警**,不是好消息:模块「死」在 init

判据是 `alpha` 必须离开 0(见 `neural_inspired/README.md` 与 `pilot_10k_surprise.py`)。
动力学与 r0 逐点重合,恰恰是 **surprise 模块从没参与计算**的症状。用 autograd 在 init
处核实(`TemporalSurprisePool(dim).forward` + 反传,见提交里的回归测试):

| init | init 处 output==mean(不退化) | `alpha` 梯度 | `gamma` 梯度 | `key` 梯度 |
|---|---|---|---|---|
| **`gamma=0, alpha=0`(原实现)** | 是 | **0** | **0** | **0** |
| `gamma=1, alpha=0`(修复) | 是 | **2.0e-3** | 0(次步起 ≠0) | 0(次步起 ≠0) |

**根因**:`alpha` 自己的梯度是 `dL/d(out)·(pooled−mean)`。原实现 `gamma` 也从 0 起 ⇒
`softmax(0)` 均匀 ⇒ `pooled ≡ mean` ⇒ `(pooled−mean)=0` ⇒ **`alpha/gamma/key` 三者
梯度全为 0**。init 是个**零梯度不动点**,AdamW 对零梯度不更新(`alpha`、`gamma` 又都
`wd=0`),模块永远停在 mean 池化——所以它和 r0 逐位一样。预期 ckpt 读数:
`alpha=0.0`、`gamma=0.0`、`key≈init`(仅 `key` 有 `wd=0.2`,170 步权重衰减 ~0.03%)。

> 对比 `TemporalAttentionPool`(`mean + proj(out)`,`proj` 零初始化)为何没死:它零初始化的是
> **输出投影**,而被投影的 `out`(注意力输出)在 init 非零 ⇒ `dL/dproj = dL/d(out)·outᵀ ≠ 0`,
> 门能开(判据三实测 `proj` L2 从 0 练到 0.18)。surprise 是用**标量 alpha 乘 (pooled−mean)**,
> 而 `pooled−mean` 自己在 init 就是 0——双零门,连 alpha 自己的梯度都被掐断。

**为何必须现在修**:H100 的 A/B 臂 `nA1_surprisepool` 用的是同一个模块。不修的话它会
**静默地拿 mean-vs-mean 做对照**,然后得出「惊奇池化没用」——一个把这条脑启发方向
错误枪毙的假阴性。

## 修复(本提交)

- `neural_inspired/temporal_surprise_pool.py`:`gamma` 由 `torch.zeros(1)` 改为
  `torch.ones(1)`。`pooled ≠ mean` 于是给 `alpha` 一个活梯度;**`alpha=0` 不变**,所以
  init 处 `output == mean` 的严格不退化起步**照旧**(与 `gamma` 取值无关)。
- `tests/test_neural_inspired.py`:加回归测试
  `test_temporal_surprise_pool_gradient_flows_at_init` —— 同时断言 init 处
  ①`output==mean`(不退化)且 ②`alpha.grad ≠ 0`(门能开)。torch-only,`4 passed`。
  这条测试会在 `gamma=0` 的老实现上失败,正是能拦住本 bug 的那道闸。

## 待办(服务器侧,修复后重跑确认)

```bash
git pull origin main
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_surprise.py 8   # 重跑
python - <<'PY'                                                                # 读 ckpt 确认门开了
import torch
f = "experiments/runs/pilot_10k_surprise/ckpt_09.pth/mp_rank_00_model_states.pt"
sd = torch.load(f, map_location="cpu"); sd = sd.get("module", sd)
a = sd.get("vision_encoder.temporal_pool.alpha")
g = sd.get("vision_encoder.temporal_pool.gamma")
print("alpha =", None if a is None else round(a.float().flatten()[0].item(),5), " (init 0 → 判据 ≠0)")
print("gamma =", None if g is None else round(g.float().flatten()[0].item(),5), " (init 1 → 应移动)")
PY
```

修复前那次的 ckpt 若还在,读出来应是 `alpha=0.0 / gamma=0.0`(死模块的实证)。
修复后重跑应看到 `alpha` 离开 0——那才是链路判据真正通过。

**进度(2026-07-24):重跑已完成,判据数待读。** 服务器已重跑一遍(40:52;ep0 1.03→
ep9 0.28;无 NaN;retrieval r_mean 最好 93.33@ep7)——链路层面全绿,但**训练日志
不打印 `alpha`**,所以还不能判定。**别把"曲线与 r0 相同"当成失败信号**:修复生效时曲线
本来就该相同——`alpha` 从 0 起步、每步梯度 ~2e-3,170 步 × lr 1e-5 只挪极小量,
`output ≈ mean`,retrieval 自然不动。"曲线相同"同时兼容【修复生效】与【没 pull 到修复】,
**只有上面那段读 ckpt 能区分**:`gamma` 判修复在不在(≈1 = 在;恰好 0 = 跑的还是旧代码),
`alpha` 判模块动没动(≠0 = 判据通过)。命令与判读表同步在
[`notes/live/from-claude.md`](../notes/live/from-claude.md) 第 (7) 块。
