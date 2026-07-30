# 1M 验证层(sub1m_*)—— 在子集上「真·验证」臂能否赢基线

介于两个已有档位之间的**中间层**:

| 档位 | 数据量 | 用途 | 能得出什么 |
|------|--------|------|-----------|
| `pilot_10k_*` | ~10k(0.1%) | 链路冒烟 | 只有「能不能跑通 / 大致方向」,**不是排序** |
| **`sub1m_*`(本层)** | **1M(~10%)** | **验证决策** | 「值不值得上全量」的高置信判断 + 否决 |
| `r0_baseline` 等全量 | 10M | 论文数字 | 唯一能对比 paper 的结果 |

为什么需要它:10k 下 40M 参数模型**严重欠拟合**,学到的池化/损失机制**表达不出优势**,
差值淹没在 MSR-VTT 1k 的 ±SE 噪声里 —— 所以 10k 只能冒烟。1M 让模型进入「真的在学」
的区间,臂的机制才有机会显形,同时 wall-clock 仍远低于全量。

---

## 设计(8 个 config = 4 臂 × 2 seed)

全部继承 `scripts/.../mobileviclip_small/config.py`,只改必要项:

- **数据**:`train_corpus="internvid_v1"`(单个 1M 分片;`train_file` 由 `${...}` 模板惰性解析)。
- **batch**:`batch_size=128`(与**在跑的全量 R0 一致**,经模板传到 `inputs.batch_size.video`)。
- **epoch**:继承基线 `3`(**与全量同配方**,只是数据 10%)。
- **测试集**:继承基线 `msrvtt_1k_test`(**标准 MSR-VTT 1k,绝不缩测试集** —— 缩只会加噪、毁可比性)。
- **seed**:42 / 43 两套孪生,判「是不是单 seed 假象」。
- 臂改动(相对 sub1m_r0,各自唯一变化):
  - `sub1m_r1a` —— `use_sigmoid_loss=True` + 多正例 + logit_scale/bias 大 lr(**损失轴**,与池化正交)。
  - `sub1m_r1b` —— `temporal_pooling="attn"`(**池化轴**,工程基线)。
  - `sub1m_nA1` —— `temporal_pooling="surprise"` + **`temporal_pool` different_lr=1e-4**(脑启发孪生;
    开 different_lr 是防止惊奇门 alpha 不动、退化成 mean)。

各臂 `output_dir` 相互隔离(`experiments/runs/sub1m_<臂>_s<seed>`),不会互相覆盖 ckpt。

### ➕ R1 文本负例库 A/B(+2 臂 × 2 seed;**彼此相减,不减 r0**)

把 R1 文本负例库的「真正发威档」A/B 也放到本层跑(不等全量 10M)。它是一对**自配对** A/B,判读方式
与上面四臂不同(**处理减自己的对照,不减 r0**),所以单列:

- `sub1m_r1both_s42/s43` —— **对照**:SigLIP 制度(sigmoid + attn 池化 + caption_idx + t/b 大 lr),**不含** bank。
  = 10k 试点对照 `pilot_10k_r1.py` 的 1M 直放大;也顺带补上四臂里缺的「r1a+r1b 合并」臂。
- `sub1m_nR1_s42/s43` —— **处理**:在对照基础上**唯一新增** `criterion.text_bank`(1M 缓存 `textbank_1m_*`,
  K=16384、dedup_cos=0.90;每步 randperm 抽 16384 条重投影当 sigmoid 负列)。
- **为什么 1M 才是这个 A/B 的正确尺度**:全局 batch 512 远低于 SigLIP 饱和点(~32k),bank 补的 +16384 正好
  推进甜区;1M×3 epoch(~5.9k 步)**不过拟合**,所以领先=真硬负例收益,**不是** 10k 试点那种「挡早塌」的
  小数据伪影(见 `r1-implementation-plan.md` §9.4 / §10)。
- **判胜负**:ΔR-Sum = `sub1m_nR1_sXX` − `sub1m_r1both_sXX`(**同 seed**),两个 seed 同向且超 ±SE 才算赢;
  两方向都报(bank 硬化 v2t/`txt_r1`,t2v 只间接受益)。**先建缓存再跑处理臂**(命令见 §跑法 & 时长 / COMMANDS §3.5)。
  全套设计见 [`r1-implementation-plan.md`](r1-implementation-plan.md) §10。

---

## 怎么读结果(判胜负的规矩)

每臂 epoch 末在**标准 MSR-VTT 1k** 上出 retrieval;取最好一轮的 R@1/R@5/R@10(或 R-Sum)。

1. **同 seed 相减**:`sub1m_r1a_s42` 减 `sub1m_r0_s42`;`sub1m_r1a_s43` 减 `sub1m_r0_s43`。
2. **两个 seed 必须方向一致**,且差值 **超过 MSR-VTT 1k 的 ±SE 阈值**(见 §1.8 的
   ±1.6→2.4 二项 SE 约定)。单 seed 的差、或小于阈值的差 = **不确定**,不能下结论。
3. **nA1 的基准是 r1b(attn),不只是 R0** —— 惊奇池化必须同时赢 mean(R0)和 attn(r1b)才算贡献。
4. **nA1 收尾必须连最终 `alpha` 一起报**(读法见 `neural_inspired/README.md`):
   `alpha` 仍 ≲0.01 ⇒ `output≈mean` ⇒ 这一臂已退化,此时「输给 r1b」不代表方向错,
   只能靠 alpha 区分,**不得据此裁掉方向**。

### 判定
- **臂在两个 seed 都显著赢基线** → 高置信「有戏」→ 值得花 7 天上全量拿论文数字。
- **臂在 1M 两个 seed 都赢不了基线** → 强**否决**:全量几乎不可能翻盘,省下 7 天。
- **矛盾/小差** → 1M 定不了,只能上全量裁决。

---

## ⚠️ 不可消除的警告

**方法排序可能随数据尺度翻转** —— 单一子集尺度**不能保证**结论迁移到 10M。所以:

- sub1m 是**决策工具**(要不要烧全量算力)和**高置信否决器**,不是最终结论。
- 想把「随尺度稳定」也测掉,需要**两个子集尺度**(如 500k + 1M)看差值趋势 —— 本层按需求只做**一个尺度(1M)**,故保留此风险。
- **sub1m 永远给不了论文数字**;对比 paper 只能用**全量 R0**。

---

## 跑法 & 时长

**可用卡 = 0–3(共 4 张,是维护者在这台 H100 的全部配额;没有 4–7 可用)。** 全量 R0、本验证层、
R1 A/B **都挤在这 4 张上 → 只能时分复用、串行**:R0 在跑时本层排队,等 R0 空出/暂停后再跑。**不能**
像本文早先设想的「R0 占 0–3、验证层用空出来的 4–7」那样分两组并行 —— 4–7 对维护者不可用。下面命令一律用 `0,1,2,3`:

```bash
# 挂后台并记 PID(共用 root,停止只能按 PID/PGID,严禁 pkill)
nohup bash experiments/run_verify_1m.sh 0,1,2,3 > experiments/runs/verify_1m.log 2>&1 &
echo "verify-1m PID = $!"

# 只先出 seed 42 的四臂(~2.5 天先看一版 A/B):
bash experiments/run_verify_1m.sh 0,1,2,3 \
    sub1m_r0_s42 sub1m_r1a_s42 sub1m_r1b_s42 sub1m_nA1_s42

# 收结果(每条 log 的 epoch 末评测行):
grep -hE "t2v_r1|v2t_r1|_match|R@" experiments/runs/sub1m_*_s4?.log | tail -60
```

**R1 文本负例库 A/B(+4 config;先建一次 1M 缓存,再串行两对)**:
```bash
# 1) 建 1M 文本负例库缓存(单卡 forward-only,一行别断行;~10–30 min,~1GB):
python experiments/r1_build_textbank.py --anno anno_downstream/InternVid/InternVid-10M-flt_1.json --out-feats experiments/cache/textbank_1m_feats.fp16.npy --out-idx experiments/cache/textbank_1m_idx.npy
# 2) 跑两对(复用同一 runner;彼此相减判胜负):
bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_r1both_s42 sub1m_nR1_s42 sub1m_r1both_s43 sub1m_nR1_s43
grep -hE "t2v_r1|v2t_r1|_match|R@" experiments/runs/sub1m_{r1both,nR1}_s4?.log | tail -40
```

**时长(单个 4-GPU slot、batch 128、9 s/step 量级)**:1M×3 epoch ≈ 1953 步/epoch ×3 ≈
5.9k 步 ≈ **~15 h/config**;8 个串行 ≈ **~5 天**(R1 A/B 那 4 条另 ~2.5 天)。**只有 0–3 这 4 张 ⇒
只能与 R0 串行,没有第二组卡可并行**;要更快只能减 config(先只跑 1 个 seed = 减半)。**断点续训**:单独重跑某条并追加 `auto_resume True`
(output_dir 已隔离)。

相关:全量 R0 在 `r0_baseline.py`(当前以 batch 128 在 0–3 跑);长跑运维/选卡/续训见
`COMMANDS.md` §4;各臂全量版见 `r1a_sigmoid.py` / `r1b_attnpool.py` / `nA1_surprisepool.py`。

---

## 6. 首个实测结果:`sub1m_r0_s42`(2026-07-29)—— 管线验证通过,但**测量功效告急**

离线补评(`COMMANDS.md` §3.8,权重加载已核实:日志有 `Load pretrained model from
.../ckpt_02.pth/mp_rank_00_model_states.pt`),标准 MSR-VTT 1k、`num_frames_test=8`:

| 参照 | DSL t2v R@1 | DSL r_mean | **raw t2v R@1** | **raw r_mean** |
|---|---|---|---|---|
| 地板:骨干初始化 + **随机**时序算子 | 34.9 | 56.92 | 32.2 | 54.82 |
| **`sub1m_r0_s42`(1M × 3 epoch)** | **42.7** | **61.28** | **41.8** | **60.65** |
| 天花板:官方权重(10.6M 全量) | 42.5 | 62.08 | 41.3 | 61.47 |

### ✅ 好消息:复现管线端到端验证通过

**用 10% 的数据、3 个 epoch,MSR-VTT t2v R@1 就已经追平官方全量模型**
(raw 41.8 vs 41.3、DSL 42.7 vs 42.5,两口径同向且都在 ±1.6 带内 ⇒ 判"打平")。
⇒ 数据管线、训练循环、评测口径**都是对的**;`sub1m_*` 这一层可以当可信的代理档。

### 🔴 坏消息(更重要):**MSR-VTT 1k 在这个尺度上已经没有分辨力了**

**数据翻 10 倍(1M → 10.6M),raw r_mean 只涨 +0.82、DSL 只涨 +0.80。**
而仓库级噪声约定(`experiments/README.md` §6.1)是:**阴性用 ±1.6 带,阳性主张要 |Δ| > 2.4**。

⇒ **10× 数据这个"巨大"处理效应,本身就落在噪声带以内。**
任何一条臂(r1a/r1b/nA1/nR1)的效应几乎必然**比 10× 数据还小** ⇒
**在 MSR-VTT 1k 上按现有判据,1M 层什么都排不出来。**

这不是 1M 层设计错了 —— 是**测量仪器精度不够**。三条出路,按性价比排:

1. **配对 bootstrap(最高优先,且本来就欠着)** —— 各臂评的是**同一** 1000 条查询,
   属**配对比较**;配对差值的方差远小于"各自对一个绝对值"的 ±1.6 无配对带。
   ⇒ **U10-1 的 query-level bootstrap CI 不是锦上添花,它是让 1M 层能用的前提。**
   (对应 ROADMAP 完整性自查 **G4**,至今未实现。)
2. **多基准聚合 R-Sum**(README §6 既有纪律):MSR-VTT + DiDeMo + ANet 同向,
   单基准本来就不够 —— 本结果把这条从"纪律"升级为"必须"。
3. **接受分辨率下限**:1M 层只用于**否决**(掉点明显的臂砍掉),
   小的正向差异一律留到全量定夺,不在 1M 层下结论。

### 顺带确认的两件事

- **地板 → 1M 是 +5.83 raw r_mean** ⇒ 视频训练确实在起作用,不是没学到东西;
  但 **1M → 10.6M 只有 +0.82** ⇒ 收益早已进入长尾。这与 GATE-0 的结论一致
  (时序通路贡献小、大部分性能来自图像骨干)。
- 本轮跑的是 `deepspeed.enable=False` 的评测路径,权重经目录自动解析加载;
  `missing_keys` 只剩**冻结的文本塔**(训练时 `exclude_frozen_parameters=True` 不存),
  它在建模时已从 `mobileclip_s2.pt` 载入 ⇒ **属预期,不影响结果**。

---

## 7. 第一条臂 A/B:`sub1m_r1a`(SigLIP sigmoid 损失) vs `sub1m_r0`(2026-07-30)

r1a 补评已就位(权重加载已核实:`Load pretrained model from
.../sub1m_r1a_s42/ckpt_02.pth/mp_rank_00_model_states.pt`),配对 bootstrap
(`collect_results.py --compare`,§3.9,10000 次重采样、seed=0)出第一份 A/B。

**seed 42,raw 主口径 `ret_test_emb/`:r1a − r0**

| 指标 | r1a | r0 | Δ | 95% CI | 判定 |
|---|---:|---:|---:|---|---|
| txt_r1 | 40.60 | 41.80 | **−1.20** | [−3.10, +0.70] | NS(含 0) |
| img_r1 | 39.10 | 39.00 | +0.10 | [−1.80, +2.00] | NS(含 0) |
| r_mean | 60.02 | 60.65 | **−0.63** | [−1.40, +0.13] | NS(含 0) |
| R_sum  | 360.10 | 363.90 | **−3.80** | [−8.40, +0.80] | NS(含 0) |

配对把 R-Sum 的 95% CI 半宽从无配对 ±15.1 缩到 **±4.6**(约 1/3.3 分辨率)。

**seed 42,DSL 口径 `ret_test/`:r1a − r0**(2026-07-30 补齐,两口径纪律闭合)

| 指标 | r1a | r0 | Δ | 95% CI | 判定 |
|---|---:|---:|---:|---|---|
| txt_r1 | 41.30 | 42.70 | −1.40 | [−3.30, +0.50] | NS(含 0) |
| img_r1 | 39.40 | 40.00 | −0.60 | [−2.50, +1.30] | NS(含 0) |
| r_mean | 60.33 | 61.28 | **−0.95** | **[−1.68, −0.20]** | **显著(不含 0)** |
| R_sum  | 362.00 | 367.70 | **−5.70** | **[−10.10, −1.20]** | **显著(不含 0)** |

**读法(两口径合看)**:
- **方向:两口径几乎所有指标一致为负** —— r1a 全面略输 r0,纪律「两口径同向」满足。
- **显著性:DSL 口径的聚合量(r_mean/R_sum)CI 已不含 0 = 显著为负**;raw 口径同向但卡边界
  (r_mean [−1.40,+0.13]、R_sum [−8.40,+0.80],差一点没到显著)。DSL(dual-softmax)是更锐的
  排序口径,先探到这条差距。**别把「显著 vs 边界」当矛盾 —— 纪律要的是方向一致,而方向一致。**
- **结论(seed 42):r1a 不是「没赢」,是在更锐的口径上「显著输」**。**符合理论**:SigLIP sigmoid
  要**大 batch(~32k)**才发威,本层全局 batch 512(128×4)远低于甜区 ⇒ softmax(r0)胜出意料之中
  (与 R1 文本负例库 A/B 的批量论证同源,见 §设计 ➕ 段 / `r1-implementation-plan.md`)。

**还差 seed 43 才能定案**:`sub1m_r1a_s43` − `sub1m_r0_s43` 尚未训练(排在 `r1b_s42`/`nA1_s42`
之后,还要几天)。按 §怎么读结果,**两个 seed 方向一致**才下判决。seed 42 已是**较强负向**
(DSL 显著);若 seed 43 同向 ⇒ **否决 r1a**(这个 batch 尺度下 sigmoid 不划算),把算力让给
r1b(attn)/nA1(surprise)那两条更有戏的臂。
