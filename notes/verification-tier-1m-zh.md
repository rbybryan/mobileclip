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

在 R0 全量占着 0–3 时,用空出来的 4 张卡(如 4–7)串行跑:

```bash
# 挂后台并记 PID(共用 root,停止只能按 PID/PGID,严禁 pkill)
nohup bash experiments/run_verify_1m.sh 4,5,6,7 > experiments/runs/verify_1m.log 2>&1 &
echo "verify-1m PID = $!"

# 只先出 seed 42 的四臂(~2.5 天先看一版 A/B):
bash experiments/run_verify_1m.sh 4,5,6,7 \
    sub1m_r0_s42 sub1m_r1a_s42 sub1m_r1b_s42 sub1m_nA1_s42

# 收结果(每条 log 的 epoch 末评测行):
grep -hE "t2v_r1|v2t_r1|_match|R@" experiments/runs/sub1m_*_s4?.log | tail -60
```

**时长(单个 4-GPU slot、batch 128、9 s/step 量级)**:1M×3 epoch ≈ 1953 步/epoch ×3 ≈
5.9k 步 ≈ **~15 h/config**;8 个串行 ≈ **~5 天**。要更快:只跑 1 个 seed(4 config,减半),
或等全量 R0 腾出更多卡后分两组并行。**断点续训**:单独重跑某条并追加 `auto_resume True`
(output_dir 已隔离)。

相关:全量 R0 在 `r0_baseline.py`(当前以 batch 128 在 0–3 跑);长跑运维/选卡/续训见
`COMMANDS.md` §4;各臂全量版见 `r1a_sigmoid.py` / `r1b_attnpool.py` / `nA1_surprisepool.py`。
