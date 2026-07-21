# 10k 抽样审查实测结论（InternVid-zh v0.1）

> 机器：`101vector` 8×V100，数据 `/home/vca1/datasets/internvid_zh_10k`（1 万条预抽样）。
> 流程与决策门见 [`tools/data_audit/SETUP_V100.md`](../tools/data_audit/SETUP_V100.md)，
> 方案见 [`internvid-zh-data-audit-plan.md`](internvid-zh-data-audit-plan.md)。
>
> **为什么有这个文件**：`outputs/` 被 gitignore，`s1_analysis.md` / `report.md`
> 都是不入库的产物。结论必须落到这里才不会随会话丢失（CLAUDE.md §0）。
>
> 本文件是**统计细节与推理过程**；流水线状态、问题登记表和待决策看
> **[`tools/data_audit/DATA_ISSUES.md`](../tools/data_audit/DATA_ISSUES.md)**。

## S1 嵌入打分（2026-07，已完成）

命令：`bash tools/data_audit/run_v100.sh s1` → `s1-report`。

### 实测分布

| 列 | min | p5 | p25 | 中位 | p75 | p95 | max | sd |
|---|---|---|---|---|---|---|---|---|
| `sig_zh_mean` | -0.018 | 0.079 | 0.109 | **0.128** | 0.148 | 0.177 | 0.247 | 0.030 |
| `sig_en_mean` | -0.002 | 0.091 | 0.128 | **0.152** | 0.177 | 0.216 | 0.322 | 0.038 |
| `xling_cos` | 0.013 | 0.611 | 0.711 | **0.767** | 0.813 | 0.877 | 1.000 | 0.082 |

`error 0/10000`，无 NaN，`sig_zh` 唯一值 1484（无塌缩）。
相关性：ρ(sig_zh, sig_en)=**0.461**、ρ(sig_zh, xling)=**0.039**、
ρ(sig_en, xling)=**-0.018**、ρ(xling, d)=**-0.059**（d = sig_en − sig_zh）。

> ⚠️ **0% 解码失败不可外推到全量 10M**：这 1 万条是预抽样，抽样时大概率已按
> 「视频可读」筛过，H 类工程问题在本机天然为空。H100 全量跑必然高于 0。

### 结论 1：S1 机制正常，SigLIP 信号可用

ρ(sig_zh, sig_en)=0.461 是健康的——两个塔共享「视频好不好描述」的公共难度
分量，量级合理。分布铺开、无塌缩。**S1 本身没问题**。

### 结论 2：`xling_cos` 是**弱工具**，不能单独用于分桶

ρ(sig_zh, xling)=0.039、ρ(sig_en, xling)=-0.018——LaBSE 的文本忠实度与画面
对齐**几乎完全无关**。且动态范围极窄：p25–p75 只有 0.711–0.813，sd=0.082，
整批压在一条窄带里。

**推测机制**：InternVid 的 caption 又短又套路（"a man is talking" 类），
LaBSE 在这种短文本上区分度本就低，余弦被挤在中段。这与方案 §1 的 E 类
（粒度粗）是同一个病因的两个侧面。

### 结论 3：翻译损伤真实存在，但**小且只在极低尾**

- 落差 d 的基线（xling 最高十分位内）= **+0.0171** ← 编码器语言尺度 +
  FLT 按英文 CLIP 分选样造成的偏移，**不是翻译的锅**
- 低 xling 十分位内 = +0.0277，**超额落差 = +0.0106**（约 0.3 sd）

方向对、量级小。配合 ρ(xling,d)=-0.059 这个近乎为零的相关：**效应不是线性
铺开的，而是集中在极低尾**。中段的 xling 变化基本是噪声/文风差异，不代表
质量。→ xling 只有极低尾（p5≈0.611 以下）值得当信号。

### 结论 4：语言污染要看 S0 旗标，**xling 抓不到**

先前担心的「未翻译=xling 满分」陷阱**方向对但量级很小**：

- xling ≥ 0.95 的只有 **19 条（0.2%）**，其中 12 条确实被 S0 判为照抄
- 但 S0 判为「中文栏其实是英文」的有 **327 条（3.3%）**，
  **其中 315 条（96%）xling < 0.95，完全漏检**
- 剔除照抄后 xling 中位数 0.767→0.765，几乎没动（占比太小）

即：数据里的语言污染主要不是**逐字照抄**，而是**部分未翻 / 中英混杂**，
LaBSE 对这类给中间分，xling 系统性漏检。
→ **RETRANSLATE 的语言污染判据必须以 S0 旗标为准。**

## 待 S2 裁决的头号问题

解读矩阵预览（低 = 各自最低 25% 分位，临时阈值）：

| 桶 | 条件 | 条数 | 占比 |
|---|---|---|---|
| RETRANSLATE 候选 | en 高 + zh 低 + xling 低 | 387 | 3.9% |
| RECAPTION 候选 | en 低 + zh 低 | 1195 | 11.9% |
| **翻译可疑（画面未受损）** | en 高 + zh 高 + xling 低 | **1522** | **15.2%** |
| KEEP 候选 | en 高 + zh 高 + xling 高 | 4678 | 46.8% |
| 其余混合 | — | 2218 | 22.2% |

最大的异常格是「**翻译可疑**」1522 条：中英两侧画面对齐都正常，只有 LaBSE
说译文不忠实。按结论 2，这一格**很可能主要是 xling 噪声而非真问题**。
**S2 必须先回答这一格里 VLM 的 `trans_score` 究竟差不差**，再谈分桶。

## 对 S3 的具体影响（跑 s3 前必须处理）

`aggregate_report.py` 的 `--xling-threshold` **默认 0.7**，且分桶规则里
**低 xling 单独就能触发 RETRANSLATE**：

```python
elif (fl & RETRANSLATE_FLAGS) or \
        (rec.get("xling_cos") is not None and rec["xling_cos"] < args.xling_threshold):
    bucket = "RETRANSLATE"
```

本批 xling 的 p25 = 0.711 —— **0.7 这个绝对阈值正好落在分布最稠密处**，
会把约 **1/5 的语料**仅凭一个 ρ≈0 的弱信号扫进重翻队列。这是「用未校准的
绝对阈值切未校准的分数」的典型错误。

**决策（等 S2 判定回来再定，不要现在改死）**：统计 VLM `trans_score≤2` 的
差率随 xling 十分位的变化——

- 曲线**平坦** → 把阈值压到极低尾（≈p5=0.611），或让 xling 完全退出分桶
  规则，只留 S0 旗标 + `sig_*` 作判据；
- 曲线**单调下降** → 说明 xling 在本任务上确有判别力，按 §7.5 的验收标准
  正常选阈值即可。

`s1-report` 现在会自动打印这个阈值预演和警告。

## 结论 5：**共享字幕率这项指标,10k 抽样根本测不出来**（2026-07-21）

`make_pilot_anno.py` 在 10k 上测到共享字幕 = **1 组 / 2 条 clip（0.0%）**。
这个数字很容易被误读成「重复 caption 不是问题、假负例治理可以不做」。**不能这么读。**

抽样率 p = 10k / 10M = **0.1%**。一组全局出现 k 次的字幕，抽样后要落进 ≥2 条
才可见，概率 ≈ (kp)²/2：

| 全局重复次数 k | 抽样里出现 ≥2 次的概率 |
|---|---|
| 2 | 1e-6 |
| 10 | 5e-5 |
| 100 | 5e-3 |
| 1000 | 0.26 |

即：**这个抽样只对「全局重复上千次」的模板句有检出力**，对 k=2…100 的中低频
重复完全瞎。反推一下——假设全量有 5%（50 万条）clip 处在 k=2 的重复对里，
那是 25 万组，抽样里的期望命中 = 25 万 × 1e-6 ≈ **0.5 组**。实测 1 组，
与该假设完全相容。**观测到 0.0% 和「全量 5% 重复」在统计上无法区分。**

另外两个把数字往下压的因素（相对次要）：

1. **只算精确重复。** 训练侧 `caption_to_idx`（`dataset/pt_dataset.py:20`）是
   `pre_text` 归一化后取 md5，"a man is talking to the camera" 与 "a man talks
   to the camera" 是两个 idx。语义近重复一概不计。
   *（2026-07-21 已修：`make_pilot_anno.py` 之前按**原始**串统计，比训练口径还严；
   现已改为先过同一套 `pre_text` 归一化再计数。）*
2. **InternVid 的构造方式本就不产生共享字幕。** 每个 clip 由 captioner 独立
   生成描述，不存在「一个视频配 20 条 caption」（MSR-VTT）或「视频级 caption
   复制给所有分段」（部分 WebVid 变体）那种结构性重复。所以重复只会偶发。

### 由此得到的两条行动项

- **判「重复 caption 率 >10%」这个决策门（`DATA_ISSUES.md` §2），必须用全量
  `profile_jsonl.py` 的 `duplicate_zh_caption_rate`，不能用任何抽样数字。**
  10k 上的 `outputs/audit_10k/s0_summary.json` 里已经有这个字段，也同样只是
  抽样值，同样不可外推。
- **口径对不上要注意**：`profile_jsonl.py` 数的是 **zh** caption 的重复，而
  试点训练用的是 **en** caption（`make_pilot_anno.py --caption-field` 默认
  `en`）。英文侧重复经 LLM 逐条翻译后可能被打散成不同中文串，所以
  `duplicate_zh_caption_rate` 是英文侧重复率的**下界**。真要给 `caption_idx`
  定论，得在全量上按训练实际用的那一侧字幕统计。

### 顺带：真正的假负例大头可能不是「同字幕」

InternVid 里同一支源视频会被切成多个 clip，这些 clip 画面高度相似但字幕各不
相同 —— 它们在 batch 里互为负例，是**真的**假负例，而按字幕哈希的
`caption_idx` **一条也抓不到**。若全量重复率证明偏低，比「共享字幕多正例」
更值得做的是**按源视频 id 分组**（同 `video_id` 的 clip 共享 idx，或干脆在
采样器层面禁止同视频 clip 进同一 batch）。这条留给 v0.2 方案权衡。

## 下一步

```bash
bash tools/data_audit/run_v100.sh s2-sample   # 10k 全覆盖
bash tools/data_audit/run_v100.sh s2          # 8 卡 ~2–3 小时
JUDGE_N=500 bash tools/data_audit/run_v100.sh s2-judge   # 需 JUDGE_API_KEY
bash tools/data_audit/run_v100.sh s3
```

S2 回来后要**先**回答两件事，再跑 s3：① 上面那 1522 条的真实差率；
② 8B vs 裁判 action 一致率 ≥75%（§4 决策门）。
