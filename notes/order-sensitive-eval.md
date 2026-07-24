# 顺序敏感评测：选 RTime，但先跑不要钱的那一步

**日期:** 2026-07-24
**上游:** [`experiments/gate0_results.md`](../experiments/gate0_results.md)（MSR-VTT 已被实证取消时序指标资格）
**下游:** [`notes/neural-inspired-round2.md`](neural-inspired-round2.md) §3 R2（可恢复顺序的读出头）

---

## 0. 一句话

**基准选定 RTime**（它就是拿"时间反转"当难负例，与我们要测的东西一一对应）。
**但核实下来 RTime 不分发视频** —— 官方只给原始视频 URL，要自己去源站抓、且要自行取得授权。
所以**先做那件不要钱的**：仓库里已经写好、已自测、但**至今没跑过**的静态审计
（`experiments/temporal_capacity_audit.py`），它从**参数层面**回答同一个问题，零数据零 GPU。
**只有当它显示模型确实编码了时间方向，才值得为 RTime 付视频获取的代价。**

> ⚠️ 这一条修正了 `notes/live/from-claude.md` (20) 里的排序。(20) 把 RTime 列为"最对口"
> 且暗示比 SSv2 省事（"有公开 repo"）。**公开的是标注和代码，不是视频。** 见 §2。

---

## 1. 为什么需要它

GATE-0 实测：MSR-VTT 上打乱 8 帧顺序 **+0.1 / +0.2**（两口径，≈零），而单帧重复 **−13.0 / −12.1**。
⇒ 帧多样性值 ~13 个 R@1，帧顺序值 ~0。**MSR-VTT 对"顺序"这件事没有分辨力** ——
即使 R2 完美恢复了顺序信息，在 MSR-VTT 上也只能测出 ≈0。没有顺序敏感基准，R2 无法证伪，
也就不该开跑。

---

## 2. RTime 核实到的事实（2026-07-24 亲自核对，非二手转述）

| 项 | 值 |
|---|---|
| 论文 | *Reversed in Time: A Novel Temporal-Emphasized Benchmark for Cross-Modal Video-Text Retrieval* |
| 会议 | **ACM MM 2024**(MM '24, Melbourne, 2024-10-28–11-01), DOI `10.1145/3664647.3680731` |
| arXiv | [`2412.19178`](https://arxiv.org/abs/2412.19178) |
| 官方 repo | <https://github.com/qyr0403/Reversed-in-Time>(ACM MM 2024 Poster) |
| 规模 | **21k 视频**、每视频约 **10 条字幕**、合计约 **122 小时** |
| 反转对 | **16,530 个（76.8%）**视频有"时间反转"难负例配对 |
| 三个任务 | **RTime-Origin** / **RTime-Hard** / **RTime-Binary** |
| 标注文件 | repo 内 `train.json` / `valid.json` / `test.json` |

**关键的一条 —— 成本全在这里：**

> repo 原文：**"To avoid potential copyright implications, we only provide original video urls,
> and you can download the corresponding videos after obtaining necessary permission from the
> source website."**

⇒ **视频不随数据集分发**，只给原始 URL，要自己抓 + 自行取得源站授权。
搜索也**没有找到 HuggingFace 镜像**。这与本项目在 InternVid 上已经吃过的苦是同一类
（见 `docs/internvid-download-zh.md`）：链接腐烂、抓取慢、成功率不可控。

另注：官方说明**验证/测试集评测时不使用 GPT-4 改写的字幕**——若自建流程，别把改写稿混进 test。

---

## 3. 先做这件不要钱的：任务 A 静态审计（已合入 · 已自测 · **尚未跑过**）

GATE-0 结果文件 §3 留了一个明确的缺口：目前只能说"MSR-VTT 测不出顺序"，
**还不能区分**「模型真的没有顺序信息」vs「模型有、但 MSR-VTT 不奖励」。

**静态审计正好补这个缺口，而且不需要任何数据。** 它直接读权重：

- **时间卷积核的反对称分量占比** `‖w−flip(w)‖ / (2‖w‖) ∈ [0,1]`
  —— 时间反转把 k=3 核 `[a,b,c]` 变成 `[c,b,a]`，**对称核 = 恰好反转盲**。
- **`temporal_embedding` 的范数** —— 它是**零初始化**的；若训练后仍 ≈0，stage-4 那条通路
  对帧顺序就是等变的（且按 round-2 的 F2，它是**加性**的，本来就被均值池化精确抹掉）。

```bash
cd ~/mobileclip && git pull origin main
python experiments/temporal_capacity_audit.py --self-test
python experiments/temporal_capacity_audit.py ./checkpoints/mobileviclip_small.pt
```

**判读 → 决策：**

| 静态审计结果 | 含义 | 该做什么 |
|---|---|---|
| 反对称占比中位数 **< 0.05** 且 DC/AC **> 0.95**，`temporal_embedding` ≈ 0 | 模型在**参数层面**就几乎不编码时间方向 | **R2 的前提被证实是真问题**，而且此时**不必先买 RTime** —— R2 的成败可以先用"反转敏感度"自身衡量（让嵌入对反转产生差异），RTime 只在需要证明**下游任务收益**时才必需 |
| 反对称占比明显 > 0 | 模型有方向信息，是 MSR-VTT 不奖励 | **值得**为 RTime 付代价：模型有信息、只差一个能测出来的基准 |

⚠️ 该脚本**没有在真权重上跑过**（Claude 侧无权重）。训练态 checkpoint 里时间分支尚未融合成
`reparam_conv_1d`；若默认键名对不上，脚本会把找到的**候选键打印出来** —— 把那段原样贴回来即可。

---

## 4. RTime 落地步骤（等 §3 判读为"值得"再执行）

**只需要 test 划分。** 零样本评测不训练，`train.json` 不需要 —— 这能把抓取量砍掉一大截。
（test 的确切条数请 clone 后从 `test.json` 读，别照抄 21k，那是全量。）

```bash
# ① 标注（免费、秒级）
git clone https://github.com/qyr0403/Reversed-in-Time.git /tmp/rtime
ls -la /tmp/rtime            # 找 train.json / valid.json / test.json 与视频 URL 清单
python -c "import json;d=json.load(open('/tmp/rtime/test.json'));print(type(d),len(d));print(json.dumps(d[:2] if isinstance(d,list) else d,ensure_ascii=False)[:800])"
```

**② 看懂字段后写一个转换器** —— 本仓库的 `load_anno()` 期望每条记录里有 `video` 键
（`dataset/utils.py`，见 `dataset/base_dataset.py` 的 `media_path`）+ 字幕字段。
RTime 的 JSON 结构官方 README **没有文档化**，所以转换器要按 ① 打印出来的真实结构写，
产出 `anno_downstream/rtime_test.json`。

**③ 抓视频**（这是主要成本）：按 repo 给的原始 URL 逐条下载到
`data/rtime/videos/`，文件名与转换后 anno 的 `video` 字段对齐。
弱网/断点续传配方复用 `docs/internvid-download-zh.md`。**记录失败率** ——
链接腐烂会让实际可用集小于官方规模，这个数字必须写进论文的评测说明。

**④ 接进仓库（三处，都很小）：**

```python
# configs/data.py —— 与 ssv2_mc_val 同形
available_corpus["rtime_test"] = dict(
    anno_path=__anno("rtime_test.json"),
    data_root=__data("rtime/videos"),
    media_type="video",
)
```

```python
# experiments/check_readiness.py 的 BENCHMARKS
"rtime": "rtime_test",
```

```
# 新建 scripts/evaluation/clip/zero_shot/mobileviclip_small/config_rtime.py
# 照抄 config_msrvtt.py，把 test_file 指到 rtime_test 即可
```

**接线到此即可用**：`run_eval.sh` 只是拼 `config_${BENCH}.py` 并检查文件存在
（`TASK` 仅在 `ssv2_mc` 时切到 `retrieval_mc.py`），所以加了 `config_rtime.py` 之后
`bash experiments/run_eval.sh small rtime ...` **自动可用**，走标准检索入口 ——
这对 **RTime-Origin / RTime-Hard** 是对的。

⚠️ **RTime-Binary 需要单独的评测入口**：它是"给字幕，在正放 vs 倒放之间二选一"，
形态更接近多选（`retrieval_mc.py`）而非检索。落地时按 `retrieval_mc.py` 改，别硬塞进 `retrieval.py`。

**为什么 RTime-Binary 是最锐利的探针：** 若视频嵌入对时间反转**完全不变**，正放与倒放
对任何字幕都拿到**同一个分数** ⇒ 只能掷硬币 ⇒ **恰好 50%**，零假设是解析已知的。
（严格说：GATE-0 测到反转让 t2v 动了 −1.2/−0.1，说明嵌入并非**精确**反转不变，
所以实测会**接近**而非**恰好** 50%。别把"恰好 50%"写成预测值，它是**不变情形下的**下界锚点。）

---

## 5. 备选对比（都要视频，没有免费午餐）

| 方案 | 标注 | 视频 | 对口程度 | 备注 |
|---|---|---|---|---|
| **RTime** | 免费(repo) | **自抓 URL + 授权** | **最高** —— 反转即难负例，Binary 有解析零假设 | 无 HF 镜像;需写转换器 + Binary 单独入口 |
| SSv2-MC | 缺(`ssv2_mc_val.json`) | 官方站注册下载 | 高(SSv2 本就时序敏感) | 本仓库**已有** `config_ssv2_mc.py` + `retrieval_mc.py`,接线成本≈0;卡在数据 |
| SSv2-template / label 检索 | 需自建(Lei et al. ACL 2023) | 同上,依赖 SSv2 视频 | 高,且比 MC 更严(MC 会漏外观线索) | 视频与 SSv2-MC 共用 ⇒ 一次下载两用 |

**成本上的实话：** 若最终要下 SSv2 视频，**SSv2-MC 的接线成本是零**（配置和 MC 入口都已在仓库里，
GATE-0 那次崩溃只是因为缺 anno+视频），而 RTime 还要额外写转换器和 Binary 入口。
所以"哪个更省"取决于**哪份视频更容易拿到**，不取决于代码。

---

## 6. 决策树（照这个走）

```
1. 跑静态审计（零成本，今天就能跑）
     ├─ 反对称占比 ≈0  ⇒ 模型参数层面就时间盲
     │     ⇒ R2 的前提确认成立；先用"反转敏感度"自测 R2，暂不买基准
     │     ⇒ 只在要证明下游收益时再回到第 2 步
     └─ 反对称占比 >0  ⇒ 模型有方向信息、是基准测不出
           ⇒ 值得买基准，进第 2 步
2. 比较 RTime 视频 vs SSv2 视频哪个实际拿得到（这是唯一的决定因素）
     ├─ RTime  ⇒ 按 §4 走（多写：转换器 + Binary 入口）
     └─ SSv2   ⇒ 接线成本≈0，直接补 anno+视频即可复跑 GATE-0 的 SSv2-MC 半边
3. 拿到基准后再上 R2（round-2 §3 R2 的消融梯子：mean / additive / slope / phase，
   其中 additive 按 F2 预测应当无增益 —— 那是判据本身）
```

---

*核实说明：§2 的论文/会议/DOI/arXiv/repo/规模/三任务/"只给 URL 不给视频"均于 2026-07-24 联网核对；
RTime 的 JSON 字段结构官方未文档化，§4 ② 因此写成"按真实结构写转换器"而非给死代码。*
