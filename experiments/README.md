# experiments/ — 优化方案 A/B 测试框架(对照原 repo 基线)

**目的:** 拿到数据集路径后,不改任何代码、按下面的步骤顺序执行,就能
(1)锁定原 repo 模型的基线数字,(2)逐个 A/B 我们的优化组件,(3)自动汇总成
与基线的对比表。

> **数据集/权重到位了没有 → `python experiments/check_readiness.py`**
> 自动检查全部基准、训练语料和 checkpoints,并报告已解锁的步骤。

## 结果报告一览(已跑完的,直接看数字)

| 报告 | 回答什么 |
|---|---|
| [`gate0_results.md`](gate0_results.md) | GATE-0 时间维容量审计 —— 这个模型编不编码时间方向?(结论:不编码) |
| [`pilot_10k_results.md`](pilot_10k_results.md) | V100/10k 试点的链路 go/no-go 记录 |
| [`v100_sweep_results.md`](v100_sweep_results.md) | V100 上的零训练扫描(分辨率、算子计数等) |
| [`frame_select_results.md`](frame_select_results.md) | 选帧基线 —— C2 必须打赢的那条平凡基线 |
| [`dropblock_results.md`](dropblock_results.md) | 逐块深度敏感度扫描(P11) |
| [`export_smoke_results.md`](export_smoke_results.md) | ONNX / Core ML / TFLite 导出冒烟 |

1M 验证层 6 个模型改造方案的结果在
[`../model-optimization/docs/retrospective-brief.md`](../model-optimization/docs/retrospective-brief.md)。

---

## 两级实验:V100/10k 只筛链路,H100/全量 定胜负 + 终训

优化方案(P3 注意力池化、P4 sigmoid、`caption_idx` 多正例……)的验证分两台机器、
两种角色,**别把职责搞混**——尤其别指望用 10k 来给方案「选型」:

| 机器 | 数据 | 角色 | 判据 |
|---|---|---|---|
| **101vector · 8×V100** | 只有 10k 预抽样(`/path/to/user/datasets/internvid_zh_10k`) | 每个方案的**链路 / 稳定性 go-no-go**:能不能跑通、loss 正不正常降、新参数是否真收到梯度 | 试点动力学档「只看动力学」:loss 下降、`logit_scale/logit_bias` 离开初值、`temporal_pool.proj` 非零。**过 = 值得拿去 H100;不等于这个方案更好。** |
| **b100 主机 · 8×H100(默认前 4 张)** | 全量 10M(`/workspace/mount/.../InternVid-10M-FLT`) | 方案之间**真正排序**(R0 vs r1a/r1b/r1_both 的 A/B)+ 胜者的**终训** | 真检索基准 + 足够步数下的方向性差异(§3 的 B0 口径) |

**为什么 10k 不能给方案排序**(关键,别拿它选型):试点的 held-out 自检索在开训前
就接近饱和——2026-07-22 实测 epoch-0 R@1 已是 79.6 / 83.0,只剩 ~20pt 空间,而 1000 条
query 的二项标准误就有 1.3pp,再叠加 BN running-stats 适应的混淆,两个方案在 10k 上的
差别基本落在噪声里(详见 §2b)。所以 10k 只能回答「这个方案实现对不对、稳不稳」,
回答不了「哪个方案更好」。

**这个分工本来也是被数据位置逼的**:全量 10M 只在 H100 机上,V100 机只有那 1 万条——
就算想在 V100 上排序也没有数据。

⇒ **流程**:在 V100/10k 上逐个方案做 go/no-go 筛查(链路对不对、稳不稳)→ 把过筛的
方案拿到 H100 上做真 A/B 排序 → 只把胜者跑完整训。各机怎么跑:
V100/10k 见 **§2b** + `make_pilot_anno.py`;H100/全量见 [`SETUP_B100.md`](SETUP_B100.md) + **§4**。

---

## 0. 本分支包含的代码变更(一次性说明)

| 类别 | 变更 | 对基线数字的影响 |
|------|------|------------------|
| 卫生 H1 | `tasks_clip/pretrain.py` 的 `set_detect_anomaly(True)` 改为仅 `debug=True` 时开启 | 无(纯提速 2–5×) |
| 卫生 H4 | 移除 spikingjelly/snntorch 死依赖(`retrieval_utils.py`、`mobileone.py`、环境文件) | 无 |
| 冻结 bug | `hasattr(name,'_1d')`(恒 False)修为 `name.startswith("head.")` | 现有配置 `freeze_vision=False`,无影响;冻结类实验从此正确 |
| 路径参数化 | `configs/data.py` 支持 `MVC_DATA_DIR` / `MVC_ANNO_DIR` 环境变量 | 无(默认值 = 原相对路径) |
| 组件 P4 | `model.use_sigmoid_loss`(默认 **False**):接通 `vtc_loss_sigmoid`,可学习 t/b(SigLIP 初始化),共享字幕按多正例处理 | 默认关闭 = 原行为 |
| 组件 P3 | `model.vision_encoder.temporal_pooling`(默认 **"mean"**):`"attn"` 启用零初始化时间维注意力池化 | 默认关闭 = 原行为;开启时 step 0 与 mean 逐位等价 |
| 卫生 H2 | `with_cp`(梯度检查点)接到 `gradient_checkpointing` 配置(原先写死 True) | 默认 True = 原行为;显存足够时置 False 可省 20–40% 反向时间 |
| 数据 idx | `caption_idx`(corpus 级,默认 **False**):置 True 时 idx = 字幕哈希,共享字幕的 clip 共享 idx,多正例掩码才真正生效 | 默认关闭 = 原行为(idx 逐样本唯一) |

两个组件开关都写进了全部 16 个配置文件(默认关闭),因此**基线路径的数学
行为与原 repo 完全一致**。

## 1. 前置准备(拿到机器 + 数据路径后做一次)

```bash
# 1) 环境(细节见 COMMANDS.md;spikingjelly/snntorch 已不再需要)
bash scripts/setup_env.sh

# 2) 数据路径:两个环境变量,建议写进 ~/.bashrc 或训练脚本
export MVC_DATA_DIR=/path/to/data              # 视频根目录
export MVC_ANNO_DIR=/path/to/anno_downstream   # 标注 json 根目录
# 期望的子目录/文件名见 configs/data.py(MSRVTT_Videos/、msrvtt_1k_test.json 等)

# 3) checkpoints/ 下需要:
#    mobileclip_s2.pt        (MobileCLIP-S2 底座,训练用)
#    mobileviclip_small.pt   (官方发布权重,锁基线用)
```

## 2. 冒烟测试(任何长训之前)

```bash
# 单测:sigmoid 损失默认行为不变 + 注意力池化零初始化 == mean
python -m pytest tests/test_optimization_components.py -v

# 单测:C2 T 无关化(C2 · T 无关化)——T=8 与旧硬编码路径逐位一致、任意 T 前向、
# temporal_embedding 插值、弹性 T 采样器、per-T BN 工具(纯 CPU 可跑,~4 分钟)
python -m pytest tests/test_t_agnostic.py -v

# 短训冒烟(单卡、debug 模式几十步自动停,验证数据/模型/损失全链路):
bash experiments/run_pretrain.sh experiments/configs/r0_baseline.py 1 debug True
bash experiments/run_pretrain.sh experiments/configs/r1_both.py    1 debug True
```

## 2b. 10k 试点(只有审查抽样、没有全量数据时;101vector 8×V100)

全量 InternVid 未到位时,用数据审查的 1 万条抽样把训练链路和组件全部调通。
**试点数字一律不外推、不进任何结论**(全量的千分之一,~180 步)。

### 能在 V100/10k 上测什么样的模型改动(推广到 r0/r1 之外)

下面的 r0/r1 只是**第一对**改动。任何新模型改动(新模块、新损失、新算子)在烧
H100 之前都该先在这里过一遍——但只测得了**实现,不是效果**:

- ✅ **能确定性回答的机械判据**:开关关闭时与原 repo **逐位等价**(单测)、
  能建模能前后向不 NaN、新参数进优化器表且**离开初值**、loss 单调降无 `OVERFLOW`、
  显存放得下(峰值 23.2 GiB / bs=64)。**这些每个改动都该做。**
- ❌ **测不了「这个改动好不好 / A/B 谁赢」**:10k 评测开训前就近饱和
  (R@1 79.6/83.0)、BN running-stats 混淆、二项 SE 1.3pp ⇒ 差别落进噪声。
  排序一律上 H100 全量(`SETUP_B100.md`)。试点里 r1 低 r0 6.5 点**不代表改动有害**。

| 模型改动 | 10k 上测什么 | 注意 |
|---|---|---|
| C3 注意力池化(`temporal_pool`) | ✅ 完整 plumbing:zero-init==mean 单测、建模、梯度、稳定性 | 已在 r1 跑过 |
| C3 sigmoid 多正例损失 | ✅ 建模、可学 t/b 离开初值、稳定性 | 已在 r1 跑过 |
| C2 T 无关化(动态 T / `temporal_embedding` 插值 / 随机 T) | ✅ **代码已落地(2026-07-24)**:单测 `tests/test_t_agnostic.py` 18 项全过(T=8 逐位一致 ✓);V100 随机 T 冒烟:`bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r3e.py 8 debug True` | C2 · T 无关化;开关 `elastic_t=dict(enable,choices)`,默认关=零改动 |
| **C3 `caption_idx` 多正例掩码** | ⚠️ **10k 上测不了**:全库仅 1 对共享字幕,掩码触发不了。要 plumbing 测得自造「字幕故意重复」的小合成集,否则等 H100 全量 | 见下方 07-21 共享字幕说明 |
| C1 教师蒸馏(DR) | ✅ 但测的是**成本不是效果**:10k 上试点教师每 clip 耗时/存储,定 L vs 1B | C1 教师管线 |
| **C2 per-T BN 漂移诊断 + 校准**(U3) | ✅ 工具已落地:`experiments/per_t_bn.py`(snapshot / reset+recalibrate / drift report / per-T bank),单测在 `tests/test_t_agnostic.py`;对真 ckpt 的诊断随 r3e/R3 试点跑(用法见该文件 docstring,CPU/单卡即可) | 评审 U3;随机 T 训练的 BN 统计混淆正是 10k 排序禁令的原因之一,这里只测机械正确性 |

**V100 硬件红线**(决定改动**能不能跑起来**,与选型无关):必须用带 `sm_70` 的
cu118 torch(否则第一个 `barrier()` 全崩,见 §2b 第 3 步预检);**无 bf16、无
flash-attn**(sm_70 装不了,只能 fp16 AMP + 回退)——**若某改动的正确性依赖
flash-attn 专用 kernel 或 bf16 数值,它在 V100 上验证不了,直接上 H100**;
A/B 两臂 batch size 必须一致(=批内负例数=对比学习任务本身)。

```bash
# 1) jsonl -> 训练 anno(9k train + 1k held-out 自检索 eval;
#    同时打印共享字幕统计 = caption_idx 多正例的实际影响面)
python tools/make_pilot_anno.py \
    --input /path/to/user/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /path/to/user/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

# 2) 数据路径:videos 在 /path/to/user/datasets/internvid_zh_10k/videos
export MVC_DATA_DIR=/path/to/user/datasets   # anno 用默认 anno_downstream/ 即可

# 3) 开跑前必做的 V100 预检:torch 必须带 sm_70 内核
#    (env 里若是 cu126/cu128/CUDA-13 构建,8 卡会在第一个 barrier 就全部
#     报 "no kernel image is available";修法见 TROUBLESHOOTING.md 末节)
python -c "import torch; a=torch.cuda.get_arch_list(); print(torch.__version__, a); \
  assert 'sm_70' in a, 'torch 无 Volta 内核 → 见 TROUBLESHOOTING.md'; print(torch.zeros(1).cuda())"

# 4) 冒烟(debug 几十步自停)→ 微型 A/B(等步数,~180 步)
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
```

> ⚠️ **batch size 不是自由旋钮,r0 与 r1 必须一致。** 它决定批内负例数量,
> 直接改变对比学习任务本身 —— bs=16 的运行和 bs=64 的运行不可比。冒烟测试可以
> 随便调小(只验证链路),但 A/B 那一对必须同批大小,优先用配置里记录的
> 64/卡(全局 512)。另注意本机是共享的:若别的任务(如 S2 审核,每卡约 17 GiB)
> 正占着卡,会 OOM —— 先 `nvidia-smi` 看清楚,别急着改配置,详见 `TROUBLESHOOTING.md`。

试点通过标准(全是动力学检查,不看绝对精度):两条损失曲线正常下降;
r1 训完从 ckpt 读 `logit_scale`/`logit_bias` 已离开初始值(exp(t')≠10、
b≠−10,否则 different_lr 没生效);`temporal_pool.proj` 权重非零;
held-out 自检索 R@1 相对方向仅作参考。评测走 `ret_test`(1k 自检索),
不依赖 MSR-VTT——MSR-VTT 下载完成后再走 §3 的 B0。

> **已测(2026-07-22,冒烟通过):这个 held-out 评测在开训前就接近饱和,
> 不能用它判组件优劣。** 第 0 个 epoch(只跑了 5 步、lr≈2e-6,等于没训)
> `ret_test/` 已经是 t2v R@1 **79.6** / v2t **83.0**(raw 口径 78.1 / 81.9)。
> 原因是两个塔都从 `mobileclip_s2.pt` 载入、本来就对齐,试点训练只是在一个
> 已经很强的模型上动了 170 步。
> 剩余空间只有 ~20 个点,而 1000 条 query 的二项标准误就有 1.3pp
> (`sqrt(.8×.2/1000)`),两次运行相差 <3.5pp 都在噪声里。
> **所以试点判据只看:(a) 损失曲线形态,(b) 上面那几条组件存活检查。**
>
> 还有一个混淆项:视觉塔没冻结(`freeze_vision=False`)且 MobileOne 主干
> 全是 BatchNorm(默认 momentum 0.1)。训练期 running stats 会在几十步内
> 被这批视频数据重估一遍,而**评测用的正是 running stats**。所以 R@1 的变化
> 可能纯粹来自 BN 统计量适应,而不是梯度学到了东西。两臂同样受影响,A/B 仍然
> 公平,但「R@1 涨了」≠「这个组件有用」。
>
> **开局 loss ≈ 1.13**(不是随机初始化该有的 ~13.9),同样是因为塔已对齐 ——
> 见 `TROUBLESHOOTING.md`「反向出 NaN」一节的 2026-07-22 更正。

**跑真 A/B 前的三件事(实测于 2026-07-22 冒烟):**

1. **别加 `debug True`。** 它每个 epoch 只跑 5 步就 break,但 epoch 循环照走,
   于是变成「5 步训练 + 整轮评测」×10 —— 只验证链路,出不了曲线。
2. **两臂的 `deepspeed.initial_scale_power` 必须相同。** 两个 pilot 配置都从
   同一个 base config 继承(默认 8),所以**不传 CLI 覆盖**就是对称的。真要改
   就两臂一起改。溢出会跳步,跳步数不同即破坏等步数前提 —— 跑完
   `grep -c OVERFLOW` 两个 `train.log`,数字差得多就要记进结论。
   注意两臂数值特性本就不同:r0 走 `/temp` 路径,r1 是 sigmoid(`temp=1.0`,
   改用 `logit_scale`/`logit_bias`),没有 `1/temp²` 放大。
3. **先 `nvidia-smi` 确认卡是空的。** 实测峰值 `max mem 23.2 GiB / res mem 27.2 GiB`
   (bs=64/卡,评测阶段最高),32 GiB 的 V100 放得下但余量不多,来个同租户就 OOM。

**耗时(实测外推):** 稳态约 **8.7 s/步**(其中数据 3.7–4.6 s,decord 解码是
瓶颈之一),17 步/epoch ≈ 2.5 min,epoch 末评测 ≈ 1.1 min ⇒ **≈3.6 min/epoch,
单臂 ≈40 min,一对 ≈80 min**。实跑核对:r0 `Training time 0:38:10`、r1 `0:37:46`,
稳态 9.6–9.9 s/步 —— 估算成立。

### 试点结果(2026-07-22,170 步 × 2 臂,全程零 `OVERFLOW`)

**链路判据:通过。** 两条损失曲线都单调下降后走平,无发散、无跳步:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| r0 `loss_vtc`(softmax InfoNCE) | 1.014 | 0.568 | 0.441 | 0.382 | 0.341 | 0.296 | 0.297 | 0.271 | 0.274 | 0.272 |
| r1 `loss_vtc`(**sigmoid**) | 7.093 | 6.208 | 5.486 | 4.940 | 4.569 | 4.371 | 4.262 | 4.197 | 4.175 | 4.165 |

> ⚠️ **两臂的 loss 数值不可互相比较** —— r0 是 softmax InfoNCE、r1 是 sigmoid,
> 量纲和零点都不同。只能各看各的形态。

**组件存活:通过**(直接可从 `train.log` 读出,无需开 ckpt):
`diff_names: ['logit_scale', 'logit_bias'], diff_lr: 0.001` 加
`optimizer -- lr=0.001 wd=0.2 len(p)=2` ⇒ different_lr 生效;
`vision_encoder.temporal_pool.{q,k,v,proj}` 出现在优化器参数表 ⇒ 注意力池化已建起并可训;
`train_file` 带 `caption_idx: True` ⇒ 多正例开关已挂上(但见上文,这批 10k 上是空操作)。

**held-out 检索(`ret_test/`,DSL 口径 r_mean):**

| epoch | 0 | 1 | 2 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|---|---|
| r0 | 92.00 | 92.32 | 92.77 | 93.12 | 93.28 | 93.27 | **93.37** |
| r1 | 89.75 | 85.42 | 85.07 | 85.68 | 86.62 | 86.77 | **86.83** |

末轮 r0 t2v/v2t R@1 = **83.8 / 83.3**,r1 = **77.2 / 71.9**;raw(`_emb/`)口径同向
(⚠️ 2026-07-22 更正:此前把这两个标签写反了 —— `img_r1` 才是 t2v、`txt_r1` 是 v2t,依据见 §3.4)
(93.17 vs 86.50)。**r1 落后 6.5 个 r_mean 点**,远超 ~3.5pp 的噪声带。

**但这不能读成「这些组件有害」。** 三条理由:

1. **判据本来就说了这个评测判不了组件优劣**(见上一节:开训前已近饱和 + BN 混淆)。
   预注册口径是「不外推」,这里照办。
2. **r1 是四个改动的捆绑包**,试点无法归因。要拆只能用 §4 的
   `r1a_sigmoid.py` / `r1b_attnpool.py` 分开跑。
3. **r1 还没收敛**:ep2 起单调回升(85.07 → 86.83)且末轮仍在涨,而 r0 已在 93.3 走平。

**一条线索(假说,未验证):**
注意力池化**不是**元凶 —— `TemporalAttentionPool` 的 `proj` 是**零初始化**的
(`models/mobileviclip_small.py:1351`),`forward` 返回 `mean + proj(out)`,
所以初始时刻它**精确等价于 mean 池化**,不存在「随机新模块拖后腿」的开局。
更可能是 **sigmoid 损失的 SigLIP 默认初值对一个已对齐的模型失配**:
`t=exp(logit_scale)=10`、`b=logit_bias=−10`,正样本 sim≈0.5 时
`t·sim+b ≈ −5` ⇒ `sigmoid ≈ 0.007`,等于「所有正样本都判错」,开局 loss 7.45
(与日志首步 7.4531 吻合),于是对已经对齐好的嵌入空间施加一记很大的推力;
随 `logit_scale`/`logit_bias`(lr=1e-3)重新标定,检索再慢慢爬回来。
**要验证**:r1a 单开 sigmoid,或把 `logit_bias` 初值调到与预训练相似度分布匹配。

**结论:§2b 完成。** 试点该回答的问题(链路能不能跑、组件是否挂上、要多久)都有了
答案;组件优劣留给 §3/§4 的全量实验。

> **已测(2026-07-21):这批 10k 上共享字幕 = 1 组 / 2 条 clip(0.0%)。**
> 直接后果:r1 里的 **`caption_idx` 多正例这一项在试点上是空操作** —— 全库就
> 一对同字幕的 clip,它俩落进同一个 batch(512)的概率单个 epoch 只有约 5.7%,
> 即便撞上也只修正 512×511 个 pair 里的 1 个。所以试点 A/B 的差异只反映
> sigmoid 损失 + 注意力池化 + t/b 大学习率这三项,**别把 r1 - r0 的差异记到
> 多正例头上**。
>
> **但这个 0.0% 不是「全量重复率低」的证据,它只是抽样的必然结果。** 10k 是
> 10M 的 0.1% 抽样;一组 k=2 的重复字幕要两条**都**被抽中才看得见,概率
> = 0.001² = 1e-6。就算全量有 5% 的 clip(50 万条)处在重复对里,这 10k 里
> 的期望命中也只有 25 万 × 1e-6 ≈ 0.5 组 —— 和实测的 1 组无法区分。反过来说,
> 这个抽样只对**全局重复上千次**的模板句有检出力(k=1000 时 P(≥2)≈26%),
> 对 k=2…100 的中低频重复完全瞎。
> **要判定多正例值不值得做,只能看全量:`profile_jsonl.py` 的
> `duplicate_zh_caption_rate`**(决策门见 `recaption/docs/data-issues.md`:
> 重复率 >10% → 假负例治理升为必做项)。

## 3. B0 — 锁定基线数字(官方权重,双口径)

> 🚩 **2026-07-22 实测:这一步在 101vector 上跑不了 —— 5 个评测全崩,一个数字都没出。**
> 不是配置问题,是**本机没有下游评测数据**。`recaption/docs/setup-v100.md` 开篇
> 就写明该机只有 `/path/to/user/datasets/internvid_zh_10k` 这 1 万条预抽样,
> **没有全量数据**。崩溃日志与两类失败的区分见
> [`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md) 的
> 「§3 B0 五个评测全部失败 —— 本机没有下游评测数据」。

### 3.0 先跑预检(10 秒,别跳)

> ⚠️ **这台机器上两批数据不在同一个根下,预检必须跑两遍**,只跑一遍必然误报
> (2026-07-22 就这么把一份**已经下好**的 HMDB51 报成了缺失):
>
> | 数据 | 落在哪 | 怎么让预检看到 |
> |---|---|---|
> | **评测数据集**(msrvtt / hmdb51 / ucf101 …) | 仓库的 `data/` —— 各下载脚本写死的相对路径,如 `tools/download_hmdb51.sh` 的 `DEST_DIR="data/hmdb51/videos_test"` | **`unset MVC_DATA_DIR`** |
> | **10k 试点语料** | `/path/to/user/datasets/internvid_zh_10k` | `export MVC_DATA_DIR=/path/to/user/datasets` |

```bash
cd "$(git rev-parse --show-toplevel)"

# ① 评测数据集(B0 用的)—— 不要 export
unset MVC_DATA_DIR
python experiments/check_readiness.py

# ② 试点语料(§2b 用的)—— 要 export
MVC_DATA_DIR=/path/to/user/datasets python experiments/check_readiness.py
```

它逐个数据集报「anno 在不在、data_root 在不在、抽查的视频在不在」,末尾直接列出
**当前已解锁的 runbook 步骤**。B0 和 §4 的长跑之前都先跑它 —— 这次就是省掉的
10 秒预检换来了 5 次崩溃。

> 若某个数据集**你记得下过、预检却报 `data_root 缺失`**,先别重下 ——
> 见 `TROUBLESHOOTING.md`「下过的数据集被预检报成缺失」,那里有定位命令。

### 3.1 B0 的现实范围(按 2026-07-22 的机器状态)

> ✅ **更正(2026-07-22 晚,全盘搜索实测):101vector 上这四个基准的视频
> 早就下好了,一个字节都不用再下。** 本节此前两版都说"要下 ~15 GB",是错的 ——
> 错因是预检带着 `MVC_DATA_DIR=/path/to/user/datasets` 跑,而数据全在
> **`~/mobileclip/data/`**(即 `/path/to/user/rby/mobileclip/data/`)。

盘上实测(2026-07-22 全盘 `find` 聚合):

| 基准 | anno | 盘上视频目录 | 数量 | 与 `configs/data.py` |
|---|---|---|---|---|
| `hmdb51` | ✅ 1511 | `data/hmdb51/videos_test` | 6754 | ✅ 路径完全一致 |
| `ucf101` | ✅ 3783 | `data/ucf101/videos` | 13320 | ✅ 路径完全一致 |
| `k400` | ✅ 10117 | `data/kinetics_400_val_10s_320p` | 10119 | ✅ 路径完全一致 |
| `msrvtt` | ✅ 1000 | `data/MSRVTT/videos/all` | 10004 | ⚠️ 需软链成 `data/MSRVTT_Videos` |
| `didemo` / `anet` / `ssv2_mc` | ❌ 缺 | — | — | 近期不做(anno 在受限 Drive) |

另有全量 `data/kinetics-dataset/k400/{train,val,test}`(train 约 24 万条)与
`data/hmdb51_hf/hmdb51.zip`(2117581749 字节,与下载脚本的期望值一致)。

> 📖 **读 `find` 输出时当心重复计数。** 若命令写成 `find ~ /path/to/user ...`,
> 而 `~` = `/path/to/user/rby` 正是 `/path/to/user` 的子目录,**同一棵树会被遍历两遍**,
> 命中行重复、计数翻倍。上表已折半。

anno 的补法见 `TROUBLESHOOTING.md`「Missing `anno_downstream/*.json`」与
「Running the next retrieval dataset (DiDeMo, ActivityNet)」两节,这里不重复。

⇒ **B0 = `msrvtt` + `hmdb51` + `ucf101` + `k400` 四个基准,零下载**,
覆盖检索 + 零样本分类。其余三个近期不做。

> ⚠️ **只有 MSR-VTT 有可对照的外部数字**(t2v R@1≈42.5)。HMDB51 / UCF101 在本仓库
> 里没有公开的 small 变体参考值,所以它们是「锁我们自己的基线」——
> 能支撑 A/B,但**接线错误不会被外部数字兜住**。先把 MSR-VTT 对上,再信另外两个。

### 3.2 让路径对上(101vector 上**已经全绿**,零操作)

> ✅ **2026-07-22 实测复核:`unset MVC_DATA_DIR` 后四个基准直接全部 `[OK ]`**,
> 连 MSR-VTT 的软链都不用建 —— `data/MSRVTT_Videos` 本来就存在(它是**软链**,
> 所以上一轮那次 `find` 没跟进去,才没出现在按目录聚合的结果里)。
> 下面的软链命令用了 `||`,已就位时是空操作,照跑无害。
>
> ```
> [OK ] msrvtt   anno 1000 条;抽查 20 条,缺失 0
> [OK ] k400     anno 10117 条;抽查 20 条,缺失 0
> [OK ] ucf101   anno 3783 条;抽查 20 条,缺失 0
> [OK ] hmdb51   anno 1511 条;抽查 20 条,缺失 0
> ```

**统一规则:B0 全程 `unset MVC_DATA_DIR`。** 评测数据都在仓库默认的 `data/` 下,
而 `/path/to/user/datasets` 是 10k 试点的地盘 —— 分开放,互不干扰。
(`configs/data.py` 只是按 `MVC_DATA_DIR` 拼路径,不导出就落回 `data/`。)

**101vector 上的全部操作就这几行**(其余三个基准的目录名已经和 config 一致):

```bash
cd "$(git rev-parse --show-toplevel)" && unset MVC_DATA_DIR

# MSR-VTT:盘上是 data/MSRVTT/videos/all,config 要的是 data/MSRVTT_Videos
ls data/MSRVTT/videos/all | head -3          # 确认是 videoXXXX.mp4 这种裸文件名
ls -ld data/MSRVTT_Videos 2>/dev/null || \
  ln -s "$(pwd)/data/MSRVTT/videos/all" "$(pwd)/data/MSRVTT_Videos"

python experiments/check_readiness.py --sample 50 --decode   # 四行应全部 [OK ]
```

> 盘上还有个 `data/MSRVTT/MSRVTT/videos/all`(同样 10004 个),疑似解压嵌套多出来
> 的一份副本。**先别删** —— 确认软链指向的那份能正常解码后,再回收另一份的空间。

<details>
<summary>如果换了机器、数据真的没有:从公开源下载(约 15 GB)</summary>

**① HMDB51(~2.1 GB,一条命令,先做这个)** —— 脚本已内置 hf-mirror 的坑的解法
(见 `TROUBLESHOOTING.md`「HMDB51 download」),幂等,可重复跑:

```bash
bash tools/download_hmdb51.sh
```

**② MSR-VTT(~6.7 GB,主指标)** —— Frozen-in-Time 公开镜像,含全部 10000 个视频:

```bash
mkdir -p data && cd data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip -q MSRVTT.zip
ln -s "$(pwd)/MSRVTT/videos/all" "$(pwd)/MSRVTT_Videos"
cd .. && ls data/MSRVTT_Videos/ | wc -l         # 期望 10000
```

**③ UCF101(~6.5 GB)** —— CRCV 官网整包,解包后需平铺到 `data/ucf101/videos/`,
详细步骤见 `README_zh.md` §6.2(含 `.rar` 解包与平铺命令)。

**下载完统一验一次**(就是预检脚本,只是这次该亮绿灯了):

```bash
unset MVC_DATA_DIR
python experiments/check_readiness.py --sample 50 --decode
# 期望:msrvtt / hmdb51 / ucf101 三行都是 [OK ],且「试解码 OK(... 帧)」
```

> 若 `data/` 所在分区放不下,可把视频放别处再
> `export MVC_DATA_DIR=<那个目录的父目录>`,只要最终路径能拼成
> `$MVC_DATA_DIR/MSRVTT_Videos/videoXXXX.mp4` 等即可 —— 但那样 10k 试点的
> corpus 路径也会跟着变,**别在同一个 shell 里混用**。

</details>

### 3.3 跑 B0 并汇总

`run_eval.sh` 的 benchmark 名和 `check_readiness.py` 的完全对齐,直接循环即可
(哪个数据集预检没亮 `[OK ]` 就把它从列表里删掉,别让它崩在中间):

```bash
cd "$(git rev-parse --show-toplevel)" && unset MVC_DATA_DIR
CKPT=./checkpoints/mobileviclip_small.pt
for b in msrvtt hmdb51 ucf101 k400; do
  CUDA_VISIBLE_DEVICES=0 \
    bash experiments/run_eval.sh small $b $CKPT experiments/evals/baseline_official/$b
done

# 只扫 evals/,别把 runs/ 里的试点数字混进 BASELINE.md(见下面的坑)
python experiments/collect_results.py \
    --dirs experiments/evals --baseline baseline_official \
    --out experiments/BASELINE.md
```

> **`CUDA_VISIBLE_DEVICES` 要自己挑。** `run_eval.sh` 默认 `0`,而这台卡是共享的
> (§2b 的 OOM 就是这么来的)。先 `nvidia-smi` 看哪张空着。

核对 `ret_test/`(DSL 口径)是否对上论文/README 数字(MSR-VTT t2v R@1≈42.5);
`ret_test_emb/` 是原始余弦口径。**此后所有 A/B 结论要求两个口径同向成立。**

### 3.4 B0 基线(2026-07-22 实测锁定,官方权重 `mobileviclip_small.pt`)

**先读懂列名,否则会把分类数字读高 20 个点:**

| 任务 | 该看哪一列 | 含义 |
|---|---|---|
| 检索(msrvtt) | `img_r1` = **t2v**;`txt_r1` = **v2t** | `itm_eval()` 里 `txt_*` 是 Images→Text(逐视频找文本),`img_*` 是 Text→Images(逐文本找视频) |
| 分类(ucf101/hmdb51/k400) | `txt_r1` = **零样本 Top-1** | 逐**视频**给类别排序,分母 = 视频数 |
| 分类 | ~~`img_r1`~~ **不是准确率** | 逐**类别**给视频排序,分母 = **类别数**,详见下方 ⚠️ |

**锁定的基线数字**(`act_val/` / `ret_test/` = DSL 口径;`_emb/` = 原始余弦):

| 基准 | 指标 | DSL | raw(`_emb/`) | 外部对照 |
|---|---|---|---|---|
| `msrvtt` | **t2v R@1**(`img_r1`) | **42.50** | 40.80 | 论文 ≈**42.5** ✅ 完全对上 |
| `msrvtt` | v2t R@1(`txt_r1`) | 43.30 | 42.50 | — |
| `ucf101` | **Top-1**(`txt_r1`) | **80.97** | 80.94 | 无(Tiny 为 75.44) |
| `hmdb51` | **Top-1**(`txt_r1`) | **53.47** | 53.87 | 无(Tiny 为 48.76) |
| `k400` | **Top-1**(`txt_r1`) | **63.13** | 63.49 | 无(Tiny 为 56.25) |

⇒ **整条评测链路验证通过**:唯一有外部数字可对照的 MSR-VTT t2v R@1 打出 **42.50**,与论文的 ≈42.5
分毫不差 —— 权重、解码、划分、prompt 全部正确。另外三个基准 Small 在 `txt_r1` 上**全面高于 Tiny**
(+5.5 / +4.7 / +6.9),方向自洽。

> ⚠️ **`img_r1` 在分类任务上不是 Top-1 准确率。** `README_zh.md` §6.2–6.4 此前四处这么写,已更正。
> 证据链:① `itm_eval()`(`tasks_clip/retrieval_utils.py:463`)里两个方向的分母不同;
> ② `dataset/ret_dataset.py:295-296` —— `img2txt` 把视频映到类标签,`txt2img` 长度 = 类别数;
> ③ `retrieval_utils.py:125` 把 29 个 prompt 集成掉,打分矩阵是 视频 × 类别;
> ④ **算术自检**:实测 6 个 `img_r1`(Small 三个 + README_zh 里 Tiny 三个)**全部恰好是
> 1/类别数 的整数倍** —— 90.10=91/101、60.78=31/51、83.00=332/400、83.17=84/101、
> 62.75=32/51、80.75=323/400;⑤ 若按 `img_r1` 读,HMDB51 的 Small(60.78)会**低于** Tiny(62.75),
> 而按 `txt_r1` 读则是 53.47 > 48.76,方向才对。
>
> 顺带:K400 若按 `img_r1` 读会得到 **83.00**,而真实零样本 Top-1 是 **63.13** —— 差 20 个点,
> 且 83% 的零样本 K400 会离谱地超过当时的 SOTA。这是"数字看着太好"该起疑的典型。

> 📌 **分类任务上 DSL 反而略低**:hmdb51 53.47(DSL) vs 53.87(raw)、k400 63.13 vs 63.49、
> ucf101 80.97 vs 80.94(基本持平)。DSL 用的是**测试集先验**,对检索有利,对分类并非天然合适。
> 这不改判据 —— **A/B 仍要求两个口径同向成立**,只是解读分类数字时心里有数。

> ⚠️ **BASELINE.md 会骗人。** `collect_results.py` 默认同时扫 `experiments/runs`
> 和 `experiments/evals`。2026-07-22 那次 5 个评测全崩、`evals/` 下一个
> `eval_res.json` 都没有,它照样写出了一个名叫 `BASELINE.md`、内容却是
> **10k 试点 A/B 数字**(`pilot_10k_r0` / `pilot_10k_r1`,r_mean 93.37 / 86.83)
> 的文件,全程零报错。那些是自检索数字,**不是** MSR-VTT 基线。
> 现在生成的文件顶部会写明扫了哪些目录、含哪些 run —— 读之前先看那两行。

> **官方权重不含文本塔主体(2026-07-22 从加载日志读出)。**
> 载入 `checkpoints/mobileviclip_small.pt` 时 `missing_keys` 恰好是整个
> `text_encoder.{embedding_layer, positional_embedding, transformer.*,
> final_layer_norm}`,而 `text_encoder.projection_layer` **在**权重里、
> 视觉塔与 `temp` 也都在。文本塔主体是建模时从
> `checkpoints/mobileclip_s2.pt` 载入的(上游 `freeze_text=True`、只训投影,
> 自洽)。**含义:B0 的数字由这两个权重文件共同决定**,复现基线时
> `mobileclip_s2.pt` 也必须是对的那一份。

## 4. R0 → R1 — 训练 A/B(等步数、等数据、等超参)

> 📍 **在 b100 主机（ZTE 集群，8×H100，全量 10M）上从零把这条线跑起来** —— 联网(反向隧道+代理)、
> Hopper/sm_90 的 torch 构建（默认只用前 4 张卡）、`/workspace/mount/.../InternVid-10M-FLT` 的探盘+软链接线、
> 预检、启动 —— 见 [`SETUP_B100.md`](SETUP_B100.md)。本节只讲 R0/R1 *是什么*;那份手册讲
> 在那台机器上*怎么搭起来*。与 §2b 的 V100 试点并行、互不阻塞。

> 🚩 **跑之前先修这两处,否则要白烧几个小时**(2026-07-22 从试点日志溯源发现,
> 尚未验证修法,请先用下面的 1 分钟预检确认)。§2b 的 pilot 配置**不受影响**
> —— 它们各自覆盖了 `test_file` / `test_types`,所以试点评测正常出表;而
> `r0_baseline.py` / `r1*.py` 只改了 `output_dir`,全盘继承 base config:
>
> 1. **训练中的评测会被静默跳过。** base config 里
>    `test_file = dict(act_val=...)` 但 `test_types = ["ret_test"]`。
>    数据集名取自 `test_file` 的**键**(`dataset/__init__.py:283`),
>    `"act_val" not in ["ret_test"]` ⇒ `tasks_clip/pretrain.py:315` 直接
>    `continue`,日志里只留一行 `Skip eval act_val split`。
>    预检:`bash experiments/run_pretrain.sh experiments/configs/r0_baseline.py 8 debug True`
>    跑一分钟,`grep "Skip eval" experiments/runs/r0_baseline/train.log` ——
>    有输出就是中招了。修法是把键名对齐(`test_file = dict(ret_test=...)`)。
> 2. **`ckpt_best.pth` 不会存在。** base config `stop_key = None`
>    ⇒ `tasks_clip/pretrain.py:332` 的 best 分支永不进入。实际产物是
>    **每个 epoch 一个 DeepSpeed 目录** `ckpt_{epoch:02d}.pth/`,权重在
>    里面的 `mp_rank_00_model_states.pt`(参见 `tasks/pretrain.py:521`)。
>    所以下面那段 `ckpt_best.pth` 的循环跑不通,得改成最后一个 epoch 的目录,
>    并确认 `run_eval.sh` 能吃 DeepSpeed 目录(否则先用
>    `zero_to_fp32.py` 转成单文件)。

```bash
# R0 复现基线(与原配方逐项一致)
bash experiments/run_pretrain.sh experiments/configs/r0_baseline.py 8

# R1a sigmoid 损失   / R1b 注意力池化   / R1 两者叠加
bash experiments/run_pretrain.sh experiments/configs/r1a_sigmoid.py 8
bash experiments/run_pretrain.sh experiments/configs/r1b_attnpool.py 8
bash experiments/run_pretrain.sh experiments/configs/r1_both.py 8
```

训练中每个 epoch 会自动跑 MSR-VTT 评测(结果在各自 output_dir 的
`eval_res_latest.json`)。训完再对每个 ckpt 跑全量基准:

```bash
for run in r0_baseline r1a_sigmoid r1b_attnpool r1_both; do
  for b in msrvtt didemo anet k400 ssv2_mc; do
    bash experiments/run_eval.sh small $b \
         experiments/runs/$run/ckpt_best.pth \
         experiments/evals/$run/$b \
         model.use_sigmoid_loss $( [[ $run == *sigmoid* || $run == *both* ]] && echo True || echo False ) \
         model.vision_encoder.temporal_pooling $( [[ $run == *attnpool* || $run == *both* ]] && echo attn || echo mean )
  done
done
```

> 注意:评测**带注意力池化**的 ckpt 时必须传
> `model.vision_encoder.temporal_pooling attn`,否则模型构建时没有该模块,
> `strict=False` 加载会静默丢掉它的权重、数字退回 mean 池化。

## 5. 汇总对比

```bash
python experiments/collect_results.py --baseline r0_baseline --out experiments/RESULTS.md
# 与官方权重对照:--baseline baseline_official
```

## 6. 判据(07-20 审核后加严)

- **保留组件**:等步数下相对 R0 的 **多基准聚合 R-Sum**(MSR-VTT + DiDeMo +
  ANet)提升同向,且 DSL 与 raw 两口径同向;单基准 +0.3 R-Sum 量级低于
  种子噪声,**临界情况(聚合优势 < 0.5)必须补第二种子**再下结论;
- 论文**主口径 = raw 余弦**(`_emb/`),DSL 仅用于与原论文数字对齐;
- R1 叠加若低于单项最优 → 取单项最优作为后续 DR 实验(R2/R3)的底座;
- R0 若无法逼近官方权重的量级 → 先排查复现问题,不进入组件结论;
- R0 启动第一天记录**单 epoch wall-clock**,据此校准论文线的
  Wk4–6 排期(并实测 `gradient_checkpointing False` 能否放下、省多少时间)。

### 6.1 MSR-VTT 1K 上的噪声带:**±1.6 是下界,不是阈值**

多处文档在用 `±1.6 R@1` 作为 MSR-VTT 1K 的判据带宽。它来自二项抽样误差
—— p≈0.42 时 `sqrt(p(1-p)/1000) ≈ 1.6`。**这个数偏松,别当阈值用。**

**实测证据**(1.8 选帧基线,2026-07-27,[`frame_select_results.md`](frame_select_results.md) §2.4):
`nativeT2` 与 `seluniform_K2` 两条臂采样时刻几乎重合 —— 前者取视频的
{0.250, 0.750},后者取 16 帧母序列的第 4、12 帧即 {0.281, 0.781},
**只差 3.1% 时长**。两者却相差 **raw +2.40 / DSL +1.40**。

内容近乎相同的一对臂能拉开 2.4 个 raw R@1 ⇒ 二项公式假设 1000 条查询**相互独立**,
而检索错误在查询间**高度相关**,真实波动大于该公式给出的值。

**使用规则:**

| 结论方向 | 1.6 带的作用 | 该怎么办 |
|---|---|---|
| **阴性**(「没有效应」) | 保守 —— 真带更宽只会让阴性更稳 | 照用,结论有效 |
| **阳性**(「有增益」) | **偏松** —— 可能把噪声读成效应 | 必须两口径同向,且优先要求 \|Δ\| 明显大于 2.4 |

> ⚠️ **2.4 的适用面(2026-07-29 收窄,见 评审 §6.1.1)。**
> 这条 2.4 来自 `nativeT2` vs `seluniform_K2` —— 两条臂**抽样时刻不同**(输入本就不同)、
> 且是**绝对值**比较。它量的是「**模型对无关输入扰动的敏感度** + 查询间相关性」。
> ⇒ **适用于非配对 / 跨抽样的阳性主张;不适用于配对的同 T 比较。**
> 当两条臂在**同一个 T、同一套抽帧、同一批 1000 条查询**上比较时(R3 的每-T 非劣、
> R1 A/B、1M 层同 seed 相减都属此类),抽样 nuisance 与查询抽样噪声**都已因设计抵消**,
> 应改用**配对 bootstrap CI**(`collect_results.py --compare`,`COMMANDS.md` §3.9)判
> 「CI 是否含 0」;套 2.4 会**过度保守**,把本来可测的主张判成不可测。
> 此时**剩下的地板是训练种子方差** —— 它**不因配对抵消**,且尚未实测;
> 1M 层「4 臂 × 2 seed」的**同臂双种子之差**就是它。

**这不推翻任何已有结论**:GATE-0 判 `reverse` 的 −1.2 为噪声,带更宽只会更成立;
`freeze` 的 −13 远超任何合理带宽。受影响的是**未来的阳性主张**。

**两口径同向那条规则是有效的**:1.8 里 `seluniform_K2` 的 +2.40(raw)在 DSL 上
只有 +1.40,规则正确地拒绝了它 —— 单看 raw 会误报一个 37% 的"缺口补回"。

`r_mean` 是 6 个高度相关召回的平均,**没有干净的闭式 SE**,一律只当描述性数字,
不要给它套 R@1 的阈值。
**(2026-07-29 部分解除:该说法针对_解析公式_;配对 bootstrap 给了 `r_mean` **经验 CI**,
可以用。但主表仍建议用 `R-Sum`,与既有判据同口径。)**

## 7. 后续(本框架预留、尚未实现)

- **R2**(+ 视频 DR 蒸馏损失,C1)与 **R3**(+ 随机 T 弹性训练,C2)
  需要新增教师 embedding 管线与 T 无关化改造,作为下一批 PR 进入本目录
  (届时新增 `r2_dr.py` / `r3_elastic.py` 配置,复用同一套 run/collect 脚本);
- 评测按 rank 分片(H5)在扩展到 ActivityNet/DiDeMo 大规模评测前加入。

## 故障排查

### 启动即崩在 `import deepspeed`:pydantic `root_validator` 报错,或 `cannot import name 'log' from torch.distributed.elastic`

老版 DeepSpeed 在 torch 2.x 环境里有**两重不兼容**(101vector 2026-07-21
两种都踩了):pydantic ≥ 2 炸 `@root_validator`;torch ≥ 2.0 移除了它要
import 的 `log`。**`pip install "pydantic<2"` 只修第一重**,修完会接着死在
第二重——torch 2.x 环境里老 deepspeed 就是跑不起来。完整分析见
`TROUBLESHOOTING.md` 对应小节。按序处理:

```bash
# 第 0 步:先 git pull!PR #84 起 import 已有防护(deepspeed 坏了也不炸);
#          堆栈若指向 shared_utils.py:8 的裸 import deepspeed = 还在跑旧代码。

# 正解(torch 2.x 环境):升级 deepspeed(需要 pydantic>=2,降过的升回来)
pip install -U pydantic
DS_BUILD_OPS=0 pip install -U "deepspeed>=0.14"
python -c "import deepspeed; print(deepspeed.__version__)"   # 不报错才算好

# 或绕开(试点/冒烟专用,DDP+AMP 路径,完全不碰 deepspeed):
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 deepspeed.enable False
```

注意:**正式 R0–R3 复现必须回到 DeepSpeed 路径**(原配方是 DeepSpeed
stage-1 + fp16),绕开只用于试点和链路调通;A/B 双方用同一路径即可公平。
按 `mobileviclip.yml` 装的原版环境(torch 1.13.1)两重都不会踩。

其余日志噪音无害:pydantic 的 `Valid config keys have changed`、torch 的
`custom_fwd deprecated`、`FusedMLP/DropoutAddRMSNorm of flash_attn is not
installed`(P1 里那些被忽略的旗标)都可以无视。

## 目录结构

```
experiments/
├── README.md            # 本文件
├── configs/             # 实验配置(继承 scripts/pretraining 基础配置,只写差异)
│   ├── r0_baseline.py
│   ├── r1a_sigmoid.py
│   ├── r1b_attnpool.py
│   └── r1_both.py
├── run_pretrain.sh      # 训练入口(torchrun 包装,支持 CLI 覆盖)
├── run_eval.sh          # 评测入口(任意 ckpt × 任意基准 × 指定输出目录)
├── collect_results.py   # 结果汇总 -> markdown 对比表(双口径、Δ 列)
├── runs/                # 训练产出(gitignore)
└── evals/               # 评测产出(gitignore)
```
