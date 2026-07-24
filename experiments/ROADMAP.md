# ROADMAP — 当前该做什么(总入口,随进度勾选更新)

**日期:** 2026-07-24(**实时进展看文末「进展日志」+「状态速览」**,每个会话更新)
**分工:** 战略层(为什么/判据)看 `notes/paper-plan-zh.md`;每条命令看
`experiments/README.md`;本文件回答"现在轮到哪一步"。
**第 0 步(任何时候)**:`python experiments/check_readiness.py`
——自动报告哪些测试集/权重已到位、下面哪些步骤已解锁。

三条线并行,互不阻塞;主时钟由第三线(全量数据)决定。

---

## 最终优化方案(bullet 速览 — 详见 `notes/paper-plan-zh.md` §一–三)

一句话故事:**数据强化(DR)训练 + 时间维弹性蒸馏,让移动端视频 CLIP 用单一
checkpoint 在任意帧预算 T∈{2…16} 下逼近大模型,推理零额外开销。** 贡献分四层:

- **C1 · 视频数据强化(主线,预期收益最大)** — InternVideo2 教师离线过一遍
  InternVid-10M-FLT,存视频+文本 embedding(真实字幕 + CapRL 合成重字幕);
  学生用 `VTC + embedding 蒸馏 + 相似度矩阵蒸馏` 训练。图像侧 DR 无视频版先例。
- **C2 · 时间维弹性蒸馏(新颖性核心)** — 时序算子 T 无关化 + 学生随机
  T∈{2,4,8,16} 训练,全部对齐同一份固定 16 帧教师目标 ⇒ 帧数即部署旋钮的
  单一模型;目标是每个预算点的 Pareto 曲线都胜过固定 T 基线。
- **C3 · 配方组件(出消融表)** — ① sigmoid 多正例损失(可学习 t/b,SigLIP 初始化,
  `caption_idx` 掩码解决共享字幕假负例);② 时间维注意力池化头(**零初始化,
  step-0 严格等价 mean,不退化**)。两者默认关闭,基线路径与原 repo 逐位一致。
- **C4 · 结果** — 多帧数零样本检索(MSR-VTT/DiDeMo/ANet)+ 零样本分类/多选
  (K400/UCF101/HMDB51/SSv2-MC)+ **端侧延迟表**,对比 MobileViCLIP /
  InternVideo2-S/B / **FluxViT-S(GFLOPs 对齐行)**;TempMe 属微调协议,
  只进单独的微调/延迟-only 对比区(07-23 评审 §3.5)。
- **卫生闸门(任何基线数字之前强制)** — 移 `set_detect_anomaly`、清 spikingjelly
  死依赖、修 `hasattr(name,'_1d')` 冻结 bug、DSL/raw 两口径分开报并钉死主口径。
- **条件项** — R4(P7 非对称帧编码,双触发才回收)、P9(导出阻塞时的分离注意力变体)。
- **⚠️ 07-23 计划评审(`notes/plan-review-upgrades.md`,U1–U10)** — C2 新颖性按
  FluxViT(ICCV'25)收窄为「帧数旋钮×移动端骨干×固定 T 锚点×端侧」组合格;导出
  冒烟提前(下方 1.5);C2 增 per-T BN 校准;R0 全局 batch 对齐 512;预算阶梯;
  统计口径三件套。paper-plan §七 为计划层准绳。

> ⚙️ **验证分两台机(别把职责搞混)**:V100/10k 只做**链路 go/no-go**(能跑通、
> loss 正常降、新参数收到梯度),**排不了方案序**(评测饱和 + BN 混淆 + 二项 SE);
> H100/全量 10M 才做**真 A/B 排序 + 胜者终训**。详见 `README.md`「两级实验」。

---

## 每一步的命令写在哪(文件 × 小节速查)

| 步骤 | 命令所在 md → 小节 |
|---|---|
| 第 0 步 就绪自检 | 本文件开头(`python experiments/check_readiness.py`);B0 前预检 = `README.md` §3.0 |
| 1.1 下载(测试集/权重) | `tools/data_audit/SETUP_V100.md` §0(环境 + 权重后台下载)、§0.5;链接大全 `CHECKPOINTS.md`(Pretrained weights / Google Drive 两节);InternVid 分片抓取 `docs/internvid-download-zh.md` §3–§4;通用下载 `COMMANDS.md` §2 |
| 1.2 数据审查流水线 | `tools/data_audit/SETUP_V100.md`:§1 冒烟 → §2 S0 → §3 S1(§3.5/3.6 体检)→ §4 S2(4.2 主力 / 4.3 裁判 / 4.5 体检)→ §5 S3 报告;§6.5 一条命令看全状态 |
| 1.3 10k 试点(✅ 已关闭) | `experiments/README.md` §2b 全套(anno 生成、`export MVC_DATA_DIR`、r0/r1、pytest);结论见 `pilot_10k_results.md`(#199) |
| 1.4 B0 锁基线 | `experiments/README.md` §3.0 预检 → §3.2 路径对上 → §3.3 跑 B0 + `collect_results.py`;已锁数字在 §3.4 |
| 1.5 导出冒烟(U1) | 任务与验收:`notes/plan-review-upgrades.md` U1。**具体脚本待写**——落地时命令按约定补进 `COMMANDS.md` |
| 1.6 B0 多 T 曲线(U2) | 任务与验收:`notes/plan-review-upgrades.md` U2;2.1 已落地 ⇒ 用既有零样本 eval 脚本对官方权重覆盖 `num_frames_test <T>`(T∈{2,4,8,16})即可,模型侧自动插值 |
| 2.1 C2 T 无关化(✅ 代码落地) | 单测 `python -m pytest tests/test_t_agnostic.py -v`(README §2);V100 冒烟命令见本文件 2.1;per-T BN 工具 `experiments/per_t_bn.py`;设计准绳 `paper-plan-zh.md` §七.3 |
| 2.2 R2 教师管线 | 设计决策清单:本文件 2.2 + `paper-plan-zh.md` §六.9 + 评审 U7;**提取脚本待写** |
| 2.4/2.5 A1/A2/B1(#197) | `notes/neural-inspired-implementation.md` Tier A(配置模板 + 单测)/ Tier B(B1 接线与三臂);配置文件待创建 |
| 3.1 H100 全量数据审查 | `tools/data_audit/SETUP_H100.md` §0 环境 → §1 冒烟 → §2 S0 → §3 S1 → §4 S2 → §5 S3 |
| 3.2 R0→R1 全量起跑 | `experiments/SETUP_B100.md`(**文件顶部有进度标记清单,当前位置=§6 冒烟→R0**,§0–§5 已全绿):**§6 冒烟 / R0 / R1 启动命令**已随 `from-claude.md`「07-24 (5) §5 全绿」交接 → §7 两个坑 → §8 交回;预算阶梯 = 评审 U6 |
| 汇总与判据 | `experiments/README.md` §5 汇总对比、§6 判据 |
| 环境/评测/git 备查 | `COMMANDS.md` §1 环境、§3 零样本评测、§4 训练、§5 git、§6 handoff |

---

## 第一线:101vector 机器上执行(GPU + 人工,现在就能开跑)

- [ ] **1.1 后台挂下载**(不占 GPU,越早越好):MSR-VTT 1k 测试集、
      官方 `mobileviclip_small.pt`、底座 `mobileclip_s2.pt`。
      弱网断点续传配方:`tools/data_audit/SETUP_V100.md` §0、
      `docs/internvid-download-zh.md`。已有的测试集放好后用
      `check_readiness.py` 确认识别。
- [x] **1.2 数据审查流水线** —— ✅ 2026-07-22 完成 S0–S3(10k 自动化审查):
      正式报告 `docs/internvid-zh-audit-report.md`,台账
      `tools/data_audit/DATA_ISSUES.md`,统计细节 `notes/audit-10k-findings.md`。
      后续 s4-merged 重字幕落地走 `notes/recaption-master-plan.md`
      (v1.1,#223;V100 Stage 0 命令见 from-claude「07-24 (4)」)。
- [x] **1.3 10k 试点**(README §2b,GPU 空档)—— ✅ 2026-07-23 关闭:
      `make_pilot_anno.py`(记下共享字幕比例)→ `export MVC_DATA_DIR` →
      `pilot_10k_r0` 冒烟 + 正跑 → `pilot_10k_r1`;
      外加 `pytest tests/test_optimization_components.py`。
      **通过标准(只看动力学)**:损失正常下降;r1 的 ckpt 里
      `logit_scale/logit_bias` 离开初始值(t≠10、b≠−10);
      `temporal_pool.proj` 非零。不满足 = 配置/代码有问题,先修再前进。
      **结果见 [`experiments/pilot_10k_results.md`](pilot_10k_results.md)** —— 三条判据
      2026-07-23 已全部核实通过(ckpt_09):r0 全过;r1 损失 7.07→4.2、
      `exp(logit_scale)=10.48`、`logit_bias=−9.76`、`temporal_pool.proj` 从零练到 L2=0.18。
      **1.3 关闭**,下一步 H100 全量 A/B(`experiments/SETUP_B100.md`)。
- [ ] **1.4 B0 锁基线**(1.1 完成后,README §3):官方权重跑已到位的
      基准 → `collect_results.py --out experiments/BASELINE.md`。
      核对 MSR-VTT DSL 口径 t2v R@1 ≈ 42.5;**论文主口径 = raw(`_emb/`)**。
- [ ] **1.5 导出冒烟(U1,零训练权重,CPU 可做,立即)**:随机权重实例化
      `MobileViCLIP_Small`(带全部时序算子),分走 ONNX / TFLite / coremltools
      转换,按 T∈{2,4,8,16} 各出静态图;记录每路 pass/fail + 失败算子。
      产出决定 P9 是否提前触发(真机延迟档仍在第 8 周)。
- [x] **1.6 B0 多 T 退化曲线(U2)** —— ✅ 2026-07-24 V100 实测完成(官方
      `mobileviclip_small.pt`,MSR-VTT 1K,fp32,middle 采样;from-claude (18)):

      | T | t2v R@1 | v2t R@1 | r_mean |(ret_test_emb 口径)
      |---|--------|--------|--------|
      | 2 | 35.7 | 35.8 | 55.42 |
      | 4 | 39.6 | 39.2 | 60.03 |
      | 8 | 40.8 | 42.5 | 61.55 |
      | 16 | 42.2 | 43.1 | 62.25 |

      **锚点核验**:T=8 DSL 口径 t2v R@1=42.5 与 B0 锁定值 42.50 一致 ⇒ 2.1
      动态化对官方权重**零回归**(插值 identity 路径服务器级验证)。三个读数:
      ① 退化平滑无断崖,T=2 相对 T=8 −6.1 r_mean(−5.1 t2v R@1)= **C2 要弥合的
      gap**;② **T=16 免训练反超 T=8**(+0.7 r_mean)——嵌入插值升采样即增益,
      支持 T 无关设计;③ 8× 帧数只差 ~6.9 r_mean,弹性推理动机成立。
      原始 JSON:服务器 `experiments/evals/b0_multiT/msrvtt_T{2,4,8,16}/eval_res.json`。
      该失配曲线在视频-文本模型上未见发表(评审 §3.3)⇒ 论文 Fig-1 v1 数据到手。
- [ ] **1.7 GATE-0 时间维容量审计(零训练,现在就能跑;命令见 `COMMANDS.md` §3.1)**
      —— **任何时序类改动之前的强制闸门**,判读与后续决策见
      [`notes/neural-inspired-round2.md`](../notes/neural-inspired-round2.md) §2。
      (a) 静态:`python experiments/temporal_capacity_audit.py checkpoints/mobileviclip_small.pt`
      —— 时间卷积核反对称占比 / DC-AC 比 + `temporal_embedding` 范数,只读权重,秒级;
      (b) 动态:`MVC_FRAME_PERTURB={shuffle,reverse,freeze}` 在 **MSR-VTT + SSv2-MC**
      上各跑一遍(默认不设 = 逐位一致基线);
      (c) 评测集补 **RTime(尤其 RTime-Binary)** 与 SSv2-template —— 现有套件测不出时序
      (单帧偏置:图像模型 BLIP 的 MSR-VTT 零样本 R@1 已高于我们锁定的 42.50);
      (d) **基线换种子重跑一次全量** → 拿到噪声地板(§1.1:±1.5~3 R-Sum,现有
      "+0.3 R-Sum" 判据很可能欠功效)。
      **产出**:无论结果如何都是一张"可测性上界"论文图(与 1.6 同一格)。

## 第二线:代码开发(不依赖数据,随时推进)

- [x] **2.1 C2 T 无关化** —— ✅ 2026-07-24 代码落地(R3/R3b 前置解除):
      动态 T(帧数上下文 `set/get_num_frames`,small/tiny 共 11 处硬编码全替)、
      `temporal_embedding` 线性插值(t==8 原样返回)、随机 T 采样器
      (`utils/elastic_t.py`,(seed,step) 决定性 ⇒ 跨 rank 同 T)、训练开关
      `elastic_t=dict(enable,choices)`(`tasks_clip/pretrain.py`,默认关=零改动)、
      per-T BN 诊断/校准工具(`experiments/per_t_bn.py`,U3)。
      **验收已过**:`tests/test_t_agnostic.py` 18 项全过(本地 CPU)——含
      T=8 与旧代码**逐位一致**(MHSA/RepMixer 训练+reparam/AttentionBlock)。
      **服务器验收已过(07-24,from-claude (17))**:V100 pytest 18 项全过;
      r3e 8 卡随机 T 冒烟三判据全过——自停干净(10 epoch,best epoch 9)/
      elastic_t 四值全出现且 50 步 T 序列与本地复算**逐位一致**(均值 7.8≈理论 7.5)/
      loss 全程有限,T=16 峰值显存 23.3GB(V100 32GB 内)⇒ **C2 链路 ✅**。
      观察项:epoch3 评测一次性 `sim has nan`(fp16 评测侧上溢,非训练发散,
      见 TROUBLESHOOTING 末节);真 ckpt 的 BN 漂移诊断(per_t_bn.py)待跑。
- [ ] **2.2 R2 教师管线**(1.2 的报告出来后设计更准):InternVideo2
      embedding 离线提取脚本,先在 10k 上试点实测每 clip 耗时/存储,
      同时敲定审核修正 9 的三个设计细节(教师维度投影、帧索引记录、
      低 T KD 加权)。用 10k 成本外推 10M,决定教师用 L 还是 1B。
      **U7 三项一并定**:教师语料 curated 2–3M 先行 vs 全量 10M(等步对照);
      ~1M 子集加存 per-frame 教师 CLS(≈45GB,解锁低 T 可达性消融/时序显著性
      蒸馏/R4 复用);合成字幕 K≥2。相对 KD(`vtc_kd`/`kd_geo`)单一所有权
      并入同一消融表(U9)。
- [ ] **2.3(条件项)** H5 评测分片(大基准评测前)、P1 torch 升级
      (仅当显存/速度真卡住)。
- [ ] **2.4 R1 精确文本负例库**(`notes/neural-inspired-round2.md` §3 R1;**不依赖 GATE-0,
      期望收益最高**):文本塔冻结 ⇒ 投影前特征全程恒定 ⇒ 缓存后每步用**当前**投影重算,
      负例库**数学上精确、零陈旧**(无需动量编码器)。全局 batch 512 距 SigLIP 饱和点约
      30–60×,是目前最大的未开采杠杆。顺带一并定:**温度**(固定 1/100 偏锐,最优值随负例
      数移动)与 **sigmoid 损失**(已实现,`r1a_sigmoid.py`)。
      ⚠️ fp16 溢出坑见 `TROUBLESHOOTING.md`;**提前检查**:banked 负例占 softmax 质量 <5%
      ⇒ 库是惰性的,直接砍。
- [ ] **2.5 P1 逐帧轨迹外露 + R2 可恢复顺序的读出头**(§3 R2、§5 P1):
      `FastViT.forward:1377` 造出 `[T,B,C]`、`:1384` 只返回池化后的 `[B,C]`,损失层根本看不到
      —— 这一处一次性改动解锁 R2/R3 全家。R2 消融梯子:`mean / additive / slope / phase`,
      其中 **additive 按 F2 预测应当无增益**(这是判据本身)。
      主指标 **RTime-Binary**(时间反转不变 ⇒ 恰好 50.0%,零假设已知),MSR-VTT 作预注册的
      不退化检查。**T1 披露**:一阶项的最小二乘解 ≡ approximate rank pooling(Bilen *CVPR* 2016),
      `α_t = 84·w_t`,精确恒等,必须自己先讲。

## 第三线:全量数据(外部依赖,主时钟)

- [ ] **3.1** 10M InternVid 到位(✅ 07-24 确认:b100 本地 jsonl 即全量
      **10,647,458** 条,免下载,from-claude「07-24 (3)」)+ H100 机全量审查
      (`SETUP_H100.md`,未跑;**不阻塞 R0**,可与训练并行排)。
- [ ] **3.2** 🟡 **进行中**:§4 anno/软链 + §5 预检已全绿(11 片 `[OK ]` + 权重
      在位,#225),冒烟→R0 命令已交接(from-claude「07-24 (5) §5 全绿」)。
      等首光回报 → R0 起跑 → **首日实测单 epoch wall-clock**,据此触发
      U6 预算阶梯(超预算裁剪顺序 R0 > R2 > R3 > R3b > R1)。

## 汇合点 → 主战役(paper-plan §四)

三线汇合条件:全量数据就绪 + B0 已锁 + 10k 试点通过 + C2 代码合入。
然后按闸门推进:**R0(复现)→ R1(组件)→ R2(DR 蒸馏,预期最大收益)
→ R3(弹性 T,新颖性核心)→ R3b(对照)**;R4/P9 按触发条件
(paper-plan §四、§六)。目标 CVPR 2027(约 11 月截稿)。

## 状态速览(手动维护)

| 里程碑 | 状态 | 备注 |
|--------|------|------|
| 计划文档(P1–P10 → C1–C4 → 审核修正) | ✅ | notes/ 四份,均已 merge |
| 实验框架 + 组件开关 + 卫生修复 | ✅ | PR #64 |
| 10k 试点配置 | ✅ | PR #67 |
| 10k 试点跑通(链路 go/no-go) | ✅ | 三条判据全过(exp(ls)=10.48≠10·lb=−9.76≠−10·proj 从零练到 L2=0.18),见 `pilot_10k_results.md` |
| B0 基线锁定 | ✅ | MSR-VTT t2v R@1=42.50 对上论文,见 README §3.4(msrvtt/hmdb51/ucf101/k400) |
| C2 T 无关化 | ✅ | 代码+单测+V100 冒烟全过 07-24(#222 + r3e 三判据,T 序列服务器↔本地逐位一致,from-claude (17));真 A/B 与终训在 H100 全量;BN 漂移诊断待跑 |
| A1 惊奇池化(#197 Tier A) | 🟡 | **10k 链路 ✅ 07-24 实测结案**(#220 修复后 alpha=-4.7e-5≠0、gamma=1 在位,见 `pilot_10k_results.md` 结案节);排队 H100 1-epoch 排序臂(≥ attn 池化才保留) |
| 重字幕主计划(zh Stage 0) | 🟡 | v1.1(#223,全自动验收 Panel-V);V100 `run_v100.sh recap` 命令已交接(from-claude 07-24 (4)/(5)) |
| R2 教师管线 | ⬜ | ← 第二线 2.2(含 U7/U9 设计项) |
| 全量数据(H100 真 A/B + 终训) | 🟡 | b100 §0–§5 全绿(#225):数据=全量 10.6M 确认、11 片预检 `[OK ]`、权重在位;§6 冒烟→R0 命令已交接,等首光回报 |
| 计划评审 + 文献网扫(U1–U10) | ✅ | `notes/plan-review-upgrades.md`;C2 按 FluxViT 收窄、导出冒烟=1.5、多 T 曲线=1.6 |
| 导出冒烟(U1) | ⬜ | ← 第一线 1.5,零权重即可,决定 P9 触发 |
| B0 多 T 曲线(U2/1.6) | ✅ | 2026-07-24 实测:emb r_mean 55.4/60.0/61.6/62.3 @ T=2/4/8/16;T8-DSL t2v=42.5 复现 B0 锚点(2.1 零回归);T16 免训练反超;表在 1.6 + from-claude (18) |
| **GATE-0 时间维容量审计** | ⬜ | ← 第一线 1.7,**零训练**,阻塞一切时序类改动;`notes/neural-inspired-round2.md` §2 |
| 脑启发第二轮(F1–F4 + R1/R2/R3) | ✅ | `notes/neural-inspired-round2.md`;四份独立评审收敛,含对第一轮 A1/B1 的修正 |

## 进展日志(实时,倒序;每个会话在此加一行,格式:日期:事件(出处))

- 2026-07-24:**B0 多 T 曲线实测完成,1.6/U2 关闭**:官方权重 MSR-VTT emb 口径 r_mean = 55.4/60.0/61.6/62.3 @ T=2/4/8/16;T=8 DSL t2v R@1=42.5 复现 B0 锚点 ⇒ 2.1 零回归;T=16 插值升采样免训练反超;T=2 gap −6.1 r_mean = C2 目标;Fig-1 v1 数据落档(from-claude (18))
- 2026-07-24:**C2 V100 验收全过 → 转 ✅**:pytest 18 项 + r3e 8 卡冒烟三判据全过,50 步 T 序列服务器↔本地逐位一致;观察项 fp16 评测偶发 sim NaN 记入 TROUBLESHOOTING 末节(from-claude (17));B0 多 T 曲线(任务 C)服务器跑动中
- 2026-07-24:**脑启发第二轮合入**(`notes/neural-inspired-round2.md`):四条已数值验证的结构事实
  (F1 均值池化**无条件**时间盲 / F2 **加性**位置编码被精确抹掉——直接命中 stage-4 的
  `temporal_embedding` / F3 一阶项的最小二乘解 ≡ approximate rank pooling / F4 检索损失在帧残差
  子空间上**恰好平坦**⇒ 帧塌缩免费、轨迹类辅助损失一阶无效);新增 **GATE-0 零训练闸门**
  (`experiments/temporal_capacity_audit.py` + `MVC_FRAME_PERTURB`,命令见 `COMMANDS.md` §3.1);
  提出 R1 精确文本负例库 / R2 可恢复顺序读出头 / R3 曲率仅测量。
  **顺带修正**:`criterion.loss_weight` 只是日志开关、不进损失(B1 配置会静默失效);A1 比第一轮
  所述更弱(重加权集合仍时间盲)
- 2026-07-24:**A1 惊奇池化 10k 链路结案 ✅**(服务器实测 alpha=-4.7e-5 离零、gamma=1 修复在位)——`nA1` 进 H100 A/B 队列(from-claude (8))
- 2026-07-24:b100 §2–§5 全绿(env/权重/anno/预检),**§6 冒烟→R0 命令交接**(#225,from-claude (5));主时钟起跑在即
- 2026-07-24:**C2 T 无关化代码合入**(#222):11 处 T=8 硬编码动态化 + 弹性 T 采样器 + per-T BN 工具 + r3e 冒烟配置;18 项单测全过 ⇒ R3/R3b 前置解除、1.6 解锁
- 2026-07-24:重字幕主计划升 **v1.1**(#223 + handoff live):Panel-V 全自动验收、撤人力标定;Stage 0 命令不变;V100 recap 命令交接(from-claude (4))
- 2026-07-24:确认 b100 本地 jsonl = **全量 10.6M** InternVid-10M-FLT(非子集),免 gdown(from-claude (3));zh 源字幕路径钉死(#213)
- 2026-07-23:**10k 试点三判据全过,1.3 关闭**(#199);计划评审 U1–U10 + C2 按 FluxViT 收窄(#200);命令位置索引(#204);A1 惊奇池化代码合入(#205);b100 §0–§1 就绪(#201)
- 2026-07-22:**B0 基线锁定**(MSR-VTT t2v R@1=42.50 对上论文,4 基准);10k 数据审查 S0–S3 完成(`docs/internvid-zh-audit-report.md`)
- 2026-07-20 及以前:计划文档四份、实验框架/组件开关/卫生修复(#64)、10k 试点配置(#67)
