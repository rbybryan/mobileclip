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
- [~] **1.5 导出冒烟(U1,零训练权重,CPU 可做)—— 脚本已就绪,待在 101vector 上跑全量**:
      随机权重实例化视觉塔(架构参数与 `build_vision_encoder` 逐项一致 ⇒ 带全部时序
      算子),分走 ONNX / TFLite / coremltools 转换,按 T∈{2,4,8,16} 各出静态图;
      记录每路 pass/fail + 失败算子。产出决定 P9 是否提前触发(真机延迟档仍在第 8 周)。
      **脚本** `experiments/export_smoke.py`,**命令** `COMMANDS.md` §3.6,
      **测试** `tests/test_export_smoke.py`(17 项)。三条路线互相独立,缺依赖记 skip 不崩。
      ⚠️ **skip ≠ fail**:脚本把「缺依赖」「缺传递依赖」都归 skip —— 把环境问题记成 fail
      会产生**假的 P9 触发**(实测踩过:torch≥2.6 的 ONNX 导出器需 `onnxscript`)。
      **本机 CPU 预跑(torch 2.12)**:全尺寸 **ONNX 路线 T=2 / T=4 pass**
      (导出 + `onnx.checker` + onnxruntime 加载推理全过)⇒ 联合注意力在 ONNX 这条路上
      **暂无导出障碍**。T=8/16 本机未跑完,TFLite / Core ML 两路本机无依赖未测 ——
      **这两点是 P9 判据的缺口,101vector 上补齐即可结案。**
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
- [~] **1.7 GATE-0 时间维容量审计(零训练,现在就能跑;命令见 `COMMANDS.md` §3.1)**
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
      **结果(2026-07-24)见 [`experiments/gate0_results.md`](gate0_results.md)**:
      (b) MSR-VTT 四个扰动跑完 —— `shuffle` 相对 B0 **+0.1/+0.2**(两口径,≈零)、
      `reverse` −1.2/−0.1(两口径不同向 ⇒ 判噪声)、`freeze` **−13.0/−12.1**(两口径同向,巨大)。
      ⇒ **顺序值 ≈0,帧多样性值 ≈13 个 R@1**;`freeze` 同时充当阳性对照,证明"shuffle 的 0"是真零而非坏开关。
      ⇒ **MSR-VTT 已被实证取消作为时序指标的资格**,R2 上马前必须先有顺序敏感基准。
      **未完成**:SSv2-MC 四个扰动全崩于 `FileNotFoundError: anno_downstream/ssv2_mc_val.json`
      —— **数据缺失**(B0 当初也只锁了 msrvtt/hmdb51/ucf101/k400),不是代码问题。
      (a) 静态审计、(c) RTime、(d) 种子复现 仍未做。
- [x] **1.8 选帧基线(零训练)—— 已跑完 2026-07-27,结论:C2 不被平凡基线解释掉** ——
      **C2 必须打赢的那条平凡基线**,由 GATE-0 的结果直接逼出来:帧的**多样性
      值 ~13 个 R@1、顺序值 ~0**,而 1.6 显示 T=2 相对 T=8 掉 **5.1 个 t2v R@1**。
      ⇒ 低帧预算下"喂哪几帧"理应很要紧,而这一点**从未量过**。
      C2 的卖点是"弹性 T 在低 T 优于固定 T",审稿人必然会问有没有跟"零训练、
      只是把帧挑得聪明一点"比过 —— 这张对照表现在不存在。
      **9 个臂**(K∈{2,4} × {uniform 受控基线 / divmax 候选 / random 地板}
      + pool16 与 nativeT2/T4 三个参照),单卡串行、可续跑:
      `bash experiments/run_frame_select.sh` → `python experiments/collect_frame_select.py`。
      **判据先写死**:t2v R@1、两口径同向、|Δ| ≥ 1.6(1K 测试集的二项抽样误差,
      与 GATE-0 判 `reverse` 同一条带);`r_mean` 只当描述性数字。**先看地板** ——
      uniform 打不过 random 就说明该 K 上选帧问题不存在。
      **两种结果都有用**:补回缺口一大块 ⇒ C2 净增益要扣掉这块再报;补不回来
      ⇒ C2 故事更硬且多一张必需的对照表。
      落档模板 [`experiments/frame_select_results.md`](frame_select_results.md)。
      **结果(msrvtt/1K,官方权重,raw 主口径)**:三个参照臂逐位复现 1.6
      (42.20/35.70/39.60)⇒ 链路与"钩子零操作"双验证。
      (a) **K=2 上选帧确实要紧**:uniform 相对 random **+3.40/+3.90**(两口径同向,
      远超带宽)—— 唯一稳稳成立的正向效应;
      (b) **但 divmax 是错的判据**:相对 uniform **−2.80/−2.40**,且**与纯随机无可测
      差别**(+0.60/+1.50,均在带内)⇒ 贪心 k-center 等价于不选;
      (c) **零训练选帧补不回低 T 缺口**:K=2 缺口 6.50,最好候选名义补回 2.40(37%)
      但 DSL 仅 +1.40、**未过两口径判据**,divmax 补回 −0.40
      ⇒ **C2 低 T 增益不被这条基线解释掉,论文净增益无需扣减**。
      (d) ⚠️ **副产物**:`nativeT2` 与 `seluniform_K2` 采样只差 3.1% 时长
      ({0.250,0.750} vs {0.281,0.781})却差 raw +2.40 ⇒ **±1.6 二项带低估真实波动**
      (查询间误差相关,非独立),后续把 1.6 当**下界**用,阳性主张要更高门槛。
      **下一步(零 GPU,~10 min;脚本已就绪)**:`python experiments/frame_select_diag.py`
      (命令与判读见 `COMMANDS.md` §3.5)dump divmax 在 K=2 的选帧下标,用 |i−j| 分布与
      端点命中率分开"偏好离群帧"与"像素距离≠时间距离"两个假说
      (相反预测见 [`frame_select_results.md`](frame_select_results.md) §4.1)。
      (e) **±1.6 已提升为仓库级约定** —— [`README.md`](README.md) §6.1:阴性结论照用,
      阳性主张要求 |Δ| 明显大于 2.4;`gate0_results.md` 已加补注(带更宽只让其阴性结论更稳)。
      **定位**:C2 的对照臂,**不是新贡献线** —— paper-plan 已判 whitespace §4
      被 C2 吸收、round-2 §4 已判"任何新的 token 合并/剪枝/选择"赛道关闭。

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
      ⚠️ **已审查(2026-07-24,[`notes/r1-examination.md`](../notes/r1-examination.md)):**
      「精确、无动量编码器」成立但落在 **v2t** 一侧,而头条指标是 **t2v**(需**视频**负例 ⇒ 仍是陈旧队列);
      精确性只因文本支路 dropout 全为 0(train 态 dropout 仍开)且草图的 `cross_entropy(arange)` 必须改成
      `get_mask` 多正例否则注入假负例;30–60× 那个杠杆是**从头训练**制度下测的,本项目是对齐初始化的 3-epoch 微调,
      收益是**开放问题**。**先做零成本的离线假负例密度探针(`experiments/r1_fn_density.py`,命令见 `COMMANDS.md` §3.2),再按 SigLIP 制度(sigmoid+温度+池子)开一条臂。** **[A 已测 2026-07-27:精确重复 ≈0(4998/5000 distinct)⇒ 免精确去重;近重复(B)与 go/no-go(训练期饱和探针)仍未测。见 `notes/r1-examination.md` §7]** **落地方案见 [`notes/r1-implementation-plan.md`](../notes/r1-implementation-plan.md);Part B 现在一条命令(`r1_textfeat_dump.py`→`r1_fn_density.py --feats`)。**
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
| 顺序敏感基准(RTime/SSv2) | ⏸ **暂缓** | GATE-0 三条证据都说模型没有方向信息 ⇒ **不买**(省掉一整轮视频抓取)。只有要证 R2 的**下游收益**时才回收。见 [`notes/order-sensitive-eval.md`](../notes/order-sensitive-eval.md) |
| **GATE-0 时间维容量审计** | ✅ | **已结案**:三条独立证据(行为 shuffle≈0 / 核反对称 0.5011=随机 null / 时序闸门仅空间的 6.1% 且同初值)一致 ⇒ 模型不编码时间方向。见 [`experiments/gate0_results.md`](gate0_results.md) §8 |
| **选帧基线(C2 的平凡对照)** | ✅ | 9 臂跑完(2026-07-27),参照臂逐位复现 1.6。**K=2 选帧要紧**(uniform−random +3.4/+3.9)**但 divmax 是错判据**(−2.8/−2.4,且≈随机)⇒ **零训练选帧补不回 6.50 的低 T 缺口,C2 净增益无需扣减**。副产物:±1.6 带偏松。落档 [`frame_select_results.md`](frame_select_results.md) |
| ~~GATE-0(旧状态)~~ | 🟡 | MSR-VTT 半边 ✅ 结案(**shuffle 不掉分、freeze 掉 13 点** ⇒ 顺序值≈0、帧多样性值 13);SSv2-MC 半边被**数据缺失**卡住。见 [`experiments/gate0_results.md`](gate0_results.md) |
| 脑启发第二轮(F1–F4 + R1/R2/R3) | ✅ | `notes/neural-inspired-round2.md`(中文全文 `notes/neural-inspired-round2_zh.md`);四份独立评审收敛,含对第一轮 A1/B1 的修正 |

## 进展日志(实时,倒序;每个会话在此加一行,格式:日期:事件(出处))

- 2026-07-27:**1.8 选帧基线跑完 —— C2 不被平凡基线解释掉(阴性结论,对 C2 有利)**:
  9 臂零训练,参照臂逐位复现 1.6(42.20/35.70/39.60)。**K=2 上选帧确实要紧**
  (uniform 相对 random +3.40/+3.90,两口径同向)**但 divmax 是错的判据** ——
  相对 uniform −2.80/−2.40,且**与纯随机无可测差别**(+0.60/+1.50 均在带内),
  贪心 k-center 等价于不选。K=2 缺口 6.50,零训练最多名义补回 2.40 且未过两口径
  判据 ⇒ **论文里 C2 的净增益无需扣减**,这张表进正文当基线行。
  与 GATE-0 的调和:`freeze` 崩 13 分说明**需要一些分散**,本轮说明**最大化分散
  是过头的** —— 均匀时间间隔已拿走"多样性"里有用的部分。
  ⚠️ 副产物:`nativeT2` 与 `seluniform_K2` 采样只差 3.1% 时长却差 raw +2.40,
  **±1.6 二项带低估真实波动**(查询间误差相关),后续当下界用。
  过程中修了汇总脚本两个 bug(结果文件名 `eval_res_latest.json`、口径键带尾斜杠),
  已加 8 项模式回归测试。落档 [`frame_select_results.md`](frame_select_results.md)。
- 2026-07-27:**选帧基线代码合入,1.8 开条(待跑)**:GATE-0 的"多样性值 13 个 R@1/顺序值 0"
  直接逼出一条 C2 必须打赢的平凡基线 —— 零训练、从 16 帧母序列里选 K 帧、模型真跑 K 帧
  (同成本)。`utils/frame_select.py`(uniform/divmax/random 三模式,divmax 只用降采样像素
  打分不过编码器 ⇒ 可部署)+ `dataset/base_dataset.py` 环境变量钩子(未设=返回同一对象)
  + `experiments/run_frame_select.sh`(9 臂、可续跑)+ `collect_frame_select.py`(判据钉死
  在 t2v R@1、两口径同向、±1.6 二项带;r_mean 仅描述)+ 34 项单测。命令 `COMMANDS.md` §3.4

- 2026-07-24:**GATE-0 结案 ✅**(`experiments/gate0_results.md` §8):三条独立证据一致 ——
  ①行为 打乱8帧 +0.1/+0.2(≈0)、单帧重复 −13;②核形状 反对称占比 0.5011 vs 随机 null 0.4999
  (逐分位点吻合,n=9920);③闸门 时序 `layer_scale_1d` 仅为同层空间 `layer_scale` 的 **6.1%**,
  且二者**同初值 1e-5**(`mobileviclip_small.py:636-641`)⇒ 16 倍差距是训练结果而非初始化。
  **⇒ 模型不编码时间方向;顺序敏感基准暂缓采购(省一轮视频抓取);下一步投 R1。**
  ⚠️ 期间修正了一个自己的错误判据:反对称占比的随机 null 是 **0.500 而非 0**,已固化为 self-test 断言(#266)
- 2026-07-24:**GATE-0 动态审计 MSR-VTT 半边结案**(`experiments/gate0_results.md`):打乱 8 帧顺序
  **不掉分**(t2v 42.50→42.6 DSL / 40.80→41.0 raw),换成单帧重复**掉 13 个点**(→29.5/28.7,两口径同向)
  ⇒ **帧多样性值 ~13 R@1、帧顺序值 ~0**;`freeze` 兼作阳性对照,证明零效应为真零。
  **MSR-VTT 就此失去作为时序指标的资格**(R2 被卡,需先拿顺序敏感基准);R1 不受影响、优先级不变。
  SSv2-MC 四扰动全崩于缺 `anno_downstream/ssv2_mc_val.json`(数据缺失,非代码)
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
