# ROADMAP — 当前该做什么(总入口,随进度勾选更新)

**日期:** 2026-07-28(**实时进展看文末「进展日志」+「状态速览」**,每个会话更新)
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
  step-0 严格等价 mean,不退化**);③ **R1 精确文本负例库**(文本塔冻结 ⇒ 缓存特征、每步用当前投影重算
  ⇒ 数学上精确、零陈旧、无需动量编码器)—— **试点级 A/B 判 GO,但耐久差经 40-epoch 长档修正为
  ΔR-Sum ≈ +3.9**(10-epoch 快照的 +11.0/+13.1 是在对照尚未从早塌恢复时测的;详见下方 2.4 与
  `r1-implementation-plan.md` §9.4)。三者默认关闭,基线路径与原 repo 逐位一致。
- **C4 · 结果** — 多帧数零样本检索(MSR-VTT/DiDeMo/ANet)+ 零样本分类/多选
  (K400/UCF101/HMDB51/SSv2-MC)+ **端侧延迟表**,对比 MobileViCLIP /
  InternVideo2-S/B / **FluxViT-S(GFLOPs 对齐行)**;TempMe 属微调协议,
  只进单独的微调/延迟-only 对比区(07-23 评审 §3.5)。
- **卫生闸门(任何基线数字之前强制)** — 移 `set_detect_anomaly`、清 spikingjelly
  死依赖、修 `hasattr(name,'_1d')` 冻结 bug、DSL/raw 两口径分开报并钉死主口径。
- **条件项** — R4(P7 非对称帧编码,双触发才回收)、P9(导出阻塞时的分离注意力变体;
  **可导出蒸馏实现 = 下方「架构探索 backlog」AC1a**)。架构第三轮最终稿的更多候选(AC1–AC9)
  见文末「架构探索 backlog」——均为顾问式、未做 A/B、不进主关键路径。
- **⚠️ 07-23 计划评审(`notes/plan-review-upgrades.md`,U1–U10)** — C2 新颖性按
  FluxViT(ICCV'25)收窄为「帧数旋钮×移动端骨干×固定 T 锚点×端侧」组合格;导出
  冒烟提前(下方 1.5);C2 增 per-T BN 校准;R0 全局 batch 对齐 512;预算阶梯;
  统计口径三件套。paper-plan §七 为计划层准绳。

> ⚙️ **验证分三档(2026-07-28 由两档扩为三档,别把职责搞混)**:
> ① **V100 / 10k(`pilot_10k_*`)= 链路 go/no-go** —— 能跑通、loss 正常降、新参数收到梯度;
> **排不了方案序**(评测饱和 + BN 混淆 + 二项 SE),且 40M 参数在 10k 上**严重欠拟合**,机制表达不出来;
> ② **1M 子集(`sub1m_*`,#329)= 验证决策档** —— 「值不值得上全量」的高置信判断 + 否决,
> 8 个 config = 4 臂(r0/r1a/r1b/nA1)× 2 seed,**测试集仍是标准 MSR-VTT 1k(绝不缩)**,
> 见 [`notes/verification-tier-1m-zh.md`](../notes/verification-tier-1m-zh.md);
> ③ **H100 / 全量 10M = 论文数字** —— 真 A/B 排序 + 胜者终训。
> 详见 `README.md`「两级实验」(**该节措辞已落后于本三档结构,待同步**)。

---

## 每一步的命令写在哪(文件 × 小节速查)

| 步骤 | 命令所在 md → 小节 |
|---|---|
| 第 0 步 就绪自检 | 本文件开头(`python experiments/check_readiness.py`);B0 前预检 = `README.md` §3.0 |
| 1.1 下载(测试集/权重) | `tools/data_audit/SETUP_V100.md` §0(环境 + 权重后台下载)、§0.5;链接大全 `CHECKPOINTS.md`(Pretrained weights / Google Drive 两节);InternVid 分片抓取 `docs/internvid-download-zh.md` §3–§4;通用下载 `COMMANDS.md` §2 |
| 1.2 数据审查流水线 | `tools/data_audit/SETUP_V100.md`:§1 冒烟 → §2 S0 → §3 S1(§3.5/3.6 体检)→ §4 S2(4.2 主力 / 4.3 裁判 / 4.5 体检)→ §5 S3 报告;§6.5 一条命令看全状态 |
| 1.3 10k 试点(✅ 已关闭) | `experiments/README.md` §2b 全套(anno 生成、`export MVC_DATA_DIR`、r0/r1、pytest);结论见 `pilot_10k_results.md`(#199) |
| 1.4 B0 锁基线 | `experiments/README.md` §3.0 预检 → §3.2 路径对上 → §3.3 跑 B0 + `collect_results.py`;已锁数字在 §3.4 |
| 1.5 导出冒烟(U1) | 任务与验收:`notes/plan-review-upgrades.md` U1;**脚本 `experiments/export_smoke.py`,命令 `COMMANDS.md` §3.6,测试 `tests/test_export_smoke.py`(22 项)**;结果 `experiments/export_smoke_results.md`。TFLite 补测用独立 venv(结果文档 §3) |
| 1.6 B0 多 T 曲线(U2) | 任务与验收:`notes/plan-review-upgrades.md` U2;2.1 已落地 ⇒ 用既有零样本 eval 脚本对官方权重覆盖 `num_frames_test <T>`(T∈{2,4,8,16})即可,模型侧自动插值 |
| 2.1 C2 T 无关化(✅ 代码落地) | 单测 `python -m pytest tests/test_t_agnostic.py -v`(README §2);V100 冒烟命令见本文件 2.1;per-T BN 工具 `experiments/per_t_bn.py`;设计准绳 `paper-plan-zh.md` §七.3 |
| 2.2 R2 教师管线 | **提取脚本 `experiments/extract_teacher.py` 已落地(2026-07-29)**,命令 `COMMANDS.md` §3.10,单测 `tests/test_extract_teacher.py`(44 项);教师权重条目 `CHECKPOINTS.md`;设计决策清单:本文件 2.2 + `paper-plan-zh.md` §六.9 + 评审 U7 |
| 2.4/2.5 A1/A2/B1(#197) | `notes/neural-inspired-implementation.md` Tier A(配置模板 + 单测)/ Tier B(B1 接线与三臂);配置文件待创建 |
| **2.4 R1 负例库(建缓存 + 跑 A/B + 长档)** | 探针 `COMMANDS.md` §3.2;建池与 A/B 一条龙 `notes/r1-implementation-plan.md` §8;结果 §9.1/§9.2;长档两臂配置 `experiments/configs/nR1_textbank_pilot_long.py` + `pilot_10k_r1_long.py`,启动命令 `notes/live/from-claude.md` (54) |
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
      **🟡 重字幕线当前定版见该文件 §0.9(2026-07-28 全案复审,与旧段冲突以它为准)**:
      流水线定为 **①独立多遍(A dense×5 / B 固定槽位×3)→ ②判断式直融(§5.7 v3,弃硬投票)
      → ③验真+✗修复复验 → ④训练 captions = 直融短句版**。复审 **3 红 / 4 黄 / 4 绿**:
      🔴 **GDINO 至今未装**(服务器连不上 HF,scp 方案已给四次未执行)⇒ 外部幻觉门只剩 OCR,
      **当前最大裸露面**;🔴 直融 token 上限 512→**640 已修**(#313,防 v3 长输出截断→JSON 不闭合→整片置空);
      🔴 长稿不过验真/修复 ⇒ **仅作目检参考**,下游(KD/EN 线)消费长稿前必须先补长稿验真。
      🟡 **迭代快于证据**(判断式/slot/双稿均零目检,"caption→检索增益"因果链零训练证据)⇒ **冻结 prompt,先目检再小训练**。
      **该线不在 EN 论文主关键路径上(D2)**,不占 R0–R3 主时钟。
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
      **测试** `tests/test_export_smoke.py`(22 项)。三条路线互相独立,缺依赖记 skip 不崩。
      ⚠️ **skip ≠ fail**:脚本把「缺依赖」「缺传递依赖」都归 skip —— 把环境问题记成 fail
      会产生**假的 P9 触发**(实测踩过:torch≥2.6 的 ONNX 导出器需 `onnxscript`)。
      **101vector 实测(2026-07-27,全尺寸 `[1,T,3,256,256]`,torch 2.5.1)**:
      **ONNX 与 Core ML 在 T∈{2,4,8,16} 上全部 pass**(ONNX 走完导出 +
      `onnx.checker` + onnxruntime 加载推理三步;Core ML 约 4343 个 MIL op 全过)
      ⇒ **联合注意力不构成导出障碍,P9 暂不触发**,弹性模型可按现形态训练。
      **TFLite 受 torch 版本阻塞,本机不评估**:两轮报错都是工具链问题而非导不出
      —— ① `ai-edge-torch` 已改名 `litert-torch`(旧包是空壳);② `litert-torch` 需
      **torch>=2.6** 的 `torch.int1`,而 101vector 锁 **2.5.1+cu118** 是为了 V100 的
      **sm_70 内核**(换版本会让 8 卡在第一个 barrier 全崩,见 TROUBLESHOOTING 末节)。
      ⇒ **别为一格导出结论动训练环境的 torch**;导出是 CPU-only + 随机权重,
      开独立 venv 补测即可(配方见结果文档 §3)。两轮都已修为 skip 而非 fail。
      落档 [`export_smoke_results.md`](export_smoke_results.md)。
      ⚠️ **对 C2 部署叙事的观察**:trace 显示 **T 被烘成常量**(`int(t)`、reshape、
      以及 `temporal_embedding` 走不走插值那个 `if`)⇒ "帧数即部署旋钮"在**权重层面**
      成立(单 checkpoint),**产物层面是 1 份权重 → N 张图**。
      **未验证**:ONNX `dynamic_axes` 能否一张图服务所有 T —— 便宜且直接影响 C4 端侧表,
      建议在真机延迟档之前搞清楚。
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
- [ ] **1.9 未见过的 T 泛化(零训练,单卡串行,~1 晚)—— 答"C2 能不能无上限 T"**
      分析全文见 [`notes/c2-arbitrary-t.md`](../notes/c2-arbitrary-t.md)。
      **结论先行**:时序算子**代码层面已无上限**(`resize_temporal_embedding` 任意插值、
      `reparam_conv_1d` 任意长度、单测已覆盖 T=3 奇数),但有**四堵墙**:
      🧱**O(T²) 的 stage-4 联合注意力**(N=64·T,T=32 的注意力项是 T=16 的 4 倍;
      T=16 实测已占 23.3GB)=**唯一的真墙**;🧱**教师锚点只有 16 帧**(T>16 无对齐目标,
      要重跑 10M 教师 pass);🧱**dataloader 故意卡死** `t > num_frames` 时 raise
      (`elastic_t.py:23`,母序列 16);🧱**导出把 T 烘成常量**(U1)⇒ 1 份权重 → N 张图。
      ⇒ **训练集合保持 {2,4,8,16} 不动**;**真要无上限先做 AC1a**(可分离注意力把 T 从
      O(T²) 降到 O(T),同时消掉导出墙)。
      **本条要做的是免费那半**:官方权重上评**训练没见过的 T** —— 内插 T∈{3,5,6,12}、
      外推 T∈{24,32}。命令 `COMMANDS.md` §3.7。判据沿用 §6.1(两口径同向;阳性要 |Δ|>2.4)。
      **两种结果都有用**:内插平滑+外推不崩 ⇒ 多一张「训 4 点、任意点可用」的图;
      外推崩 ⇒ 把"16 帧锚点"从实现细节升级为**被实证的设计边界**,主动写进 limitations。
      ⚠️ **别把卖点改成「任意 T」** —— 评审 §3.1 的差异化格是「帧数旋钮×移动端骨干×
      固定 T 锚点×端侧」,「无上限」更靠近 FluxViT 已占的地盘;这张表只当**附加鲁棒性证据**。

## 第二线:代码开发(不依赖数据,随时推进)

> ### 🚨 读本段前必看:`R1/R2/R3` 与 `P1–P4` 在本仓库有**两套含义**,而且本段两套都在用
>
> | 编号 | ①**训练运行**(paper-plan §四 / 第三线) | ②**第二轮提案**(`neural-inspired-round2.md` §3) |
> |---|---|---|
> | `R1` | 基线 + C3 组件 | **精确文本负例库** |
> | `R2` | 基线 + C1 DR 损失(= 下方「2.2 R2 教师管线」) | **可恢复顺序的读出头** |
> | `R3` | 基线 + C2 随机 T(头条模型;另有 `R3b`/`R0`/`R4`) | **曲率仅作测量** |
>
> **本段内就并存**:`2.2 R2 教师管线` 是①,`2.4 R1 精确文本负例库` 与 `2.5 … R2 读出头`
> 是②,相隔十几行。`P1` 同样撞车:`2.3` 的 `P1 torch 升级` 是 paper-plan 的 P1,
> `2.5` 的 `P1 逐帧轨迹外露` 是 round2 §5 的 P1(round2 的 P2/P3/P4 亦与 P1–P10 撞)。
> 第三处:架构 backlog 的 `AC1–AC9` 在 `model-architecture-improvements-round3-zh.md`
> 原文里写作 `C1–C9`,与 paper-plan 的贡献层 `C1–C4` 撞 —— **本文件一律用 `AC*`**。
>
> **约定**:上下文不能一眼分辨时,写 `round2-R1` / `run-R1`(`round2-P2` / `plan-P2`)。
> 已合入的 PR 大量引用旧编号,**不做追溯重命名**,这是阅读规则不是重构。

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
- [x] **2.2a 教师提取脚本** —— ✅ **2026-07-29 落地**:`experiments/extract_teacher.py`
      (教师 = IV2-stage2,U7-4 定案)。分片 + 定长块 + **原子落盘 ⇒ 断点续跑**;
      解码失败**保留占位行不删**(删行会让 `rows` 与嵌入静默错位),单块失败率超
      `--fail-rate-abort` 直接抛错;`merge` / `verify` / `self-test` 三个子命令。
      单测 `tests/test_extract_teacher.py` **44 项全过**(纯 CPU,不需要权重与视频;
      含一个用极小真 `InternVideo2` 验证 token 布局与读出头假设的接口层)。
      命令见 `COMMANDS.md` §3.10,权重条目见 `CHECKPOINTS.md`。
      **修正 9 的三个设计细节落点**:① 教师维度不投影、原样存 D 并写进 manifest
      (投影是训练侧的事);② 逐条存 `frame_indices` + 协议进 manifest,
      采样默认 `middle`(**确定性**,不依赖种子);③ 低 T 可达性 → `--per-frame`
      的 `subset_emb`。⚠️ **口径**:`subset_emb` 是「教师**读出**只看这些帧的 token」,
      **不等于**「教师**只看**这些帧重跑一遍」(主干注意力仍跨全 16 帧),
      manifest 记作 `readout_restricted`;要后者得按子集重跑主干(需 pos-embed 插值)。
      另:`--per-frame` 只多几次极廉价池化,**不是 T× 成本**;按 IV2-1B 的 D=768 算
      ≈ **24 KB/片**(U7 原文的 ~45KB 是按更大的 D 估的)。
- [ ] **2.2b 跑起来 + 定语料范围**(**这才是 G1 的实体,脚本只是让它可跑**):
      先跑 `--limit 100000` 成本切片拿实测 `clip/s`,再按它外推决定
      **curated 2–3M 先行 vs 全量 10M**(U7-1,等步对照)。教师用 1B 还是 6B
      同法先试再定(U7-4)。合成字幕 K≥2 由 CapRL 产出侧保证(U7-3);
      相对 KD(`vtc_kd`/`kd_geo`)单一所有权并入同一消融表(U9)——
      ⚠️ U9 前先修 `loss_weight` 只是日志开关那个坑(round2 §5 P2)。
- [ ] **2.2c 训练侧接线**(脚本不管这一段):教师嵌入读取 Dataset、
      D→512 投影、`vtc_kd` 的 ~30 行新写(`data-quality-caprl-scheme.md` §6.2),
      低 T KD 加权/课程。
- [ ] **2.3(条件项)** H5 评测分片(大基准评测前)、P1 torch 升级
      (仅当显存/速度真卡住)。
- [~] **2.4 R1 精确文本负例库 —— 🟢 试点级 A/B 已判 GO(两个种子),下一步 40-epoch 长档**
      (`notes/neural-inspired-round2.md` §3 R1;**不依赖 GATE-0,
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

      **✅ 已完成(2026-07-27 → 07-28,详见 `r1-implementation-plan.md` §7.6/§8/§9):**
      - **探针 Part B(近重复密度)= 低 ⇒ GO 建池**(§7.6);代码 2.4c 落地(建缓存 + 跑 A/B 一条龙,`r1_build_textbank.py`,默认关)。
      - **A/B seed-42(§9.1)**:处理臂 `nR1_textbank_pilot`(bank ON,pool=(8999,512),K′=8999 ⇒ 每步整库当 sigmoid 负列)
        vs 对照 `pilot_10k_r1`(同配置去掉 `criterion.text_bank`)。末轮 **R-Sum 533.2 vs 522.2 ⇒ ΔR-Sum = +11.0**,
        kill 判据是 **< 0.5 ⇒ 停**,**清过 ~22×**;池在 **全部 10 个 epoch、两个方向**都领先。
      - **A/B seed-43 复现(§9.2)**:**ΔR-Sum = +13.1**(ep9:v2t txt_r1 **+3.4**、t2v img_r1 **+2.9**);
        两臂 torch 版本一致(`2.5.1+cu118`)⇒ **环境混淆保留意见已消除**。
      - **机制**:检索 R@k 只看 L2 归一化余弦 ⇒ **对 `logit_scale`/`logit_bias` 免疫**,差不可能由温度制造;
        seed-43 看得最清楚 —— **对照在 ep1 就塌**(r_mean 90.2→85.4 且没完全回来),池几乎不掉(90.9→平台 ~88.9)
        ⇒ 收益形态是**减小微调退化 / 更高更快的平台**,不是相对零样本初始化的绝对增益。
      - **⚠️ 诚实登记**:两臂都低于各自 epoch-0(冻结初始化)峰值 —— 9000 片 / 17 batch 的试点**测不到** R1 真正
        针对的制度(大数据 + 多步 + 批内负例真的成为信息瓶颈)。**试点级 GO ≠ 论文级结论。**
      **✅ 40-epoch 长档已跑完(2026-07-28,§9.4)—— 试点级问题就此收尾**:
      平台(ep30–39 均值)池 **88.57** vs 对照 **87.92**,**两条平台带不重叠**(池最差 88.33 > 对照最好 88.10)
      ⇒ 耐久 **Δr_mean +0.65 / ΔR-Sum ≈ +3.9**(末轮 +3.7 一致),仍是 kill 阈值 0.5 的 **~8×**。
      **对照 ep0→ep1 又塌了一次**(90.00→85.18,**第三次复现**),再花约 12 epoch 爬回平台
      ⇒ 池挡的就是这个早塌;4× 步数让对照**追回约 2/3**。
      ⇒ **结论修正(refine, not overturn)**:诚实说法是「**大幅早期稳定 + 温和耐久提升**」,
      **不是** 10-epoch 快照那个 +11~13 —— 那是在对照尚未恢复时测的。
      ⚠️ **且这仍不是定论**:10k 上池挡的是**小数据过拟合伪影**;R1 真正针对的机制(大数据下
      in-batch 负例成为信息瓶颈)**根本没被触发**。**V100 不需要再跑任何试点级实验。**
      **⚠️ 大数据 A/B = H100 的活,不是 V100**(§9.3):`internvid_all`(11 个 `InternVId-FLT_*` 分片)只在 H100 上;
      **V100(101vector)只有 10k 子样本**,把 `train_corpus="internvid_all"` 指过去会因数据缺失直接失败。
      挪到 H100 时须**在 H100 重建更大的 text-bank 缓存**(现有 `textbank_pilot_*` 只覆盖 10k)。
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
      **起跑前先读两条已归档的坑(2026-07-28,#316,`TROUBLESHOOTING.md`)**:
      ① b100 `patch_embed` 秒级 OOM 而 `nvidia-smi` 显示 0% util —— 别等"空卡",按剩余显存裁 batch;
      ② DDP + 梯度检查点崩 `Parameter ... layer_scale_2 has been marked as ready twice`
      —— **已修**:两处 `checkpoint()` 调用点加 `use_reentrant=False`(`models/mobileviclip_small.py`)。

## 架构探索 backlog(第三轮最终稿落地顺序 · 条件项,不进 R0–R3 主关键路径)

**来源**:[`notes/model-architecture-improvements-round3-zh.md`](../notes/model-architecture-improvements-round3-zh.md)
(逐点源码核查·最终稿,PR #310)。

> ⚠️ **编号防撞(务必先读)**:那份 note 内部把架构候选编号为 **C1–C9**,与本文件的
> **贡献层 C1–C4**、以及**弹性-T(= 贡献 C2)** 是**完全不同的东西**(例:note 的 C2 = 时序金字塔
> 池化,贡献 C2 = 弹性帧训练)。为免混淆,**本节一律加前缀 `AC-`(Arch-Candidate)**,括号内标注
> note 的原始编号。

**定位**:全部为顾问式研究方向,**均未做等步 A/B**,**不占 R0–R3 主时钟**;按下列闸门择机推进,
落地任一条前先补对照实验(评测集/判据沿用 §「最终优化方案」等步纪律 + §1.8 的 ±1.6→2.4 门槛)。
选点的两条绑定约束:①逐帧卷积 ×T 主导 ~55G FLOPs;②stage-4 联合 T·H·W softmax = 峰值显存 + 导出瓶颈
(`models/convert_model.py` 绕过全部时序算子,`whitespace §0`)。

> ⚠️ **2026-07-28 重排(U1 推翻了本节原来的排序前提)** —— 详见
> [`notes/c2-vs-alternatives.md`](../notes/c2-vs-alternatives.md) §1。原顺序把 **AC1a 放在「主攻」**,
> 理由是它「产出 whitespace §0 点名的首要缺失物:时序模型端侧导出」。**U1 实测证明那个缺失物
> 不存在** —— 含全部时序算子的视觉塔在 **ONNX + Core ML 于 T∈{2,4,8,16} 全 pass**。⇒
> 🔻 **AC1a 降为条件项**(剩余价值只有峰值显存,却要付整套蒸馏成本);🔻 **AC4 失去存在理由**
> (定位就是「AC1b 的可导出退化」);🔻 **AC1b 动机减半**;🔺 **AC9 / AC3′ 的 Step 0 前置已清**。
> **未被推翻**:TFLite 未评估、**导出≠快**(真机延迟一个没测)。
> **新下注:AC5 + AC9 = 论文「柱 2」候选 → AC-b 基础设施 → AC2。**

落地顺序(2026-07-28 重排后):

- [ ] **AC-a 先手 · 免费/前提 —— 且现在是「柱 2」的第一块砖**(可与主战役并行,零精度风险):
      - **AC5(note C5)时序多分支过参数化块** —— `RepMixer.reparameterize()` 已把
        `mixer_1d`+`norm_1d` 折叠进单个 `reparam_conv_1d`(`mobileviclip_small.py:762/801`);
        训练期并联 k=5 / dilated 分支、推理期折叠,**推理成本 +0**。风险最低,**建议最先做**。
      - **AC9(note C9)量化前拆 INT8 重参数化地雷** —— AC1/AC5 与现架构均重度依赖重参数化,
        folded kernel 动态范围大;按 QARepVGG 约束分支 scale + 用 QAT(非 PTQ),否则效率在 NPU 上蒸发。
- [ ] **AC-b 基础设施(AC2 的硬前置)** —— 把全局 `set_num_frames` 改成**逐阶段**帧数 + 单测。
      现状 `set_num_frames(T)` 在 `forward` 顶部只调一次、全网恒定(`:1431`);中途时序池化会让
      `get_num_frames()` 返回旧 T、后续 reshape(`:525/726/1160`)错位崩。**不做这步 AC2 必崩。**
- [ ] **AC-c 主攻(现在只剩一条)**:
      - **AC2(note C2)早期时序收缩(8→4→2 池化调度)** —— 深层最贵阶段少见帧,砍 55G FLOPs
        主导项 + 减 stage-4 token 数。**须先完成 AC-b。** 风险:过早池化冲淡短动作 →
        在 MSR-VTT/DiDeMo 动作切片消融池化深度。
- [ ] **AC-c′ 条件项(2026-07-28 从主攻降下来)—— 由 TFLite 补测 + 真机延迟触发,不是由 1.5 触发**:
      - **AC1a 蒸馏成可分离时空注意力** —— ~~产出首要缺失物:端侧导出~~ **已作废**(U1:ONNX+Core ML 全 pass)。
        剩余价值 = 消 stage-4 峰值显存 + 一点 FLOPs;代价 = 整套蒸馏工程 + 掉点风险。**性价比已不成立,除非
        TFLite 那格挂在联合注意力上**(那才重新触发)。
      - **AC1b(高风险赌注)** —— 推理彻底无注意力。动机减半(只剩显存),仍要付预期 0.5–1.5 R@1。
- [ ] **AC-d near-free** —— **AC6′(note C6)文本塔 LoRA**:冻结文本塔加 rank-8 LoRA,导出期
      merge、推理零成本;与贡献 C3 的 sigmoid(`r1a_sigmoid.py`)叠加。注:note 里"温度可学"
      部分**已被 R1a 覆盖**(`self.temp` 被 clamp 钉死 `:90/166`;可学 `logit_scale/bias` 在
      sigmoid 路径 `:145-146` 且 R1a 已训),净新增只有 LoRA。**与 §2.4 R1 负例库/温度项相邻,
      别重复接线。**
- [ ] **AC-e 后备/独立线(两条都已减值)**:
      - ~~**AC4 register 瓶颈注意力**~~ —— **失去存在理由**:它的定位就是「AC1b 掉点时的**可导出**退化」,
        而可导出性不再是问题(U1)。除非出现别的动机,否则**移出 backlog**。
      - **AC3′ 因果时序卷积 + 流式缓冲** —— Step 0 前置**已清**(U1),但**改被 GATE-0 卡死**:
        模型不编码时间方向 ⇒ **因果 vs 非因果在现有基准上测不出差别**,无法证明收益 ⇒
        要复活得先有顺序敏感基准(该采购已「暂缓」)。
- [ ] **AC-f 低优先 add-on** —— AC7 RGB-diff 运动 token / AC8 低秩双线性池化(视评测集是否
      运动重再定;两者预期增益小,均可整包移除)。

**死路(核查未翻案,别投端侧)**:delta/skip-conv 稀疏计算、Mamba/SSM 时序混合、文本条件化
视频池化、CoCa 解码头 —— 理由见 note §3。

---

## 汇合点 → 主战役(paper-plan §四)

三线汇合条件:全量数据就绪 + B0 已锁 + 10k 试点通过 + C2 代码合入。
然后按闸门推进:**R0(复现)→ R1(组件)→ R2(DR 蒸馏,预期最大收益)
→ R3(弹性 T,新颖性核心)→ R3b(对照)**;R4/P9 按触发条件
(paper-plan §四、§六)。目标 CVPR 2027(约 11 月截稿)。

## ⚠️ 方案完整性自查(2026-07-28;答「目前的优化方案完善吗」)

**不完善。** 逐条列出**已知的缺口**,按严重度排。这一节的作用是**不让缺口靠"没人提"活下去** ——
每条都要么有 owner 条目,要么在这里挂着。

| # | 缺口 | 严重度 | 现状 |
|---|---|---|---|
| **G1** | **C1(视频数据强化)被定为「主线、预期收益最大」,却是唯一零代码的贡献层** | 🟡 **2026-07-29 降级:提取脚本已落地,但一条嵌入都还没抽** | ✅ **提取脚本已写**(`experiments/extract_teacher.py`,44 项单测,命令 §3.10,教师权重条目已补进 `CHECKPOINTS.md`)⇒「零代码」不再成立。**仍红的部分**:① **一条嵌入都还没抽** —— 下一步是 `--limit 100000` 成本切片;② **U7-1 语料范围(2–3M vs 10M)仍未定**,要等切片的实测吞吐;③ **训练侧接线整段未动**(教师嵌入 Dataset、D→512 投影、`vtc_kd` 那 ~30 行);④ 教师权重**是否已下载到机器上未确认**。⇒ 重心与投入仍然反向,只是不再卡在「没有工具」 |
| **G2** | **U10-3 的非劣边际 δ=0.3 R@1 与实测噪声地板冲突** | 🟡 **2026-07-29 已定案,待填一个实测数** | **选出路 ②**,详见 `plan-review-upgrades.md` **§6.1.1**。要点:2.4 与 δ **量的不是同一种噪声** —— 2.4 来自**跨抽样**的绝对值比较(源:输入 nuisance),而每-T 非劣是**同 T、同抽帧、同 1000 查询**的配对比较,该 nuisance 与查询抽样噪声**都已抵消** ⇒ 拿 2.4 当 δ 下界是**标尺错配**(出路 ① 据此否决;出路 ③ 只压查询噪声、对种子无效,降为备选)。真正的地板是**训练种子方差**,**尚未实测**,但 1M 层「4 臂 × 2 seed」的**同臂双种子之差**就是它,**零额外算力**。⇒ 判据 = 多基准聚合 R-Sum + 两口径同向 + **配对 CI 不含 0**;**δ 由实测种子地板逐 T 重定,0.3 作废**。**不阻塞 R3 起跑**,只阻塞头条主张定稿 |
| **G3** | **U8 的 EMA 至今不在 `tasks_clip/pretrain.py`** | 🟡 | 全仓 grep 无 `model_ema`;U8 原定「R1 起默认开」 |
| ~~**G4**~~ | ~~U10-1 的 bootstrap CI~~ | ✅ **2026-07-29 已实现** | `experiments/paired_bootstrap.py`(纯 numpy,11 项单测)+ `collect_results.py --compare A B`;评测时自动落盘逐查询 rank(~8KB/臂)。**实测分辨率提升约 11 倍**:同一对臂 R-Sum 的 95% CI 半宽 配对 **±1.5** vs 无配对 ±17 ⇒ +0.8 的 R@1 差可判显著。**判读改为「CI 是否含 0」,不再套 ±1.6**(那是无配对带)。命令 `COMMANDS.md` §3.9。⚠️ rank 是 07-29 才开始存的,更早评过的臂需重跑一次评测(~1 min) |
| **G5** | **柱 2(AC5+AC9)只有论证,没有可执行条目** | 🟡 | `c2-vs-alternatives.md` 判它是唯一值得补的架构线,但 backlog 里 AC5 仍只是一行描述,**无配置、无单测、无判据** |
| **G6** | **U1 的 TFLite 那格 + `dynamic_axes` 一图多 T 未验** | 🟡 | 前者决定 Android 结论与 AC9 落点,后者直接影响 C4 端侧表;**都零 GPU**,却没有 owner 条目 |
| **G7** | **GDINO 未装(重字幕线)** | 🟡 | scp 方案已给四次未执行 ⇒ 外部幻觉门只剩 OCR。**不占 EN 主时钟(D2)**,但那条线的验真是空的 |
| **G8** | **`README.md`「两级实验」措辞已落后于三档结构** | 🟢 | 1M 层(#329)插入后,那节仍写「两级」 |
| ~~**G9**~~ | ~~主时钟 R0 是否已起跑~~ | ✅ **2026-07-29 已答(部分)** | **1M 子集 R0(`sub1m_r0_s42`)已在 H100 跑完 3 epoch**(4 卡 × batch 128 = 全局 512,`Training time 15:11:58`)。**全量 10M R0 是否另有一路在跑仍未确认**,但已不阻塞:实测吞吐见下方 U6 条 |
| **G10** | **1M 层首臂白跑 15 小时 —— 训练内评测被静默跳过** | ✅ **已修** | `test_file` 键 `act_val` 与 `test_types=["ret_test"]` 对不上 ⇒ 每轮跳过 ⇒ `Empty DataFrame`。#334 修了 8 个臂,**#335 新增的 4 个臂又漏了 ⇒ 本次补齐 12/12**,并在 `pretrain.py` 加**开跑即失败**的守卫(代价 15h → 秒级)。ckpt 完好,**离线补评测即可,不用重训**(`COMMANDS.md` §3.8) |
| **U6 预算阶梯:实测 wall-clock 已到手** | 🔴 **需要决策** | 由 1M 实测外推 | 9.1821 s/it @ 全局 batch 512 ⇒ **55.8 samples/s**。全量 10.65M ⇒ **20,796 it/epoch ≈ 53.0 h/epoch**,**R0 三个 epoch ≈ 159 h ≈ 6.6 天(4 卡)**。U6 的「宽裕档 = 33E」≈ **73 天,直接不可行**;按常态档(R0 3ep + 排序臂各 1ep + 胜者补 3ep)在 4 卡上 ≈ **26 天**。⇒ ⚠️ **「加卡」这条路不存在**(#339:0–3 就是维护者在这台 H100 的**全部配额,没有 4–7**)⇒ **6.6 天/R0 是硬约束**,只能在**臂数/epoch 数**上裁,按 U6 常态档执行并接受 4 卡串行;全量 R0 与 1M 验证层**共用这 4 张卡 ⇒ 必须串行排队** |

**已经补齐、不再是缺口的**(免得重复提):导出可行性(U1)、多 T 曲线(U2)、per-T BN 工具(U3)、
预算阶梯(U6)、MobileCLIP2-S2 条目(U8 之一)、GATE-0、选帧平凡基线、R1 试点级结论、1M 验证层配置。

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
| 导出冒烟(U1) | 🟡 **两路已结论** | 2026-07-27 于 101vector 实测:**ONNX + Core ML 在 T∈{2,4,8,16} 全 pass** ⇒ **联合注意力不是导出障碍,P9 暂不触发**;TFLite 受 `litert-torch` 需 torch≥2.6 阻塞(本机锁 2.5.1 是为 V100 sm_70),**独立 venv 补测**即闭环。⚠️ trace 显示 **T 被烘成常量** ⇒ 部署是「1 份权重 → N 张图」,`dynamic_axes` 能否一图多 T **未验证**。见 [`export_smoke_results.md`](export_smoke_results.md) |
| B0 多 T 曲线(U2/1.6) | ✅ | 2026-07-24 实测:emb r_mean 55.4/60.0/61.6/62.3 @ T=2/4/8/16;T8-DSL t2v=42.5 复现 B0 锚点(2.1 零回归);T16 免训练反超;表在 1.6 + from-claude (18) |
| 顺序敏感基准(RTime/SSv2) | ⏸ **暂缓** | GATE-0 三条证据都说模型没有方向信息 ⇒ **不买**(省掉一整轮视频抓取)。只有要证 R2 的**下游收益**时才回收。见 [`notes/order-sensitive-eval.md`](../notes/order-sensitive-eval.md) |
| **GATE-0 时间维容量审计** | ✅ | **已结案**:三条独立证据(行为 shuffle≈0 / 核反对称 0.5011=随机 null / 时序闸门仅空间的 6.1% 且同初值)一致 ⇒ 模型不编码时间方向。见 [`experiments/gate0_results.md`](gate0_results.md) §8 |
| **选帧基线(C2 的平凡对照)** | ✅ | 9 臂跑完(2026-07-27),参照臂逐位复现 1.6。**K=2 选帧要紧**(uniform−random +3.4/+3.9)**但 divmax 是错判据**(−2.8/−2.4,且≈随机)⇒ **零训练选帧补不回 6.50 的低 T 缺口,C2 净增益无需扣减**。副产物:±1.6 带偏松。落档 [`frame_select_results.md`](frame_select_results.md) |
| ~~GATE-0(旧状态)~~ | 🟡 | MSR-VTT 半边 ✅ 结案(**shuffle 不掉分、freeze 掉 13 点** ⇒ 顺序值≈0、帧多样性值 13);SSv2-MC 半边被**数据缺失**卡住。见 [`experiments/gate0_results.md`](gate0_results.md) |
| 脑启发第二轮(F1–F4 + R1/R2/R3) | ✅ | `notes/neural-inspired-round2.md`(中文全文 `notes/neural-inspired-round2_zh.md`);四份独立评审收敛,含对第一轮 A1/B1 的修正 |
| **C2 换成别的创新点?(架构侧)** | ✅ **已判:不换柱,加第二柱** | 逐条评估 AC1–AC9 后**没有任何单个架构候选能当柱子**;C2 有代码/单测/冒烟/动机图/平凡基线/闸门全套实测,候选全是零 A/B。**U1 顺带推翻了 backlog 的排序前提**(「时序算子导不出」是推测,实测 ONNX+Core ML 全 pass)⇒ AC1a/AC4/AC1b 降级、AC9/AC3′ 的 Step 0 已清。**推荐柱 2 = AC5+AC9**(重参数化在时序轴:训练期免费 / INT8 期收账),与 C2 互补、共用 C4 端侧表。见 [`c2-vs-alternatives.md`](../notes/c2-vs-alternatives.md) |
| **C2 无上限 T?(1.9)** | 🟡 **已判:代码无上限,四堵墙有一堵是真墙** | 时序算子代码层面已支持任意 T(含奇数,单测覆盖 T=3);真墙 = **stage-4 联合注意力 O(T²)**(N=64·T;T=16 实测 23.3GB)⇒ **训练集合保持 {2,4,8,16}**,真要无上限先做 **AC1a**。**待跑(零训练)**:未见过的 T 泛化(内插 3/5/6/12、外推 24/32),`COMMANDS.md` §3.7。⚠️ 别改卖点为「任意 T」(FluxViT 地盘)。见 [`c2-arbitrary-t.md`](../notes/c2-arbitrary-t.md) |
| **R1 文本负例库(2.4)** | 🟢 **试点级结案:GO,但幅度被修正** | 三个种子/档位都领先:10-epoch ΔR-Sum **+11.0**(s42)/ **+13.1**(s43);**40-epoch 长档修正为耐久 ΔR-Sum ≈ +3.9**(平台 88.57 vs 87.92,两条平台带**不重叠**,池在最后 10 epoch 每轮都领先,仍是 kill 阈值 0.5 的 ~8×)。机制=**挡住对照 ep0→ep1 的早塌**(90.0→85.2,**三次复现**),4× 步数让对照追回约 2/3。⇒ 诚实说法:**大幅早期稳定 + 温和耐久提升**。见 §9.1/§9.2/**§9.4** |
| R1 长档 40-epoch(对照能否追上) | ✅ **已跑完(2026-07-28)** | 答案:**大部分追上但没追平**。**试点级问题就此收尾,V100 不需要再跑**;决定性检验仍是 H100 大数据 A/B(§9.3) |
| **1M 验证层:首个结果(`sub1m_r0_s42`)** | ✅ **管线验证通过** / 🔴 **但仪器精度不够** | **1M(10% 数据)× 3 epoch 就追平官方全量**:raw t2v R@1 **41.8** vs 官方 41.3、DSL **42.7** vs 42.5(两口径同向、均在带内 ⇒ 打平);地板(骨干初始化+随机时序)raw r_mean 54.82 → 1M **60.65** → 全量 61.47。🔴 **10× 数据只买到 +0.82 raw r_mean,落在 ±1.6 噪声带以内** ⇒ **MSR-VTT 1k 排不出臂序**,必须先补配对 bootstrap(G4)+ 多基准聚合。见 [`verification-tier-1m-zh.md`](../notes/verification-tier-1m-zh.md) §6 |
| 1M 验证层 其余 11 臂 | 🟡 **待跑** | 填 10k 冒烟与 10M 论文数字之间的空档:**8 config = 4 臂(r0/r1a/r1b/nA1)× 2 seed**,`train_corpus=internvid_v1`(单个 1M 分片)、`batch_size=128`、epoch=3、**测试集仍是标准 MSR-VTT 1k**;runner `experiments/run_verify_1m.sh`。判法=同 seed 相减 + 两 seed 同向。**+4 config = R1 文本负例库自配对 A/B**(`sub1m_r1both`/`sub1m_nR1` × 2 seed,彼此相减,见下一行)。见 [`verification-tier-1m-zh.md`](../notes/verification-tier-1m-zh.md) |
| R1 大数据 A/B(真正发威档) | 🟡 **配置已入库(1M 子集档)** | 不等全量,**放到 1M 验证层跑**:处理 `sub1m_nR1_s42/s43`(+bank) vs 对照 `sub1m_r1both_s42/s43`(SigLIP 制度无 bank),**彼此相减**;需先建 1M 缓存 `textbank_1m_*`(命令 `COMMANDS.md` §3.3.2)。为何 1M 才发威:全局 batch 512 ≪ SigLIP 饱和 ~32k + 不过拟合 ⇒ 领先=真硬负例(非 10k 挡早塌伪影)。全量 10M=论文数字。设计见 [`r1-implementation-plan.md`](../notes/r1-implementation-plan.md) §10 |
| 重字幕线定版(§0.9 全案复审) | 🟡 | 2026-07-28:判断式直融 v3 定版 + 供给取代旧段;**3 红**(GDINO 未装=最大裸露面 / 512→640 已修 / 长稿未验真)、4 黄(迭代快于证据 ⇒ 冻结 prompt)、4 绿。**不占 EN 主时钟(D2)**。见 [`recaption-master-plan.md`](../notes/recaption-master-plan.md) §0.9 |
| b100 训练侧两个坑 | ✅ 已修/已档 | `patch_embed` 秒级 OOM(别等空卡,按剩余显存裁 batch)+ DDP×梯度检查点 `marked as ready twice` ⇒ 两处 `checkpoint()` 加 `use_reentrant=False`(#316,`TROUBLESHOOTING.md`) |
| 架构探索 backlog(AC1–AC9,第三轮最终稿) | ⬜ | 顾问式,**均未 A/B、不占主时钟**;落地顺序在「架构探索 backlog」段;AC1a=可导出 P9 由 §1.5 触发、AC-b 是 AC2 前置、AC6′ 净新增仅文本塔 LoRA;死路 4 条已判。源码核查稿 #310 |

## 进展日志(实时,倒序;每个会话在此加一行,格式:日期:事件(出处))

- 2026-07-29:**C1 教师提取脚本落地 —— G1 由 🔴 降为 🟡**(`experiments/extract_teacher.py`,
  命令 `COMMANDS.md` §3.10,单测 `tests/test_extract_teacher.py` 44 项全过)。
  教师 = **IV2-stage2**(U7-4 定案),权重条目补进 `CHECKPOINTS.md`(原生 4 帧,
  按 T=16 抽取时用 `interpolate_pos_embed(orig_t_size=4)`,与 `mobileviclip_small.py`
  载 stage2 视觉塔同一套处理)。工程要点:分片 + 定长块 + 原子落盘 ⇒ **断点续跑**;
  **解码失败保留占位行绝不删行**(删行 ⇒ `rows` 与嵌入静默错位 ⇒ 下游只表现为
  「蒸馏没效果」,G10 同类事故);单块失败率超阈值**直接抛错**,不写半空的 shard;
  载权重 `missing>20` 直接抛错(防 #344 那种「没加载却照跑」)。
  **修正 9 的三条**各自落点见 2.2a。**per-frame(U7-2)实现方式与原设想不同**:
  不按帧重跑教师(那是 T× 成本),而是 hook 住 `clip_projector` 的输入 token,
  把**同一个读出头**作用在帧子集的 token 上 —— 单测证明「用全量 token 重放
  = 模型自己的输出」逐位相等,所以这个口径是干净的;代价近乎为零,存储 ≈24 KB/片
  (不是 U7 估的 45KB)。⚠️ 但它是 `readout_restricted`,**不等于**教师只看子集重跑,
  拿它答 §六.9 决策 3 时必须按这个口径陈述。
  单测过程中抓到一个真 bug:`pool()` 会重入 forward hook 把已捕获的 token 覆盖掉,
  已改为池化期间冻结 hook,并加回归测试。
  **下一步 = 2.2b**:`--limit 100000` 成本切片 → 实测 `clip/s` → 定 U7-1 语料范围。

- 2026-07-29:**G2 定案 —— δ=0.3 作废,选出路 ②,δ 改由实测种子地板重定**
  (`plan-review-upgrades.md` **§6.1.1**)。核心诊断:**2.4 与 δ 量的不是同一种噪声**。
  README §6.1 的 2.4 来自 `nativeT2` vs `seluniform_K2` —— **抽样时刻不同**的**绝对值**比较,
  量的是「模型对无关输入扰动的敏感度」;而 R3 的每-T 非劣是**同 T、同抽帧、同 1000 查询**的
  **配对**比较,该 nuisance(源③)因 T 对齐而抵消、查询抽样噪声(源①)因配对而抵消
  ⇒ **拿 2.4 当 δ 下界是标尺错配**,会把本来可测的主张判成不可测 ⇒ **出路 ① 否决**;
  出路 ③(换大评测集)只压源①、对种子完全无效 ⇒ **降为备选**。
  **真正的地板 = 训练种子方差(源②),不因配对抵消,且至今未实测** ——
  但 1M 层「4 臂 × 2 seed」的**同臂双种子之差**(如 `sub1m_r0_s42` − `s43`)就是它,
  **零额外算力**,不需要专门排跑。⇒ 判据写死为**多基准聚合 R-Sum + 两口径同向 + 配对 CI 不含 0**;
  δ 逐 T 由实测地板重定(低 T 地板可能更大,别用一个数糊全部)。
  **不阻塞 R3 起跑**,只阻塞头条主张定稿。顺带:`README.md` §6.1 补了 2.4 的**适用面收窄**
  (只管非配对/跨抽样),并解除 `r_mean` 的「无 CI」限制(配对 bootstrap 给了经验 CI)。

- 2026-07-28:**R1 大数据 A/B 设计定档 = 1M 子集档**(本次):既然 H100 现训 1M 子集,就把 R1 文本负例库 A/B
  放到 1M 验证层跑,不等全量。入库 4 config(处理 `sub1m_nR1_s42/s43` +bank / 对照 `sub1m_r1both_s42/s43` 无 bank,
  彼此相减)+ 1M 缓存构建命令(`textbank_1m_*`,`COMMANDS.md` §3.3.2)。为何 1M 才是正确尺度:全局 batch 512 ≪
  SigLIP 饱和 ~32k、1M×3ep 不过拟合 ⇒ 领先=真硬负例,而非 10k 的挡早塌伪影。全量 10M=论文数字,1M=决策门。
  设计见 `r1-implementation-plan.md` §10、`verification-tier-1m-zh.md`。
- 2026-07-29:**配对 bootstrap 落地,G4 结案 —— 1M 层的判据从此换掉 ±1.6**
  (`experiments/paired_bootstrap.py` + `collect_results.py --compare`,`COMMANDS.md` §3.9,11 项单测):
  评测时自动把**逐查询 rank** 落盘(~8KB/臂),两条臂用**同一组重采样下标**做差 ⇒ 共享方差被抵消。
  **实测分辨率提升约 11 倍**:同一对臂 R-Sum 的 95% CI 半宽,**配对 ±1.5 vs 无配对 ±17**
  ⇒ 一个 +0.80 的 R@1 差可以判显著,而无配对带永远判不出来。
  ⇒ **A/B 判读改为「95% CI 是否含 0」**;`README.md` §6.1 的 ±1.6 仍适用于**单臂绝对值**的阴性结论,
  但**不再用于 A/B 比较**。两口径同向的纪律不变。
  ⚠️ rank 是今天才开始存的 ⇒ 之前评过的臂(首版 `sub1m_r0_s42`)要重跑一次评测(纯前向,~1 min)。
- 2026-07-29:**1M 层首个真实数字到手 —— 管线验证通过,但仪器精度告急**
  (`verification-tier-1m-zh.md` §6;权重加载已核实):**1M(10% 数据)× 3 epoch 就追平官方全量模型** ——
  raw t2v R@1 **41.8** vs 官方 41.3、DSL **42.7** vs 42.5,两口径同向且都在 ±1.6 带内 ⇒ 判打平。
  三点参照:地板(骨干初始化+**随机**时序算子)raw r_mean **54.82** → 1M **60.65** → 全量 **61.47**。
  ✅ 复现管线端到端正确,`sub1m_*` 可当可信代理档。
  🔴 **但同一组数字暴露了更要紧的问题:数据翻 10 倍只买到 +0.82 raw r_mean(DSL +0.80),
  落在 ±1.6 噪声带以内** —— 连"10× 数据"这种巨大处理效应都测不出来,各臂的效应只会更小
  ⇒ **MSR-VTT 1k 在 1M 尺度上排不出臂序**。这不是 1M 层设计错,是**测量仪器精度不够**。
  ⇒ **G4(U10-1 的 query-level bootstrap CI)由 🟡 升为 🔴**:各臂评的是**同一** 1000 条查询,
  属**配对**比较,配对差值方差远小于无配对 ±1.6 带 ⇒ **它是让 1M 层能用的前提,不是锦上添花**;
  同时把 README §6 的「多基准聚合 + 两口径同向」从纪律升级为**必须**。
- 2026-07-29:**1M 层首臂白跑 15 小时 —— 训练内评测被静默跳过;已修 + 加开跑即失败守卫**:
  `sub1m_r0_s42` 在 4×H100 上跑满 `Training time 15:11:58`,却只得到 `Empty DataFrame`。
  根因**不是评测代码**,而是 `test_file` 的键(基线写 `act_val`)与 `test_types=["ret_test"]` 对不上,
  `pretrain.py` 的 epoch 末循环把唯一的测试集整个 `continue` 掉 —— 正是评审 **U8「训练内 held-out 评测修 key」**那条。
  #334 已为 8 个臂修过,但 **#335 新增的 4 个臂(`sub1m_nR1_*` / `sub1m_r1both_*`)又原样漏了**
  ⇒ 本次**补齐 12/12**,并在 `tasks_clip/pretrain.py` 加**开跑前**的键交集守卫(为空直接 raise 并打印两边的键)
  —— **代价从 15 小时降到几秒**。**ckpt 完好,离线补评测即可,不用重训**(`COMMANDS.md` §3.8;
  DeepSpeed 目录式 ckpt 由 `shared_utils.py:150` 的 `osp.isdir` 分支直接支持)。
  **顺带拿到 U6 等了很久的那个数**:9.1821 s/it @ 全局 batch 512 = **55.8 samples/s**
  ⇒ 全量 10.65M **≈ 53.0 h/epoch**、**R0 三个 epoch ≈ 6.6 天(4 卡)**;U6 的宽裕档 ≈ 73 天不可行,
  常态档 4 卡 ≈ 26 天。⚠️ **加卡这条路后被 #339 否掉**(0–3 即全部配额,无 4–7)⇒ 6.6 天/R0 是硬约束,只能裁臂数/epoch 数。
  **G9 部分结案**(1M 子集 R0 确认已在 H100 跑完;全量 R0 是否另有一路仍未确认),**新增 G10**。
- 2026-07-28:**R1 长档 40-epoch 跑完 → 结论修正(refine, not overturn)+ 新增「方案完整性自查」节**(#330 / 本次):
  平台(ep30–39)池 88.57 vs 对照 87.92、两带**不重叠** ⇒ 耐久 **ΔR-Sum ≈ +3.9**,而非 10-epoch 快照的 +11~13
  —— 那是**对照尚未从早塌恢复时**测的。对照 ep0→ep1 塌 90.00→85.18(**第三次复现**),4× 步数追回约 2/3。
  ⇒ 诚实说法「**大幅早期稳定 + 温和耐久提升**」;仍是 kill 阈值的 ~8×;**试点级问题收尾,V100 不再跑**。
  同时把 **1M 验证层(`sub1m_*`,#329)** 补进本文件:8 config = 4 臂 × 2 seed,填 10k 冒烟与 10M 论文数字之间的空档
  ⇒ 顶部「验证分两台机」**改为三档**。**新增「⚠️ 方案完整性自查」节,挂出 G1–G9 九条已知缺口**,
  其中 🔴 **G1 = C1 被定为主线却零代码**(教师提取脚本至今未写)、🔴 **G2 = δ=0.3 与噪声地板冲突未解**。
- 2026-07-28:**U1 导出冒烟合入主干(此前完工但滞留在 `claude/export-smoke` 未合)+ U1–U10 落地状态复核**
  (见 [`notes/plan-review-upgrades.md`](../notes/plan-review-upgrades.md) §6 的逐条审计表):
  **1.5 从「脚本待写」改为「两路已结论」** —— 101vector 2026-07-27 实测 **ONNX + Core ML 在 T∈{2,4,8,16} 全 pass**
  ⇒ **P9 暂不触发**;TFLite 卡在 `litert-torch` 需 torch≥2.6(本机锁 2.5.1 为 V100 sm_70),**独立 venv** 补测即闭环。
  ⚠️ **对 C2 部署叙事的限定**:trace 显示 **T 被烘成常量** ⇒「帧数即部署旋钮」在**权重层**成立(单 checkpoint),
  **产物层是 1 份权重 → N 张图**;`dynamic_axes` 能否一图服务所有 T **未验证**(便宜,且直接影响 C4 端侧表)。
  复核同时点出两个真缺口:**U8 的 EMA 至今不在 `tasks_clip/pretrain.py`**、**U10-1 的 bootstrap CI 不在 `collect_results.py`**;
  以及一处**判据冲突**:U10-3 预注册的非劣边际 **δ=0.3 R@1** 落在 README §6.1 实测噪声带(阳性需 |Δ|>2.4)**之下约 8 倍**,
  R3 头条主张前必须重定。
- 2026-07-28:**R1 文本负例库 A/B 判 GO —— 两个种子都清过 kill 判据约 20×**(#305 / #315,
  [`r1-implementation-plan.md`](../notes/r1-implementation-plan.md) §9.1/§9.2):seed-42 末轮 R-Sum
  **533.2 vs 522.2 = ΔR-Sum +11.0**;seed-43 **复现且更大,+13.1**(ep9:v2t txt_r1 +3.4、t2v img_r1 +2.9),
  两臂 torch 版本一致(`2.5.1+cu118`)⇒ 环境保留意见消除。池在**全部 10 个 epoch、两个方向**领先;
  R@k 只看 L2 归一化余弦 ⇒ **对 `logit_scale`/`logit_bias` 免疫**,差不可能由温度制造。
  **机制**=减小微调退化:**对照 ep1 就塌**(r_mean 90.2→85.4 没完全回来),池几乎不掉(90.9→平台 ~88.9)。
  ⚠️ 两臂都低于各自冻结初始化峰值 —— 9000 片试点**测不到** R1 真正针对的大数据制度,**试点级 GO ≠ 论文级结论**。
  **下一步**:V100 跑 40-epoch 长档两臂(#315,答"对照多跑步能否追上");**大数据 A/B 是 H100 的活**
  (#317:`internvid_all` 11 分片只在 H100,V100 只有 10k;上机前须在 H100 重建更大 text-bank 缓存)。
- 2026-07-28:**b100 训练侧两个坑归档 + DDP 修复**(#316,`TROUBLESHOOTING.md`):
  ① `patch_embed` 秒级 `torch.OutOfMemoryError` 而 `nvidia-smi` 显 0% util —— 别等"空卡",按**剩余显存**裁 batch;
  ② DDP + 梯度检查点崩 `Parameter ... layer_scale_2 has been marked as ready twice`
  —— **已修**:两处 `checkpoint()` 加 `use_reentrant=False`(`models/mobileviclip_small.py`)。
- 2026-07-28:**重字幕线出定版一页图 + 全案复审(§0.9,3 红/4 黄/4 绿)**(#312/#313/#314):
  融合从**硬投票改判断式**(§5.7 v3:抓重点 → 类型感知辨真伪 → 先长稿后短句 + 带理由 dropped),
  B 臂改**固定槽位**(§5.6b),训练 captions 定为**直融短句版**;旧段 §5.5/§4-schema/D5-G2 阶梯/§5.7 v1-v2 **均被取代**。
  🔴 **GDINO 仍未装**(scp 已给四次)= 外部幻觉门只剩 OCR,**当前最大裸露面**;🔴 直融 token 上限 512→**640 已修**;
  🔴 长稿不过验真 ⇒ 仅目检参考。🟡 **迭代快于证据 ⇒ 冻结 prompt,先目检后小训练**。**不占 EN 主时钟(D2)**。
- 2026-07-28:**第三轮架构最终稿落地顺序写入 ROADMAP(新增「架构探索 backlog」AC1–AC9)**:
  round-3 头脑风暴稿经逐点源码核查后(#310)出最终稿——改正 3 处过时前提(弹性 T/冻结 bug/
  temporal_pool 均已实现)+ 2 处机制错误(C1 联合注意力塞不进深度时序卷积→改可分离蒸馏=可导出 P9;
  C2 与全局 `set_num_frames` 冲突→须先改逐阶段帧数)+ 1 处表述不准(C6 温度已被 R1a 覆盖,净新增仅
  文本塔 LoRA)。落地顺序 **AC5+AC9(免费/前提)→ AC-b(改 set_num_frames 基础设施)→ AC1a+AC2(主攻)
  → AC6′(near-free)→ AC4/AC3′(后备)**;死路 4 条已判。**为避免与贡献层 C1–C4 撞号统一加前缀 `AC-`。**
  均未 A/B、不占主时钟。
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
