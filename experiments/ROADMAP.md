# ROADMAP — 当前该做什么(总入口,随进度勾选更新)

**日期:** 2026-07-20
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

## 第一线:101vector 机器上执行(GPU + 人工,现在就能开跑)

- [ ] **1.1 后台挂下载**(不占 GPU,越早越好):MSR-VTT 1k 测试集、
      官方 `mobileviclip_small.pt`、底座 `mobileclip_s2.pt`。
      弱网断点续传配方:`tools/data_audit/SETUP_V100.md` §0、
      `docs/internvid-download-zh.md`。已有的测试集放好后用
      `check_readiness.py` 确认识别。
- [ ] **1.2 数据审查流水线**(GPU ~3–4h,产出 10k 诊断报告):
      `bash tools/data_audit/run_v100.sh smoke → s0 → s1 → s2-sample → s2 → s3`。
      报告决定 R2 的真实/合成字幕配比。
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
- [ ] **1.6 B0 多 T 退化曲线(U2,依赖 2.1 代码,纯评测)**:官方
      `mobileviclip_small.pt` 在 T∈{2,4,8,16} 下评 MSR-VTT
      (`temporal_embedding` 插值、BN 原样)→ 论文 Fig-1 动机图 +
      标定 C2 要跨越的 gap。该失配曲线在视频-文本模型上未见发表(评审 §3.3)。

## 第二线:代码开发(不依赖数据,随时推进)

- [ ] **2.1 C2 T 无关化**(R3/R3b 前置):动态 T 替换三处 `reshape(8,...)`
      硬编码、`temporal_embedding` 插值、随机 T 采样器。
      验收:T=8 路径与旧代码**逐位一致**的单测;**外加 per-T BN 漂移诊断
      (U3,README §2b 矩阵有行)**——同一 ckpt 纯 T=2 vs 纯 T=16 各前向
      若干 batch,量化 BN running stats 偏移;并附训后 per-T BN 校准脚本
      (纯前向刷新 buffer,4 套,MB 级)。
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

## 第三线:全量数据(外部依赖,主时钟)

- [ ] **3.1** 10M InternVid 到位 + H100 机全量审查(`SETUP_H100.md`)。
- [ ] **3.2** 到位当天:`export MVC_DATA_DIR/MVC_ANNO_DIR` → R0 起跑 →
      **首日实测单 epoch wall-clock**,校准 Wk4–6 排期
      (超预算裁剪顺序 R0 > R2 > R3 > R3b > R1)。

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
| C2 T 无关化 | ⬜ | ← 第二线 2.1(验收含 per-T BN 诊断,U3) |
| R2 教师管线 | ⬜ | ← 第二线 2.2(含 U7/U9 设计项) |
| 全量数据(H100 真 A/B + 终训) | ⬜ | ← 第三线;`SETUP_B100.md` 就绪,`from-server.md` 尚无回报=未起跑 |
| 计划评审 + 文献网扫(U1–U10) | ✅ | `notes/plan-review-upgrades.md`;C2 按 FluxViT 收窄、导出冒烟=1.5、多 T 曲线=1.6 |
| 导出冒烟(U1) | ⬜ | ← 第一线 1.5,零权重即可,决定 P9 触发 |
