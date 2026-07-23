# ROADMAP — 当前该做什么(总入口,随进度勾选更新)

**日期:** 2026-07-20
**分工:** 战略层(为什么/判据)看 `notes/paper-plan-zh.md`;每条命令看
`experiments/README.md`;本文件回答"现在轮到哪一步"。
**第 0 步(任何时候)**:`python experiments/check_readiness.py`
——自动报告哪些测试集/权重已到位、下面哪些步骤已解锁。

三条线并行,互不阻塞;主时钟由第三线(全量数据)决定。

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
- [ ] **1.3 10k 试点**(README §2b,GPU 空档):
      `make_pilot_anno.py`(记下共享字幕比例)→ `export MVC_DATA_DIR` →
      `pilot_10k_r0` 冒烟 + 正跑 → `pilot_10k_r1`;
      外加 `pytest tests/test_optimization_components.py`。
      **通过标准(只看动力学)**:损失正常下降;r1 的 ckpt 里
      `logit_scale/logit_bias` 离开初始值(t≠10、b≠−10);
      `temporal_pool.proj` 非零。不满足 = 配置/代码有问题,先修再前进。
      **结果见 [`experiments/pilot_10k_results.md`](pilot_10k_results.md)**:r0 全过、
      r1 判据一(损失下降)过、判据二三待服务器读 ckpt 核实(文档内附命令)。
- [ ] **1.4 B0 锁基线**(1.1 完成后,README §3):官方权重跑已到位的
      基准 → `collect_results.py --out experiments/BASELINE.md`。
      核对 MSR-VTT DSL 口径 t2v R@1 ≈ 42.5;**论文主口径 = raw(`_emb/`)**。

## 第二线:代码开发(不依赖数据,随时推进)

- [ ] **2.1 C2 T 无关化**(R3/R3b 前置):动态 T 替换三处 `reshape(8,...)`
      硬编码、`temporal_embedding` 插值、随机 T 采样器。
      验收:T=8 路径与旧代码**逐位一致**的单测。
- [ ] **2.2 R2 教师管线**(1.2 的报告出来后设计更准):InternVideo2
      embedding 离线提取脚本,先在 10k 上试点实测每 clip 耗时/存储,
      同时敲定审核修正 9 的三个设计细节(教师维度投影、帧索引记录、
      低 T KD 加权)。用 10k 成本外推 10M,决定教师用 L 还是 1B。
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
| 10k 试点跑通 | 🟡 | r0 过·r1 判据一过·二三待核实,见 pilot_10k_results.md |
| B0 基线锁定 | ⬜ | ← 第一线 1.4 |
| C2 T 无关化 | ⬜ | ← 第二线 2.1 |
| R2 教师管线 | ⬜ | ← 第二线 2.2 |
| 全量数据 | ⬜ | ← 第三线 |
