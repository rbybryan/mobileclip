# experiments/ — 优化方案 A/B 测试框架(对照原 repo 基线)

**目的:** 拿到数据集路径后,不改任何代码、按下面的步骤顺序执行,就能
(1)锁定原 repo 模型的基线数字,(2)逐个 A/B 我们的优化组件,(3)自动汇总成
与基线的对比表。实验设计对应 `notes/paper-plan-zh.md` 的第 1–2 周
(卫生闸门 + C3 组件 A/B)。

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

# 短训冒烟(单卡、debug 模式几十步自动停,验证数据/模型/损失全链路):
bash experiments/run_pretrain.sh experiments/configs/r0_baseline.py 1 debug True
bash experiments/run_pretrain.sh experiments/configs/r1_both.py    1 debug True
```

## 3. B0 — 锁定基线数字(官方权重,双口径)

```bash
CKPT=./checkpoints/mobileviclip_small.pt
for b in msrvtt didemo anet k400 ssv2_mc; do
  bash experiments/run_eval.sh small $b $CKPT experiments/evals/baseline_official/$b
done
python experiments/collect_results.py --out experiments/BASELINE.md
```

核对 `ret_test/`(DSL 口径)是否对上论文/README 数字(MSR-VTT t2v R@1≈42.5);
`ret_test_emb/` 是原始余弦口径。**此后所有 A/B 结论要求两个口径同向成立。**

## 4. R0 → R1 — 训练 A/B(等步数、等数据、等超参)

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

## 6. 判据(与 paper-plan 一致)

- **保留组件**:等步数下相对 R0 **≥ +0.3 R-Sum**,且 DSL 与 raw 两口径同向;
- R1 叠加若低于单项最优 → 取单项最优作为后续 DR 实验(R2/R3)的底座;
- R0 若无法逼近官方权重的量级 → 先排查复现问题,不进入组件结论。

## 7. 后续(本框架预留、尚未实现)

- **R2**(+ 视频 DR 蒸馏损失,paper-plan C1)与 **R3**(+ 随机 T 弹性训练,C2)
  需要新增教师 embedding 管线与 T 无关化改造,作为下一批 PR 进入本目录
  (届时新增 `r2_dr.py` / `r3_elastic.py` 配置,复用同一套 run/collect 脚本);
- 评测按 rank 分片(H5)在扩展到 ActivityNet/DiDeMo 大规模评测前加入。

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
