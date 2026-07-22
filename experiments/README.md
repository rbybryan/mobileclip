# experiments/ — 优化方案 A/B 测试框架(对照原 repo 基线)

**目的:** 拿到数据集路径后,不改任何代码、按下面的步骤顺序执行,就能
(1)锁定原 repo 模型的基线数字,(2)逐个 A/B 我们的优化组件,(3)自动汇总成
与基线的对比表。实验设计对应 `notes/paper-plan-zh.md` 的第 1–2 周
(卫生闸门 + C3 组件 A/B)。

> **当前该做哪一步 → 看 `experiments/ROADMAP.md`(总入口)。**
> 数据集/权重到位了没有 → `python experiments/check_readiness.py`
> 自动检查全部基准、训练语料和 checkpoints,并报告已解锁的步骤。

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

# 短训冒烟(单卡、debug 模式几十步自动停,验证数据/模型/损失全链路):
bash experiments/run_pretrain.sh experiments/configs/r0_baseline.py 1 debug True
bash experiments/run_pretrain.sh experiments/configs/r1_both.py    1 debug True
```

## 2b. 10k 试点(只有审查抽样、没有全量数据时;101vector 8×V100)

全量 InternVid 未到位时,用数据审查的 1 万条抽样把训练链路和组件全部调通。
**试点数字一律不外推、不进任何结论**(全量的千分之一,~180 步)。

```bash
# 1) jsonl -> 训练 anno(9k train + 1k held-out 自检索 eval;
#    同时打印共享字幕统计 = caption_idx 多正例的实际影响面)
python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

# 2) 数据路径:videos 在 /home/vca1/datasets/internvid_zh_10k/videos
export MVC_DATA_DIR=/home/vca1/datasets   # anno 用默认 anno_downstream/ 即可

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
> `duplicate_zh_caption_rate`**(决策门见 `tools/data_audit/DATA_ISSUES.md`:
> 重复率 >10% → 假负例治理升为必做项)。

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

## 6. 判据(与 paper-plan 一致,07-20 审核后加严)

- **保留组件**:等步数下相对 R0 的 **多基准聚合 R-Sum**(MSR-VTT + DiDeMo +
  ANet)提升同向,且 DSL 与 raw 两口径同向;单基准 +0.3 R-Sum 量级低于
  种子噪声,**临界情况(聚合优势 < 0.5)必须补第二种子**再下结论;
- 论文**主口径 = raw 余弦**(`_emb/`),DSL 仅用于与原论文数字对齐;
- R1 叠加若低于单项最优 → 取单项最优作为后续 DR 实验(R2/R3)的底座;
- R0 若无法逼近官方权重的量级 → 先排查复现问题,不进入组件结论;
- R0 启动第一天记录**单 epoch wall-clock**,据此校准 paper-plan 的
  Wk4–6 排期(并实测 `gradient_checkpointing False` 能否放下、省多少时间)。

## 7. 后续(本框架预留、尚未实现)

- **R2**(+ 视频 DR 蒸馏损失,paper-plan C1)与 **R3**(+ 随机 T 弹性训练,C2)
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
