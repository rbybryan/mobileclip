# 怎么跑:训练 → 补评 → 配对比较

> 复现 6 个方案的**唯一入口**是 [`run_model_opt.sh`](../scripts/run_model_opt.sh)——
> 一条脚本三段。本文给一键跑法 + 分段跑法 + 手动兜底 + 判读规则。
> 机制见 [`model-changes.md`](model-changes.md),数字见 [`retrospective-brief.md`](retrospective-brief.md)。
> 命令细节的权威是 `COMMANDS.md` §3.8 / §3.9,本文与之一致、不重复解释每个坑。

## 0. 前置

```bash
cd "$(git rev-parse --show-toplevel)"
source experiments/env_b100.sh          # b100 必做:设 PYTHONUSERBASE,否则 termcolor 等导不进
export MVC_DATA_DIR=/path/to/data  MVC_ANNO_DIR=/path/to/anno_downstream   # 见 experiments/README.md
# checkpoints/mobileclip_s2.pt 就位(视觉塔初值,见 CHECKPOINTS.md)
```

## 1. 一键(推荐)

```bash
# 训练 → 离线补评(落 ranks)→ 配对 bootstrap,一次跑完。挂后台记 PID
# ⚠️ 共用 root:停止只能按这个 PID / 其 PGID,严禁 pkill
nohup bash model-optimization/scripts/run_model_opt.sh 0,1,2,3 > experiments/runs/model_opt.log 2>&1 &
echo "model-opt PID = $!"
```

默认清单 = 精简报告的 6 个方案 × seed:`r0 / P1trained / r1b / nA1 / r1a × {42,43}` +
`P9 × {42}`(二阶统计单 seed 即否决,见 [`model-changes.md`](model-changes.md) §4)。
**断点续训内置**(`auto_resume True`):被打断后原样重跑本脚本,已有 ckpt 的臂从 `ckpt_latest`
续、没 ckpt 的从头起。按 r0 实测约 15 h/臂,11 臂串行 ≈ 一周。

## 2. 分段 / 挑方案跑

```bash
# 只跑某几段(train,eval,compare 逗号分隔)
PHASE=eval,compare bash model-optimization/scripts/run_model_opt.sh 3     # 训练已完成,只要 ranks + CI
PHASE=compare      bash model-optimization/scripts/run_model_opt.sh 3     # 纯 CPU,GPU 参数被忽略

# 只跑指定方案(短名不带 .py)
bash model-optimization/scripts/run_model_opt.sh 0,1,2,3 sub1m_r0_s42 sub1m_P1trained_s42
```

**三段各干什么**:

| 段 | 干什么 | 落点 |
|---|---|---|
| `train` | 逐臂训练(`run_pretrain.sh` + `auto_resume`,日志追加不覆盖) | `experiments/runs/<臂>/ckpt_02.pth`、`<臂>.log` |
| `eval` | 补评 MSR-VTT 1k、落 `ranks_*.npz`(纯前向 ~1 min/臂,已有 ranks 则跳) | `experiments/evals/sub1m/<臂>/`(含 `eval_res.json`) |
| `compare` | 每条非基线臂减**同 seed 的 `r0`**,raw + DSL 两口径配对 bootstrap | stdout(可 `--out` 落 md) |

## 3. 手动兜底(想单独跑某一步 / 排错时)

**训练单臂**(冒烟先加 `debug True` 几十步即停):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/run_pretrain.sh \
    experiments/configs/sub1m_P1trained_s42.py 4 use_bf16 True bf16 True auto_resume True
```

**补评单臂**(ckpt 指到**目录** `ckpt_02.pth`,不是里面的 `.pt`):
```bash
export CUDA_VISIBLE_DEVICES=3
RUN=experiments/runs/sub1m_P1trained_s42
bash experiments/run_eval.sh small msrvtt "${RUN}/ckpt_02.pth" \
     experiments/evals/sub1m/$(basename ${RUN}) num_frames_test 8 batch_size_test 32
# ⚠️ 必做自检:没看到 "Load pretrained model from" = 权重没加载、数字无效(未加载也会打印像样的 r_mean)
grep -E "Load pretrained model from|training from scratch" \
     experiments/evals/sub1m/$(basename ${RUN})/train.log
```

**配对比较单对**(`--compare` 两参数就是**装着 npz 的目录**;ranks 可能在 `runs/` 或 `evals/`,`ls` 一眼再传):
```bash
python experiments/collect_results.py --compare \
       experiments/evals/sub1m/sub1m_P1trained_s42 \
       experiments/evals/sub1m/sub1m_r0_s42                    # raw 主口径
python experiments/collect_results.py --compare \
       experiments/evals/sub1m/sub1m_P1trained_s42 \
       experiments/evals/sub1m/sub1m_r0_s42 --prefix 'ret_test/'   # DSL 口径
```

**自比对自检**(拿到第一份 ranks 后必做一次:同一目录既当臂又当基线,每指标 Δ 必须恰好 0.00):
```bash
python experiments/collect_results.py --compare \
    experiments/evals/sub1m/sub1m_r0_s42 experiments/evals/sub1m/sub1m_r0_s42 --prefix ret_test_emb/
```

## 4. 怎么判读(veto-only,别再套 ±1.6)

1. **一律减同 seed 的 `r0`**,看**配对 bootstrap 95% CI 是否含 0**(默认 `--n-boot 10000`)。
2. **两口径都要看**:raw(`ret_test_emb/`)与 DSL(`ret_test/`)——sigmoid 那臂就是 raw 卡边界、
   DSL 才现出显著负,漏看一个会误判。
3. **加法方案**(三种池化 / sigmoid):要**两 seed 同向 + CI 不含 0** 才算「显著赢」;单 seed 正向
   差是噪声(注意力池化的 seed 42 +2.8 归零就是活教材)。
4. **减法方案**(删时序分支):方向相反——**不需要赢,只要不显著输**即放行全量;但「非劣 ≠ 等价」,
   1M 只能否决。
5. **新池化头必须连门的范数一起报**:`alpha`(surprise)/ `‖W2‖`(stats)仍≈0 ⇒ 退化成均值,
   「打平基线」不代表方向错,只代表没学动(见 [`model-changes.md`](model-changes.md) §4/§5)。

## 5. 常见坑(详见 COMMANDS §3.7–§3.9 / TROUBLESHOOTING)

- **训练跑完却没有任何评测数字**(`Empty DataFrame`):环内评测被跳过,靠 `eval` 段离线补评收官;
  别只看 `runs/` 就判「中断」。
- **`--compare` 报 `FileNotFoundError: ranks_*.npz`**:那条臂的 ranks 在**另一个目录**(runs↔evals);
  `run_model_opt.sh` 的 compare 段已自动指对,手动跑时先 `ls experiments/{runs,evals/sub1m}/<臂>/ranks_*.npz`。
- **墙钟看着过短被判 🔴**:`auto_resume` 续跑 + 日志**追加**才对;覆盖写会抹掉前一腿墙钟。本脚本已用 `>>`。
