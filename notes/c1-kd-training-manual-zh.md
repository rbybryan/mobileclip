# C1 教师蒸馏(KD)· 1M 训练执行手册(2026-08-04)

§11.2 阶梯 ③④⑤ 的**逐步执行单**:训练侧接线代码已全部落地(本轮 PR),
本手册按闸门顺序把 `sub1m_C1kd_s42` 送上 H100。设计依据:
[`verification-tier-1m-zh.md`](verification-tier-1m-zh.md) §11.2、C1 KD、
评审 U7/U9(评审清单)。

**判读纪律不变**:1M 层只有否决权;配对 bootstrap 两口径;单指标显著性不作数
(§10.2 的实测教训);种子地板 δ=2.5(保守 5.5)。

---

## 落地了什么(代码地图)

| 文件 | 内容 |
|---|---|
| `utils/teacher_bank.py` | 教师嵌入库:按 **anno 行号**查 merge 后的分片(memmap),miss 行零向量+mask;`verify`/`align`/`self-test` 三个 CLI 闸门 |
| `utils/loss_weighting.py` | **U9 修复**:`criterion.loss_weight` 从纯日志开关变成真权重;无权重的损失键**立刻抛错**(不再隐式 1.0);权重恰为 1.0 跳过乘法 ⇒ 基线臂计算图逐位不变 |
| `tasks_clip/pretrain.py` | `build_teacher_bank`(启动即炸的对齐闸门:单语料、无 caption_idx、样本数=库行数)+ 每步查表 + `kd_hit` 累计命中率 meter + epoch 末 `[KD] cumulative hit rate` 日志行 |
| `models/mobileviclip_small.py` | `_init_vtc_kd`/`_kd_losses`:`loss_kd_feat`(嵌入蒸馏,教师 D→512 可学习投影 + 1−cos,mask 归一)与 `loss_kd_rel`(batch 内 v2v 关系 KL,τ²,去对角;= U9 里 kd_geo 的近亲,同表消融位)。默认 OFF ⇒ 不建参数、不碰 RNG,r0 路径零接触 |
| `experiments/configs/sub1m_C1kd_s42.py` | 臂 config:唯一变化 = `kd_feat=1.0` + `criterion.vtc_kd` |
| `tests/test_teacher_bank.py` | 13 项(查表/miss/重复行/长度失配/U9 求和/align CLI),numpy-only |

**为什么是嵌入蒸馏而不是 §6.2 的相似度矩阵 KD**:那要教师**文本**嵌入
(LLaMA 塔,`--stage text`,依赖重、至今未抽);现在库里只有视频嵌入。
video-text 矩阵 KD 等文本阶段落地再作为独立臂进消融表,**不冒充**。

### 🔴 本设计押注的一条对齐链(全部闸门都在保它)

```
训练 idx ── pt_dataset(jump_filter=True ⇒ 不过滤不重排)──> anno 行号
教师行号 ── extract_teacher(失败保占位行,shard 按行号取模)──> 同一份 anno 的行号
```

任何一环断掉(两份 anno 不同、语料被过滤、caption_idx、多语料拼接),KD 就静默
变成「拿别的视频的教师教这个视频」——损失照降、检索照出数,蒸馏白做。
代码里已放三道自动闸门,§0 还有一道人工闸门,**四道全绿才开卡**。

---

## §0. 零 GPU 闸门(现在就能做,~5 分钟,不占卡)

```bash
cd "$(git rev-parse --show-toplevel)" && git pull

# 0a) 单测 + 自检(秒级)
python utils/teacher_bank.py self-test
python -m pytest tests/test_teacher_bank.py -q          # 13 项

# 0b) 🔴 anno 对齐硬闸门 —— 抽取用的 anno vs 训练用的 anno,逐行比对视频文件名。
#     抽取命令里 --anno 传的是哪份就填哪份(§3.10 的示例路径 data/anno_pretrain/
#     internvid_v1.jsonl 在仓库里不存在,b100 上是你手工放的 —— 正因如此才必须比)。
python utils/teacher_bank.py align \
    --anno-a anno_downstream/InternVid/InternVid-10M-flt_1.json \
    --anno-b <抽取时传给 --anno 的那份文件>
# ✅ 「N 行全部一致」才继续;❌ 任何错位 ⇒ 停,抽取必须对着 --anno-a 那份重跑

# 0c) 库体检(抽取 merge 完成后;merge/verify 命令在 COMMANDS.md §3.10)
python experiments/extract_teacher.py verify --out experiments/teacher/iv2s2_1m
python utils/teacher_bank.py verify --out experiments/teacher/iv2s2_1m \
    --anno anno_downstream/InternVid/InternVid-10M-flt_1.json
# 看三样:coverage=100%(分片全跑完)、anno 行数一致、embed_dim ——
# embed_dim ≠ 768 就改 sub1m_C1kd_s42.py 里的 teacher_dim(config 有注释位)
```

## §1. 冒烟门(b100,单卡 ~10 分钟;可与在跑训练共卡,峰值余量够)

```bash
source experiments/env_b100.sh
export CUDA_VISIBLE_DEVICES=3
bash experiments/run_pretrain.sh experiments/configs/sub1m_C1kd_s42.py 1 \
    use_bf16 True bf16 True debug True output_dir experiments/runs/_kd_smoke \
    > experiments/runs/_kd_smoke.log 2>&1
grep -E "\[KD\]|loss_kd_feat|loss_vtc|kd_hit" experiments/runs/_kd_smoke.log | tail -20
```

**必须同时看到,缺一不过**:

1. `[KD] teacher bank ready: ... 'coverage': 1.0` —— 库建起来且全覆盖;
2. `[KD] vtc_kd ON: teacher_dim=768 feat=True rel=False` —— 模型侧开对了;
3. 日志里 `kd_hit: 1.0000` 且 `loss_kd_feat` **有限**(不是 nan/inf;首值 ~1.0 量级,
   随机投影下 cos≈0 ⇒ 1−cos≈1,属预期);
4. `loss_vtc` 与 r0 冒烟同量级(~7 起步)—— KD 没把对比损失打坏。

**默认关 = 零扰动,不用另跑对照**:vtc_kd 不写进 config 时不建任何参数、不碰 RNG
流、不改计算图(权重 1.0 跳乘法有单测钉着)⇒「关 KD 时 bitwise 等于 r0」由构造
保证。真要眼见为实:重跑一次 `sub1m_r0_s42` 的 debug 冒烟,首 20 步 `loss_vtc`
应与该臂日志开头的历史值同轨。

## §2. 排队(15 h;当前队列跑完、`status_1m.py` 转 ⚪ 且无 🔴/⚠️ 后)

```bash
python experiments/status_1m.py        # ⚪ 才排(块 82 修掉了三条假警报)
nohup bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_C1kd_s42 \
    > experiments/runs/verify_1m_c1kd.log 2>&1 &
echo "c1kd PID = $!"
```

跑动后头几分钟看一眼:`grep -m3 "kd_hit" experiments/runs/sub1m_C1kd_s42.log` ——
`1.0000` 以外的任何值 = 行号对齐出了问题,**立刻按 PID 停**,别烧满 15 h。
每个 epoch 末还有一行 `[KD] cumulative hit rate = 1.000000` 兜底。

## §3. 判读(跑完后,CPU 分钟级)

```bash
# 同 seed 减 r0(⚠️ r0_s42 的 ranks 在 evals/ 下,块 82 的教训)
python experiments/collect_results.py --compare \
    experiments/runs/sub1m_C1kd_s42 experiments/evals/sub1m/sub1m_r0_s42
python experiments/collect_results.py --compare \
    experiments/runs/sub1m_C1kd_s42 experiments/evals/sub1m/sub1m_r0_s42 --prefix ret_test/
```

| 结果(两口径同向,R_sum/r_mean 为准) | 动作 |
|---|---|
| 显著正(CI 下界 > 0) | 补 seed 43 确认;两 seed 同向 ⇒ C1 进 10M 配方 |
| NS(CI 含 0) | **一次预登记重试**:λ_feat 1.0→0.25(唯一改动),重跑 s42;再 NS ⇒ 本形态结案,等文本阶段的矩阵 KD |
| 显著负 | 不重试不辩解,结案记入 §10 风格战绩表;KD 线转向文本阶段/低 T 蒸馏(§11.7) |

**预登记(2026-08-04,先于任何结果写下)**:首跑 λ_feat=1.0;唯一重试位 λ_feat=0.25;
τ=0.07 不动;kd_rel 不与 feat 同臂首跑(单变量纪律),它的消融位与 kd_geo 合并
(U9)。**除上述外任何超参改动都视为新臂,需重新立项,不占「一次重试」名额。**

收尾连带报告:`temporal_pool` 无(r0 配方)、`kd_proj.weight` 的范数(零 GPU,
P9 那条命令改个路径即可)—— 若训完 ‖W‖ 几乎没离开初始化,说明梯度没到投影层,
结果**不可判读**(nA1/P9 的同款陷阱)。

## §4. 依赖与顺位(谁先谁后)

```
① 1M 抽取(块 79 已排,~1.6 天)──> §0c 库体检 ──> §1 冒烟 ──> §2 排队
② §0a/0b(单测 + anno 对齐)——现在就能做,不等抽取
③ round-2 臂(mc2init/lean,notes/verification-tier-1m-zh.md)与本臂共享队列,
   谁先过完自己的闸门谁先上;C1kd 依赖抽取完成,大概率排在它们后面
```
