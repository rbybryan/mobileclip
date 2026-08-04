# from-claude → Server

Claude writes here in reply to [`from-server.md`](from-server.md). **Newest block at the TOP.**
Server reads this and acts. See [git-handoff-workflow.md](../git-handoff-workflow.md).

Reusable commands/fixes/plans also get logged into `COMMANDS.md` / `TROUBLESHOOTING.md` /
`notes/…` per [CLAUDE.md](../../CLAUDE.md) — this file is the live scratch channel.

<!-- Template — Claude copies this block to the top for each reply:

## YYYY-MM-DD HH:MM — <short topic>

**Diagnosis:** <what's going on>

**Do this next:**
```bash
<exact commands to run from repo root>
```

**Then report back** in from-server.md: <what output Claude needs to see next>

-->

---

## 2026-08-04 (83) — ✍️ C1 KD 训练侧代码已全部落地(含 U9 `loss_weight` 假开关修复);执行手册 = notes/c1-kd-training-manual-zh.md

维护者拍板「写 C1 代码 + 1M 训练手册」—— 本块交付。**§11.2 阶梯 ③ 关闭**,
⑤(`sub1m_C1kd_s42`,15 h)只差抽取(块 79 已排)+ 手册里的闸门。

**落了什么**(代码地图在手册开头):`utils/teacher_bank.py`(按 anno 行号查教师嵌入
+ align/verify 闸门 CLI)、`utils/loss_weighting.py`(**U9 修复:`loss_weight` 变真权重,
无权重键启动即炸;权重 1.0 跳乘法 ⇒ 既有臂计算图逐位不变**)、模型侧 `loss_kd_feat`
(教师 D→512 投影 + 1−cos)与 `loss_kd_rel`(v2v 关系 KL,默认 0)、`sub1m_C1kd_s42.py`、
13 项单测(numpy-only,已在本机全绿)。默认 OFF ⇒ r0/一切既有臂零接触。

**Do this next(零 GPU,现在就能做,不等抽取)**:

```bash
git pull
python -m pytest tests/test_teacher_bank.py -q && python utils/teacher_bank.py self-test

# 🔴 anno 对齐硬闸门:训练读 anno_downstream/InternVid/InternVid-10M-flt_1.json,
# 抽取读你传给 --anno 的那份(b100 上手工放的 jsonl)。两份必须逐行同名,否则
# KD = 拿别的视频的教师教这个视频,静默白做。<路径> 换成抽取实际用的:
python utils/teacher_bank.py align \
    --anno-a anno_downstream/InternVid/InternVid-10M-flt_1.json \
    --anno-b <抽取时传给 --anno 的那份文件>
```

**抽取 merge 完之后**:手册 §0c(库体检)→ §1(b100 冒烟,10 分钟,4 条判据)→
§2(排队,15 h)→ §3(判读;λ 与唯一重试位已预登记)。速查在 `COMMANDS.md` §3.12。

**Then report back** in from-server.md:① pytest 与 align 的输出(尤其 align 是否全行一致);
② 抽取 merge 后 `teacher_bank.py verify` 打印的 coverage 与 embed_dim(≠768 要改 config,
手册有注释位);③ 冒烟的 4 条判据行。

---

## 2026-08-04 (82) — 块 77 四项收尾判读:P9 正式结案;三条 🔴/⚠️ 全是假警报(还会卡住块 79 的排队闸门,已修);两条命令因路径盲区重发

**一句话**:你贴的四段输出(块 77 的 §11.0 收尾命令)里,**唯一的真数字(P9 的 W₂ 范数)
已把 P9 线关死**;其余三条(status 的 🔴/⚠️、compare 的 FileNotFoundError、grep 无文件)
都是**块 77 给的路径/工具有盲区**,不是实验出了问题。工具已修、文档已改;
**尤其注意:不修的话,块 79 那条「必须 ⚪ 才排」的闸门会被这三条假警报永远卡死。**

### 判读(对应四段输出)

1. **P9 归因 —— ✅ 结案,想法本身被否。** `‖W₂‖ = 1.688`(零初始化起步,10k 门 0.0929 的
   **18 倍**),`ln_std` 的 weight/bias 也都离开初值 ⇒ 模块确实收到梯度并大幅移动,
   「没学动、锅在 `different_lr`」这条退路**关死**。−4.3 归因给 stats 池化本身;
   **§11.5-2 的条件臂 P9′ 资格取消,永不排**。已写进 §10.3(d) / §10.6 / §11.0-2。
2. **status_1m 的 3 条 🔴/⚠️ 全是假警报 —— 队列其实是空的,什么都不用"续"。**
   - `r0_s42` / `r1a_s42`:环内评测被跳过(块 60 讲过的 #334 时间差),数字靠 §3.8 离线补评,
     落在 `experiments/evals/sub1m/` —— 旧脚本只扫 `runs/` ⇒ 误报 ⚠️ 中断。
   - `r1b_s42`:runner 每次调用**覆盖写**日志,07-30 那次 auto_resume 把前一腿抹了,
     只剩续跑腿 2:12 ⇒ 误报 🔴。臂是完整的(它就是 §10.2 种子地板的一半)。
   - 已修(`TROUBLESHOOTING.md`「status_1m 把已收官的臂报成 🔴/⚠️」节):status_1m 兜底扫
     evals/、加看 ckpt mtime 跨度、
     顺带监测 `extract_teacher.py`;runner 改追加写(`>>`)。
   - **⚠️ 千万别跑它上次附的三条「续跑命令」** —— 没坏的东西不用修。
3. **r1both−r1a 比较 —— 块 77 给的路径错了**:r1a_s42 是离线补评的臂,ranks 在
   `evals/sub1m/` 不在 `runs/`(落点规矩已记 `COMMANDS.md` §3.9)。修正版在下面。
   **同一个坑也埋在块 79 的判读命令里**:那两条 compare 的基线写的是
   `experiments/runs/sub1m_r0_s42`,而 r0_s42 的 ranks 同样只在
   `experiments/evals/sub1m/sub1m_r0_s42` —— 到时候照下面第 4 步的修正版跑。
4. **C1 干净吞吐 —— 块 77 的 grep 路径也错了**:`extract_teacher.py` 只往 stdout 打,
   日志文件=当时重定向到哪(100k 那条怎么挂的我这边看不到)。**改用块文件 mtime 读吞吐,
   不依赖日志**(已记 §3.10),命令在下面。

### Do this next(H100,按序;0–2 全是分钟级只读,3–4 就是块 79 的原计划)

```bash
git pull        # 仓库根目录

# 0) 验证假警报已清:预期 r0_s42/r1a_s42 → ✅(离线补评)、r1b_s42 → ✅(续跑截断件)
#    若仍有 🔴/⚠️ ⇒ 贴回输出,先别排队
python experiments/status_1m.py

# 1) r1both−r1a(路径已修正;先 ls 确认 ranks 落点,r1a 若两处都没有则按 §3.8 补评一次 ~1 分钟)
ls experiments/{runs,evals/sub1m}/sub1m_r1{both,a}_s42/ranks_*.npz 2>/dev/null
python experiments/collect_results.py --compare experiments/runs/sub1m_r1both_s42 experiments/evals/sub1m/sub1m_r1a_s42
python experiments/collect_results.py --compare experiments/runs/sub1m_r1both_s42 experiments/evals/sub1m/sub1m_r1a_s42 --prefix ret_test/

# 2) C1 干净吞吐:块 mtime 直接算(P9 于 08-03 06:12 结束,看这之后的块;顺带看 100k 到没到 49/49)
python - <<'PY'
import glob, os, datetime
fs = sorted(glob.glob('experiments/teacher/iv2s2_100k/**/block_*.npz', recursive=True))
prev = None
for f in fs:
    t = os.path.getmtime(f)
    note = '' if prev is None else f'{t-prev:7.1f}s  {2048/(t-prev):5.2f} clip/s'
    print(os.path.basename(f), datetime.datetime.fromtimestamp(t).strftime('%m-%d %H:%M'), note)
    prev = t
PY

# 3) flash-attn A/B(§3.10 现成命令,~1 小时一张闲卡;pgrep -af extract_teacher.py 确认 100k 没还在跑)
# 4) A/B 出结果 ⇒ 块 79 的一条命令排「s43 训练 → 1M 抽取」(flash 胜就给 extract 加 --use-flash-attn)
#    ⚠️ 15 h 后块 79 的判读命令请用修正版(r0_s42 的 ranks 在 evals/ 下):
#    python experiments/collect_results.py --compare experiments/runs/sub1m_P1trained_s43 experiments/evals/sub1m/sub1m_r0_s42
#    python experiments/collect_results.py --compare experiments/runs/sub1m_P1trained_s43 experiments/evals/sub1m/sub1m_r0_s42 --prefix ret_test/
```

**读法预判(§11.0-3)**:由 §10.1 绝对分推 r1both−r1a 点估计 ≈ −1.2 raw / −1.8 DSL,
都在种子地板 δ=2.5 内 ⇒ 大概率「≈0」那支(5 分=sigmoid@512 的已知代价),以配对 CI 为准。
V100 侧照块 81 不变。

**Then report back** in from-server.md:① 新 status_1m 全文(应无 🔴/⚠️);
② r1both−r1a 两口径的 Δ 与 95% CI;③ 块 mtime 表的尾部(独占卡后的 clip/s、100k 是否已到
49/49);④ flash A/B 的两个 clip/s + cos 等价自检;⑤ 块 79 队列的 PID。

---

## 2026-08-04 (81) — ✅ 阶段 0/7/10 + gate 两口径全收;C-4 判「方差」但 LSE 是免费增益候选;下一步:probe v2 重跑阶段 10 + 多基准补已训臂

**Diagnosis(全部已落盘,细节见 `experiments/v100_sweep_results.md` §8 与改判后的 §1/§2):**

1. **gate4/P5 两口径认证通过**(raw +0.70*/DSL +0.67*,R_sum +4.2/+4.0,CI 均不含 0)⇒ **可写进论文**。
2. **gate3「测不出差别」改判收回**:raw 口径显著为负(r_mean −0.43*,R_sum −2.60*)⇒
   零训练删时序分支要付 ≈0.4 的真实小代价;P1 的立论回到 **P1trained**(训练把它买回来了)。
3. **C-4 判「方差缩减」**(oracle−mean = −1.27/−0.65,远低于覆盖门槛)⇒ P6/FILIP **训练**族维持死亡。
   **但 LSE(τ=0.05)+1.15/+1.56 同时赢 mean 和 hard-max** —— 评测侧唯一还开着的精度线索,
   且 LSE@T=8 > mean@T=16(聚合器买得动帧预算)。裸差不算数,等下面第 1 步的配对 CI。
4. A7 成本轴(224² = 0.765× MACs)、B1 词表(100%@8k,**仅 10k-zh 口径**,饱和干净反而要警惕)、
   A9/A10/A13 第二轮复现 —— 全落地。
5. **阶段 0 的 `MSRVTT 视频根` FAIL 是假警报**:`find` 默认不解引用**末级软链**,数据一直是好的
   (同轮阶段 10 照常出数就是铁证)。已修 `find -L` + `TROUBLESHOOTING.md` 末节。**数据根别动。**

**Do this next(细节与判读纪律见 `notes/v100-10k-advance-tests.md` §9):**

```bash
cd ~/mobileclip && git pull
unset MVC_DATA_DIR

# 1) 阶段 10 重跑(probe v2:τ 扫描 + 配对 CI + 两口径 + rank 落盘,~40 min)
bash experiments/run_v100_all.sh 10

# 2) 多基准补已训臂(§8 第 3 步原文照跑;软链已就位,只差 ckpt):
#    rsync 两个 ckpt(sub1m_r0_s42 / sub1m_P1trained_s42)→ 每臂 × k400/ucf101/hmdb51
#    → collect_results --compare(--prefix 按实际键名);命令块在 v100 §8 第 3 步。
```

判读要点:头条只认 **τ=0.05**(预登记,不许事后挑 τ);LSE@0.05 的 `[CI]` 行
**两口径同向且不含 0** ⇒ 认证,连帧预算结论一起进论文;任一口径含 0 ⇒ 降级为附录趋势。

**Then report back** in from-server.md:① 阶段 10 的 `[verdict]` + `[CI]` 行(整段贴回);
② 多基准的 6 份 `--compare` 表;③ 若 rsync 不通,贴 b100 侧 `ls experiments/runs/*/ckpt_02.pth`。

---

## 2026-08-03 (80) — 📋 下一轮 refinement 训练手册已落:MC2-init + lean(config/flag 全就位)

选臂理由与阶梯见 `verification-tier-1m-zh.md` §11.6;**逐步执行手册(含全部命令与决策表)=
[`next-training-round-zh.md`](../next-training-round-zh.md)**。本轮落了:
`disable_temporal_embedding` flag(`models/mobileviclip_small.py`,P1 同模式,缺省不变)、
`sub1m_mc2init_s42.py`(双塔换 MobileCLIP2-S2 init)、`sub1m_lean_s42.py`(两减法合体)。

**现在就能做(不占训练卡,~10 分钟)** —— 手册 §0:

```bash
cd ~/mobileclip && git pull
wget -O checkpoints/mobileclip2_s2.pt \
  "https://huggingface.co/apple/MobileCLIP2-S2/resolve/main/mobileclip2_s2.pt"
# 然后跑手册 §0 的键集一致性闸门(强制)+ 配置解析闸门
```

**当前队列(s43 训练 → 1M 抽取)跑完后**:先手册 §1 冒烟门(10 分钟),再:

```bash
python experiments/status_1m.py        # ⚪ 才排
nohup bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_mc2init_s42 sub1m_lean_s42 \
  > experiments/runs/verify_1m_round2.log 2>&1 &
echo "round2 PID = $!"
```

**Then report back** in from-server.md:手册 §0 闸门 A 输出、§1 三行断言、跑完后 §3 的 CI 表。

---

## 2026-08-03 (79) — ▶️ 维护者拍板:现在就训改进版模型 —— 一条命令排「P1trained_s43 训练 → 1M 教师抽取」

改进版模型 = **P1trained**(删时序分支,全轮唯一过关的改动)。它的下一步训练就是
`sub1m_P1trained_s43`(config 已在,#421)—— 这既是「训改进版」也是它上 10M 前的第二个 seed 保险。
再往上的两档(**lean** = P1trained+删 `temporal_embedding`;**10M 论文数版**)都还缺 config/flag,
回「写吧」我落地;10M 按 §11.4 仍要等两道保险。

**先跑 flash A/B(块 77 第 2 步,~1 小时)定抽取用不用 `--use-flash-attn`,然后一条命令排完两件事**
(runner 串行;训练 ~15 h 完了自动接抽取 ~39 h,总 ~55 h 不用管):

```bash
cd ~/mobileclip && git pull
python experiments/status_1m.py        # 必须 ⚪ 才排
nohup bash -c '
  bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_P1trained_s43
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k python experiments/extract_teacher.py run \
      --anno data/anno_pretrain/internvid_v1.jsonl \
      --out  experiments/teacher/iv2s2_1m \
      --teacher-ckpt checkpoints/InternVideo2-stage2_1b-224p-f4.pt \
      --shard-id $k --num-shards 4 --batch-size 8 --num-workers 8 --amp bf16 \
      --per-frame --subset-ts 2,4,8 > experiments/teacher/extract_1m_$k.log 2>&1 &
  done
  wait
' > experiments/runs/queue_0803.log 2>&1 &
echo "queue PID = $!"
```

(flash A/B 若胜,给上面 extract 那条加 `--use-flash-attn` 再启动;全语料只用一种设置。)

**训练跑完的判读(15 h 后,CPU 即可)** —— 跨 seed 对 `r0_s42`,§10.3(b) 惯例,δ=2.5:

```bash
python experiments/collect_results.py --compare experiments/runs/sub1m_P1trained_s43 experiments/runs/sub1m_r0_s42
python experiments/collect_results.py --compare experiments/runs/sub1m_P1trained_s43 experiments/runs/sub1m_r0_s42 --prefix ret_test/
```

两口径点估计为正或 CI 下界 > −2.5 ⇒ 非劣在第二个 seed 复现,10M 的第一道保险到手;
显著负过 δ ⇒ 保险丝熔断,10M 停,先归因。

---

## 2026-08-03 (78) — 🔍 「两台都没有 k400/ucf101/hmdb51」先别下结论 —— §5/§6 的数字就是在它们上测的

**矛盾**:`v100_sweep_results.md` §5/§6(P7 分辨率否决、多基准锚点)是这三个基准的**实测数**
(07-31,#414)。最可能的解释:视频一直在 V100 旧 `MVC_DATA_DIR` 根(`/home/vca1/datasets`)下,
#418 修复 `unset` 后只软链了 10k ⇒ 三个基准被**同一个根机制**断开,「没有」= 路径断开。

**V100 上先探(1 分钟)**:
```bash
find /home/vca1/datasets -maxdepth 2 -type d \
     \( -iname "*kinetics*" -o -iname "*ucf*" -o -iname "*hmdb*" \) 2>/dev/null
```

**找到 ⇒ 三条幂等软链(前缀按 find 输出改),readiness 应转 `[OK ]`,然后 §8 第 3 步照跑**:
```bash
ln -sfn /home/vca1/datasets/kinetics_400_val_10s_320p data/kinetics_400_val_10s_320p
ln -sfn /home/vca1/datasets/ucf101                    data/ucf101
ln -sfn /home/vca1/datasets/hmdb51                    data/hmdb51
python experiments/check_readiness.py
```

**真没有(find 为空)⇒** ① 🔴 把空输出贴回 from-server.md,§5/§6 的出处要审计;
② 兜底二选一:下载 UCF101(~6.9 GB)+ HMDB51(~2 GB)补齐(k400 先跳过),
或放弃多基准保险、P1trained 只靠 MSR-VTT 配对 CI + s43。

**b100 队列不变**(flash A/B → 1M 教师抽取,COMMANDS §3.10);
块 77 里「若 b100 有三基准就在 b100 评」那一条**作废**。

---

## 2026-08-03 (77) — 🏁 全轮收官后的下一步:H100 = 一条确认臂 + C1 阶梯;V100 = 阶段 10/7 重跑 + 多基准面

**现状一句话**:1M 层 8 臂全收(§10):5 个精度改进**全否**,唯一通过的是**减法**
(P1trained 非劣);H100 队列已空。⇒ 配方微调这条线停了,下一轮的完整排队理由见
[`verification-tier-1m-zh.md`](../verification-tier-1m-zh.md) **§11**(H100)与
[`v100-10k-advance-tests.md`](../v100-10k-advance-tests.md) **§8**(V100)。

### H100(b100)· 按序

```bash
cd ~/mobileclip && git pull
# 1) 零成本收尾(§11.0,全部只读/分钟级):
python experiments/status_1m.py                     # 应显示 ⚪ 可排新任务
python -c "import torch;sd=torch.load('experiments/runs/sub1m_P9_s42/ckpt_02.pth/mp_rank_00_model_states.pt',map_location='cpu');sd=sd.get('module',sd.get('model',sd));[print(k,float(v.float().norm())) for k,v in sd.items() if 'temporal_pool' in k]"
python experiments/collect_results.py --compare experiments/runs/sub1m_r1both_s42 experiments/runs/sub1m_r1a_s42
python experiments/collect_results.py --compare experiments/runs/sub1m_r1both_s42 experiments/runs/sub1m_r1a_s42 --prefix ret_test/
grep -A2 "block 1[1-9]/" experiments/runs/extract_teacher*.log | tail -20   # C1 干净吞吐(P9 结束后的 block)

# 2) 确认 ⚪ 后排唯一的确认臂(config 本次已建,15 h):
nohup bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_P1trained_s43 > experiments/runs/verify_1m_p1s43.log 2>&1 &
echo "verify-1m PID = $!"
```

**读法**:P9 的 `W₂` 范数明显长大 ⇒ stats 池化想法本身被否;仍≈0 ⇒ 归因 `different_lr`、
想法未被检验(§11.0-2)。`r1both−r1a ≈ 0` ⇒ 那 5 分就是 sigmoid@512 的已知代价(§11.0-3)。
P1trained_s43 跑完按 §11.1 跨 seed 对 `r0_s42` 配对判读(δ=2.5)。

**C1 阶梯(§11.2,§10.4 指定的主方向)**:① 干净吞吐(上面那句 grep)→ 定 U7-1;
② `internvid_v1` 1M 分片教师嵌入(4 卡 `--num-shards 4 --shard-id 0..3`,悲观 ~1.6 天)
可在 s43 臂之后排;③ **训练侧接线(Dataset / 投影 / vtc_kd + 先修 `loss_weight` 假开关)
还没有代码 —— 回「写吧」我就落地**,那是 KD 臂(⑤)的前提。

### V100(101vector)· 按 §8 的序

```bash
cd ~/mobileclip && git pull
unset MVC_DATA_DIR
bash experiments/run_v100_all.sh 0 10      # 阶段 10:C-2/C-4 —— 唯一可能再开一条精度线的实验
bash experiments/run_v100_all.sh 7         # 阶段 7:A7 成本轴 + B1 词表覆盖率
# 然后照 §8 第 3 步给 r0_s42 / P1trained_s42 补 k400/ucf101/hmdb51 三个测量面(命令在 §8)
# 第 4 步:gate3/gate4 的 raw 配对 CI(块 75 开头那四条,若上轮没跑)
```

**需要我先写代码的两件(§8 第 5 步 / §11.3)**:BN 重校准 CLI、P1 结构性旁路 + 导出 diff;
另有可选臂 `sub1m_lean`(P1trained + 删 `temporal_embedding`)要一个 ~5 行训练期 flag。
**回「写吧」并指明哪件。**

**Then report back** in from-server.md:① `status_1m.py` 全文;② P9 的 `temporal_pool.*` 范数;
③ r1both−r1a 两口径 CI;④ C1 block 11+ 的干净 clip/s;⑤ 阶段 10 的 C-2/C-4 判别结果;
⑥ P1trained vs r0 的三基准配对 CI;⑦ 要不要写(以及先写哪个):KD 接线 / BN CLI / P1 旁路 / lean flag。

---

## 2026-07-31 (76) — ✅ **10k 门全过 ⇒ H100 两条臂现在就能排**;阶段 10 挂在数据根冲突(已修)

**结论先行**:阶段 0/1/6 全 PASS。两条强制检查都过了 ——
**P1**:`flag 生效,40 个 layer_scale_1d 已冻结`;
**P9**:`‖W2‖ = 0.092874`,比阈值 `1e-4` 大近 1000 倍 ⇒ 二阶统计**确实学动了**,
不是 nA1 那种「alpha ≲0.01 ⇒ 结果不可判读」的空跑。
**链路 go/no-go 通过,H100 那两条 15 小时的臂可以上。**

### 1. H100:**先查现状,再排队**(顺序不能反)

```bash
cd ~/mobileclip && git pull
python experiments/status_1m.py          # ← 30 秒,只读,训练在跑时也能跑
```

只有当它打印 **`⚪ 没检测到在跑的训练进程 ⇒ 可以排新任务`** 且**没有 🔴/⚠️ 行**时,才执行:

```bash
bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_P1trained_s42 sub1m_P9_s42
```

**为什么加这一步**:`run_verify_1m.sh` 是**串行**的,盲排会和在跑的臂抢同一组卡;
而且「跑完」不等于「跑满」—— `sub1m_r1b_s42` 曾**只跑 2:12:39**(参考 15:11:58)
就出现在完成清单里,那种臂的结论是**无效**,不是「结果为负」。
把这两件事交给 `status_1m.py` 自动判,不靠人扫日志(用法见 `COMMANDS.md` §3.7b)。

它输出五种状态:`未排队` / `🟢 运行中` / `✅ 已完成` / `🔴 可疑`(有结果但墙钟过短)/
`⚠️ 中断`(有 ckpt 无 eval,`auto_resume` 可续),后两种会直接附上续跑命令。

**如果 🔴/⚠️ 里出现 `sub1m_r1b_s42` 或 `sub1m_nA1_s42`**:先把它们续完再排新臂 ——
§8 把这两条列为「仅剩的两条方向未知臂」,它们的结论是整个 C3 配方判断的支点,
用一条只跑了 2 小时的臂去下结论,比没有结论更糟。

排队理由见 [`verification-tier-1m-zh.md`](../verification-tier-1m-zh.md) §9。
提醒:1M 档**只有否决权,不能排名**(§2)—— 它能判 P9/P1 是不是白给,
判不了「P9 比 P1 好多少」。

### 2. V100:阶段 10 重跑(先 pull,数据根冲突已在 stage 0 自动修)

```bash
cd ~/mobileclip && git pull
unset MVC_DATA_DIR            # ← 关键:别再 export 它,理由见下
bash experiments/run_v100_all.sh 0 10
```

### 🔴 数据根冲突:`export MVC_DATA_DIR` 是个**换边**,不是修复

上一轮我让你 `export MVC_DATA_DIR=/home/vca1/datasets`。它修好了 10k 训练,
**同时打掉了所有 MSR-VTT 评测**:

```
RuntimeError: Error reading /home/vca1/datasets/MSRVTT_Videos/video7020.mp4...
```

因为 `configs/data.py` 把**所有**数据集拼在**同一个**根下(`__data()`, :20/:29):

| 数据集 | 配置里的路径 | 101vector 上的真实位置 |
|---|---|---|
| 10k pilot | `$ROOT/internvid_zh_10k/videos` (:107/:114) | `/home/vca1/datasets/internvid_zh_10k/videos` |
| MSR-VTT 1K-A | `$ROOT/MSRVTT_Videos` (:50) | 仓库本地 `data/MSRVTT_Videos` |

两份东西**不在同一个父目录下**,所以 `MVC_DATA_DIR` 取任何值都会顶掉另一个 ——
一个变量,两个根,无解。**正解是补一条软链,让两者落进同一个 `data/`**:

```bash
ln -sfn /home/vca1/datasets/internvid_zh_10k data/internvid_zh_10k
unset MVC_DATA_DIR
```

`data/` 是 gitignore 的,软链幂等,`rm data/internvid_zh_10k` 即可撤销。
`run_v100_all.sh` 阶段 0 现在**自动做这件事**:缺软链就建、`MVC_DATA_DIR` 里没有
`MSRVTT_Videos` 就本次 unset、并对**本轮真正会用到的每一个**视频根分别判 PASS/FAIL。
完整记录见 `TROUBLESHOOTING.md` 的
「`MVC_DATA_DIR` root collision」一节。

**顺带**:阶段 10 这次崩在 dataloader,说明 #416 的文本路径修复**已经生效**
(文本编码跑在取视频之前,先过了才轮到解码器报错)——
`'Tensor' object has no attribute 'attention_mask'` 那条已经不会再出现。

### 这两轮的两条教训(已写进阶段 0)

1. **能自动判定的事,不许只打印。** 第一版阶段 0 把 `MVC_DATA_DIR=<未设→data>`
   打在屏幕上还记 PASS,两次冒烟就死在它上面。
2. **预检要验完这一轮会用到的全部数据根,不能只验刚出过事的那一个。** 第二版只验了
   10k,于是把 MSR-VTT 换到了坏的一边。

---

## 2026-07-31 (75) — V100 两小时执行单(逐条命令)+ H100「新想法」排队;顺带:**gate3/gate4 早就跑完了,没人写下来**

**背景**:P11 全 40 块测完、判据②触发、该线已关(#400),P14 随之关闭
⇒ **round4-P7(分辨率)是唯一存活的效率线**。完整计划 #401:
执行单 [`v100-10k-advance-tests.md`](../v100-10k-advance-tests.md) **§6**,
1M 排队 [`verification-tier-1m-zh.md`](../verification-tier-1m-zh.md) **§8/§9**。

### 🔴 先看这个:两条早就测完、却没进任何文档的结果

`collect_results.py` 的输出里一直有 `eval_gate3` 和 `eval_gate4`。查了
`gate_checks.py:534-535`:**`gate3` = 删时序分支(融合核 := δ)、`gate4` = 置零
`temporal_embedding`** —— 也就是 **P1 和 P5 各自的那一次判定评测**。基线 raw 61.55 / DSL 62.28:

| 臂 | 是什么 | raw r_mean | DSL r_mean | Δ raw | Δ DSL |
|---|---|---|---|---|---|
| `eval_gate3` | **P1**:删时序分支 | 61.12 | 61.95 | **−0.43** | **−0.33** |
| `eval_gate4` | **P5**:删 `temporal_embedding` | 62.25 | 62.95 | **+0.70** | **+0.67** |

- **P1 的预登记判据是「Δr_mean > 1.0(两口径同号)⇒ 杀」。实测 −0.43 / −0.33,
  远在阈值内 ⇒ P1 通过,时序分支可删。** 代价是 −0.43 r_mean,换来
  **440 个 ONNX 节点(27.8%)、80 次全张量转置、CPU mixer 路径 −18%**。
- **`temporal_embedding` 删掉之后两个口径都更好(+0.70/+0.67,同号)。** 与 F2 的预测一致
  (加性码被 mean readout 湮灭),但**实测是「删了更好」而不只是「删了不亏」**。

⚠️ **这两条都还没走配对 bootstrap,现在只是裸差。** 先补上(每条约 1 分钟,纯前向):

```bash
cd ~/mobileclip && git pull
python experiments/collect_results.py --compare eval_gate3 eval_base
python experiments/collect_results.py --compare eval_gate3 eval_base --prefix ret_test/
python experiments/collect_results.py --compare eval_gate4 eval_base
python experiments/collect_results.py --compare eval_gate4 eval_base --prefix ret_test/
```

**判读**:两口径同号 + CI 不含 0 ⇒ 可写进论文。gate4 若显著为正,那是**免费的 +0.7**,
而且它让 C2 的「未见过的 T」从插值技巧变成**结构性**的(塔里再没有 T 形状的参数)。

---

### V100 · 两小时执行单(照序跑,约 100 分钟)

**第 0 步(1 分钟)—— 它决定第 5 步做不做:**
```bash
cd ~/mobileclip && git pull
export MVC_DATA_DIR=/home/vca1/datasets      # 按本机实际改
python experiments/check_readiness.py
```
盯 `msrvtt`(应当 `[OK ]`)以及 **`k400` / `ucf101` / `hmdb51` 这三行**。

**第 1 步(5 分钟)· A12 BN 双更新** —— 跑一步训练,打印任一 BN 的 `num_batches_tracked`;
`== 2` 即确认。**后果**:此后所有 BN 重校准必须 `with_cp=False`。
⚠️ **不要因此改动正在跑的 1M campaign**(见下方「不要动」)。

**第 2 步(20 分钟)· A7 分辨率成本轴** —— `torch.utils.flop_counter`,随机输入,零数据,
对 `img_size ∈ {128,160,181,224,256}` 各数一次 MACs;顺带复核「每块 157.29 MMACs」。

**第 3 步(5 分钟)· 🔺 P7 分辨率精度轴 —— 唯一存活效率线的第一份实测:**
```bash
bash experiments/run_resolution_sweep.sh
python experiments/collect_results.py --dirs . --out experiments/resolution_all.md
```
脚本会同时改 `inputs.image_res` 与 `model.vision_encoder.img_size`(只改一个会静默不一致)。
🔴 **这是趋势筛不是判据**:换分辨率会改**每一层**的激活统计,BN 漂移远大于 P11 删块。

**第 4 步(10 分钟)· B1 词表覆盖率** —— 见 `v100-10k-advance-tests.md` B1;结论标 **10k-zh 口径**。

**第 5 步(30 分钟,仅当第 0 步显示三个基准 `[OK ]`)· 多基准 R-Sum 探路** ——
用官方权重把 k400 / ucf101 / hmdb51 各跑一遍(`COMMANDS.md` §3.11 配方,换 config)。
**为什么值**:`verification-tier-1m-zh.md` §6 已把多基准 R-Sum 从「纪律」升级为**「必须」**
(MSR-VTT 1k 在 1M 尺度上 10× 数据只涨 +0.82,落在 ±1.6 带内);DiDeMo/ANet 的 anno 拿不到,
**但这三个的 anno 一直是齐的**。通了 ⇒ 每条已训 ckpt 多三个独立测量面,
**比再花 15 小时训一条臂划算得多**。

**第 6 步(有余量才做)· A5 空间 mixer 的 ρ 直方图** —— 单独决定 AC9 的生死,照 §10.4 的收紧规格。

---

### H100 · **新模型改动**里,哪些真的需要训练(答你的问题)

第四/五轮的提案里,**绝大多数是零训练的评测项,不该占 H100**:

| 提案 | 需要训练? | 怎么办 |
|---|---|---|
| P1 删时序分支 · P5 β/`temporal_embedding` · P6 聚合 · P7 分辨率 · P13 词表/量化 · P3/P15 等价改写 | ❌ **不需要** | 全在 V100 上评测决定(P1/P5 **已经测完**,见本块开头) |
| **P9 统计池化** | ✅ | **排第 1** |
| **P8 图文回放(T=1 通道)** | ✅ | **排第 2** |
| **P1-trained(整条时序分支删掉后重训)** | ✅ | **排第 3** |
| **P4 修正版 AC5** | ✅(还要自带对照 = 2 臂) | **排最后** |

**建议排队(等当前在跑的臂空出来之后):**
```bash
# 先确认当前在跑什么(块 74:r1b_s42 曾被信号打断并重挂)
nvidia-smi; ls -lt experiments/runs/ | head
# 然后追加(runner 串行):
bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_P9_s42 sub1m_P8_s42
```
⚠️ **`sub1m_P9_s42` / `sub1m_P8_s42` 这两个 config 还不存在** —— 需要我先写。
理由与优先级见 [`verification-tier-1m-zh.md`](../verification-tier-1m-zh.md) **§9**。
**回一句「写吧」我就把两个 config + 必要的模型改动落地。**

---

### 不要动的两件事

1. **A12 的 BN 双更新(动量实际 0.19 而非 0.10)不要在 campaign 中途修** ——
   它对所有臂一视同仁、不偏袒任何一臂;**中途改会让新旧 run 不可比**,毁掉已花掉的 GPU 小时。
   记进论文 limitations,不修改。
2. **不要把 8 条 `sub1m_*` 排满** —— §6 已定:1M 层**只能否决,不能排序**。

**Then report back** in from-server.md:①第 0 步 `check_readiness` 的三行;
②第 3 步分辨率表;③开头那四条配对 bootstrap 的 CI;④要不要我写 P9/P8 的 config。

---

## 2026-07-30 (74) — ⚠️ `sub1m_r1b_s42` 被信号打断,整条 runner 死了;续训重跑(挂后台)

**发生了什么**:`sub1m_r1b_s42.log` 尾部 `SignalException: Process 4004093 got signal:` ——
训练被**终止信号**打断(不是代码 bug、不是 OOM)。根因 = **前台跑的 runner + tunnel 掉线/Ctrl-C**。
因为 runner 是 `set -e`,被打断的臂返回非零 ⇒ **整条 `run_verify_1m.sh` 退出** ⇒ `nA1_s42` 和
所有 seed-43 臂**没启动**。`r0_s42`/`r1a_s42` 已跑完,不受影响。(详见 `TROUBLESHOOTING.md` 新增段。)

**Do this next:**
```bash
cd ~/mobileclip && git pull origin main
# 1) 先确认 0–3 没残留进程占显存(应接近 0):
nvidia-smi --query-gpu=index,memory.used --format=csv
# 2) 挂后台续跑剩余 6 臂(r1b_s42 会从 ckpt_latest 续训,其余从头;runner 已内置 auto_resume True):
nohup bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_r1b_s42 sub1m_nA1_s42 sub1m_r0_s43 sub1m_r1a_s43 sub1m_r1b_s43 sub1m_nA1_s43 > experiments/runs/verify_1m_resume.log 2>&1 & echo "verify-1m PID = $!"
```
**关键:这次必须 `nohup ... &` 后台跑**(上次前台是被打断的根因)。记下 PID —— 共用 root,
以后要停只能按这个 PID / 其 PGID,**严禁 pkill**。

**验证续训生效**:重跑后 `grep -m1 -E "auto_resume|Resume|ckpt_latest|Load pretrained" experiments/runs/sub1m_r1b_s42.log`
—— 看到 resume/加载既有 ckpt = 续上了;若是从 epoch 0 step 0 全新开始,说明 r1b 之前没存到 ckpt(那也没关系,重训即可)。

**Then report back** in from-server.md:上面 nvidia-smi 的四行 + 新 runner 的 PID + `sub1m_r1b_s42.log` 头几行(确认续训/全新)。

---

## 2026-07-30 (73) — 1M 验证层首条臂 A/B:`sub1m_r1a`(SigLIP sigmoid)**显著输** `r0`(仅 seed 42)

**结果**(配对 bootstrap,§3.9;补评已核实权重加载 `Load pretrained model from .../sub1m_r1a_s42/ckpt_02.pth/mp_rank_00_model_states.pt`):
- **DSL 口径 `ret_test/`**:r_mean Δ **−0.95** [−1.68,−0.20]、R_sum Δ **−5.70** [−10.10,−1.20] ⇒ **显著为负**。
- **raw 口径 `ret_test_emb/`**:同向,r_mean Δ −0.63 [−1.40,+0.13]、R_sum Δ −3.80 [−8.40,+0.80](卡边界,未显著)。
- **两口径一致为负** ⇒ r1a 不如 r0。**符合理论**:sigmoid 要 ~32k batch 才发威,本层全局 batch 512 远低于甜区。

**Do this next:**
1. r1a 无立即动作 —— 定案还差 **seed 43**(`sub1m_r1a_s43` − `sub1m_r0_s43`),由 runner 自动训,别手动插队。
2. `r1b_s42`/`nA1_s42` 是 post-fix 启动、会自动出评测(带 `ranks_*.npz`)。**跑完后**跑这两条(nA1 的真基线是 r1b,不是 r0):
```bash
python experiments/collect_results.py --compare experiments/evals/sub1m/sub1m_nA1_s42 experiments/evals/sub1m/sub1m_r1b_s42
python experiments/collect_results.py --compare experiments/evals/sub1m/sub1m_r1b_s42 experiments/evals/sub1m/sub1m_r0_s42
```
   nA1 收尾**连最终 `alpha` 一起报**(§怎么读结果 rule 4;`alpha ≲ 0.01` = 惊奇门退化成 mean)。

**已落库**:`notes/verification-tier-1m-zh.md` §7(两口径表)+ `ROADMAP.md` 结果表新增一行。

**Then report back** in from-server.md:`sub1m_r1b_s42` / `sub1m_nA1_s42` 是否已出 `ranks_*.npz`,以及上面两条 compare 的输出。

---

## 2026-07-29 (72) — ⚠️ 并行合并事故通告: #363 陈旧树 squash 曾覆盖删除 #354–#361 多个文件(已部分恢复)

**事故**：PR #363（arch round 4）的分支基于过期 main，squash 后把它碰过的共享文件全部
退回旧版：删掉了 `docs/recaption-ab-pipeline_zh.md` 整个文件、run_v100.sh 12 行计时、
计划 §5.1c 回填、TROUBLESHOOTING rm 配方、from-claude 块(69)(71)、#359 的判错原因列
（make_pipeline_tsv/make_style_compare/make_summary_table/test）。**以上 10 个文件已由
本 PR 从 fc2c86b(#361) 整文件恢复**，#363 自己的新增（architecture-round4.md 等）未动。

**尚未恢复、请 R1/arch 线会话自查**（#363 同样回退了你们的内容，但这三个文件混着
#363 的合法新增，不便代并）：`COMMANDS.md`（配对 bootstrap 判读命令段被删）、
`experiments/ROADMAP.md`（GATE-0 结案行被删）、`notes/r1-implementation-plan.md`
（#347/#354–#356 的 paired-bootstrap 读法段被删）。恢复配方：
```bash
git diff fc2c86b 49af082 -- COMMANDS.md experiments/ROADMAP.md notes/r1-implementation-plan.md
# 逐段核对: '-' 行里属于 #346-#356 的内容 = 被误删,从 fc2c86b 取回后再叠 #363 的新增
```

**防再犯（所有会话）**：开分支前 `git fetch && git rebase origin/main`；合并后不但要
grep 自己的块，还要 `git ls-tree origin/main -- <自己的文件>` 验文件还在。

---

## 2026-07-29 (71) — [101vector] 细粒度耗时收到并已回填(doc §6.1 + 计划 §5.1c);下一步还是 GDINO + 50 片目检

你贴的 N=10 双臂 12 行计时已进仓库：全表在 `docs/recaption-ab-pipeline_zh.md` §6.1，
决策摘要进了计划 §5.1c 行名注。读数三句话：**A 大头=②派生 58 s/片 ≫ ④b2 直融 31.8**
（§5.9 的裁减对象没变）；**B 全链 35.0 = A(118.1) 的 0.3×**；坍缩哨兵两臂绿、B 投票
0 fallback（数票前提健康）。④b3 B>A、⑤c A>B 是 10 片噪声级，不用管。

**Do this next（两件都是老待办，顺序执行）：**
```bash
# 1) GDINO 本地下载+scp 装进 ~/.cache/huggingface/hub/ —— 配方见 TROUBLESHOOTING
#    「hf-mirror 不可达」节。装完删验真+修复+终稿三组缓存让 ⑤ 真跑 GDINO
#    (refine 档按 video 续跑,不删会复用 OCR-only 旧口径的修复结果):
rm -f outputs/audit_10k/pipe*_verified.jsonl \
      outputs/audit_10k/pipe*_r.*.jsonl outputs/audit_10k/pipe*_final.jsonl
# 2) 升到 50 片出目检包(§0.9 行动序②; 这 10 片缓存自动复用,只补 40 片):
SAMPLE_OFFSET=300 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
```

**Then report back**：目检四查结论（B 漏动画/游戏标记？漏关键交互物？物体栏堆琐碎？
短稿核心 vs 众数遍核心一致率）+ 单选导出 TSV。GDINO 装上后 ⑤ 行耗时若明显涨，把新
「⏱」块再贴一次，我更新 §5.1c 口径。

---

## 2026-07-29 (70) — 答"测的是不是判错的caption":是, 试点只跑 RECAPTION 桶; 评审物料新增「判错原因」列

**回答:是。** 我们所有试点(50 片 pipeline-compare、表格页)都只从
`v0_2/recaption_tasks.jsonl` 里抽——它是 apply_buckets 的 RECAPTION 桶队列
(S2 审计判「英文 caption 与画面不符」的片, TARGET_PRECISION=0.3 校准, 全桶 2,532/10k)。
`pipeline-sample` 就是 `head -N`(或 SAMPLE_OFFSET 偏移)这个队列;评审页也一直把
判错的 en 源和旧译并排放给你对照。

**新增(已合入):** compare 表 / 单臂表 / 表格页(html/csv/xlsx)在「原zh」后面都加了
**「判错原因」列**(来自 buckets 的 reasons)——目检时直接对着错因看新字幕是否修好,
不用再回 TSV 查。表格页角标升 **vT20260729a**。重新生成(全部缓存命中, 分钟级):
```bash
N_SAMPLE=50 PIPE_LEAN=1 bash tools/data_audit/run_v100.sh pipeline-compare
bash tools/data_audit/run_v100.sh summary-table
tar czf outputs/audit_10k/table.tgz -C outputs/audit_10k table_compare
```

**登记的空白(可选, 待你拍板):** 我们从没在 KEEP 桶(判「好」的 caption)上跑过管线
作对照——部署上不需要(只替换 RECAPTION 桶), 但作为标定对照有价值:若在判好的片上
我们的重述≈旧 caption, 说明分桶和重述都靠谱。要做的话一条命令就能抽 50 片 KEEP 对照,
回「做 KEEP 对照」即可。

## 2026-07-29 (69) — [101vector] A/B 管线图解文档上线 + 计时拆到每小步,请回一轮冷样本细粒度耗时

**新东西（已合并 main）：**
1. **`docs/recaption-ab-pipeline_zh.md`** —— A/B 两臂分步图解 + 模型清单 + 7 条 prompt
   全文 + 计时读法，一页看懂现行管线（§0.9 定版的落地说明书）。
2. **计时拆细**：`pipeline-sample` 的 `${P}_timing.tsv` 从 7 行拆到 12 行，一行=一个
   真实小步骤。原来 ④b 一行看不出「直融两调用 vs 投票打包」谁贵、⑤ 一行看不出三路
   验真各占多少——现在 ④b1 计票/④b2 直融/④b3 投票打包、⑤a 视图/⑤b 直融稿/⑤c 投票稿、
   ⑥TSV/⑦评审包 全部分列。汇总/外推逻辑不变，`pipeline-compare` 结尾照常打印两臂对照。

**Do this next（约 1.5–2.5 h，多在 ①②④b 的 GPU 段）：**
```bash
git pull origin main
# 冷样本细粒度计时,双臂一条命令。offset=300 避开 §5.1c 那轮已缓存的 200-249 段;
# 若中断,原命令重跑即按 video 续走。GDINO 未装也照跑(⑤ 行=OCR-only 口径,回填时会标注)
SAMPLE_OFFSET=300 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
# 赶时间变体(维护者已确认可用): N_SAMPLE=10, ~40min。①行 s/片 会被模型装载虚高约 2×、
# ②行约 +15%(每个单卡段重装一次 8B, 10 片摊不薄), 看各小步相对贵贱够用, 精确全量外推
# 仍以 50 片为准; 之后升 50 只补跑 40 片(按 video 续跑), 目检包(§0.9 行动序②)要 50 片的量
```

**Then report back**：把结尾「⏱ A/B 两臂分段耗时」整块贴回 from-server.md（或直接贴
`outputs/audit_10k/pipe_timing.tsv` + `pipe_sl_timing.tsv` 原文）。我用它回填计划
§5.1c 的 slot 行与细粒度表——这也是 §5.9 剩余精简项（派生 55 s/片、投票 LLM）拍板的
数据底座。这轮的 `bundle_compare/` 同时就是 §0.9 行动序②的目检材料，可顺手
`tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare` 取回。

---

## 2026-07-29 (68) — gifs/ 文件名改带表格序号前缀(001_), 文件夹按名排序=表格行序

按你要求:GIF 文件名现在是 `001_<video>.gif` `002_...`,**文件夹里按名排序就是表格顺序**,
按行号即可定位。配套:表格页视频格下方、csv/xlsx 首列都加了 `序号`(001 起,三处一致)。
旧命名的既有 GIF 会**自动改名复用,不重新编码**——重跑还是秒级:
```bash
bash tools/data_audit/run_v100.sh summary-table
tar czf outputs/audit_10k/table.tgz -C outputs/audit_10k table_compare
```
(序号=review_compare.tsv 的行序,同一抽样文件下每次重跑稳定不变。)

## 2026-07-29 (65) — `ranks_*.npz` 没生成,是因为**那次 pull 根本没跑**(我给的 `cd ~/mobileclip` 是错的)

你贴的日志第一行:

```
-bash: cd: /root/mobileclip: No such file or directory
```

`cd ~/mobileclip && git fetch ... && git reset ...` —— **`cd` 失败,`&&` 把整条链短路**,
fetch/reset 一条都没执行。仓库还停在 `7fc9183`,而配对 bootstrap 是 **#346**,更晚。
所以没有 rank 落盘的代码,`ls ranks_*.npz` 自然找不到。

**这是我写错的**:b100 的仓库在 `/workspace/mount/b100_zaip_data/rby/mobileclip`,
`~` 是 `/root`,没有 `mobileclip`。仓库里 **39 处 `cd ~/mobileclip` 已全部改成**
`cd "$(git rev-parse --show-toplevel)"`(机器无关),并把这个坑写进了 TROUBLESHOOTING。

### 重跑(去掉 cd 即可,你本来就在仓库里)

```bash
git fetch origin main && git reset --hard origin/main
git log --oneline -1        # ← 必须看到 #346 或更新;还是旧号就是没 pull 到,贴回来
source experiments/env_b100.sh
export CUDA_VISIBLE_DEVICES=3
bash experiments/run_eval.sh small msrvtt \
     experiments/runs/sub1m_r0_s42/ckpt_02.pth \
     experiments/evals/sub1m/sub1m_r0_s42 num_frames_test 8 batch_size_test 32
ls -l experiments/evals/sub1m/sub1m_r0_s42/ranks_*.npz    # ← 这次应该有两个
```

**以后每次 pull 之后先 `git log --oneline -1` 确认 commit 真的动了**,再跑后续步骤 ——
这条已写进 TROUBLESHOOTING。

### 顺带:这次评测有个好消息

数字与上一次**逐位一致**(`ret_test/` 42.7/40.0/61.28,`ret_test_emb/` 41.8/39.0/60.65)
⇒ **评测流程是确定性的**,同一 ckpt 重跑不会漂。后面各臂的 Δ 不用担心被评测噪声污染,
所有波动都来自训练本身。

## 2026-07-29 (67) — 表格新增 csv + xlsx 导出:summary-table 一条命令三种产物

答复「能不能存成 xlsx/csv」:**能,已实现**。`summary-table` 现在同目录一次出三样:

- `table.html` —— 动图版(GIF 会循环播放),展示用;
- `table.csv` —— utf-8-sig 逗号分隔,Excel/WPS **双击直开中文不乱码**;多条重述在
  单元格内按 `1. 2. 3.` 换行(Excel 里开「自动换行」即逐行显示);列=video|en源|原zh|
  A重述|A长稿|A直融|B重述|B长稿|B直融|video_path;
- `table.xlsx` —— **audit_env 里 `pip install openpyxl` 后才出**(纯 python 小依赖,
  不碰 torch,装了重跑本命令即补出):首列内嵌 GIF **首帧静态缩略图**(Excel 不播动图,
  动图仍看 html),表头加粗冻结、全列自动换行。没装 openpyxl 时 csv/html 照出,只提示跳过。

**服务器(按 (66) 对齐并跑完 compare 后):**
```bash
pip install openpyxl        # 一次性, 可选(只为 xlsx)
bash tools/data_audit/run_v100.sh summary-table   # GIF 有缓存, 重跑秒级
tar czf outputs/audit_10k/table.tgz -C outputs/audit_10k table_compare
```

(块号说明:R1 线 07-29 又用了 (63)(64),与我此前的 (64) 表格页/(65) (66) 是不同内容,
按日期+标题区分;我继续单调递增用 (67)。)

## 2026-07-29 (64) — 配对 bootstrap 写好了(G4 结案):**分辨率提升约 11 倍**,判据换掉 ±1.6

昨天那个"MSR-VTT 排不出臂序"的问题解决了。

**核心**:两条臂评的是**同一批**查询 ⇒ **配对**比较。同一次重采样里两臂用**同一组下标**,
共享方差被抵消。实测同一对臂,R-Sum 的 95% CI 半宽:

| | 半宽 |
|---|---|
| **配对** | **±1.5** |
| 无配对 | ±17 |

⇒ **约 11 倍**。一个 **+0.80 的 R@1 差可以判显著**,而用无配对带永远判不出来。

### 你要做的两件事

**① 重跑一次 `sub1m_r0_s42` 的评测**(纯前向,~1 分钟)——
逐查询 rank 是**今天才开始存的**,那条臂评的时候还没有,所以现在没法比。
(往后跑的臂都会自动存,不用管。)

```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
source experiments/env_b100.sh
export CUDA_VISIBLE_DEVICES=3
bash experiments/run_eval.sh small msrvtt \
     experiments/runs/sub1m_r0_s42/ckpt_02.pth \
     experiments/evals/sub1m/sub1m_r0_s42 num_frames_test 8 batch_size_test 32
ls -l experiments/evals/sub1m/sub1m_r0_s42/ranks_*.npz   # ← 应该看到两个(raw + DSL)
```

**② 等某条臂跑完,直接出带 CI 的对比表**(零 GPU):

```bash
python experiments/collect_results.py --compare \
       experiments/evals/sub1m/sub1m_r1a_s42 \
       experiments/evals/sub1m/sub1m_r0_s42
```

输出长这样(判定列直接给结论):

```
| 指标   |   臂  | 基线  |   Δ   |     95% CI     | 判定 |
| txt_r1 | 39.80 | 39.00 | +0.80 | [+0.30, +1.40] | 显著 |
| R_sum  |247.30 |245.10 | +2.20 | [+0.90, +3.90] | 显著 |
```

### 判据变更(重要,请记一下)

- **A/B 比较:以 95% CI 是否含 0 为准,不再套 ±1.6。**
- `README.md` §6.1 的 ±1.6 **仍然适用**于「单臂绝对值」的阴性结论(如 GATE-0 判 reverse 是噪声),
  只是**不再用于 A/B**。
- **两口径同向**(`ret_test/` 与 `ret_test_emb/`)的纪律**不变** —— 加 `--prefix 'ret_test/'` 跑第二遍。

实现:`experiments/paired_bootstrap.py`(纯 numpy,不依赖 torch)+ `collect_results.py --compare`,
11 项单测本地全过。命令详见 `COMMANDS.md` §3.9。

## 2026-07-29 (63) — ✅ 数字出来了,而且**1M 就追平了官方全量**;但同一组数字暴露出更要紧的问题

**权重这次真加载上了**(日志有 `自动解析到 .../mp_rank_00_model_states.pt` + `Load pretrained model from`)。

| 参照 | DSL t2v R@1 | DSL r_mean | **raw t2v R@1** | **raw r_mean** |
|---|---|---|---|---|
| 地板:骨干初始化 + **随机**时序算子 | 34.9 | 56.92 | 32.2 | 54.82 |
| **`sub1m_r0_s42`(1M × 3 epoch)** | **42.7** | **61.28** | **41.8** | **60.65** |
| 天花板:官方权重(10.6M 全量) | 42.5 | 62.08 | 41.3 | 61.47 |

### ✅ 好消息:复现管线端到端是对的

**10% 的数据、3 个 epoch,MSR-VTT t2v R@1 就追平官方全量**(raw 41.8 vs 41.3、DSL 42.7 vs 42.5,
两口径同向、都在 ±1.6 带内 ⇒ 打平)。数据/训练/评测三条链路都验证过了。

### 🔴 坏消息(更重要):**MSR-VTT 1k 在这个尺度上已经排不出臂序了**

**数据翻 10 倍,raw r_mean 只涨 +0.82(DSL +0.80)** —— 而仓库判据是「阴性 ±1.6 带、阳性要 >2.4」。
⇒ **连"10× 数据"这么大的处理效应本身都落在噪声带以内。** 各臂(r1a/r1b/nA1/nR1)的效应
只会**比它更小** ⇒ 按现在的判法,后面 11 臂跑完也是一堆"测不出差别"。

**这不是 1M 层设计错了,是测量仪器精度不够。** 出路按性价比:

1. **配对 bootstrap(最高优先)** —— 各臂评的是**同一** 1000 条查询,是**配对**比较;
   配对差值的方差远小于"各自对一个绝对值"的无配对 ±1.6 带。
   这正是评审 **U10-1** 欠着的那件事(ROADMAP 自查 G4,今天由 🟡 升 🔴)。
   **它不是锦上添花,是让 1M 层能用的前提。**
2. **多基准聚合**(MSR-VTT + DiDeMo + ANet 同向)—— README §6 本来就写了,现在从纪律升级为必须。
3. **降低期望**:1M 层只用来**否决**(掉点明显的砍掉),小的正向差异一律留到全量。

### 建议你现在怎么办

**继续让 runner 把剩下的臂跑完**(它们不白跑 —— 是否决用的,且换判据后可复用同一批 ckpt 重算)。
**我这边下一步做配对 bootstrap 进 `collect_results.py`**,等你臂跑完就能直接出带 CI 的对比表。

要不要我现在就动手写 G4 那个 bootstrap?(纯评测器改动,不占 GPU,不影响你在跑的任何东西。)

## 2026-07-29 (62) — ⛔ 那组数字**无效**:模型根本没加载训练权重(已修,重跑即可)

**先别记那两行数字。** 你贴的日志里有这一句:

```
WARNING ... | tasks_clip.shared_utils: No pretrained checkpoint provided, training from scratch
```

⇒ 评的是「MobileCLIP-S2 骨干初始化 + **随机**时序算子」,**不是**你训了 15 小时的模型。
旁证:`_IncompatibleKeys(missing_keys=[... temporal_embedding, 全部 *_1d ...])` —— 时序算子
全是随机的;raw `r_mean 54.82` 也明显低于官方权重同口径的 **61.47**。

### 根因(不是 auto_resume)

`Auto resuming` / `Not found checkpoint in <eval 输出目录>` 那两行**无害** —— 找不到就不覆盖
`pretrained_path`。真正的问题是:**DeepSpeed 存的是目录式 ckpt**(`ckpt_02.pth/` 里面才是
`mp_rank_00_model_states.pt`),而 `shared_utils.py` 里**只有 `deepspeed.enable=True` 那一支**
用 `osp.isdir` 处理目录;**评测配置默认 `deepspeed.enable=False`**,走的是 `osp.isfile(...)` ——
**目录恒为 False ⇒ 静默跳过加载,只留一句 warning,评测照跑。**

我上一条说「`shared_utils.py:150` 的 isdir 分支支持目录」——那行确实存在,但它在 deepspeed 分支里,
评测走不到。这条是我判断错了。

### 已修(两道)

1. **自动解析目录** → 指向里面的 `mp_rank_00_model_states.pt`(其 `module` 键正好被既有加载逻辑支持);
2. **加载失败直接 raise** —— 显式给了 `pretrained_path` 却没加载上、或 `evaluate=True` 却无权重,
   一律炸。**绝不再在未加载权重的模型上出数字。**

### 重跑(同一条命令,pull 后即可)

```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
source experiments/env_b100.sh
export CUDA_VISIBLE_DEVICES=3
bash experiments/run_eval.sh small msrvtt \
     experiments/runs/sub1m_r0_s42/ckpt_02.pth \
     experiments/evals/sub1m/sub1m_r0_s42 num_frames_test 8 batch_size_test 32
```

**跑完先自检**(这一步以后每次都做):
```bash
grep -E "Load pretrained model from|training from scratch" \
     experiments/evals/sub1m/sub1m_r0_s42/train.log
```
必须看到 `Load pretrained model from .../ckpt_02.pth/mp_rank_00_model_states.pt`。

### 顺带:那组无效数字其实也有用

「S2 初始化 + 随机时序算子」= raw `r_mean 54.82`。它是一个**免费的地板参照** ——
1M 训练如果连它都超不过,那就是训练本身有问题。官方权重同口径 61.47 是天花板参照。

## 2026-07-29 (66) — 版本对了(#344✓可继续);病根找到:服务器一直站在 claude/export-smoke 分支上

你贴的 `git log` 第一行就是答案:`(HEAD -> claude/export-smoke, origin/main)` ——内容已对齐
#344(比 #343 新,✓),**但服务器不在 main 上,而是滞留在 (#320 时代的)工作分支 claude/export-smoke**。
这就是几周来反复 divergent/「pull 了没生效」的病根:每次 `git pull origin main` 都在错误分支上执行。

**一次性根治(内容不变,只是把脚站回 main),然后继续 (65) 的序列:**
```bash
cd ~/mobileclip
git checkout -B main origin/main        # 切回 main 并对齐(-B: 已存在则重置)
git branch -D claude/export-smoke       # 删掉滞留分支(刚 reset 过,无未推提交,安全)
git log --oneline -1                    # ← 括号里应显示 (HEAD -> main, origin/main)
# 继续:五槽+新融合 prompt 的 compare 重跑 → 表格页
rm -f outputs/audit_10k/recap_slot.*.jsonl outputs/audit_10k/pipe_sl_*.jsonl       outputs/audit_10k/pipe_sl_timing.tsv outputs/audit_10k/pipe*fused_d*.jsonl
PIPE_LEAN=1 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
bash tools/data_audit/run_v100.sh summary-table
tar czf outputs/audit_10k/table.tgz -C outputs/audit_10k table_compare
```
以后每次对齐后**看一眼括号**:`(HEAD -> main, origin/main)` 才是正常态。TROUBLESHOOTING
的 divergent 条目已补这个根因。GDINO scp 仍最高优先。

## 2026-07-29 (65) — 你那条 summary-table 没跑起来:服务器 checkout 又落后了(帮助列表里没这命令=铁证)

刚才那次失败的读法:**脚本打印整屏用法=子命令没被识别**,而打印出的列表里根本没有
summary-table —— 说明本机还在 #341 之前的旧代码上(git pull 又没生效,老毛病,
TROUBLESHOOTING 新增了对应条目)。table_compare/ 自然没生成,tar 才报 No such file。

**恢复序列(逐行,第 2 行必须核对):**
```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
git log --oneline -1     # ← 必须 #342 或更新;还是旧号就把输出贴回来
# 若还没按 (60) 块重跑过 compare(五槽+新融合 prompt),先跑它(可加 PIPE_LEAN=1, 见 (62)):
rm -f outputs/audit_10k/recap_slot.*.jsonl outputs/audit_10k/pipe_sl_*.jsonl       outputs/audit_10k/pipe_sl_timing.tsv outputs/audit_10k/pipe*fused_d*.jsonl
PIPE_LEAN=1 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
# 然后表格页:
bash tools/data_audit/run_v100.sh summary-table
tar czf outputs/audit_10k/table.tgz -C outputs/audit_10k table_compare
```
注意:若跳过中间的 compare 重跑直接出表,表格显示的会是**旧版内容**(四槽 B/旧融合),
别拿它当新方案的目检依据。

顺手加固:从 #342 起,`run_v100.sh` 遇到未知子命令会显式 `❌ 未知子命令: xxx` 并
exit 1(不再静默打帮助装作成功),并直接提示上面的对齐命令——同样的坑下次 10 秒内定位。

## 2026-07-29 (64) — 表格版总结页就位:GIF + 九列(en源|原zh|A/B 重述+长稿+短稿), compare 之后一条命令

按你要的形式做好了(注:R1 线 07-29 的块把 (58)-(60) 编号复用了,与我 07-28 的 (58)-(60)
是不同内容,按日期区分;为避混我直接跳到 (64))。

**用法(pipeline-compare 跑完后, CPU ~1-2 分钟):**
```bash
bash tools/data_audit/run_v100.sh summary-table
tar czf outputs/audit_10k/table.tgz -C outputs/audit_10k table_compare
# 本地: scp rby@101vector:~/mobileclip/outputs/audit_10k/table.tgz . && tar xzf table.tgz
#       打开 table_compare/table.html
```

- 每行 = 一片:**循环 GIF**(16 帧×150ms, 最长边 300px, 用管线同款 decord 解码, 单片失败
  只缺那一格不拖死整表) | 源caption(en) | 源翻译(黄) | **方案A** 重述(1.2.3. 编号)/直融
  长稿/直融短稿(绿底加粗) | **方案B** 同三列(重述带【人物】【物体】蓝徽标)。
- 数据直接读 review_compare.tsv,**零重算**;GIF 有缓存,重跑只补缺的。右上角版本角标
  **vT20260728a**(旧文件识别用)。50 片的包约 25-75MB。
- 表格页是给"看结果/给人展示"用的静态页;要边看边填单选仍用卡片包 bundle_compare。

顺序建议:先跑 (60) 块(07-28 那个五槽重跑序列, 含 PIPE_LEAN=1 变体见 (62)) → 再跑本块
两行。**GDINO scp 仍最高优先**。
## 2026-07-29 (61) — `termcolor` 的真因是**没 source `env_b100.sh`**;另外**没有 GPU 4**(我上条给错了)

我上一条给的命令有**两个错**,都已修:

**错 ①:`CUDA_VISIBLE_DEVICES=4` —— 这台机没有 4-7**(#339 已定:0-3 就是你的全部配额)。
**错 ②:漏了 `source experiments/env_b100.sh`。**

**`termcolor` 的真因**(诊断行已经把它指出来了):`python : /usr/bin/python`。
这在 b100 上**是对的、不是 bug** —— `SETUP_B100.md` §2 早就写明:「`mvc_env310` 提示符虽在,
但 `python` 实际指向 `/usr/bin/python`,不再修它」,策略是**沿用系统 torch(2.5.1+cu124,含 sm_90)
+ 缺的小包装进私有 user-site**,由 `env_b100.sh` 设 `PYTHONUSERBASE` 生效。
**你这个新 shell 没 source 它 ⇒ user-site 不在 `sys.path` ⇒ termcolor 导不进。**
(训练那 15 小时之所以没事,是因为启动 runner 的那个 shell source 过。)

⇒ **别 `pip install termcolor`** —— 裸装会落到共用系统目录,是事故级操作(会改到 wky/zyc 的环境)。

### 正确命令(两处都已改对)

```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
source experiments/env_b100.sh                 # ← 关键的一步
export CUDA_VISIBLE_DEVICES=3                  # ← 0-3 里挑一张,不是 4
bash experiments/run_eval.sh small msrvtt \
     experiments/runs/sub1m_r0_s42/ckpt_02.pth \
     experiments/evals/sub1m/sub1m_r0_s42 num_frames_test 8 batch_size_test 32
```

**先看头两行输出**:
```
python    : /usr/bin/python                    ← b100 上这就是对的
userbase  : /workspace/mount/b100_zaip_data/rby/pyuser   ← 这行必须有值,`<未设置>` 就是没 source
```

### 与在跑的 r1a 共卡,可以

`sub1m_r1a_s42` 占 0-3 训练,峰值 ~42G / reserved ~46G,**80G 卡还剩 ~34G**;
补评是单卡、1k 条、`batch_size_test 32`,放得下。会让训练略慢几分钟,可接受。
(不想共卡就等 r1a 约 16:25 UTC 跑完再补评,反正它也要补。)

### 脚本已经自己会兜了

`run_eval.sh` / `run_pretrain.sh` 现在:**① 检测到 `PYTHONUSERBASE` 没设且 user-site 存在
就自动 source `env_b100.sh`;② 起 torchrun 之前先试导一次 `termcolor, torch`,失败就直接
打印修法并退出** —— 不会再让你等三层栈之后才看到一个含糊的缺包错误。

## 2026-07-29 (60) — 别取消 `sub1m_r1a_s42`:训练是好的,只是环内评测会空;后面 6 个臂正常

**回答:不用取消。** 三条:

1. **训练本身没问题** —— 坏的只是「环内评测被跳过」这一步,loss / ckpt 全部正常存。
2. **数字能离线拿回来**(和 `r0_s42` 一样的路子,`COMMANDS.md` §3.8),取消只会白丢已经跑掉的小时数。
3. **不冲突** —— 它在 GPU 0-3,你补评用的是 GPU 4,可以并行。

### 但有三件事要有心理准备

**① `r1a_s42` 会和 `r0_s42` 一样出 `Empty DataFrame` —— 预期之内,别慌。**
它 `START 01:13:46 UTC`,而修复(#334)是 `01:47:10 UTC` 落地的 —— **早了 33 分钟**,
所以它进程里load 的是旧 config。按 r0 的 15:11:58 推算,**约 07-29 16:25 UTC 跑完**,跑完照 §3.8 补评。

**② 从 `sub1m_r1b_s42` 开始的 6 个臂是好的。**
runner 每个臂都重新 `bash experiments/run_pretrain.sh` ⇒ **重读磁盘上的 config**,
而你 02:07 UTC 那次 `git reset --hard` 已经把 config 更新了。所以只有 r0/r1a 两个臂需要补评。
(顺带确认:那次 pull **没有动 `run_verify_1m.sh`** ⇒ 不存在「bash 读到一半脚本被改」的风险。)

**③ 默认清单是 8 臂 ≈ 5 天,#335 新增的 4 个臂不在里面。**
`CONFIGS` 默认只有 r0/r1a/r1b/nA1 × seed{42,43}。#335 的 `sub1m_nR1_*` / `sub1m_r1both_*`
**要显式指定才会跑**(加上就是 12 臂 ≈ 7.6 天,所以没给你悄悄塞进去):

```bash
bash experiments/run_verify_1m.sh 0,1,2,3 \
     sub1m_nR1_s42 sub1m_r1both_s42 sub1m_nR1_s43 sub1m_r1both_s43
```

### 你现在只要做一件事

等 (59) 的补评命令跑出 `r0_s42` 的两行数字贴回来。**别动那个正在跑的 r1a。**

## 2026-07-29 (59) — `termcolor` 不是真问题:裸 `torchrun` 把评测跑到**系统 python** 去了(已全仓修)

**先看你贴的栈,不是看报错**:venv 是 `mvc_env310`,可栈里全是
`/usr/local/lib/python3.10/dist-packages/...`、`binary: /usr/bin/python3`
⇒ **评测根本没在你激活的环境里跑**,所以 venv 装没装 `termcolor` 不相干。

**根因**:裸 `torchrun` 走 PATH → `/usr/local/bin/torchrun`,其 shebang 指向 `/usr/bin/python3`。
**这个坑 7-24 咬过一次**(当时训练崩,只修了 `run_pretrain.sh`),`run_eval.sh` 没同步到,7-29 又咬一次。

**已修**:全仓 **17 个启动器**统一改成 `python -m torch.distributed.run`
(`experiments/run_eval.sh` + `scripts/evaluation/.../eval_*.sh` 全部 + `scripts/pretraining/.../run.sh` 全部),
`COMMANDS.md` / `README_zh.md` 的手写配方同步。`run_eval.sh` 现在会先打印 `python : <路径>`,
**那一行对不上你激活的环境就是这个坑**。

### 重跑(同一条命令,pull 后即可)

```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
export CUDA_VISIBLE_DEVICES=4
RUN=experiments/runs/sub1m_r0_s42
bash experiments/run_eval.sh small msrvtt "${RUN}/ckpt_02.pth" \
     experiments/evals/sub1m/sub1m_r0_s42 num_frames_test 8
```

**先看第一行输出的 `python :`** —— 应该是 `/workspace/.../mvc_env310/bin/python`。
不是的话先 `source` 你的 venv 再跑,**别急着 pip install termcolor**(装在系统 python 上治标不治本)。

跑完把 `experiments/evals/sub1m/sub1m_r0_s42/eval_res.json` 里
`ret_test/` 与 `ret_test_emb/` 两行贴回来 —— 那是 1M 层的第一个基准数字,后面所有臂都减它。

(上一条 (58) 的第 ② 步「用 runner 起剩下 11 个臂」不受影响,可以并行开。)

## 2026-07-29 (58) — 15 小时白跑的根因找到了:评测键对不上,**不用重训**;顺带拿到全量 wall-clock

**先说结论:`sub1m_r0_s42` 那 15 小时没白费 —— ckpt 是好的,离线补评测就行。**

**根因**(不是评测代码坏了):基线 config 写 `test_file = dict(act_val=...)`,而 `test_types = ["ret_test"]`
—— 键对不上,`pretrain.py` 的 epoch 末循环把唯一的测试集整个跳过,所以只剩 `Empty DataFrame`。
`pilot_10k_*` 没踩到是因为它们显式写了 `ret_test`。#334 已为 8 个臂修过,**但 #335 新增的 4 个臂又漏了**
(`sub1m_nR1_s42/s43`、`sub1m_r1both_s42/s43`)—— **本次补齐 12/12**,并在 `pretrain.py` 加了**开跑前**就检查
键交集的守卫(为空直接 raise),**代价从 15 小时降到几秒**。

### 你现在做两件事

**① 先补评已跑完的那臂(一张卡,几分钟,零重训)**

```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
export CUDA_VISIBLE_DEVICES=4
RUN=experiments/runs/sub1m_r0_s42
bash experiments/run_eval.sh small msrvtt "${RUN}/ckpt_02.pth" \
     experiments/evals/sub1m/sub1m_r0_s42 num_frames_test 8
python -c "import json;d=json.load(open('experiments/evals/sub1m/sub1m_r0_s42/eval_res.json'));print(json.dumps(d,indent=2)[:800])"
```
(ckpt 路径指到**目录** `ckpt_02.pth`,不是里面的 `.pt`;`shared_utils.py:150` 的 `osp.isdir` 分支支持它。)

**② 再用 runner 起剩下的臂**(上次没走 runner —— `verify_1m.log` 里没有 START/DONE,所以只有 r0_s42 跑了)

```bash
nohup bash experiments/run_verify_1m.sh 4,5,6,7 > experiments/runs/verify_1m.log 2>&1 &
grep -E "START|DONE" experiments/runs/verify_1m.log     # 现在应该能看到标记
```

### ⏱ 顺带:U6 等了很久的那个数拿到了

`9.1821 s/it` @ 全局 batch 512(4 卡 × 128)= **55.8 samples/s**。外推全量 10.65M:

| | 4 卡(现状) | 8 卡(每卡 batch 64,全局仍 512) |
|---|---|---|
| 单 epoch | **53.0 h** | ~26.5 h |
| **R0 三个 epoch** | **≈ 159 h ≈ 6.6 天** | **≈ 3.3 天** |

⇒ U6 的「宽裕档 = 33E」≈ **73 天,直接不可行**;常态档(R0 3ep + 排序臂各 1ep + 胜者补 3ep)
在 4 卡上 ≈ **26 天**。**几乎必然要申请后 4 张卡** —— 8 卡且每卡 batch 64 可保持全局 512,
**不改任何对照条件**(`SETUP_B100.md` 红线要求你明确指示才能用后 4 张)。

**⚠️ 外推假设**:全量分片的 IO 与 1M 单分片同速。真起跑后请用首日实测校正。

### 还想请你回一句

**全量 10M 的 R0 是不是另有一路在跑?** 上面这 15 小时是 **1M 子集**那一路(`sub1m_r0_s42`)。
ROADMAP 3.2 还写着「等首光回报 → R0 起跑」,两边对不上。

## 2026-07-28 (63) — R1 大数据 A/B 定档 = 1M 子集档:入库 4 config + 1M 缓存命令,先建缓存再跑两对

既然 H100 现在训 1M 子集(`internvid_v1`),就把 R1 文本负例库的「真正发威档」A/B **放到 1M 验证层跑**(不等全量 10M)。
已入库(合入 main):处理 `sub1m_nR1_s42/s43`(+bank)、对照 `sub1m_r1both_s42/s43`(SigLIP 制度、无 bank),
**彼此相减判胜负**(不减 r0)。全套设计 `notes/r1-implementation-plan.md` §10、`verification-tier-1m-zh.md`。

**为什么 1M 才测得到(10k 测不到)**:全局 batch 512 ≪ SigLIP 饱和点 ~32k,bank 每步补 +16384 正好进甜区;
1M×3 epoch(~5.9k 步)**不过拟合** ⇒ 领先=真硬负例收益,而不是 10k 试点那种「挡早塌」的小数据伪影(§9.4)。

**⚠️ 卡 = 0,1,2,3(你在这台 H100 的全部配额,没有 4–7)。全量 R0 也在这 4 张上 → 与本 A/B 串行:
R0 在跑就先排队,等 R0 空出/暂停后再跑(全局 batch 仍 128×4=512,设计不变)。**
**(全局更正:本仓早先交接与 `verification-tier-1m-zh.md`/`COMMANDS.md` 里凡写 `4,5,6,7` / 「空出来的 4–7」的,
一律按 `0,1,2,3` 执行 —— 4–7 对你不可用;R1 A/B 路径的命令已在本次改成 0–3。)**

**服务器执行(0–3 空出后;先建一次缓存,再串行两对,共 4 config)**:
```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
export MVC_DATA_DIR=/home/vca1/datasets
ls -l experiments/configs/sub1m_nR1_s42.py experiments/configs/sub1m_r1both_s42.py   # 应都在
# ① 建 1M 文本负例库缓存(单卡、纯前向、~10–30min、~1GB;一行,勿折行):
python experiments/r1_build_textbank.py --anno anno_downstream/InternVid/InternVid-10M-flt_1.json --out-feats experiments/cache/textbank_1m_feats.fp16.npy --out-idx experiments/cache/textbank_1m_idx.npy
# ② 跑两对(~15h/config → 4 条 ~2.5 天;处理臂启动日志应见 `[R1] text bank ON: pool=(~1e6, 512) ...`):
bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_r1both_s42 sub1m_nR1_s42 sub1m_r1both_s43 sub1m_nR1_s43
```
**报告回来**:每 seed 的 `sub1m_nR1` − `sub1m_r1both`(标准 MSR-VTT 1k,best epoch,六个 recall + R-Sum)。
两 seed 同向且超 ±SE ⇒ 真硬负例、带 bank 上全量 10M 终训;≈0/矛盾 ⇒ 试点 +3.9 是 10k 专属、撤 bank。
(先建缓存再跑,别忘 `git reset --hard` 那行 —— 少 pull 会崩在 `config.debug`,见 TROUBLESHOOTING。)

## 2026-07-28 (62) — SigLIP打分可跳、A′可跳、多样性诊断要留; PIPE_LEAN=1 开关已就位

⚠️ **先说一件事**:并行分支合并把我此前的 (57)-(60) 四个块覆盖丢了,已从提交历史恢复并
**整体重编号 +1**(现为 (58)-(61),给 R1 线的 40-epoch 块让出 (57));你若按旧编号读过/
执行过,内容一一对应,命令没变。

答复"SigLIP 打分和 A′切句+诊断必要吗":**SigLIP 和 A′ 都可以跳,多样性诊断不能跳**
(计时表里它俩绑在一行,但不是一回事):

- **SigLIP 打分(0.76 s/片)→ 跳**。直融终稿完全不消费它;现在只喂三处:A− 选择(已退役)、
  评审页 ⟨⟩ 徽章(装饰)、投票代表句排序(投票已降为诊断)。跳过后这三处缺省,下游全部
  出空列不崩。
- **A′ 句切 → 跳**。G1=A′ 已被直融短稿取代,只剩全景对照列。
- **多样性诊断(CPU, ~0.1 s/片)→ 留**。它是「N 遍独立采样」这个判断式融合核心前提的
  **唯一哨兵**(COLLAPSE/IDENT 旗);砍了省不到一分钟,却瞎了核心假设的眼睛。

已实现 `PIPE_LEAN=1` 开关(默认关,两臂通用)。用法=在 (60) 块的重跑命令前加个变量:
```bash
PIPE_LEAN=1 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
```
(其余 rm/grep/tar 行照 (60) 块不变,徽标仍 v20260728f。)两项合计省 ~0.86 s/片,全桶约 0.6h。

**提醒真正的大头**:② 派生 55 s/片(A 臂三分之二的成本,直融根本不读它)+ ④b 投票
LLM ~10 s/片。§5.9 整案还差这两项 + A 的 N 5→3 + 开关是否默认开——**回「§5.9 做」
我就全部并进 PIPE_LEAN 并改默认开**(A 臂 88.6→~19 s/片, 全桶 ~12h→~6h)。

## 2026-07-28 (61) — A/B 重述终审通过; 顺带修 B 例句三处零成本瑕疵((60) 命令不变)

终审结论:**结构与契约全部对上,可以跑**。核对项:五栏目↔正则↔行数上限逐项一致;「无」/
超行/杂话处理与三级兜底在;legacy【主体】渲染在;REFINE 30 字=短稿 30 字;A/B 语义覆盖
授权对等(两臂都强制 人+物+外观+时序+场景+画面文字,A 段落/B 栏目——compare 隔离的是
结构变量,不是内容授权差)。

趁 B 缓存已失效**零成本修三处**(已合入 main):
1. B 例句动作行点名施动者「男子先把黄瓜…」——多人物片里动作行必须自带主语才可消歧;
2. 例句 gloss `[]`→`[cucumber,plate]`——原例教模型少给 gloss,和规则 6 矛盾,会饿着 MISS 哨兵;
3. B 规则 3 补「拍摄时间」禁推断,对齐 A。

登记不改:A 首句「用于训练的描述」措辞已过时(训练字幕=直融短稿),但改它白费 A 全部缓存,
留待 A 臂下次必须重跑时顺带;投票门槛 A 2/5 vs B 2/3 是已登记设计(投票现仅诊断)。
**B v2 至今零样本**——静态审到此为止,(60) 首跑后的目检三查才是动态验收。

**服务器操作:仍是 (60) 那段命令,一字不变**(git log 应看到 #327 或更新)。若你已经跑过
(60):再执行同一段命令一次即可——只有 B 臂会重算(~15 分钟),其余全部缓存命中。

## 2026-07-28 (60) — 属性表 v2:按 ASID 分开 人物/物体 → 五槽定版; B臂重述缓存失效须重跑(徽标v20260728f)

按你指令把 Character/Object 像 paper 那样分开。**定版 v2(取代 (59) 的四槽版):
人物≤2 / 物体≤2 / 动作≤2 / 场景 1 / 文字 1**(动作/场景本就分开,不变;文字是我们补的
OCR 可验槽,保留)。要点:

- 分开后每行修饰**仍内嵌行内**((59) 说的绑定错误是针对单列「属性槽」的,人物/物体分栏
  不触发);投票同槽约束现在人物对人物、物体对物体精确对齐;专设物体栏=显式回忆线索。
- 新登记风险:物体栏清单式堆琐碎物件——prompt 写死「只写对内容有意义的主要物品」+「无」
  纪律,目检加第三查。每遍行数上限 6→8,解码约 +10-15%。
- Camera/Speech/Emotion/Narrative 落选结论与三条重开条件维持((59))。
- 旧 slot/struct 缓存在评审页仍按【主体】正常渲染(legacy 映射),不会花屏。

**槽位 prompt 变了 → B 臂重述及其下游全部失效,须重跑**(A 臂重述/投票缓存不受影响;
若 (58) 还没跑,直接跑本块,总约 45-60 分钟;跑过 (58) 则约 30-40 分钟):
```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
git log --oneline -1        # ← 必须比 #324 新; 还是旧号就贴回来
rm -f outputs/audit_10k/recap_slot.*.jsonl outputs/audit_10k/pipe_sl_*.jsonl \
      outputs/audit_10k/pipe_sl_timing.tsv outputs/audit_10k/pipe*fused_d*.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
grep -c v20260728f outputs/audit_10k/bundle_compare/index.html   # ← 必须 ≥1
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
```
页面徽标这次是 **v20260728f**;B重述应出现【人物】【物体】蓝徽标。**GDINO scp 仍最高优先**。

## 2026-07-28 (59) — 属性表定版:四槽维持并冻结, Camera 终审不加(无新服务器操作, (58) 照跑)

你让我定最终方案。**定版:四槽不动、随冻结令冻结——主体≤2 / 动作≤2 / 场景 1 / 文字 1。**
每槽过了四条硬标准(8 帧可见 / 可外验或可投票 / 检索区分度 / 进得了 30 字预算);完整逐槽
理由表在主计划 §5.6b 定版段。要点:

- **修饰不单列是设计而非缺失**:外观/数量内嵌主体行、时序内嵌动作行——两主体并存时单列
  属性槽必产生「红色归谁」的跨槽绑定错误(新幻觉源);ASID 也把它们作子属性,同构。
- **Camera 终审不加**(上一块留的拍板项就此关闭):① 检索查询/MSR-VTT 式字幕几乎不含镜头
  语言,区分度低;② GDINO/OCR 均验不了+8 帧判运动弱,纯增裸露面;③ 与「禁画面/镜头元词」
  纪律正面冲突。Speech 无音频不适用;Emotion/Narrative 反幻觉纪律排除。
- **定版但可证伪**,三条重开条件写死在 §5.6b:(a) B直融同一缺因系统性输 A(≥5/50);
  (b) ≥10% 片的 A 重述含 B 槽装不下的高区分度信息;(c) 动画/游戏**呈现形式** A 有 B 无
  ≥5/50(唯一登记的残余疑点——现靠模型自发写进主体行,未加 prompt 硬要求)。
- 目检清单加两项:**看 B 是否漏「动画/游戏画面」标记、是否漏关键交互物体**。

**本块零代码改动、零缓存失效——服务器不用做任何新操作,(58) 的重融序列照跑即可**(徽标
仍是 v20260728e)。GDINO scp 仍最高优先。

## 2026-07-28 (58) — 直融改两次调用:长稿不限字数→短稿一条; ASID属性对照答复(徽标v20260728e)

三条回复(前两条已实施合入 main, 第三条是答复+待拍板):

1. **长稿去字数上限**:融合调用 1 只做判断+写长稿(**长度不设限制**, 以完整覆盖过审细节为准,
   禁流水账/禁新增)+带理由 dropped。
2. **长短稿分开生成**:调用 2 **只读长稿**提炼一条 ≤30 字训练短字幕(闭集, 判断不重做)。
   每片 +1 次小调用(~2-4s); 调用 1 失败时短稿降级用原始重述(`short_src` 落盘)。
   `--max-new-tokens` 640→768。
3. **B 臂 4 属性 vs 参考文献**:ASID(2602.13013)用 **8 个核心属性**(项目页确认):
   Scene/Object/Character/Action/Emotion/Camera/Narrative Element/Speech, 每视频 8 条单属性字幕。
   映射:Object+Character→我们的主体(≤2 行); Action→动作; Scene→场景; Speech(音频)→不适用
   (只看帧无音频; 我们的「文字」=画面文字, ASID 没这个顶层属性、是我们补的且 OCR 可验);
   Emotion/Narrative→被反幻觉规则明令排除(情绪/前因后果=易幻觉, 不该进检索训练字幕);
   **Camera(镜头运动)=唯一真正缺席的候选**。**判断:对 32-token 检索字幕 4 槽够用**;
   要不要加可选第 5 槽「镜头」(1 行空写无; 代价: GDINO/OCR 验不了镜头运动, 8 帧判运动偏弱)
   ——**待你拍板, 默认不加**。详见主计划 §5.6b 属性对照段。

**融合 prompt 又变了 → 直融缓存再清一次重融**(重述/投票缓存照旧复用; 若 (56) 还没跑, 直接跑本块,
约 35-45 分钟):
```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
git log --oneline -1        # ← 必须比 #320 新; 还是旧号就贴回来
rm -f outputs/audit_10k/pipe*fused_d*.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
grep -c v20260728e outputs/audit_10k/bundle_compare/index.html   # ← 必须 ≥1
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
```
页面徽标这次是 **v20260728e**。**GDINO scp 仍是最高优先待办**((52) 块步骤)。

## 2026-07-28 (57) — 长档 40-epoch 两臂跑完:池仍领先,但**耐久差只有 ~+3.9 R-Sum**(不是 +11~13)→ 试点级问题收尾

两臂都跑通(各 ~2:27h,`nR1_textbank_pilot_long` 然后 `pilot_10k_r1_long`,都 `torch 2.5.1+cu118`)。
回答唯一未决问题「对照多跑步能否追上」:

- **对照 ep0→ep1 又塌了**(r_mean 90.00→85.18,−4.8;seed42/43 之外第三次复现),再花 ~12 epoch 爬回平台。
  说明池在试点上的好处**机制上=挡住这个早期塌陷**(10k 太小的过拟合伪影)。
- **平台对比(ep30–39 均值,LR≈0 都收敛)**:处理臂 **88.57** [88.33,88.77] vs 对照 **87.92** [87.80,88.10]
  → Δr_mean **+0.65 → ΔR-Sum ≈ +3.9**。两条平台带**不重叠**(池最差晚期 88.33 > 对照最好 88.10),
  **池在最后 10 个 epoch 每一轮都领先**。末轮(ep39)R-Sum 531.0 vs 527.3 = **+3.7**,与平台一致。

**结论:改措辞,不推翻。** 池仍有稳定耐久领先(~+3.9,约判据 0.5 的 8 倍,晚期每轮都赢),所以**不是纯"加速"**;
但耐久差只有 10-epoch 快照(+11~13)的约 30%——多跑 4× 步,对照追回了约 2/3 的早塌亏损。诚实的试点级说法是
**「大幅早期稳定 + 适度耐久提升」**,而不是 +11~13 的耐久胜。之前那个大数字是在对照还没爬回来时测的。

**这不是最终定论**:10k 上池挡的塌陷是小数据过拟合伪影;R1 真正针对的档(大数据、in-batch 负样本变成
信息瓶颈)是**另一套机制**。试点 +3.9 是好兆头但不决定性。**决定性实验仍是 H100 大数据 A/B((55) 块),
等你说何时上、用哪几个分片。V100 这边不需要再跑了——试点级问题到此收尾。**(全表见 `notes/r1-implementation-plan.md §9.4`)

## 2026-07-28 (56) — 直融短稿改为一条≤30字训练字幕 + B臂属性徽标 + 每条caption一个编号(徽标v20260728d)

按你三条指令改完(已合入 main):

1. **B 方案重述的属性** = 固定四栏:**主体≤2 行 / 动作≤2 行 / 场景 1 行 / 文字 1 行**(空栏写「无」,
   struct 变体同一套四属性、每遍 5-8 条)。现在评审页 B重述每行行首渲染蓝色小徽标
   【主体】【动作】【场景】【文字】——从已缓存的 `views.attr` 显示端重建,**重述缓存不用重跑**。
2. **分点改法**:每条 caption(每遍重述)一个编号,不再按「;」把一条切成多点。
3. **直融两稿定版**:**短稿 = 一条 ≤30 字**的自然短句(32-token 文本编码器墙内,[CLS]+30字+[SEP])
   ——这条就是最终训练字幕;**长稿 = 60-120 字**尽量保留全部过审细节(目检参考/后备)。
   融合 prompt 输出键改 `short`,验真/refine/TSV 全链兼容(views 单元素),REFINE 上限同步 28→30 字。

**融合 prompt 变了 → 直融缓存必须清掉重融**(重述/SigLIP/投票缓存全部复用,约 25-35 分钟):
```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
git log --oneline -1        # ← 必须比 #318 新(本块对应 PR 见提交信息);还是旧号就贴回来
rm -f outputs/audit_10k/pipe*fused_d*.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
grep -c v20260728d outputs/audit_10k/bundle_compare/index.html   # ← 必须 ≥1
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
```
本地解包后页面左上角看 **v20260728d** 徽标;B重述应出现蓝色属性徽标,A重述 5 条=5 个编号,
直融=一条短句、直融长稿=一段。**GDINO scp 仍是最高优先待办**((52) 块的下载+scp 步骤,5 分钟)——
判断式融合放宽单源细节后,外部验真门现在实际只剩 OCR。

## 2026-07-28 (55) — 数据分工已明确:V100 只有 10k,全量 InternVid 在 H100 → 大数据 A/B 是 H100 的活

维护者答复:**V100(101vector)只有 10k 子样本;全量 InternVid 只在 H100 上。** 所以:
- **现在在 V100 上跑的**:仍是 (54) 的 40-epoch 长档两臂(同一 10k 试点,复用 bank 缓存)。这是 V100 能做的
  下一个、也是最后一个纯试点级实验(回答"对照多跑步能否追上")。
- **大数据 A/B(R1 真正发威的档)= H100 实验**,不是 V100。把 `train_corpus="internvid_all"` 指到 V100 会因
  数据缺失而失败。等挪到 H100 时,要在 H100 上**重建更大的 text-bank 缓存**(现有 `textbank_pilot_*` 只覆盖 10k)。

已把这条服务器数据分工写进 `notes/r1-implementation-plan.md §9.3` 和记忆(`server-data-split`)。V100 这边就按
(54) 的命令跑长档、报逐 epoch `r_mean` 即可;H100 的大数据 A/B 等你说何时上、用哪几个分片再单独设计配置。

## 2026-07-28 (54) — seed-43 **复现了**:ΔR-Sum=+13.1(seed42 是 +11.0)→ GO,下一步跑长档

reset 后两臂都跑通了。**seed-43 把差复现了,还更大**:末轮 R-Sum 处理臂 533.8 vs 对照 520.7 =
**+13.1**(seed42 是 +11.0;判据 <0.5,两个种子都是 ~20 倍)。**池在全部 10 个 epoch、两个方向都领先**
(ep9:v2t txt_r1 +3.4、t2v img_r1 +2.9)。torch 版本两臂一致(都 `2.5.1+cu118`),**环境保留意见已消除**。
seed-43 还把机制看得更清楚:**对照在 ep1 就塌了**(r_mean 90.2→85.4 再没完全回来),池几乎不掉
(90.9→平台 ~88.9)——正是"减小微调退化 / 更高平台"。两臂并排见 `notes/r1-implementation-plan.md §9.2`。

**下一步(不需要新数据、最便宜、直接回答唯一未决问题):长档 40 epoch 同一试点两臂**,看对照多跑步能否追上,
还是池保持稳定领先。配置已入库(seed 42,除 `epochs=40` 外与 10-epoch 那对完全一致,复用现有 bank 缓存,
不用重建;每臂约 2.5h):
```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
ls -l experiments/configs/nR1_textbank_pilot_long.py experiments/configs/pilot_10k_r1_long.py
bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot_long.py 8
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1_long.py 8
```
(**注意**:先跑 `reset --hard` 那行再启动 —— 上两次就是没对齐/没 pull 才崩在 `config.debug`;`ls` 两个文件都在再跑。)
**报告回来**:两臂逐 epoch 的 `r_mean`(和末轮六个 recall)。若池领先在 40 epoch **保持/扩大** → 稳,不只是收敛更快
→ 上真正的大数据 A/B。若对照到 ep30-40 追到 <0.5 → 池主要是"加速",要改措辞(不是推翻)。

**真正该发威的大数据档(卡在数据):** 配置里 `internvid_all`=11 个 `InternVId-FLT_*` 分片。上大数据 A/B 前,
请确认服务器上这些分片 + 对应更大的 text-bank 缓存都在。这个你比我清楚——**回一句服务器上现在有哪些 InternVid 数据可用**。

## 2026-07-28 (53) — 渲染器本地已验证无误:你看到的是旧文件;新增版本徽标一眼辨新旧 + 投票版撤下页面

**判定**:我在本地用合成表端到端跑了 make_review_bundle 并模拟浏览器渲染——编号分行
(‖ 与「；」双切)、长稿独立成块、投票列移除,**全部正常**。你反复看到旧样式的根因
= checkout divergent(块 50),`git pull` 一直没真正生效,页面是旧代码产的。

**三个变更已合并**:① 页面左上角新增**版本徽标 `v20260728c`**——打开先看它,没有
=旧文件;② **投票版从所有页面撤下**(compare 单选改 A直融/B直融 二选;单臂页只显示
直融长稿+直融;投票数据仍在全景 TSV 作诊断);③ compare 提示语同步更新。

**防呆执行序(每步都有可验证输出)**:

```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
git log --oneline -1        # ← 必须看到 #314 或更新;还是旧号=对齐失败,贴回来
rm -f outputs/audit_10k/pipe_fused_d*.jsonl outputs/audit_10k/pipe_sl_fused_d*.jsonl \
      outputs/audit_10k/pipe_st_fused_d*.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
grep -c v20260728c outputs/audit_10k/bundle_compare/index.html   # ← 必须 ≥1
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
# 本地: scp 解包 → 打开 index.html → 左上角看到 v20260728c 徽标才算新页
```

(GDINO scp 仍是最高优先,命令在块 (52)。)

**Then report back**: `git log --oneline -1` 输出 + 徽标是否出现 + 四版目检结论
+ 20 条 dropped 理由抽查。

## 2026-07-28 (52) — 全案复审出 3 红:GDINO 必须装了(5 分钟)、直融上限已修 640、长稿不过修复

**复审记录在主计划 §0.9(当前定版一页图,与旧段冲突以它为准)。服务器只有一件必办**:

🔴 **装 GDINO**——判断式融合放宽了单源细节,外部幻觉门现在实际只剩 OCR 查引号。
这台机连 hf-mirror 也不通,只能**本地下载后 scp**(第四次给出,请务必执行):

```bash
# 你【本地能上网的机器】:
pip install -U huggingface_hub
huggingface-cli download IDEA-Research/grounding-dino-base   # ~700MB
huggingface-cli download hfl/minirbt-h288                    # ~40MB
ssh rby@101vector 'mkdir -p ~/.cache/huggingface/hub'
scp -r ~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base rby@101vector:~/.cache/huggingface/hub/
scp -r ~/.cache/huggingface/hub/models--hfl--minirbt-h288    rby@101vector:~/.cache/huggingface/hub/
# scp 完成后在服务器(清掉"GDINO 缺席"时代的旧验真判定让它重验):
rm -f outputs/audit_10k/pipe*_verified.jsonl outputs/audit_10k/pipe*_final.jsonl
```

其余两红已处理/登记:直融 512→**640** token(防 v3 长输出截断置空,本次已修——跑 (51)
的命令时顺带生效);长稿不过验真/修复=已登记为"仅目检参考",下游要用长稿前必须补验。
黄灯提醒:目检结论出来前**冻结 prompt 迭代**;§5.9 精简仍在等拍板。

**Then report back**: 照 (51) 跑完后——四版目检结论 + 抽 20 条 dropped 看取舍理由
是否服信 + GDINO 装好后 ✗ 数量变化(此前 OCR-only 的 ✗ 都是引号文字类)。

## 2026-07-28 (51) — 融合改判断式(弃硬投票) + 页面小点/长稿两个显示问题的修法

**三件事已实施合并**:

1. **判断式融合(§5.7 v3)**:直融 prompt 弃「≥2 段支持才写」的机械票规,改三步编辑
   判断——抓重点(核心内容优先)→ 辨真伪(跨版本一致=证据非铁律;矛盾取多数或模糊化;
   单源细节按类型裁:动作/位置可留,数字/画面文字/人名/意图无旁证即舍或模糊化;引号
   文字仍须多段一致才照抄)→ 成稿(先长稿后短句)。dropped 每条现附括号取舍理由;
   规则投票照跑作对照、GDINO/OCR 验真+修复仍是硬门。
2. **页面小点**:渲染改为按 ‖ **和「；」**双重切分编号——重述里"；"连接的短句现在
   会 1. 2. 3. 分行。
3. **长稿不显示的成因**:多半是**直融缓存没清**(旧输出里没有 long 字段)或浏览器缓存了
   旧 index.html。照下面跑并强刷(Ctrl+F5):

```bash
# 你的 checkout 已 divergent(见块 50)——服务器按约定只读,直接硬对齐最干净:
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
# prompt 又变了 → 直融缓存再次全部作废必清:
rm -f outputs/audit_10k/pipe_fused_d*.jsonl outputs/audit_10k/pipe_sl_fused_d*.jsonl \
      outputs/audit_10k/pipe_st_fused_d*.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
# 本地: scp 解包后打开 index.html, 若样式仍旧务必强刷 Ctrl+F5 (浏览器会缓存同名页面)
```

**Then report back**: 长稿/短稿是否都显示、小点是否分行;目检重点=判断式融合的
dropped 理由是否合理(它现在是编辑判断的唯一透明窗口)、长稿有无超出重述范围的编造。

## 2026-07-28 (50) — pull **失败了**(divergent),所以还是没 pull → 同样的 `config.debug` 崩

这次 `git pull origin main` 报了
`fatal: Need to specify how to reconcile divergent branches`,**pull 没生效**,
s43 配置和那个"报清楚错误"的修复都没进服务器,于是又崩在旧的 `config.debug`。
(fetch 是成功的:`cddee56..e9cbbbd`,只是本地分支和 origin/main 岔开了,merge 被拒。)

**服务器按约定是只读的**(不该有本地代码改动,缓存/权重/数据都 gitignore),
所以直接对齐 origin/main 最干净、最安全 —— 用 `reset --hard`,不要纠结怎么 merge:

```bash
cd ~/mobileclip && git fetch origin main && git reset --hard origin/main
ls -l experiments/configs/nR1_textbank_pilot_s43.py experiments/configs/pilot_10k_r1_s43.py
```
`ls` 两个文件都在(非空)再启动。然后:
```bash
bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot_s43.py 8
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1_s43.py 8
```
对齐后 `utils/config.py` 已是新版:哪怕配置真缺,也会报
`FileNotFoundError: config file not found …`,不会再伪装成 `config.debug`。
详见 TROUBLESHOOTING「服务器 git pull 报 divergent」+「`config.debug` AttributeError」两节。
**报告回来**:两臂逐 epoch 的 `txt_r1 / img_r1`。

## 2026-07-28 (49) — seed-43 崩在 `config.debug` = **没 pull**,配置文件不在服务器上

**不是 bug,是没拉代码。** 你直接跑了 `bash experiments/run_pretrain.sh ...s43.py`,
**跳过了 (47) 那行 `git pull origin main`**,所以 `nR1_textbank_pilot_s43.py` /
`pilot_10k_r1_s43.py` **根本不在服务器上**。`Config.get_config()` 旧逻辑对不存在的
配置**静默跳过**→ 配置为空 `{}` → 第一句 `if config.debug` 就抛
`AttributeError: 'EasyDict' object has no attribute 'debug'`。`debug` 是障眼法,配置是空的。
(已核实:入库的 s43 文件本身没问题,`debug=False`、`seed=43` 都能正确收集。)

**已修**:`utils/config.py` 现在遇到不存在的配置路径直接报清楚的
`FileNotFoundError: config file not found: '<path>'. … run 'git pull origin main' …`,
下次不会再伪装成 `config.debug`。也补了 TROUBLESHOOTING 一节。

**现在这样做(pull 和跑分行,但先跑 pull 那行,别再跳)**:
```bash
cd ~/mobileclip && git pull origin main
ls -l experiments/configs/nR1_textbank_pilot_s43.py experiments/configs/pilot_10k_r1_s43.py
bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot_s43.py 8
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1_s43.py 8
```
`ls` 那行确认两个文件都在再启动。顺手 10 秒确认 torch 一致:`python -c "import torch;print(torch.__version__)"`。
**报告回来**:两臂逐 epoch 的 `txt_r1 / img_r1`。seed43 也复现 +8~+11 的 R-Sum 差 → 放大到更多
InternVid 数据/更多步;塌到 <2 → seed42 走运,回头审。

## 2026-07-28 (48) — 直融改双稿(长稿+浓缩)、评审页短句改编号分行——需 rm 直融缓存重跑

**两条指令已实施合并**:

1. **直融一次调用出两版**:`long` 长稿(60-120 字连贯段落,主体→动作→场景→文字组织,
   不受 28 字限制,仍只写 ≥2 段支持的信息)+ `views` 浓缩短句(现行 2-5 条;**训练
   captions 仍用短句版**——32-token 墙不变,长稿是目检参考/后备产物)。评审包与
   对比包新增「直融长稿 / A长稿 / B长稿」列;
2. **页面显示**:多条短句一律 1. 2. 3. 编号分行(弃 ‖ 竖线;重述/直融/投票各列通用)。

```bash
git pull origin main
# 直融 prompt 变了 → 全部直融缓存作废必清(投票版/重述/派生缓存不受影响):
rm -f outputs/audit_10k/pipe_fused_d*.jsonl outputs/audit_10k/pipe_sl_fused_d*.jsonl \
      outputs/audit_10k/pipe_st_fused_d*.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
# 增量 = B 臂 slot 重述(首次, ~5-8min) + 两臂直融重产(~2min) + 验真修复(~2min)
```

**Then report back**(并入 (46) 四项): 长稿目检感受——长稿是否明显比浓缩短句信息更全、
有没有超出重述范围的编造(它与短句同源共享投票约束,但不单独过验真,先靠目检把关)。

## 2026-07-28 (47) — R1 A/B 判完:**池赢了 kill 判据,ΔR-Sum=+11.0** → GO(试点级)

对照臂 `pilot_10k_r1`(无池、seed42、同 10 epoch)跑通。两臂并排见
`notes/r1-implementation-plan.md §9.1`。要点:**池在全部 10 个 epoch、两个方向都领先**
(ep9:v2t txt_r1 +2.8、t2v img_r1 +2.4;末轮 R-Sum 533.2 vs 522.2 = **+11.0**,判据是 <0.5 就停,
现在是 22 倍)。**为何可信**:检索 R@k 按归一化余弦排序,**与 logit_scale/bias 无关**——池对温度的
额外梯度造不出这个差,差来自更分得开的嵌入;而且差是**系统性**的(每 epoch 都领先,不是噪声抖动)。

**诚实的保留**:两臂都**低于 epoch0(冻结初始)峰值**——在这条 9000 片的易域内 val 上,池的作用是
**减小微调退化 / 更高更快的平台**,不是相对初始的绝对增益。R1 真正该发威的大数据/多步/批内负例真受限
的场景,这个试点测不到。

**建议先做一个 seed-43 复现(两臂),排除随机性**(配置已入库,paste-proof):
```bash
git pull origin main
bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot_s43.py 8
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1_s43.py 8
```
顺手 10 秒确认两臂同一 torch(对照日志打了 `torch 2.5.1+cu118`):`python -c "import torch;print(torch.__version__)"`。
**报告回来**:两臂逐 epoch 的 `txt_r1 / img_r1`。若 seed43 也复现出 +8~+11 的 R-Sum 差,就把 A/B 放大到更多
InternVid 数据 / 更多步(R1 真正针对的设置)。若 seed43 差塌到 <2,那说明 seed42 是走运,得回头审。

## 2026-07-28 (46) — B 臂已切换为固定槽位(§5.6b 落地):行式免 JSON、空槽写「无」、同槽投票

**维护者批「做」,已实施合并**:compare 的 **B 臂 = slot 固定槽位×3 遍**(旧 struct 臂
保留 `RECAP_STYLE=struct` 手动可用,产物互不覆盖)。每遍输出四栏行式(主体≤2/动作≤2/
场景/文字,行尾 `[gloss]`,空栏写「无」不许编),正则解析零 JSON;投票加了同槽约束
(主体绝不与场景并组);解析空仍有 0.3 温重试+dense 兜底。预期:解析失败趋零、
B 单遍解码 token 降 ~40-50%、按槽数票更准。

```bash
git pull origin main
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
# B 臂全新产物 recap_slot.* / pipe_sl_*——无需 rm 任何旧文件;A 臂全缓存秒过
# (要连计时一起测: SAMPLE_OFFSET=200 N_SAMPLE=50 ... pipeline-compare)
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
# 本地: scp rby@101vector:~/mobileclip/outputs/audit_10k/compare.tgz . && tar xzf compare.tgz
```

**Then report back**: ① B 臂 ①段耗时(对比 struct 的 14.46 s/片——验证降 40-50% 的预期);
② `grep -o '"n_dense_fallback": [0-9]*' outputs/audit_10k/recap_slot.*.jsonl | sort | uniq -c`
(预期几乎全 0);③ ④b 的 `[done]` 行(同槽投票后 units/回退,对比 struct 的 3.0/9);
④ compare 包目检四版结论(现在 B=slot 版)。

## 2026-07-28 (45) — ⏱ 计时收到已记档(§5.1c):B 慢因=解码 token 数;分片后 A/B 全量成本打平(~12h)

**你贴的两张 ⏱ 表读完,实测数已写进主计划 §5.1c。** 三个结论:

1. **B 重述 3 遍比 A 5 遍慢的定因**:单遍 4.8s vs 1.3s(3.6×)——struct 每遍吐 5-8 条
   JSON(键名+脚手架+gloss ≈300-450 token) vs dense 一段 ≈130 token,V100 逐 token 解码
   是瓶颈;JSON 失败遍还付重试且常顶满 512 上限。看帧成本两臂一样。
2. **全桶 2,532 片排期**:照现在拓扑(下游单卡)A≈62h / B≈27h;**②派生④b融合⑤验真改
   8 卡分片后 A≈11.7h、B≈12.9h——两臂打平**,A/B 取舍回归纯质量(等你 compare 包目检)。
3. **降本序**:分片(免费)→ struct max_new_tokens 512→384 → 全量只产目检胜者那版融合
   (④b 减半)。这三项都等目检结论一起定,现在不用动。

**无需新操作**——就差 compare 包目检四版的结论(A直融|A投票|B直融|B投票 哪版能当训练
字幕),回一句即可,我按胜者定全量生产配方并落 §5 排期。

## 2026-07-28 (44) — R1 处理臂跑通了(bank ON、无崩);还差对照臂才能判 go/kill

**处理臂 `nR1_textbank_pilot` 成功**:启动打了 `[R1] text bank ON: pool=(8999, 512) K'=16384
dedup_cos=0.9`,10 epoch 无崩。逐 epoch(`ret_test/` 行)见 `notes/r1-implementation-plan.md §9`。
要点:epoch0(接近冻结初始)最高 90.97,随后掉到 88.4 再慢爬回 ~88.9。这是"微调在这条容易的
域内 val 上略低于强零样本初始"的形状,**两臂都会有**——所以判据是**差值**,不是绝对值。
(epoch5 eval 打过一次 `sim_v2t has nan`,是已知 fp16 评测小毛病,没崩,数照出。)

**还不能判 go/kill**:kill 判据是"池 − 无池 < 0.5 R-Sum ⇒ 停",得有对照臂(`pilot_10k_r1`,
除 `criterion.text_bank` 外与处理臂完全相同)。**就差这一条**(约 38 分钟):
```bash
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
```
**报告回来**:对照臂逐 epoch 的 `txt_r1 / img_r1`(以及 best-epoch 六个 recall)。我据此算
两臂 R-Sum 差值,按 §5 判 go/kill——v2t(txt_r1)是文本池直接硬化的方向,t2v(img_r1)只间接受益。

## 2026-07-27 (43) — ASID 三修复已入库:✗改修复、JSON三级兜底、漏说哨兵——先 pull 再走 (41) 的两步

**维护者批「按建议执行」,§5.8 的 1-3 已实施合并**:

1. **✗ 句从删除改修复**:两版融合稿验真后若有 ✗,自动发一次"只修问题处"的最小修改
   调用并复验;终稿一律 `*_fused_*_final.jsonl`(无 ✗ 时=首轮验真件拷贝)。上轮 A 臂
   那 10 条直融 ✗ 这次会被修而不是扔;
2. **struct JSON 三级兜底**:重试@0.3 → 残缺 JSON 逐条抢救 → 该遍改跑 dense 段落
   兜底——**B 臂不会再丢片**(上轮丢 5/50);兜底次数落在 `n_dense_fallback`;
3. **漏说哨兵**:TSV 旗标新增 `MISS_D/MISS_V`(融合产出对 ≥2 票单元 gloss 覆盖
   <70% = 可能把主要信息浓缩丢了)。

**操作不变,只是先 pull**——(41) 的两步照走,修复自动生效:

```bash
git pull origin main
# ① 取包目检(命令同 (41)); ② 冷样本计时:
SAMPLE_OFFSET=200 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
```

**Then report back**(在 (41) 报告项上加三样): ⑤ 段的修复行(`n_repaired`/仍败数)、
`grep -o '"n_dense_fallback": [0-9]*' outputs/audit_10k/recap_struct.*.jsonl | sort | uniq -c`、
TSV 旗标列里 `MISS_` 的出现次数。

## 2026-07-27 (42) — R1 A/B 崩因是**粘贴把建缓存命令折断**,不是代码;缓存本身是对的,改名即可

**收到的输出说明两件事,都不是 bug:**

1. **`git pull origin main` 报 `divergent branches`**,pull 没生效(但 R1 代码更早那次已进来,
   所以 `_init_text_bank` 在;错在缓存缺文件)。修法见 TROUBLESHOOTING 新增节
   「服务器 git pull 报 divergent」——服务器只读,`git fetch && git reset --hard origin/main`。

2. **建缓存命令的 `\` 续行被粘贴拆断** → `--ckpt/--out-feats/--out-idx` 全丢,脚本用默认名把缓存
   建到了 `experiments/cache/textbank_feats.fp16.npy`(配置要 `..._pilot_...`),于是 A/B
   `FileNotFoundError`。**但那份缓存内容是对的**:投影前特征只由 eval 配置里的**冻结文本底座**
   (`mobileclip_s2.pt`,建模型时加载)决定,与 `--ckpt` 无关——`--ckpt` 只覆盖视觉 + 训练过的
   `projection_layer`,都在返回的「投影前特征」之后。所以 `--ckpt` 丢了不影响。缓存已是 `(8999,512)`、
   distinct idx 8999,正是要的。

**这就干(不必重建,改个名再跑处理臂)。⚠️ 每行都短到不会被终端折行——长 `cp` 全路径折行后
粘贴会拆成两条,`cp` 只拿到一个参数就报 `missing destination file operand`;先 `cd` 进 cache 目录用短名。
行尾也别带注释(注释被折行会当命令跑,上次 `cache/weights: command not found` 就是这么来的):**
```bash
cd ~/mobileclip/experiments/cache
cp textbank_feats.fp16.npy textbank_pilot_feats.fp16.npy
cp textbank_idx.npy textbank_pilot_idx.npy
cd ~/mobileclip
bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot.py 8
```
启动应打 `[R1] text bank ON: pool=(8999, 512) K'=8999 dedup_cos=0.9`。对照臂(若还没跑):
`bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8`。
(`git reset --hard origin/main` 上次已成功,HEAD=0b150b9,divergent 已修,这次不用再跑。)
> 顺手加固已推:① `_init_text_bank` 缺缓存时不再抛 8 条 numpy traceback,而是一行提示 + 正确的
> **单行**建缓存命令;② `r1_build_textbank.py` 文档串改成单行(别再用反斜杠折行——就是它被粘贴拆断的)。
> 下次真要重建,用 COMMANDS §3.3.1 里那条一行的命令。

**Then report back** in from-server.md:两臂逐 epoch 的 `txt_r1 / img_r1`(文本池硬化 v2t=txt_r1
直接、t2v=img_r1 间接),以及 `[R1]` 那行有没有出现。**kill 判据不变**:池 vs 无池 < 0.5 R-Sum ⇒
(d) 误判,停。

## 2026-07-27 (41) — A/B 对比跑通判读:三个发现 + 两步走(先取包目检,再拉计时跑冷样本)

**你贴的 pipeline-compare 全跑通**:A 臂 50 片、B 臂 45 片、对比包 50 片(both=45)。
三个发现:

1. **B 臂丢了 5 片(50→45)**:struct 重述三遍 JSON 全解析失败的片被静默丢弃(§5.6 登记
   的代价,但 10% 偏高)。先诊断再修——把这个贴回来:
   ```bash
   grep -o '"n_samples": [0-9]*' outputs/audit_10k/recap_struct.*.jsonl | sort | uniq -c
   ```
   (n_samples=0 的就是全失败片;若 1-2 遍的也多,说明 JSON 稳定性要加固)
2. **直融被 OCR 抓到 10 条 ✗(A 臂),投票浓缩 0 条**(B 臂: 直融 3 vs 投票 1)——方向与
   §5.7 登记的风险一致(融合调用可能改写引号文字/新增细节)。目检时**优先看直融的红色
   ✗ 条目**(recap_pipeline.tsv 的「直融✗」「直融dropped」列):是真幻觉还是 OCR 误杀?
3. **k=2 后投票健康度**:A 臂 2.1 单元/片、回退 20/50;B 臂 3.0 单元/片、回退仅 9/45
   ——struct 的属性结构+gloss 让跨遍投票明显更好使(早期信号,记档)。

**两步走**:

```bash
# ① 取包目检(A/B 并排,每片四版: A直融|A投票|B直融|B投票):
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
# 本地: scp rby@101vector:~/mobileclip/outputs/audit_10k/compare.tgz . && tar xzf compare.tgz

# ② 你这次跑时计时代码还没进你的 checkout(刚合并)——拉一次再跑冷样本出全量耗时表:
git pull origin main
SAMPLE_OFFSET=200 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
```

**Then report back**: ①的四版目检结论(哪版最像你要的训练字幕) + ②结尾两张 ⏱ 表 +
上面 n_samples 分布。顺带说一句:B 臂重述这轮肉眼计时 ~12.5 分钟/50 片(8卡)——struct
的 JSON 长输出比 dense 段落更慢,全量估算等 ⏱ 表定数。

## 2026-07-27 (40) — 流水线加了分段计时:跑一次冷样本,全量耗时表自动算出来

**变更(pull 生效)**:pipeline-sample 每段墙钟自动记录(`pipe[_st]_timing.tsv`),结尾打印
外推表:每段「秒/片 → 全桶(按队列实数)小时 → 每10万片小时」;pipeline-compare 结尾并排打
A/B 两臂耗时。新增 `SAMPLE_OFFSET=k`(从队列第 k+1 片取样)。

**关键:测真实成本必须用冷样本**——前 50 片各段全有缓存,计时会虚低到 ~0s。照这个跑:

```bash
git pull origin main
SAMPLE_OFFSET=200 N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
# 结尾自动出 A/B 两臂的分段耗时 + 全桶(2532片)/每10万片外推
```

**读数注意**:
- 外推按「①8卡并行 + 下游单卡串行」的现状算——全量生产时 ②派生/④b融合 也能
  `--num-shards 8` 分片,对应行可 ÷8;
- ①行含 8×模型装载(~1-2min)固定开销,大批量摊薄后更低;
- 每段原始秒数在 `outputs/audit_10k/pipe_timing.tsv`(A) / `pipe_st_timing.tsv`(B)。

**Then report back** in from-server.md: 把结尾两张 ⏱ 表整段贴回来——我据此把主计划
§5 成本表换成实测数,并给出全量(2532片)与语料级的排期建议。

## 2026-07-27 (39) — 「LLM 直接融合」已落地,且按最新指令改为与投票浓缩**双版并出**(§5.7 v2)

**指令已落地**:让 LLM 直接读 3/5 遍独立重述,一次调用完成「语义数票(≥2 遍支持才写)+
压缩成 2-5 条 ≤28 字短句」——参考 ASID(arXiv 2602.13013, 南开+字节+清华)的
Verify-refine 思路;模型必须**自报 dropped**(被排除的单源细节,落盘可查),GDINO/OCR
验真照旧是硬性后置门。**v2 更新(维护者:都出一版,要对比)**:不再设 FUSE_IMPL 二选一——
**每次同时产两版**:直融(`*_fused_d.*`) + 投票浓缩(`*_fused_v.*`),各自验真;
单臂评审包并排「直融|投票浓缩」两列,A/B 对比包每片并排**四版**:
A直融|A投票|B直融|B投票 + 四选一「最佳浓缩」。

```bash
git pull origin main
rm -f outputs/audit_10k/pipe_fused*.jsonl outputs/audit_10k/pipe_st_fused*.jsonl  # 清旧命名的融合产物
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare   # A/B 双臂 × 直融/投票 四版一次出
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
# 本地: scp rby@101vector:~/mobileclip/outputs/audit_10k/compare.tgz . && tar xzf compare.tgz
# 只跑单臂: N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample (A) 或 RECAP_STYLE=struct 同命令 (B)
```

**Then report back** in from-server.md: bundle_compare 四选一统计(50 片里 A直融/A投票/
B直融/B投票/都不行 各多少——多数版本即生产配方)+ 随手抽一片看
`outputs/audit_10k/pipe_fused_d.0.jsonl` 里 `verdict.dropped` 的自报淘汰是否合理。

## 2026-07-27 (38) — R1 池代码已落地(2.4c):建缓存 + 跑 A/B 一条龙

**GO 之后的池代码写完并推上来了,默认关=逐位不变。** 改动:`models/criterions.py`(池损失
`vtc_loss_sigmoid(text_bank=…)` + `_text_bank_nll`)、`models/mobileviclip_small.py`(`_init_text_bank`/
`_sample_text_bank`/forward 接线,**每步用当前 projection_layer 重投影 = 精确池**)、
`text_encoder.py`(`return_pre_proj` 暴露投影前特征)、`experiments/r1_build_textbank.py`(离线缓存)、
`experiments/configs/nR1_textbank_pilot.py`(A/B 处理臂,对照 = `pilot_10k_r1`)。

**池的数学我这边已 CPU 单测**(正例识别、去重恰好丢近重复假负例、fp16 往返都对齐参考实现);
`caption_to_idx` 与训练用的 `dataset.pt_dataset` 逐字节一致(池 idx = 训练 idx)。
**端到端前向/缓存只能你那边验**——建模型走的是 Part B 已跑通的同一条路。

**服务器一条龙(pull 后):**
```bash
cd ~/mobileclip && git pull origin main
export MVC_DATA_DIR=/home/vca1/datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ① 建缓存(投影前文本特征 + idx;单卡纯前向,几分钟):
python experiments/r1_build_textbank.py \
    --anno anno_downstream/internvid_zh_10k_pilot.json \
    --ckpt checkpoints/mobileviclip_small.pt \
    --out-feats experiments/cache/textbank_pilot_feats.fp16.npy \
    --out-idx   experiments/cache/textbank_pilot_idx.npy
# ② A/B 处理臂(启动日志应有一行 `[R1] text bank ON: ...`):
bash experiments/run_pretrain.sh experiments/configs/nR1_textbank_pilot.py 8
# 对照臂若没跑过:bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
```
判读:处理 vs 对照 held-out R@1,**两个方向都报**(池硬化 v2t=txt_r1;t2v=img_r1 只间接受益)。
kill 判据:池 vs 无池 < 0.5 R-Sum ⇒ 停。细节见 [`notes/r1-implementation-plan.md`](../r1-implementation-plan.md) §8 / COMMANDS §3.3.1。

**报告回来:** ① 缓存的 `feats.shape` 与 distinct idx 数;② 两臂逐 epoch 的 `txt_r1 / img_r1`(以及有没有报错 / `[R1]` 那行)。
若 ① 建模型报错,贴完整 traceback。

---

## 2026-07-27 (37) — Part B 跑通,近重复密度**低** → **GO(建 R1)**;下一步 2.4c 建池

**Part B 这次跑通了(路径修好、模型建起来、encode_text 形状正常,4998 条字幕全编码)。近重复密度低:**
| 余弦 ≥ | 平均近邻/条 | 有≥1近邻占比 | 每对概率 | ⇒ K=65k 时每查询假负例 |
|---|---|---|---|---|
| 0.90 | 0.166 | 7.3% | 3.3e-5 | ~2.2 |
| 0.95 | 0.040 | 2.0% | 8.0e-6 | ~0.5 |
| 0.99 | 0.002 | 0.14% | 4.0e-7 | ~0.03 |

**判定 = GO(建 R1)。** 对照 (35):`maxneg` 在 512 仍涨(+0.2/翻倍),而 Part B 说这些高相似负例里
**几乎都是真难负例、不是假负例**(≥0.95「基本同一句」的只占 2%)。这正是「仍涨 + 近重复低」那一行 ⇒ **建**。
去重掩码定 `dedup_cos=0.90`(保守),而且**近乎免费**——65k 池每查询才 ~2 对被标(0.95 时 ~0.5),稀疏掩码,预算算好即用。

**这不是「R1 一定有效」,是「前提成立,值得建臂 + 跑 A/B」。** kill 判据不变:512→65k 负例 <0.5 R-Sum ⇒ 停。

**下一步 2.4c 建池(方案 [`notes/r1-implementation-plan.md`](../r1-implementation-plan.md) §2 / §7.6):**
离线缓存**投影前**文本特征 + idx → `forward` 接入 bank → 扩 `get_mask`(把 bank 里同 idx 的当正例、
按 `dedup_cos` 掩近重复)→ fp32/分块算扩展 logits → 默认关的开关配置 `nR1_textbank.py`。
**默认关 = 逐位不变**(和探针同样的纪律)。这步是新代码,我会像探针那样尽量 CPU 单测池的数学、并标注哪些只能在服务器验。

**暂时不用在服务器跑任何东西**——等我把 2.4c 的代码写好推上来,再给你一条「建缓存 + 跑 A/B」的命令。
(校准注意:这是单个 InternVid 分片、zh 试点文本分布;建缓存那趟会在**真实池**上顺带复测一次密度。)

---

## 2026-07-27 (36) — pipeline-sample 判读 + 四条指令已落地:浓缩=最终字幕、覆盖优先、评审包精简、OwlCap 结构化=平行B臂(A/B对比)

**你贴的运行读完了。全绿**:重述/派生/浓缩全部按 video 秒级续跑(#274 的早退生效)、
多样性健康(Jaccard 0.226 无坍缩)、评审包 50/50 已打好。
**但 ④b 暴露真问题**:`34 fallback / 50` —— 68% 的片投不出任何 ≥3/5 票单元,浓缩列
退化成保底单句(重述被刻意做得多样,同一细节要 3/5 独立出现太严)。

**四条指令全部已实现并入库,pull 即生效**:

1. **浓缩短句 = 最终训练字幕**(生产主配方;RECAPTION 桶训练 `captions` 即浓缩短句
   去 ✗;G2 消融降级为验证性对照)——主计划 §5.5「定位更新」;
2. **不过度浓缩、覆盖优先**:投票门默认 3→**2**、全淘保底从留 1 条改**留前 3 条**、
   浓缩输出 2-4→**2-5 条**且 prompt 明令"宁多一条、不丢单元信息";
3. **评审包只显示四样**:英文en / 原zh(黄) / 重述×5(⟨SigLIP分⟩+★A−) / **浓缩**
   (+两道单选:重述质量、浓缩质量)。全景各阶段列仍在 `recap_pipeline.tsv`;
4. **OwlCap 式结构化重述 = 方案B 平行备选(§5.6,不覆盖方案A)**:每遍一次解码产
   **5-8 条带属性短句**(subject/action/scene/text,≤28 字+内联 gloss),**独立跑 3 遍**
   (temp0.9+帧抖动),派生步裁撤;**产物全部 `_st` 后缀隔离,方案A(稠密×5)主线不动**。
   为什么不能只跑一遍:一遍产的条目共享上下文、彼此不独立,一个幻觉会连贯污染整组,
   票数变假信号。新增 `pipeline-compare`:同批 50 片双跑 A/B,出并排对比包。

```bash
# ① pull 后一键 A/B 对比。A 臂重述/派生全缓存秒过;A 的浓缩因 prompt 更新要清一次
#   (只清 A,不碰 _st);B 臂全新但只有 3 遍×50 片。总计 ~20-30 分钟:
git pull origin main
rm -f outputs/audit_10k/pipe_fused*.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-compare
tar czf outputs/audit_10k/compare.tgz -C outputs/audit_10k bundle_compare
# 本地: scp rby@101vector:~/mobileclip/outputs/audit_10k/compare.tgz . && tar xzf compare.tgz
#       打开 bundle_compare/index.html —— 每片并排: en源|原zh|A重述|A浓缩|B重述|B浓缩 + 单选
# 顺手贴回 B 臂 JSON 健康度: grep -o '"n_samples": [0-9]*' outputs/audit_10k/recap_struct.*.jsonl | sort | uniq -c
```

**② GDINO + MiniRBT:这台机连 hf-mirror 也不通,只能本地下载后 scp 缓存**。不装的话
浓缩句验真永远缺"物体在不在画面里"这个轴、A′ 一直用 30 字回退预算(TROUBLESHOOTING 有存档):

```bash
# 在你【本地能上网的机器】:
pip install -U huggingface_hub
huggingface-cli download IDEA-Research/grounding-dino-base   # ~700MB
huggingface-cli download hfl/minirbt-h288                    # ~40MB
ssh rby@101vector 'mkdir -p ~/.cache/huggingface/hub'
scp -r ~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base rby@101vector:~/.cache/huggingface/hub/
scp -r ~/.cache/huggingface/hub/models--hfl--minirbt-h288    rby@101vector:~/.cache/huggingface/hub/

# scp 完成后在服务器上——验真按 video 续跑,必须清掉"GDINO 缺席"时代的旧判定让它重验:
rm -f outputs/audit_10k/pipe_verified.jsonl outputs/audit_10k/pipe_fused_verified.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
tar czf outputs/audit_10k/pipeline.tgz -C outputs/audit_10k bundle_pipeline
```

①② 相互独立;都做就按顺序来,② 的重跑会顺带把新浓缩句也过一遍 GDINO。
(另:上次末尾那行 `-bash: 重述/派生/打分...: No such file or directory` 是把注释行
当命令贴进终端了,无害。)

**Then report back** in from-server.md: A/B 两臂各自 ④b 的 `[done]` 行(kept/dropped/
fallback)+ **bundle_compare 四选一统计**(见块 (39):A直融/A投票/B直融/B投票/都不行)
+ B 臂 `n_samples` 分布(3 遍齐不齐)。

## 2026-07-27 (35) — 新指标读完:池杠杆在 512 **没饱和** → 倾向「建」,只差 Part B(其路径 bug 已修)

**重跑读到了修正后的两项指标。结论翻转:从上一条的「偏向砍」→ 现在「倾向建,取决于 Part B」。**

**关键读数 `maxneg_by_pool`(最难负例 logit,64/128/256/512 嵌套前缀):**
- step160: `-4.77 → -4.48 → -4.19 → -3.996`,每翻倍 +0.29/+0.29/+0.20 —— **到 512 仍在涨,增量近似恒定、没有变平**。
- 这正是 `最难负例 ~ log(池大小)` 还在**线性段**的特征。外推 512→65k(≈7 次翻倍)≈ 再 +1.4 ⇒
  **更大的池会持续捞到更难的负例。** 这**推翻**了我上一条基于「对齐初始化⇒杠杆已平」的先验猜测——数据不支持。
- `neg_grad_mass ≈ 0.7 且已稳`:批内负例还活着(不是全死),但批内难负例梯度已饱和,而更难的负例明明就在
  512 采样之外(见上)。这正是 SigLIP「批受限」的特征,轻微利好 R1。

**所以判据不再是「砍」。现在整件事只压在 Part B 上**(`maxneg` 涨,但探针分不清这些更难的负例是**真**难负例
还是**近重复假负例**——后者相似度也高,会同样抬高 `maxneg`):
| `maxneg`@512 | 近重复密度 (Part B) | 结论 |
|---|---|---|
| 仍涨(实测:是) | **低** | **建 R1**(按 §2 建 SigLIP 池;涨的是真难负例) |
| 仍涨(实测:是) | **高** | **先砍/修标签**(涨的多是假负例,不去重的大池放大错误排斥) |

**Part B 这次没跑成 —— 是我脚本的 bug,已修:** `r1_textfeat_dump.py` 直接当脚本跑时仓库根不在
`sys.path` 上(训练入口是通过 launcher 加的),报 `ModuleNotFoundError: No module named 'utils'`。
本轮已在脚本顶部插入仓库根;我这边验证过:`import utils.config` 现在能越过 `utils` 路径、进到正常的
依赖加载(easydict,服务器上有)。

**下一步(pull 后只需重跑 Part B ②;① 不用再跑,`maxneg` 已读到):**
```bash
cd ~/mobileclip && git pull origin main
python experiments/r1_textfeat_dump.py \
    --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
    --ckpt checkpoints/mobileviclip_small.pt --out /tmp/txtfeats.npy --sample 10000
python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy --thresholds 0.9,0.95,0.99
```
若建模型那步又报错,把**完整 traceback** 贴回(最可能是 `Config.from_file` 字段或 `encode_text` 返回形状,
按报错微调;字幕读取已验证 4998 条 OK)。完整推导见 [`notes/r1-implementation-plan.md`](../r1-implementation-plan.md) **§7.5**。

**报告回来:** Part B 三个阈值(0.9/0.95/0.99)下「有≥1 近邻」的占比。低 ⇒ 建 R1;高 ⇒ 砍/先修标签。

---

## 2026-07-27 (34) — R1 go/no-go 试点结果:不建、暂不砍(探针指标当时选错,已修)

**你贴回的 `[SAT]` 8 行读完了。结论:go/no-go 是 inconclusive,偏向砍,但先别动手建池。**

**数字:** `top1_is_pos` 全程 ≈0.67–0.77(随机基线 ≈1/512≈0.002),`mean(pos-hardestneg)`
0.25→0.60 且**还在涨**。试点 val 稳在 t2v R@1≈76–77 / R@5≈92。

**关键更正(我上一版探针错了,这轮已改 `models/criterions.py`):**
1. 旧判据「`top1_is_pos→1 ⇒ 批内饱和 ⇒ 大池无益 ⇒ 砍」是**错的科学**。最难负例的相似度随池
   `~log(池大小)` 增长——512 里全分开,32k 池几乎仍能捞到更难的负例。「512 全分开」和
   「大池能加梯度」**不互斥**。而且 `top1=0.73`(不是 →1),这条判据本来就没触发。
2. `top1/margin` 是**排序/分离度代理**,**不是**决定「大池有没有用」的量。sigmoid 下负例的梯度
   `∝ σ(t·s+b)`,只有落在边界活跃区的负例才有梯度。判据应看**梯度质量**,不是排序。

**修正后探针现在多记两项(同一 `[B,B]` 矩阵即可算,不额外训练):**
`maxneg_by_pool`(最难负例随 64/128/256/512 前缀是否仍在涨——直接回答「大池会不会加更难负例」)
和 `neg_grad_mass`(批内负例的有效梯度质量,≈0 ⇒ 负例已「死」)。

**为什么 0.73 既不能砍也不能建:** 27% 的查询批内有负例压过正例——**没饱和**;但这被两件事污染:
(a) 10k/3-epoch 上「margin 还在涨」多半是**记忆**,不代表全量语料;(b)**假负例**——近重复但
`idx` 不同的字幕被当负例,它们本就是高 logit 的「难负例」,同时**压低 top1、抬高 margin**,却不是真信号。

**下一步(两项零训练检查,按 §7 决策表判):**
```bash
cd ~/mobileclip && git pull origin main
export MVC_DATA_DIR=/home/vca1/datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ① 重跑同一试点,读新指标 maxneg_by_pool / neg_grad_mass:
MVC_LOG_SAT=20 bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
grep '\[SAT\]' experiments/runs/pilot_10k_r1/train.log | tail -20
# ② Part B 近重复密度(定去重掩码;dump 建模型部分仍未在真权重验证,报错贴 traceback):
python experiments/r1_textfeat_dump.py \
    --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
    --ckpt checkpoints/mobileviclip_small.pt --out /tmp/txtfeats.npy --sample 10000
python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy --thresholds 0.9,0.95,0.99
```
**判:** `maxneg_by_pool` 到 512 已平 ⇒ 砍。仍涨 + 近重复密度高 ⇒ 砍/先修标签。
仍涨 + 密度低 ⇒ 才建 R1(三种里最不可能——对齐初始化+冻结文本塔+小数据,预期这杠杆很平)。

**顺带一个建池前置结论:** 近重复去重掩码从「Part B 高才加」升级为**建池必需**——假负例正是
sigmoid 大池里主导梯度的高 logit 对,不去重 = 放大错误排斥。完整推导见
[`notes/r1-implementation-plan.md`](../r1-implementation-plan.md) **§7**(附 `deep-learning-researcher` 复核)。

**报告回来:** 新 `[SAT]` 行(`maxneg_by_pool[...]` 与 `neg_grad_mass`)+ Part B 的近邻占比。

---

## 2026-07-27 (33) — pipeline-sample 新增 C_fuse 投票浓缩段：pull 后重跑一次（增量只需几分钟）

**变更**（维护者指令：不满足于 A′ 机械切句 → 跨重述投票筛细节、浓缩成高浓度短句；
主计划 §5.5）：`pipeline-sample` 在派生/打分之后新增 ④b——`fuse_condense.py` 把
A_llm 短句跨 5 条重述聚类计票（纯规则、不占卡），support≥3 的单元交 Qwen 文本-only
打包成 2–4 条 ≤28 字浓缩短句，再照走 GDINO/OCR 验真。TSV/评审包新增列：
`投票单元(票数) | 淘汰单元 | ✓/·/✗浓缩` + 第三道判读单选「浓缩 vs 逐条」。

**Do this next**（重述/派生/打分全部按 video 续跑秒过，增量只有浓缩+验真+重出表）：
```bash
cd ~/mobileclip && git pull origin main
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
tar czf outputs/audit_10k/pipeline.tgz -C outputs/audit_10k bundle_pipeline
# 本地: scp rby@101vector:~/mobileclip/outputs/audit_10k/pipeline.tgz . && tar xzf pipeline.tgz
```

**Then report back** in from-server.md：`pipe_units.jsonl` 尾部打印的
`voted units kept (X/clip), Y dropped, Z fallback` 三个数 + 目检 3–5 片：
「淘汰单元」列里被投掉的是幻觉还是真细节？「✓/·浓缩」读起来比「A′视图/✓视图」
更好吗（第三道单选顺手勾）？若 fallback 多（>20%），下次跑加 `FUSE_MIN_SUPPORT=2`。

(31) 说"要我写那两行饱和探针吗" —— 已写并合入,默认关、只读、单测过
(`models/criterions.py` 的 `saturation_stats` / `_maybe_log_saturation`,`MVC_LOG_SAT` 打开)。
所以三档下一步现在**全是精确命令**,见 [`COMMANDS.md`](../../COMMANDS.md) §3.3。最该先跑的:

```bash
cd ~/mobileclip && git pull origin main
export MVC_DATA_DIR=/home/vca1/datasets
MVC_LOG_SAT=5  bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 debug True   # 冒烟:确认打印 [SAT]
MVC_LOG_SAT=20 bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8              # 正跑 ~180 步
grep '\[SAT\]' experiments/runs/pilot_10k_r1/train.log | tail -20
```

判读:`top1_is_pos`→≈1.0 且 `mean(pos-hardestneg)` 明显为正 ⇒ 批内 512 已饱和 ⇒ **R1 更大负例池无益,砍**;
否则 ⇒ (d) 是真瓶颈 ⇒ 值得按 `notes/r1-implementation-plan.md` 建池。

**Then report back:** `train.log` 里最后几行 `[SAT]`(top1_is_pos 与 margin 的收敛值)。
这一条就能定 R1 的 go/no-go —— 不用先建任何池子。

---

## 2026-07-27 (31) — 「都做」:Part B 现在一条命令 + R1 落地方案已成文;真正的 go/no-go 是训练期饱和探针

**两件都落库了。**

**① Part B(近重复密度)现在一条命令能出** —— 之前卡在"要自己写建模型的脚本"。
新 `experiments/r1_textfeat_dump.py` **复用 GATE-0 评测在服务器上跑通的同一条构建路径**
(`Config.from_file(eval配置)` → `MobileViCLIP_Small(config)` → 模型自己的 `load_checkpoint`
做键重映射,再 `strict=False` 覆盖 small.pt 取训练后的投影),不重造 setup_model:

```bash
python experiments/r1_textfeat_dump.py \
    --anno anno_downstream/InternVid/InternVid-10M-flt_1.json \
    --ckpt checkpoints/mobileviclip_small.pt --out /tmp/txtfeats.npy --sample 10000
python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy --thresholds 0.9,0.95,0.99
```
判读:「有≥1近邻」占比高 ⇒ R1 需余弦去重掩码;低 ⇒ 免。
⚠️ 该脚本的**建模型/前向未在真权重上验证**(我这边没权重)——字幕读取/去重/分块已自测,
构建若报错把**完整报错**贴回,最可能是 `Config.from_file` 字段或 `encode_text` 返回形状,微调即可。

**② R1 落地方案成文** —— [`notes/r1-implementation-plan.md`](../r1-implementation-plan.md)。
按 examination §6 重构为「SigLIP 制度」一条臂,含:触点(criterions/forward/config)、
**穿过 `get_mask` 的损失参考实现**(不是 `cross_entropy(arange)`,否则注入假负例)、
fp16 的 fp32+分块(且指出 sigmoid 下 logit 尺度≈10 而非 100,本就远离溢出)、
默认关=逐位一致的开关、以及离线缓存投影前特征的"精确"机制(附 dropout=0 的不变量断言)。

**但最关键的一句(别被"精确负例库"带跑偏):**
**R1 的 go/no-go 不是 Part A/B,是训练期的批内饱和探针。** 而且它最便宜的形式**不需要**先建池子:
拿现成的 `r1a_sigmoid.py`(sigmoid+可学习 t/b,已实现)+ 加两行日志(banked/批内负例的 softmax 质量占比、
正样本在 512 内的排名),跑一个 10k 试点即可回答"(d) 在我们这个对齐初始化的 3-epoch 制度下到底真不真"。
若显示 512 就已饱和 ⇒ 不必建池子,直接砍。方案 §4/§5 写了这条最省的路径。

**Do this next(三选,按信息价值排序):**
1. **最省的 go/no-go**:`r1a_sigmoid.py` + 两行饱和日志,10k 试点(方案 §4)。**这一条最该先跑。**
2. Part B 近重复密度(上面 ①):定 `dedup_cos`,顺便验证 `r1_textfeat_dump.py` 在真权重上能跑。
3. Part A 收紧到 55k:`--anno …flt_*.json`(秒级,可选)。

**要我把 §4 的两行饱和探针直接写进 `criterions.py` / `r1a` 配置吗?** 那是能立刻产生 go/no-go 的最小改动。
说一声我就写(会是默认关、可跑的最小 diff,不是又一份方案)。

---

## 2026-07-27 (30) — R1 前置探针 A 出数:精确重复 ≈0 ⇒ 免做精确去重;近重复(B)成为唯一待测设计项

**你跑的 Part A(InternVid-10M-flt_1,5000 条):** 4998 条不同(**99.96%**),
随机一对同字幕概率 **≈1.6e-7(0.0000%)** ⇒ 65k 池里期望 **0** 个精确假负例。

**含义(这条结果替 R1 减了一项风险、也缩了要写的代码):**
- **examination §3 的"精确匹配假负例"担忧基本不成立。** 65k 池几乎抽不到精确重复,
  所以即使负例库**不做**精确 `idx` 去重,也几乎没有精确假负例污染。穿过 `get_mask` 仍是
  好习惯,但**不是能把 R1 搞砸的那一环** —— 关键路径上少一个风险点。
- **顺带**:这也说明 B0/R0 没设 `caption_idx`(只有 r1a 设了)其实几乎零代价 ——
  InternVid 上那条多正例项本来就近乎空。
- **风险现在全压在 Part B(近重复)** —— idx 抓不到的、投影空间里语义相近的字幕。
  样本里那种通用句("a woman with red hair singing into a microphone")正是"精确重复 0、
  近重复多"的典型。**Part B 决定要不要加基于余弦阈值的去重掩码。**

**边界(别读过头):** Part A / B 都**不是 R1 的 go/no-go**。go/no-go 是**训练期**的饱和探针
(banked 负例占 softmax 质量 <5% ⇒ 库惰性;正样本在 512 批内已 rank-1 ⇒ 512 就饱和)。
前置探针只是(i)排除一个失败模式、(ii)定一个设计旋钮(去重掩码要不要)。

**接下来二选一(你定):**

**① 想把前置探针做全**(都很便宜):
```bash
# A 收紧到 55k(11 片,仍是秒级)
python experiments/r1_fn_density.py --anno anno_downstream/InternVid/InternVid-10M-flt_*.json
# B 近重复:先在训练环境生成文本特征(片段见下),再算密度
python experiments/r1_fn_density.py --help-feats     # 打印生成 .npy 的片段(复用 eval 的 encode_text)
python experiments/r1_fn_density.py --feats /tmp/txtfeats.npy --thresholds 0.9,0.95,0.99
```
Part B 的判读:「有≥1近邻」占比高 ⇒ 必须上余弦去重掩码;低 ⇒ 免了。

**② 直接开 R1 训练臂**(Part A 已说可跳过精确去重;近重复掩码可等 B 或按需再加)。
按 examination §6 重构为 **SigLIP 制度**一条臂:sigmoid 损失(已实现 `r1a_sigmoid.py`)+
可学习温度 + 大文本负例池,**臂内埋饱和探针**(才是 go/no-go)。
你说一声要走 ②,我把「离线缓存投影前文本特征 → 每步重投影拼进 `get_mask` → 三件套(池子/温度/sigmoid)」
的落地步骤写进 `COMMANDS.md` + 新配置,连 fp16 溢出的 fp32/分块处理一起给全。

**要我现在写 ② 的落地方案吗?** 还是你先把 B 跑了我们再定池子大小?

---

## 2026-07-27 (29) — 2 卡冒烟几乎全过！只栽在别人占了卡的 OOM(不是我们的问题)

**Diagnosis:** 这是**大进展**。冒烟这次一路走通:导入链、`scripts` 包、配置加载、
**InternVid 预训练视频真的解码进了 forward**(崩在 `patch_embed`,说明 dataloader 产出了
真实 batch → §4 的预训练数据确实读得动)、MSR-VTT 评测 anno 载入 1000/1000、从
`mobileclip_s2.pt` 载入双塔、bf16、优化器、dataloader(83183 batch)、`Start training /
Epoch: 0` 全部到位。

崩在第一个训练步的 `CUDA out of memory`——但**不是模型太大**。看报错的 per-process 分解:
GPU0/1 各有一个**别人的**进程占了 ~70GB,我们的进程才占 ~9GB。`CUDA_VISIBLE_DEVICES=0,1`
恰好选中了两张被别人吃满的卡。MobileViCLIP-small 本身只要 ~9GB,**换两张空卡就能打印 loss**。

冒烟日志里这两条**是正常的,别当错误**:
- `_IncompatibleKeys(missing_keys=[... mixer_1d ... temporal_embedding ...])`:视频专用时序
  参数,图像底座里没有,按设计新初始化(`unexpected_keys=[]` 是关键)。
- `No pretrained checkpoint provided, training from scratch`:指没有整模型续训 ckpt;双塔已
  从 `mobileclip_s2.pt` 载入。R0 本该如此。

**Do this next**(`git pull` 拿到我加的选卡工具):

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip
git pull origin main
source experiments/env_b100.sh

# 1) 挑两张真空闲的卡(free>70000 MiB 才算空)
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s  free=%6s MiB  used=%6s MiB  util=%s%%  %s\n",
               $1,$2,$3,$4,($2>70000?"kong-OK":"占用")}'

# 2) 把 G 换成上面 OK 的两张卡号,重跑冒烟
G=4,5   # ← 改成两张空卡号,别盲抄
CUDA_VISIBLE_DEVICES=$G bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 2 \
    use_bf16 True bf16 True deepspeed.enable False debug True
```

**Then report back**(from-server.md):冒烟有没有打印出 loss(几十步自停)。
- **打印了 loss** → 冒烟正式通过,接 §6 步骤 2 真 R0;**先用同一句选卡工具确认要用的 4 张卡
  都空**,前 4 张被占是要和你/中文线协调的事(等空或换空卡),不要 kill 别人进程、也别开满 8 张。
- **又 OOM** → 贴新的内存分解,大概率还是没挑到真空的卡。

⚠️ 共用机纪律(§0):只挑空卡,不 kill 占卡的进程。坑已固化到 `TROUBLESHOOTING.md`
「冒烟 CUDA OOM——共用机上别人占了卡」+ §6 步骤 1 现在带选卡工具、不再硬编码 0,1。

---

## 2026-07-24 (28) — 八卡各挂 17GB 零利用率 = 滞留的重述分片；核对→清理→重跑（两处已加固）

**Diagnosis:** 你贴的 nvidia-smi 判断正确：**0% 利用率 + 17GB 常驻 = 进程持显存不干活**。
PID 932606–932613 连号、一卡一个、各 17032MiB——就是一次 8 卡重述分发装载的 Qwen3-VL-8B
残留（我们自己的分片；GPU6 那堆 404–504MB `python3` 是别的小任务，**不要动**）。
两处根因已修（本块随附 PR）：① `vlm_audit_hf` 原来**先装 17GB 模型再查有没有活干**——
续跑重启时 8 个 todo=0 的分片也会各压 17GB；现在改为**先查 todo、为空直接退出、不碰 GPU**。
② `recap` 子命令加了防重复启动守卫（已有分片在跑时拒绝再叠 8 份装载）。

**Do this next（101vector；先核对再清理，共用机器不盲杀）：**
```bash
# ① 确认那 8 个进程是重述分片（cmd 应含 vlm_audit_hf.py --mode recap）:
ps -o pid,user,etime,cmd -p 932606 932607 932608 932609 932610 932611 932612 932613
# ② 是的话清掉（只会杀你自己有权限的进程）:
pkill -f "vlm_audit_hf.py.*--mode recap"
sleep 3; nvidia-smi | head -20        # 各卡应回到 ~0-1GB 占用
# ③ 拉修复并重跑（重述已缓存, 全程 ~半小时）:
git pull origin main
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
tar czf outputs/audit_10k/pipeline.tgz -C outputs/audit_10k bundle_pipeline
```
若 ① 显示的不是 recap 分片（cmd 不同或 user 不是你），**别杀**，把 ps 输出贴回来我再看。
顺手贴一下 `tail -n 3 outputs/audit_10k/recap_shard*.log`——我想确认分片当时停在哪一步。

---

## 2026-07-24 (27) — 派生步 OOM：GPU0 被占（16.6GB 外部进程），已改为自动选空卡

**Diagnosis:** 你的 pipeline-sample 在步② 装 Qwen3-VL-8B 时 OOM——不是代码错：GPU0 上有个
16.63 GiB 的外部进程（PID 922896，多半是并行实验），而单卡步骤默认钉死 `PIPE_GPU=0`；
8B fp16 装载需 ~19GB，只剩 13.9GB。已修（本块随附 PR）：`PIPE_GPU` 未指定时**自动选剩余
显存最大的卡**，全卡都 <19GB 时会明确报错并提示等待/指定。tar 失败只是链条中断的连带
（bundle 没生成），修复后重跑即得。

**Do this next（101vector）：**
```bash
git pull origin main
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample   # 会打印 [gpu] 选了哪张卡
tar czf outputs/audit_10k/pipeline.tgz -C outputs/audit_10k bundle_pipeline
```
（不想 pull 的立即绕法：`nvidia-smi` 挑张空卡，`PIPE_GPU=<卡号> N_SAMPLE=50 bash … pipeline-sample`。）
本地取回同 (26)：`scp rby@101vector:~/mobileclip/outputs/audit_10k/pipeline.tgz . && tar xzf pipeline.tgz`

**Then report back** 同 (26)：页头「视频 X/50」、重述 vs 旧译的判感、红✗ 视图真幻觉还是误杀。

---

## 2026-07-24 (26) — 目检改成 better_zh 同款评审包：index.html + 视频 + 打分单选（取代 (24) 的单文件 HTML）

**Diagnosis:** 应维护者要求，pipeline 目检改用与 better_zh/merge 抽检完全同款的离线评审包
（`make_review_bundle.py`，本次给它加了「重述/视图」块的渲染与 `pipeline` 判读提示）。
`pipeline-sample` 第 ⑥⑦ 步现在自动产 `review_pipeline.tsv` → `bundle_pipeline/`
（index.html + videos/ + README.txt）：每片一卡 = **视频播放器 | en 源 | 黄色旧译 |
重述 1–5（⟨SigLIP 分⟩、★A− 徽标）| A′ 视图 | 绿✓/灰·/红✗ 视图块** + 两道判读单选
（重述质量、被滤视图判读）+ 备注，进度存浏览器、填完可导出 TSV。

**Do this next（101vector）：**
```bash
git pull origin main
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample   # 续跑秒过, 只补打分/表/包
tar czf outputs/audit_10k/pipeline.tgz -C outputs/audit_10k bundle_pipeline
```
**本地机器**取回并打开：
```bash
scp rby@101vector:~/mobileclip/outputs/audit_10k/pipeline.tgz . && tar xzf pipeline.tgz
# Windows 双击 bundle_pipeline/index.html；macOS: open …；Linux: xdg-open …
```
页头会显示「视频 X/50」——若是 0/50 说明打包时没找到视频（报回来我查路径），个别放不了
多半是 HEVC 编码换 Firefox（README.txt 有说明）。

**Then report back**：页头视频计数、你点过 10 片后对「重述 vs 旧译」的总体判感、
红✗ 视图看着是真幻觉还是误杀（那两道单选填了导出 TSV 传回更好）。

---

## 2026-07-24 (25) — MSR-VTT 视频"全部缺失"：软链接错了一层（我的文档写错了）

**Diagnosis:** anno 本身完全没问题，是**我上一版 §4b 步骤 3 写错了**。

你贴回来的探盘结果里有关键一行：

```
{"video": "TestVideo/video9770.mp4", "caption": [...]}
```

`video` 字段**带子目录前缀**。而仓库拼视频路径只有一种写法、**任何地方都不取 basename**：

```python
join(data_root, anno["video"])        # dataset/utils.py:121，check_readiness.py 同款
```

所以它要找的是 `data/MSRVTT_Videos/TestVideo/video9770.mp4`，
而步骤 3 让你把 mp4 **拍平**软链，给出的是 `data/MSRVTT_Videos/video9770.mp4` —— **差一层**。
10000 个软链一个都对不上，就是这个原因。

其余都是好的：anno 是仓库原生格式（list/1000 条）、`caption` 是列表也没关系
（`ret_dataset.py:323` 有 list 分支）、`TestVideo/` 覆盖 1000/1000。**MSR-VTT 不用下载任何东西。**

我在本地用 fixture 复现了这个失败、也验证了下面这个修法能翻成 `[OK ]`。

**Do this next:**

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip
git pull origin main
source experiments/env_b100.sh

MSRVTT=/workspace/mount/b100_zaip_data/video_datasets/MSR-VTT

# 清掉拍平的那批（-type l 保证只删软链，真文件一个都碰不到）
find data/MSRVTT_Videos -maxdepth 1 -type l -name '*.mp4' -delete

# 改成把子目录整个软链进去
for sub in TestVideo TrainValVideo; do
  [ -d "$MSRVTT/$sub" ] && ln -sfn "$MSRVTT/$sub" "data/MSRVTT_Videos/$sub"
done

ls -l data/MSRVTT_Videos/                          # 只剩 TestVideo、TrainValVideo 两条软链
python experiments/check_readiness.py --decode 2>&1 | grep -i msrvtt
```

期望：`[OK ] msrvtt   anno 1000 条;抽查 20 条,缺失 0;试解码 OK(N 帧)`。

**顺带改了 `check_readiness.py`**：以后报"全部缺失"会把**它试过的完整路径**打出来，
不用再猜是哪一层错了。剩下 6 个基准接线时，**每个都要先看一眼 `anno[0]["video"]`**
是纯文件名还是带子目录（各家 anno 不一样），别照抄 MSR-VTT 的接法 —— 规则写在
§4b 步骤 3 的表里，坑记在 `TROUBLESHOOTING.md`「预检报"抽查 N 条视频全部缺失"」。

**Then report back** in from-server.md：上面那条 `grep -i msrvtt` 的输出。

⚠️ **但真正的关键路径仍然是 (19) 的 2 卡冒烟** —— §4b 这一整节都不挡 R0–R3
（预训练只用 `internvid_all`，§4 已全绿）。冒烟先跑，MSR-VTT 顺手修。

---

## 2026-07-24 (24) — pipeline-sample 首跑判读：链路 ✅、多样性 ✅；SigLIP 路径 bug 已修（#265），重跑补两列

**判读（好消息为主）：** 50 片全链跑通。**温度坍缩红旗未触发**——片内成对 3-gram Jaccard
中位数 0.226、完全相同对 0/50，五条重述有真方差，投票信号活着（主计划 §7 第一行担心的
失效模式不存在）。OCR 档已经在干活（9 条含引号文字的视图被滤，TSV 的 ✗ 列可直接目检
这 9 条是真幻觉还是花体字漏检）。验真档缺 GDINO 时保守保留 97%，符合设计。
重述吞吐实测 ≈ 250 样本/5.5 分钟/8 卡 → 全桶 2,532×5 ≈ **4–5 小时**（§5.1 表可更新）。

**两个问题与修法：**
1. **步③ SigLIP 打分 0/50**（`ERROR opening .../videos/videos/...` 双重前缀）——
   `score_alignment` 解析器缺 `vlm_audit` 的前缀回退，**#265 已修**；TSV 的 `⟨SigLIP分⟩`
   与 `A_minus_keep` 两列因此为空，重跑即补。见 `TROUBLESHOOTING.md` 新条目。
2. **hf-mirror 当时连不上** → MiniRBT tokenizer 落到 30 字符预算（可接受）、GDINO 未加载
   （gloss 检查全部 unverified）。网络恢复后预下载这两个小件即可补上。

**Do this next（101vector）：**
```bash
git pull origin main
# ① 重跑（重述/派生/验真按 video 续跑秒过；只重算打分 + TSV，几分钟）
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
# ② 网络通时预下载两个非 LLM 小件（GDINO ~700MB + MiniRBT ~50MB）：
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download as d; d('IDEA-Research/grounding-dino-base'); d('hfl/minirbt-h288')"
# ③ 下载成功后补 GDINO 验真档（必须先删验真缓存让它重验）：
rm -f outputs/audit_10k/pipe_verified.jsonl
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
```
② 若 mirror 持续不通就跳过（TSV 除 gloss 验真列外全部可用），报回来我给替代下载路线。

**Then report back**：重跑后 TSV 任挑 3 行贴 `en_source / zh_translated / recap_1(含⟨分⟩) /
A_minus_keep / views_fail` 五列 + `pipe_diag.txt` 全文；顺带说一句那 9 条 ✗ 视图看着像
真幻觉还是 OCR 漏检。

---

## 2026-07-24 (24) — ✅ **GATE-0 结案**:闸门比 0.0613 + 同初值 ⇒ 三条证据齐了,**不买基准,转投 R1**

**Diagnosis:** 你跑出的第 3 节把最后一块拼图补上了,而且我去核了初始化,**排除了它最主要的替代解释**。

| 量 | 中位数 |
|---|---|
| `\|layer_scale_1d\|`(时序闸门) | **0.0008** |
| `\|layer_scale\|`(同层空间闸门) | **0.0130** |
| **时序/空间 闸门比** | **0.0613 ≈ 1/16** |

**关键:两者初始化完全一样。** `models/mobileviclip_small.py:636-641`:

```python
self.layer_scale    = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)))
self.layer_scale_1d = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1)))
#                     ^^^^ 同一个 layer_scale_init_value = 1e-5
```

从**同一起点 1e-5** 出发,训练把**空间**闸门推到 0.0130(约 **1300×**),
把**时序**闸门只推到 0.0008(约 **80×**)。
⇒ **这 16 倍差距是训练结果,不是初始化差异。** 时序支路不是"没训",是"训得少得多"。

**三条独立证据现在齐了,方向完全一致:**

| # | 证据 | 数字 |
|---|---|---|
| 1 | 行为 | 打乱 8 帧 **+0.1/+0.2**(≈0);单帧重复 **−13.0/−12.1** |
| 2 | 核形状 | 反对称占比 **0.5011** vs 随机 null **0.4999**(逐分位点吻合,n=9920) |
| 3 | 闸门 | 时序/空间 = **0.0613**,同初值 1e-5 |

**⇒ MobileViCLIP 这份官方权重里的"时序重参数化"支路:闸门被压低约 16 倍、
核在方向性上与随机无异、行为上打乱帧序不掉分。**
措辞要准:我们说的是**这份公开权重**里该支路贡献小且无方向性,**不是**"该设计原理上无效"。

**⇒ 决策(GATE-0 闭环):**
1. **不买顺序敏感基准**(RTime / SSv2 都暂缓)—— 省掉一整轮视频抓取。
   只有将来要证明 R2 的**下游收益**时才回收(`notes/order-sensitive-eval.md` §6 已标注终止)。
2. **下一步投 R1(精确文本负例库)** —— 它打的是瓶颈 (d)(全局 batch 512,距 SigLIP 饱和点
   30–60×),用普通检索就能测,而 MSR-VTT 测这个是**有效**的。设计见
   `notes/neural-inspired-round2.md` §3 R1;ROADMAP 第二线 2.4。

**你这边暂时没有必须跑的东西。** GATE-0 三项(静态/动态/判读)都已闭环并落库:
`experiments/gate0_results.md` §8 是结论页。recap 主线不受影响,照常推进。

**要不要现在开 R1?** 它需要先把 InternVid 字幕的**投影前**文本特征离线缓存一遍
(文本塔冻结 ⇒ 这份特征全程恒定 ⇒ 负例库数学上精确、零陈旧)。
你说一声,我把缓存脚本 + 配置 + 三臂 A/B 的落地步骤写进 `COMMANDS.md` 和一份新 note。

---

## 2026-07-24 (23) — 探盘范围收窄：只在 `video_datasets/` 里找，别扫整个挂载点

**Diagnosis:** 维护者要求：**找数据集只在
`/workspace/mount/b100_zaip_data/video_datasets/` 之内**。已把这条写成 SETUP_B100 §0 的
标准规则（和"杀进程只按 PID"并列），并改掉仓库里两条会越界的命令：

| 位置 | 原来 | 现在 |
|---|---|---|
| `SETUP_B100.md` §4 | `find "$FULL" /workspace/mount/b100_zaip_data* -maxdepth 4 …` | 只在 `$FULL` 和 `video_datasets/vd-foundation___InternVid-10M-FLT` 两个**确切目录**里找 |
| `tools/data_audit/SETUP_V100.md` | `find /workspace/mount/b100_zaip_data -maxdepth 4 -iname "*InternVL*"` | 只在 `model_zoo/` 里找 |

**(22) 的 §4b 探盘命令本来就已经限定在 `$VD` 内**（`for d in "$VD"/*/` 和
`find "$VD" -maxdepth 4`），无需改动，照原样跑即可 —— 现在只是多了一行显式提醒。

理由有两条，都不是形式主义：整挂载点底下是**别人的工作目录**（wky / zyc / model_zoo），
既不该翻，几百万文件的 `find` 也会实打实拖慢这台共用机器。

**顺带一个观察**：`video_datasets/` 里有个 `vd-foundation___InternVid-10M-FLT/` —— 和 §4
现在用的 `b100_zaip_data_luan/InternVid-10M-FLT` 可能是同一份数据的另一个副本。**§4 已经
全绿，不要去动它**；只是万一将来 `_luan` 那份出问题，这里有个备选，顺手记一笔。

**Do this next：** 没有新命令。`git pull` 拿到收窄后的文档，然后**继续 (22) 的步骤 1
三段探盘**（那三条本来就是合规的）。真正的关键路径仍是 **(19) 的 2 卡冒烟**。

---

## 2026-07-24 (22) — 评测数据**本机就有**：MSR-VTT 自带现成的 `msrvtt_1k_test.json`，别再 gdown

**Diagnosis:** 你贴的目录是个大发现。`/workspace/mount/b100_zaip_data/video_datasets/`
里存着上百个视频数据集，而且 **`MSR-VTT/mobileviclip_annotations/` 里就有一个
`msrvtt_1k_test.json`** —— 文件名和 `configs/data.py:48-52` 要的**一字不差**，说明有人
（多半是同机的另一条线）已经按 MobileViCLIP 的格式备好了。旁边还有 `README.md` 和
`status.json`，先读一眼能知道备到什么程度。

同一目录下还能对上另外三个基准（DiDeMo / ActivityNet / Kinetics-400）；**UCF101、HMDB51、
SSv2 在你贴的列表里没看到**（`ucf_crimes` 是另一个数据集，别认错）。这和 (20) 里
"SSv2-MC 是缺数据"的结论一致。

接线只用**软链**：源目录是多人共用的，只读着用，绝不复制、不改、不往里写。

**Do this next（b100，仓库根；步骤 1 是零成本探盘，先只跑它）：**

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip && git pull
VD=/workspace/mount/b100_zaip_data/video_datasets

# ① 还有哪些数据集也备好了 MobileViCLIP 格式的 anno（MSR-VTT 就是这么发现的）
for d in "$VD"/*/; do [ -d "$d/mobileviclip_annotations" ] && echo "$d"; done

# ② 仓库要的 7 个 anno 文件名，逐个在盘上找
for f in msrvtt_1k_test.json didemo_ret_test.json anet_ret_val.json \
         kinetics400_validate.json ucf101_act_test.json hmdb51_act_test.json ssv2_mc_val.json; do
  printf '== %s\n' "$f"; find "$VD" -maxdepth 4 -name "$f" 2>/dev/null | head -3
done

# ③ 那个目录里的说明读一眼
cat "$VD/MSR-VTT/mobileviclip_annotations/README.md"
cat "$VD/MSR-VTT/mobileviclip_annotations/status.json"
```

拿到结果再走 **§4b 的步骤 2/3**（验 anno 格式 → 数 `TestVideo`/`TrainValVideo` 覆盖率 →
软链 `anno_downstream/msrvtt_1k_test.json` + `data/MSRVTT_Videos/`）。命令都在
`experiments/SETUP_B100.md` §4b 里，`CHECKPOINTS.md` 也加了一张"仓库期望 ↔ b100 实际路径"
对照表。

**⚠️ 优先级提醒：这一整节都不挡 R0–R3 预训练。** 预训练只吃 `internvid_all`（§4 早就全绿）。
**当前的关键路径仍然是 (19) 的冒烟** —— `git pull` 后先确认 `import scripts` 指到仓库自身、
再跑 2 卡冒烟。评测数据这条线可以等冒烟过了、R0 挂上去之后再慢慢接。

**Then report back** in from-server.md：①②③ 三段输出。我据此判断哪几个基准能就地接线、
哪几个真的要另外下（目前看至少 UCF101 / HMDB51 / SSv2 三个存疑）。
## 2026-07-24 (23) — ⚠️ **我上一条的判据是错的,结论反过来**:那个 0.50 是随机 null,不是"编码了方向"

**先说结论:脚本给你打印的"值得为顺序敏感基准付代价"是错的,请忽略。正确结论是「先别买」。**

**错在哪。** (22) 的判据写的是「反对称占比 ≥0.05 ⇒ 参数层面编码了方向」。
但反对称占比是一个**尺度不变的比值** —— **未经训练的随机 3 抽头核,这个值就 ≈0.500**。
我把 null 当成 0 了,于是把随机噪声读成了信号。蒙特卡洛(20 万次,高斯/均匀/不同方差一致):

| 统计量 | 随机核 null | 你的实测 |
|---|---|---|
| 中位数 | 0.4999 | **0.5011** |
| 均值 | 0.5000 | **0.4994** |
| p05 | 0.0504 | **0.0487** |
| p95 | 0.9496 | **0.9497** |

**逐个分位点都对得上随机核(n=9920)。偏离 = +0.0011。**
另外两个量同样落在各自 null 上:DC/AC 实测 0.52–0.68 vs null 0.589;
`temporal_embedding` 反转反对称 0.601–0.652 vs null 0.706(略**低于** null,偏回文)。

**⇒ 没有证据表明模型学到了时间方向选择性。**
这与 GATE-0 行为侧**完全一致**:打乱 8 帧不掉分,是因为**模型本来就没有方向信息**,
而不是"有信息但基准测不出"。参数侧与行为侧现在互相印证。

**⇒ 决策:先别买顺序敏感基准(RTime 也好 SSv2 也好)。**
R2 的前提在两侧都成立;R2 的成败可以先用「反转敏感度」自测,
RTime 只在需要证明**下游任务收益**时才必需。这省掉了一整轮视频抓取。

**已修的东西(`git pull` 即可):**
- 脚本改为**与随机 null 比**,不再与 0 比;三个 null 常数已用蒙特卡洛**固化成 `--self-test` 断言**,
  防止同类错误再犯。
- 新增第 3 节 **`layer_scale_1d`** —— 这是我首轮漏掉的量,也是时序支路的**总闸门**:
  分支是 `x + layer_scale_1d*(mixer_1d − norm_1d)`,**闸门小则核长什么样都不重要**。
  脚本会同时与同层**空间** `layer_scale` 对比,给出「时序/空间闸门比」。
- `gate0_results.md` §7.3/§7.4 与 `order-sensitive-eval.md` §3 的判读表都已按正确 null 改写。

**Do this next(一条命令,秒级,不用 GPU 不用数据):**

```bash
cd ~/mobileclip && git pull origin main
python experiments/temporal_capacity_audit.py ./checkpoints/mobileviclip_small.pt
```

这次只为拿**第 3 节**的新数:`|layer_scale_1d|` 的分布 + **时序/空间闸门比**。
若闸门比 ≪ 1,"时序支路贡献很小"就有了**第三条独立证据**(行为 / 核形状 / 闸门),三条完全自洽,
这个结论就可以写进论文了 —— 而且它本身是个有意思的发现:
**MobileViCLIP 的时序重参数化分支,在这份官方权重里似乎并没有学到方向性。**

**Then report back**:第 3 节那几行(`|layer_scale_1d|` 统计 + 闸门比 + 那句判读)。
第 1、2 节不用再贴,已经记录在案。

---

## 2026-07-24 (22) — 静态审计判读：`temporal_embedding` **不是零**；时间核那半边不用 reparameterize 了，**同一条命令重跑即可**

**Diagnosis:** 两个发现，一个意外、一个已修好。

**① `temporal_embedding` 不是零 —— 与"它可能从未被训起来"的猜测相反。**
四块 stage-4 全部离开零初始化(‖E‖ = 0.107 / 0.101 / 0.115 / 0.150)，逐位置范数也彼此不同
(相对离散度 0.14–0.22) ⇒ **参数层面确实区分了帧位置**，这条通路没死。

**但它不等于顺序敏感性。** 按 round-2 **F2**，`temporal_embedding` 是**加性**的，
而加性位置编码会被均值池化**精确抹掉** —— 只能经由下游非线性间接起作用，是**弱通路**。
和行为侧(打乱 8 帧不掉分)合起来看：**参数上有弱位置信息，没转化成行为上的顺序敏感性。**
(已加一个新指标：E 沿时间翻转的**反对称占比**；≈0 ⇒ 沿时间回文 ⇒ **原则上编码不了方向**。下次跑直接给。)

**② 时间核没测到 —— 官方权重是训练态。但不用 reparameterize 了。**
这是我这边补上的一个恒等式(已数值验证)：

```
等效核 w = id_tensor + λ·(mixer_1d − norm_1d)
  id_tensor / norm_1d(仅BN) / rbr_scale(k=1)  → 全是中心抽头 → 对称 → 反对称贡献 **恰好 0**
  rbr_conv(k=3)                              → **唯一**的反对称来源
⇒ antisym(等效核) = λ·(逐通道 BN 缩放)·antisym(k=3 分支)
⇒ **k=3 分支对称 ⟺ 等效核恰好对称 ⟺ 恰好时间反转盲**
```

所以脚本已改成**自动回退**到 k=3 分支，并把数标注为**上界**(融合后 id_tensor 往分母加对称质量，
真实值只会更小)。**不需要 build 模型、不需要存推理态权重、不用加任何参数。**

**Do this next（一条命令，秒级，不用 GPU 不用数据）：**

```bash
cd ~/mobileclip && git pull origin main
python experiments/temporal_capacity_audit.py ./checkpoints/mobileviclip_small.pt
```

**这次会直接打印判定结论。** 判读表(它自己也会印)：

| k=3 反对称占比(上界)中位数 | 含义 | 决策 |
|---|---|---|
| **< 0.05** | 等效核也近似对称 ⇒ stage 1-3 基本编码不了方向;与"打乱不掉分"自洽 | **先别买基准** —— R2 可先用反转敏感度自测，RTime 只在要证下游收益时才买 |
| **≥ 0.05** | 参数层面确实有方向信息 ⇒ GATE-0 的零效应是**基准测不出** | **值得买** —— 按 `notes/order-sensitive-eval.md` §4 走 |

**顺带,你上一轮的两条已经收进仓库了:**
- `none` 基线 r_mean 补齐(DSL **62.28** / raw **61.55**),并且 **`none` 臂与 B0 §3.4 锁定值逐项完全一致** ——
  这本身验证了"不设 `MVC_FRAME_PERTURB` 时钩子是零操作"。另外 **`shuffle` 的 raw r_mean 与 `none` 完全相同(都 61.55)**。
- `check_readiness` 的输出确认:msrvtt/k400/ucf101/hmdb51 [OK]，**didemo / anet / ssv2_mc 三个 anno 缺失** ——
  与 GATE-0 里 SSv2-MC 崩掉的原因对上了。

**Then report back**（from-server.md）：这次审计的**末尾判读段**(「全部 k=3 分支 反对称占比」那两行 +
下面的「判读」几行)，外加 `temporal_embedding` 那节新增的**时间反转反对称占比**四个数。
**拿到这两样，我就能给"买不买基准 / 买哪个"的定论** —— 这是当前唯一的关键路径阻塞点。

---

## 2026-07-24 (21) — 基准定 RTime，但**它不发视频**（已核实）；先跑那件不要钱的：静态审计

**Diagnosis / 修正 (20)：** 你选 RTime，我去核了官方 repo —— 有一条 (20) 没说清、而且是**成本的全部**：

> repo 原文：**"To avoid potential copyright implications, we only provide original video urls,
> and you can download the corresponding videos after obtaining necessary permission from the
> source website."**

**公开的是标注和代码，不是视频。** 也没有 HuggingFace 镜像。(20) 把 RTime 列在 SSv2 前面、
并说"有公开 repo"，容易读成"比 SSv2 省事" —— **不对，视频一样要自己抓**，而且是 URL 逐条抓 +
链接腐烂，和我们在 InternVid 上吃过的苦同一类。这一条已写进
[`notes/order-sensitive-eval.md`](../order-sensitive-eval.md)（含核实过的论文/DOI/规模/三任务）。

**所以先做不要钱的那一步 —— 任务 A 静态审计，它至今没跑过。**
GATE-0 结果文件 §3 留了个明确缺口：现在只能说"MSR-VTT 测不出顺序"，**还不能区分**
「模型真没有顺序信息」vs「模型有、但 MSR-VTT 不奖励」。静态审计**正好补这个**，
而且**零数据、零 GPU、秒级** —— 它直接读权重里时间卷积核的**反对称分量占比**
（时间反转把 `[a,b,c]` 变 `[c,b,a]`，**对称核 = 恰好反转盲**）和 `temporal_embedding` 的范数。

**Do this next（哪台机器都行，不用 GPU、不用数据）：**

```bash
cd ~/mobileclip && git pull origin main
python experiments/temporal_capacity_audit.py --self-test
python experiments/temporal_capacity_audit.py ./checkpoints/mobileviclip_small.pt
```

**判读直接决定要不要花钱买基准：**

| 结果 | 含义 | 接下来 |
|---|---|---|
| 反对称占比中位数 **<0.05**、DC/AC **>0.95**、`temporal_embedding`≈0 | 模型在**参数层面**就几乎不编码时间方向 | **先别买基准**。R2 的前提确认成立，且它的成败可以先用"反转敏感度"自测；RTime 只在要证明**下游收益**时才必需 |
| 反对称占比明显 **>0** | 模型有方向信息，是基准测不出 | **值得买**，按 `order-sensitive-eval.md` §4 走 |

⚠️ 该脚本**没在真权重上跑过**。训练态 checkpoint 里时间分支还没融合成 `reparam_conv_1d`，
若键名对不上，它会把**候选键列表打印出来** —— 把那段原样贴回来，我按真实键名调（或给你一条
先 reparam 再存推理态权重的命令）。

**另外两条小事（顺手）：**
```bash
# ① 把 GATE-0 的 none 基线 r_mean 补进结果表（本轮已生成该目录，10 秒）
python -c "import json;d=json.load(open('experiments/evals/gate0/msrvtt_none/eval_res.json'));print(json.dumps(d,ensure_ascii=False)[:400])"
# ② 看清到底缺哪些评测数据（顺带确认 SSv2 的缺口）
unset MVC_DATA_DIR && python experiments/check_readiness.py
```

**一条成本上的实话（`order-sensitive-eval.md` §5）：** 如果最后视频还是得下 SSv2，
那么 **SSv2-MC 的接线成本是零** —— `config_ssv2_mc.py` 和 `retrieval_mc.py` 仓库里都已经有了，
GATE-0 那次崩溃纯粹是缺 anno+视频；而 RTime 还得**额外写标注转换器**（官方没文档化 JSON 结构）
**外加 RTime-Binary 的单独评测入口**（它是"正放 vs 倒放"二选一，形态像多选，不能硬塞进 `retrieval.py`）。
⇒ **选哪个，取决于哪份视频你实际拿得到，不取决于代码。**

**Then report back**（from-server.md）：静态审计的**完整 stdout**（`temporal_embedding` 一节 +
时间核统计一节 + 末尾"全部时间核"两行汇总），键名对不上就贴候选键；外加 ① 的 json 和
② 的 readiness 输出。拿到这三样我就能给出"买不买基准 / 买哪个"的定论。

---

## 2026-07-24 (20) — GATE-0 判读：**顺序值 ≈0、帧多样性值 13 个点**；MSR-VTT 就此失去时序指标资格；SSv2-MC 是缺数据

**Diagnosis:** (16) 的任务 B 跑完了 MSR-VTT 四个扰动，结论**非常干净**，完整表格已落
[`experiments/gate0_results.md`](../../experiments/gate0_results.md)。相对 B0 锁定值
（§3.4：t2v R@1 = 42.50 DSL / 40.80 raw；**`img_r1` 才是 t2v**）：

| 扰动 | t2v ΔDSL | t2v Δraw | 判读 |
|---|---|---|---|
| `shuffle` 打乱 8 帧 | **+0.10** | **+0.20** | **零效应** —— 顺序全打乱居然不掉分 |
| `reverse` 时间反转 | −1.20 | −0.10 | 两口径不同向、在 ±1.6 噪声带内 ⇒ **判噪声** |
| `freeze` 单帧×8 | **−13.00** | **−12.10** | 两口径同向的**巨大**效应 |

**⇒ 帧的"多样性"值 ~13 个 R@1，帧的"顺序"值 ~0。**

**为什么这个"零"可信 —— `freeze` 是阳性对照。** 零效应最常见的死法是开关没生效；
本轮同一个 `MVC_FRAME_PERTURB` 钩子在 `freeze` 上打出 −13，说明钩子确实在改帧、评测确实
对帧内容敏感 ⇒ **`shuffle` 的 0 是真零，不是坏开关**。这一条比那三个数字本身更重要。

**三条不能多说的（已写进结果文件 §3）：**
1. `freeze` **不等于**"单帧模型不行" —— 拿 8 帧训练的模型只喂 1 帧是**分布偏移**，这 13 点是
   "饿掉多帧信息"的**上界**，不是 ATP/单帧偏置那种干净对照。别写成"我们反驳了单帧偏置"。
2. 只测了一半 ⇒ 目前只能说"MSR-VTT 测不出顺序"，**还不能**区分"模型真没顺序信息" vs
   "模型有、但 MSR-VTT 不奖励"。
3. `reverse` 的 −1 别当效应。

**SSv2-MC 四个全崩 —— 是缺数据，不是 bug：**
`FileNotFoundError: anno_downstream/ssv2_mc_val.json`。属 `TROUBLESHOOTING.md`
「Missing `anno_downstream/*.json`」那一类。佐证：B0 当初也只锁了 msrvtt/hmdb51/ucf101/k400
四个，**SSv2-MC 从来没在这台机器上到位过**。
（下次先跑 `python experiments/check_readiness.py`，它认识 `ssv2_mc`，能在开跑前就报缺，
省掉这 4 次崩溃 —— §3.1 的第一行就是它。）

**对计划的影响（GATE-0 的用途就在这里）：**
- **R1 精确文本负例库：不受影响，优先级不变，而且相对更值得做了** —— 它打的是小 batch 对比学习
  这个瓶颈，用普通检索就能测，MSR-VTT 测这个是**有效**的。
- **R2 可恢复顺序的读出头：被卡住。** MSR-VTT 已被实证取消资格 —— 就算 R2 完美恢复顺序信息，
  在 MSR-VTT 上也只能测出 ≈0。**上 R2 之前必须先有顺序敏感基准。**
- **C2 弹性 T：被支持** —— 帧数值 13 个点，帧预算旋钮的动机成立（不成立的只是"顺序"）。

**Do this next（101vector；都不训练，仍低于 recap 主线优先级）：**

**① 把 `none` 的 r_mean 补进表（10 秒，本轮已生成该目录）**
```bash
cd ~/mobileclip && git pull origin main
python -c "import json;d=json.load(open('experiments/evals/gate0/msrvtt_none/eval_res.json'));print(json.dumps(d,ensure_ascii=False)[:400])"
```

**② 任务 A 静态审计还没跑（不用 GPU、不用数据、秒级）**
```bash
python experiments/temporal_capacity_audit.py --self-test
python experiments/temporal_capacity_audit.py ./checkpoints/mobileviclip_small.pt
```
它直接读权重里时间卷积核的**反对称占比**和 `temporal_embedding` 的范数 ——
能从**参数层面**回答"模型到底有没有编码时间方向"，正好补上 §3 那条"还不能区分"的缺口。
⚠️ 没在真权重上跑过；键名对不上时它会把候选键打印出来，**把那段原样贴回来**即可。

**③ 顺序敏感基准（这才是关键路径，决定 R2 能不能上）** —— 三选一，从易到难：
```bash
# 先确认到底缺什么
unset MVC_DATA_DIR && python experiments/check_readiness.py
```
- **SSv2-MC**：缺 `anno_downstream/ssv2_mc_val.json` + `data/ssv2/20bn-something-something-v2`。
  SSv2 视频要在官方站注册后下载（20BN/Qualcomm），anno 走 `TROUBLESHOOTING.md` 那一节的路子。
- **RTime**(`Reversed in Time`, ACM MM 2024)：**最对口** —— 它就是拿"时间反转"做难负例，
  RTime-Binary 的零假设是**恰好 50%**（时间盲的模型必然掷硬币）。有公开 repo。
- **SSv2-template / SSv2-label 检索**(Lei et al. ACL 2023)：仍依赖 SSv2 视频。

**Then report back**（from-server.md）：① 的 json 原文；② 的**完整 stdout**（两节 + 末尾汇总两行），
键名对不上就贴候选键；③ `check_readiness.py` 的输出 + 你倾向走哪条基准路线（我按你选的那条写落地步骤）。

---

## 2026-07-24 (19) — `No module named 'scripts.pretraining'`：仓库的 `scripts` 被 site-packages 里的同名包顶掉了（已修，pull 即可）

**Diagnosis:** 依赖这一关**过了**——`userbase` 已生效、import 链一路走到配置加载才崩。
新崩点的关键在于报的是 **`'scripts.pretraining'` 而不是 `'scripts'`**：`scripts` 本身
import 成功了，只是**导进来的不是本仓库那个**。

原因：仓库的 `scripts/` 原先**一个 `__init__.py` 都没有**，是 PEP 420 命名空间包；而命名
空间包的优先级**低于普通包**——Python 扫 `sys.path` 时，遇到无 `__init__.py` 的同名目录
只记一个 "portion" 就**继续往后扫**，后面任何一个带 `__init__.py` 的 `scripts` 都会**反超**
它，哪怕仓库根在 `PYTHONPATH` 里排得更前。b100 共用 site-packages 里恰好有这么一个包；
V100（101vector）上没有，所以同一份代码只在 b100 发作。

已给 `scripts/` 及其子目录补上 `__init__.py`（本地验证：`import scripts` 现在稳定指到仓库
自己的 `scripts/__init__.py`）。**别删这些文件**，每个里面都写了原因。

**⚠️ 顺带一件要先处理的事：那个 4 卡 R0 可能还活着。** 你整段粘贴时，冒烟崩了以后
`nohup ... &` 的 R0 照样被拉起来了（`[1] 195353`），而 **`&` 后台任务不吃 Ctrl-C**——你按的
`^C` 只停掉了前台的 `tail -f`。它大概率也已经因同样的 `scripts.pretraining` 崩掉了，但要确认。

**Do this next（b100，仓库根）：**

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip

# ① 先确认那个 R0 是死是活（按 PID，禁止 pkill -f —— 本机 root 共用，别人的中文线
#    跑的是同一个 tasks_clip/pretrain.py）
ps -p 195353 -o pid,etime,cmd
tail -5 experiments/runs/r0_baseline.log
#   —— 没有输出 = 已经死了，什么都不用做；
#   —— 还活着就先 ls -l /proc/195353/cwd 确认是 rby/ 下的作业，再 kill 195353

# ② 拉修复
git pull

# ③ 自检：必须打印仓库自己的路径，不是 site-packages 里的
source experiments/env_b100.sh
python -c "import scripts; print(scripts.__file__)"
# 期望 /workspace/mount/b100_zaip_data/rby/mobileclip/scripts/__init__.py

# ④ 只跑冒烟这一步（2 卡）。跑之前 nvidia-smi 看 0、1 号卡空不空。
CUDA_VISIBLE_DEVICES=0,1 bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 2 \
    use_bf16 True bf16 True deepspeed.enable False debug True
```

**⛔ 闸门：冒烟没跑到打印 loss，就不要执行 §6 步骤 2 的 4 卡 R0。** SETUP_B100 §6 已经把
三步拆成三个独立代码块并加了这条闸门 —— 之前写成一整块，整段粘贴会把没验证过的 4 卡作业
直接拉起来，在共用机器上这是要避免的。

**Then report back** in from-server.md：① `ps -p 195353` 有没有输出；② `import scripts`
打印的路径；③ 冒烟日志——跑通就贴到出现 loss 的前 30 行，崩了就贴完整 traceback。
## 2026-07-24 (18) — (13) 任务 C 判读:全过,1.6/U2 关闭,Fig-1 v1 数据到手;⚠️ recap 抽样启动是空跑

**判定:任务 C ✅,(13) 三个任务全部收官。**

四组 `ret_test_emb` 读数(官方权重,MSR-VTT 1K,fp32,middle 采样):

| T | t2v R@1 | v2t R@1 | r_mean |
|---|--------|--------|--------|
| 2 | 35.7 | 35.8 | 55.42 |
| 4 | 39.6 | 39.2 | 60.03 |
| 8 | 40.8 | 42.5 | 61.55 |
| 16 | 42.2 | 43.1 | 62.25 |

三个关键读数:

1. **锚点核验通过(最重要)**:T=8 的 DSL 口径 t2v R@1 = **42.5**,与 B0 锁定值
   42.50 完全一致。T=8 走的是 2.1 插值 identity 路径 ⇒ **2.1 的 11 处动态化改动
   对官方权重零回归**,这是继本地 CPU 逐位单测之后的服务器 GPU 级证据。
2. **T=16 免训练反超 T=8**(emb r_mean +0.7,DSL t2v +0.8):`temporal_embedding`
   从 8 插值升采样到 16(2.1 的 resize 路径首次在官方权重上实跑)不仅没坏,还白拿
   增益——直接支持 T 无关设计与弹性推理叙事。
3. **T=2 gap = C2 的靶子**:相对 T=8 掉 6.1 r_mean(t2v R@1 −5.1);退化平滑无
   断崖,8× 帧数差距只有 ~6.9 r_mean。这条曲线就是论文 **Fig-1 v1**(该失配曲线
   在视频-文本模型上未见发表),绘图在 Claude 侧做,服务器无需再动。

结果已落 `experiments/ROADMAP.md` 1.6(含表)+ 状态速览;原始 JSON 在服务器
`experiments/evals/b0_multiT/msrvtt_T{2,4,8,16}/eval_res.json`,**别删**。

**⚠️ 顺带发现:recap 抽样那次启动是空跑。** 粘贴时说明文字连命令一起进了 shell,
真正的 `head -n 50 <队列> > sample.jsonl` 没有执行(就是报 `head: command not found`
那行,但重定向仍创建了**空的** sample.jsonl),于是 `[input] sample.jsonl （0 片）`,
8 个 shard 秒退,什么也没算。重跑三步:① 单独执行 head 那行,`<队列>` 换成 (14)
里写的实际队列文件;② `wc -l sample.jsonl` 确认 =50;③ 再
`RECAP_INPUT=sample.jsonl bash tools/data_audit/run_v100.sh recap`。续跑按 video
去重,不会重复计费已算过的片。

**V100 队列剩余**:(16) 的 GATE-0 任务 A/B(优先,阻塞时序类改动)→ recap 抽样
修正重跑 → per-T BN 漂移诊断(用 r3e ckpt,`experiments/per_t_bn.py`,不急)。

---

## 2026-07-24 (17) — (13) 任务 A/B 判读:全过,C2 转 ✅;一个观察项;只差 (13) 任务 C 回报

> **编号提醒**:本块判读的是 **(13) 的任务 A/B/C**(pytest / r3e 冒烟 / 多 T 曲线)。
> 下面 (16) 的 GATE-0「任务 A/B」是**另一组新任务**,不受本判读影响,照 (16) 跑。

**判定:**

- **任务 A ✅** —— `pytest tests/test_t_agnostic.py` 18 passed,与本地一致。
- **任务 B ✅(三判据全过)→ C2 状态 🟡→✅(V100 链路层面)**
  1. **自停干净**:debug 每 epoch 5 步 ×10 epoch,`Training time 0:19:28`,
     best epoch 9,ckpt 落盘,退出只有 NCCL destroy 的良性 warning。
  2. **elastic_t 正常跳变**:{2,4,8,16} 全部出现;日志 10 个 epoch 的 T 序列
     已与本地 `choose_step_t(choices, seed=42, step)` 复算逐位比对,
     **50 步全一致**(含用 epoch 均值反推的每轮第 5 步)——跨 rank 决定性
     采样器端到端验证通过;均值 7.8 ≈ 理论 E[T]=7.5。
  3. **无 OOM、训练 loss 全程有限**:1.09→0.37 平滑下降;首个 T=16 步显存
     12.2GB→23.3GB 后稳定(V100 32GB 内,余量约 8GB)。
  - 附带验证:T=16 步在 gradient checkpointing 重算下正常(帧上下文按步单值
    ⇒ 重算同 T,设计如预期);评测恒用 `num_frames_test=8`,不随训练 T 漂移。

**观察项(不阻塞,已记 TROUBLESHOOTING 末节):** epoch3 评测窗口打印过一次
`sim_v2t has nan / sim_t2v has nan`。出处 `models/criterions.py:55`,调用方
`tasks_clip/retrieval_utils.py:139`(evaluation_dot)——**评测侧** fp16 特征提取
偶发上溢(每个 rank 独立提全量特征,先算完的 rank 先打分,所以裸 print 穿插在
rank0 进度条中间;不是训练路径)。训练 loss 从未 NaN,epoch4–9 评测再未出现,
当轮指标仍正常(r_mean 87.4)。b100 走 bf16 天然免疫;**仅当 V100 后续评测持续
出现**才处理(该次评测置 `use_half_precision=False` 即可),单次出现不用管。

**接下来:** 只差**任务 C**回报——命令仍按 (13)(记得 `unset MVC_DATA_DIR` 的
干净 shell),要的是 T=2/4/8/16 四组 `ret_test_emb` 的 t2v/v2t R@1,那就是
Fig-1 v1 的数据。r3e 的 ckpt 已在 `experiments/runs/pilot_10k_r3e/`,之后可
顺手跑 per-T BN 漂移诊断(`experiments/per_t_bn.py` docstring),不急。

---

## 2026-07-24 (16) — GATE-0 时间维容量审计：两个任务（秒级静态 + 8 次评测），做任何时序改动之前先跑（#246）

**Diagnosis:** 脑启发第二轮已合入（#246，`notes/neural-inspired-round2.md`）。它证了四条
**已数值验证**的结构事实，其中三条直接改写待办优先级：

- **F1** 均值池化头**无条件**时间盲——集合的均值对任意重排都不变（对任何轨迹都成立，不只
  直线）。所以模型全部的时序敏感性只能来自 backbone（stage 1–3 的时间卷积 + stage 4 的
  `temporal_embedding`），而且必须以「所有帧共享的 clip 级信号」形式写进**每一帧**，才能在
  平均之后幸存。
- **F2** **加性**位置编码被均值**精确抹掉**，只有**乘性**绑定能幸存。而 stage 4 恰恰是加性的
  （`x = x + resize_temporal_embedding(...)`，`mobileviclip_small.py:1105`）⇒ 它对池化后嵌入的
  顺序敏感性贡献**恰好为零**（只能经由下游非线性间接起作用）。
- **F4** 检索损失在帧残差子空间上**恰好平坦** ⇒ 帧塌缩是**免费**方向，「对比损失能防塌缩」在
  这里是**假的**（它只钉住均值）；straightening / 潜在预测 / 时间去相关这类轨迹辅助损失
  一阶无效。

外加一条：现有评测套件很可能**测不出**时序（单帧偏置；纯图像模型 BLIP 的 MSR-VTT 零样本
R@1 已高于我们锁定的 42.50）。所以在动手改任何时序结构之前，先把「**模型到底有没有编码
时间方向 + 我们的基准能不能测出来**」这两件事测掉。两个任务**都不训练**。

**Do this next（101vector；优先级仍低于 recap 主线，插空档跑）：**

**任务 A · 静态审计（不用 GPU、不用数据、秒级；b100 上也能跑）**

```bash
cd ~/mobileclip && git pull origin main
python experiments/temporal_capacity_audit.py --self-test          # 先自检，应打印 self-test 通过
python experiments/temporal_capacity_audit.py ./checkpoints/mobileviclip_small.pt
```

判读：时间卷积核**反对称分量占比**中位数 < 0.05 ⇒ 核近似对称 ⇒ 对时间反转基本不敏感；
**DC/AC** 中位数 > 0.95 ⇒ 近似纯低通 ⇒ 帧间残差被抹平。两者同时成立 ⇒ 模型几乎不编码时间
方向。另外看 `temporal_embedding` 的范数——它是**零初始化**的，若训练后仍≈0，stage 4 那条
通路对帧顺序就是等变的。

> ⚠️ 这个脚本在**合成** checkpoint 上端到端验过，但**没在真权重上跑过**（我这边没有权重）。
> 训练态 checkpoint 里时间分支还没融合成 `reparam_conv_1d`，若默认键名对不上，脚本会把找到的
> 候选键**打印出来**——把那段输出原样贴回来即可，我按真实键名调（或给你一条先 reparam 再存
> 一份推理态权重的命令）。

**任务 B · 动态审计（评测 ⇒ 要 `unset`，与 (13) 任务 C 同一套 `run_eval.sh`）**

```bash
unset MVC_DATA_DIR
python experiments/check_readiness.py            # msrvtt / ssv2_mc 应为 [OK ]

for DS in msrvtt ssv2_mc; do
  for P in none shuffle reverse freeze; do
    echo "===== $DS / perturb=$P ====="
    MVC_FRAME_PERTURB=$P CUDA_VISIBLE_DEVICES=0 \
      bash experiments/run_eval.sh small $DS \
        ./checkpoints/mobileviclip_small.pt \
        experiments/evals/gate0/${DS}_${P}
  done
done
```

汇总成表（主口径 = `ret_test_emb/` 原始余弦）：

```bash
python - <<'PY'
import json, glob, os
rows = {}
for p in sorted(glob.glob("experiments/evals/gate0/*/eval_res.json")):
    ds, perturb = os.path.basename(os.path.dirname(p)).rsplit("_", 1)
    rows.setdefault(ds, {})[perturb] = json.load(open(p))
for ds, d in rows.items():
    print(f"\n=== {ds} ===")
    for perturb in ("none", "shuffle", "reverse", "freeze"):
        if perturb in d:
            print(f"  {perturb:8s} {json.dumps(d[perturb], ensure_ascii=False)[:300]}")
PY
```

三个扰动的含义：`shuffle` = 随机置换 8 帧（按样本 index 确定性设种，可复现）测总的时序敏感度；
`reverse` = 时间反转，专测「时间之箭」；`freeze` = 全换成中间帧，测多帧信息一共值多少。
**`none` 是干净基线，必须跑**——它同时验证「不设环境变量 = 与原路径逐位一致」。

> ⏱️ **时间紧就先跑 4 个**：`msrvtt/none`、`msrvtt/reverse`、`ssv2_mc/none`、`ssv2_mc/reverse`。
> `reverse` 是最锐利的探针，这 4 个数就足以做判读。
>
> ⚠️ `MVC_FRAME_PERTURB` **只在审计评测时设，绝不要在训练时打开**。打开时日志会出现一行
> `帧顺序正在被人为扰动（GATE-0 审计模式）` 的 warning —— 训练日志里出现这行就是配置错了。

**判读（决定后面投哪条线，`notes/neural-inspired-round2.md` §2c）：**

| 结果 | 含义 | 接下来 |
|---|---|---|
| MSR-VTT 与 SSv2-MC **都**几乎不变 | 模型时间盲**且**基准测不出来 | 时序线只留 R2（可恢复顺序的读出头），且必须补 RTime 才测得到；其余时序想法全砍 |
| MSR-VTT 不变、**SSv2-MC 掉** | 模型有顺序信息，是**基准**的问题 | 推进 R2 + 补 RTime/Vinoground，按 Pareto 汇报 |
| MSR-VTT 也明显掉 | 与 F1 预期不符 | 先别改，把数贴回来重新推导 |

无论哪种结果都是一张论文图（在**我们自己的模型**上测出的「可测性上界」，与 1.6 多 T 曲线同一格）。

**顺带两条已落库的修正（不用你操作，知道即可）：**
- `criterion.loss_weight` 只是**日志开关**，不进损失——`tasks_clip/pretrain.py` 对 forward
  返回的 loss dict 做**无权求和**，权重从未参与。以后加辅助损失必须在 `forward` 内部自己乘。
  已写进 `TROUBLESHOOTING.md`。
- A1 惊奇池化比第一轮所述更弱：重加权一个**集合**仍然是时间盲的，它只能利用**外观**显著性
  （压低糊帧/无信息帧），永远拿不到时序顺序。与 main 上已钉的「A1 不是贡献」一致。

**Then report back**（from-server.md）：
① 任务 A 的**完整 stdout**（两段：`temporal_embedding` 那节 + 时间核统计那节，包括最后的
「全部时间核」两行汇总）；若报「未找到」就把它打印的候选键列表贴回来。
② 任务 B 的汇总表输出（或至少 4 个 `reverse`/`none` 组合的 `ret_test_emb` 原文）。
③ 跑之前确认一句：`git pull origin main` 后 `experiments/temporal_capacity_audit.py` 在不在。

---

## 2026-07-24 (15) — 冒烟只差几个小包；但**别再往共用系统 site-packages 里装**

**Diagnosis:** 2 卡冒烟的这次崩溃其实是**进展**——deepspeed 已经起来、`models` 也
import 成功了，说明 (8) 之前的两关（peft 的 `cannot import name 'Cache'`、
`No module named 'open_clip'`）都过了。现在只剩 `utils/logger.py:14` 缺 `termcolor`
这种纯 python 小包。

两个必须先说清楚的判断：

1. **不再修 `mvc_env310`，改用系统 torch。** 日志里 `python : /usr/bin/python`——
   提示符上的环境名只是 PS1 文本，实际解释器一直是系统的。而系统 torch 是
   **2.5.1+cu124、`get_arch_list()` 含 `sm_90`**，Hopper 支持本来就是好的。为本项目再
   下一套几个 G 的 torch 没有收益，**沿用系统 torch 是更省事也更稳的选择**。

2. **⚠️ 但装包的落点必须改。** 这一路 `pip install` 全落进了
   `/usr/local/lib/python3.10/dist-packages`——那是 wky / zyc 共用的目录。**尤其别照抄
   `scripts/setup_env.sh` Stage 4 那串带版本号的命令**（`transformers==4.28.1`
   `numpy==1.26.4` `timm==0.5.4` …）：那是给 V100 的 torch 1.13.1+cu117 写的，在 b100 上
   会把别人在用的公共依赖**降级**，属于事故级操作。

**方案：私有 user-site。** 缺的包装进 `rby/pyuser/`，系统目录一个字节不动；user-site
在 `sys.path` 里优先，所以只对 source 过环境文件的 shell 生效。仓库里新加了两个文件
（`experiments/env_b100.sh`、`experiments/check_imports.py`），先 `git pull`。

**Do this next（b100，仓库根，全程零 GPU）：**

```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip
git pull

# ① 每个新 shell 都先 source 一次（只设环境变量，不装任何东西，重复 source 无副作用）
source experiments/env_b100.sh

# ② 一次列全缺哪些包（不用再一轮崩一个）——末尾会打印可直接复制的 pip 命令
python experiments/check_imports.py
```

`check_imports.py` 把整条 pretrain import 链上的第三方模块逐个 try 一遍，并顺带打印
`sys.executable` / `torch.__version__` / `get_arch_list()`。按它末尾给的命令补装，
**只用这一种形式**：

```bash
python -m pip install --user --no-deps <它列出的包名>
python experiments/check_imports.py      # 重跑，直到打印“全部 [OK]”
```

- `--user` + `PYTHONUSERBASE`（`env_b100.sh` 里设好了）→ 落到 `rby/pyuser/`，不碰共用目录。
- `--no-deps` → 不让 pip 顺手升/降级 torch / transformers / numpy。
- 用 `python -m pip` 而不是裸 `pip`，保证装到当前解释器。

**③ 依赖齐了再冒烟**（等有空闲卡；`nvidia-smi` 先看清前 4 张的占用）：

```bash
source experiments/env_b100.sh           # 新 shell 必须重新 source
CUDA_VISIBLE_DEVICES=0,1 bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 2 \
    use_bf16 True bf16 True deepspeed.enable False debug True
```

脚本开头现在会打印 `python` 和 `userbase` 两行——**看到 `userbase : <未设置>` 就是忘了
source**，停下来重来（子进程靠继承环境变量，漏了就等于没装）。

**Then report back** in from-server.md：`check_imports.py` 的完整输出（尤其是
`[缺失]`/`[异常]` 行和最后那条 pip 命令），以及补装后重跑是否打印“全部 [OK]”。
冒烟真跑到 loss 打印就贴前 30 行日志。

对应文档已更新：`experiments/SETUP_B100.md` §2「现行方案」+ §6 步骤 0、
`TROUBLESHOOTING.md`「b100 共用系统 site-packages」。
## 2026-07-24 (14) — pipeline-sample 缺队列已修：先跑 s3(0.3)+新 s4 两条秒级命令（#243）

**Diagnosis:** 你贴的报错是对的——`v0_2/recaption_tasks.jsonl` 从没在本机落地过，而旧的落地
命令 `s4-merged` 硬性依赖 better_zh 融合稿（`s2_merged.*`，主计划 v1.1 已存档弃用）。
#243 已加不依赖融合稿的 **`s4`** 落地命令，并把报错文案换成自救指令。另外盘上现存的
`buckets.jsonl` 可能是审计结论 15 里**作废的 0.8 版**（RECAPTION 空桶），所以 s3 要带
`TARGET_PRECISION=0.3` 重跑一次保险（秒级、幂等覆盖）。

**Do this next（101vector，接着你刚才的终端；①② 均为 CPU 秒级）：**
```bash
git pull origin main    # 拿到新的 s4 子命令（#243）
TARGET_PRECISION=0.3 bash tools/data_audit/run_v100.sh s3   # ① 重算分桶（0.3 版才有效）
bash tools/data_audit/run_v100.sh s4                        # ② 落地队列, 预期打印 ~2,5xx 片
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample   # ③ 原命令重跑
```
③ 的回报格式照 (10)：`pipe_diag.txt` 全文 + TSV 任挑 3 行 + 起止时间。
（此条也补进 (13) 的待办总汇语境：pipeline-sample 任务前多了 ①② 两个前置。）

---

## 2026-07-24 (13) — V100 待办命令总汇（三个任务按环境变量分组；更正 (12) 一处错误）

**Diagnosis / 更正：** (12) 说「② 评测集不吃 `MVC_DATA_DIR`」**是错的**——
`configs/data.py` 的 `__data()` 对**所有**语料生效，评测数据在仓库 `data/`
（下载脚本写死的相对路径 + 软链），export 后会被重定向到
`/home/vca1/datasets/…` 下不存在的路径。正确纪律 = README §3.0 的**双根规则**
（2026-07-22 曾因混用把已下好的 HMDB51 误报为缺失）：
**训练/试点 shell 要 `export`，评测 shell 要 `unset`，别在同一个 shell 里混。**
下面把当前 V100 全部待办按此分组，直接照抄即可（优先级都低于 recap 主线，插空档）。

**任务 A · C2 单测（与环境变量无关；纯 CPU ~4 分钟）：**

```bash
cd ~/mobileclip && git pull origin main
python -m pytest tests/test_t_agnostic.py -v      # 预期 18 passed
```

**任务 B · r3e 弹性冒烟（试点语料 ⇒ 要 export；同一 shell 连着跑）：**

```bash
export MVC_DATA_DIR=/home/vca1/datasets
python experiments/check_readiness.py             # internvid_zh_10k_pilot 应 [OK ]
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r3e.py 8 debug True
# 判读三条：debug 自停干净；日志 elastic_t 在 2/4/8/16 间跳；无 OOM/NaN。
# 全过 ⇒ C2 状态 🟡→✅（ROADMAP 2.1 关闭）。
```

**任务 C · B0 官方权重多 T 退化曲线 = 论文 Fig-1（评测数据 ⇒ 要 unset；换个 shell 或先 unset）：**

```bash
unset MVC_DATA_DIR
python experiments/check_readiness.py             # msrvtt 应 [OK ]（§3.0 ① 口径）
for T in 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 bash experiments/run_eval.sh small msrvtt \
      ./checkpoints/mobileviclip_small.pt \
      experiments/evals/b0_multiT/msrvtt_T${T} \
      num_frames ${T} num_frames_test ${T}
done
# 看数（主口径 = ret_test_emb/ 原始余弦；T=8 的 DSL 口径 t2v R@1 应≈42.5 对上 §3.4 锚点）：
python - <<'PY'
import json, glob
for p in sorted(glob.glob("experiments/evals/b0_multiT/msrvtt_T*/eval_res.json")):
    print(p.split("/")[-2], json.dumps(json.load(open(p)), ensure_ascii=False)[:400])
PY
```

**b100：** 无变化，照既有序列跑（#222 改动默认关闭，R0 不受影响）。

**Then report back**（from-server.md）：A 的 pass 数；B 的判读三条 + 日志尾部几行；
C 的四组 t2v/v2t R@1（`ret_test_emb` 贴原文）——C 的四个数就是 Fig-1 第一版 +
C2 难度标定。

---

## 2026-07-24 (12) — r3e 冒烟崩溃已定位：新 shell 忘了 `export MVC_DATA_DIR`，一行修复重跑

**Diagnosis:** 你贴的 r3e 全量崩溃**不是弹性 T 代码的问题**——崩在 dataloader 取
第一个 batch 之前，模型一行没跑到。证据链：(a) **每一条**视频都 `No such file or
directory`（不是零星坏文件）；(b) 尝试的是**相对路径** `data/internvid_zh_10k/videos/…`
= `MVC_DATA_DIR` 未设时的默认值（`configs/data.py` 的 `internvid_zh_10k_pilot` 注释
原文就写着 101vector 要 export）；(c) 报错末尾的提示语本身就指向了这个原因。
r0/r1 当时能跑是因为那个 shell 照 §2b 第 2 步 export 过；这次是全新会话，环境变量
不继承。TROUBLESHOOTING 已有该机制的完整条目（`data_root` 解析一节），不再重复。

**V100 · Do this next（repo 根，同一个 shell 里连着跑）：**

```bash
# 1) 一行修复（README §2b 第 2 步原文；每个新 shell 都要）：
export MVC_DATA_DIR=/home/vca1/datasets

# 2) 10 秒自检（第 0 步；任何新 shell 开跑前都该先跑它，这次就能拦住）：
python experiments/check_readiness.py     # internvid_zh_10k_pilot 应为 [OK ]

# 3) 重跑 r3e 冒烟（debug 几十步自停）：
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r3e.py 8 debug True
```

**判读（(6)③ 的三条不变）：** 自停干净、日志里 `elastic_t` 在 2/4/8/16 间跳、
无 OOM/NaN。三条都过 ⇒ C2 状态 🟡→✅（ROADMAP 2.1 关闭）。
顺带：(6) 的 ①（`pytest tests/test_t_agnostic.py`，纯 CPU）与 ②（B0 多 T 曲线）
仍待跑；~~② 用的评测集不吃这个环境变量~~ **此句有误，见 (13) 更正**：评测数据在
仓库 `data/` 下，② 反而要 `unset MVC_DATA_DIR`（README §3.0 的双根纪律）。

**Then report back**（from-server.md）：check_readiness 那行输出；r3e 尾部几行
（有没有 `elastic_t` 数值在跳）；①② 的输出。

---

## 2026-07-24 (11) — 确认：pipeline-sample 全链唯一 LLM = 本机 Qwen3-VL-8B（#235 加固）

**Diagnosis:** 维护者确认 V100 上唯一 LLM 是 Qwen3-VL-8B。已逐环节核对并加固（#235）：
① 重述与派生都走 `$S2_MODEL`（本地 `ckpts/Qwen3-VL-8B-Instruct` 优先）——重述 temp0.9 采样
（制造投票方差）、派生 temp0 贪心（闭集确定性），启动横幅会打印这行确认；② SigLIP2 打分是
非 LLM 嵌入模型且 s1 轮已缓存；③ GDINO/RapidOCR 是非 LLM 可选件，装不上自动降级为
unverified 且不再白解码视频、也不会阻塞 TSV；④ **红线**：验真器缺了就让视图保持
unverified，**绝不用 Qwen 自验**（§3.5 纪律）。(10) 的命令不变，直接跑即可。

---

## 2026-07-24 (10) — 完整优化流水线可小样本端到端了：pipeline-sample 一条命令 → 全景 TSV（#233）

**Diagnosis:** 维护者要的是**整条 caption 优化方案**的小样本测评，不只是重述段。缺口已补齐
（#233 已并入 main）：`--mode derive`（每条重述派生 2–3 条短视图 + en_gloss，文本-only 闭集，
复用已缓存的 Qwen3-VL 当文本 LLM，零新权重）、`score_alignment --tag` 逐重述 SigLIP 打分
（A− top-3 选择）、`tools/fuse_verify.py` v0（Panel-V 检测器档：GroundingDINO fp32 按
en_gloss 查 + RapidOCR 引号文字模糊匹配；装不上自动降级为 unverified 保守留、不阻塞）、
`make_pipeline_tsv.py` 全景表。这等于把主计划 G2 三臂（A_llm / A− / C_v）在样本级预览了一遍。
**取代 (8)「重述抽样检验」的命令**——pipeline-sample 是它的超集（recap-tsv 仍可作轻量档）。

**Do this next（101vector；单卡前台全程 ~1h，含等 8 卡重述；可随时中断续跑）：**
```bash
source ~/audit_env/bin/activate
pip install rapidocr-onnxruntime onnxruntime   # 可选 OCR 档；失败不阻塞
N_SAMPLE=50 bash tools/data_audit/run_v100.sh pipeline-sample
# 产物: outputs/audit_10k/recap_pipeline.tsv （utf-8-sig, Excel/WPS 直开）
# 列: en源caption | zh翻译 | 5重述(⟨SigLIP分⟩) | A−保留 | A′句切 | ✓过验真 | ·未验 | ✗被滤[gdino/ocr] | 旗标
```
首次会经 hf-mirror 下载 GroundingDINO-base（~700MB）。

**Then report back** in from-server.md：① `pipe_diag.txt` 全文；② TSV 任挑 3 行贴
en_source / zh_translated / recap_1 / views_pass / views_fail 五列；③ 全程起止时间。
我据此核：五条是否有差异、验真过滤率是否合理（fail 占比预期 5–20%，>40% 说明 gdino 阈值或
gloss 质量有问题，我来调）。

---

## 2026-07-24 (9) — b100 每次 `git pull` 都要密码：凭据只在 `rby/` 下生效，不落共用 `$HOME`

**Diagnosis:** b100 的 root 账号多人共用。`TROUBLESHOOTING.md`「b100 上一切 git 命令都报
`not a git repository`」一节已经把"全局"配置重定向进了 `rby/.gitconfig`（`.gitenv` 里的
`GIT_CONFIG_GLOBAL`），但**凭据这一块一直没配**，所以每次 `git pull` 都问用户名/密码。
顺带：GitHub 早就取消了密码认证，这里要填的"密码"是 **PAT**，不是账号密码。

下面这套把凭据 helper 关进 `rby/` 自己的配置，并用 `includeIf gitdir:` **双重门控**：
无论有没有 `source .gitenv`，都**只对 `rby/` 下的仓库生效**，`rby/` 外的仓库一律不受影响；
token 文件也只落在 `rby/` 内，不进共用 `$HOME`。四种组合已在 Curta 上逐一验过（见文末表）。

**Do this next（b100，任意目录）：**

```bash
RBY=$(cd /workspace/mount/b100_zaip_data/rby && pwd -P)   # 取真实路径，软链已解析

# ① 身份 + 凭据 helper：单独一个文件，只放 rby/ 内
cat > "$RBY/.gitconfig-cred" <<EOF
[user]
    name = rbybryan
    email = br2566@columbia.edu
[credential]
    helper = store --file $RBY/.git-credentials
EOF

# ② rby/.gitconfig 里按 gitdir 门控（source 过 .gitenv 的 shell 走这条）
git config --file "$RBY/.gitconfig" "includeIf.gitdir:$RBY/.path" "$RBY/.gitconfig-cred"

# ③ 兜底：忘了 source .gitenv 时走共用 $HOME —— 只写一行路径，不含 token（可选，见下）
git config --file "$HOME/.gitconfig" "includeIf.gitdir:$RBY/.path" "$RBY/.gitconfig"

# ④ 写 token（read -s，不进 shell history）
read -rsp "GitHub PAT: " TOK; echo
umask 077
printf 'https://rbybryan:%s@github.com\n' "$TOK" > "$RBY/.git-credentials"
unset TOK
chmod 600 "$RBY/.git-credentials"
```

②③ 用 `git config --file` 而不是 `>>` 追加，所以**重复跑不会写重**（同键覆盖）。

**自检**（**新开一个 shell、不要 source 任何东西**；第一条应打印 helper 指向
`rby/.gitconfig-cred`，第二条应全程不问密码）：

```bash
cd /workspace/mount/b100_zaip_data/rby/MobileViCLIP
git config --show-origin --get credential.helper
git pull origin main
```

**要不要每次登录都 `source .gitenv`？——做了 ③ 就不用。** 身份和凭据靠 `$HOME/.gitconfig`
里那行 `includeIf` 自动加载，登录后直接用。**唯一的例外是 `safe.directory`**：git 判断
「可疑的仓库所有权」时读的是 *early / protected config*，那个时刻仓库还没被认出来，
`gitdir:` 条件无从匹配，所以**放在 conditional include 里的 `safe.directory` 不生效**
（Curta 实测：`git config --get-all safe.directory` **看得见**这个值，但同一个仓库照样报
`detected dubious ownership`；`source .gitenv` 后就正常）。所以：

- 从没见过 `dubious ownership` 报错（b100 上大家都是 root、文件也归 root）→ **完全不用 source**。
- 见过 → 二选一：每个 shell 先 `source "$RBY/.gitenv"`，**或**把这一行直接写进共用
  `$HOME/.gitconfig`（只是路径，不含 token，一次即可、之后永久免 source）：

  ```bash
  git config --global --add safe.directory "$RBY/MobileViCLIP"
  ```

`.gitenv` 里另一半 `GIT_CONFIG_SYSTEM=/dev/null`（屏蔽 `/etc/gitconfig` 里别人塞的东西）
同理只在 source 后生效；除非系统配置里有 helper/代理来捣乱，一般不用管。

**几个坑：**

- **共用 root 下 `chmod 600` 不提供任何隔离** —— 同账号的人都是 root，照样读得到
  `.git-credentials`。所以请用 **fine-grained PAT、只授权 `rbybryan/mobileclip` 这一个仓库**，
  别用带完整 `repo` scope 的经典 token。`upstream`（MCG-NJU）是公开仓库，本来就不需要凭据。
- **`gitdir:` 末尾的 `/` 不能省** —— `gitdir:…/rby/` 等价于 `…/rby/**`（整棵子树）；
  少了斜杠就只匹配 gitdir 恰好等于该路径的那一个仓库。
- **`credential.helper` 是叠加的，不是覆盖的** —— 若共用 `$HOME/.gitconfig` 里已经有别人配的
  全局 helper，它会**先**被问到，这里配的就轮不上。先查一下
  `git config --get-all --show-origin credential.helper`，有多余的再处理。
- **匹配的是 `.git` 目录、不是当前工作目录** —— 建在 `rby/` 外的 linked worktree 不会命中。
- ③ 会往共用 `$HOME/.gitconfig` 写一行（**只有路径，没有 token**）。若坚持"什么都不落共用
  `$HOME`"，**跳过 ③ 即可**，代价是每个 shell 都必须先 `source .gitenv`，否则又开始问密码。

**Curta 上验过的四种组合**（`git config --get credential.helper`）：

| 仓库位置 | 已 `source .gitenv` | 未 source |
|---|---|---|
| `rby/` 内 | ✅ 命中 | ✅ 命中（走 ③ 兜底） |
| `rby/` 外 | ⬜ 无（正确） | ⬜ 无（正确） |

固化版见 `TROUBLESHOOTING.md`「b100 上一切 git 命令都报 …」一节的**共用账号的隔离**小节。

**Then report back** in from-server.md：自检那两条的输出 —— `--show-origin` 那行指向哪个文件，
以及 `git pull origin main` 是不是全程没再问密码。

---

## 2026-07-24 (8) — surprise 判据已读：✅ 通过，A1 试点结案；V100 只剩 (6) 三件事

**Diagnosis:** 贴回的三个数命中 (7) 判据表**第一行**：`gamma=1.0`（#220 修复在位，
非旧代码的 0）、`alpha=-4.74e-5 ≠ 0`（模块活着，量级与 Adam×170 步预估自洽）、
`key.weight L2=13.06`（≈ Linear(512,512) kaiming 初始化期望范数 13.07）。
**A1 链路判据真正通过，试点关闭**；`nA1_surprisepool` 排进 H100 全量 A/B
（U6 阶梯下 1-epoch 排序臂，淘汰线 = 等步数 ≥ attn 池化）。
结案记录 + 一个易错点已写进 `experiments/pilot_10k_results.md` 结案节：
**gamma/key 在试点里“纹丝不动”是数学预期**（二者梯度都被 `alpha·(…)` 缩放，
alpha~1e-5 时步长被压到 ~1e-10），别当失败信号——试点判据只看 alpha；
alpha 的负号在该量级是噪声，不构成方向性结论（10k 禁令）。

**V100 · Do this next:** A1 无后续动作。**(6) 的三件事仍待跑**（插 recap 空档）：
① `python -m pytest tests/test_t_agnostic.py -v`（纯 CPU）；
② 1.6 多 T 退化曲线四条评测（命令在 (6)，单卡）；③ r3e 弹性冒烟（debug 自停）。

**b100:** 照 (5) 冒烟→R0，无任何变化。

**Then report back**（from-server.md）：(6) 三项的输出；另贴一行 `nvidia-smi`
的占卡概况，好排 ②③ 什么时候插空。

---

## 2026-07-24 (8) — 重述抽样检验已可跑：50 片全流程 → TSV（源caption/翻译/5重述/A′视图 并排）

**Diagnosis:** 主计划 Stage 0 的抽样检验链已实现并入 main（#227）：`split_views.py`
（A′ 句切，MiniRBT 32-token 预算，离线自动落 30 字符）、`recap_diag.py`（坍缩红旗 +
侧批对照）、`make_recap_tsv.py` + `run_v100.sh recap-tsv`。**先抽 50 片把整条链跑通、
出 TSV 目检，再决定放全量**——抽样输出与全量共用 `recap5.*.jsonl`、按 video 去重续跑，
50 片不会重算。TSV 自带 `en_source`（源 caption）与 `zh_translated`（翻译源 caption）
对比列（队列文件 `recaption_tasks.jsonl` 本就携带 en/zh_old）。

**Do this next（101vector；≈10 分钟 GPU + 秒级 CPU；与 (7) 的 V100 三件事不冲突，可排后）：**
```bash
source ~/audit_env/bin/activate
head -n 50 outputs/audit_10k/v0_2/recaption_tasks.jsonl > outputs/audit_10k/v0_2/recap_sample.jsonl
RECAP_INPUT=outputs/audit_10k/v0_2/recap_sample.jsonl bash tools/data_audit/run_v100.sh recap
bash tools/data_audit/run_v100.sh status     # 等 recap 行到 50/50
bash tools/data_audit/run_v100.sh recap-tsv  # 产 outputs/audit_10k/recap_inspect.tsv
```

**Then report back** in from-server.md：① recap_diag 摘要（`recap_diag.txt` 全文贴回，
含坍缩红旗行）；② TSV 里任挑 3 行贴回（en_source / zh_translated / recap_1 / views 四列）；
③ 50 片的起止时间（吞吐三元组，填主计划 §5.1）。目检若五条雷同（COLLAPSE 多）→ 先停，
等我调 temp 再放全量。

---

---

## 2026-07-24 (7) — 脑启发试点 `TemporalSurprisePool`：重跑已完成，但判据数还没读出来（V100，一条命令）

**Diagnosis:** `pilot_10k_surprise` 重跑正常收尾（40:52，ep0 1.03→ep9 0.28，无 NaN，
retrieval r_mean 最好 93.33@ep7）——**链路层面全绿**。但**训练日志读不出判据**：
试点的通过条件是 `alpha ≠ 0`（模块真的离开了 mean pooling），而这个数只存在 ckpt 里，
日志从头到尾不打印。

注意别被"曲线和 r0 一模一样"误导：**修复生效时曲线本来就该和 r0 一样**。`alpha` 从 0 起步，
每步梯度才 ~2e-3，170 步 × lr 1e-5 只能挪动极小量 → `output ≈ mean` → retrieval 自然不动。
所以"曲线相同"同时兼容【修复生效】和【没 pull 到修复】两种情况，**只有读 ckpt 能区分**。

背景：#220 修掉的是 `gamma=0 且 alpha=0` 这个**零梯度不动点**（softmax(0) 均匀 →
`pooled≡mean` → `(pooled−mean)=0` → alpha/gamma/key 三者梯度全为 0，模块永远醒不过来）。
修法是 `gamma` 初值 0→1，`alpha` 仍为 0 所以**零回退起点分毫不变**。

**V100（101vector）· Do this next（repo 根；纯 CPU 读盘，几秒，不占卡）：**

```bash
git pull origin main

# ckpt_09.pth 是 DeepSpeed ZeRO-1 的“目录”不是文件，权重在 module 键下
python - <<'PY'
import torch
p  = "experiments/runs/pilot_10k_surprise/ckpt_09.pth/mp_rank_00_model_states.pt"
sd = torch.load(p, map_location="cpu")["module"]
b  = "vision_encoder.temporal_pool."
print("alpha         =", sd[b+"alpha"].float().tolist())
print("gamma         =", sd[b+"gamma"].float().tolist())
print("key.weight L2 =", sd[b+"key.weight"].float().norm().item())
PY
```

**读到的两个数怎么判（`gamma` 判「修复在不在」，`alpha` 判「模块动没动」）：**

| `gamma` | `alpha` | 结论 |
|---|---|---|
| ≈ **1.0**（略微偏离） | **≠ 0** | ✅ 修复已生效且模块活着——链路判据**真正通过**，可以放行 H100 的 `nA1_surprisepool` A/B |
| ≈ 1.0 | **恰好 0** | ⚠️ 修复在但 alpha 没动——是新问题，把三个数贴回来我查 |
| **恰好 0.0** | 恰好 0.0 | ❌ 跑的还是旧代码（没 pull 到 #220）。执行下面的补跑 |

**只有第三行才需要补跑**（约 41 分钟，8 卡）：

```bash
git pull origin main
python -c "import neural_inspired,inspect,re;s=inspect.getsource(neural_inspired.TemporalSurprisePool.__init__);print('gamma init OK' if 'ones' in s else 'STILL OLD CODE')"
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_surprise.py 8
# 跑完再执行上面那段读 ckpt 的 python
```

**优先级：低于 recap 主线（(4)/(5)）**，读 ckpt 那条几秒钟随时可插；补跑只在 GPU 空档做。

**Then report back** 在 from-server.md：把 `alpha` / `gamma` / `key.weight L2` 三个数原样贴回来。
我会据此把结论写进 `experiments/pilot_10k_results.md`（现在那节还挂着"待服务器重跑确认"）。

---

## 2026-07-24 (6) — C2 弹性 T 代码已合入（#222）：V100 三件事（插空档），b100 零影响照跑

**Diagnosis:** ROADMAP 2.1 落地：模型里全部 11 处 `reshape(8,...)` 硬编码已动态化，
`temporal_embedding` 自动插值，弹性 T 训练开关 `elastic_t` **默认关闭 = 基线路径逐位
不变**（18 项单测已在 CPU 全过，含 T=8 与旧代码逐位一致）。**对 (5) 的 b100 冒烟→R0
完全零影响，照跑，勿改任何参数。** 新解锁三件事，都在 V100（101vector），且
**优先级全部低于 recap 主线（(4)/(5)），只插 GPU 空档**。

**V100 · Do this next（repo 根，先 `git pull origin main`）：**

```bash
# ① 单测（纯 CPU、~4 分钟、跑一次即可；预期 18 passed）：
python -m pytest tests/test_t_agnostic.py -v

# ② 1.6 · B0 官方权重多 T 退化曲线 = 论文 Fig-1 第一版（单卡、每个 T 几分钟、纯评测）
#    T=8 那组应对上 §3.4 的锚点（DSL t2v R@1≈42.5），作为链路自检：
for T in 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=0 bash experiments/run_eval.sh small msrvtt \
      ./checkpoints/mobileviclip_small.pt \
      experiments/evals/b0_multiT/msrvtt_T${T} \
      num_frames ${T} num_frames_test ${T}
done
# 看数（论文主口径 = ret_test_emb/ 原始余弦；ret_test/ 是 DSL 对齐口径）：
python - <<'PY'
import json, glob
for p in sorted(glob.glob("experiments/evals/b0_multiT/msrvtt_T*/eval_res.json")):
    print(p.split("/")[-2], json.dumps(json.load(open(p)), ensure_ascii=False)[:400])
PY

# ③ r3e · 弹性 T 链路冒烟（README §2b 矩阵 C2 行；8 卡但 debug 几十步自停）：
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r3e.py 8 debug True
```

**b100:** 无新动作——照 (5) 跑。R0 起跑后若想顺手验环境（CPU、不占卡）：
`python -m pytest tests/test_t_agnostic.py -v`。

**Then report back**（from-server.md）：① 是否 `18 passed`；② 四个 T 的 t2v/v2t R@1
（`ret_test_emb` 口径贴原文即可）——这就是 Fig-1 的第一版数据 + C2 难度标定；
③ r3e 是否自停干净、日志 `elastic_t` 是否在 2/4/8/16 间跳、有无 OOM/NaN。

> 备注：下方 (4)/(5) 各有**两块**（b100 数据线与重字幕线并行推进导致撞号）——
> 均有效，以标题区分；v1.1 块对服务器的指示 = Stage 0 照「重字幕主计划定稿」(4) 跑。

---

---

## 2026-07-24 (8) — 装错了包（`openai-clip` ≠ `open_clip_torch`）；且 `python` 根本不是 conda 环境

**两个问题，第二个更要紧。**

### (a) 包装错了：`openai-clip` 不是 `open_clip`

它们是**两个不相干的项目**：

| 你装的 | 仓库要的 |
|---|---|
| `openai-clip==1.0.1`（OpenAI CLIP）→ 提供模块 **`clip`** | **`open_clip_torch==2.20.0`**（LAION/mlfoundations）→ 提供模块 **`open_clip`** |

`mobileclip/tokenizer.py` 第 7 行 `import open_clip`，要的是后者。装 `openai-clip` 再多次也
补不上——模块名都不一样。仓库的规范装法（`scripts/setup_env.sh` Stage 6）：

```bash
python -m pip install --no-cache-dir --no-deps "open_clip_torch==2.20.0"
```
**`--no-deps` 是硬性要求**：它的 metadata 写了不带上界的 torch，pip 会顺手把 torch 升级掉。

### (b) ⚠️ 你的 `python` 是 `/usr/bin/python`，**不是** `mvc_env310`

你这条输出是实锤：
```
/usr/bin/python 2.5.1+cu124 ['sm_50',...,'sm_90']
```
提示符虽然亮着 `(mvc_env310)`，但 `sys.executable` 是 **`/usr/bin/python`**（系统解释器，
包在共用的 `/usr/local/lib/python3.10/dist-packages`）。含义有三：

1. **好消息**：系统 torch 是 **2.5.1+cu124 且 arch 含 `sm_90`**——Hopper 支持没问题，
   之前担心的 sm_90 这一关实际上是过的。
2. **坏消息**：你这一路 `pip install` 很可能**装进了共用的系统 site-packages**——这台是
   **共用 root 账号**，等于在改 wky/zyc 也在用的环境。**先停手，别再往系统里装。**
3. `run_pretrain.sh` 现在用 `python -m torch.distributed.run`，`python` 是谁就用谁——
   所以必须先把 `python` 指对，否则改了启动器也没用。

**先诊断环境到底怎么了：**
```bash
ENVP=/workspace/mount/b100_zaip_data/rby/mvc_env310
echo "CONDA_PREFIX=$CONDA_PREFIX"
which -a python python3 pip | head
ls -l "$ENVP/bin/python" 2>&1          # 这个文件在不在？
"$ENVP/bin/python" -V 2>&1              # 能不能跑？
"$ENVP/bin/python" -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())" 2>&1 | tail -1
```

**按结果二选一：**

- **情况 1：`$ENVP/bin/python` 存在且里面有 torch** → 只是 PATH 没生效。
  ```bash
  export PATH="$ENVP/bin:$PATH"
  which python && python -c "import sys;print(sys.executable)"   # 必须是 $ENVP/bin/python
  python -m pip install --no-cache-dir --no-deps "open_clip_torch==2.20.0"
  ```
- **情况 2：`$ENVP/bin/python` 不存在，或里面没有 torch** → 那个 conda env 等于没建成。
  **别再往系统里装**，最省事的隔离办法是**基于系统 py3.10 建带 `--system-site-packages`
  的 venv**：直接继承那套已经能用的 torch 2.5.1+cu124(sm_90)，新装的包只落在自己目录，
  既不用重下 torch，也不污染共用环境：
  ```bash
  VENV=/workspace/mount/b100_zaip_data/rby/mvc_venv310
  /usr/bin/python -m venv --system-site-packages "$VENV"
  ls -l "$VENV/bin/pip" || "$VENV/bin/python" -m ensurepip --upgrade   # 上次踩过 venv 没 pip 的坑
  export PATH="$VENV/bin:$PATH"
  which python pip && python -c "import sys;print(sys.executable)"
  python -m pip install --no-cache-dir --no-deps "open_clip_torch==2.20.0"
  ```

**装完复验（仍然零 GPU）：**
```bash
cd /workspace/mount/b100_zaip_data/rby/mobileclip
python -c "import sys,torch;print(sys.executable, torch.__version__, torch.cuda.get_arch_list())"
PYTHONPATH=. python -c "import models; print('models import OK')"
```

**Then report back**：诊断那五行的输出（判断走情况 1 还是 2）、以及复验两行是否都过。
`import models` 一过，就只剩等空闲卡冒烟了。

> `openai-clip==1.0.1` 对本仓库没用（模块名是 `clip`，没人 import 它）。它多半也装进了
> 共用系统环境；**不建议**再去系统里 uninstall（同样是动共用环境），留着不碍事即可。

---

## 2026-07-24 (7) — 8 张卡都满了：先做**不需要 GPU** 的验证，再按 PID 甄别占卡进程

**Diagnosis：卡满不挡事。** 上次崩在 `tasks_clip/pretrain.py:15` 的 `from models import *`
——**纯 import 阶段、纯 CPU**。所以 #229 那两处修复（解释器 + peft 惰性导入）**现在就能验，
一张卡都不用**。等卡是等冒烟，不是等验证。

**① 先验修复（不碰 GPU，现在就做）：**
```bash
git pull origin main
which python
python -c "import sys,torch; print(sys.executable, torch.__version__, torch.cuda.get_arch_list())"
#   ^ 只读 torch 的构建属性，不分配显存、不需要空闲卡。
#     python 必须在 .../rby/mvc_env310/bin/ 下；arch 列表必须含 sm_90。
#     若报 No module named 'torch' => 环境里根本没装 torch，回 §2（这也解释了上次为何跑成系统解释器）
PYTHONPATH=. python -c "import models; print('models import OK -- peft 惰性导入生效')"
#   ^ 这正是上次崩掉的那一行，纯 CPU。能过 = peft 已彻底离开训练 import 链
```
这两条过了，就说明 #229 修对了，剩下的**只差 GPU 排队**。

**② 甄别占卡的是谁——只看，不杀：**
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# 共用 root 账号下 ps 的 user 全是 root，**用 cwd 区分归属**：
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | cut -d, -f1); do
  echo "== PID $p"; ps -o pid,etime,cmd -p "$p" --no-headers
  ls -l /proc/"$p"/cwd 2>/dev/null | sed 's/.*-> //'
done
```
判读：
- cwd 在 **`rby/mobileclip`** → 是**你自己的**（上次崩溃/nohup 的残留），确认 `ps` 里
  确实是 `pretrain.py` 后可 `kill <PID>` 回收；
- cwd 在 **`rby/MobileViCLIP`**（中文线 wky）或 **`zyc/...`** → **别人的，别碰**。
- ⚠️ 重申 §0：共用账号，**只准按 PID kill，禁止 `pkill -f`/`killall`**——中文线跑的是
  同一个 `tasks_clip/pretrain.py`，模式匹配会误杀别人。

**③ 卡空出来再冒烟——不必等齐 4 张，1 张就能冒烟：**
```bash
CUDA_VISIBLE_DEVICES=<某张空闲卡号> bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 1 \
    use_bf16 True bf16 True deepspeed.enable False debug True
```
冒烟只为验证 arch/数据/前反向通不通，1 卡足够、也最容易插空。**真 R0 才需要前 4 张齐**
（等步纪律要求 4 卡/batch 256，见 §6；别为了赶进度用 3 卡跑 R0，那会毁掉 A/B 可比性）。

**Then report back**：① 的两行输出（解释器路径 + arch 含不含 sm_90；`import models` 过没过）、
② 占卡进程的 cwd 归属（判断要不要等别人）、③ 有空闲卡后冒烟的结果。

> 若 8 张卡长期被别人占满，那是**排期问题不是技术问题**——需要你跟 wky/zyc 或维护者协调
> 时间窗；Claude 这边没法代劳。

---

## 2026-07-24 (6) — 冒烟崩了：跑成了系统解释器 + peft 被白白 import。两处已修，先杀掉 R0

**Diagnosis：不是 sm_90 的问题，是两个故障叠在一起。**

**(a) 主因——整个训练没在你的 conda 环境里跑。** 报错路径是
`/usr/local/lib/python3.10/dist-packages/...`，binary 是 `/usr/bin/python3`，torchrun 是
`/usr/local/bin/torchrun`。`run_pretrain.sh` 从前调用**裸 `torchrun`**，按 `PATH` 解析时命中
了系统的 `/usr/local/bin/torchrun`（shebang → `/usr/bin/python3`），**绕开了 `mvc_env310`**。
→ 已把启动器改成 `python -m torch.distributed.run`，并打印 `python : <路径>`。

**(b) `peft` 纯属被捎带。** `backbones/internvideo2/__init__.py` 第 4 行急切
`from .internvideo2_clip_text import LLaMA, Tokenizer` → `import peft`。但 MobileViCLIP
两个塔**从不用 `LLaMA`/`Tokenizer`**，全仓库也只有那一个文件依赖 peft。
→ 已改为**惰性导入**（PEP 562 `__getattr__`），训练路径**完全不碰 peft**，peft 装没装、
什么版本都不再影响预训练。（这条是把 b100 中文线早就做过的通用增强移植进 fork。）

**另外**：`check_readiness.py` **从不 import torch**（只 import decord + configs.data），
所以 §5 全绿只证明数据接对了，**不证明环境能训**——这次就是这么被骗过去的。已在 §5 加警示。

**Do this next：**
```bash
# 0) 处理那个 4 卡 R0。⚠️ 千万别 pkill -f "tasks_clip/pretrain.py"：
#    这台是**共用 root 账号**，中文线(wky)跑的是**同一个** tasks_clip/pretrain.py，
#    模式匹配会连别人的训练一起杀掉。只按 PID 处理自己的。
#    它多半已经用同样的 ImportError 自己崩了 —— 先看，再决定要不要杀。
tail -20 experiments/runs/r0_baseline.log     # 若结尾是 ImportError/ChildFailedError => 已死，跳过下面
ps -p 1192760 -o pid,etime,cmd                # 1192760 = 你启动时打印的 "[1] 1192760"
jobs -l                                       # 若还在这个 shell 里，能看到它

# 只有在「确认还活着」且「确认是自己的」时才杀（先核对 cwd 是你的 repo）：
#   ls -l /proc/1192760/cwd     # 应指向 /workspace/mount/b100_zaip_data/rby/mobileclip
#   kill 1192760                # 按 PID，不要 pkill/killall
nvidia-smi                                    # 看前 4 张卡是否已释放

# 1) 拉修复
git pull origin main

# 2) 核实解释器与 sm_90（这一步才是真正的 §2 自检，之前一直没验过）
which python
python -c "import sys,torch; print(sys.executable, torch.__version__, torch.cuda.get_arch_list())"
#   python 必须在 /workspace/mount/b100_zaip_data/rby/mvc_env310/bin/ 下
#   arch 列表必须含 sm_90；若报 No module named 'torch' → 环境里压根没装 torch，回 §2 装
python -c "import torch; print(torch.zeros(1).cuda())"   # 应打印 tensor([0.], device='cuda:0')

# 3) 重跑 2 卡冒烟（现在应打印 python : .../mvc_env310/bin/python）
CUDA_VISIBLE_DEVICES=0,1 bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 2 \
    use_bf16 True bf16 True deepspeed.enable False debug True
```

**Then report back**：② 的三行输出（解释器路径 / torch 版本 / arch 列表含不含 sm_90），
以及 ③ 冒烟是否跑起来。冒烟过了我再放行 4 卡 R0。
**先别急着重开 R0**——这次务必等冒烟绿了再挂后台。

---

## 2026-07-24 (5) — §5 全绿，放行 §6：先 2 卡冒烟（顺带确认 sm_90）→ 再 R0 四卡

**结论：数据/权重全部就绪，R0–R3 解锁。** 预检 11 个 `internvid_v*` 全 `[OK ]`（抽查
20/20 在、真解码 OK），三个权重都在（`mobileclip_s2.pt` 398MB 等），预检末尾也标了
`[可跑] R0–R3 全量预训练(§4)`。§2–§5 我已在 `SETUP_B100.md` 进度清单里勾掉。

**Do this next（`rby/mobileclip` 根；先 `nvidia-smi` 确认前 4 张卡空闲——本机共享）：**
```bash
# 0) 首光冒烟：2 卡 + DDP+AMP + bf16 + debug，几十步自停。这一步同时最终确认 torch 有
#    Hopper(sm_90) 内核——若这里 CUDA kernel 报错，就是 §2 的 torch 没带 sm_90，回去换 cu12x。
CUDA_VISIBLE_DEVICES=0,1 bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 2 \
    use_bf16 True bf16 True deepspeed.enable False debug True
```
冒烟通了（跑完几十步没崩、loss 有数）再起真 R0：
```bash
# 1) 真 R0：4 卡、bf16、后台 nohup（掉隧道不影响），batch=64×4=256
mkdir -p experiments/runs
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash experiments/run_pretrain.sh \
    experiments/configs/r0_baseline.py 4 \
    use_bf16 True bf16 True > experiments/runs/r0_baseline.log 2>&1 &
tail -f experiments/runs/r0_baseline.log     # 头几百步看 loss 是否正常下降/发散/NaN
```

**纪律**：默认只用前 4 张卡（后 4 张留别人）；`<num_gpus>` 绝不能省（省了默认 8）；
R1 三臂要与 R0 同 batch/卡数/bf16/deepspeed 开关，才是等步对照（§6）。真跑装了
`deepspeed>=0.14` 后可去掉 `deepspeed.enable False`，但**一旦一臂用 deepspeed，所有臂都用**。

**Then report back**：冒烟是否通过（有没有 sm_90/CUDA 报错）、R0 头几百步的 loss 走势
+ 一个 epoch 的墙钟估算、显存占用（`nvidia-smi` 峰值，判断 bs 还能不能上调）。

---

---

## 2026-07-24 (5) — 主计划升 v1.1：全自动验收、零重人力标注、附全流程图（#223 已并入）

**Diagnosis:** 三路专家终审（自动评测设计 × 统计效度 × 可行性）后，
[`notes/recaption-master-plan.md`](../recaption-master-plan.md) 升 **v1.1**。对服务器操作的
影响：**Stage 0 的命令一字不变**（照 (4) 跑即可）；变化都在 G1 之后——原 ~600 判定的人力
k\* 标定与 300 片人评盲评**全部撤销**，改为 Panel-V 自动验真面板（InternVL2.5-8B +
GroundingDINO + OCR，全栈无 Qwen，§5.4）与视图级验真臂 C_v；G1 主臂改为零工具风险的句切
A′；G2 种子自适应（期望省 ~60% 训练算力）；§5.1 的「V100 跑 300k×5 ≈7 天」系错误估计已
撤销（实为 25–118 天），**H100 空档成为 Stage 3 硬前置**。全计划人力只剩 ≤75 分钟的
无约束力抽查。§3 顶部新增 mermaid 全流程图（GitHub 上看渲染版）。

**Do this next:** 与 (4) 相同——V100 跑 `run_v100.sh recap` + 两个侧批；b100 照 (3) 继续。
另外（可选、CPU、随时）：预下载 `hfl/minirbt-h288`、`sentence-transformers/LaBSE`，
D11 探针不再等 b100。

---

---

## 2026-07-24 (4) — 重字幕主计划定稿：执行以 notes/recaption-master-plan.md 为准

**Diagnosis:** 应维护者要求，对 `caption-fusion-scheme.md` 与 `data-quality-caprl-scheme.md`
两份重字幕方案做了全面评估（含两路独立评审：VLM 方法论 + 统计设计），裁决合并为唯一执行层
**[`notes/recaption-master-plan.md`](../recaption-master-plan.md)**（已并入 main，#221）。要点：
S0 五样本重述照跑不变；**训练门 G1 前置**（不需要任何人力标注），150–300 片人评标定收缩为
~600 单元判定（≈1 人·日）且**后置**到 G1 通过之后；融合（CFU）降为门控假设（外部评审先验
偏低，但保留 §6.4 可证伪测试）；RETRANSLATE 并入 RECAPTION；两份旧方案的分阶段清单不再照做
（各自文首已加取代声明）。

**Do this next（V100 审计机 101vector；= 主计划 Stage 0；与 b100 的 (3) 互不阻塞）:**
```bash
# ① 五样本重述主批（前置: s3+s4 已产 $OUT/v0_2/recaption_tasks.jsonl; 2,532 片当天级）
bash tools/data_audit/run_v100.sh recap
# ② 跑完记录: 总生成数 + 起止时间 + 卡数（吞吐是全量排期的唯一可信输入）
# ③ 两个侧批诊断（各 300 片, 主批完再跑; 精确命令见主计划 §3 Stage 0）
```
b100 侧照 **(3)** 继续英文线 anno → 预检 → §6 冒烟/R0（即主计划 §1 的 EN 论文线，两线并行）。

**Then report back** in from-server.md：recap5.* 行数、吞吐三元组、抽 3 片贴 `verdict.recaps`
（我要目检五条是否真有差异 = 温度坍缩检查）。

---

## 2026-07-24 (4) — anno 已建好；就差软链让预检的 data_root 目录存在

**Diagnosis:** 11 片 anno 全建成了（计数完全对：1–10 各 100 万、11 是 647458）。预检那
11 行报的是 `data_root 缺失: data/video_sample/InternVId-FLT_i`——你**跳过了 ③ 建软链**，
且 `MVC_DATA_DIR` 是默认 `data`，所以那些目录还不存在。**这一步只是让目录存在**：`video`
是 `abs_path`，训练根本不走 data_root，只是 `check_readiness` 要 `isdir(data_root)`。

**Do this next（就在 `rby/mobileclip` 根，不用 export MVC_DATA_DIR）：**
```bash
FULL=/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT   # abs_path 的根：InternVid(小写)
mkdir -p data/video_sample
for i in $(seq 1 11); do ln -sfn "$FULL/InternVId-FLT_$i" "data/video_sample/InternVId-FLT_$i"; done
ls -ld data/video_sample/InternVId-FLT_*/     # 应 11 条软链，各解析到真实分片目录
python experiments/check_readiness.py --decode 2>&1 | grep -E "internvid_v"
```

**Then report back**：那 11 行现在是不是 `[OK ]`。
- 若还 `data_root 缺失` → `$FULL/InternVId-FLT_i` 目录名大小写没对上（根 `InternVid`、分片
  `InternVId-FLT_i`，用 `ls $FULL` 核对真实名字再改软链 target）。
- 若变成 `抽查…全部缺失` → abs_path 指的 mp4 真不在盘上（拿一条 abs_path `ls -l` 验证）。

**顺带（§3，R0 之前必须）**：`checkpoints/mobileclip_s2.pt` 还缺，预检也标了。R0–R3 底座
必需——按 §3 用 gdown 下好它，否则 §6 起不来。评测基准那 7 个 anno 缺失（msrvtt/didemo…）
只影响 B0 评测，不挡 R0–R3 预训练，先不用管。

---

---

## 2026-07-24 (3) — 覆盖确认为全量 10.6M，放行：建 11 片 anno → 预检 → §6

**结论：这份 jsonl 就是作者 InternVid-10M-FLT 全量，非子集。** 你的计数：total
**10,647,458**、`video` 前缀 0 不匹配、FLT_1..10 各 100 万、FLT_11 64.7 万。可直接替代 gdown。

**Do this next（`rby/mobileclip` 根，接着方案 C 往下）：**
1. 照 §4「方案 C ②」建 11 片 anno（`video=abs_path`、`caption=caption`）。
2. 照 §4「方案 C ③」建软链——`$FULL=/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT`
   （**根是 `InternVid`，分片是 `InternVId-FLT_i`，大小写别写反**，以 abs_path 为准），
   `export MVC_DATA_DIR="$WORK/data"`，再 `python experiments/check_readiness.py --decode`。
3. 11 个 `internvid_v*` 全 `[OK ]`（含真解码）后，进 §6：先 2 卡冒烟，再 R0 4 卡。

**Then report back**：② 每片生成条数（应对上上面的计数）、③ 预检 11 行、§6 冒烟头几百步
loss 走势。若某片 `data_root 缺失` 就是软链没对上 `$FULL/InternVId-FLT_i`；若 `全部缺失`
则是软链目标层级不对（abs_path 已是绝对路径，通常不该出现）。

---

## 2026-07-24 (2) — probe 结果确认可用；定稿：video 用 abs_path，caption 用 caption

**结论：可以用 zyc 这份 jsonl 替代 gdown。** 你贴的 probe 已确认：
- **源英文 caption = `caption`**（`'woman using a computer mouse and keyboard'`），中文在
  `raw_caption_zh_translated`——英文 R0–R3 用 `caption`。
- 每条带绝对路径 **`abs_path`**，且 `video` 是**深度嵌套**（`InternVId-FLT_i/shard_xxxxx/InternVId-FLT_i/<file>.mp4`）。
- 因为 loader 是 `os.path.join(data_root, rec["video"])` 且 `os.path.join(root, 绝对路径)=绝对路径`，
  **直接把 `abs_path` 当 `video` 字段最稳**，跟软链/嵌套层级无关。已把 §4「方案 C」改成定稿版。

**还差一件事**：你上一次 probe 的**分片计数那段被 `^S`（Ctrl-S）卡住没打出来**。先补跑它，
我要用 total / 每片条数判断这是作者 FLT 全量还是子集（子集=换了训练集，得有意为之）。

**Do this next（`rby/mobileclip` 项目根，conda 环境已激活）：**
```bash
SRC=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
# ① 覆盖核查（若终端再卡按 Ctrl-Q 解冻）
python - <<PY
import json,re,collections
rx=re.compile(r"^(InternVId-FLT_(?:1[01]|[1-9]))/"); c=collections.Counter(); n=bad=nocap=0
for l in open("$SRC"):
    if not l.strip(): continue
    n+=1; r=json.loads(l); m=rx.match(r.get("video",""))
    c[m.group(1)]+=1 if m else 0; bad+=0 if m else 1
    cap=r.get("caption"); nocap+=0 if (isinstance(cap,str) and cap.strip()) else 1
print("total:",n,"| video 前缀不匹配:",bad,"| caption 空/非串:",nocap)
for i in range(1,12): print(f"  InternVId-FLT_{i}: {c[f'InternVId-FLT_{i}']}")
PY
```
① 出数后，直接照 §4「方案 C」的 ②（生成 11 片，`video=abs_path`）+ ③（建软链过预检）跑；
命令都在 md 里、已定稿，不用再填占位。

**Then report back** in from-server.md：贴 ① 的 total + 每片条数（判断全量/子集）、②
每片生成条数、③ `check_readiness --decode` 的 11 行 `internvid_v*` 结果。全 `[OK ]` 就进 §6 冒烟。

---

## 2026-07-24 (1) — 英文线 anno：能否用 zyc 合并 jsonl 的源 caption 免 gdown？先 probe/test

**Diagnosis:** zyc 的翻译合并文件里**同时含英文源 caption**，理论上可直接生成 11 个训练
anno，省掉 §4 的 gdown。但两件事我这边看不到、必须你在 b100 上 probe 确认：**(a)** 源英文
caption 是哪个 key；**(b)** `video` 前缀是否覆盖 `InternVId-FLT_1..11`、条数是不是你要的规模
（远小于作者 FLT 全量=子集，等于换训练集，要有意为之）。英文 R0–R3 要的是**源英文 caption**，
不是中文 `raw_caption_zh_translated`。详见 `experiments/SETUP_B100.md` §4「方案 C」。

**Do this next（在 `rby/mobileclip` 项目根，已 `source .gitenv` 且 conda 环境激活）：**
```bash
SRC=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/aligned_internvid_zh_translated_retry_merged.jsonl
# ① probe：认出源英文 caption 字段名 + 看分片覆盖/条数
python - <<PY
import json,re,collections
r=json.loads(next(l for l in open("$SRC") if l.strip()))
print("keys:",list(r.keys()))
for k,v in r.items(): print(f"  {k!r:34s} {type(v).__name__:6s} {repr(v)[:150]}")
rx=re.compile(r"^(InternVId-FLT_(?:1[01]|[1-9]))/"); c=collections.Counter(); n=bad=0
for l in open("$SRC"):
    if not l.strip(): continue
    n+=1; m=rx.match(json.loads(l).get("video",""))
    c[m.group(1)]+=1 if m else 0; bad+=0 if m else 1
print("total:",n,"| video 前缀不匹配:",bad)
for i in range(1,12): print(f"  InternVId-FLT_{i}: {c[f'InternVId-FLT_{i}']}")
PY
```
拿到 ① 输出后，按 §4「方案 C」的 ②（生成 shard 1 + `check_readiness` 验 `internvid_v1`，
用来确定 `CAPKEY` 和 `STRIP`），③（验好再一次性生成 11 片）执行——那两段命令都在 md 里，照抄。

**Then report back** in from-server.md：贴 ① 的完整输出（keys 那几行 + 每分片条数）、
② 里 `internvid_v1` 那一行是 `[OK ]` 还是 `全部缺失`、以及你最终用的 `CAPKEY`/`STRIP`。
据此我把 §4「方案 C」的占位改成对准你真实字段的定稿脚本，并判断是否算“全量”。

---

## 2026-07-23 (6) — §2 返工：先清理误装进 conda base 的包，再用 conda py3.10 重建

**Diagnosis:** 你的 §2 输出显示两个叠加故障：**(a)** `python -m venv` 建的环境没带
pip，`pip` 落到共用 conda base——torch 2.6+cu124 等全装进了 `/root/miniforge3`，venv
是空的（所以 arch 自检报 `No module named 'torch'`；提示符 `(mvc_env) (base)` 叠着就是
警报）；**(b)** miniforge 是 py3.12，tokenizers 0.13.3 没有 cp312 轮子、源码编译要
Rust，所以 transformers 那整条 pip 一个包都没装上。已把 §2 配方改成 conda python=3.10
+ 全程 `python -m pip`（`SETUP_B100.md` §2），故障记入 `TROUBLESHOOTING.md`
「b100 §2：pip 装错解释器」。

**为什么不换个方式继续用 venv：** venv 只能复制机器上已有的解释器——本机唯一的
python 是 miniforge 的 3.12，venv 建出来永远还是 3.12，而 tokenizers 0.13.3 没有
cp312 轮子（源码编译要 Rust），装到它必死；`ensurepip` 只能修好"venv 没 pip"这一层，
修不了 3.12 这一层。要拿到仓库钉的 python 3.10，本机只有 conda 能提供——所以
conda create 是唯一同时解决两个问题的路，不是偏好。

**Do this next**（照顺序；③④ 需要代理开着）：

```bash
# ① 退出坏 venv 并删除
deactivate 2>/dev/null || true
rm -rf /workspace/mount/b100_zaip_data/rby/mvc_env

# ② 清理误装进共用 base 的包（这些装之前 base 里都没有，卸载=还原，不影响别人已有的东西）
/root/miniforge3/bin/python -m pip uninstall -y \
  torch torchvision torchaudio triton sympy mpmath numpy networkx jinja2 MarkupSafe \
  fsspec pillow open_clip_torch deepspeed py-cpuinfo hjson msgpack ninja einops \
  psutil pydantic pydantic-core annotated-types typing-inspection \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12

# ③ conda 建 py3.10 环境（仍在 rby/ 内）并自检解释器归属
source /root/miniforge3/etc/profile.d/conda.sh
conda create -y -p /workspace/mount/b100_zaip_data/rby/mvc_env310 python=3.10
conda activate /workspace/mount/b100_zaip_data/rby/mvc_env310
which python && which pip   # 必须都在 .../rby/mvc_env310/bin/ 下；不是就停下来贴回来

# ④ 重装（全程 python -m pip；torch 768MB 走隧道约 20 分钟，正常）
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install "numpy<2"
python -m pip install open_clip_torch --no-deps
python -m pip install transformers==4.28.1 tokenizers==0.13.3 timm==0.5.4 openai-clip==1.0.1 \
    einops decord==0.6.0 av==12.0.0 opencv-python ftfy regex pandas scipy scikit-learn \
    pyyaml easydict yacs tabulate tensorboard wandb webdataset imageio \
    accelerate safetensors huggingface-hub setuptools
python -m pip install "deepspeed>=0.14" || true

# ⑤ arch 自检（这回 python = py3.10 环境里的）
python - <<'ARCH'
import torch
a = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| archs:", a, "| dev:", torch.cuda.get_device_name(0))
assert "sm_90" in a, "torch 无 Hopper(sm_90) 内核 -> 换 cu118+/cu12x 的 torch 2.x"
print(torch.zeros(1).cuda())   # 必须打印 tensor([0.], device='cuda:0')
ARCH
```

> 若 ③ 的 `conda create` 走不通（频道被墙/代理不认），把报错贴回来，我再给
> rustup+py3.12 的备选路线。

**Then report back** in from-server.md：② 卸载是否顺利、③ 的 `which python`/`which pip`
输出、⑤ arch 自检整段输出（torch 版本 + arch 列表 + tensor 打印）。

---

## 2026-07-23 (5) — 并排 clone 英文改进线：`rby/mobileclip`（中文线原目录不动）

**Diagnosis:** 快照 `87b812e` 已收到并逐行审读完毕（记录：`notes/b100-local-changes.md`；
通用修复已按方案 B 移植进 fork main，PR #195）。现在把两条线在盘上分开：
现有 `rby/MobileViCLIP` **保留给中文 NebulaCV 线**（上游 clone，wky 协作，别动）；
英文原版模型改进线（R0–R3 全量预训练）**另 clone 本 fork** 到并排的 `rby/mobileclip`。
`SETUP_B100.md` §0/§1 已按此更新，venv 也改到 rby/ 内（$HOME 共用）。

**Do this next**（代理开着；PAT 上次 push 已存进 `rby/.git-credentials`，不会再问）：

```bash
source /workspace/mount/b100_zaip_data/rby/.gitenv
cd /workspace/mount/b100_zaip_data/rby
git clone https://github.com/rbybryan/mobileclip.git mobileclip
cd mobileclip
git log --oneline -3          # 应在 main，且能看到 PR #195 的合并提交
git config core.fileMode false
```

之后照 [`experiments/SETUP_B100.md`](../../experiments/SETUP_B100.md) **§2 起**逐节走
（arch 自检 → 权重 → 数据软链 → `check_readiness.py` 预检 → §6 冒烟）。注意：
- venv 用 `/workspace/mount/b100_zaip_data/rby/mvc_env`（§2 已更新，别装进 ~）；
- 数据软链的 `WORK=/workspace/mount/b100_zaip_data/rby/mvc_stage` 与项目目录无关，照旧；
- 中文线目录 `rby/MobileViCLIP` 里的任何东西都不要挪/删。

**Then report back** in from-server.md：clone 是否成功 + `git log -3` 输出；到 §2 后按
`SETUP_B100.md` §8 的清单回报（arch 自检、探盘、预检、首日墙钟）。

---

## 2026-07-23 (4) — 改动清单已收到；push 403 的原因 + 重做快照（排除权重）推到 fork

**Diagnosis:** 三个新事实：
1. b100 的 `MobileViCLIP` 是**上游 `MCG-NJU/MobileViCLIP` 的 clone**（`origin` 指向上游），
   所以快照推送被 `403 Permission denied` 拒了——推错了远端，不是 token 问题。
2. clone 我们的 fork（`rbybryan/mobileclip`，私有库）时输的是账号密码——GitHub 已废弃
   密码认证，必须用 **PAT**（个人访问令牌）。
3. 快照 `1dabf71` **作废**：`git add -A` 把 `checkpoints/nebulacv-clip-24m.pt` 权重也收
   进去了（上游 .gitignore 不覆盖它）。重做前先用 `.git/info/exclude` 排除。

文件级改动清单已据你回传的 `git status`/diffstat 记入
[`notes/b100-local-changes.md`](../b100-local-changes.md)（主题：NebulaCV 中文 CLIP
文本塔 + 中文 InternVid 标注管线）；还差逐行 diff，按下面走完就齐了。

**Do this next**（需要代理开着；PAT 在手边——就是你在别处 push 这个 fork 用的那个 token）：

```bash
source /workspace/mount/b100_zaip_data/rby/.gitenv
cd /workspace/mount/b100_zaip_data/rby/MobileViCLIP

# ① origin 是上游（只读）。加一个 fork 远端指向我们的私有仓库：
git remote add fork https://github.com/rbybryan/mobileclip.git 2>/dev/null
git remote -v    # 应看到 origin=MCG-NJU（保留）、fork=rbybryan

# ② 凭据也圈在 rby/ 内（不落共用账号的 $HOME）。首次 push 会问一次：
#    Username 填 rbybryan，Password 填 **PAT**（不是账号密码）
git config --global credential.helper "store --file=/workspace/mount/b100_zaip_data/rby/.git-credentials"

# ③ 权重不进快照（只写本仓库的 exclude，不动工作区、不改 tracked 的 .gitignore）
printf 'checkpoints/\n' >> .git/info/exclude

# ④ 重做快照并推到 fork
git add -A
git status -s | grep -c '^A.*checkpoints/' || true   # 必须是 0
STASH=$(git stash create "b100 nebulacv local changes snapshot $(date +%F_%H%M)")
git reset -q
echo "snapshot commit: $STASH"
git push fork "$STASH":refs/heads/server/b100-local-opts
```

> push 时会把上游那份历史一并传到 fork（两边没有共同历史，git 会补齐对象）——预期行为，
> 别慌。若 PAT 不在手边、push 实在走不通，替代：
> ```bash
> git diff HEAD > ../b100_nebulacv.patch
> tar czf ../b100_new_files.tgz $(git ls-files --others --exclude-standard)
> ```
> 把 patch 全文贴进 `from-server.md`（新文件先只贴文件名，逐个要）。

**Then report back**：`snapshot commit` 新哈希 + push 是否成功。成功后 Claude 直接
`git fetch` 读快照，把 `notes/b100-local-changes.md` 的「待确认」全部填实。

---

## 2026-07-23 (3) — 候选字幕融合 F0：`s2-merge` 融合 + 可视化人审（同 better_zh）

**背景：** 你已人工复核 `better_zh`（P0 通过）：新候选更准更全、3 条各带不同细节，
融合有望更好。完整融合方案见 [`notes/caption-fusion-scheme.md`](../caption-fusion-scheme.md)。
**好消息：F0（最简版）不用新写代码** —— 仓库里 `s2-merge`（`vlm_audit_hf.py --mode merge`
+ `MERGE_PROMPT`）已实现「单次 LLM 调用把 2–3 条候选融合成一条、**闭集防幻觉**（绝不新增
候选里没有的细节）、保时序」。它**还没有** support 计票／单例验真／CSER 双向验收（那些是
方案的 F1+）。**F0 的目的：用 300 片人审判断「融合是否真比单条更全、且幻觉 <10%」**，
达标才值得建 F1。人审用**和 better_zh 完全同款的离线可视化包**（视频+候选+融合稿同屏点按钮，
见第 3 步），不用 Gradio、不用手拼 TSV。

**做这几步（repo 根目录，数据机）：**

```bash
git pull                                   # 取到 caption-fusion-scheme.md
# 1) 跑融合（不解码视频、8 卡、可续跑；产出 outputs/audit_10k/s2_merged.*.jsonl）
bash tools/data_audit/run_v100.sh s2-merge
bash tools/data_audit/run_v100.sh status   # 看进度；跑完 8 分片行数之和 ≈ 有 ≥2 候选的片数
```

```bash
# 2) 融合体检（只看健康、不建表；不健康就别急着审）
source ~/audit_env/bin/activate && cd ~/mobileclip
python - <<'PY'
import json, glob, random
random.seed(0)
OUT="outputs/audit_10k"
cand={}
for f in glob.glob(f"{OUT}/s2_verdicts.*.jsonl"):
    for l in open(f,encoding="utf-8"):
        r=json.loads(l); v=(r.get("verdict") or {})
        cs=[c for c in (v.get("better_zh") or []) if isinstance(c,str) and c.strip()]
        if r.get("video"): cand[r["video"]]=cs
tot=real=passthru=nullc=err=0; ex=[]
for f in glob.glob(f"{OUT}/s2_merged.*.jsonl"):
    for l in open(f,encoding="utf-8"):
        r=json.loads(l); v=(r.get("verdict") or {}); tot+=1
        cs=cand.get(r.get("video")) or []; m=v.get("merged_zh")
        if v.get("error"): err+=1; continue
        if len(cs)>=2 and isinstance(m,str) and m.strip(): real+=1; ex.append((cs,m))
        elif len(cs)==1: passthru+=1
        else: nullc+=1
print(f"总行数 {tot} | 真融合(≥2候选) {real} | 单候选透传 {passthru} | 空(0候选/黑屏) {nullc} | 判定失败 {err}")
print("--- 抽 3 例肉眼看融合质量 ---")
for cs,m in random.sample(ex, min(3,len(ex))):
    print(f"候选: {' ／ '.join(cs)}\n融合: {m}\n")
PY
```

**先看这几个数**（不健康就别急着审）：真融合应 ≈9000+（`better_zh` 是 9951×3）；
空的应 ≈49（黑屏片，`better_zh` 为空）；**判定失败应接近 0**——若很多，先别审，把数字贴 Claude 查。
抽的 3 例也扫一眼：融合稿有没有把候选的细节合起来、读着通顺。

```bash
# 3) 可视化人审：一条命令建表 + 打离线包（视频+候选+融合稿同屏，和 better_zh 完全同款）
bash tools/data_audit/run_v100.sh merge-review        # 默认抽 300；MERGE_REVIEW_N=100 可调
# 取回本地看画面：
#   服务器  tar czf ~/bundle_merge.tgz -C outputs/audit_10k bundle_merge
#   本地    scp vca1@10.235.31.101:~/bundle_merge.tgz . && tar xzf bundle_merge.tgz
#           浏览器打开 bundle_merge/index.html  （视频 + 3候选 + 绿色融合稿 + 按钮同屏，进度自动存）
```

**在浏览器里每条点两个按钮**（页头有提示）：
- **幻觉 [y/n]**：绿色融合稿有没有说**任何候选都没提、且画面里也没有**的东西？有=y（最要紧，验闭集守没守）。
- **更全 [y/n]**：融合稿比**其中写得最好的那一条候选**含更多**真实**细节？是=y（验融合有增益）。
- 进度存浏览器本地，关掉再开不丢；填一半也能先统计。

```bash
# 4) 统计决策门（导出的 TSV 传回服务器后跑；复用现成 review-score，自动判两个门）
bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/review_merge.tsv
```

**决策门（`review-score` 会自动打印，对齐方案 §6.2 F0）：**
- **幻觉率 <10%** 且 **更全率 >50%** → F0 通过。落地：`bash tools/data_audit/run_v100.sh s4-merged`
  把 `merged_zh` 写进训练 JSON，再按 §6.3 做 **融合 vs best-single vs uniform-K** 消融（这才是「融合有用」的硬判据）。
- **幻觉率 ≥10%** → 现成 `MERGE_PROMPT` 的闭集约束不够，需上方案 F1 的 support 计票 + 单例验真
  （`support≥2` 采信、单例回看 8 帧 yes/no）。这要新写 `tools/fuse_verify.py`，跟我说一声我来写。

> 纯文本快速版（不想看画面时）：`merge-review` 建的 `review_merge.tsv` 也能直接在 Excel 里填
> `幻觉[y/n]`/`更全[y/n]` 两列，再跑同一个 `review-score`。但**判「更全」看画面更靠谱**——能直接核对某个细节到底在不在视频里。

**Then report back** in from-server.md：贴第 4 步 `review-score` 的「幻觉率 / 更全率 + 判定」，我据此定 F1 要不要做、怎么做。

---

## 2026-07-23 (2) — b100 目录不是 git 仓库：嫁接 `.git` 再采集；git 配置全部圈在 rby/ 内

**Diagnosis:** 上一块的命令全报 `fatal: not a git repository` ⇒
`/workspace/mount/b100_zaip_data/rby/MobileViCLIP` **没有 `.git`**——代码当年不是按
`SETUP_B100.md` §1 clone 的，而是直接拷贝过去的，所以 git 无法对比。修法：用
`git clone --no-checkout` 只取 `.git` 嫁接进现有目录，把它变成"基于 origin/main 的脏
工作区"，改动就全部现形。另外账号是**共用的**：所有 git "全局"配置一律通过
`GIT_CONFIG_GLOBAL` 圈在 `rby/` 目录内，不碰共用的 `~/.gitconfig`。
（已同步记入 `TROUBLESHOOTING.md`。）

**Do this next**（① 每次开工都要 source；②需要代理，见 `SETUP_B100.md` §0）：

```bash
# ① git 环境隔离（只此一次生成；以后每个要用 git 的 shell 先 source 它）
cat > /workspace/mount/b100_zaip_data/rby/.gitenv <<'EOF'
export GIT_CONFIG_GLOBAL=/workspace/mount/b100_zaip_data/rby/.gitconfig
export GIT_CONFIG_SYSTEM=/dev/null
EOF
source /workspace/mount/b100_zaip_data/rby/.gitenv
git config --global safe.directory /workspace/mount/b100_zaip_data/rby/MobileViCLIP
git config --global user.name  "rby (b100)"
git config --global user.email "rbybryan1995@gmail.com"

# ② 嫁接 .git（代理开着；只下载版本库，不碰现有工作区任何文件）
cd /workspace/mount/b100_zaip_data/rby
git clone --no-checkout https://github.com/rbybryan/mobileclip.git MobileViCLIP_gitref
mv MobileViCLIP_gitref/.git MobileViCLIP/.git
rm -rf MobileViCLIP_gitref
cd MobileViCLIP
git config core.fileMode false      # 挂载盘的权限位噪声不算改动（写进仓库自己的 .git/config）
git reset -q origin/main            # 索引对齐 main；工作区一个字节都不动

# ③ 现在改动全部可见了——把这些输出贴回 from-server.md：
git status -sb | head -60
git diff --stat
git ls-files --others --exclude-standard | head -60

# ④ 快照并推回（不动工作区、不切分支、不影响正在跑的任务）
git add -A
STASH=$(git stash create "b100 local optimizations snapshot $(date +%F_%H%M)")
git reset -q
echo "snapshot commit: $STASH"
git push origin "$STASH":refs/heads/server/b100-local-opts
```

> 注意：③ 的 diff 是相对**当前 main HEAD** 的——如果这份拷贝当初基于旧版本，里面会混入
> "仓库后来的演进"。不用管：Claude 拿到快照后会自己在历史里找最小 diff 的基准 commit，
> 把真正的本地优化和版本漂移分开。
> `git add -A` 遵守仓库 `.gitignore`，`data/`、`checkpoints/` 等大文件不会进快照。

**Then report back** in from-server.md：③ 的三段输出 + `snapshot commit` 哈希 + ④ push
是否成功。之后这个目录就是正规 git 仓库，`SETUP_B100.md` §1 的 `git pull origin main`
也能用了（记得先 `source /workspace/mount/b100_zaip_data/rby/.gitenv`）。

---

## 2026-07-23 — b100 本地"优化"改动盘点：先把 diff 完整传回来

**Diagnosis:** b100 上 `/workspace/mount/b100_zaip_data/rby/MobileViCLIP` 的代码相对
远程仓库做过少量优化，但这些改动**只存在于服务器工作区**——仓库里没有任何记录
（`from-server.md` 为空、也没有对应分支；`exp/opt-vs-baseline` 是早已合入 main 的旧
分支，与此无关）。Claude 连不上 b100，所以第一步是在服务器上把改动完整采集回来；
收到后 Claude 会把「改了哪里、为什么」记录到 `notes/b100-local-changes.md`，再评审
是否合入 main。

**Do this next**（①–③ 纯本地 git 操作，不需要外网；只有 ④ 的 push 需要隧道+代理，
见 `experiments/SETUP_B100.md` §0）：

```bash
cd /workspace/mount/b100_zaip_data/rby/MobileViCLIP

# ① 全景：基于哪个 commit、改了哪些文件、有没有本地 commit / stash
git status -sb
git log --oneline -5
git log --oneline origin/main..HEAD
git stash list
git diff HEAD --stat

# ② 未跟踪的新文件（数据/权重等 gitignored 的不会出现在这里）
git ls-files --others --exclude-standard

# ③ 把完整改动（含未跟踪新文件）做成一个快照 commit —— 不动工作区、不切分支、
#    不影响正在跑的任务：
git add -A
STASH=$(git stash create "b100 local optimizations snapshot $(date +%F_%H%M)")
git reset -q
echo "snapshot commit: $STASH"

# ④ 推回（需要代理开着；SETUP_B100.md §0 的 http_proxy 设好后执行）
git push origin "$STASH":refs/heads/server/b100-local-opts
```

> ④ 推不动（隧道没起）时的替代：`git diff HEAD > /workspace/mount/b100_zaip_data/rby/b100_local_changes.patch`，
> 把 patch 内容连同 ① 的输出一起贴进 `from-server.md` 顶部新块再推 handoff；
> patch 太长贴不下就先只贴 ① ② 的输出。

**Then report back** in from-server.md：① ② 的完整输出 + `snapshot commit` 哈希 +
④ push 是否成功。Claude 拿到 `server/b100-local-opts` 后会 `git diff <snapshot>^1 <snapshot>`
读出全部改动并记录。

---

## 2026-07-22 (20) — 评审包视频全空：**路径 bug 已修**，重打包即可

**Diagnosis:** 你本地 `videos/` 全空 = 服务器打包时一个视频都没定位到。根因是**重复
实现漏洞**：抽检表用的是简化版路径解析（直接用 `abs_path` 或盲拼），而 `abs_path`
是**别的主机**的绝对路径、`video` 字段又带 `videos/` 这类本机没有的前缀，拼出来的
路径本机不存在。审查流水线（S1/S2）当初能成功解码，靠的是 `vlm_audit.resolve_video_path`
——它会剥掉前缀、在多个候选里返回**第一个真实存在**的。抽检表没复用它。**已修：改成
复用同一个解析器，并在建表时打印 `[videos] X/N` 命中数。**

**Do this next:**

```bash
cd ~/mobileclip && git pull

# ① 重新生成表——现在会打印 [videos] X/15，先确认不是 0
bash tools/data_audit/run_v100.sh review
```

**看那行 `[videos] X/15`：**

- **是 15/15** → 路径修好了，继续 ②。
- **还是 0/15** → 视频不在默认 `$CLIPS`（`/home/vca1/datasets/internvid_zh_10k/videos`）下。
  把下面三行的输出贴到 `from-server.md`，我据此调 `CLIPS`：
  ```bash
  head -1 outputs/audit_10k/s2_sample.jsonl | python3 -m json.tool | grep -iE 'video|abs_path'
  ls /home/vca1/datasets/internvid_zh_10k/videos/ | head -3
  find /home/vca1/datasets/internvid_zh_10k -name '*.mp4' | head -3
  ```

```bash
# ② 命中正常后打包
bash tools/data_audit/run_v100.sh review-bundle
tar czf ~/review_bundles.tgz -C outputs/audit_10k $(cd outputs/audit_10k && ls -d bundle_*)
```

```bash
# ③ 本地（Linux 用 xdg-open，不是 open；包内附 README.txt）
scp vca1@10.235.31.101:~/review_bundles.tgz . && tar xzf review_bundles.tgz
xdg-open bundle_conflicts/index.html
```

页头会显示 **「视频 X/N」**——一眼确认有没有打包进去；缺失的行会显示占位+路径，
不再是静默坏掉的播放器。

**Then report back** in from-server.md：`review` 那行 `[videos] X/15` 的数字。
若是 0，连同上面三行诊断输出一起贴。

---

## 2026-07-22 (19) — 抽检怎么开跑：**别装 gradio**，用离线评审包

**Diagnosis:** 你那个 `readonly database` 报错**与 gradio 无关，是权限**——
`gradio_video_text_review.py` 把标注库写在**它自己所在的目录**
（`/home/vca1/...` 下），你是 `rby` 只有读权限。报错落在
`PRAGMA journal_mode=WAL` 而不是 `connect()` 正是佐证：SQLite 开 WAL 要在
**库文件所在目录**建 `-wal`/`-shm`，库文件本身可读也没用。

而且**就算装上也不好用**：那个 Gradio 是**浏览数据集**的，没有抽检表的填写字段——
你得一边看 Gradio 一边在 Excel 里按 video id 对着填。

所以走离线评审包：视频 + 自带填写界面的 HTML 打成一个目录，本地浏览器里评，
**服务器上什么都不用装**。

**Do this next:**

```bash
git pull

# ① 打包（纯 stdlib；默认跳过 random 那 300 条视频，太大）
bash tools/data_audit/run_v100.sh review-bundle
tar czf ~/review_bundles.tgz -C outputs/audit_10k $(cd outputs/audit_10k && ls -d bundle_*)
```

```bash
# ② 本地机器上（不是服务器）
scp vca1@10.235.31.101:~/review_bundles.tgz . && tar xzf review_bundles.tgz
open bundle_conflicts/index.html     # 先过这 15 条，几分钟
open bundle_better_zh/index.html     # 再过 50 条
```

页面里视频／原字幕／候选字幕／单选按钮**同屏**；进度存浏览器本地，关掉再开不丢；
按 `n` 跳下一条未填；填完点**「导出 TSV」**。

```bash
# ③ 把导出的 TSV 传回服务器统计（脚本自己判决策门，不用手算比例）
scp review_conflicts.tsv vca1@10.235.31.101:~/mobileclip/outputs/audit_10k/
bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/review_conflicts.tsv
bash tools/data_audit/run_v100.sh review-score outputs/audit_10k/review_better_zh.tsv
```

### 三张表填什么、为什么

| 表 | 填的列 | 判什么 | 为什么要紧 |
|---|---|---|---|
| `conflicts`（15 条） | `谁对` = `s0`/`vlm`/`unclear` | S0 说该重翻、实测说翻译没问题，谁对 | 了结「语言污染 3.3%」的**待核**标记（问题 12） |
| `better_zh`（50 条） | `准确性`/`具体性` = `better`/`same`/`worse`；**`幻觉` = `y`/`n`** | 重写的字幕到底有没有变好 | **v0.2 落地前的硬前置**——`apply_buckets.py` 会把未验证的 `better_zh` 直接写进训练数据（问题 13） |
| `random`（300 条） | `同意打分` = `y`/`n` | 单一质量分准不准 | 顺带能澄清裁判判 `recaption` 63% vs S3 给 25.32% 的分歧 |

**`better_zh` 那张表，「幻觉」那一列最要紧**：准确性/具体性变差顶多是浪费算力，
幻觉是直接往对比学习的**正样本**里掺假信息。两道门脚本会自动判：
幻觉率 >10% → 不能直接入库；`worse` 多于 `better` → 重写是净损失。

> **填一半就跑统计完全可以**，未填的行自动跳过。

### 非要用 Gradio 的话

装到**独立 venv**（你已经建了 `~/gradio_env`，对的，别进 `audit_env`），
然后二选一绕开权限：① `python gradio_video_text_review.py --help` 看有没有
`--result-dir` 之类的参数，指到自己家目录；② 把脚本拷到 `~/review_gui/` 再跑。
**别 `chmod` 别人的目录、也别 `sudo`。** 完整条目见 `TROUBLESHOOTING.md` 首条。

**Then report back** in from-server.md：`review-score` 的输出（它会直接打印
「幻觉率 X% → 能不能入库」「更准确 vs 更差」这类判决），我据此决定 v0.2 能不能落地。

---

## 2026-07-22 (18) — B0 锁定 ✅ **MSR-VTT t2v R@1 = 42.50,与论文 42.5 分毫不差**

四个基准全部跑完。**唯一有外部数字可对照的 MSR-VTT 打出 42.50,论文 ≈42.5** ——
整条评测链路(权重、解码、划分、prompt)验证通过,另外三个基准的数字现在可信了。

| 基准 | 指标 | DSL | raw | 对照 |
|---|---|---|---|---|
| `msrvtt` | **t2v R@1** | **42.50** | 40.80 | 论文 ≈42.5 ✅ |
| `msrvtt` | v2t R@1 | 43.30 | 42.50 | — |
| `ucf101` | **Top-1** | **80.97** | 80.94 | Tiny 75.44 |
| `hmdb51` | **Top-1** | **53.47** | 53.87 | Tiny 48.76 |
| `k400` | **Top-1** | **63.13** | 63.49 | Tiny 56.25 |

### ⚠️ 但你那份 BASELINE.md 有个**读数陷阱**,会把分类数字读高 20 个点

**分类任务的 Top-1 是 `txt_r1`,不是 `img_r1`。** `README_zh.md` §6.2–6.4 有四处写成
"`img_r1` = Top-1",**是错的,我已更正**。

- `txt_r*` = Images→Text:逐**视频**给类别文本排序,分母 = 视频数 ⇒ **这才是零样本 Top-1**
- `img_r*` = Text→Images:逐**类别**给视频排序,分母 = **类别数** ⇒ 问的是"这个类至少有一个
  视频排第一吗",**不是准确率**

证据(都在代码里,不是猜):`itm_eval()` 两个方向分母不同(`retrieval_utils.py:463`);
`ret_dataset.py:295-296` 里 `img2txt` 把视频映到类标签、`txt2img` 长度 = 类别数;
`retrieval_utils.py:125` 把 29 个 prompt 集成掉,所以打分矩阵是 视频 × 类别。

**最直观的自检**:6 个 `img_r1` 全部恰好是 1/类别数 的整数倍 ——
`90.10=91/101`、`60.78=31/51`、`83.00=332/400`,连 README_zh 里 Tiny 的
`83.17=84/101`、`62.75=32/51`、`80.75=323/400` 也一样。视频数根本除不出这些值。

**后果有多大**:K400 按 `img_r1` 读是 **83.00**,真实零样本 Top-1 是 **63.13**,差 20 个点 ——
而且 83% 的零样本 K400 会离谱地超过 SOTA,这本身就该起疑。另一个反证:按 `img_r1` 读,
HMDB51 的 Small(60.78)会**低于** Tiny(62.75);按 `txt_r1` 读则 53.47 > 48.76,方向才对。

**顺带两条**:① 日志里 `number of classes: 11600` 打印的其实是 类别数×prompt 数(400×29),
不是类别数;② 我在块 (12) 里把试点的 t2v/v2t 标签写反了(应为 r0 **83.8 / 83.3**、
r1 **77.2 / 71.9**),已更正 —— 结论不受影响,差距仍是 6.5 个 r_mean 点。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
sed -n '/### 3.4 B0 基线/,/^### /p' experiments/README.md    # 锁定的基线表 + 读数说明
```

B0 到此完成。**下一步只剩一件事能在这台机器上做:§2b 的 r1a/r1b 归因拆包**
(用那 1 万条,数据已在),把 r1 落后的 6.5 点拆到 sigmoid / 注意力池化 头上:

```bash
export MVC_DATA_DIR=/home/vca1/datasets      # 注意:试点语料要 export,和 B0 相反
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.vision_encoder.temporal_pooling mean \
    output_dir experiments/runs/pilot_10k_r1a
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.use_sigmoid_loss False \
    output_dir experiments/runs/pilot_10k_r1b
```

各约 40 min。跑完把两份 `train.log` 的末轮 `ret_test/` 贴回来。

**§4 全量预训练仍然卡在数据**(InternVid 视频要逐条抓 YouTube,见块 (14)),那是独立的一步。

**Then report back**:上面 r1a/r1b 的结果;或者你决定先解决 §4 的数据获取。

---

## 2026-07-22 (17) — 四基准全绿 ✅ **B0 现在就能跑,零操作**;附 §4 的一条线索

预检确认:`unset MVC_DATA_DIR` 后 msrvtt / k400 / ucf101 / hmdb51 **全部 `[OK ]`**,
连那条软链都不用建 —— `data/MSRVTT_Videos` 本来就在(它是**软链**,所以上一轮
`find` 没跟进去,才没出现在聚合结果里)。**没有任何前置操作了,直接跑:**

```bash
cd ~/mobileclip && unset MVC_DATA_DIR
CKPT=./checkpoints/mobileviclip_small.pt
for b in msrvtt hmdb51 ucf101 k400; do
  CUDA_VISIBLE_DEVICES=0 \
    bash experiments/run_eval.sh small $b $CKPT experiments/evals/baseline_official/$b
done
python experiments/collect_results.py --dirs experiments/evals \
       --baseline baseline_official --out experiments/BASELINE.md
```

> `nvidia-smi` 先看哪张卡空着再定 `CUDA_VISIBLE_DEVICES`(卡是共享的)。
> **先盯 MSR-VTT 的 `ret_test/` t2v R@1 是否 ≈42.5** —— 唯一有外部数字可对照的,
> 它对上了,另外三个的数字才可信。K400 有 1 万条视频,是四个里最慢的。

### 你那份输出里的两条附带信息

**① §4 线索:shard 1 的抓取开过头,没抓完。** 11 个分片的报错**不一样**:

```
internvid_v1   anno 5000 条,抽查 20 条视频全部缺失(检查 MVC_DATA_DIR / 目录层级)
internvid_v2   anno 5000 条,但 data_root 缺失: data/video_sample/InternVId-FLT_2
```

v2–v11 是**目录都不存在**(没开始过);v1 是**目录存在但视频一条都不在** ——
有人跑过 `tools/download_internvid.py`(它会 `mkdir -p` 出 `--out`)但没抓成,
或者抓下来的文件名和 anno 对不上。查一下,别重复劳动:

```bash
ls data/video_sample/InternVId-FLT_1 | wc -l
tail -20 data/video_sample/InternVId-FLT_1/_download_log.tsv 2>/dev/null
python -c "import json;print([e['video'] for e in json.load(open('anno_downstream/InternVid/InternVid-10M-flt_1.json'))[:3]])"
```

大量 `fail` = YouTube 永久损耗;文件在但命名不同 = 只是对不齐,**不必重抓**。

**② 最后那行提示是误导,我已经改掉了。** 它写"pilot anno 未生成的话,先跑
`make_pilot_anno.py`" —— 但你的 pilot anno 好好的(9000 条读入正常),缺的是
**视频目录**,因为这一遍没 export `MVC_DATA_DIR`。照着那句去重建 anno 就白干了。
现在它会分清死因:anno 真缺才劝重建,否则直接告诉你换个根再跑一遍。
同时"已解锁的步骤"标题会带上本次用的根,提醒你这份结论只对这个根成立。

**Then report back**:`BASELINE.md` 的内容,尤其 MSR-VTT 的
`ret_test/` 与 `ret_test_emb/` 两行。

---

## 2026-07-22 (16) — 数据**全都在** ✅ B0 四个基准零下载,只差一条软链

搜索结果很干脆:**四个基准的视频一个都没丢**,全在 `~/mobileclip/data/` 下。
我之前让你下的 15 GB **完全不用下**,那个说法是错的(错因就是预检带着
`MVC_DATA_DIR=/home/vca1/datasets` 跑,而数据在仓库的 `data/`)。

先说个读数陷阱:你那条 `find ~ /home/vca1 ...` 里 `~` = `/home/vca1/rby`,
**正是 `/home/vca1` 的子目录**,同一棵树被遍历了两遍 —— 所以第 3 步每条命中
打印两次、第 4 步的计数**全部翻倍**。折半后:

| 基准 | 盘上目录 | 实际数量 | 与 config |
|---|---|---|---|
| `hmdb51` | `data/hmdb51/videos_test` | 6754 | ✅ 完全一致 |
| `ucf101` | `data/ucf101/videos` | 13320 | ✅ 完全一致 |
| `k400` | `data/kinetics_400_val_10s_320p` | 10119 | ✅ 完全一致(anno 10117) |
| `msrvtt` | `data/MSRVTT/videos/all` | 10004 | ⚠️ 只差软链名 `MSRVTT_Videos` |

`hmdb51_hf/hmdb51.zip` 也在,字节数 `2117581749`,与脚本 `EXPECTED_SIZE` 分毫不差。
另有全量 `data/kinetics-dataset/k400/{train,val,test}`(train 约 24 万条)。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
unset MVC_DATA_DIR                    # 关键:评测数据在仓库 data/,不是 /home/vca1/datasets

# 唯一要做的事:给 MSR-VTT 补一条软链
ls data/MSRVTT/videos/all | head -3   # 确认是 videoXXXX.mp4 这种裸文件名
ls -ld data/MSRVTT_Videos 2>/dev/null || \
  ln -s "$(pwd)/data/MSRVTT/videos/all" "$(pwd)/data/MSRVTT_Videos"

python experiments/check_readiness.py --sample 50 --decode    # 四行应全部 [OK ]
```

四行都亮了就直接跑 **B0 四基准**(零下载):

```bash
CKPT=./checkpoints/mobileviclip_small.pt
for b in msrvtt hmdb51 ucf101 k400; do
  CUDA_VISIBLE_DEVICES=0 \
    bash experiments/run_eval.sh small $b $CKPT experiments/evals/baseline_official/$b
done
python experiments/collect_results.py --dirs experiments/evals \
       --baseline baseline_official --out experiments/BASELINE.md
```

先看 **MSR-VTT 的 `ret_test/` t2v R@1 是否≈42.5** —— 这是唯一有外部数字可对照的,
对上了才说明整条评测链路(权重、解码、划分)是通的,另外三个的数字才可信。

**两个提醒:**

1. 盘上有 `data/MSRVTT/MSRVTT/videos/all`(同样 10004 个),疑似解压嵌套多出的
   副本。**先别删** —— 等软链指向的那份确认能解码,再回收空间。
2. `/home/vca1/datasets` **没出现**在按目录聚合的结果里,但预检确认那 1 万条试点
   视频存在 —— 多半因为它是**软链**而 `find` 默认不跟随(要 `find -L`)。
   §2b 的数据没问题,别被这个吓到。

**Then report back**:`check_readiness.py` 的四行,以及 `BASELINE.md` 的内容
(尤其 MSR-VTT 那两行 `ret_test/` 与 `ret_test_emb/`)。

---

## 2026-07-22 (15) — HMDB51 **多半没丢**:是预检看错了根(这坑是我埋的)

`tools/download_hmdb51.sh` 写死的是**仓库相对路径**:

```
RAW_DIR="data/hmdb51_hf"            ->  ~/mobileclip/data/hmdb51_hf
DEST_DIR="data/hmdb51/videos_test"  ->  ~/mobileclip/data/hmdb51/videos_test
```

而你跑预检时按我 §3.0 写的 `export MVC_DATA_DIR=/home/vca1/datasets`,它就去
`/home/vca1/datasets/hmdb51/videos_test` 找了 —— **那儿从来没放过评测数据**。
**这台机器上两批数据不在一个根下**:评测集在仓库 `data/`,试点语料在
`/home/vca1/datasets`。我上一轮 §3.0 只给了 export 那一种跑法,是我的错,已修成跑两遍。

**旁证(挺硬的):`hmdb51_act_test.json` 有 1511 条,而 split-1 测试集应是
51×30 = 1530 条。** 少的 19 条正是 `build_act_rec_anno.py` 的 `--videos`
磁盘过滤丢掉的("Keep only entries whose video exists in videos_dir")。
**能过滤出 1511 条,说明生成 anno 的那一刻视频就在盘上。**

**Do this next(按顺序,大概率第 1 步就完事):**

```bash
cd ~/mobileclip && git pull origin main

# 1) 换个根再看一眼 —— 评测集不要 export MVC_DATA_DIR
unset MVC_DATA_DIR
ls -ld data/hmdb51/videos_test data/hmdb51_hf 2>/dev/null
find data/hmdb51/videos_test -maxdepth 1 -name '*.avi' 2>/dev/null | wc -l   # 期望 ~6766
python experiments/check_readiness.py            # hmdb51 应该就亮 [OK ] 了
```

如果目录在、但文件数是 0 或校验仍报缺失,是**第 2 种情况:软链悬空**
(`videos_test/` 里全是指向 `data/hmdb51_hf/` 的软链;若为回收 2.1 GB 把
`hmdb51_hf/` 删了就全断):

```bash
# 2) 悬空检查 + 就地修复(不用重下,只要 hmdb51_hf 里的 .avi 还在)
find data/hmdb51/videos_test -maxdepth 1 -xtype l | wc -l        # >0 = 悬空
find data/hmdb51/videos_test -maxdepth 1 -xtype l -delete
find "$(cd data/hmdb51_hf && pwd)" -name '*.avi' -exec ln -sf {} data/hmdb51/videos_test/ \;
```

两步都没找到,再全盘搜(拿 anno 里的**真实文件名**做外科式搜索,比翻目录快):

```bash
# 3) 外科式搜索:自动从 anno 取真实文件名再全盘找(不用手动粘贴)
NAME=$(python -c "import json;print(json.load(open('anno_downstream/hmdb51_act_test.json'))[0]['video'])")
echo "找这个文件:$NAME"
find ~ /home/vca1 -name "$NAME" 2>/dev/null

# 4) 或者按目录聚合看视频都堆在哪
for d in ~ /home/vca1 /data /mnt; do
  [ -d "$d" ] && find "$d" \( -name '*.avi' -o -name '*.mp4' \) -printf '%h\n' 2>/dev/null
done | sort | uniq -c | sort -rn | head -20

# 5) 原始 zip 还在的话不用重下(脚本会校验字节数后跳过下载)
find ~ /home/vca1 -name 'hmdb51*.zip' -size +1G -printf '%10s  %p\n' 2>/dev/null
# 期望 2117581749 字节;放回 data/hmdb51_hf/hmdb51.zip 再跑 bash tools/download_hmdb51.sh
```

**Then report back**:第 1 步那三条命令的输出(`ls -ld` + `.avi` 计数 +
不带 MVC_DATA_DIR 的 `check_readiness.py`)。若已亮 [OK ],直接跑 hmdb51 的 B0:

```bash
CKPT=./checkpoints/mobileviclip_small.pt
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_eval.sh small hmdb51 $CKPT experiments/evals/baseline_official/hmdb51
```

> 同理,MSR-VTT / UCF101 下载前也先用**不带 `MVC_DATA_DIR`** 的预检确认一遍 ——
> 说不定也早就下过了,别白下 13 GB。

---

## 2026-07-22 (14) — 预检坐实了诊断,但多出两条:**B0 能做三个基准**;**§4 我上轮说错了**

预检输出确认了 (13) 的判断,同时给出两条新信息 —— 一条把 B0 做大,一条推翻我自己。

### ① B0 不止 MSR-VTT:**四个基准的 anno 都在盘上**,三个视频是公开源

我上轮只按 §3 那个 5 基准循环去看,漏了 `ucf101` / `hmdb51` —— 预检显示它们
anno 齐(3783 / 1511 条),而且下载量比 K400 小得多:

| 基准 | 任务 | anno | 下载量 | 怎么拿 |
|---|---|---|---|---|
| `hmdb51` | 分类 | ✅ 1511 | **~2.1 GB** | `bash tools/download_hmdb51.sh` 一条命令,幂等 |
| `msrvtt` | 检索 | ✅ 1000 | ~6.7 GB | 公开 wget(主指标) |
| `ucf101` | 分类 | ✅ 3783 | ~6.5 GB | CRCV 官网整包 |
| `k400` | 分类 | ✅ 10117 | 很大 | 可选,源不稳,不做 |

⇒ **B0 = msrvtt + hmdb51 + ucf101**(约 15 GB,全公开源,零授权),检索+分类都覆盖。
**先跑 HMDB51** —— 2.1 GB 一条命令,是验证整条评测链路最便宜的一次。

⚠️ 但只有 **MSR-VTT 有外部数字可对照**(t2v R@1≈42.5)。HMDB51/UCF101 是锁我们
自己的基线,接线错了不会被外部数字兜住 —— 所以先让 MSR-VTT 对上,再信另外两个。

### ② 更正:§4 不是"换台机器"能解决的,我上轮写错了

我在 (13) 里写"§4 需要先解决 InternVid **全量数据**的落盘位置(H100 机或换机)"。
**两处都不准**,查了 `docs/internvid-download-zh.md` 才发现:

1. **盘上那 11 个 anno 不是全量**,是本仓库自己生成的子集 —— 每片恰好 5000 条、
   合计 55000 条。生成脚本里明写着 `LIMIT = 5000  # 先做子集;要全量把这行设为 None`
   和 `PER_SHARD = 5000`。"全量 10M"从来没进过这份 anno。
2. **不存在"别处有一份视频、搬过来就行"**。InternVid 只发 URL 列表,视频要
   **逐条从 YouTube 抓**(`tools/download_internvid.py`),而且**必然有永久损耗**
   —— `Private video` / `Video unavailable` 的条目是真没了。

正确说法:**§4 是一个数据获取任务,不是机器搬迁任务。** 换 H100 也一样要从头抓,
只在带宽/并发/磁盘上占优。开跑前要先定三件事:抓多少条(现有 anno 是 55k,要更多
得重跑转换脚本)、在哪抓、能接受多大损耗率。**建议单独排期,别混进 §4 的训练排期。**

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
unset MVC_DATA_DIR          # B0 全程不要 export,让视频落回仓库默认的 data/
df -h .                     # 三个合计 ~15 GB,解压时并存,留 >25 GB 更稳

# 最便宜的一步:HMDB51 2.1 GB,一条命令,先把评测链路跑通
bash tools/download_hmdb51.sh
python experiments/check_readiness.py --sample 50 --decode   # 期望 hmdb51 [OK ]

CKPT=./checkpoints/mobileviclip_small.pt
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_eval.sh small hmdb51 $CKPT experiments/evals/baseline_official/hmdb51
```

这一步跑通了,再下 MSR-VTT(~6.7 GB)和 UCF101(~6.5 GB),三个一起汇总 ——
完整命令见 `experiments/README.md` §3.2 / §3.3。

**另外:把盘上那份 `experiments/BASELINE.md` 删掉。** 它是 (13) 说的那个陷阱产物,
内容是 10k 试点自检索数字、文件名却叫 BASELINE。重新生成时脚本会在顶部写明出处。

**Then report back**:`bash tools/download_hmdb51.sh` 之后的
`check_readiness.py` 输出,以及 hmdb51 的 `eval_res.json`。

---

## 2026-07-22 (13) — §3 B0 全崩 ❌ 不是配置问题:**本机没有下游评测数据**

5 个评测一个数字都没出。**别再重跑一遍** —— 重跑还是全崩。报错分两类:

- **A 类 anno 就缺**(`didemo` / `anet` / `ssv2_mc`):秒崩在
  `FileNotFoundError: anno_downstream/didemo_ret_test.json`。
- **B 类 anno 有、视频没有**(`msrvtt` / `k400`):走得很远 —— 配置打印、标注读入
  (`1000/1000`)、权重载入、**文本特征都算完了**,才崩在第一批视频:
  `ERROR opening: /home/vca1/datasets/MSRVTT_Videos/video7020.mp4`。
  B 类特别容易误判成偶发解码错误,因为 `data_root` 只在真正读视频那一刻才被触碰。

**根因**:`tools/data_audit/SETUP_V100.md` 开篇就写了 —— 这台机器
**只有 `/home/vca1/datasets/internvid_zh_10k` 那 1 万条抽样,没有全量数据**。
MSR-VTT / K400 / DiDeMo / ANet / SSv2 的视频一个都不在。

**连带影响(更要紧):§4 的 R0→R1 全量 A/B 在这台机器上同样跑不了。**
`internvid_v1..v11` 的 `data_root` 是 `$MVC_DATA_DIR/video_sample/InternVId-FLT_{i}`,
同一个"没有全量数据"的结论对它一样成立。所以 101vector 上 §2b 之后**还能做的只有两件**:
① r1a/r1b 归因拆包(数据已在);② 补完 MSR-VTT 视频后的 B0 单基准。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

# 1) 先跑预检(10 秒)。仓库里一直有这个脚本,这次就是省掉它换来 5 次崩溃
export MVC_DATA_DIR=/home/vca1/datasets
python experiments/check_readiness.py
```

把这份输出贴回来 —— 它会一次性说清哪些 anno/视频在、哪些步骤已解锁,
我据此定 §4 到底在哪台机器上跑。

```bash
# 2) 想继续 B0 的话:MSR-VTT 是唯一近在咫尺的(anno 已在,只差视频,公开 wget ~6.7GB)
#    注意:跑 MSR-VTT 时不要 export MVC_DATA_DIR,让它落回仓库默认的 data/
cd ~/mobileclip && unset MVC_DATA_DIR
df -h .                     # 先确认 >8 GB 余量
mkdir -p data && cd data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip -q MSRVTT.zip && ln -s "$(pwd)/MSRVTT/videos/all" "$(pwd)/MSRVTT_Videos"
cd .. && ls data/MSRVTT_Videos/ | wc -l        # 期望 10000
python experiments/check_readiness.py --sample 50 --decode   # 期望 [OK ] msrvtt

CKPT=./checkpoints/mobileviclip_small.pt
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/run_eval.sh small msrvtt $CKPT experiments/evals/baseline_official/msrvtt
python experiments/collect_results.py --dirs experiments/evals \
       --baseline baseline_official --out experiments/BASELINE.md
```

`didemo` / `anet` / `ssv2_mc` **近期不做**(anno 在受限 Drive + 视频要大量下载),
`k400` 可选。B0 收窄成 MSR-VTT 单基准,足够支撑 §6 的判据(主指标本来就是它)。

**两个新发现,已写进文档:**

1. **`BASELINE.md` 会骗人。** 那次 5 个评测全崩、`evals/` 下一个结果都没有,
   `collect_results.py` 照样从 `runs/` 捞到试点数字,写出一个名叫 **BASELINE.md**、
   内容是 **10k 试点自检索** r_mean 93.37 / 86.83 的文件,退出码 0、零警告。
   那**不是** MSR-VTT 基线(基线 t2v R@1≈42.5)。已加两条防线:`--dirs experiments/evals`,
   以及脚本现在会把「扫了哪些目录、含哪些 run」写进产物顶部、基线缺失时显式报警。
   **你现在盘上那份 BASELINE.md 是误导性的,建议直接删掉重生成。**
2. **官方 `mobileviclip_small.pt` 不含文本塔主体**(从加载日志读出):`missing_keys`
   恰好是整个 `text_encoder.{embedding_layer, positional_embedding, transformer.*,
   final_layer_norm}`,而 `text_encoder.projection_layer` **在**权重里,视觉塔和
   `temp` 也都在。文本塔是建模时从 `mobileclip_s2.pt` 载入的(上游 `freeze_text=True`
   只训投影,自洽)。**含义:B0 的数字由这两个权重共同决定**,复现时
   `mobileclip_s2.pt` 也必须是对的那一份。

**Then report back**:`check_readiness.py` 的完整输出;以及若下了 MSR-VTT,
`experiments/evals/baseline_official/msrvtt/eval_res.json` 的内容。

---

## 2026-07-22 (12) — §2b 完成 ✅ 试点跑通了;r1 落后 6.5 点但**不能**据此否定组件

**两臂都跑完,全程零 `OVERFLOW`**(r0 38:10 / r1 37:46,和我估的 ≈40 min 对上)。
判据两条都过:

- **损失曲线**:r0 `1.014 → 0.272`、r1 `7.093 → 4.165`,都单调下降后走平,不发散。
  ⚠️ **两臂 loss 不可互比** —— r0 是 softmax InfoNCE,r1 是 sigmoid,量纲不同。
- **组件存活**:不用开 ckpt,`train.log` 里就能读到 ——
  `diff_names: ['logit_scale','logit_bias'], diff_lr: 0.001` 加
  `optimizer -- lr=0.001 wd=0.2 len(p)=2`(different_lr 生效);
  `vision_encoder.temporal_pool.{q,k,v,proj}` 在优化器参数表里(注意力池化已建起);
  `train_file` 带 `caption_idx: True`。

### r1 的检索明显低:93.37 vs 86.83(r_mean)。**先别下结论。**

末轮 t2v/v2t R@1:r0 **83.8 / 83.3**,r1 **77.2 / 71.9**,两个口径同向。差距 6.5 点
远超 ~3.5pp 噪声带 —— 但**这正是上一轮就写明「这个评测判不了组件优劣」的场景**,
理由没变(开训前已近饱和 + BN running stats 混淆),而且:
r1 是**四个改动的捆绑包**,试点无法归因;r1 从 ep2 起还在**单调回升**
(85.07 → 86.83)、末轮仍在涨,**根本没收敛**,而 r0 早在 93.3 走平。

**我先前猜「注意力池化是随机初始化所以拖后腿」—— 查了源码,这个猜测是错的。**
`TemporalAttentionPool` 的 `proj` 是**零初始化**(`models/mobileviclip_small.py:1351`),
`forward` 返回 `mean + proj(out)`,初始时刻**精确等于 mean 池化**。
更像是 **sigmoid 的 SigLIP 默认初值对「已经对齐好的模型」失配**:
`t=10, b=−10`,正样本 sim≈0.5 ⇒ `t·sim+b≈−5` ⇒ `sigmoid≈0.007`,
等于开局宣布「所有正样本都判错」(首步 loss 7.4531 正好对上),
于是对预训练好的嵌入空间来了一记大推力,然后再慢慢标定回来。**这是假说,没验证。**

### 另外两处更正 / 新发现

1. **我上一轮说「`temp` 被 clamp 钉死」是错的。** 真跑里 r0 的 temp **往上走**:
   `0.0100 → 0.0101 → 0.0102 → 0.0103`,然后随 lr 衰减走平。clamp 是下界,梯度
   根本没往下顶。所以「冻结 temp 近似空操作」**收回** —— 冻结会改变 r0。
   (r1 全程 0.0100 是另一回事:sigmoid 分支传的是 `temp=1.0`,`self.temp`
   **完全没有梯度**,是个没接线的参数。)
2. **`no_weight_decay()` 是死代码。** `utils/optimizer.py:99-107` 把
   `no_decay = model.no_weight_decay()` **注释掉了**、只留 `pass`;而
   `temp`/`logit_scale`/`logit_bias` 是 **0 维**标量(`torch.Size([])`,长度 0),
   也躲过了 `len(shape)==1` 的 bias/BN 兜底。结果三者都吃到 **wd 0.2**。
   量级:`logit_scale`/`logit_bias` 在 lr=1e-3 下 170 步被拉向 0 约 **3.3%**。
   二阶效应,但「声明了 no-decay 却被静默忽略」正是会悄悄污染消融的那类缺陷。
   上游代码,**我没改**。

### Do this next

试点到此为止,别再在 10k 上追组件结论。下一步是 §3 的 B0(需要 MSR-VTT 到位):

```bash
cd ~/mobileclip && git pull origin main
ls -la anno_downstream/msrvtt_1k_test.json checkpoints/mobileviclip_small.pt   # 到位了吗
```

**两个都在** → 走 §3;**缺** → 先补下载(见 `CHECKPOINTS.md`)。

真要在试点上再挖一手,唯一值得跑的是**拆包归因**(各 ≈40 min),验证上面那个假说:

```bash
export MVC_DATA_DIR=/home/vca1/datasets
# 关掉注意力池化(只剩 sigmoid + different_lr)
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.vision_encoder.temporal_pooling mean \
    output_dir experiments/runs/pilot_10k_r1a
# 关掉 sigmoid(只剩注意力池化 + different_lr)
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8 \
    model.use_sigmoid_loss False \
    output_dir experiments/runs/pilot_10k_r1b
```

> 这两条是拿 r1 配置**关掉**其中一项,`different_lr` 仍开着,不是严格单因子;
> 严格拆分要用 §4 的 `r1a_sigmoid.py` / `r1b_attnpool.py`。
> `merge_list` 只认**已存在**的键 —— `output_dir` 存在,可以这样覆盖。
> 若假说成立:r1a(仍带 sigmoid)应该复现这次的掉点,r1b(无 sigmoid)不该掉。

**⚠️ 跑 §4 之前务必先做 `experiments/README.md` §4 顶部标的两项预检**
(训练中评测被静默 Skip、`ckpt_best.pth` 不存在),否则几小时白烧。

---

## 2026-07-22 (11) — 冒烟通过 ✅ 可以跑真 A/B 了(但评测已近饱和,判据要改)

**链路全通了。** 8 卡起来、数据读到、前向反向都过、epoch 末评测出表、下一个 epoch
接着跑。`initial_scale_power 5` 全程**零** `OVERFLOW`,anomaly detection 按预期
被跳过(日志里有那行中文说明)。这是试点第一次真正跑出损失。

**但先别把 1.3846 → 0.9709 读成「在学习」。** 那两个 epoch 的
`Averaged stats` 差异**几乎肯定是批次差异,不是训练效果**,理由有三条:

1. `debug` 每个 epoch 只跑 **5 步**,lr 还在 warmup(1e-6 → 1e-5)。Adam 的单步
   位移约等于 lr,5 步累计把权重挪动 ~5e-6 的相对量级 —— 对 35M 参数的模型
   等于没动。
2. 日志里每行 loss 是**原始单步值**(`pretrain.py:55` 的 `fmt="{value:.4f}"`,
   不是滑窗中位数)。epoch 0 单步在 [1.13, 1.66] 之间跳,epoch 1 在 [0.95, 1.18],
   两个区间是重叠的 —— n=5 的样本判不出趋势。
3. 每 epoch 只有 17 个 batch、抽 5 个,批次组成本身的方差就够解释这个差。

**真正的损失曲线要等去掉 `debug` 的那次(17 步 × 10 epoch = 170 步)。**

**先纠正我上一轮的预测。** 我说开局 loss 应该在 13–14,实际是 **1.1257**。我错了,
而且错的原因很重要:我那个数是按**随机初始化**的归一化嵌入算的,但两个塔都从
`mobileclip_s2.pt` 载入、**开训前就已经对齐**。证据就在你这份日志里 —— 第 0 个
epoch(只跑 5 步、lr≈2e-6,等于没训)held-out 检索已经是 **R@1 79.6 / 83.0**。
所以 `|dL/dtemp|` 远小于我估的 1.2e3,`scale=54 上限`是随机初始化的最坏情况,
不是这个模型的工作点。已在 `TROUBLESHOOTING.md` 和 base config 注释里更正。

**顺带查清了 `temperature: 0.0100` 为什么一步不动:** `temp` 和 `temp_min` 都是
1/100,而 `clip_contrastive_temperature()` 每次 forward 前都 `clamp_(min=temp_min)`。
对比学习想让温度更低(更锐),每步都被夹回去 —— 所以 `temp` **事实上已经是冻结的**,
却还在贡献完整的 `1/temp²` 放大梯度。付了 fp16 的代价,什么也没买到。这让上一轮
留给你的那个决定从「权衡」变成了「一边倒」:冻结 `temp` 在语义上几乎是空操作。
仍然归你拍板,但不着急 —— 真跑没有 OVERFLOW 就先不用动。

### ⚠️ 一个会影响结论的发现:这个 held-out 评测判不了组件优劣

开训前就 79.6 / 83.0,天花板 100,只剩 ~20 个点;而 1000 条 query 的二项标准误
就有 1.3pp,两次运行差 <3.5pp 全在噪声里。**再加一个混淆项**:视觉塔没冻结,
MobileOne 主干全是 BatchNorm,训练期 running stats 会被这批视频重估一遍,而评测
用的正是 running stats —— R@1 变化可能纯粹来自 BN 统计量适应,不是学到了东西。
两臂同样受影响,A/B 仍然公平,但**「R@1 涨了」不等于「这个组件有用」**。

所以试点判据收窄成两条(已写进 `experiments/README.md` §2b):
**(a) 损失曲线形态;(b) 组件存活检查**(`logit_scale`/`logit_bias` 离开
exp(t')=10、b=−10;`temporal_pool.proj` 权重非零)。R@1 只当参考。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main
nvidia-smi          # 先确认 8 张卡是空的:峰值 res mem 27.2 GiB,余量不多

export MVC_DATA_DIR=/home/vca1/datasets

# 真 A/B 一对(等步数 ~170 步)。不要加 debug,不要传 initial_scale_power ——
# 两个配置继承同一个 base(默认 8),不覆盖才是对称的。单臂 ≈40 min。
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r1.py 8
```

**Then report back** in from-server.md:

```bash
# 1) 两条损失曲线(每个 epoch 的 Averaged stats 就够)
grep "Averaged stats" experiments/runs/pilot_10k_r0/train.log
grep "Averaged stats" experiments/runs/pilot_10k_r1/train.log

# 2) 跳步对称性 —— 两个数字差得多就说明不是等步数对照
grep -c OVERFLOW experiments/runs/pilot_10k_r0/train.log
grep -c OVERFLOW experiments/runs/pilot_10k_r1/train.log

# 3) r1 的组件存活检查(这才是试点真正要回答的)
#    注意:pilot 配置 stop_key=None ⇒ **不会**生成 ckpt_best.pth;
#    DeepSpeed 存的是**目录** experiments/runs/pilot_10k_r1/ckpt_09.pth/,
#    真正的权重在里面的 mp_rank_00_model_states.pt(和 tasks/pretrain.py:521 同款)。
python - <<'PY'
import glob, torch
d = sorted(glob.glob("experiments/runs/pilot_10k_r1/ckpt_*.pth"))[-1]
print("读取:", d)
sd = torch.load(f"{d}/mp_rank_00_model_states.pt", map_location="cpu")["module"]
for k in ("logit_scale", "logit_bias"):
    if k in sd:
        print(k, "=", sd[k].float().item())
print("  ^ 初值 logit_scale=ln10≈2.3026, logit_bias=-10;没动=different_lr 没生效")
pk = [k for k in sd if "temporal_pool" in k]
print("temporal_pool 参数数:", len(pk), "(0 = 注意力池化没建起来)")
for k in pk[:4]:
    print("  ", k, "absmax =", sd[k].float().abs().max().item())
PY
```

以及最后一个 epoch 的评测表(两臂各一份)。
## 2026-07-22 (10) — 裁判（中转机）：清掉重复进程和重复行，单进程跑完 500

> 这一条只关 **S2 裁判**（中转机 `10.55.87.68`），与上面第 (9) 条的训练崩溃无关，可并行。

**现状：链路已全通，卡的是并发不是端点。** 已实测通过的：端点连通
（`172.30.85.198:8000` / `172.30.62.177:8000` 返回 200，4 卡那个不通）、
conda base 环境零安装（用 `cv2` 后端）、`CLIPS` 指向 b100 上的 10k 子集、
`enable_thinking=False` 生效、冒烟 5/5 无失败。

**踩到两个坑，都已修：**

1. **两个进程在写同一个文件** —— 500 条的输入却有 644 行。
   信号是 `nohup … &` 返回 `[2]`（说明 `[1]` 还在跑）。
2. **看起来卡在 `[progress] 200/476`** —— 不是端点慢，是 `vlm_audit.py` 的两个缺陷：
   `pool.map` 按提交顺序产出，一条慢请求把后面所有已完成结果堵住不写盘
   （`--timeout 600` + 重试 ≈ 堵 20 分钟）；进度又写死每 200 条才打一次。
   **已改成 `as_completed` + `--progress-every 25`**（实测首条落盘从 3.05s → 0.16s）。

**Do this next**（逐条，注意标注的机器）：

```bash
# ① [中转] 只留一个进程
jobs -l; pgrep -af vlm_audit.py
pkill -f vlm_audit.py

# ② [101] 送新版脚本（上面两处修复都在里面，旧版跑还会再卡一次）
cd ~/mobileclip && git pull origin main
scp -P 31179 tools/data_audit/vlm_audit.py root@10.55.87.68:~/judge/
```

```bash
# ③ [中转] 去重：按 video 唯一，成功的判定覆盖失败的
cd ~/judge
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open("s2_verdicts_judge.jsonl")]
best={}
for r in rows:
    v=r.get("video")
    if v is None: continue
    ver=r.get("verdict") or {}
    ok = "action" in ver and "error" not in ver
    if v not in best or (ok and not best[v][1]):
        best[v]=(r, ok)
with open("s2_verdicts_judge.dedup.jsonl","w",encoding="utf-8") as f:
    for r,_ in best.values(): f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"原 {len(rows)} 行 -> 去重后 {len(best)} 条唯一 video")
PY
mv s2_verdicts_judge.jsonl s2_verdicts_judge.raw.bak
mv s2_verdicts_judge.dedup.jsonl s2_verdicts_judge.jsonl
```

```bash
# ④ [中转] 单进程跑完剩下的（已完成的按 video 自动跳过）
export CLIPS=/workspace/mount/b100_zaip_data/zyc/vibe_coding/low_resource_cv/MobileViCLIP/outputs/internvid_alignment/caption_translation_qwen36_v0_1/20_outputs/internvid_zh_10k
export JUDGE_EP=http://172.30.85.198:8000/v1
pgrep -af vlm_audit.py || echo "确认无残留进程，可以开跑"
nohup python vlm_audit.py \
  --input s2_judge_sample.jsonl --video-root "$CLIPS" \
  --api-base "$JUDGE_EP" --api-key EMPTY \
  --model /workspace/mount/b100_zaip_data/model_zoo/Qwen3.6-27B \
  --frames 8 --concurrency 8 --frame-backend opencv \
  --output s2_verdicts_judge.jsonl > judge.log 2>&1 &
watch -n 30 'wc -l ~/judge/s2_verdicts_judge.jsonl; tail -3 ~/judge/judge.log'
# 判断是否真卡住：看 wc -l 一分钟内涨没涨，别看进度行
```

```bash
# ⑤ [中转] 跑完先自查（这决定裁判有没有价值），再回传
python3 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("s2_verdicts_judge.jsonl")]
v=[r["verdict"] for r in rows if "action" in (r.get("verdict") or {})]
def same(a,b): return sum(1 for x in v if x.get(a)==x.get(b))/max(len(v),1)
print(f"n={len(v)}  总行数={len(rows)}（两者应相等，不等说明又有重复）")
print(f"en==trans: {same('en_score','trans_score'):.1%}   (主力 8B 92.5%)")
print(f"zh==trans: {same('zh_score','trans_score'):.1%}   (主力 8B 98.9%)")
print("action 分布:", dict(collections.Counter(x.get("action") for x in v)))
PY

# ⑥ [101] 取回并算一致率 + Cohen's κ
scp -P 31179 root@10.55.87.68:~/judge/s2_verdicts_judge.jsonl outputs/audit_10k/
bash tools/data_audit/run_v100.sh judge-agree
```

**一个已经站住的中间结论**（n=13 时）：`en==trans` **30.8%** vs 主力 **92.5%**、
`zh==trans` **30.8%** vs **98.9%**。若裁判和 8B 一样塌缩，这个结果的概率是
`4e-08` / `2e-15` —— 即**裁判确实在独立评估三个维度，halo 是 8B 这个模型的毛病，
不是 rubric 的毛病**。它的 `trans_score` 因此可以拿来裁决决策 1。

**一个预测**：action 分布已透出 T4 会怎么走 —— 主力最大桶是 `retranslate`（41.2%），
裁判至今一条没给、全判 `recaption`。**五类一致率大概率过不了 75%**，
但二值口径（主力 `keep+fix_text` 35.7% vs 裁判 30.8%）可能没问题。
真出现这个结果**别简单判 T4 不达标**——它说明的是主力的 `action` 不可信，
而 `apply_buckets.py` 正是按 action 分流任务队列的。

**Then report back** in `from-server.md`：⑤ 的自查输出（n / 两个同值率 / action 分布）、
错误率、以及 ⑥ `judge-agree` 打印的三个门和各维度 κ。拿到就能把 T4 和决策 1 一起结掉。

完整手册：[`tools/data_audit/SETUP_JUDGE_RELAY.md`](../../tools/data_audit/SETUP_JUDGE_RELAY.md)
（§6 跑 500、§7 回传算分、§9 常见问题含「重复写入」一节）。

---

## 2026-07-22 (9) — 崩溃是 `debug True` 自己造成的;顺带查出 temp 把 loss scale 压死

**卡空出来了,bs=64 装得下,前向也跑完了** —— 这次死在**反向**。报错位置很具体:

```
File "models/criterions.py", line 52, in get_sim
    sim_v2t = vision_proj @ text_proj.T / temp
RuntimeError: Function 'DivBackward0' returned nan values in its 1th output.
```

**「1th output」要读准**:`DivBackward0` 有两个输出,0 是对分子的梯度,**1 是对
分母的梯度** —— 也就是 `self.temp`(可学习参数,初值 0.01)。NaN 恰恰在温度的梯度上。

**这里其实是两件事,得分开说。**

**① 为什么会「崩」—— 是 `debug True` 把预期内的事件变成了致命错误。**

`pretrain.py` 原本只要 `debug` 为真就开 `torch.autograd.set_detect_anomaly(True)`。
它和 fp16 **动态 loss scaling** 根本不兼容:动态缩放的做法就是先给一个偏大的初始
系数,**故意**让前几步溢出,DeepSpeed 检出 inf/nan 后跳过该步、系数减半(就是日志里
那句 `OVERFLOW! ... Skipping step`)。而 anomaly detection 一见反向出现 nan 就立刻
抛异常 —— 赶在 DeepSpeed 检查梯度**之前**,于是自愈机制没机会生效。

**已修**:现在 deepspeed+fp16 时自动跳过 anomaly(并打日志说明);bf16 / fp32 /
不用 deepspeed 时照常开启,那些场景下它是安全且有用的。

**② 为什么会有 NaN —— 可学习的 temp 把可用 loss scale 压到了 54。**

`sim = (v @ tᵀ) / temp`,temp=0.01 让温度的梯度带 `1/temp² = 1e4` 的放大。
我在本机用 fp32 实测(B=512,即 all_gather 后的全局批,随机归一化嵌入):

| | |
|---|---|
| `\|dL/dtemp\|` | **1.2e3** |
| fp16 上限 | 65504 |
| **temp 不溢出所允许的最大 loss scale** | **≈ 54** |
| 塔内嵌入允许的最大 scale(`\|dL/dvision_proj\|` 最大 5.7e-2) | ≈ 1.15e6 |

DeepSpeed 默认 `initial_scale_power: 16`(=65536)因此要**连续减半 11 次**才可能有
一步落地;而 `debug` 模式**只跑 5 步**就 break。就算不崩,冒烟也全耗在退避上、
什么都看不到。

**已修**:`initial_scale_power` 改为从配置读取(默认降到 **8**=256),也能命令行覆盖。
loss scaling 纯粹是数值手段(更新前会反缩放回去),**不改变实验语义**,只决定开头
浪费几步。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 \
    deepspeed.initial_scale_power 5 debug True
```

(`5`=32,刚好在实测可用区 54 以内,开头基本不浪费步数。想更稳可以不带这一项,
用默认的 8,代价是多两三步退避 —— 但只有 5 步预算,所以我建议先用 5。)

**Then report back:** `Train Epoch: [0]` 之后带 `loss_vtc` / `lr` / `temperature`
的那几行。判据:
- loss_vtc 是个**有限数**(初值应在 13–14 附近,和我本机算的 13.93 同量级)就算通了;
- 开头有一两条 `OVERFLOW! Skipping step` 仍属正常;
- 若 5 步全是 OVERFLOW,把日志里的 loss scale 数值贴回来。

**还有一件要你拍板的事(我没动)。** 上面那张表说明:**光是可学习的 `temp` 就把
可用 loss scale 压低了约 2 万倍**(54 vs 1.15e6)。其它所有参数的梯度因此只能按
~32 缩放而不是 ~1e6,小梯度会更靠近 fp16 下溢。这是 Volta+fp16 在这种参数化下的
真实代价。两条标准出路:把温度改成 **log 参数化**(r1 的 `logit_scale` 已经是这么做的,
放大从 `1/temp²` 降到 `1/temp`),或者**冻结 `temp`**。两者都会改变 r0 的定义,
所以我不擅自动手 —— 等正式跑起来后看 loss scale 稳定在多少,低于 ~16 就该处理。

---

## 2026-07-22 (8) — dtype 那条线**全通了**;这次 OOM 是因为卡上还跑着 S2 审核

**先说结论:训练代码这边已经没问题了。** 三个 dtype 修复全部生效,前向真正跑进了
视觉塔的 stage-2/3 做实际计算,报错性质完全变了:`torch.OutOfMemoryError`。

**Diagnosis:别改配置,先看报错的第二、三行。**

```
GPU 4 has a total capacity of 31.73 GiB of which 404.50 MiB is free.
Process 2621830 has 16.93 GiB memory in use.          <- 这不是我们的进程
Including non-PyTorch memory, this process has 14.39 GiB memory in use.
```

把账算一下:

| | |
|---|---|
| 单卡总量 | 31.73 GiB |
| **别人的进程** | **约 16.9–17.0 GiB** |
| 我们的训练进程 | 14.36–14.52 GiB(PyTorch 已分配 12.46 GiB) |
| 死时剩余 | 268–436 MiB(它要 480 MiB) |

17.0 + 14.5 ≈ 31.5 —— **卡是被两个任务塞满的**。空卡上 bs=64 只要约 14.5 GiB,
还剩 17 GiB 富余。**训练配置本身一点问题都没有。**

**占卡的是谁:** PID **2621826–2621833,连续 8 个,一卡一个** —— 这正是
`run_v100.sh s2` 的形状(每张卡起一个 `vlm_audit_hf.py`,`CUDA_VISIBLE_DEVICES=$i`),
而 Qwen3-VL-8B 的 fp16 权重就是约 16 GB,`DATA_ISSUES.md` 里 S2 也标着「运行中」。
**是 S2 审核在跑。** 另外 GPU 0 和 GPU 6 上还挂着约 11 个 ~404 MiB 的残留进程,
值得顺手清掉,但不是这次 OOM 的原因。

**Do this next —— 两条路,建议选第一条:**

**① 让 S2 先跑完(推荐)。** 它是整条审核流水线的阻塞项(后面每一步都吃它的产物),
还剩不到 2–3 h。跑完再用空卡按记录的 bs=64 跑试点。先看进度:

```bash
nvidia-smi
bash tools/data_audit/run_v100.sh status
```

**② 想现在就把训练循环验通**,用小 batch 挤进 S2 剩下的约 14.7 GiB:

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 有 654 MiB 是已保留未分配的碎片
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 \
    batch_size 16 debug True
```

命令行覆盖 `batch_size` **确实会传到 dataloader**:`utils/config.py:get_config`
里 `merge_list(cfg, args.opts)` 在 `eval_dict_leaf(cfg)` **之前**执行,所以
`inputs.batch_size.{image,video}` 那个 `"${batch_size}"` 插值是按覆盖后的值解析的
(我查过源码顺序,不是猜的)。

> ⚠️ **但 batch size 不是自由旋钮。** 它决定批内负例数量,直接改变对比任务本身 ——
> bs=16 和 bs=64 的结果不可比。**小 batch 只能用于冒烟**;真正的 r0/r1 那一对
> 必须同批大小,优先用 64/卡(全局 512)。这条已写进 `experiments/README.md` §2b。

**Then report back:** 若走②,给我 `Train Epoch: [0]` 之后带 `loss_vtc` / `lr` /
`temperature` 的头十几行 —— 那就说明全链路通了,只等空卡跑正式 A/B。判据同前:
开头几条 `OVERFLOW! ... Skipping step` 正常,`nan` 或 loss scale 掉到底才是问题。

---

## 2026-07-22 (7) — 视觉塔通了,死在文本塔的 LayerNorm;同族的雷已一次扫干净

**又往前走了一层。** 上次死在视觉塔第一个卷积,这次那里过了,死在**文本塔的
第一个 LayerNorm** —— 同一个病根(DeepSpeed 关掉外层 autocast),换了个受害者。

**Diagnosis:**

```
File "models/backbones/internvideo2/mobileclip/transformer.py", line 46, in forward
    return super().forward(x.to(torch.float32)).to(inp_dtype)
RuntimeError: expected scalar type Float but found Half
```

`LayerNormFP32` 只把**输入**转 fp32,`self.weight` / `self.bias` 被 `model.half()`
变成了 fp16,而 `F.layer_norm` 要求输入与参数同 dtype。**这个类从来就不自洽**,
只是以前有外层 autocast 替它兜底 —— `layer_norm` 在 autocast 的 fp32 名单里,
而该名单会把**所有**输入(含 weight/bias)一起转 fp32。engine 关掉 autocast,
它就得自己做完整。

**Fix(已合进 main,pull 即可):** 两个副本(`mobileclip/transformer.py` 和
`mobileclip/modules/common/transformer.py`,近似重复文件)都改成参数一并转 fp32。
本地验证:旧写法对 `.half()` 模块复现同类报错,新写法与全 fp32 LayerNorm 再转回
fp16 **逐位相等**,`elementwise_affine=False`(参数为 None)分支也正常。

**这次顺手把同族的雷一次扫完了** —— 起了个子代理把 r0 整条前向路径(视觉塔、
文本塔、VTC 损失、all-gather)全读了一遍,专找「靠 autocast 兜底」的写法:

- **不会再有 dtype 崩溃**。文本塔注意力里的 `softmax(attn.float())` 是**对的**
  (转回后才做矩阵乘,且 softmax 无参数),不用动。
- **但有一处静默的精度回退**:VTC 损失里的 `log_softmax` / `sum` 本来在 autocast
  的 fp32 名单里,现在退化成 fp16。已在 `models/criterions.py` 把相似度矩阵转回
  fp32 再进 `log_softmax`。

  **说清楚它到底值多少**:实测 B=512、temp=0.01 下,损失 fp16 `13.929688`
  vs fp32 `13.929939`,只差 **2.5e-4**。`log_softmax` 内核本来就用 fp32 累加,
  所以**不会**出现我上一条里担心的「阶梯状损失」。留这两行只是把 r0 基线恢复成
  仓库原本的意图,顺便在 loss scale 出问题时少一个嫌疑人 —— 不要指望它改变什么。
- **有一处我改了又撤回**:视觉塔的注意力 softmax 也没有 fp32 提升,看着像同一个
  雷,实测却与直接 fp16 softmax **逐元素差 0.0**(那个内核同样 fp32 累加)。
  纯空操作,撤了,不留在 diff 里当噪音。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

`make_pilot_anno.py` **不用再跑**(你刚跑过,9000 + 1000 已经在位;那次输出里的
「共享字幕 0.0%」和下面的抽样率提示是新加的说明文字,不是异常)。

**这次仍然盯损失。** 判据同上一条:开头刷几条 `OVERFLOW! ... Skipping step`
正常;loss 从第 0 步就 `nan`、或 loss scale 一路掉到 `min_loss_scale` 不动才是
真问题。

**Then report back:** `Train Epoch: [0]` 之后带 `loss_vtc` / `lr` / `temperature`
的头十几行,以及 17 个 batch 跑完有没有干净退出。若又崩,traceback 最后 30 行。

---

## 2026-07-21 (6) — DeepSpeed 起来了、训练循环也起来了;死在第一次 forward 的 dtype

**大进展:** FusedAdam 那关过了 —— `Use deepspeed to initialize model!!!` 之后
显存分配正常(211M),`Start training` / `Epoch: 0` / `MetaLoader has 1
dataloaders, 17 batches in total` 全部打出来了。**优化器、DeepSpeed、数据集、
decord 这四层现在都是通的**,只差最后一步。

**Diagnosis:** 8 个 rank 全部死在第一个卷积:

```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor) should be the same
```

关键线索是崩溃前那行 warning:

```
[WARNING] [torch_autocast.py:122:autocast_if_enabled] torch.autocast is enabled
outside DeepSpeed but disabled within the DeepSpeed engine.
```

仓库的混合精度是「**模型整体 `.half()` + 训练循环外套一层 `torch.cuda.amp.autocast`**」:
权重是 fp16,batch 还是 fp32,靠 autocast 在算子层面把输入转过去。而新版 DeepSpeed
的 `engine.forward` 会**显式关掉外层 autocast**(`torch_autocast.py` 的
`autocast_if_enabled`:没配 `torch_autocast` 段时,用 `enabled=False` 覆盖外层
上下文)。于是那层 autocast 在 DeepSpeed 路径上成了空操作,fp32 输入直接撞上
fp16 权重。**这是 DeepSpeed 版本回归,不是机器配置问题,也跟 Volta/sm_70 无关。**

**Fix(已合进 main,pull 即可):** `tasks_clip/pretrain.py` 在 forward 前把 batch
转成**模型参数的真实 dtype**。我在本地用 `nn.Conv2d(...).half()` 复现了同一类
报错并验证了修复,fp32 路径下是空操作。

用「模型参数的 dtype」而不是现成的 `data_type`,是因为顺手查出精度开关在仓库里
**有三处、两个键名、两个默认值**,只有配置把它们都显式写对时才一致:

| 文件 | 键 | 默认 | 决定 |
|---|---|---|---|
| `shared_utils.py:57` | **`bf16`** | `True` | 模型转 bf16 还是 fp16 |
| `pretrain.py` | `use_bf16` | `True` | autocast 的 `data_type` |
| `config_utils.py:130` | `use_bf16` | `False` | ds 配置写 bf16 段还是 fp16 段 |

试点配置三个都显式写了 False,所以这次没出事;但以后新配置只写一个就会踩雷。
直接问模型要 dtype 就绕开了整个问题。(细节全部记进 `TROUBLESHOOTING.md`。)

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**这次要看的是损失能不能正常打出来。** 心里先有个数:

- 外层 autocast 被 DeepSpeed 关掉后,前向是**纯 fp16**,靠 DeepSpeed 的动态
  loss scaling(`initial_scale_power: 16`)兜底。**开头刷几条
  `OVERFLOW! ... Skipping step` 是正常的**,那是缩放系数在找合适的量级。
- **不正常的是**:loss 从第 0 步就是 `nan`,或者 loss scale 一路掉到
  `min_loss_scale` 就再也不动了。真出现这个,就得把对比损失(`log_softmax`
  那两行)single out 到 fp32 算 —— 但先别改,让数据说话。

**Then report back:** `Train Epoch: [0]` 之后的头十几行(带 `loss_vtc`、`lr`、
`temperature` 的那些),以及跑完 17 个 batch 有没有干净退出。若又崩,把 traceback
最后 30 行贴回来即可。

---

## 2026-07-21 (5) — 更正:那个「共享字幕 0.0%」不能当成「全量重复率低」

**问的是:** 为什么共享字幕会是 0%?

**答:主要是抽样造成的,不是数据的性质。** 上一条笔记里我写的
「多正例要等全量数据才谈得上验证」说对了一半,但结论下得太软 —— 准确的说法是:
**这个 0.0% 对全量几乎没有任何约束力,不能拿它去判 `DATA_ISSUES.md` 里
「重复 caption 率 >10%」那个决策门。**

三条原因,按重要性排:

1. **抽样率只有 0.1%(10k / 10M),这是决定性的。** 一组出现 2 次的重复字幕,
   要**两条都**被抽中才看得见,概率 = 0.001² = **1e-6**。假设全量有 5% 的
   clip(50 万条 = 25 万组)处在重复对里,这 10k 里的期望命中 =
   25 万 × 1e-6 ≈ **0.5 组**。实测 1 组 —— 和「全量 5% 重复」完全相容。
   换句话说,**观测 0.0% 和全量重复 5%,在统计上区分不开**。
   这个抽样只对**全局重复上千次**的模板句有检出力(k=1000 时 P(≥2)≈26%),
   对 k=2…100 的中低频重复完全瞎。
2. **只算精确重复。** 训练侧 `caption_to_idx` 是 `pre_text` 归一化后取 md5,
   "a man is talking to the camera" 和 "a man talks to the camera" 是两个 idx。
   语义近重复一律不计。(顺带修了个小口径 bug:`make_pilot_anno.py` 之前按
   **原始**串统计,比训练口径还严;现已改成先过同一套归一化再计数,
   held-out 评测集的判重也跟着改了。)
3. **InternVid 的构造方式本来就不产生共享字幕。** 每个 clip 由 captioner 独立
   生成描述,不存在 MSR-VTT「一个视频 20 条 caption」那种结构性重复。

**没变的结论:** 试点 A/B 里 `caption_idx` 仍然是空操作(全库就一对同字幕 clip,
落进同一 batch=512 的概率单个 epoch 才 ~5.7%),所以 **r1 - r0 的差异照旧只能
归给 sigmoid 损失 + 注意力池化 + t/b 大学习率**。这点不受上面影响。

**要真判多正例值不值得做,唯一的办法是全量跑一遍:**

```bash
# 只有拿到全量 10M jsonl 时才做(H100 那边),CPU ~20 分钟
python tools/data_audit/profile_jsonl.py --input <全量 jsonl> --out outputs/audit_v0_1/s0_summary.json
python -m json.tool outputs/audit_v0_1/s0_summary.json | grep -A3 duplicate
```

看 `duplicate_zh_caption_rate`。注意口径:它数的是 **zh** 侧,试点训练喂的是
**en** 侧字幕;英文重复经逐条翻译可能被打散,所以它是英文侧重复率的**下界**。

**另外提一个可能更值钱的方向(留给 v0.2,先不动):** InternVid 里同一支源视频
被切成多个 clip,画面高度相似但字幕各不相同 —— 它们在 batch 里互为负例,是
**真的**假负例,而按字幕哈希的 `caption_idx` 一条也抓不到。若全量重复率证明偏
低,比「共享字幕多正例」更值得做的是**按源视频 id 分组**(同 `video_id` 共享
idx,或采样器层面禁止同视频 clip 进同一 batch)。

**这条不需要服务器做任何事** —— 推导已落到 `notes/audit-10k-findings.md`
结论 5、`experiments/README.md` §2b、`tools/data_audit/DATA_ISSUES.md`。
**当前待办仍然是下面 (4) 里那三行命令。**

---

## 2026-07-21 (4) — anno 成了;死在 DeepSpeed 现编 FusedAdam(nvcc 配不上 gcc 11)

**先说好消息:** anno 这步彻底通了 —— `可用 10000 / 缺失 0 / root 外 0`,
9k train + 1k val 都写出来了。冒烟也比上次走远得多:数据集建好(9000 条已序列化)、
`mobileclip_s2.pt` 双塔权重加载、优化器分组(458 + 875 个参数)全过,
说明 sm_70、路径、配置这三层都对了。

**Diagnosis:** 新报错在 `deepspeed.initialize` → `FusedAdam` 这一步。
DeepSpeed 默认的 Adam 是 **FusedAdam**,它要在首次运行时用**系统 `/usr/bin/nvcc`
现编一个 CUDA 扩展**(不是 torch wheel 里自带的那份 CUDA)。101vector 上系统
nvcc 比系统 g++ 老,老 nvcc 解析不了 gcc 11 的 `<functional>`,于是:

```
/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’
```

其余 7 张卡刷的 `fused_adam.so: cannot open shared object file` 是**下游噪音** ——
只有一张卡负责编译,其它卡在等那个从没被生成出来的 `.so`。编译修好就一起没了。
跟上次的 Volta 问题**无关**:这条 nvcc 命令行里 `-gencode=…sm_70` 是对的。

**修法(代码里已经改好,你只要 pull):** `utils/config_utils.py` 现在往生成的
`deepspeed_config.json` 里写 `"torch_adam": true`,DeepSpeed 就改用
`torch.optim.AdamW` —— 同一个算法,只是没有融合核,**什么都不用编译**。
(顺带修了两个坑:`adam_w_mode` 原来写在 `optimizer` 顶层,DeepSpeed 只从
`params` 里 pop,一直被静默忽略;还有 `bias_correction` 是 FusedAdam 专有参数,
原样传给 torch AdamW 会直接 `TypeError`,torch 路径下已去掉。)

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main

# 上次失败留下的半成品编译缓存,清掉免得干扰
rm -rf ~/.cache/torch_extensions/py310_cu118/fused_adam

export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

anno 那两个 json 已经生成好了,**不用再跑 `make_pilot_anno.py`**。

**顺手记一笔(重要,别忘):** 共享字幕 = **1 组 / 2 条 clip(0.0%)**。
意味着 r1 里的 **`caption_idx` 多正例在这批 10k 上是空操作** —— 批内基本撞不上
同字幕的两条 clip。所以试点 A/B 的差异只能归给 sigmoid 损失 + 注意力池化 +
t/b 大学习率这三项,**不能记到多正例头上**;多正例要等全量数据才谈得上验证。
这条已经写进 `experiments/README.md` §2b。

**Then report back** in from-server.md:冒烟的头几十行 —— 具体看
`Use deepspeed to initialize model!!!` 之后有没有顺利进到训练循环、
损失有没有正常打印。若还卡在编译上,把 `nvcc --version` / `g++ --version` /
`echo $CUDA_HOME` 三条的输出一起贴回来。

---

## 2026-07-21 (3) — `没有可用记录` 已修:片段在 InternVId-*/ 嵌套目录里

**Diagnosis:** 不是你命令打错,是 `make_pilot_anno.py` 的 bug。片段**不是平铺在
`videos/` 下**,而是在嵌套的 `InternVId-*/…` 目录里,jsonl 的 `video` 字段常常只是
裸文件名,真正定位文件的是记录里的 `abs_path`。审查工具一直有这套三候选解析
(`tools/data_audit/score_alignment.py: resolve_video_path()`),但
`make_pilot_anno.py` 只做了朴素的 `os.path.join(clips_root, video)` ——
于是 1 万条全部判为「文件缺失」。而且它在打印计数**之前**就 `sys.exit`,
所以那句话没法区分「字段名不对」还是「路径不对」。两个问题都已修并合进 `main`。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin main     # 必须先拉,否则还是旧脚本

python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

ls -l anno_downstream/internvid_zh_10k_pilot*.json

# 成功后接 §2b 第 2–4 步
export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**若还是没记录**:新脚本会打印**分项计数** + 首条记录的**真实字段名** + 试过的路径。
把那一整段贴回来,一轮就能定位。对应处理见 `TROUBLESHOOTING.md`:
- 缺 video/caption 多 → 用新加的 `--video-key` / `--caption-key` 指定真实字段名;
- 文件缺失多 → `--clips-root` 指错了,或记录没有 `abs_path`;
- `root 外` 多 → 文件在 `--clips-root` 之外,anno 写不出相对路径。

⚠️ **不要**因为跑不通就改 `--caption-field zh`。文本塔是 CLIP 英文 BPE,
中文字幕喂进去是**另一个实验**,不是等价替代,会让 r0/r1 对照失效。
英文字幕若真的缺失,先报回来再一起定。

**Then report back** in from-server.md:脚本打印的那两行(可用条数 / 共享字幕比例),
或失败时的完整分项计数。

---

## 2026-07-21 (2) — sm_70 修好了;现在缺的是试点 anno(§2b 第 1 步没跑)

**Diagnosis:** 好消息:**torch 的问题解决了**。这次报错发生在
`setup_dataloaders` → `create_dataset`,说明 CUDA 初始化、模型构建、
`init_distributed_mode` 全都过了 —— 上一条的 sm_70 修复生效了。

现在只是 `anno_downstream/internvid_zh_10k_pilot.json` **还没生成**:它不在仓库里、
也不在审查抽样目录里,要用 `tools/make_pilot_anno.py` 从审查用的 10k jsonl 现做,
也就是 `experiments/README.md` §2b 的**第 1 步**。直接跳到训练命令就会缺这个文件。

**路径本身没问题** —— 已逐项对过 `tools/data_audit/`:

| 项 | 值 | 出处 |
|---|---|---|
| 审查 jsonl | `/home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl` | `run_v100.sh` `$DATA` |
| 视频根目录 | `/home/vca1/datasets/internvid_zh_10k/videos` | `run_v100.sh` `$CLIPS` |
| `data_root` | `$MVC_DATA_DIR/internvid_zh_10k/videos` | `configs/data.py` |
| `anno_path` | `$MVC_ANNO_DIR/internvid_zh_10k_pilot.json` | 同上,`MVC_ANNO_DIR` 默认 `anno_downstream` |

只要 `MVC_DATA_DIR=/home/vca1/datasets`,`data_root` 就正好是审查工具用的那个
`videos/`,和脚本写出的相对 `video` 字段对得上(脚本会拿 `--clips-root` 逐条校验存在性)。

**Do this next:**

```bash
cd ~/mobileclip && git pull origin handoff

# §2b 第 1 步(只需跑一次)
python tools/make_pilot_anno.py \
    --input /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl \
    --clips-root /home/vca1/datasets/internvid_zh_10k/videos \
    --out-train anno_downstream/internvid_zh_10k_pilot.json \
    --out-eval  anno_downstream/internvid_zh_10k_pilot_val.json

ls -l anno_downstream/internvid_zh_10k_pilot*.json     # ~9k + 1k

# §2b 第 2–4 步
export MVC_DATA_DIR=/home/vca1/datasets
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

⚠️ 看一眼脚本打印的两行再往下走:
- `文件缺失 M` 若很大 → jsonl 的 `video` 字段不是相对 `--clips-root` 的(可能带子目录
  或绝对路径),先看 schema:
  `head -1 /home/vca1/datasets/internvid_zh_10k/internvid_zh_10k.jsonl | python -m json.tool`
- `共享字幕: … 组 / 覆盖 … 条 clip (…%)` → 这就是 `caption_idx` 多正例修复在这批数据上的
  影响面上限,**记下来**,后面 r1 结论要用。

**Then report back** in from-server.md:`make_pilot_anno.py` 打印的那两行(可用条数 /
文件缺失 / 共享字幕比例),以及冒烟跑起来后损失有没有正常打印。

---

## 2026-07-21 — pilot_10k_r0 起不来:torch 没有 Volta 内核(**不是 deepspeed 的问题**)

**Diagnosis:** deepspeed 已经装好了(0.19.2,import 正常),这一步已经过去了。
新报错是**另一个更底层的问题**:`mobileviclip` 环境里的 **torch 2.11.0 是 CUDA 13
构建**(依赖里能看到 `cuda-toolkit 13.0.2` / `nvidia-cudnn-cu13` / `nvidia-nccl-cu13`)。
CUDA 13 已经**彻底删掉 Volta(sm_70)**,所以 warning 里列出的架构是
`7.5 / 8.0 / 8.6 / 9.0 / 10.0 / 12.0` —— **没有 7.0**。V100 正是 CC 7.0,
于是**任何一个 CUDA kernel 都跑不了**,第一个碰到的 `torch.distributed.barrier()`
就炸了(`utils/distributed.py:119`),8 张卡全一样。

所以 `deepspeed.enable False` 这条旁路救不了:崩溃发生在 `init_distributed_mode`,
比选不选 deepspeed 更早,跟框架无关。这正是 `tools/data_audit/SETUP_V100.md` §0
早就写过的 V100 三大坑之一 —— 当时只对**审查环境**做了处理,**训练环境
`mobileviclip` 漏了同样的处理**。

**Do this next:**

```bash
# 0) 先拉最新(本条修复 + §2b 的 sm_70 预检都在里面)
cd ~/mobileclip && git pull origin handoff

conda activate mobileviclip

# 1) 卸掉 CUDA-13 构建
pip uninstall -y torch torchvision torchaudio

# 2) 换成带 sm_70 的 cu118 构建(版本与审查环境一致、已在本机验证过;
#    torch>=2.5 才能配 deepspeed 0.19)
pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu118

# 3) decord 0.6 在 numpy 2 下会报 _ARRAY_API,数据加载器是 decord 的
pip install "numpy<2"

# 4) 自检:arch list 里必须出现 sm_70,且 .cuda() 不报错
python -c "import torch, decord; print(torch.__version__, torch.cuda.get_arch_list()); \
  print(torch.zeros(1).cuda(), torch.cuda.get_device_name(0))"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"

# 5) sm_70 出现后再重开冒烟
bash experiments/run_pretrain.sh experiments/configs/pilot_10k_r0.py 8 debug True
```

**冒烟过了之后**:回 `experiments/README.md` §2b 按原顺序续跑,不用另记命令 ——
§2b 的 4 步已经是完整的(1 建 anno → 2 `export MVC_DATA_DIR` → 3 sm_70 预检 →
4 冒烟 + 等步数 A/B)。两个提醒:

- §2b 第 1 步 `tools/make_pilot_anno.py` **必须先跑过一次**,否则 r0 配置找不到
  `anno_downstream/internvid_zh_10k_pilot.json`;已经跑过就跳过。
- 第 4 步的等步数 A/B 是**成对**的,两条都要跑完才能比:
  `pilot_10k_r0.py 8` 与 `pilot_10k_r1.py 8`。

⚠️ 以后**不要再在这个环境里 `pip install -U torch/torchvision/torchaudio`,
也不要裸 `pip install torchvision`** —— PyPI 默认 wheel 是 CUDA 12.8/13 构建,
一升就把 Volta 支持删掉。101vector 上任何 torch 升级都必须走 cu118 index。

顺带:`FusedMLP of flash_attn is not installed!!!` 这类是**正常回退提示**不是错误
(sm_70 装不了 flash-attn,见 README_zh Q9);V100 无 bf16 已在配置里处理好
(`use_bf16=False` / `bf16=False`,fp16 AMP)。

**Then report back** in from-server.md:第 4 步自检的完整输出(尤其 arch list 里
有没有 `sm_70`),以及第 5 步冒烟跑起来后的头几十行(损失有没有正常打印)。

完整版已写进 `TROUBLESHOOTING.md` 末节;`experiments/README.md` §2b 也加了一条
开跑前的 sm_70 预检命令。

---

## 2026-07-23 · 10k 试点跑完 —— 只差一条 ckpt 核实命令(ROADMAP 1.3 判据二三)

`pilot_10k_r0` / `pilot_10k_r1` 都**正跑完**了(结果 + go/no-go 记在
`experiments/pilot_10k_results.md`)。**r0 全过;r1 判据一(损失 7.07→4.2 单调降)过**。
r1 的 held-out 分数低于 r0 是**预期**(新 attn 池化 init + sigmoid 重标定 + 仅 170 步),
按两级设计 10k 不用来排方案,只看动力学。

只剩判据二三要读 r1 的 ckpt —— **只能你在服务器上跑**,Claude 端无 GPU/ckpt。

⚠️ ckpt 是 **DeepSpeed 目录**不是单文件:`ckpt_09.pth/` 里权重在
`mp_rank_00_model_states.pt` 的 `module` 键下。直接 `torch.load` 目录会 `IsADirectoryError`。

```bash
# 仓库根目录，conda activate mobileviclip
python - <<'PY'
import torch
f = "experiments/runs/pilot_10k_r1/ckpt_09.pth/mp_rank_00_model_states.pt"
obj = torch.load(f, map_location="cpu")
sd = obj.get("module", obj)
ls, lb = sd.get("logit_scale"), sd.get("logit_bias")
w  = sd.get("vision_encoder.temporal_pool.proj.weight")
print("present:", ("logit_scale" in sd), ("logit_bias" in sd),
      ("vision_encoder.temporal_pool.proj.weight" in sd))
print("exp(logit_scale) =", None if ls is None else round(ls.float().exp().item(),4), " (init 10 → ≠10)")
print("logit_bias       =", None if lb is None else round(lb.float().item(),4),       " (init -10 → ≠-10)")
print("temporal_pool.proj.weight L2 =", None if w is None else round(w.float().norm().item(),4),
      " mean|w| =", None if w is None else round(w.float().abs().mean().item(),6), " (非零)")
PY
```

判据:`exp(logit_scale)≠10` 且 `logit_bias≠-10`(**关键** —— 证明 `different_lr @1e-3`
在更新它们)、proj 范数非零。三行都满足 → **1.3 全过**,V100/10k 链路筛完,下一步去
H100 跑全量 A/B(`experiments/SETUP_B100.md`,r1a/r1b 分开跑)。顺带
`pytest tests/test_optimization_components.py`。

**把这三行输出贴回 `from-server.md`**,我据此把 1.3 状态 🟡→✅。
细节见 `experiments/pilot_10k_results.md`。
