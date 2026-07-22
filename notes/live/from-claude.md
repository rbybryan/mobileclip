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

## 2026-07-22 (10) — 冒烟通过 ✅ 可以跑真 A/B 了(但评测已近饱和,判据要改)

**链路全通了。** 8 卡起来、数据读到、前向反向都过、epoch 末评测出表、下一个 epoch
接着跑。`initial_scale_power 5` 全程**零** `OVERFLOW`,anomaly detection 按预期
被跳过(日志里有那行中文说明)。这是试点第一次真正跑出损失。

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
