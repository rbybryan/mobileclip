# 脚本级优化清单(训练 / 评测 / 数据链路 · 逐项核查)

**日期:** 2026-07-16
**性质:** 对现有模型脚本(`tasks_clip/`、`models/`、`dataset/`)的逐文件检查结果:
哪些文件/函数存在可进一步优化的热点。与已合入的两份模型本体报告分工明确:

- `notes/model-architecture-improvements.md` / `-zh.md` —— **模型架构与训练目标**(P1–P10);
- **本清单 —— 脚本与工程层**:训练循环、评测循环、模型装配、数据管线。

所有条目均已在源码中逐行核实(行号以 2026-07-16 仓库状态为准,漂移后按符号名定位)。
数据侧 decord 解码链路(`dataset/video_utils.py:read_frames_decord`)检查过**没有问题**
(`get_batch` 只解码采样帧、bridge 已设 torch),无需改动。

---

## 一、高优先级(收益大、风险低)

### H1. `tasks_clip/pretrain.py:362` — 全局开启 `torch.autograd.set_detect_anomaly(True)`

入口处无条件打开 autograd 异常检测,给**每一步反向传播**加完整图检查,
典型代价是训练慢 **2–5×**。这是调试残留,正式训练必须移除
(或改为 `config.debug` 控制)。**全仓库性价比最高的一行改动。**

### H2. `models/mobileviclip_{small,tiny}.py` — `basic_blocks(..., with_cp=True)` 硬编码

FastViT 构建时把梯度检查点**写死为 True**(全部 stage 全量重算),
配置里的 `gradient_checkpointing` 并没有接进来。建议:

1. 把 `with_cp` 接到 config;
2. 实测显存余量后**选择性检查点**:只对激活最大的 stage 1–2 开启,
   stage 3–4 关闭;显存足够则全关。

预期反向速度提升 20–40%(V100-32GB、batch 64×8 帧,需实测确认)。

### H3. `tasks_clip/retrieval_utils.py:147-148` — DSL 打分口径(正确性,先于性能)

`evaluation_video_clip` 返回的主分数做了 dual-softmax:
`i2t_scores * i2t_scores.softmax(dim=0)`,即用**整个测试集的查询分布**修正打分
(测试时先验);`"_emb/"` 前缀才是原始余弦分。复现对齐论文数字前必须先确认
论文报告的是哪一个口径,否则 R@1 差距可达数个点,**所有 A/B 实验结论都会被污染**。
建议:在 eval 结果里同时报告两个口径并显式命名(`dsl/` 与 `raw/`),
`best_key` 绑定到明确口径。

### H4. `tasks_clip/retrieval_utils.py:17,137` + `models/.../mobileone.py:13-19` — spikingjelly 残留

- `retrieval_utils.py` 顶部 import 脉冲神经网络库 spikingjelly,并在评测循环里
  **每个 batch** 调一次 `functional.reset_net(model)`——模型中没有任何 LIF 神经元,
  该调用只是逐层遍历整个模块树,纯开销;
- `mobileone.py` 同样 import(`MultiStepLIFNode` 等)但从未使用;
- `mobileviclip.yml` 还带着 `snntorch==0.9.4`。

全部清除:去掉两处 import、`reset_net` 调用、环境依赖。收益 = 少量评测提速 +
**环境瘦身**(装机快、隔离面小)+ 未来端侧导出不被无关依赖绊住。

### H5. `tasks_clip/retrieval_utils.py:evaluation_video_clip` — 每张卡冗余编码全量测试集

代码注释自述 "this computes all features in each GPU":8 个 rank 各自把**全部**
视频与文本编码一遍,没有任何分片。MSR-VTT 1k 尚可,ActivityNet / DiDeMo
(数千条长视频 × 8 帧)时评测时间 ≈ 单卡时间。改法:按 rank 分片视频列表 →
各自编码 → `all_gather` 拼接 → rank0 算相似度。预期评测提速近 **8×**;
训练每个 epoch 都要评测,属累积收益。
(对照:同文件 `evaluation()` 的 rerank 部分早已做了 rank 分片,模式现成。)

---

## 二、中优先级

### M1. `tasks_clip/pretrain.py:73` — 主进程逐步 tokenize

`text_input = tokenizer(text).to(device)` 在训练循环里、主进程上做 BPE 分词,
与 GPU 计算串行。挪进 dataset `__getitem__` 或 collate_fn
(6 个 dataloader worker 并行、与计算重叠)。单步数毫秒 × 全部步数。

### M2. `RepMixer.forward` / `AttentionBlock.forward` / `MHSA.forward` — reshape 往返

每个 block 做两轮 `reshape(8,...) → permute → reshape`(时间轴混合前后各一次),
全是非连续内存搬运,且 stage 1 发生在 64×64×80 的大激活上。
短期:合并/消除相邻 block 间的布局转换;
这与已合入的提案 **P1(高效注意力内核)、P2(解除 T=8 硬编码)** 是同一处代码,
建议同一个 PR 一起动。

### M3. `tasks_clip/shared_utils.py:72` — `find_unused_parameters=True`

DeepSpeed 关闭、退回 DDP 时,该标志使每步额外遍历一次计算图。
文本塔冻结参数本就 `requires_grad=False`,不需要它;实测能关就关。

---

## 三、低优先级(顺手改)

| # | 位置 | 问题 | 改法 |
|---|------|------|------|
| L1 | `pretrain.py:84,91` | `optimizer.zero_grad()` 未用 `set_to_none=True`(torch 1.13 默认 False) | 加参数 |
| L2 | `pretrain.py:106` | 每步 `temp.item()` 强制 GPU 同步 | 降频到 `log_freq` 再取值 |
| L3 | `retrieval_utils.py:itm_eval` | 逐行 numpy 全排序 O(N²logN) | 大测试集改 GPU `torch.topk` / 向量化 |
| L4 | `pretrain.py:122-145` | `save_iter=500` 全量存 ckpt(含 optimizer 状态),I/O 阻塞主循环 | 加大间隔或异步保存 |
| L5 | `shared_utils.py:77` | GradScaler 在 DeepSpeed 路径下创建但不用 | 条件创建(仅整洁性) |

---

## 四、建议动手顺序与验证

```
第一批(一个 PR,训练端立竿见影): H1 → H4 → H2(可配置化)
第二批(先对口径,再定基准):     H3(核对论文 DSL 口径)
第三批(评测扩展前做):           H5(评测分片)
第四批(与 P1/P2 合并):          M1 / M2 / M3
低优先级:                        随其它 PR 顺手带上
```

**验证方法:** H1/H4/M1 改完后跑 `debug=True` 短训 + MSR-VTT 零样本评测,
确认损失曲线与 R@1 数值不变(这些改动都不应影响数学结果);
H2 记录改动前后显存峰值与 step time;H5 对比分片前后评测分数逐位一致。
