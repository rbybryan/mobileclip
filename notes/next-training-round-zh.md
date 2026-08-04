# 下一轮 refinement 训练:执行手册(2026-08-03)

**选中的两条臂**(候选阶梯与淘汰理由见 [`verification-tier-1m-zh.md`](verification-tier-1m-zh.md) §11.6):

| 臂 | config | 时长 | 为什么是它 |
|---|---|---|---|
| **MC2-init** | `sub1m_mc2init_s42` | 15 h | 图像骨干供了 61.47 里的 54.82 ⇒ **换更好的图像初始化是可得的最大单项 delta**(round4 §9.4 原话);架构一致可直换(§3.4 已核),**零新模型代码** |
| **lean** | `sub1m_lean_s42` | 15 h | 两个已验证的减法合体(P1trained 非劣 + P5 评测版显著正)= **出货配置的保险丝**;flag 已实现(`models/mobileviclip_small.py` `disable_temporal_embedding`) |

Survivor B(精确 InfoNCE 负例库)期望价值同样高,但**差一段真代码**(bank 移植到
`vtc_loss` + 缓存构建),排下一批 —— 见 §4。

两臂都在 **b100(H100)** 跑,排在当前队列(`P1trained_s43` 训练 → 1M 教师抽取,~55 h)之后。
**§0 和 §1 的前半现在就能做,不占训练卡。**

---

## §0. 现在就做:下载 MC2 权重 + 两道零 GPU 闸门(~10 分钟)

```bash
cd ~/mobileclip && git pull

# 1) 下载 MobileCLIP2-S2(~600 MB,HF 直链;条目见 CHECKPOINTS.md)
wget -O checkpoints/mobileclip2_s2.pt \
  "https://huggingface.co/apple/MobileCLIP2-S2/resolve/main/mobileclip2_s2.pt"
```

**闸门 A · 键集一致性(强制,CPU 30 秒)** —— 加载器是 `strict=False`,键对不上会
**静默空载**(gate5 同款坑),所以必须先断言 v2 与 v1 的键/形状完全一致:

```bash
python - <<'PY'
import torch
a = torch.load('checkpoints/mobileclip_s2.pt',  map_location='cpu')
b = torch.load('checkpoints/mobileclip2_s2.pt', map_location='cpu')
ka, kb = set(a.keys()), set(b.keys())
print('v1 keys:', len(ka), ' v2 keys:', len(kb))
print('only v1:', sorted(ka-kb)[:5]); print('only v2:', sorted(kb-ka)[:5])
assert ka == kb, '键集不一致 ⇒ 需要转换脚本,停手,把上面两行贴回 from-server.md'
bad = [k for k in ka if a[k].shape != b[k].shape]
assert not bad, f'形状不一致: {bad[:5]}'
print('OK: 键与形状完全一致,可直接作为 init')
PY
```

**闸门 B · 配置可解析(CPU 10 秒)**:

```bash
for c in sub1m_mc2init_s42 sub1m_lean_s42; do
  python -c "
import sys; sys.path.insert(0,'.')
from utils.config import Config
c = Config.from_file('experiments/configs/$c.py')
ve = c.model.vision_encoder
print('$c', c.model.vision_ckpt_path, ve.get('disable_temporal_branch', False), ve.get('disable_temporal_embedding', False))
"
done
# 期望:mc2init 行显示 mobileclip2_s2.pt / False / False;lean 行显示 mobileclip_s2.pt / True / True
```

---

## §1. 冒烟门(卡一空先做这个,~10 分钟,单卡)

两臂各跑几十步 debug,**看的不是 loss,是三行日志断言**:

```bash
export CUDA_VISIBLE_DEVICES=0
bash experiments/run_pretrain.sh experiments/configs/sub1m_lean_s42.py 1 debug True \
  2>&1 | tee /tmp/smoke_lean.log
grep -E "\[P1\]|\[P5\]" /tmp/smoke_lean.log
# 必须同时出现:
#   [P1] 时序分支已删除:40 个 layer_scale_1d 置零并冻结
#   [P5] temporal_embedding 已置零并冻结(塔中不再有任何 T 形状参数)

bash experiments/run_pretrain.sh experiments/configs/sub1m_mc2init_s42.py 1 debug True \
  2>&1 | tee /tmp/smoke_mc2.log
grep -E "Load vision_encoder checkpoint|missing_keys" /tmp/smoke_mc2.log | head -4
# 必须看到:Load vision_encoder checkpoint from checkpoints/mobileclip2_s2.pt
# missing_keys 应与 r0 先例相同(只剩时序件与 temporal_pool;见 §6 的先例记录)
```

任一断言缺失 ⇒ **不排队**,把日志贴回 `from-server.md`。

---

## §2. 排队(当前队列跑完、`status_1m.py` 转 ⚪ 后)

```bash
python experiments/status_1m.py        # 必须 ⚪、无 🔴/⚠️ 才排
nohup bash experiments/run_verify_1m.sh 0,1,2,3 sub1m_mc2init_s42 sub1m_lean_s42 \
  > experiments/runs/verify_1m_round2.log 2>&1 &
echo "round2 PID = $!"
```

串行 2 × 15 h ≈ **30 小时**。MC2-init 排前(信息量更大:它可能改变**所有**后续臂的默认配置)。

---

## §3. 判读(每臂结束后,CPU 分钟级)

```bash
# ── MC2-init:同 seed 减 r0 ──
python experiments/collect_results.py --compare experiments/runs/sub1m_mc2init_s42 experiments/runs/sub1m_r0_s42
python experiments/collect_results.py --compare experiments/runs/sub1m_mc2init_s42 experiments/runs/sub1m_r0_s42 --prefix ret_test/

# ── lean:先对父臂(隔离 P5 的训练期效应),再对 r0(整体非劣)──
python experiments/collect_results.py --compare experiments/runs/sub1m_lean_s42 experiments/runs/sub1m_P1trained_s42
python experiments/collect_results.py --compare experiments/runs/sub1m_lean_s42 experiments/runs/sub1m_P1trained_s42 --prefix ret_test/
python experiments/collect_results.py --compare experiments/runs/sub1m_lean_s42 experiments/runs/sub1m_r0_s42
python experiments/collect_results.py --compare experiments/runs/sub1m_lean_s42 experiments/runs/sub1m_r0_s42 --prefix ret_test/
```

**决策表**(终点只认 `R_sum`/`r_mean`,单指标不作数 —— §10.2 的实测纪律):

| 臂 | 结果 | 动作 |
|---|---|---|
| MC2-init | 两口径显著正 | **改进臂统一换 MC2 init**(R0 仍锚 v1 复现,U8 计划);10M 最终模型用 MC2 |
| MC2-init | NS | 记 null:「初始化优势没穿过 1M 视频微调」,保持 v1,故事更简单 |
| MC2-init | 显著负 | 先回 §0 闸门 A 查加载,**别急着下结论** |
| lean | 对 r0 非劣(CI 下界 > −2.5) | **lean 成为出货配置**;对 P1trained 若还显著正 ⇒ P5 的 +0.7 在训练期也成立,可写 |
| lean | 显著负超 δ | 保险丝熔断:出货配置退回只删时序分支(P1trained) |

---

## §4. 这轮之后解锁什么(按依赖)

| 项 | 差什么 | 谁做 |
|---|---|---|
| **Survivor B**:精确 InfoNCE 负例库 K′∈{0,16k,128k}(§11.6-1,期望价值最高的未测臂) | bank 移植到 `vtc_loss` + 投影前缓存构建脚本 | Claude 写(回「写吧」) |
| **α 扫描 + LiNeS 剖面**(反遗忘族的免费闸门,可能白拿 +1~+4) | ~30 行插值脚本 | Claude 写(回「写吧」) |
| **EMA 搭车**(U8 唯一真缺口;落地后任何下臂免费出 EMA/raw 双对照) | `tasks_clip/pretrain.py` 加 EMA | Claude 写(回「写吧」) |
| KD 首臂 `sub1m_C1kd_s42`(§11.2 台阶 ⑤) | 教师抽取跑完 + 训练侧接线(先修 `loss_weight` 假开关) | 抽取在跑;接线 Claude 写 |

**汇报**:§0 闸门 A 的输出、§1 的三行断言、§3 的八张 CI 表,照旧贴 `from-server.md`。
