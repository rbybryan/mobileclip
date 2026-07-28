# 导出冒烟结果(ROADMAP 1.5 / 评审 U1)

**状态:** 🟡 两路已结论,TFLite 待补 · **日期:** 2026-07-27 · **机器:** 101vector(CPU)
**脚本:** `experiments/export_smoke.py` · **命令:** [`COMMANDS.md`](../COMMANDS.md) §3.5
**环境:** torch 2.5.1+cu118 · onnx 1.22 / onnxruntime 1.23.2 / onnxscript 0.7.1 · coremltools 9.0

导出对象:**视觉塔**(`FastViT`,架构参数与 `build_vision_encoder` 逐项一致 ⇒ 含
全部时序算子),随机权重,输入 `[1, T, 3, 256, 256]`,ONNX opset 17。**不含延迟**
—— 真机延迟档仍在第 8 周。

---

## 1. 结果

| T | ONNX | Core ML | TFLite |
|---|---|---|---|
| 2 | ✅ pass | ✅ pass | ⏭️ 待测 |
| 4 | ✅ pass | ✅ pass | ⏭️ 待测 |
| 8 | ✅ pass | ✅ pass | ⏭️ 待测 |
| 16 | ✅ pass | ✅ pass | ⏭️ 待测 |

- **ONNX**:`torch.onnx.export` + `onnx.checker.check_model` + **onnxruntime 加载并推理**
  三步全过(U1 明确要求后半步)。
- **Core ML**:`torch.jit.trace` → `ct.convert(convert_to="mlprogram")`,
  全尺寸约 **4343 个 MIL op**,MIL 五段 pipeline 全过。
- **TFLite**:本轮**未取得有效结论**,原因见 §3(是工具链问题,不是模型问题)。

## 2. 结论:P9 暂不触发

**联合注意力在 ONNX 与 Core ML 两条路上都不构成导出障碍**,且四个 T 一致通过。

⇒ 按 U1 的判据,**P9(分离注意力变体)不需要提前进入 R3 架构决策视野** ——
弹性模型可以按现有形态训练,不必为了可导出性改架构。

⚠️ **这个结论待 TFLite 补齐后才完整**。Android 侧走 TFLite,若那条路挂在联合
注意力上,P9 仍可能被触发。

## 3. TFLite 那一格为什么没有结论

**是工具链改名,不是导不出。** `ai-edge-torch` 已废弃并**改名为 `litert-torch`**;
`pip install ai-edge-torch` 只会装到一个 1.8 kB 的空壳,`ai_edge_torch.convert`
不存在,报 `AttributeError`。

第一版脚本把它记成了 **fail** —— 那正是**假 P9 触发**:让"环境问题"伪装成
"导不出",进而让架构决策建立在不存在的证据上。已修:两个包名都试,都没有
convert 入口就抛 `BackendUnavailable` ⇒ 记 **skip**,并在原因里写明该装什么。

**补测命令:**

```bash
pip install litert-torch
python experiments/export_smoke.py            # 只有 tflite 那列会变
```

## 4. ⚠️ 一个对 C2 部署叙事有影响的观察

trace 期间稳定出现这几条 `TracerWarning`(`models/mobileviclip_small.py`):

```
:50    _NUM_FRAMES = int(t)
:513   x = x.reshape(t_frames, int(x.shape[0]/t_frames), ...)
:727   同上
:1149  同上
:65    if t == emb.shape[-1]          # temporal_embedding 插值分支
```

含义:**T 在导出图里被烘成常量**,包括 `temporal_embedding` 走不走插值这个分支。

这**不是失败** —— U1 要的就是"按 T 各出一张静态图",现状符合要求。但部署叙事
要说准:C2 的"帧数即部署旋钮"在**权重层面**成立(单一 checkpoint),在**产物层面**
是 **1 份权重 → N 张图**。

**未验证的下一问**:ONNX 用 `dynamic_axes` 把 T 设成动态维,能否一张图服务所有 T?
若能,部署叙事可以强化成"单图任意 T";若不能(很可能,因为 `int()` 与 `if` 把 T
烘死了),论文里就该按"N 张图共享权重"写。这是**便宜且直接影响 C4 端侧表的**一问,
建议在真机延迟档之前搞清楚。

## 5. 复现

```bash
python experiments/export_smoke.py --mini     # 自检脚本本身,秒级
python experiments/export_smoke.py            # 本表
python experiments/export_smoke.py --inference-mode   # 重参数化后的部署形态(未测)
```

原始产物:`experiments/exports/export_smoke.{json,md}` 与各 `vision_T*.{onnx,mlpackage}`。

**读表规则**(脚本里也写死了):skip **不是**结论 —— 既别读成 pass,也别读成 fail。
只有 **fail 且失败算子指向 attention / `temporal_embedding`** 才是 P9 的证据。
