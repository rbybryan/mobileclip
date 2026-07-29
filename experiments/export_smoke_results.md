# 导出冒烟结果(ROADMAP 1.5 / 评审 U1)

**状态:** ✅ ONNX / Core ML 已结论(全 pass);TFLite 顺延至第 8 周部署档 · **日期:** 2026-07-27(TFLite 处置 07-29) · **机器:** 101vector(CPU)
**脚本:** `experiments/export_smoke.py` · **命令:** [`COMMANDS.md`](../COMMANDS.md) §3.6
**环境:** torch 2.5.1+cu118 · onnx 1.22 / onnxruntime 1.23.2 / onnxscript 0.7.1 · coremltools 9.0

导出对象:**视觉塔**(`FastViT`,架构参数与 `build_vision_encoder` 逐项一致 ⇒ 含
全部时序算子),随机权重,输入 `[1, T, 3, 256, 256]`,ONNX opset 17。**不含延迟**
—— 真机延迟档仍在第 8 周。

---

## 1. 结果

| T | ONNX | Core ML | TFLite |
|---|---|---|---|
| 2 | ✅ pass | ✅ pass | ⏭️ 顺延 |
| 4 | ✅ pass | ✅ pass | ⏭️ 顺延 |
| 8 | ✅ pass | ✅ pass | ⏭️ 顺延 |
| 16 | ✅ pass | ✅ pass | ⏭️ 顺延 |

- **ONNX**:`torch.onnx.export` + `onnx.checker.check_model` + **onnxruntime 加载并推理**
  三步全过(U1 明确要求后半步)。
- **Core ML**:`torch.jit.trace` → `ct.convert(convert_to="mlprogram")`,
  全尺寸约 **4343 个 MIL op**,MIL 五段 pipeline 全过。
- **TFLite**:**顺延至第 8 周部署档**,原因见 §3 —— 三轮报错全是工具链问题,
  **无一指向模型导不出**。

## 2. 结论:P9 暂不触发

**联合注意力在 ONNX 与 Core ML 两条路上都不构成导出障碍**,且四个 T 一致通过。

⇒ 按 U1 的判据,**P9(分离注意力变体)不需要提前进入 R3 架构决策视野** ——
弹性模型可以按现有形态训练,不必为了可导出性改架构。

**TFLite 未评估,但不阻塞这个结论。** P9 的触发条件很窄 ——「联合注意力导不出」。
联合注意力就是标准 SDPA / matmul+softmax,已过两条独立路线,且 **Core ML 的 MIL
算子集比 TFLite 还窄**(4343 个 op 全过)。TFLite 若真挂,大概率挂在 5D reshape
上 —— 那是**部署工程问题而非 P9**,不影响 R3 的架构决策。

⚠️ 行文口径:**只能写「ONNX 与 Core ML 两路无障碍,TFLite 未评估」**,不能写
「三路全通」。上面那段是推断,不是实测。

## 3. TFLite 那一格为什么顺延

**三轮尝试,没有一条报错指向模型导不出。** 全部已归 skip 而非 fail。

| 轮次 | 报错 | 真实原因 |
|---|---|---|
| 1 | `module 'ai_edge_torch' has no attribute 'convert'` | `ai-edge-torch` 已废弃**改名** `litert-torch`,旧包只是 1.8 kB 空壳 |
| 2 | `module 'torch' has no attribute 'int1'` | `litert-torch` 需 **torch ≥ 2.6** 的次字节 dtype;本机 **2.5.1+cu118** |
| 3 | 见下「三轮栽在同一件事上」 | 旁路环境始终没建成,pip 一直在往系统 python 装 |

第 2 条是**硬约束,不能在 101vector 上解**:那台机器的 torch 版本是为 V100 的
**sm_70 内核**锁定的(见 `TROUBLESHOOTING.md` 末节 —— 换成 cu126/cu128/CUDA-13
构建会让 8 卡在第一个 barrier 就全部报 `no kernel image is available`)。
**为一格导出结论去动训练环境的 torch,风险远大于收益。**

### 三轮栽在同一件事上:旁路环境没真的建起来

根因不是依赖,是**没有验证隔离是否生效**。两次日志里 pip 都在往系统 python 装:

- `Defaulting to user installation because normal site-packages is not writeable`
  ⇒ 落到 `~/.local/lib/python3.10/site-packages`
- `Permission denied: '/usr/local/lib/python3.10/dist-packages/test'`
  ⇒ 落到系统 dist-packages(无权限)

⚠️ **`~/.local` 按 python 版本共享,不按环境隔离。** `mobileviclip` 也是 python 3.10,
而 user site 在 `sys.path` 里**排在 conda 环境自己的 site-packages 之前**。所以往
`~/.local` 装 torch 会**盖住训练环境的 2.5.1**。已清理并复验干净(2026-07-29:
`2.5.1+cu118`,路径在 `miniconda3/envs/mobileviclip/` 下)。

另有两个坑,顺带记下:

- **`(env)` 提示符会骗人**。`rm -rf` 掉 venv 目录后 PS1 残留,PATH 指向不存在的
  路径,`pip`/`python` 静默回落到系统解释器。**看提示符不算数,要看 `sys.prefix`。**
- **裸 venv 走不通**。导出虽是 CPU-only + 随机权重,但 `models/mobileviclip_small.py`
  拉的是整条 InternVideo2 链(`torchvision`/`PIL`/`einops`/…),装 `timm` 远远不够;
  且 `litert-torch` 与 torch/numpy 同轮解析时,pip 23.0.1 会从 torch 2.13 回溯到 2.4、
  再从 numpy 2.2.6 回溯到 **1.19.2 并试图源码编译**,挂在 Cython
  (`_sfc64.pyx: Cannot assign type ... noexcept`),途中已下载数 GB nvidia wheel。

### 第 8 周补测配方(**每一步都先验,验不过就停**)

```bash
conda create -y --name tflite-export --clone mobileviclip
conda activate tflite-export

# ↓ 这两条是关卡,不是装饰。打不出 envs/tflite-export 就别往下走
python -c "import sys; print(sys.prefix)"
export PYTHONNOUSERSITE=1                    # 挡住 ~/.local,防止回落污染

pip install -U pip                                              # 旧 pip 的回溯是灾难源头
pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.12.1" torchvision                                 # CPU 轮子:~200MB vs ~3.5GB
pip install litert-torch                                        # 单独一步,别和 torch 同轮解析

cd "$(git rev-parse --show-toplevel)"
python experiments/export_smoke.py --ts 2,16
```

三个要点:

1. **`torch==2.12.1`**,不是 2.13 —— `litert-torch` 0.9.2 声明 `torch<2.13.0`,
   钉 2.13 会被它降级,那 526 MB 白下。
2. **CPU-only 轮子**。导出不需要 CUDA;装 CUDA 版会在 `import torch` 时炸
   `OSError: libcudart.so.13: cannot open shared object file`(torch 与 nvidia 库
   落在不同 site-packages,路径对不上)。
3. **只跑 T=2 与 T=16 两端** —— 中间两个 T 在另外两条路线上都与两端同结论,
   没有理由单独在 TFLite 上分叉。

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
