# b100 工作区本地改动记录（vs remote repo）

b100 主机 `/workspace/mount/b100_zaip_data/rby/MobileViCLIP` 里的代码相对远程仓库做过
修改。本文件是这些改动的**权威记录**：改了哪个文件、改了什么、为什么、如何处置。

**状态：✅ 逐行 diff 已审读完毕（2026-07-23，快照 `87b812e` 推至 `server/b100-local-opts`）。
处置决定待维护者拍板（见文末）。**

## 基本事实

- b100 的 `MobileViCLIP` **不是本 fork 的 clone，而是上游 `MCG-NJU/MobileViCLIP` 的
  clone**（`origin` 指向上游）。diff 基准 = 上游 main HEAD（`3706de2`，无版本漂移）。
  本 fork 与上游无共同 git 历史。
- 改动规模：**19 个文件，+1123/−50 行**（8 改 + 11 新增；另有权重
  `checkpoints/nebulacv-clip-24m.pt` 不进 git，已用 `.git/info/exclude` 排除）。
- 快照分支：fork 上的 `server/b100-local-opts` = commit
  `87b812ebfb95220c2e6a2ff44783692b4c286e66`（`git stash create` 产物，父 1 = 上游 main）。
  首个快照 `1dabf71` 因混入权重作废，未推出 b100。

## 改动的目的（一句话）

**把 MobileViCLIP-Tiny 的双塔底座从 Apple MobileCLIP-S2（英文）换成
NebulaCV-CLIP-24M（中文：FastViT-MCI0 视觉塔 + MiniRBT-H288 中文文本塔），
在 InternVid 中文翻译字幕（11 分片，人审覆盖）上做预训练。**
训练数据与 HF 缓存路径都指向 `wky_b100` 的工作区（`internvid-cap-zh`、
`wky_b100/cache/hfhub`），即这是与 wky 协作的中文训练线，与本 fork 的
InternVid-10M-FLT 英文 R0–R3 计划（`experiments/SETUP_B100.md`）**平行、互不冲突**。

## 修改的已有文件（8 个）

| 文件 | 改动 | 性质 |
|------|------|------|
| `.gitignore` | +1：忽略输出目录 `pretrain_mobileviclip_nebulacv_24m/` | 本地便利 |
| `dataset/__init__.py` | train/test transform 参数化：`inputs.image_mean/image_std` 可覆盖、`inputs.interpolation` 可选（默认 bilinear 不变）、新增 `inputs.normalize` 开关（默认 False=旧行为；NebulaCV 需要 OpenCLIP 归一化）| **通用增强，向后兼容** |
| `models/backbones/internvideo2/__init__.py` | 重写为惰性导入（模块级 `__getattr__`）：import mobileclip 组件不再连带加载 InternVideo2/LLaMA 重依赖；导出 `HFTextEncoder`/`HFTokenizer` | **通用增强** |
| `models/backbones/internvideo2/mobileclip/__init__.py` | +2：导出 `HFTextEncoder`、`HFTokenizer` | 配套 |
| `models/mobileviclip_small.py` | 删除未用的 `InternVideo2` 顶层导入（配合惰性导入） | 配套 |
| `models/mobileviclip_tiny.py` | 最大改动（+80/−17），见下方细目 | 核心 |
| `tasks_clip/pretrain.py` | `image.to(device)` → 同时 cast 到 `data_type`（bf16/fp16），仅当 `use_half_precision` | **dtype 对齐修复**（正对本 fork fp16/`LayerNormFP32` 那串故障的病灶） |
| `tasks_clip/shared_utils.py` | **bug 修复**：`config.get('bf16', True)` → `config.get('use_bf16', True)`——上游读的 key 和配置里写的 key 不一致；本 fork 的 `SETUP_B100.md` §6 之所以要同时传 `use_bf16 True bf16 True` 就是在绕这个坑 | **bug 修复** |

### `models/mobileviclip_tiny.py` 细目

1. 文本塔可插拔：配置有 `hf_model_name` 时用 `HFTextEncoder`/`HFTokenizer`（HF
   AutoModel/AutoTokenizer），否则回落原 `TextTransformer`/`ClipTokenizer`；HF 相关
   键（`hf_tokenizer_name`、`hf_pooler_type`、`hf_proj_type`、`local_files_only`、
   `cache_dir`、`config_path`、`tokenizer_max_length`）都能从训练配置覆盖 JSON。
2. **冻结逻辑修复**：上游 `hasattr(name, '_1d')`（对 str 恒为 False）是死代码；
   重写为显式判断 temporal 部件（`mixer_1d/norm_1d/layer_scale_1d`，**新增
   `temporal_embedding`**）与投影头（`head.*`），文本侧冻结额外匹配 HF 塔的
   `proj.` 前缀。**上游真 bug，值得单独上游/进 fork。**
3. `GlobalPool2D`/`FastViT` 新增 `head_bias` 选项（NebulaCV 权重的投影头带 bias）。
4. 归一化 mean/std 从配置读（默认 ImageNet 值不变）。
5. `checkpoint_format="openclip"` 时走新的 `load_openclip_mobileclip_checkpoint`，
   并把加载报告存到 `self.checkpoint_load_report`。

## 新增文件（11 个 + 1 个权重）

| 文件 | 行数 | 说明 |
|------|------|------|
| `models/checkpoint_utils.py` | 190 | OpenCLIP/timm FastViT 键名 → 本仓库键名的转换器（`stem.`→`patch_embed.`、`stages.N`→`network.M`、`mlp`→`convffn`、`se.fc1/fc2`→`se.reduce/expand`、`head.fc.weight`→`head.proj` 转置、`logit_scale`→`temp=exp(−s)`）+ 带**匹配率守门**的加载器（vision/text 任一 0 命中或低于 `min_match_ratio` 直接 raise，防静默随机初始化） |
| `models/.../mobileclip/hf_encoder.py` | 76 | `HFTextEncoder`：AutoModel（仅本地配置构建，不下权重）、mean/cls pooling、mlp/linear/none 投影、支持梯度检查点 |
| `models/.../mobileclip/hf_tokenizer.py` | 45 | `HFTokenizer`：返回 input_ids 张量，空白归一、max_length padding/截断 |
| `models/.../configs/nebulacv_clip_24m.json` | 15 | 文本塔配置：embed_dim 512、fastvit_mci0@256、hfl/minirbt-h288、mean_pooler、mlp 投影 |
| `preprocess/prepare_internvid_chinese_annotations.py` | 273 | 中文标注管线：机翻 `raw_caption_zh_translated` + 人审 verdict 合并（通过/改后通过→用 `corrected_translation` 覆盖；拒绝→剔除；不确定→`drop/raw/corrected` 三种策略），按 volume 切成 11 个 `InternVId-FLT_{i}.json`（原子写入） |
| `scripts/pretraining/clip/mobileviclip_nebulacv/config.py` | 143 | 训练配置：11 中文分片（`ANNO_DIR`/`ANNO_FILES`/`ANNO_PATH` 三种指定法）、8 帧、batch 32/卡、max_txt 32、OpenCLIP 归一化 + bicubic 256px、视觉塔全放开、文本塔冻结只开 MLP 投影、双塔均从 `nebulacv-clip-24m.pt` 初始化（match_ratio ≥0.98）、仅 VTC loss、lr 1e-5、3 epoch cosine、bf16、deepspeed stage-1 |
| `scripts/pretraining/clip/mobileviclip_nebulacv/run.sh` | 24 | torchrun 启动器；HF 离线模式；`HF_HOME` 指向 wky 缓存；**默认 `NUM_GPUS=8`** |
| `docs/nebulacv_chinese_training.md` | 51 | 中文训练操作文档（数据/权重/MiniRBT 快照路径、`LD_LIBRARY_PATH` PyAV 坑、`Loaded 888/888 converted tensors` 自检） |
| `tests/test_checkpoint_utils.py` | 30 | 键名转换单测 |
| `tests/test_internvid_chinese_annotations.py` | 89 | 标注管线单测 |
| `AGENTS.md` | 27 | AI 助手仓库说明（Codex 风格，b100 侧自建） |
| `checkpoints/nebulacv-clip-24m.pt` | — | NebulaCV-CLIP-24M 权重（不进 git） |

## 评审意见与注意点

1. **代码质量整体不错**：向后兼容（不设新配置键则行为与上游一致）、带单测、
   加载器有防静默失败的守门。不是随手 hack。
2. ⚠️ `run.sh` 默认 `NUM_GPUS=8`，与本 fork 的 b100 纪律「**默认只用前 4 张卡**」
   （`SETUP_B100.md`）冲突——这是 wky 协作线的脚本，跑之前需与维护者对齐卡数分配。
3. 三处改动是**独立于中文线的真修复/增强**，值得移植进本 fork（fork 同源文件已
   各自演化，移植需比对，不能盲 cherry-pick）：
   - `shared_utils.py` 的 `use_bf16` key 修复；
   - `mobileviclip_tiny.py` 的冻结逻辑修复（`hasattr(name,'_1d')` 死代码）；
   - `pretrain.py` 的输入 dtype cast。
4. `dataset/__init__.py` 的 transform 参数化与惰性导入也是无害通用增强，可一并评估。

## 处置（待维护者决定）

| 选项 | 内容 |
|------|------|
| A | 全量作为 `nebulacv` 特性分支进 fork（保留中文线完整可复现） |
| B | 只移植上面第 3、4 条通用修复进 fork main，中文线留在 b100 |
| C | 仅存档（快照分支 `server/b100-local-opts` 已在 fork 上，随时可取） |
