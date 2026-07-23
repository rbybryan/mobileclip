# b100 工作区本地改动记录（vs remote repo）

b100 主机 `/workspace/mount/b100_zaip_data/rby/MobileViCLIP` 里的代码相对远程仓库做过
少量优化。本文件是这些改动的**权威记录**：改了哪个文件、改了什么、为什么、是否已合入 main。

**状态：🟡 文件级清单已到手（2026-07-23 服务器回传 `git status`/diffstat）；
逐行内容等 `server/b100-local-opts` 快照推上来后补全。**

## 基本事实（2026-07-23 确认）

- b100 的 `MobileViCLIP` **不是本 fork 的 clone，而是上游 `MCG-NJU/MobileViCLIP` 的
  clone**（`origin` 指向上游，所以直接 push 被 403 拒）。所有"本地优化"的 diff 基准是
  **上游 main**，与本 fork 的 `docs/`、`experiments/`、`tools/` 等扩展无关。
- 本 fork 与上游**没有共同 git 历史**（fork 是独立初始化的仓库）。
- 改动主题：**接入 NebulaCV 中文 CLIP 文本编码器 + 中文 InternVid 标注管线**
  （nebulacv-clip-24m 底座、HuggingFace encoder/tokenizer 封装、`mobileviclip_nebulacv`
  训练配置）。
- 首个快照 commit `1dabf71bf395ee6896e5e9d8adacb3870b8ca5ed`（只存在于 b100 本地）**作废**：
  `git add -A` 时把未被 .gitignore 覆盖的 `checkpoints/nebulacv-clip-24m.pt` 权重也收了进去。
  重做时用 `.git/info/exclude` 排除 `checkpoints/`。

## 改动清单（相对上游 main；行数来自服务器回传的 diffstat）

### 修改的已有文件（8 个，共 +160/−50 行）

| 文件 | 改动量 | 内容（待逐行 diff 确认） |
|------|--------|--------------------------|
| `.gitignore` | +1 | 待确认 |
| `dataset/__init__.py` | ~55 行 | 待确认（疑与中文标注/数据集注册相关） |
| `models/backbones/internvideo2/__init__.py` | ~42 行 | 待确认（疑注册 nebulacv 文本塔） |
| `models/backbones/internvideo2/mobileclip/__init__.py` | 4 行 | 待确认 |
| `models/mobileviclip_small.py` | 3 行 | 待确认 |
| `models/mobileviclip_tiny.py` | ~97 行 | 待确认（改动最大的模型文件） |
| `tasks_clip/pretrain.py` | 6 行 | 待确认 |
| `tasks_clip/shared_utils.py` | 2 行 | 待确认 |

### 新增文件（12 个）

| 文件 | 说明 |
|------|------|
| `AGENTS.md` | AI 助手工作说明（b100 侧自建） |
| `checkpoints/nebulacv-clip-24m.pt` | NebulaCV 中文 CLIP 权重（**不进 git**） |
| `docs/nebulacv_chinese_training.md` | 中文训练文档 |
| `models/backbones/internvideo2/mobileclip/configs/nebulacv_clip_24m.json` | 文本塔配置 |
| `models/backbones/internvideo2/mobileclip/hf_encoder.py` | HF 文本编码器封装 |
| `models/backbones/internvideo2/mobileclip/hf_tokenizer.py` | HF tokenizer 封装 |
| `models/checkpoint_utils.py` | 权重加载工具 |
| `preprocess/prepare_internvid_chinese_annotations.py` | 中文 InternVid 标注预处理 |
| `scripts/pretraining/clip/mobileviclip_nebulacv/config.py` | nebulacv 预训练配置 |
| `scripts/pretraining/clip/mobileviclip_nebulacv/run.sh` | nebulacv 预训练启动脚本 |
| `tests/test_checkpoint_utils.py` | 单测 |
| `tests/test_internvid_chinese_annotations.py` | 单测 |

## 待办

1. 服务器按 `live/from-claude.md`「2026-07-23 (4)」块重做快照（排除 `checkpoints/`）并
   推到 fork 的 `server/b100-local-opts` 分支（需要 PAT，见该块）。
2. Claude 收到后 `git diff <snapshot>^1 <snapshot>` 逐行审读，把上表「待确认」填实。
3. 评审处置：合入本 fork main / 仅保留在 b100 / 回退。
