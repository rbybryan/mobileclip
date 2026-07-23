# NebulaCV 中文 MobileViCLIP 训练

专用配置使用 `NebulaCV-CLIP-24M` 的 FastViT-MCI0、MiniRBT-H288、中文词表、MLP 文本投影和对比学习温度初始化 MobileViCLIP。加载器会转换 OpenCLIP/timm 键名，并在匹配率低于 98% 时终止，防止静默地随机初始化。

训练标注是按 InternVid volume 分片的 JSON 数组，每个文件包含最小的
`video`/`caption` 字段：

```json
[
  {"video": "000001.mp4", "caption": "一只猫在窗边晒太阳"}
]
```

`video` 相对于 `DATA_ROOT`；中文文本保持为 UTF-8。完整 InternVid 中文数据由
`InternVId-FLT_1.json` 到 `InternVId-FLT_11.json` 组成。MiniRBT 最多支持 512
token，专用配置默认截断到 32，可通过配置覆盖。

如需重新生成分片，可运行：

```bash
PYTHONPATH=. python preprocess/prepare_internvid_chinese_annotations.py \
  --source /path/aligned_internvid_zh_translated_retry_merged.jsonl \
  --manual-reviews /path/manual_review_annotations.jsonl \
  --output-dir /path/internvid-cap-zh \
  --uncertain-policy drop
```

脚本默认使用 `raw_caption_zh_translated`，并用人工审核中“通过”或“修改后通过”的
`corrected_translation` 覆盖；“不确定”和“拒绝/删除”样本不进入训练。

从仓库根目录启动：

```bash
conda activate mobileviclip

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

ANNO_DIR=/workspace/mount/b100_zaip_data/wky_b100/project/video-text-dataset/internvid-cap-zh \
DATA_ROOT=/workspace/mount/b100_zaip_data_luan/InternVid-10M-FLT \
HF_MODEL_PATH=/workspace/mount/b100_zaip_data/wky_b100/cache/hfhub/hub/models--hfl--minirbt-h288/snapshots/dc4eebb0cf6f9e7094142ac28fbf971517c6a366 \
NUM_GPUS=8 \
bash scripts/pretraining/clip/mobileviclip_nebulacv/run.sh
```

配置会自动加载 `ANNO_DIR` 下的 11 个分片。小规模调试时也可以设置
`ANNO_PATH=/path/train.json` 使用单个 JSON 数组，或用 `ANNO_FILES`（以系统路径分隔符
分隔）指定任意分片列表。

权重默认读取 `checkpoints/nebulacv-clip-24m.pt`。`LD_LIBRARY_PATH` 确保 PyAV 使用 Conda 环境中的 C++ 运行库；`HF_MODEL_PATH` 指向包含 `config.json`、`vocab.txt` 的 MiniRBT 本地快照。

首次启动时检查日志中类似 `Loaded 888/888 converted tensors` 的信息。输出写入 `pretrain_mobileviclip_nebulacv_24m/`；恢复训练前设置配置中的 `auto_resume=True`，不要再把 MobileCLIP 初始 checkpoint 当作优化器恢复点。
