import os as __os

from configs.model import *


# The default launcher uses eleven InternVid volume shards.  ANNO_FILES can
# be used for an explicit path-separated list, while ANNO_PATH preserves the
# single-file mode for small/debug datasets.
__anno_dir = __os.environ.get(
    "ANNO_DIR", "../video-text-dataset/internvid-cap-zh"
)
__anno_path = __os.environ.get("ANNO_PATH")
__anno_files = __os.environ.get("ANNO_FILES")
if __anno_files:
    __anno_paths = [path for path in __anno_files.split(__os.pathsep) if path]
elif __anno_path:
    __anno_paths = [__anno_path]
else:
    __anno_paths = [
        __os.path.join(__anno_dir, f"InternVId-FLT_{shard_id}.json")
        for shard_id in range(1, 12)
    ]

__data_root = __os.environ.get("DATA_ROOT", "data/chinese_videos")
train_file = [
    dict(
        anno_path=anno_path,
        data_root=__data_root,
        media_type="video",
        jump_filter=True,
    )
    for anno_path in __anno_paths
]
test_file = dict()
test_types = []
num_workers = 6

stop_key = None
best_key = None

# NebulaCV-CLIP-24M was trained at 256 px with the OpenCLIP normalization.
num_frames = 8
num_frames_test = 8
batch_size = 32
batch_size_test = 64
max_txt_l = 32

inputs = dict(
    image_res=256,
    normalize=True,
    interpolation="bicubic",
    image_mean=(0.48145466, 0.4578275, 0.40821073),
    image_std=(0.26862954, 0.26130258, 0.27577711),
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

model = dict(
    model_cls="MobileViCLIP_Tiny",
    vision_encoder=dict(
        name="nebulacv_clip_s0",
        img_size=256,
        head_drop_path_rate=0.0,
        attn_pool_num_heads=16,
        clip_embed_dim=512,
        align_dim=512,
        head_bias=True,
    ),
    text_encoder=dict(
        name="nebulacv_clip_24m",
        hf_model_name="hfl/minirbt-h288",
        hf_tokenizer_name="hfl/minirbt-h288",
        hf_pooler_type="mean_pooler",
        hf_proj_type="mlp",
        tokenizer_max_length="${max_txt_l}",
        local_files_only=True,
    ),
    temp=1 / 100.0,
    temp_min=1 / 1000.0,
    freeze_vision=False,
    open_vision_clip_projector=True,
    freeze_text=True,
    open_text_projection=True,
    open_text_lora=False,
    checkpoint_format="openclip",
    checkpoint_min_match_ratio=0.98,
    vision_ckpt_path="checkpoints/nebulacv-clip-24m.pt",
    text_ckpt_path="checkpoints/nebulacv-clip-24m.pt",
    load_vision_ckpt_from_internvideo2_stage2=False,
)

criterion = dict(loss_weight=dict(vtc=1.0))

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],
    weight_decay=0.2,
    max_grad_norm=1.0,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.6)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

wandb = dict(enable=False, entity=None, project="NebulaCV-MobileViCLIP")
dist_url = "env://"
device = "cuda"
mode = "pt"

output_dir = "pretrain_mobileviclip_nebulacv_24m"
resume = False
debug = False
log_freq = 10
seed = 42

save_latest = True
save_iter = 5000
save_ckpt_iter = 5000
auto_resume = False
pretrained_path = ""

deepspeed = dict(enable=True, stage=1)
