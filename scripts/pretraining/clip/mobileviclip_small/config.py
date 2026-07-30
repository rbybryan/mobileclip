from configs.data import *
from configs.model import *

# ========================= data ==========================
train_corpus = "internvid_all"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
# 键必须落在 test_types 里,否则 epoch 末的评测会被整个 `continue` 掉。
# 原来写的是 act_val 而 test_types 是 ["ret_test"] ⇒ sub1m_r0_s42 白跑 15 小时;
# 2026-07-29 加的守卫(pretrain.py:248)会在第一个 step 前直接 raise —— 但基线 config
# 本身一直没跟着改,于是守卫上线后**预训练完全跑不起来**。2026-07-30 一并修掉。
# 见 TROUBLESHOOTING.md「训练跑完但没有任何评测数字」。
test_file = dict(ret_test=available_corpus["msrvtt_1k_test"])
test_types = ["ret_test"]
num_workers = 6 

stop_key = None
best_key = ["msrvtt_1k_test_match", "t2v_r1"]
# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 64  # 8x V100-32GB -> global batch 8*64=512 (same as paper)
batch_size_test = 64
max_txt_l = 32

inputs = dict(
    image_res=256, 
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

# ========================= model ==========================
model = dict(
    model_cls="MobileViCLIP_Small",
    vision_encoder=dict(
        name="mobileclip_s2",
        img_size=256, 
        head_drop_path_rate=0.,
        attn_pool_num_heads=16,
        clip_embed_dim=512, 
        align_dim=512,
    ),
    text_encoder=dict(
        name="mobileclip_s2"
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_vision=False,
    open_vision_clip_projector=True,
    freeze_text=True,
    open_text_projection=True,
    open_text_lora=False,
    vision_ckpt_path="checkpoints/mobileclip_s2.pt",
    load_vision_ckpt_from_internvideo2_stage2=False,
    text_ckpt_path="checkpoints/mobileclip_s2.pt",
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
    ),  # 0: disabled.
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.2,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.6)


evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

# V100 (Volta) has no bfloat16 tensor cores -> use fp16, not bf16.
# NOTE: `use_bf16` is the single precision key, read by utils/config_utils.py
# (DeepSpeed fp16/bf16 block), tasks_clip/pretrain.py (autocast dtype) and
# tasks_clip/shared_utils.py (model cast; its default is True, so set False
# explicitly here). Legacy `bf16` is kept only as a fallback spelling for
# shared_utils.py in old configs/commands - keep the two in sync if set.
use_half_precision = True
use_bf16 = False
bf16 = False
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="likunchang",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="InternVideo2_CLIP",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = "pretrain_mobileviclip_small_with_internvid" # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 500
save_ckpt_iter = 500
auto_resume = False#True
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=True,
    stage=1,
    # fp16 动态 loss scaling 的**初始**缩放系数(2 的幂)。只影响数值,不影响
    # 数学——梯度在更新前会被反缩放回去,所以调它不改变实验语义,只决定开头
    # 要浪费几步在退避上。
    # DeepSpeed 默认 16(=65536)在这个模型上过高:可学习的 `temp` 初值 0.01,
    # 其梯度带 1/temp² = 1e4 的放大,**随机初始化**下 |dL/dtemp| ≈ 1.2e3(B=512),
    # fp16 上限 65504 ⇒ 允许的最大 scale 约 54,从 65536 起要连续减半 11 次。
    # 而冒烟只跑 5 步,全耗在退避上就什么都看不到。
    #
    # 但那是**随机初始化的最坏情况**:两个塔都从 mobileclip_s2.pt 载入、开训前
    # 就已对齐(试点开局 loss = 1.13,不是随机初始化预测的 ~13.9),真实梯度远小
    # 于该估计。2026-07-22 实测:scale=32 全程**零** OVERFLOW。
    # 8(=256)因此是有余量的保守值。详见 TROUBLESHOOTING.md「反向出 NaN」一节。
    #
    # ⚠️ A/B 两臂必须用**同一个值**:溢出会跳步,两臂跳步数不同就不再是等步数对照。
    initial_scale_power=8,
)

# ========== optimization-component switches (defaults = original baseline) ==========
# Set via experiment config or CLI override (e.g. `model.vision_encoder.temporal_pooling attn`)
# when training with / evaluating a checkpoint that uses these components.
model["use_sigmoid_loss"] = False
model["vision_encoder"]["temporal_pooling"] = "mean"
