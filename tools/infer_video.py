"""Zero-shot inference for MobileViCLIP on a single custom video.

Given one video and a list of text labels/captions, this loads the authors'
checkpoints (via a standard eval config) and prints the video<->text similarity,
reusing the exact model + preprocessing path from `tasks_clip/retrieval.py`.

Example:
    export PYTHONPATH=${PYTHONPATH}:.
    python tools/infer_video.py \
        --config scripts/evaluation/clip/zero_shot/mobileviclip_tiny/config_msrvtt.py \
        --video path/to/clip.mp4 \
        --labels "a person cooking" "a dog running" "people playing soccer"

Notes:
- The config supplies model_cls, vision/text base weights (vision_ckpt_path /
  text_ckpt_path) and the full MobileViCLIP weights (pretrained_path). Point
  --config at any zero_shot config; tiny/small are auto-detected from it.
- Preprocessing mirrors the eval pipeline: `sample_type_test` frame sampling
  (default "middle"), bilinear resize to `inputs.image_res`, div-by-255, and
  NO mean/std normalization (matches `dataset/__init__.py: get_test_transform`).
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Repo imports (require PYTHONPATH=.)
from models import *  # noqa: F401,F403  -> registers MobileViCLIP_Tiny / _Small
from utils.config import Config, eval_dict_leaf
from dataset.video_utils import read_frames_decord
from models.backbones.internvideo2.mobileclip.modules.common.mobileone import (
    reparameterize_model,
)

try:
    from spikingjelly.clock_driven import functional as sj_functional
except Exception:  # spikingjelly optional at import time
    sj_functional = None


def load_config(config_path):
    cfg = Config.from_file(config_path)
    cfg = eval_dict_leaf(cfg)  # resolve ${...} lazy references
    return cfg


def build_model(cfg, device):
    """Build model, load base + pretrained weights, reparameterize. Mirrors
    tasks_clip/shared_utils.py:setup_model (eval path, no DDP/deepspeed)."""
    model_cls = eval(cfg.model.get("model_cls", "MobileViCLIP"))
    # __init__ loads vision_ckpt_path / text_ckpt_path (base MobileCLIP weights)
    model = model_cls(config=cfg, is_pretrain=False)

    # Load the full MobileViCLIP weights (pretrained_path), same key remapping.
    pretrained_path = cfg.get("pretrained_path", "")
    if os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "module" in ckpt:
            state_dict = ckpt["module"]
        else:
            state_dict = ckpt
        remapped = {}
        for k, v in state_dict.items():
            if "image_encoder.model" in k:
                remapped[k.replace("image_encoder.model.", "vision_encoder.")] = v
            else:
                remapped[k] = v
        msg = model.load_state_dict(remapped, strict=False)
        print(f"[infer] loaded pretrained: {pretrained_path}")
        print(f"[infer] load_state_dict: missing={len(msg.missing_keys)} "
              f"unexpected={len(msg.unexpected_keys)}")
    else:
        print(f"[infer][WARN] pretrained_path not found: {pretrained_path!r} "
              f"(using base weights only)")

    model = reparameterize_model(model)  # eval uses reparameterize=True
    model = model.to(device).eval()
    return model


def get_eval_transform(cfg):
    """Replicate dataset/__init__.py:get_test_transform (normalize is disabled)."""
    image_res = cfg.inputs.image_res
    return transforms.Compose([
        transforms.Resize(
            (image_res, image_res), interpolation=InterpolationMode.BILINEAR
        ),
        transforms.Lambda(lambda x: x.float().div(255.0)),
    ])


def load_video_tensor(cfg, video_path):
    """Return preprocessed video tensor of shape [1, T, C, H, W]."""
    num_frames = cfg.inputs.video_input.num_frames_test
    sample_type = cfg.inputs.video_input.get("sample_type_test", "middle")
    # frames: uint8 (T, C, H, W)
    frames, frame_indices, _ = read_frames_decord(
        video_path, num_frames=num_frames, sample=sample_type
    )
    transform = get_eval_transform(cfg)
    frames = transform(frames)              # (T, C, H, W) float in [0,1]
    frames = frames.unsqueeze(0)            # (1, T, C, H, W)
    print(f"[infer] sampled {len(frame_indices)} frames "
          f"(sample='{sample_type}') indices={list(frame_indices)}")
    return frames


@torch.no_grad()
def run(cfg, model, device, video_path, labels):
    video = load_video_tensor(cfg, video_path).to(device)

    # Vision feature (test=True returns pooled [1, C]); reset spiking neurons.
    video_feat = model.encode_vision(video, test=True)
    if sj_functional is not None:
        sj_functional.reset_net(model)

    # Text features: MobileViCLIP uses its own ClipTokenizer on a list of strings.
    text_input = model.tokenizer(list(labels)).to(device)
    text_feat = model.encode_text(text_input)
    if isinstance(text_feat, (tuple, list)):
        text_feat = text_feat[0]

    video_feat = F.normalize(video_feat.float(), dim=-1)   # [1, C]
    text_feat = F.normalize(text_feat.float(), dim=-1)     # [N, C]

    sims = (video_feat @ text_feat.T).squeeze(0)           # [N]
    probs = sims.softmax(dim=-1)

    order = torch.argsort(sims, descending=True)
    print("\n=== Zero-shot results (video vs. text) ===")
    print(f"video: {video_path}\n")
    print(f"{'rank':<5}{'similarity':<12}{'softmax':<10}label")
    for rank, i in enumerate(order.tolist(), 1):
        print(f"{rank:<5}{sims[i].item():<12.4f}{probs[i].item()*100:<9.2f}% "
              f"{labels[i]}")
    return sims.cpu(), probs.cpu()


def main():
    ap = argparse.ArgumentParser(description="MobileViCLIP single-video zero-shot inference")
    ap.add_argument("--config", required=True,
                    help="path to a zero_shot config_*.py (tiny or small)")
    ap.add_argument("--video", required=True, help="path to a video file")
    ap.add_argument("--labels", required=True, nargs="+",
                    help="candidate text labels / captions")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.device = args.device
    device = torch.device(args.device)

    model = build_model(cfg, device)
    run(cfg, model, device, args.video, args.labels)


if __name__ == "__main__":
    main()
