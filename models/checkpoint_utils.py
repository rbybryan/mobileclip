import logging
import re
from collections.abc import Mapping

import torch


logger = logging.getLogger(__name__)


_FASTVIT_STAGE_MAP = {
    (0, "blocks"): 0,
    (1, "downsample"): 1,
    (1, "blocks"): 2,
    (2, "downsample"): 3,
    (2, "blocks"): 4,
    (3, "downsample"): 5,
    (3, "pos_emb"): 6,
    (3, "blocks"): 7,
}


def _to_legacy_fastvit_names(key):
    """Translate current timm FastViT module names to this repository's names."""
    replacements = (
        ("identity.", "rbr_skip."),
        ("conv_kxk.", "rbr_conv."),
        ("conv_scale.", "rbr_scale."),
        ("large_conv.", "lkb_origin."),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    return key


def _torch_load(path):
    """Load a trusted training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint):
    state_dict = checkpoint
    for key in ("state_dict", "model", "module"):
        if isinstance(state_dict, Mapping) and isinstance(state_dict.get(key), Mapping):
            state_dict = state_dict[key]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return state_dict


def _convert_fastvit_key(key, value):
    if key.startswith("stem."):
        suffix = _to_legacy_fastvit_names(key[len("stem."):])
        return "patch_embed." + suffix, value

    match = re.match(r"stages\.(\d+)\.(blocks|downsample|pos_emb)\.(.+)", key)
    if match:
        stage = int(match.group(1))
        component = match.group(2)
        suffix = match.group(3)
        network_index = _FASTVIT_STAGE_MAP.get((stage, component))
        if network_index is None:
            return None, value
        if component == "blocks":
            suffix = suffix.replace(".mlp.", ".convffn.")
            suffix = suffix.replace(".token_mixer.layer_scale.gamma", ".token_mixer.layer_scale")
            suffix = suffix.replace(".layer_scale_1.gamma", ".layer_scale_1")
            suffix = suffix.replace(".layer_scale_2.gamma", ".layer_scale_2")
            suffix = suffix.replace(".layer_scale.gamma", ".layer_scale")
        elif component == "pos_emb":
            suffix = suffix.replace("pos_enc.", "pe.")
        suffix = _to_legacy_fastvit_names(suffix)
        return f"network.{network_index}.{suffix}", value

    if key.startswith("final_conv."):
        suffix = _to_legacy_fastvit_names(key[len("final_conv."):])
        suffix = suffix.replace("se.fc1.", "se.reduce.")
        suffix = suffix.replace("se.fc2.", "se.expand.")
        return "conv_exp." + suffix, value
    if key == "head.fc.weight":
        return "head.proj", value.t().contiguous()
    if key == "head.fc.bias":
        return "head.bias", value
    return None, value


def convert_openclip_mobileclip_state_dict(checkpoint):
    """Convert OpenCLIP MobileCLIP and native MobileCLIP keys to MobileViCLIP."""
    converted = {}
    for raw_key, value in _extract_state_dict(checkpoint).items():
        key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key

        if key.startswith("visual.trunk."):
            vision_key, value = _convert_fastvit_key(key[len("visual.trunk."):], value)
            if vision_key is not None:
                converted["vision_encoder." + vision_key] = value
        elif key.startswith("image_encoder.model."):
            converted["vision_encoder." + key[len("image_encoder.model."):]] = value
        elif key.startswith("text."):
            converted["text_encoder." + key[len("text."):]] = value
        elif key.startswith(("vision_encoder.", "text_encoder.")) or key == "temp":
            converted[key] = value
        elif key == "logit_scale":
            converted["temp"] = value.float().neg().exp()
    return converted


def load_openclip_mobileclip_checkpoint(
    model,
    vision_checkpoint,
    text_checkpoint,
    extra_checkpoint=None,
    min_match_ratio=0.9,
):
    """Load converted image/text weights and reject silently incompatible checkpoints."""
    checkpoint_cache = {}

    def converted(path):
        if path not in checkpoint_cache:
            logger.info("Loading trusted checkpoint from %s", path)
            checkpoint_cache[path] = convert_openclip_mobileclip_state_dict(_torch_load(path))
        return checkpoint_cache[path]

    candidates = {}
    vision_state = converted(vision_checkpoint)
    candidates.update(
        (key, value)
        for key, value in vision_state.items()
        if key.startswith("vision_encoder.") or key == "temp"
    )
    text_state = converted(text_checkpoint)
    candidates.update(
        (key, value) for key, value in text_state.items() if key.startswith("text_encoder.")
    )
    if extra_checkpoint:
        candidates.update(converted(extra_checkpoint))

    target_state = model.state_dict()
    compatible = {}
    shape_mismatches = {}
    unexpected = []
    for key, value in candidates.items():
        if key not in target_state:
            unexpected.append(key)
        elif target_state[key].shape != value.shape:
            shape_mismatches[key] = (tuple(value.shape), tuple(target_state[key].shape))
        else:
            compatible[key] = value

    match_ratio = len(compatible) / max(len(candidates), 1)
    vision_matches = sum(key.startswith("vision_encoder.") for key in compatible)
    text_matches = sum(key.startswith("text_encoder.") for key in compatible)
    if vision_matches == 0 or text_matches == 0 or match_ratio < min_match_ratio:
        raise RuntimeError(
            "Checkpoint conversion matched "
            f"{len(compatible)}/{len(candidates)} tensors ({match_ratio:.1%}); "
            f"vision={vision_matches}, text={text_matches}, "
            f"shape_mismatches={len(shape_mismatches)}, unexpected={len(unexpected)}"
        )

    incompatible = model.load_state_dict(compatible, strict=False)
    logger.info(
        "Loaded %d/%d converted tensors (%.1f%%): vision=%d, text=%d, "
        "shape mismatches=%d, unexpected=%d, model missing=%d",
        len(compatible),
        len(candidates),
        100 * match_ratio,
        vision_matches,
        text_matches,
        len(shape_mismatches),
        len(unexpected),
        len(incompatible.missing_keys),
    )
    if shape_mismatches:
        logger.warning("Checkpoint shape mismatches: %s", shape_mismatches)
    if unexpected:
        logger.warning("Unmapped checkpoint tensors: %s", unexpected)
    return {
        "matched": len(compatible),
        "candidates": len(candidates),
        "match_ratio": match_ratio,
        "vision_matches": vision_matches,
        "text_matches": text_matches,
        "shape_mismatches": shape_mismatches,
        "unexpected": unexpected,
        "missing": incompatible.missing_keys,
    }
