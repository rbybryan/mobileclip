"""InternVideo backbones with lightweight imports for MobileViCLIP."""

from .mobileclip import ClipTokenizer, HFTextEncoder, HFTokenizer, TextTransformer


def __getattr__(name):
    if name == "internvl_clip_6b":
        from .internvl_clip_vision import internvl_clip_6b

        return internvl_clip_6b
    if name in ("pretrain_internvideo2_1b_patch14_224", "pretrain_internvideo2_6b_patch14_224"):
        from . import internvideo2

        return getattr(internvideo2, name)
    if name == "InternVideo2":
        from .internvideo2_clip_vision import InternVideo2

        return InternVideo2
    if name in ("LLaMA", "Tokenizer"):
        from . import internvideo2_clip_text

        return getattr(internvideo2_clip_text, name)
    raise AttributeError(name)


__all__ = [
    "ClipTokenizer",
    "HFTextEncoder",
    "HFTokenizer",
    "TextTransformer",
    "internvl_clip_6b",
    "pretrain_internvideo2_1b_patch14_224",
    "pretrain_internvideo2_6b_patch14_224",
    "InternVideo2",
    "LLaMA",
    "Tokenizer",
]
