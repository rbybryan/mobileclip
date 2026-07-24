"""InternVideo backbones, imported lazily so the MobileViCLIP path stays light.

Only `internvideo2_clip_text` needs `peft`, and the MobileViCLIP towers never
touch `LLaMA`/`Tokenizer`. Importing it eagerly here dragged `peft` — and
through it a `transformers` version conflict — into every training run; see
TROUBLESHOOTING.md「冒烟 import 崩在 peft / transformers `Cache`」.
"""

from .mobileclip import ClipTokenizer, TextTransformer


def __getattr__(name):
    if name == "internvl_clip_6b":
        from .internvl_clip_vision import internvl_clip_6b

        return internvl_clip_6b
    if name in (
        "pretrain_internvideo2_1b_patch14_224",
        "pretrain_internvideo2_6b_patch14_224",
    ):
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
    "TextTransformer",
    "internvl_clip_6b",
    "pretrain_internvideo2_1b_patch14_224",
    "pretrain_internvideo2_6b_patch14_224",
    "InternVideo2",
    "LLaMA",
    "Tokenizer",
]
