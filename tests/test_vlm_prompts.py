"""Prompt invariants: .format-consumed prompts need doubled braces; raw
prompts must not contain format placeholders (a stray brace would crash
.format or leak literal braces into the model input)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "recaption", "scripts"))
from vlm_audit import (CONDENSE_PROMPT, DERIVE_PROMPT,  # noqa: E402
                       FUSE_LONG_KEYS, FUSE_LONG_PROMPT, FUSE_SHORT_KEYS,
                       FUSE_SHORT_PROMPT, RECAP_PROMPT, REFINE_KEYS,
                       REFINE_PROMPT, SLOT_RECAP_PROMPT)


def test_refine_prompt_formats_cleanly():
    assert REFINE_KEYS == {"views"}
    filled = REFINE_PROMPT.format(items="1. 男子切菜（未通过: 画面检验）")
    assert '"views"' in filled and "{items}" not in filled


def test_fuse_long_prompt_formats_cleanly():
    # §5.7 v5 call 1: judgment fusion -> long draft, NO length cap
    assert FUSE_LONG_KEYS == {"long"}
    filled = FUSE_LONG_PROMPT.format(n=3, recaps="描述1：x")
    assert '"long"' in filled and '"dropped"' in filled
    assert '"short"' not in filled   # v5: 长短稿分开生成 (maintainer 2026-07-28)
    assert "不设限制" in filled       # 长稿不用字数限制
    assert "60-120" not in filled
    assert "{" + "n}" not in filled  # no unfilled placeholder survives


def test_fuse_short_prompt_formats_cleanly():
    # §5.7 v5 call 2: distill the long draft -> ONE <=30-char training caption
    assert FUSE_SHORT_KEYS == {"short"}
    filled = FUSE_SHORT_PROMPT.format(long="一段完整长描述")
    assert '"short"' in filled and '"gloss"' in filled
    assert "30 个字" in filled       # 32-token 文本编码器墙 -> ≤30 字硬上限
    assert '"views"' not in filled and '"long"' not in filled
    assert "{long}" not in filled    # placeholder consumed


def test_slot_prompt_is_raw_with_single_braces():
    """B 臂 slot 提示不经 .format，示例里的方括号/引号必须原样保留。"""
    assert "{" not in SLOT_RECAP_PROMPT          # 无占位符 -> 直接送模型
    for slot in ("人物", "物体", "动作", "场景", "文字"):
        assert f"{slot}：" in SLOT_RECAP_PROMPT
    assert "[man,apron]" in SLOT_RECAP_PROMPT    # 行尾 gloss 的示例


def test_format_prompts_have_doubled_braces():
    # these go through .format -> literal JSON braces must be doubled
    assert "{{" in DERIVE_PROMPT and "{dense}" in DERIVE_PROMPT
    assert "{{" in CONDENSE_PROMPT and "{units}" in CONDENSE_PROMPT


def test_dense_recap_prompt_is_plain_text():
    assert "{" not in RECAP_PROMPT
