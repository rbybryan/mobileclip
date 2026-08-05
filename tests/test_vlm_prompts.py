"""Prompt invariants: .format-consumed prompts need doubled braces; raw
prompts must not contain format placeholders (a stray brace would crash
.format or leak literal braces into the model input)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "recaption", "scripts"))
from vlm_audit import (CONDENSE_PROMPT, DERIVE_PROMPT,  # noqa: E402
                      FUSE_LONG_KEYS, FUSE_LONG_PROMPT, FUSE_SHORT_KEYS,
                      FUSE_SHORT_PROMPT, RECAP_PROMPT,
                      REFINE_KEYS, REFINE_PROMPT, STRUCT_KEYS,
                      STRUCT_RECAP_PROMPT, salvage_struct_items)


def test_refine_prompt_formats_cleanly():
    assert REFINE_KEYS == {"views"}
    filled = REFINE_PROMPT.format(items="1. 男子切菜（未通过: 画面检验）")
    assert '"views"' in filled and "{items}" not in filled


def test_salvage_struct_items_rescues_valid_objects():
    broken = ('{"captions": [{"attr": "subject", "text": "男子在切菜", '
              '"gloss": ["man"]}, {"attr": "scene", "text": "厨房里", '
              '"gloss": ["kitchen"]}, {"attr": "text", "text": ')  # truncated
    items = salvage_struct_items(broken)
    assert [i["text"] for i in items] == ["男子在切菜", "厨房里"]


def test_salvage_struct_items_empty_on_garbage():
    assert salvage_struct_items("") == []
    assert salvage_struct_items("完全不是JSON的输出") == []
    assert salvage_struct_items('{"text": "   "}') == []


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


def test_struct_prompt_is_raw_with_single_braces():
    # used WITHOUT .format -> JSON example must keep single braces
    assert "{{" not in STRUCT_RECAP_PROMPT
    assert '"captions"' in STRUCT_RECAP_PROMPT
    assert STRUCT_KEYS == {"captions"}


def test_struct_prompt_covers_attr_vocab():
    for attr in ("subject", "action", "scene", "text"):
        assert attr in STRUCT_RECAP_PROMPT


def test_format_prompts_have_doubled_braces():
    # these go through .format -> literal JSON braces must be doubled
    assert "{{" in DERIVE_PROMPT and "{dense}" in DERIVE_PROMPT
    assert "{{" in CONDENSE_PROMPT and "{units}" in CONDENSE_PROMPT


def test_dense_recap_prompt_is_plain_text():
    assert "{" not in RECAP_PROMPT
