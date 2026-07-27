"""Prompt invariants: .format-consumed prompts need doubled braces; raw
prompts must not contain format placeholders (a stray brace would crash
.format or leak literal braces into the model input)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "data_audit"))
from vlm_audit import (CONDENSE_PROMPT, DERIVE_PROMPT,  # noqa: E402
                      FUSE_DIRECT_KEYS, FUSE_DIRECT_PROMPT, RECAP_PROMPT,
                      STRUCT_KEYS, STRUCT_RECAP_PROMPT)


def test_fuse_direct_prompt_formats_cleanly():
    # .format-consumed: doubled braces + exactly the {n}/{recaps} slots
    assert FUSE_DIRECT_KEYS == {"views"}
    filled = FUSE_DIRECT_PROMPT.format(n=3, recaps="描述1：x")
    assert '"views"' in filled and '"dropped"' in filled
    assert "{" + "n}" not in filled  # no unfilled placeholder survives


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
