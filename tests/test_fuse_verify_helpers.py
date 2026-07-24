"""Unit tests for tools/fuse_verify.py pure helpers (no torch/GPU needed).

    python tests/test_fuse_verify_helpers.py   # or pytest
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from fuse_verify import fuzzy_contains, quoted_spans  # noqa: E402


def test_quoted_spans():
    assert quoted_spans('屏幕上显示“季度报告”和“2024”。') == ["季度报告", "2024"]
    assert quoted_spans("没有引号的句子") == []
    assert quoted_spans("") == [] and quoted_spans(None) == []


def test_fuzzy_exact_and_substring():
    assert fuzzy_contains("季度报告", "xx季度报告yy") == 1.0
    assert fuzzy_contains("", "abc") == 1.0
    assert fuzzy_contains("abc", "") == 0.0


def test_fuzzy_tolerates_one_char_ocr_miss():
    assert fuzzy_contains("季度报告", "xx季度报吿yy") >= 0.7


def test_fuzzy_rejects_unrelated():
    assert fuzzy_contains("季度报告", "完全无关的文字串") < 0.5


def test_fuzzy_ignores_whitespace():
    assert fuzzy_contains("季度 报告", "季度报告") == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} tests passed")
