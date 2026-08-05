"""Unit tests for the A' sentence-split rule (recaption/scripts/split_views.py).

Pure-function tests using the char-budget fallback (no transformers needed):
    python tests/test_split_views.py   # or pytest tests/test_split_views.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "recaption", "scripts"))
from split_views import split_dense  # noqa: E402

FITS30 = lambda t: len(t) <= 30  # noqa: E731  char-budget stand-in for MiniRBT@32


def test_basic_sentence_split():
    text = "一名男子先在厨房切菜。接着他把菜倒入锅中翻炒。最后他把菜装盘。"
    views, dropped = split_dense(text, FITS30)
    assert views == ["一名男子先在厨房切菜。", "接着他把菜倒入锅中翻炒。", "最后他把菜装盘。"]
    assert dropped == 0


def test_overlong_sentence_packs_on_commas():
    text = "画面里有一名穿红色上衣的女子，她坐在窗边的木桌前，桌上放着一杯咖啡和一台银色笔记本电脑，窗外是傍晚的街道。"
    views, dropped = split_dense(text, FITS30)
    assert views, "comma packing must produce views"
    assert all(len(v) <= 30 for v in views)
    assert dropped == 0
    assert "红色上衣" in "".join(views) and "笔记本电脑" in "".join(views)


def test_single_overlong_clause_dropped():
    text = "这是一段完全没有任何标点符号因此绝对无法在预算之内被切开的超长连续中文描述文字所以只能被丢弃"
    views, dropped = split_dense(text, FITS30)
    assert views == []
    assert dropped == 1


def test_filters_short_and_punct_and_dedupes():
    text = "嗯。一名男子在跑步。！！！一名男子在跑步。"
    views, _ = split_dense(text, FITS30)
    assert views == ["一名男子在跑步。"]


def test_empty_and_none_safe():
    assert split_dense("", FITS30) == ([], 0)
    assert split_dense(None, FITS30) == ([], 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} tests passed")
