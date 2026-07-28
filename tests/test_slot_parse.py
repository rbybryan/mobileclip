"""Unit tests for the §5.6b fixed-slot line parser (pure function)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "data_audit"))
from vlm_audit import SLOT_RECAP_PROMPT, parse_slot_lines  # noqa: E402


def test_happy_path():
    out = parse_slot_lines(
        "主体：一名穿红色围裙的中年男子 [man,apron]\n"
        "主体：案板上放着一把菜刀 [knife]\n"
        "动作：先切黄瓜，随后装盘 []\n"
        "场景：明亮的家用厨房 [kitchen]\n"
        "文字：无")
    assert [(v["attr"], v["text"]) for v in out] == [
        ("subject", "一名穿红色围裙的中年男子"),
        ("subject", "案板上放着一把菜刀"),
        ("action", "先切黄瓜，随后装盘"),
        ("scene", "明亮的家用厨房")]
    assert out[0]["gloss"] == ["man", "apron"]
    assert out[2]["gloss"] == []


def test_none_slots_and_garbage_skipped():
    out = parse_slot_lines(
        "好的，以下是描述：\n"          # chatter line -> no slot name, ignored
        "文字：无\n"
        "场景：无。\n"                   # 无 with trailing punctuation
        "动作：一只狗跑过草地 [dog,grass]\n"
        "备注：这不是合法栏目\n")
    assert [(v["attr"], v["text"]) for v in out] == [("action", "一只狗跑过草地")]


def test_slot_caps_enforced():
    out = parse_slot_lines(
        "主体：主体一 [a]\n主体：主体二 [b]\n主体：主体三 [c]\n"
        "场景：场景一 [x]\n场景：场景二 [y]\n")
    assert [v["text"] for v in out] == ["主体一", "主体二", "场景一"]


def test_fullwidth_colon_and_missing_brackets():
    out = parse_slot_lines("动作: 挥手致意\n文字：屏幕显示“你好”")
    assert [(v["attr"], v["gloss"]) for v in out] == [
        ("action", []), ("text", [])]


def test_prompt_is_raw_lines_not_json():
    assert "{" not in SLOT_RECAP_PROMPT
    for slot in ("主体", "动作", "场景", "文字"):
        assert slot in SLOT_RECAP_PROMPT
    assert "无" in SLOT_RECAP_PROMPT
