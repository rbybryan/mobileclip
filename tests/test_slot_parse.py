"""Unit tests for the §5.6b fixed-slot line parser (pure function)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "data_audit"))
from vlm_audit import (SLOT_RECAP_PROMPT, parse_slot_lines,  # noqa: E402
                       slot_recap_cells)


def test_happy_path():
    out = parse_slot_lines(
        "人物：一名穿红色围裙的中年男子 [man,apron]\n"
        "物体：案板上放着一把菜刀 [knife]\n"
        "动作：先切黄瓜，随后装盘 []\n"
        "场景：明亮的家用厨房 [kitchen]\n"
        "文字：无")
    assert [(v["attr"], v["text"]) for v in out] == [
        ("character", "一名穿红色围裙的中年男子"),
        ("object", "案板上放着一把菜刀"),
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
        "人物：人物一 [a]\n人物：人物二 [b]\n人物：人物三 [c]\n"
        "场景：场景一 [x]\n场景：场景二 [y]\n")
    assert [v["text"] for v in out] == ["人物一", "人物二", "场景一"]
    assert parse_slot_lines("主体：旧栏目名不再解析 [x]") == []  # v2 词表


def test_fullwidth_colon_and_missing_brackets():
    out = parse_slot_lines("动作: 挥手致意\n文字：屏幕显示“你好”")
    assert [(v["attr"], v["gloss"]) for v in out] == [
        ("action", []), ("text", [])]


def test_prompt_is_raw_lines_not_json():
    assert "{" not in SLOT_RECAP_PROMPT
    for slot in ("人物", "物体", "动作", "场景", "文字"):
        assert slot in SLOT_RECAP_PROMPT
    assert "主体：" not in SLOT_RECAP_PROMPT  # v2: Character/Object 分开
    assert "无" in SLOT_RECAP_PROMPT


def test_slot_recap_cells_rebuilds_labels_per_pass():
    # display-side 【属性】 rebuild from cached views (no re-run needed);
    # attr="subject" here is the v1/struct LEGACY name -> must still render 主体
    verdict = {
        "recaps": ["男子切菜；明亮厨房", "男子随后装盘"],
        "views": [
            {"text": "一名男子在切菜", "attr": "subject", "src_recap": 0},
            {"text": "明亮的家用厨房", "attr": "scene", "src_recap": 0},
            {"text": "男子随后把菜装盘", "attr": "action", "src_recap": 1},
        ]}
    cells = slot_recap_cells(verdict)
    assert cells == ["【主体】一名男子在切菜 【场景】明亮的家用厨房",
                     "【动作】男子随后把菜装盘"]


def test_slot_recap_cells_dense_fallback_pass_keeps_plain_text():
    # a pass that fell back to the dense prompt has no attributed views;
    # its cell must keep the plain recap paragraph instead of going empty
    verdict = {
        "recaps": ["【无属性】这遍是稠密兜底段落", "男子装盘"],
        "views": [{"text": "男子把菜装盘", "attr": "action", "src_recap": 1}]}
    cells = slot_recap_cells(verdict)
    assert cells[0] == "【无属性】这遍是稠密兜底段落"
    assert cells[1] == "【动作】男子把菜装盘"


def test_slot_recap_cells_none_for_dense_mode():
    # dense recaps carry no attr/src_recap-int views -> caller falls back
    assert slot_recap_cells({"recaps": ["一段稠密描述"]}) is None
    assert slot_recap_cells({"recaps": ["x"], "views": [
        {"text": "融合句", "src_recap": "fuse"}]}) is None
    assert slot_recap_cells(None) is None


def test_slot_recap_cells_v2_character_object_labels():
    verdict = {
        "recaps": ["x"],
        "views": [
            {"text": "一名男子", "attr": "character", "src_recap": 0},
            {"text": "一把菜刀", "attr": "object", "src_recap": 0},
        ]}
    assert slot_recap_cells(verdict) == ["【人物】一名男子 【物体】一把菜刀"]
