"""Unit tests for the summary-table renderer (pure text functions --
GIF encoding needs a video backend and is exercised on the server)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "data_audit"))
from make_summary_table import cell_caps, deco, safe_name  # noqa: E402


def test_cell_caps_numbers_one_item_per_caption():
    out = cell_caps("一名男子在厨房切菜；随后装盘 ‖ 男子先洗菜")
    assert out.startswith("<ol>") and out.count("<li>") == 2
    assert "切菜；随后装盘" in out          # ；stays INSIDE one item


def test_cell_caps_single_and_empty():
    assert cell_caps("只有一条") == "只有一条"
    assert cell_caps("") == ""
    assert cell_caps(None) == ""


def test_deco_badges_and_escaping():
    out = deco("【人物】一名<男子> & 【物体】菜刀")
    assert '<span class="slot">人物</span>' in out
    assert '<span class="slot">物体</span>' in out
    assert "&lt;男子&gt;" in out and "&amp;" in out   # escaped BEFORE badging


def test_deco_legacy_subject_badge():
    assert '<span class="slot">主体</span>' in deco("【主体】旧缓存行")


def test_safe_name_flattens_paths():
    assert safe_name("videos/a b/c.mp4") == "videos_a_b_c.mp4.gif"
