"""Unit tests for the C_fuse vote stage (pure functions, no model)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "tools", "data_audit"))
from fuse_condense import (cluster_views, jaccard, ngrams,  # noqa: E402
                           select_units)


def V(text, src, gloss=()):
    return {"text": text, "src_recap": src, "gloss": list(gloss)}


def test_ngrams_strip_punct():
    assert ngrams("男人。切菜！") == ngrams("男人切菜")
    assert ngrams("") == set()


def test_same_fact_merges_across_recaps():
    views = [V("一名男子在厨房的案板上切菜", 0, ["man", "kitchen"]),
             V("男子在厨房案板上切菜", 1, ["man", "kitchen"]),
             V("男子在厨房的案板上切着蔬菜", 2, ["man", "kitchen"]),
             V("窗外是夜晚的街道", 0, ["street"])]
    units = cluster_views(views)
    assert len(units) == 2
    assert units[0]["support"] == 3          # the kitchen fact, 3 recaps
    assert units[1]["support"] == 1          # the street fact, 1 recap
    assert len(units[0]["members"]) == 3


def test_support_counts_distinct_recaps_not_members():
    # two near-identical views from the SAME recap must count support=1
    views = [V("画面下方出现字幕“新品上市”", 0),
             V("画面下方出现字幕“新品上市”啦", 0)]
    units = cluster_views(views)
    assert len(units) == 1
    assert units[0]["support"] == 1


def test_gloss_rescue_low_text_overlap():
    # paraphrases with low char overlap but same detected objects merge via
    # the gloss rescue path (jac_lo <= text-jaccard < jac_th)
    a = V("狗在草地上追飞盘", 0, ["dog", "frisbee"])
    b = V("一只狗追着飞盘跑过草地", 1, ["dog", "frisbee"])
    t = jaccard(ngrams(a["text"]), ngrams(b["text"]))
    assert t < 0.35                          # would NOT merge on text alone
    units = cluster_views([a, b])
    assert len(units) == 1 and units[0]["support"] == 2


def test_no_gloss_no_rescue():
    a = V("狗在草地上追飞盘", 0)
    b = V("一只狗追着飞盘跑过草地", 1)
    assert len(cluster_views([a, b])) == 2   # low text overlap, no gloss


def test_best_recap_representative():
    views = [V("男子在厨房切菜做饭准备晚餐", 0, ["man"]),
             V("男子在厨房切菜", 2, ["man"])]
    units = cluster_views(views, best_recap=2)
    assert units[0]["text"] == "男子在厨房切菜"


def test_empty_and_blank_views():
    assert cluster_views([]) == []
    assert cluster_views([V("", 0), V("  ", 1)]) == []


def test_tiebreak_prefers_more_detectable_nouns():
    # all-tied support: the unit with more gloss nouns (more verifiable, likely
    # the subject fact) must sort first -- it becomes the fallback keeper
    units = cluster_views([V("天空有几只飞鸟", 1, ["bird"]),
                           V("狗在草地上奔跑", 0, ["dog", "grass"])])
    assert units[0]["text"] == "狗在草地上奔跑"


def test_select_units_gate_no_fallback():
    units = cluster_views([V("男子在厨房切菜", 0, ["man", "kitchen"]),
                           V("男子在厨房里切菜", 1, ["man", "kitchen"]),
                           V("窗外是夜晚的街道", 2, ["street"])])
    kept, fb = select_units(units, 2)
    assert not fb and [u["support"] for u in kept] == [2]


def test_select_units_fallback_keeps_top3_not_top1():
    # nothing clears the gate -> coverage-first fallback keeps 3 units, so a
    # clip degrades to "weakly voted" rather than "single sentence"
    units = cluster_views([V("男子在厨房切菜", 0, ["man", "kitchen"]),
                           V("窗外是夜晚的街道", 1, ["street"]),
                           V("桌上放着一杯咖啡", 2, ["cup", "coffee"]),
                           V("天空有几只飞鸟", 3, ["bird"])])
    kept, fb = select_units(units, 2)
    assert fb and len(kept) == 3
    # both 2-gloss subject facts survive; a 1-gloss incidental is what drops
    texts = {u["text"] for u in kept}
    assert {"男子在厨房切菜", "桌上放着一杯咖啡"} <= texts
