"""汇总脚本的模式回归测试。

存在的理由:第一版 collect_frame_select.py 同时写错了两处,而且**静默失败** ——
一个表都没打,只输出一句「臂不齐,无法判读」,看上去像是数据缺失而不是解析错误。
两处错误都在这里钉死:

1. 文件名:`evaluate: True` 时只写 `eval_res_latest.json`(tasks_clip/pretrain.py),
   不是 `eval_res.json`。
2. 口径键:带尾部斜杠 `ret_test/` / `ret_test_emb/`
   (tasks_clip/retrieval_utils.py:83-84 的 `prefix + "/"`)。

这里的 fixture 用的是 2026-07-27 那轮 msrvtt K=4 三个臂的真实读数,顺带把
「divmax 未过判据 + 地板告警」这条判读路径也走一遍。
"""
import importlib.util
import json
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "collect_frame_select",
    os.path.join(os.path.dirname(__file__), "..", "experiments", "collect_frame_select.py"),
)
cfs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cfs)


def _res(dsl_img_r1, raw_img_r1, dsl_r_mean, raw_r_mean):
    """按 pd.DataFrame(eval_res).to_json() 的真实布局造一份结果。"""
    return {
        "ret_test/": {"txt_r1": 0.0, "img_r1": dsl_img_r1, "r_mean": dsl_r_mean},
        "ret_test_emb/": {"txt_r1": 0.0, "img_r1": raw_img_r1, "r_mean": raw_r_mean},
    }


# 2026-07-27 msrvtt 实测(官方权重,零训练)
ARMS_K4 = {
    "seluniform_K4": _res(41.7, 39.8, 61.17, 60.15),
    "seldivmax_K4": _res(40.2, 39.7, 60.43, 59.92),
    "selrandom_K4": _res(39.2, 38.6, 59.08, 58.55),
}


@pytest.fixture
def root(tmp_path):
    for arm, res in ARMS_K4.items():
        d = tmp_path / f"msrvtt_{arm}"
        d.mkdir()
        (d / "eval_res_latest.json").write_text(json.dumps(res))
    return str(tmp_path)


def test_calibers_carry_trailing_slash():
    """口径键必须带尾斜杠,否则 get() 全取不到值。"""
    assert [c for c, _ in cfs.CALIBERS] == ["ret_test_emb/", "ret_test/"]


def test_latest_is_preferred_result_file():
    """评测模式只产出 eval_res_latest.json,它必须排在第一位。"""
    assert cfs.RESULT_FILES[0] == "eval_res_latest.json"
    assert set(cfs.RESULT_FILES) >= {"eval_res_best.json", "eval_res.json"}


def test_load_arms_reads_latest(root):
    arms = cfs.load_arms(root)
    assert set(arms) == {"msrvtt"}
    assert set(arms["msrvtt"]) == set(ARMS_K4)


def test_load_arms_ignores_dir_without_results(root, tmp_path):
    (tmp_path / "msrvtt_halfbaked").mkdir()
    assert "halfbaked" not in cfs.load_arms(root)["msrvtt"]


def test_get_extracts_real_numbers(root):
    arms = cfs.load_arms(root)["msrvtt"]
    assert cfs.get(arms["seluniform_K4"], "ret_test_emb/", "img_r1") == 39.8
    assert cfs.get(arms["seluniform_K4"], "ret_test/", "img_r1") == 41.7


def test_divmax_fails_the_criterion(root):
    """raw −0.1 / DSL −1.5:既没过噪声带,两口径也不同向 ⇒ 未过判据。"""
    arms = cfs.load_arms(root)["msrvtt"]
    ok, d = cfs.both_calibers_agree(arms, "seldivmax_K4", "seluniform_K4")
    assert ok is False
    assert d == pytest.approx(-0.1, abs=0.01)


def test_floor_check_also_fails(root):
    """uniform 相对 random raw 只有 +1.2,没过 1.6 ⇒ 地板告警必须触发。"""
    arms = cfs.load_arms(root)["msrvtt"]
    ok, d = cfs.both_calibers_agree(arms, "seluniform_K4", "selrandom_K4")
    assert ok is False
    assert d == pytest.approx(1.2, abs=0.01)


def test_main_prints_tables_not_silent_failure(root, capsys):
    """回归核心:必须真的打出表和判读,不能退化成一句「臂不齐」。"""
    cfs.main.__globals__["__name__"]  # noqa: B018 - 保证模块已加载
    import sys

    argv = sys.argv
    sys.argv = ["collect", "--root", root]
    try:
        cfs.main()
    finally:
        sys.argv = argv

    out = capsys.readouterr().out
    assert "raw(主口径)" in out and "`ret_test_emb/`" in out
    assert "39.80" in out  # 数字真的被读出来了
    assert "臂不齐" not in out
    assert "没有一个能取出" not in out  # 静默失败保护未触发
    assert "未过判据" in out  # divmax 判读走到了
    assert "地板先看" in out  # 地板告警触发
