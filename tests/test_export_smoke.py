"""导出冒烟脚本的报告逻辑测试(纯 CPU,不需要任何导出后端)。

这张表是 **P9 触发判据的全部依据**(评审 U1),所以它的失败模式不是"跑不了",
而是**把状态记错**:

- 缺环境记成 fail ⇒ **假的 P9 触发**,R3 架构决策建立在不存在的证据上;
- 导不出记成 skip ⇒ 漏掉真的触发。

这两个方向都在下面钉死。算子抽取则决定失败清单有没有用。
"""
import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "export_smoke", os.path.join(_ROOT, "experiments", "export_smoke.py")
)
es = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(es)


# ------------------------------------------------------------ 算子抽取
@pytest.mark.parametrize("text,expected", [
    ("RuntimeError: Exporting the operator 'aten::unflatten' to ONNX opset 17", "unflatten"),
    ("torch.onnx.errors.UnsupportedOperatorError: Unsupported: ONNX export of "
     "operator adaptive_avg_pool3d", "adaptive_avg_pool3d"),
    ('ValueError: Op "einsum" is not implemented in coremltools', "einsum"),
    ("NotImplementedError: conversion for torch operator 'scaled_dot_product_attention' "
     "not implemented", "scaled_dot_product_attention"),
    ("PyTorch convert function for op 'repeat_interleave' not implemented", "repeat_interleave"),
])
def test_extract_ops_finds_the_operator(text, expected):
    assert expected in es.extract_ops(text)


def test_extract_ops_dedups_and_keeps_order():
    text = "aten::foo failed; aten::bar failed; aten::foo again"
    assert es.extract_ops(text) == ["foo", "bar"]


def test_extract_ops_empty_on_unrelated_text():
    assert es.extract_ops("Segmentation fault (core dumped)") == []
    assert es.extract_ops("") == []
    assert es.extract_ops(None) == []


# ------------------------------------------------------------ 状态归类(最关键)
def _run(route_fn, deps=("os",)):
    """用假路线跑一次 run_one,拿到状态归类结果。"""
    return es.run_one("fake", route_fn, "bin", deps, None, None, "/tmp", 8, 17)


def test_missing_declared_dep_is_skip_not_fail():
    r = es.run_one("fake", lambda *a: {}, "bin", ("definitely_not_installed_xyz",),
                   None, None, "/tmp", 8, 17)
    assert r["status"] == "skip"
    assert "definitely_not_installed_xyz" in r["reason"]


def test_missing_transitive_dep_is_skip_not_fail():
    """回归:torch 2.12 的 ONNX 导出器缺 onnxscript 时抛 ModuleNotFoundError。

    第一版把它记成 fail —— 那会让「环境缺件」伪装成「导不出」,产生假的 P9 触发。
    """
    def boom(*_):
        raise ModuleNotFoundError("No module named 'onnxscript'", name="onnxscript")
    r = _run(boom)
    assert r["status"] == "skip", "缺传递依赖必须是 skip,否则是假 P9 触发"
    assert "onnxscript" in r["reason"]
    assert r["ops"] == []


def test_real_export_failure_is_fail_with_ops():
    """真的导不出必须记 fail 并带算子名 —— 否则漏掉真的 P9 触发。"""
    def boom(*_):
        raise RuntimeError("Exporting the operator 'aten::unflatten' to ONNX opset 17")
    r = _run(boom)
    assert r["status"] == "fail"
    assert "unflatten" in r["ops"]
    assert r["traceback_tail"]


def test_route_success_is_pass():
    r = _run(lambda *a: {"bytes": 123})
    assert r["status"] == "pass"
    assert r["bytes"] == 123


def test_one_route_failing_does_not_raise():
    """三条路线必须互不影响 —— run_one 永不抛出。"""
    for exc in (RuntimeError("x"), ValueError("y"), KeyboardInterrupt):
        def boom(*_, _e=exc):
            raise _e if not isinstance(_e, type) else _e()
        if exc is KeyboardInterrupt:
            with pytest.raises(KeyboardInterrupt):   # 中断必须透传,不能被吞
                _run(boom)
        else:
            assert _run(boom)["status"] == "fail"


# ------------------------------------------------------------ 报告渲染
def _results(status_by_route):
    return {"8": {name: dict(status=s, reason="缺依赖 foo", ops=["aten::bar"])
                  for name, s in status_by_route.items()}}


def test_render_marks_all_three_statuses_distinctly():
    res = _results({"onnx": "pass", "tflite": "fail", "coreml": "skip"})
    out = es.render(res, [8], {"model": "m", "inference_mode": False, "batch": 1,
                               "img": 256, "opset": 17, "torch": "2.12"})
    assert "✅ pass" in out and "❌ fail" in out and "⏭️ skip" in out
    assert "别把 skip 读成 pass" in out          # 判读指引必须在


def test_render_lists_failing_ops_for_p9_decision():
    res = _results({"onnx": "fail", "tflite": "pass", "coreml": "pass"})
    out = es.render(res, [8], {"model": "m", "inference_mode": False, "batch": 1,
                               "img": 256, "opset": 17, "torch": "2.12"})
    assert "失败算子清单" in out
    assert "bar" in out


def test_render_handles_no_failures():
    res = _results({"onnx": "pass", "tflite": "pass", "coreml": "pass"})
    out = es.render(res, [8], {"model": "m", "inference_mode": False, "batch": 1,
                               "img": 256, "opset": 17, "torch": "2.12"})
    assert "无失败路线" in out


# ------------------------------------------------------------ 模型构造
def test_mini_tower_forward_shape():
    """包装后的输入签名必须是 [B,T,C,H,W](与 encode_vision 对齐),输出 [B,512]。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    tower = es._wrap(es.build_vision_tower(mini=True))
    for t in (2, 8):
        with torch.no_grad():
            out = tower(torch.randn(1, t, 3, 64, 64))
        assert out.shape == (1, 512), f"T={t} -> {out.shape}"


def test_full_size_params_match_build_vision_encoder():
    """全尺寸架构参数必须与 models/mobileviclip_small.py:build_vision_encoder 一致,
    否则导出结论对不上真模型。用源码文本核对,不需要建全尺寸模型。
    """
    src = open(os.path.join(_ROOT, "models", "mobileviclip_small.py")).read()
    body = src[src.index("def build_vision_encoder"):]
    body = body[:body.index("def build_text_encoder")]
    for token in ("[4, 12, 24, 4]", "[80, 160, 320, 640]", "[3, 3, 3, 3]",
                  "[False, False, True, True]", "spatial_shape=(7, 7)"):
        assert token in body, f"build_vision_encoder 里找不到 {token},export_smoke 需同步"
    es_src = open(os.path.join(_ROOT, "experiments", "export_smoke.py")).read()
    for token in ("[4, 12, 24, 4]", "[80, 160, 320, 640]", "[3, 3, 3, 3]",
                  "[False, False, True, True]", "spatial_shape=(7, 7)"):
        assert token in es_src, f"export_smoke 里找不到 {token},与真模型不一致"
