"""选帧诊断脚本的统计与判读逻辑测试(不碰视频,纯 CPU)。

诊断脚本的价值全在「(A)/(B) 两个假说给出相反预测」这一点上 —— 如果统计口径
或判读阈值写错,跑出来的表会把两个机制混掉,而那种错误在真实数据上**看不出来**
(没有 ground truth 可对)。所以这里用**构造好的极端下标**把两条预测各自钉死。
"""
import importlib.util
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "frame_select_diag", os.path.join(_ROOT, "experiments", "frame_select_diag.py")
)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


# ------------------------------------------------------------ 解析基线
def test_endpoint_rate_matches_hand_calculation():
    """n=16,k=2:1 − C(14,2)/C(16,2) = 1 − 91/120 = 29/120。"""
    assert diag.analytic_endpoint_rate(16, 2) == pytest.approx(29 / 120)


def test_endpoint_rate_saturates_when_k_large():
    """k > n−2 时不可能避开两个端点。"""
    assert diag.analytic_endpoint_rate(16, 15) == 1.0
    assert diag.analytic_endpoint_rate(16, 16) == 1.0


def test_endpoint_rate_rejects_bad_k():
    for k in (0, -1, 17):
        with pytest.raises(ValueError):
            diag.analytic_endpoint_rate(16, k)


def test_mean_pair_gap_is_n_plus_1_over_3():
    """E|i−j| = (n+1)/3;n=16 ⇒ 17/3。"""
    assert diag.analytic_mean_pair_gap(16) == pytest.approx(17 / 3)


def test_mean_pair_gap_matches_brute_force():
    """小 n 上与穷举一致,防止公式记反。"""
    from itertools import combinations
    for n in (2, 3, 5, 9, 16):
        pairs = list(combinations(range(n), 2))
        brute = sum(b - a for a, b in pairs) / len(pairs)
        assert diag.analytic_mean_pair_gap(n) == pytest.approx(brute)


# ------------------------------------------------------------ 统计
def test_index_stats_on_uniform_k2():
    """uniform 在 16 帧上取 K=2 恒为 {4,12}:不碰端点,间隔 8。"""
    from utils.elastic_t import subsample_frame_indices
    idx = subsample_frame_indices(16, 2)
    assert idx == [4, 12]
    s = diag.index_stats([idx] * 10, 16)
    assert s["endpoint_rate"] == 0.0
    assert s["mean_pair_gap"] == pytest.approx(8.0)
    assert s["mean_span"] == pytest.approx(8.0)


def test_index_stats_counts_either_endpoint():
    s = diag.index_stats([[0, 5], [5, 15], [6, 7]], 16)
    assert s["endpoint_rate"] == pytest.approx(2 / 3)


def test_index_stats_rejects_empty():
    with pytest.raises(ValueError):
        diag.index_stats([], 16)


# ------------------------------------------------------------ 判读
def test_interior_cv_zero_when_flat():
    """内部完全均匀 ⇒ CV = 0。"""
    assert diag.interior_cv([99] + [10] * 14 + [99]) == pytest.approx(0.0)


def test_interior_cv_ignores_endpoints():
    """端点次数再极端也不该影响内部 CV —— 否则端点偏好会伪装成「内部有结构」。"""
    flat = [10] * 14
    assert diag.interior_cv([0] + flat + [0]) == diag.interior_cv([9999] + flat + [9999])


def test_verdict_endpoint_bias_flat_is_the_measured_case():
    """恒选 {0, 中段}:端点率 1.0、内部只有一个位置 —— 但那正是「平坦」的退化情形。

    用真实实测的形状更有意义:端点过采 + 内部近似均匀。
    """
    import random
    rng = random.Random(0)
    idx = []
    for _ in range(4000):
        if rng.random() < 0.42:                       # 实测 K=2 端点率 0.423
            idx.append(sorted([rng.choice([0, 15]), rng.randrange(1, 15)]))
        else:
            idx.append(sorted(rng.sample(range(1, 15), 2)))
    d = diag.analyze_mechanism(diag.index_stats(idx, 16), 16, 2)
    assert d["verdict"] == "endpoint_bias_flat"
    assert d["endpoint_ratio_vs_random"] > diag.ENDPOINT_BIAS_RATIO
    assert d["interior_cv"] < diag.FLAT_INTERIOR_CV


def test_verdict_uninformative_when_indistinguishable_from_random():
    """纯随机采样 ⇒ 端点率≈理论值、内部平坦 ⇒ 指标不携带信息。"""
    import random
    rng = random.Random(1)
    idx = [sorted(rng.sample(range(16), 2)) for _ in range(8000)]
    d = diag.analyze_mechanism(diag.index_stats(idx, 16), 16, 2)
    assert d["verdict"] == "uninformative"


def test_verdict_structured_no_bias_when_fixed_interior_pair():
    """恒选 {2,13}:不碰端点,但内部极不均匀 ⇒ 有结构、不追端点。"""
    d = diag.analyze_mechanism(diag.index_stats([[2, 13]] * 50, 16), 16, 2)
    assert d["verdict"] == "structured_no_bias"


def test_verdict_endpoint_bias_structured():
    """恒选 {0,7}:端点率 1.0 且内部只压在一个位置 ⇒ 端点偏好 + 内部有结构。"""
    d = diag.analyze_mechanism(diag.index_stats([[0, 7]] * 50, 16), 16, 2)
    assert d["verdict"] == "endpoint_bias_structured"


def test_mean_pair_gap_is_not_used_by_verdict():
    """回归:间隔曾被当判据并导致 K=4 误判(gap 6.33≈随机 5.67 就判了「像素≠时间」,
    而同一批数据端点率是随机的 1.57×)。

    构造一对**端点率相同、内部分布同构、间隔差 4 倍**的输入:恒选 {0,3} vs {0,12}。
    两者内部都是单点尖峰(CV 与位置无关),端点率都是 1.0,间隔 3 vs 12。
    判读必须一致 —— 若还看间隔就会分叉。
    """
    a = diag.analyze_mechanism(diag.index_stats([[0, 3]] * 50, 16), 16, 2)
    b = diag.analyze_mechanism(diag.index_stats([[0, 12]] * 50, 16), 16, 2)
    assert a["interior_cv"] == pytest.approx(b["interior_cv"])
    assert a["endpoint_ratio_vs_random"] == pytest.approx(b["endpoint_ratio_vs_random"])
    assert a["verdict"] == b["verdict"] == "endpoint_bias_structured"
    # 间隔确实差很多,证明这一对确实能暴露「判读看了间隔」这个 bug
    gaps = [diag.index_stats([[0, j]] * 50, 16)["mean_pair_gap"] for j in (3, 12)]
    assert gaps == [3.0, 12.0]


def test_interior_cv_degenerate_when_no_interior_picks():
    """全选端点(内部零次)时 CV 走 mean==0 分支返回 0,不能崩。

    真实 divmax 不会这样(K=2 时 58% 的选择落在内部),这里只保证不炸。
    """
    assert diag.interior_cv([50] + [0] * 14 + [50]) == 0.0


def test_every_verdict_has_prose():
    """判读文本必须覆盖所有分支,否则报告阶段 KeyError。"""
    assert set(diag.VERDICT_TEXT) == {
        "endpoint_bias_flat", "endpoint_bias_structured",
        "uninformative", "structured_no_bias",
    }


# ------------------------------------------------------------ corpus 包装
# 这一组是真实事故的回归:第一版把 available_corpus[...] 这个**普通 dict**
# 直接塞进 VidTxtRetEvalDataset,在 V100 上崩于
#   dataset/utils.py:102  data_root = d.data_root
#   AttributeError: 'dict' object has no attribute 'data_root'
# 本机没有 decord / anno 文件,跑不了真集成测试,所以这里用静态契约把它钉死。

def _load_anno_source() -> str:
    """把 dataset/utils.py 里 load_anno 的函数体切出来(纯文本,不 import)。"""
    src = open(os.path.join(_ROOT, "dataset", "utils.py")).read()
    start = src.index("def load_anno(")
    rest = src[start + 1:]
    nxt = rest.find("\ndef ")
    return rest[:nxt] if nxt != -1 else rest


def _attrs_read_by_load_anno():
    """load_anno 用**属性**读的字段名(`d.xxx`),排除 dict 方法。"""
    import re
    names = set(re.findall(r"\bd\.([A-Za-z_][A-Za-z0-9_]*)", _load_anno_source()))
    return names - {"get", "keys", "items", "values"}


def test_plain_dict_lacks_attribute_access():
    """记录 bug 本身:普通 dict 上 d.data_root 就是会炸。"""
    with pytest.raises(AttributeError):
        {"data_root": "/x"}.data_root


def test_attrdict_supports_load_anno_access_pattern():
    """逐行复现 load_anno 开头对 corpus 条目的访问。"""
    d = diag.AttrDict({"anno_path": "a.json", "data_root": "/root", "media_type": "video"})
    assert isinstance(d, dict)          # load_anno 靠这个决定要不要包成单元素列表
    assert d.data_root == "/root"
    assert d.get("data_root_prefix", "") == ""
    assert d.anno_path == "a.json"
    assert d.media_type == "video"


def test_attrdict_missing_key_raises_attributeerror():
    """必须是 AttributeError 而不是 KeyError,否则 getattr(o, x, default) 会炸。"""
    d = diag.AttrDict({"a": 1})
    with pytest.raises(AttributeError):
        d.nope
    assert getattr(d, "nope", "fallback") == "fallback"


def test_corpus_entry_provides_every_attr_load_anno_reads():
    """msrvtt corpus 条目必须覆盖 load_anno 读的所有属性 —— 少一个就是下一次 V100 崩。"""
    from configs.data import available_corpus
    entry = available_corpus["msrvtt_1k_test"]
    missing = _attrs_read_by_load_anno() - set(entry)
    assert not missing, f"corpus 条目缺少 load_anno 需要的属性: {sorted(missing)}"


def test_attrs_probe_actually_found_something():
    """防止上面的正则因为 load_anno 改写而静默匹配到空集(测试变成永真)。"""
    assert "data_root" in _attrs_read_by_load_anno()


def test_load_anno_actually_accepts_attrdict(tmp_path):
    """真跑一次崩溃点 —— 本机缺 torchaudio/decord 会跳过,V100 上会真执行。

    上面几条是静态契约;这条是唯一能证明 AttrDict 真的喂得进 load_anno 的测试。
    """
    import json as _json
    pytest.importorskip("torchaudio")
    pytest.importorskip("decord")
    from dataset.utils import load_anno

    anno = tmp_path / "tiny.json"
    anno.write_text(_json.dumps([{"video": "v0.mp4", "caption": "a cat"}]))
    d = diag.AttrDict({
        "anno_path": str(anno), "data_root": "/videos", "media_type": "video",
    })
    out = load_anno(d)                       # 第一版就是死在这里
    assert len(out) == 1
    assert out[0]["image"] == "/videos/v0.mp4"


# ------------------------------------------------------------ 守卫
@pytest.mark.parametrize("var", ["MVC_FRAME_SELECT", "MVC_FRAME_PERTURB"])
def test_refuses_to_run_with_hook_enabled(var):
    """钩子开着会改变吐出来的帧,诊断就变成了诊断另一件事 —— 必须拒跑。"""
    env = {**os.environ, var: "divmax" if "SELECT" in var else "shuffle"}
    env.pop("MVC_DATA_DIR", None)
    r = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "experiments", "frame_select_diag.py"),
         "--limit", "1"],
        capture_output=True, text=True, env=env, cwd=_ROOT, timeout=180,
    )
    assert r.returncode != 0
    assert "拒绝执行" in (r.stdout + r.stderr)
    assert var in (r.stdout + r.stderr)
