"""C1 KD 教师库 + U9 权重求和的单测(ROADMAP 2.2c)。

只依赖 numpy —— 本地无 torch/decord 也能全跑(与 test_paired_bootstrap 同一约定)。
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss_weighting import weighted_loss_sum          # noqa: E402
from utils.teacher_bank import TeacherBank, cli_align       # noqa: E402


# ------------------------------------------------------------------ fixtures


def make_bank_dir(tmp_path, n=10, d=8, shards=2, bad_row=7, missing_row=9):
    """合成 extract_teacher merge 后的目录:分片按行号取模,`bad_row` 标 ok=False,
    `missing_row` 不进任何分片(模拟 --limit 截断/分片没跑完)。"""
    rng = np.random.default_rng(0)
    full = rng.normal(size=(n, d)).astype(np.float16)
    for si in range(shards):
        rows = np.array([r for r in range(n) if r % shards == si and r != missing_row],
                        dtype=np.int64)
        sd = tmp_path / f"shard{si}"
        sd.mkdir()
        np.save(sd / "rows.npy", rows)
        np.save(sd / "video_emb.npy", full[rows])
        ok = np.ones(rows.shape[0], dtype=bool)
        ok[rows == bad_row] = False
        np.save(sd / "ok.npy", ok)
        (sd / "manifest.json").write_text(json.dumps({"anno": {"n_rows_total": n}}))
    return full


# ------------------------------------------------------------------ TeacherBank


def test_take_hits_and_misses(tmp_path):
    full = make_bank_dir(tmp_path)
    bank = TeacherBank(str(tmp_path))
    assert bank.n_rows_total == 10 and bank.embed_dim == 8
    emb, m = bank.take([0, 3, 7, 9, 42, -1])
    # 7 = ok=False,9 = 不在库,42/-1 = 越界 —— 全部 mask=False 且零行
    assert m.tolist() == [True, True, False, False, False, False]
    assert np.allclose(emb[0], full[0].astype(np.float32))
    assert np.allclose(emb[1], full[3].astype(np.float32))
    assert emb[2:].sum() == 0
    assert bank.n_asked == 6 and bank.n_hit == 2


def test_expect_rows_mismatch_raises(tmp_path):
    make_bank_dir(tmp_path)
    with pytest.raises(ValueError, match="不是同一份"):
        TeacherBank(str(tmp_path), expect_rows=11)
    TeacherBank(str(tmp_path), expect_rows=10)  # 一致则不炸


def test_duplicate_rows_raise(tmp_path):
    make_bank_dir(tmp_path)
    # 把 shard0 的行号改成与 shard1 重叠
    np.save(tmp_path / "shard0" / "rows.npy",
            np.load(tmp_path / "shard1" / "rows.npy"))
    # 行数也得对齐才能走到重复检查
    np.save(tmp_path / "shard0" / "video_emb.npy",
            np.load(tmp_path / "shard1" / "video_emb.npy"))
    np.save(tmp_path / "shard0" / "ok.npy", np.load(tmp_path / "shard1" / "ok.npy"))
    with pytest.raises(ValueError, match="重复"):
        TeacherBank(str(tmp_path))


def test_unmerged_dir_raises(tmp_path):
    (tmp_path / "shard0").mkdir()
    np.savez(tmp_path / "shard0" / "block_000000.npz", rows=np.arange(3))
    with pytest.raises(FileNotFoundError, match="merge"):
        TeacherBank(str(tmp_path))


def test_coverage_reported(tmp_path):
    make_bank_dir(tmp_path)  # 行 9 缺席
    bank = TeacherBank(str(tmp_path))
    assert abs(bank.coverage - 0.9) < 1e-9


# ------------------------------------------------------------------ U9 权重求和


def test_weighted_sum_basic():
    total = weighted_loss_sum({"loss_vtc": 2.0, "loss_kd_feat": 3.0},
                              {"vtc": 1.0, "kd_feat": 0.5})
    assert total == 2.0 + 1.5


def test_weight_one_is_identity_object():
    # 权重恰为 1.0 时跳过乘法 ⇒ 单损失基线的计算图与修复前逐位相同
    class Marker:  # 最小可加对象,记录是否被乘过
        def __mul__(self, w):
            raise AssertionError("weight=1.0 不应触发乘法")
    m = Marker()
    assert weighted_loss_sum({"loss_vtc": m}, {"vtc": 1.0}) is m


def test_missing_weight_raises():
    with pytest.raises(KeyError, match="loss_weight"):
        weighted_loss_sum({"loss_vtc": 1.0, "loss_new": 1.0}, {"vtc": 1.0})


def test_zero_weight_on_computed_loss_raises():
    with pytest.raises(ValueError, match="权重是 0"):
        weighted_loss_sum({"loss_vtc": 1.0, "loss_kd_feat": 1.0},
                          {"vtc": 1.0, "kd_feat": 0.0})


def test_empty_loss_dict_raises():
    with pytest.raises(ValueError):
        weighted_loss_sum({}, {"vtc": 1.0})


# ------------------------------------------------------------------ anno 对齐 CLI


def _write_anno_json(path, names):
    path.write_text(json.dumps([{"image": f"vids/{n}", "caption": "x"} for n in names]))


def _write_anno_jsonl(path, names):
    path.write_text("\n".join(
        json.dumps({"abs_path": f"/mnt/data/{n}", "caption": "x"}) for n in names))


def test_align_ok(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.jsonl"
    _write_anno_json(a, ["v1.mp4", "v2.mp4"])
    _write_anno_jsonl(b, ["v1.mp4", "v2.mp4"])
    assert cli_align(str(a), str(b)) == 0


def test_align_catches_reorder(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.jsonl"
    _write_anno_json(a, ["v1.mp4", "v2.mp4"])
    _write_anno_jsonl(b, ["v2.mp4", "v1.mp4"])
    assert cli_align(str(a), str(b)) == 1


def test_align_catches_length(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.jsonl"
    _write_anno_json(a, ["v1.mp4", "v2.mp4"])
    _write_anno_jsonl(b, ["v1.mp4"])
    assert cli_align(str(a), str(b)) == 1
