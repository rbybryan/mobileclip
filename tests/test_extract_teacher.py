"""`experiments/extract_teacher.py` 的单测(C1 教师提取,C1 教师管线)。

全部 CPU、不需要教师权重、不需要视频文件。分两层:

* **纯函数层** —— 抽帧子集、token 切片、分片、块续跑、merge 对齐、失败占位行。
* **接口层** —— 用一个**极小配置的真 `InternVideo2`** 验证脚本对教师的三条结构假设:
  输入是 `[B, C, T, H, W]`、token 布局是 `[cls] + T*L`、`clip_projector` 是 token 集合的
  纯函数(所以子集池化才成立)。这层是重点:一旦上游改了 token 布局,提取出来的
  `frame_emb` / `subset_emb` 会**静默错位**,而嵌入本身看不出毛病。

跑:`python -m pytest tests/test_extract_teacher.py -v`
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.extract_teacher import (  # noqa: E402
    ProjectorCapture,
    frame_token_slice,
    merge_blocks,
    pick_captions,
    pick_video_path,
    read_anno,
    save_npz_atomic,
    shard_rows,
    subset_token_slice,
    uniform_subset,
    verify,
    write_json_atomic,
)


# --------------------------------------------------------------- 纯函数层


class TestUniformSubset:
    def test_full_is_identity(self):
        assert uniform_subset(16, 16) == list(range(16))

    @pytest.mark.parametrize("t,want", [
        (2, [4, 12]),
        (4, [2, 6, 10, 14]),
        (8, [1, 3, 5, 7, 9, 11, 13, 15]),
    ])
    def test_known_subsets(self, t, want):
        """低 T 子集必须钉死:它决定 `subset_emb` 的语义,改了就对不上学生看到的帧。"""
        assert uniform_subset(16, t) == want

    @pytest.mark.parametrize("t", [1, 2, 3, 4, 5, 8, 16])
    def test_valid_and_unique(self, t):
        s = uniform_subset(16, t)
        assert len(s) == t == len(set(s))
        assert all(0 <= i < 16 for i in s)
        assert s == sorted(s)

    def test_deterministic(self):
        assert uniform_subset(16, 4) == uniform_subset(16, 4)

    @pytest.mark.parametrize("t", [0, -1, 17])
    def test_rejects_bad_size(self, t):
        with pytest.raises(ValueError):
            uniform_subset(16, t)


class TestTokenSlicing:
    B, T, L, C = 2, 4, 3, 8

    def _tokens(self):
        n = 1 + self.T * self.L
        return torch.arange(self.B * n * self.C, dtype=torch.float32).reshape(self.B, n, self.C)

    def test_frame_slice_shape_and_content(self):
        tok = self._tokens()
        for t in range(self.T):
            fs = frame_token_slice(tok, t, self.T)
            assert fs.shape == (self.B, 1 + self.L, self.C)
            assert torch.equal(fs[:, :1], tok[:, :1])           # cls 保留
            assert torch.equal(fs[:, 1:], tok[:, 1 + t * self.L: 1 + (t + 1) * self.L])

    def test_subset_slice_concatenates_in_order(self):
        tok = self._tokens()
        ss = subset_token_slice(tok, [1, 3], self.T)
        assert ss.shape == (self.B, 1 + 2 * self.L, self.C)
        assert torch.equal(ss[:, 1:1 + self.L], tok[:, 1 + self.L: 1 + 2 * self.L])
        assert torch.equal(ss[:, 1 + self.L:], tok[:, 1 + 3 * self.L: 1 + 4 * self.L])

    def test_rejects_inconsistent_layout(self):
        """token 数不能被帧数整除 ⇒ 布局假设不成立,必须**抛错而不是猜**。"""
        bad = torch.zeros(1, 1 + 7, 4)
        with pytest.raises(RuntimeError, match="布局假设"):
            frame_token_slice(bad, 0, 4)


class TestShardRows:
    def test_partition_is_exact_cover(self):
        got = []
        for k in range(4):
            got += shard_rows(100, k, 4)
        assert sorted(got) == list(range(100))

    def test_strided_not_contiguous(self):
        """跨步切片:相邻行常同源,连续切会让各 shard 失败率/时长分布系统性不同。"""
        assert shard_rows(10, 0, 2) == [0, 2, 4, 6, 8]

    def test_limit_applies_before_sharding(self):
        assert shard_rows(100, 0, 1, limit=7) == list(range(7))
        assert shard_rows(100, 0, 2, limit=10) == [0, 2, 4, 6, 8]

    def test_limit_larger_than_corpus_is_noop(self):
        assert shard_rows(5, 0, 1, limit=999) == list(range(5))


class TestRecordFields:
    def test_video_path_prefers_abs_path(self):
        assert pick_video_path({"abs_path": "/a.mp4", "video": "b.mp4"}) == "/a.mp4"
        assert pick_video_path({"video": "b.mp4"}) == "b.mp4"
        assert pick_video_path({"nothing": 1}) is None

    def test_captions_accepts_str_and_list(self):
        assert pick_captions({"caption": "a"}, "caption", 4) == ["a"]
        assert pick_captions({"caption": ["a", "b"]}, "caption", 4) == ["a", "b"]

    def test_captions_dedup_and_cap(self):
        """U7-3 要 K≥2;但重复字幕不能充数,否则 KD 的多正例信号是假的。"""
        rec = {"caption": ["a", "a", "b"], "captions": ["b", "c", "d"]}
        assert pick_captions(rec, "caption", 3) == ["a", "b", "c"]

    def test_captions_merges_synthetic_channel(self):
        rec = {"caption": "real", "recaption": ["syn1", "syn2"]}
        assert pick_captions(rec, "caption", 4) == ["real", "syn1", "syn2"]


class TestReadAnno:
    RECS = [{"video": "/a.mp4", "caption": "x"}, {"video": "/b.mp4", "caption": "y"}]

    def _roundtrip(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "anno")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            return read_anno(p)

    def test_jsonl(self):
        import json
        text = "\n".join(json.dumps(r) for r in self.RECS) + "\n"
        assert self._roundtrip(text) == self.RECS

    def test_json_array(self):
        """仓库 anno_downstream/InternVid/*.json 是 `[{...}]` 数组(configs/data.py
        指向的正是它们)—— 必须和 jsonl 一样直接可读,行号语义一致。"""
        import json
        assert self._roundtrip(json.dumps(self.RECS)) == self.RECS

    def test_json_array_with_leading_whitespace(self):
        import json
        assert self._roundtrip("\n  " + json.dumps(self.RECS)) == self.RECS

    def test_malformed_line_raises_not_silently_skips(self):
        """坏行必须炸而不是被静默吞掉 —— 行号错位的下游表现是「蒸馏没效果」。"""
        import json as _json
        with pytest.raises(ValueError):
            self._roundtrip('{"video": "/a.mp4"}{"video": "/b.mp4"}\n')
        assert issubclass(_json.JSONDecodeError, ValueError)


class TestBlocksAndMerge:
    def _write_block(self, d, idx, rows, ok, D=4):
        emb = np.random.RandomState(idx).randn(len(rows), D).astype(np.float16)
        emb[~ok] = 0
        save_npz_atomic(
            os.path.join(d, f"block_{idx:06d}.npz"),
            rows=np.asarray(rows, dtype=np.int64), ok=ok, video_emb=emb,
            frame_indices=np.zeros((len(rows), 16), np.int32),
            vlen=np.zeros(len(rows), np.int32), fps=np.zeros(len(rows), np.float32),
        )

    def test_atomic_write_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_block(tmp, 0, [0, 1], np.array([True, True]))
            assert os.listdir(tmp) == ["block_000000.npz"]

    def test_merge_sorts_and_aligns_rows(self):
        """块是乱序写的(shard 跨步切片),merge 必须按 rows 排回去。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "all")
            os.makedirs(d)
            self._write_block(d, 0, [4, 0, 2], np.array([True, True, True]))
            self._write_block(d, 1, [1, 3], np.array([True, True]))
            s = merge_blocks(tmp)
            assert s["all"]["n"] == 5 and s["all"]["dup_rows"] == 0
            assert np.array_equal(np.load(os.path.join(d, "rows.npy")), np.arange(5))

    def test_failed_rows_are_kept_as_placeholders(self):
        """**核心不变量**:失败条保留占位行。丢行 ⇒ 下游对齐悄悄错位(下游对齐类事故)。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "all")
            os.makedirs(d)
            ok = np.array([True, False, True])
            self._write_block(d, 0, [0, 1, 2], ok)
            s = merge_blocks(tmp)
            assert s["all"]["n"] == 3                     # 行数不因失败而减少
            assert s["all"]["n_ok"] == 2
            assert abs(s["all"]["fail_rate"] - 1 / 3) < 1e-9
            emb = np.load(os.path.join(d, "video_emb.npy"))
            assert np.abs(emb[1]).sum() == 0              # 失败条是零向量
            assert np.abs(emb[0]).sum() > 0

    def test_verify_flags_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "all")
            os.makedirs(d)
            self._write_block(d, 0, [0, 1], np.array([True, True]))
            self._write_block(d, 1, [1, 2], np.array([True, True]))   # 行号撞车
            merge_blocks(tmp)
            write_json_atomic(os.path.join(d, "manifest.json"), {})
            assert verify(tmp)["all"]["rows_unique"] is False

    def test_verify_flags_zero_embedding_marked_ok(self):
        """ok=True 却是零向量 ⇒ 教师前向出了问题(不是解码问题),必须能被 verify 抓到。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "all")
            os.makedirs(d)
            ok = np.array([True, True])
            emb = np.zeros((2, 4), np.float16)
            save_npz_atomic(os.path.join(d, "block_000000.npz"),
                            rows=np.arange(2, dtype=np.int64), ok=ok, video_emb=emb,
                            frame_indices=np.zeros((2, 16), np.int32),
                            vlen=np.zeros(2, np.int32), fps=np.zeros(2, np.float32))
            merge_blocks(tmp)
            write_json_atomic(os.path.join(d, "manifest.json"), {})
            assert verify(tmp)["all"]["zero_emb_but_ok"] == 2


# --------------------------------------------------------------- 接口层


@pytest.fixture(scope="module")
def tiny_teacher():
    """极小配置的**真** `InternVideo2`。验证结构假设,不验证数值。"""
    from models.backbones.internvideo2.internvideo2_clip_vision import InternVideo2

    torch.manual_seed(0)
    return InternVideo2(
        img_size=32, patch_size=16, num_frames=4, tubelet_size=1,
        embed_dim=32, depth=2, num_heads=2, mlp_ratio=2.0,
        clip_embed_dim=24, attn_pool_num_heads=2,
        use_flash_attn=False, use_fused_rmsnorm=False, use_fused_mlp=False,
        qk_normalization=False, drop_path_rate=0.0,
    ).eval()


class TestTeacherInterface:
    T, S = 4, 32

    def test_accepts_bcthw(self, tiny_teacher):
        """脚本把 dataloader 的 [B,T,C,H,W] permute 成 [B,C,T,H,W];PatchEmbed 是 Conv3d。"""
        with torch.no_grad():
            out = tiny_teacher(torch.randn(2, 3, self.T, self.S, self.S))
        assert out.shape == (2, 24)

    def test_embed_dim_probe_matches_output(self, tiny_teacher):
        """脚本用 `clip_projector.cross_attn.proj.out_features` 探测 D,写进 manifest。"""
        probed = tiny_teacher.clip_projector.cross_attn.proj.out_features
        with torch.no_grad():
            out = tiny_teacher(torch.randn(1, 3, self.T, self.S, self.S))
        assert probed == out.shape[1] == 24

    def test_token_layout_is_cls_plus_t_times_l(self, tiny_teacher):
        cap = ProjectorCapture(tiny_teacher)
        try:
            with torch.no_grad():
                tiny_teacher(torch.randn(2, 3, self.T, self.S, self.S))
            L = (self.S // 16) ** 2
            assert cap.tokens.shape[1] == 1 + self.T * L
        finally:
            cap.close()

    def test_capture_pool_reproduces_forward_exactly(self, tiny_teacher):
        """**最关键的一条**:hook 抓到的 token 重新过 projector+fc_norm,必须逐位复现
        模型自己的输出。它成立,才说明 `subset_emb` 真是「同一个读出头、只看子集 token」。"""
        cap = ProjectorCapture(tiny_teacher)
        try:
            x = torch.randn(2, 3, self.T, self.S, self.S)
            with torch.no_grad():
                out = tiny_teacher(x)
                reconstructed = cap.pool(cap.tokens)
            assert torch.allclose(reconstructed, out, atol=1e-6)
        finally:
            cap.close()

    def test_subset_pool_differs_from_full_pool(self, tiny_teacher):
        """子集池化必须真的不同于全量池化 —— 否则 `subset_emb` 是废数据。"""
        cap = ProjectorCapture(tiny_teacher)
        try:
            with torch.no_grad():
                tiny_teacher(torch.randn(2, 3, self.T, self.S, self.S))
                full = cap.pool(cap.tokens)
                sub = cap.pool(subset_token_slice(cap.tokens, uniform_subset(self.T, 2), self.T))
            assert sub.shape == full.shape
            assert not torch.allclose(sub, full, atol=1e-4)
        finally:
            cap.close()

    def test_per_frame_pool_shapes(self, tiny_teacher):
        cap = ProjectorCapture(tiny_teacher)
        try:
            with torch.no_grad():
                tiny_teacher(torch.randn(3, 3, self.T, self.S, self.S))
                femb = torch.stack(
                    [cap.pool(frame_token_slice(cap.tokens, t, self.T)) for t in range(self.T)], 1)
            assert femb.shape == (3, self.T, 24)
        finally:
            cap.close()

    def test_pool_does_not_clobber_capture(self, tiny_teacher):
        """回归测试(2026-07-29 首次单测就抓到):`pool()` 内部会再次调用
        `clip_projector`,从而**重入** forward_pre_hook。若不设防,第一次子集池化就把
        `cap.tokens` 覆盖成那个子集,之后每一帧都从被污染的 token 流上切 —— 而且
        大概率不抛错、只是**悄悄算错** `frame_emb`。hook 必须在 `pool()` 期间冻结。"""
        cap = ProjectorCapture(tiny_teacher)
        try:
            with torch.no_grad():
                tiny_teacher(torch.randn(2, 3, self.T, self.S, self.S))
                before = cap.tokens.clone()
                cap.pool(frame_token_slice(cap.tokens, 0, self.T))
                cap.pool(subset_token_slice(cap.tokens, [0, 1], self.T))
            assert torch.equal(cap.tokens, before), "pool() 污染了已捕获的 token 流"
        finally:
            cap.close()

    def test_per_frame_pool_is_stable_without_prebinding(self, tiny_teacher):
        """逐帧池化直接读 `cap.tokens`(不预先绑定局部变量)也必须正确 ——
        否则脚本的正确性就依赖于书写顺序,是个随时会被重构踩爆的地雷。"""
        cap = ProjectorCapture(tiny_teacher)
        try:
            with torch.no_grad():
                tiny_teacher(torch.randn(2, 3, self.T, self.S, self.S))
                femb = torch.stack(
                    [cap.pool(frame_token_slice(cap.tokens, t, self.T))
                     for t in range(self.T)], 1)
            assert femb.shape == (2, self.T, 24)
            # 各帧的池化结果必须互不相同,否则说明切片没生效
            assert not torch.allclose(femb[:, 0], femb[:, 1], atol=1e-5)
        finally:
            cap.close()

    def test_hook_is_removed_on_close(self, tiny_teacher):
        n_before = len(tiny_teacher.clip_projector._forward_pre_hooks)
        cap = ProjectorCapture(tiny_teacher)
        assert len(tiny_teacher.clip_projector._forward_pre_hooks) == n_before + 1
        cap.close()
        assert len(tiny_teacher.clip_projector._forward_pre_hooks) == n_before


# --------------------------------------------------------------- 端到端


class TestRunVideoStageEndToEnd:
    """跑通整条 `run_video_stage`:块循环 / 续跑 / manifest / 失败占位 / merge。

    视频路径全部指向不存在的文件 ⇒ 每条都走解码失败分支。这既省掉了造视频的麻烦,
    又正好把**最危险的那条路径**(失败条如何落盘)完整走一遍。教师用极小真模型,
    `--fail-rate-abort` 放宽到 1.1 才允许写出全失败的块。
    """

    def _args(self, tmp, anno, **kw):
        import argparse

        a = argparse.Namespace(
            anno=anno, out=os.path.join(tmp, "out"), shard_id=0, num_shards=1,
            shard_tag="", limit=0, batch_size=2, num_workers=0, block_size=3,
            device="cpu", captions_key="caption", max_captions=4,
            stage="video", teacher_ckpt="fake.pt", num_frames=4, sample="middle",
            img_size=32, vision_ckpt_t_size=4, use_flash_attn=False, amp="off",
            per_frame=False, subset_ts="2", fail_rate_abort=1.1,
            text_tokenizer="", text_model_kwargs="",
        )
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def _anno(self, tmp, n=7):
        p = os.path.join(tmp, "anno.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for i in range(n):
                f.write('{"abs_path": "/nonexistent/%d.mp4", "caption": "c%d"}\n' % (i, i))
        return p

    def test_end_to_end_with_resume(self, tiny_teacher, monkeypatch):
        import experiments.extract_teacher as ET

        monkeypatch.setattr(ET, "build_teacher", lambda *a, **k: tiny_teacher)

        with tempfile.TemporaryDirectory() as tmp:
            anno = self._anno(tmp, 7)
            args = self._args(tmp, anno)
            ET.run_video_stage(args)

            d = os.path.join(args.out, "all")
            blocks = sorted(f for f in os.listdir(d) if f.startswith("block_"))
            assert blocks == ["block_000000.npz", "block_000001.npz",
                              "block_000002.npz"]              # 7 行 / 块 3 ⇒ 3 块
            assert os.path.exists(os.path.join(d, "manifest.json"))

            import json as _json
            man = _json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
            assert man["teacher"]["identity"] == "InternVideo2-stage2"   # U7-4 定案
            assert man["sampling"]["num_frames"] == 4
            assert man["sampling"]["deterministic"] is True
            assert man["teacher"]["embed_dim"] == 24

            # 续跑:已有块必须被跳过(用 mtime 证明没被重写)
            mtimes = {b: os.path.getmtime(os.path.join(d, b)) for b in blocks}
            ET.run_video_stage(args)
            for b in blocks:
                assert os.path.getmtime(os.path.join(d, b)) == mtimes[b]

            s = merge_blocks(args.out)
            assert s["all"]["n"] == 7                    # 7 行一条不少
            assert s["all"]["n_ok"] == 0                 # 全部解码失败
            assert s["all"]["dup_rows"] == 0
            rows = np.load(os.path.join(d, "rows.npy"))
            assert np.array_equal(rows, np.arange(7))
            fidx = np.load(os.path.join(d, "frame_indices.npy"))
            assert (fidx == -1).all()                    # 失败条帧号记 -1

    def test_aborts_when_failure_rate_exceeds_threshold(self, tiny_teacher, monkeypatch):
        """半空的 shard 绝不能悄悄落盘 —— 它在下游只表现为『蒸馏没效果』。"""
        import experiments.extract_teacher as ET

        monkeypatch.setattr(ET, "build_teacher", lambda *a, **k: tiny_teacher)
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, self._anno(tmp, 4), fail_rate_abort=0.2)
            with pytest.raises(RuntimeError, match="失败率"):
                ET.run_video_stage(args)

    def test_missing_path_field_fails_fast(self, tiny_teacher, monkeypatch):
        """开跑即失败:anno 没有路径字段时立刻退出,而不是跑满一轮才发现(G10 教训)。"""
        import experiments.extract_teacher as ET

        monkeypatch.setattr(ET, "build_teacher", lambda *a, **k: tiny_teacher)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bad.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"caption": "no path here"}\n')
            with pytest.raises(SystemExit, match="视频路径字段"):
                ET.run_video_stage(self._args(tmp, p))

    def test_per_frame_arrays_have_expected_shapes(self, tiny_teacher, monkeypatch):
        import experiments.extract_teacher as ET

        monkeypatch.setattr(ET, "build_teacher", lambda *a, **k: tiny_teacher)
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, self._anno(tmp, 3), per_frame=True, subset_ts="2")
            ET.run_video_stage(args)
            z = np.load(os.path.join(args.out, "all", "block_000000.npz"))
            assert z["frame_emb"].shape == (3, 4, 24)     # [N, T, D]
            assert z["subset_emb"].shape == (3, 1, 24)    # [N, |subset_ts|, D]
            assert z["video_emb"].dtype == np.float16
