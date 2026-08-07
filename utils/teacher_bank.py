#!/usr/bin/env python3
"""C1 教师嵌入库(C1 KD)—— 训练时按 anno 行号查教师视频嵌入。

数据来源是 `experiments/extract_teacher.py` **merge 之后**的目录布局:

    <out_dir>/<shard_tag>/rows.npy        int64 [N_s]  anno 行号(升序,分片间不重叠)
    <out_dir>/<shard_tag>/video_emb.npy   f16   [N_s, D]
    <out_dir>/<shard_tag>/ok.npy          bool  [N_s]  解码/前向是否成功(失败=占位零行)
    <out_dir>/<shard_tag>/manifest.json   含 anno 路径与 n_rows_total

对齐的唯一凭据是 **anno 行号**:训练侧 `pt_dataset` 的 `idx` 就是样本在 anno 数组里的
下标(`jump_filter=True` ⇒ 不过滤不重排;解码失败重采样时返回的是**实际加载样本**的
下标,天然对齐)。任何会改变行号语义的设置(`caption_idx`、多语料 Concat、字幕过滤)
都必须在接线处被挡下 —— 见 `tasks_clip/pretrain.py` 的 build_teacher_bank。

🔴 **已知雷(必须先排):两份 anno 可能不是同一份。** `COMMANDS.md` §3.10 的抽取命令
用的是 b100 上的 `data/anno_pretrain/internvid_v1.jsonl`,而训练读的是
`configs/data.py` 指到的 `anno_downstream/InternVid/InternVid-10M-flt_1.json`。
两份文件只要有一行错位,KD 就静默变成「拿别的视频的教师教这个视频」——损失照降、
检索照出数,只是蒸馏白做。所以:
  - 本类构造时**强制**校验 `n_rows_total`(manifest)与训练集样本数一致;
  - `python utils/teacher_bank.py align --anno-a A --anno-b B` 逐行比对两份 anno 的
    视频文件名(basename),这是上抽取/上训练之前的**硬闸门**(零 GPU,秒级)。

本模块只依赖 numpy(torch-free):`take()` 返回 numpy,张量化交给调用方 ——
这样单测和 CLI 在没有 torch 的机器上也能跑。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

REQUIRED = ("rows.npy", "video_emb.npy", "ok.npy")


class TeacherBank:
    """按 anno 行号查教师嵌入。miss(行不在库里 / ok=False)返回零行 + mask=False。"""

    def __init__(self, out_dir: str, expect_rows: Optional[int] = None):
        if not os.path.isdir(out_dir):
            raise FileNotFoundError(
                f"[KD] teacher dir 不存在:{out_dir!r}。先跑 extract_teacher run(各分片)"
                f"再 merge + verify,见 COMMANDS.md §3.10。")
        tags = sorted(
            d for d in os.listdir(out_dir)
            if os.path.isdir(os.path.join(out_dir, d))
            and all(os.path.exists(os.path.join(out_dir, d, f)) for f in REQUIRED))
        if not tags:
            raise FileNotFoundError(
                f"[KD] {out_dir!r} 下没有任何 merge 过的分片(缺 rows.npy/video_emb.npy)。"
                f"块还在 block_*.npz 形态 ⇒ 先 `extract_teacher.py merge --out {out_dir}`。")
        self.out_dir = out_dir
        self.tags = tags
        self._emb: List[np.ndarray] = []   # 每分片 memmap [N_s, D]
        self._ok: List[np.ndarray] = []
        n_total = None
        dim = None
        for tag in tags:
            d = os.path.join(out_dir, tag)
            rows = np.load(os.path.join(d, "rows.npy"))
            emb = np.load(os.path.join(d, "video_emb.npy"), mmap_mode="r")
            ok = np.load(os.path.join(d, "ok.npy"))
            if rows.shape[0] != emb.shape[0] or rows.shape[0] != ok.shape[0]:
                raise ValueError(f"[KD] {tag}: rows/video_emb/ok 行数不一致 —— merge 坏了")
            if dim is None:
                dim = int(emb.shape[1])
            elif dim != emb.shape[1]:
                raise ValueError(f"[KD] 分片维度不一致:{tag} 是 {emb.shape[1]},先前是 {dim}")
            mpath = os.path.join(d, "manifest.json")
            if os.path.exists(mpath):
                with open(mpath) as f:
                    man = json.load(f)
                nrt = man.get("anno", {}).get("n_rows_total") or man.get("n_rows_total")
                if nrt is not None:
                    if n_total is not None and n_total != int(nrt):
                        raise ValueError(
                            f"[KD] 分片 manifest 的 n_rows_total 不一致({n_total} vs {nrt})"
                            f" —— 各分片不是对着同一份 anno 抽的,对齐无从谈起")
                    n_total = int(nrt)
            self._emb.append(emb)
            self._ok.append(ok.astype(bool))
            self._rows_tmp = rows  # 只为下面建索引
            if not hasattr(self, "_all_rows"):
                self._all_rows: List[np.ndarray] = []
            self._all_rows.append(rows.astype(np.int64))

        if n_total is None:
            n_total = int(max(r.max() for r in self._all_rows)) + 1
        self.n_rows_total = n_total
        self.embed_dim = int(dim)

        # 行号 → (分片, 分片内下标)。1M 行 ≈ 10 MB,常驻可接受。
        self._shard_of = np.full(n_total, -1, dtype=np.int16)
        self._local_of = np.zeros(n_total, dtype=np.int64)
        for si, rows in enumerate(self._all_rows):
            if rows.size and int(rows.max()) >= n_total:
                raise ValueError(f"[KD] {tags[si]} 的行号超出 n_rows_total={n_total} —— "
                                 f"分片和 manifest 不是同一份 anno 的产物")
            dup = self._shard_of[rows] >= 0
            if dup.any():
                raise ValueError(
                    f"[KD] 行号在多个分片重复出现(如 {rows[dup][:5].tolist()})—— "
                    f"同一 --out 被不同 --num-shards 跑过?清掉旧分片再 merge")
            self._shard_of[rows] = si
            self._local_of[rows] = np.arange(rows.shape[0], dtype=np.int64)
        del self._all_rows, self._rows_tmp

        cov = float((self._shard_of >= 0).mean())
        self.coverage = cov
        # 运行期命中统计(闸门要求 100% 命中,见 notes/c1-kd-training-manual-zh.md §2)
        self.n_asked = 0
        self.n_hit = 0

        if expect_rows is not None and expect_rows != n_total:
            raise ValueError(
                f"[KD] 教师库按 n_rows_total={n_total} 抽取,但训练集有 {expect_rows} 个样本。"
                f"两边 anno 不是同一份(或训练侧被过滤/拼接过)。KD 的行号对齐已破坏,"
                f"拒绝启动 —— 先跑 `python utils/teacher_bank.py align` 对齐两份 anno。")

    def take(self, rows: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """按 anno 行号取教师嵌入。

        Returns:
            emb  float32 [B, D] —— miss 行为全 0(调用方用 mask 排除,不会除到)。
            mask bool    [B]    —— 行在库里且 ok=True。
        """
        rows = np.asarray(rows, dtype=np.int64)
        B = rows.shape[0]
        out = np.zeros((B, self.embed_dim), dtype=np.float32)
        mask = np.zeros(B, dtype=bool)
        in_range = (rows >= 0) & (rows < self.n_rows_total)
        for i in np.nonzero(in_range)[0]:
            si = self._shard_of[rows[i]]
            if si < 0:
                continue
            li = self._local_of[rows[i]]
            if not self._ok[si][li]:
                continue
            out[i] = self._emb[si][li]
            mask[i] = True
        self.n_asked += B
        self.n_hit += int(mask.sum())
        return out, mask

    @property
    def hit_rate(self) -> float:
        return self.n_hit / self.n_asked if self.n_asked else float("nan")

    def summary(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "shards": self.tags,
            "n_rows_total": self.n_rows_total,
            "embed_dim": self.embed_dim,
            "coverage": round(self.coverage, 6),
        }


# ------------------------------------------------------------------ CLI 工具


def _read_anno(path: str) -> list:
    """读 anno:JSON 数组或 JSONL 自动识别(与 extract_teacher.read_anno 同语义,
    在这里复制 20 行以免 utils/ 反向依赖 experiments/)。"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        while head.isspace():
            head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(ln) for ln in f if ln.strip()]


def _video_name(rec: dict) -> str:
    """取记录的视频文件名(basename)。仓库 json 用 `image`(视频也用它,
    见 dataset/base_dataset.py),b100 jsonl 用 `abs_path`,老格式用 `video`。"""
    for k in ("image", "abs_path", "video"):
        if k in rec:
            return os.path.basename(str(rec[k]))
    raise KeyError(f"记录里找不到视频路径键(image/abs_path/video):{list(rec)[:8]}")


def cli_align(a: str, b: str, n_show: int = 5) -> int:
    """逐行比对两份 anno 的视频文件名。任何错位 ⇒ 非零退出。"""
    ra, rb = _read_anno(a), _read_anno(b)
    print(f"A = {a}  ({len(ra)} 行)")
    print(f"B = {b}  ({len(rb)} 行)")
    if len(ra) != len(rb):
        print(f"❌ 行数不一致({len(ra)} vs {len(rb)})—— 不是同一份语料,KD 对齐不成立")
        return 1
    bad = [i for i, (x, y) in enumerate(zip(ra, rb)) if _video_name(x) != _video_name(y)]
    if bad:
        print(f"❌ {len(bad)} 行文件名不一致,前 {n_show} 个:")
        for i in bad[:n_show]:
            print(f"   行 {i}: A={_video_name(ra[i])}  B={_video_name(rb[i])}")
        return 1
    print(f"✅ {len(ra)} 行全部一致 —— 行号对齐成立,可以拿这份抽取喂 KD")
    return 0


def cli_verify(out_dir: str, anno: Optional[str]) -> int:
    bank = TeacherBank(out_dir)
    print(json.dumps(bank.summary(), ensure_ascii=False, indent=2))
    rc = 0
    if bank.coverage < 1.0:
        n_miss = int((bank._shard_of < 0).sum())
        print(f"⚠️ 覆盖率 {bank.coverage:.4%},缺 {n_miss} 行 —— 分片没跑完或 --limit 截断。"
              f" 训练时这些行会被 mask 掉(命中率闸门会把它暴露出来)。")
    if anno:
        n = len(_read_anno(anno))
        if n != bank.n_rows_total:
            print(f"❌ anno 有 {n} 行,库按 {bank.n_rows_total} 行抽取 —— 不是同一份 anno")
            rc = 1
        else:
            print(f"✅ anno 行数与库一致({n})")
    return rc


def _self_test() -> int:
    """合成两分片小库,验证查表 / miss / 越界 / 重复行报错。零 GPU、零真数据。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        D, N = 8, 10
        rng = np.random.default_rng(0)
        full = rng.normal(size=(N, D)).astype(np.float16)
        # 分片 0 拿偶数行,分片 1 拿奇数行;行 7 标记 ok=False;行 9 不在任何分片(缺)
        for si, rows in enumerate([np.arange(0, N, 2), np.arange(1, N - 1, 2)]):
            d = os.path.join(td, f"shard{si}")
            os.makedirs(d)
            np.save(os.path.join(d, "rows.npy"), rows.astype(np.int64))
            np.save(os.path.join(d, "video_emb.npy"), full[rows])
            ok = np.ones(rows.shape[0], dtype=bool)
            if si == 1:
                ok[rows == 7] = False
            np.save(os.path.join(d, "ok.npy"), ok)
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump({"anno": {"n_rows_total": N}}, f)
        bank = TeacherBank(td)
        assert bank.embed_dim == D and bank.n_rows_total == N
        emb, m = bank.take([0, 3, 7, 9, 42, -1])
        assert m.tolist() == [True, True, False, False, False, False], m
        assert np.allclose(emb[0], full[0].astype(np.float32))
        assert np.allclose(emb[1], full[3].astype(np.float32))
        assert emb[2].sum() == 0 and emb[3].sum() == 0
        assert abs(bank.hit_rate - 2 / 6) < 1e-9
        # expect_rows 不匹配必须炸
        try:
            TeacherBank(td, expect_rows=N + 1)
            raise AssertionError("expect_rows 不匹配竟然没炸")
        except ValueError:
            pass
        # 行号重复(同一行进两个分片)必须炸
        d = os.path.join(td, "shard0")
        np.save(os.path.join(d, "rows.npy"), np.arange(1, N, 2).astype(np.int64))
        try:
            TeacherBank(td)
            raise AssertionError("重复行号竟然没炸")
        except ValueError:
            pass
    print("self-test OK —— 查表/miss/命中率/长度校验/重复行 全过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="检查 merge 后的库:分片/覆盖率/维度,可选对 anno 行数")
    v.add_argument("--out", required=True)
    v.add_argument("--anno", default=None, help="训练用 anno(json 数组或 jsonl)")
    al = sub.add_parser("align", help="逐行比对两份 anno 的视频文件名(抽取前的硬闸门)")
    al.add_argument("--anno-a", required=True, help="训练侧 anno(configs/data.py 指到的那份)")
    al.add_argument("--anno-b", required=True, help="抽取侧 anno(--anno 传给 extract_teacher 的那份)")
    sub.add_parser("self-test", help="合成数据自检(零依赖真数据)")
    a = ap.parse_args()
    if a.cmd == "verify":
        return cli_verify(a.out, a.anno)
    if a.cmd == "align":
        return cli_align(a.anno_a, a.anno_b)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
