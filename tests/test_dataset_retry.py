"""`dataset.pt_dataset.retry_load_item` 的边界测试。

不需要训练环境:装了 decord 就直接 import 真函数,没装就从源码里把这一个函数
抽出来单独执行(它只依赖 numpy + logging)。所以在开发机和服务器上都能跑:

    python -m pytest tests/test_dataset_retry.py -v

覆盖三件事:
1. 全部样本都读不到时(data_root 配错 / 视频没下载)必须**有界地**报错,
   而不是无限递归成 RecursionError —— 这是 2026-07-22 试点崩溃的直接原因;
2. 偶发坏文件仍然要能重采样恢复(不能因为修 bug 把容错也修没了);
3. media_type="audio_video" 时错误信息里的路径查找不能二次崩溃。
"""
import ast
import logging

import numpy as np
import pytest


def _get_retry_load_item():
    """优先用真的;缺 decord 之类的重依赖时从源码抽这一个函数。"""
    try:
        from dataset.pt_dataset import MIN_LOAD_RETRIES, retry_load_item
        return retry_load_item, MIN_LOAD_RETRIES
    except ImportError:
        pass

    import os
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "pt_dataset.py",
    )
    with open(src_path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "retry_load_item"),
        None,
    )
    if fn is None:
        pytest.fail("retry_load_item 不在 dataset/pt_dataset.py 里了,测试需要同步更新")
    floor = next(
        (n.value.value for n in tree.body
         if isinstance(n, ast.Assign)
         and getattr(n.targets[0], "id", None) == "MIN_LOAD_RETRIES"),
        None,
    )
    if floor is None:
        pytest.fail("MIN_LOAD_RETRIES 不在 dataset/pt_dataset.py 里了,测试需要同步更新")
    ns = {"np": np, "logger": logging.getLogger("test_dataset_retry"),
          "MIN_LOAD_RETRIES": floor}
    exec(compile(ast.Module([fn], []), "<retry_load_item>", "exec"), ns)
    return ns["retry_load_item"], floor


retry_load_item, MIN_LOAD_RETRIES = _get_retry_load_item()


class _AllMissing:
    """每条样本都读不到 —— 视频没下载 / data_root 配错的场景。"""
    media_type = "video"
    num_tries = 3

    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 10000

    def get_anno(self, index):
        return {"video": f"data/video_sample/InternVId-FLT_2/x{index}.mp4"}

    def _load_item(self, index):
        self.calls += 1
        raise RuntimeError("Error reading ...mp4")


class _FlakyThenOK:
    """第一次失败,重采样到的下一条就成功。"""
    media_type = "video"
    num_tries = 3

    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 10

    def get_anno(self, index):
        return {"video": f"v{index}.mp4"}

    def _load_item(self, index):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("corrupt file")
        return ("VIDEO", "a caption", index)


def test_all_missing_raises_bounded_not_recursion():
    """全缺失时有界报错,且错误信息要能指向真正的原因。"""
    ds = _AllMissing()
    logging.disable(logging.WARNING)
    try:
        with pytest.raises(RuntimeError) as exc:
            retry_load_item(ds, 0)
    finally:
        logging.disable(logging.NOTSET)

    # 递归版会在这里炸 RecursionError;有界版只试 num_tries + 1 次
    assert not isinstance(exc.value, RecursionError)
    expected = max(ds.num_tries, MIN_LOAD_RETRIES) + 1
    assert ds.calls == expected, f"应尝试 {expected} 次,实际 {ds.calls}"
    # 必须远低于 Python 递归上限,否则等于没修
    assert ds.calls < 500

    msg = str(exc.value)
    # 报错要告诉操作者去查什么,而不是只丢一个栈
    assert "data_root" in msg
    assert "check_readiness.py" in msg


def test_transient_failure_still_recovers():
    """修掉无限递归不能把偶发坏文件的容错一起修没。"""
    ds = _FlakyThenOK()
    logging.disable(logging.WARNING)
    try:
        media, caption, _ = retry_load_item(ds, 0)
    finally:
        logging.disable(logging.NOTSET)

    assert media == "VIDEO"
    assert caption == "a caption"
    assert ds.calls == 2, f"应在第 2 次成功,实际调用 {ds.calls} 次"


def test_audio_video_media_type_does_not_crash_path_lookup():
    """media_type='audio_video' 不是 anno 的 key,路径查找要回退而不是二次崩溃。"""
    class _AV(_AllMissing):
        media_type = "audio_video"

        def get_anno(self, index):
            return {"video": "v.mp4", "audio": "a.wav"}

    logging.disable(logging.WARNING)
    try:
        with pytest.raises(RuntimeError):
            retry_load_item(_AV(), 0)
    finally:
        logging.disable(logging.NOTSET)


def test_realistic_attrition_does_not_trip_the_bound():
    """InternVid 有永久性的 YouTube 损耗,健康语料也有一定比例的死链。

    边界必须高到"随机损耗打不穿",否则修掉无限递归反而引入了新的随机崩溃:
    单样本崩溃概率 ~ p^(n+1),配置默认 n=3 时 p=0.3 就是 8e-3,
    一个 55k 样本的 epoch 会崩几百次。
    """
    rng = np.random.RandomState(0)

    class _Attrition:
        media_type = "video"
        num_tries = 3  # 配置默认值:必须被 MIN_LOAD_RETRIES 抬上去

        def __len__(self):
            return 1000

        def get_anno(self, index):
            return {"video": f"v{index}.mp4"}

        def _load_item(self, index):
            if rng.rand() < 0.3:          # 30% 死链,比实际更悲观
                raise RuntimeError("dead link")
            return ("VIDEO", "cap", index)

    ds = _Attrition()
    logging.disable(logging.WARNING)
    try:
        # 2000 次取样一次都不该抛;n=3 的话期望崩 ~16 次
        for i in range(2000):
            media, _, _ = retry_load_item(ds, i % len(ds))
            assert media == "VIDEO"
    finally:
        logging.disable(logging.NOTSET)
