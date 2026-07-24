#!/usr/bin/env python
"""一次性探测预训练链路缺哪些第三方模块（零 GPU、零下载）。

用法：
    PYTHONPATH=. python experiments/check_imports.py

为什么需要它：`tasks_clip/pretrain.py` 的 import 链很长，缺一个包就崩一次，
一轮只能发现一个（b100 2026-07-24 就这么来回了好几轮：peft → open_clip →
termcolor → ...）。这个脚本把整条链路上的第三方模块**逐个 try import**，
一次列全所有缺失项，并直接拼好可复制的 pip 命令。

输出末尾给出的 pip 命令刻意带 `--user --no-deps`，原因见 TROUBLESHOOTING.md
「b100 共用系统 site-packages，别再往里装」一节：
  * `--user` + `PYTHONUSERBASE` → 装进自己的私有目录，不动共用的系统 site-packages；
  * `--no-deps` → 不让 pip 顺手升/降级 torch、transformers、numpy 这些共用重依赖。
"""

import importlib
import sys

# 模块名 -> pip 包名（同名的写 None）。仅列**非可选**依赖：
# apex / flash_attn / petrel_client / coremltools / onnx 都在 try 块里，缺了不影响训练。
REQUIRED = {
    "torch": None,
    "torchvision": None,
    "numpy": None,
    "yaml": "PyYAML",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "av": None,
    "decord": None,
    "imageio": None,
    "einops": None,
    "easydict": None,
    "scipy": None,
    "pandas": None,
    "tqdm": None,
    "termcolor": None,
    "timm": None,
    "transformers": None,
    "open_clip": "open_clip_torch",
    "deepspeed": None,
    "fvcore": None,
    "iopath": None,
    "wandb": None,
}

# 仓库自身的模块：第三方都齐了之后再验这几个，能一路 import 就等于冒烟前置全过。
REPO = ["models", "dataset.video_utils", "utils.config_utils", "tasks_clip.pretrain"]


def main() -> int:
    print(f"python     : {sys.executable}")
    try:
        import torch

        print(f"torch      : {torch.__version__}")
        print(f"arch_list  : {torch.cuda.get_arch_list()}")
    except Exception as e:  # noqa: BLE001
        print(f"torch      : 导入失败 -> {e!r}")

    missing = []
    print("\n--- 第三方依赖 ---")
    for mod, pkg in REQUIRED.items():
        try:
            importlib.import_module(mod)
            print(f"[OK  ] {mod}")
        except ImportError:
            print(f"[缺失] {mod}")
            missing.append(pkg or mod)
        except Exception as e:  # noqa: BLE001  # 装了但坏了（版本冲突等），要看清楚
            print(f"[异常] {mod} -> {type(e).__name__}: {e}")

    print("\n--- 仓库模块（第三方齐了才有意义）---")
    repo_bad = []
    for mod in REPO:
        try:
            importlib.import_module(mod)
            print(f"[OK  ] {mod}")
        except Exception as e:  # noqa: BLE001
            print(f"[失败] {mod} -> {type(e).__name__}: {e}")
            repo_bad.append(mod)

    print()
    if missing:
        print("缺失的第三方包，一条命令补齐（先 export PYTHONUSERBASE，见 SETUP_B100.md §6）：")
        print()
        print("    python -m pip install --user --no-deps " + " ".join(sorted(missing)))
        print()
        print("装完重跑本脚本，直到这里不再打印命令。")
    elif repo_bad:
        print("第三方依赖齐了，但仓库模块仍失败——看上面 [失败] 行的真实异常，")
        print("不是缺包问题，查 TROUBLESHOOTING.md。")
    else:
        print("全部 [OK]：可以进 §6 冒烟测试了。")

    return 1 if (missing or repo_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
