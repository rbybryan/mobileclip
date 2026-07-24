#!/usr/bin/env bash
# b100 每次开新 shell 先 source 一次：
#
#     cd /workspace/mount/b100_zaip_data/rby/mobileclip
#     source experiments/env_b100.sh
#
# 只设环境变量，不装任何东西、不碰共用目录，重复 source 无副作用。
#
# 为什么是这套（背景见 TROUBLESHOOTING.md「b100 共用系统 site-packages」）：
#   * b100 是**共用 root 账号**，wky / zyc 也在用同一套 /usr/local/lib/python3.10/
#     dist-packages。往里 pip install 会改到别人的环境，带版本号的更会把
#     transformers / numpy / timm **降级**，属于事故级操作。
#   * 系统 torch 是 2.5.1+cu124，`get_arch_list()` 含 sm_90 → H100 本来就支持，
#     没必要为了本项目再装一套 torch（重下几个 G，还不一定编得对）。
#   * 所以策略是：**沿用系统 torch，缺的小包装进自己的 user-site**。
#     PYTHONUSERBASE 指向 rby 自己的目录，user-site 在 sys.path 里优先于系统
#     site-packages，因此既补得上、又只对 source 过本文件的 shell 生效。

# --- 私有 user-site：pip install --user 的落点 ---
export PYTHONUSERBASE=/workspace/mount/b100_zaip_data/rby/pyuser

# --- 仓库根加入 import 路径（models/ dataset/ utils/ 都是仓库内包）---
export PYTHONPATH=".:${PYTHONPATH:-}"

# --- 数据与标注目录（experiments/README.md 约定，见 SETUP_B100.md §4）---
export MVC_DATA_DIR="${MVC_DATA_DIR:-data}"
export MVC_ANNO_DIR="${MVC_ANNO_DIR:-anno_downstream}"

echo "PYTHONUSERBASE=$PYTHONUSERBASE"
echo "python         = $(command -v python)  ($(python -V 2>&1))"
echo "MVC_DATA_DIR=$MVC_DATA_DIR MVC_ANNO_DIR=$MVC_ANNO_DIR"
echo
echo "提示：装包一律用   python -m pip install --user --no-deps <包>"
echo "     禁止裸  pip install <包>==<版本>  —— 会改到共用系统环境。"
