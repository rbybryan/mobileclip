"""Marker file — 别删。

`scripts/` 没有 `__init__.py` 时是 PEP 420 **命名空间包**，而命名空间包的优先级
**低于普通包**：Python 扫 `sys.path` 时，遇到无 `__init__.py` 的同名目录只记一个
"portion" 就继续往后扫，后面任何一个带 `__init__.py` 的 `scripts` 都会反超它。
b100 的 site-packages 里恰好有这么一个，于是
`from scripts.pretraining.clip.mobileviclip_small.config import *` 报的是
`No module named 'scripts.pretraining'`（注意不是 `'scripts'` —— 名字被别人占了）。
加上本文件后 `scripts` 成为普通包，在 PYTHONPATH 首次命中即返回，不会再被顶掉。
详见 TROUBLESHOOTING.md「`No module named 'scripts.pretraining'`」。
"""
