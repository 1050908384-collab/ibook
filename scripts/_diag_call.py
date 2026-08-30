# -*- coding: utf-8 -*-
"""看 OllamaEmbeddingFunction 内部 call 函数实现和所用端点。"""

# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------

import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in sys.path: sys.path.insert(0, str(_PROJ_ROOT_))
_BE_ = Path(_PROJ_ROOT_, "backend")
if str(_BE_) not in sys.path: sys.path.insert(0, str(_BE_))

import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

import sys, inspect, re
from chromadb.utils import embedding_functions

module = inspect.getmodule(embedding_functions.OllamaEmbeddingFunction)
src = inspect.getsource(module)
# 找 call 函数定义
idx = src.find("def call(")
if idx >= 0:
    lines = src[idx:].splitlines()[:120]
    print("\n".join(lines))
else:
    print("没找到 call 函数定义，打印所有顶层定义:")
    for m in re.findall(r"^(?:def|class) \w+", src, re.MULTILINE):
        print(" ", m)

# 自删
try: os.remove(__file__)
except: pass
