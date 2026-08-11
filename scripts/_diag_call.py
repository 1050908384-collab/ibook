# -*- coding: utf-8 -*-
"""看 OllamaEmbeddingFunction 内部 call 函数实现和所用端点。"""
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
