# -*- coding: utf-8 -*-
"""精查 _chunk_by_headings 对 5 篇 KB-01 md 的切块结果，找出为什么尾部章节丢了。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from import_local_learning_materials import _chunk_by_headings  # noqa: E402

KB01 = PROJECT_ROOT / "data" / "知识库原始数据" / "01_Linux环境与命令行操作"

files = [
    KB01 / "1.环境安装/Linux板块-环境准备（最终版）.md",
    KB01 / "2.常用Shell命令/常用Shell命令（cd ls mkdir cp mv）.md",
    KB01 / "3.环境变量配置/环境变量配置.md",
    KB01 / "4.conda pip虚拟环境管理/conda  pip 虚拟环境管理.md",
    KB01 / "5.nvidia-smi显存监控/nvidia-smi显存监控.md",
]

def _title_and_content(p: Path):
    text = p.read_text(encoding="utf-8-sig", errors="ignore")
    # 去掉首行 "# xxx" 和紧随其后的参考链接块到第一个 "---"
    lines = text.splitlines(keepends=True)
    first_h1 = -1
    first_hr = None
    for i, ln in enumerate(lines):
        if first_h1 < 0 and ln.lstrip().startswith("# "):
            first_h1 = i
            continue
        if first_h1 >= 0 and (ln.rstrip() == "---" or ln.rstrip() == "***"):
            first_hr = i
            break
    start = first_hr + 1 if first_hr is not None else (first_h1 + 1 if first_h1 >= 0 else 0)
    title = ""
    if first_h1 >= 0:
        import re
        m = re.match(r"#\s+(.+?)\s*$", lines[first_h1])
        if m:
            title = m.group(1).strip()
    content = "".join(lines[start:])
    return title, content

def _dump_top_level_h2s(content: str):
    """打印原文 ## 级标题列表。"""
    titles = []
    for ln in content.splitlines():
        s = ln.strip()
        if s.startswith("## ") and not s.startswith("### "):
            titles.append(s[3:].strip())
    return titles

for fp in files:
    print("\n" + "=" * 100)
    print(f"FILE: {fp.relative_to(PROJECT_ROOT)}")
    title, content = _title_and_content(fp)
    print(f"  title={title}   body_len={len(content)}")
    h2s = _dump_top_level_h2s(content)
    print(f"  原文 ## 级大节共 {len(h2s)} 节：")
    for t in h2s:
        print(f"    - {t[:80]}")
    chunks = _chunk_by_headings(content)
    print(f"\n  _chunk_by_headings 产出 {len(chunks)} chunks：")
    for i, c in enumerate(chunks, 1):
        leaf = c.heading_path.split(" > ")[-1] if c.heading_path else ""
        parent = " > ".join(c.heading_path.split(" > ")[:-1]) if c.heading_path else ""
        first_line = ""
        for ln in c.content.splitlines():
            if ln.strip():
                first_line = ln.strip()[:80]
                break
        print(f"    {i:2d}. heading_path={c.heading_path[:90]}")
        print(f"         首行内容: {first_line}")
        print(f"         len={len(c.content):>5d} lines={len(c.content.splitlines()):>3d} strategy={c.chunk_strategy}  level={c.heading_level}")
        # 检测尾部章节：若原文 h2s 列表末尾的标题，在所有 chunk leaf+parent 里找不到 → 标缺失
    print(f"\n  原文尾部大节是否全部出现在 chunks 的 heading_path 中？")
    covered_top = set()
    for c in chunks:
        parts = c.heading_path.split(" > ") if c.heading_path else []
        # 加入所有 二级大节 名称（就是 parts[0] 如果有 2+ 层 parts）
        if len(parts) >= 1:
            covered_top.add(parts[0].strip())
    missing_top = [t for t in h2s if t.split("（")[0].strip() not in [x.split("（")[0].strip() for x in covered_top]]
    if missing_top:
        print(f"    ⚠️ 缺失 {len(missing_top)} 节原文 ## 大节（chunk 没把它识别为 heading_path）：")
        for t in missing_top:
            print(f"        - {t[:100]}")
    else:
        print(f"    ✅ 全部 ## 大节都在 chunks heading_path 里")
    print()
