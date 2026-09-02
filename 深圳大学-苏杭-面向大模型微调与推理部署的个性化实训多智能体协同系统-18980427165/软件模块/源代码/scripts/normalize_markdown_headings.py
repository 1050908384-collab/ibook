# -*- coding: utf-8 -*-
"""一次性脚本：统一 73 篇学习文章的 Markdown 标题层级。

修复规则（对齐 docs/知识库Markdown编写规范.md）：
- 缺少 H1：从种子文件取子知识点名，在文件开头补 `# {name}`
- 多个 H1：保留第一个 H1；从第二个 H1 起，所有标题层级 +1（H1→H2, H2→H3, H3→H4），
  H4 封顶不升 H5
- H5/H6：降为 H4，输出 warning
- 标题跳级（如 H1→H3 无 H2）：输出 warning，不自动修复

安全保证：
- 严格忽略 fenced code block（``` 包裹）内的 `#`
- 不删除/修改任何正文、代码、表格、链接
- 可安全重复运行（已修复的文件再跑不会变）
"""
from __future__ import annotations


# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------
import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from import_local_learning_materials import discover_lessons, _load_subtopics


def _clean_subtopic_name(name: str) -> str:
    """去掉子知识点名外层包裹括号，作为 H1 标题。"""
    name = name or ""
    while len(name) >= 2 and name[0] in "（(" and name[-1] in "）)":
        name = name[1:-1]
    return name.strip()


def _parse_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """解析 ATX 标题，忽略 code block。返回 [(line_idx, level, text), ...]。"""
    headings: list[tuple[int, int, str]] = []
    in_code = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        # ATX 标题：行首 1-6 个 # 后跟空格
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))
    return headings


def _set_heading_level(line: str, new_level: int) -> str:
    """替换标题行的 # 数量，保留后续文本和换行。"""
    m = re.match(r"^(#{1,6})(\s+.*)$", line)
    if not m:
        return line
    return "#" * new_level + m.group(2)


def normalize_file(path: Path, subtopic_name: str) -> dict:
    """归一化单个 Markdown 文件，返回变更摘要。"""
    content = path.read_text(encoding="utf-8-sig")
    lines = content.splitlines(keepends=True)
    changes = {
        "missing_h1_fixed": False,
        "multi_h1_demoted": 0,
        "h5_h6_capped": 0,
        "level_skip_warnings": [],
    }

    headings = _parse_headings(lines)

    # --- 修复 1：缺少 H1 ---
    h1_indices = [h[0] for h in headings if h[1] == 1]
    if not h1_indices:
        clean_name = _clean_subtopic_name(subtopic_name) or path.stem
        # 在文件最前面补 H1 + 空行
        lines.insert(0, f"# {clean_name}\n\n")
        changes["missing_h1_fixed"] = True
        # 重新解析（行号已偏移）
        headings = _parse_headings(lines)
        h1_indices = [h[0] for h in headings if h[1] == 1]

    # --- 修复 2：多个 H1 ---
    # 从第二个 H1 起，所有标题层级 +1（H4 封顶）
    if len(h1_indices) > 1:
        second_h1_line = h1_indices[1]
        for line_idx, level, _text in headings:
            if line_idx < second_h1_line:
                continue
            new_level = min(level + 1, 4)
            if new_level != level:
                lines[line_idx] = _set_heading_level(lines[line_idx], new_level)
                if level == 1:
                    changes["multi_h1_demoted"] += 1
                elif level >= 5:
                    changes["h5_h6_capped"] += 1

    # --- 修复 3：H5/H6 → H4（处理未被多 H1 修复覆盖的部分）---
    headings = _parse_headings(lines)
    for line_idx, level, _text in headings:
        if level >= 5:
            lines[line_idx] = _set_heading_level(lines[line_idx], 4)
            changes["h5_h6_capped"] += 1

    # --- 检查 4：标题跳级 warning ---
    headings = _parse_headings(lines)
    prev_level = 0
    for line_idx, level, text in headings:
        if prev_level > 0 and level > prev_level + 1:
            changes["level_skip_warnings"].append(
                f"    L{line_idx + 1}: H{prev_level}→H{level} 跳级: {text[:60]}"
            )
        prev_level = level

    # 写回（仅在内容变化时）
    new_content = "".join(lines)
    if new_content != content:
        path.write_text(new_content, encoding="utf-8")

    return changes


def main() -> None:
    lessons = discover_lessons()
    subtopics_by_block = _load_subtopics()
    # 构建 subtopic_id → subtopic_name 映射
    subtopic_names: dict[str, str] = {}
    for records in subtopics_by_block.values():
        for record in records:
            subtopic_names[record["subtopic_id"]] = record.get("subtopic_name", "")

    print(f"{'=' * 70}")
    print(f"标题归一化：共 {len(lessons)} 篇文章")
    print(f"{'=' * 70}")

    total_missing = 0
    total_multi = 0
    total_h56 = 0
    total_skip = 0
    total_changed = 0

    for lesson in lessons:
        name = subtopic_names.get(lesson.subtopic_id, lesson.subtopic_id)
        changes = normalize_file(lesson.path, name)

        changed = (
            changes["missing_h1_fixed"]
            or changes["multi_h1_demoted"]
            or changes["h5_h6_capped"]
        )
        if changed:
            total_changed += 1
        if changes["missing_h1_fixed"]:
            total_missing += 1
        if changes["multi_h1_demoted"]:
            total_multi += 1
        if changes["h5_h6_capped"]:
            total_h56 += 1
        if changes["level_skip_warnings"]:
            total_skip += 1

        # 每篇输出摘要
        flags = []
        if changes["missing_h1_fixed"]:
            flags.append("补H1")
        if changes["multi_h1_demoted"]:
            flags.append(f"多H1降{changes['multi_h1_demoted']}")
        if changes["h5_h6_capped"]:
            flags.append(f"H5/6→H4({changes['h5_h6_capped']})")
        if changes["level_skip_warnings"]:
            flags.append(f"跳级warning({len(changes['level_skip_warnings'])})")

        status = " | ".join(flags) if flags else "无需修改"
        rel_path = lesson.path.relative_to(PROJECT_ROOT)
        print(f"  {lesson.subtopic_id}: {status}  ({rel_path})")
        for warn in changes["level_skip_warnings"]:
            print(warn)

    print(f"\n{'=' * 70}")
    print(f"汇总：修改 {total_changed} 篇 | 补H1 {total_missing} | 多H1降级 {total_multi} | H5/6→H4 {total_h56} | 跳级warning {total_skip}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
