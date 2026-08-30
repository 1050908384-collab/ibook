"""Audit local Markdown lessons and render a reviewable structural report.

This script is read-only with respect to the source lessons and database.  It
ignores fenced code blocks, because source-code comments are not Markdown
headings, and writes a report under ``docs`` for editorial review.
"""
from __future__ import annotations


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

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "知识库原始数据"
REPORT_PATH = PROJECT_ROOT / "docs" / "知识库Markdown标题层级审计报告.md"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    line_number: int
    offset: int


@dataclass(frozen=True)
class CandidateChunk:
    title_path: str
    level: int
    chars: int
    start_line: int
    end_line: int


def lesson_files() -> list[Path]:
    """Return exactly the per-subtopic lesson files used by the importer."""
    files: list[Path] = []
    for block_dir in sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir()):
        for subtopic_dir in sorted(path for path in block_dir.iterdir() if path.is_dir()):
            files.extend(sorted(subtopic_dir.glob("*.md")))
    return files


def headings_in(content: str) -> list[Heading]:
    headings: list[Heading] = []
    in_fence = False
    offset = 0
    for line_number, line in enumerate(content.splitlines(keepends=True), start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not in_fence:
            match = HEADING_RE.match(line.rstrip("\r\n"))
            if match:
                headings.append(Heading(
                    level=len(match.group(1)),
                    title=match.group(2),
                    line_number=line_number,
                    offset=offset,
                ))
        offset += len(line)
    return headings


def candidate_chunks(content: str, headings: list[Heading]) -> tuple[int, list[CandidateChunk]]:
    """Use H3 chunks when present; otherwise use H2 chunks, retaining its path."""
    cut_level = 3 if any(item.level == 3 for item in headings) else 2
    starts = [item for item in headings if item.level == cut_level]
    if not starts:
        return cut_level, []

    chunks: list[CandidateChunk] = []
    for index, heading in enumerate(starts):
        end_offset = len(content)
        # A candidate ends at the next heading at this level or a shallower level.
        for next_heading in headings:
            if next_heading.offset > heading.offset and next_heading.level <= cut_level:
                end_offset = next_heading.offset
                break

        # The most recent heading at each parent level is the relevant context.
        path_parts: list[str] = []
        for parent_level in range(1, cut_level):
            parent = next(
                (item.title for item in reversed(headings)
                 if item.offset <= heading.offset and item.level == parent_level),
                None,
            )
            if parent:
                path_parts.append(parent)
        path_parts.append(heading.title)
        end_line = content.count("\n", 0, end_offset) + 1
        chunks.append(CandidateChunk(
            title_path=" > ".join(path_parts),
            level=cut_level,
            chars=len(content[heading.offset:end_offset].strip()),
            start_line=heading.line_number,
            end_line=end_line,
        ))
    return cut_level, chunks


def issues_for(headings: list[Heading]) -> list[str]:
    issues: list[str] = []
    if not headings:
        return ["无 ATX 标题"]
    h1_count = sum(item.level == 1 for item in headings)
    if h1_count == 0:
        issues.append("缺少 H1 文档标题")
    elif h1_count > 1:
        issues.append(f"存在 {h1_count} 个 H1；需保留一个文档标题，其余降级")
    if not any(item.level == 3 for item in headings):
        issues.append("缺少 H3；候选块将回退至 H2")
    for previous, current in zip(headings, headings[1:]):
        if current.level > previous.level + 1:
            issues.append(f"标题越级：L{previous.line_number} H{previous.level} -> L{current.line_number} H{current.level}")
    if max(item.level for item in headings) > 4:
        issues.append("使用 H5/H6；不符合拟定的四级上限")
    return issues or ["无自动检测到的结构异常"]


def render_tree(headings: list[Heading]) -> list[str]:
    return [f"{'  ' * (item.level - 1)}- H{item.level} L{item.line_number}: {item.title}" for item in headings]


def main() -> None:
    files = lesson_files()
    if len(files) != 73:
        raise RuntimeError(f"Expected 73 lesson files, found {len(files)}")

    records = []
    total_candidates = 0
    for path in files:
        content = path.read_text(encoding="utf-8-sig")
        headings = headings_in(content)
        cut_level, candidates = candidate_chunks(content, headings)
        issues = issues_for(headings)
        records.append((path, content, headings, cut_level, candidates, issues))
        total_candidates += len(candidates)
    candidate_sizes = [chunk.chars for record in records for chunk in record[4]]

    lines = [
        "# 知识库 Markdown 标题层级审计报告",
        "",
        "- 审计对象：`data/知识库原始数据` 中被导入器发现的 73 篇子知识点文章。",
        "- 审计方式：忽略 fenced code block；仅识别 ATX 标题（`#` 至 `######`）。",
        "- 候选分块：当前结构中有 H3 的文章按 H3 估算；没有 H3 的文章按 H2 估算。字符数包含该标题至下一同级或更高层标题之前的文本。",
        "- 注意：候选块仅用于评审当前标题结构，不代表最终入库规则；最终规则见 `知识库Markdown编写规范.md`。",
        "",
        "## 汇总",
        "",
        f"- 文档数：{len(records)}",
        f"- 当前候选块数：{total_candidates}",
        f"- 多 H1 文档：{sum(sum(item.level == 1 for item in record[2]) > 1 for record in records)} 篇",
        f"- 缺 H1 文档：{sum(not any(item.level == 1 for item in record[2]) for record in records)} 篇",
        f"- 缺 H3 文档：{sum(not any(item.level == 3 for item in record[2]) for record in records)} 篇",
        f"- 使用 H4 的文档：{sum(any(item.level == 4 for item in record[2]) for record in records)} 篇",
        f"- 候选块小于 300 字符：{sum(size < 300 for size in candidate_sizes)} 个",
        f"- 候选块在 300 至 1,800 字符：{sum(300 <= size <= 1800 for size in candidate_sizes)} 个",
        f"- 候选块大于 1,800 字符：{sum(size > 1800 for size in candidate_sizes)} 个",
        f"- 候选块字符数范围：{min(candidate_sizes)} 至 {max(candidate_sizes)}",
        "",
        "## 逐篇审计",
        "",
    ]

    for number, (path, _content, headings, cut_level, candidates, issues) in enumerate(records, start=1):
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        levels = Counter(item.level for item in headings)
        distribution = ", ".join(f"H{level}={levels[level]}" for level in range(1, 7) if levels[level]) or "无"
        lines.extend([
            f"### {number:02d}. `{relative_path}`",
            "",
            f"- 标题分布：{distribution}",
            f"- 候选切分层级：H{cut_level}",
            f"- 自动检测：{'；'.join(issues)}",
            "",
            "#### 标题树",
            "",
            *render_tree(headings),
            "",
            "#### 候选块字符数",
            "",
        ])
        if candidates:
            lines.extend(["| 序号 | 标题路径 | 行范围 | 字符数 |", "| --- | --- | --- | ---: |"])
            for index, chunk in enumerate(candidates, start=1):
                title = chunk.title_path.replace("|", "\\|")
                lines.append(f"| {index} | {title} | {chunk.start_line}-{chunk.end_line} | {chunk.chars} |")
        else:
            lines.append("无可用候选块；需要先补充 H2/H3 标题。")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
