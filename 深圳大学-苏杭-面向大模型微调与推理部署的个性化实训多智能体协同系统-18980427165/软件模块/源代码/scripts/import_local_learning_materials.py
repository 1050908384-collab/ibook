"""Import reviewed local Markdown lessons into the runtime knowledge database.

The source directory remains the editable, version-controlled source of truth.
This importer owns only ``LOCAL-*`` documents and their chunks, so existing seed
documents, videos, questions, and user data are left untouched.

分块策略（对齐 docs/知识库Markdown编写规范.md）：
- 按标题层级分块，而非纯字符累计
- 有 H3 的文章：每个 H3 及其内容 = 候选块（heading_h3）
- 无 H3 的文章：每个 H2 及其内容 = 候选块（heading_h2_fallback）
- 候选块 < 300 字符：与下一个同父级候选块合并
- 候选块 > 1800 字符：优先按 H4 拆分（heading_h4_split）；仍超长再按段落拆分（paragraph_overflow）
- 每个 chunk 记录 heading_path / heading_level / source_line_start / source_line_end / chunk_strategy
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

import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "知识库原始数据"
SUBTOPICS_FILE = PROJECT_ROOT / "data" / "seeds" / "knowledge_subtopics.json"
DATABASE_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

MIN_CHUNK_CHARS = 300
MAX_CHUNK_CHARS = 1800


@dataclass(frozen=True)
class LessonFile:
    path: Path
    block_id: str
    subtopic_id: str
    difficulty: str


@dataclass
class Chunk:
    """一个标题级知识分块。"""
    heading_path: str
    heading_level: int
    content: str
    source_line_start: int   # 1-based
    source_line_end: int     # 1-based
    chunk_strategy: str
    chunk_type: str = "text"


# ------------------------------------------------------------------
# 文章发现与元数据（保持不变）
# ------------------------------------------------------------------

def _directory_order(name: str) -> int:
    match = re.match(r"(\d+)(?:[_.、]|$)", name)
    if not match:
        raise ValueError(f"目录缺少数字顺序前缀: {name}")
    return int(match.group(1))


def _load_subtopics() -> dict[str, list[dict]]:
    records = json.loads(SUBTOPICS_FILE.read_text(encoding="utf-8"))
    by_block: dict[str, list[dict]] = {}
    for record in records:
        by_block.setdefault(record["block_id"], []).append(record)
    for records_in_block in by_block.values():
        records_in_block.sort(key=lambda item: item["sequence_index"])
    return by_block


def discover_lessons() -> list[LessonFile]:
    subtopics_by_block = _load_subtopics()
    lessons: list[LessonFile] = []
    discovered_subtopic_ids: set[str] = set()

    for block_dir in sorted(SOURCE_DIR.iterdir()):
        if not block_dir.is_dir():
            continue
        block_number = _directory_order(block_dir.name)
        block_id = f"KB-{block_number:02d}"
        subtopics = subtopics_by_block.get(block_id, [])
        if not subtopics:
            raise ValueError(f"{block_dir} does not map to a seeded knowledge block")
        for subtopic_dir in sorted((item for item in block_dir.iterdir() if item.is_dir()), key=lambda item: _directory_order(item.name)):
            index = _directory_order(subtopic_dir.name) - 1
            if index < 0 or index >= len(subtopics):
                raise ValueError(f"{subtopic_dir} 无法映射到 {block_id} 的子知识点")
            markdown_files = sorted(subtopic_dir.glob("*.md"))
            if len(markdown_files) != 1:
                raise ValueError(f"{subtopic_dir} 应包含且只包含一个 Markdown 文件")
            markdown_file = markdown_files[0]
            if markdown_file.stat().st_size == 0:
                continue
            subtopic_id = subtopics[index]["subtopic_id"]
            if subtopic_id in discovered_subtopic_ids:
                raise ValueError(f"Duplicate local lesson mapping for {subtopic_id}")
            discovered_subtopic_ids.add(subtopic_id)
            lessons.append(LessonFile(
                path=markdown_file,
                block_id=block_id,
                subtopic_id=subtopic_id,
                difficulty=subtopics[index].get("difficulty") or "中级",
            ))

    expected_subtopic_ids = {
        record["subtopic_id"]
        for records in subtopics_by_block.values()
        for record in records
    }
    missing_subtopic_ids = sorted(expected_subtopic_ids - discovered_subtopic_ids)
    if missing_subtopic_ids:
        raise ValueError(
            "Missing local lesson directories for seeded subtopics: "
            + ", ".join(missing_subtopic_ids)
        )
    return lessons


def _title_and_content(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError(f"资料为空: {path}")
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip(), content
    return path.stem, content


def _reference_url(content: str) -> str:
    """Use the first curated external link as the learner-facing article URL."""
    match = re.search(r"https?://[^\s)>\]}`]+", content)
    return match.group(0).rstrip(".,，。;；") if match else ""


# ------------------------------------------------------------------
# 标题级分块核心
# ------------------------------------------------------------------

def _parse_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """解析 ATX 标题，忽略 fenced code block。返回 [(line_idx_0based, level, text), ...]。"""
    headings: list[tuple[int, int, str]] = []
    in_code = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))
    return headings


def _parent_path(heading_path: str) -> str:
    """取 heading_path 的父路径（去掉最后一段）。"""
    if " > " in heading_path:
        return heading_path.rsplit(" > ", 1)[0]
    return ""


def _chunk_by_headings(content: str) -> list[Chunk]:
    """按标题层级分块。

    严格对齐《知识库Markdown编写规范.md》：
    - 有 H3 的文章：H3 为唯一块边界，H2 章节内容并入最近的 H3 块
      （首个 H3 前的 H2 → 前导；H3 之间的 H2 → 前一个 H3 的结尾；
      末尾 H3 后的 H2 → 最后一个 H3 的收尾）
    - 无 H3 的文章：H2 为块边界（heading_h2_fallback）
    """
    lines = content.splitlines(keepends=True)
    headings = _parse_headings(lines)

    if not headings:
        return [Chunk(
            heading_path="", heading_level=0, content=content.strip(),
            source_line_start=1, source_line_end=len(lines),
            chunk_strategy="paragraph_overflow",
        )]

    has_h3 = any(h[1] == 3 for h in headings)

    if has_h3:
        return _chunk_with_h3(lines, headings)
    else:
        return _chunk_without_h3(lines, headings)


def _chunk_with_h3(lines: list[str], headings: list[tuple[int, int, str]]) -> list[Chunk]:
    """有 H3 的文章：**H2 与 H3 均视为块边界**。

    原问题：旧实现只把 H3 当块边界，导致没有 `### x.x` 小节的「## 总结」
    「## 常见问题」「## 为什么需要 xx」这类 H2 大节，整段内容被"吞并"到
    了文件中最后一个 H3 块的 content 末尾，导致 heading_path 缺失该大节。

    新规则：
    - 任何 `##` 标题（H2）单独产出 1 块，heading_path = "H1 > H2标题"，
      heading_level=2，chunk_strategy="heading_h2_standalone"。
    - 任何 `###` 标题（H3）仍产出 1 块，heading_path = "H1 > H2 > H3标题"，
      heading_level=3，chunk_strategy="heading_h3"。
    - H1/首行引言（第一个 H2/H3 之前的文字）**并入第一块开头**，保持原开篇不丢。
    """
    # 先筛选：只看 L=1 / L=2 / L=3 的标题，按行号排序（headings 已经有序）
    relevant_h = [(idx, lv, tx) for idx, lv, tx in headings if lv in (1, 2, 3)]
    # 构建 2/3 级标题序列（作为真正切分边界的"锚"）
    anchors: list[tuple[int, int, str, str]] = []  # (line_idx, level, text, heading_path)
    h1_text = ""
    h2_text = ""
    for idx, lv, tx in relevant_h:
        if lv == 1:
            h1_text = tx
            h2_text = ""
            continue
        if lv == 2:
            h2_text = tx
            path = " > ".join(p for p in (h1_text, h2_text) if p)
            anchors.append((idx, 2, tx, path))
        elif lv == 3:
            path = " > ".join(p for p in (h1_text, h2_text, tx) if p)
            anchors.append((idx, 3, tx, path))

    if not anchors:
        return [Chunk(
            heading_path="", heading_level=0,
            content="".join(lines).strip(),
            source_line_start=1, source_line_end=len(lines),
            chunk_strategy="paragraph_overflow",
        )]

    raw_chunks: list[Chunk] = []
    for i, (start_idx, level, _text, hpath) in enumerate(anchors):
        # 第一个锚：chunk_start_idx = 0（把文章开头的 H1/引言纳入）
        chunk_start_idx = 0 if i == 0 else start_idx
        # chunk_end_idx = 下一个锚之前 1 行；末尾锚到文件最后一行
        chunk_end_idx = (anchors[i + 1][0] - 1) if i + 1 < len(anchors) else (len(lines) - 1)
        chunk_lines = lines[chunk_start_idx : chunk_end_idx + 1]
        chunk_content = "".join(chunk_lines).strip()
        strategy = "heading_h2_standalone" if level == 2 else "heading_h3"
        raw_chunks.append(Chunk(
            heading_path=hpath,
            heading_level=level,
            content=chunk_content,
            source_line_start=chunk_start_idx + 1,
            source_line_end=chunk_end_idx + 1,
            chunk_strategy=strategy,
        ))

    # 后处理：合并小块 / 拆分大块（继承原流程）
    raw_chunks = _merge_small_chunks(raw_chunks)
    raw_chunks = _split_large_chunks(raw_chunks, lines)
    raw_chunks = _merge_small_chunks(raw_chunks)
    return raw_chunks


def _chunk_without_h3(lines: list[str], headings: list[tuple[int, int, str]]) -> list[Chunk]:
    """无 H3 的文章：H2 为块边界。"""
    h2_headings = [(idx, text) for idx, level, text in headings if level == 2]

    if not h2_headings:
        return [Chunk(
            heading_path="", heading_level=0,
            content="".join(lines).strip(),
            source_line_start=1, source_line_end=len(lines),
            chunk_strategy="paragraph_overflow",
        )]

    # 为每个 H2 构建 heading_path（H1 > H2）
    h1_text = ""
    h2_paths: dict[int, str] = {}
    for idx, level, text in headings:
        if level == 1:
            h1_text = text
        elif level == 2:
            parts = [p for p in [h1_text, text] if p]
            h2_paths[idx] = " > ".join(parts)

    raw_chunks: list[Chunk] = []
    for i, (h2_idx, _) in enumerate(h2_headings):
        chunk_start_idx = h2_idx
        if i == 0:
            chunk_start_idx = 0

        chunk_end_idx = len(lines) - 1
        if i + 1 < len(h2_headings):
            chunk_end_idx = h2_headings[i + 1][0] - 1

        chunk_lines = lines[chunk_start_idx : chunk_end_idx + 1]
        chunk_content = "".join(chunk_lines).strip()

        heading_path = h2_paths.get(h2_idx, "")

        raw_chunks.append(Chunk(
            heading_path=heading_path,
            heading_level=2,
            content=chunk_content,
            source_line_start=chunk_start_idx + 1,
            source_line_end=chunk_end_idx + 1,
            chunk_strategy="heading_h2_fallback",
        ))

    raw_chunks = _merge_small_chunks(raw_chunks)
    raw_chunks = _split_large_chunks(raw_chunks, lines)
    raw_chunks = _merge_small_chunks(raw_chunks)

    return raw_chunks


def _merge_small_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """合并 < MIN_CHUNK_CHARS 的块（但 H2 独立块绝不合并，这是用户硬要求）。

    - 第一轮：向前合并（小块 + 后一个同父级块），H2 独立块跳过；
    - 第二轮：向后合并仍小的尾块并入前一个块（不限父级），H2 独立块跳过。
    """
    if len(chunks) <= 1:
        return chunks

    _H2_STANDALONE = "heading_h2_standalone"

    # 第一轮：向前合并（小块并入后一个同父级块）
    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        while (
            current.chunk_strategy != _H2_STANDALONE
            and i + 1 < len(chunks)
            and chunks[i + 1].chunk_strategy != _H2_STANDALONE
            and len(current.content) < MIN_CHUNK_CHARS
            and _parent_path(current.heading_path) == _parent_path(chunks[i + 1].heading_path)
        ):
            nxt = chunks[i + 1]
            current = Chunk(
                heading_path=current.heading_path,
                heading_level=current.heading_level,
                content=current.content + "\n\n" + nxt.content,
                source_line_start=current.source_line_start,
                source_line_end=nxt.source_line_end,
                chunk_strategy=current.chunk_strategy,
            )
            i += 1
        merged.append(current)
        i += 1

    # 第二轮：向后合并仍小的块（不限父级，并入前一个块）—— H2 独立块永不参与
    if len(merged) <= 1:
        return merged
    final: list[Chunk] = []
    for chunk in merged:
        if (
            chunk.chunk_strategy != _H2_STANDALONE
            and len(chunk.content) < MIN_CHUNK_CHARS
            and final
            and final[-1].chunk_strategy != _H2_STANDALONE
        ):
            prev = final[-1]
            prev = Chunk(
                heading_path=prev.heading_path,
                heading_level=prev.heading_level,
                content=prev.content + "\n\n" + chunk.content,
                source_line_start=prev.source_line_start,
                source_line_end=chunk.source_line_end,
                chunk_strategy=prev.chunk_strategy,
            )
            final[-1] = prev
        else:
            final.append(chunk)
    return final


def _split_large_chunks(chunks: list[Chunk], lines: list[str], _depth: int = 0) -> list[Chunk]:
    """拆分 > MAX_CHUNK_CHARS 的块：优先按 H4，再按段落。代码围栏不可截断。"""
    if _depth > 3:  # 递归深度保护，超过则强制段落拆分
        return _split_large_chunks_fallback(chunks, lines)

    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.content) <= MAX_CHUNK_CHARS:
            result.append(chunk)
            continue

        # 尝试按 H4 拆分
        h4_positions = _find_h4_in_range(lines, chunk.source_line_start - 1, chunk.source_line_end)
        if h4_positions:
            sub_chunks = _split_by_h4(chunk, lines, h4_positions)
            # 如果 H4 拆分后仍有超长块，递归深度+1继续；超过3层则用段落兜底
            still_large = [c for c in sub_chunks if len(c.content) > MAX_CHUNK_CHARS]
            if still_large:
                sub_chunks = _split_large_chunks(sub_chunks, lines, _depth + 1)
            result.extend(sub_chunks)
        else:
            # 按段落拆分
            sub_chunks = _split_by_paragraphs(chunk, lines)
            result.extend(sub_chunks)
    return result


def _split_large_chunks_fallback(chunks: list[Chunk], lines: list[str]) -> list[Chunk]:
    """递归深度耗尽时的兜底：所有超长块强制按段落拆分。"""
    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.content) <= MAX_CHUNK_CHARS:
            result.append(chunk)
        else:
            sub_chunks = _split_by_paragraphs(chunk, lines)
            result.extend(sub_chunks)
    return result


def _find_h4_in_range(lines: list[str], start_idx: int, end_idx: int) -> list[tuple[int, str]]:
    """在行范围内查找 H4 标题（忽略 code block）。"""
    h4s: list[tuple[int, str]] = []
    in_code = False
    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = re.match(r"^(#{4})\s+(.+)$", line)
        if m:
            h4s.append((i, m.group(2).strip()))
    return h4s


def _split_by_h4(chunk: Chunk, lines: list[str], h4_positions: list[tuple[int, str]]) -> list[Chunk]:
    """按 H4 标题拆分超长块。前导内容过小（纯父标题）时并入首个 H4 子块，避免标题-only 块。"""
    # H4 之前的部分（属于父块的前导内容）
    first_h4_idx = h4_positions[0][0]
    pre_content = ""
    pre_start = chunk.source_line_start - 1
    if first_h4_idx > pre_start:
        pre_lines = lines[pre_start : first_h4_idx]
        pre_content = "".join(pre_lines).strip()

    # 每个 H4 形成一个子块
    h4_sub_chunks: list[Chunk] = []
    for i, (h4_idx, h4_text) in enumerate(h4_positions):
        # 子块结束：下一个 H4 或父块结束
        if i + 1 < len(h4_positions):
            sub_end_idx = h4_positions[i + 1][0] - 1
        else:
            sub_end_idx = chunk.source_line_end - 1

        sub_lines = lines[h4_idx : sub_end_idx + 1]
        sub_content = "".join(sub_lines).strip()

        # heading_path 加上 H4 标题
        sub_path = f"{chunk.heading_path} > {h4_text}"

        h4_sub_chunks.append(Chunk(
            heading_path=sub_path,
            heading_level=4,
            content=sub_content,
            source_line_start=h4_idx + 1,
            source_line_end=sub_end_idx + 1,
            chunk_strategy="heading_h4_split",
        ))

    # 前导内容处理：足够大则独立成块；过小（通常只有父标题）则并入首个 H4 子块
    sub_chunks: list[Chunk] = []
    if pre_content and len(pre_content) >= MIN_CHUNK_CHARS:
        sub_chunks.append(Chunk(
            heading_path=chunk.heading_path,
            heading_level=chunk.heading_level,
            content=pre_content,
            source_line_start=pre_start + 1,
            source_line_end=first_h4_idx,
            chunk_strategy=chunk.chunk_strategy,
        ))
        sub_chunks.extend(h4_sub_chunks)
    elif pre_content and h4_sub_chunks:
        first = h4_sub_chunks[0]
        h4_sub_chunks[0] = Chunk(
            heading_path=first.heading_path,
            heading_level=first.heading_level,
            content=pre_content + "\n\n" + first.content,
            source_line_start=pre_start + 1,
            source_line_end=first.source_line_end,
            chunk_strategy=first.chunk_strategy,
        )
        sub_chunks.extend(h4_sub_chunks)
    else:
        sub_chunks.extend(h4_sub_chunks)

    return sub_chunks


def _split_by_paragraphs(chunk: Chunk, lines: list[str]) -> list[Chunk]:
    """按段落拆分超长块，不截断代码围栏。"""
    start_idx = chunk.source_line_start - 1
    end_idx = chunk.source_line_end - 1

    # 按空行分段，但不拆分 code block
    segments: list[tuple[int, int]] = []
    seg_start = start_idx
    in_code = False
    for i in range(start_idx, end_idx + 1):
        line = lines[i] if i < len(lines) else ""
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
        if not in_code and not line.strip() and i > seg_start:
            segments.append((seg_start, i - 1))
            seg_start = i + 1
    if seg_start <= end_idx:
        segments.append((seg_start, end_idx))

    # 将 `---` 分隔线段并入前一段，避免产生独立微型块
    filtered: list[tuple[int, int]] = []
    for seg_s, seg_e in segments:
        seg_text = "".join(lines[seg_s : seg_e + 1]).strip()
        if seg_text == "---" and filtered:
            prev_s, _ = filtered[-1]
            filtered[-1] = (prev_s, seg_e)
        else:
            filtered.append((seg_s, seg_e))
    segments = filtered

    # 合并段落直到接近 MAX_CHUNK_CHARS
    sub_chunks: list[Chunk] = []
    cur_start = None
    cur_end = None
    cur_len = 0
    for seg_s, seg_e in segments:
        seg_text = "".join(lines[seg_s : seg_e + 1])
        if cur_start is None:
            cur_start = seg_s
            cur_end = seg_e
            cur_len = len(seg_text)
        elif cur_len + len(seg_text) + 2 > MAX_CHUNK_CHARS:
            # 输出当前块，开始新块
            content = "".join(lines[cur_start : cur_end + 1]).strip()
            sub_chunks.append(Chunk(
                heading_path=chunk.heading_path,
                heading_level=chunk.heading_level,
                content=content,
                source_line_start=cur_start + 1,
                source_line_end=cur_end + 1,
                chunk_strategy="paragraph_overflow",
            ))
            cur_start = seg_s
            cur_end = seg_e
            cur_len = len(seg_text)
        else:
            cur_end = seg_e
            cur_len += len(seg_text) + 2

    if cur_start is not None:
        content = "".join(lines[cur_start : cur_end + 1]).strip()
        sub_chunks.append(Chunk(
            heading_path=chunk.heading_path,
            heading_level=chunk.heading_level,
            content=content,
            source_line_start=cur_start + 1,
            source_line_end=cur_end + 1,
            chunk_strategy="paragraph_overflow",
        ))

    return sub_chunks if sub_chunks else [chunk]


# ------------------------------------------------------------------
# 导入主流程
# ------------------------------------------------------------------

def import_lessons(database_path: Path | None = None) -> tuple[int, int]:
    lessons = discover_lessons()
    if not lessons:
        raise RuntimeError("未发现可导入的本地学习资料")

    doc_count = 0
    chunk_count = 0
    connection = sqlite3.connect(database_path or DATABASE_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            for lesson in lessons:
                title, content = _title_and_content(lesson.path)
                doc_id = f"LOCAL-{lesson.subtopic_id}"
                source_url = _reference_url(content)
                source_file_path = str(lesson.path.relative_to(PROJECT_ROOT)).replace("\\", "/")

                # 删除旧块
                connection.execute("DELETE FROM knowledge_chunks WHERE doc_id = ?", (doc_id,))

                # upsert knowledge_docs（新增 source_file_path）
                connection.execute("""
                    INSERT INTO knowledge_docs (
                        doc_id, block_id, subtopic_id, title, content, doc_type, source, difficulty, source_file_path
                    ) VALUES (
                        ?, ?, ?, ?, ?, 'tutorial', ?, ?, ?
                    )
                    ON CONFLICT(doc_id) DO UPDATE SET
                        block_id = excluded.block_id,
                        subtopic_id = excluded.subtopic_id,
                        title = excluded.title,
                        content = excluded.content,
                        source = excluded.source,
                        difficulty = excluded.difficulty,
                        source_file_path = excluded.source_file_path
                """, (doc_id, lesson.block_id, lesson.subtopic_id, title, content,
                         source_url, lesson.difficulty, source_file_path))

                # 标题级分块
                chunks = _chunk_by_headings(content)
                for index, chunk in enumerate(chunks, start=1):
                    chunk_id = f"{doc_id}-C{index:03d}"
                    connection.execute("""
                        INSERT INTO knowledge_chunks (
                            chunk_id, doc_id, block_id, chunk_index, content, chunk_type,
                            char_start, char_end,
                            heading_path, heading_level, source_line_start, source_line_end, chunk_strategy
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?,
                            NULL, NULL,
                            ?, ?, ?, ?, ?
                        )
                    """, (
                        chunk_id, doc_id, lesson.block_id, index, chunk.content, chunk.chunk_type,
                        chunk.heading_path, chunk.heading_level,
                        chunk.source_line_start, chunk.source_line_end, chunk.chunk_strategy,
                    ))
                    chunk_count += 1
                doc_count += 1
    finally:
        connection.close()
    return doc_count, chunk_count


if __name__ == "__main__":
    documents, chunks = import_lessons()
    print(f"[DONE] 已导入 {documents} 个本地课程文档，生成 {chunks} 个知识分块。")
