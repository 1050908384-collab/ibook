import logging
import re
from typing import Any

from tools.agent3_tools.vector_search_tool import VectorSearchTool

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


_MISSING_TUTORIAL_TEXT = "本知识点暂未匹配到本地学习资料。"


def ensure_stage_subtopics(stage: dict[str, Any]) -> dict[str, Any]:
    """Ensure stage always has subtopic_ids + subtopic_names arrays (repair legacy sessions).

    Old stages generated before the fix only had single subtopic_id/subtopic_name, which
    breaks frontend resource filtering and article URL lookups. This function reconstructs
    the arrays from the knowledge_docs table so article backfill & pagination keep working.
    """
    if not isinstance(stage, dict):
        return stage or {}

    block_id = str(stage.get("block_id") or "")
    existing_ids = list(stage.get("subtopic_ids") or [])
    existing_names = list(stage.get("subtopic_names") or [])

    # 已经有完整数组 → 直接返回
    if existing_ids and existing_names and len(existing_ids) == len(existing_names):
        return stage

    # 缺 name 或 id，从数据库 knowledge_docs + knowledge_subtopics 表反查
    if not block_id:
        return stage

    try:
        import sqlite3
        from tools.agent3_tools.vector_search_tool import _get_db_path

        db_path = _get_db_path()
        if not db_path.exists():
            return stage
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            # 如果已存在单个 subtopic_id，按它过滤；否则取该 block 的所有 subtopic
            single_id = str(stage.get("subtopic_id") or "").strip()
            if existing_ids:
                placeholders = ", ".join("?" for _ in existing_ids)
                rows = conn.execute(f"""
                    SELECT s.subtopic_id, s.subtopic_name
                    FROM knowledge_subtopics s
                    WHERE s.block_id = ? AND s.subtopic_id IN ({placeholders})
                    ORDER BY s.sequence_index
                """, [block_id, *existing_ids]).fetchall()
            elif single_id:
                # 旧会话只有单个 subtopic_id，需要补全该 block 下的全部？
                # 策略：按 sequence_index 排序，最多取5个，保证旧会话也有文章列表
                rows = conn.execute("""
                    SELECT s.subtopic_id, s.subtopic_name
                    FROM knowledge_subtopics s
                    WHERE s.block_id = ?
                    ORDER BY s.sequence_index
                    LIMIT 5
                """, [block_id]).fetchall()
            else:
                # 完全缺失：取该 block 下的所有 subtopic（最多5个）
                rows = conn.execute("""
                    SELECT s.subtopic_id, s.subtopic_name
                    FROM knowledge_subtopics s
                    WHERE s.block_id = ?
                    ORDER BY s.sequence_index
                    LIMIT 5
                """, [block_id]).fetchall()
        finally:
            conn.close()

        if rows:
            if not existing_ids:
                existing_ids = [str(r[0]) for r in rows]
            if not existing_names:
                existing_names = [str(r[1] or r[0]) for r in rows]
            stage = {
                **stage,
                "subtopic_ids": existing_ids,
                "subtopic_names": existing_names,
            }
            # 兼容单值字段
            if not stage.get("subtopic_id") and existing_ids:
                stage["subtopic_id"] = existing_ids[0]
            if not stage.get("subtopic_name") and existing_names:
                stage["subtopic_name"] = existing_names[0]
            logger.info(
                "ensure_stage_subtopics repaired legacy stage: block=%s ids=%s names=%s",
                block_id, existing_ids, existing_names,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ensure_stage_subtopics failed: %s", exc)

    return stage

# 匹配 .md 文件中的文章参考链接行（① 底稿：《xxx》/网址：http... 等）
_URL_REF_LINE = re.compile(
    r"^\s*[\u3000\t >]*"                                   # 前导空白/全角空格/引用符
    r"(?:[（(]?[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+[\s\+]*)*[）)]?"  # 序号标记
    r"[\s:：]*"
    r"(?:底稿|参考文章|参考|权威|来源|原文|官网|链接)"      # 标记词
    r"[\s:：]*"
    r"(?:《[^》\n]*》)?"                                    # 《标题》
    r".*"
    r"(?:网\s*址|URL|链接|来源|原\s*文|官\s*网)"
    r"[\s:：]*https?://\S+",
    re.MULTILINE | re.UNICODE,
)

# 纯链接行（整行只有 URL）
_BARE_URL_LINE = re.compile(
    r"^\s*[\u3000\t >]*(?:网\s*址|URL|链接|来源|原\s*文|官\s*网)\s*[:：]\s*https?://\S+\s*$",
    re.MULTILINE | re.UNICODE,
)

# 参考标题行（① 底稿：《xxx》/ ② 参考文章：《xxx》 等无 URL 的引用行）
_REF_TITLE_LINE = re.compile(
    r"^\s*[\u3000\t >]*"
    r"(?:[（(]?[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+[\s\+]*)*[）)]?"
    r"[\s:：]*"
    r"(?:底稿|参考文章|参考|权威|来源|原文|官网|链接)"
    r"[\s:：]*"
    r"(?:《[^》\n]*》)?"
    r"[\s]*$",
    re.MULTILINE | re.UNICODE,
)


def _strip_preamble_before_suitable_audience(content: str) -> str:
    """Strip everything up to (and excluding) the "适合人群" line.

    Intended to clean the lesson preamble shown to end-users in the frontend:
    remove top-level H1, reference header blocks ("> **参考文章链接**" quote /
    "## 参考资料" H2 / newly-added vLLM version banner), all circled reference
    entries (① 底稿 / 权威 / 参考文章 + 网址), version/hardware hint quote
    blocks, and leading "---" separators —- everything that sits **before** the
    "适合人群" metadata line. The "适合人群" line itself and every line after
    it (预计学习时长 / 环境准备 / 版本说明 / 正文章节) are preserved as-is.

    SAFETY: if the document contains no "适合人群" marker, return the original
    content unchanged to avoid accidentally wiping the whole lesson.
    """
    if not content:
        return content
    lines = content.splitlines(keepends=True)
    i = 0
    n = len(lines)
    while i < n:
        if "适合人群" in lines[i]:
            break
        i += 1
    if i >= n:
        return content
    return "".join(lines[i:]).lstrip()


def _strip_reference_urls(content: str) -> str:
    """Remove article reference URL lines from lesson content.

    These lines (① 底稿：《xxx》/ 网址：https://...) are already exposed as
    clickable links in the "文章网址" section, so they should not clutter the
    rendered markdown tutorial.
    """
    if not content:
        return content
    cleaned = _URL_REF_LINE.sub("", content)
    cleaned = _BARE_URL_LINE.sub("", cleaned)
    cleaned = _REF_TITLE_LINE.sub("", cleaned)
    # 压缩连续空行（删除 URL 行后会留下空行）
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# 匹配标题末尾的章节序号标记，如 "1.2 xxx 到底选哪个？（①）" 中的 "（①）"，
# 也兼容 "Linux 基本命令入门（新增②）" 这种括号内带前缀 + 圈号的情况。
_TRAILING_SECTION_MARK = re.compile(
    r"[\s\u3000]*[（(][^）)]*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳][^）)]*[）)]\s*$"
)

# 匹配标题开头的"面包屑路径 + >"，例如：
#   "环境准备 > 一、先搞清楚 > 1.2 xxx到底选哪个？" → 只保留 "1.2 xxx到底选哪个？"
_HEADING_CRUMB = re.compile(r"^(\s*#{1,6}\s+)(?:[^#\n>]*[>》]\s*)*(.+)$")


def _clean_headings(content: str) -> str:
    """Remove breadcrumb prefixes and section-number circled marks from markdown headings.

    The imported .md lessons contain nested "breadcrumb" titles like:
      ## 环境准备 > 一、先搞清楚（看完再动手...）（①） > 1.2 xxx 到底选哪个？（①）
    We want them rendered as clean headings:
      ## 1.2 "虚拟机"、"WSL2"、"云服务器"到底选哪个？
    """
    lines = content.splitlines()
    cleaned_lines: list[str] = []
    in_fence = False
    for line in lines:
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            cleaned_lines.append(line)
            continue
        if not in_fence and re.match(r"^\s*#{1,6}\s+", line):
            # 去末尾（①）
            line = _TRAILING_SECTION_MARK.sub("", line)
            # 去面包屑前缀
            m = _HEADING_CRUMB.match(line)
            if m:
                line = f"{m.group(1)}{m.group(2).strip()}"
            # 再去一次末尾标记（嵌套标题可能有多个）
            line = _TRAILING_SECTION_MARK.sub("", line).rstrip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _nest_document_markdown(content: str) -> str:
    """Place a full lesson below a topic H2 without changing fenced code."""
    # 1) 语义级：删除"适合人群"之前的整段前言（参考文章链接块/版本头/H1/分隔线等），保留适合人群本身
    content = _strip_preamble_before_suitable_audience(content)
    # 2) 行级兜底：去掉散落在正文中的"① 底稿...网址..."单行（它们已在"拓展网址"区单独展示）
    content = _strip_reference_urls(content)
    # 3) 清理标题面包屑和（①）标记
    content = _clean_headings(content)
    lines = content.strip().splitlines()
    if lines and re.match(r"^#\s+", lines[0]):
        lines = lines[1:]

    nested: list[str] = []
    in_fence = False
    for line in lines:
        if re.match(r"^\s*(```|~~~)", line):
            in_fence = not in_fence
            nested.append(line)
            continue
        if not in_fence:
            heading = re.match(r"^(#{1,6})(\s+.*)$", line)
            if heading:
                level = min(len(heading.group(1)) + 1, 6)
                line = f"{'#' * level}{heading.group(2)}"
        nested.append(line)
    result = "\n".join(nested).strip()
    # 压缩连续空行
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def backfill_stage_tutorial(package: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    """Replace legacy placeholder tutorials with imported full lessons.

    ⚠️ 仅对「旧会话」生效：
      - tutorial 为空 / 占位符文本；
      - 或 tutorial 是 source_material 的纯拼接（不含任何 subtopic 级 ## heading）。
    新会话 Agent3/HybridContentAgent 流水线已经把个性化内容写进 tutorial + content_blocks，
    这里绝对不能覆盖，否则 practice / theory 两版本的表达层差异化会被统一回
    VectorSearchTool 知识库原文正文 → 两学习包一模一样。
    """
    # 1. 确保 subtopic_ids/subtopic_names 被补全（旧会话兼容）
    stage = ensure_stage_subtopics(stage)

    tutorial = str(package.get("tutorial") or "")
    # ═══════════════════════════════════════════════════════════════════════
    # Guard 1：新会话流水线一旦写出了 content_blocks，tutorial 就是
    #   "块标题 + content_blocks.personalized_content" 拼接或 Agent3.run
    #   整篇 LLM 生成的结果，必然是个性化版本。如果此时仍强行 backfill，
    #   会用知识库原文覆盖掉 practice / theory 的表达差异，对用户呈现
    #   为"修改半天一模一样"的假失败。
    # ═══════════════════════════════════════════════════════════════════════
    blocks = package.get("content_blocks") or []
    if isinstance(blocks, list) and blocks:
        return package
    if tutorial.strip() and _MISSING_TUTORIAL_TEXT not in tutorial:
        # 额外检查：tutorial 是否包含至少一个 subtopic 对应的 ## heading。
        # 旧会话可能存储了 source_material（所有知识点内容的纯拼接，无 ## 分隔），
        # 前端 getTopicTutorial 无法按知识点切片，导致所有页显示同一份合并内容。
        # 这种 tutorial 不含占位符但结构错误，必须重新生成。
        subtopic_names_for_check = list(stage.get("subtopic_names") or [])
        if subtopic_names_for_check:
            has_subtopic_heading = False
            for raw_name in subtopic_names_for_check:
                clean_name = _TRAILING_SECTION_MARK.sub("", str(raw_name)).strip()
                if clean_name.startswith("（") and clean_name.endswith("）"):
                    clean_name = clean_name[1:-1].strip()
                if not clean_name:
                    continue
                # ═══════════════════════════════════════════════════════════════
                # 放宽匹配：从『整行完全等于 clean_name』改为『## 行包含子串』。
                # 之前严格正则 r"^##\s+{clean_name}\s*$" 会把 Agent3 写出的
                # 带风格前缀 / 多级 heading_path 的标题（例如
                #   ## 立即动手 · nvidia-smi 显存监控 > ... > 2.1 最简命令（①）
                #   ## 概念与设计动机：nvidia-smi 显存监控 > ... > 2.1 最简命令（①）
                # ）误判为"缺少 subtopic 级 ## heading"→ 触发 backfill 覆盖。
                # 放宽后只要同一行、同一 ## 层级中出现子知识点名即可判定为已有
                # 子标题；后面的完整路径 / 风格前缀并不破坏切片。
                # ═══════════════════════════════════════════════════════════════
                if re.search(
                    r"^##\s+.*\b" + re.escape(clean_name) + r"\b.*$",
                    tutorial,
                    re.MULTILINE | re.IGNORECASE,
                ):
                    has_subtopic_heading = True
                    break
                # 次选：clean_name 中可能含有空格变体 / 中英文标点差异，
                # 做一次"去掉所有空白和中英文符号后的包含匹配"。
                import unicodedata as _uni
                def _compact(text: str) -> str:
                    chars: list[str] = []
                    for ch in _uni.normalize("NFKC", text or ""):
                        category = _uni.category(ch)
                        if category.startswith("Z") or category.startswith("P"):
                            continue
                        chars.append(ch.lower())
                    return "".join(chars)
                compact = _compact(clean_name)
                if compact:
                    for line in tutorial.splitlines():
                        if not line.lstrip().startswith("##"):
                            continue
                        if compact in _compact(line):
                            has_subtopic_heading = True
                            break
                    if has_subtopic_heading:
                        break
            if has_subtopic_heading:
                return package
            # 没有 subtopic 级 ## heading → 继续走 backfill 重新生成
        else:
            return package

    documents = VectorSearchTool().get_local_lessons(
        stage.get("block_id", ""),
        list(stage.get("subtopic_ids") or ([stage["subtopic_id"]] if stage.get("subtopic_id") else [])),
    )
    if not documents:
        return package

    documents_by_subtopic = {document.get("subtopic_id"): document for document in documents}
    subtopic_ids = list(stage.get("subtopic_ids") or [])
    subtopic_names = list(stage.get("subtopic_names") or [])
    if not subtopic_ids and stage.get("subtopic_id"):
        subtopic_ids = [stage["subtopic_id"]]
    if not subtopic_names and stage.get("subtopic_name"):
        subtopic_names = [stage.get("subtopic_name")]

    # 不再生成 # 块标题（前端已有标题），直接按每个 subtopic 输出 ## 子标题 + 正文
    sections: list[str] = []
    for index, subtopic_id in enumerate(subtopic_ids):
        document = documents_by_subtopic.get(subtopic_id)
        if not document or not str(document.get("content") or "").strip():
            continue
        topic_name = subtopic_names[index] if index < len(subtopic_names) else document.get("title")
        # 清理 subtopic 名称尾部的括号序号和全角括号
        clean_name = str(topic_name or document.get("title") or "知识点")
        clean_name = _TRAILING_SECTION_MARK.sub("", clean_name).strip()
        if clean_name.startswith("（") and clean_name.endswith("）"):
            clean_name = clean_name[1:-1].strip()
        sections.extend([
            f"## {clean_name}",
            _nest_document_markdown(str(document["content"])),
        ])

    if not sections:
        return package
    return {**package, "tutorial": "\n\n".join(sections)}


_SPACE = r"[ \t\u3000>]"

# 匹配每个参考链接行常见格式，例如：
#   ① 底稿：《xxx》
#   　　网址：https://...
#   ② 参考文章：《xxx》
#   　　网址：https://...
#   > ① 底稿：《xxx》
#   > 　　网址：https://...
#   ① 权威：《xxx》
#   　　网址：https://...
#   ①② 参考文章：《xxx》
#   　　网址：https://...
#   （①②③）参考文章：《xxx》  网址：https://...
# 或者
#   > **参考文章链接**
#   > ① 底稿：《Python语法》
#   > 　　网址：https://...
_REF_ENTRY = re.compile(
    r"(?:(?<=\n)|(?<=^))"                                    # 行首
    + _SPACE + r"*"
    + r"[（(]?[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+(?:[\s\+]+[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+)*[）)]?"  # 序号标记（支持括号包裹）
    + r"[\s:：]*"                                             # 可选空白和冒号
    + r"(?P<label>[^《\n：:]{0,40}?)"                         # 标记词（底稿/参考文章/权威…  限长避免吃到标题）
    + r"[:：]?\s*"
    + r"(?:《(?P<title>.*?)》)?\s*"                           # 《xxx》可选（书名号内是文章正式标题）
    + r"[ \t\u3000]*\n"                                       # 换行
    + _SPACE + r"*"
    + r"(?:网\s*址|URL|链接|来源|原\s*文|官\s*网|底\s*稿|参\s*考)"
    + r"[ \t\u3000]*[:：][ \t\u3000]*"                        # "网址："
    + r"(?P<url>https?://[^\s《》<>\"'|，,；;）)【】]+)",      # URL
    re.MULTILINE | re.UNICODE,
)

# 宽松匹配：整行形如 "任意文本 网址：http..."
_LOOSE_URL = re.compile(
    r"(?:^|\n)" + _SPACE + r"*"
    + r"(?P<title>[^\n：:]{0,100}?)"
    + r"(?:网\s*址|URL|链接|来源|原\s*文|官\s*网)"
    + r"[ \t\u3000]*[:：][ \t\u3000]*"
    + r"(?P<url>https?://[^\s《》<>\"'|，,；;）)【】]+)",
    re.MULTILINE | re.UNICODE,
)

# Markdown 超链接：[显示文字](https://...)
#   例：① 权威：《HuggingFace课程》  网址：[HF LLM Chapter 0-1](https://huggingface.co/...)
# 必须用 #3 兜底，因为 _REF_ENTRY / _LOOSE_URL 的 URL 捕获要求
# "http..." 直接出现在 "网址：" 之后，而 markdown 写法会被方括号挡住。
_MD_LINK = re.compile(
    r"\[(?P<label>[^\]\n]{1,180})\]\((?P<url>https?://[^\s)《》<>\"'|，,；;]+)\)",
    re.MULTILINE | re.UNICODE,
)


def extract_all_reference_urls(content: str) -> list[tuple[str, str]]:
    """Return all (label+title, url) reference links from a local lesson md.

    Each entry's first field is human-readable text combining the label
    (底稿/参考文章/权威…) and 《xxx》 title; the second field is the
    normalized URL. Duplicate URLs are de-duplicated (first occurrence wins).
    """
    if not content:
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(label: str, title: str, url: str) -> None:
        clean_url = url.strip().rstrip("。，,；;")
        if not clean_url or clean_url in seen:
            return
        seen.add(clean_url)
        parts: list[str] = []
        if label:
            cleaned_label = label.strip().rstrip("：:").strip()
            if cleaned_label:
                parts.append(cleaned_label)
        cleaned_title = (title or "").strip()
        if cleaned_title:
            parts.append(cleaned_title)
        combined = "：".join(p for p in parts if p) if parts else clean_url
        pairs.append((combined, clean_url))

    # 1. 标准带序号格式优先（有①②标记 + 标题书名号 + 网址：）
    for match in _REF_ENTRY.finditer(content):
        _add(match.group("label") or "", match.group("title") or "", match.group("url"))

    # 2. 宽松形式补齐（捕获没有①②标记但写着"网址：http..."的行）
    for match in _LOOSE_URL.finditer(content):
        title_match = re.search(r"《(.*?)》", match.group("title") or "")
        title = title_match.group(1) if title_match else ""
        label = ""
        if not title:
            pre = (match.group("title") or "").strip()
            if pre and len(pre) <= 20:
                label = pre
        _add(label, title, match.group("url"))

    # 3. Markdown 超链接兜底（[标题/说明文字](https://...)）
    #    这是 HuggingFace/官方文档风格 md 常用的写法。
    #    从 #1 / #2 没抓到过的 URL 里，解析 label 并加上。
    for match in _MD_LINK.finditer(content):
        raw_label = (match.group("label") or "").strip()
        if not raw_label:
            continue
        # 丢弃明显是"锚点"或代码风格伪链接（如 [SOURCE]:... 这种引用式定义出现在别处时一般无括号 URL，
        # 但保险起见过滤纯数字/符号占位）
        if re.fullmatch(r"[\W_]+", raw_label):
            continue
        # 提取书名号片段作为标题，余下作为 label
        title = ""
        tmatch = re.search(r"《(.*?)》", raw_label)
        if tmatch:
            title = tmatch.group(1).strip()
        # 去除行首的 ①②③... 序号前缀与常见的中文标签词分隔符号
        prefix_clean = re.sub(
            r"^[\s\u3000]*[（(]?[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳\d]+[）)]?[\s\.、．:：\-]*",
            "",
            raw_label,
        )
        # 如果已经书名号提取了 title，label 就是剩下的文字
        label = prefix_clean.replace(f"《{title}》", "").strip(" ：:，,、\t\u3000") if title else prefix_clean
        _add(label, title, match.group("url"))

    return pairs


def build_stage_document_resources(stage: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose article URL resources for a stage (one resource per reference link)."""
    stage = ensure_stage_subtopics(stage)
    block_id = stage.get("block_id")
    subtopic_ids = list(stage.get("subtopic_ids") or [])
    if not subtopic_ids and stage.get("subtopic_id"):
        subtopic_ids = [stage["subtopic_id"]]
    logger.info(
        "build_stage_document_resources: block_id=%s subtopic_ids=%s subtopic_id=%s",
        block_id, subtopic_ids, stage.get("subtopic_id"),
    )
    if not block_id or not subtopic_ids:
        logger.warning("build_stage_document_resources: missing block_id or subtopic_ids, returning []")
        return []

    documents = VectorSearchTool().get_local_lessons(block_id, subtopic_ids)
    logger.info(
        "build_stage_document_resources: get_local_lessons returned %d documents", len(documents),
    )
    documents_by_subtopic: dict[str, dict[str, Any]] = {
        doc.get("subtopic_id"): doc for doc in documents if isinstance(doc, dict) and doc.get("subtopic_id")
    }

    # 预构建：每个 subtopic_id 对应的知识点名称列表，用于检测 ref_title 面包屑是否跨知识点
    # 例：ref_title 以 "环境变量配置 > " 开头 → 尽管 URL 在当前文档里，实际归属是 "环境变量配置"
    # 这才是真正解决"第三部分的文章出现在第二部分"的核心
    stage_subtopic_names: list[tuple[str, str]] = []  # [(subtopic_id, subtopic_name), ...]
    if isinstance(stage.get("subtopic_names"), list) and isinstance(stage.get("subtopic_ids"), list):
        stage_subtopic_names = list(zip(
            [str(x) for x in stage["subtopic_ids"]],
            [str(x).strip() for x in stage["subtopic_names"]],
        ))
    # 再从 knowledge_subtopics 查真实 display_name 补全
    if block_id:
        try:
            import sqlite3 as _sqlite
            from tools.agent3_tools.vector_search_tool import _get_db_path
            _db_path = _get_db_path()
            if _db_path.exists():
                _conn = _sqlite.connect(f"file:{_db_path}?mode=ro", uri=True)
                try:
                    _rows = _conn.execute("""
                        SELECT subtopic_id, subtopic_name FROM knowledge_subtopics
                        WHERE block_id = ? ORDER BY sequence_index
                    """, [block_id]).fetchall()
                    for _rid, _rname in _rows:
                        _sname = str(_rname or "").strip()
                        _sid = str(_rid or "")
                        if _sid and _sname and not any(s[0] == _sid for s in stage_subtopic_names):
                            stage_subtopic_names.append((_sid, _sname))
                finally:
                    _conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _resolve_ref_subtopic_id(ref_title: str, doc_subtopic_id: str) -> str:
        """根据 ref_title 开头的面包屑判断实际归属知识点。

        例如 "环境变量配置 > 一、环境变量是什么？ > 1.1 通俗理解"
        的第一级面包屑是 "环境变量配置"，即使该 URL 被写在 subtopic 2（常用Shell命令）
        的 md 文档里，也必须强制归属到 "环境变量配置" 对应的 subtopic_id，
        否则前端就会看到「第三部分的文章出现在第二部分」。
        """
        title_head = (ref_title or "").split(">", 1)[0].strip()
        # 去掉序号前缀再比较（如 "3 环境变量配置" → "环境变量配置"）
        title_head_clean = re.sub(r"^[\s\d①②③④⑤⑥⑦⑧⑨⑩一二三四五六七八九十]+[.、．:：)\]\s]*", "", title_head).strip()
        if not title_head_clean:
            return doc_subtopic_id
        # 先精确匹配
        for sid, sname in stage_subtopic_names:
            sname_clean = re.sub(r"^[\s\d①②③④⑤⑥⑦⑧⑨⑩一二三四五六七八九十]+[.、．:：)\]\s]*", "", sname).strip()
            if sname_clean == title_head_clean:
                return sid
        # 再去掉括号后缀比较
        title_head_core = re.sub(r"[\s（(【\[].*$", "", title_head_clean).strip()
        if title_head_core and title_head_core != title_head_clean:
            for sid, sname in stage_subtopic_names:
                sname_clean = re.sub(r"^[\s\d①②③④⑤⑥⑦⑧⑨⑩一二三四五六七八九十]+[.、．:：)\]\s]*", "", sname).strip()
                sname_core = re.sub(r"[\s（(【\[].*$", "", sname_clean).strip()
                if sname_core == title_head_core:
                    return sid
        return doc_subtopic_id

    result: list[dict[str, Any]] = []
    order_counter = 0

    for subtopic_id in subtopic_ids:
        document = documents_by_subtopic.get(subtopic_id)
        if not isinstance(document, dict):
            logger.info("build_stage_document_resources: no document for subtopic_id=%s", subtopic_id)
            continue
        doc_id = str(document.get("doc_id") or "")
        if doc_id and not doc_id.startswith("LOCAL-"):
            continue
        doc_subtopic_id = str(document.get("subtopic_id") or subtopic_id or "")
        doc_title = str(document.get("title") or "")
        content = str(document.get("content") or "")
        refs = extract_all_reference_urls(content)
        if not refs:
            # 退回到数据库的 source 字段（兼容未完整更新 import 的库）
            fallback = str(document.get("source") or "").strip()
            if fallback.startswith(("http://", "https://")):
                refs = [(doc_title or "学习文章", fallback)]
            else:
                # 本地 Markdown 可能没有外部网址，但它仍是该知识点的
                # 垂直领域来源。保留一个 documentation 资源，前端可显示
                # 来源名称/本地文档标识，而不是把“参考文章”渲染成空白。
                local_locator = f"knowledge_doc:{doc_id or subtopic_id}"
                result.append({
                    "resource_id": f"{doc_id or 'LOCAL'}-DOC",
                    "resource_type": "documentation",
                    "title": doc_title or f"{subtopic_id} 学习资料",
                    "url": "",
                    "canonical_url": "",
                    "platform": "local",
                    "description": "本地领域知识库文档",
                    "content_summary": doc_title or "本地领域知识库文档",
                    "content_ref": local_locator,
                    "locator": local_locator,
                    "recommendation_level": "primary",
                    "role": "primary",
                    "order": order_counter + 1,
                    "subtopic_id": doc_subtopic_id,
                })
                order_counter += 1
        logger.info(
            "build_stage_document_resources: subtopic=%s doc_id=%s refs_count=%d",
            subtopic_id, doc_id, len(refs),
        )
        # ★ 注意：绝对不能跨 subtopic 先到先得去重！
        # 同一个 URL 出现在多个 subtopic 的参考列表里是正常的（如拓展阅读、后续章节预告），
        # 但必须严格按它所在文档的 subtopic_id 归属（或按 ref_title 面包屑修正）。
        # 重复问题由 extract_all_reference_urls 的单文档 seen 控制，以及前端 uniqueResources
        # 按 (subtopic_id, URL) 二元组去重。
        for ref_index, (ref_title, ref_url) in enumerate(refs, start=1):
            order_counter += 1
            resolved_subtopic_id = _resolve_ref_subtopic_id(ref_title, doc_subtopic_id)
            if resolved_subtopic_id != doc_subtopic_id:
                logger.info(
                    "build_stage_document_resources: ref re-assigned by breadcrumb: "
                    "doc_subtopic=%s → resolved=%s (title=%s url=%s)",
                    doc_subtopic_id, resolved_subtopic_id, ref_title[:40], ref_url[:60],
                )
            result.append({
                "resource_id": f"{doc_id or 'LOCAL'}-ART-{order_counter:02d}",
                "resource_type": "article",
                "title": ref_title or doc_title or f"学习文章 {order_counter}",
                "url": ref_url,
                "canonical_url": ref_url,
                "platform": "",
                "description": doc_title or "",
                "content_summary": doc_title or "",
                "recommendation_level": "primary",
                "role": "primary",
                "order": order_counter,
                # 1) 先看文档本身的 subtopic_id；2) 若 ref_title 面包屑显示属别的知识点则修正
                "subtopic_id": resolved_subtopic_id,
            })

    # 后端再次保险：同一个 (subtopic_id, url) 只保留一条（前端也会做）
    deduped: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for item in result:
        pair = (str(item.get("subtopic_id") or ""), str(item.get("url") or item.get("canonical_url") or ""))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append(item)
    logger.info(
        "build_stage_document_resources: returning %d article resources with subtopics: %s",
        len(deduped),
        sorted({r["subtopic_id"] for r in deduped}),
    )
    return deduped
