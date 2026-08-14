import re
from typing import Any

from tools.agent3_tools.vector_search_tool import VectorSearchTool


_MISSING_TUTORIAL_TEXT = "本知识点暂未匹配到本地学习资料。"


def _nest_document_markdown(content: str) -> str:
    """Place a full lesson below a topic H2 without changing fenced code."""
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
    return "\n".join(nested).strip()


def backfill_stage_tutorial(package: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    """Replace legacy placeholder tutorials with imported full lessons."""
    tutorial = str(package.get("tutorial") or "")
    if tutorial.strip() and _MISSING_TUTORIAL_TEXT not in tutorial:
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
        subtopic_names = [stage["subtopic_name"]]

    sections: list[str] = [f"# {stage.get('block_name') or package.get('block_name') or '学习资料'}"]
    for index, subtopic_id in enumerate(subtopic_ids):
        document = documents_by_subtopic.get(subtopic_id)
        if not document or not str(document.get("content") or "").strip():
            continue
        topic_name = subtopic_names[index] if index < len(subtopic_names) else document.get("title")
        sections.extend([
            f"## {topic_name or document.get('title') or '知识点'}",
            _nest_document_markdown(str(document["content"])),
        ])

    if len(sections) == 1:
        return package
    return {**package, "tutorial": "\n\n".join(sections)}


def build_stage_document_resources(stage: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose the source records behind a stage, including legacy packages."""
    block_id = stage.get("block_id")
    subtopic_ids = list(stage.get("subtopic_ids") or [])
    if not subtopic_ids and stage.get("subtopic_id"):
        subtopic_ids = [stage["subtopic_id"]]
    if not block_id or not subtopic_ids:
        return []

    documents = VectorSearchTool().get_local_lessons(block_id, subtopic_ids)
    return [
        {
            "resource_id": document.get("doc_id") or f"stage_doc_{index}",
            "resource_type": "article" if str(document.get("source") or "").startswith(("http://", "https://")) else "documentation",
            "title": document.get("title") or f"文章来源 {index}",
            "url": document.get("source") or "",
            "platform": "",
            "description": document.get("source") or "知识库学习资料",
            "recommendation_level": "primary",
            "role": "source",
            "order": index,
        }
        for index, document in enumerate(documents, start=1)
        if isinstance(document, dict)
    ]
