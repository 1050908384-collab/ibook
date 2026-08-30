"""Build auditable knowledge-base submission artifacts from repository sources.

The raw Markdown articles are the source of truth for references. Runtime
knowledge_docs.json is treated as a derived index and is never overwritten.
"""

from __future__ import annotations


import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in sys.path: sys.path.insert(0, str(_PROJ_ROOT_))
_BE_ = Path(_PROJ_ROOT_, "backend")
if str(_BE_) not in sys.path: sys.path.insert(0, str(_BE_))

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "知识库原始数据"
SEEDS = ROOT / "data" / "seeds"
OUT = ROOT / "测试数据与评测材料" / "08_知识库提交材料"

URL_RE = re.compile(r"https?://[^\s)）>】]+")
AUTHOR_RE = re.compile(r"(?:作者|Author)\s*[:：]\s*([^\n]+)", re.I)
BLOCK_RE = re.compile(r"^(\d{2})_")
SEQ_RE = re.compile(r"^(\d+)[.、]")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean_url(url: str) -> str:
    return url.rstrip("，。；;、,.)]】")


def raw_articles():
    grouped = defaultdict(list)
    for path in RAW_ROOT.rglob("*.md"):
        if path.name in {"知识清单-学习文章.md", "知识清单-学习视频链接收集.md"}:
            continue
        block_match = BLOCK_RE.match(path.relative_to(RAW_ROOT).parts[0])
        if not block_match:
            continue
        seq_match = SEQ_RE.match(path.relative_to(RAW_ROOT).parts[-2])
        if not seq_match:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        title = next((x.lstrip("# ").strip() for x in lines if x.startswith("#")), path.stem)
        urls = []
        for url in URL_RE.findall(text):
            url = clean_url(url)
            if url not in urls:
                urls.append(url)
        authors = [m.group(1).strip() for m in AUTHOR_RE.finditer(text)]
        refs = []
        for index, line in enumerate(lines):
            if "网址" in line or "参考资料" in line or "参考文章" in line or "权威" in line:
                context = " ".join(x.strip() for x in lines[max(0, index - 1): index + 2] if x.strip())
                if context not in refs:
                    refs.append(context)
        grouped[block_match.group(1)].append({
            "path": path,
            "sequence": int(seq_match.group(1)),
            "title": title,
            "urls": urls,
            "authors": authors,
            "references": refs,
        })
    for values in grouped.values():
        values.sort(key=lambda x: x["sequence"])
    return grouped


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = read_json(SEEDS / "knowledge_blocks.json")
    subtopics = read_json(SEEDS / "knowledge_subtopics.json")
    docs = read_json(SEEDS / "knowledge_docs.json")
    questions = read_json(SEEDS / "test_questions.json")
    grouped = raw_articles()
    docs_by_subtopic = defaultdict(list)
    for doc in docs:
        docs_by_subtopic[doc.get("subtopic_id")].append(doc)

    block_names = {x["block_id"]: x["block_name"] for x in blocks}
    raw_by_subtopic = {}
    for block_id in sorted(block_names):
        number = block_id.split("-")[-1]
        block_subtopics = sorted(
            [x for x in subtopics if x["block_id"] == block_id],
            key=lambda x: int(x.get("sequence_index", 0)),
        )
        block_articles = grouped.get(number, [])
        if len(block_subtopics) != len(block_articles):
            raise ValueError(
                f"raw article count mismatch for {block_id}: "
                f"{len(block_articles)} articles vs {len(block_subtopics)} subtopics"
            )
        # Some legacy seed blocks start sequence_index at 0 and others at 1.
        # Pairing sorted entries avoids silently shifting an entire block.
        for subtopic, article in zip(block_subtopics, block_articles):
            raw_by_subtopic[subtopic["subtopic_id"]] = article

    source_rows = []
    article_rows = []
    subtopic_rows = []
    question_counts = Counter(x.get("subtopic_id") for x in questions)
    question_types = defaultdict(Counter)
    for question in questions:
        question_types[question.get("subtopic_id")][question.get("question_type")] += 1

    for subtopic in sorted(subtopics, key=lambda x: x["subtopic_id"]):
        sid = subtopic["subtopic_id"]
        raw = raw_by_subtopic.get(sid, {})
        urls = raw.get("urls", [])
        authors = raw.get("authors", [])
        runtime_docs = docs_by_subtopic.get(sid, [])
        source_ids = []
        for index, url in enumerate(urls, 1):
            source_id = f"SRC-{sid}-{index:02d}"
            source_ids.append(source_id)
            source_rows.append({
                "source_id": source_id,
                "subtopic_id": sid,
                "block_id": subtopic["block_id"],
                "source_url": url,
                "author": "; ".join(authors) if authors else "原文未声明",
                "source_role": "原始知识文章参考来源",
                "verification_status": "待人工核验",
                "notes": "网址由原始Markdown自动提取；提交前核验可访问性、标题和授权信息。",
            })
        types = question_types[sid]
        subtopic_rows.append({
            "subtopic_id": sid,
            "block_id": subtopic["block_id"],
            "block_name": block_names[subtopic["block_id"]],
            "subtopic_name": subtopic["subtopic_name"],
            "difficulty": subtopic.get("difficulty", ""),
            "estimated_hours": subtopic.get("estimated_hours", ""),
            "prerequisites": "|".join(subtopic.get("prerequisites", [])),
            "raw_markdown_path": str(raw.get("path", "").relative_to(ROOT)) if raw.get("path") else "",
            "raw_reference_count": len(urls),
            "runtime_doc_count": len(runtime_docs),
            "question_count": question_counts[sid],
            "question_types": ";".join(f"{k}:{v}" for k, v in sorted(types.items())),
            "source_ids": "|".join(source_ids),
            "content_status": "有原始文章" if raw else "缺少原始文章映射",
        })
        for doc in runtime_docs:
            article_rows.append({
                "doc_id": doc.get("doc_id", ""),
                "subtopic_id": sid,
                "block_id": doc.get("block_id", ""),
                "title": doc.get("title", ""),
                "doc_type": doc.get("doc_type", ""),
                "source_label_in_runtime": doc.get("source", ""),
                "raw_markdown_path": str(raw.get("path", "").relative_to(ROOT)) if raw.get("path") else "",
                "source_url_count_from_raw": len(urls),
                "source_ids": "|".join(source_ids),
                "content_review_status": "待人工复核",
            })

    write_csv(OUT / "knowledge_subtopic_catalog.csv", subtopic_rows, list(subtopic_rows[0]))
    write_csv(OUT / "knowledge_document_catalog.csv", article_rows, list(article_rows[0]))
    write_csv(OUT / "knowledge_source_registry.csv", source_rows, list(source_rows[0]))

    question_rows = []
    for sid in sorted(question_counts):
        types = question_types[sid]
        question_rows.append({
            "subtopic_id": sid,
            "question_count": question_counts[sid],
            "at_least_two": question_counts[sid] >= 2,
            "question_types": ";".join(f"{k}:{v}" for k, v in sorted(types.items())),
            "has_choice": "single_choice" in types,
            "has_fill_blank": "fill_blank" in types,
            "has_true_false": "true_false" in types,
        })
    write_csv(OUT / "knowledge_question_coverage.csv", question_rows, list(question_rows[0]))

    summary = {
        "raw_markdown_articles": len(raw_by_subtopic),
        "knowledge_subtopics": len(subtopics),
        "runtime_knowledge_docs": len(docs),
        "source_records_extracted": len(source_rows),
        "subtopics_with_at_least_two_questions": sum(v >= 2 for v in question_counts.values()),
        "question_type_totals": dict(Counter(x.get("question_type") for x in questions)),
        "runtime_docs_without_raw_article_mapping": sum(1 for x in article_rows if not x["raw_markdown_path"]),
        "subtopics_without_raw_article_mapping": sum(1 for x in subtopic_rows if not x["raw_markdown_path"]),
    }
    (OUT / "knowledge_submission_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
