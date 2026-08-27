# -*- coding: utf-8 -*-
"""Build the competition-facing knowledge-base slice and source-review aids."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "知识库原始数据"
SEEDS = ROOT / "data" / "seeds"
OUT = ROOT / "测试数据与评测材料" / "08_知识库提交材料"
URL_RE = re.compile(r"https?://[^\s)）>】]+")
BLOCK_RE = re.compile(r"^(\d{2})_")
SEQ_RE = re.compile(r"^(\d+)[.、]")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean_url(value: str) -> str:
    return value.rstrip("，。；;、,.)]】`'")


def get_articles():
    grouped = defaultdict(list)
    for path in RAW_ROOT.rglob("*.md"):
        if path.name in {"知识清单-学习文章.md", "知识清单-学习视频链接收集.md"}:
            continue
        relative = path.relative_to(RAW_ROOT)
        block_match = BLOCK_RE.match(relative.parts[0])
        sequence_match = SEQ_RE.match(relative.parts[-2])
        if not block_match or not sequence_match:
            continue
        grouped[block_match.group(1)].append(
            {
                "path": path,
                "sequence": int(sequence_match.group(1)),
                "title": path.stem,
                "text": path.read_text(encoding="utf-8", errors="replace"),
            }
        )
    for articles in grouped.values():
        articles.sort(key=lambda item: item["sequence"])
    return grouped


def write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = load(SEEDS / "knowledge_blocks.json")
    subtopics = load(SEEDS / "knowledge_subtopics.json")
    questions = load(SEEDS / "test_questions.json")
    videos = load(ROOT / "data" / "knowledge_base" / "video_resources.json")
    grouped = get_articles()
    block_names = {item["block_id"]: item["block_name"] for item in blocks}
    raw_by_subtopic = {}
    for block_id in sorted(block_names):
        block_number = block_id.split("-")[-1]
        selected = [x for x in subtopics if x["block_id"] == block_id]
        selected.sort(key=lambda x: int(x.get("sequence_index", 0)))
        articles = grouped.get(block_number, [])
        if len(selected) != len(articles):
            raise ValueError(f"{block_id}: {len(selected)} subtopics vs {len(articles)} articles")
        for subtopic, article in zip(selected, articles):
            raw_by_subtopic[subtopic["subtopic_id"]] = article

    video_counts = defaultdict(int)
    video_ids = defaultdict(list)
    for item in videos.get("subtopic_mappings", []):
        video_counts[item.get("subtopic_id", "")] += 1
        video_ids[item.get("subtopic_id", "")].append(item.get("resource_id", ""))
    question_ids = defaultdict(list)
    for item in questions:
        question_ids[item.get("subtopic_id", "")].append(item.get("question_id", ""))

    article_review = []
    citation_rows = []
    for subtopic in sorted(subtopics, key=lambda item: item["subtopic_id"]):
        sid = subtopic["subtopic_id"]
        article = raw_by_subtopic[sid]
        relative = str(article["path"].relative_to(ROOT))
        lines = article["text"].splitlines()
        heading = "正文"
        urls_in_article = []
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
            for raw_url in URL_RE.findall(line):
                url = clean_url(raw_url)
                if not url:
                    continue
                if url not in urls_in_article:
                    urls_in_article.append(url)
                context_start = max(0, line_no - 2)
                context_end = min(len(lines), line_no + 1)
                context = " ".join(x.strip() for x in lines[context_start:context_end] if x.strip())
                citation_rows.append(
                    {
                        "subtopic_id": sid,
                        "block_id": subtopic["block_id"],
                        "article_path": relative,
                        "section_heading": heading,
                        "source_url": url,
                        "source_context": context[:1000],
                        "channel_check": "待人工核验",
                        "title_check": "待人工核验",
                        "author_check": "待人工核验",
                        "date_check": "待人工核验",
                        "license_check": "待人工核验",
                        "review_note": "",
                    }
                )
        article_review.append(
            {
                "subtopic_id": sid,
                "block_id": subtopic["block_id"],
                "block_name": block_names[subtopic["block_id"]],
                "subtopic_name": subtopic["subtopic_name"],
                "article_path": relative,
                "article_title": article["title"],
                "source_url_count": len(urls_in_article),
                "video_resource_count": video_counts[sid],
                "video_resource_ids": "|".join(video_ids[sid]),
                "question_count": len(question_ids[sid]),
                "question_ids": "|".join(question_ids[sid]),
                "content_review": "待人工核验",
                "source_channel_review": "待人工核验",
                "citation_review": "待人工核验",
                "privacy_review": "待人工核验",
                "reviewer": "",
                "review_date": "",
                "review_note": "",
            }
        )

    mapping_fields = [
        "subtopic_id", "block_id", "block_name", "subtopic_name", "article_path",
        "video_resource_ids", "question_ids", "mapping_review"
    ]
    mapping_rows = []
    for row in article_review:
        mapping_rows.append({
            "subtopic_id": row["subtopic_id"],
            "block_id": row["block_id"],
            "block_name": row["block_name"],
            "subtopic_name": row["subtopic_name"],
            "article_path": row["article_path"],
            "video_resource_ids": row["video_resource_ids"],
            "question_ids": row["question_ids"],
            "mapping_review": "程序生成，抽查确认",
        })
    write_csv(OUT / "knowledge_subtopic_article_video_question_map.csv", mapping_rows, mapping_fields)

    write_csv(
        OUT / "knowledge_article_review.csv",
        article_review,
        list(article_review[0]),
    )
    write_csv(
        OUT / "knowledge_article_citations.csv",
        citation_rows,
        list(citation_rows[0]) if citation_rows else ["subtopic_id", "article_path"],
    )
    manifest = {
        "version": "KB-20260828-r2",
        "competition_basis": "PDF page 5, submission data package",
        "knowledge_slice": "data/知识库原始数据/ 旗下按知识点组织的 Markdown 文章",
        "article_count": len(article_review),
        "knowledge_subtopic_count": len(subtopics),
        "citation_record_count": len(citation_rows),
        "articles_with_source_urls": sum(row["source_url_count"] > 0 for row in article_review),
        "articles_without_source_urls": sum(row["source_url_count"] == 0 for row in article_review),
        "manual_review_required": True,
        "runtime_doc_count": len(load(SEEDS / "knowledge_docs.json")),
        "note": "73篇文章是知识库切片主体；运行时资料索引数量不作为知识点数量口径。",
    }
    (OUT / "knowledge_slice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    hash_targets = [
        ROOT / "data" / "sqlite_data" / "project.db",
        ROOT / "data" / "graph_data" / "runtime" / "knowledge_graph.gpickle",
        ROOT / "data" / "graph_data" / "current" / "nodes.jsonl",
        ROOT / "data" / "graph_data" / "current" / "edges.jsonl",
        ROOT / "data" / "chroma_db" / "chroma.sqlite3",
    ]
    hash_rows = []
    for path in hash_targets:
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            hash_rows.append((str(path.relative_to(ROOT)), digest))
    lines = [
        "# 知识库提交总说明", "",
        "## 比赛数据包对应关系", "",
        "比赛 PDF 第 5 页要求至少提交一套垂直领域专业知识库切片。项目提交主体为 `data/知识库原始数据/` 下的 15 个板块、73 个知识点文章。", "",
        "## 版本与数量", "",
        "- 知识库版本：`KB-20260828-r2`。", "- 知识点：73 个。", "- 原始 Markdown 文章：73 篇。", "- 运行时资料索引：68 条（系统派生索引，不等于知识点数量）。", "- 题目：323 道，73/73 个知识点有题目。", "",
        "## 来源、引用和映射", "",
        "文章顶部保留多个来源网址，正文使用 `①②③` 等圈号标记借鉴来源。来源、作者、发布时间和渠道以文章内原始信息及人工核验为准，不凭空补写。", "",
        "知识点到文章、视频、题目的程序索引见 `knowledge_subtopic_article_video_question_map.csv`；该表用于回溯，不替代文章正文。", "",
        "## 关键事实人工核验", "",
        "关键事实人工核验记录使用 `测试数据与评测材料/06_指标统计/fact_check_review.csv`，重点核验版本、CUDA/PyTorch兼容性、显存、LoRA/QLoRA参数、vLLM参数和实际命令结果。", "",
        "## 数据 Hash（SHA-256）", "",
    ]
    for rel, digest in hash_rows:
        lines.append(f"- `{rel}`：`{digest}`")
    lines += ["", "## 授权、引用和脱敏", "", "第三方文章、官方文档、论文和视频仅作为学习资料引用；提交包保留来源和引用说明，不声称拥有第三方版权。提交前清除 API Key、账号、Cookie、个人信息和本机敏感路径；`backend/.env` 不提交。", "", "## 人工审核记录", "", "人工审核结果填写在 `knowledge_article_review.csv`，完成后再勾选 `knowledge_submission_checklist.md`。"]
    (OUT / "知识库提交总说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
