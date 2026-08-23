"""将原始视频 Markdown 清单转换为知识库资源目录。"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEEDS_DIR = DATA_DIR / "seeds"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
VIDEO_CATALOG_PATH = KNOWLEDGE_BASE_DIR / "video_resources.json"
RAW_FILE = next((DATA_DIR / "知识库原始数据").glob("*.md"))


def normalize(value: str) -> str:
    value = re.sub(r"<[^>]+>|~~|\*", "", value)
    value = re.sub(r"[-－]\s*\d+(?:\.\d+)?\s*h\s*$", "", value, flags=re.I)
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", value).lower()


def canonicalize_url(raw_url: str) -> str:
    parts = urlsplit(raw_url.replace("\\_", "_"))
    query = [(key, value) for key, value in parse_qsl(parts.query) if key not in {"vd_source", "share_source"}]
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), urlencode(query), ""))


def score_match(source_heading: str, subtopic_name: str) -> float:
    source = normalize(source_heading)
    target = normalize(subtopic_name)
    if not source or not target:
        return 0.0
    if source == target:
        return 1.0
    if source in target or target in source:
        return 0.95
    return SequenceMatcher(None, source, target).ratio()


def load_subtopics() -> dict[str, list[dict]]:
    subtopics = json.loads((SEEDS_DIR / "knowledge_subtopics.json").read_text(encoding="utf-8"))
    result: dict[str, list[dict]] = defaultdict(list)
    for subtopic in subtopics:
        result[subtopic["block_id"]].append(subtopic)
    return result


def parse_resources(subtopics_by_block: dict[str, list[dict]]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    block_lookup = json.loads((SEEDS_DIR / "knowledge_blocks.json").read_text(encoding="utf-8"))
    block_by_name = {normalize(item["block_name"]): item["block_id"] for item in block_lookup}
    resources, mappings, block_mappings, review_items = [], [], [], []
    current_block_id = None
    current_heading = None
    mapping_order = defaultdict(int)
    block_mapping_order = defaultdict(int)
    seen_canonical_urls: set[str] = set()

    for line_number, raw_line in enumerate(RAW_FILE.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        block_match = re.match(r"^#{6}\s+\*\*(.+?)\*\*\s*$", line)
        if block_match:
            current_block_id = block_by_name.get(normalize(block_match.group(1)))
            current_heading = None
            continue

        heading_match = re.match(r"^\*\*(.+?)\*\*\s*$", line)
        # Environment preparation is a real ``-00`` learning node, not a
        # board-level decoration.  Its videos must be reachable by 个性化学习资源生成 Agent
        # when the installation stage is scheduled.
        if heading_match and "~~" not in line:
            current_heading = heading_match.group(1)
            continue

        url_match = re.search(r"https?://\S+", line)
        if not url_match:
            continue

        raw_url = url_match.group(0).rstrip("。）；")
        title = line[:url_match.start()].strip(" 【】") or "未命名视频"
        canonical_url = canonicalize_url(raw_url)
        if canonical_url in seen_canonical_urls:
            continue
        seen_canonical_urls.add(canonical_url)
        video_match = re.search(r"/video/(BV[0-9A-Za-z]+)", canonical_url)
        part_match = re.search(r"(?:[?&])p=(\d+)", canonical_url)
        resource_id = f"RES-VID-{len(resources) + 1:03d}"
        candidate = None
        confidence = 0.0
        if current_block_id and current_heading:
            candidates = subtopics_by_block[current_block_id]
            candidate = max(candidates, key=lambda item: score_match(current_heading, item["subtopic_name"]))
            confidence = score_match(current_heading, candidate["subtopic_name"])

        confirmed = candidate is not None and confidence >= 0.85
        summary = (
            f"用于学习“{candidate['content_summary']}”的推荐视频；导入后需补充视频内容摘要或字幕。"
            if candidate
            else "用于当前知识板块的总览或环境准备视频；导入后需补充视频内容摘要或字幕。"
        )
        resources.append({
            "resource_id": resource_id,
            "resource_type": "video",
            "title": title,
            "raw_url": raw_url,
            "canonical_url": canonical_url,
            "platform": "bilibili" if "bilibili.com" in canonical_url else "unknown",
            "external_id": video_match.group(1) if video_match else None,
            "external_part": part_match.group(1) if part_match else None,
            "content_summary": summary,
            "recommendation_level": "supplementary",
            "review_status": "pending",
            "metadata": {"raw_file": str(RAW_FILE.relative_to(PROJECT_ROOT)), "line_number": line_number},
        })
        if candidate:
            mapping_order[candidate["subtopic_id"]] += 1
            role = "primary" if mapping_order[candidate["subtopic_id"]] == 1 else "supplementary"
            resources[-1]["recommendation_level"] = role
            mappings.append({
                "resource_id": resource_id,
                "block_id": candidate["block_id"],
                "subtopic_id": candidate["subtopic_id"],
                "role": role,
                "sequence_index": mapping_order[candidate["subtopic_id"]],
                "relevance_score": round(confidence, 3),
                "mapping_status": "confirmed" if confirmed else "pending",
                "mapping_notes": f"从原始清单标题“{current_heading}”自动匹配。",
            })
        elif current_block_id:
            block_mapping_order[current_block_id] += 1
            role = "primary" if block_mapping_order[current_block_id] == 1 else "supplementary"
            resources[-1]["recommendation_level"] = role
            block_mappings.append({
                "resource_id": resource_id,
                "block_id": current_block_id,
                "role": role,
                "sequence_index": block_mapping_order[current_block_id],
                "relevance_score": 0.9,
                "mapping_status": "confirmed",
                "mapping_notes": "原始清单位于板块标题之后、首个子知识点之前，按板块级总览/环境准备资源处理。",
            })
        if not confirmed and not current_block_id:
            review_items.append({
                "resource_id": resource_id,
                "title": title,
                "line_number": line_number,
                "detected_heading": current_heading,
                "proposed_subtopic_id": candidate["subtopic_id"] if candidate else None,
                "proposed_subtopic_name": candidate["subtopic_name"] if candidate else None,
                "match_confidence": round(confidence, 3),
                "reason": "未检测到子知识点标题或自动匹配置信度低于 0.85。",
            })
    # Every stage, including an installation node with no dedicated video,
    # needs a safe video fallback.  Reuse the highest-confidence confirmed
    # subtopic resource as a block-level recommendation when necessary.
    mapped_blocks = {mapping["block_id"] for mapping in block_mappings}
    for block_id, subtopics in subtopics_by_block.items():
        if block_id in mapped_blocks:
            continue
        candidates = [
            mapping for mapping in mappings
            if mapping["block_id"] == block_id and mapping["mapping_status"] == "confirmed"
        ]
        if not candidates:
            continue
        fallback = sorted(
            candidates,
            key=lambda item: (-item["relevance_score"], item["sequence_index"], item["resource_id"]),
        )[0]
        block_mappings.append({
            "resource_id": fallback["resource_id"],
            "block_id": block_id,
            "role": "supplementary",
            "sequence_index": 99,
            "relevance_score": fallback["relevance_score"],
            "mapping_status": "confirmed",
            "mapping_notes": "该板块没有独立总览视频，复用已确认子知识点视频作为阶段回退资源。",
        })

    return resources, mappings, block_mappings, review_items


def main() -> None:
    resources, mappings, block_mappings, review_items = parse_resources(load_subtopics())
    catalog = {
        "schema_version": "1.0",
        "resource_type": "video",
        "resources": resources,
        "subtopic_mappings": mappings,
        "block_mappings": block_mappings,
        "manual_review": review_items,
    }
    VIDEO_CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"resources={len(resources)} subtopic_mappings={len(mappings)} block_mappings={len(block_mappings)} pending_review={len(review_items)}")


if __name__ == "__main__":
    main()
