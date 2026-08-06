# -*- coding: utf-8 -*-
"""Agent1 knowledge taxonomy retrieval and grounding helpers."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import DATA_DIR


BLOCKS_SEED_FILE = DATA_DIR / "seeds" / "knowledge_blocks.json"
SUBTOPICS_SEED_FILE = DATA_DIR / "seeds" / "knowledge_subtopics.json"


EXTRA_BLOCK_ALIASES: dict[str, tuple[str, ...]] = {
    "KB-01": ("Linux", "Shell", "命令行", "终端", "WSL", "环境配置", "conda", "pip", "nvidia-smi"),
    "KB-02": ("Python", "Python基础", "编程基础", "Python语法"),
    "KB-03": ("数据处理", "数据分析", "数据清洗", "NumPy", "Pandas"),
    "KB-04": ("PyTorch", "torch", "深度学习框架", "自动求导", "DataLoader"),
    "KB-05": ("NLP", "自然语言处理", "文本表示"),
    "KB-06": ("注意力", "Attention", "Self-Attention", "Multi-Head Attention", "QKV"),
    "KB-07": ("Tokenizer", "分词器", "分词", "子词切分", "BPE"),
    "KB-08": ("Transformer", "Decoder-Only", "GPT", "Llama", "Qwen", "RoPE", "KV Cache"),
    "KB-09": ("LLM", "大语言模型", "大模型", "上下文窗口", "Prompt"),
    "KB-10": ("LLM训练", "大模型训练", "预训练", "SFT", "RLHF", "指令微调", "对齐训练"),
    "KB-11": ("HuggingFace", "transformers库", "datasets库", "Trainer", "Model Hub"),
    "KB-12": (
        "LoRA", "QLoRA", "PEFT", "参数高效微调", "大模型微调", "模型微调", "微调",
        "量化微调", "NF4", "4-bit量化", "4bit量化",
    ),
    "KB-13": ("PyTorch进阶", "训练优化", "混合精度", "梯度累积", "梯度检查点"),
    "KB-14": ("DeepSpeed", "分布式训练", "多卡训练", "ZeRO"),
    "KB-15": (
        "vLLM", "模型部署", "推理部署", "部署", "推理服务", "PagedAttention",
        "Continuous Batching", "OpenAI兼容接口",
    ),
}


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


@lru_cache(maxsize=1)
def _catalog() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    blocks = _load_json_list(BLOCKS_SEED_FILE)
    subtopics = _load_json_list(SUBTOPICS_SEED_FILE)
    subtopics_by_block: dict[str, list[dict[str, Any]]] = {}
    for subtopic in subtopics:
        block_id = str(subtopic.get("block_id", ""))
        if block_id:
            subtopics_by_block.setdefault(block_id, []).append(subtopic)
    for rows in subtopics_by_block.values():
        rows.sort(key=lambda item: (int(item.get("sequence_index") or 0), str(item.get("subtopic_id", ""))))
    return blocks, subtopics_by_block


class KnowledgeCatalogTool:
    """Retrieve the canonical block/subtopic taxonomy before LLM classification."""

    def list_blocks(self) -> list[dict[str, Any]]:
        blocks, _ = _catalog()
        return [dict(block) for block in blocks]

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        normalized_query = _normalize(query)
        if not normalized_query:
            return []

        blocks, subtopics_by_block = _catalog()
        candidates: list[dict[str, Any]] = []
        for block in blocks:
            block_id = str(block.get("block_id", ""))
            matched_terms: list[str] = []
            score = 0.0
            for alias in self._block_aliases(block):
                normalized_alias = _normalize(alias)
                if block_id == "KB-09" and normalized_alias == "llm" and "vllm" in normalized_query:
                    continue
                if len(normalized_alias) < 2 or normalized_alias not in normalized_query:
                    continue
                matched_terms.append(alias)
                score += 20.0 + min(len(normalized_alias), 20) * 0.5
                if alias == block.get("block_name"):
                    score += 8.0

            matched_subtopics: list[dict[str, Any]] = []
            for subtopic in subtopics_by_block.get(block_id, []):
                subtopic_name = str(subtopic.get("subtopic_name", ""))
                normalized_name = _normalize(subtopic_name)
                if len(normalized_name) >= 2 and normalized_name in normalized_query:
                    score += 18.0 + min(len(normalized_name), 20) * 0.4
                    matched_subtopics.append(self._compact_subtopic(subtopic))

            if score <= 0:
                continue
            candidates.append({
                "block_id": block_id,
                "block_name": block.get("block_name", block_id),
                "description": block.get("description", ""),
                "difficulty": block.get("difficulty", ""),
                "estimated_hours": block.get("estimated_hours", 0),
                "prerequisite_ids": list(block.get("prerequisites", [])),
                "matched_terms": self._dedupe(matched_terms),
                "matched_subtopics": matched_subtopics,
                "score": round(score, 3),
            })

        candidates.sort(key=lambda item: (-float(item["score"]), str(item["block_id"])))
        return candidates[: max(1, top_k)]

    def prompt_candidates(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        matches = self.search(query, top_k=top_k)
        if matches:
            enriched_matches = []
            for item in matches:
                block = self.get_block(str(item.get("block_id", ""))) or {}
                enriched = dict(item)
                enriched["subtopics"] = block.get("subtopics", [])
                enriched_matches.append(enriched)
            return [self._prompt_candidate(item) for item in enriched_matches]

        blocks, subtopics_by_block = _catalog()
        return [
            {
                "block_id": block.get("block_id", ""),
                "block_name": block.get("block_name", ""),
                "aliases": list(block.get("corresponding_gaps", [])),
                "subtopics": [
                    self._compact_subtopic(item)
                    for item in subtopics_by_block.get(str(block.get("block_id", "")), [])[:3]
                ],
            }
            for block in blocks
        ]

    def merge_recall_results(
        self,
        lexical_candidates: list[dict[str, Any]],
        semantic_hits: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Union precise alias matches with semantic document evidence by block."""
        by_block_id = {str(item.get("block_id", "")): dict(item) for item in lexical_candidates}
        for hit in semantic_hits:
            block_id = str(hit.get("block_id", ""))
            block = self.get_block(block_id)
            if not block:
                continue
            candidate = by_block_id.setdefault(block_id, {
                "block_id": block_id,
                "block_name": block.get("block_name", block_id),
                "description": block.get("description", ""),
                "difficulty": block.get("difficulty", ""),
                "estimated_hours": block.get("estimated_hours", 0),
                "prerequisite_ids": list(block.get("prerequisites", [])),
                "matched_terms": [],
                "matched_subtopics": [],
                "score": 0.0,
            })
            candidate.setdefault("subtopics", block.get("subtopics", []))
            candidate.setdefault("semantic_hits", []).append(hit)
            candidate["score"] = float(candidate.get("score") or 0.0) + float(hit.get("similarity") or 0.0) * 24.0
            subtopic_id = str(hit.get("subtopic_id", ""))
            active_subtopic_ids = {item["subtopic_id"] for item in block.get("subtopics", [])}
            if (
                subtopic_id in active_subtopic_ids
                and all(item.get("subtopic_id") != subtopic_id for item in candidate["matched_subtopics"])
            ):
                candidate["matched_subtopics"].append({
                    "subtopic_id": subtopic_id,
                    "subtopic_name": hit.get("title", ""),
                    "difficulty": hit.get("difficulty", ""),
                    "estimated_hours": 0,
                    "content_summary": hit.get("content_summary", ""),
                    "direct_prerequisite_ids": hit.get("direct_prerequisite_ids", []),
                })

        results = []
        for candidate in by_block_id.values():
            block = self.get_block(str(candidate.get("block_id", "")))
            if block:
                candidate.setdefault("subtopics", block.get("subtopics", []))
            lexical_score = float(candidate.get("score") or 0.0)
            semantic_score = max((float(hit.get("similarity") or 0.0) for hit in candidate.get("semantic_hits", [])), default=0.0)
            candidate["mapping_confidence"] = round(min(0.99, max(
                0.0,
                0.58 + min(0.32, lexical_score / 120.0),
                0.45 + semantic_score * 0.45,
            )), 3)
            results.append(candidate)
        results.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("block_id", ""))))
        return results[:max(1, top_k)]

    def resolve_block(self, value: str, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        normalized_value = _normalize(value)
        if not normalized_value:
            return None

        blocks, subtopics_by_block = _catalog()
        candidate_ids = {
            str(candidate.get("block_id", ""))
            for candidate in (candidates or [])
            if candidate.get("block_id")
        }
        ordered_blocks = sorted(
            blocks,
            key=lambda block: 0 if str(block.get("block_id", "")) in candidate_ids else 1,
        )
        partial_matches: list[tuple[int, dict[str, Any]]] = []
        for block in ordered_blocks:
            block_id = str(block.get("block_id", ""))
            for alias in self._block_aliases(block):
                normalized_alias = _normalize(alias)
                if not normalized_alias:
                    continue
                if normalized_value == normalized_alias:
                    return self._resolved_block(block, subtopics_by_block.get(block_id, []), 1.0)
                if len(normalized_value) >= 2 and (
                    normalized_value in normalized_alias or normalized_alias in normalized_value
                ):
                    partial_matches.append((min(len(normalized_value), len(normalized_alias)), block))

        if not partial_matches:
            return None
        _, block = max(partial_matches, key=lambda item: item[0])
        block_id = str(block.get("block_id", ""))
        return self._resolved_block(block, subtopics_by_block.get(block_id, []), 0.8)

    def get_block(self, block_id: str) -> dict[str, Any] | None:
        blocks, subtopics_by_block = _catalog()
        for block in blocks:
            if block.get("block_id") == block_id:
                return self._resolved_block(block, subtopics_by_block.get(block_id, []), 1.0)
        return None

    def resolve_subtopic_ids(self, block_id: str, values: list[Any]) -> list[str]:
        block = self.get_block(block_id)
        if not block:
            return []
        subtopics = block.get("subtopics", [])
        resolved: list[str] = []
        for value in values or []:
            raw = value.get("subtopic_id") or value.get("subtopic_name") or "" if isinstance(value, dict) else str(value)
            normalized_raw = _normalize(raw)
            if not normalized_raw:
                continue
            for subtopic in subtopics:
                if normalized_raw in {
                    _normalize(subtopic.get("subtopic_id", "")),
                    _normalize(subtopic.get("subtopic_name", "")),
                }:
                    resolved.append(str(subtopic["subtopic_id"]))
                    break
        return self._dedupe(resolved)

    def default_target_subtopic_ids(self, block_id: str, query: str) -> list[str]:
        """Choose a canonical target when recall finds a block but no valid node."""
        normalized_query = _normalize(query)
        if block_id == "KB-12":
            if "qlora" in normalized_query or "4bit" in normalized_query or "nf4" in normalized_query:
                return ["KBS-12-04"]
            if any(term in normalized_query for term in ("merge", "导出", "合并")):
                return ["KBS-12-05"]
            return ["KBS-12-03"]
        return []

    def canonical_skill_label(self, block: dict[str, Any], query: str) -> str:
        normalized_query = _normalize(query)
        aliases = sorted(self._block_aliases(block), key=lambda item: len(_normalize(item)), reverse=True)
        for alias in aliases:
            normalized_alias = _normalize(alias)
            if normalized_alias and normalized_alias in normalized_query:
                if re.fullmatch(r"[a-zA-Z0-9+_. -]+", alias) or alias in block.get("corresponding_gaps", []):
                    return alias
        return str(block.get("block_name") or block.get("block_id") or query)

    def _block_aliases(self, block: dict[str, Any]) -> list[str]:
        block_id = str(block.get("block_id", ""))
        aliases = [
            block_id,
            str(block.get("block_name", "")),
            *[str(item) for item in block.get("corresponding_gaps", [])],
            *EXTRA_BLOCK_ALIASES.get(block_id, ()),
        ]
        if block_id == "KB-06":
            aliases = [alias for alias in aliases if _normalize(alias) != "transformer"]
        return self._dedupe([alias for alias in aliases if alias])

    @staticmethod
    def _compact_subtopic(subtopic: dict[str, Any]) -> dict[str, Any]:
        return {
            "subtopic_id": subtopic.get("subtopic_id", ""),
            "subtopic_name": subtopic.get("subtopic_name", ""),
            "difficulty": subtopic.get("difficulty", ""),
            "estimated_hours": subtopic.get("estimated_hours", 0),
            "sequence_index": subtopic.get("sequence_index", 0),
            "direct_prerequisite_ids": list(subtopic.get("prerequisites", [])),
            "content_summary": subtopic.get("content_summary", ""),
        }

    @staticmethod
    def _prompt_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "block_id": candidate.get("block_id", ""),
            "block_name": candidate.get("block_name", ""),
            "matched_terms": candidate.get("matched_terms", []),
            "matched_subtopics": candidate.get("matched_subtopics", []),
            "description": candidate.get("description", ""),
            "difficulty": candidate.get("difficulty", ""),
            "estimated_hours": candidate.get("estimated_hours", 0),
            "prerequisite_ids": candidate.get("prerequisite_ids", []),
            "subtopics": candidate.get("subtopics", []),
            "semantic_hits": candidate.get("semantic_hits", []),
        }

    def _resolved_block(
        self,
        block: dict[str, Any],
        subtopics: list[dict[str, Any]],
        confidence: float,
    ) -> dict[str, Any]:
        payload = dict(block)
        payload["subtopics"] = [self._compact_subtopic(item) for item in subtopics]
        payload["mapping_confidence"] = confidence
        return payload

    @staticmethod
    def _dedupe(values: list[Any]) -> list[Any]:
        seen: set[Any] = set()
        result: list[Any] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result
