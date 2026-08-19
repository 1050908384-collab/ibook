# -*- coding: utf-8 -*-
"""Agent4 checker for whether a block applies Agent3's personalization plan."""
from __future__ import annotations

from typing import Any


_PRACTICE_MARKERS = ("实践", "实操", "动手", "示例", "practice")
_VALID_ACTIONS = {
    "keep", "remediate",
    "explain",
    "explain_and_practice",
    "compress",
    "reorder",
    "contextualize",
}


class PersonalizationCheckTool:
    """Validate structured teaching-layer metadata without judging source facts."""

    async def check(self, block: dict[str, Any], profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        profile = profile or {}
        block = block or {}
        issues: list[dict[str, Any]] = []
        action = str(block.get("personalization_action") or "").strip()
        reason = str(block.get("personalization_reason") or "").strip()
        source_ids = [item for item in block.get("fact_source_chunk_ids", []) if item]
        tasks = [item for item in block.get("teaching_tasks", []) if isinstance(item, dict)]
        evidence_refs = [item for item in block.get("evidence_refs", []) if item]
        style = str(profile.get("learning_style") or "").lower()

        if action not in _VALID_ACTIONS:
            issues.append({
                "error_type": "personalization_action_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "内容块未声明有效的个性化处理动作。",
                "suggestion": "声明 keep、compress、explain、reorder、contextualize 或 explain_and_practice。",
            })
        if not reason:
            issues.append({
                "error_type": "personalization_reason_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "内容块未记录本次个性化的用户画像依据。",
                "suggestion": "记录学习目标、学习方式或实训条件中至少一项作为生成依据。",
            })
        if not source_ids:
            issues.append({
                "error_type": "teaching_source_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "个性化教学层未声明对应的事实来源块。",
                "suggestion": "在 fact_source_chunk_ids 中保留当前 source_chunk_id。",
            })
        if action == "remediate" and not evidence_refs:
            issues.append({
                "error_type": "remediation_evidence_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "前置补救内容未绑定学习证据。",
                "suggestion": "在 evidence_refs 中记录对应的前置知识点或低分阶段。",
            })
        if action == "remediate" and not tasks:
            issues.append({
                "error_type": "remediation_task_missing",
                "severity": "major",
                "location": "教学任务",
                "description": "前置补救内容缺少可完成的快速检查。",
                "suggestion": "提供 5-15 分钟内可完成的最小回顾或验证任务。",
            })
        if any(marker in style for marker in _PRACTICE_MARKERS) and not tasks:
            issues.append({
                "error_type": "practice_task_missing",
                "severity": "major",
                "location": "教学任务",
                "description": "实操型学习者的内容块缺少学习检查或验证任务。",
                "suggestion": "补充不引入新技术事实的操作验证、观察记录或自我检查任务。",
            })
        return issues
