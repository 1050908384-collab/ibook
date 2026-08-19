"""Build a compact, source-of-truth learner evidence package for Agent3."""
from __future__ import annotations

import json
from typing import Any

from models.contracts import LearningEvidence
from models.stage import StageInfo


class LearningEvidenceService:
    def __init__(self, db_client: Any):
        self.db_client = db_client

    def build(self, session_id: str, stage: StageInfo) -> LearningEvidence:
        statuses = self.db_client.list_subtopic_statuses(session_id)
        by_id = {str(item.get("subtopic_id")): item for item in statuses}
        target_ids = list(dict.fromkeys(stage.subtopic_ids or ([stage.subtopic_id] if stage.subtopic_id else [])))
        prerequisites: list[str] = []
        for subtopic_id in target_ids:
            raw = (by_id.get(str(subtopic_id)) or {}).get("prerequisites") or []
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = []
            prerequisites.extend(str(item) for item in raw if item)
        prerequisites.extend(str(item) for item in stage.prerequisites if item in by_id)
        prerequisites = list(dict.fromkeys(prerequisites))

        unstable_prerequisites = []
        for subtopic_id in prerequisites:
            item = by_id.get(subtopic_id)
            if not item:
                continue
            accuracy = item.get("last_accuracy")
            mastered = int(item.get("status") or 0) == 1
            stable = mastered and (accuracy is None or float(accuracy) >= 0.7)
            if not stable:
                unstable_prerequisites.append({
                    "subtopic_id": subtopic_id,
                    "subtopic_name": item.get("subtopic_name", subtopic_id),
                    "status": "掌握不稳" if mastered else "未掌握",
                    "last_accuracy": accuracy,
                    "mastery_score": item.get("mastery_score", 0),
                })

        completed_stages = [
            {
                "stage_num": item.get("stage_num"), "block_name": item.get("block_name"),
                "subtopic_id": item.get("subtopic_id"), "accuracy": item.get("accuracy"),
                "strategy_used": item.get("strategy_used"),
            }
            for item in self.db_client.list_stage_progress(session_id)
            if int(item.get("stage_num") or 0) < stage.stage_num and item.get("status") in {"completed", "retreat"}
        ]
        weak_topics = [
            item for item in completed_stages
            if item.get("accuracy") is not None and float(item["accuracy"]) < 0.7
        ]
        recent_questions = [
            str(item.get("question", "")).strip()[:180]
            for item in self.db_client.get_recent_qa(session_id, limit=5)
            if str(item.get("question", "")).strip()
        ]
        preflight_review = None
        if unstable_prerequisites:
            names = "、".join(item["subtopic_name"] for item in unstable_prerequisites[:3])
            preflight_review = {
                "title": "开始前快速回顾",
                "estimated_minutes": min(15, max(5, len(unstable_prerequisites) * 5)),
                "reason": f"本阶段依赖 {names}，其中存在未掌握或掌握不稳的内容。",
                "markdown": (
                    "## 开始前快速回顾\n\n"
                    f"本阶段会用到：{names}。建议先用 5-15 分钟回顾相关基础，再进入新内容。\n\n"
                    "- 写出一个最小示例，确认自己能读懂关键语法或操作。\n"
                    "- 遇到无法解释的地方，在本阶段学习时优先记录并提问。"
                ),
            }
        return LearningEvidence(
            completed_stages=completed_stages,
            mastered_subtopic_ids=[key for key, item in by_id.items() if int(item.get("status") or 0) == 1],
            unstable_prerequisites=unstable_prerequisites,
            weak_topics=weak_topics,
            recent_questions=recent_questions,
            completed_topic_names=[str(item.get("block_name") or "") for item in completed_stages if item.get("block_name")],
            preflight_review=preflight_review,
        )
