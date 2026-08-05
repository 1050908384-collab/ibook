# -*- coding: utf-8 -*-
#调用了backend/db/里面的文件夹
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from db.sqlite_client import SQLiteClient
from models.profile import UserProfile
from models.stage import LearningPackage, StageInfo
from models.user_data import SessionData


class StateManager:
    def __init__(self, db_client: SQLiteClient | None = None):
        self.db_client = db_client or SQLiteClient()
        self._cache: Dict[str, SessionData] = {}

    def create_session(self, session_id: str | None = None) -> Dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        state = SessionData(session_id=session_id, created_at=now, updated_at=now)
        self.db_client.create_session(session_id)
        self._save(state)
        return state.model_dump(mode="json")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        return state.model_dump(mode="json") if state else None

    def require_session(self, session_id: str) -> SessionData:
        state = self._get(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")
        if state.status != "active":
            raise ValueError(f"Session is not active: {session_id}")
        return state

    def update_profile(self, session_id: str, profile: UserProfile | dict) -> None:
        state = self.require_session(session_id)
        state.profile = UserProfile.model_validate(profile)
        state.user_id = state.profile.user_id
        state.goal = state.profile.goal
        state.target_daily_hours = state.profile.daily_hours
        self.db_client.save_profile(session_id, state.profile.model_dump(mode="json"))
        self._save(state)

    def get_profile(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        return state.profile.model_dump(mode="json") if state and state.profile else None

    def update_stage(self, session_id: str, stage: StageInfo | dict) -> None:
        state = self.require_session(session_id)
        state.stage_info = StageInfo.model_validate(stage)
        state.current_stage = state.stage_info.stage_num
        self.db_client.save_stage(session_id, state.stage_info.model_dump(mode="json"))
        self._save(state)

    def get_current_stage(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        return state.stage_info.model_dump(mode="json") if state and state.stage_info else None

    def update_learning_package(self, session_id: str, package: LearningPackage | dict) -> None:
        state = self.require_session(session_id)
        state.learning_package = LearningPackage.model_validate(package)
        self._save(state)

    def get_learning_package(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        return state.learning_package.model_dump(mode="json") if state and state.learning_package else None

    def update_next_stage(self, session_id: str, next_stage: StageInfo | dict) -> None:
        state = self.require_session(session_id)
        state.next_stage = StageInfo.model_validate(next_stage)
        self._save(state)

    def get_next_stage(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        return state.next_stage.model_dump(mode="json") if state and state.next_stage else None

    def clear_next_stage(self, session_id: str) -> None:
        """清空已应用过的 next_stage 和 planning_notes，避免重复套用
        """
        state = self.require_session(session_id)
        state.next_stage = None
        state.next_stage_planning_notes = None
        self._save(state)

    def update_planning_notes(self, session_id: str, notes: str) -> None:
        state = self.require_session(session_id)
        state.next_stage_planning_notes = notes
        self._save(state)

    def get_planning_notes(self, session_id: str) -> Optional[str]:
        state = self._get(session_id)
        return state.next_stage_planning_notes if state else None

    def add_qa_record(self, session_id: str, question: str, answer: str, tech_pass: bool = True, quality_pass: bool = True) -> None:
        state = self.require_session(session_id)
        record = {"question": question, "answer": answer, "tech_pass": tech_pass, "quality_pass": quality_pass}
        state.qa_history.append(record)
        self.db_client.save_qa_record(session_id, question, answer, tech_pass, quality_pass, state.current_stage)
        self._save(state)

    def get_qa_history(self, session_id: str) -> List[Dict[str, Any]]:
        state = self._get(session_id)
        return list(state.qa_history) if state else []

    def add_answer_record(self, session_id: str, question_id: str, user_answer: str, is_correct: bool, time_spent: int = 0) -> None:
        state = self.require_session(session_id)
        record = {
            "stage": state.current_stage,
            "question_id": question_id,
            "user_answer": user_answer,
            "is_correct": is_correct,
            "time_spent": time_spent,
        }
        state.answer_records.append(record)
        self.db_client.save_answer_record(session_id, question_id, user_answer, is_correct, time_spent, state.current_stage)
        self._save(state)

    def get_answer_records(self, session_id: str) -> List[Dict[str, Any]]:
        state = self._get(session_id)
        return list(state.answer_records) if state else []

    def calculate_accuracy(self, session_id: str) -> float:
        state = self._get(session_id)
        if not state:
            return 0.0
        answers = [
            answer
            for answer in state.answer_records
            if answer.get("stage", state.current_stage) == state.current_stage
        ]
        return sum(1 for answer in answers if answer["is_correct"]) / len(answers) if answers else 0.0

    def add_interaction(self, session_id: str, interaction_type: str, data: Dict[str, Any]) -> None:
        state = self.require_session(session_id)
        state.interactions.append({"type": interaction_type, "data": data})
        self._save(state)

    def get_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        state = self._get(session_id)
        return list(state.interactions) if state else []

    def set_status(self, session_id: str, status: str) -> None:
        state = self._get(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")
        state.status = status
        self.db_client.update_session_status(session_id, status)
        self._save(state)

    def end_session(self, session_id: str) -> None:
        self.set_status(session_id, "completed")

    def abandon_session(self, session_id: str) -> None:
        self.set_status(session_id, "abandoned")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """获取所有会话列表"""
        return self.db_client.list_sessions()

    def update_field(self, session_id: str, field: str, value: Any) -> None:
        """更新 SessionData 的任意字段"""
        state = self._get(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")
        setattr(state, field, value)
        self._save(state)

    def refresh_cached_user_profile(self, user_id: str) -> None:
        for session_id, state in self._cache.items():
            if state.user_id != user_id:
                continue
            profile = self.db_client.get_latest_profile(session_id)
            if profile:
                state.profile = UserProfile.model_validate(profile)

    def refresh_cached_target(self, session_id: str) -> None:
        state = self._cache.get(session_id)
        if not state:
            return
        target = self.db_client.get_target_settings(session_id) or {}
        state.goal = target.get("goal") or ""
        state.target_total_hours = target.get("target_total_hours") or 0.0
        state.target_daily_hours = target.get("target_daily_hours") or 2.0
        state.target_progress = target.get("target_progress") or 0.0
        profile = self.db_client.get_latest_profile(session_id)
        if profile:
            state.profile = UserProfile.model_validate(profile)

    def refresh_cached_stage_duration(self, session_id: str, stage_id: str, estimated_hours: float) -> None:
        state = self._cache.get(session_id)
        if state and state.stage_info and state.stage_info.stage_id == stage_id:
            state.stage_info.estimated_hours = estimated_hours

    def _get(self, session_id: str) -> SessionData | None:
        if session_id in self._cache:
            return self._cache[session_id]
        payload = self.db_client.load_session_state(session_id)
        if not payload:
            return None
        state = SessionData.model_validate(payload)
        target = self.db_client.get_target_settings(session_id) or {}
        state.user_id = target.get("user_id") or state.user_id
        state.goal = target.get("goal") or state.goal
        state.target_total_hours = target.get("target_total_hours") or 0.0
        state.target_daily_hours = target.get("target_daily_hours") or state.target_daily_hours
        state.target_progress = target.get("target_progress") or 0.0
        composed_profile = self.db_client.get_latest_profile(session_id)
        if composed_profile:
            state.profile = UserProfile.model_validate(composed_profile)
        self._cache[session_id] = state
        return state

    def _save(self, state: SessionData) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self._cache[state.session_id] = state
        payload = state.model_dump(mode="json")
        # The combined profile is a compatibility DTO. Stable learner fields
        # live in learner_profiles and must not be duplicated in session JSON.
        payload["profile"] = None
        self.db_client.save_session_state(state.session_id, payload)
