# -*- coding: utf-8 -*-
#调用了backend/db/里面的文件夹
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from db.sqlite_client import SQLiteClient
from models.profile import PracticeProfile, UserProfile
from models.stage import LearningPackage, StageInfo
from models.user_data import SessionData


class StateManager:
    def __init__(self, db_client: SQLiteClient | None = None):
        self.db_client = db_client or SQLiteClient()
        self._cache: Dict[str, SessionData] = {}

    def create_session(self, session_id: str | None = None, user_id: str = "local-user") -> Dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        state = SessionData(session_id=session_id, user_id=user_id, created_at=now, updated_at=now)
        self.db_client.create_session(session_id, user_id)
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

    def update_practice_profile(self, session_id: str, profile: PracticeProfile | dict) -> None:
        state = self.require_session(session_id)
        practice_profile = PracticeProfile.model_validate(profile)
        # Hardware belongs to the user-level profile. PracticeProfile is only
        # the Agent3 execution view and must not become another source of truth.
        self.db_client.upsert_user_profile(
            self.db_client.get_session_user_id(session_id),
            {"hardware": practice_profile.hardware},
        )
        state.practice_profile = practice_profile
        self._save(state)

    def get_practice_profile(self, session_id: str) -> Dict[str, Any]:
        state = self.require_session(session_id)
        user_profile = self.db_client.get_user_profile(
            self.db_client.get_session_user_id(session_id)
        ) or {}
        state.practice_profile = PracticeProfile(
            hardware=user_profile.get("hardware", "")
        )
        return state.practice_profile.model_dump(mode="json")

    def refresh_practice_profile(self, session_id: str) -> None:
        """Refresh the runtime Agent3 DTO from the user-level hardware field."""
        state = self.require_session(session_id)
        user_profile = self.db_client.get_user_profile(
            self.db_client.get_session_user_id(session_id)
        ) or {}
        state.practice_profile = PracticeProfile(
            hardware=user_profile.get("hardware", "")
        )
        self._save(state)

    def update_stage(self, session_id: str, stage: StageInfo | dict) -> None:
        state = self.require_session(session_id)
        next_stage = StageInfo.model_validate(stage)
        if state.stage_info and state.stage_info.stage_id != next_stage.stage_id:
            self._save_current_stage_snapshot(state)
            # A new stage starts with an independent dialogue context. The
            # completed stage's history remains in its snapshot.
            state.qa_history = []
        state.stage_info = next_stage
        state.current_stage = state.stage_info.stage_num
        self.db_client.save_stage(session_id, state.stage_info.model_dump(mode="json"))
        self._save(state)

    def get_current_stage(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        return state.stage_info.model_dump(mode="json") if state and state.stage_info else None

    def update_learning_package(self, session_id: str, package: LearningPackage | dict) -> None:
        state = self.require_session(session_id)
        state.learning_package = LearningPackage.model_validate(package)
        self._save_current_stage_snapshot(state)
        self._save(state)

    def get_learning_package(self, session_id: str) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        return state.learning_package.model_dump(mode="json") if state and state.learning_package else None

    def get_stage_snapshot(self, session_id: str, stage_num: int) -> Optional[Dict[str, Any]]:
        state = self._get(session_id)
        if not state:
            return None
        if state.stage_info and state.stage_info.stage_num == stage_num:
            return {
                "stage_info": state.stage_info.model_dump(mode="json"),
                "learning_package": state.learning_package.model_dump(mode="json") if state.learning_package else None,
                # ``qa_history`` in state_json is a runtime cache. Older
                # sessions can retain records from a previous stage there,
                # so the stage-aware table is the source of truth for the
                # user-visible conversation.
                "qa_history": self.db_client.list_qa_records_for_stage(session_id, stage_num),
            }
        snapshot = state.stage_snapshots.get(str(stage_num))
        if snapshot:
            snapshot = dict(snapshot)
            snapshot["qa_history"] = self.db_client.list_qa_records_for_stage(session_id, stage_num)
            return snapshot

        # Sessions created before stage snapshots were introduced only retain
        # their completed stages in stage_progress. Keep those stages
        # navigable instead of making the entire stage switch fail.
        legacy_stage = next(
            (item for item in self.db_client.list_stage_progress(session_id)
             if item.get("stage_num") == stage_num),
            None,
        )
        if not legacy_stage:
            return None
        legacy_stage = dict(legacy_stage)
        legacy_stage.setdefault("block_ids", [legacy_stage.get("block_id")] if legacy_stage.get("block_id") else [])
        legacy_stage.setdefault("subtopic_ids", [legacy_stage.get("subtopic_id")] if legacy_stage.get("subtopic_id") else [])
        legacy_stage.setdefault("subtopic_names", [])
        legacy_stage.setdefault("gap_skill", legacy_stage.get("block_name") or "")
        return {
            "stage_info": legacy_stage,
            "learning_package": None,
            "qa_history": self.db_client.list_qa_records_for_stage(session_id, stage_num),
        }

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

    def add_qa_record(
        self,
        session_id: str,
        question: str,
        answer: str,
        tech_pass: bool = True,
        quality_pass: bool = True,
        stage_num: Optional[int] = None,
    ) -> None:
        state = self.require_session(session_id)
        target_stage_num = stage_num if stage_num is not None else state.current_stage
        record = {
            "stage": target_stage_num,
            "question": question,
            "answer": answer,
            "tech_pass": tech_pass,
            "quality_pass": quality_pass,
        }
        self.db_client.save_qa_record(session_id, question, answer, tech_pass, quality_pass, target_stage_num)
        # Runtime state only represents the active stage. Historical-stage
        # messages remain isolated in user_qa and are read by stage number.
        if target_stage_num == state.current_stage:
            state.qa_history.append(record)
            self._save_current_stage_snapshot(state)
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

    def delete_session(self, session_id: str, user_id: str | None = "local-user") -> Dict[str, int]:
        """物理删除 session：清除 DB 所有关联快照 + 内存缓存（不可逆）。"""
        deletes = self.db_client.delete_session(session_id, user_id)
        # 同步清掉内存缓存，避免后续请求还读到旧数据
        self._cache.pop(session_id, None)
        return deletes

    def delete_last_stage(self, session_id: str, user_id: str | None = "local-user") -> Dict[str, object]:
        """删除 session 的最后一个学习阶段；若该 session 已无阶段，自动删整个 session。"""
        details = self.db_client.delete_last_stage(session_id, user_id)
        # 同步丢内存缓存，下一次请求会从 DB 重新加载最新状态
        self._cache.pop(session_id, None)
        return details

    def list_sessions(self, user_id: str = "local-user") -> List[Dict[str, Any]]:
        """获取指定用户的会话列表。"""
        return self.db_client.list_sessions(user_id)

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
            user_profile = self.db_client.get_user_profile(user_id) or {}
            state.practice_profile = PracticeProfile(
                hardware=user_profile.get("hardware", "")
            )

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
        if state.stage_info and state.stage_info.stage_id:
            duration = self.db_client.get_stage_duration(session_id, state.stage_info.stage_id)
            if duration:
                state.stage_info.estimated_hours = duration["estimated_hours"]

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
        user_profile = self.db_client.get_user_profile(state.user_id) or {}
        state.practice_profile = PracticeProfile(
            hardware=user_profile.get("hardware", "")
        )
        self._cache[session_id] = state
        return state

    def _save(self, state: SessionData) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self._cache[state.session_id] = state
        payload = state.model_dump(mode="json")
        # The combined profile is a compatibility DTO. Stable learner fields
        # live in learner_profiles and must not be duplicated in session JSON.
        payload["profile"] = None
        # PracticeProfile is rebuilt from learner_profiles on restore. Keeping
        # it out of session JSON avoids duplicating user hardware per target.
        payload.pop("practice_profile", None)
        self.db_client.save_session_state(state.session_id, payload)

    @staticmethod
    def _save_current_stage_snapshot(state: SessionData) -> None:
        """Copy the active stage's user-visible state before it is replaced."""
        if not state.stage_info:
            return
        state.stage_snapshots[str(state.stage_info.stage_num)] = {
            "stage_info": state.stage_info.model_dump(mode="json"),
            "learning_package": state.learning_package.model_dump(mode="json") if state.learning_package else None,
            "qa_history": list(state.qa_history),
        }
