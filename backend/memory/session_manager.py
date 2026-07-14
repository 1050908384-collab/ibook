from typing import Dict, Any, Optional, List
import uuid

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str = None) -> Dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        self.sessions[session_id] = {
            "session_id": session_id,
            "profile": None,
            "current_stage": None,
            "learning_package": None,
            "qa_history": [],
            "answer_records": [],
            "interactions": [],
            "created_at": None,
            "updated_at": None
        }
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def update_profile(self, session_id: str, profile: Dict[str, Any]) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["profile"] = profile

    def get_profile(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id, {}).get("profile")

    def update_stage(self, session_id: str, stage_info: Dict[str, Any]) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["current_stage"] = stage_info

    def get_current_stage(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id, {}).get("current_stage")

    def update_learning_package(self, session_id: str, package: Dict[str, Any]) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["learning_package"] = package

    def get_learning_package(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id, {}).get("learning_package")

    def add_qa_record(self, session_id: str, question: str, answer: str, tech_pass: bool = True, quality_pass: bool = True) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["qa_history"].append({
                "question": question,
                "answer": answer,
                "tech_pass": tech_pass,
                "quality_pass": quality_pass
            })

    def get_qa_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, {}).get("qa_history", [])

    def add_answer_record(self, session_id: str, question_id: str, user_answer: str, is_correct: bool, time_spent: int = 0) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["answer_records"].append({
                "question_id": question_id,
                "user_answer": user_answer,
                "is_correct": is_correct,
                "time_spent": time_spent
            })

    def get_answer_records(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, {}).get("answer_records", [])

    def add_interaction(self, session_id: str, interaction_type: str, data: Dict[str, Any]) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["interactions"].append({
                "type": interaction_type,
                "data": data
            })

    def get_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, {}).get("interactions", [])

    def end_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]
