from typing import Dict, Any, Optional, List
from memory.session_manager import SessionManager
from db.sqlite_client import SQLiteClient

class StateManager:
    def __init__(self):
        self.session_manager = SessionManager()
        self.db_client = SQLiteClient()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.session_manager.get_session(session_id)

    def create_session(self, session_id: str) -> Dict[str, Any]:
        self.db_client.create_session(session_id)
        return self.session_manager.create_session(session_id)

    def update_profile(self, session_id: str, profile: Dict[str, Any]) -> None:
        self.session_manager.update_profile(session_id, profile)
        self.db_client.save_profile(session_id, profile)

    def get_profile(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.session_manager.get_profile(session_id)

    def update_stage(self, session_id: str, stage_info: Dict[str, Any]) -> None:
        self.session_manager.update_stage(session_id, stage_info)
        self.db_client.save_stage(session_id, stage_info)

    def get_current_stage(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.session_manager.get_current_stage(session_id)

    def update_learning_package(self, session_id: str, package: Dict[str, Any]) -> None:
        self.session_manager.update_learning_package(session_id, package)

    def get_learning_package(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.session_manager.get_learning_package(session_id)

    def add_qa_record(self, session_id: str, question: str, answer: str, tech_pass: bool = True, quality_pass: bool = True) -> None:
        self.session_manager.add_qa_record(session_id, question, answer, tech_pass, quality_pass)
        self.db_client.save_qa_record(session_id, question, answer, tech_pass, quality_pass)

    def get_qa_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.session_manager.get_qa_history(session_id)

    def add_answer_record(self, session_id: str, question_id: str, user_answer: str, is_correct: bool, time_spent: int = 0) -> None:
        self.session_manager.add_answer_record(session_id, question_id, user_answer, is_correct, time_spent)
        self.db_client.save_answer_record(session_id, question_id, user_answer, is_correct, time_spent)

    def calculate_accuracy(self, session_id: str) -> float:
        answers = self.session_manager.get_answer_records(session_id)
        if not answers:
            return 0.0
        correct = sum(1 for a in answers if a["is_correct"])
        return correct / len(answers)

    def get_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        return self.session_manager.get_interactions(session_id)

    def end_session(self, session_id: str) -> None:
        self.session_manager.end_session(session_id)
        self.db_client.update_session_status(session_id, "completed")

    def abandon_session(self, session_id: str) -> None:
        self.session_manager.end_session(session_id)
        self.db_client.update_session_status(session_id, "abandoned")
