# -*- coding: utf-8 -*-
#SQLite 读写（9张表）
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from config import BACKEND_DIR, settings
from pathlib import Path
import json
import uuid

class SQLiteClient:
    def __init__(self, database_url: str = None):
        self.database_url = self._resolve_database_url(database_url or settings.DATABASE_URL)
        self._ensure_data_dir()
        self.engine = create_engine(self.database_url, connect_args={"check_same_thread": False})
        event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.Session = sessionmaker(bind=self.engine)
        self._init_tables()
        self._migrate_tables()

    def close(self) -> None:
        self.engine.dispose()

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    def _resolve_database_url(self, database_url: str) -> str:
        if not database_url.startswith("sqlite:///"):
            return database_url
        raw_path = database_url.replace("sqlite:///", "", 1)
        if raw_path == ":memory:":
            return database_url
        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = (BACKEND_DIR / db_path).resolve()
        return f"sqlite:///{db_path.as_posix()}"

    def _ensure_data_dir(self):
        db_url = self.database_url
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_tables(self):
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(50) PRIMARY KEY,
                    current_stage INT DEFAULT 1,
                    status VARCHAR(20) CHECK (status IN ('active', 'completed', 'abandoned')) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    profile_id VARCHAR(50) PRIMARY KEY,
                    session_id VARCHAR(50) NOT NULL,
                    version INT DEFAULT 1,
                    skill_tags TEXT,
                    knowledge_gaps TEXT,
                    goal TEXT,
                    daily_hours FLOAT,
                    learning_style VARCHAR(20),
                    learning_rhythm VARCHAR(20),
                    confidence_index FLOAT,
                    confusion_markers TEXT,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stage_progress (
                    stage_id VARCHAR(50) PRIMARY KEY,
                    session_id VARCHAR(50) NOT NULL,
                    stage_num INT NOT NULL,
                    block_id VARCHAR(50),
                    block_name VARCHAR(100),
                    subtopic_id VARCHAR(50),
                    status VARCHAR(20) CHECK (status IN ('pending', 'learning', 'tested', 'completed', 'retreat')) DEFAULT 'pending',
                    accuracy FLOAT,
                    strategy_used VARCHAR(20),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_qa (
                    qa_id VARCHAR(50) PRIMARY KEY,
                    session_id VARCHAR(50) NOT NULL,
                    stage INT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT,
                    tech_pass BOOLEAN,
                    quality_pass BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_answers (
                    answer_id VARCHAR(50) PRIMARY KEY,
                    session_id VARCHAR(50) NOT NULL,
                    stage INT NOT NULL,
                    question_id VARCHAR(50) NOT NULL,
                    user_answer TEXT,
                    is_correct BOOLEAN,
                    time_spent INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS knowledge_blocks (
                    block_id VARCHAR(50) PRIMARY KEY,
                    block_name VARCHAR(100) NOT NULL,
                    description TEXT,
                    estimated_hours FLOAT DEFAULT 1.0,
                    difficulty VARCHAR(20) CHECK (difficulty IN ('初级', '中级', '高级')),
                    prerequisites TEXT,
                    corresponding_gaps TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS knowledge_docs (
                    doc_id VARCHAR(50) PRIMARY KEY,
                    block_id VARCHAR(50) NOT NULL,
                    title VARCHAR(200) NOT NULL,
                    content TEXT NOT NULL,
                    doc_type VARCHAR(20) CHECK (doc_type IN ('tutorial', 'code', 'spec', 'best_practice', 'pitfall')),
                    source VARCHAR(200),
                    difficulty VARCHAR(20) CHECK (difficulty IN ('初级', '中级', '高级')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS knowledge_subtopics (
                    subtopic_id VARCHAR(50) PRIMARY KEY,
                    block_id VARCHAR(50) NOT NULL,
                    subtopic_name VARCHAR(200) NOT NULL,
                    estimated_hours FLOAT DEFAULT 1.0,
                    sequence_index INT NOT NULL,
                    difficulty VARCHAR(20) CHECK (difficulty IN ('初级', '中级', '高级')),
                    prerequisites TEXT,
                    content_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id VARCHAR(50) PRIMARY KEY,
                    doc_id VARCHAR(50) NOT NULL,
                    block_id VARCHAR(50) NOT NULL,
                    chunk_index INT,
                    content TEXT NOT NULL,
                    chunk_type VARCHAR(20) CHECK (chunk_type IN ('text', 'code', 'table')),
                    char_start INT,
                    char_end INT,
                    FOREIGN KEY (doc_id) REFERENCES knowledge_docs(doc_id),
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS subtopic_status (
                    session_id VARCHAR(50) NOT NULL,
                    subtopic_id VARCHAR(50) NOT NULL,
                    block_id VARCHAR(50) NOT NULL,
                    status INT NOT NULL DEFAULT -1 CHECK (status IN (-1, 0, 1)),
                    mastery_score FLOAT DEFAULT 0.0,
                    last_accuracy FLOAT,
                    attempt_count INT DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    PRIMARY KEY (session_id, subtopic_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (subtopic_id) REFERENCES knowledge_subtopics(subtopic_id),
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_questions (
                    question_id VARCHAR(50) PRIMARY KEY,
                    block_id VARCHAR(50) NOT NULL,
                    difficulty VARCHAR(20) CHECK (difficulty IN ('初级', '中级', '高级')),
                    question_type VARCHAR(20) CHECK (question_type IN ('选择', '填空', '判断', '简答')),
                    question_text TEXT NOT NULL,
                    options TEXT,
                    correct_answer TEXT NOT NULL,
                    hint TEXT,
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS faq_items (
                    faq_id VARCHAR(50) PRIMARY KEY,
                    block_id VARCHAR(50) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty VARCHAR(20) CHECK (difficulty IN ('初级', '中级', '高级')),
                    tags TEXT,
                    views INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS session_states (
                    session_id VARCHAR(50) PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """))
            conn.commit()

    def _migrate_tables(self):
        with self.engine.connect() as conn:
            self._ensure_column(conn, "user_profiles", "user_id", "VARCHAR(50)")
            self._ensure_column(conn, "user_profiles", "grade", "VARCHAR(50)")
            self._ensure_column(conn, "user_profiles", "hardware", "VARCHAR(100)")
            self._ensure_column(conn, "user_profiles", "profile_confidence", "FLOAT")
            self._ensure_column(conn, "user_profiles", "last_updated", "TIMESTAMP")
            self._ensure_column(conn, "stage_progress", "subtopic_id", "VARCHAR(50)")
            self._ensure_column(conn, "knowledge_docs", "subtopic_id", "VARCHAR(50)")
            conn.commit()

    @staticmethod
    def _ensure_column(conn, table_name: str, column_name: str, column_sql: str) -> None:
        columns = {
            row[1]
            for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        }
        if column_name in columns:
            return
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))

    def create_session(self, session_id: str) -> None:
        with self.Session() as session:
            session.execute(text("INSERT OR IGNORE INTO sessions (session_id) VALUES (:session_id)"),
                           {"session_id": session_id})
            session.commit()
        self.init_subtopic_status(session_id)

    def session_exists(self, session_id: str) -> bool:
        with self.Session() as session:
            row = session.execute(
                text("SELECT 1 FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).first()
            return row is not None

    def save_session_state(self, session_id: str, state: dict) -> None:
        payload = json.dumps(state, ensure_ascii=False)
        with self.Session() as session:
            session.execute(text("""
                INSERT INTO session_states (session_id, state_json, updated_at)
                VALUES (:session_id, :state_json, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
            """), {"session_id": session_id, "state_json": payload})
            session.execute(text("""
                UPDATE sessions
                SET current_stage = :current_stage,
                    status = :status,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = :session_id
            """), {
                "current_stage": state.get("current_stage", 1),
                "status": state.get("status", "active"),
                "session_id": session_id,
            })
            session.commit()

    def load_session_state(self, session_id: str) -> dict | None:
        with self.Session() as session:
            row = session.execute(text("""
                SELECT state_json
                FROM session_states
                WHERE session_id = :session_id
            """), {"session_id": session_id}).first()
            return json.loads(row[0]) if row else None

    def get_question(self, question_id: str) -> dict | None:
        with self.Session() as session:
            row = session.execute(text("""
                SELECT question_id, block_id, difficulty, question_type,
                       question_text, options, correct_answer, hint
                FROM test_questions
                WHERE question_id = :question_id
            """), {"question_id": question_id}).mappings().first()
            if not row:
                return None
            result = dict(row)
            result["options"] = json.loads(result["options"]) if result["options"] else None
            return result

    def update_session_status(self, session_id: str, status: str) -> None:
        with self.Session() as session:
            session.execute(text("UPDATE sessions SET status = :status, updated_at = CURRENT_TIMESTAMP WHERE session_id = :session_id"),
                           {"status": status, "session_id": session_id})
            session.commit()

    def save_profile(self, session_id: str, profile: dict) -> None:
        with self.Session() as session:
            profile_id = str(uuid.uuid4())
            version = session.execute(text("""
                SELECT COALESCE(MAX(version), 0) + 1
                FROM user_profiles
                WHERE session_id = :session_id
            """), {"session_id": session_id}).scalar_one()
            session.execute(text("""
                INSERT INTO user_profiles (
                    profile_id, session_id, user_id, version, grade, goal,
                    daily_hours, learning_style, hardware, skill_tags, knowledge_gaps,
                    learning_rhythm, confidence_index, confusion_markers,
                    confidence, profile_confidence, last_updated
                ) VALUES (
                    :profile_id, :session_id, :user_id, :version, :grade, :goal,
                    :daily_hours, :learning_style, :hardware, :skill_tags, :knowledge_gaps,
                    :learning_rhythm, :confidence_index, :confusion_markers,
                    :confidence, :profile_confidence, CURRENT_TIMESTAMP
                )
            """), {
                "profile_id": profile_id,
                "session_id": session_id,
                "user_id": profile.get("user_id", session_id),
                "version": version,
                "grade": profile.get("grade", ""),
                "hardware": profile.get("hardware", ""),
                "skill_tags": json.dumps(profile.get("skill_tags", [])),
                "knowledge_gaps": json.dumps(profile.get("knowledge_gaps", [])),
                "goal": profile.get("goal", ""),
                "daily_hours": profile.get("daily_hours", 2.0),
                "learning_style": profile.get("learning_style", "mixed"),
                "learning_rhythm": profile.get("learning_rhythm", "medium"),
                "confidence_index": profile.get("confidence_index", 0.5),
                "confusion_markers": json.dumps(profile.get("confusion_markers", [])),
                "confidence": profile.get("confidence", 0.7),
                "profile_confidence": profile.get("profile_confidence", profile.get("confidence", 0.7)),
            })
            session.commit()

    def save_stage(self, session_id: str, stage_info: dict) -> None:
        with self.Session() as session:
            session.execute(text("""
                INSERT OR REPLACE INTO stage_progress (
                    stage_id, session_id, stage_num, block_id, block_name, subtopic_id, status, accuracy, strategy_used
                ) VALUES (:stage_id, :session_id, :stage_num, :block_id, :block_name, :subtopic_id, :status, :accuracy, :strategy_used)
            """), {
                "stage_id": stage_info.get("stage_id", str(uuid.uuid4())),
                "session_id": session_id,
                "stage_num": stage_info.get("stage_num", 1),
                "block_id": stage_info.get("block_id", ""),
                "block_name": stage_info.get("block_name", ""),
                "subtopic_id": stage_info.get("subtopic_id"),
                "status": stage_info.get("status", "pending"),
                "accuracy": stage_info.get("accuracy"),
                "strategy_used": stage_info.get("strategy_used")
            })
            session.commit()

    def init_subtopic_status(self, session_id: str) -> None:
        with self.Session() as session:
            existing = session.execute(text("""
                SELECT 1
                FROM subtopic_status
                WHERE session_id = :session_id
                LIMIT 1
            """), {"session_id": session_id}).first()
            if existing:
                return
            rows = session.execute(text("""
                SELECT subtopic_id, block_id
                FROM knowledge_subtopics
            """)).fetchall()
            for subtopic_id, block_id in rows:
                session.execute(text("""
                    INSERT OR IGNORE INTO subtopic_status (
                        session_id, subtopic_id, block_id, status, mastery_score, updated_at
                    ) VALUES (
                        :session_id, :subtopic_id, :block_id, -1, 0.0, CURRENT_TIMESTAMP
                    )
                """), {
                    "session_id": session_id,
                    "subtopic_id": subtopic_id,
                    "block_id": block_id,
                })
            session.commit()

    def ensure_subtopic_status(self, session_id: str) -> None:
        self.init_subtopic_status(session_id)

    def get_subtopic_status(self, session_id: str, subtopic_id: str) -> dict | None:
        self.ensure_subtopic_status(session_id)
        with self.Session() as session:
            row = session.execute(text("""
                SELECT session_id, subtopic_id, block_id, status, mastery_score,
                       last_accuracy, attempt_count, updated_at, started_at, completed_at
                FROM subtopic_status
                WHERE session_id = :session_id
                  AND subtopic_id = :subtopic_id
            """), {
                "session_id": session_id,
                "subtopic_id": subtopic_id,
            }).mappings().first()
        return dict(row) if row else None

    def set_subtopic_status(
        self,
        session_id: str,
        subtopic_id: str,
        status: int,
        mastery_score: float | None = None,
        last_accuracy: float | None = None,
        increment_attempt: bool = False,
    ) -> None:
        self.ensure_subtopic_status(session_id)
        existing = self.get_subtopic_status(session_id, subtopic_id)
        if not existing:
            return
        attempt_count = int(existing.get("attempt_count") or 0) + (1 if increment_attempt else 0)
        with self.Session() as session:
            session.execute(text("""
                UPDATE subtopic_status
                SET status = :status,
                    mastery_score = COALESCE(:mastery_score, mastery_score),
                    last_accuracy = COALESCE(:last_accuracy, last_accuracy),
                    attempt_count = :attempt_count,
                    started_at = CASE
                        WHEN :status = 0 AND started_at IS NULL THEN CURRENT_TIMESTAMP
                        ELSE started_at
                    END,
                    completed_at = CASE
                        WHEN :status = 1 THEN CURRENT_TIMESTAMP
                        ELSE completed_at
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = :session_id
                  AND subtopic_id = :subtopic_id
            """), {
                "session_id": session_id,
                "subtopic_id": subtopic_id,
                "status": status,
                "mastery_score": mastery_score,
                "last_accuracy": last_accuracy,
                "attempt_count": attempt_count,
            })
            session.commit()

    def get_completed_subtopic_ids(self, session_id: str, block_id: str) -> list[str]:
        self.ensure_subtopic_status(session_id)
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT subtopic_id
                FROM subtopic_status
                WHERE session_id = :session_id
                  AND block_id = :block_id
                  AND status = 1
            """), {
                "session_id": session_id,
                "block_id": block_id,
            }).fetchall()
        return [row[0] for row in rows if row and row[0]]

    def list_subtopic_statuses(self, session_id: str) -> list[dict]:
        self.ensure_subtopic_status(session_id)
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT s.subtopic_id, s.block_id, s.status, s.mastery_score, s.last_accuracy,
                       s.attempt_count, k.subtopic_name, k.estimated_hours, k.sequence_index,
                       k.difficulty, k.prerequisites, k.content_summary
                FROM subtopic_status s
                JOIN knowledge_subtopics k ON s.subtopic_id = k.subtopic_id
                WHERE s.session_id = :session_id
                ORDER BY k.block_id ASC, k.sequence_index ASC
            """), {"session_id": session_id}).mappings().all()
        return [dict(row) for row in rows]

    def list_learnable_subtopics(
        self,
        session_id: str,
        preferred_block_id: str | None = None,
        max_hours: float | None = None,
    ) -> list[dict]:
        rows = self.list_subtopic_statuses(session_id)
        status_index = {row["subtopic_id"]: int(row["status"]) for row in rows}
        candidates: list[dict] = []
        for row in rows:
            if int(row["status"]) != -1:
                continue
            prerequisites = json.loads(row["prerequisites"]) if row.get("prerequisites") else []
            if all(status_index.get(prerequisite_id) == 1 for prerequisite_id in prerequisites):
                row["prerequisites"] = prerequisites
                candidates.append(row)

        def sort_key(item: dict) -> tuple:
            hours = float(item.get("estimated_hours") or 0)
            return (
                0 if preferred_block_id and item["block_id"] == preferred_block_id else 1,
                0 if max_hours is not None and hours <= max_hours else 1,
                hours if max_hours is not None else 0,
                int(item.get("sequence_index") or 0),
                item["subtopic_id"],
            )

        return sorted(candidates, key=sort_key)

    def get_next_subtopic(
        self,
        session_id: str,
        block_id: str,
        max_hours: float | None = None,
        additional_completed: list[str] | None = None,
    ) -> dict | None:
        completed = set(self.get_completed_subtopic_ids(session_id, block_id))
        completed.update(item for item in (additional_completed or []) if item)
        rows = [
            row for row in self.list_learnable_subtopics(
                session_id=session_id,
                preferred_block_id=block_id,
                max_hours=max_hours,
            )
            if row["block_id"] == block_id and row["subtopic_id"] not in completed
        ]
        return rows[0] if rows else None

    def get_block_status_summary(self, session_id: str, block_id: str) -> dict | None:
        rows = [row for row in self.list_subtopic_statuses(session_id) if row["block_id"] == block_id]
        if not rows:
            return None
        total = len(rows)
        mastered = sum(1 for row in rows if int(row["status"]) == 1)
        learning = sum(1 for row in rows if int(row["status"]) == 0)
        progress = mastered / total if total else 0.0
        if mastered == total:
            status = "已掌握"
        elif learning > 0 or mastered > 0:
            status = "学习中"
        else:
            status = "未掌握"
        return {
            "block_id": block_id,
            "status": status,
            "progress": round(progress, 3),
            "mastered_subtopics": mastered,
            "total_subtopics": total,
        }

    def get_subtopic_completion_ratio(
        self,
        session_id: str,
        block_id: str,
        additional_completed: list[str] | None = None,
    ) -> tuple[float, int, int]:
        completed = set(self.get_completed_subtopic_ids(session_id, block_id))
        completed.update(item for item in (additional_completed or []) if item)
        with self.Session() as session:
            total = session.execute(text("""
                SELECT COUNT(*)
                FROM knowledge_subtopics
                WHERE block_id = :block_id
            """), {"block_id": block_id}).scalar_one()
        if total <= 0:
            return 1.0, 0, 0
        completed_count = min(len(completed), total)
        return completed_count / total, completed_count, total

    def save_qa_record(self, session_id: str, question: str, answer: str, tech_pass: bool, quality_pass: bool, stage: int = 1) -> None:
        with self.Session() as session:
            qa_id = str(uuid.uuid4())
            session.execute(text("""
                INSERT INTO user_qa (qa_id, session_id, stage, question, answer, tech_pass, quality_pass)
                VALUES (:qa_id, :session_id, :stage, :question, :answer, :tech_pass, :quality_pass)
            """), {
                "qa_id": qa_id,
                "session_id": session_id,
                "stage": stage,
                "question": question,
                "answer": answer,
                "tech_pass": tech_pass,
                "quality_pass": quality_pass
            })
            session.commit()

    def save_answer_record(self, session_id: str, question_id: str, user_answer: str, is_correct: bool, time_spent: int, stage: int = 1) -> None:
        with self.Session() as session:
            answer_id = str(uuid.uuid4())
            session.execute(text("""
                INSERT INTO user_answers (answer_id, session_id, stage, question_id, user_answer, is_correct, time_spent)
                VALUES (:answer_id, :session_id, :stage, :question_id, :user_answer, :is_correct, :time_spent)
            """), {
                "answer_id": answer_id,
                "session_id": session_id,
                "stage": stage,
                "question_id": question_id,
                "user_answer": user_answer,
                "is_correct": is_correct,
                "time_spent": time_spent
            })
            session.commit()

    def search_faq(self, block_id: str = None, query: str = None, limit: int = 5) -> list[dict]:
        with self.Session() as session:
            params = {}
            if block_id and query:
                sql = """
                    SELECT faq_id, block_id, question, answer, difficulty, tags, views
                    FROM faq_items
                    WHERE block_id = :block_id AND (question LIKE :query OR answer LIKE :query)
                    ORDER BY views DESC, difficulty ASC
                    LIMIT :limit
                """
                params = {"block_id": block_id, "query": f"%{query}%", "limit": limit}
            elif block_id:
                sql = """
                    SELECT faq_id, block_id, question, answer, difficulty, tags, views
                    FROM faq_items
                    WHERE block_id = :block_id
                    ORDER BY views DESC, difficulty ASC
                    LIMIT :limit
                """
                params = {"block_id": block_id, "limit": limit}
            elif query:
                sql = """
                    SELECT faq_id, block_id, question, answer, difficulty, tags, views
                    FROM faq_items
                    WHERE question LIKE :query OR answer LIKE :query
                    ORDER BY views DESC, difficulty ASC
                    LIMIT :limit
                """
                params = {"query": f"%{query}%", "limit": limit}
            else:
                return []

            rows = session.execute(text(sql), params).mappings().all()
            results = []
            for row in rows:
                result = dict(row)
                result["tags"] = json.loads(result["tags"]) if result["tags"] else []
                results.append(result)
            return results

    def increment_faq_views(self, faq_id: str) -> None:
        with self.Session() as session:
            session.execute(text("""
                UPDATE faq_items
                SET views = views + 1
                WHERE faq_id = :faq_id
            """), {"faq_id": faq_id})
            session.commit()

    def get_faq_by_id(self, faq_id: str) -> dict | None:
        with self.Session() as session:
            row = session.execute(text("""
                SELECT faq_id, block_id, question, answer, difficulty, tags, views
                FROM faq_items
                WHERE faq_id = :faq_id
            """), {"faq_id": faq_id}).mappings().first()
            if not row:
                return None
            result = dict(row)
            result["tags"] = json.loads(result["tags"]) if result["tags"] else []
            return result
