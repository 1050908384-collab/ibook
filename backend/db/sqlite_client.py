# -*- coding: utf-8 -*-
#SQLite 读写（9张表）
from sqlalchemy import bindparam, create_engine, event, text
from sqlalchemy.orm import sessionmaker
from config import BACKEND_DIR, settings
from pathlib import Path
import json
import random
import uuid

DEFAULT_USER_ID = "local-user"

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
                    user_id VARCHAR(50) NOT NULL DEFAULT 'local-user',
                    goal TEXT DEFAULT '',
                    target_knowledge_gaps TEXT NOT NULL DEFAULT '[]',
                    target_total_hours FLOAT DEFAULT 0.0,
                    target_daily_hours FLOAT DEFAULT 2.0,
                    target_progress FLOAT DEFAULT 0.0,
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
                    estimated_hours FLOAT DEFAULT 1.0,
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
                    source_file_path TEXT NOT NULL DEFAULT '',
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
                    heading_path TEXT NOT NULL DEFAULT '',
                    heading_level INTEGER NOT NULL DEFAULT 0,
                    source_line_start INTEGER NOT NULL DEFAULT 0,
                    source_line_end INTEGER NOT NULL DEFAULT 0,
                    chunk_strategy TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (doc_id) REFERENCES knowledge_docs(doc_id),
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS learning_resources (
                    resource_id VARCHAR(50) PRIMARY KEY,
                    resource_type VARCHAR(20) CHECK (resource_type IN ('video', 'article')) NOT NULL,
                    title VARCHAR(300) NOT NULL,
                    raw_url TEXT NOT NULL,
                    canonical_url TEXT NOT NULL UNIQUE,
                    platform VARCHAR(50),
                    external_id VARCHAR(100),
                    external_part VARCHAR(30),
                    content_summary TEXT NOT NULL,
                    recommendation_level VARCHAR(20) CHECK (recommendation_level IN ('primary', 'supplementary')) NOT NULL DEFAULT 'supplementary',
                    review_status VARCHAR(20) CHECK (review_status IN ('pending', 'confirmed', 'rejected', 'unavailable')) NOT NULL DEFAULT 'pending',
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS subtopic_resources (
                    resource_id VARCHAR(50) NOT NULL,
                    block_id VARCHAR(50) NOT NULL,
                    subtopic_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) CHECK (role IN ('primary', 'supplementary')) NOT NULL DEFAULT 'supplementary',
                    sequence_index INT NOT NULL DEFAULT 1,
                    relevance_score FLOAT CHECK (relevance_score >= 0 AND relevance_score <= 1) NOT NULL DEFAULT 0.5,
                    mapping_status VARCHAR(20) CHECK (mapping_status IN ('pending', 'confirmed', 'rejected')) NOT NULL DEFAULT 'pending',
                    mapping_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (resource_id, subtopic_id),
                    FOREIGN KEY (resource_id) REFERENCES learning_resources(resource_id),
                    FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id),
                    FOREIGN KEY (subtopic_id) REFERENCES knowledge_subtopics(subtopic_id)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS block_resources (
                    resource_id VARCHAR(50) NOT NULL,
                    block_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) CHECK (role IN ('primary', 'supplementary')) NOT NULL DEFAULT 'supplementary',
                    sequence_index INT NOT NULL DEFAULT 1,
                    relevance_score FLOAT CHECK (relevance_score >= 0 AND relevance_score <= 1) NOT NULL DEFAULT 0.5,
                    mapping_status VARCHAR(20) CHECK (mapping_status IN ('pending', 'confirmed', 'rejected')) NOT NULL DEFAULT 'pending',
                    mapping_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (resource_id, block_id),
                    FOREIGN KEY (resource_id) REFERENCES learning_resources(resource_id),
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
                CREATE TABLE IF NOT EXISTS session_states (
                    session_id VARCHAR(50) PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """))
            self._create_user_scope_tables(conn)
            self._create_assessment_tables(conn)
            self._create_review_protocol_tables(conn)
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_learning_resources_type_status ON learning_resources(resource_type, review_status)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_subtopic_resources_lookup ON subtopic_resources(block_id, subtopic_id, role, sequence_index)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_block_resources_lookup ON block_resources(block_id, role, sequence_index)"))
            conn.commit()

    def _migrate_tables(self):
        with self.engine.connect() as conn:
            self._ensure_column(conn, "user_profiles", "user_id", "VARCHAR(50)")
            self._ensure_column(conn, "user_profiles", "grade", "VARCHAR(50)")
            self._ensure_column(conn, "user_profiles", "hardware", "VARCHAR(100)")
            self._ensure_column(conn, "user_profiles", "profile_confidence", "FLOAT")
            self._ensure_column(conn, "user_profiles", "last_updated", "TIMESTAMP")
            self._ensure_column(conn, "stage_progress", "subtopic_id", "VARCHAR(50)")
            self._ensure_column(conn, "stage_progress", "estimated_hours", "FLOAT DEFAULT 1.0")
            self._ensure_column(conn, "knowledge_docs", "subtopic_id", "VARCHAR(50)")
            # 标题级分块新增字段（见《知识库分块新增字段说明.md》）
            self._ensure_column(conn, "knowledge_docs", "source_file_path", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "knowledge_chunks", "heading_path", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "knowledge_chunks", "heading_level", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "knowledge_chunks", "source_line_start", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "knowledge_chunks", "source_line_end", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "knowledge_chunks", "chunk_strategy", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "sessions", "user_id", f"VARCHAR(50) DEFAULT '{DEFAULT_USER_ID}'")
            self._ensure_column(conn, "sessions", "goal", "TEXT")
            self._ensure_column(conn, "sessions", "target_knowledge_gaps", "TEXT DEFAULT '[]'")
            self._ensure_column(conn, "sessions", "target_total_hours", "FLOAT DEFAULT 0.0")
            self._ensure_column(conn, "sessions", "target_daily_hours", "FLOAT DEFAULT 2.0")
            self._ensure_column(conn, "sessions", "target_progress", "FLOAT DEFAULT 0.0")
            self._create_user_scope_tables(conn)
            self._ensure_column(conn, "learner_profiles", "foundation_confirmed", "BOOLEAN NOT NULL DEFAULT 0")
            # 结构化设备配置：型号→显存/架构自动补全后落库（配合 device_spec 查表）。
            self._ensure_column(conn, "learner_profiles", "device_type", "VARCHAR(20) DEFAULT ''")
            self._ensure_column(conn, "learner_profiles", "gpu_model", "VARCHAR(100) DEFAULT ''")
            self._ensure_column(conn, "learner_profiles", "vram_gb", "INTEGER")
            self._ensure_column(conn, "learner_profiles", "gpu_architecture", "VARCHAR(50) DEFAULT ''")
            self._migrate_user_scope_data(conn)
            self._migrate_test_questions_table(conn)
            self._create_assessment_tables(conn)
            self._migrate_review_protocol_tables(conn)
            self._migrate_review_protocol_foreign_keys(conn)
            self._migrate_learning_package_source_types(conn)
            self._repair_legacy_session_references(conn)
            self._create_knowledge_chunks_fts(conn)
            conn.commit()

    @staticmethod
    def _create_knowledge_chunks_fts(conn) -> None:
        """创建并维护 Agent3 技术问答使用的 FTS5/BM25 索引。"""
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                block_id UNINDEXED,
                subtopic_id UNINDEXED,
                content,
                tokenize='trigram'
            )
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS knowledge_chunks_fts_insert
            AFTER INSERT ON knowledge_chunks BEGIN
                INSERT INTO knowledge_chunks_fts(chunk_id, block_id, subtopic_id, content)
                SELECT NEW.chunk_id, NEW.block_id, COALESCE(d.subtopic_id, ''), NEW.content
                FROM knowledge_docs d WHERE d.doc_id = NEW.doc_id;
            END
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS knowledge_chunks_fts_delete
            AFTER DELETE ON knowledge_chunks BEGIN
                DELETE FROM knowledge_chunks_fts WHERE chunk_id = OLD.chunk_id;
            END
        """))
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS knowledge_chunks_fts_update
            AFTER UPDATE OF chunk_id, doc_id, block_id, content ON knowledge_chunks BEGIN
                DELETE FROM knowledge_chunks_fts WHERE chunk_id = OLD.chunk_id;
                INSERT INTO knowledge_chunks_fts(chunk_id, block_id, subtopic_id, content)
                SELECT NEW.chunk_id, NEW.block_id, COALESCE(d.subtopic_id, ''), NEW.content
                FROM knowledge_docs d WHERE d.doc_id = NEW.doc_id;
            END
        """))
        fts_count = conn.execute(text("SELECT COUNT(*) FROM knowledge_chunks_fts")).scalar_one()
        chunk_count = conn.execute(text("SELECT COUNT(*) FROM knowledge_chunks")).scalar_one()
        if fts_count != chunk_count:
            conn.execute(text("DELETE FROM knowledge_chunks_fts"))
            conn.execute(text("""
                INSERT INTO knowledge_chunks_fts(chunk_id, block_id, subtopic_id, content)
                SELECT c.chunk_id, c.block_id, COALESCE(d.subtopic_id, ''), c.content
                FROM knowledge_chunks c
                LEFT JOIN knowledge_docs d ON d.doc_id = c.doc_id
            """))

    def _create_user_scope_tables(self, conn) -> None:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                migration_id VARCHAR(100) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS learner_profiles (
                user_id VARCHAR(50) PRIMARY KEY,
                grade VARCHAR(100) DEFAULT '',
                skill_tags TEXT NOT NULL DEFAULT '[]',
                foundation_confirmed BOOLEAN NOT NULL DEFAULT 0,
                learning_style VARCHAR(20) DEFAULT 'mixed',
                hardware VARCHAR(200) DEFAULT '',
                device_type VARCHAR(20) DEFAULT '',
                gpu_model VARCHAR(100) DEFAULT '',
                vram_gb INTEGER,
                gpu_architecture VARCHAR(50) DEFAULT '',
                default_daily_hours FLOAT DEFAULT 2.0,
                learning_rhythm VARCHAR(20) DEFAULT 'medium',
                confidence_index FLOAT DEFAULT 0.5,
                confusion_markers TEXT NOT NULL DEFAULT '[]',
                confidence FLOAT DEFAULT 0.7,
                version INT DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_subtopic_status (
                user_id VARCHAR(50) NOT NULL,
                subtopic_id VARCHAR(50) NOT NULL,
                block_id VARCHAR(50) NOT NULL,
                status INT NOT NULL DEFAULT -1 CHECK (status IN (-1, 0, 1)),
                mastery_score FLOAT NOT NULL DEFAULT 0.0,
                is_initial_foundation BOOLEAN NOT NULL DEFAULT 0,
                attempt_count INT NOT NULL DEFAULT 0,
                last_accuracy FLOAT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                PRIMARY KEY (user_id, subtopic_id),
                FOREIGN KEY (subtopic_id) REFERENCES knowledge_subtopics(subtopic_id),
                FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id)
            )
        """))
        if "user_id" in self._get_table_columns(conn, "sessions"):
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_subtopic_block ON user_subtopic_status(user_id, block_id, status)"))

    def _migrate_user_scope_data(self, conn) -> None:
        """Idempotently move legacy session-scoped learner data to local-user."""
        migration_id = "2026-08-user-scope-v1"
        applied = conn.execute(
            text("SELECT 1 FROM schema_migrations WHERE migration_id = :migration_id"),
            {"migration_id": migration_id},
        ).first()
        if applied:
            return
        conn.execute(text("UPDATE sessions SET user_id = :user_id WHERE user_id IS NULL OR user_id = ''"), {"user_id": DEFAULT_USER_ID})
        legacy_profile = conn.execute(text("""
            SELECT grade, skill_tags, learning_style, hardware, daily_hours,
                   learning_rhythm, confidence_index, confusion_markers,
                   confidence, version
            FROM user_profiles
            ORDER BY created_at DESC, version DESC
            LIMIT 1
        """)).mappings().first()
        if legacy_profile:
            conn.execute(text("""
                INSERT OR IGNORE INTO learner_profiles (
                    user_id, grade, skill_tags, learning_style, hardware,
                    default_daily_hours, learning_rhythm, confidence_index,
                    confusion_markers, confidence, version
                ) VALUES (
                    :user_id, :grade, :skill_tags, :learning_style, :hardware,
                    :daily_hours, :learning_rhythm, :confidence_index,
                    :confusion_markers, :confidence, :version
                )
            """), {"user_id": DEFAULT_USER_ID, **dict(legacy_profile)})

        conn.execute(text("""
            UPDATE sessions
            SET goal = COALESCE(NULLIF(goal, ''), (
                    SELECT up.goal FROM user_profiles up
                    WHERE up.session_id = sessions.session_id
                    ORDER BY up.version DESC LIMIT 1
                ), ''),
                target_knowledge_gaps = CASE
                    WHEN target_knowledge_gaps IS NULL OR target_knowledge_gaps = '[]' THEN COALESCE((
                        SELECT up.knowledge_gaps FROM user_profiles up
                        WHERE up.session_id = sessions.session_id
                        ORDER BY up.version DESC LIMIT 1
                    ), '[]')
                    ELSE target_knowledge_gaps
                END,
                target_daily_hours = COALESCE((
                    SELECT up.daily_hours FROM user_profiles up
                    WHERE up.session_id = sessions.session_id
                    ORDER BY up.version DESC LIMIT 1
                ), target_daily_hours, 2.0)
        """))
        conn.execute(text("""
            INSERT INTO user_subtopic_status (
                user_id, subtopic_id, block_id, status, mastery_score,
                is_initial_foundation, attempt_count, last_accuracy,
                updated_at, started_at, completed_at
            )
            SELECT :user_id, ss.subtopic_id, ss.block_id,
                   MAX(ss.status), MAX(ss.mastery_score), 0,
                   SUM(COALESCE(ss.attempt_count, 0)), MAX(ss.last_accuracy),
                   MAX(ss.updated_at), MIN(ss.started_at), MAX(ss.completed_at)
            FROM subtopic_status ss
            GROUP BY ss.subtopic_id, ss.block_id
            ON CONFLICT(user_id, subtopic_id) DO UPDATE SET
                status = MAX(user_subtopic_status.status, excluded.status),
                mastery_score = MAX(user_subtopic_status.mastery_score, excluded.mastery_score),
                attempt_count = MAX(user_subtopic_status.attempt_count, excluded.attempt_count),
                last_accuracy = COALESCE(excluded.last_accuracy, user_subtopic_status.last_accuracy),
                updated_at = CURRENT_TIMESTAMP
        """), {"user_id": DEFAULT_USER_ID})
        state_rows = conn.execute(text("SELECT session_id, state_json FROM session_states")).fetchall()
        for session_id, state_json in state_rows:
            try:
                payload = json.loads(state_json)
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("profile") is None:
                continue
            payload["profile"] = None
            conn.execute(text("""
                UPDATE session_states SET state_json = :state_json
                WHERE session_id = :session_id
            """), {
                "session_id": session_id,
                "state_json": json.dumps(payload, ensure_ascii=False),
            })
        conn.execute(
            text("INSERT INTO schema_migrations (migration_id) VALUES (:migration_id)"),
            {"migration_id": migration_id},
        )

    def _migrate_learning_package_source_types(self, conn) -> None:
        """安全扩展 learning_package_sources 的 CHECK 枚举以支持 video。"""
        table_sql = conn.execute(text("SELECT sql FROM sqlite_master WHERE type='table' AND name='learning_package_sources'")).scalar() or ""
        if "'video'" in table_sql:
            return

        tables = [
            "learning_package_sources", "learning_package_claims",
            "review_debate_issues", "review_debate_responses",
        ]
        snapshots = {}
        for table in tables:
            columns = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]
            if columns:
                snapshots[table] = (columns, [dict(row._mapping) for row in conn.execute(text(f"SELECT * FROM {table}")).fetchall()])

        conn.execute(text("PRAGMA foreign_keys = OFF"))
        for table in ("review_debate_responses", "review_debate_issues", "learning_package_claims", "learning_package_sources"):
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        self._create_review_protocol_tables(conn)
        for table in tables:
            columns, rows = snapshots.get(table, ([], []))
            if not rows:
                continue
            column_sql = ", ".join(columns)
            value_sql = ", ".join(f":{column}" for column in columns)
            conn.execute(text(f"INSERT INTO {table} ({column_sql}) VALUES ({value_sql})"), rows)
        conn.execute(text("PRAGMA foreign_keys = ON"))

    def _migrate_review_protocol_foreign_keys(self, conn) -> None:
        """Rebuild only legacy review tables whose composite source FK is stale.

        SQLite cannot alter foreign-key definitions in place.  Older databases
        referenced a temporary ``learning_package_sources_legacy`` table, so
        preserve all review rows while recreating the small review-table group
        with the current schema.
        """
        if not self._table_exists(conn, "learning_package_claims"):
            return

        foreign_keys = conn.execute(
            text("PRAGMA foreign_key_list(learning_package_claims)")
        ).fetchall()
        grouped: dict[int, list[tuple]] = {}
        for row in foreign_keys:
            grouped.setdefault(row[0], []).append(tuple(row))
        has_current_source_fk = any(
            rows[0][2] == "learning_package_sources"
            and [item[3] for item in sorted(rows, key=lambda item: item[1])] == ["trace_id", "source_ref"]
            and [item[4] for item in sorted(rows, key=lambda item: item[1])] == ["trace_id", "source_id"]
            for rows in grouped.values()
        )
        if has_current_source_fk:
            return

        tables = [
            "learning_package_sources",
            "learning_package_claims",
            "review_debate_issues",
            "review_debate_rounds",
            "review_debate_responses",
        ]
        snapshots: dict[str, tuple[list[str], list[dict]]] = {}
        for table in tables:
            if not self._table_exists(conn, table):
                continue
            columns = [row[1] for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]
            rows = [dict(row._mapping) for row in conn.execute(text(f"SELECT * FROM {table}")).fetchall()]
            snapshots[table] = (columns, rows)

        # PRAGMA foreign_keys can only change outside an active transaction.
        conn.commit()
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        try:
            for table in reversed(tables):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            self._create_review_protocol_tables(conn)
            for table in tables:
                columns, rows = snapshots.get(table, ([], []))
                if not rows:
                    continue
                column_sql = ", ".join(columns)
                value_sql = ", ".join(f":{column}" for column in columns)
                conn.execute(text(f"INSERT INTO {table} ({column_sql}) VALUES ({value_sql})"), rows)
            conn.commit()
        finally:
            conn.execute(text("PRAGMA foreign_keys = ON"))

    def _repair_legacy_session_references(self, conn) -> None:
        """Retain legacy learning records by recreating their missing sessions."""
        tables = ["user_profiles", "stage_progress", "user_qa", "user_answers", "session_states"]
        for table in tables:
            if not self._table_exists(conn, table):
                continue
            has_session_id = any(
                row[1] == "session_id"
                for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            )
            if not has_session_id:
                continue
            conn.execute(text(f"""
                INSERT OR IGNORE INTO sessions (session_id, user_id)
                SELECT DISTINCT legacy.session_id, :user_id
                FROM {table} AS legacy
                LEFT JOIN sessions AS current ON current.session_id = legacy.session_id
                WHERE legacy.session_id IS NOT NULL AND current.session_id IS NULL
            """), {"user_id": DEFAULT_USER_ID})


    @staticmethod
    def _ensure_column(conn, table_name: str, column_name: str, column_sql: str) -> None:
        columns = {
            row[1]
            for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        }
        if column_name in columns:
            return
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))

    def _migrate_test_questions_table(self, conn) -> None:
        columns = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(test_questions)")).fetchall()
        }
        expected_columns = {
            "question_id", "block_id", "subtopic_id", "difficulty", "question_type",
            "question_text", "options", "correct_answer", "explanation",
            "expected_output", "grading_mode", "source", "hint"
        }
        if columns == expected_columns:
            return

        conn.execute(text("ALTER TABLE test_questions RENAME TO test_questions_legacy"))
        conn.execute(text("""
            CREATE TABLE test_questions (
                question_id VARCHAR(50) PRIMARY KEY,
                block_id VARCHAR(50) NOT NULL,
                subtopic_id VARCHAR(50),
                difficulty VARCHAR(20),
                question_type VARCHAR(50),
                question_text TEXT NOT NULL,
                options TEXT,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                expected_output TEXT,
                grading_mode VARCHAR(50),
                source VARCHAR(200),
                hint TEXT,
                FOREIGN KEY (block_id) REFERENCES knowledge_blocks(block_id),
                FOREIGN KEY (subtopic_id) REFERENCES knowledge_subtopics(subtopic_id)
            )
        """))
        conn.execute(text("""
            INSERT INTO test_questions (
                question_id, block_id, subtopic_id, difficulty, question_type,
                question_text, options, correct_answer, explanation,
                expected_output, grading_mode, source, hint
            )
            SELECT
                question_id,
                block_id,
                NULL,
                difficulty,
                question_type,
                question_text,
                options,
                correct_answer,
                hint,
                NULL,
                'exact_match',
                'legacy_seed',
                hint
            FROM test_questions_legacy
        """))
        conn.execute(text("DROP TABLE test_questions_legacy"))

    def _create_assessment_tables(self, conn) -> None:
        """测试服务（AssessmentService）三张表（Agent46_API接口任务说明.md 第 2 节）。

        - assessment_attempts：一次测试的生命周期（每次独立 attempt_id、状态、正确率）。
        - assessment_attempt_questions：测试开始时冻结的题目快照（正确答案/解析随快照保存，
          用于提交后统一判题与公布，不再依赖 test_questions 的后续变更）。
        - assessment_answers：每题只保存一条最终答案（PRIMARY KEY 保证覆盖而非追加）。
        """
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assessment_attempts (
                attempt_id VARCHAR(50) PRIMARY KEY,
                session_id VARCHAR(50) NOT NULL,
                stage_id VARCHAR(50),
                stage_num INT,
                block_id VARCHAR(50),
                subtopic_id VARCHAR(50),
                status VARCHAR(20) CHECK (status IN ('in_progress', 'submitted')) NOT NULL DEFAULT 'in_progress',
                total_questions INT NOT NULL DEFAULT 0,
                answered_count INT NOT NULL DEFAULT 0,
                correct_count INT,
                accuracy FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                submitted_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assessment_attempt_questions (
                attempt_id VARCHAR(50) NOT NULL,
                question_id VARCHAR(50) NOT NULL,
                sequence_index INT NOT NULL DEFAULT 0,
                block_id VARCHAR(50),
                subtopic_id VARCHAR(50),
                difficulty VARCHAR(20),
                question_type VARCHAR(50),
                question_text TEXT NOT NULL,
                options TEXT,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                hint TEXT,
                is_required BOOLEAN NOT NULL DEFAULT 1,
                PRIMARY KEY (attempt_id, question_id),
                FOREIGN KEY (attempt_id) REFERENCES assessment_attempts(attempt_id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assessment_answers (
                attempt_id VARCHAR(50) NOT NULL,
                question_id VARCHAR(50) NOT NULL,
                user_answer TEXT,
                is_correct BOOLEAN,
                time_spent INT DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (attempt_id, question_id),
                FOREIGN KEY (attempt_id) REFERENCES assessment_attempts(attempt_id)
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_assessment_attempts_session ON assessment_attempts(session_id, status)"))

    def _create_review_protocol_tables(self, conn) -> None:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_traces (
                trace_id VARCHAR(50) PRIMARY KEY,
                session_id VARCHAR(50) NOT NULL,
                stage_id VARCHAR(50),
                target_type VARCHAR(20) CHECK (target_type IN ('learning_package', 'answer')) NOT NULL,
                package_version INT DEFAULT 1,
                package_status VARCHAR(20) CHECK (package_status IN ('draft', 'approved', 'provisional', 'rejected')) DEFAULT 'draft',
                review_mode VARCHAR(20) CHECK (review_mode IN ('standard', 'debate')) DEFAULT 'debate',
                unresolved_issue_count INT DEFAULT 0,
                unresolved_must_fix_count INT DEFAULT 0,
                final_status VARCHAR(30) CHECK (final_status IN ('in_progress', 'passed', 'needs_human_review', 'failed')) DEFAULT 'in_progress',
                final_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (stage_id) REFERENCES stage_progress(stage_id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS learning_package_sources (
                trace_id VARCHAR(50) NOT NULL,
                source_id VARCHAR(50) NOT NULL,
                source_type VARCHAR(30) CHECK (source_type IN ('knowledge_doc', 'best_practice', 'manual', 'external_note', 'code_example', 'video')) NOT NULL,
                title VARCHAR(200),
                locator TEXT,
                content_ref TEXT,
                source_version VARCHAR(50),
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trace_id, source_id),
                FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS learning_package_claims (
                trace_id VARCHAR(50) NOT NULL,
                claim_id VARCHAR(50) NOT NULL,
                text TEXT NOT NULL,
                source_ref VARCHAR(50),
                source_span TEXT,
                section_ref TEXT NOT NULL,
                claim_type VARCHAR(20) CHECK (claim_type IN ('fact', 'procedure', 'code', 'config', 'concept')) NOT NULL,
                is_key_technical BOOLEAN DEFAULT 0,
                confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1) DEFAULT 0.5,
                status VARCHAR(20) CHECK (status IN ('active', 'revised', 'dropped')) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trace_id, claim_id),
                FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id),
                FOREIGN KEY (trace_id, source_ref) REFERENCES learning_package_sources(trace_id, source_id),
                CHECK (NOT (is_key_technical = 1 AND source_ref IS NULL))
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_debate_issues (
                issue_id VARCHAR(50) PRIMARY KEY,
                trace_id VARCHAR(50) NOT NULL,
                canonical_error_id VARCHAR(64) NOT NULL,
                error_id VARCHAR(64) NOT NULL,
                claim_id VARCHAR(50),
                error_type VARCHAR(50),
                severity VARCHAR(20) CHECK (severity IN ('critical', 'major', 'minor')) DEFAULT 'major',
                must_fix BOOLEAN DEFAULT 0,
                must_fix_reason TEXT,
                description TEXT,
                location TEXT,
                evidence TEXT,
                suggestion TEXT,
                source_refs TEXT,
                change_request TEXT,
                current_status VARCHAR(30) CHECK (current_status IN ('open', 'accepted', 'rejected', 'partially_accepted', 'deferred', 'overruled', 'withdrawn', 'needs_human_review')) DEFAULT 'open',
                final_disposition VARCHAR(30) CHECK (final_disposition IN ('accepted', 'rejected', 'partially_accepted', 'deferred', 'overruled', 'withdrawn', 'needs_human_review')),
                opened_in_round_num INT,
                resolved_in_round_num INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id),
                FOREIGN KEY (trace_id, claim_id) REFERENCES learning_package_claims(trace_id, claim_id),
                UNIQUE (trace_id, canonical_error_id)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_debate_rounds (
                round_id VARCHAR(50) PRIMARY KEY,
                trace_id VARCHAR(50) NOT NULL,
                round_num INT NOT NULL,
                review_iteration INT,
                round_status VARCHAR(20) CHECK (round_status IN ('open', 'closed')) DEFAULT 'open',
                orchestrator_verdict VARCHAR(30) CHECK (orchestrator_verdict IN ('pending', 'all_resolved', 'partial_resolved', 'deadlocked', 'needs_human_review')) DEFAULT 'pending',
                verdict_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id),
                UNIQUE (trace_id, round_num)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS review_debate_responses (
                response_id VARCHAR(50) PRIMARY KEY,
                round_id VARCHAR(50) NOT NULL,
                issue_id VARCHAR(50) NOT NULL,
                actor VARCHAR(20) CHECK (actor IN ('agent3', 'agent4', 'orchestrator')) NOT NULL,
                response_type VARCHAR(30) CHECK (response_type IN ('author_reply', 'reviewer_verdict', 'orchestrator_ruling')) NOT NULL,
                disposition VARCHAR(30) CHECK (disposition IN ('accepted', 'rejected', 'partially_accepted', 'persist', 'withdraw', 'downgrade', 'overruled', 'deferred')) NOT NULL,
                rationale TEXT,
                source_refs TEXT,
                patched_sections TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (round_id) REFERENCES review_debate_rounds(round_id),
                FOREIGN KEY (issue_id) REFERENCES review_debate_issues(issue_id),
                UNIQUE (round_id, issue_id, actor, response_type)
            )
        """))

    def _create_review_protocol_indexes(self, conn) -> None:
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_review_traces_session_stage ON review_traces(session_id, stage_id)",
            "CREATE INDEX IF NOT EXISTS idx_review_traces_status ON review_traces(final_status, package_status)",
            "CREATE INDEX IF NOT EXISTS idx_learning_package_sources_type ON learning_package_sources(trace_id, source_type)",
            "CREATE INDEX IF NOT EXISTS idx_learning_package_claims_source ON learning_package_claims(trace_id, source_ref)",
            "CREATE INDEX IF NOT EXISTS idx_learning_package_claims_status ON learning_package_claims(trace_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_review_debate_issues_trace_status ON review_debate_issues(trace_id, current_status)",
            "CREATE INDEX IF NOT EXISTS idx_review_debate_issues_claim ON review_debate_issues(trace_id, claim_id)",
            "CREATE INDEX IF NOT EXISTS idx_review_debate_rounds_trace ON review_debate_rounds(trace_id, round_num)",
            "CREATE INDEX IF NOT EXISTS idx_review_debate_responses_issue ON review_debate_responses(issue_id, actor)",
        ]
        for statement in index_statements:
            conn.execute(text(statement))

    def _migrate_review_protocol_tables(self, conn) -> None:
        """增量迁移：表不存在则 CREATE，缺列则 ALTER ADD COLUMN，永不 DROP，保护已写入的审核数据。

        设计原则：
        - CREATE TABLE 保留完整约束（PK/NOT NULL/CHECK/UNIQUE/FK），仅首次建表生效
        - ALTER ADD COLUMN 仅用于后续加列，不带 CHECK/UNIQUE/PK（SQLite 限制），靠 DEFAULT 保证非空
        - 复合主键和外键只在 CREATE 时建立，不通过 ALTER 修改
        - 若未来需要改列类型/删列/改约束，须另写带数据迁移的 rebuild 逻辑，不要回到 drop+recreate
        """
        # 每张表：create_sql 首次建表用；columns 增量补列用 (列名, ADD COLUMN 语句片段)
        # ADD COLUMN 片段不含 PK/CHECK/UNIQUE/FK（SQLite ALTER 限制），仅 TYPE + DEFAULT
        specs = [
            {
                "name": "review_traces",
                "create_sql": """
                    CREATE TABLE review_traces (
                        trace_id VARCHAR(50) PRIMARY KEY,
                        session_id VARCHAR(50) NOT NULL,
                        stage_id VARCHAR(50),
                        target_type VARCHAR(20) CHECK (target_type IN ('learning_package', 'answer')) NOT NULL,
                        package_version INT DEFAULT 1,
                        package_status VARCHAR(20) CHECK (package_status IN ('draft', 'approved', 'provisional', 'rejected')) DEFAULT 'draft',
                        review_mode VARCHAR(20) CHECK (review_mode IN ('standard', 'debate')) DEFAULT 'debate',
                        unresolved_issue_count INT DEFAULT 0,
                        unresolved_must_fix_count INT DEFAULT 0,
                        final_status VARCHAR(30) CHECK (final_status IN ('in_progress', 'passed', 'needs_human_review', 'failed')) DEFAULT 'in_progress',
                        final_summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        closed_at TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                        FOREIGN KEY (stage_id) REFERENCES stage_progress(stage_id)
                    )
                """,
                "columns": [
                    ("session_id", "VARCHAR(50) NOT NULL DEFAULT ''"),
                    ("stage_id", "VARCHAR(50)"),
                    ("target_type", "VARCHAR(20) NOT NULL DEFAULT 'learning_package'"),
                    ("package_version", "INT DEFAULT 1"),
                    ("package_status", "VARCHAR(20) DEFAULT 'draft'"),
                    ("review_mode", "VARCHAR(20) DEFAULT 'debate'"),
                    ("unresolved_issue_count", "INT DEFAULT 0"),
                    ("unresolved_must_fix_count", "INT DEFAULT 0"),
                    ("final_status", "VARCHAR(30) DEFAULT 'in_progress'"),
                    ("final_summary", "TEXT"),
                    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                    ("closed_at", "TIMESTAMP"),
                ],
            },
            {
                "name": "learning_package_sources",
                "create_sql": """
                    CREATE TABLE learning_package_sources (
                        trace_id VARCHAR(50) NOT NULL,
                        source_id VARCHAR(50) NOT NULL,
                        source_type VARCHAR(30) CHECK (source_type IN ('knowledge_doc', 'best_practice', 'manual', 'external_note', 'code_example', 'video')) NOT NULL,
                        title VARCHAR(200),
                        locator TEXT,
                        content_ref TEXT,
                        source_version VARCHAR(50),
                        metadata_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (trace_id, source_id),
                        FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id)
                    )
                """,
                "columns": [
                    ("source_type", "VARCHAR(30) NOT NULL DEFAULT 'knowledge_doc'"),
                    ("title", "VARCHAR(200)"),
                    ("locator", "TEXT"),
                    ("content_ref", "TEXT"),
                    ("source_version", "VARCHAR(50)"),
                    ("metadata_json", "TEXT"),
                    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ],
            },
            {
                "name": "learning_package_claims",
                "create_sql": """
                    CREATE TABLE learning_package_claims (
                        trace_id VARCHAR(50) NOT NULL,
                        claim_id VARCHAR(50) NOT NULL,
                        text TEXT NOT NULL,
                        source_ref VARCHAR(50),
                        source_span TEXT,
                        section_ref TEXT NOT NULL,
                        claim_type VARCHAR(20) CHECK (claim_type IN ('fact', 'procedure', 'code', 'config', 'concept')) NOT NULL,
                        is_key_technical BOOLEAN DEFAULT 0,
                        confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1) DEFAULT 0.5,
                        status VARCHAR(20) CHECK (status IN ('active', 'revised', 'dropped')) DEFAULT 'active',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (trace_id, claim_id),
                        FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id),
                        FOREIGN KEY (trace_id, source_ref) REFERENCES learning_package_sources(trace_id, source_id),
                        CHECK (NOT (is_key_technical = 1 AND source_ref IS NULL))
                    )
                """,
                "columns": [
                    ("text", "TEXT NOT NULL DEFAULT ''"),
                    ("source_ref", "VARCHAR(50)"),
                    ("source_span", "TEXT"),
                    ("section_ref", "TEXT NOT NULL DEFAULT ''"),
                    ("claim_type", "VARCHAR(20) NOT NULL DEFAULT 'concept'"),
                    ("is_key_technical", "BOOLEAN DEFAULT 0"),
                    ("confidence", "FLOAT DEFAULT 0.5"),
                    ("status", "VARCHAR(20) DEFAULT 'active'"),
                    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ],
            },
            {
                "name": "review_debate_issues",
                "create_sql": """
                    CREATE TABLE review_debate_issues (
                        issue_id VARCHAR(50) PRIMARY KEY,
                        trace_id VARCHAR(50) NOT NULL,
                        canonical_error_id VARCHAR(64) NOT NULL,
                        error_id VARCHAR(64) NOT NULL,
                        claim_id VARCHAR(50),
                        error_type VARCHAR(50),
                        severity VARCHAR(20) CHECK (severity IN ('critical', 'major', 'minor')) DEFAULT 'major',
                        must_fix BOOLEAN DEFAULT 0,
                        must_fix_reason TEXT,
                        description TEXT,
                        location TEXT,
                        evidence TEXT,
                        suggestion TEXT,
                        source_refs TEXT,
                        change_request TEXT,
                        current_status VARCHAR(30) CHECK (current_status IN ('open', 'accepted', 'rejected', 'partially_accepted', 'deferred', 'overruled', 'withdrawn', 'needs_human_review')) DEFAULT 'open',
                        final_disposition VARCHAR(30) CHECK (final_disposition IN ('accepted', 'rejected', 'partially_accepted', 'deferred', 'overruled', 'withdrawn', 'needs_human_review')),
                        opened_in_round_num INT,
                        resolved_in_round_num INT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id),
                        FOREIGN KEY (trace_id, claim_id) REFERENCES learning_package_claims(trace_id, claim_id),
                        UNIQUE (trace_id, canonical_error_id)
                    )
                """,
                "columns": [
                    ("trace_id", "VARCHAR(50) NOT NULL DEFAULT ''"),
                    ("canonical_error_id", "VARCHAR(64) NOT NULL DEFAULT ''"),
                    ("error_id", "VARCHAR(64) NOT NULL DEFAULT ''"),
                    ("claim_id", "VARCHAR(50)"),
                    ("error_type", "VARCHAR(50)"),
                    ("severity", "VARCHAR(20) DEFAULT 'major'"),
                    ("must_fix", "BOOLEAN DEFAULT 0"),
                    ("must_fix_reason", "TEXT"),
                    ("description", "TEXT"),
                    ("location", "TEXT"),
                    ("evidence", "TEXT"),
                    ("suggestion", "TEXT"),
                    ("source_refs", "TEXT"),
                    ("change_request", "TEXT"),
                    ("current_status", "VARCHAR(30) DEFAULT 'open'"),
                    ("final_disposition", "VARCHAR(30)"),
                    ("opened_in_round_num", "INT"),
                    ("resolved_in_round_num", "INT"),
                    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ],
            },
            {
                "name": "review_debate_rounds",
                "create_sql": """
                    CREATE TABLE review_debate_rounds (
                        round_id VARCHAR(50) PRIMARY KEY,
                        trace_id VARCHAR(50) NOT NULL,
                        round_num INT NOT NULL,
                        review_iteration INT,
                        round_status VARCHAR(20) CHECK (round_status IN ('open', 'closed')) DEFAULT 'open',
                        orchestrator_verdict VARCHAR(30) CHECK (orchestrator_verdict IN ('pending', 'all_resolved', 'partial_resolved', 'deadlocked', 'needs_human_review')) DEFAULT 'pending',
                        verdict_summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        closed_at TIMESTAMP,
                        FOREIGN KEY (trace_id) REFERENCES review_traces(trace_id),
                        UNIQUE (trace_id, round_num)
                    )
                """,
                "columns": [
                    ("trace_id", "VARCHAR(50) NOT NULL DEFAULT ''"),
                    ("round_num", "INT NOT NULL DEFAULT 0"),
                    ("review_iteration", "INT"),
                    ("round_status", "VARCHAR(20) DEFAULT 'open'"),
                    ("orchestrator_verdict", "VARCHAR(30) DEFAULT 'pending'"),
                    ("verdict_summary", "TEXT"),
                    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                    ("closed_at", "TIMESTAMP"),
                ],
            },
            {
                "name": "review_debate_responses",
                "create_sql": """
                    CREATE TABLE review_debate_responses (
                        response_id VARCHAR(50) PRIMARY KEY,
                        round_id VARCHAR(50) NOT NULL,
                        issue_id VARCHAR(50) NOT NULL,
                        actor VARCHAR(20) CHECK (actor IN ('agent3', 'agent4', 'orchestrator')) NOT NULL,
                        response_type VARCHAR(30) CHECK (response_type IN ('author_reply', 'reviewer_verdict', 'orchestrator_ruling')) NOT NULL,
                        disposition VARCHAR(30) CHECK (disposition IN ('accepted', 'rejected', 'partially_accepted', 'persist', 'withdraw', 'downgrade', 'overruled', 'deferred')) NOT NULL,
                        rationale TEXT,
                        source_refs TEXT,
                        patched_sections TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (round_id) REFERENCES review_debate_rounds(round_id),
                        FOREIGN KEY (issue_id) REFERENCES review_debate_issues(issue_id),
                        UNIQUE (round_id, issue_id, actor, response_type)
                    )
                """,
                "columns": [
                    ("round_id", "VARCHAR(50) NOT NULL DEFAULT ''"),
                    ("issue_id", "VARCHAR(50) NOT NULL DEFAULT ''"),
                    ("actor", "VARCHAR(20) NOT NULL DEFAULT 'agent3'"),
                    ("response_type", "VARCHAR(30) NOT NULL DEFAULT 'author_reply'"),
                    ("disposition", "VARCHAR(30) NOT NULL DEFAULT 'accepted'"),
                    ("rationale", "TEXT"),
                    ("source_refs", "TEXT"),
                    ("patched_sections", "TEXT"),
                    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ],
            },
        ]

        for spec in specs:
            if not self._table_exists(conn, spec["name"]):
                # 首次建表：带完整约束
                conn.execute(text(spec["create_sql"]))
            else:
                # 已存在：仅补缺失的列（不 DROP、不改列类型、不动数据）
                for col_name, add_sql in spec["columns"]:
                    self._ensure_column(conn, spec["name"], col_name, add_sql)
        self._create_review_protocol_indexes(conn)

    def _rebuild_table_if_needed(self, conn, table_name: str, expected_columns: set[str], create_sql: str) -> None:
        if not self._table_exists(conn, table_name):
            conn.execute(text(create_sql))
            return

        existing_columns = self._get_table_columns(conn, table_name)
        if existing_columns == expected_columns:
            return

        legacy_table = f"{table_name}_legacy_migration"
        if self._table_exists(conn, legacy_table):
            conn.execute(text(f"DROP TABLE {legacy_table}"))
        conn.execute(text(f"ALTER TABLE {table_name} RENAME TO {legacy_table}"))
        conn.execute(text(create_sql))

        legacy_columns = self._get_table_columns(conn, legacy_table)
        common_columns = [column for column in expected_columns if column in legacy_columns]
        if common_columns:
            column_list = ", ".join(common_columns)
            conn.execute(text(f"""
                INSERT INTO {table_name} ({column_list})
                SELECT {column_list}
                FROM {legacy_table}
            """))
        conn.execute(text(f"DROP TABLE {legacy_table}"))

    @staticmethod
    def _table_exists(conn, table_name: str) -> bool:
        row = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :name"),
            {"name": table_name},
        ).first()
        return row is not None

    @staticmethod
    def _get_table_columns(conn, table_name: str) -> set[str]:
        return {
            row[1]
            for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        }

    def create_session(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> None:
        with self.Session() as session:
            session.execute(text("INSERT OR IGNORE INTO sessions (session_id, user_id) VALUES (:session_id, :user_id)"),
                           {"session_id": session_id, "user_id": user_id})
            session.commit()
        self.ensure_user_subtopic_status(user_id)
        self.refresh_target_metrics(session_id)

    def refresh_target_metrics(self, session_id: str) -> dict:
        target = self.get_target_settings(session_id)
        if not target:
            raise ValueError(f"Session not found: {session_id}")
        with self.Session() as session:
            gaps_json = session.execute(
                text("SELECT target_knowledge_gaps FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).scalar() or "[]"
        gaps = json.loads(gaps_json)
        target_ids = {
            subtopic_id
            for gap in gaps if isinstance(gap, dict)
            for subtopic_id in gap.get("target_subtopic_ids", [])
            if subtopic_id
        }
        target_blocks = {
            gap.get("block_id") for gap in gaps
            if isinstance(gap, dict) and gap.get("block_id")
        }
        rows = self.list_subtopic_statuses(session_id)
        selected = [
            row for row in rows
            if row["subtopic_id"] in target_ids
            or (not target_ids and row["block_id"] in target_blocks)
        ]
        total_hours = round(sum(float(row.get("estimated_hours") or 0) for row in selected), 2)
        # The active path may start with prerequisite stages outside the
        # learner's final target block. A target total must never shrink below
        # time already completed on that path.
        with self.Session() as session:
            completed_stage_hours = session.execute(text("""
                SELECT COALESCE(SUM(estimated_hours), 0)
                FROM stage_progress
                WHERE session_id = :session_id AND status = 'completed'
            """), {"session_id": session_id}).scalar_one()
        total_hours = max(total_hours, round(float(completed_stage_hours or 0), 2))
        progress = round(
            sum(float(row.get("mastery_score") or 0) for row in selected) / len(selected), 4
        ) if selected else 0.0
        return self.update_target_settings(session_id, {
            "target_total_hours": total_hours,
            "target_progress": progress,
        })

    def session_exists(self, session_id: str) -> bool:
        with self.Session() as session:
            row = session.execute(
                text("SELECT 1 FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).first()
            return row is not None

    def get_session_user_id(self, session_id: str) -> str:
        with self.Session() as session:
            user_id = session.execute(
                text("SELECT user_id FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).scalar()
        if not user_id:
            if not self.session_exists(session_id):
                raise ValueError(f"Session not found: {session_id}")
            return DEFAULT_USER_ID
        return str(user_id)

    def get_user_profile(self, user_id: str = DEFAULT_USER_ID) -> dict | None:
        with self.Session() as session:
            row = session.execute(text("""
                SELECT user_id, grade, skill_tags, foundation_confirmed, learning_style, hardware,
                       device_type, gpu_model, vram_gb, gpu_architecture,
                       default_daily_hours, learning_rhythm, confidence_index,
                       confusion_markers, confidence, version, created_at, updated_at
                FROM learner_profiles
                WHERE user_id = :user_id
            """), {"user_id": user_id}).mappings().first()
        if not row:
            return None
        result = dict(row)
        result["foundation_confirmed"] = bool(result.get("foundation_confirmed"))
        result["skill_tags"] = json.loads(result.get("skill_tags") or "[]")
        result["confusion_markers"] = json.loads(result.get("confusion_markers") or "[]")
        return result

    def upsert_user_profile(self, user_id: str, profile: dict) -> dict:
        existing = self.get_user_profile(user_id) or {}
        values = {
            "user_id": user_id,
            "grade": profile.get("grade", existing.get("grade", "")),
            "skill_tags": json.dumps(profile.get("skill_tags", existing.get("skill_tags", [])), ensure_ascii=False),
            "foundation_confirmed": bool(profile.get("foundation_confirmed", existing.get("foundation_confirmed", False))),
            "learning_style": profile.get("learning_style", existing.get("learning_style", "mixed")),
            "hardware": profile.get("hardware", existing.get("hardware", "")),
            "device_type": profile.get("device_type", existing.get("device_type", "")),
            "gpu_model": profile.get("gpu_model", existing.get("gpu_model", "")),
            "vram_gb": profile.get("vram_gb", existing.get("vram_gb")),
            "gpu_architecture": profile.get("gpu_architecture", existing.get("gpu_architecture", "")),
            "default_daily_hours": profile.get("default_daily_hours", profile.get("daily_hours", existing.get("default_daily_hours", 2.0))),
            "learning_rhythm": profile.get("learning_rhythm", existing.get("learning_rhythm", "medium")),
            "confidence_index": profile.get("confidence_index", existing.get("confidence_index", 0.5)),
            "confusion_markers": json.dumps(profile.get("confusion_markers", existing.get("confusion_markers", [])), ensure_ascii=False),
            "confidence": profile.get("confidence", existing.get("confidence", 0.7)),
        }
        with self.Session() as session:
            session.execute(text("""
                INSERT INTO learner_profiles (
                    user_id, grade, skill_tags, foundation_confirmed, learning_style, hardware,
                    device_type, gpu_model, vram_gb, gpu_architecture,
                    default_daily_hours, learning_rhythm, confidence_index,
                    confusion_markers, confidence, version
                ) VALUES (
                    :user_id, :grade, :skill_tags, :foundation_confirmed, :learning_style, :hardware,
                    :device_type, :gpu_model, :vram_gb, :gpu_architecture,
                    :default_daily_hours, :learning_rhythm, :confidence_index,
                    :confusion_markers, :confidence, 1
                )
                ON CONFLICT(user_id) DO UPDATE SET
                    grade = excluded.grade,
                    skill_tags = excluded.skill_tags,
                    foundation_confirmed = excluded.foundation_confirmed,
                    learning_style = excluded.learning_style,
                    hardware = excluded.hardware,
                    device_type = excluded.device_type,
                    gpu_model = excluded.gpu_model,
                    vram_gb = excluded.vram_gb,
                    gpu_architecture = excluded.gpu_architecture,
                    default_daily_hours = excluded.default_daily_hours,
                    learning_rhythm = excluded.learning_rhythm,
                    confidence_index = excluded.confidence_index,
                    confusion_markers = excluded.confusion_markers,
                    confidence = excluded.confidence,
                    version = learner_profiles.version + 1,
                    updated_at = CURRENT_TIMESTAMP
            """), values)
            session.commit()
        return self.get_user_profile(user_id) or {}

    def get_target_settings(self, session_id: str) -> dict | None:
        with self.Session() as session:
            row = session.execute(text("""
                SELECT session_id, user_id, goal, target_total_hours,
                       target_daily_hours, target_progress, updated_at
                FROM sessions WHERE session_id = :session_id
            """), {"session_id": session_id}).mappings().first()
        return dict(row) if row else None

    def update_target_settings(self, session_id: str, updates: dict) -> dict:
        allowed = {"goal", "target_total_hours", "target_daily_hours", "target_progress"}
        fields = {key: value for key, value in updates.items() if key in allowed and value is not None}
        adjusted_stages: list[dict] = []
        with self.Session() as session:
            exists = session.execute(
                text("SELECT 1 FROM sessions WHERE session_id = :session_id"),
                {"session_id": session_id},
            ).first()
            if not exists:
                raise ValueError(f"Session not found: {session_id}")

            if "target_total_hours" in fields:
                stages = session.execute(text("""
                    SELECT stage_id, estimated_hours, status
                    FROM stage_progress
                    WHERE session_id = :session_id
                    ORDER BY stage_num ASC
                """), {"session_id": session_id}).mappings().all()
                completed_hours = round(sum(
                    float(stage["estimated_hours"] or 0)
                    for stage in stages
                    if stage["status"] == "completed"
                ), 2)
                requested_total = round(float(fields["target_total_hours"]), 2)
                if requested_total < completed_hours:
                    raise ValueError(
                        f"Target hours cannot be less than completed stage hours ({completed_hours})"
                    )

                unfinished = [stage for stage in stages if stage["status"] != "completed"]
                remaining_hours = round(requested_total - completed_hours, 2)
                if stages and not unfinished:
                    pass  # 全部阶段已完成，无需再分配学时，直接更新目标值
                if unfinished:
                    current_total = sum(float(stage["estimated_hours"] or 0) for stage in unfinished)
                    if remaining_hours == 0 and current_total == 0:
                        adjusted_stages.extend({
                            "stage_id": stage["stage_id"],
                            "estimated_hours": 0.0,
                        } for stage in unfinished)
                    else:
                        allocated = 0.0
                        for index, stage in enumerate(unfinished):
                            if index == len(unfinished) - 1:
                                stage_hours = round(remaining_hours - allocated, 2)
                            elif current_total > 0:
                                stage_hours = round(
                                    remaining_hours * float(stage["estimated_hours"] or 0) / current_total,
                                    2,
                                )
                            else:
                                stage_hours = round(remaining_hours / len(unfinished), 2)
                            if stage_hours <= 0:
                                raise ValueError("Each unfinished stage must retain a positive duration")
                            session.execute(text("""
                                UPDATE stage_progress SET estimated_hours = :estimated_hours
                                WHERE session_id = :session_id AND stage_id = :stage_id
                            """), {
                                "session_id": session_id,
                                "stage_id": stage["stage_id"],
                                "estimated_hours": stage_hours,
                            })
                            adjusted_stages.append({
                                "stage_id": stage["stage_id"],
                                "estimated_hours": stage_hours,
                            })
                            allocated = round(allocated + stage_hours, 2)

            if fields:
                assignments = ", ".join(f"{key} = :{key}" for key in fields)
                result = session.execute(
                    text(f"UPDATE sessions SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE session_id = :session_id"),
                    {"session_id": session_id, **fields},
                )
                if result.rowcount == 0:
                    raise ValueError(f"Session not found: {session_id}")
            session.commit()
        target = self.get_target_settings(session_id)
        if not target:
            raise ValueError(f"Session not found: {session_id}")
        target["adjusted_stages"] = adjusted_stages
        return target

    def get_stage_duration(self, session_id: str, stage_id: str) -> dict | None:
        with self.Session() as session:
            row = session.execute(text("""
                SELECT stage_id, session_id, stage_num, estimated_hours
                FROM stage_progress
                WHERE session_id = :session_id AND stage_id = :stage_id
            """), {"session_id": session_id, "stage_id": stage_id}).mappings().first()
        return dict(row) if row else None

    def update_stage_duration(self, session_id: str, stage_id: str, estimated_hours: float) -> dict:
        with self.Session() as session:
            result = session.execute(text("""
                UPDATE stage_progress SET estimated_hours = :estimated_hours
                WHERE session_id = :session_id AND stage_id = :stage_id
            """), {"session_id": session_id, "stage_id": stage_id, "estimated_hours": estimated_hours})
            if result.rowcount == 0:
                raise ValueError(f"Stage not found: {stage_id}")
            session.commit()
        return self.get_stage_duration(session_id, stage_id) or {}

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
                    goal = COALESCE(:goal, goal),
                    target_total_hours = COALESCE(:target_total_hours, target_total_hours),
                    target_daily_hours = COALESCE(:target_daily_hours, target_daily_hours),
                    target_progress = COALESCE(:target_progress, target_progress),
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = :session_id
            """), {
                "current_stage": state.get("current_stage", 1),
                "status": state.get("status", "active"),
                "goal": state.get("goal"),
                "target_total_hours": state.get("target_total_hours"),
                "target_daily_hours": state.get("target_daily_hours"),
                "target_progress": state.get("target_progress"),
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

    _TRUE_FALSE_TYPES = {"true_false", "判断", "判断题", "yes_no"}

    def get_question(self, question_id: str) -> dict | None:
        with self.Session() as session:
            row = session.execute(text("""
                SELECT question_id, block_id, subtopic_id, difficulty, question_type,
                       question_text, options, correct_answer, explanation,
                       expected_output, grading_mode, source, hint
                FROM test_questions
                WHERE question_id = :question_id
            """), {"question_id": question_id}).mappings().first()
            if not row:
                return None
            result = dict(row)
            if result.get("options"):
                try:
                    result["options"] = json.loads(result["options"])
                except (TypeError, ValueError):
                    result["options"] = None
            else:
                result["options"] = None
            qt = (result.get("question_type") or "").strip().lower() if result.get("question_type") else ""
            if not result["options"] and qt in self._TRUE_FALSE_TYPES:
                result["options"] = ["正确", "错误"]
            return result

    def list_questions_for_stage(
        self,
        block_id: str,
        subtopic_ids: list[str] | None = None,
        limit: int = 2,
        questions_per_subtopic: int = 2,
    ) -> list[dict]:
        """Return stage questions strictly from each subtopic's own pool.

        Policy (as requested by user):
        - For every given ``subtopic_id``, sample up to
          ``questions_per_subtopic`` questions **that actually belong to
          that subtopic**. Default: each knowledge point gets ≤2 questions.
        - If a subtopic has 0 questions, skip it (don't "steal" from other
          subtopics or the block-level pool).
        - If a subtopic has exactly 1 question, return exactly 1 question.
        - When ``subtopic_ids`` is empty or not provided, fall back to the
          legacy block-level global ``limit`` (only used by legacy callers
          that pass no subtopic dimension).
        """
        if not block_id:
            return []
        subtopic_ids = list(dict.fromkeys(item for item in (subtopic_ids or []) if item))
        base_cols = (
            "question_id, block_id, subtopic_id, difficulty, question_type, "
            "question_text, options, correct_answer, explanation, "
            "expected_output, grading_mode, source, hint"
        )
        picked_ids: set[str] = set()
        combined_rows: list[Any] = []

        def _itemize(rows) -> list[dict]:
            out: list[dict] = []
            for row in rows:
                item = dict(row)
                if item.get("options"):
                    try:
                        item["options"] = json.loads(item["options"])
                    except (TypeError, ValueError):
                        item["options"] = None
                else:
                    item["options"] = None
                qt = (item.get("question_type") or "").strip().lower() if item.get("question_type") else ""
                if not item["options"] and qt in self._TRUE_FALSE_TYPES:
                    item["options"] = ["正确", "错误"]
                out.append(item)
            return out

        with self.Session() as session:
            if subtopic_ids:
                # 严格按 subtopic 抽：每个 subtopic 最多 questions_per_subtopic 题，
                # 0 题跳过，1 题就出 1 题。绝不从其他 subtopic / block 池凑题。
                # 随机化：先取该 subtopic 全部题，再 Python 端 random.sample，
                # 保证每次出题顺序与题目都不同（用户要求测试题目必须随机）。
                per_subtopic_sql = text(f"""
                    SELECT {base_cols}
                    FROM test_questions
                    WHERE block_id = :block_id AND subtopic_id = :subtopic_id
                    ORDER BY difficulty ASC, question_id ASC
                """)
                for stid in subtopic_ids:
                    pool = session.execute(
                        per_subtopic_sql,
                        {"block_id": block_id, "subtopic_id": stid},
                    ).mappings().all()
                    take = min(len(pool), questions_per_subtopic)
                    sampled = random.sample(list(pool), take) if take > 0 else []
                    for r in sampled:
                        if r["question_id"] in picked_ids:
                            continue
                        picked_ids.add(r["question_id"])
                        combined_rows.append(r)
            else:
                # 未指定 subtopic → 保留旧的全局 limit 行为（同样随机化）
                global_pool = session.execute(
                    text(f"""
                        SELECT {base_cols}
                        FROM test_questions
                        WHERE block_id = :block_id
                        ORDER BY difficulty ASC, question_id ASC
                    """),
                    {"block_id": block_id},
                ).mappings().all()
                take = min(len(global_pool), limit)
                combined_rows = random.sample(list(global_pool), take) if take > 0 else []

        return _itemize(combined_rows)

    def update_session_status(self, session_id: str, status: str) -> None:
        with self.Session() as session:
            session.execute(text("UPDATE sessions SET status = :status, updated_at = CURRENT_TIMESTAMP WHERE session_id = :session_id"),
                           {"status": status, "session_id": session_id})
            session.commit()

    def delete_session(self, session_id: str, user_id: str | None = DEFAULT_USER_ID) -> dict:
        """物理删除一个 session 及其所有关联快照/历史（不可逆）。

        按「从子到父」的顺序清理所有带 session_id 外键的表，避免 SQLite
        外键约束报错。对无直接 session_id 外键、但通过 attempt_id / trace_id
        链接的子表，先通过 session 维度查出对应的 attempt_id / trace_id，再批量
        删除它们的明细。最终返回各表删除的行数明细。
        """
        if not session_id:
            raise ValueError("session_id is required")
        deletes: dict[str, int] = {}

        with self.Session() as session:
            # 临时关闭 FK 检查：历史 DB 的 learning_package_claims 表上残留了
            # 指向已废弃表 learning_package_sources_legacy 的外键约束，一旦启用
            # foreign_keys=ON，任何 DELETE FROM learning_package_claims 都会触发
            # "no such table: main.learning_package_sources_legacy" 报错。我们在
            # 单个事务内手动按子→父顺序清理，所以可以安全关闭 FK 检查。
            session.execute(text("PRAGMA foreign_keys = OFF"))

            # 先做所有权校验（若传了 user_id）：不允许跨用户删
            if user_id:
                owner = session.execute(
                    text("SELECT user_id FROM sessions WHERE session_id = :sid"),
                    {"sid": session_id},
                ).scalar()
                if owner is None:
                    session.execute(text("PRAGMA foreign_keys = ON"))
                    raise LookupError(f"Session not found: {session_id}")
                if str(owner) != str(user_id):
                    session.execute(text("PRAGMA foreign_keys = ON"))
                    raise PermissionError(f"Session {session_id} does not belong to user {user_id}")

            # --- 子表 1/2：通过 attempt_id / trace_id 级联的明细 ---
            attempt_ids = [
                r[0] for r in session.execute(
                    text("SELECT attempt_id FROM assessment_attempts WHERE session_id = :sid"),
                    {"sid": session_id},
                ).fetchall()
            ]
            trace_ids = [
                r[0] for r in session.execute(
                    text("SELECT trace_id FROM review_traces WHERE session_id = :sid"),
                    {"sid": session_id},
                ).fetchall()
            ]

            def _del(sql: str, params: dict) -> int:
                return session.execute(text(sql), params).rowcount or 0

            if attempt_ids:
                placeholders = ",".join(f":aid{i}" for i in range(len(attempt_ids)))
                attempt_params: dict = {f"aid{i}": aid for i, aid in enumerate(attempt_ids)}
                deletes["assessment_answers"] = _del(
                    f"DELETE FROM assessment_answers WHERE attempt_id IN ({placeholders})", attempt_params
                )
                deletes["assessment_attempt_questions"] = _del(
                    f"DELETE FROM assessment_attempt_questions WHERE attempt_id IN ({placeholders})", attempt_params
                )
            if trace_ids:
                placeholders = ",".join(f":tid{i}" for i in range(len(trace_ids)))
                trace_params: dict = {f"tid{i}": tid for i, tid in enumerate(trace_ids)}
                # debate responses 关联 round_id / issue_id → 先于 rounds/issues 删除
                # 安全上限：单 session trace_id 不可能超 500，超过就跳过（避免超大 SQL）
                if len(trace_ids) <= 500:
                    deletes["review_debate_responses"] = _del(
                        f"DELETE FROM review_debate_responses WHERE round_id IN (SELECT round_id FROM review_debate_rounds WHERE trace_id IN ({placeholders}))",
                        trace_params,
                    ) + _del(
                        f"DELETE FROM review_debate_responses WHERE issue_id IN (SELECT issue_id FROM review_debate_issues WHERE trace_id IN ({placeholders}))",
                        trace_params,
                    )
                else:
                    deletes["review_debate_responses"] = 0
                deletes["review_debate_rounds"] = _del(
                    f"DELETE FROM review_debate_rounds WHERE trace_id IN ({placeholders})", trace_params
                )
                deletes["learning_package_claims"] = _del(
                    f"DELETE FROM learning_package_claims WHERE trace_id IN ({placeholders})", trace_params
                )
                deletes["learning_package_sources"] = _del(
                    f"DELETE FROM learning_package_sources WHERE trace_id IN ({placeholders})", trace_params
                )
                deletes["review_debate_issues"] = _del(
                    f"DELETE FROM review_debate_issues WHERE trace_id IN ({placeholders})", trace_params
                )
                deletes["review_traces"] = _del(
                    f"DELETE FROM review_traces WHERE trace_id IN ({placeholders})", trace_params
                )
            if attempt_ids:
                placeholders = ",".join(f":aid{i}" for i in range(len(attempt_ids)))
                attempt_params = {f"aid{i}": aid for i, aid in enumerate(attempt_ids)}
                deletes["assessment_attempts"] = _del(
                    f"DELETE FROM assessment_attempts WHERE attempt_id IN ({placeholders})", attempt_params
                )

            # --- 子表 2/2：直接带 session_id 外键的表 ---
            direct_session_tables = [
                # 按依赖排序：无被依赖者优先
                "user_answers", "user_qa",
                "subtopic_status",
                "user_profiles",
                "stage_progress",
                "session_states",
            ]
            for table in direct_session_tables:
                deletes[table] = _del(
                    f"DELETE FROM {table} WHERE session_id = :sid", {"sid": session_id}
                )

            # --- 父表：sessions ---
            deletes["sessions"] = _del("DELETE FROM sessions WHERE session_id = :sid", {"sid": session_id})
            session.execute(text("PRAGMA foreign_keys = ON"))
            session.commit()

        deletes["total"] = sum(v for v in deletes.values() if isinstance(v, int))
        return deletes

    def delete_last_stage(self, session_id: str, user_id: str | None = DEFAULT_USER_ID) -> dict:
        """物理删除一个学习目标（session）的最后一个学习阶段，并清理所有关联快照。

        - 删 stage_progress 里 stage_num 最大的那条记录（只允许删最后一个，避免阶段中间出现空洞）。
        - 级联清理该 stage_id 关联的：review_traces（及其子表：claims/sources/debate）、
          assessment_attempts（及 attempt_questions / answers）。
        - 同时清理 session_states.state_json 里的 stage_info / learning_package / next_stage / result 等
          阶段相关缓存，并把 sessions.current_stage 回退到删除后的最大 stage_num（没有则 0）。
        - 如果 session 在删除完后 stage_progress 已经没有任何阶段，则调用 delete_session
          把整个学习目标连同 sessions / session_states / user_profiles 全部清掉。
        """
        if not session_id:
            raise ValueError("session_id is required")

        with self.Session() as session:
            # 临时关 FK：避开 learning_package_claims → learning_package_sources_legacy 残留约束
            session.execute(text("PRAGMA foreign_keys = OFF"))

            # 所有权校验
            if user_id:
                owner = session.execute(
                    text("SELECT user_id FROM sessions WHERE session_id = :sid"),
                    {"sid": session_id},
                ).scalar()
                if owner is None:
                    session.execute(text("PRAGMA foreign_keys = ON"))
                    raise LookupError(f"Session not found: {session_id}")
                if str(owner) != str(user_id):
                    session.execute(text("PRAGMA foreign_keys = ON"))
                    raise PermissionError(f"Session {session_id} does not belong to user {user_id}")

            # 1. 找到最后一个 stage_progress
            last_stage = session.execute(
                text("""
                    SELECT stage_id, stage_num, block_id, subtopic_id, status
                    FROM stage_progress
                    WHERE session_id = :sid
                    ORDER BY stage_num DESC
                    LIMIT 1
                """), {"sid": session_id},
            ).first()
            if last_stage is None:
                session.execute(text("PRAGMA foreign_keys = ON"))
                raise LookupError(f"No stage to delete for session {session_id}")

            stage_id, stage_num, block_id, subtopic_id, status = last_stage
            trace_ids = [
                r[0] for r in session.execute(
                    text("SELECT trace_id FROM review_traces WHERE session_id = :sid AND (stage_id = :stage_id OR stage_id IS NULL)"),
                    {"sid": session_id, "stage_id": stage_id},
                ).fetchall()
            ]
            attempt_ids = [
                r[0] for r in session.execute(
                    text("SELECT attempt_id FROM assessment_attempts WHERE session_id = :sid AND (stage_id = :stage_id OR stage_id IS NULL)"),
                    {"sid": session_id, "stage_id": stage_id},
                ).fetchall()
            ]

            def _del(sql: str, params: dict) -> int:
                return session.execute(text(sql), params).rowcount or 0

            details: dict[str, int] = {"stage_id": stage_id, "stage_num": stage_num}

            # 2. 删 assessment 链条
            if attempt_ids:
                placeholders = ",".join(f":aid{i}" for i in range(len(attempt_ids)))
                attempt_params = {f"aid{i}": aid for i, aid in enumerate(attempt_ids)}
                details["assessment_answers"] = _del(
                    f"DELETE FROM assessment_answers WHERE attempt_id IN ({placeholders})", attempt_params
                )
                details["assessment_attempt_questions"] = _del(
                    f"DELETE FROM assessment_attempt_questions WHERE attempt_id IN ({placeholders})", attempt_params
                )
                details["assessment_attempts"] = _del(
                    f"DELETE FROM assessment_attempts WHERE attempt_id IN ({placeholders})", attempt_params
                )
            else:
                details["assessment_attempts"] = 0

            # 3. 删 review_traces 链条（同 delete_session 的子→父顺序）
            if trace_ids:
                placeholders = ",".join(f":tid{i}" for i in range(len(trace_ids)))
                trace_params = {f"tid{i}": tid for i, tid in enumerate(trace_ids)}
                if len(trace_ids) <= 500:
                    details["review_debate_responses"] = _del(
                        f"DELETE FROM review_debate_responses WHERE round_id IN (SELECT round_id FROM review_debate_rounds WHERE trace_id IN ({placeholders}))",
                        trace_params,
                    ) + _del(
                        f"DELETE FROM review_debate_responses WHERE issue_id IN (SELECT issue_id FROM review_debate_issues WHERE trace_id IN ({placeholders}))",
                        trace_params,
                    )
                else:
                    details["review_debate_responses"] = 0
                details["review_debate_rounds"] = _del(
                    f"DELETE FROM review_debate_rounds WHERE trace_id IN ({placeholders})", trace_params
                )
                details["learning_package_claims"] = _del(
                    f"DELETE FROM learning_package_claims WHERE trace_id IN ({placeholders})", trace_params
                )
                details["learning_package_sources"] = _del(
                    f"DELETE FROM learning_package_sources WHERE trace_id IN ({placeholders})", trace_params
                )
                details["review_debate_issues"] = _del(
                    f"DELETE FROM review_debate_issues WHERE trace_id IN ({placeholders})", trace_params
                )
                details["review_traces"] = _del(
                    f"DELETE FROM review_traces WHERE trace_id IN ({placeholders})", trace_params
                )
            else:
                details["review_traces"] = 0

            # 4. 删 stage_progress
            details["stage_progress"] = _del(
                "DELETE FROM stage_progress WHERE session_id = :sid AND stage_id = :stage_id",
                {"sid": session_id, "stage_id": stage_id},
            )

            # 5. 清理 session_states.state_json 里阶段相关字段，并调整 sessions.current_stage
            state_row = session.execute(
                text("SELECT state_json FROM session_states WHERE session_id = :sid"),
                {"sid": session_id},
            ).first()
            state_json = state_row[0] if state_row else None
            import json as _json
            if state_json:
                try:
                    state = _json.loads(state_json)
                except (TypeError, ValueError):
                    state = {}
                # 清掉阶段相关字段；保留 profile / user_id 等画像信息
                for key in ("stage_info", "learning_package", "next_stage",
                            "planning_notes", "result", "assessment_state"):
                    state.pop(key, None)
                # phase 回退到 stage_ready，让用户从下一个阶段重新开始
                state["phase"] = "stage_ready"
                state["available_actions"] = ["start_stage", "plan_next_stage"]
                session.execute(
                    text("UPDATE session_states SET state_json = :payload, updated_at = CURRENT_TIMESTAMP WHERE session_id = :sid"),
                    {"payload": _json.dumps(state, ensure_ascii=False), "sid": session_id},
                )

            # 6. 计算剩余阶段数；若为 0 则整 session 删掉，否则 current_stage 回退
            remaining = session.execute(
                text("SELECT MAX(stage_num) FROM stage_progress WHERE session_id = :sid"),
                {"sid": session_id},
            ).scalar()
            if remaining is None:
                # 没阶段了 → 整 session 删
                session.execute(text("PRAGMA foreign_keys = ON"))
                session.commit()
                # 在新的事务里用 delete_session 物理删整个 session
                cascade = self.delete_session(session_id, user_id)
                details["session_deleted"] = True
                details["cascade"] = cascade
                return details

            session.execute(
                text("UPDATE sessions SET current_stage = :stage, updated_at = CURRENT_TIMESTAMP WHERE session_id = :sid"),
                {"stage": remaining, "sid": session_id},
            )

            session.execute(text("PRAGMA foreign_keys = ON"))
            session.commit()

        details["remaining_stage_num"] = remaining
        details["session_deleted"] = False
        return details

    def list_sessions(self, user_id: str = DEFAULT_USER_ID) -> list[dict]:
        """获取指定用户的会话列表，禁止跨用户返回历史目标。"""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT s.session_id, s.user_id, s.status, s.current_stage, s.created_at, s.updated_at,
                       s.target_total_hours, s.target_daily_hours,
                       ss.state_json
                FROM sessions s
                LEFT JOIN session_states ss ON s.session_id = ss.session_id
                WHERE s.user_id = :user_id
                ORDER BY s.created_at DESC, s.updated_at DESC
            """), {"user_id": user_id}).mappings().all()
            
            result = []
            for row in rows:
                state_json = json.loads(row["state_json"]) if row["state_json"] else {}
                profile = self.get_latest_profile(row["session_id"]) or {}
                stage_info = state_json.get("stage_info") or {}
                next_stage = state_json.get("next_stage") or {}
                knowledge_gaps = profile.get("knowledge_gaps") or []
                phase = state_json.get("phase", "diagnosing")
                status = row["status"]

                # A planned stage is the learner's actual next destination.
                # Surface it in project summaries before it is formally started.
                display_stage = next_stage if phase == "next_stage_ready" and next_stage else stage_info

                # 只有 stage_info 非空（含 block_name 或 stage_id）才算有阶段，否则 stage_num=0
                has_real_stage = bool(display_stage and (
                    display_stage.get("stage_id") or
                    display_stage.get("block_name") or
                    display_stage.get("subtopic_name")
                ))
                current_stage_num = (display_stage.get("stage_num") if has_real_stage else 0) or 0

                # progress 计算规则：
                # - abandoned: 0
                # - completed: 100
                # - diagnosing（未生成阶段）: 0
                # - 其余: 阶段号粗估（最多80，留20给最终评估）
                if status == "abandoned":
                    progress = 0
                elif status == "completed":
                    progress = 100
                elif phase == "diagnosing" or current_stage_num == 0:
                    progress = 0
                else:
                    progress = min(current_stage_num * 20, 80)

                # current_stage_name: 只在有真实阶段时填
                current_stage_name = display_stage.get("block_name", "") if has_real_stage else ""

                result.append({
                    "session_id": row["session_id"],
                    "title": state_json.get("title", ""),
                    "status": status,
                    "phase": phase,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    # profile_summary：账号级"所有学习画像/所有项目"列表
                    "profile_summary": {
                        "goal": profile.get("goal", ""),
                        "grade": profile.get("grade", ""),
                        "skill_tags": profile.get("skill_tags", []),
                        "knowledge_gaps": [
                            {
                                "skill": gap.get("skill", ""),
                                "status": gap.get("status", "pending"),
                                "progress": progress,
                            }
                            for gap in knowledge_gaps
                            if isinstance(gap, dict)
                        ] if knowledge_gaps else [],
                        "daily_hours": profile.get("daily_hours"),
                        "target_total_hours": row["target_total_hours"] or 0.0,
                        "target_daily_hours": row["target_daily_hours"] or profile.get("daily_hours") or 2.0,
                        "learning_style": profile.get("learning_style", ""),
                        "version": profile.get("version", 1),
                        "current_stage_num": current_stage_num,
                        "current_stage_name": current_stage_name,
                        "progress": progress,
                    },
                })
            return result

    def save_profile(self, session_id: str, profile: dict) -> None:
        """Split the compatibility profile DTO into user and target records."""
        user_id = self.get_session_user_id(session_id)
        self.upsert_user_profile(user_id, profile)
        with self.Session() as session:
            session.execute(text("""
                UPDATE sessions
                SET goal = :goal,
                    target_knowledge_gaps = :knowledge_gaps,
                    target_daily_hours = :daily_hours,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = :session_id
            """), {
                "session_id": session_id,
                "goal": profile.get("goal", ""),
                "knowledge_gaps": json.dumps(profile.get("knowledge_gaps", []), ensure_ascii=False),
                "daily_hours": profile.get("daily_hours", 2.0),
            })
            session.commit()
        self.refresh_target_metrics(session_id)

    def save_stage(self, session_id: str, stage_info: dict) -> None:
        with self.Session() as session:
            session.execute(text("""
                INSERT OR REPLACE INTO stage_progress (
                    stage_id, session_id, stage_num, block_id, block_name, subtopic_id,
                    estimated_hours, status, accuracy, strategy_used
                ) VALUES (
                    :stage_id, :session_id, :stage_num, :block_id, :block_name, :subtopic_id,
                    :estimated_hours, :status, :accuracy, :strategy_used
                )
            """), {
                "stage_id": stage_info.get("stage_id", str(uuid.uuid4())),
                "session_id": session_id,
                "stage_num": stage_info.get("stage_num", 1),
                "block_id": stage_info.get("block_id", ""),
                "block_name": stage_info.get("block_name", ""),
                "subtopic_id": stage_info.get("subtopic_id"),
                "estimated_hours": stage_info.get("estimated_hours", 1.0),
                "status": stage_info.get("status", "pending"),
                "accuracy": stage_info.get("accuracy"),
                "strategy_used": stage_info.get("strategy_used")
            })
            session.commit()

    def init_subtopic_status(self, session_id: str) -> None:
        self.ensure_user_subtopic_status(self.get_session_user_id(session_id))

    def ensure_user_subtopic_status(self, user_id: str = DEFAULT_USER_ID) -> None:
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT subtopic_id, block_id
                FROM knowledge_subtopics
            """)).fetchall()
            for subtopic_id, block_id in rows:
                session.execute(text("""
                    INSERT OR IGNORE INTO user_subtopic_status (
                        user_id, subtopic_id, block_id, status, mastery_score,
                        is_initial_foundation, attempt_count, updated_at
                    ) VALUES (
                        :user_id, :subtopic_id, :block_id, -1, 0.0, 0, 0, CURRENT_TIMESTAMP
                    )
                """), {
                    "user_id": user_id,
                    "subtopic_id": subtopic_id,
                    "block_id": block_id,
                })
            session.commit()

    def ensure_subtopic_status(self, session_id: str) -> None:
        self.init_subtopic_status(session_id)

    def get_subtopic_status(self, session_id: str, subtopic_id: str) -> dict | None:
        user_id = self.get_session_user_id(session_id)
        self.ensure_user_subtopic_status(user_id)
        with self.Session() as session:
            row = session.execute(text("""
                SELECT user_id, subtopic_id, block_id, status, mastery_score,
                       is_initial_foundation, last_accuracy, attempt_count,
                       updated_at, started_at, completed_at
                FROM user_subtopic_status
                WHERE user_id = :user_id
                  AND subtopic_id = :subtopic_id
            """), {
                "user_id": user_id,
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
        is_initial_foundation: bool | None = None,
    ) -> None:
        user_id = self.get_session_user_id(session_id)
        self.ensure_user_subtopic_status(user_id)
        existing = self.get_subtopic_status(session_id, subtopic_id)
        if not existing:
            return
        attempt_count = int(existing.get("attempt_count") or 0) + (1 if increment_attempt else 0)
        with self.Session() as session:
            session.execute(text("""
                UPDATE user_subtopic_status
                SET status = :status,
                    mastery_score = COALESCE(:mastery_score, mastery_score),
                    last_accuracy = COALESCE(:last_accuracy, last_accuracy),
                    attempt_count = :attempt_count,
                    is_initial_foundation = COALESCE(:is_initial_foundation, is_initial_foundation),
                    started_at = CASE
                        WHEN :status = 0 AND started_at IS NULL THEN CURRENT_TIMESTAMP
                        ELSE started_at
                    END,
                    completed_at = CASE
                        WHEN :status = 1 THEN CURRENT_TIMESTAMP
                        ELSE completed_at
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :user_id
                  AND subtopic_id = :subtopic_id
            """), {
                "user_id": user_id,
                "subtopic_id": subtopic_id,
                "status": status,
                "mastery_score": mastery_score,
                "last_accuracy": last_accuracy,
                "attempt_count": attempt_count,
                "is_initial_foundation": is_initial_foundation,
            })
            session.commit()

    def set_user_subtopic_status(
        self,
        user_id: str,
        subtopic_id: str,
        *,
        status: int,
        mastery_score: float,
        is_initial_foundation: bool | None = None,
        attempt_count: int | None = None,
    ) -> dict:
        self.ensure_user_subtopic_status(user_id)
        with self.Session() as session:
            result = session.execute(text("""
                UPDATE user_subtopic_status
                SET status = :status,
                    mastery_score = :mastery_score,
                    is_initial_foundation = COALESCE(:is_initial_foundation, is_initial_foundation),
                    attempt_count = COALESCE(:attempt_count, attempt_count),
                    completed_at = CASE WHEN :status = 1 THEN CURRENT_TIMESTAMP ELSE completed_at END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :user_id AND subtopic_id = :subtopic_id
            """), {
                "user_id": user_id,
                "subtopic_id": subtopic_id,
                "status": status,
                "mastery_score": mastery_score,
                "is_initial_foundation": is_initial_foundation,
                "attempt_count": attempt_count,
            })
            if result.rowcount == 0:
                raise ValueError(f"Subtopic not found: {subtopic_id}")
            session.commit()
        return self.get_user_subtopic_status(user_id, subtopic_id) or {}

    def replace_user_initial_foundation(self, user_id: str, mastered_subtopic_ids: list[str]) -> None:
        """Replace only the mastery records that came from the user's initial foundation."""
        self.ensure_user_subtopic_status(user_id)
        selected = set(mastered_subtopic_ids)
        with self.Session() as session:
            known_ids = {
                str(row[0])
                for row in session.execute(text("SELECT subtopic_id FROM knowledge_subtopics")).all()
            }
            unknown_ids = selected - known_ids
            if unknown_ids:
                raise ValueError(f"Subtopic not found: {sorted(unknown_ids)[0]}")

            session.execute(text("""
                UPDATE user_subtopic_status
                SET status = CASE WHEN attempt_count > 0 THEN status ELSE -1 END,
                    mastery_score = CASE WHEN attempt_count > 0 THEN mastery_score ELSE 0 END,
                    is_initial_foundation = 0,
                    completed_at = CASE WHEN attempt_count > 0 THEN completed_at ELSE NULL END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :user_id
                  AND is_initial_foundation = 1
            """), {"user_id": user_id})
            if selected:
                session.execute(text("""
                    UPDATE user_subtopic_status
                    SET status = 1,
                        mastery_score = 1,
                        is_initial_foundation = 1,
                        completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = :user_id
                      AND subtopic_id IN :subtopic_ids
                """).bindparams(bindparam("subtopic_ids", expanding=True)), {
                    "user_id": user_id,
                    "subtopic_ids": sorted(selected),
                })
            session.commit()

    def get_user_subtopic_status(self, user_id: str, subtopic_id: str) -> dict | None:
        self.ensure_user_subtopic_status(user_id)
        with self.Session() as session:
            row = session.execute(text("""
                SELECT user_id, subtopic_id, block_id, status, mastery_score,
                       is_initial_foundation, attempt_count, last_accuracy,
                       updated_at, started_at, completed_at
                FROM user_subtopic_status
                WHERE user_id = :user_id AND subtopic_id = :subtopic_id
            """), {"user_id": user_id, "subtopic_id": subtopic_id}).mappings().first()
        return dict(row) if row else None

    def get_completed_subtopic_ids(self, session_id: str, block_id: str) -> list[str]:
        user_id = self.get_session_user_id(session_id)
        self.ensure_user_subtopic_status(user_id)
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT subtopic_id
                FROM user_subtopic_status
                WHERE user_id = :user_id
                  AND block_id = :block_id
                  AND status = 1
            """), {
                "user_id": user_id,
                "block_id": block_id,
            }).fetchall()
        return [row[0] for row in rows if row and row[0]]

    def list_subtopic_statuses(self, session_id: str) -> list[dict]:
        return self.list_user_subtopic_statuses(self.get_session_user_id(session_id))

    def list_user_subtopic_statuses(self, user_id: str = DEFAULT_USER_ID) -> list[dict]:
        self.ensure_user_subtopic_status(user_id)
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT s.subtopic_id, s.block_id, s.status, s.mastery_score, s.last_accuracy,
                       s.is_initial_foundation, s.attempt_count,
                       k.subtopic_name, k.estimated_hours, k.sequence_index,
                       k.difficulty, k.prerequisites, k.content_summary
                FROM user_subtopic_status s
                JOIN knowledge_subtopics k ON s.subtopic_id = k.subtopic_id
                WHERE s.user_id = :user_id
                ORDER BY k.block_id ASC, k.sequence_index ASC
            """), {"user_id": user_id}).mappings().all()
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

        # Stage planning must stay inside the learner's selected knowledge
        # block.  Sorting alone allowed an unrelated block to be selected when
        # the preferred block had no currently-unlocked subtopic.
        if preferred_block_id:
            candidates = [item for item in candidates if item["block_id"] == preferred_block_id]
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

    def list_qa_records_for_stage(self, session_id: str, stage: int) -> list[dict]:
        """Return a stage-local dialogue history for legacy session recovery."""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT question, answer, tech_pass, quality_pass, created_at
                FROM user_qa
                WHERE session_id = :session_id AND stage = :stage
                ORDER BY created_at ASC, qa_id ASC
            """), {"session_id": session_id, "stage": stage}).mappings().all()
        return [dict(row) for row in rows]

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

    # ========================================================================
    # 测试服务（AssessmentService）落库接口（Agent46_API接口任务说明.md 第 2 节）
    # 约束：独立 attempt_id / 开始冻结题目 / 单题只存一条最终答案 /
    #      未答完必答题不能提交 / 提交后不能改 / 只从已提交完整 attempt 算正确率。
    # 本层只做落库与查询，判题与生命周期约束由 AssessmentService 负责。
    # ========================================================================

    def create_assessment_attempt(
        self,
        session_id: str,
        frozen_questions: list[dict],
        stage_id: str | None = None,
        stage_num: int | None = None,
        block_id: str | None = None,
        subtopic_id: str | None = None,
    ) -> str:
        """创建一次测试并冻结题目快照，返回独立 attempt_id。"""
        attempt_id = f"att_{uuid.uuid4().hex[:16]}"
        with self.Session() as session:
            session.execute(text("""
                INSERT INTO assessment_attempts (
                    attempt_id, session_id, stage_id, stage_num, block_id, subtopic_id,
                    status, total_questions, answered_count
                ) VALUES (
                    :attempt_id, :session_id, :stage_id, :stage_num, :block_id, :subtopic_id,
                    'in_progress', :total_questions, 0
                )
            """), {
                "attempt_id": attempt_id,
                "session_id": session_id,
                "stage_id": stage_id,
                "stage_num": stage_num,
                "block_id": block_id,
                "subtopic_id": subtopic_id,
                "total_questions": len(frozen_questions),
            })
            for index, question in enumerate(frozen_questions):
                options = question.get("options")
                session.execute(text("""
                    INSERT INTO assessment_attempt_questions (
                        attempt_id, question_id, sequence_index, block_id, subtopic_id,
                        difficulty, question_type, question_text, options, correct_answer,
                        explanation, hint, is_required
                    ) VALUES (
                        :attempt_id, :question_id, :sequence_index, :block_id, :subtopic_id,
                        :difficulty, :question_type, :question_text, :options, :correct_answer,
                        :explanation, :hint, :is_required
                    )
                """), {
                    "attempt_id": attempt_id,
                    "question_id": question.get("question_id") or f"q_{index}",
                    "sequence_index": index,
                    "block_id": question.get("block_id"),
                    "subtopic_id": question.get("subtopic_id"),
                    "difficulty": question.get("difficulty"),
                    "question_type": question.get("question_type"),
                    "question_text": question.get("question_text", ""),
                    "options": json.dumps(options, ensure_ascii=False) if options is not None else None,
                    "correct_answer": question.get("correct_answer", ""),
                    "explanation": question.get("explanation"),
                    "hint": question.get("hint"),
                    "is_required": bool(question.get("is_required", True)),
                })
            session.commit()
        return attempt_id

    def get_assessment_attempt(self, attempt_id: str) -> dict | None:
        """读取 attempt 元信息（不含题目/答案）。"""
        with self.Session() as session:
            row = session.execute(text("""
                SELECT attempt_id, session_id, stage_id, stage_num, block_id, subtopic_id,
                       status, total_questions, answered_count, correct_count, accuracy,
                       created_at, submitted_at
                FROM assessment_attempts
                WHERE attempt_id = :attempt_id
            """), {"attempt_id": attempt_id}).mappings().first()
        return dict(row) if row else None

    def get_in_progress_attempt(
        self,
        session_id: str,
        stage_id: str | None = None,
    ) -> dict | None:
        """查同一 session（可选叠加 stage）当前未提交的 attempt（取最近一条）。

        用于 create_attempt 幂等复用：同一 session+stage 未正式交卷前只保留一份
        in_progress attempt，避免前端双击/刷新重复创建多份未完成试卷。
        复用现有 idx_assessment_attempts_session(session_id, status) 索引，只读查询。
        """
        params: dict = {"session_id": session_id}
        stage_filter = ""
        if stage_id:
            stage_filter = " AND stage_id = :stage_id"
            params["stage_id"] = stage_id
        with self.Session() as session:
            row = session.execute(text(f"""
                SELECT attempt_id, session_id, stage_id, stage_num, block_id, subtopic_id,
                       status, total_questions, answered_count, correct_count, accuracy,
                       created_at, submitted_at
                FROM assessment_attempts
                WHERE session_id = :session_id AND status = 'in_progress'{stage_filter}
                ORDER BY created_at DESC
                LIMIT 1
            """), params).mappings().first()
        return dict(row) if row else None

    def get_assessment_attempt_questions(self, attempt_id: str) -> list[dict]:
        """读取冻结题目快照（含正确答案/解析，供提交后公布使用）。"""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT question_id, sequence_index, block_id, subtopic_id, difficulty,
                       question_type, question_text, options, correct_answer, explanation,
                       hint, is_required
                FROM assessment_attempt_questions
                WHERE attempt_id = :attempt_id
                ORDER BY sequence_index ASC
            """), {"attempt_id": attempt_id}).mappings().all()
        results = []
        for row in rows:
            item = dict(row)
            item["options"] = json.loads(item["options"]) if item["options"] else None
            item["is_required"] = bool(item["is_required"])
            results.append(item)
        return results

    def get_assessment_answers(self, attempt_id: str) -> dict[str, dict]:
        """读取当前已保存答案，按 question_id 索引（每题一条）。"""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT question_id, user_answer, is_correct, time_spent, updated_at
                FROM assessment_answers
                WHERE attempt_id = :attempt_id
            """), {"attempt_id": attempt_id}).mappings().all()
        return {row["question_id"]: dict(row) for row in rows}

    def upsert_assessment_answer(
        self,
        attempt_id: str,
        question_id: str,
        user_answer: str,
        is_correct: bool | None = None,
        time_spent: int = 0,
    ) -> None:
        """保存/覆盖单题答案（每题只保留一条最终答案），并刷新 answered_count。"""
        with self.Session() as session:
            session.execute(text("""
                INSERT INTO assessment_answers (attempt_id, question_id, user_answer, is_correct, time_spent, updated_at)
                VALUES (:attempt_id, :question_id, :user_answer, :is_correct, :time_spent, CURRENT_TIMESTAMP)
                ON CONFLICT(attempt_id, question_id) DO UPDATE SET
                    user_answer = excluded.user_answer,
                    is_correct = excluded.is_correct,
                    time_spent = excluded.time_spent,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                "attempt_id": attempt_id,
                "question_id": question_id,
                "user_answer": user_answer,
                "is_correct": is_correct,
                "time_spent": time_spent,
            })
            session.execute(text("""
                UPDATE assessment_attempts
                SET answered_count = (
                    SELECT COUNT(*) FROM assessment_answers
                    WHERE attempt_id = :attempt_id
                      AND user_answer IS NOT NULL AND TRIM(user_answer) != ''
                )
                WHERE attempt_id = :attempt_id
            """), {"attempt_id": attempt_id})
            session.commit()

    def finalize_assessment_attempt(
        self,
        attempt_id: str,
        graded_answers: list[dict],
        correct_count: int,
        accuracy: float,
    ) -> None:
        """提交测试：写回逐题判题结果、置 status=submitted、记录正确率。"""
        with self.Session() as session:
            for graded in graded_answers:
                session.execute(text("""
                    INSERT INTO assessment_answers (attempt_id, question_id, user_answer, is_correct, time_spent, updated_at)
                    VALUES (:attempt_id, :question_id, :user_answer, :is_correct, :time_spent, CURRENT_TIMESTAMP)
                    ON CONFLICT(attempt_id, question_id) DO UPDATE SET
                        user_answer = excluded.user_answer,
                        is_correct = excluded.is_correct,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "attempt_id": attempt_id,
                    "question_id": graded["question_id"],
                    "user_answer": graded.get("user_answer"),
                    "is_correct": graded.get("is_correct"),
                    "time_spent": graded.get("time_spent", 0),
                })
            session.execute(text("""
                UPDATE assessment_attempts
                SET status = 'submitted',
                    correct_count = :correct_count,
                    accuracy = :accuracy,
                    answered_count = total_questions,
                    submitted_at = CURRENT_TIMESTAMP
                WHERE attempt_id = :attempt_id
            """), {
                "attempt_id": attempt_id,
                "correct_count": correct_count,
                "accuracy": accuracy,
            })
            session.commit()

    def get_submitted_attempt_accuracy(
        self,
        session_id: str,
        stage_id: str | None = None,
    ) -> dict | None:
        """只从已提交的完整 attempt 计算正确率（取最近一次提交）。

        约束（第 2 节）：正确率不能从零散答题记录计算，必须来自已提交 attempt。
        """
        params: dict = {"session_id": session_id}
        stage_filter = ""
        if stage_id:
            stage_filter = " AND stage_id = :stage_id"
            params["stage_id"] = stage_id
        with self.Session() as session:
            row = session.execute(text(f"""
                SELECT attempt_id, accuracy, correct_count, total_questions, submitted_at
                FROM assessment_attempts
                WHERE session_id = :session_id AND status = 'submitted'{stage_filter}
                ORDER BY submitted_at DESC
                LIMIT 1
            """), params).mappings().first()
        return dict(row) if row else None

    def get_submitted_attempt_records(
        self,
        session_id: str,
        stage_id: str | None = None,
    ) -> dict | None:
        """读取最近一次已提交 attempt 的「逐题判题记录（含耗时）」，供 Agent6 决策使用。

        - 定位口径与 get_submitted_attempt_accuracy 一致：阶段维度优先，退回会话维度，
          均取最近一次已提交（submitted_at DESC）的完整 attempt（约束 7）。
        - JOIN 冻结题目快照与逐题答案，输出每题 question_id / skill(=subtopic_id) /
          question_type / is_correct / time_spent，供 BKT Wt 修正与阶段总结用时统计使用。
        - 返回 {attempt_id, accuracy, question_records: [...], total_time_spent(秒)}；
          没有已提交 attempt 时返回 None（调用方退化为交互耗时估算）。
        """
        attempt = self.get_submitted_attempt_accuracy(session_id, stage_id=stage_id)
        # 阶段维度无已提交 attempt 时，退回会话维度最近一次
        if attempt is None and stage_id is not None:
            attempt = self.get_submitted_attempt_accuracy(session_id)
        if not attempt:
            return None

        attempt_id = attempt["attempt_id"]
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT q.question_id      AS question_id,
                       q.subtopic_id      AS skill,
                       q.question_type    AS question_type,
                       a.is_correct       AS is_correct,
                       a.time_spent       AS time_spent
                FROM assessment_attempt_questions q
                LEFT JOIN assessment_answers a
                    ON a.attempt_id = q.attempt_id AND a.question_id = q.question_id
                WHERE q.attempt_id = :attempt_id
                ORDER BY q.sequence_index ASC
            """), {"attempt_id": attempt_id}).mappings().all()

        question_records: list[dict] = []
        total_time_spent = 0
        for row in rows:
            time_spent = int(row["time_spent"] or 0)
            total_time_spent += time_spent
            question_records.append({
                "question_id": row["question_id"],
                "skill": row["skill"],
                "question_type": row["question_type"],
                "is_correct": bool(row["is_correct"]) if row["is_correct"] is not None else False,
                "time_spent": time_spent,
            })
        return {
            "attempt_id": attempt_id,
            "accuracy": attempt.get("accuracy"),
            "question_records": question_records,
            "total_time_spent": total_time_spent,
        }

    def list_assessment_attempts(self, session_id: str) -> list[dict]:
        """列出会话下的所有 attempt（供仪表盘正确率趋势聚合）。"""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT attempt_id, stage_id, stage_num, block_id, status,
                       total_questions, correct_count, accuracy, created_at, submitted_at
                FROM assessment_attempts
                WHERE session_id = :session_id
                ORDER BY created_at ASC
            """), {"session_id": session_id}).mappings().all()
        return [dict(row) for row in rows]

    # ========================================================================
    # 仪表盘聚合查询（Agent46_API接口任务说明.md 第 4 节）
    # 后端直接从 user_profiles / stage_progress / subtopic_status /
    # assessment_attempts / user_qa 聚合，不属于 Agent4 运行职责。
    # ========================================================================

    def get_latest_profile(self, session_id: str) -> dict | None:
        """Compose the legacy profile DTO from normalized user and target data."""
        with self.Session() as session:
            row = session.execute(text("""
                SELECT session_id, user_id, goal, target_knowledge_gaps,
                       target_daily_hours, created_at, updated_at
                FROM sessions WHERE session_id = :session_id
            """), {"session_id": session_id}).mappings().first()
        if not row:
            return None
        user_profile = self.get_user_profile(row["user_id"] or DEFAULT_USER_ID)
        if not user_profile:
            return None
        return {
            "profile_id": f"profile_{row['user_id']}",
            "session_id": session_id,
            "user_id": row["user_id"] or DEFAULT_USER_ID,
            "version": user_profile.get("version", 1),
            "grade": user_profile.get("grade", ""),
            "skill_tags": user_profile.get("skill_tags", []),
            "foundation_confirmed": bool(user_profile.get("foundation_confirmed", False)),
            "knowledge_gaps": json.loads(row["target_knowledge_gaps"] or "[]"),
            "goal": row["goal"] or "",
            "daily_hours": row["target_daily_hours"] or user_profile.get("default_daily_hours", 2.0),
            "learning_style": user_profile.get("learning_style", "mixed"),
            "hardware": user_profile.get("hardware", ""),
            "learning_rhythm": user_profile.get("learning_rhythm", "medium"),
            "confidence_index": user_profile.get("confidence_index", 0.5),
            "confusion_markers": user_profile.get("confusion_markers", []),
            "confidence": user_profile.get("confidence", 0.7),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_stage_progress(self, session_id: str) -> list[dict]:
        """读取会话所有阶段记录（供当前阶段摘要与阶段历史聚合）。"""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT stage_id, stage_num, block_id, block_name, subtopic_id, estimated_hours,
                       status, accuracy, strategy_used, started_at, completed_at
                FROM stage_progress
                WHERE session_id = :session_id
                ORDER BY stage_num ASC
            """), {"session_id": session_id}).mappings().all()
        return [dict(row) for row in rows]

    def get_recent_qa(self, session_id: str, limit: int = 5) -> list[dict]:
        """读取最近问答记录（供仪表盘 recent_qa）。"""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT qa_id, stage, question, answer, tech_pass, quality_pass, created_at
                FROM user_qa
                WHERE session_id = :session_id
                ORDER BY created_at DESC, qa_id DESC
                LIMIT :limit
            """), {"session_id": session_id, "limit": limit}).mappings().all()
        return [dict(row) for row in rows]

    def get_subtopic_mastery_counts(self, session_id: str) -> tuple[int, int]:
        """统计已掌握子知识点数量与总数（mastered_count / total_count）。"""
        user_id = self.get_session_user_id(session_id)
        self.ensure_user_subtopic_status(user_id)
        with self.Session() as session:
            row = session.execute(text("""
                SELECT
                    COALESCE(SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END), 0) AS mastered,
                    COUNT(*) AS total
                FROM user_subtopic_status
                WHERE user_id = :user_id
            """), {"user_id": user_id}).mappings().first()
        return int(row["mastered"]), int(row["total"])

    def get_block_progress(self, session_id: str) -> list[dict]:
        """按知识块聚合掌握进度（block_progress）。"""
        rows = self.list_subtopic_statuses(session_id)
        blocks: dict[str, dict] = {}
        for row in rows:
            block_id = row["block_id"]
            bucket = blocks.setdefault(block_id, {"block_id": block_id, "mastered": 0, "total": 0})
            bucket["total"] += 1
            if int(row["status"]) == 1:
                bucket["mastered"] += 1
        result = []
        for block_id, bucket in blocks.items():
            total = bucket["total"]
            mastered = bucket["mastered"]
            result.append({
                "block_id": block_id,
                "mastered_count": mastered,
                "total_count": total,
                "progress": round(mastered / total, 3) if total else 0.0,
            })
        return result

    # ========================================================================
    # Agent3-4 审核辩论协议 — 落库接口（P2，按 trace_id 组织）
    # source_id / claim_id / canonical_error_id 在单个 trace 内唯一，
    # 所有写入必须携带 trace_id，不允许脱离 trace_id 操作。
    # ========================================================================

    def save_review_trace(
        self,
        trace_id: str,
        session_id: str,
        stage_id: str | None = None,
        target_type: str = "learning_package",
        package_version: int = 1,
        package_status: str = "draft",
        review_mode: str = "debate",
        final_status: str = "in_progress",
        final_summary: str | None = None,
    ) -> None:
        """创建或更新审核 trace（review_traces 表）。"""
        with self.Session() as session:
            session.execute(text("""
                INSERT INTO review_traces
                    (trace_id, session_id, stage_id, target_type, package_version,
                     package_status, review_mode, final_status, final_summary,
                     created_at, updated_at)
                VALUES
                    (:trace_id, :session_id, :stage_id, :target_type, :package_version,
                     :package_status, :review_mode, :final_status, :final_summary,
                     CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(trace_id) DO UPDATE SET
                    package_version = excluded.package_version,
                    package_status = excluded.package_status,
                    final_status = excluded.final_status,
                    final_summary = excluded.final_summary,
                    updated_at = CURRENT_TIMESTAMP
            """), {
                "trace_id": trace_id,
                "session_id": session_id,
                "stage_id": stage_id,
                "target_type": target_type,
                "package_version": package_version,
                "package_status": package_status,
                "review_mode": review_mode,
                "final_status": final_status,
                "final_summary": final_summary,
            })
            session.commit()

    def save_learning_package_sources(
        self,
        trace_id: str,
        sources: list[dict],
    ) -> None:
        """批量写入来源清单（learning_package_sources，复合主键 trace_id+source_id）。"""
        if not sources:
            return
        with self.Session() as session:
            for src in sources:
                session.execute(text("""
                    INSERT INTO learning_package_sources
                        (trace_id, source_id, source_type, title, locator,
                         content_ref, source_version, metadata_json,
                         created_at, updated_at)
                    VALUES
                        (:trace_id, :source_id, :source_type, :title, :locator,
                         :content_ref, :source_version, :metadata_json,
                         CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(trace_id, source_id) DO UPDATE SET
                        source_type = excluded.source_type,
                        title = excluded.title,
                        locator = excluded.locator,
                        content_ref = excluded.content_ref,
                        source_version = excluded.source_version,
                        metadata_json = excluded.metadata_json,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "trace_id": trace_id,
                    "source_id": src.get("source_id", ""),
                    "source_type": src.get("source_type", "knowledge_doc"),
                    "title": src.get("title"),
                    "locator": src.get("locator"),
                    "content_ref": src.get("content_ref"),
                    "source_version": src.get("source_version"),
                    "metadata_json": json.dumps(src.get("metadata_json"), ensure_ascii=False) if src.get("metadata_json") else None,
                })
            session.commit()

    def save_learning_package_claims(
        self,
        trace_id: str,
        claims: list[dict],
    ) -> None:
        """批量写入 claim 声明（learning_package_claims，复合主键 trace_id+claim_id）。"""
        if not claims:
            return
        with self.Session() as session:
            for clm in claims:
                session.execute(text("""
                    INSERT INTO learning_package_claims
                        (trace_id, claim_id, text, source_ref, source_span,
                         section_ref, claim_type, is_key_technical, confidence,
                         status, created_at, updated_at)
                    VALUES
                        (:trace_id, :claim_id, :text, :source_ref, :source_span,
                         :section_ref, :claim_type, :is_key_technical, :confidence,
                         :status, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(trace_id, claim_id) DO UPDATE SET
                        text = excluded.text,
                        source_ref = excluded.source_ref,
                        source_span = excluded.source_span,
                        section_ref = excluded.section_ref,
                        claim_type = excluded.claim_type,
                        is_key_technical = excluded.is_key_technical,
                        confidence = excluded.confidence,
                        status = excluded.status,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "trace_id": trace_id,
                    "claim_id": clm.get("claim_id", ""),
                    "text": clm.get("text", ""),
                    "source_ref": clm.get("source_ref"),
                    "source_span": clm.get("source_span"),
                    "section_ref": clm.get("section_ref", ""),
                    "claim_type": clm.get("claim_type", "concept"),
                    "is_key_technical": 1 if clm.get("is_key_technical") else 0,
                    "confidence": clm.get("confidence", 0.5),
                    "status": clm.get("status", "active"),
                })
            session.commit()

    def save_review_debate_issues(
        self,
        trace_id: str,
        issues: list[dict],
    ) -> None:
        """批量写入争议单（review_debate_issues，主键 issue_id，UNIQUE trace_id+canonical_error_id）。

        source_refs / change_request 为列表/字典，序列化为 JSON 存储。
        """
        if not issues:
            return
        with self.Session() as session:
            for iss in issues:
                session.execute(text("""
                    INSERT INTO review_debate_issues
                        (issue_id, trace_id, canonical_error_id, error_id, claim_id,
                         error_type, severity, must_fix, must_fix_reason,
                         description, location, evidence, suggestion,
                         source_refs, change_request, current_status,
                         opened_in_round_num, created_at, updated_at)
                    VALUES
                        (:issue_id, :trace_id, :canonical_error_id, :error_id, :claim_id,
                         :error_type, :severity, :must_fix, :must_fix_reason,
                         :description, :location, :evidence, :suggestion,
                         :source_refs, :change_request, :current_status,
                         :opened_in_round_num, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(trace_id, canonical_error_id) DO UPDATE SET
                        error_id = excluded.error_id,
                        must_fix = excluded.must_fix,
                        must_fix_reason = excluded.must_fix_reason,
                        current_status = excluded.current_status,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "issue_id": iss.get("issue_id") or f"iss_{trace_id}_{iss.get('canonical_error_id', '')}",
                    "trace_id": trace_id,
                    "canonical_error_id": iss.get("canonical_error_id", ""),
                    "error_id": iss.get("error_id", ""),
                    "claim_id": iss.get("claim_id"),
                    "error_type": iss.get("error_type"),
                    "severity": iss.get("severity", "major"),
                    "must_fix": 1 if iss.get("must_fix") else 0,
                    "must_fix_reason": iss.get("must_fix_reason"),
                    "description": iss.get("description"),
                    "location": iss.get("location"),
                    "evidence": iss.get("evidence"),
                    "suggestion": iss.get("suggestion"),
                    "source_refs": json.dumps(iss.get("source_refs", []), ensure_ascii=False) if iss.get("source_refs") else None,
                    "change_request": json.dumps(iss.get("change_request"), ensure_ascii=False) if iss.get("change_request") else None,
                    "current_status": iss.get("current_status", "open"),
                    "opened_in_round_num": iss.get("opened_in_round_num"),
                })
            session.commit()

    def save_review_debate_round(
        self,
        trace_id: str,
        round_id: str,
        round_num: int,
        review_iteration: int | None = None,
        round_status: str = "open",
        orchestrator_verdict: str = "pending",
        verdict_summary: str | None = None,
    ) -> None:
        """写入答辩复核轮次（review_debate_rounds，UNIQUE trace_id+round_num）。"""
        with self.Session() as session:
            session.execute(text("""
                INSERT INTO review_debate_rounds
                    (round_id, trace_id, round_num, review_iteration,
                     round_status, orchestrator_verdict, verdict_summary,
                     created_at)
                VALUES
                    (:round_id, :trace_id, :round_num, :review_iteration,
                     :round_status, :orchestrator_verdict, :verdict_summary,
                     CURRENT_TIMESTAMP)
                ON CONFLICT(trace_id, round_num) DO UPDATE SET
                    round_status = excluded.round_status,
                    orchestrator_verdict = excluded.orchestrator_verdict,
                    verdict_summary = excluded.verdict_summary,
                    closed_at = CASE WHEN excluded.round_status = 'closed' THEN CURRENT_TIMESTAMP ELSE NULL END
            """), {
                "round_id": round_id,
                "trace_id": trace_id,
                "round_num": round_num,
                "review_iteration": review_iteration,
                "round_status": round_status,
                "orchestrator_verdict": orchestrator_verdict,
                "verdict_summary": verdict_summary,
            })
            session.commit()

    def save_review_debate_responses(
        self,
        round_id: str,
        responses: list[dict],
    ) -> None:
        """批量写入每轮回复（review_debate_responses，UNIQUE round_id+issue_id+actor+response_type）。

        source_refs / patched_sections 为列表，序列化为 JSON 存储。
        """
        if not responses:
            return
        with self.Session() as session:
            for resp in responses:
                session.execute(text("""
                    INSERT INTO review_debate_responses
                        (response_id, round_id, issue_id, actor, response_type,
                         disposition, rationale, source_refs, patched_sections, created_at)
                    VALUES
                        (:response_id, :round_id, :issue_id, :actor, :response_type,
                         :disposition, :rationale, :source_refs, :patched_sections, CURRENT_TIMESTAMP)
                    ON CONFLICT(round_id, issue_id, actor, response_type) DO UPDATE SET
                        disposition = excluded.disposition,
                        rationale = excluded.rationale,
                        source_refs = excluded.source_refs,
                        patched_sections = excluded.patched_sections
                """), {
                    "response_id": resp.get("response_id") or f"rsp_{round_id}_{resp.get('issue_id', '')}_{resp.get('actor', '')}",
                    "round_id": round_id,
                    "issue_id": resp.get("issue_id", ""),
                    "actor": resp.get("actor", "agent3"),
                    "response_type": resp.get("response_type", "author_reply"),
                    "disposition": resp.get("disposition", "accepted"),
                    "rationale": resp.get("rationale"),
                    "source_refs": json.dumps(resp.get("source_refs", []), ensure_ascii=False) if resp.get("source_refs") else None,
                    "patched_sections": json.dumps(resp.get("patched_sections", []), ensure_ascii=False) if resp.get("patched_sections") else None,
                })
            session.commit()

    def update_review_trace_status(
        self,
        trace_id: str,
        final_status: str,
        package_status: str | None = None,
        unresolved_issue_count: int | None = None,
        unresolved_must_fix_count: int | None = None,
        final_summary: str | None = None,
    ) -> None:
        """更新 trace 最终状态（超轮次兜底 / 审核通过时调用）。"""
        with self.Session() as session:
            session.execute(text("""
                UPDATE review_traces
                SET final_status = :final_status,
                    package_status = COALESCE(:package_status, package_status),
                    unresolved_issue_count = COALESCE(:unresolved_issue_count, unresolved_issue_count),
                    unresolved_must_fix_count = COALESCE(:unresolved_must_fix_count, unresolved_must_fix_count),
                    final_summary = COALESCE(:final_summary, final_summary),
                    updated_at = CURRENT_TIMESTAMP,
                    closed_at = CASE WHEN :final_status IN ('passed', 'needs_human_review', 'failed') THEN CURRENT_TIMESTAMP ELSE closed_at END
                WHERE trace_id = :trace_id
            """), {
                "trace_id": trace_id,
                "final_status": final_status,
                "package_status": package_status,
                "unresolved_issue_count": unresolved_issue_count,
                "unresolved_must_fix_count": unresolved_must_fix_count,
                "final_summary": final_summary,
            })
            session.commit()

    def update_review_debate_issue_status(
        self,
        issue_id: str,
        current_status: str | None = None,
        final_disposition: str | None = None,
        resolved_in_round_num: int | None = None,
    ) -> None:
        """更新争议单状态（裁决后调用）。"""
        with self.Session() as session:
            session.execute(text("""
                UPDATE review_debate_issues
                SET current_status = COALESCE(:current_status, current_status),
                    final_disposition = COALESCE(:final_disposition, final_disposition),
                    resolved_in_round_num = COALESCE(:resolved_in_round_num, resolved_in_round_num),
                    updated_at = CURRENT_TIMESTAMP
                WHERE issue_id = :issue_id
            """), {
                "issue_id": issue_id,
                "current_status": current_status,
                "final_disposition": final_disposition,
                "resolved_in_round_num": resolved_in_round_num,
            })
            session.commit()
