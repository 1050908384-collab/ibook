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
            self._ensure_column(conn, "knowledge_docs", "subtopic_id", "VARCHAR(50)")
            self._migrate_test_questions_table(conn)
            self._create_assessment_tables(conn)
            self._migrate_review_protocol_tables(conn)
            self._migrate_learning_package_source_types(conn)
            conn.commit()

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
                source_type VARCHAR(30) CHECK (source_type IN ('knowledge_doc', 'faq', 'best_practice', 'manual', 'external_note', 'code_example', 'video')) NOT NULL,
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
                        source_type VARCHAR(30) CHECK (source_type IN ('knowledge_doc', 'faq', 'best_practice', 'manual', 'external_note', 'code_example', 'video')) NOT NULL,
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
                SELECT question_id, block_id, subtopic_id, difficulty, question_type,
                       question_text, options, correct_answer, explanation,
                       expected_output, grading_mode, source, hint
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
        """读取最新版本的用户画像（version 最大者）。"""
        with self.Session() as session:
            row = session.execute(text("""
                SELECT profile_id, session_id, version, skill_tags, knowledge_gaps, goal,
                       daily_hours, learning_style, learning_rhythm, confidence_index,
                       confusion_markers, confidence, grade, hardware, created_at
                FROM user_profiles
                WHERE session_id = :session_id
                ORDER BY version DESC
                LIMIT 1
            """), {"session_id": session_id}).mappings().first()
        if not row:
            return None
        result = dict(row)
        for field in ("skill_tags", "knowledge_gaps", "confusion_markers"):
            result[field] = json.loads(result[field]) if result[field] else []
        return result

    def list_stage_progress(self, session_id: str) -> list[dict]:
        """读取会话所有阶段记录（供当前阶段摘要与阶段历史聚合）。"""
        with self.Session() as session:
            rows = session.execute(text("""
                SELECT stage_id, stage_num, block_id, block_name, subtopic_id,
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
        self.ensure_subtopic_status(session_id)
        with self.Session() as session:
            row = session.execute(text("""
                SELECT
                    COALESCE(SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END), 0) AS mastered,
                    COUNT(*) AS total
                FROM subtopic_status
                WHERE session_id = :session_id
            """), {"session_id": session_id}).mappings().first()
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
