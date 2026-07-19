import sqlite3
import os

# 获取项目根目录（scripts文件夹的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 如果 data 文件夹不存在，自动创建
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 数据库文件路径：放在 data/knowledge.db
DB_PATH = os.path.join(DATA_DIR, 'knowledge.db')

# 1. 连接到数据库
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 2. 关闭外键约束检查（防止建表顺序报错）
cursor.execute("PRAGMA foreign_keys=OFF;")

# 3. 一次性执行所有建表语句
cursor.executescript('''

-- =============================================
-- 第四层：系统基础数据
-- =============================================

-- 9. 会话管理表
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    current_stage INT DEFAULT 1,
    status VARCHAR(20) CHECK (status IN ('active', 'completed', 'abandoned')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- 第一层：知识库数据
-- =============================================

-- 1. 知识板块清单表
CREATE TABLE IF NOT EXISTS knowledge_blocks (
    block_id VARCHAR(50) PRIMARY KEY,
    block_name VARCHAR(100) NOT NULL,
    description TEXT,
    estimated_hours FLOAT DEFAULT 1.0,
    difficulty VARCHAR(20) CHECK (difficulty IN ('初级', '中级', '高级')),
    prerequisites TEXT,
    corresponding_gaps TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 知识文档表
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
);

-- 3. 文档分块表
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
);

-- 4. 测试题库
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
);

-- =============================================
-- 第二层：用户个性化数据
-- =============================================

-- 5. 用户画像演进表
CREATE TABLE IF NOT EXISTS user_profiles (
    profile_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    version INT DEFAULT 1,
    grade VARCHAR(50),
    goal TEXT,
    daily_hours FLOAT,
    learning_style VARCHAR(20),
    hardware VARCHAR(100),
    skill_tags TEXT,
    knowledge_gaps TEXT,
    learning_rhythm VARCHAR(20),
    confidence_index FLOAT,
    confusion_markers TEXT,
    profile_confidence FLOAT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- 6. 阶段进度表
CREATE TABLE IF NOT EXISTS stage_progress (
    stage_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    stage_num INT NOT NULL,
    block_id VARCHAR(50),
    block_name VARCHAR(100),
    status VARCHAR(20) CHECK (status IN ('pending', 'learning', 'tested', 'completed', 'retreat')),
    accuracy FLOAT,
    strategy_used VARCHAR(20),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- =============================================
-- 第三层：用户交互数据
-- =============================================

-- 7. 用户问答记录表
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
);

-- 8. 用户答题记录表
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
);

''')

# 4. 重新开启外键约束检查
cursor.execute("PRAGMA foreign_keys=ON;")

# 5. 提交事务并关闭连接
conn.commit()
conn.close()

print("✅ 9张表创建成功！")
print(f"📁 数据库文件位置: {DB_PATH}")