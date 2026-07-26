# AI Agent Trainer - 多Agent个性化辅导系统

一个基于多Agent协同的AI个性化学习辅导系统，专注于大模型微调与推理部署领域的教学。

## 核心特性

- **个性化画像**：通过对话式交互生成用户学习画像
- **阶段式学习**：按“子知识点 + 每日可学时长”逐阶段推送学习内容，动态调整学习路径
- **实时答疑**：学习过程中随时提问，系统即时回答
- **双重审核**：技术审核+质量评审，确保学习内容准确性和教学效果
- **策略调整**：根据测试结果自动调整下一阶段学习策略
- **可视化报告**：实时展示学习进度和能力变化

## 技术栈

- **后端**：FastAPI, Python 3.12+
- **数据库**：SQLite, Chroma Vector DB, NetworkX
- **LLM**：Ollma本地模型 (deepseek-r1/ qwen2.5)
- **前端**：React + TypeScript

后端默认使用 `EXECUTION_MODE=hybrid`：状态管理、阶段规划和答题判定保持确定性，教程与问答由 Ollama 生成。设置为 `deterministic` 后，无需启动 Ollama 也可以跑通完整闭环。

## 项目结构

```
project-root/
├── backend/          # 后端服务
├── frontend/         # 前端界面
├── data/             # 数据存储
│   ├── seeds/        # 结构化种子数据（5个JSON文件）
│   ├── sqlite_data/  # 主SQLite数据库 project.db
│   ├── chroma_db/    # FAQ/知识检索向量库
│   ├── graph_data/   # 知识依赖图谱数据
│   └── test_data/    # 测试样例数据
├── scripts/          # 工具脚本
└── docs/             # 项目文档
```

## 快速开始

### 环境要求

- Python 3.12+
- Ollma (本地模型推理)
- Node.js 18+ (前端)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件
```

### 初始化数据

```bash
python scripts/init_db.py
python scripts/insert_knowledge_data.py
python scripts/init_faq_vectors.py
```

当前数据导入脚本只读取 `data/seeds/` 下的 5 个结构化文件：

- `knowledge_blocks.json`
- `knowledge_subtopics.json`
- `knowledge_docs.json`
- `faq_items.json`
- `test_questions.json`

## 数据库约定

- 当前唯一主 SQLite 库为 `data/sqlite_data/project.db`
- `project.db` 是运行时唯一真相源，包含知识板块、子知识点、知识文档、FAQ、题库、会话与状态快照
- 历史库 `data/knowledge.db` 不再参与运行链路，仅保留备份归档用途
- 所有后端代码、初始化脚本、FAQ 向量化脚本统一读取 `DATABASE_URL`
- 所有 Chroma 数据统一读取 `CHROMA_PATH`
- Agent2 当前按 `knowledge_subtopics` 的 `estimated_hours` 与 `sequence_index` 规划阶段
- Agent3 当前按 `knowledge_docs` 检索教程与答疑内容；`knowledge_docs.subtopic_id` 用于将文档精确关联到子知识点

## FAQ 问答链路
- Agent3 仅在用户提问时调用 FAQ 检索，不在教程生成阶段调用
- FAQ 检索使用“完整问题直查向量库”方案，不先调用 LLM 提取关键词
- FAQ 检索返回候选后，只有最高相似度达到阈值时才直接返回 FAQ 标准答案
- FAQ 未命中或最高相似度低于阈值时，直接回退到知识库检索与 LLM 普通答疑
- FAQ 向量数据来源于 `faq_items`，通过 `python scripts/init_faq_vectors.py` 写入 Chroma 的 `faq_collection`

## PowerShell 调试流程

```powershell
# 1. 创建 Session
$session = Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/session/
$sessionId = $session.session_id
$sessionId

# 2. 初始化用户画像 + 生成当前阶段
$body = "{`"session_id`": `"$sessionId`", `"message`": `"我想学习大模型微调，我有Python和Linux基础，每天可以学习1小时，偏实操型`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/chat/ -Body $bytes -ContentType "application/json; charset=utf-8"

# 3. 启动当前阶段，生成教程并触发 Agent4/5 审核
$body = "{`"session_id`": `"$sessionId`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/stage/start -Body $bytes -ContentType "application/json; charset=utf-8"

# 4. 在当前阶段内提问
$body = "{`"session_id`": `"$sessionId`", `"message`": `"什么是LoRA微调？`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/chat/question -Body $bytes -ContentType "application/json; charset=utf-8"

# 5. 查看当前 Session 状态
Invoke-RestMethod -Method GET -Uri "http://localhost:8500/api/session/$sessionId"

# 6. 如果要提交题目答案
$questionId = "你的题目ID"
$body = "{`"session_id`": `"$sessionId`", `"question_id`": `"$questionId`", `"user_answer`": `"LoRA`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/stage/submit-answer -Body $bytes -ContentType "application/json; charset=utf-8"

# 7. 完成当前阶段，触发 Agent6 决策
Invoke-RestMethod -Method POST -Uri "http://localhost:8500/api/stage/complete/$sessionId"
```

### 启动服务

```bash
cd backend
python main.py
```

### 运行确定性闭环测试

```bash
python -m unittest discover -s tests -v
```

## Agent架构

| Agent | 职责 | 核心功能 |
|-------|------|----------|
| Agent1 | 学情诊断 | 生成用户画像JSON |
| Agent2 | 目录生成 | 规划下一阶段学习内容 |
| Agent3 | 内容生成 | 生成学习包、回答问题 |
| Agent4 | 技术审核 | 检查幻觉、语法、规范 |
| Agent5 | 质量评审 | 检查难度、负荷、风格 |
| Agent6 | 阶段决策 | 更新画像、选择策略 |

## 学习流程

1. 用户首次输入 → Agent1生成画像
2. Agent2根据知识盲区、子知识点顺序和 `daily_hours` 生成下一阶段
3. Agent3生成学习包 → Agent4+5并行审核
4. 用户学习 + 实时答疑
5. 完成测试 → Agent6更新画像
6. 若当前大板块仍有未完成子知识点，则继续推进下一子知识点
7. 循环直到所有知识盲区掌握
