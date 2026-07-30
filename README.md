# AI Agent Trainer - 多Agent个性化辅导系统

一个基于多Agent协同的AI个性化学习辅导系统，专注于大模型微调与推理部署领域的教学。

## 核心特性

- **个性化画像**：通过对话式交互生成用户学习画像
- **阶段式学习**：逐阶段推送学习内容，动态调整学习路径
- **实时答疑**：学习过程中随时提问，系统即时回答
- **统一审核**：Agent4 合并技术审核与质量评审，确保内容准确且适配用户
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
│   ├── seeds/        # 结构化种子数据
│   ├── sqlite_data/  # 主库 project.db
│   ├── chroma_db/    # FAQ 向量库
│   ├── graph_data/   # 知识图谱结果层
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
python scripts/init_knowledge_base.py
python scripts/build_knowledge_graph.py
python scripts/seed_test_data.py
python scripts/init_faq_vectors.py
```

## 数据库约定

- 当前唯一主 SQLite 库为 `data/sqlite_data/project.db`
- `project.db` 是运行时唯一真相源，包含知识库表、FAQ 表、会话表和 `session_states`
- 历史库 `data/knowledge.db` 不再参与运行链路，仅保留备份归档用途
- 所有后端代码、初始化脚本、FAQ 向量化脚本统一读取 `DATABASE_URL`
- 所有 Chroma 数据统一读取 `CHROMA_PATH`

## FAQ 问答链路
- Agent3 仅在用户提问时调用 FAQ 检索，不在教程生成阶段调用
- FAQ 检索使用“完整问题直查向量库”方案，不先调用 LLM 提取关键词
- FAQ 检索返回候选后，只有最高相似度达到阈值时才直接返回 FAQ 标准答案
- FAQ 未命中或最高相似度低于阈值时，直接回退到知识库检索与 LLM 普通答疑
- FAQ 向量数据来源于 `faq_items`，通过 `python scripts/init_faq_vectors.py` 写入 Chroma 的 `faq_collection`

## PowerShell 调试流程

```powershell
# 1. 创建 Session
$session = Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/session/
$sessionId = $session.session_id
$sessionId

# 2. 初始化用户画像 + 生成当前阶段
$body = "{`"session_id`": `"$sessionId`", `"message`": `"我想学习大模型微调`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/chat/ -Body $bytes -ContentType "application/json; charset=utf-8"

# 3. 启动当前阶段，生成教程并触发 Agent4 统一审核
$body = "{`"session_id`": `"$sessionId`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/stage/start -Body $bytes -ContentType "application/json; charset=utf-8"

# 4. 在当前阶段内提问
$body = "{`"session_id`": `"$sessionId`", `"message`": `"什么是LoRA微调？`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/chat/question -Body $bytes -ContentType "application/json; charset=utf-8"

# 5. 查看当前 Session 状态
Invoke-RestMethod -Method GET -Uri "http://localhost:8000/api/session/$sessionId"

# 6. 如果要提交题目答案
$questionId = "你的题目ID"
$body = "{`"session_id`": `"$sessionId`", `"question_id`": `"$questionId`", `"user_answer`": `"LoRA`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8000/api/stage/submit-answer -Body $bytes -ContentType "application/json; charset=utf-8"

# 7. 完成当前阶段，触发 Agent6 决策
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/api/stage/complete/$sessionId"
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
| Agent3 | 内容生成 | 生成学习包、回答问题、按 `block_id` 抽取测试题 |
| Agent4 | 统一审核 | 同时检查技术正确性与教学质量 |
| Agent6 | 阶段决策 | 更新画像、选择策略、推进下一阶段 |

说明：

- Agent5 已合并进 Agent4，不再作为独立运行代理
- 当前阶段测试题由 Agent3 从 `test_questions` 表中抽取，不由 Agent6 生成

## 学习流程

1. 用户首次输入 → Agent1生成画像
2. Agent2生成下一阶段内容
3. Agent3生成学习包 → Agent4统一审核
4. 用户学习 + 实时答疑
5. 进入阶段测试 → Agent3按当前 `block_id` 抽取 1~3 题
6. 完成测试 → Agent6更新画像
7. 循环直到所有知识盲区掌握

## DAG + mastery state

- 当前阶段推进已经从“板块级串行推进”下沉到“知识点状态推进”
- 运行时真实状态来源于 `subtopic_status`
- 状态语义：`-1` 未掌握，`0` 学习中，`1` 已掌握
- Agent2 依据知识图谱前置关系与 `subtopic_status` 选择下一学习内容
- `knowledge_gaps` 继续保留为板块级画像视图，不作为唯一推进依据
