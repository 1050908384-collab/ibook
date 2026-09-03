# AI Agent Trainer

面向大模型微调与推理部署的个性化实训多智能体协同系统。系统围绕学习者画像、知识点级阶段规划、个性化学习资源生成、专业知识与教学质量审核、阶段测试和动态决策形成学习闭环。

## 运行边界

- 前端、FastAPI 后端、SQLite 主数据库、ChromaDB 向量索引和 NetworkX 知识图谱均在本地运行。
- 系统可按配置调用外部 DeepSeek API；异常时可回退到本地 Ollama；两者均不可用时保留确定性状态处理和明确的人工复核状态。
- 当前运行时包括四个 Agent：学情诊断与阶段规划 Agent、个性化学习资源生成 Agent、专业知识与教学质量审核 Agent、学情评估与动态决策 Agent。
- vLLM、LoRA 和 QLoRA 是系统教授的实训内容，不是本系统当前的运行时生成服务。

## 目录结构

```text
源代码/
├─ backend/                 # FastAPI 后端、四个 Agent、Orchestrator 和工具
├─ frontend/                # React + TypeScript 前端工作台
├─ data/                    # 种子数据、主数据库、向量索引和知识图谱
├─ scripts/                 # 初始化、构图、评测和校验脚本
├─ tests/                   # Python unittest 单元与集成回归测试
├─ requirements.txt         # Python 依赖
└─ README.md                # 本说明
```

完整的安装、模型配置、一键启动和故障排查步骤请阅读上一级目录的《部署说明.md》。

## 快速启动

在本目录执行：

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
cd frontend
npm ci
cd ..
```

复制 `backend/.env.example` 为 `backend/.env`，再按《部署说明.md》配置模型和数据库路径。准备本地模型时，可使用 Ollama：

```bash
ollama pull deepseek-r1:7b
ollama pull qwen3-embedding:0.6b
```

启动后端：

```bash
cd backend
python main.py
```

另开终端启动前端：

```bash
cd frontend
npm run dev
```

默认前端地址为 `http://localhost:5173`，后端健康检查地址为 `http://localhost:8500/health`。提交包也提供上一级目录中的“一键运行（可执行程序）/一键启动.bat”。

## 主要功能

- 学习者画像：记录基础、目标、每日时长、学习风格和硬件条件。
- 阶段规划：依据 15 个知识板块、73 个原子知识点、先修关系和掌握状态选择当前阶段。
- 个性化学习包：围绕当前阶段组织教程、代码、实操指南和测试题。
- 专业审核：检查技术事实、代码、来源、难度、认知负荷和硬件可行性，支持结构化修订与复核。
- 实时答疑：结合当前阶段上下文检索资料，生成回答并经过审核。
- 阶段测试与错题本：提交答案后记录正误和耗时；错题按学习目标和阶段聚合，支持查看题目、答案、解析和关联知识点。
- 动态决策：综合正确率、错题、问答历史和学习行为选择下一阶段策略，并生成阶段总结。

## 数据规模

当前提交版本的核心知识与评测数据包括：15 个知识板块、73 个原子知识点、1539 个知识内容块和 323 道测试题。主数据库为 `data/sqlite_data/project.db`。

## 测试

正式 Python 测试位于 `tests/`，运行：

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

前端状态、知识网络和错题本专项测试位于 `frontend/tests/`，运行：

```bash
npm run test:agent-flow
```

错题本专项测试文件：

- `tests/test_wrong_questions_api.py`
- `frontend/tests/wrongQuestionBook.test.mjs`

`tests/` 是正式测试目录；`scripts/` 中的初始化、评测和调试脚本不等同于单元测试。

## 说明

- 数据库表结构和知识底座说明：`数据库框架.md`。
- 系统架构和 Agent 协同说明：`框架设计.md` 与 `docs/` 目录。
- 运行时配置不得提交真实 API Key；请使用 `backend/.env.example` 创建本地配置文件。
