# AI Agent Trainer - 多Agent个性化辅导系统

一个基于多Agent协同的AI个性化学习辅导系统，专注于大模型微调与推理部署领域的教学。

## 核心特性

- **个性化画像**：通过对话式交互生成用户学习画像
- **阶段式学习**：逐阶段推送学习内容，动态调整学习路径
- **实时答疑**：学习过程中随时提问，系统即时回答
- **双重审核**：技术审核+质量评审，确保学习内容准确性和教学效果
- **策略调整**：根据测试结果自动调整下一阶段学习策略
- **可视化报告**：实时展示学习进度和能力变化

## 技术栈

- **后端**：FastAPI, Python 3.12+
- **数据库**：SQLite, Chroma Vector DB, NetworkX
- **LLM**：Ollma本地模型 (deepseek-r1/ qwen2.5)
- **前端**：React + TypeScript

后端当前默认使用 `EXECUTION_MODE=deterministic`，无需启动 Ollama 即可跑通诊断、阶段规划、学习包、答题和结业闭环。

## 项目结构

```
project-root/
├── backend/          # 后端服务
├── frontend/         # 前端界面
├── data/             # 数据存储
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
python scripts/init_knowledge_base.py
python scripts/build_knowledge_graph.py
python scripts/seed_test_data.py
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
2. Agent2生成下一阶段内容
3. Agent3生成学习包 → Agent4+5并行审核
4. 用户学习 + 实时答疑
5. 完成测试 → Agent6更新画像
6. 循环直到所有知识盲区掌握
