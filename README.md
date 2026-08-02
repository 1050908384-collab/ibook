# AI Agent Trainer - 多Agent个性化辅导系统

一个基于多Agent协同的AI个性化学习辅导系统，专注于大模型微调与推理部署领域的教学。

## 核心特性

- **个性化画像**：通过对话式交互生成用户学习画像
- **阶段式学习**：逐阶段推送学习内容，动态调整学习路径
- **实时答疑**：学习过程中随时提问，系统即时回答
- **统一审核**：Agent4 合并技术审核与质量评审，确保内容准确且适配用户
- **结构化争议处理**：Agent3↔Agent4 不再单向报错，而是“审核 → 答辩 → 复核 → 裁决”闭环，issue 带 source、Agent3 逐条回复、Orchestrator 终审，全程落库可回放
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
│   ├── chroma_db/    # FAQ 与 Agent1 知识语义索引
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
python scripts/init_agent1_semantic_index.py
python scripts/build_knowledge_graph.py
python scripts/seed_test_data.py
python scripts/init_faq_vectors.py
```

### 手动校验 Agent3-4 审核辩论表结构

在 Agent3 / Agent4 开始向 `review_traces` 等 6 张表写入真实审核数据前，建议先手动执行一次 schema 校验脚本：

```bash
python scripts/check_review_schema.py
```

这个脚本会检查：

- 6 张审核/辩论表是否存在
- 外键是否都指向正式表，而不是 `*_legacy_migration`
- 复合外键、唯一约束、关键列是否与协议一致
- 通过事务回滚方式验证 claim/source/issue/round 的关键约束是否真正生效

说明：

- 这是**手动执行**的检查工具，**不会**伴随后端每次启动自动运行
- 建议在以下场景执行：
  - 首次定版审核辩论表结构后
  - 修改 review/debate 迁移逻辑后
  - Agent3 / Agent4 开始正式写入审核数据前
  - 后续再次调整 review/debate schema 后

### 初始化后推荐顺序

```bash
python scripts/init_db.py
python scripts/init_knowledge_base.py
python scripts/init_agent1_semantic_index.py
python scripts/build_knowledge_graph.py
python scripts/seed_test_data.py
python scripts/init_faq_vectors.py
python scripts/check_review_schema.py
```

## 数据库约定

- 当前唯一主 SQLite 库为 `data/sqlite_data/project.db`
- `project.db` 是运行时唯一真相源，包含知识库表、FAQ 表、会话表和 `session_states`
- 历史库 `data/knowledge.db` 不再参与运行链路，仅保留备份归档用途
- 所有后端代码、初始化脚本、FAQ 向量化脚本统一读取 `DATABASE_URL`
- 所有 Chroma 数据统一读取 `CHROMA_PATH`
- Agent1 使用 `agent1_knowledge_docs` collection 对 `knowledge_docs` 的 68 条文档做语义召回；初始化或更新种子文档后执行 `python scripts/init_agent1_semantic_index.py`

## Agent3-4 审核辩论协议

为提升 Agent3 产出可追溯性与 Agent4 审核可执行性，审核链路从“单向报错”升级为“结构化争议处理”：

- **流程**：Agent4 出具带来源的 issue（含 `canonical_error_id`、`source_refs`、`must_fix`） → Agent3 逐条回复 `accepted / rejected / partially_accepted` → Agent4 复核 → Orchestrator 终审；最多 2 轮
- **数据落库**：6 张表（`review_traces` / `learning_package_sources` / `learning_package_claims` / `review_debate_issues` / `review_debate_rounds` / `review_debate_responses`），均以 `trace_id` 维度组织；`source_id` / `claim_id` / `canonical_error_id` 在单个 trace 内唯一
- **超轮次策略**：达到最大轮次仍有未决 must_fix 时，`final_status=needs_human_review`、`package_status=provisional`，学习包降级可用但不进入正式知识沉淀
- **完整协议**：见 [`docs/Agent34审核辩论协议.md`](docs/Agent34审核辩论协议.md)；表结构说明见 [`数据库框架.md`](数据库框架.md) 第九节

## FAQ 问答链路
- Agent3 仅在用户提问时调用 FAQ 检索，不在教程生成阶段调用
- FAQ 检索使用“完整问题直查向量库”方案，不先调用 LLM 提取关键词
- FAQ 检索返回候选后，只有最高相似度达到阈值时才直接返回 FAQ 标准答案
- FAQ 未命中或最高相似度低于阈值时，直接回退到知识库检索与 LLM 普通答疑
- FAQ 向量数据来源于 `faq_items`，通过 `python scripts/init_faq_vectors.py` 写入 Chroma 的 `faq_collection`

## PowerShell 调试流程

```powershell
# 0. 启动前先校验 Agent3-4 审核辩论表结构（仅首次/变更后执行）
python scripts/check_review_schema.py

# 1. 创建 Session
$session = Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/session/
$sessionId = $session.session_id
$sessionId

# 2. 初始化用户画像 + 生成当前阶段
$body = "{`"session_id`": `"$sessionId`", `"message`": `"我想学习大模型微调`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/chat/ -Body $bytes -ContentType "application/json; charset=utf-8"

# 3. 启动当前阶段，生成教程并触发 Agent4 统一审核
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

# 7. 完成当前阶段：只执行 Agent6 决策 + 保存画像，按需推进下一阶段
#   - 返回 can_plan_next_stage=true 说明仍有知识盲区，可继续规划下一阶段
#   - all_mastered=true 说明所有盲区已掌握，session 被真正结束
$completeResult = Invoke-RestMethod -Method POST -Uri "http://localhost:8500/api/stage/complete/$sessionId"
$completeResult.status
$completeResult.all_mastered
$completeResult.can_plan_next_stage

# 8. （按需调用）用户点击"生成下一阶段"后，规划真正的 next_stage
#   - 需要先 complete_stage 且 can_plan_next_stage=true
$body = "{`"session_id`": `"$sessionId`", `"adjustment_notes`": `"想先学QLoRA再看部署`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$planResult = Invoke-RestMethod -Method POST -Uri "http://localhost:8500/api/stage/plan-next" -Body $bytes -ContentType "application/json; charset=utf-8"
$planResult.next_stage          # 规划出的下一阶段 StageInfo
$planResult.planning_notes      # 本次规划说明（含 adjustment_notes）

# 9. 启动 next_stage：start_stage 会自动把 next_stage 切换为当前 stage_info
$body = "{`"session_id`": `"$sessionId`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/stage/start -Body $bytes -ContentType "application/json; charset=utf-8"
```

### P3 按需阶段规划 — 阶段推进接口契约

| 接口 | 何时调用 | 核心职责 |
|------|----------|----------|
| `POST /api/stage/complete/{sid}` | 用户答完测试题后点击"完成本阶段" | 仅调用 Agent6 + 保存更新后的画像；返回 `can_plan_next_stage` / `all_mastered` / `session_status`；不生成 next_stage，不提前结束 session |
| `POST /api/stage/plan-next` | complete 后 `can_plan_next_stage=true`，且用户主动点击"生成下一阶段" | 基于最新画像（可选带用户 adjustment_notes）调用 Agent1 规划链路生成 `next_stage`，保存到 session.next_stage |
| `POST /api/stage/start` | 用户确认下一阶段后点击"开始" | 若 session 中已有 `next_stage` 则先切换为当前 stage_info，再走 Agent3 生成学习包 → Agent4 审核的原流程 |

`complete_stage` 返回结构：
```jsonc
{
  "status": "continue" | "repeat" | "completed",
  "session_id": "...",
  "accuracy": 0.0~1.0,
  "updated_profile": { /* UserProfile */ },
  "stage_summary": "Agent6 生成的阶段总结 Markdown",
  "next_strategy": "reinforce" | "simplify" | "finish" | "backtrack" | ...,
  "can_plan_next_stage": true,  // 还有盲区可继续规划时为 true
  "session_status": "active" | "completed",
  "all_mastered": false,        // 仅当所有 knowledge_gaps.progress>=1.0 时为 true
  "final_report": null          // 仅 all_mastered=true 时有值
}
```

`plan-next` 返回结构：
```jsonc
{
  "status": "planned",
  "session_id": "...",
  "next_stage": { /* StageInfo，阶段编号已根据 accuracy 推进或重试 */ },
  "updated_profile": { /* 最新画像快照 */ },
  "planning_notes": "本阶段(1)已达标，推进到下一阶段；想先学QLoRA再看部署",
  "session_status": "active"
}
```

何时会真正结束 session：
- 只有 `all_mastered == true`（即用户画像中每个 `knowledge_gaps[i].progress >= 1.0`）时，`complete_stage` 才会调用 `end_session` 把 `session_status` 置为 `completed`。
- 其它任何情况（哪怕 accuracy=1.0），只要仍存在未掌握的知识盲区，session 就保持 `active`，等待 `plan-next` 继续推进。

前端推荐调用顺序（一个阶段循环）：
1. `POST /api/chat/`（首次：生成画像 + stage 1）
2. `POST /api/stage/start`（启动当前阶段 → 学习包+审核）
3. （学习中）`POST /api/chat/question`（可多次）
4. （测试）`POST /api/stage/submit-answer`（可多次）
5. `POST /api/stage/complete/{sid}`（Agent6 决策，返回 can_plan_next_stage / all_mastered）
6. 如果 `all_mastered=true` → 学习结束；否则：
   a. 若用户想继续 → 点击按钮触发 `POST /api/stage/plan-next`（可填 `adjustment_notes`）
   b. 用户确认 `next_stage` 后 → 回到第 2 步 `POST /api/stage/start`（自动把 next_stage 切为当前并启动）

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
