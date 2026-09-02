# AI Agent Trainer - 多Agent个性化辅导系统

<p style="background:#b91c1c;color:#fff;padding:14px 18px;border-radius:8px;font-size:16px;line-height:1.75;">
⚠️  <strong style="font-size:18px">评委复现第一步：必须先 <code>pip install -r requirements.txt</code></strong><br>
&nbsp;&nbsp;① 使用同一个 Python 虚拟环境（推荐 <code>python -m venv venv</code> → <code>venv\Scripts\activate</code>），<br>
&nbsp;&nbsp;② 激活后执行 <code>python -m pip install -r requirements.txt</code>，<br>
&nbsp;&nbsp;③ <strong>禁止</strong>在未激活的情况下直接全局 <code>pip install</code>，否则 <code>python scripts\*.py</code> 仍会报 ModuleNotFound。<br>
&nbsp;&nbsp;④ 或直接双击打开 <code>快捷方式\_launcher.ps1</code>，脚本会自动检测缺失依赖并补装。
</p>

一个基于多Agent协同的AI个性化学习辅导系统，专注于大模型微调与推理部署领域的教学。

## 核心特性

- **个性化画像**：通过对话式交互生成用户学习画像
- **阶段式学习**：逐阶段推送学习内容，动态调整学习路径
- **实时答疑**：学习过程中随时提问，系统即时回答
- **统一审核**：专业知识与教学质量审核 Agent 合并技术审核与质量评审，确保内容准确且适配用户
- **结构化争议处理**：个性化学习资源生成 Agent↔专业知识与教学质量审核 Agent 不再单向报错，而是“审核 → 答辩 → 复核 → 裁决”闭环，issue 带 source、个性化学习资源生成 Agent 逐条回复、Orchestrator 终审，全程落库可回放
- **策略调整**：根据测试结果自动调整下一阶段学习策略
- **可视化报告**：实时展示学习进度和能力变化

## 技术栈

- **后端**：FastAPI, Python 3.12+
- **数据库**：SQLite, Chroma Vector DB, NetworkX
- **LLM**：Ollama 本地模型 (deepseek-r1/ qwen2.5)；个性化学习资源生成 Agent 技术问答使用 `qwen3-embedding:0.6b` 进行 Chroma 语义检索
- **前端**：React + TypeScript

后端默认使用 `EXECUTION_MODE=hybrid`：学情诊断与阶段规划 Agent 先以用户表单字段为事实源，使用别名精确匹配与轻量语义召回定位候选知识板块，LLM 仅在候选内细化盲区；阶段选路、状态管理和答题判定保持确定性，教程与问答由 Ollama 生成。设置为 `deterministic` 后，无需启动 Ollama 也可以跑通完整闭环。

## 项目结构

```
project-root/
├── backend/          # 后端服务
├── frontend/         # 前端界面
├── data/             # 数据存储
│   ├── seeds/        # 结构化种子数据
│   ├── sqlite_data/  # 主库 project.db
│   ├── chroma_db/    # 学情诊断与阶段规划 Agent 知识语义索引
│   ├── graph_data/   # 知识图谱结果层
│   └── test_data/    # 测试样例数据
├── scripts/          # 评测、数据构建、部署校验及开发辅助脚本（非单元测试目录）
└── docs/             # 项目文档
```

## 快速开始

### 环境要求

- Python 3.12+
- Ollama (本地模型推理)
- Node.js 18+ (前端)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
copy backend\\.env.example backend\\.env
# 或手动复制 backend/.env.example 为 backend/.env
```

`backend/.env` 是后端实际读取的环境变量文件；不要把 API Key 写进 `backend/config.py`，也不要直接修改模板文件。

DeepSeek 最小必填配置：

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

队友克隆仓库后的使用方式：

1. 复制 `backend/.env.example` 为 `backend/.env`
2. 填入自己的 `DEEPSEEK_API_KEY`
3. 启动或重启后端服务

不要提交 `backend/.env`；仓库中只保留 `.env.example` 作为模板。

个性化学习资源生成 Agent 的技术问答需要 Ollama embedding 模型。首次部署先下载并配置：

```bash
ollama pull qwen3-embedding:0.6b
```

```env
EMBEDDING_MODEL=qwen3-embedding:0.6b
```

该模型只负责编码问题和知识片段，用于技术资料检索，不负责生成回答；日常问候、闲聊和情绪表达不会调用它。切换 embedding 模型或更新 `knowledge_chunks` 后，必须重新执行 `python scripts/init_knowledge_base.py`。

### 初始化数据

```bash
python scripts/init_db.py
python scripts/init_knowledge_base.py
python scripts/init_agent1_semantic_index.py
python scripts/build_knowledge_graph.py
python scripts/seed_test_data.py
```

### 知识目录与视频资源同步

首次使用项目，以及修改 `知识清单-带学习时长版（新版）.md`、知识点种子或学习视频清单后，必须按以下顺序执行。它会同步 15 个知识板块、知识点时长、知识图谱和视频资源到唯一主库 `data/sqlite_data/project.db`：

```bash
python scripts/sync_knowledge_catalog.py
python scripts/build_knowledge_graph.py
python scripts/prepare_video_resources.py
python scripts/insert_knowledge_data.py
python scripts/render_knowledge_graph.py
python scripts/verify_catalog_runtime.py
```

- `sync_knowledge_catalog.py`：从新版清单同步知识点、安装节点及学习时长。
- `build_knowledge_graph.py`：生成 学情诊断与阶段规划 Agent 使用的运行时图谱。
- `prepare_video_resources.py` 与 `insert_knowledge_data.py`：重建视频目录和视频映射并导入主库。
- `render_knowledge_graph.py`：生成 `docs/知识图谱可视化.html`，可直接在浏览器打开查看。
- `verify_catalog_runtime.py`：校验 学情诊断与阶段规划 Agent 图谱依赖查询，以及 个性化学习资源生成 Agent 的数据库/目录视频 URL 查询。

### 学情诊断与阶段规划 Agent 与 个性化学习资源生成 Agent 语义索引

`python scripts/init_knowledge_base.py` 使用 Ollama 的 `EMBEDDING_MODEL`（默认 `qwen3-embedding:0.6b`）读取 SQLite 的 `knowledge_chunks`，重建 Chroma 的 `knowledge_base` collection，供 个性化学习资源生成 Agent 技术类问答按当前阶段的多个知识板块和知识点进行语义检索。脚本会先验证 embedding 服务，再重建 collection 并校验条目数；SQLite FTS5/BM25 保留为 Chroma 不可用时的回退。

`python scripts/init_agent1_semantic_index.py` 首次执行会下载并缓存 FastEmbed 的 `BAAI/bge-small-zh-v1.5`（模型文件约 90MB，CPU 运行，运行时通常额外占用约 200MB 内存），然后为 68 条 `knowledge_docs` 建立 学情诊断与阶段规划 Agent 专用索引。首次下载耗时取决于网络，通常是几十秒级。下载与建索引只应由源码开发者或部署方执行；可用下面的命令检查中文召回：

```bash
python scripts/evaluate_agent1_retrieval.py
```

面向普通用户的发行包应预置 `data/fastembed_models/` 和已初始化的 `data/chroma_db/`，用户无需下载 embedding 模型、启动 Ollama embedding 服务或手动初始化索引。

### 手动校验 个性化学习资源生成 Agent-4 审核辩论表结构

在 个性化学习资源生成 Agent / 专业知识与教学质量审核 Agent 开始向 `review_traces` 等 6 张表写入真实审核数据前，建议先手动执行一次 schema 校验脚本：

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
  - 个性化学习资源生成 Agent / 专业知识与教学质量审核 Agent 开始正式写入审核数据前
  - 后续再次调整 review/debate schema 后

### 初始化后推荐顺序

```bash
python scripts/init_db.py
python scripts/init_knowledge_base.py
python scripts/init_agent1_semantic_index.py
python scripts/build_knowledge_graph.py
python scripts/seed_test_data.py
python scripts/check_review_schema.py
```

## 数据库约定

- 当前唯一主 SQLite 库为 `data/sqlite_data/project.db`
- `project.db` 是运行时唯一真相源，包含知识库表、会话表和 `session_states`
- 历史库 `data/knowledge.db` 不再参与运行链路，仅保留备份归档用途
- 所有后端代码和初始化脚本统一读取 `DATABASE_URL`
- 所有 Chroma 数据统一读取 `CHROMA_PATH`
- 个性化学习资源生成 Agent 使用 `knowledge_base` collection 对 `knowledge_chunks` 做技术问答语义检索；首次部署须执行 `ollama pull qwen3-embedding:0.6b`，并在知识块或 embedding 模型变更后执行 `python scripts/init_knowledge_base.py`
- 学情诊断与阶段规划 Agent 使用 `agent1_knowledge_docs` collection 对 `knowledge_docs` 的 68 条文档做语义召回；初始化或更新种子文档后执行 `python scripts/init_agent1_semantic_index.py`
- 学情诊断与阶段规划 Agent 索引独立使用 FastEmbed 的 `BAAI/bge-small-zh-v1.5`（约 90MB、CPU 运行）与 cosine 距离，不依赖 Ollama。
- **源码开发/部署者**：执行一次 `python scripts/init_agent1_semantic_index.py` 与 `python scripts/evaluate_agent1_retrieval.py`，模型缓存和索引会生成在 `data/fastembed_models/`、`data/chroma_db/`。
- **普通用户**：发行包应预置上述两个目录；不需要下载模型或执行初始化。两个目录被 Git 忽略，不能依赖源码仓库自动提供。
- `KnowledgeGap` 中的 `block_id`、`block_name`、`target_subtopic_ids`、`mapping_confidence` 是 学情诊断与阶段规划 Agent 的映射与选路元数据；学情评估与动态决策 Agent 不重新推断这些字段，只在更新学习状态时原样保留。

## 个性化学习资源生成 Agent-4 审核辩论协议

为提升 个性化学习资源生成 Agent 产出可追溯性与 专业知识与教学质量审核 Agent 审核可执行性，审核链路从“单向报错”升级为“结构化争议处理”：

- **流程**：专业知识与教学质量审核 Agent 出具带来源的 issue（含 `canonical_error_id`、`source_refs`、`must_fix`） → 个性化学习资源生成 Agent 逐条回复 `accepted / rejected / partially_accepted` → 专业知识与教学质量审核 Agent 复核 → Orchestrator 终审；最多 3 轮（由 `MAX_REVIEW_ITERATIONS` 配置）
- **数据落库**：6 张表（`review_traces` / `learning_package_sources` / `learning_package_claims` / `review_debate_issues` / `review_debate_rounds` / `review_debate_responses`），均以 `trace_id` 维度组织；`source_id` / `claim_id` / `canonical_error_id` 在单个 trace 内唯一
- **超轮次策略**：达到最大轮次仍有未决 must_fix 时，`final_status=needs_human_review`、`package_status=provisional`，学习包降级可用但不进入正式知识沉淀
- **完整协议**：见 [`docs/个性化学习资源生成 专业知识与教学质量审核 Agent审核辩论协议.md`](docs/个性化学习资源生成 专业知识与教学质量审核 Agent审核辩论协议.md)；表结构说明见 [`数据库框架.md`](数据库框架.md) 第九节

## 个性化学习资源生成 Agent 答疑链路
- 个性化学习资源生成 Agent 使用当前阶段信息和知识文档检索结果作为上下文。
- 答案由本地 LLM 直接生成，并统一交由 专业知识与教学质量审核 Agent 审核与必要改稿。

## PowerShell 调试流程

```powershell
# 0. 启动前先校验 个性化学习资源生成 Agent-4 审核辩论表结构（仅首次/变更后执行）
python scripts/check_review_schema.py

# 1. 创建 Session
$session = Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/session/
$sessionId = $session.session_id
$sessionId

# 2. 初始化用户画像 + 生成当前阶段
$body = "{`"session_id`": `"$sessionId`", `"message`": `"我想学习大模型微调`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod -Method POST -Uri http://localhost:8500/api/chat/ -Body $bytes -ContentType "application/json; charset=utf-8"

# 3. 启动当前阶段，生成教程并触发 专业知识与教学质量审核 Agent 统一审核
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

# 7. 完成当前阶段：只执行 学情评估与动态决策 Agent 决策 + 保存画像，按需推进下一阶段
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
| `POST /api/stage/complete/{sid}` | 用户答完测试题后点击"完成本阶段" | 仅调用 学情评估与动态决策 Agent + 保存更新后的画像；返回 `can_plan_next_stage` / `all_mastered` / `session_status`；不生成 next_stage，不提前结束 session |
| `POST /api/stage/plan-next` | complete 后 `can_plan_next_stage=true`，且用户主动点击"生成下一阶段" | 基于最新画像（可选带用户 adjustment_notes）调用 学情诊断与阶段规划 Agent 规划链路生成 `next_stage`，保存到 session.next_stage |
| `POST /api/stage/start` | 用户确认下一阶段后点击"开始" | 若 session 中已有 `next_stage` 则先切换为当前 stage_info，再走 个性化学习资源生成 Agent 生成学习包 → 专业知识与教学质量审核 Agent 审核的原流程 |

`complete_stage` 返回结构：
```jsonc
{
  "status": "continue" | "repeat" | "completed",
  "session_id": "...",
  "accuracy": 0.0~1.0,
  "updated_profile": { /* UserProfile */ },
  "stage_summary": "学情评估与动态决策 Agent 生成的阶段总结 Markdown",
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
5. `POST /api/stage/complete/{sid}`（学情评估与动态决策 Agent 决策，返回 can_plan_next_stage / all_mastered）
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

当前核心模块测试共 141 个用例，覆盖多智能体调度、领域知识检索与生成、专业审核、数据库状态、题目评估、并发控制和模型回退；最近一次执行结果为 141 个测试全部通过。该测试结果属于软件模块级验证，50 组主评测则用于系统级个性化效果统计。

### `scripts/` 与 `tests/` 的职责边界

`tests/` 保存正式的 Python `unittest` 单元与集成回归用例；`scripts/` 保存可重复执行的系统评测、指标统计、数据库/知识库初始化、图谱构建、运行状态校验和开发辅助工具。正式评测主要使用 `run_real_evaluation.py`、`calculate_real_metrics.py` 及 `run_fallback_tests.py`；带有 `debug_`、`diagnose_`、`patch_` 或交互式 `test_` 名称的脚本属于开发辅助，不计入141个单元测试。历史 `run_ablation_tests.py` 不属于当前提交的正式实验结果。

## Agent架构

| Agent | 职责 | 核心功能 |
|-------|------|----------|
| 学情诊断与阶段规划 Agent | 学情诊断与阶段选路 | 以表单事实源生成画像、混合召回并通过 `generate_stage` 规划当前/下一阶段 |
| 阶段规划 | 学情诊断与阶段规划 Agent 的阶段选路能力 | 由 `generate_stage` 承担 |
| 个性化学习资源生成 Agent | 内容生成 | 生成学习包、回答问题、按 `block_id` 抽取测试题 |
| 专业知识与教学质量审核 Agent | 统一审核 | 同时检查技术正确性与教学质量 |
| 学情评估与动态决策 Agent | 阶段决策 | 更新画像、选择策略、推进下一阶段 |

说明：

- 专业知识与教学质量审核 Agent 统一承担技术审核和教学质量审核
- 当前阶段测试题由 个性化学习资源生成 Agent 从 `test_questions` 表中抽取，不由 学情评估与动态决策 Agent 生成

## 学习流程

1. 用户首次输入 → 学情诊断与阶段规划 Agent 生成受候选目录约束的画像
2. 学情诊断与阶段规划 Agent 的 `generate_stage` 生成当前阶段
3. 个性化学习资源生成 Agent生成学习包 → 专业知识与教学质量审核 Agent统一审核
4. 用户学习 + 实时答疑
5. 进入阶段测试 → 个性化学习资源生成 Agent按当前 `block_id` 抽取 1~3 题
6. 完成测试 → 学情评估与动态决策 Agent更新画像
7. 循环直到所有知识盲区掌握

## DAG + mastery state

- 当前阶段推进已经从“板块级串行推进”下沉到“知识点状态推进”
- 运行时真实状态来源于 `subtopic_status`
- 状态语义：`-1` 未掌握，`0` 学习中，`1` 已掌握
- 当前运行时由 学情诊断与阶段规划 Agent 的 `generate_stage` 依据知识图谱前置关系与 `subtopic_status` 选择下一学习内容
- `knowledge_gaps` 继续保留为板块级画像视图，不作为唯一推进依据
