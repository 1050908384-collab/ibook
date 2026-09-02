# SKILL名称：content_generation

## 1. 能力描述
- 功能：为个性化学习资源生成 Agent（HybridContentAgent）提供**基于原始知识块的个性化改写与改稿方法论**。Agent 3 只依据当前原始块，以用户画像调整表达深浅、术语解释与原文已有步骤的呈现重点，并依据专业知识与教学质量审核 Agent 的意见定点修订。在 `HybridContentAgent.run` 中被加载为 `content_skill_context`，作为 LLM 块级改写的约束。
- 适用场景（3个）：
  1. **初始化教程**：阶段开始时由 `HybridContentAgent.run` 触发，根据 `stage.block_name` 与 `profile.goal` 生成结构化Markdown教程，交 `DeterministicContentAgent` 包装为 LearningPackage
2. **审核后改稿**：专业知识与教学质量审核 Agent统一审核未通过时由 `revise_learning_package` 触发，接收兼容性的 `technical_errors` 与 `quality_errors` 字段逐条修订
  3. **版本迭代**：每次修订 `learning_package.version` 自增，记录迭代轮次直至 `MAX_REVIEW_ITERATIONS`
- 前置依赖：
  - 当前阶段 StageInfo（`block_name`、`gap_skill`、`estimated_hours`）
  - 用户画像 UserProfile（`goal`、`learning_style`、`skill_tags` 用于难度适配与已知回顾）
  - code_style 技能（代码片段风格约束，作为 `content_skill_context` 一同注入）
- 审核反馈（修订时：`technical_errors` 与 `quality_errors` 均来自 专业知识与教学质量审核 Agent 的统一审核结果）

## 2. 核心方法论
- **步骤1：定位教程主题与难度面向**——读取 `stage.block_name` 作为教程主题，结合 `profile.goal` 确定面向（如"掌握LoRA微调"），参照 `profile.skill_tags` 判断已知基线，避免重复已掌握内容或跨度太大
- **步骤2：按原始知识块个性化改写**——调用 LLM 保持原文标题层级、段落顺序与技术事实；结合 `goal`、`learning_style`、`skill_tags`、年级和硬件调整讲解口吻、术语解释及原文已有代码/步骤的突出程度。不新增固定章节、反思题、学习检查或原文不存在的代码示例。
- **步骤3：交付审核**——tutorial 交 `DeterministicContentAgent` 包装为 `LearningPackage(tutorial, questions, code_examples, version=1)`，由 Orchestrator 送 专业知识与教学质量审核 Agent 统一审核
- **步骤4：格式化审核反馈**——将 `technical_errors`（ReviewError数组）与 `quality_errors`（QualityIssue数组）合并为反馈清单，按 severity 降序排列（critical > major > minor）
- **步骤5：逐条修订**——调用LLM按反馈清单逐条修订教程，保留正确内容，仅修改 errors 指向的 location，返回完整新 tutorial，`version` 自增
- **步骤6：迭代终止判定**——审核通过或迭代轮次达 `MAX_REVIEW_ITERATIONS`（默认3）时终止；未通过则保留最后版本并标注未通过状态

## 3. 知识/规则定义

### 规则1：块级改写规范
- 一块内容只对应一个 `source_chunk_id`，不得引入该块之外的技术事实、命令、参数、代码或结论。
- 保留原文标题层级和段落顺序；可对句子做压缩、换说法与术语解释，但不制作固定学习卡、表格、思考题、反思题或学习检查。
- 原文有命令、代码、报错或操作步骤时，按用户画像调整说明重点；原文没有时不得补造。
- 正式测试由用户主动进入测试流程后从 `test_questions` 题库抽取，教程正文不出题。

### 规则2：内容要素规范
| 要素 | 要求 | 违规处置 |
|:---|:---|:---|
| 术语解释 | 核心概念首次出现必须配简明解释 | 标记 `unexplained_term`，major |
| 代码示例 | 遵循 code_style（PEP8、docstring、异常处理） | 交 专业知识与教学质量审核 Agent `rule_check_tool` 检测 |
| 难度适配 | 面向 `profile.goal` 与 `skill_tags` 基线，不超 ZPD | 标记 `difficulty_mismatch`，major |
| 篇幅控制 | 每块篇幅与原文相称，避免无依据扩写 | 超载建议压缩，minor |
| 事实准确 | 技术陈述须有依据，不编造参数/原理 | 交 专业知识与教学质量审核 Agent 幻觉检测（sim<0.65判critical） |

### 规则3：改稿优先级
**修订顺序（从高到低）**：
1. **critical 错误**（LoRA机制错误、参数量错误、安全违规、路径硬编码）——必须修复
2. **major 错误**（幻觉、结构缺失、难度不当、未捕获异常）——必须修复
3. **minor 错误**（风格、命名、冗余）——尽量修复

**改稿规则**：
- 逐条解决审核 errors，不得遗漏 critical/major 项
- 保留审核未指出的正确内容，不做无谓改写
- 审核意见为空时保留原 tutorial，仅 `version` 自增
- 改稿后必须保持原始块的标题层级、段落顺序与事实边界

## 4. 输入输出规范

### 输入（生成）
`ContentGenerationInput` 对象：
- `session_id`：会话标识
- `stage`：StageInfo（`block_name`、`gap_skill`、`estimated_hours`）
- `profile`：UserProfile（`goal`、`learning_style`、`skill_tags`）

### 输入（修订）
`ContentRevisionInput` 对象：
- `session_id`
- `learning_package`：LearningPackage（当前 tutorial、`version`）
- `stage`、`profile`
- `technical_errors`：List[ReviewError]
- `quality_errors`：List[QualityIssue]

### 输出
`ContentGenerationOutput` 对象，含 `learning_package: LearningPackage`：
```json
{
  "learning_package": {
    "tutorial": "# LoRA微调入门\n\nLoRA冻结原权重W，训练低秩增量ΔW=(α/r)BA。\n\n```python\nclass LoRALayer(nn.Module):\n    ...\n```",
    "questions": [],
    "code_examples": [
      {"title": "LoRA最小示例", "language": "python", "code": "..."}
    ],
    "version": 2
  }
}
```

**字段说明**：
- `tutorial`：由原始知识块改写得到的 Markdown 教程正文
- `questions`：自测题列表（教程生成阶段为空，用户主动点击自测时从 `test_questions` 表抽取填充）
- `code_examples`：从 tutorial 提取的代码示例
- `version`：迭代版本号，首次为1，每次改稿+1

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| LLM返回空或Error前缀 | 回退 `DeterministicContentAgent` 保留原始知识块正文，version=1 |
| LLM生成内容破坏原文事实边界或段落顺序 | 标记结构问题交 专业知识与教学质量审核 Agent，由 `quality_errors` 反馈后定点修订 |
| 修订后LLM响应无效 | 保留原tutorial，追加"审核修订"清单到末尾，version自增 |
| 审核轮次达 `MAX_REVIEW_ITERATIONS`(3) | 保留最后版本，标注"未通过审核"，终止迭代 |
| `technical_errors` 与 `quality_errors` 均为空 | 保留原tutorial，仅version自增，判通过 |
| `stage.block_name` 为空 | 抛出 ValueError，终止生成 |

## 6. 附录

### 附录A：教程生成理论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | 认知负荷理论（Cognitive Load Theory） | John Sweller | 1988 | 工作记忆容量7±2，内在+外在+相关负荷总和不能超限，超限导致认知过载 | 规则2篇幅控制与新概念密度限制（每段≤2个新概念）的依据 |
| A-2 | 掌握学习理论（Mastery Learning） | Benjamin Bloom | 1968 | 学习者需达到掌握标准（80%正确率）方可进阶 | 自测段TestQuestion的依据——检验阶段掌握度，配合学情评估与动态决策 Agent策略选择 |
| A-3 | 建构主义学习理论（Constructivism） | Piaget / Vygotsky | 1950s | 学习者在真实情境中主动建构知识，而非被动接受 | 规则1实操演示段要求真实场景案例的依据 |
| A-4 | 最近发展区理论（ZPD） | Lev Vygotsky | 1978 | 学习内容应处于"已知但未掌握"的最近发展区，难度略高于当前水平 | 规则2难度适配的依据——参照 `skill_tags` 基线不超ZPD上限 |

### 附录B：完整场景示例

#### 场景1：首次生成教程
**输入**：
```json
{
  "session_id": "sess-001",
  "stage": {"block_name": "LoRA", "gap_skill": "LoRA", "estimated_hours": 3.0},
  "profile": {"goal": "掌握LoRA微调", "learning_style": "practice", "skill_tags": ["Transformer", "注意力机制"]}
}
```

**处理过程**：
1. 步骤1：主题=LoRA，面向=掌握LoRA微调，已知基线=[Transformer,注意力机制]
2. 步骤2：LLM 仅改写当前 LoRA 原始知识块；根据已知基线减少重复基础解释，保留原文已有的概念与代码，不新增固定章节或题目。
3. 步骤3：包装为 LearningPackage(version=1)，送 专业知识与教学质量审核 Agent 统一审核

**输出**：基于原始块改写的 tutorial + questions(空) + 原文已有 code_examples，version=1

#### 场景2：审核未通过后改稿
**输入**：
```json
{
  "technical_errors": [
    {"error_type": "lora_mechanism_error", "severity": "critical", "location": "核心概念段", "description": "将LoRA描述为SVD分解"}
  ],
  "quality_errors": [],
  "learning_package": {"tutorial": "...", "version": 1}
}
```

**处理过程**：
1. 步骤4：合并反馈，critical优先（LoRA机制错误）
2. 步骤5：LLM修订——将"SVD分解"改为"冻结W训练低秩增量ΔW=BA"；保留其余正确内容，不追加自测或学习检查
3. 步骤6：version=1→2，重新送审

**输出**：修订后tutorial（机制错误已修复），version=2

**说明**：场景2体现规则3改稿优先级——critical 的 LoRA 机制错误必须先修复，正确内容保留不改。正式测试由用户进入测试流程后从 `test_questions` 表抽取，不纳入教程生成与改稿阶段。

## 7. 代码纠错场景

### 7.1 触发条件
当用户输入包含以下关键词时，触发代码纠错流程：
- "纠错"、"修改代码"、"报错"、"bug"、"错误"、"修复"
- 用户提供了代码片段

### 7.2 处理流程
1. **识别意图**：检测用户输入是否为代码纠错请求
2. **提取代码**：从用户输入中提取代码片段
3. **执行验证**：调用 `code_executor_tool` 执行代码，获取真实执行反馈或报错信息
4. **加载策略**：优先使用 `code_correction_pedagogy.skill.md` 中的策略
5. **生成回答**：按照"错误分析→修改步骤→修改后完整代码→修改说明"四个板块输出
6. **验证修改**：对修改后的代码再次执行验证，确保修复成功

### 7.3 注意事项
- **必须执行验证**：严禁凭空猜测错误原因，必须先通过 `code_executor_tool` 获取真实反馈
- **结合当前阶段**：错误分析应联系当前学习阶段的知识点
- **遵循教学法**：严格按照 `code_correction_pedagogy.skill.md` 的解释风格和输出结构
- **仅修改必要部分**：保留用户代码中的正确部分，只修改有问题的地方

### 7.4 与 个性化学习资源生成 Agent 的衔接
个性化学习资源生成 Agent 在 `generate_answer` 方法中检测到代码纠错意图时：
1. 调用 `code_executor_tool.execute_code()` 获取执行结果
2. 加载 `code_correction_pedagogy` 技能作为 prompt 规范
3. 生成结构化纠错回答

### 7.5 输入输出示例

**输入**：
```json
{
  "session_id": "sess-001",
  "question": "帮我看看这段代码哪里错了：\n```python\ndef add(a, b)\n    return a + b\n```",
  "stage": {"block_name": "Python基础"}
}
```

**输出**：
```json
{
  "answer": "# 代码纠错分析\n\n## 错误分析\n语法错误：第1行 `def add(a, b)` 缺少冒号 `:`。Python函数定义语法要求函数头以冒号结尾。\n\n## 修改步骤\n1. 在 `def add(a, b)` 后面添加冒号 `:`\n2. 确保函数体正确缩进\n\n## 修改后完整代码\n```python\ndef add(a, b):\n    return a + b\n```\n\n## 修改说明\n本次修改学习要点：\n- Python函数定义必须以冒号结尾\n- 函数体需要4个空格缩进\n- 这是Python基础语法，后续所有函数定义都要遵守"
}
```

## 8. 版本信息
- 版本：v3.4，更新日期：2026-08-02
- 变更记录：v3.4 移除旧的直接标准答案检索；用户答疑以知识文档检索结果作为可选上下文，由本地 LLM 生成回答，并继续走 专业知识与教学质量审核 Agent 审核与改稿链路。

<!-- RUNTIME_POLICY_START -->
## Runtime Policy

- Treat each source chunk as the complete factual boundary. Personalize by selecting, compressing, reordering, explaining, and emphasizing source-supported material for the learner.
- Preserve factual traceability. Do not add unsupported technical facts, commands, parameters, or cross-topic claims.
- 个性化学习资源生成 Agent must return `personalization_action`, `personalization_reason` and `fact_source_chunk_ids` alongside the learner-facing Markdown body. `teaching_tasks` remains empty unless a future, separately designed feature explicitly needs it.
- Treat `LearningEvidence` as teaching input: unstable prerequisites may change the explanation focus; low-accuracy history may request clearer source-supported explanation. Do not add a visible consolidation check.
- Do not remove a source block solely to shorten the tutorial. A removal or compression must record its evidence-based reason, and must not remove a stage's essential operation or source-stated common pitfall.
- The final non-whitespace character count may be up to 150% of the source chunk only for source-supported explanation.
- When a review requires a correction, change only the cited part. Do not append a revision note to learner-facing content.
- Return Markdown body only. Do not describe the rewrite process.
<!-- RUNTIME_POLICY_END -->
