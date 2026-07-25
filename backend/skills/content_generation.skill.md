# SKILL名称：content_generation

## 1. 能力描述
- 功能：为Agent3（HybridContentAgent）提供**教程生成与改稿的方法论**——定义"一篇合格阶段教程长什么样"：五段式结构规范、内容要素清单、改稿优先级。Agent3调用本SKILL生成结构化Markdown教程，并依据Agent4技术审核与Agent5教学审核的意见迭代修订。在 `HybridContentAgent.run` 中被加载为 `content_skill_context`（与 code_style 一同注入），作为LLM生成 tutorial 的 prompt 规范；在 `revise_learning_package` 中作为改稿依据
- 适用场景（3个）：
  1. **初始化教程**：阶段开始时由 `HybridContentAgent.run` 触发，根据 `stage.block_name` 与 `profile.goal` 生成结构化Markdown教程，交 `DeterministicContentAgent` 包装为 LearningPackage
  2. **审核后改稿**：Agent4/Agent5审核未通过时由 `revise_learning_package` 触发，接收 `technical_errors` 与 `quality_errors` 逐条修订
  3. **版本迭代**：每次修订 `learning_package.version` 自增，记录迭代轮次直至 `MAX_REVIEW_ITERATIONS`
- 前置依赖：
  - 当前阶段 StageInfo（`block_name`、`gap_skill`、`estimated_hours`）
  - 用户画像 UserProfile（`goal`、`learning_style`、`skill_tags` 用于难度适配与已知回顾）
  - code_style 技能（代码片段风格约束，作为 `content_skill_context` 一同注入）
  - 审核反馈（修订时：`technical_errors` 来自 Agent4，`quality_errors` 来自 Agent5）

## 2. 核心方法论
- **步骤1：定位教程主题与难度面向**——读取 `stage.block_name` 作为教程主题，结合 `profile.goal` 确定面向（如"掌握LoRA微调"），参照 `profile.skill_tags` 判断已知基线，避免重复已掌握内容或跨度太大
- **步骤2：一次性生成四段式结构教程**——调用LLM按"已知回顾→核心概念拆解→实操演示→常见坑点"四段生成Markdown；核心概念配代码示例（遵循 code_style）；自测题由用户主动点击自测时从 `test_questions` 表抽取，教程中不包含
- **步骤3：交付审核**——tutorial 交 `DeterministicContentAgent` 包装为 `LearningPackage(tutorial, questions, code_examples, version=1)`，由 Orchestrator 送 Agent4/Agent5 双重审核
- **步骤4：格式化审核反馈**——将 `technical_errors`（ReviewError数组）与 `quality_errors`（QualityIssue数组）合并为反馈清单，按 severity 降序排列（critical > major > minor）
- **步骤5：逐条修订**——调用LLM按反馈清单逐条修订教程，保留正确内容，仅修改 errors 指向的 location，返回完整新 tutorial，`version` 自增
- **步骤6：迭代终止判定**——审核通过或迭代轮次达 `MAX_REVIEW_ITERATIONS`（默认3）时终止；未通过则保留最后版本并标注未通过状态

## 3. 知识/规则定义

### 规则1：四段式结构规范
| 段落 | 必含要素 | 缺失处置 | severity |
|:---|:---|:---|:---|
| 已知回顾 | 链接 `profile.skill_tags` 中相关已掌握技能 | 标记 `missing_review` | major |
| 核心概念拆解 | `block_name` 的定义/原理，每段≤2个新概念 | 标记 `concept_overload` | major |
| 实操演示 | 可运行代码示例（遵循code_style）或真实场景案例 | 标记 `missing_context_case` | major |
| 常见坑点 | 2~3个典型误区/陷阱 | 标记 `missing_pitfalls` | minor |

**结构完整性依据**：四段式对应"已知→未知→实践→反思"的认知闭环，与 `pedagogy.skill.md` 规则2的渐进式讲解规范一致。自测题由用户主动点击自测时从 `test_questions` 表抽取，不纳入教程生成阶段。

### 规则2：内容要素规范
| 要素 | 要求 | 违规处置 |
|:---|:---|:---|
| 术语解释 | 核心概念首次出现必须配简明解释 | 标记 `unexplained_term`，major |
| 代码示例 | 遵循 code_style（PEP8、docstring、异常处理） | 交 Agent4 `rule_check_tool` 检测 |
| 难度适配 | 面向 `profile.goal` 与 `skill_tags` 基线，不超 ZPD | 标记 `difficulty_mismatch`，major |
| 篇幅控制 | 800~2000字为宜，避免认知超载 | 超载建议拆分，minor |
| 事实准确 | 技术陈述须有依据，不编造参数/原理 | 交 Agent4 幻觉检测（sim<0.65判critical） |

### 规则3：改稿优先级
**修订顺序（从高到低）**：
1. **critical 错误**（LoRA机制错误、参数量错误、安全违规、路径硬编码）——必须修复
2. **major 错误**（幻觉、结构缺失、难度不当、未捕获异常）——必须修复
3. **minor 错误**（风格、命名、冗余、缺少坑点）——尽量修复

**改稿规则**：
- 逐条解决审核 errors，不得遗漏 critical/major 项
- 保留审核未指出的正确内容，不做无谓改写
- 审核意见为空时保留原 tutorial，仅 `version` 自增
- 改稿后 tutorial 必须仍保持四段式结构完整性

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
    "tutorial": "# LoRA微调入门\n\n## 已知回顾\n你已经掌握了Transformer的注意力机制...\n\n## 核心概念拆解\nLoRA冻结原权重W，训练低秩增量ΔW=(α/r)BA...\n\n## 实操演示\n```python\nclass LoRALayer(nn.Module):\n    ...\n```\n\n## 常见坑点\n- 秩r过大导致过拟合",
    "questions": [],
    "code_examples": [
      {"title": "LoRA最小示例", "language": "python", "code": "..."}
    ],
    "version": 2
  }
}
```

**字段说明**：
- `tutorial`：Markdown教程正文（四段式）
- `questions`：自测题列表（教程生成阶段为空，用户主动点击自测时从 `test_questions` 表抽取填充）
- `code_examples`：从 tutorial 提取的代码示例
- `version`：迭代版本号，首次为1，每次改稿+1

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| LLM返回空或Error前缀 | 回退 `DeterministicContentAgent` 生成基础三段式教程（概述/概念/示例），version=1 |
| LLM生成的tutorial无五段结构 | 保留内容，标记结构问题交Agent5审核，由 quality_errors 反馈后改稿补全 |
| 修订后LLM响应无效 | 保留原tutorial，追加"审核修订"清单到末尾，version自增 |
| 审核轮次达 `MAX_REVIEW_ITERATIONS`(3) | 保留最后版本，标注"未通过审核"，终止迭代 |
| `technical_errors` 与 `quality_errors` 均为空 | 保留原tutorial，仅version自增，判通过 |
| `stage.block_name` 为空 | 抛出 ValueError，终止生成 |

## 6. 附录

### 附录A：教程生成理论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | 认知负荷理论（Cognitive Load Theory） | John Sweller | 1988 | 工作记忆容量7±2，内在+外在+相关负荷总和不能超限，超限导致认知过载 | 规则2篇幅控制与新概念密度限制（每段≤2个新概念）的依据 |
| A-2 | 掌握学习理论（Mastery Learning） | Benjamin Bloom | 1968 | 学习者需达到掌握标准（80%正确率）方可进阶 | 自测段TestQuestion的依据——检验阶段掌握度，配合Agent6策略选择 |
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
2. 步骤2：LLM生成四段式——已知回顾链接Transformer注意力机制；核心概念讲ΔW=(α/r)BA；实操给LoRALayer代码；坑点讲r过大过拟合；questions为空（自测题由用户主动点击时从 `test_questions` 表抽取）
3. 步骤3：包装为 LearningPackage(version=1)，送 Agent4/Agent5 审核

**输出**：tutorial（四段式）+ questions(空) + code_examples(LoRALayer)，version=1

#### 场景2：审核未通过后改稿
**输入**：
```json
{
  "technical_errors": [
    {"error_type": "lora_mechanism_error", "severity": "critical", "location": "核心概念段", "description": "将LoRA描述为SVD分解"}
  ],
  "quality_errors": [
    {"issue_type": "missing_self_test", "severity": "minor", "location": "末尾"}
  ],
  "learning_package": {"tutorial": "...", "version": 1}
}
```

**处理过程**：
1. 步骤4：合并反馈，critical优先（LoRA机制错误）
2. 步骤5：LLM修订——将"SVD分解"改为"冻结W训练低秩增量ΔW=BA"；保留其余正确内容；自测题不纳入改稿范围
3. 步骤6：version=1→2，重新送审

**输出**：修订后tutorial（机制错误已修复），version=2

**说明**：场景2体现规则3改稿优先级——critical的LoRA机制错误必须先修复，正确内容保留不改。自测题由用户主动点击自测时从 `test_questions` 表抽取，不纳入教程生成与改稿阶段。

## 8. 代码纠错场景

### 8.1 触发条件
当用户输入包含以下关键词时，触发代码纠错流程：
- "纠错"、"修改代码"、"报错"、"bug"、"错误"、"修复"
- 用户提供了代码片段

### 8.2 处理流程
1. **识别意图**：检测用户输入是否为代码纠错请求
2. **提取代码**：从用户输入中提取代码片段
3. **执行验证**：调用 `code_executor_tool` 执行代码，获取真实执行反馈或报错信息
4. **加载策略**：优先使用 `code_correction_pedagogy.skill.md` 中的策略
5. **生成回答**：按照"错误分析→修改步骤→修改后完整代码→修改说明"四个板块输出
6. **验证修改**：对修改后的代码再次执行验证，确保修复成功

### 8.3 注意事项
- **必须执行验证**：严禁凭空猜测错误原因，必须先通过 `code_executor_tool` 获取真实反馈
- **结合当前阶段**：错误分析应联系当前学习阶段的知识点
- **遵循教学法**：严格按照 `code_correction_pedagogy.skill.md` 的解释风格和输出结构
- **仅修改必要部分**：保留用户代码中的正确部分，只修改有问题的地方

### 8.4 与 Agent3 的衔接
Agent3 在 `generate_answer` 方法中检测到代码纠错意图时：
1. 调用 `code_executor_tool.execute_code()` 获取执行结果
2. 加载 `code_correction_pedagogy` 技能作为 prompt 规范
3. 生成结构化纠错回答

### 8.5 输入输出示例

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

## 10. FAQ混合检索规则

### 10.1 触发条件
仅在用户主动提问时触发FAQ检索，其他场景（生成学习包、修改学习包、修改回答）不触发。

### 10.2 混合检索流程
```
用户问题
    ↓
步骤1：LLM提取关键词（3-5个）
    ↓
步骤2：Chroma向量检索（faq_collection集合）
    ↓ 命中 → 返回FAQ答案
    ↓ 未命中
步骤3：SQL LIKE关键词匹配兜底
    ↓ 命中 → 返回FAQ答案
    ↓ 未命中
步骤4：返回空列表（由上层调用LLM生成答案）
```

### 10.3 检索优先级
| 优先级 | 检索方式 | 适用场景 |
|:---|:---|:---|
| 1 | Chroma向量检索 | 语义相似度高的查询 |
| 2 | SQL LIKE关键词匹配 | 向量检索未命中时兜底 |

### 10.4 关键词提取规则
- 使用轻量prompt，仅提取3-5个关键词
- 控制token消耗：max_tokens=50，temperature=0.1
- 提取失败时，使用原始问题作为关键词

### 10.5 命中判定与返回
- 向量检索：similarity >= 0.6 视为命中
- SQL LIKE：匹配到任意关键词视为命中
- 命中时直接返回FAQ中的answer字段
- 未命中时返回空列表，由上层继续执行向量检索+LLM生成流程

### 10.6 与Agent3的衔接
Agent3 在 `answer_question` 方法中：
1. 先检测是否为代码纠错场景
2. 若非代码纠错，调用 `faq_search_tool.search()` 执行混合检索
3. 命中FAQ则直接返回答案
4. 未命中则继续执行原有的向量检索+LLM生成流程

### 10.7 注意事项
- **Token消耗控制**：LLM关键词提取仅在用户提问时触发，且使用轻量prompt
- **性能保障**：向量检索失败时自动降级为SQL LIKE，不影响系统可用性
- **数据一致性**：FAQ数据变更后需重新运行 `scripts/init_faq_vectors.py` 更新向量库

## 9. 版本信息
- 版本：v3.3，更新日期：2026-07-25
- 变更记录：能力描述明确3个适用场景与 `content_skill_context` 加载机制；核心方法论重构为6步（定位主题→生成四段式→送审→格式化反馈→逐条修订→迭代终止）；规则1新增四段式结构表（含缺失处置/severity）；规则2新增内容要素规范表；规则3新增改稿优先级与改稿规则；输入输出规范补完整 LearningPackage JSON示例与字段说明；异常处理扩展为6场景表格；附录A补4个教程生成理论（Sweller/Bloom/Piaget/Vygotsky）；附录B新增2个完整场景（首次生成+审核改稿，含输入JSON→处理过程→输出）；v3.1修订：自测题从教程生成阶段移除，改为用户主动点击自测时从 `test_questions` 表抽取；教程结构从五段式调整为四段式（已知回顾→核心概念拆解→实操演示→常见坑点）；v3.2修订：新增第8章代码纠错场景，定义代码纠错触发条件、处理流程、与Agent3的衔接方式及输入输出示例；v3.3修订：新增第10章FAQ混合检索规则，定义触发条件、混合检索流程、检索优先级、关键词提取规则、命中判定与返回、与Agent3的衔接方式及注意事项
