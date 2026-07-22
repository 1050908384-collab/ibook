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
- **步骤2：一次性生成五段式结构教程**——调用LLM按"已知回顾→核心概念拆解→实操演示→常见坑点→自测"五段生成Markdown；核心概念配代码示例（遵循 code_style），自测段含1~2道TestQuestion（填空/选择优先）
- **步骤3：交付审核**——tutorial 交 `DeterministicContentAgent` 包装为 `LearningPackage(tutorial, questions, code_examples, version=1)`，由 Orchestrator 送 Agent4/Agent5 双重审核
- **步骤4：格式化审核反馈**——将 `technical_errors`（ReviewError数组）与 `quality_errors`（QualityIssue数组）合并为反馈清单，按 severity 降序排列（critical > major > minor）
- **步骤5：逐条修订**——调用LLM按反馈清单逐条修订教程，保留正确内容，仅修改 errors 指向的 location，返回完整新 tutorial，`version` 自增
- **步骤6：迭代终止判定**——审核通过或迭代轮次达 `MAX_REVIEW_ITERATIONS`（默认3）时终止；未通过则保留最后版本并标注未通过状态

## 3. 知识/规则定义

### 规则1：五段式结构规范
| 段落 | 必含要素 | 缺失处置 | severity |
|:---|:---|:---|:---|
| 已知回顾 | 链接 `profile.skill_tags` 中相关已掌握技能 | 标记 `missing_review` | major |
| 核心概念拆解 | `block_name` 的定义/原理，每段≤2个新概念 | 标记 `concept_overload` | major |
| 实操演示 | 可运行代码示例（遵循code_style）或真实场景案例 | 标记 `missing_context_case` | major |
| 常见坑点 | 2~3个典型误区/陷阱 | 标记 `missing_pitfalls` | minor |
| 自测 | 1~2道TestQuestion（填空/选择优先） | 标记 `missing_self_test` | minor |

**结构完整性依据**：五段式对应"已知→未知→实践→反思→检验"的认知闭环，与 `pedagogy.skill.md` 规则2的渐进式讲解规范一致。

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
- 改稿后 tutorial 必须仍保持五段式结构完整性

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
    "tutorial": "# LoRA微调入门\n\n## 已知回顾\n你已经掌握了Transformer的注意力机制...\n\n## 核心概念拆解\nLoRA冻结原权重W，训练低秩增量ΔW=(α/r)BA...\n\n## 实操演示\n```python\nclass LoRALayer(nn.Module):\n    ...\n```\n\n## 常见坑点\n- 秩r过大导致过拟合\n\n## 自测\n1. LoRA训练哪两个矩阵？",
    "questions": [
      {"question_type": "填空", "question_text": "LoRA可训练参数量公式：r*(?)+r*(?)", "correct_answer": "d_in, d_out"}
    ],
    "code_examples": [
      {"title": "LoRA最小示例", "language": "python", "code": "..."}
    ],
    "version": 2
  }
}
```

**字段说明**：
- `tutorial`：Markdown教程正文（五段式）
- `questions`：自测题（`DeterministicContentAgent` 基类生成1道填空题，LLM可补充）
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
2. 步骤2：LLM生成五段式——已知回顾链接Transformer注意力机制；核心概念讲ΔW=(α/r)BA；实操给LoRALayer代码；坑点讲r过大过拟合；自测1道填空（参数量公式）
3. 步骤3：包装为 LearningPackage(version=1)，送 Agent4/Agent5 审核

**输出**：tutorial（五段式）+ questions(1道填空) + code_examples(LoRALayer)，version=1

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
1. 步骤4：合并反馈，critical优先（LoRA机制错误），其次minor（缺自测）
2. 步骤5：LLM修订——将"SVD分解"改为"冻结W训练低秩增量ΔW=BA"；补1道自测题；保留其余正确内容
3. 步骤6：version=1→2，重新送审

**输出**：修订后tutorial（机制错误已修复+自测已补），version=2

**说明**：场景2体现规则3改稿优先级——critical的LoRA机制错误必须先修复，minor的缺自测一并补全，正确内容保留不改。

## 7. 版本信息
- 版本：v3.0，更新日期：2026-07-23
- 变更记录：从v2.0骨架升级为v3.0详尽格式；能力描述明确3个适用场景与 `content_skill_context` 加载机制；核心方法论重构为6步（定位主题→生成五段式→送审→格式化反馈→逐条修订→迭代终止）；规则1新增五段式结构表（含缺失处置/severity）；规则2新增内容要素规范表；规则3新增改稿优先级与改稿规则；输入输出规范补完整 LearningPackage JSON示例与字段说明；异常处理扩展为6场景表格；附录A补4个教程生成理论（Sweller/Bloom/Piaget/Vygotsky）；附录B新增2个完整场景（首次生成+审核改稿，含输入JSON→处理过程→输出）
