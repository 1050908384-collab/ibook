# SKILL名称：question_answering

## 1. 能力描述
- 功能：为Agent3（HybridContentAgent）提供**答疑生成与改稿的方法论**——定义"一次合格的学习答疑长什么样"：结论先行、原理支撑、不编造来源、聚焦当前阶段。Agent3调用本SKILL针对用户提问生成简明答疑，并依据Agent4技术审核与Agent5教学审核意见迭代修订。在 `HybridContentAgent.generate_answer` 中作为 prompt 规范；在 `revise_answer` 中作为改稿依据
- 适用场景（3个）：
  1. **实时答疑**：用户在学习过程中提问，由 `generate_answer` 触发，结合 `stage` 与 `profile` 生成回答
  2. **审核后改稿**：Agent4/Agent5审核答疑未通过时由 `revise_answer` 触发，接收 `technical_errors` 与 `quality_errors` 逐条修订
  3. **跨阶段问题处理**：用户提问与当前阶段无关时，简短作答并提示聚焦本阶段
- 前置依赖：
  - 用户提问文本（`question`）
  - 当前阶段 StageInfo（`block_name`，判断问题相关性）
  - 用户画像 UserProfile（`goal`、`learning_style`，调整答疑深度）
  - 审核反馈（修订时：`technical_errors` 来自 Agent4，`quality_errors` 来自 Agent5）

## 2. 核心方法论
- **步骤1：定位问题与当前阶段关系**——读取 `stage.block_name` 与 `profile.goal`，判断提问是否属于当前阶段；属于则深入答，无关则简短答并提示聚焦
- **步骤2：调用LLM生成简明答疑**——先给结论，再解释原理，需要时配示例；遵循"结论→原理→示例"三段结构
- **步骤3：交付审核**——answer 交 Orchestrator 送 Agent4（技术准确性、幻觉检测）与 Agent5（教学清晰度）双重审核
- **步骤4：格式化审核反馈**——将 `technical_errors` 与 `quality_errors` 合并为反馈清单，按 severity 降序排列
- **步骤5：逐条修订**——调用LLM按反馈清单逐条修订回答，保留正确内容，仅修改 errors 指向部分，返回完整新 answer
- **步骤6：迭代终止判定**——审核通过或迭代轮次达 `MAX_REVIEW_ITERATIONS`（默认3）时终止

## 3. 知识/规则定义

### 规则1：答疑内容规范
| 检查项 | 要求 | 违规处置 | severity |
|:---|:---|:---|:---|
| 结论先行 | 首句给明确结论，再展开原理 | 标记 `missing_conclusion` | major |
| 不编造来源 | 不得编造参数、模型原理、论文；无法确认时明确说明"暂无确切依据" | 标记 `hallucination`，交Agent4 sim<0.65判critical | critical |
| 原理支撑 | 结论后须有原理/机制解释，非空泛断言 | 标记 `unsupported_claim` | major |
| 示例佐证 | 需要时配代码/数值示例（代码遵循code_style） | 标记 `missing_example` | minor |
| 聚焦阶段 | 与当前阶段无关的问题简短作答，提示回归本阶段 | 标记 `off_topic` | minor |

### 规则2：答疑风格规范
| 检查项 | 要求 | 违规处置 | severity |
|:---|:---|:---|:---|
| 简洁直击 | 回答简洁，复杂原理分段解释，避免冗长 | 标记 `too_verbose` | minor |
| 难度适配 | 面向 `profile.goal` 与 `learning_style`，practice风格多给代码，theory风格多给原理 | 标记 `difficulty_mismatch` | major |
| 术语解释 | 答疑中首次出现的术语配简明解释 | 标记 `unexplained_term` | minor |
| 中性客观 | 不带主观评价，不贬低提问质量 | 标记 `subjective_tone` | minor |

### 规则3：改稿优先级
**修订顺序（从高到低）**：
1. **critical 错误**（幻觉 sim<0.65、编造参数/原理、安全违规）——必须修复
2. **major 错误**（缺结论、无原理支撑、难度不当、事实错误）——必须修复
3. **minor 错误**（冗长、缺示例、术语未解释、偏题）——尽量修复

**改稿规则**：
- 逐条解决审核 errors，不得遗漏 critical/major 项
- 保留审核未指出的正确内容，不做无谓改写
- 审核意见为空时保留原 answer，直接通过
- 幻觉类错误改稿时须删除无法验证的声明或补充可靠依据，不得换个说法重述

## 4. 输入输出规范

### 输入（生成）
`AnswerGenerationInput` 对象：
- `session_id`
- `question`：用户提问文本
- `stage`：StageInfo（`block_name`）
- `profile`：UserProfile（`goal`、`learning_style`）

### 输入（修订）
`AnswerRevisionInput` 对象：
- `session_id`
- `question`、`answer`（当前回答）
- `stage`、`profile`
- `technical_errors`：List[ReviewError]
- `quality_errors`：List[QualityIssue]

### 输出
`AnswerGenerationOutput` 对象，含 `answer: str`：
```json
{
  "answer": "LoRA训练的是两个低秩矩阵 A 和 B，而非原始权重 W。\n\n原理：LoRA 冻结原权重 W，仅训练低秩增量 ΔW=(α/r)BA，其中 A 是 r×d_in、B 是 d_out×r，r 远小于 d_in/d_out，因此可训练参数量仅为 r×(d_in+d_out)，远小于原参数 d_in×d_out。\n\n示例：d_in=d_out=4096、r=8 时，原参数约1678万，LoRA参数约6.5万，降幅超99%。"
}
```

**字段说明**：
- `answer`：答疑正文字符串，遵循"结论→原理→示例"结构

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| LLM返回空或Error前缀 | 回退 `DeterministicContentAgent.generate_answer` 基础话术（"关于该问题，建议参考当前阶段教程的XX部分"） |
| 提问与当前阶段无关 | 简短作答（1-2句），提示"建议在XX阶段深入学习此问题" |
| LLM疑似编造来源/参数 | 交 Agent4 幻觉检测（sim<0.65），回流 critical error 后改稿删除或补依据 |
| 修订后LLM响应无效 | 保留原 answer，追加"补充修订"说明，重新送审 |
| 审核轮次达 `MAX_REVIEW_ITERATIONS`(3) | 保留最后版本，标注"未通过审核"，终止迭代 |
| `question` 为空或过短（<5字符） | 返回引导话术"请详细描述你的问题，例如XX方面的疑问" |

## 6. 附录

### 附录A：答疑方法论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | 苏格拉底式提问法（Socratic Method） | Socrates / 经由 Plato 记录 | 公元前5世纪 | 通过追问引导学习者自行发现答案，而非直接灌输；答疑应启发而非替代思考 | 规则1结论先行+原理支撑的依据——给结论但同时解释原理，让学习者理解"为什么" |
| A-2 | 形成性评价理论（Formative Assessment） | Michael Scriven / Benjamin Bloom | 1967/1968 | 在学习过程中提供即时反馈，帮助学习者纠偏；反馈应具体、及时、可操作 | 答疑作为学习过程中的即时反馈机制，回答须具体可操作（配示例） |
| A-3 | 认知负荷理论（Cognitive Load Theory） | John Sweller | 1988 | 工作记忆容量有限，信息呈现应减少外在负荷，结论先行可降低认知负荷 | 规则1结论先行+规则2简洁直击的依据——先给结论降低认知负荷，再补原理 |
| A-4 | 信息真实性原则（RAG/检索增强范式） | Lewis 等 | 2020 | 生成式回答须有可追溯依据，低置信声明应明确标注不确定性，避免幻觉 | 规则1不编造来源的依据——无法确认时明确说明，交Agent4幻觉检测兜底 |

### 附录B：完整场景示例

#### 场景1：实时答疑（合格回答）
**输入**：
```json
{
  "session_id": "sess-001",
  "question": "LoRA微调训练的是哪些矩阵？为什么参数量会减少？",
  "stage": {"block_name": "LoRA"},
  "profile": {"goal": "掌握LoRA微调", "learning_style": "practice"}
}
```

**处理过程**：
1. 步骤1：提问与当前阶段(LoRA)强相关，深入答
2. 步骤2：LLM生成——结论"训练A和B两矩阵非W"；原理"冻结W训练ΔW=BA，参数r(d_in+d_out)"；示例"d_in=d_out=4096,r=8降幅99%"
3. 步骤3：送 Agent4（技术准确+幻觉检测）与 Agent5（教学清晰）审核

**输出**：answer（结论→原理→示例三段），审核通过

#### 场景2：审核未通过（幻觉）后改稿
**输入**：
```json
{
  "question": "LoRA微调训练的是哪些矩阵？",
  "answer": "LoRA通过对原始权重W做SVD分解，训练U、Σ、V三个矩阵来实现参数压缩。",
  "technical_errors": [
    {"error_type": "lora_mechanism_error", "severity": "critical", "location": "原理段", "description": "将LoRA描述为SVD分解，混淆LoRA与权重矩阵分解", "evidence": "rule_check_tool正则命中：lora.*?(?:svd|奇异值分解)"}
  ]
}
```

**处理过程**：
1. 步骤4：反馈清单——critical（LoRA机制错误，将LoRA误述为SVD）
2. 步骤5：LLM修订——删除"SVD分解训练U/Σ/V"的错误陈述，改为"冻结W训练低秩增量ΔW=(α/r)BA，A与B是两个低秩矩阵"；保留结论部分
3. 步骤6：重新送审，version无需（answer无version字段，直接替换）

**输出**：修订后 answer（机制错误已修复，结论保留）

**说明**：场景2体现规则3改稿优先级——critical的幻觉/机制错误必须先修复，且不得换个说法重述错误（须删除错误陈述改为正确机制），符合"幻觉类错误改稿时须删除无法验证的声明"。

## 7. 版本信息
- 版本：v3.0，更新日期：2026-07-23
- 变更记录：从v2.0骨架升级为v3.0详尽格式；能力描述明确3个适用场景（实时答疑/审核改稿/跨阶段处理）；核心方法论重构为6步（定位关系→生成简明答疑→送审→格式化反馈→逐条修订→迭代终止）；规则1新增答疑内容规范表（含不编造来源的critical处置）；规则2新增答疑风格规范表；规则3新增改稿优先级与幻觉改稿规则；输入输出规范补完整 AnswerGenerationOutput JSON示例；异常处理扩展为6场景表格；附录A补4个答疑方法论（苏格拉底法/形成性评价/认知负荷/RAG真实性）；附录B新增2个完整场景（合格答疑+幻觉改稿，含输入JSON→处理过程→输出）
