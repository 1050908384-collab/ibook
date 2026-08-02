# SKILL名称：learner_profiling

## 1. 能力描述
- 功能：从用户自然语言输入提取**标准化学习者画像**——技能标签（skill_tags）、知识盲区（knowledge_gaps，每项含 skill 与 level）、学习目标（goal）、每日时长（daily_hours）、学习风格（learning_style），并评估置信度（confidence），置信不足时生成追问。对齐 Agent1 `HybridDiagnosticAgent` 的单次 LLM 提取 + 正则回退行为
- 适用场景（3个）：
  1. **首次对话诊断**：新会话首条消息触发 `HybridDiagnosticAgent.run`，从用户输入提取画像
  2. **置信不足追问**：confidence < `CONFIDENCE_THRESHOLD`(0.7) 时返回 `needs_more_info` + follow_up_questions 补齐信息
  3. **LLM失败回退**：LLM返回非JSON时回退 `DeterministicDiagnosticAgent` 正则匹配 `SKILL_NAMES`
- **休眠状态说明**：当前新框架 `HybridDiagnosticAgent` 未通过 `load_skills` 加载本SKILL（诊断逻辑内联于代码），本SKILL内容已对齐其行为，作为休眠方法论备查；若将来恢复 skill 驱动诊断可直接启用
- 前置依赖：
  - 用户原始对话文本（`user_input`）
  - 可选历史交互（当前新框架未使用）
  - `SKILL_NAMES` 常量（确定性回退路径的正则词表）

## 2. 核心方法论
- **步骤1：识别输入中的画像要素**——识别技术术语（已掌握技能）、学习目标、每日时长、学习基础与风格偏好
- **步骤2：输出标准化画像字段**——`skill_tags`（已掌握技能列表）、`knowledge_gaps`（待学项，每项含 `skill` 与 `level`：beginner/intermediate/advanced）
- **步骤3：提取补充字段**——`goal`、`daily_hours`（未知取2）、`learning_style`（theory/practice/mixed，未知取 mixed）
- **步骤4：评估置信度**——按加权公式计算 `confidence = clamp(base×0.5 + completeness×0.5, 0.3, 0.95)`（base：LLM=0.9/回退=0.75；completeness 为各维度评分之和，详见规则2）；低于 `CONFIDENCE_THRESHOLD`(0.7) 时生成 `follow_up_questions` 补齐
- **步骤5：LLM失败回退**——LLM返回非JSON或解析失败时，回退 `DeterministicDiagnosticAgent` 用正则匹配 `SKILL_NAMES` 提取技能标签，confidence=0.75

## 3. 知识/规则定义

### 规则1：画像字段规范
| 字段 | 类型 | 要求 | 违规处置 |
|:---|:---|:---|:---|
| `skill_tags` | List[str] | 已掌握技能名称列表 | 为空时不影响confidence，但触发追问 |
| `knowledge_gaps` | List[Dict] | 每项含 `skill` 与 `level`(beginner/intermediate/advanced) | 缺 level 时默认 beginner |
| `goal` | str | 学习目标描述 | 为空时默认"通用学习目标"，confidence=0.75 |
| `daily_hours` | float | 每日学习时长 | 未知时取2 |
| `learning_style` | str | theory/practice/mixed | 未知时取 mixed |
| `confidence` | float | 0~1 置信度 | <0.7触发追问 |

### 规则2：置信度计算公式与追问机制

置信度采用**加权公式**（对齐 `calculate_confidence_tool._calculate`），非固定值：

```
base = 0.9（LLM正常返回）| 0.75（LLM回退正则）
completeness = skills_score + gaps_score + goal_score + hours_score + style_score
confidence = clamp(base * 0.5 + completeness * 0.5, 0.3, 0.95)
```

**完整度各维度评分**：
| 维度 | 分值 | 条件 |
|:---|:---|:---|
| skills_score | min(len(skill_tags) × 0.1, 0.3) | 已掌握技能标签，最多0.3 |
| gaps_score | min(len(knowledge_gaps) × 0.1, 0.3) | 知识盲区，最多0.3 |
| goal_score | 0.2 | 有学习目标 |
| hours_score | 0.1 | 有每日时长且>0 |
| style_score | 0.1 | 有学习风格 |

**置信度边界**：
| 场景 | base | completeness | confidence | 处置 |
|:---|:---|:---|:---|:---|
| LLM+信息完整(理论最大) | 0.9 | 1.0 | 0.95 | 直接输出画像 |
| LLM+典型完整(2技能1盲区+目标+时长+风格) | 0.9 | 0.7 | 0.80 | 直接输出画像 |
| LLM+部分信息 | 0.9 | 0.3~0.5 | 0.45~0.55 | <0.7 触发追问 |
| LLM+无信息 | 0.9 | 0.0 | 0.45 | <0.7 触发追问 |
| LLM回退+信息完整(理论最大) | 0.75 | 1.0 | 0.875 | 标注"估算画像"，输出 |
| LLM回退+典型完整 | 0.75 | 0.6~0.7 | 0.675~0.725 | 临界，可能触发追问 |
| LLM回退+无信息 | 0.75 | 0.0 | 0.375 | <0.7 触发追问 |
| 信息严重不足 | — | — | clamp下限0.3 | 触发追问 |

> **注**：completeness 理论最大值 1.0 需同时满足 3+技能(0.3) + 3+盲区(0.3) + 目标(0.2) + 时长(0.1) + 风格(0.1)

**追问触发条件**：confidence < `CONFIDENCE_THRESHOLD`(0.7) 时返回 `needs_more_info=true` + follow_up_questions

**追问生成规则**：
- 缺目标 → 追问"你希望达成什么学习目标？"
- 缺基础 → 追问"你目前已掌握哪些技能？"
- 缺时长 → 追问"你每天可投入多少小时学习？"
- 缺风格 → 追问"你偏好哪种学习方式？（理论学习/动手实践/两者结合）"

### 规则3：UserProfile 契约对齐
- 画像字段须符合 `UserProfile` 契约（`backend/models/contracts.py` 与 `profile.py`）
- `knowledge_gaps` 每项完整结构：`skill`、`level`、`status`（默认"未开始"）、`progress`（默认0.0）、`prerequisites`（默认[]）
- LLM必须输出严格JSON，解析失败回退确定性诊断

## 4. 输入输出规范

### 输入
`DiagnosticInput` 对象：
- `session_id`
- `user_input`：用户原始对话文本

### 输出
`DiagnosticOutput` 对象：
```json
{
  "profile": {
    "skill_tags": ["Python", "PyTorch"],
    "knowledge_gaps": [
      {"skill": "LoRA", "level": "beginner", "status": "未开始", "progress": 0.0, "prerequisites": ["Transformer"]},
      {"skill": "Transformer", "level": "intermediate", "status": "未开始", "progress": 0.0, "prerequisites": ["注意力机制"]}
    ],
    "goal": "掌握大模型微调技术",
    "daily_hours": 2,
    "learning_style": "practice",
    "confidence": 0.9
  },
  "follow_up_questions": []
}
```

**字段说明**：
- `profile`：UserProfile 对象
- `follow_up_questions`：追问列表（confidence≥0.7时为空数组）

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 用户输入过短/无有效信息 | 降低confidence<0.7，触发追问 |
| LLM返回非JSON | 回退 `DeterministicDiagnosticAgent` 正则匹配 `SKILL_NAMES`，confidence=0.75 |
| 无识别到目标 | 默认"通用学习目标"，confidence=0.75 |
| knowledge_gaps为空 | 不生成阶段，置信度压低并由 Agent1 返回追问或完成状态（无盲区时阶段选路抛异常） |
| LLM提取的level非枚举值 | 自动修正为 beginner |

## 6. 附录

### 附录A：学习者画像理论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | 贝叶斯知识追踪（BKT） | Corbett & Anderson | 1995 | 用隐马尔可夫模型追踪学习者每个知识点的掌握概率，基于答题更新 | knowledge_gaps 的 status/progress 字段依据——追踪每项盲区的掌握状态与进度 |
| A-2 | 深度知识追踪（DKT） | Piech 等 | 2015 | 用循环神经网络建模学习者知识状态，捕捉知识点间依赖关系 | knowledge_gaps 的 prerequisites 字段依据——盲区间存在先修依赖 |
| A-3 | 最近发展区理论（ZPD） | Lev Vygotsky | 1978 | 学习内容应处于"已知但未掌握"的最近发展区，需先识别已知基线 | skill_tags（已知）与 knowledge_gaps（未知）的划分依据 |
| A-4 | 学习风格模型（VARK） | Fleming & Mills | 1992 | 学习者偏好分为视觉/听觉/读写/动觉，适配风格可提升效率 | learning_style 字段（theory/practice/mixed）的简化映射依据 |

### 附录B：完整场景示例

#### 场景1：首次对话诊断（LLM正常路径）
**输入**：
```json
{
  "session_id": "sess-001",
  "user_input": "我已经会Python和PyTorch，想学大模型微调，每天能学2小时，喜欢动手实践。"
}
```

**处理过程**：
1. 步骤1：识别已掌握[Python,PyTorch]、目标=大模型微调、时长=2、风格=practice
2. 步骤2：knowledge_gaps 推导=[LoRA(beginner), Transformer(intermediate)]
3. 步骤3：goal/daily_hours/learning_style 填充
4. 步骤4：信息完整，confidence=0.9，无需追问

**输出**：profile（含2个gap）+ 空追问列表

#### 场景2：LLM失败回退正则
**输入**：`{"user_input": "我想学LoRA和Transformer"}`，LLM返回非JSON
**处理过程**：
1. 步骤5：LLM解析失败 → 回退 `DeterministicDiagnosticAgent`
2. 正则匹配 `SKILL_NAMES`：命中"LoRA"、"Transformer" → knowledge_gaps=[LoRA, Transformer]
3. 缺 goal/daily_hours/learning_style → goal默认"通用学习目标"，daily_hours=2，learning_style=mixed
4. confidence=0.75（标注"估算画像"）

**输出**：profile（估算）+ 空追问列表（0.75≥0.7不触发追问）

**说明**：场景2体现规则2的回退机制——LLM失败时不阻断流程，用确定性正则保底产出可用画像，confidence降级标注估算。

## 7. 版本信息
- 版本：v3.1，更新日期：2026-08-02
- 变更记录：v3.0基础同上；v3.1修订：规则2置信度从固定值（LLM=0.9/回退=0.75）改为加权公式 `clamp(base×0.5 + completeness×0.5, 0.3, 0.95)`，新增完整度各维度评分表与置信度边界表，追问规则补充"缺风格"分支，对齐 `calculate_confidence_tool._calculate` 实际实现
