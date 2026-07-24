# SKILL名称：learner_model

## 1. 能力描述
- 功能：为Agent⑤提供**学习者特征基准库**，定义"不同人适合什么"——三类用户基准（在校学生/工程师/研究者）、难度适配映射、风格适配规则、环境容忍度规则。Agent⑤调用本SKILL根据学习者类型调整难度和风格判断标准，输出用户类型判定和三维匹配度评分。在 `HybridQualityReviewAgent` 中被加载为 `skill_context`，结合用户画像评估内容难度、风格、环境是否匹配
- 适用场景（3个）：
  1. **判断难度匹配度**：根据用户 `knowledge_gaps[].level` 和内容概念密度，评估难度是否处于i+1区间
  2. **判断风格适配度**：根据用户 `learning_style` 和内容理论/实践比例，评估风格是否匹配
  3. **评估环境可行性**：根据用户环境经验（Docker/conda/GPU），评估内容中的部署示例是否可在用户环境运行
- 前置依赖：
  - 画像中的 `professional_background`（职业背景，用于识别用户类型）
  - 画像中的 `learning_style`（学习风格：theory/practice/mixed）
  - 画像中的 `skills_mastered_list`（已掌握技能列表，对应 `UserProfile.skill_tags`）
  - 画像中的 `knowledge_gaps`（含 level 字段，用于难度适配）
  - 画像中的 `daily_hours`（每日学习时长，用于时间适配）

## 2. 核心方法论
- **步骤1：识别用户类型（在校学生/工程师/研究者）**——根据 `professional_background`、`skill_tags`、`goal` 字段综合判定：
  - 含"学生/大一/大二/研一/本科/研究生"等关键词 → 在校学生
  - 含"工程师/开发/后端/前端/算法工程师/从业"等关键词 → 工程师
  - 含"研究/博士/论文/学术/教授"等关键词 → 研究者
  - 无明确标识 → 默认在校学生
- **步骤2：根据类型映射难度基准和风格偏好**——查规则2难度适配映射表得到该用户的概念密度上限和代码行数上限，查规则3风格适配规则得到该用户的理实比例偏好
- **步骤3：比对内容特征输出匹配度评分**——计算三维匹配度：
  - `difficulty_match`(0-1)：内容概念密度与用户level基准的匹配度
  - `style_match`(0-1)：内容理论/实践比例与用户 `learning_style` 的匹配度
  - `environment_fit`(0/1)：内容部署示例与用户环境经验的可行性

## 3. 知识/规则定义

### 规则1：三类用户基准
| 用户类型 | 识别特征 | 内容偏好 | 容忍度低项 | 推荐教学风格 | 概念密度建议 |
|:---|:---|:---|:---|:---|:---|
| 在校学生 | `professional_background`含"学生/大一/大二/研一/本科" | 结构化、循序渐进的讲解，先理论后实践，需要完整前置知识铺垫 | 跳过基础直接讲高阶、概念无铺垫直接使用 | mixed（理论40%+实践60%） | ≤5个/千字 |
| 工程师 | `professional_background`含"工程师/开发/后端/前端/从业" | 案例驱动、可直接运行的代码，对代码质量敏感 | 大段原理讲解无代码、代码片段不可运行 | practice（理论20%+实践80%） | 4~7个/千字 |
| 研究者 | `professional_background`含"研究/博士/论文/学术/教授" | 公式推导、论文溯源、原理解析，需要理论深度 | 简单的实操细节无深度、缺乏文献引用 | theory（理论70%+实践30%） | 6~10个/千字 |

**默认规则**：若用户画像无明确类型标识，默认按"在校学生"基准评估，并在评估结果中标注 `"note": "用户类型未识别，采用默认基准"`

### 规则2：难度适配映射
| 用户level | 概念密度上限 | 代码行数上限 | 难度特征 |
|:---|:---|:---|:---|
| beginner（初级） | ≤3个/千字 | ≤50行 | 概念逐一解释，代码为最小可运行示例 |
| intermediate（中级） | 4~7个/千字 | 50~150行 | 概念适度展开，代码含完整训练流程 |
| advanced（高级） | ≥8个/千字 | ≥150行 | 概念高密度，代码含工程化实践（分布式、混合精度） |

**匹配度计算**：
- `difficulty_match` = 1.0（内容密度在用户level区间内）
- `difficulty_match` = 0.6（内容密度偏离用户level区间±2个/千字）
- `difficulty_match` = 0.3（内容密度严重偏离，如beginner用户遇到advanced内容）
- 若 `difficulty_match` < 0.5，标记 issue_type=`"difficulty_mismatch"`，severity=major

### 规则3：风格适配规则
| 用户 `learning_style` | 匹配的内容特征 | 不匹配的内容特征 |
|:---|:---|:---|
| `"动手操作"` / `"practice"` | 案例驱动、代码实操、可运行示例占主体 | 大段公式推导、纯理论讲解无代码 |
| `"理论讲解"` / `"theory"` | 概念拆解、原理推导、公式与文献引用 | 仅代码示例无原理解释 |
| `"混合型"` / `"mixed"` | 理论40%+实践60%，先概念后代码 | 极端偏向（纯理论或纯代码） |

**匹配度计算**：
- 统计内容中理论段落字数与代码+实操段落字数的比例
- `style_match` = 1.0（比例在用户偏好区间内）
- `style_match` = 0.5（比例偏离用户偏好±20%）
- `style_match` < 0.5时，标记 issue_type=`"style_mismatch"`，severity=major

### 规则4：环境容忍度规则
**环境可行性检查**：
- 内容中涉及Docker部署示例（检测关键词：`docker`、`docker run`、`Dockerfile`、`docker-compose`）时，检查用户画像是否有Docker经验
- 内容中涉及GPU显存要求（检测关键词：`cuda`、`gpu`、`显存`、`device_map`）时，检查用户环境是否支持GPU
- 内容中涉及分布式训练（检测关键词：`distributed`、`DDP`、`FSDP`、`多卡`）时，检查用户是否有多卡环境

**容忍度规则**：
| 用户环境经验 | 内容环境要求 | environment_fit | 处置 |
|:---|:---|:---|:---|
| 无Docker经验 | 内容使用Docker部署 | 0（不可行） | 标记 issue_type=`"environment_mismatch"`，severity=major，suggestion="提供conda环境替代方案：`conda create -n lora python=3.10`" |
| 无GPU环境 | 内容需GPU推理 | 0（不可行） | 标记 issue_type=`"environment_mismatch"`，severity=major，suggestion="提供CPU替代方案或Google Colab免费GPU使用说明" |
| 无多卡环境 | 内容需分布式训练 | 0（不可行） | 标记 issue_type=`"environment_mismatch"`，severity=minor，suggestion="提供单卡替代方案（如减小batch_size或使用梯度累积）" |
| 环境经验充足 | 内容环境要求匹配 | 1（可行） | 通过 |

**替代方案库**：
- Docker → conda：`conda create -n <name> python=3.10 && conda activate <name>`
- GPU → CPU：`device = torch.device("cpu")`，并提示性能下降预期
- 多卡 → 单卡梯度累积：`for _ in range(accumulation_steps): loss.backward()`

## 4. 输入输出规范

### 输入
- `user_profile`（UserProfile对象，含 `skill_tags`、`learning_style`、`knowledge_gaps[].level`、`professional_background`、`daily_hours`、`goal`）
- 内容特征标签（由Agent⑤提取，含 概念密度、代码行数、理论/实践比例、环境要求）

### 输出（数据流向quality_metrics）
**本SKILL输出的核心数据将直接提供给 `quality_metrics` 用于加权评分**：
- `difficulty_match`(0-1) → 对应 `quality_metrics` 的 **难度匹配度** 维度（权重20%）
- `style_match`(0-1) → 对应 `quality_metrics` 的 **风格适配度** 维度（权重10%）
- `environment_fit`(0/1) → 对应 `quality_metrics` 的 **环境可行性** 维度（权重10%），**同时用于红线检查**
- `issues` 数组 → 合并到 `QualityReviewResult.errors`（用于附加条件检查）
学习者适配评估结果，LLM返回JSON格式：
```json
{
  "user_type": "工程师",
  "difficulty_match": 0.8,
  "style_match": 0.4,
  "environment_fit": 1,
  "issues": [
    {
      "issue_type": "style_mismatch",
      "severity": "major",
      "location": "内容整体",
      "description": "用户learning_style为practice，但内容理论讲解占70%，与用户偏好不匹配",
      "suggestion": "减少公式推导部分，增加完整的LoRA微调代码示例（含数据加载、训练、保存、推理全流程）"
    }
  ],
  "summary": "用户为工程师型，偏好案例驱动内容。当前内容理论比例过高（70%），风格适配度0.4低于0.5阈值，难度和环境适配良好。"
}
```

**字段说明**：
- `user_type`：`"在校学生"` / `"工程师"` / `"研究者"` / `"未识别"`
- `difficulty_match`：0-1浮点数，难度匹配度
- `style_match`：0-1浮点数，风格适配度
- `environment_fit`：0或1，环境可行性（0=不可行，1=可行）
- `issues`：问题列表，每项含 issue_type/severity/location/description/suggestion
- `summary`：人类可读的评估总结

**与QualityReviewResult的映射**：`difficulty_match`×100映射到 `scores["difficulty"]`，`style_match`×100映射到 `scores["style_match"]`，`environment_fit`×100映射到 `scores["environment_fit"]`，issues数组映射到 `QualityReviewResult.errors`。

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 画像数据不完整（`professional_background` 为空或无法识别用户类型） | 默认按"在校学生"类型评估，在issues中标注 issue_type=`"profile_incomplete"`，severity=minor，suggestion="建议补充用户职业背景信息以提升适配精度" |
| 混合型用户（同时具备工程师和研究者特征，如"算法工程师兼博士生"） | 取各维度的加权平均：difficulty_match = 0.5×工程师基准 + 0.5×研究者基准，在summary中标注"混合型用户，采用加权评估" |
| `learning_style` 值不在 `{"theory","practice","mixed"}` 范围内 | 默认按 `"mixed"` 处理，记录日志，在issues中标注 issue_type=`"invalid_learning_style"`，severity=minor |
| `knowledge_gaps[].level` 为空 | 默认按 `"beginner"` 处理，difficulty_match按初级基准计算 |
| 内容无代码块（无法统计代码行数） | 跳过代码行数检查，仅按概念密度评估difficulty_match，在summary中注明"无代码块，仅按概念密度评估" |
| 用户环境经验字段缺失 | 默认按"无特殊环境经验"处理，environment_fit检查时若内容涉及Docker/GPU则标记待人工确认 |
| 内容审核循环达上限（Agent4+5并行审核循环≥3次仍未通过） | Orchestrator强制终止循环，status=`"review_needed"`，summary标注"审核循环达3次上限，建议人工介入修订内容" |

## 6. 附录

### 附录A：学习者风格与认知理论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | Felder-Silverman学习风格模型 | Richard Felder & Linda Silverman | 1988 | 将学习风格分为4个维度：主动-反思（信息加工）、感官-直觉（感知）、视觉-言语（输入）、序列-全局（理解）。不同风格的学习者对不同呈现方式有偏好 | 规则3风格适配规则的依据——`learning_style` 为practice对应"主动+感官+视觉"偏好，theory对应"反思+直觉+言语"偏好，混合型对应多维均衡 |
| A-2 | Kolb经验学习理论 | David Kolb | 1984 | 学习是一个循环过程：具体经验→反思观察→抽象概念化→主动实验。不同学习者在循环中偏好不同阶段，形成4种学习风格：发散者、同化者、聚合者、适应者 | 规则1三类用户基准的依据——工程师偏好"具体经验+主动实验"（适应者），研究者偏好"抽象概念化+反思观察"（同化者），学生偏好均衡循环 |
| A-3 | Gardner多元智能理论 | Howard Gardner | 1983 | 智能分为8种类型（语言、逻辑数学、空间、身体动觉等），不同智能组合影响学习偏好 | 辅助解释为何工程师偏好代码实操（逻辑数学+身体动觉），研究者偏好公式推导（逻辑数学+语言） |

### 附录B：工程师型用户 vs 学生型用户的内容偏好对比示例

#### 场景：同一份LoRA微调内容，不同用户类型的适配评估

**待评估内容特征**：
```
内容字数：约1000字
新概念密度：6个/千字（LoRA、低秩矩阵、秩r、缩放系数α、增量矩阵ΔW、冻结权重）
代码行数：80行（完整LoRA训练流程）
教学风格：理论讲解占30%，代码实操占70%
环境要求：需GPU（device_map="auto"）
```

#### 工程师型用户评估
**用户画像**：
```json
{
  "professional_background": "后端工程师，3年Python开发经验",
  "skill_tags": ["Python", "PyTorch"],
  "learning_style": "practice",
  "knowledge_gaps": [{"skill": "LoRA", "level": "beginner"}],
  "daily_hours": 3.0
}
```

**评估结果**：
```json
{
  "user_type": "工程师",
  "difficulty_match": 0.9,
  "style_match": 0.95,
  "environment_fit": 1,
  "issues": [
    {
      "issue_type": "difficulty_slightly_high",
      "severity": "minor",
      "location": "核心概念段落",
      "description": "用户level为beginner，内容概念密度6个/千字略高于初级建议（≤5个/千字）",
      "suggestion": "工程师实践能力强，可接受略高密度，建议对'低秩近似'概念增加1句话说明"
    }
  ],
  "summary": "工程师型用户，偏好案例驱动。内容理论30%+实践70%高度匹配practice风格，难度略高但可接受，环境适配良好。"
}
```

**评分解析**：
- difficulty_match=0.9：概念密度6个/千字，beginner建议≤5但工程师实践能力强，容忍度上调
- style_match=0.95：理论30%+实践70%接近工程师偏好（理论20%+实践80%）
- environment_fit=1：工程师有GPU环境

#### 学生型用户评估（同一内容）
**用户画像**：
```json
{
  "professional_background": "大三计算机专业学生",
  "skill_tags": ["Python"],
  "learning_style": "mixed",
  "knowledge_gaps": [{"skill": "LoRA", "level": "beginner"}],
  "daily_hours": 2.0
}
```

**评估结果**：
```json
{
  "user_type": "在校学生",
  "difficulty_match": 0.6,
  "style_match": 0.7,
  "environment_fit": 0,
  "issues": [
    {
      "issue_type": "difficulty_mismatch",
      "severity": "major",
      "location": "核心概念段落",
      "description": "用户level为beginner且为学生型，内容概念密度6个/千字超过学生建议（≤5个/千字），且缺少PyTorch基础回顾",
      "suggestion": "在开头增加PyTorch基础回顾，并对每个新概念增加通俗解释，建议拆分为两个阶段"
    },
    {
      "issue_type": "style_mismatch",
      "severity": "minor",
      "location": "内容整体",
      "description": "用户偏好mixed（理论40%+实践60%），内容实践比例70%略高",
      "suggestion": "增加原理推导段落，平衡理论与实践比例"
    },
    {
      "issue_type": "environment_mismatch",
      "severity": "major",
      "location": "代码示例",
      "description": "内容需GPU（device_map='auto'），但学生通常无GPU环境",
      "suggestion": "提供CPU替代方案：device=torch.device('cpu')，并提示性能下降预期；或提供Google Colab免费GPU使用说明"
    }
  ],
  "summary": "在校学生型，偏好结构化讲解。内容难度略高（6个/千字超过学生建议≤5），风格基本匹配但环境不匹配（无GPU），建议增加基础回顾和CPU替代方案。"
}
```

**评分解析**：
- difficulty_match=0.6：概念密度6个/千字，学生型beginner建议≤5，偏离区间
- style_match=0.7：实践比例70%略高于mixed偏好的60%
- environment_fit=0：学生无GPU环境，需提供替代方案

#### 对比结论
| 评估维度 | 工程师型 | 学生型 |
|:---|:---|:---|
| difficulty_match | 0.9（实践能力强，容忍略高密度） | 0.6（学生需更低密度+基础回顾） |
| style_match | 0.95（理论30%+实践70%高度匹配） | 0.7（实践比例略高） |
| environment_fit | 1（有GPU） | 0（无GPU，需替代方案） |
| 主要问题 | 无major问题 | 2个major（难度+环境） |

**说明**：同一份内容对不同用户类型的适配度差异显著，体现了规则1"不同人适合什么"的核心思想——Agent⑤必须根据用户类型动态调整判断标准，而非一刀切评估。

## 7. 版本信息
- 版本：v3.1，更新日期：2026-07-24
- 变更记录：能力描述明确3个适用场景（难度匹配度/风格适配度/环境可行性）；核心方法论重构为3步（识别用户类型→映射基准→比对输出匹配度）；规则2难度适配映射新增代码行数维度（≤50/50-150/≥150行）；规则3风格适配规则新增匹配度计算方式；新增规则4环境容忍度规则（Docker→conda、GPU→CPU、多卡→单卡梯度累积替代方案库）；输入输出规范新增user_type/difficulty_match(0-1)/style_match(0-1)/environment_fit(0/1)完整JSON示例；异常处理新增混合型用户加权平均评估场景；附录A补充3个学习者风格理论；附录B新增工程师型vs学生型同一内容对比评估示例。**v3.1变更：新增"审核循环≥3次"异常处置路径**
