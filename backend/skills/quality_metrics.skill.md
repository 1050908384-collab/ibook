# SKILL名称：quality_metrics

## 0. Agent⑤协作关系图与过关判定流程

### 0.1 三个Skill的分工与协作关系
Agent⑤（质量评审）由三个Skill协同工作，**本SKILL（quality_metrics）是核心大脑**，负责定义评分框架和最终裁决；另外两个Skill提供评估数据：

| Skill名称 | 角色 | 职责方向 | 提供的数据维度 |
|:---|:---|:---|:---|
| **quality_metrics**（本文件） | **评分框架（大脑）** | 定义评分体系、权重、及格线、红线规则，**本身不做具体评估** | 不提供原始数据，仅定义评分规则 |
| **pedagogy** | 教学法专家（眼睛） | 评估内容的教学逻辑是否科学，认知负荷是否合理，情绪是否友好 | 提供 **3个维度**：`cognitive_load`（认知负荷合理度）、`pedagogy`（教学逻辑严谨度）、`emotion_friendliness`（情绪友好度） |
| **learner_model** | 画像专家（尺子） | 评估内容与学习者画像的匹配程度 | 提供 **3个维度**：`difficulty`（难度匹配度）、`style_match`（风格适配度）、`environment_fit`（环境可行性） |

> 六个维度全部由 pedagogy（3个）和 learner_model（3个）提供，无需额外工具。quality_metrics 仅负责接收成品评分、加权汇总和红线判定。

### 0.2 过关判定流程（三级判定）
**不是"6个维度全部过关"，而是以下三级判定流程**：

```
输入：pedagogy提供的2维度数据 + learner_model提供的3维度数据 + 情绪工具提供的1维度数据
      ↓
步骤1：数据标准化（规则4映射函数）
      ↓ 映射为0-100分
步骤2：红线检查（规则3）
      ↓ 任一触发？
      ├── 是 → 直接判定 status="fail"，passed=false（无需后续步骤）
      └── 否 → 进入步骤3
      ↓
步骤3：加权总分计算（规则1权重 × 各项得分）
      ↓
步骤4：及格线检查（规则2）
      ↓
步骤5：附加条件检查（是否有major/critical级错误）
      ↓
最终输出：status="pass"/"fail"，passed=true/false
```

**三级判定标准**：
1. **第一级：红线否决（一票否决）**
   - 触发条件：`cognitive_load_level > 85`（认知负荷严重超载）或 `environment_fit_score = 0`（环境完全不可行）
   - 判定：**直接不通过**，无需计算总分

2. **第二级：加权总分及格**
   - 计算方式：`total_score = Σ(维度得分 × 权重)`
   - 判定标准：**加权总分 ≥ 70分** → 通过

3. **第三级：附加条件（错误级别检查）**
   - 即使总分 ≥ 70，若存在 `severity` 为 `"major"` 或 `"critical"` 的ReviewError
   - 判定：**仍需修改**，passed=false

**示例**：
- 场景A：6维度得分均在80分以上，无major/critical错误 → 加权总分>70 → **通过**
- 场景B：加权总分85分，但存在1个major错误（如认知负荷超标） → **不通过**
- 场景C：加权总分90分，cognitive_load_level=87（触发红线） → **直接不通过**（红线否决优先）

---

## 1. 能力描述
- 功能：为Agent⑤提供**综合评分标准与权重**，定义"怎么打分和算总分"——六维评分体系、权重分配、及格线、单项红线。Agent⑤调用本SKILL将各工具输出的原始数据（难度匹配度、认知负荷、环境可行性、风格适配度、情绪友好度、教学逻辑严谨度）转换为统一的评分报告，包含加权总分和通过/不通过结论。在 `HybridQualityReviewAgent` 中被加载为 `skill_context`，作为LLM计算综合评分的标准
- 适用场景：所有工具数据采集完成后进入评分阶段。`DeterministicQualityReviewAgent` 检查内容长度红线，`HybridQualityReviewAgent` 使用LLM进行六维评分并按权重计算总分
- 前置依赖：
  - 两个子SKILL的返回结果（共6个维度，无额外工具依赖）：
    - 认知负荷合理度 + 认知负荷水平 + 教学逻辑严谨度 + 情绪友好度（来自 `pedagogy.skill.md` 规则3/1-2-4/5）
    - 难度匹配度 + 环境可行性 + 风格适配度（来自 `learner_model.skill.md` 规则2/4/3）
  - 用户画像（`ReviewInput.profile`）
  - 学习内容（`ReviewInput.content`）

## 2. 核心方法论
- **步骤1：接收子SKILL成品评分并标准化**——接收 pedagogy 提供的 cognitive_load_score / cognitive_load_level / pedagogy_score / emotion_friendliness_score（0-100成品分，直接使用）和 learner_model 提供的 difficulty_match(0-1) / style_match(0-1) / environment_fit(0/1)（需×100标准化为0-100分），汇总为 dimension_scores
- **步骤2：按权重计算加权总分**——应用规则1的六维权重，计算 `total_score = Σ(dimension_score × weight)`
- **步骤3：检查红线条件**——应用规则3的单项红线：若 cognitive_load_level > 85（来自pedagogy）或 environment_fit = 0（来自learner_model），直接判定不通过，无需再看总分
- **步骤4：输出通过/不通过结论和建议**——应用规则2及格线（total_score ≥ 70 通过），生成包含 total_score、dimension_scores、status、red_line_triggered、suggestions 的完整评分报告

## 3. 知识/规则定义

### 规则1：六维评分权重
| 维度 | 字段名 | 权重 | 说明 | 数据来源 |
|:---|:---|:---|:---|:---|
| 难度匹配度 | `difficulty` | 20% | 内容难度与用户水平的匹配程度 | `learner_model.skill.md` 规则2 |
| 认知负荷合理度 | `cognitive_load` | 25% | 内容复杂度是否在合理范围 | `pedagogy.skill.md` 规则3 |
| 环境可行性 | `environment_fit` | 10% | 部署示例是否可在用户环境运行 | `learner_model.skill.md` 规则4 |
| 风格适配度 | `style_match` | 10% | 内容风格与用户偏好匹配度 | `learner_model.skill.md` 规则3 |
| 情绪友好度 | `emotion_friendliness` | 15% | 内容的鼓励性、亲和力、挫败感控制 | `pedagogy.skill.md` 规则5 |
| 教学逻辑严谨度 | `pedagogy` | 20% | 是否遵循脚手架、渐进式、建构主义原则 | `pedagogy.skill.md` 规则1/2/4 |

**权重设计依据**：认知负荷(25%)和教学逻辑(20%)权重最高，因为这两项直接决定学习者能否跟上内容；难度匹配(20%)次之，确保内容在i+1区间；情绪友好(15%)影响学习持续性；环境和风格各10%为辅助维度。

**总分计算公式**：
```
total_score = difficulty×0.20 + cognitive_load×0.25 + environment_fit×0.10 
            + style_match×0.10 + emotion_friendliness×0.15 + pedagogy×0.20
```

### 规则2：及格线
- **加权总分 ≥ 70** → status=`"pass"`，通过
- **加权总分 < 70** → status=`"fail"`，触发修改建议
- **附加条件**：即使总分≥70，若存在severity为"major"或"critical"的ReviewError，status仍为`"fail"`（综合判定规则）

### 规则3：单项红线（Red Line）
**关键概念区分**：本规则检查的是**认知负荷水平**（`cognitive_load_level`，0-100，**越高=负荷越重=越糟**），而非维度评分（`dimension_scores["cognitive_load"]` 为合理度，越高=越合理=越好）。两者关系：`dimension_score = 100 - cognitive_load_level`。

**红线条件（满足任一即直接不通过，无需计算总分）**：
1. **认知负荷红线**：`cognitive_load_level > 85`（即概念密度严重超载，对应 `dimension_scores["cognitive_load"] < 15`）→ 直接 status=`"fail"`，red_line_triggered=`"cognitive_load_overload"`
2. **环境不可行红线**：`environment_fit_score = 0`（即environment_fit=0） → 直接 status=`"fail"`，red_line_triggered=`"environment_infeasible"`

**红线优先级**：当多个红线同时触发时，以认知负荷红线为最高优先级（red_line_triggered取 `"cognitive_load_overload"`），因为认知过载是最难通过修改内容解决的问题，需强制拆分阶段。

**红线与及格线的关系**：
- 红线检查在步骤3，先于及格线检查（步骤4）
- 红线触发时，red_line_triggered非空，status强制为fail，suggestions优先给出红线修复建议
- 红线未触发时，才进入及格线判断

### 规则4：评分映射函数
各工具原始输出到0-100分的映射：

#### 4.1 难度匹配度映射
```python
def map_difficulty(difficulty_match: float) -> int:
    """difficulty_match: 0-1浮点数（来自learner_model规则2）"""
    return round(difficulty_match * 100)
```
- difficulty_match=0.9 → score=90
- difficulty_match=0.6 → score=60

#### 4.2 认知负荷映射
认知负荷需计算两个值：**维度评分**（合理度，越高越好，用于加权总分）和**负荷水平**（越高越糟，用于红线检查）。

```python
def map_cognitive_load_score(concept_density: float) -> int:
    """返回维度评分（合理度，0-100，越高越好，用于加权总分）
    concept_density: 新概念个数/千字（来自pedagogy规则3）"""
    if concept_density <= 5:
        return 100 - round((5 - concept_density) * 2)  # 3~5个: 96~100
    elif concept_density <= 7:
        return 90 - round((concept_density - 5) * 10)  # 6~7个: 80~70
    elif concept_density <= 10:
        return 69 - round((concept_density - 7) * 6)   # 8~10个: 51~69
    else:
        return max(0, 49 - round((concept_density - 10) * 5))  # >10个: <49

def map_cognitive_load_level(concept_density: float) -> int:
    """返回负荷水平（0-100，越高越糟，用于红线检查）
    与维度评分互逆：level = 100 - score"""
    return 100 - map_cognitive_load_score(concept_density)
```
- density=4 → score=98（合理度高），level=2（负荷低，不触发红线）
- density=6 → score=80，level=20
- density=9 → score=51，level=49
- density=12 → score=34，level=66
- density=15 → score=14，level=86 > 85（**触发认知负荷红线**）

**红线检查**：`map_cognitive_load_level(concept_density) > 85` 时触发规则3红线。等价于 `map_cognitive_load_score(concept_density) < 15`。

#### 4.3 环境可行性映射
```python
def map_environment(environment_fit: int) -> int:
    """environment_fit: 0或1（来自learner_model规则4）"""
    return environment_fit * 100
```
- environment_fit=1 → score=100
- environment_fit=0 → score=0（触发红线）

#### 4.4 风格适配度映射
```python
def map_style(style_match: float) -> int:
    """style_match: 0-1浮点数（来自learner_model规则3）"""
    return round(style_match * 100)
```
- style_match=0.95 → score=95
- style_match=0.4 → score=40

#### 4.5 情绪友好度映射
```python
def map_emotion(positive_ratio: float, has_discouraging_words: bool) -> int:
    """positive_ratio: 正面情绪语句占比(0-1)
       has_discouraging_words: 是否含'这很简单''你应该懂'等挫败性表达"""
    base = round(positive_ratio * 100)
    if has_discouraging_words:
        base = max(0, base - 20)  # 挫败性表达扣20分
    return base
```
- positive_ratio=0.8, 无挫败词 → score=80
- positive_ratio=0.7, 有挫败词 → score=50

#### 4.6 教学逻辑严谨度映射
```python
def map_pedagogy(section_completeness: int, has_scaffold: bool, has_case: bool) -> int:
    """section_completeness: 五段式完整段数(0-5)
       has_scaffold: 是否遵循脚手架原则
       has_case: 是否含情境化案例"""
    score = section_completeness * 15  # 五段每段15分，满分75
    if has_scaffold:
        score += 15
    if has_case:
        score += 10
    return min(100, score)
```
- 五段全完整+scaffold+case → score=100
- 三段+scaffold+case → score=70
- 两段+无scaffold+无case → score=30

## 4. 输入输出规范

### 输入
**来自 pedagogy 的成品评分**（3个维度）+ **来自 learner_model 的成品评分**（3个维度）+ **合并的 issues 数组**：
```json
{
  "cognitive_load_score": 80,
  "cognitive_load_level": 20,
  "pedagogy_score": 85,
  "emotion_friendliness_score": 75,
  "difficulty_match": 0.8,
  "environment_fit": 1,
  "style_match": 0.9,
  "review_errors": []
}
```

> **数据契约说明**：pedagogy 输出 cognitive_load_score / cognitive_load_level / pedagogy_score / emotion_friendliness_score（均为0-100成品分），learner_model 输出 difficulty_match(0-1) / style_match(0-1) / environment_fit(0/1)。quality_metrics 的映射函数（规则4）仅作为**评分参考说明**，实际计算时直接使用子SKILL提供的成品分（0-1值需×100标准化），不重复从原始值计算。

### 输出
综合评分报告，LLM返回JSON格式：
```json
{
  "total_score": 82,
  "dimension_scores": {
    "difficulty": 80,
    "cognitive_load": 80,
    "environment_fit": 100,
    "style_match": 90,
    "emotion_friendliness": 75,
    "pedagogy": 85
  },
  "status": "pass",
  "red_line_triggered": null,
  "suggestions": [
    "情绪友好度75分偏低，建议增加鼓励性语句，如'别担心，这个概念一开始看起来复杂，我们一步步来'"
  ],
  "passed": true,
  "errors": [],
  "review_details": "六维评分：难度80、认知负荷80（合理度高，负荷水平20，未触发红线）、环境100、风格90、情绪75、教学85。加权总分82分≥70通过，无红线触发。建议优化情绪友好度。"
}
```

**字段说明**：
- `total_score`：0-100，加权总分
- `dimension_scores`：字典，六维各项0-100分
- `status`：`"pass"` / `"fail"` / `"review_needed"`（边缘通过65~69分时）
- `red_line_triggered`：红线标识，`null`（未触发）/ `"cognitive_load_overload"` / `"environment_infeasible"`
- `suggestions`：改进建议列表（按优先级排序，红线问题优先）
- `passed`：布尔值，与status联动（pass→true，fail/review_needed→false）
- `errors`：ReviewError列表，合并各工具的issues
- `review_details`：人类可读的评分总结

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 某维度数据缺失（如pedagogy未返回emotion_friendliness_score） | 该维度权重重新分配到其他5个维度（按原权重比例分配），在review_details中标注 `"emotion_friendliness数据缺失，权重重新分配"`。缺失维度按60分基准计算（保守估计） |
| 多红线触发（认知负荷>85且环境不可行） | 以认知负荷红线为最高优先级，red_line_triggered取 `"cognitive_load_overload"`，suggestions优先给出认知负荷拆分建议，环境建议列为次要 |
| 总分接近及格线（65~69分） | 判定 status=`"review_needed"`，passed=false，suggestions标注"边缘通过，建议优化提升至70分以上" |
| 内容过短（`content.strip()`<40字符） | `DeterministicQualityReviewAgent` 直接返回 passed=false，error_type=`"too_short"`，不执行LLM六维评分，所有dimension_scores=0 |
| LLM审核结果无法解析 | 添加 ReviewError(error_type=`"review_system_error"`, severity=critical, description="Agent4统一质量评分结果无法解析")，passed=false，status=`"fail"` |
| LLM返回的severity值不在 `{"critical","major","minor"}` 范围内 | 自动修正为 `"major"` |
| 六维数据全部缺失 | 降级为仅检查内容长度红线，返回 status=`"fail"`，review_details标注"评分数据完全缺失，请检查工具链" |
| 内容审核循环达上限（Agent4统一审核循环≥3次仍未通过） | Orchestrator强制终止循环，status=`"review_needed"`，summary标注"审核循环达3次上限，建议人工介入修订内容" |

## 6. 附录

### 附录A：评分理论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | Bloom教育目标分类学（修订版） | Lorin Anderson & David Krathwohl（修订Bloom 1956原版） | 2001 | 将认知过程分为6个层次：记忆→理解→应用→分析→评价→创造，学习目标应明确对应认知层次 | 规则1六维评分中"教学逻辑严谨度"维度的依据——评估内容是否设计了从低阶（记忆/理解）到高阶（应用/分析）的认知阶梯，对应规则2及格线70分确保学习者达到"应用"层级 |
| A-2 | 认知负荷理论（Cognitive Load Theory） | John Sweller | 1988 | 工作记忆容量有限（7±2），内在负荷+外在负荷+相关负荷总和不能超限，超限导致认知过载 | 规则3单项红线的直接依据——认知负荷水平>85（对应concept_density严重超载、合理度评分<15）直接判不通过，因为认知过载无法通过局部修改解决，必须拆分阶段 |
| A-3 | Kirkpatrick培训评估四级模型 | Donald Kirkpatrick | 1959（1994完善） | 培训效果评估分四级：反应→学习→行为→结果，其中"反应"级评估学习者对内容的情绪接受度 | 规则1六维评分中"情绪友好度"维度的依据——评估学习者对内容的反应（鼓励性、挫败感控制），对应Kirkpatrick第一级"反应"评估 |

### 附录B：完整场景示例

#### 场景1：总分通过示例
**输入**（六个工具原始输出）：
```json
{
  "difficulty_match": 0.85,
  "concept_density": 4,
  "environment_fit": 1,
  "style_match": 0.9,
  "positive_ratio": 0.8,
  "has_discouraging_words": false,
  "section_completeness": 5,
  "has_scaffold": true,
  "has_case": true,
  "review_errors": []
}
```

**评分计算过程**：
1. 步骤1映射（计算维度评分与负荷水平）：
   - difficulty = 0.85×100 = 85
   - cognitive_load_score = map_score(4) = 100-(5-4)×2 = 98（合理度高）
   - cognitive_load_level = 100 - 98 = 2（负荷水平低，远未触发红线85）
   - environment_fit = 1×100 = 100
   - style_match = 0.9×100 = 90
   - emotion_friendliness = 0.8×100 = 80（无挫败词）
   - pedagogy = 5×15+15+10 = 100
2. 步骤2加权总分：
   - total = 85×0.20 + 98×0.25 + 100×0.10 + 90×0.10 + 80×0.15 + 100×0.20
   - total = 17 + 24.5 + 10 + 9 + 12 + 20 = 92.5 ≈ 93
3. 步骤3红线检查：cognitive_load_level=2，远低于红线阈值85；environment_fit=100≠0 → **无红线触发**
4. 步骤4输出：total_score=93 ≥ 70，无major/critical错误，status=`"pass"`

**输出**：
```json
{
  "total_score": 93,
  "dimension_scores": {
    "difficulty": 85,
    "cognitive_load": 98,
    "environment_fit": 100,
    "style_match": 90,
    "emotion_friendliness": 80,
    "pedagogy": 100
  },
  "status": "pass",
  "red_line_triggered": null,
  "suggestions": [
    "情绪友好度80分良好但可提升，建议增加鼓励性语句"
  ],
  "passed": true,
  "errors": [],
  "review_details": "六维评分均高于80分，加权总分93分远超70分及格线，无红线触发。内容质量优秀，建议通过。"
}
```

#### 场景2：红线阻断示例（认知负荷超载）
**输入**（六个工具原始输出）：
```json
{
  "difficulty_match": 0.4,
  "concept_density": 17,
  "environment_fit": 1,
  "style_match": 0.6,
  "positive_ratio": 0.5,
  "has_discouraging_words": true,
  "section_completeness": 2,
  "has_scaffold": false,
  "has_case": false,
  "review_errors": [
    {"error_type": "cognitive_load", "severity": "major", "description": "新概念密度17个/千字，严重超载"}
  ]
}
```

**评分计算过程**：
1. 步骤1映射（计算维度评分与负荷水平）：
   - difficulty = 0.4×100 = 40
   - cognitive_load_score = map_score(17) = max(0, 49-round((17-10)×5)) = max(0, 49-35) = 14（合理度极低）
   - cognitive_load_level = 100 - 14 = 86（负荷水平86，**超过红线阈值85**）
   - environment_fit = 1×100 = 100
   - style_match = 0.6×100 = 60
   - emotion_friendliness = 0.5×100 - 20 = 30（有挫败词扣20分）
   - pedagogy = 2×15+0+0 = 30
2. 步骤2加权总分（参考值，红线触发后不以此为准）：
   - total = 40×0.20 + 14×0.25 + 100×0.10 + 60×0.10 + 30×0.15 + 30×0.20
   - total = 8 + 3.5 + 10 + 6 + 4.5 + 6 = 38
3. 步骤3红线检查：cognitive_load_level=86 > 85 → **触发认知负荷红线！**
   - red_line_triggered = `"cognitive_load_overload"`
   - 直接判定 status=`"fail"`，**无需再看总分**（红线优先于及格线）
4. 步骤4输出：红线触发，suggestions优先给出认知负荷拆分建议

**输出**：
```json
{
  "total_score": 38,
  "dimension_scores": {
    "difficulty": 40,
    "cognitive_load": 14,
    "environment_fit": 100,
    "style_match": 60,
    "emotion_friendliness": 30,
    "pedagogy": 30
  },
  "status": "fail",
  "red_line_triggered": "cognitive_load_overload",
  "suggestions": [
    "【红线-紧急】认知负荷严重超载（概念密度17个/千字，负荷水平86>85红线，合理度评分仅14分），必须将内容拆分为3~4个独立阶段，每阶段新概念控制在5个以内",
    "教学逻辑严谨度仅30分，五段式仅2段且无脚手架、无情境化案例，建议重构为完整五段结构",
    "情绪友好度30分，含挫败性表达，建议删除'这很简单你应该懂'类语句，增加鼓励性表述",
    "难度匹配度40分，内容难度严重超出用户水平，建议增加基础扫盲段落"
  ],
  "passed": false,
  "errors": [
    {
      "error_type": "cognitive_load",
      "severity": "major",
      "description": "新概念密度17个/千字，负荷水平86超过85红线，触发认知负荷红线"
    }
  ],
  "review_details": "红线触发：认知负荷超载（level=86>85，合理度评分=14<15）。加权总分38分<70。红线优先于及格线判定，直接判fail。优先解决认知负荷问题，建议拆分阶段。"
}
```

**说明**：场景2体现了规则3单项红线的强制阻断作用——认知负荷水平86超过85红线，即使不计算总分（38分也确实<70），红线触发即直接判不通过，且suggestions将红线修复建议置于最高优先级。这确保了认知过载这一最难通过局部修改解决的问题得到优先处理。

**场景1 vs 场景2 对比**：
| 指标 | 场景1（通过） | 场景2（红线阻断） |
|:---|:---|:---|
| concept_density | 4（合理） | 17（严重超载） |
| cognitive_load_score | 98（合理度高） | 14（合理度极低） |
| cognitive_load_level | 2（负荷低） | 86（负荷高，**触发红线**） |
| red_line_triggered | null | cognitive_load_overload |
| total_score | 93 | 38 |
| status | pass | fail |

## 7. 版本信息
- 版本：v3.1，更新日期：2026-07-24
- 变更记录：评分体系从四维扩展为**六维**（新增 environment_fit 10%、style_match 10%、emotion_friendliness 15%，原 clarity 并入 pedagogy 改为"教学逻辑严谨度"20%）；核心方法论重构为4步（映射→加权→红线检查→输出）；规则1新增六维权重表和数据来源说明；规则3新增单项红线（认知负荷水平>85或环境fit=0直接不通过）及多红线优先级，**区分"合理度评分"（越高越好，用于加权总分）与"负荷水平"（越高越糟，用于红线检查，两者互逆）**；规则4新增6个评分映射函数（含Python代码示例和数值示例，认知负荷提供score与level两个函数）；输入输出规范新增total_score/dimension_scores/status/red_line_triggered/suggestions完整JSON示例；异常处理新增数据缺失权重重分配和多红线优先级场景；附录A补充3个评分理论；附录B新增2个完整场景。**v3.1变更：①情绪友好度数据来源从"情绪分析工具"（不存在）改为pedagogy规则5，消除对未定义工具的依赖；②明确数据契约——输入接收子SKILL成品评分（非原始值），映射函数改为参考说明；③前置依赖修正为仅依赖pedagogy+learner_model两个子SKILL；④新增"审核循环≥3次"异常处置路径**
- **与代码对齐说明**：当前 `QualityReviewResult.scores` 代码实现为四维（difficulty/cognitive_load/clarity/pedagogy），本SKILL v3.0按扩展指南升级为六维。后续代码实现可扩展 scores 字典增加 environment_fit/style_match/emotion_friendliness 键，并调整 `HybridQualityReviewAgent` 的评分prompt以匹配六维体系
