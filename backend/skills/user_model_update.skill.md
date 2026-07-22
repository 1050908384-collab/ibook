# SKILL名称：user_model_update

## 1. 能力描述
- 功能：为Agent⑥提供**画像字段更新规则库**，定义"算出数据后怎么写入画像"——掌握度计算（BKT模型）、学习节奏推断、信心指数计算、困惑标记生成、情绪规则。原 `knowledge_tracing.skill.md`（BKT参数）和 `emotion_model.skill.md`（情绪规则）的内容已合并至此，本SKILL同时包含BKT参数规则、情绪规则和画像更新规则。Agent⑥将原始交互数据（答题记录、耗时、情绪得分）转化为画像更新字段（knowledge_gaps/pace/confidence/confusion_tags）。`DeterministicStageDecisionAgent` 根据阶段正确率更新 `UserProfile` 中的 knowledge_gaps状态、confidence_index和version字段
- 适用场景：掌握度精算和画像组装步骤（Orchestrator调用 `complete_stage` 时触发）。Agent⑥在阶段结束时将原始交互数据转化为结构化画像更新，传递给Agent①重新生成画像并驱动下一阶段规划
- 前置依赖：
  - 答题记录（`question_records`，含每题正确性、题型、耗时）
  - 情绪得分（`emotion_score`，-1~+1浮点数）
  - 当前画像（`current_profile`，UserProfile对象）
  - 耗时数据（每题实际耗时 vs 预期平均耗时）

## 2. 核心方法论
- **步骤1：用BKT模型计算各知识点掌握概率（含Wt和We修正）**——应用规则1的BKT四参数和规则2的修正因子，逐题更新掌握概率 P(L)，最终得到 P_corrected = P_BKT × Wt × We
- **步骤2：将掌握概率映射为status**——应用规则3的状态映射，将 P_corrected 映射为"已掌握"/"学习中"/"未掌握"
- **步骤3：推断学习节奏和信心指数**——应用规则4推断学习节奏（过快/适中/过慢），应用规则5计算信心指数（accuracy×0.6 + emotion×0.4）
- **步骤4：生成困惑标记**——应用规则6，对答错≥2次或提问≥3次的知识点写入 confusion_tags
- **步骤5：组装profile_updates**——合并上述结果，输出包含 knowledge_gaps/dynamic_state/pace_adjustment/confusion_tags/next_strategy 的完整画像更新和 update_rationale

## 3. 知识/规则定义

### 规则1：BKT四参数（Bayesian Knowledge Tracing）
BKT模型使用四个参数描述知识点的学习状态：

| 参数 | 含义 | 默认值 | 取值依据 |
|:---|:---|:---|:---|
| P(L₀) | 初始掌握概率（学习前已掌握的概率） | 有相似经验=0.6，无经验=0.2 | 根据 `profile.skill_tags` 判断：若已掌握相似技能（如学过PyTorch则LoRA的P(L₀)=0.6），否则0.2 |
| P(T) | 学习跃迁概率（每次练习后从"未掌握"变为"掌握"的概率） | 0.4 | 默认值，基于Corbett & Anderson (1995)的典型参数设置 |
| P(G) | 猜对概率（未掌握但答对的概率） | 选择题=0.25，问答题=0.1，实操题=0.05 | 选择题4选1猜对率1/4=0.25；问答题需理解才能作答；实操题需实际操作 |
| P(S) | 粗心概率（已掌握但答错的概率） | 0.1 | 默认值，基于Corbett & Anderson (1995)的典型参数设置 |

**BKT更新公式**：
- 答对后：`P(L|correct) = P(L)×(1-P(S)) / [P(L)×(1-P(S)) + (1-P(L))×P(G)]`
- 答错后：`P(L|incorrect) = P(L)×P(S) / [P(L)×P(S) + (1-P(L))×(1-P(G))]`
- 学习跃迁：`P(L)_new = P(L|response) + (1-P(L|response))×P(T)`

### 规则2：修正因子（Wt时长权重 + We情绪权重）
BKT基础概率 P_BKT 需通过时长和情绪修正因子调整，得到最终掌握概率：

**Wt（时长权重）**：
| 用户平均耗时 / 预期平均耗时 | Wt | 说明 |
|:---|:---|:---|
| 0.8~1.2倍 | 1.0 | 正常节奏，不修正 |
| <0.5倍 或 >2.0倍 | 0.7 | 过快（可能猜对）或过慢（可能查阅资料），降低置信 |
| 其余（0.5~0.8倍 或 1.2~2.0倍） | 0.85 | 略偏离正常，轻度修正 |

**We（情绪权重）**：
| 情绪得分 | We | 说明 |
|:---|:---|:---|
| > 0.5 | 1.1 | 正面情绪，学习投入度高，提升置信 |
| < -0.3 | 0.9 | 负面情绪（挫败/焦虑），降低置信 |
| -0.3 ~ 0.5 | 1.0 | 中性情绪，不修正 |

**修正公式**：
```
P_corrected = min(1.0, max(0.0, P_BKT × Wt × We))
```
注：通过min/max限制在[0,1]范围内，防止超出。

### 规则3：状态映射
将修正后的掌握概率 P_corrected 映射为知识点状态：

| P_corrected 范围 | status | 处置 |
|:---|:---|:---|
| ≥ 0.8 | 已掌握 | 从 knowledge_gaps 中移除（或标记为已掌握），progress=1.0 |
| 0.4 ~ 0.8（含0.4，不含0.8） | 学习中 | 保留在 knowledge_gaps 中，progress=P_corrected |
| < 0.4 | 未掌握 | 保留在 knowledge_gaps 中，progress=P_corrected |

**与代码对齐**：当前 `DeterministicStageDecisionAgent` 代码使用 accuracy≥0.8 判定"已掌握"。本规则使用BKT修正后的 P_corrected≥0.8 判定，更精确地反映真实掌握度（考虑了猜对率和粗心率的修正）。

### 规则4：学习节奏推断
根据阶段总时长与预期时长的比值推断学习节奏：

| 实际总时长 / 预期总时长 | learning_rhythm | 说明 |
|:---|:---|:---|
| < 0.8 | 过快（fast） | 可能未充分理解，建议放慢节奏 |
| 0.8 ~ 1.2 | 适中（medium） | 正常学习节奏 |
| > 1.2 | 过慢（slow） | 可能遇到困难，建议补充辅助材料 |

### 规则5：信心指数计算
```
confidence_index = accuracy × 0.6 + emotion_score × 0.4
```
- accuracy：本阶段正确率（0~1）
- emotion_score：情绪均分（-1~+1，需归一化到0~1：`(emotion_score + 1) / 2`）

**完整公式**：
```
confidence_index = accuracy × 0.6 + ((emotion_score + 1) / 2) × 0.4
```
结果限制在[0, 1]范围内。

**与代码对齐**：当前代码使用 `confidence_index = min(1.0, max(0.0, accuracy))`（纯正确率）。本规则升级为正确率×0.6 + 情绪×0.4的加权计算，更全面反映学习者信心。

### 规则6：困惑标记生成
对每个知识点检查以下条件，满足任一即写入 `confusion_tags`：
- 答错次数 ≥ 2 次（同一知识点的题目答错2次以上）
- 提问次数 ≥ 3 次（在学习过程中针对该知识点提问3次以上）

**困惑标记格式**：`confusion_tags` 为字符串列表，每项为知识点名称，如 `["LoRA秩选择", "矩阵分解"]`

### 规则7：情绪规则（原 emotion_model.skill.md 合并）
**情绪标签分类**——基于关键词和行为信号：

| 情绪标签 | 关键词触发 | 行为信号触发 | 情绪得分 |
|:---|:---|:---|:---|
| 开心（happy） | "原来如此"、"明白了"、"懂了"、"学会了" | 主动完成自测题 | +0.5 ~ +1.0 |
| 满足（satisfied） | "还行"、"差不多"、"可以做" | 正常完成答题 | +0.1 ~ +0.5 |
| 中性（neutral） | 无明显情绪关键词 | 正常浏览无特殊行为 | -0.3 ~ +0.1 |
| 挫败（frustrated） | "好难"、"看不懂"、"不会"、"放弃了" | 求助点击 ≥ 3 次 | -0.5 ~ -0.3 |
| 焦虑（anxious） | "来不及"、"做不完"、"怎么办" | 反复刷新页面 ≥ 5 次 | -1.0 ~ -0.5 |

**情绪得分计算**：
- 逐题分析：对每道题的答题行为和上下文进行关键词匹配和行为信号检测，得出每题情绪得分（-1~+1）
- 情绪均分 = 所有题目情绪得分的算术平均
- 情绪得分范围：-1（极度焦虑）~ +1（极度开心）

**情绪得分→We映射**：见规则2的We情绪权重表。

## 4. 输入输出规范

### 输入
- `question_records`：答题记录列表，每项含 question_id、skill（关联知识点）、is_correct（布尔）、question_type（"选择题"/"问答题"/"实操题"）、time_spent（耗时秒数）
- `emotion_score`：情绪均分（-1~+1浮点数）
- `current_profile`：当前用户画像（UserProfile对象）

### 输出
```json
{
  "profile_updates": {
    "knowledge_gaps": [
      {
        "skill": "LoRA",
        "level": "beginner",
        "status": "学习中",
        "progress": 0.797,
        "prerequisites": ["Transformer"]
      }
    ],
    "dynamic_state": {
      "mastery_probability": 0.797,
      "confidence_index": 0.60
    },
    "pace_adjustment": "过慢",
    "confusion_tags": ["LoRA秩选择"],
    "next_strategy": "巩固"
  },
  "update_rationale": "BKT计算P_BKT=0.852，经Wt=0.85（耗时比1.3）和We=1.1（情绪0.6>0.5）修正后P_corrected=0.797，处于[0.4,0.8)区间，状态映射为'学习中'。信心指数=0.6×0.6+0.6×0.4=0.60。节奏过慢（耗时比1.3>1.2）。LoRA秩选择答错2次，写入confusion_tags。"
}
```

**字段说明**：
- `profile_updates.knowledge_gaps`：更新后的知识盲区列表，status和progress基于BKT修正后的P_corrected
- `profile_updates.dynamic_state`：动态状态，含 mastery_probability（BKT修正后掌握概率）和 confidence_index（信心指数）
- `profile_updates.pace_adjustment`：学习节奏推断结果（"过快"/"适中"/"过慢"）
- `profile_updates.confusion_tags`：困惑标记列表
- `profile_updates.next_strategy`：建议的下一阶段策略（基于P_corrected和accuracy的综合判断）
- `update_rationale`：人类可读的更新理由，包含BKT计算过程、修正因子取值、状态映射依据

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 答题记录为空（question_records=[]） | 返回空更新（profile_updates为空对象），update_rationale标注"无答题记录，画像未更新" |
| 样本量不足（<3题） | Wt固定1.0（不修正时长），update_rationale标注"样本量不足(<3题)，Wt采用fallback=1.0"。BKT计算仍正常执行，但置信度降低 |
| 情绪数据缺失（emotion_score为None） | We固定1.0（不修正情绪），confidence_index仅基于accuracy计算（accuracy×0.6+0.5×0.4=accuracy×0.6+0.2），update_rationale标注"情绪数据缺失，We采用fallback=1.0" |
| 题型未知（question_type不在"选择题"/"问答题"/"实操题"范围内） | P(G)默认取0.15（选择题和问答题的中间值），update_rationale标注"题型未知，P(G)采用默认0.15" |
| 知识点在画像中不存在（question_records中的skill不在knowledge_gaps中） | 新增KnowledgeGap到knowledge_gaps列表，status初始为"未掌握"，progress=P_corrected |
| P_BKT计算异常（如除零错误） | 跳过BKT计算，降级为accuracy直接映射（accuracy≥0.8→已掌握，<0.8→部分掌握），update_rationale标注"BKT计算异常，降级为accuracy直接映射" |
| 画像数据缺失 | 使用UserProfile默认值（confidence_index=0.5，version=1），标注"数据不足" |

## 6. 附录

### 附录A：BKT与情绪模型参考
| 编号 | 论文/理论 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | Knowledge Tracing: Modeling the Acquisition of Human Knowledge | Albert Corbett & John Anderson | 1995 | 提出贝叶斯知识追踪（BKT）模型，用四个参数（P(L₀)、P(T)、P(G)、P(S)）描述知识点的学习状态转移。每次答题后通过贝叶斯更新掌握概率，引入学习跃迁项模拟"练习促进掌握"的过程。论文中典型参数设置为P(T)=0.3~0.5、P(S)=0.1、P(G)=0.2~0.3 | 规则1 BKT四参数的直接来源——P(L₀)初始概率、P(T)=0.4学习跃迁、P(G)按题型区分（选择题0.25/问答0.1/实操0.05）、P(S)=0.1粗心概率，均参考Corbett & Anderson的典型设置 |
| A-2 | Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing | Zachary Pardos & Neil Heffernan | 2010 | 提出改进型BKT，引入个体化参数和情境修正因子。通过学习者的历史表现动态调整P(L₀)和P(T)，并引入外部信号（如耗时、情绪）作为修正权重，弥补标准BKT未考虑行为情境的不足 | 规则2修正因子的依据——Wt（时长权重）和We（情绪权重）作为外部情境信号修正BKT基础概率，P_corrected = P_BKT × Wt × We。这种引入情境修正的思路来自Pardos & Heffernan的改进型BKT |
| A-3 | The Behavior of Tutoring Systems | Kurt VanLehn | 2006 | 智能辅导系统的"外循环-内循环"架构，外循环基于学习者模型选择任务，内循环在任务内提供步骤级反馈。学习者模型应包含知识状态、情绪状态和行为特征三维度 | 规则5信心指数（accuracy×0.6+emotion×0.4）和规则6困惑标记的设计依据——画像更新需综合知识维度（BKT掌握概率）、情绪维度（情绪得分）和行为维度（耗时、求助行为） |

### 附录B：完整场景示例

**场景**：用户学习LoRA阶段，答5道选择题，答对3道（序列：对、对、错、对、错），情绪均分0.6，各题耗时记录完整。

**输入**：
```json
{
  "question_records": [
    {"question_id": "Q1", "skill": "LoRA", "is_correct": true, "question_type": "选择题", "time_spent": 100},
    {"question_id": "Q2", "skill": "LoRA", "is_correct": true, "question_type": "选择题", "time_spent": 110},
    {"question_id": "Q3", "skill": "LoRA", "is_correct": false, "question_type": "选择题", "time_spent": 200},
    {"question_id": "Q4", "skill": "LoRA", "is_correct": true, "question_type": "选择题", "time_spent": 90},
    {"question_id": "Q5", "skill": "LoRA", "is_correct": false, "question_type": "选择题", "time_spent": 150}
  ],
  "emotion_score": 0.6,
  "current_profile": {
    "session_id": "sess-001",
    "skill_tags": ["Python"],
    "knowledge_gaps": [
      {"skill": "LoRA", "level": "beginner", "status": "未开始", "progress": 0.0, "prerequisites": ["Transformer"]}
    ],
    "confidence_index": 0.5,
    "version": 1
  }
}
```

**处理过程**：

**步骤1：BKT计算掌握概率**

BKT参数确定：
- P(L₀) = 0.2（用户skill_tags=["Python"]，无PyTorch/Transformer经验，与LoRA无相似经验）
- P(T) = 0.4（默认学习跃迁概率）
- P(G) = 0.25（选择题猜对概率）
- P(S) = 0.1（默认粗心概率）

逐题BKT更新（序列：对、对、错、对、错）：

**Q1（答对）**：P(L)=0.2
```
P(L|correct) = 0.2×(1-0.1) / [0.2×0.9 + 0.8×0.25]
             = 0.18 / [0.18 + 0.20] = 0.18/0.38 = 0.4737
P(L)_new = 0.4737 + (1-0.4737)×0.4 = 0.4737 + 0.2105 = 0.6842
```

**Q2（答对）**：P(L)=0.6842
```
P(L|correct) = 0.6842×0.9 / [0.6842×0.9 + 0.3158×0.25]
             = 0.6158 / [0.6158 + 0.0790] = 0.6158/0.6948 = 0.8863
P(L)_new = 0.8863 + 0.1137×0.4 = 0.8863 + 0.0455 = 0.9318
```

**Q3（答错）**：P(L)=0.9318
```
P(L|incorrect) = 0.9318×0.1 / [0.9318×0.1 + 0.0682×0.75]
               = 0.0932 / [0.0932 + 0.0512] = 0.0932/0.1444 = 0.6454
P(L)_new = 0.6454 + 0.3546×0.4 = 0.6454 + 0.1418 = 0.7872
```

**Q4（答对）**：P(L)=0.7872
```
P(L|correct) = 0.7872×0.9 / [0.7872×0.9 + 0.2128×0.25]
             = 0.7085 / [0.7085 + 0.0532] = 0.7085/0.7617 = 0.9301
P(L)_new = 0.9301 + 0.0699×0.4 = 0.9301 + 0.0280 = 0.9581
```

**Q5（答错）**：P(L)=0.9581
```
P(L|incorrect) = 0.9581×0.1 / [0.9581×0.1 + 0.0419×0.75]
               = 0.0958 / [0.0958 + 0.0314] = 0.0958/0.1272 = 0.7533
P(L)_new = 0.7533 + 0.2467×0.4 = 0.7533 + 0.0987 = 0.8520
```

**P_BKT = 0.8520**

修正因子计算：
- 平均耗时 = (100+110+200+90+150)/5 = 650/5 = 130s
- 预期平均耗时 = 100s（选择题典型耗时）
- 耗时比 = 130/100 = 1.3 → 在1.2~2.0区间 → **Wt = 0.85**
- 情绪得分 = 0.6 > 0.5 → **We = 1.1**

**P_corrected = min(1.0, max(0.0, 0.8520 × 0.85 × 1.1)) = 0.7966**

**步骤2：状态映射**
- P_corrected = 0.7966，在[0.4, 0.8)区间 → status = "学习中"
- progress = 0.797

> **关键对比**：未经修正的 P_BKT=0.852 ≥ 0.8 会判为"已掌握"，但经Wt和We修正后降至0.797，判为"学习中"。这体现了修正因子的价值——用户耗时偏长（1.3倍）暗示对内容不够熟练，情绪虽积极但不足以弥补熟练度不足，因此不应判为已掌握。

**步骤3：推断学习节奏和信心指数**
- 节奏：耗时比1.3 > 1.2 → learning_rhythm = "过慢"
- 信心指数：accuracy = 3/5 = 0.6
  ```
  confidence_index = 0.6 × 0.6 + ((0.6 + 1) / 2) × 0.4
                   = 0.36 + 0.8 × 0.4
                   = 0.36 + 0.32
                   = 0.68
  ```

**步骤4：生成困惑标记**
- Q3和Q5答错，均关联"LoRA"知识点 → 答错2次 ≥ 2次 → confusion_tags = ["LoRA"]
- 进一步分析Q3和Q5的题目内容均为"LoRA秩选择" → confusion_tags = ["LoRA秩选择"]

**步骤5：组装profile_updates**
- knowledge_gaps：LoRA的status更新为"学习中"，progress=0.797
- next_strategy：accuracy=0.6在[0.6, 0.8)区间 → "巩固"（reinforce）
- version = 1 + 1 = 2

**输出**：
```json
{
  "profile_updates": {
    "knowledge_gaps": [
      {
        "skill": "LoRA",
        "level": "beginner",
        "status": "学习中",
        "progress": 0.797,
        "prerequisites": ["Transformer"]
      }
    ],
    "dynamic_state": {
      "mastery_probability": 0.797,
      "confidence_index": 0.68
    },
    "pace_adjustment": "过慢",
    "confusion_tags": ["LoRA秩选择"],
    "next_strategy": "巩固"
  },
  "update_rationale": "BKT逐题更新：P(L)从0.2经5题（对对错对错）更新至P_BKT=0.852。修正因子：Wt=0.85（耗时比1.3，位于1.2~2.0区间）、We=1.1（情绪0.6>0.5）。P_corrected=0.852×0.85×1.1=0.797，处于[0.4,0.8)区间，状态映射为'学习中'。信心指数=0.6×0.6+0.8×0.4=0.68。节奏过慢（耗时比1.3>1.2）。Q3和Q5答错LoRA秩选择相关题目，答错2次≥2，写入confusion_tags。建议下一策略：巩固。"
}
```

**关键数据汇总**：
| 指标 | 值 | 计算依据 |
|:---|:---|:---|
| P(L₀) 初始概率 | 0.2 | 无相似经验（skill_tags仅Python） |
| P_BKT 最终掌握概率 | 0.852 | 5题BKT逐题更新 |
| Wt 时长权重 | 0.85 | 耗时比1.3（130s/100s） |
| We 情绪权重 | 1.1 | 情绪0.6 > 0.5 |
| P_corrected 修正后概率 | 0.797 | 0.852×0.85×1.1 |
| status 状态 | 学习中 | 0.797 ∈ [0.4, 0.8) |
| confidence_index 信心指数 | 0.68 | 0.6×0.6+0.8×0.4 |
| pace_adjustment 节奏 | 过慢 | 耗时比1.3 > 1.2 |
| confusion_tags 困惑标记 | ["LoRA秩选择"] | Q3+Q5答错2次 |
| next_strategy 下一步策略 | 巩固 | accuracy=0.6 ∈ [0.6, 0.8) |

**说明**：本场景完整展示了BKT四参数模型从初始概率0.2经5题迭代更新至0.852的全过程，以及修正因子（Wt时长权重+We情绪权重）将P_BKT从0.852修正至0.797的关键作用——修正前会判为"已掌握"，修正后判为"学习中"，体现了BKT+修正因子相比纯accuracy判定（accuracy=0.6→部分掌握）的精度提升：BKT考虑了答对/答错的序列模式（前2题答对拉高概率，后2题答错拉低概率），而纯accuracy仅看3/5=0.6的比例。

## 7. 版本信息
- 版本：v3.0，更新日期：2026-07-23
- 变更记录：合并原 `knowledge_tracing.skill.md`（BKT参数）和 `emotion_model.skill.md`（情绪规则）全部内容；能力描述明确"掌握度精算和画像组装"适用场景；核心方法论重构为5步（BKT计算→状态映射→节奏信心推断→困惑标记→组装画像）；新增规则1 BKT四参数表（P(L₀)/P(T)/P(G)/P(S)含取值依据和BKT更新公式）；新增规则2修正因子（Wt时长权重三档+We情绪权重三档+P_corrected公式）；规则3状态映射升级为三档（≥0.8已掌握/0.4~0.8学习中/<0.4未掌握，基于P_corrected而非accuracy）；新增规则4学习节奏推断（耗时比三档）；规则5信心指数升级为加权公式（accuracy×0.6+emotion×0.4）；新增规则6困惑标记生成（答错≥2次或提问≥3次）；新增规则7情绪规则（5类情绪标签关键词+行为信号+得分-1~+1映射）；输入输出规范新增profile_updates和update_rationale完整JSON示例；异常处理新增样本量不足Wt=1.0 fallback和情绪缺失We=1.0 fallback；附录A补充3个参考论文（Corbett&Anderson 1995 BKT原始论文、Pardos&Heffernan 2010改进型BKT、VanLehn 2006辅导系统架构）；附录B新增完整BKT计算场景（5题逐题更新全公式+修正因子+状态映射+输出汇总表）
- **与代码对齐说明**：当前 `DeterministicStageDecisionAgent` 代码使用纯accuracy判定（accuracy≥0.8→已掌握，confidence_index=accuracy）。本SKILL v3.0升级为BKT四参数模型+修正因子，更精确地反映真实掌握度。后续代码可引入BKT计算模块，将 `confidence_index` 升级为加权公式，并增加 `mastery_probability`/`pace_adjustment`/`confusion_tags` 字段
