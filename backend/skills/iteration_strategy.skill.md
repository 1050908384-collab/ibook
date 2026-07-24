# SKILL名称：iteration_strategy

## 1. 能力描述
- 功能：为Agent⑥提供**策略选择规则库**，定义"什么条件下选什么策略"——五档策略表（结业/进阶/巩固/降维/回溯）、方向切换条件、终止规则。Agent⑥接收 `user_model_update.skill.md` 输出的 `next_strategy`（BKT基于掌握度的策略建议），结合本阶段正确率、覆盖率和迭代历史进行最终策略决策，输出 `next_strategy`（最终策略标签）、决策理由和终止标志。`DeterministicStageDecisionAgent` 根据 accuracy 阈值选择策略标签，并输出决策类型
- 适用场景：阶段结束后进入策略选择步骤（Orchestrator调用 `complete_stage` 时触发）。Agent⑥在阶段结束时根据BKT建议的策略、答题正确率、知识覆盖率和历史迭代轮次决定下一阶段行为
- 前置依赖：
  - `user_model_update.skill.md` 输出的 `next_strategy`（BKT基于掌握度的策略建议）和 `profile_updates`（含 knowledge_gaps、dynamic_state、confusion_tags 等）
  - 本阶段正确率（`accuracy`，0~1）
  - 覆盖率（`coverage`，0~1，已掌握知识点数/总知识点数）
  - 当前画像（`current_profile`，含 `knowledge_gaps` 列表、`confidence_index`、`version`）
  - 迭代轮次历史（`iteration_history`，含历史accuracy序列、已用策略序列、当前迭代轮次）

## 2. 核心方法论
- **步骤1：接收BKT策略建议并计算正确率和覆盖率**——接收 `user_model_update.skill.md` 输出的 `next_strategy`（BKT建议）和 `profile_updates`；正确率=正确题数/总题数（由 `StateManager.calculate_accuracy` 计算）；覆盖率=已掌握知识点数（status="已掌握"的gap数，由 user_model_update 基于P_corrected判定）/总知识点数（knowledge_gaps总数）
- **步骤2：结合BKT建议查策略表匹配策略标签**——以 `next_strategy`（BKT建议）为基础，应用规则1的五档策略表，结合 accuracy 和 coverage 验证并调整策略标签（结业/进阶/巩固/降维/回溯）。BKT建议与策略表结果冲突时，以策略表结果为准（因为策略表考虑了覆盖率和终止条件）
- **步骤3：检查是否触发方向切换或终止条件**——应用规则2检查降维是否连续2轮无效需切换回溯；应用规则3检查是否满足终止条件（结业/最大迭代/主动完成/死循环检测）
- **步骤4：输出最终 next_strategy 和决策理由**——生成 `next_strategy`（最终策略标签）、`reason`（决策理由，说明BKT建议、策略表匹配、方向切换/终止检查的综合过程）、`should_terminate`（是否终止）、`decision`（continue/repeat/completed）

## 3. 知识/规则定义

### 规则1：五档策略表
| 条件 | 策略标签 | decision | 说明 |
|:---|:---|:---|:---|
| accuracy ≥ 0.85 **且** coverage ≥ 0.90 **且** 所有gaps已掌握 | 结业（finish） | completed | 终止迭代，进入结业报告 |
| accuracy ≥ 0.80（非全部已掌握） | 进阶（advance） | continue | 推送下一阶段内容 |
| 0.60 ≤ accuracy < 0.80 | 巩固（reinforce） | repeat | 重新学习当前阶段，补充讲解 |
| accuracy < 0.60 | 降维（simplify） | repeat | 降低内容难度，简化概念 |
| 降维连续2轮无效时触发 | 回溯（backtrack） | repeat | 回退到前置知识阶段重新学习 |

**策略优先级**：结业判定 > 方向切换（回溯） > 四档策略匹配 > BKT建议参考。先检查所有盲区是否已掌握，再检查是否需回溯，再按accuracy匹配四档策略，最后参考BKT建议验证合理性（如BKT建议与策略表结果冲突，以策略表结果为准，因为策略表考虑了覆盖率和终止条件）。

**掌握状态判定（由 user_model_update 提供）**：本SKILL的 coverage 计算依赖的 gap.status（"已掌握"/"学习中"/"未掌握"）由 `user_model_update.skill.md` 基于 BKT 修正后的 P_corrected 判定（P_corrected≥0.8→已掌握）。本SKILL不重复计算掌握度，仅读取已更新的画像进行策略匹配。

### 规则2：方向切换（降维→回溯）
**触发条件**：当策略为"降维"且连续2轮降维后正确率提升 < 5%（即 `accuracy[当前] - accuracy[上轮] < 0.05` 连续2次），自动切换为"回溯"策略。

**回溯逻辑**：
- 回溯到当前知识点的前置知识阶段（通过 `KnowledgeGap.prerequisites` 字段定位）
- 若无前置知识（prerequisites为空），则回溯到最基础的入门阶段并标注"基础不足，建议从头学习"
- 回溯后重置当前知识点的迭代计数，从前置阶段重新开始

**切换示例**：
```
第1轮：accuracy=0.50 → 降维（simplify）
第2轮：accuracy=0.52 → 降维（提升0.02 < 0.05，累计1次无效）
第3轮：accuracy=0.54 → 降维（提升0.02 < 0.05，累计2次无效）→ 切换为回溯（backtrack）
```

### 规则3：终止条件
| 终止条件 | 触发判定 | should_terminate | 说明 |
|:---|:---|:---|:---|
| 结业判定 | 所有knowledge_gaps状态均为"已掌握" | true | 正常完成所有学习目标 |
| 最大迭代次数 | 当前迭代轮次 ≥ max_iterations（默认5） | true | 防止无限循环，强制终止并生成结业报告（含未完成项） |
| 学习者主动完成 | 用户主动点击"完成学习" | true | 尊重学习者意愿，终止迭代 |
| 死循环检测 | 连续2轮改进 < 3%（`accuracy[当前] - accuracy[上轮] < 0.03` 连续2次）且已触发回溯 | true | 停滞终止，建议人工介入或调整学习目标 |

**防死循环阈值**：
- `max_iterations = 5`：单个知识点最多迭代5轮，超过即终止
- 停滞阈值：连续2轮 accuracy 提升 < 3%（0.03），判定为停滞
- 停滞后处置：should_terminate=true，在结业报告中标注"该知识点学习停滞，建议调整学习目标或寻求人工辅导"

**终止与策略的关系**：
- should_terminate=true 时，next_strategy 仍输出当前匹配的策略，但 decision 强制为 "completed"
- 结业报告中包含已完成和未完成的知识点列表

## 4. 输入输出规范

### 输入
StageDecisionInput对象，包含：
- `next_strategy`：来自 `user_model_update.skill.md` 的BKT基于掌握度的策略建议（`"finish"`/`"advance"`/`"reinforce"`/`"simplify"`）
- `accuracy`：本阶段正确率（0~1浮点数）
- `coverage`：知识覆盖率（0~1浮点数，已掌握gaps数/总gaps数）
- `current_profile`：当前用户画像（UserProfile，含 knowledge_gaps、confidence_index、version）
- `iteration_history`：迭代历史列表，每轮含 accuracy、strategy_used、iteration_num

### 输出
```json
{
  "next_strategy": "advance",
  "reason": "BKT建议next_strategy='reinforce'（P_corrected=0.797∈[0.60,0.80)），但accuracy=0.85≥0.80匹配进阶策略，以策略表结果为准。未触发方向切换，未满足终止条件。",
  "should_terminate": false,
  "decision": "continue"
}
```

**字段说明**：
- `next_strategy`：最终策略标签（`"finish"` / `"advance"` / `"reinforce"` / `"simplify"` / `"backtrack"`），作为Agent⑥的最终输出字段传递给下游
- `reason`：决策理由文本，说明BKT建议、策略表匹配结果、是否触发方向切换、是否满足终止条件
- `should_terminate`：布尔值，是否终止迭代
- `decision`：`"continue"` / `"repeat"` / `"completed"`（与代码对齐）

> **职责边界**：本SKILL负责最终策略决策（`next_strategy`），`user_model_update.skill.md` 负责画像更新（knowledge_gaps/status/progress）和BKT策略建议。Orchestrator先调用 user_model_update 更新画像并生成BKT建议，再调用本SKILL读取BKT建议和已更新画像进行最终策略匹配。两SKILL的 `next_strategy` 字段含义不同：user_model_update输出的是**纯数据驱动建议**，本SKILL输出的是**结合迭代历史的最终决策**。

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 数据不完整（accuracy或coverage缺失） | 默认策略"巩固"（reinforce），decision="repeat"，reason标注"数据不完整，采用默认巩固策略" |
| 轮次信息缺失（iteration_history为空） | 按首次迭代处理（iteration_num=1），不触发方向切换和死循环检测，正常按accuracy匹配四档策略 |
| 答题记录不足（无答题数据） | accuracy=0.0，匹配"降维"策略（<0.60），decision="repeat" |
| 知识盲区列表为空（knowledge_gaps=[]） | coverage=1.0（0/0按1.0处理），触发结业判定，strategy="finish"，should_terminate=true |
| accuracy超出[0,1]范围 | confidence_index计算时通过min/max限制在[0,1]范围内 |
| 迭代历史中accuracy序列长度不足（<2轮） | 无法判定"连续2轮无效"，跳过方向切换检查，正常匹配四档策略 |
| 回溯时前置知识为空（prerequisites=[]） | 回溯到入门阶段，标注"基础不足，建议从头学习"，strategy="backtrack" |
| 内容审核循环达上限（Agent4/5审核循环≥3次仍未通过） | should_terminate=true，decision="completed"，reason标注"内容审核循环达3次上限，强制终止迭代，建议人工介入修订内容" |

## 6. 附录

### 附录A：自适应学习策略参考
| 编号 | 论文/理论 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | The Behavior of Tutoring Systems | Kurt VanLehn | 2006 | 提出智能辅导系统的"外循环-内循环"架构：外循环选择学习任务（对应本SKILL的策略选择），内循环在任务内提供步骤级帮助。外循环策略包括"宏观适应"（根据学习者整体水平调整任务顺序）和"微观适应"（根据当前任务表现调整即时反馈）。提出学习者模型驱动的外循环决策应基于正确率、耗时和知识状态三要素 | 规则1五档策略表的依据——策略选择基于accuracy（正确率）和coverage（知识覆盖率），对应VanLehn的"宏观适应"外循环决策。规则3终止条件中的max_iterations和停滞检测对应VanLehn提出的"防止学习者陷入无效循环"原则 |
| A-2 | Mastery Learning Theory | Benjamin Bloom | 1968 | 学习者需达到前一阶段的掌握标准（80%正确率）后方可进入下一阶段，否则需补偿性学习。掌握标准是进阶的必要条件而非充分条件 | 规则1中"进阶"策略阈值accuracy≥0.80的直接依据——Bloom掌握学习理论要求80%正确率方可进阶 |
| A-3 | Zone of Proximal Development (ZPD) | Lev Vygotsky | 1978 | 学习内容应处于最近发展区（i+1），难度略高于当前水平。当内容超出ZPD上限时，需降低难度（降维）；当低于ZPD下限时，需提升难度 | 规则1中"降维"策略（accuracy<0.60时降低难度）和"回溯"策略的依据——accuracy过低说明内容超出ZPD上限，需降维或回溯到前置知识 |

### 附录B：完整场景示例

#### 场景1：进阶触发示例
**输入**：
```json
{
  "next_strategy": "reinforce",
  "accuracy": 0.85,
  "coverage": 0.5,
  "current_profile": {
    "skill_tags": ["Python"],
    "knowledge_gaps": [
      {"skill": "LoRA", "level": "beginner", "status": "未开始", "progress": 0.0, "prerequisites": ["Transformer"]},
      {"skill": "Transformer", "level": "beginner", "status": "未开始", "progress": 0.0, "prerequisites": []}
    ],
    "confidence_index": 0.5,
    "version": 1
  },
  "iteration_history": [
    {"accuracy": 0.85, "strategy_used": "advance", "iteration_num": 1}
  ]
}
```

**处理过程**：
1. 步骤1：接收BKT建议 `next_strategy="reinforce"`（来自user_model_update的P_corrected=0.797建议）；accuracy=0.85，coverage=0.5（0个已掌握/2个总gap）
2. 步骤2：查策略表——accuracy≥0.85但coverage=0.5<0.90且非全部已掌握 → 不满足结业条件；accuracy≥0.80 → 匹配"进阶"策略。BKT建议为"reinforce"与策略表结果"advance"冲突，以策略表结果为准
3. 步骤3：检查方向切换——iteration_history仅1轮，无降维历史，不触发回溯；检查终止条件——非全部已掌握，iteration_num=1<5，无停滞 → 不终止
4. 步骤4：输出最终 next_strategy="advance"

**输出**：
```json
{
  "next_strategy": "advance",
  "reason": "BKT建议next_strategy='reinforce'（P_corrected=0.797∈[0.60,0.80)），但accuracy=0.85≥0.80匹配进阶策略，以策略表结果为准。非全部已掌握（Transformer仍未开始），不满足结业条件。迭代历史仅1轮，未触发方向切换。iteration_num=1<5、无停滞，不满足终止条件。",
  "should_terminate": false,
  "decision": "continue"
}
```

#### 场景2：降维切回溯示例
**输入**：
```json
{
  "next_strategy": "simplify",
  "accuracy": 0.54,
  "coverage": 0.0,
  "current_profile": {
    "knowledge_gaps": [
      {"skill": "LoRA", "level": "beginner", "status": "部分掌握", "progress": 0.54, "prerequisites": ["Transformer", "PyTorch"]}
    ],
    "confidence_index": 0.3,
    "version": 3
  },
  "iteration_history": [
    {"accuracy": 0.50, "strategy_used": "simplify", "iteration_num": 1},
    {"accuracy": 0.52, "strategy_used": "simplify", "iteration_num": 2},
    {"accuracy": 0.54, "strategy_used": "simplify", "iteration_num": 3}
  ]
}
```

**处理过程**：
1. 步骤1：接收BKT建议 `next_strategy="simplify"`（来自user_model_update的P_corrected<0.60建议）；accuracy=0.54，coverage=0.0（0个已掌握/1个总gap）
2. 步骤2：查策略表——accuracy=0.54<0.60 → 匹配"降维"策略。BKT建议"simplify"与策略表结果一致
3. 步骤3：检查方向切换——iteration_history显示连续3轮降维：
   - 第1→2轮：accuracy从0.50→0.52，提升0.02 < 0.05（第1次无效）
   - 第2→3轮：accuracy从0.52→0.54，提升0.02 < 0.05（第2次无效）
   - 连续2轮降维无效 → **自动切换为"回溯"策略**
   - 回溯目标：LoRA的prerequisites=["Transformer", "PyTorch"] → 回溯到Transformer阶段
   - 检查终止条件——iteration_num=3<5，但检查死循环：第2→3轮提升0.02<0.03，连续2轮改进<3% **但尚未触发回溯后的停滞**（死循环检测要求"已触发回溯"后的停滞）→ 不终止，执行回溯
4. 步骤4：输出最终 next_strategy="backtrack"，回溯到Transformer阶段，重置LoRA迭代计数

**输出**：
```json
{
  "next_strategy": "backtrack",
  "reason": "BKT建议next_strategy='simplify'（P_corrected<0.60），与策略表结果一致（accuracy=0.54<0.60匹配降维策略）。但连续3轮降维（0.50→0.52→0.54），每轮提升仅0.02<0.05，连续2轮无效，自动切换为回溯。回溯目标：LoRA的前置知识Transformer（prerequisites[0]）。iteration_num=3<5，未触发死循环终止（回溯后需重新检测停滞）。",
  "should_terminate": false,
  "decision": "repeat",
  "backtrack_target": "Transformer",
  "iteration_reset": true
}
```

**说明**：场景2体现了规则2方向切换的核心逻辑——当降维策略连续2轮无效（正确率提升<5%）时，系统自动判断当前难度仍超出学习者的ZPD上限，需回溯到前置知识阶段打牢基础。回溯后重置迭代计数，从前置阶段重新开始学习，避免在当前难度上无效循环。

## 7. 版本信息
- 版本：v3.2，更新日期：2026-07-24
- 变更记录：策略体系从三档扩展为**五档**（新增降维simplify、回溯backtrack）；能力描述新增BKT策略建议作为前置依赖；核心方法论重构为4步（接收BKT建议→结合建议查策略表→检查方向切换和终止→输出最终next_strategy）；规则1新增coverage≥0.90结业条件、accuracy<0.60降维策略和BKT建议参考优先级；新增规则2方向切换（降维连续2轮无效→回溯，含前置知识定位逻辑）；新增规则3终止条件（max_iterations=5、死循环检测连续2轮改进<3%停滞终止）；输入输出规范新增next_strategy字段（输入为BKT建议，输出为最终决策）；异常处理新增轮次信息缺失和回溯前置为空场景；附录A补充3个自适应学习理论；附录B新增2个完整场景。**v3.2变更：新增next_strategy作为最终输出字段（替代原strategy_label），明确与user_model_update的协作流程——接收BKT建议，结合迭代历史做最终决策，解决文档3要求的Agent⑥输出字段缺失问题**
- **与代码对齐说明**：当前 `DeterministicStageDecisionAgent` 代码实现为三档策略（advance/reinforce/finish，阈值0.80），本SKILL v3.2按扩展指南升级为五档（新增simplify/backtrack，阈值0.85/0.80/0.60）。`strategy_selector_tool.py` 已实现五档策略表（`_select_strategy`函数），与本SKILL规则1一致。后续代码需扩展 `_select_strategy` 接收BKT建议参数，并将输出字段从 `strategy_label` 重命名为 `next_strategy` 以对齐本SKILL规范
