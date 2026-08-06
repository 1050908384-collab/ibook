# SKILL名称：stage_planning

## 1. 能力描述
- 功能：从学习者画像中选取**下一个待学知识盲区**，生成单阶段学习目录信息（StageInfo）。对齐 Agent1 `DeterministicDiagnosticAgent.generate_stage` 的确定性实现——取首个未掌握 gap，用 uuid5 派生确定性 ID，estimated_hours 固定 2.0
- 适用场景（3个）：
  1. **规划下一阶段**：阶段开始时由 `DeterministicDiagnosticAgent.generate_stage` 触发，从 profile.knowledge_gaps 选取下一个待学盲区
  2. **回溯后重规划**：Agent6 触发回溯策略后，重新规划前置知识阶段
  3. **结业判定前置**：无未掌握盲区时抛出异常，由上层判定结业
- **休眠状态说明**：当前新框架 `DeterministicDiagnosticAgent.generate_stage` 为纯确定性实现，未通过 `load_skills` 加载本SKILL（逻辑内联于代码），本SKILL内容已对齐其行为，作为休眠方法论备查
- 前置依赖：
  - 学习者画像 UserProfile（`knowledge_gaps` 列表、`stage_num`）
  - 会话标识 `session_id`（用于 uuid5 派生）

## 2. 核心方法论
- **步骤1：选取首个未掌握盲区**——遍历 `profile.knowledge_gaps`，选取首个 `status ≠ "已掌握"` 的 gap
- **步骤2：派生确定性ID**——以 `session_id + stage_num + gap.skill` 为输入用 uuid5 派生 `stage_id` 与 `block_id`，保证同一会话同一阶段幂等
- **步骤3：填充阶段字段**——`block_name = gap.skill`，`estimated_hours = 2.0`（固定值），`prerequisites = gap.prerequisites`，`stage_num` 取当前阶段序号
- **步骤4：无未掌握盲区处理**——所有 gap 均已掌握时抛出 `ValueError("No unmastered knowledge gap remains")`，由上层判定结业

## 3. 知识/规则定义

### 规则1：阶段选取规范
| 检查项 | 要求 | 违规处置 |
|:---|:---|:---|
| 单阶段输出 | 仅输出1个学习阶段，禁止一次生成多阶段路线 | 违反则不符合 Agent1 契约 |
| 盲区选取顺序 | 取 knowledge_gaps 中首个 `status ≠ "已掌握"` 的 gap | 全部已掌握时抛异常 |
| gap_skill 来源 | `stage.gap_skill = gap.skill` | 不生成临时技能名 |

### 规则2：确定性ID派生
| 字段 | 派生方式 | 幂等性 |
|:---|:---|:---|
| `stage_id` | `uuid5(NAMESPACE_DNS, f"{session_id}-{stage_num}-{gap.skill}")` | 同 session_id+stage_num+skill → 同 stage_id |
| `block_id` | `uuid5(NAMESPACE_DNS, f"{gap.skill}")` 或沿用 gap.skill | 同 skill → 同 block_id |

**幂等性依据**：uuid5 基于命名空间+名称的 SHA-1 哈希，相同输入恒定输出，保证同一会话重复规划不产生新ID，便于状态追踪与断点续学。

### 规则3：阶段字段规范
| 字段 | 取值 | 说明 |
|:---|:---|:---|
| `block_name` | `gap.skill` | 阶段主题 |
| `estimated_hours` | 2.0 | 固定值，新框架不再读图谱 block 字段 |
| `prerequisites` | `gap.prerequisites` | 沿用盲区自带先修依赖 |
| `stage_num` | 当前阶段序号 | 从1递增 |
| `gap_skill` | `gap.skill` | 匹配的盲区技能名 |

### 规则4：StageInfo 契约对齐
- 字段须符合 `StageInfo` 契约（`backend/models/contracts.py` 与 `stage.py`）
- 关键字段：`stage_id`、`stage_num`、`block_id`、`block_name`、`estimated_hours`、`gap_skill`、`prerequisites`

## 4. 输入输出规范

### 输入
`StagePlanningInput` 对象：
- `session_id`
- `profile`：UserProfile（`knowledge_gaps`、`stage_num`）
- `stage_num`：当前阶段序号

### 输出
`StagePlanningOutput` 对象：
```json
{
  "stage": {
    "stage_id": "a1b2c3d4-...",
    "stage_num": 1,
    "block_id": "e5f6g7h8-...",
    "block_name": "LoRA",
    "estimated_hours": 2.0,
    "gap_skill": "LoRA",
    "prerequisites": ["Transformer", "注意力机制"]
  }
}
```

**字段说明**：
- `stage`：StageInfo 对象，单阶段信息

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 所有盲区均已掌握 | 抛出 `ValueError("No unmastered knowledge gap remains")`，由上层判定结业 |
| `gap.prerequisites` 为空 | `prerequisites` 输出空数组，不阻断规划 |
| `knowledge_gaps` 为空 | 同"全部已掌握"处理，抛异常 |
| `profile` 缺失或 stage_num 无效 | 抛出 ValueError，终止规划 |

## 6. 附录

### 附录A：阶段规划理论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | 掌握学习理论（Mastery Learning） | Benjamin Bloom | 1968 | 学习者需达到前一阶段掌握标准（80%）方可进入下一阶段 | 规则1盲区选取的依据——仅取 status≠"已掌握"的 gap，未掌握不进阶 |
| A-2 | 最近发展区理论（ZPD） | Lev Vygotsky | 1978 | 学习内容应处于最近发展区，难度略高于当前水平 | prerequisites 字段的依据——阶段须先修前置知识，确保不超 ZPD |
| A-3 | 逆向设计（Backward Design） | Wiggins & McTighe | 1998 | 课程设计应从目标出发逆推学习阶段，每阶段指向最终目标 | 单阶段规划服务于 `profile.goal`，逐阶段逼近最终目标 |
| A-4 | 知识空间理论（Knowledge Space Theory） | Doignon & Falmagne | 1985 | 知识结构可建模为状态空间，学习路径是状态间的合法转移序列 | knowledge_gaps 的 prerequisites 定义合法学习顺序，避免跳跃 |

### 附录B：完整场景示例

#### 场景1：正常规划下一阶段
**输入**：
```json
{
  "session_id": "sess-001",
  "profile": {
    "knowledge_gaps": [
      {"skill": "LoRA", "level": "beginner", "status": "未开始", "progress": 0.0, "prerequisites": ["Transformer", "注意力机制"]},
      {"skill": "Transformer", "level": "intermediate", "status": "已掌握", "progress": 1.0, "prerequisites": []}
    ]
  },
  "stage_num": 1
}
```

**处理过程**：
1. 步骤1：遍历 gaps——Transformer 已掌握（跳过），LoRA 未开始 → 选取 LoRA
2. 步骤2：uuid5 派生 stage_id 与 block_id
3. 步骤3：block_name=LoRA，estimated_hours=2.0，prerequisites=["Transformer","注意力机制"]

**输出**：stage（block_name=LoRA，stage_num=1）

#### 场景2：无未掌握盲区（结业）
**输入**：
```json
{
  "profile": {
    "knowledge_gaps": [
      {"skill": "LoRA", "status": "已掌握", "progress": 1.0},
      {"skill": "Transformer", "status": "已掌握", "progress": 1.0}
    ]
  },
  "stage_num": 3
}
```

**处理过程**：
1. 步骤1：遍历 gaps——全部 status="已掌握"
2. 步骤4：抛出 `ValueError("No unmastered knowledge gap remains")`

**输出**：异常，由上层 Orchestrator 捕获并触发结业流程

**说明**：场景2体现规则1的结业判定——所有盲区掌握后不再规划新阶段，转交 Agent6 的"结业"策略。

## 7. 版本信息
- 版本：v3.0，更新日期：2026-07-23
- 变更记录：能力描述明确3个适用场景与休眠状态说明（对齐 `DeterministicDiagnosticAgent.generate_stage` 行为）；核心方法论重构为4步（选取盲区→派生ID→填充字段→无盲区处理）；规则1新增阶段选取规范表；规则2新增确定性ID派生表（含uuid5幂等性说明）；规则3新增阶段字段规范表（estimated_hours固定2.0）；规则4明确 StageInfo 契约对齐；输入输出规范补完整 StagePlanningOutput JSON示例；异常处理扩展为4场景表格；附录A补4个阶段规划理论（Bloom掌握学习/Vygotsky ZPD/逆向设计/知识空间理论）；附录B新增2个完整场景（正常规划+无盲区结业）


## 8. 2026-07-30 增补

- 当前项目落地已切到知识点级 `DAG + mastery state`
- Agent1.generate_stage 每次只做三步：
  1. 找出所有 `status = -1` 的知识点
  2. 过滤出前置知识点都已 `= 1` 的候选项
  3. 从候选项中选一个最适合当前用户的知识点
- `knowledge_gaps` 保留，但只作为板块级汇总参考
- 当前版本不做回退；失败时重复当前知识点

## 9. 当前运行时覆盖规则（2026-08-06）

本节优先于上文的历史实现说明。运行时以 `backend/agents/deterministic.py` 和知识图谱数据为准。

- 一个 Stage 必须且只绑定一个 `subtopic_id`，不再以知识板块或单个 `knowledge_gap` 作为阶段粒度。
- 候选范围是目标知识点及其 `HARD_PREREQUISITE_OF` 硬前置闭包；`knowledge_gaps` 只用于诊断目标，不能直接决定阶段顺序。
- 仅当一个候选知识点的直接硬前置均已完成时，才可被选为下一阶段；推荐前置不阻塞学习。
- `estimated_hours` 读取知识点的原始时长，包括环境准备知识点的 `0.0h`，不得替换为固定 `2.0h` 或最小时长。
- 每日可学习时长只用于计算 `recommended_daily_hours` 与 `suggested_study_days`，不得将多个知识点合并到一个 Stage。`0.0h` 节点的每日时长为 `0.0h`，建议学习天数为 `1`。
- `KBS-01-00`（Linux/WSL 环境准备）到 `KBS-02-00`（Python 环境准备）是推荐关系，Python 学习路径不因该节点未完成而被阻塞。
