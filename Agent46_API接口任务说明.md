# Agent4/6 前端 API 接口任务说明

本文档覆盖 Agent4/6 队友负责的前端 API：答疑审核结果、测试服务、Agent6 阶段决策和仪表盘。

## 职责边界

- Agent4 只负责审核学习包和答疑回答。
- `AssessmentService` 负责创建测试、保存答案、判题和计算正确率，不属于 Agent4 的运行职责。
- Agent6 负责基于学习结果生成阶段总结、下一阶段策略和更新后的画像。
- 仪表盘由后端聚合数据库数据，不属于 Agent4 的运行职责。

## 1. 学习中的答疑审核

| 接口 | 用途 | 负责人 |
| --- | --- | --- |
| `POST /api/v1/sessions/{session_id}/qa` | 用户提问，Agent3 回答，Agent4 审核 | Agent1/3 调用流程；Agent4/6 提供审核结果 |
| `GET /api/v1/sessions/{session_id}/qa` | 获取当前会话的问答历史 | Agent1/3 |

本队友需要提供答疑审核结果结构。答疑返回内容：

```text
qa_id
question
answer
stage_id
review_summary
source_summary
created_at
```

Agent4 在答疑链路中的职责仅是审核 `answer`，并返回审核摘要：

```text
technical_passed
quality_passed
iterations
status
needs_human_review
```

## 2. 测试接口

本组接口属于 `AssessmentService`，由 Agent4/6 队友承担工程实现，但不属于 Agent4 的运行职责。

| 接口 | 用途 | 工程负责人 |
| --- | --- | --- |
| `POST /api/v1/sessions/{session_id}/stages/current/assessment-attempts` | 根据当前学习包创建一次测试 | Agent4/6 队友 |
| `GET /api/v1/assessment-attempts/{attempt_id}` | 刷新后恢复测试进度 | Agent4/6 队友 |
| `PUT /api/v1/assessment-attempts/{attempt_id}/answers/{question_id}` | 保存或覆盖单题答案 | Agent4/6 队友 |
| `POST /api/v1/assessment-attempts/{attempt_id}/submit` | 检查完整性、统一判题并公布结果 | Agent4/6 队友 |

### 测试模块约束

- 每次测试有独立 `attempt_id`。
- 测试开始时冻结当前学习包的题目。
- 同一题可以在提交前修改，但数据库只保存一条最终答案。
- 未完成全部必答题时不能提交。
- 提交后不能再修改答案。
- 只有整份测试提交后，才能返回正确答案和解析。
- 正确率只能从已提交的完整 attempt 计算，不能从零散答题记录计算。

### 建议新增的数据表

```text
assessment_attempts
assessment_attempt_questions
assessment_answers
```

## 3. Agent6 阶段决策

| 接口 | 用途 | 负责人 |
| --- | --- | --- |
| `POST /api/v1/sessions/{session_id}/stages/current/complete` | 测试完成后调用 Agent6，生成总结与策略 | Agent4/6 |
| `POST /api/v1/sessions/{session_id}/stages/next/plan` | Agent1 根据新画像规划下一阶段 | Agent1/3 |

### `complete` 的处理流程

```text
读取已提交测试的准确率
→ Orchestrator 读取问答历史、交互记录、画像
→ 调用 Agent6
→ 保存 Agent6 返回的更新画像、策略和总结
→ 更新子知识点掌握状态
→ 返回阶段结果
```

### 阶段结果返回内容

```text
status: continue | repeat | completed
accuracy
stage_summary
next_strategy
updated_profile
can_plan_next_stage
all_mastered
final_report
```

未结业时：

```text
phase = stage_result
available_actions = ["plan_next_stage"]
```

`plan-next` 成功后：

```text
phase = next_stage_ready
available_actions = ["start_stage"]
```

## 4. 仪表盘

| 接口 | 用途 | 工程负责人 |
| --- | --- | --- |
| `GET /api/v1/sessions/{session_id}/dashboard` | 获取仪表盘主数据 | Agent4/6 队友 |

仪表盘由后端聚合数据库数据，不属于 Agent4 的运行职责。主要数据来源：

```text
user_profiles
stage_progress
subtopic_status
assessment_attempts
user_qa
```

仪表盘返回内容：

```text
profile_summary
current_stage_summary
overall_progress
current_accuracy
confidence_index
learning_rhythm
mastered_count
total_count
block_progress
stage_history
accuracy_trend
recent_qa
```
