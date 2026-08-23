# AI 实训导师前端 API 对接说明

本文合并原 `学情诊断与阶段规划 Agent3_API接口任务说明.md` 与 `专业知识与教学质量审核 Agent6_API接口任务说明.md`，并以当前 `/openapi.json` 和后端实现为准。新前端只使用 `/api/v1/*`，基础地址为 `http://localhost:8500/api/v1`。

## 1. 职责与安全边界

- 学情诊断与阶段规划 Agent：诊断、画像、下一阶段规划；个性化学习资源生成 Agent：学习包和答疑；专业知识与教学质量审核 Agent：审核学习包和答疑回答；学情评估与动态决策 Agent：阶段决策、总结和画像更新。
- AssessmentService 负责建卷、保存答案和 `time_spent`、判题、正确率，不属于 专业知识与教学质量审核 Agent 的职责。
- 学习包、会话快照、未提交测试均不返回 `correct_answer`、`explanation`、`expected_output` 或内部判题规则；只有整卷 `submit` 成功后才返回答案与解析。
- 以 `/openapi.json` 生成 TypeScript 类型，不手写 `any`。

## 2. 页面流程与状态

```text
创建会话 -> 多轮诊断 -> 确认画像 -> 开始阶段 -> 学习/答疑/测试
-> 提交整卷测试 -> 完成阶段（学情评估与动态决策 Agent）-> 规划下一阶段（学情诊断与阶段规划 Agent）-> 开始下一阶段
```

| phase | 前端可用操作 |
| --- | --- |
| `diagnosing` | `submit_diagnosis` |
| `stage_ready` | `start_stage`、`edit_profile` |
| `learning` | `ask_question`、`start_assessment` |
| `stage_result` | `plan_next_stage` |
| `next_stage_ready` | `start_stage` |
| `completed` | 查看结业报告、仪表盘 |

## 3. 会话与左侧栏

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/sessions` | 创建学习会话 |
| GET | `/sessions` | 左侧会话栏、所有学习项目与画像摘要 |
| GET | `/sessions/{session_id}` | 刷新后恢复页面状态 |
| POST | `/sessions/{session_id}/abandon` | 放弃会话 |

`GET /sessions` 返回 `{ sessions: [...] }`。每项包括：
`session_id`、`title`、`status`、`phase`、`updated_at`、`goal`、`grade`、`skill_tags`、`knowledge_gaps`、`daily_hours`、`learning_style`、`version`、`current_stage_num`、`current_stage_name`、`progress`、`profile_summary`。

`GET /sessions/{session_id}` 返回 `SessionSnapshot`：
`session_id`、`title`、`status`、`phase`、`profile`、`profile_modified`、`current_stage`、`next_stage`、`learning_package`（公开版）、`assessment_summary`、`progress_summary`、`recent_qa`、`available_actions`、`created_at`、`updated_at`。

## 4. 诊断与画像

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST / GET | `/sessions/{session_id}/diagnosis/messages` | 提交并恢复多轮诊断对话 |
| GET / PATCH | `/sessions/{session_id}/profile` | 获取画像、保存设置页偏好 |
| POST | `/sessions/{session_id}/profile/confirm` | 确认或质疑画像 |

诊断请求：

```json
{ "content": "我有 PyTorch 基础，每天可学习两小时，目标是学习 QLoRA", "message_type": "user" }
```

诊断响应：`status`（`needs_more_info` 或 `profile_ready`）、`assistant_message`、`missing_fields`、`followup_questions`、`profile_draft`、`confidence`。历史响应为 `{ "messages": [{ "message_id", "role", "content", "created_at" }], "total": 0 }`。

画像字段：`profile_id`、`session_id`、`confirmed`、`goal`、`grade`、`skill_tags`、`daily_hours`、`learning_style`、`hardware`、`knowledge_gaps`、`learning_rhythm`、`confidence_index`、`confusion_markers`、`confidence`、`version`。

PATCH 只接受可选的 `goal`、`daily_hours`、`learning_style`、`hardware`，例如：

```json
{ "daily_hours": 2, "learning_style": "practice_first", "hardware": "RTX 4090 24GB" }
```

PATCH 只持久化设置，不重新诊断，也不重规划当前阶段。确认请求为 `{ "confirm": true, "notes": "" }`；只有确认后才能开始学习。

## 5. 当前阶段与学习包

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/sessions/{session_id}/stages/current` | 当前阶段摘要 |
| POST | `/sessions/{session_id}/stages/current/start` | 个性化学习资源生成 Agent 生成学习包，专业知识与教学质量审核 Agent 审核 |
| GET | `/sessions/{session_id}/stages/current/package` | 获取或恢复学习包 |

阶段字段：`stage_id`、`stage_num`、`block_id`、`block_name`、`subtopic_id`、`subtopic_name`、`estimated_hours`、`status`、`title`、`description`、`objectives`、`difficulty`、`recommended_daily_hours`、`suggested_study_days`、`learning_style_notes`、`hardware_notes`、`personalization_summary`。

学习包 `package` 字段：`stage_id`、`stage_num`、`block_name`、`subtopic_name`、`estimated_hours`、`tutorial`、`code_examples`、`learning_resources`、`questions`、`review_summary`。

`learning_resources[]`：`resource_id`、`resource_type`（`video`、`article`、`documentation`、`code`）、`title`、`url`、`platform`、`description`、`recommendation_level`、`role`、`order`。前端按 `resource_type` 分区；当前数据库可能只有视频，文章区应支持空状态或隐藏。不要使用 `recommended_videos`、`recommended_articles`。

## 6. 学习答疑

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST / GET | `/sessions/{session_id}/qa` | 个性化学习资源生成 Agent 答疑、专业知识与教学质量审核 Agent 审核、恢复历史 |

POST 请求：`{ "question": "QLoRA 的 NF4 量化为什么适合微调？" }`。

单条响应：`qa_id`、`session_id`、`question`、`answer`、`stage_id`、`created_at`、`source_summary`，以及 `review_summary`：`technical_passed`、`quality_passed`、`iterations`、`status`、`needs_human_review`。历史响应为 `{ "items": [...] }`。

## 7. 测试与答题

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/sessions/{session_id}/stages/current/assessment-attempts` | 创建或恢复本阶段测试 |
| GET | `/assessment-attempts/{attempt_id}` | 恢复测试进度 |
| PUT | `/assessment-attempts/{attempt_id}/answers/{question_id}` | 保存或覆盖单题答案 |
| POST | `/assessment-attempts/{attempt_id}/submit` | 完整性校验、判题、公布结果 |

创建/读取测试字段：`attempt_id`、`session_id`、`stage_id`、`status`、`total_questions`、`answered_count`、`questions`、`created_at`；创建接口另有 `resumed`，为 `true` 时表示复用了同阶段未提交试卷。

保存答案请求：

```json
{ "user_answer": "A", "time_spent": 42 }
```

`time_spent` 单位为秒。保存响应：`attempt_id`、`question_id`、`saved`、`answered_count`、`total_questions`、`status`。

约束：同一会话同一阶段只有一份 `in_progress` 测试；提交前可覆盖答案；必答题未完成不能提交；已提交不能再修改。`submit` 成功后题目才额外返回 `correct_answer`、`explanation`、`is_correct`，顶层才有 `correct_count`、`accuracy`、`submitted_at`。

## 8. 阶段完成、下一阶段与仪表盘

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/sessions/{session_id}/stages/current/complete` | 已交卷后调用 学情评估与动态决策 Agent 阶段决策 |
| POST | `/sessions/{session_id}/stages/next/plan` | 学情诊断与阶段规划 Agent 规划下一阶段 |
| GET | `/sessions/{session_id}/dashboard` | 仪表盘数据库聚合数据 |

`complete` 的前置条件是当前阶段存在 `submitted` 测试；否则返回 `409`，错误码为 `ASSESSMENT_NOT_SUBMITTED`。完成阶段主要响应：`status`（`continue`、`repeat`、`completed`）、`accuracy`、`stage_summary`、`next_strategy`、`updated_profile`、`can_plan_next_stage`、`all_mastered`、`final_report`、`phase`、`available_actions`。

下一阶段规划请求：

```json
{ "adjustment_notes": "希望增加更多 vLLM 部署练习" }
```

规划成功后：`phase=next_stage_ready`，`available_actions=["start_stage"]`。

仪表盘字段：`session_id`、`profile_summary`、`current_stage_summary`、`overall_progress`、`current_accuracy`、`confidence_index`、`learning_rhythm`、`mastered_count`、`total_count`、`block_progress`、`stage_history`、`accuracy_trend`、`recent_qa`。

## 9. 错误处理

- `404`：会话、阶段、画像、测试或题目不存在。
- `409`：状态不允许，例如未确认画像就开始学习、未交卷就完成阶段、已提交测试后继续修改答案。
- `422`：请求体字段或类型错误。
- `500`：服务端错误。开发阶段可展示 `detail`，生产环境应使用统一错误提示。
