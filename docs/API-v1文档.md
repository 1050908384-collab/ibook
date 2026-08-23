# AI Agent Trainer v1 API

Base URL: `http://localhost:8500/api/v1`

The legacy `/api/*` endpoints remain for compatibility. New frontend work must use
the v1 paths below. The backend OpenAPI document at `/openapi.json` is the source
of generated frontend types.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/sessions` | Create a learning session |
| GET | `/sessions` | Sidebar session list and profile summaries |
| GET | `/sessions/{session_id}` | Restore a safe session snapshot |
| POST | `/sessions/{session_id}/abandon` | Abandon a session |
| POST / GET | `/sessions/{session_id}/diagnosis/messages` | Submit and restore diagnosis messages |
| GET / PATCH | `/sessions/{session_id}/profile` | Read profile and persist preferences |
| POST | `/sessions/{session_id}/profile/confirm` | Confirm profile before study starts |
| GET | `/sessions/{session_id}/stages/current` | Current stage summary |
| POST | `/sessions/{session_id}/stages/current/start` | Generate and review a learning package |
| GET | `/sessions/{session_id}/stages/current/package` | Public learning package and resources |
| POST / GET | `/sessions/{session_id}/qa` | 个性化学习资源生成 Agent answer, 专业知识与教学质量审核 Agent review, and history |
| POST | `/sessions/{session_id}/stages/current/assessment-attempts` | Create or resume an assessment |
| GET | `/assessment-attempts/{attempt_id}` | Restore assessment progress |
| PUT | `/assessment-attempts/{attempt_id}/answers/{question_id}` | Save answer and `time_spent` seconds |
| POST | `/assessment-attempts/{attempt_id}/submit` | Validate, grade, and disclose results |
| POST | `/sessions/{session_id}/stages/current/complete` | 学情评估与动态决策 Agent decision after a submitted assessment |
| POST | `/sessions/{session_id}/stages/next/plan` | 学情诊断与阶段规划 Agent plans the next stage |
| GET | `/sessions/{session_id}/dashboard` | Dashboard aggregation |

## Workflow Constraints

- Public learning-package payloads never include answers, explanations, or internal grading rules.
- A profile PATCH only persists `goal`, `daily_hours`, `learning_style`, and `hardware`; it does not rediagnose or replan the current stage.
- `stage_ready` requires profile confirmation before `start`.
- In `learning`, frontend actions are `ask_question` and `start_assessment`.
- AssessmentService owns attempts, answers, time tracking, and grading. 专业知识与教学质量审核 Agent reviews learning packages and Q&A only. 学情评估与动态决策 Agent makes stage decisions and updates the profile.
- A current stage can complete only after a complete assessment attempt is submitted.
