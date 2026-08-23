# AI Agent Trainer API文档

## 一、基础信息

- **API地址**: `http://localhost:8500/api`
- **Content-Type**: `application/json`
- **认证方式**: 无（开发阶段）

## 二、对话接口

### 2.1 初始对话

**POST** `/api/chat/`

创建新会话或处理初始输入。

请求体：
```json
{
  "session_id": "可选，已有会话ID",
  "message": "用户输入内容"
}
```

响应：
```json
{
  "session_id": "会话ID",
  "response": {
    "status": "ready",
    "profile": { ... },
    "stage_info": { ... }
  }
}
```

### 2.2 处理问题

**POST** `/api/chat/question`

处理用户在学习过程中的提问。

请求体：
```json
{
  "session_id": "会话ID",
  "message": "用户问题"
}
```

响应：
```json
{
  "session_id": "会话ID",
  "response": {
    "answer": "回答内容",
    "tech_pass": true,
    "quality_pass": true,
    "iterations": 1,
    "review_history": [
      {
        "iteration": 1,
        "technical": {"passed": true, "errors": []},
        "quality": {"passed": true, "errors": []},
        "passed": true
      }
    ]
  }
}
```

## 三、会话接口

### 3.1 创建会话

**POST** `/api/session/`

创建新会话。

响应：
```json
{
  "session_id": "会话ID",
  "status": "active"
}
```

### 3.2 获取会话

**GET** `/api/session/{session_id}`

获取会话信息。

响应：
```json
{
  "session_id": "会话ID",
  "profile": { ... },
  "current_stage": { ... },
  "status": "active"
}
```

### 3.3 结束会话

**PUT** `/api/session/{session_id}/end`

结束会话。

响应：
```json
{
  "session_id": "会话ID",
  "status": "completed"
}
```

### 3.4 放弃会话

**PUT** `/api/session/{session_id}/abandon`

放弃会话。

响应：
```json
{
  "session_id": "会话ID",
  "status": "abandoned"
}
```

## 四、阶段接口

### 4.1 开始阶段

**POST** `/api/stage/start`

开始当前阶段学习。

请求体：
```json
{
  "session_id": "会话ID"
}
```

响应：
```json
{
  "learning_package": { ... },
  "review_passed": true,
  "iterations": 1,
  "review_history": [
    {
      "iteration": 1,
      "technical": {"passed": true, "errors": []},
      "quality": {"passed": true, "errors": []},
      "passed": true
    }
  ]
}
```

### 4.2 提交答案

**POST** `/api/stage/submit-answer`

提交测试题答案。

请求体：
```json
{
  "session_id": "会话ID",
  "question_id": "题目ID",
  "user_answer": "用户答案"
}
```

响应：
```json
{
  "is_correct": true
}
```

### 4.3 完成阶段

**POST** `/api/stage/complete/{session_id}`

完成当前阶段并调用 学情评估与动态决策 Agent 决策。该接口只保存当前阶段结果和更新后的画像，不直接生成下一阶段；仍有未掌握内容时返回 `can_plan_next_stage=true`。

响应：
```json
{
  "status": "continue",
  "accuracy": 0.85,
  "updated_profile": { ... },
  "next_strategy": "continue",
  "can_plan_next_stage": true,
  "all_mastered": false,
  "session_status": "active"
}
```

### 4.4 规划下一阶段

**POST** `/api/stage/plan-next`

用户确认继续后调用。Orchestrator 基于 学情评估与动态决策 Agent 更新后的画像调用 学情诊断与阶段规划 Agent 的 `generate_stage`（逻辑 Agent2 的阶段规划职责），生成并保存 `next_stage`；不会立即覆盖当前阶段。

请求体：
```json
{
  "session_id": "会话ID",
  "adjustment_notes": "可选，例如先学 QLoRA 再看部署"
}
```

响应：
```json
{
  "status": "planned",
  "session_id": "会话ID",
  "next_stage": { ... },
  "updated_profile": { ... },
  "planning_notes": "...",
  "session_status": "active"
}
```

## 五、可视化接口

### 5.1 获取画像数据

**GET** `/api/visualize/profile/{session_id}`

获取用户画像数据。

响应：
```json
{
  "session_id": "会话ID",
  "skill_tags": ["PyTorch基础", "Transformer"],
  "knowledge_gaps": [
    {
      "skill": "LoRA",
      "block_id": "KB-12",
      "block_name": "LoRA/QLoRA参数高效微调",
      "target_subtopic_ids": ["KBS-12-01"],
      "mapping_confidence": 0.9,
      "status": "未掌握",
      "progress": 0.0
    }
  ],
  "goal": "掌握LoRA微调",
  "daily_hours": 2.0,
  "learning_style": "mixed",
  "confidence": 0.85
}
```

### 5.2 获取阶段数据

**GET** `/api/visualize/stage/{session_id}`

获取当前阶段信息。

响应：
```json
{
  "stage_id": "STG-001",
  "stage_num": 1,
  "block_id": "KB-003",
  "block_name": "Transformer架构",
  "estimated_hours": 4.0,
  "prerequisites": ["KB-002"]
}
```

### 5.3 获取进度数据

**GET** `/api/visualize/progress/{session_id}`

获取学习进度数据。

响应：
```json
{
  "accuracy": 0.85,
  "knowledge_gaps": [...],
  "learning_rhythm": "medium",
  "confidence_index": 0.75
}
```

## 六、错误响应

所有错误返回以下格式：

```json
{
  "detail": "错误描述"
}
```

常见错误码：
- 404: 会话/资源不存在
- 500: 服务器内部错误

## 七、示例

### 7.1 完整流程示例

```bash
# 1. 创建会话
curl -X POST http://localhost:8500/api/session/

# 2. 初始输入
curl -X POST http://localhost:8500/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "message": "我有PyTorch基础，想学习LoRA微调"}'

# 3. 开始阶段
curl -X POST http://localhost:8500/api/stage/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx"}'

# 4. 提问
curl -X POST http://localhost:8500/api/chat/question \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "message": "什么是LoRA？"}'

# 5. 提交答案
curl -X POST http://localhost:8500/api/stage/submit-answer \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "question_id": "Q-001", "user_answer": "def"}'

# 6. 完成阶段
curl -X POST http://localhost:8500/api/stage/complete/xxx

# 7. 用户确认后规划下一阶段（未全部掌握时）
curl -X POST http://localhost:8500/api/stage/plan-next \
  -H "Content-Type: application/json" \
  -d '{"session_id": "xxx", "adjustment_notes": "先学QLoRA再看部署"}'
```
