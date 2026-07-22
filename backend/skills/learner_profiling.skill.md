# SKILL名称：learner_profiling

## 1. 能力描述
- 功能：从用户自然语言输入提取标准化学习者画像（技能标签、知识盲区、目标、时长、学习风格），置信不足时生成追问
- 适用场景：Agent1（HybridDiagnosticAgent）首次对话诊断；当前新框架未加载本技能，作为休眠方法论备查
- 前置依赖：用户原始对话文本；可选历史交互

## 2. 核心方法论
- 步骤1：识别输入中的技术术语、学习目标、每日时长、学习基础与风格
- 步骤2：输出skill_tags（已掌握）、knowledge_gaps（待学，每项含skill与level）
- 步骤3：提取goal、daily_hours（未知取2）、learning_style（theory/practice/mixed）
- 步骤4：综合信息完整度评估confidence；低于阈值时生成follow_up_questions补齐
- 步骤5：LLM响应不可解析时回退正则确定性诊断

## 3. 知识/规则定义
- 规则1：knowledge_gaps每项包含skill与level（beginner/intermediate/advanced）
- 规则2：confidence阈值由settings.CONFIDENCE_THRESHOLD控制，低于阈值返回needs_more_info
- 规则3：画像字段须符合UserProfile契约
- 规则4：LLM必须输出严格JSON，解析失败回退确定性诊断

## 4. 输入输出规范
- 输入：DiagnosticInput(session_id, user_input)
- 输出：DiagnosticOutput(profile: UserProfile, follow_up_questions: List[str])
- UserProfile关键字段：skill_tags、knowledge_gaps(skill,level,status,progress,prerequisites)、goal、daily_hours、learning_style、confidence

## 5. 异常处理
- 用户输入过短/无有效信息 → 降低confidence，触发追问
- LLM返回非JSON → 回退DeterministicDiagnosticAgent正则匹配SKILL_NAMES
- 无识别到目标 → 默认“通用学习目标”，confidence=0.75

## 6. 附录
- 附录A：UserProfile标准JSON示例
- 附录B：确定性诊断SKILL_NAMES参考列表

## 7. 版本信息
- 版本：v2.0，更新日期：2026-07-23（注：新框架Agent1当前未加载本技能，内容已对齐HybridDiagnosticAgent行为）
