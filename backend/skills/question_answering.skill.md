# SKILL名称：question_answering

## 1. 能力描述
- 功能：针对用户学习过程中的提问，结合当前阶段与用户画像生成简明答疑，并依据Agent4/Agent5审核意见迭代修订
- 适用场景：Agent3（HybridContentAgent）处理用户实时提问、答疑审核未通过后改稿
- 前置依赖：用户提问文本、当前阶段StageInfo、用户画像UserProfile

## 2. 核心方法论
- 步骤1：读取stage.block_name与profile.goal，定位问题与当前阶段关系
- 步骤2：调用LLM直接回答——先给简明结论，再解释原理，需要时给示例
- 步骤3：回答交付Agent4/Agent5审核；未通过时接收technical_errors与quality_errors
- 步骤4：将审核意见格式化为反馈，调用LLM逐条修复，返回完整修订后回答

## 3. 知识/规则定义
- 规则1：不得编造来源、参数、模型原理；无法确认时明确说明
- 规则2：回答简洁直击核心，复杂原理分段解释
- 规则3：与当前阶段无关的问题简短作答，提示聚焦本阶段
- 规则4：修订必须逐条解决审核errors，保持技术准确与教学清晰

## 4. 输入输出规范
- 输入（生成）：AnswerGenerationInput(session_id, question, stage, profile)
- 输入（修订）：AnswerRevisionInput(session_id, question, answer, stage, profile, technical_errors, quality_errors)
- 输出：AnswerGenerationOutput(answer: str)

## 5. 异常处理
- LLM返回空或Error前缀 → 回退DeterministicContentAgent.generate_answer基础话术
- 修订后LLM响应无效 → 保留原answer并追加“补充修订”清单
- 审核轮次达MAX_REVIEW_ITERATIONS → 返回未通过状态

## 6. 附录
- 附录A：简明答疑模板（结论→原理→示例）
- 附录B：审核意见反馈格式化示例

## 7. 版本信息
- 版本：v2.0，更新日期：2026-07-23
