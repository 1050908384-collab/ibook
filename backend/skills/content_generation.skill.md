# SKILL名称：content_generation

## 1. 能力描述
- 功能：为单个学习阶段生成结构化Markdown教程，并依据Agent4技术审核与Agent5教学审核意见对教程迭代修订
- 适用场景：Agent3（HybridContentAgent）初始化阶段学习包教程、审核未通过后改稿
- 前置依赖：当前阶段StageInfo（block_name、gap_skill）、用户画像UserProfile、code_style技能（代码示例风格约束）

## 2. 核心方法论
- 步骤1：读取stage.block_name与profile.goal，确定教程主题与难度面向
- 步骤2：调用LLM一次性生成结构化Markdown教程（概述→核心概念→学习步骤→示例→总结）
- 步骤3：教程交付Agent4/Agent5双重审核；未通过时接收technical_errors与quality_errors
- 步骤4：将审核意见格式化为反馈清单，调用LLM逐条修订，保留正确内容，返回完整新教程
- 步骤5：每次修订learning_package.version自增，记录迭代轮次

## 3. 知识/规则定义
- 规则1：教程面向初学者，避免未解释的术语，核心概念配示例
- 规则2：含代码片段时遵循code_style规范（命名、注释、异常处理）
- 规则3：修订必须逐条解决审核errors，不得遗漏critical/major项
- 规则4：审核意见为空时保留原教程，仅version自增

## 4. 输入输出规范
- 输入（生成）：ContentGenerationInput(session_id, stage: StageInfo)
- 输入（修订）：ContentRevisionInput(session_id, learning_package, stage, profile, technical_errors, quality_errors)
- 输出：ContentGenerationOutput(learning_package: LearningPackage)，含 tutorial(Markdown)、questions、code_examples、version

## 5. 异常处理
- LLM返回空或Error前缀 → 回退DeterministicContentAgent生成基础三段式教程
- 修订后LLM响应无效 → 保留原tutorial并追加“审核修订”清单，version自增
- 审核轮次达MAX_REVIEW_ITERATIONS → 保留未通过状态，终止迭代

## 6. 附录
- 附录A：教程Markdown三段式模板（概述/核心概念/学习步骤/示例/总结）
- 附录B：审核意见反馈清单格式化示例

## 7. 版本信息
- 版本：v2.0，更新日期：2026-07-23
