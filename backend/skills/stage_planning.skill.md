# SKILL名称：stage_planning

## 1. 能力描述
- 功能：从学习者画像中选取下一个待学知识盲区，生成单阶段学习目录信息
- 适用场景：Agent2（DeterministicStagePlannerAgent）规划下一阶段；当前新框架未加载本技能，作为休眠方法论备查
- 前置依赖：学习者画像UserProfile（knowledge_gaps、stage_num）

## 2. 核心方法论
- 步骤1：遍历profile.knowledge_gaps，选取首个status≠“已掌握”的盲区
- 步骤2：以gap.skill为block_name，生成确定性stage_id与block_id（uuid5）
- 步骤3：写入estimated_hours（默认2.0）、prerequisites（取自gap.prerequisites）、stage_num
- 步骤4：无未掌握盲区时抛出异常终止

## 3. 知识/规则定义
- 规则1：仅输出单个学习阶段，禁止一次生成多阶段路线
- 规则2：stage_id与block_id由session_id+stage_num+gap.skill确定性派生（uuid5），保证幂等
- 规则3：estimated_hours默认2.0，prerequisites沿用gap.prerequisites
- 规则4：字段须符合StageInfo契约

## 4. 输入输出规范
- 输入：StagePlanningInput(session_id, profile: UserProfile, stage_num)
- 输出：StagePlanningOutput(stage: StageInfo)
- StageInfo关键字段：stage_id、stage_num、block_id、block_name、estimated_hours、gap_skill、prerequisites

## 5. 异常处理
- 所有盲区均已掌握 → 抛出ValueError("No unmastered knowledge gap remains")
- gap.prerequisites为空 → prerequisites输出空数组，不阻断

## 6. 附录
- 附录A：StageInfo标准JSON示例
- 附录B：uuid5确定性ID派生规则

## 7. 版本信息
- 版本：v2.0，更新日期：2026-07-23（注：新框架Agent2为纯确定性实现，当前未加载本技能，内容已对齐其行为）
