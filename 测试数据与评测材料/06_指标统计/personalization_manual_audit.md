# 50组个性化学习包人工审查报告

审查日期：2026-08-28

审查人：Codex-AI-Reviewer

## 1. 审查范围与证据

本次审查按《审核与评测操作手册》、`测试3+50要求.md`、`测试数据建议方法.md`、`完整版全部流程.docx`、`框架设计.md`、`数据库框架.md`及 `docs/` 中的设计/审核协议执行。

冻结输出实际位置为：

`测试数据与评测材料/00_冻结备份/raw_outputs_20260827/`

手册中提到的 `02_50组个性化主评测/raw_outputs/` 在当前工作区不存在，因此本报告以冻结备份为唯一运行证据；输入画像仍来自 `02_50组个性化主评测/evaluation_cases.csv`。

全量读取结果：50 个案例、748 个文件、25,491,849 字节；全部 JSON 文件解析成功，核心文件（画像、诊断、阶段规划、学习包、审核轨迹、题目、作答、阶段决策）均存在。普通阶段和空阶段均读取了目录下的全部文件，包含多阶段案例的 `stage_2`。

## 2. 判定口径

- 普通阶段按 5 项联合判定：`target_path`、`difficulty`、`time_budget`、`hardware`、`learning_style`。
- 空阶段不套用 5 项，不进入普通学习包适配率分母；另按“无待学点、无普通学习包、无测试题、正确结业状态”检查。
- 证据字段以 `learner_profile.json`、`stage_planning_output.json`、`learning_package.md`、`assessment_questions.json`、`review_trace.json` 为主。
- `review_trace.review_passed=true` 仅证明 Agent4 链路执行过；`package_status=provisional` 或 `needs_human_review=true` 仍表示最终包需人工关注。

## 3. 普通阶段逐案矩阵（28组，140项）

缩写：T=目标路径，D=难度，Tm=时长，H=硬件，S=学习风格。

| 案例 | 阶段规划 | T | D | Tm | H | S | 主要证据/问题 |
|---|---|---:|---:|---:|---:|---:|---|
| case_001 | KB-02 / KBS-02-00 | pass | pass | pass | fail | fail | 零基础先做环境准备；画像硬件缺失，风格被统一为 hands-on，规划只给通用混合建议 |
| case_002 | KB-07 / KBS-07-00 | pass | pass | pass | fail | pass | Python基础后进入HF环境；硬件未入画像 |
| case_003 | KB-11 / KBS-11-00,01 | fail | pass | pass | fail | fail | 无PyTorch基础却进入HF数据处理；硬件/风格字段缺失 |
| case_004 | KB-08 / KBS-08-00..02 | fail | fail | pass | fail | fail | 仅基础Python却直接安排多项Transformer高阶内容，未补PyTorch/Tokenizer |
| case_005 | KB-01 / KBS-01-02,03 | pass | pass | pass | fail | fail | 路径可作为前置补全；硬件/风格字段缺失 |
| case_006 | KB-08 / KBS-08-00,01 | fail | fail | pass | fail | fail | 无编程基础却要求运行transformers并理解Encoder-Decoder |
| case_007 | KB-01 / KBS-01-04 | fail | pass | pass | fail | fail | 旧Python脚本经验却要求已配置环境的nvidia-smi监控 |
| case_008 | KB-11 / KBS-11-02 | fail | fail | pass | fail | pass | 无Python基础直接做datasets.map/filter/shuffle |
| case_009 | KB-11 / KBS-11-02 | pass | pass | pass | fail | fail | 阶段内容可作为LoRA前置；硬件/风格字段缺失 |
| case_010 | KB-08 / KBS-08-00..03 | pass | pass | pass | fail | fail | 目标路径合理；硬件/风格字段缺失 |
| case_011 | KB-01 / KBS-01-04 | pass | pass | pass | fail | fail | 阶段可作为前置补全；硬件/风格字段缺失 |
| case_012 | KB-08 / KBS-08-04,05 | fail | fail | pass | fail | fail | 仅Tokenizer基础即跳到FFN/SwiGLU、RoPE |
| case_013 | KB-04 / KBS-04-00,01 | pass | pass | pass | fail | fail | PyTorch前置路径合理；硬件/风格字段缺失 |
| case_014 | KB-11 / KBS-11-03 | pass | pass | pass | fail | fail | 数据处理前置合理；硬件/风格字段缺失 |
| case_015 | KB-12 / KBS-12-00 | pass | pass | pass | fail | fail | LoRA环境准备合理；硬件/风格字段缺失 |
| case_016 | KB-04 / KBS-04-02 | fail | pass | pass | fail | fail | QLoRA目标却重复自动求导，未推进KB-12 |
| case_017 | KB-04 / KBS-04-03 | fail | pass | pass | fail | pass | 已有KB-04却重复torch.nn；混合偏好体现较好 |
| case_018 | KB-12 / KBS-12-01..03 | pass | pass | pass | fail | fail | QLoRA路径与显存条件可审查，但审核轨迹仍有1个未决must-fix |
| case_019 | KB-12 / KBS-12-04 | pass | pass | pass | fail | fail | QLoRA阶段与12GB显存方向一致；硬件/风格字段缺失 |
| case_020 | KB-04 / KBS-04-04,05 | fail | pass | pass | fail | fail | 已有训练基础却回到Dataset/优化器，未进入LoRA合并/vLLM目标 |
| case_021 | KB-12 / KBS-12-05 | fail | fail | pass | fail | fail | 未完成LoRA配置/Checkpoint前置即安排merge_and_unload |
| case_022 | KB-04 / KBS-04-06,07 | fail | pass | pass | fail | pass | 目标是LoRA rank/alpha却安排通用训练循环；混合偏好体现较好 |
| case_027 | KB-15 / KBS-15-00,01 | pass | pass | pass | fail | fail | vLLM环境与服务基础合理；硬件/风格字段缺失 |
| case_028 | KB-15 / KBS-15-02 | pass | pass | pass | fail | fail | vLLM服务配置合理；硬件/风格字段缺失 |
| case_029 | KB-08 / KBS-08-06 | fail | pass | pass | fail | fail | 仅有前置基础却进入GPT-2前向传播链路 |
| case_030 | KB-15 / KBS-15-03 | pass | pass | pass | fail | fail | vLLM部署阶段合理；硬件/风格字段缺失 |
| case_031 | KB-15 / KBS-15-03,04 | pass | pass | pass | fail | fail | 高配置部署阶段合理；硬件/风格字段缺失 |
| case_034 | KB-08 / KBS-08-07 | fail | fail | pass | fail | fail | 仅Python基础却直接做GPT-2前向传播实验 |

### 3.1 普通阶段统计

| 条件 | pass | fail | error | 通过率 |
|---|---:|---:|---:|---:|
| target_path | 16 | 12 | 0 | 57.14% |
| difficulty | 22 | 6 | 0 | 78.57% |
| time_budget | 28 | 0 | 0 | 100.00% |
| hardware | 0 | 28 | 0 | 0.00% |
| learning_style | 4 | 24 | 0 | 14.29% |
| 合计 | 70 | 70 | 0 | 50.00% |

正式普通阶段适配率为 `70 / 140 = 50.00%`，低于目标 `>=85%`。因此当前 50 组真实输出不能证明达到学习者画像-资源难度适配指标。

## 4. 空阶段单独检查（22组）

空阶段案例：`case_023`、`case_024`、`case_025`、`case_026`、`case_032`、`case_033`、`case_035` 至 `case_050`（共22组）。

所有22组均出现同一类流程错误：

- `stage_planning_output.json` 的 `subtopic_ids=[]`、`estimated_hours=0`、标题为“已完成全部知识点”；
- 画像仍有未掌握 `knowledge_gaps`（1或2项），并非无待学知识点；
- `learning_package.md` 仍存在（约3.3万至10.1万字）；
- `assessment_questions.json` 仍生成8道题；
- `stage_decision` 未统一返回正确的终止状态。`case_032`、`case_033`、`case_037` 虽返回 `completed/all_mastered=true`，但与输入画像中未掌握知识点及“检验0道题、覆盖1个知识点”的矛盾仍在；其余案例返回 `continue` 或 `repeat`。

因此22/22组空阶段检查判为 `fail`。这些失败不计入上述140项普通适配率，但说明终止分支尚未满足“无待学点、不生成普通包、不抽题、正确结业”的流程契约。

## 5. 系统性问题

1. **硬件画像完全丢失**：50/50 个 `learner_profile.json` 没有 `hardware` 字段，50/50 个 `stage_planning_output.json` 的 `hardware_notes` 为空。即使学习包里出现GPU、CPU或云端字样，也不能证明它针对该案例硬件作出了规划，因此普通阶段硬件项全部 fail。
2. **学习风格被统一化**：50/50 个画像的 `learning_style` 都是 `hands-on`，与输入画像的理论型、混合型、工程实操型等不一致；普通阶段规划的 `learning_style_notes` 基本都是“混合型学习者”的固定表述。只有4组输入本身为混合/实践组合且学习包组织与之相符，判为 pass。
3. **路径与难度存在前置跳跃**：失败集中在无Python/PyTorch基础直接进入Transformer、datasets、GPT-2或LoRA合并等阶段，或已有目标基础却回退到不相关板块。
4. **审核轨迹不是“已批准”证明**：50/50 组 `review_passed=true`，但全部 `package_status=provisional`、`needs_human_review=true`；其中 `case_018`、`case_043`、`case_046`、`case_047`、`case_049` 仍有未决 must-fix。它只能证明审核链路运行过，不能替代内容适配结论。
5. **人工审核表口径冲突**：当前 `personalization_review.csv` 共250行，其中普通阶段70 pass、70 fail，空阶段110行标为 `not_applicable`；统计脚本明确支持该排除值并按140项计算50%。手册正文要求 verdict 只能是三种值，与脚本和“空阶段不入分母”的最新规则不一致，应在最终交付说明中固定这一口径。
6. **事实核验尚未完成**：`fact_check_review.csv` 的150行仍为 `not_reviewed`，不能据此声称专业知识错误率 `<5%`；本报告只结论化85%适配指标。

## 6. 结论与修复优先级

当前结论：**不通过**。普通阶段画像-资源适配率为50.00%，且22组空阶段全部错误生成学习包和题目。

建议按以下顺序修复后重新运行同一套评测：

1. 修复画像字段透传，至少保留 `hardware`、原始 `learning_style`，并让 `hardware_notes`、`learning_style_notes` 使用案例值而不是统一模板。
2. 修复空阶段守卫：存在未掌握知识点时不得返回空阶段；真正空阶段必须跳过学习包和抽题，并返回明确结业/结束状态。
3. 修复硬前置闭包和目标路径选择，禁止无前置补偿的高阶跳跃及已掌握板块重复安排。
4. 重新生成50组输出并重新填写人工适配证据；不要直接修改自动统计结果。
5. 另行完成150条事实核验后，才可报告专业知识错误率。
