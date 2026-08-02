# SKILL名称：pedagogy

## 1. 能力描述
- 功能：为Agent⑤提供**教学法原理库**，定义"好的教学长什么样"——脚手架原则、渐进式讲解规范、认知负荷理论、建构主义要求。Agent⑤调用本SKILL评估学习内容的教学逻辑是否科学，输出教学严谨度评分（pedagogy_score）和问题清单（issues）。在 `HybridQualityReviewAgent` 中被加载为 `skill_context`，作为LLM审核教学质量的理论依据
- 适用场景（3个）：
  1. **审核学习包结构**：评估Agent3生成的tutorial是否遵循"已知回顾→核心概念→实操演示→常见坑点→自测"五段逻辑，缺少任一环节扣分
  2. **评估认知负荷**：计算内容的新概念密度，判断是否超出工作记忆容量限制（5±2个），超载则建议拆分阶段
  3. **检查建构主义元素与情绪友好度**：验证内容是否提供真实情境案例让学习者在解决问题中建构知识；同时评估内容的鼓励性和挫败感控制，确保情绪友好
- 前置依赖：
  - 待审核内容（`ReviewInput.content`，tutorial或answer文本）
  - 画像中的用户水平（`ReviewInput.profile`，含 `skill_tags`、`knowledge_gaps`、`learning_style`，用于判断"已知"基线和难度适配）
  - 当前学习阶段信息（`ReviewInput.stage.block_name`）

## 2. 核心方法论
- **步骤1：检查内容是否从"已知回顾"开始**——扫描内容开头300字，检测是否存在回顾用户已掌握技能的段落（关键词：`在前面的学习中`、`你已经掌握了`、`回顾`、`前置要求`、`prerequisite`）。若 `profile.skill_tags` 非空但内容开头无链接性回顾，标记 issue_type=`"missing_review"`
- **步骤2：检查新概念的引入节奏（每步不超过2个）**——按段落切分内容，统计每段落首次出现的技术术语数量。若单一段落新概念>2个，标记 issue_type=`"concept_overload_in_paragraph"`，location指向该段落
- **步骤3：检查是否有情境化案例**——检测内容是否包含真实应用场景描述（关键词：`场景`、`案例`、`应用`、`实际项目`、`例如`、`case`）和可运行代码示例（```python围栏）。若仅有理论无案例，标记 issue_type=`"missing_context_case"`
- **步骤4：量化认知负荷**——计算三个指标：①新概念密度（首次出现的技术术语数/内容字数×1000）；②总篇幅（字数）；③复杂度（嵌套层级深度+公式数量）。综合输出 cognitive_load_score（0-100，合理度，越高越好）和 cognitive_load_level（0-100，负荷水平，越高越糟，= 100 - cognitive_load_score），超载则标记 issue_type=`"cognitive_load"`
- **步骤5：评估情绪友好度**——应用规则5，统计内容的正面情绪语句占比（positive_ratio）和是否含挫败性表达（has_discouraging_words），输出 emotion_friendliness_score（0-100）。含"这很简单""你应该懂"等居高临下表达扣分
- **步骤6（可选）：评估代码修复教学质量**——当内容为代码修复建议时（`mode == "code_error_analysis"` 或内容含代码修复标记），应用规则6，依次检查错误定位精准度、引导性解释充分性、完整答案完整性三项原则，输出 code_fix_pedagogy_score 和 code_fix_issues

## 3. 知识/规则定义

### 规则1：脚手架原则（Scaffolding）
- **要求**：新内容必须建立在"已知→未知"的递进逻辑上，每步最多引入1~2个新概念
- **判断方式**：
  - 检测内容是否存在"前置要求"章节（正则：`##\s*(前置要求| prerequisites|先修知识|预备知识)`）
  - 比对 `profile.skill_tags` 与内容首段提及的已知技能，若无交集且无基础扫盲段落，标记"跨度太大"
  - 段落级检查：每个新段落引入的新概念≤2个，否则标记 issue_type=`"concept_overload_in_paragraph"`
- **违规处置**：issue_type=`"missing_review"` 或 `"concept_overload"`，severity=major，location指向缺失段落或超载段落

### 规则2：渐进式讲解规范
- **理想顺序**：已知回顾 → 核心概念拆解 → 实操演示 → 常见坑点 → 自测
- **检查逻辑**：按上述五段顺序扫描内容，每段检测关键词：
  - 已知回顾：`回顾`、`前置`、`你已经`、`前面的学习`
  - 核心概念拆解：`核心概念`、`原理`、`是什么`、`定义`
  - 实操演示：`示例`、`代码`、`实操`、`动手`、`实践`
  - 常见坑点：`坑点`、`常见错误`、`注意`、`避坑`、`误区`
  - 自测：`练习`、`测试`、`自测`、`题目`、`验证`
- **扣分规则**：缺少任一环节扣20分（pedagogy_score），缺少2个以上判major
- **违规处置**：issue_type=`"missing_section"`，severity=major（缺2段）或minor（缺1段）

### 规则3：认知负荷理论（Cognitive Load Theory）
- **要求**：单次学习新概念不超过5±2个（即3~7个），超载则需拆分
- **理论依据**：Sweller（1988）《Cognitive Load During Problem Solving》提出工作记忆容量限制为7±2个信息块（基于Miller 1956的魔数），内在负荷+外在负荷+相关负荷总和不能超过工作记忆容量，否则导致认知过载
- **计算方式**：统计内容中首次出现的技术术语数量（基于术语词典+大写首字母启发式），新概念密度=新概念数/内容字数×1000
- **阈值**：
  - 3~5个/千字：负荷适中（cognitive_load_score 90~100）
  - 6~7个/千字：偶有密集（70~89）
  - 8~10个/千字：部分超载（50~69）
  - >10个/千字：严重超载（0~49），触发 issue_type=`"cognitive_load"`，severity=major
- **违规处置**：超载时建议拆分内容为多个阶段或提供辅助材料

### 规则4：建构主义要求（Constructivism）
- **要求**：内容需提供真实情境案例，让学习者在问题中建构知识，否则标记"缺乏情境化"
- **检查逻辑**：检测内容是否包含至少1个真实应用场景描述（非抽象理论），形式可为：
  - 真实业务场景描述（如"在推荐系统场景中..."）
  - 可运行代码示例（```python围栏且含完整逻辑）
  - 案例分析（含问题→分析→解决的叙事结构）
- **违规处置**：issue_type=`"missing_context_case"`，severity=major，suggestion建议补充真实场景案例

### 规则5：情绪友好度评估（Emotional Friendliness）
- **要求**：内容应具有鼓励性和亲和力，避免居高临下或挫败性表达，让学习者保持学习动力
- **检查逻辑**：
  - 统计正面情绪语句占比（positive_ratio）：扫描内容中含鼓励性表达的语句数 / 总语句数。鼓励性关键词：`别担心`、`一步步来`、`你能做到`、`没关系`、`我们一起来`、`这很正常`、`刚开始都这样`
  - 检测挫败性表达（has_discouraging_words）：扫描是否含居高临下或否定性表达。挫败性关键词：`这很简单`、`你应该懂`、`这都不懂`、`很简单`、`怎么连...都不会`、`这还用说`
- **评分计算**：
  ```
  emotion_friendliness_score = round(positive_ratio × 100)
  若 has_discouraging_words == true:
      emotion_friendliness_score = max(0, emotion_friendliness_score - 20)  # 挫败性表达扣20分
  ```
- **阈值**：
  - score ≥ 80：情绪友好度高（鼓励性语句丰富，无挫败表达）
  - 60 ~ 79：情绪友好度中等（有一定鼓励但不够，或偶有轻微挫败表达）
  - < 60：情绪友好度低（缺乏鼓励或含明显挫败表达），标记 issue_type=`"poor_emotion_friendliness"`，severity=minor，suggestion建议增加鼓励性语句
- **违规处置**：含挫败性表达时 issue_type=`"discouraging_words"`，severity=major，suggestion="删除'这很简单'等居高临下表达，改为'这个概念一开始看起来复杂，我们一步步来'"

### 规则6：代码修复教学原则（Code Fix Pedagogy）
- **适用场景**：当审核内容为"代码错误修复建议"时（即内容包含问题代码分析 + 修复方案，或 Agent3 基于 Agent4 的代码错误分析生成的修复内容），触发本规则评估
- **三大核心原则（必须同时满足）**：
  1. **错误定位精准**——必须明确指出错误发生的位置（行号、函数名、变量名）和错误类型，让用户清楚"哪里错了、错在哪类"。格式要求：先给出错误定位，再展开分析。判定标准：内容中是否包含具体的位置标识（如"第5行"、"函数 `foo()` 中"、"变量 `data`"）和错误类型名称（如"KeyError"、"语法错误"）。缺失任一项判定为违规
  2. **引导性解释充分**——必须在给出完整修复代码之前，先用自然语言解释错误的根本原因和修复思路，引导用户思考。要求包含：错误根因分析（为什么会错）+ 修复思路说明（怎么想到要这么改）。引导性文字的长度应 ≥ 修复代码长度的 30%。判定标准：检查是否存在独立的"原因分析"或"修复思路"段落，且该段落在完整修复代码之前
  3. **完整答案必给**——必须提供完整的、可直接运行的修复代码，不能只给提示不给答案。修复代码需满足：a) 标注关键修改点（用 `# ← 修改点：xxx` 或 diff 格式标注）；b) 关键修改点数量 ≤ 3 处时全部标注，> 3 处时标注前 3 个最重要的；c) 修复代码与原代码的上下文保持一致，可独立运行。判定标准：检查内容中是否存在完整代码块，且代码块内包含至少 1 处修改点标注
- **违规等级判定**：

| 违反原则 | issue_type | severity | 扣分说明 |
|:---|:---|:---|:---|
| 缺少错误定位（原则1） | `poor_code_fix_no_location` | major | 扣 20 分 |
| 缺少引导性解释（原则2） | `poor_code_fix_no_guidance` | major | 扣 15 分 |
| 缺少完整答案（原则3） | `poor_code_fix_no_answer` | critical | 扣 30 分 |
| 缺少修改点标注（原则3子项） | `poor_code_fix_no_annotation` | minor | 扣 10 分 |

- **输出要求**：`code_fix_pedagogy_score`（0~100，越高越符合教学原则），`code_fix_issues`（违反的原则列表）
- **触发条件**：当 `ReviewInput.mode == "code_error_analysis"` 或内容中包含代码修复建议标记时，自动启用本规则评估

## 4. 输入输出规范

### 输入
- `learning_package` 内容（tutorial文本或answer文本，通过 `ReviewInput.content` 传入）
- `user_profile`（通过 `ReviewInput.profile` 传入，含 `skill_tags`、`knowledge_gaps`、`learning_style`、`goal`）

### 输出（数据流向quality_metrics）
**本SKILL输出的核心数据将直接提供给 `quality_metrics` 用于加权评分**：
- `cognitive_load_score` → 对应 `quality_metrics` 的 **认知负荷合理度** 维度（权重25%）
- `cognitive_load_level` → 供 `quality_metrics` 的 **红线检查** 使用（>85触发认知负荷红线）
- `pedagogy_score` → 对应 `quality_metrics` 的 **教学逻辑严谨度** 维度（权重20%）
- `emotion_friendliness_score` → 对应 `quality_metrics` 的 **情绪友好度** 维度（权重15%）
- `code_fix_pedagogy_score` → 代码修复场景专用，供 `quality_metrics` 作为附加评分维度（仅当 `mode == "code_error_analysis"` 时生效）
- `issues` 数组 → 合并到 `QualityReviewResult.errors`（用于附加条件检查）
- `code_fix_issues` 数组 → 代码修复场景专用，合并到 `QualityReviewResult.errors`（仅代码修复场景）

LLM返回JSON格式：
```json
{
  "pedagogy_score": 65,
  "cognitive_load_score": 82,
  "cognitive_load_level": 18,
  "emotion_friendliness_score": 60,
  "issues": [
    {
      "issue_type": "missing_review",
      "severity": "major",
      "location": "内容开头（前300字）",
      "suggestion": "在开头增加：'在前面的学习中，你已经掌握了PyTorch的张量操作。现在我们将在此基础上学习LoRA微调...'"
    },
    {
      "issue_type": "missing_section",
      "severity": "major",
      "location": "内容末尾",
      "suggestion": "补充'常见坑点'段落（如秩r过大导致过拟合）和'自测'段落（1~2道测试题）"
    },
    {
      "issue_type": "missing_context_case",
      "severity": "major",
      "location": "核心概念段落",
      "suggestion": "补充真实应用场景，如'在推荐系统微调场景中，LoRA可在单卡GPU上完成亿级参数模型的领域适配...'"
    },
    {
      "issue_type": "discouraging_words",
      "severity": "major",
      "location": "核心概念段落",
      "suggestion": "删除'这很简单'等居高临下表达，改为'这个概念一开始看起来复杂，我们一步步来'"
    }
  ],
  "summary": "内容技术准确，但教学结构不完整：缺少已知回顾、常见坑点、自测段落，缺乏情境化案例，且含挫败性表达。pedagogy_score=65低于70分及格线。"
}
```

**字段说明**：
- `pedagogy_score`：0-100，教学严谨度评分（五段式完整性+脚手架原则+建构主义综合）
- `cognitive_load_score`：0-100，认知负荷合理度评分（新概念密度映射，越高越合理）
- `cognitive_load_level`：0-100，认知负荷水平（= 100 - cognitive_load_score，越高越糟，供红线检查）
- `emotion_friendliness_score`：0-100，情绪友好度评分（鼓励性语句占比，含挫败表达扣分）
- `issues`：问题列表，每项含 `issue_type`（缺失类型）、`severity`（critical/major/minor）、`location`（位置）、`suggestion`（修复建议）
- `summary`：人类可读的审核总结

**与QualityReviewResult的映射**：issues数组每项映射为ReviewError（`error_type` 取 issue_type，`description` 取 location+suggestion），pedagogy_score和cognitive_load_score合并到 `QualityReviewResult.scores` 字典（键 `"pedagogy"`、`"cognitive_load"`）。

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 内容过短（`content.strip()` 长度<500字） | 标记 issue_type=`"too_short"`，severity=major，suggestion="内容过简，建议补充讲解至500字以上，覆盖五段式结构"，pedagogy_score上限封顶50 |
| 内容过短（<40字符） | `DeterministicQualityReviewAgent` 直接返回 error_type=`"too_short"`，passed=false，不执行LLM审核 |
| 无法提取结构（内容无明确段落分隔，缺少 `##` 标题） | 标记 issue_type=`"unclear_structure"`，severity=major，suggestion="结构不清晰，建议使用Markdown二级标题划分五段式结构"，pedagogy_score上限封顶60 |
| 用户画像skill_tags为空 | 跳过"已知回顾"匹配检查，但仍要求内容开头有引导性回顾段落，标记 issue_type=`"missing_review"`，severity降级为minor |
| 新概念术语词典未覆盖（无法识别新概念） | 启发式统计（大写首字母+英文单词+中文术语后缀如"矩阵/系数/函数"），cognitive_load_score标注"估算值"，建议人工复核 |
| LLM审核结果无法解析 | 添加 ReviewError(error_type=`"review_system_error"`, severity=critical, description="Agent4统一教学质量审核结果无法解析")，passed=false |
| 内容审核循环达上限（Agent4统一审核循环≥3次仍未通过） | Orchestrator强制终止循环，status=`"review_needed"`，summary标注"审核循环达3次上限，建议人工介入修订内容" |

## 6. 附录

### 附录A：教学法理论参考
| 编号 | 理论名称 | 作者 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | 最近发展区理论（Zone of Proximal Development, ZPD） | Lev Vygotsky | 1978 | 学习内容应处于学习者"已知但未掌握"的最近发展区，难度略高于当前水平（i+1原则），教师/同伴支持可帮助跨越 | 规则1脚手架原则的理论基础——内容必须从用户 `skill_tags`（已知）逐步过渡到 `knowledge_gaps`（未知），难度差异不超过2个层级 |
| A-2 | 认知负荷理论（Cognitive Load Theory） | John Sweller | 1988 | 工作记忆容量有限（7±2个信息块），内在负荷（内容本身复杂度）+外在负荷（呈现方式导致）+相关负荷（图式建构所需）总和不能超限，超限导致认知过载 | 规则3认知负荷理论的直接依据——新概念密度阈值5±2个/千字来源于Sweller的工作记忆容量限制，超载需拆分阶段 |
| A-3 | 脚手架理论（Scaffolding Theory） | Jerome Bruner | 1976 | 教师提供临时支持（脚手架），帮助学习者完成独立无法完成的任务，随着学习者能力提升逐步撤除脚手架 | 规则1脚手架原则的实施依据——每步引入1~2个新概念，提供前置知识回顾作为脚手架，避免学习者掉队 |
| A-4 | 建构主义学习理论（Constructivism） | Jean Piaget / Lev Vygotsky | 1950s | 学习者在真实情境中主动建构知识，而非被动接受；知识通过与社会环境和问题的交互而内化 | 规则4建构主义要求的依据——内容必须提供真实情境案例，让学习者在解决问题中建构知识 |
| A-5 | 掌握学习理论（Mastery Learning） | Benjamin Bloom | 1968 | 学习者需达到前一阶段的掌握标准（80%正确率）后方可进入下一阶段，否则需补偿性学习 | Agent⑥策略选择的依据（accuracy≥0.8方可进阶），与本SKILL的"渐进式讲解"配合确保每阶段达标 |

### 附录B：好的教学结构示例 vs 差的教学结构示例

#### 好的教学结构示例（pedagogy_score=92）
```markdown
# LoRA微调入门

## 已知回顾
在前面的学习中，你已经掌握了PyTorch的张量操作和Transformer的注意力机制。
今天我们将在此基础上学习LoRA——一种参数高效的微调方法。

## 核心概念拆解
LoRA（Low-Rank Adaptation）通过冻结原始权重W，仅训练低秩矩阵A和B来实现参数更新。
增量矩阵 ΔW = (α/r) × B × A，其中r是秩，α是缩放系数。
- 冻结W：不更新原模型参数，降低显存占用
- 训练A和B：两个低秩矩阵，参数量仅为r×(d_in+d_out)

## 实操演示
在推荐系统微调场景中，我们用LoRA对Llama-2-7B进行领域适配：
```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
```

## 常见坑点
- 秩r过大（如r=64）会导致过拟合，建议从r=8开始调参
- target_modules选择不当会降低效果，建议优先选q_proj和v_proj

## 自测
1. LoRA通过训练哪两个矩阵实现参数更新？
2. 假设d_in=d_out=4096，r=8，计算LoRA参数量与原参数量的比值。
```

**评分解析**：
- 已知回顾✓（链接PyTorch和Transformer）
- 核心概念拆解✓（含2个新概念：LoRA、低秩矩阵）
- 实操演示✓（真实场景+可运行代码）
- 常见坑点✓（2个典型坑点）
- 自测✓（2道测试题）
- 新概念密度：4个/约400字，负荷适中
- pedagogy_score=92（五段完整+情境化案例+负荷合理）

#### 差的教学结构示例（pedagogy_score=42）
```markdown
# LoRA微调

LoRA是一种微调方法。它使用SVD分解原权重W，训练U和V矩阵。
原理：W = UΣV，通过训练U和V降低参数量。
公式：ΔW = BA，其中A是d×r，B是r×d。

参数计算：原模型10万参数，LoRA参数50万，显著减少。
```

**问题清单（issues）**：
| issue_type | severity | location | suggestion |
|:---|:---|:---|:---|
| missing_review | major | 内容开头 | 缺少已知回顾段落，未链接用户已掌握的PyTorch知识。建议增加引导性回顾 |
| missing_section | major | 内容整体 | 缺少实操演示、常见坑点、自测段落（仅核心概念一段，五段缺三段） |
| missing_context_case | major | 全文 | 仅有抽象理论，无真实应用场景案例。建议补充推荐系统/对话系统等场景示例 |
| concept_overload_in_paragraph | minor | "原理"段落 | 单段落引入SVD、U、V、Σ、ΔW、BA共6个新概念，超过每段2个上限。建议拆分为多段 |
| cognitive_load | major | 全文 | 新概念密度8个/约150字，超过7个/千字上限，认知负荷超载。建议拆分阶段或补充辅助说明 |

**评分解析**：
- pedagogy_score=42（五段缺三段+无情境化+认知超载）
- cognitive_load_score=35（严重超载）
- 判定：passed=false，存在4个major问题

## 7. 版本信息
- 版本：v3.2，更新日期：2026-07-25
- 变更记录：能力描述明确3个适用场景；核心方法论重构为4步（已知回顾→节奏检查→情境化案例→量化认知负荷）；规则1新增"前置要求"章节检测和段落级新概念检查；规则3补充Sweller(1988)理论依据和工作记忆容量来源；输入输出规范新增pedagogy_score和issues列表完整JSON示例；异常处理新增内容过简（<500字）和结构不清晰场景；附录A补充5个教学法理论；附录B新增好的教学结构vs差的教学结构完整对比示例。**v3.1变更**：新增规则5情绪友好度评估（鼓励性语句占比+挫败性表达检测），输出新增 emotion_friendliness_score（对应quality_metrics情绪友好度维度15%）和 cognitive_load_level（供红线检查）；消除quality_metrics对不存在的"情绪分析工具"的依赖；新增"审核循环≥3次"异常处置路径。**v3.2变更**：新增规则6代码修复教学原则（三大核心原则：错误定位精准、引导性解释充分、完整答案必给）；核心方法论新增步骤6（可选）评估代码修复教学质量；输出新增 code_fix_pedagogy_score 和 code_fix_issues；适配 Agent4 代码错误分析新功能，确保代码修复建议遵循教学法原则
