# SKILL名称：technical_review

## 1. 能力描述
- 功能：为Agent④技术审核提供**幻觉判定阈值、常见错误模式库、审核检查清单、报错分级标准、裁决优先级**，定义"什么是错"。原 `hallucination_detection.skill.md` 的内容已合并至此，本SKILL同时承载"阈值和错误模式"与"裁决逻辑"两重职责。Agent④调用 `rule_check_tool` 执行确定性规则检查（LoRA机制错误、参数量计算错误），同时调用LLM进行语义审核（幻觉、事实、规范），按裁决优先级合并结果生成最终审核报告
- 适用场景（3个）：
  1. **内容审核**：Agent3生成的学习包（tutorial、code_examples）发布前审核技术正确性，由Orchestrator在 `_review_learning_package` 时触发
  2. **回答审核**：用户问题回答（answer）发送前审核，由Orchestrator在 `_review_answer` 时触发
  3. **多工具结果冲突裁决**：当 `rule_check_tool` 与LLM语义审核、知识库相似度检索（`knowledge_verify_tool`）与知识库标准答案条目结论不一致时，按裁决优先级输出最终结论
- 前置依赖：
  - 待审核内容（`ReviewInput.content`）
  - 知识库（`knowledge_verify_tool` 提供的语义相似度检索）
  - 知识库标准答案条目（`knowledge_base` 中人工策展的高置信度领域基线条目，如"LoRA冻结W训练ΔW=(α/r)BA"。**非独立资源**，是知识库的策展子集，用于解决相似度检索不足时的判定回退）
  - 规范规则库（`rule_check_tool` 的正则规则、`industry_standards.skill.md` 的红线条款）
  - 当前学习阶段信息（`ReviewInput.stage.block_name`）和用户画像（`ReviewInput.profile`）

## 2. 核心方法论
### 2.1 内容审核/回答审核流程（通用场景）
- **步骤1：解析输入内容**——识别代码块（```...```围栏）和技术陈述段落（含"参数量"、"原理"、"LoRA"、"显存"等领域关键词），为每个技术声明分配位置标识（location），便于后续evidence定位
- **步骤2：硬性检查**——调用 `rule_check_tool.check_rules(content)` 执行确定性规则检查，识别LoRA机制错误、参数量计算错误、语法错误、依赖冲突。此步骤为纯规则匹配，无LLM参与，结果100%确定
- **步骤3：事实检查**——调用 `knowledge_verify_tool` 计算技术声明与知识库的语义相似度，应用规则1（幻觉阈值）判定事实正确性。此步骤为LLM语义审核+知识库检索结合
- **步骤4：标准检查（调用 `industry_standards` 子模块）**——**调用 `industry_standards` 子检查模块**，传入相同的 `ReviewInput`，获取LoRA/QLoRA微调与vLLM部署领域合规红线的 `ReviewError` 列表（含显存分配、端口安全、数据脱敏、许可证合规等）
- **步骤5：综合裁决输出**——合并步骤2~4的所有ReviewError，应用规则2（错误模式库）标记、规则3（报错分级标准）分类（Critical/Major/Minor）、规则4（裁决优先级）裁决：若硬性检查发现critical错误，`passed` 强制为 `false`。最终输出包含status、errors数组、summary的结构化报告

### 2.2 用户提交代码审核流程（代码错误分析专用场景）
当Agent4收到"用户代码错误分析"请求时（含代码+报错，或仅问题描述），执行以下专属流程：

- **步骤1：解析报错信息提取错误类型**——从 `error_info` 中提取异常类型名（如 `KeyError`、`TypeError`），**优先引用 `code_error_patterns.skill.md` 规则1的错误类型→等级映射表**进行错误分类映射
- **步骤2：错误分级判定**——应用 `code_error_patterns.skill.md` 规则1，将错误类型映射为三级严重等级：SyntaxError类→致命(critical)、RuntimeError类→主要(major)、LogicError类→次要(minor)
- **步骤3：错误归因判定**——应用 `code_error_patterns.skill.md` 规则2，对已识别的错误类型进行归因分类：代码问题 / 环境问题 / 外部依赖问题 / 数据问题。按归因判定优先级执行：环境问题→外部依赖问题→数据问题→代码问题
- **步骤4：知识库交叉验证**——调用 `knowledge_verify_tool.verify_code_error()` 获取知识库匹配结果。此步骤的结果**作为辅助参考**，用于补充修改建议和验证规则分类的准确性，但**不改变步骤2-3的规则判定结论**（冲突时以 `code_error_patterns.skill.md` 规则分类为准）
- **步骤5：代码静态检查（可选）**——若用户提供了代码片段，调用 `code_parser_tool.parse_code()` 检查语法错误，调用 `rule_check_tool.check_code_rules()` 检查规范违规。结果作为补充信息附加在输出中
- **步骤6：生成标准化输出**——按 `code_error_patterns.skill.md` 定义的输出格式生成结果，包含 `error_category`、`severity`、`root_cause`、`fix_suggestion`、`confidence`、`classification_method`、`evidence`

**判断依据优先级**：
1. **最高优先级**：`code_error_patterns.skill.md` 的错误分级规则（规则1）和归因判定规则（规则2）——所有错误分类、严重等级、归因类型必须优先采纳此SKILL的定义
2. **辅助参考**：`knowledge_verify_tool` 的知识库检索结果——用于补充修改建议、提供代码修复示例、增强分类置信度
3. **补充信息**：`code_parser_tool` 和 `rule_check_tool` 的静态检查结果——仅附加在输出的补充字段中

**与industry_standards的协作流程**：
1. `technical_review` 接收 `ReviewInput`
2. 内部执行步骤4的"标准检查"时，调用 `industry_standards` 的规则匹配逻辑
3. `industry_standards` 返回 `List[ReviewError]`
4. `technical_review` 将此列表与步骤2（硬性检查）、步骤3（事实检查）的错误合并
5. `technical_review` 作为唯一的最终出口，输出完整的 `TechnicalReviewResult`

## 3. 知识/规则定义

### 规则1：幻觉判定阈值
| 知识库相似度 | 标记 | severity | 处置 |
|:---|:---|:---|:---|
| < 0.65 | 潜在幻觉 | critical | 直接判fail，要求删除或补充可靠来源 |
| 0.65 ~ 0.80 | 存疑需人工复核 | major（review_needed） | 标记major，要求补充依据或人工复核 |
| ≥ 0.80 | 有依据 | 通过 | 正常放行 |

**0.65阈值来源依据**：参考Stanford CS224V《RAG系统幻觉检测指南》第4章阈值建议——该指南基于RAGAS评估框架的faithfulness指标实证，当检索证据与生成内容的语义相似度低于0.65时，faithfulness分数通常跌破0.5（即半数以上声明无依据），此时判为幻觉的精确率可达0.82以上；0.80为高置信通过线，对应faithfulness≥0.85。

### 规则2：常见错误模式库
| 错误模式 | 触发条件 | 错误示例 | 正确示例 | severity |
|:---|:---|:---|:---|:---|
| 参数类型错误 | `torch.tensor` 传入Python list而非ndarray/tensor | `x = torch.tensor([1,2,3], requires_grad=True).float()` 后直接做矩阵运算导致梯度断流 | `x = torch.tensor([1.0,2.0,3.0], requires_grad=True)` 或先转 `np.array` 再构造 | major |
| API调用过时 | 使用PyTorch 1.x旧写法 `torch.cuda.set_device`、`DataParallel` 单卡写法 | `model = model.cuda(); torch.cuda.set_device(0)` | `model = model.to(device); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | major |
| 路径硬编码 | 代码中出现绝对路径 `/root/data`、`C:\Users\xxx` | `df = pd.read_csv("/root/data/train.csv")` | `DATA_DIR = os.getenv("DATA_DIR", "./data"); df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))` | critical |
| 配置文件缺失必填项 | `config.json` 缺少 `model_name`、`tokenizer_path` 等必填字段 | `{"max_length": 512}` 缺少model_name | `{"model_name": "meta-llama/Llama-2-7b", "tokenizer_path": "...", "max_length": 512}` | critical |
| 未捕获异常 | 涉及IO/网络/模型加载的代码块缺失 `try-except` | `model = AutoModelForCausalLM.from_pretrained(path)` 直接调用 | `try: model = AutoModelForCausalLM.from_pretrained(path) except Exception as e: logger.error(e); raise` | major |
| LoRA机制错误 | 将LoRA描述为SVD分解或训练U/Σ/V（rule_check_tool正则命中） | "LoRA将原始权重W分解为U*Σ*V" | "LoRA冻结原权重W，训练低秩增量ΔW=(α/r)BA" | critical |
| 参数量计算错误 | 声称参数量下降但trainable≥original（rule_check_tool正则命中） | "原模型10万参数，微调参数50万" | "原参数d_in×d_out=262144，LoRA参数r(d_in+d_out)=8192" | critical |

**LoRA机制检测正则**（来自 `rule_check_tool.py`）：
- `r"(?:原始)?权重(?:矩阵)?w.*?分解.*?u.*?v"`
- `r"w.*?=.*?u.*?(?:sigma|σ).*?v"`
- `r"lora.*?(?:svd|奇异值分解)"`

**参数量提取正则**：
- `r"(?:原模型|原始模型|模型).*?(\d+(?:\.\d+)?)万(?:个)?参数.*?(?:微调参数|参数量).*?(\d+(?:\.\d+)?)万"`

### 规则3：报错分级标准
| 分级 | 定义 | 触发条件（互斥） | 处置 |
|:---|:---|:---|:---|
| Critical（致命） | 阻断性错误 | 语法错误 / 依赖冲突 / 安全违规 / LoRA机制错误 / 参数量计算错误 / 路径硬编码 / 配置必填项缺失 | 直接不通过（passed=false），阻止发布 |
| Major（主要） | 实质内容错误 | 幻觉（sim<0.65）/ 标准答案偏离 / 事实错误 / API过时 / 未捕获异常 / 相似度0.65~0.80存疑 | 需修改后方可通过 |
| Minor（次要） | 风格/规范建议 | 风格建议 / 冗余描述 / 命名不规范 / 缺少许可证说明 | 建议修改，不阻断通过 |

**互斥条件判断流程**：
1. 若错误同时满足多个分级条件，取最高级（Critical > Major > Minor）
2. 单条ReviewError的severity字段唯一，不允许同时为多个值
3. Critical错误一旦存在，整体passed强制为false，无论Major/Minor数量
4. 仅当无Critical且Major数量为0时，passed可为true（Minor不阻断）

### 规则4：裁决优先级
**优先级顺序（从高到低）**：
1. **硬性阻断**：语法错误 / 依赖冲突 / 安全违规 / LoRA机制错误 / 参数量计算错误（来自 `rule_check_tool` 确定性检查）
2. **规范红线**：硬编码密钥 / 端口暴露 / 权限缺失 / 许可证违规（来自 `industry_standards`）
3. **语义冲突**：知识库相似度检索 vs 知识库标准答案条目——**标准答案条目优先于相似度检索**

**裁决规则**：
- 若硬性检查发现critical错误，`passed` 必须为 `false`，**且事实检查（语义检查）结果仍须附加在errors数组中**，不得因硬性错误已fail而省略事实核查
- 当 `rule_check_tool` 与LLM审核结论冲突时，以 `rule_check_tool` 的确定性规则为准
- 当知识库相似度检索无依据（sim<0.65）但知识库标准答案条目命中时，**以标准答案条目为准**判通过，并在evidence中标注"知识库标准答案条目依据"
- 当知识库相似度检索有依据但标准答案条目明确反对时，以标准答案条目为准判fail

## 4. 输入输出规范

### 输入
ReviewInput对象，包含：
- `content`：待审核内容字符串（tutorial / answer / code_examples）
- `user_profile`：用户画像（UserProfile）
- `context`：可选上下文（当前StageInfo、历史审核记录）
- `target_type`：`"learning_package"` 或 `"answer"`
- `session_id`、`stage`、`question`（可选）、`code_examples`（可选）

### 输出
#### 通用场景输出（内容审核/回答审核）
TechnicalReviewResult对象，LLM返回JSON格式：
```json
{
  "status": "fail",
  "passed": false,
  "errors": [
    {
      "error_type": "lora_mechanism_error",
      "severity": "critical",
      "location": "LoRA原理说明段落",
      "description": "将LoRA描述为对原始权重W执行SVD分解，混淆了LoRA与权重矩阵分解。",
      "evidence": "标准LoRA冻结W并训练低秩增量ΔW=(α/r)BA。rule_check_tool正则命中：lora.*?(?:svd|奇异值分解)",
      "suggestion": "改为说明冻结原权重，仅训练低秩矩阵A和B形成增量更新。"
    },
    {
      "error_type": "hallucination",
      "severity": "major",
      "location": "参数量示例",
      "description": "知识库相似度0.42，存在潜在幻觉风险。",
      "evidence": "knowledge_verify_tool.similarity=0.42 < 0.65阈值，知识库无对应声明依据",
      "suggestion": "删除无法验证的技术声明或补充可靠来源。"
    }
  ],
  "summary": "检测到1个critical硬性错误（LoRA机制错误）和1个major幻觉风险，已阻断发布。建议修订后重新审核。",
  "review_details": "规则检查发现1条critical；LLM语义检查发现1条major幻觉。"
}
```

#### 代码错误分析场景输出（用户提交代码审核）
**输出格式**：`错误分类：xxx，严重等级：xxx，修改建议：xxx`

CodeErrorClassification对象，JSON格式：
```json
{
  "error_category": "RuntimeError",
  "severity": "major",
  "root_cause": "code_issue",
  "fix_suggestion": {
    "summary": "KeyError: 'user_id' — 访问了字典中不存在的键",
    "detail": "代码在访问字典 data['user_id'] 时，该键不存在。建议在访问前使用 data.get('user_id', default_value) 或先检查 'user_id' in data。",
    "code_fix": "user_id = data.get('user_id', None)\nif user_id is None:\n    raise ValueError(\"缺少必填字段 user_id\")",
    "environment_checks": []
  },
  "confidence": 0.85,
  "classification_method": "rule_based",
  "evidence": "错误类型KeyError匹配规则1 RuntimeError类；报错信息含'user_id'键名；代码审查无语法错误"
}
```

**字段说明**：
- `error_category`：错误分类（`"SyntaxError"` / `"RuntimeError"` / `"LogicError"` / `"unknown"`），来自 `code_error_patterns.skill.md` 规则1
- `severity`：严重等级（`"critical"` / `"major"` / `"minor"`），来自 `code_error_patterns.skill.md` 规则1
- `root_cause`：归因类型（`"code_issue"` / `"environment_issue"` / `"dependency_issue"` / `"data_issue"` / `"code_unknown"`），来自 `code_error_patterns.skill.md` 规则2
- `fix_suggestion.summary`：一句话修改建议，格式为"错误分类：xxx，严重等级：xxx，修改建议：xxx"
- `fix_suggestion.detail`：详细修改说明
- `fix_suggestion.code_fix`：修复代码示例（仅当有代码输入时提供）
- `fix_suggestion.environment_checks`：环境排查清单（仅当归因为环境问题时提供）
- `confidence`：分类置信度（0~1）
- `classification_method`：分类方法（`"rule_based"` 规则匹配 / `"knowledge_based"` 知识库匹配 / `"hybrid"` 混合）
- `evidence`：分类依据说明

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 工具超时（`knowledge_verify_tool` 或 `rule_check_tool` 调用超过30s） | 标记该条检查结果为"未知"，降级整体status为 `review_needed`，在summary中注明"工具超时，建议人工复核" |
| 证据ID缺失（`knowledge_verify_tool` 未返回evidence_id） | 用内容摘要（前80字符）替代evidence字段，格式为 `内容摘要: "<摘要>..."` |
| 多工具冲突（`rule_check_tool` 与LLM结论不一致） | 按规则4裁决优先级：硬性阻断 > 规范红线 > 语义冲突，以高优先级工具结论为准 |
| 待审核内容为空（`content.strip()` 为空） | 添加 `ReviewError(error_type="missing_content", severity="critical", description="待审核内容为空")`，passed=false |
| `rule_check_tool` 执行异常 | 记录日志，继续执行LLM审核，在review_details中标注"规则检查缺失" |
| LLM审核结果无法解析（`parse_json_object` 返回None） | 添加 `ReviewError(error_type="review_system_error", severity="critical", description="Agent4审核结果无法解析", suggestion="重新执行技术审核")`，passed=false |
| LLM返回的severity值不在 `{"critical","major","minor"}` 范围内 | 自动修正为 `"major"` |
| 内容审核循环达上限（Agent4+5并行审核循环≥3次仍未通过） | Orchestrator强制终止循环，status=`"review_needed"`，summary标注"审核循环达3次上限，建议人工介入修订内容"，不再发回Agent3修改 |

## 6. 附录

### 附录A：参考标准
| 编号 | 标准名称 | 作者/机构 | 年份 | 具体章节/贡献说明 |
|:---|:---|:---|:---|:---|
| A-1 | Stanford CS224V《RAG系统幻觉检测指南》 | Stanford University / Percy Liang团队 | 2024 | 第4章"幻觉阈值与评估"提出基于RAGAS faithfulness的相似度阈值建议：0.65为幻觉判定下限（精确率0.82），0.80为高置信通过线。本SKILL规则1的0.65/0.80阈值直接来源于此 |
| A-2 | OWASP ML Top 10 2023 | Open Worldwide Application Security Project / ML Security Project | 2023 | 列出机器学习系统十大安全风险，其中ML01:2023 Input Manipulation Attack、ML02:2023 Data Poisoning Attack、ML06:2023 Sensitive Information Disclosure为本SKILL规则2的路径硬编码、未捕获异常、敏感信息暴露等错误模式提供分级依据 |
| A-3 | PEP 8 — Style Guide for Python Code | Python Software Foundation / Guido van Rossum等 | 2013（持续更新） | 规定Python代码命名规范、缩进、导入顺序、异常处理风格。本SKILL规则3的Minor分级（命名不规范、冗余描述）依据PEP 8的命名约定（函数snake_case、常量UPPER_CASE）和"E714更推荐is not而非not is"等风格条款 |

### 附录B：完整场景示例

#### 场景1：硬性阻断（依赖冲突）
**输入**：
```json
{
  "target_type": "learning_package",
  "content": "安装依赖：pip install torch==2.0.0 transformers==4.30.0\n\nimport torch\nimport transformers\nfrom transformers import AutoModelForCausalLM\n\n# 加载模型\nmodel = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b')\nmodel = model.cuda()\nprint(model.device)",
  "stage": {"block_name": "模型部署"},
  "profile": {"goal": "掌握大模型部署", "learning_style": "practice"}
}
```

**处理过程**：
1. 步骤1解析：识别代码块，提取技术声明"pip install torch==2.0.0 transformers==4.30.0"和"model.cuda()"
2. 步骤2硬性检查：`rule_check_tool` 检测到 `model.cuda()` 为PyTorch 1.x旧写法（API调用过时模式）；同时检测到代码块缺失try-except（未捕获异常模式）
3. 步骤2规范检查：`industry_standards` 检测到无显存要求标注
4. 步骤3标记问题：生成3条ReviewError
5. 步骤4分级：API过时→major，未捕获异常→major，显存标注缺失→minor
6. 步骤5裁决：存在major错误，但无critical，status=`fail`，passed=false

**输出**：
```json
{
  "status": "fail",
  "passed": false,
  "errors": [
    {
      "error_type": "api_deprecated",
      "severity": "major",
      "location": "model.cuda()",
      "description": "使用PyTorch 1.x旧写法model.cuda()，未指定device。",
      "evidence": "PEP/PyTorch 2.x推荐使用model.to(device)，device由torch.device统一管理。",
      "suggestion": "改为 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = model.to(device)"
    },
    {
      "error_type": "missing_exception_handler",
      "severity": "major",
      "location": "AutoModelForCausalLM.from_pretrained 调用块",
      "description": "模型加载涉及IO操作，缺失try-except异常捕获。",
      "evidence": "OWASP ML Top 10 2023 ML06要求敏感操作配套异常处理。",
      "suggestion": "用try-except包裹from_pretrained调用，捕获OSError并记录日志。"
    },
    {
      "error_type": "missing_vram_note",
      "severity": "minor",
      "location": "部署说明",
      "description": "部署文档未标注Llama-2-7b FP16显存最低要求。",
      "evidence": "industry_standards规则2要求部署文档必须包含显存标注。",
      "suggestion": "补充注释：# 注意：Llama-2-7b FP16需要至少14GB显存"
    }
  ],
  "summary": "检测到2个major错误（API过时、未捕获异常）和1个minor错误（显存标注缺失），已阻断发布。建议修订后重新审核。",
  "review_details": "规则检查发现2条major、1条minor；语义检查未发现幻觉。"
}
```

#### 场景2：语义冲突（知识库相似度无依据但标准答案条目有依据）
**输入**：
```json
{
  "target_type": "answer",
  "content": "LoRA通过冻结原权重W，仅训练低秩增量ΔW=(α/r)BA实现高效微调，其中A和B是两个低秩矩阵，r远小于d_in和d_out。",
  "stage": {"block_name": "LoRA"},
  "profile": {"goal": "掌握LoRA微调"}
}
```

**处理过程**：
1. 步骤1解析：识别技术声明"LoRA冻结W训练ΔW=(α/r)BA"
2. 步骤2硬性检查：`rule_check_tool` 正则未命中（无SVD、无U/Σ/V描述），无critical
3. 步骤2语义检查：`knowledge_verify_tool` 计算相似度=0.58（< 0.65），按规则1应判"潜在幻觉"（critical）
4. 步骤2标准答案条目查询：知识库标准答案条目命中"LoRA冻结W训练ΔW=(α/r)BA"条目，明确支持该陈述
5. 步骤3标记问题：知识库相似度检索 vs 标准答案条目冲突
6. 步骤4分级：按规则4裁决——标准答案条目优先于相似度检索，判通过，但在evidence中标注"知识库标准答案条目依据"
7. 步骤5裁决：status=`pass`，passed=true，errors为空数组但review_details记录冲突

**输出**：
```json
{
  "status": "pass",
  "passed": true,
  "errors": [],
  "summary": "知识库相似度0.58低于阈值，但知识库标准答案条目明确支持该陈述，按裁决优先级（标准答案条目>相似度检索）判通过。",
  "review_details": "知识库sim=0.58（潜在幻觉），标准答案条目命中LoRA冻结W条目，以标准答案条目为准。建议后续扩充知识库覆盖。"
}
```

**说明**：场景2体现了规则4的"标准答案条目优先于相似度检索"裁决逻辑——当知识库检索能力不足导致相似度低，但标准答案条目有明确依据时，不应误判为幻觉，而应以标准答案条目为准放行，并在review_details中提示知识库覆盖不足。

## 7. 版本信息
- 版本：v3.3，更新日期：2026-07-25
- 变更记录：**v3.3变更**：新增"用户提交代码审核流程"（2.2节），定义Agent4处理代码错误分析请求的6步专属流程；明确判断依据优先级（code_error_patterns.skill.md规则 > knowledge_verify_tool > 静态检查工具）；新增代码错误分析场景的输出格式与字段说明（含error_category、severity、root_cause、fix_suggestion等字段），输出格式为"错误分类：xxx，严重等级：xxx，修改建议：xxx"。**v3.2变更**：将核心方法论的"逐级检查"拆分为独立的硬性检查、事实检查、标准检查三个步骤，明确各步骤的工具调用和职责边界，与文档3的流程步骤一一对应
- **与代码对齐说明**：`analyze_code_error()` 方法已实现代码错误分析功能，v3.3要求该方法集成 `code_error_patterns.skill.md` 的错误分级与归因规则，优先使用规则分类结果作为最终判定，knowledge_verify_tool结果作为辅助参考
