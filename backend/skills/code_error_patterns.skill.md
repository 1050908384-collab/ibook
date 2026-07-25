# SKILL名称：code_error_patterns

## 1. 能力描述
- 功能：为Agent④提供**代码错误行业通用分类依据**，定义"代码报错属于哪类、严重程度如何、归因于代码还是环境"。作为`technical_review.skill.md`中"用户代码审核"场景的**首要判定依据**
- 适用场景：当Agent④收到"用户代码错误分析"请求时（含代码+报错，或仅问题描述），优先引用本SKILL的分类规则进行错误归类和归因
- 前置依赖：
  - 用户提交的报错信息（`error_info`，含异常类型名、错误描述、堆栈跟踪）
  - 用户提交的代码片段（`code`，可选）
  - 代码语法检查结果（来自 `code_parser_tool`，可选）
  - 代码规范检查结果（来自 `rule_check_tool`，可选）
  - 知识库错误匹配结果（来自 `knowledge_verify_tool.verify_code_error`，可选）

## 2. 核心方法论
- **步骤1：解析报错信息提取错误类型**——从 `error_info` 中提取异常类型名（如 `KeyError`、`TypeError`），匹配规则1的错误分类映射表
- **步骤2：判定错误严重等级**——应用规则1，将错误类型映射为三级严重等级：SyntaxError类→致命(critical)、RuntimeError类→主要(major)、LogicError类→次要(minor)但需修改
- **步骤3：归因判断（代码问题 vs 环境问题）**——应用规则2，对已识别的错误类型进行归因分类：如果错误属于"环境问题"（如ModuleNotFoundError、FileNotFoundError），标记归因类型并给出环境排查建议
- **步骤4：生成标准化输出**——输出 `error_category`（错误分类）、`severity`（严重等级）、`root_cause`（归因类型）、`fix_suggestion`（修改建议）

## 3. 知识/规则定义

### 规则1：错误分级与分类映射表
**三级错误等级定义**：

| 错误等级 | 标识 | 定义 | 典型错误类型 | 处置要求 |
|:---|:---|:---|:---|:---|
| 致命 | critical | 代码无法运行，必须修复后才能继续 | SyntaxError、IndentationError、TabError、语法类错误 | 强制要求修改，不修改无法通过 |
| 主要 | major | 代码可运行但会产生错误结果或崩溃 | RuntimeError、KeyError、IndexError、TypeError、ValueError、AttributeError、NameError、ZeroDivisionError、MemoryError、ImportError | 强烈建议修改，可能导致运行时异常 |
| 次要 | minor | 代码可运行但逻辑不正确或存在隐患 | LogicError、AssertionError、NotImplementedError、自定义业务异常 | 建议修改，不影响运行但结果可能不符合预期 |

**错误类型→等级自动映射表**：

| 异常类型关键词 | 分类等级 | 说明 |
|:---|:---|:---|
| SyntaxError / IndentationError / TabError | critical | 语法错误，Python解释器无法解析 |
| RuntimeError / TypeError / ValueError / KeyError / IndexError / AttributeError / NameError / UnboundLocalError / ZeroDivisionError / MemoryError / RecursionError / OverflowError | major | 运行时错误，代码语法正确但执行时触发异常 |
| ImportError / ModuleNotFoundError / FileNotFoundError / PermissionError / IsADirectoryError / NotADirectoryError | major | 资源访问类错误，通常与环境配置或文件路径有关 |
| ConnectionError / TimeoutError / OSError / IOError | major | 网络/IO类错误，通常与环境或外部服务有关 |
| AssertionError / NotImplementedError / StopIteration / GeneratorExit / Exception | minor | 逻辑/业务类异常，可能是代码逻辑问题 |
| CUDAError / TorchError / PyTorchError / TensorFlowError | major | 框架/硬件相关错误，通常需调整环境配置或代码写法 |
| JSONDecodeError / JSONEncodeError | major | 数据格式错误，需检查输入数据格式 |
| HTTPError / RequestException / URLError | major | 网络请求错误，通常与外部服务或网络环境有关 |

**未识别错误的默认分级**：
- 若报错信息无法匹配上述任何类型，默认归类为 `major`
- 标注 `error_category="unknown"`，`severity="major"`，并在 `fix_suggestion` 中建议"请提供完整报错信息以便精确诊断"

### 规则2：错误归因判定（代码问题 vs 环境问题）

**归因分类**：

| 归因类型 | 定义 | 判定依据 | 典型错误类型 |
|:---|:---|:---|:---|
| 代码问题 | 错误根源在用户代码本身的逻辑、语法或写法 | 错误类型属于规则1中SyntaxError/LogicError类，且不涉及外部资源 | SyntaxError、IndentationError、KeyError、IndexError、TypeError、LogicError、AttributeError、NameError、ZeroDivisionError、AssertionError |
| 环境问题 | 错误根源在运行环境、依赖配置、文件路径等外部条件 | 错误类型涉及模块导入、文件访问、网络连接、硬件资源 | ModuleNotFoundError、ImportError、FileNotFoundError、PermissionError、ConnectionError、TimeoutError、CUDAError、TorchError、MemoryError |
| 外部依赖问题 | 错误根源在第三方库版本不兼容或API变更 | 报错信息含"deprecated"、"removed"、"not found in module"等关键词 | AttributeError（因库API变更）、ImportError（因版本不兼容）、TypeError（因参数变更） |
| 数据问题 | 错误根源在输入数据格式或内容不符合预期 | 报错信息含"cannot parse"、"invalid format"、"unexpected value"等关键词 | JSONDecodeError、ValueError（数据校验类）、KeyError（缺失必填字段） |

**归因判定优先级**：
1. 若错误类型在"环境问题"典型列表中 → 直接判定为环境问题
2. 若错误信息含版本/弃用关键词（如"deprecated"、"version"、"incompatible"） → 判定为外部依赖问题
3. 若错误信息含数据格式关键词（如"parse"、"format"、"invalid"） → 判定为数据问题
4. 其余错误 → 判定为代码问题
5. 无法判定时 → 默认标记为 `code_unknown`（未知归因），建议用户提供更多上下文

## 4. 输入输出规范

### 输入
CodeErrorAnalysisInput对象，包含：
- `error_info`：报错信息字符串（必填），含异常类型名、错误描述、可选堆栈跟踪
- `code`：用户提交的代码片段（可选）
- `code_parser_errors`：`code_parser_tool` 返回的语法检查结果（可选）
- `rule_violations`：`rule_check_tool` 返回的规范检查结果（可选）
- `knowledge_match`：`knowledge_verify_tool.verify_code_error` 返回的知识库匹配结果（可选）

### 输出
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
- `error_category`：错误分类（`"SyntaxError"` / `"RuntimeError"` / `"LogicError"` / `"unknown"`）
- `severity`：严重等级（`"critical"` / `"major"` / `"minor"`）
- `root_cause`：归因类型（`"code_issue"` / `"environment_issue"` / `"dependency_issue"` / `"data_issue"` / `"code_unknown"`）
- `fix_suggestion.summary`：一句话修改建议
- `fix_suggestion.detail`：详细修改说明
- `fix_suggestion.code_fix`：修复代码示例（仅当有代码输入时提供）
- `fix_suggestion.environment_checks`：环境排查清单（仅当归因为环境问题时提供）
- `confidence`：分类置信度（0~1）
- `classification_method`：分类方法（`"rule_based"` 规则匹配 / `"knowledge_based"` 知识库匹配 / `"hybrid"` 混合）

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 报错信息为空（`error_info.strip()` 为空） | 返回默认分类：`error_category="unknown"`、`severity="major"`、`root_cause="code_unknown"`，`fix_suggestion` 建议用户提供完整报错信息 |
| 报错信息无法识别任何已知错误类型 | 按"未识别错误的默认分级"处理：`severity="major"`，标注 `error_category="unknown"`，建议用户提供更详细的报错堆栈 |
| 归因判定不确定（同时符合多种归因类型特征） | 按归因判定优先级取最高匹配类型；若仍无法判定，标记为 `root_cause="code_unknown"` |
| 用户仅提供问题描述无代码和报错 | 返回 `error_category="unknown"`，`fix_suggestion` 建议用户补充报错信息和相关代码片段以获得精确诊断 |
| 知识库匹配与规则分类冲突（如知识库判定为代码问题但规则判定为环境问题） | 以规则分类为准，知识库匹配结果仅作为参考附在 `evidence` 字段中 |

## 6. 附录

### 附录A：错误分类与归因速查表
| 异常类型 | 等级 | 归因 | 常见场景 |
|:---|:---|:---|:---|
| SyntaxError | critical | code_issue | 语法拼写错误、缩进错误 |
| IndentationError | critical | code_issue | 缩进不一致、Tab/空格混用 |
| ModuleNotFoundError | major | environment_issue | 缺少依赖包、虚拟环境未激活 |
| ImportError | major | environment_issue | 模块路径错误、循环导入 |
| KeyError | major | code_issue | 字典键不存在 |
| IndexError | major | code_issue | 列表/数组索引越界 |
| TypeError | major | code_issue | 类型操作不兼容（如str+int） |
| ValueError | major | code_issue | 函数参数值非法 |
| AttributeError | major | dependency_issue | 对象无该属性（可能因API变更） |
| FileNotFoundError | major | environment_issue | 文件路径错误或文件不存在 |
| PermissionError | major | environment_issue | 文件/目录权限不足 |
| CUDAError | major | environment_issue | GPU资源不足、CUDA版本不匹配 |
| MemoryError | major | environment_issue | 内存不足，需优化算法或增加资源 |
| ConnectionError | major | environment_issue | 网络连接失败、服务不可达 |
| AssertionError | minor | code_issue | 断言条件不满足（逻辑错误） |
| LogicError | minor | code_issue | 业务逻辑不正确 |

### 附录B：完整场景示例

#### 场景1：用户提交代码+报错（KeyError）
**输入**：
```json
{
  "error_info": "KeyError: 'user_id'\n  at line 5: name = data['user_id']",
  "code": "def process_user(data):\n    name = data['user_id']\n    age = data['age']\n    return {'name': name, 'age': age}",
  "code_parser_errors": [],
  "rule_violations": [],
  "knowledge_match": {
    "is_known_error": true,
    "known_solution": "使用dict.get()方法或检查键是否存在",
    "confidence": 0.75
  }
}
```

**处理过程**：
1. 步骤1：解析错误类型 → `KeyError`
2. 步骤2：匹配规则1 → `KeyError` 属于 RuntimeError类 → `severity="major"`
3. 步骤3：匹配规则2 → KeyError属于"代码问题"典型列表 → `root_cause="code_issue"`
4. 步骤4：生成输出 → 综合知识库建议和规则判定

**输出**：
```json
{
  "error_category": "RuntimeError",
  "severity": "major",
  "root_cause": "code_issue",
  "fix_suggestion": {
    "summary": "KeyError: 'user_id' — 访问了字典中不存在的键",
    "detail": "代码在第5行访问 data['user_id'] 时，该键不存在。建议使用 dict.get() 方法提供默认值，或在访问前检查键是否存在。",
    "code_fix": "def process_user(data):\n    name = data.get('user_id', None)\n    if name is None:\n        raise ValueError(\"缺少必填字段 user_id\")\n    age = data.get('age', 0)\n    return {'name': name, 'age': age}",
    "environment_checks": []
  },
  "confidence": 0.85,
  "classification_method": "hybrid",
  "evidence": "错误类型KeyError匹配规则1 RuntimeError类；知识库匹配度0.75，建议使用dict.get()；代码检查无语法/规范错误"
}
```

#### 场景2：用户仅描述问题（无代码无报错）
**输入**：
```json
{
  "error_info": "",
  "code": "",
  "question": "我运行pip install的时候报错说找不到包，怎么办？"
}
```

**处理过程**：
1. 步骤1：无法提取错误类型 → `error_category="unknown"`
2. 步骤2：默认分级 → `severity="major"`
3. 步骤3：从问题描述推断 → 涉及"找不到包"→ 疑似环境/依赖问题 → `root_cause="environment_issue"`
4. 步骤4：生成输出 → 建议用户提供具体报错信息

**输出**：
```json
{
  "error_category": "unknown",
  "severity": "major",
  "root_cause": "environment_issue",
  "fix_suggestion": {
    "summary": "疑似依赖环境问题，请提供完整报错信息以便精确诊断",
    "detail": "问题描述涉及包安装失败，可能是环境问题（包名错误、源配置、Python版本不匹配等）。建议提供完整的报错堆栈信息和当前Python环境信息（python --version、pip list），以便进行精确诊断。",
    "code_fix": "",
    "environment_checks": [
      "确认包名拼写正确（区分大小写）",
      "检查Python版本是否满足包的要求",
      "尝试更换pip源：pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple",
      "检查是否在虚拟环境中运行pip"
    ]
  },
  "confidence": 0.3,
  "classification_method": "rule_based",
  "evidence": "无具体报错信息，仅从问题描述推断为包安装问题，置信度较低"
}
```

#### 场景3：环境问题（ModuleNotFoundError）
**输入**：
```json
{
  "error_info": "ModuleNotFoundError: No module named 'torch'\n  at line 1: import torch",
  "code": "import torch\nimport transformers\n\nmodel = transformers.AutoModel.from_pretrained('bert-base-uncased')",
  "code_parser_errors": [],
  "rule_violations": [],
  "knowledge_match": {
    "is_known_error": true,
    "known_solution": "安装缺失的依赖包：pip install torch transformers",
    "confidence": 0.9
  }
}
```

**处理过程**：
1. 步骤1：解析错误类型 → `ModuleNotFoundError`
2. 步骤2：匹配规则1 → `ModuleNotFoundError` 属于 RuntimeError类 → `severity="major"`
3. 步骤3：匹配规则2 → ModuleNotFoundError属于"环境问题"典型列表 → `root_cause="environment_issue"`
4. 步骤4：生成输出 → 提供环境排查清单

**输出**：
```json
{
  "error_category": "RuntimeError",
  "severity": "major",
  "root_cause": "environment_issue",
  "fix_suggestion": {
    "summary": "ModuleNotFoundError: No module named 'torch' — 缺少torch依赖包",
    "detail": "运行环境中未安装torch库。需在当前Python环境中安装torch和transformers包。",
    "code_fix": "",
    "environment_checks": [
      "执行 pip install torch transformers 安装缺失依赖",
      "确认当前Python环境正确（python --version）",
      "若使用GPU版本，安装时需指定CUDA版本：pip install torch --index-url https://download.pytorch.org/whl/cu121",
      "若在虚拟环境中，确认已激活正确的venv/conda环境"
    ]
  },
  "confidence": 0.95,
  "classification_method": "hybrid",
  "evidence": "错误类型ModuleNotFoundError匹配规则1 RuntimeError类；匹配规则2环境问题典型列表；知识库匹配度0.9"
}
```

## 7. 版本信息
- 版本：v1.0，更新日期：2026-07-25
- 变更记录：新建SKILL文件，定义代码错误行业通用分类依据；规则1定义三级错误分级（SyntaxError→critical、RuntimeError→major、LogicError→minor）及完整错误类型→等级映射表；规则2定义四维度归因判定（代码问题/环境问题/外部依赖问题/数据问题）及判定优先级；附录A提供错误分类与归因速查表；附录B新增3个完整场景示例（代码+报错/仅问题描述/环境问题）
- **与代码对齐说明**：本SKILL定义的错误分类规则将集成到 `TechnicalReviewAgent.analyze_code_error()` 方法中，`knowledge_verify_tool.py` 的 `_ERROR_TYPE_MAP` 已包含本SKILL规则1列出的错误类型关键词映射，后续可在 `rule_check_tool.py` 中增加错误分类功能以对齐本SKILL的完整分级标准