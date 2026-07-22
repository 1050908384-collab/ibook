# SKILL名称：code_style

## 1. 能力描述
- 功能：为Agent3（HybridContentAgent）生成教程中的 **Python 代码片段提供风格与规范约束**，作为内容生成 prompt 的提示规范。定义"教程里什么样的代码是合格的"：PEP8 风格、docstring 要求、异常处理、配置化与安全规范。在 `HybridContentAgent.run` 中与 `content_generation` 一同被加载为 `content_skill_context`，注入 LLM 生成 tutorial 的 prompt；本SKILL不执行实际校验，代码规范违规由 Agent4 `rule_check_tool` 与 `industry_standards` 兜底检测
- 适用场景（3个）：
  1. **教程代码生成**：Agent3 生成含代码示例的 Markdown 教程时，作为 `content_skill_context` 一同注入 prompt，约束生成代码风格
  2. **改稿期代码修订**：`revise_learning_package` 接到 Agent4 代码相关 errors（如 `missing_exception_handler`、`api_deprecated`）时，本SKILL指导如何修订代码
  3. **答疑代码片段**：Agent3 `generate_answer` 回答中含代码时，约束片段风格
- 前置依赖：
  - 当前阶段 StageInfo（`block_name`，决定代码主题）
  - `content_generation` 技能（本SKILL作为其子规范，不独立产出）
  - 无独立工具；仅作为提示文本被 LLM 读取

## 2. 核心方法论
- **步骤1：应用 PEP8 基本风格**——生成的 Python 代码遵循 PEP8：单行≤88字符、4空格缩进、导入分标准库/第三方/本地三层、命名 snake_case（函数/变量）与 CamelCase（类）
- **步骤2：补充 docstring 与注释**——顶层函数与类必须配中文 docstring（标注功能、入参、返回值）；模型加载、训练循环、推理等关键步骤配单行注释
- **步骤3：配套异常处理**——文件 IO、模型加载（`from_pretrained`）、网络请求、推理等步骤必须 try-except，捕获具体异常并记录日志
- **步骤4：配置化与安全**——禁止硬编码密钥/GPU编号/账号/绝对路径，统一用 `os.getenv` 或配置文件读取；敏感信息不进代码
- **步骤5：API 时效性**——使用 PyTorch 2.x 推荐 API（`model.to(device)` 而非 `model.cuda()`），避免过时写法

## 3. 知识/规则定义

### 规则1：PEP8 风格规范
| 检查项 | 要求 | 违规示例 | 正确示例 | severity |
|:---|:---|:---|:---|:---|
| 行宽 | 单行≤88字符 | `model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", torch_dtype=torch.float16)` 超长 | 换行+括号续行 | minor |
| 导入顺序 | 标准库→第三方→本地，分组空行 | 混排 `import os` 与 `import torch` | `import os\n\nimport torch` | minor |
| 命名 | 函数/变量 snake_case，类 CamelCase，常量 UPPER_CASE | `def MyFunc():` / `class my_class:` | `def my_func():` / `class MyClass:` | minor |
| 缩进 | 4空格，禁用Tab | Tab缩进 | 4空格 | minor |

### 规则2：docstring 与注释规范
| 检查项 | 要求 | 违规处置 | severity |
|:---|:---|:---|:---|
| 顶层函数/类 docstring | 必须有，标注功能、入参、返回值 | 标记 `missing_docstring` | major |
| 关键步骤注释 | 模型加载/训练循环/推理配单行注释 | 标记 `missing_comment` | minor |
| docstring 语言 | 中文为主，术语保留英文 | 标记 `docstring_lang` | minor |

### 规则3：异常处理规范
| 检查项 | 要求 | 违规示例 | 正确示例 | severity |
|:---|:---|:---|:---|:---|
| IO/模型加载/网络 | 必须 try-except | `model = AutoModelForCausalLM.from_pretrained(path)` 裸调 | `try: model = ...from_pretrained(path) except OSError as e: logger.error(e); raise` | major |
| 捕获具体异常 | 禁止裸 `except:` | `except: pass` | `except (OSError, RuntimeError) as e:` | major |
| 异常记录 | 捕获后须 logger 记录 | `except: pass` 静默吞掉 | `except Exception as e: logger.error(e)` | major |

### 规则4：配置化与安全规范
| 检查项 | 要求 | 违规示例 | 正确示例 | severity |
|:---|:---|:---|:---|:---|
| 路径 | 禁止硬编码绝对路径 | `pd.read_csv("/root/data/train.csv")` | `DATA_DIR = os.getenv("DATA_DIR", "./data")` | critical |
| 密钥/账号 | 禁止硬编码 | `API_KEY = "sk-xxx"` | `API_KEY = os.getenv("API_KEY")` | critical |
| GPU 设备 | 禁止硬编码设备号 | `torch.cuda.set_device(0)` | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | major |
| API 时效 | 用 PyTorch 2.x 写法 | `model.cuda()` | `model.to(device)` | major |

## 4. 输入输出规范

### 输入
- 无独立输入；作为提示文本注入教程生成 prompt（`content_skill_context` 拼接 `content_generation` + `code_style` 全文）

### 输出
- 约束 LLM 生成的教程内代码片段风格（不产出独立校验报告）
- 代码规范违规的实际检测与报告由 Agent4 `rule_check_tool`（正则）与 `industry_standards`（红线）产出 ReviewError

### 与 Agent4 的衔接
教程代码经 Agent4 审核后，常见的代码类 ReviewError：
```json
{
  "error_type": "missing_exception_handler",
  "severity": "major",
  "location": "AutoModelForCausalLM.from_pretrained 调用块",
  "description": "模型加载涉及IO操作，缺失try-except异常捕获。",
  "suggestion": "用try-except包裹from_pretrained调用，捕获OSError并记录日志。"
}
```

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 本SKILL为被动提示，不执行实际校验 | 代码规范违规由 Agent4 `rule_check_tool` 与 `industry_standards` 兜底检测 |
| 生成的代码缺 docstring/异常处理 | 由 Agent4 标记 `missing_docstring` / `missing_exception_handler`，回流至 `revise_learning_package` 由本SKILL指导修订 |
| 生成的代码硬编码路径/密钥 | 由 Agent4 标记 critical（`path_hardcoded` / `secret_hardcoded`），强制改稿 |
| 代码片段使用过时 API | 由 Agent4 标记 `api_deprecated`（major），改稿时参照规则4替换为推荐写法 |

## 6. 附录

### 附录A：代码风格参考标准
| 编号 | 标准名称 | 作者/机构 | 年份 | 核心观点 | 在本SKILL中的应用 |
|:---|:---|:---|:---|:---|:---|
| A-1 | PEP 8 — Style Guide for Python Code | Python Software Foundation / Guido van Rossum 等 | 2001（持续更新） | 规定 Python 命名规范、缩进、导入顺序、行宽、异常处理风格 | 规则1 PEP8 风格规范的全部依据（行宽88、snake_case、导入分层） |
| A-2 | Google Python Style Guide | Google | 2008（持续更新） | 强调 docstring 规范（Args/Returns/Raises 三段式）、模块级注释、类型注解 | 规则2 docstring 与注释规范的依据 |
| A-3 | OWASP ML Top 10 2023 | Open Worldwide Application Security Project | 2023 | ML06:2023 Sensitive Information Disclosure 要求敏感信息不进代码、配置化读取 | 规则4配置化与安全规范的依据（禁止硬编码密钥/路径） |
| A-4 | Clean Code | Robert C. Martin | 2008 | 函数应短小、单一职责、有意义命名、错误处理不掩盖异常 | 规则3异常处理规范（禁裸 except、禁静默吞异常）的依据 |

### 附录B：合规代码 vs 违规代码示例

#### 合规代码示例（通过审核）
```python
import os
import logging

import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> AutoModelForCausalLM:
    """加载因果语言模型。

    Args:
        model_path: 模型路径，优先从环境变量 MODEL_PATH 读取。
    Returns:
        加载完成的模型实例。
    """
    path = model_path or os.getenv("MODEL_PATH", "./base_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(path)
        return model.to(device)  # 统一设备管理
    except OSError as e:
        logger.error("模型加载失败: %s", e)
        raise
```

**评分解析**：导入分层✓、docstring完整✓、异常处理✓（捕获OSError+logger）、配置化✓（os.getenv）、PyTorch2.x写法✓（model.to(device)）。

#### 违规代码示例（触发4类ReviewError）
```python
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/root/models/Llama-2-7b")
model = model.cuda()
API_KEY = "sk-xxxxxxxxxxxx"
```

**问题清单**：
| error_type | severity | location | suggestion |
|:---|:---|:---|:---|
| `path_hardcoded` | critical | `from_pretrained` 调用 | 用 `os.getenv("MODEL_PATH")` 替代硬编码路径 |
| `secret_hardcoded` | critical | `API_KEY = "sk-xxx"` | 用 `os.getenv("API_KEY")`，禁止密钥进代码 |
| `missing_exception_handler` | major | `from_pretrained` 调用块 | 用 try-except 包裹，捕获 OSError 并记录日志 |
| `api_deprecated` | major | `model.cuda()` | 改为 `model.to(device)`，device 由 `torch.device` 统一管理 |

**说明**：违规代码触发2个critical（硬编码路径/密钥）+2个major（缺异常处理/API过时），按 `technical_review` 规则4裁决优先级，critical阻断→passed=false，强制改稿。

## 7. 版本信息
- 版本：v3.0，更新日期：2026-07-23
- 变更记录：能力描述明确3个适用场景与被动提示定位（实际校验由Agent4兜底）；核心方法论重构为5步（PEP8→docstring→异常处理→配置化安全→API时效）；规则1新增PEP8检查表（含违规/正确示例/severity）；规则2新增docstring规范表；规则3新增异常处理规范表（禁裸except/禁静默吞异常）；规则4新增配置化与安全规范表（含critical级硬编码处置）；输入输出规范新增与Agent4衔接的ReviewError JSON示例；异常处理扩展为4场景表格；附录A补4个参考标准（PEP8/Google Style/OWASP/Clean Code）；附录B新增合规代码 vs 违规代码完整对比示例（含4类ReviewError清单）
