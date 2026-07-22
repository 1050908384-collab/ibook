# SKILL名称：industry_standards

## 1. 能力描述
- 功能：为Agent④提供大模型微调与部署领域的**行业红线规则库**，定义"行业红线"——微调与部署领域的安全规范、部署规范、数据隐私规范。Agent④在技术审核时调用本SKILL匹配违规模式，与 `rule_check_tool`、`technical_review.skill.md` 配合，在代码/配置安全、部署规范、数据隐私三个维度执行合规检查
- 适用场景（3个）：
  1. **内容中的代码/配置/命令审核**：审核学习包中的Python代码、YAML/JSON配置、Shell命令是否违反安全规范（硬编码密钥、端口暴露、权限缺失）
  2. **部署文档审核**：审核部署教程是否标注模型许可证、显存最低要求、依赖版本等部署规范
  3. **数据操作示例审核**：审核涉及用户数据的示例代码是否脱敏、是否使用加密传输
- 前置依赖：
  - 待审核内容（`ReviewInput.content`，含代码/配置/文档）
  - 规范规则库：
    - 硬编码检测模式（API_KEY、端口、密码正则）
    - 许可证清单（MIT/Apache/GPL/Llama2/GPT 可商用性）
    - 脱敏要求（user_id/phone/email 字段处理规则）
  - LoRA领域基线（用于参数量核验的辅助依据）

## 2. 核心方法论
- **步骤1：从内容中提取配置信息、命令、API调用、数据操作语句**——通过正则和代码块识别提取：
  - 配置项：`API_KEY`、`KEY`、`password`、`secret`、`token` 等敏感字段赋值
  - 命令：`pip install`、`CUDA_VISIBLE_DEVICES`、`docker run`、`python train.py`
  - API调用：`from_pretrained`、`model.cuda()`、`requests.get`、`openai.ChatCompletion`
  - 数据操作语句：`read_csv`、`fetch_user`、`SELECT * FROM users`、`send_email`
- **步骤2：逐项匹配违规规则（正则/关键词）**——对提取出的每一项应用规则1（安全规范）、规则2（部署规范）、规则3（数据隐私）的正则模式或关键词匹配，记录命中位置和证据片段
- **步骤3：按严重程度分级（Critical/Major/Minor）**——
  - Critical：硬编码密钥、硬编码数据库账号、不可商用模型用于生产部署示例
  - Major：端口暴露0.0.0.0、缺失权限校验、缺失异常处理、使用真实用户数据、非HTTPS传输
  - Minor：缺失许可证说明、缺失显存要求标注、缺失依赖版本约束
- **步骤4：生成违规报告**——输出violations数组，每项包含level/category/clause/evidence/suggestion，并附summary汇总。违规列表合并到 `TechnicalReviewResult.errors` 中

## 3. 知识/规则定义

### 规则1：安全规范
#### 1.1 禁止硬编码API密钥
**检测模式**：
- 正则1：`API_KEY\s*=\s*["']sk-` 匹配形如 `API_KEY = "sk-..."` 的OpenAI风格密钥硬编码
- 正则2：`os\.environ\[\s*["']KEY["']\s*\]\s*=\s*["']` 匹配 `os.environ["KEY"] = "..."` 的环境变量硬编码
- 正则3：`api_key\s*=\s*["'][a-zA-Z0-9_-]{20,}["']` 匹配通用API密钥赋值

**错误示例**：
```python
API_KEY = "sk-abcdef1234567890abcdef1234567890"  # 违规：硬编码密钥
os.environ["OPENAI_API_KEY"] = "sk-xxx"  # 违规：环境变量硬编码
```

**正确示例（修复建议）**：
```python
import os
API_KEY = os.getenv("OPENAI_API_KEY")  # 合规：从环境变量读取
if not API_KEY:
    raise EnvironmentError("请设置 OPENAI_API_KEY 环境变量")
```

**违规级别**：Critical

#### 1.2 禁止暴露内部端口
**检测模式**：
- 正则：`host\s*=\s*["']0\.0\.0\.0["']` 或 `app\.run\(\s*host\s*=\s*["']0\.0\.0\.0["']` 或 `bind\s*.*0\.0\.0\.0:\s*\d+`
- 关键词：`port=8080` 且 `host=0.0.0.0` 同时出现

**错误示例**：
```python
app.run(host="0.0.0.0", port=8080)  # 违规：绑定所有网卡
uvicorn.run(app, host="0.0.0.0", port=8000)  # 违规
```

**正确示例**：
```python
app.run(host="127.0.0.1", port=8080)  # 合规：仅绑定本地回环
# 或通过反向代理（nginx）暴露，应用层仅监听127.0.0.1
```

**违规级别**：Major

#### 1.3 必须包含权限校验
**检测模式**：
- 涉及用户数据操作的API端点，必须包含 `@login_required` 装饰器 或 `auth` 中间件 或 `Depends(get_current_user)` 依赖注入
- 正则：检测路由定义 `@app\.(get|post|put|delete)` 后续行是否包含 `@login_required` 或 `Depends(get_current_user)` 或 `auth`

**错误示例**：
```python
@app.get("/api/users")
def get_users():  # 违规：无权限校验
    return User.query.all()
```

**正确示例**：
```python
from flask_login import login_required

@app.get("/api/users")
@login_required  # 合规：添加权限校验
def get_users():
    return User.query.all()
```

**违规级别**：Major

### 规则2：部署规范
#### 2.1 模型许可证合规
**许可证清单与可商用性**：
| 许可证 | 类型 | 可商用性 | 部署检查逻辑 |
|:---|:---|:---|:---|
| MIT | 宽松开源 | ✅ 可商用 | 通过，无需额外声明 |
| Apache 2.0 | 宽松开源 | ✅ 可商用 | 通过，需保留NOTICE文件 |
| GPL v3 | Copyleft | ⚠️ 需开源衍生作品 | 商用部署需检查衍生作品是否开源 |
| Llama 2 Community | 社区许可 | ⚠️ 月活>7亿需申请 | 商用需检查月活用户数，超过7亿需Meta授权 |
| Llama 3 Community | 社区许可 | ⚠️ 月活>7亿需申请 | 同Llama 2 |
| GPT系（OpenAI API） | 闭源API | ❌ 不可作为自有模型商用部署 | 仅可作为API调用，不可作为自有模型分发 |

**检查逻辑**：
- 部署文档引用模型时必须标注许可证类型
- 若模型为GPL/Llama Community许可且部署示例面向商用场景，触发Major违规
- 若将GPT系API包装为"自有模型"分发，触发Critical违规

**违规级别**：GPL未声明开源→Major；GPT系冒充自有模型→Critical；缺失许可证说明→Minor

#### 2.2 显存最低要求标注
**计算公式**：
```
显存(GB) ≈ 参数量(B) × 精度系数 × 1.2
```
其中：
- 精度系数：FP32=4、FP16/BF16=2、INT8=1、INT4=0.5
- 1.2为经验冗余系数（含激活值、KV Cache、优化器状态的余量）

**常见模型显存参考**：
| 模型 | 参数量 | 精度 | 显存估算 | 最低要求 |
|:---|:---|:---|:---|:---|
| Llama-2-7B | 7B | FP16 | 7×2×1.2≈16.8GB | 14GB（最低，无优化） |
| Llama-2-7B | 7B | INT8 | 7×1×1.2≈8.4GB | 8GB |
| Llama-2-13B | 13B | FP16 | 13×2×1.2≈31.2GB | 24GB |
| Llama-2-70B | 70B | FP16 | 70×2×1.2≈168GB | 140GB（多卡） |
| Qwen-7B | 7B | FP16 | 7×2×1.2≈16.8GB | 14GB |
| ChatGLM-6B | 6B | FP16 | 6×2×1.2≈14.4GB | 12GB |

**检查逻辑**：部署文档若涉及 `from_pretrained` 加载模型，必须包含显存要求标注（注释或文档说明），关键词检测 `显存|VRAM|GPU内存|memory`

**违规级别**：缺失显存标注→Minor

### 规则3：数据隐私
#### 3.1 涉及用户数据的操作必须脱敏
**检测字段**：
- `user_id`、`user_name`、`username`、`name` 字段在示例代码中以真实值出现
- `phone`、`mobile`、`tel` 字段为11位数字（中国手机号格式）
- `email` 字段为标准邮箱格式 `xxx@xxx.com`
- `id_card`、`身份证` 字段为18位数字

**脱敏方式**：
| 字段类型 | 真实示例 | 脱敏后示例 |
|:---|:---|:---|
| user_id | `user_28394756` | `user_001`（用序号替代） |
| phone | `13812341234` | `138****1234`（中间4位打码） |
| email | `zhangsan@company.com` | `user001@example.com`（使用example域名） |
| id_card | `110101199001011234` | `110101********1234`（出生日期打码） |

**检测正则**：
- `user_id\s*=\s*["']\d{6,}["']` 匹配疑似真实用户ID
- `phone\s*=\s*["']1[3-9]\d{9}["']` 匹配11位手机号
- `email\s*=\s*["'][^@]+@(?!example\.com)[^"']+["']` 匹配非example域名的邮箱

**违规级别**：使用真实用户数据→Major

#### 3.2 数据传输加密
**检测模式**：
- 数据传输URL必须以 `https://` 或 `wss://` 开头
- 正则：`(http://|ws://)` 命中即违规（除非明确标注为本地开发环境 `http://localhost` 或 `http://127.0.0.1`）

**错误示例**：
```python
response = requests.get("http://api.example.com/users")  # 违规：非加密传输
ws = WebSocket("ws://api.example.com/ws")  # 违规
```

**正确示例**：
```python
response = requests.get("https://api.example.com/users")  # 合规
ws = WebSocket("wss://api.example.com/ws")  # 合规
```

**违规级别**：非加密传输→Major

## 4. 输入输出规范

### 输入
ReviewInput对象，包含：
- `content`：待审核内容字符串（含代码块、配置、命令、文档说明）
- `user_profile`（可选）：用户画像，用于判断示例难度是否匹配学习者
- `context`（可选）：当前StageInfo

### 输出
违规报告，LLM返回JSON格式：
```json
{
  "violations": [
    {
      "level": "critical",
      "category": "security",
      "clause": "规则1.1 禁止硬编码API密钥",
      "evidence": "代码第3行：API_KEY = \"sk-abcdef1234567890\"",
      "suggestion": "改为使用 os.getenv(\"OPENAI_API_KEY\") 从环境变量读取"
    },
    {
      "level": "major",
      "category": "security",
      "clause": "规则1.2 禁止暴露内部端口",
      "evidence": "代码第8行：app.run(host=\"0.0.0.0\", port=8080)",
      "suggestion": "改为 host=\"127.0.0.1\"，通过nginx反向代理对外暴露"
    },
    {
      "level": "minor",
      "category": "deployment",
      "clause": "规则2.2 显存最低要求标注",
      "evidence": "部署文档未包含显存要求标注",
      "suggestion": "补充注释：# 注意：Llama-2-7b FP16需要至少14GB显存"
    }
  ],
  "summary": "检测到3处违规：1个critical（硬编码密钥）、1个major（端口暴露）、1个minor（显存标注缺失）"
}
```

**字段说明**：
- `violations`：违规数组，每项含：
  - `level`：`"critical"|"major"|"minor"`
  - `category`：`"security"|"deployment"|"privacy"`
  - `clause`：违反的具体条款（如"规则1.1 禁止硬编码API密钥"）
  - `evidence`：违规证据（代码片段+位置）
  - `suggestion`：修复建议
- `summary`：人类可读的违规汇总

**与TechnicalReviewResult的映射**：violations数组的每项映射为ReviewError，其中 `error_type` 取clause的简写（如 `"hardcoded_secret"`），`severity` 取level，`description` 取evidence，`evidence` 取clause，`suggestion` 取suggestion。

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 新规范未覆盖（出现规则库未定义的违规模式） | 归入"待评估"类别，category=`"pending_evaluation"`，level=`"minor"`，记录日志供规则库更新，不阻断发布 |
| 规则匹配引擎误报（正则命中但语义上合规，如示例代码中明确标注"仅演示用"） | 允许人工复核，在violations中标注 `"note": "疑似误报，建议人工复核"`，level降级为minor |
| 内容中无代码块/配置/命令（纯文本教程） | 跳过安全/部署规范检查，仅检查数据隐私相关文字描述，summary中注明"未检测到代码块" |
| 许可证无法识别（模型名不在已知清单中） | 标记level=`"minor"`，clause=`"规则2.1 许可证未知"`，suggestion建议作者明确标注许可证类型 |
| 显存公式无法套用（非标准模型结构，如MoE） | 跳过显存估算检查，仅检查是否包含显存标注关键词，summary中注明"显存估算不适用MoE模型" |

## 6. 附录

### 附录A：参考标准
| 编号 | 标准名称 | 作者/机构 | 年份 | 具体章节/贡献说明 |
|:---|:---|:---|:---|:---|
| A-1 | OWASP ML Top 10 2023 | Open Worldwide Application Security Project / ML Security Project | 2023 | 列出机器学习系统十大安全风险。本SKILL规则1（安全规范）的硬编码密钥检测对应ML06:2023 Sensitive Information Disclosure，端口暴露对应ML07:2023 Model Inversion Attack的攻击面扩展，权限缺失对应ML04:2023 Model Extraction的未授权访问 |
| A-2 | 各主流模型许可证汇总表 | 各模型发布方（Meta/HuggingFace/OpenAI等） | 2023-2024 | 汇总MIT、Apache 2.0、GPL v3、Llama 2/3 Community License、OpenAI API Terms 的可商用性条款。本SKILL规则2.1的许可证清单与检查逻辑直接引用此汇总表，明确区分"可商用"、"需开源衍生作品"、"需申请授权"、"不可作为自有模型分发"四类 |
| A-3 | ISO/IEC 27001:2022 信息安全管理体系标准 | International Organization for Standardization / IEC | 2022 | 附录A控制措施清单中的A.5.12（信息分类）、A.5.34（隐私与PII保护）、A.8.9（配置管理）为本SKILL规则3（数据隐私）的脱敏要求、规则1.1的密钥管理、规则2.2的配置标注提供合规依据 |

### 附录B：完整场景示例

#### 场景1：硬编码密钥检测（Critical违规）
**输入**：
```json
{
  "content": "```python\nimport openai\n\n# 配置API密钥\nAPI_KEY = \"sk-abcdef1234567890abcdef1234567890\"\nopenai.api_key = API_KEY\n\n# 调用模型\nresponse = openai.ChatCompletion.create(\n    model=\"gpt-3.5-turbo\",\n    messages=[{\"role\": \"user\", \"content\": \"你好\"}]\n)\nprint(response.choices[0].message.content)\n```\n\n以上是调用OpenAI API的示例代码。"
}
```

**处理过程**：
1. 步骤1提取：识别代码块，提取配置项 `API_KEY = "sk-abcdef..."` 和API调用 `openai.ChatCompletion.create`
2. 步骤2匹配：正则 `API_KEY\s*=\s*["']sk-` 命中规则1.1
3. 步骤3分级：硬编码密钥→Critical
4. 步骤4生成报告

**输出**：
```json
{
  "violations": [
    {
      "level": "critical",
      "category": "security",
      "clause": "规则1.1 禁止硬编码API密钥",
      "evidence": "代码第4行：API_KEY = \"sk-abcdef1234567890abcdef1234567890\"，密钥以明文形式硬编码在源码中",
      "suggestion": "改为使用 os.getenv(\"OPENAI_API_KEY\") 从环境变量读取，并将密钥存入.env文件（加入.gitignore），避免泄露到代码仓库"
    }
  ],
  "summary": "检测到1处critical违规：硬编码OpenAI API密钥。建议立即修订，密钥已泄露需在OpenAI控制台轮换。"
}
```

#### 场景2：许可证缺失检测（Minor违规）
**输入**：
```json
{
  "content": "## 部署Llama-2-7B模型\n\n以下是部署Llama-2-7B模型的步骤：\n\n```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nimport torch\n\nmodel = AutoModelForCausalLM.from_pretrained(\n    \"meta-llama/Llama-2-7b-hf\",\n    torch_dtype=torch.float16,\n    device_map=\"auto\"\n)\ntokenizer = AutoTokenizer.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n\ntry:\n    inputs = tokenizer(\"你好\", return_tensors=\"pt\").to(model.device)\n    outputs = model.generate(**inputs, max_new_tokens=50)\n    print(tokenizer.decode(outputs[0]))\nexcept Exception as e:\n    print(f\"推理失败：{e}\")\n```\n\n部署完成后即可对外提供服务。"
}
```

**处理过程**：
1. 步骤1提取：识别代码块，提取模型名 `meta-llama/Llama-2-7b-hf` 和部署说明"部署完成后即可对外提供服务"
2. 步骤2匹配：
   - 规则1.1硬编码密钥：未命中（无API_KEY）
   - 规则1.2端口暴露：未命中（无app.run）
   - 规则1.3权限校验：未命中（非Web API）
   - 规则2.1许可证合规：模型为Llama-2 Community License，部署示例未标注许可证类型，命中Minor违规
   - 规则2.2显存标注：未包含显存关键词（"显存"、"VRAM"），命中Minor违规
   - 规则3数据隐私：未涉及用户数据，未命中
3. 步骤3分级：许可证缺失→Minor，显存标注缺失→Minor
4. 步骤4生成报告

**输出**：
```json
{
  "violations": [
    {
      "level": "minor",
      "category": "deployment",
      "clause": "规则2.1 模型许可证合规",
      "evidence": "部署文档引用 meta-llama/Llama-2-7b-hf 但未标注许可证类型，Llama 2 为社区许可（月活>7亿需Meta授权）",
      "suggestion": "补充许可证说明：# 模型许可证：Llama 2 Community License（商用需注意月活用户数限制，>7亿需向Meta申请授权）"
    },
    {
      "level": "minor",
      "category": "deployment",
      "clause": "规则2.2 显存最低要求标注",
      "evidence": "部署文档未包含显存要求标注，Llama-2-7B FP16按公式7×2×1.2≈16.8GB，最低需14GB显存",
      "suggestion": "补充注释：# 注意：Llama-2-7b FP16需要至少14GB显存（推荐16GB以上），INT8量化版本需8GB"
    }
  ],
  "summary": "检测到2处minor违规：许可证说明缺失、显存要求标注缺失。代码本身合规（无硬编码、有异常处理），建议补充文档说明后即可通过。"
}
```

**说明**：场景2体现了Minor违规不阻断通过的特性——代码本身已合规（使用device_map而非.cuda()、包含try-except异常处理），仅文档说明不完整，建议补充即可，不阻止内容发布。

## 7. 版本信息
- 版本：v3.0，更新日期：2026-07-23
- 变更记录：核心方法论扩展为4步（提取→匹配→分级→报告）；规则1安全规范细化3个子项（1.1硬编码密钥含3种正则模式、1.2端口暴露含0.0.0.0检测、1.3权限校验含@login_required/Depends检测），每项含错误示例和正确示例；规则2部署规范新增许可证清单表（MIT/Apache/GPL/Llama2/Llama3/GPT系5类）和显存计算公式（参数量×精度×1.2，含6个常见模型估算表）；规则3数据隐私新增脱敏方式表（user_id/phone/email/id_card 4类字段脱敏规则）和传输加密检测；输入输出规范新增violations数组完整JSON示例（含level/category/clause/evidence/suggestion字段）；异常处理表新增5种场景（新规范未覆盖、误报、无代码块、许可证未知、MoE模型）；附录A扩充至3个参考标准（OWASP ML Top 10 2023、模型许可证汇总表、ISO/IEC 27001:2022）；附录B新增2个完整场景示例（硬编码密钥Critical+许可证缺失Minor）
