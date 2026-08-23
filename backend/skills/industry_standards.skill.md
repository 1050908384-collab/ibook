# SKILL名称：industry_standards

## 1. 能力描述
- **功能定位**：Agent④技术审核的**子检查模块**（Sub-module）。专注于**大模型微调（LoRA/QLoRA）与推理部署（vLLM）领域特有**的行业红线规则检查。与 `technical_review` 配合，`technical_review` 负责总调度和最终裁决，本SKILL负责领域合规性检查并返回 `ReviewError` 列表
- **适用场景**：当 `technical_review` 调用 `_run_industry_standards_check()` 子流程时触发，检查学习内容中的代码/配置/命令是否违反大模型微调与部署的特定规范
- **前置依赖**：待审核内容（`ReviewInput.content`）、大模型微调与部署的规范规则库

## 2. 核心方法论
- **步骤1：提取领域特有信息**——从内容中识别：
  - LoRA/QLoRA配置代码（`r=`、`lora_alpha=`、`target_modules=`）
  - vLLM部署命令（`vllm serve`、`--model`、`--gpu-memory-utilization`、`--port`）
  - 模型加载/推理代码（`from_pretrained`、`device_map`、`quantize_config`）
- **步骤2：逐项匹配领域红线规则**——应用规则1（LoRA/QLoRA安全）、规则2（vLLM部署安全）、规则3（数据与许可证）进行正则/关键词匹配
- **步骤3：生成标准化ReviewError**——将违规检测结果转换为 `ReviewError` 对象（`error_type`、`severity`、`location`、`description`、`evidence`、`suggestion`），与 `technical_review` 的错误列表格式完全一致
- **步骤4：返回ReviewError列表**——将违规列表返回给 `technical_review`，由其统一合并到 `TechnicalReviewResult.errors` 中

## 3. 知识/规则定义

### 规则1：LoRA/QLoRA微调安全规范
#### 1.1 LoRA秩（r）值安全
**检测模式**：
- 正则：`lora_config.*r\s*=\s*(\d+)` 或 `r\s*=\s*(\d+)`（在LoraConfig上下文）

**判定逻辑**：
| r值范围 | 判定 | 说明 |
|:---|:---|:---|
| r ≥ 64 | ⚠️ 警告（Minor） | r值过大容易导致过拟合，建议r∈[2, 32]。触发 error_type=`"lora_r_too_high"`，severity=`"minor"` |
| r ≤ 1 | ❌ 错误（Critical） | r=1时LoRA退化为原始矩阵，失去低秩适应性意义。触发 error_type=`"lora_r_invalid"`，severity=`"critical"` |
| r ∈ [2, 32] | ✅ 正常 | 推荐的参数高效微调区间 |

**错误示例**：
```python
# 违规：r=64 过大
config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj"])
```

**正确示例**：
```python
# 合规：r=16 推荐值
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
```

#### 1.2 QLoRA量化精度安全
**检测模式**：
- 正则：`bitsandbytes.*quantize.*config` 或 `QuantizationConfig` 上下文

**判定逻辑**：
| 量化精度 | 判定 | 说明 |
|:---|:---|:---|
| load_in_4bit=True + compute_dtype=bfloat16 | ✅ 正常 | QLoRA推荐配置 |
| load_in_8bit=True + compute_dtype=float16 | ✅ 正常 | 标准8位量化 |
| load_in_4bit=True + compute_dtype=float32 | ⚠️ 警告 | 量化与计算精度不匹配，可能导致数值溢出。触发 error_type=`"quantize_mismatch"`，severity=`"major"` |

#### 1.3 PEFT配置合法性
**检测模式**：
- 检查 `target_modules` 有效性：正则 `target_modules\s*=\s*\[(.*?)\]`

**判定逻辑**：
- 若 `target_modules` 包含不存在的层（如`["embedding", "pooling"]`），触发 error_type=`"peft_invalid_target"`，severity=`"major"`
- 若 `target_modules` 为空列表 `[]`，触发 error_type=`"peft_empty_target"`，severity=`"critical"`

### 规则2：vLLM推理部署安全规范
#### 2.1 显存分配参数安全
**检测模式**：
- 正则：`gpu.memory.utilization\s*=\s*([0-9.]+)` 或 `--gpu-memory-utilization\s+([0-9.]+)`

**判定逻辑**：
| gpu_memory_utilization值 | 判定 | 说明 |
|:---|:---|:---|
| > 0.95 | ❌ 错误（Critical） | 显存占用过高可能导致OOM，建议≤0.90。触发 error_type=`"vllm_vram_overallocation"`，severity=`"critical"` |
| > 0.90 | ⚠️ 警告（Major） | 显存占用偏高，存在OOM风险。触发 error_type=`"vllm_vram_high"`，severity=`"major"` |
| 0.80 ~ 0.90 | ✅ 正常 | 推荐范围 |
| < 0.50 | ⚠️ 警告（Minor） | 显存利用过低，效率未达最优。触发 error_type=`"vllm_vram_low"`，severity=`"minor"` |

**错误示例**：
```python
# 违规：显存占用过高
llm = LLM(model="meta-llama/Llama-2-7b", gpu_memory_utilization=0.98)
```

**正确示例**：
```python
# 合规：显存占用合理
llm = LLM(model="meta-llama/Llama-2-7b", gpu_memory_utilization=0.85)
```

#### 2.2 端口配置安全
**检测模式**：
- 正则：`port\s*=\s*(\d+)` 或 `--port\s+(\d+)`

**判定逻辑**：
- 端口为 80/443/8080/8000 等常见端口 → 触发 error_type=`"vllm_port_conflict"`，severity=`"major"`（在生产环境中存在端口冲突风险，建议使用10000+高位端口）
- 端口 < 1024 → 触发 error_type=`"vllm_port_privileged"`，severity=`"major"`（需要root权限）

#### 2.3 模型权重完整性校验
**检测模式**：
- 检查是否加载 `.safetensors` 权重文件或通过 `from_pretrained` 加载
- 正则：`\.bin["']` 引用PyTorch旧格式权重

**判定逻辑**：
- 使用 `.bin` 格式而非 `.safetensors` → 触发 error_type=`"model_weight_format"`，severity=`"minor"`（建议迁移至安全格式）
- 使用本地路径加载模型但路径下无权重文件检测 → 触发 error_type=`"model_weight_missing"`，severity=`"critical"`

### 规则3：数据隐私与模型许可证
#### 3.1 训练数据脱敏
**检测模式**：
- 正则：`dataset\.load_dataset\(.+)` 或 `read_csv\(.+)` 上下文
- 关键词：`user_id`、`phone`、`email`、`personal`、`敏感`

**判定逻辑**：
- 加载涉及用户个人信息的数据集且未标注脱敏处理 → 触发 error_type=`"data_privacy"`，severity=`"critical"`
- 数据预处理步骤中无脱敏操作（如`dropna`而非`anonymize`） → 触发 error_type=`"data_desensitization_missing"`，severity=`"major"`

#### 3.2 模型许可证合规
**检测模式**：
- 检测 `from_pretrained` 中加载的模型名

**判定逻辑**：
| 模型许可证 | 判定 |
|:---|:---|
| Llama-2/3 Community License | ⚠️ 建议（Minor）：标注"商用需检查月活用户数（>7亿需Meta授权）" |
| GPT系（仅API） | ❌ 错误（Critical）：不可作为自有模型商用部署，仅可通过API调用 |
| Apache 2.0 / MIT | ✅ 正常 |
| GPL v3 | ⚠️ 建议（Major）：商用部署需开源衍生作品 |

## 4. 输入输出规范

### 输入
**与 `technical_review` 完全一致**，使用 `ReviewInput` 对象：
- `content`：待审核内容字符串（tutorial、answer或code示例）
- `user_profile`：用户画像（UserProfile）
- `context`：可选上下文
- `target_type`：`"learning_package"` 或 `"answer"`

### 输出
**标准化ReviewError数组**，与 `technical_review` 的错误列表格式完全一致：
```json
[
  {
    "error_type": "lora_r_too_high",
    "severity": "minor",
    "location": "LoRA配置行",
    "description": "LoRA秩r=64过大，容易导致过拟合。",
    "evidence": "规则1.1 LoRA秩安全：推荐r∈[2,32]，r≥64触发过拟合警告。",
    "suggestion": "将r值调整为8~32之间，如r=16。"
  },
  {
    "error_type": "vllm_vram_overallocation",
    "severity": "critical",
    "location": "LLM初始化行",
    "description": "vLLM显存分配率0.98过高，可能导致OOM。",
    "evidence": "规则2.1 显存分配安全：gpu_memory_utilization>0.95触发critical。",
    "suggestion": "将gpu_memory_utilization调整为≤0.90。"
  }
]
```

**字段说明**：每个ReviewError包含：
- `error_type`：违规类型标识（如`lora_r_too_high`、`vllm_vram_overallocation`）
- `severity`：`"critical"` / `"major"` / `"minor"`
- `location`：违规在内容中的位置
- `description`：违规描述
- `evidence`：违反的规则条款及依据
- `suggestion`：修复建议

**与TechnicalReviewResult的关系**：本SKILL返回的ReviewError数组将由 `technical_review` 的步骤3合并到 `TechnicalReviewResult.errors` 中。

## 5. 异常处理
| 异常场景 | 处置方式 |
|:---|:---|
| 内容中无LoRA/QLoRA/vLLM相关代码 | 跳过领域规则检查，返回空ReviewError数组，标注reason="未检测到微调/部署相关代码" |
| 规则正则匹配引擎超时 | 跳过该条规则检查，在返回结果中追加 `error_type="rule_check_timeout"`（severity=minor），标注"建议人工复核" |
| 模型名无法识别许可证 | 追加 `error_type="license_unknown"`（severity=minor），建议人工确认许可证 |
| PEFT配置字段拼写错误（如`target_modules`拼写为`target_moduels`） | 追加 `error_type="peft_typo"`（severity=critical），提示常见拼写错误 |
| 内容审核循环达上限（专业知识与教学质量审核 Agent统一审核循环≥3次仍未通过） | Orchestrator强制终止循环，本子模块返回的ReviewError仍合并到TechnicalReviewResult，但整体status=`"review_needed"`，summary标注"审核循环达3次上限，建议人工介入修订内容" |

## 6. 附录

### 附录A：通用Web安全扩展参考（非核心规则）
> **说明**：以下通用Web安全规则不属于当前项目核心范围，仅作为扩展参考。若学习内容涉及完整的Web应用开发，可参考执行。

| 规则类别 | 检测模式 | 违规级别 | 说明 |
|:---|:---|:---|:---|
| 禁止硬编码API密钥 | `api_key\s*=\s*["']sk-` | critical | 通用安全 |
| 禁止暴露内部端口 | `host\s*=\s*["']0\.0\.0\.0["']` | major | 通用Web安全 |
| 必须包含权限校验 | `@login_required`缺失（用户API） | major | 通用Web安全 |
| 数据传输加密 | `http://`开头（非localhost） | major | 通用安全 |

### 附录B：领域规则完整场景示例

#### 场景1：LoRA+QLoRA配置审核
**输入**：
```
以下是QLoRA微调的代码示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# QLoRA配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # 注意：这里使用了float32
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config
)

# LoRA配置 - r=128
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
```
```

**处理过程**：
1. 步骤1提取：识别QLoRA配置、LoRA配置
2. 步骤2匹配：
   - 规则1.1：正则匹配`r=128` → r≥64 → 警告
   - 规则1.2：正则匹配`bnb_4bit_compute_dtype=torch.float32` → 量化不匹配 → major
   - 规则1.3：target_modules包含7个层 → 合法
3. 步骤3：生成ReviewError数组

**输出**：
```json
[
  {
    "error_type": "lora_r_too_high",
    "severity": "minor",
    "location": "LoRA配置行",
    "description": "LoRA秩r=128过大，容易导致过拟合。",
    "evidence": "规则1.1 LoRA秩安全：推荐r∈[2,32]，r≥64触发过拟合警告。",
    "suggestion": "将r值调整为8~32之间，如r=16。"
  },
  {
    "error_type": "quantize_mismatch",
    "severity": "major",
    "location": "BitsAndBytesConfig行",
    "description": "QLoRA使用load_in_4bit量化，但compute_dtype设为float32，存在数值溢出风险。",
    "evidence": "规则1.2 QLoRA量化安全：4位量化推荐搭配bfloat16或float16计算精度。",
    "suggestion": "将bnb_4bit_compute_dtype改为torch.bfloat16或torch.float16。"
  }
]
```

#### 场景2：vLLM部署命令审核
**输入**：
```
以下是vLLM部署的代码示例：
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-chat",
    gpu_memory_utilization=0.98,
    port=8080
)

params = SamplingParams(temperature=0.7, max_tokens=1024)
output = llm.generate(["你好"], params)
```
```

**处理过程**：
1. 步骤1提取：识别LLM初始化配置
2. 步骤2匹配：
   - 规则2.1：正则匹配`gpu_memory_utilization=0.98` → >0.95 → critical
   - 规则2.2：正则匹配`port=8080` → 常见端口 → major
3. 步骤3：生成ReviewError数组

**输出**：
```json
[
  {
    "error_type": "vllm_vram_overallocation",
    "severity": "critical",
    "location": "LLM初始化行",
    "description": "vLLM显存分配率0.98过高，可能导致OOM崩溃。",
    "evidence": "规则2.1 显存分配安全：gpu_memory_utilization>0.95触发critical。",
    "suggestion": "将gpu_memory_utilization调整为0.85~0.90之间。"
  },
  {
    "error_type": "vllm_port_conflict",
    "severity": "major",
    "location": "端口配置行",
    "description": "使用端口8080为常见端口，在生产环境中存在冲突风险。",
    "evidence": "规则2.2 端口安全：建议使用10000+高位端口避免冲突。",
    "suggestion": "将port改为18080或其他高位端口。"
  }
]
```

## 7. 版本信息
- 版本：v3.0，更新日期：2026-07-23
- 变更记录：
  - **范围聚焦**：从通用Web安全收窄到LoRA/QLoRA+vLLM领域特有规则。删除端口暴露/权限校验/HTTPS加密等通用规则，降级为附录A扩展参考
  - **新增领域规则**：规则1.1（LoRA秩r值安全，含≥64过拟合警告和≤1错误阈值）、规则1.2（QLoRA量化精度匹配检查）、规则1.3（PEFT配置合法性检查）、规则2.1（vLLM显存分配安全，分四档阈值）、规则2.2（vLLM端口冲突检测）、规则2.3（模型权重格式和完整性校验）、规则3.1（训练数据脱敏检测）、规则3.2（模型许可证合规）
  - **I/O统一**：输入改为ReviewInput（与technical_review一致），输出改为标准化ReviewError数组（与technical_review错误列表格式完全一致）
  - **架构定位**：明确标注为technical_review的子检查模块，输出直接合并到TechnicalReviewResult.errors
  - **附录示例**：新增2个完整领域场景（LoRA+QLoRA配置审核、vLLM部署命令审核）
  - **v3.1变更**：新增"审核循环≥3次"异常处置路径
- **与代码对齐说明**：当前`HybridTechnicalReviewAgent`直接加载本SKILL为`skill_context`。v3.0明确本SKILL输出为ReviewError列表，后续代码可增加`_run_industry_standards_check()`子流程调用本SKILL的规则检查，将返回的ReviewError合并到最终errors中
