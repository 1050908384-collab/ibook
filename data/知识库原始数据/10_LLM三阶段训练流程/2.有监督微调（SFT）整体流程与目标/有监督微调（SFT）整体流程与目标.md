# 有监督微调（SFT）：让基座模型学会"听懂话、办好事"

> **参考文章链接**
> 
> ① 底稿：《SFT微调过程》
> 　　网址：https://blog.csdn.net/jack_wgj/article/details/147093720
> 
> ② 参考文章1：《垂域微调实践：打造金融企业年报分析助手》
> 　　网址：https://blog.csdn.net/m0_59164520/article/details/147822058
> 
> ③ 参考文章2：《一文搞懂大模型预训练PT与有监督微调SFT》
> 　　网址：https://blog.csdn.net/Python_cocola/article/details/141070679

---

**适合人群**：已掌握预训练（Pretrain）流程，希望理解如何将"会接话"的基座模型转化为"会办事"的指令模型的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将掌握 SFT 的完整流程、数据构建方法、训练配置与评估指标，为后续"RLHF 对齐人类偏好"打下坚实基础。

**预计学习时长**：约 1.5 小时  
**环境准备**：Python 3.10+，基础依赖 `transformers`、`peft`、`datasets`、`accelerate`；若要用 QLoRA 还需 `bitsandbytes`、`auto-gptq`/`autoawq`（视量化方案选其一）。

```bash
# 最低可跑通依赖（LoRA + BF16 或 FP16）
pip install transformers peft datasets accelerate

# 要使用 QLoRA（4-bit/8-bit 量化，大幅降低显存门槛）再加以下任一：
pip install bitsandbytes                            # 8-bit / NF4-QLoRA（最通用）
# pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # GPTQ 量化方案
# pip install autoawq                                                         # AWQ 量化方案
```

> **硬件门槛与适配人群提示（SFT 的显存/并行策略参照表）**：
> - **只学概念/不跑训练**：本章第 1~2 章、第 6 章（常见陷阱）**完全不需要 GPU**，直接阅读就能搞清楚 SFT 数据构造、prompt-only loss 屏蔽、评估思路这些核心知识；第 3~5 章代码也可以当教学伪代码看。这是绝大多数入门学习者的首选路径。
> - **想跑训练但只有消费级 GPU（1 张 8GB~12GB 显存，如 RTX 3060/3080/4060/4070）**：只适合跑 **≤2B~3B 小模型的 QLoRA/4-bit LoRA**，MAX_LENGTH 限制在 512~1024，batch_size 要很小（1~2），可以加 gradient_accumulation_steps 来凑等效 batch。7B 级别基本无法在此档显存下跑通。
> - **有单卡 16GB~24GB（如 RTX 3090/4090、A10/A5000）**：可以用 **QLoRA/4-bit** 跑 **7B~14B 模型**，MAX_LENGTH 可开到 2048~4096；BF16 LoRA 可稳跑 7B，batch_size 视 MAX_LENGTH 在 1~8 区间调整。这是个人用户做 SFT 实操的**甜点档**。
> - **有单机 4~8 卡（A100 40G/80G、H100 等）**：可以用 **BF16 LoRA** 稳跑 **7B~70B 级别**；7B~13B 还能做 Full Fine-Tuning（全参微调）。推荐用 accelerate launch 的多卡模式或 DeepSpeed ZeRO-2/ZeRO-3 启动。
> - **没 GPU / 没达到上述档位**：有两条替代路线——（1）用 AutoDL、恒源云、RunPod、Lambda 等按量 GPU 云平台临时租卡；（2）把模型改成 0.5B/1.5B 级别玩具模型，只做"能跑通 loss 下降"的教学验证，不追求模型最终质量。
> - **启动方式提醒**：训练代码里不要写 device_map="auto"，要用 `accelerate launch` / `deepspeed` 命令启动脚本，让并行框架自己分配设备；精度显式指定 bf16=True 或 fp16=True（不要 torch_dtype="auto"）。

---

## 第一章：SFT 的定位——承上启下的关键一步

### 1.1 三阶段训练中的 SFT

大语言模型从"出生"到"上岗"，通常经历三个阶段：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  预训练 PT   │ → │  有监督微调 SFT │ → │   对齐 RLHF  │
│ Pre-training│    │ Supervised  │    │ Reinforcement│
│             │    │ Fine-Tuning │    │ from Human   │
└─────────────┘    └─────────────┘    └─────────────┘
     ↓                  ↓                  ↓
学习通用语言能力      学习指令遵循能力      对齐人类价值观
（读遍互联网文本）    （学习问答对话格式）   （安全、有用、诚实）
```

**SFT 的核心定位**：
- **承上**：利用预训练模型学到的通用语言知识
- **启下**：为 RLHF 提供"会遵循指令"的基础模型
- **核心作用**：让模型学会解析任务指令，并在垂直领域输出专业、准确的响应

### 1.2 SFT 与预训练的本质区别

| 维度 | 预训练（PT） | 有监督微调（SFT） |
|------|-------------|------------------|
| **目标** | 学习语言统计规律 | 适配特定任务指令 |
| **数据** | TB 级无标注文本 | 1万~10万条高质量标注样本 |
| **数据格式** | 纯文本序列 | 指令-输入-输出三元组 |
| **训练复杂度** | 极高（千卡级集群，数周） | 中等（单机多卡，数小时~数天） |
| **典型硬件** | 64×A100 | 4~8×A100/H100 |
| **输出能力** | 文本续写 | 遵循指令、对话、问答 |

**为什么这样写**：预训练让模型"识字"，SFT 让模型"懂事"。没有 SFT，基座模型只会做"自动补全"，无法理解"请翻译""请总结"这类指令。

---

## 第二章：SFT 数据构建——质量大于数量

### 2.1 数据格式：指令-输入-输出

SFT 的核心数据格式是**指令-输入-输出（Instruction-Input-Output）**三元组：

```json
{
  "instruction": "请计算华为2023年研发投入占比",
  "input": "华为2023年年报：研发支出1647亿元，总收入6423亿元。",
  "output": "研发投入占比 = 研发支出 ÷ 总收入 × 100% = (1647 ÷ 6423) × 100% ≈ 25.65%。"
}
```

**为什么这样写**：
- `instruction`：告诉模型"要做什么"（任务描述）
- `input`：具体的输入内容（可为空，如纯问答任务）
- `output`：期望模型生成的标准答案
- **⚠️ 指令与答案必须严格对应**：上面的问题问"研发投入占比"，答案就计算"研发投入占比"。如果指令问的是"营收增长率"，答案却去算"研发投入占比"，就属于**样本错配**——问题与答案不是一回事。这种数据会教坏模型：让它学会"答非所问"。构造时务必逐条核对指令、输入与输出在语义与数值口径上一致。

### 2.2 数据构建流程

```
原始数据（PDF/网页/文档）
    ↓
文本抽取 & OCR 清洗（去除页眉页脚、纠正识别错误）
    ↓
关键信息标注（人工或半自动）
    ↓
指令-答案对生成（自动+人工校对）
    ↓
JSONL 格式的 SFT 数据集
```

**为什么这样写**：SFT 数据的质量直接决定微调效果。金融、医疗、法律等垂直领域尤其需要**专业、准确、合规**的标注数据。

### 2.3 数据增强技巧

| 技巧 | 说明 | 示例 |
|------|------|------|
| **同义改写** | 自动工具+人工扩写，保证问法多样性 | "华为营收多少？"→"华为2023年收入？" |
| **纠错式样本（正确示范）** | 把"错误问法/含错前提"作为输入，但 `output` 仍是**正确**的纠正与解答 | 输入含错误数字，output 指出错误并给出正确答案 |
| **推理链引导（CoT）** | 要求模型先展示推理过程 | "请先列出计算公式，再计算..." |
| **上下文扩展** | 多段信息的问题提供完整上下文 | 对比多家公司时提供完整表格 |

> **⚠️ 不要把"错误答案"直接当作 `output` 混入训练**：SFT 是**模仿学习**——模型会去拟合、复现 `output` 里的内容。如果把错的答案放进 `output`，等于在**教模型输出错误**，而不是"让它学会甄别错误"。若想提升模型的纠错/鉴别能力，正确做法是：把"错误"放在**输入侧**（如带错误前提的问题、需要被反驳的说法），而 `output` 始终是**正确的纠正与解答**；模型对错误答案本身的偏好抑制应交给后续的偏好对齐（RLHF/DPO，用"选/拒"成对数据），而非 SFT 的监督目标。

### 2.4 数据量与质量的关系

**关键结论**：SFT 是**质量远大于数量**的任务。

- 通用领域：1万~5万条高质量样本即可看到显著提升
- 垂直领域：甚至 5000 条精心标注的样本就能超过 10 万条低质量样本
- 混合策略：保留 10%~20% 通用指令数据，防止**灾难性遗忘**

**为什么这样写**：预训练需要"海量数据+简单任务"，SFT 需要"精准数据+明确目标"。少量高质量样本足以让模型学会任务模式。

---

## 第三章：SFT 训练流程——从加载到保存

### 3.1 完整训练流程概览

```
选择预训练模型 → 加载模型权重 → 准备 SFT 数据 → 配置训练参数
        ↓
    定义数据加载器 → 定义损失函数和优化器 → 训练模型
        ↓
    验证评估 → 测试模型 → 保存和部署
```

### 3.2 加载预训练模型与训练启动方式

> ⚠️ **不要在训练脚本中用 `device_map="auto"` 做多卡训练**：`device_map` 主要用于**推理/加载阶段**的单层分片，它与 `Trainer`、DDP、DeepSpeed 的数据并行训练逻辑常互相冲突（例如参数被预分片后，并行框架无法再正确分配梯度、优化器状态）。**多卡训练应使用 `accelerate launch` 或 `deepspeed` 启动脚本**，让并行框架自行管理设备。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# =====================================================================
# 两种常见 SFT 起点，请按你的流程对应选择：
#
# 【选项 A：标准三阶段流程中的 SFT】← 本章默认讲解的场景
#   起点是【基础模型（Base）】，即"预训练阶段输出、还没经过任何指令/对话对齐"的版本。
#   本章 3.3 的 chat template + chat SFT 数据，就是把"只会续写文本的 Base"教成"会按指令回答"。
BASE_MODEL = "Qwen/Qwen2.5-7B"           # Base：没有 Instruct 后缀
#
# 【选项 B：二次 SFT / 领域适配（Domain Adaptation）】
#   起点是【已经做好通用指令对齐的 Instruct/Chat 模型】，
#   想进一步在某个垂直领域（医疗/法律/代码企业内部风格）上强化能力。
#   这种做法合法、很常见，但属于"在已有对齐模型上继续微调"，
#   不是三阶段流程里严格意义上的"从 Base 做第一轮 SFT"。
# INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
#
# ⚠️ 两种场景的 chat SFT 数据策略不同：
#    - 选项 A 需要包含完整的"指令遵循 + 角色扮演"等通用对齐能力样本；
#    - 选项 B 建议以领域专业样本为主，同时混入少量（约 10%~20%）通用 SFT 数据防灾难性遗忘。
# =====================================================================
model_name = BASE_MODEL   # 或改为 INSTRUCT_MODEL

# 加载模型和分词器：训练时【显式指定精度】，不要依赖 "auto"
# - torch_dtype="auto" 会按 checkpoint 的 config.torch_dtype 加载，
#   不是按 A100/H100 显卡型号自动选；实际训练精度需要与
#   TrainingArguments 中的 bf16=True / fp16=True 对应。
# - 不写 device_map：让 accelerate / DeepSpeed / Trainer 管理设备
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,   # 显式指定，与 TrainingArguments.bf16 保持一致
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
```

**正确的多卡启动方式（命令行，不要在 Python 脚本里手工 device_map）**：

```bash
# 方式一：用 accelerate（推荐，统一封装单卡/多卡/DeepSpeed）
accelerate config          # 首次：交互式配置并行策略、精度、DeepSpeed 等
accelerate launch train_sft.py

# 方式二：直接用 DeepSpeed（配合 ds_config.json）
deepspeed --num_gpus=8 train_sft.py \
  --deepspeed ds_config.json

# 方式三：单机多卡 DDP（不写 device_map）
torchrun --nproc_per_node=8 train_sft.py
```

**为什么这样写**：
- `torch_dtype=torch.bfloat16`：**显式指定**权重加载/运算精度，与 `TrainingArguments.bf16=True` 保持一致，避免 checkpoint 配置和运行环境不一致导致的 dtype 错配。若确需使用 float16，就传 `torch.float16` 并同时打开 `TrainingArguments.fp16=True`。
- **不设置 device_map**：在训练场景下，并行策略（DDP/DeepSpeed ZeRO/FSDP）会接管参数、梯度、优化器状态的切分与放置；提前写 `device_map="auto"` 等于绕过并行框架做了一层分片，后续容易出现参数不在预期设备、优化器找不到参数、DDP 进程间权重不同步等问题。
- 推理/演示场景（单脚本本地加载看输出）可以临时用 `device_map="auto"`，但**训练脚本不要这么做**。

### 3.3 数据加载与格式化

```python
from datasets import load_dataset

# 加载自定义 JSONL 数据集
# 格式: {"instruction": "...", "input": "...", "output": "..."}
dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

IGNORE_INDEX = -100       # cross_entropy 默认忽略 -100，用来屏蔽不该算 loss 的位置
MAX_LENGTH = 4096         # 与 TrainingArguments 的 max_length / 模型上下文窗口对齐，避免爆显存

def format_and_tokenize(example):
    """用【目标模型自带的 chat template】拼接对话，并只对 assistant 回复计算 loss"""
    user_content = example["instruction"]
    if example["input"]:
        user_content += "\n\n" + example["input"]

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]

    # 关键 1：用 tokenizer.apply_chat_template，套用【该模型训练时真正使用的对话格式】，
    #         而不是自己手写 "### 指令：/### 回答："。不同模型（Qwen/Llama/ChatGLM...）的
    #         特殊 token、角色标记、系统位都不同，手写格式会与模型预期不一致，损害效果。
    # 关键 2：设置 MAX_LENGTH + truncation=True，超长样本在 tokenizer 阶段就截断，
    #         避免后续 collator 拼接 batch 时因序列过长导致显存暴涨或报错。
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # 再单独算出 "prompt 部分"（到 assistant 开始生成之前）的长度，用于屏蔽 prompt 的 loss。
    # prompt 同样要截断到 MAX_LENGTH（或更小，保证 prompt_len <= len(full_ids)）。
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    prompt_len = min(len(prompt_ids), len(full_ids))

    # labels 与 input_ids 等长；prompt 段落用 IGNORE_INDEX 屏蔽，只保留 assistant 回复参与 loss
    # 若被截断导致截断点落在 assistant 段内，截断之后的 token 自然会被忽略。
    labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
    return {"input_ids": full_ids, "labels": labels}

# batched=False 保持一条一条处理；如果你想进一步提速，可以改成 batched=True
# 并在函数内按批处理 apply_chat_template。
tokenized_dataset = dataset.map(
    format_and_tokenize, remove_columns=dataset.column_names, batched=False
)
```

**为什么这样写**：
- **必须使用目标模型的 chat template**（`tokenizer.apply_chat_template`）：SFT 要让模型在**它未来推理时真正会遇到的对话格式**下学习。手写 `### 指令：/### 回答：` 与模型实际的角色标记、特殊 token 不一致，会造成训练/推理格式错配，效果打折。
- **显式设置 MAX_LENGTH + truncation=True**：长样本（instruction+input+output 总和超过 MAX_LENGTH）必须在 tokenizer 阶段截断，否则：
  - 同一个 batch 里若混入超长样本，会导致激活显存暴涨直接 OOM；
  - 不同长度样本拼接时若出现形状不匹配，可能直接报错；
  - 也可以同时过滤掉 `len(input_ids) < min_tokens` 的过短样本或空样本。
  MAX_LENGTH 具体取值要与模型支持的上下文窗口、显存预算、任务平均长度一起定；常见选择有 1024/2048/4096/8192 等。
- 不在这里做定长 `padding`：填充留给 `data_collator` 动态处理，避免大量无意义的 pad 位。

### 3.4 配置 LoRA 进行高效微调

相比全参数微调，LoRA 只训练少量低秩适配器、冻结原权重，因此显存占用显著降低（具体数值取决于模型规模、序列长度、batch、量化与实现，下面的数字仅为数量级参考）：

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 配置
lora_config = LoraConfig(
    r=8,                    # 低秩矩阵的秩，控制参数量
    lora_alpha=32,          # 缩放因子，实际缩放 = alpha / r = 4
    target_modules=[        # 要注入 LoRA 的模块
        "q_proj",           # Query 投影
        "k_proj",           # Key 投影
        "v_proj",           # Value 投影
        "o_proj",           # Output 投影
    ],
    lora_dropout=0.05,      # Dropout 防止过拟合
    bias="none",            # 不训练偏置
    task_type=TaskType.CAUSAL_LM  # 因果语言模型
)

# 将 LoRA 应用到模型
model = get_peft_model(model, lora_config)

# 查看可训练参数量
model.print_trainable_parameters()
# 输出示例: trainable params: 20,971,520 || all params: 7,615,616,000 || trainable%: 0.2754
```

**参数详解**：

| 参数 | 含义 | 建议 |
|------|------|------|
| `r` | 低秩维度 | 8~64，小任务用 8，复杂任务用 32~64 |
| `lora_alpha` | 缩放因子 | 通常设为 `r` 的 2~4 倍 |
| `target_modules` | 注入位置 | Attention 的 Q/K/V/O 投影层是标准选择 |
| `lora_dropout` | 正则化 | 0.05~0.1 |

**为什么这样写**：LoRA 只训练少量低秩适配器参数（约 0.3%），冻结原模型权重。这样：
- 显存占用大幅降低（相比全参训练可省下梯度+优化器状态的开销，量级上是数倍差距，具体数值随配置而定）
- 训练速度更快
- **有助于缓解**灾难性遗忘：原权重被冻结、更新范围小，通常比全参微调更不易大幅覆盖旧知识；但 LoRA 仍会改变模型的整体输出与行为，**并不能完全避免**遗忘，必要时仍需混入通用数据等手段

### 3.5 训练配置与启动

```python
import torch
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=4,  # 每张卡的 batch size
    gradient_accumulation_steps=4,   # 梯度累积，等效 batch = 4×4 = 16
    learning_rate=2e-4,             # LoRA 可用较大学习率
    warmup_ratio=0.03,              # 前 3% 步数 warmup
    lr_scheduler_type="cosine",     # 余弦退火
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,             # 只保留最近 3 个检查点
    bf16=True,                      # 优先 bfloat16 混合精度（无需 GradScaler）
    gradient_checkpointing=True,    # 梯度检查点，省 30%~40% 显存
    report_to="none",
)

IGNORE_INDEX = -100

def sft_data_collator(features):
    """动态 padding，并同时屏蔽 padding 位置的 loss"""
    pad_id = tokenizer.pad_token_id
    max_len = max(len(f["input_ids"]) for f in features)

    input_ids, attention_mask, labels = [], [], []
    for f in features:
        ids = f["input_ids"]
        lab = f["labels"]                       # 已在预处理中屏蔽了 prompt 段
        pad_len = max_len - len(ids)

        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        # 关键：padding 位置的 label 设为 IGNORE_INDEX(-100)，不参与 loss
        labels.append(lab + [IGNORE_INDEX] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=sft_data_collator,
)

# 开始训练
trainer.train()
```

**为什么这样写**：
- `gradient_accumulation_steps=4`：显存不够大 batch 时，累积 4 步再更新，等效 batch size = 16
- `gradient_checkpointing=True`：用计算换显存，长文本训练必备
- `warmup_ratio=0.03`：前 3% 步数线性升温，防止初期学习率过大破坏预训练权重
- **padding 的 loss 必须屏蔽**：padding 不是真实内容，padding 段的 `labels` 一定要置为 `-100`，否则模型会拟合无意义的 pad token。
- **prompt 的 loss：通常建议屏蔽（chat SFT 主流做法），但不是"必须"的硬性规则**。当前 chat SFT 最常见的配置是：仅让 assistant 回复段参与损失计算，prompt（用户指令/上下文）段置为 `-100`。这是因为 chat 场景中我们关心的是"模型怎么回答"，而不是"模型背诵用户说过的话"。但以下场景也有**合法的全序列训练（prompt 也参与 loss）**用法，并非"优化错误目标"：
  - **纯文本续写/领域继续预训练风格的 SFT**：没有严格的 user/assistant 角色，希望模型对整段文档都建模；
  - **SFT 前期混合少量通用语料**（防遗忘）：沿用 CLM 全序列 loss 可以更自然地复习通用能力；
  - **一些研究设置**发现少量 prompt loss 对复杂指令跟随有正向作用。
  选型建议：如果做标准的对话/指令模型，保持本示例"只训练 assistant"即可；如果做上述特殊场景，再根据实验决定是否放开 prompt loss。

### 3.6 保存与合并

```python
# 保存 LoRA 权重（仅保存适配器，体积小）
model.save_pretrained("./lora_weights")

# 合并 LoRA 权重到原模型（可选，用于部署）
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

**为什么这样写**：
- 保存的 LoRA 权重仅几十 MB，便于分享和加载
- `merge_and_unload()` 将 LoRA 合并到原模型，推理时无需额外计算，速度更快

---

## 第四章：关键技术点详解

### 4.1 层级微调策略

不是所有层都需要微调，根据资源选择策略：

| 策略 | 做法 | 适用场景 |
|------|------|----------|
| **仅微调最后一层** | 只训练 LM Head | 任务与预训练高度相似 |
| **微调最后几层** | 解冻顶部 2~4 层 Transformer | 通用 SFT 的折中方案 |
| **全参数微调** | 训练所有参数 | 数据充足、算力充沛 |
| **LoRA/QLoRA** | 训练低秩适配器 | **推荐**，资源受限场景 |
| **Prefix-Tuning** | 训练前缀嵌入 | 极轻量、快速实验 |

**为什么这样写**：底层参数学习的是通用语法和语义，高层参数学习的是任务特定模式。微调高层或添加适配器，既能适应新任务，又能保护底层通用知识。

### 4.2 学习率设置

```python
# 全参数微调：学习率要小，防止破坏预训练知识
full_ft_lr = 1e-5  # 或 2e-5

# LoRA 微调：学习率可以稍大，因为原权重被冻结
lora_lr = 1e-4  # 或 2e-4

# 分层学习率：底层更小，顶层更大（高级技巧）
from transformers import AdamW

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if "layers.0" in n],
        "lr": 1e-6,  # 底层极小学习率
    },
    {
        "params": [p for n, p in model.named_parameters() if "layers.31" in n],
        "lr": 1e-4,  # 顶层较大学习率
    },
]
```

**为什么这样写**：预训练模型底层已经学到了稳定的语言表示，不需要大幅调整。分层学习率让底层"微调"、顶层"大调"，平衡稳定性和适应性。

### 4.3 灾难性遗忘与缓解

**灾难性遗忘**：模型在垂直领域数据上微调后，忘记了通用知识（如"1+1=2"都不会了）。

**缓解策略**：

```python
# 策略1：混合训练数据（最常用、通常最见效的手段之一）
# SFT 数据 = 80% 领域数据 + 20% 通用指令数据（比例为起始建议，需按任务调整）
# 通用数据可用 Alpaca、ShareGPT 等开源数据集

# 策略2：L2 正则化（权重衰减）
from transformers import TrainingArguments
training_args = TrainingArguments(
    weight_decay=0.01,  # 限制参数变化幅度
    # ...
)

# 策略3：EWC（弹性权重巩固）——高级技巧
# 计算预训练参数的重要性，对重要参数施加更强约束
```

**为什么这样写**：混合通用数据是最简单有效的策略。模型在微调时同时"复习"通用知识和"学习"领域知识，避免"偏科"。

---

## 第五章：评估与验证——模型微调得怎么样

### 5.1 自动化评估指标

| 指标 | 含义 | 适用场景 |
|------|------|----------|
| **Exact Match** | 生成内容与标准答案的精确匹配率 | 数值、短文本回答 |
| **BLEU/ROUGE** | n-gram 重叠度 | 文本生成、摘要 |
| **Perplexity** | 语言模型对数据集的预测难度 | 整体生成质量 |
| **推理步骤一致率** | CoT 中过程逻辑与标准步骤的匹配度 | 多步推理任务 |
| **人工评估** | 专家打分（相关性、准确性、流畅性） | 最终验收 |

### 5.2 评估代码示例

```python
def evaluate_exact_match(test_set, model, tokenizer):
    """精确匹配评估"""
    correct = 0
    model.eval()
    
    with torch.no_grad():
        for sample in test_set:
            # 构建 prompt：必须与【训练时相同的 chat template】保持一致，
            # 否则会造成训练/推理格式错配（切勿手写 "### 指令/### 回答"，见 §3.3）
            user_content = sample["instruction"]
            if sample.get("input"):
                user_content += "\n\n" + sample["input"]
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # 贪婪解码，确定性输出
            )
            
            pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred = pred.strip()
            gold = sample["output"].strip()
            
            if pred == gold:
                correct += 1
    
    accuracy = correct / len(test_set)
    print(f"Exact Match Accuracy: {accuracy:.2%}")
    return accuracy
```

**为什么这样写**：`do_sample=False` 使用贪婪解码，消除随机性，使评估结果可复现。对于数值类任务，Exact Match 是最严格的指标。

---

## 第六章：全参数微调 vs 参数高效微调对比

> 下表数值为**数量级参考**，实际显存/速度取决于序列长度、batch、量化方案与实现，请以实测为准。注意：**全参数微调的显存不止"存一份参数"**，还要叠加梯度与优化器状态（详见预训练章的显存分析）。

| 维度 | 全参数微调 | LoRA | QLoRA |
|------|-----------|------|-------|
| **可训练参数** | 100% | ~0.3% | ~0.3% |
| **7B 模型显存（数量级）** | 上百 GB（参数+梯度+优化器状态+激活） | 显著更低 | 在 LoRA 基础上再靠 4bit 量化进一步降低 |
| **训练速度** | 基准 | 快 | 中等 |
| **存储占用** | 完整模型（~14GB） | LoRA 权重（~50MB） | LoRA 权重（~50MB） |
| **推理速度** | 快 | 稍慢（需计算 LoRA） | 稍慢 |
| **灾难性遗忘风险** | 高 | 低 | 低 |
| **适用场景** | 数据充足、算力充沛 | **推荐通用场景** | 单卡消费级 GPU |

**为什么这样写**：除非有充足算力和数据，否则**优先使用 LoRA**。它在效果、效率、防遗忘之间取得了最佳平衡。

---

## 第七章：常见陷阱与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 数据质量差（错误标注、格式混乱） | 投入 50% 精力在数据清洗和验证上 |
| 指令与答案错配（问 A 答 B） | 逐条核对指令/输入/输出的语义与数值口径一致 |
| 把错误答案当 `output` 混入训练 | SFT 只用正确答案；纠错能力靠"错在输入、对在输出"或后续偏好对齐 |
| 手写 `### 指令/### 回答` 格式 | 用目标模型的 `tokenizer.apply_chat_template`，与推理格式一致 |
| 对 prompt 和 padding 也算 loss | 把 prompt 段与 padding 段的 `labels` 置为 `-100`，只训练回复部分 |
| 学习率过大导致灾难性遗忘 | 全参数用 1e-5，LoRA 用 1e-4，务必 warmup |
| 只用领域数据微调 | 混合 10%~20% 通用数据，防止遗忘 |
| 训练轮数过多导致过拟合 | 通常 1~3 个 epoch 足够，监控验证集 loss |
| 不评估就部署 | 建立自动化评估流水线，Exact Match + 人工抽检 |
| 忽略 EOS token | 确保每个样本末尾有 `eos_token`，否则模型不会停止生成 |
| 不保存 tokenizer | `tokenizer.save_pretrained()` 与模型一起保存 |
| 长文本不截断 | 设置 `max_length`，防止 OOM |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **SFT 定位** | PT 之后、RLHF 之前，让模型学会遵循指令 |
| **数据格式** | 指令-输入-输出三元组，JSONL 格式 |
| **数据量** | 质量 > 数量，1万~5万条高质量样本即可 |
| **数据增强** | 同义改写、纠错式样本（错在输入、对在输出）、CoT 引导、上下文扩展 |
| **LoRA 配置** | r=8, alpha=32, target_modules=Q/K/V/O_proj |
| **学习率** | 全参数 1e-5，LoRA 1e-4，必加 warmup |
| **显存优化** | gradient_accumulation + gradient_checkpointing |
| **灾难性遗忘** | 混合 10%~20% 通用数据，或分层学习率 |
| **评估指标** | Exact Match、BLEU、Perplexity、人工评估 |
| **保存** | LoRA 权重可单独保存，也可 merge 后保存 |

**养成的好习惯**：
1. **数据质量是第一生产力**，宁可少而精，不要多而杂
2. 微调前**先跑通推理**，确保基座模型能正常生成
3. **从小学习率开始**，逐步调整，观察 loss 曲线
4. **每轮保存检查点**，并保留验证集最佳模型
5. **部署前做人工抽检**，自动化指标无法覆盖所有问题

**下一步预告**：掌握了 SFT 后，你将进入 **基于人类反馈的强化学习（RLHF）** 的学习——理解如何用奖励模型（Reward Model）配合 PPO、DPO、GRPO 等对齐算法，让模型不仅"会做事"，还要"做对事"，对齐人类的安全、有用、诚实的价值观。这是大语言模型从"助手"进化为"可信助手"的关键一步。继续加油！