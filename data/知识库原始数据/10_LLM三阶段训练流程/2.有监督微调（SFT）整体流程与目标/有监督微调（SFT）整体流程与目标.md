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
**环境准备**：Python 3.10+，安装 `transformers`、`peft`、`datasets`、`accelerate`

```bash
pip install transformers peft datasets accelerate
```

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
  "instruction": "请分析华为2023年营收增长率",
  "input": "",
  "output": "根据华为2023年年报，研发支出1647亿元，总收入6423亿元，研发投入占比 = (1647 ÷ 6423) × 100% ≈ 25.65%。"
}
```

**为什么这样写**：
- `instruction`：告诉模型"要做什么"（任务描述）
- `input`：具体的输入内容（可为空，如纯问答任务）
- `output`：期望模型生成的标准答案

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
| **负样本注入** | 插入 5%~10% 错误答案，训练模型甄别 | 故意给出错误数字，让模型纠正 |
| **推理链引导（CoT）** | 要求模型先展示推理过程 | "请先列出计算公式，再计算..." |
| **上下文扩展** | 多段信息的问题提供完整上下文 | 对比多家公司时提供完整表格 |

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

### 3.2 加载预训练模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或 Llama-2-7b, Baichuan2-7B 等

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",      # 自动选择 float16/bfloat16
    device_map="auto"        # 自动分配到可用 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
```

**为什么这样写**：
- `torch_dtype="auto"`：根据硬件自动选择精度，A100 用 float16，H100 用 bfloat16
- `device_map="auto"`：自动将模型层分配到多张 GPU，单卡放不下时自动拆分

### 3.3 数据加载与格式化

```python
from datasets import load_dataset

# 加载自定义 JSONL 数据集
# 格式: {"instruction": "...", "input": "...", "output": "..."}
dataset = load_dataset("json", data_files="sft_data.jsonl", split="train")

def format_prompt(example):
    """将数据格式化为模型训练所需的文本格式"""
    if example["input"]:
        prompt = f"### 指令：\n{example['instruction']}\n\n### 输入：\n{example['input']}\n\n### 回答：\n"
    else:
        prompt = f"### 指令：\n{example['instruction']}\n\n### 回答：\n"
    
    # 完整文本 = prompt + output + EOS
    full_text = prompt + example["output"] + tokenizer.eos_token
    return {"text": full_text}

# 应用格式化
dataset = dataset.map(format_prompt)

# 分词
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

**为什么这样写**：
- 使用 **Chat Template 风格** 的格式（`### 指令：` `### 回答：`），让模型明确区分"指令"和"回答"
- `truncation=True` + `max_length=2048`：截断超长样本，防止显存溢出
- `padding="max_length"`：填充到固定长度，便于批处理

### 3.4 配置 LoRA 进行高效微调

全参数微调 7B 模型需要约 28GB 显存（FP16），而 LoRA 仅需约 8GB：

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
- 显存占用大幅降低（7B 模型从 28GB → 8GB）
- 训练速度更快
- 防止灾难性遗忘（原知识被冻结保护）

### 3.5 训练配置与启动

```python
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
    fp16=True,                      # 混合精度训练
    gradient_checkpointing=True,    # 梯度检查点，省 30%~40% 显存
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda x: {        # 动态 padding，提高效率
        "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in x]),
        "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in x]),
        "labels": torch.stack([torch.tensor(d["input_ids"]) for d in x]),  # 自回归：labels = input_ids
    },
)

# 开始训练
trainer.train()
```

**为什么这样写**：
- `gradient_accumulation_steps=4`：显存不够大 batch 时，累积 4 步再更新，等效 batch size = 16
- `gradient_checkpointing=True`：用计算换显存，长文本训练必备
- `warmup_ratio=0.03`：前 3% 步数线性升温，防止初期学习率过大破坏预训练权重

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
# 策略1：混合训练数据（最有效）
# SFT 数据 = 80% 领域数据 + 20% 通用指令数据
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
            # 构建 prompt
            prompt = f"### 指令：\n{sample['instruction']}\n\n### 回答：\n"
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

| 维度 | 全参数微调 | LoRA | QLoRA |
|------|-----------|------|-------|
| **可训练参数** | 100% | ~0.3% | ~0.3% |
| **7B 模型显存** | ~28GB (FP16) | ~8GB | ~6GB |
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
| **数据增强** | 同义改写、负样本注入、CoT 引导、上下文扩展 |
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

**下一步预告**：掌握了 SFT 后，你将进入 **基于人类反馈的强化学习（RLHF）** 的学习——理解如何用奖励模型（Reward Model）和 PPO 算法，让模型不仅"会做事"，还要"做对事"，对齐人类的安全、有用、诚实的价值观。这是大语言模型从"助手"进化为"可信助手"的关键一步。继续加油！