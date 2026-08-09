# 了解 ChatTemplate 对话模板格式

权威文章：《Chat templates》  
　　网址：https://huggingface.co/docs/transformers/main/en/chat_templating

① 底稿：《tokenizer.apply_chat_template()》  
　　网址：https://blog.csdn.net/KIKI3666/article/details/156544210

② 参考文章：《tokenizer的apply_chat_template》  
　　网址：https://blog.csdn.net/l963852k/article/details/148840510

> **适合人群**：已完成"熟悉 padding、truncation、max_length 等关键参数"章节，能够使用 `AutoTokenizer` 进行编码的初学者。本节时长约 0.5 小时，目标是理解 ChatTemplate 的作用，掌握 `apply_chat_template` 的用法，能够区分不同模型的对话格式，并在推理和微调场景中正确应用。（①+②）
>
> **前置说明**：上节我们学会了如何将文本编码为模型输入。但在使用现代对话模型（如 `Qwen/Qwen2.5-0.5B-Instruct`、`meta-llama/Llama-2-7b`）时，模型期望的输入不是原始文本，而是**带有特殊控制 token 的对话格式**。ChatTemplate 就是用来完成这个转换的。所有代码请在已安装 `transformers` 的 `pytorch` conda 环境中运行。

## 一、为什么需要 ChatTemplate？（①+②）

现代大语言模型（尤其是 Chat 模型）在微调时会使用特定的对话格式训练。不同模型使用不同的**控制 token**来区分说话人角色（system/user/assistant）和消息边界。（①）

例如：
- **`Qwen/Qwen2.5-0.5B-Instruct`** 使用 `<|im_start|>user\n...<|im_end|>\n`
- **`meta-llama/Meta-Llama-3-8B-Instruct`** 使用 `<|start_header_id|>user<|end_header_id|>\n...<|eot_id|>`
- **`HuggingFaceH4/zephyr-7b-beta`** 使用 `<|user|>\n...<|assistant|>\n`

> **关键认知**：这些控制 token 不是装饰，而是模型训练时的"语法"。用错格式，模型效果通常会很差。（②）

> **本地运行配置要求**：本节代码均为纯 Python 操作，无需 GPU，CPU 即可运行，内存建议 ≥ 4GB。首次加载模型时需要联网下载 tokenizer 配置文件（体积通常在数 MB 以内）。

## 二、apply_chat_template 基本用法（①+②）

`apply_chat_template` 将结构化的消息列表转换为模型可接受的格式。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
    {"role": "user", "content": "你能解释一下量子计算吗？"}
]

# 应用聊天模板，返回字符串
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)
```

> 输出结果会随模型 checkpoint 和 transformers 版本变化，以下为示意格式，实际以运行结果为准：（①）

## 三、关键参数详解（①+②）

| 参数 | 类型 | 说明 |
|---|---|---|
| `conversation` | `List[Dict]` | 消息列表，每个 dict 含 `"role"`（system/user/assistant）和 `"content"` |
| `tokenize` | `bool` | `False` → 返回字符串；`True` → 返回 token IDs 列表 |
| `add_generation_prompt` | `bool` | `True` 时在末尾添加 assistant 开始标记，提示模型生成 |
| `return_tensors` | `str` | 当 `tokenize=True` 时，返回 `"pt"`/`"np"` 张量 |
| `continue_final_message` | `bool` | 让模型继续最后一条消息（用于 prefill），不能与 `add_generation_prompt` 同用 |

## 四、推理 vs 微调：add_generation_prompt 的区别（权威+①+②）

### 4.1 推理时：通常需要 add_generation_prompt=True（①）

推理时只需要提供历史对话，模型需要知道"该轮到我生成了"：

```python
# 推理：只给 user 输入，添加 generation prompt
messages = [{"role": "user", "content": "你好！"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # ← 关键！
    return_tensors="pt"
)
# 输出末尾会添加 assistant 开始标记
```

**对比效果**（②）：

| 设置 | 输出末尾 |
|---|---|
| `add_generation_prompt=False` | 以最后一条消息的结束标记结尾 |
| `add_generation_prompt=True` | 添加 assistant 开始标记，提示模型生成 |

### 4.2 微调时：根据训练方式选择（权威+①）

微调时 `add_generation_prompt` 的设置取决于你的数据整理流程，**核心原则是避免重复套模板和重复添加 generation prompt**：（权威+①）

**场景一：完整对话序列训练（最常用）**

训练数据已经包含完整的 user + assistant 对话，且由预处理函数显式调用 `apply_chat_template`，或由明确支持 conversational dataset 的训练框架统一套模板。此时应设为 `False`，因为 assistant 的回复已经在训练数据中：

```python
def format_example(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False  # 已有完整 assistant 回复
        )
    }

dataset = raw_dataset.map(format_example)
```

**场景二：只训练 assistant 回复**

某些框架（如部分版本的 TRL）会在内部自动调用 `apply_chat_template`，此时你的原始数据不应预先套模板，只需提供原始 messages，由框架统一处理格式化。

> **注意**：自动套用 ChatTemplate 只负责对话格式化，不代表训练时只计算 assistant 回复的 loss。只训练 assistant 回复还需要使用框架提供的 assistant-only loss 配置（如支持时设置 `assistant_only_loss=True`，并使用能够生成 assistant mask 的模板），或自行将非 assistant 位置的 labels 设为 `-100`。

**场景三：在 dataset.map() 预处理阶段显式套模板（推荐做法）**

如果你使用标准的 `Trainer` 或自定义训练循环，推荐在数据预处理阶段通过 `dataset.map()` 显式调用 `apply_chat_template`，将 messages 转换为模型输入字符串。数据整理器（Data Collator）只负责 padding 和 tensor 拼接，**不会自动套用 ChatTemplate**：

```python
def format_example(example):
    # 显式调用 apply_chat_template，在预处理阶段完成格式转换
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = raw_dataset.map(format_example)
# 后续由 DataCollator 或自定义 collator 处理 padding
```

> **注意**：`DataCollatorForSeq2Seq` 等普通数据整理器不会自动调用 `apply_chat_template`。如果你在预处理阶段没有显式套模板，训练时模型看到的将是未经格式化的原始文本，导致效果下降。（权威+①）

**场景四：使用 TRL 等框架的自动处理**

如果你使用 TRL（Transformer Reinforcement Learning）库进行微调，部分版本（如 TRL ≥ 0.8）的 `SFTTrainer` 支持在内部自动处理 conversational 数据集。此时需要查阅对应版本的文档，确认是否需要在预处理阶段手动调用 `apply_chat_template`。例如，某些 TRL 版本要求数据格式为 `{"messages": [...]}`，由 `SFTTrainer` 内部调用模板；而另一些版本则要求预处理后的文本格式。请以你使用的 TRL 版本官方文档为准。

> **关键提醒**：不要在数据预处理阶段手动套模板后，又在 collator 或训练器中再次套模板；也不要在已有 assistant 回复的数据上额外添加 `add_generation_prompt=True`，否则会导致标签错位。（权威+①）

## 五、不同模型的模板格式对比（②）

| 模型 checkpoint | 格式名称 | 控制 token 示例 |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | ChatML | `<|im_start|>user\n...<|im_end|>\n` |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Llama-3 | `<|start_header_id|>user<|end_header_id|>\n...<|eot_id|>` |
| `HuggingFaceH4/zephyr-7b-beta` | Zephyr | `<|user|>\n...<|assistant|>\n` |
| `mistralai/Mistral-7B-Instruct-v0.2` | Mistral | `<s>[INST] ... [/INST]` |

> **重要**：不同模型模板**通常不能混用**。用 `Qwen/Qwen2.5-0.5B-Instruct` 的模板去调 `meta-llama/Meta-Llama-3-8B-Instruct` 模型，效果通常会很差。（②）

## 六、查看和自定义模板（①）

```python
# 查看当前 tokenizer 的模板
print(tokenizer.chat_template)

# 自定义模板（高级）
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{message['role']}}: {{message['content']}}\n"
    "{% endfor %}"
)
```

## 七、完整推理示例（①+②）

### 7.1 CPU 推理（无需 GPU）

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# CPU 推理使用 float32（bfloat16 在部分 CPU 上可能不被支持）
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

messages = [
    {"role": "system", "content": "你是一个友好的助手。"},
    {"role": "user", "content": "你好！"}
]

# 构造输入（推理时 add_generation_prompt=True）
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

# 生成
with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **本地运行配置要求**：CPU 推理 `Qwen2.5-0.5B` 约需 4GB 内存，无需 GPU。

### 7.2 CUDA 推理（需要 NVIDIA GPU）

> **环境依赖**：CUDA 推理示例需要安装 `accelerate` 库（`pip install accelerate`），用于 `device_map="auto"` 的自动设备分配。经测试，`accelerate` ≥ 0.20 版本通常可与 `transformers` 配合正常工作，具体兼容版本请以官方文档为准。
>
> **硬件兼容性说明**：`torch.bfloat16` 需要 GPU 支持 BF16（如 NVIDIA Ampere 架构及以后的 A100、RTX 30/40 系列等）。如果你的 GPU 不支持 BF16（如 Pascal、Turing 架构的 GTX 1080 Ti、RTX 2080 等），请将 `torch_dtype` 改为 `torch.float16`；CPU 推理请使用 `torch.float32`（见 7.1 节）。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# CUDA 推理：支持 BF16 的 GPU（Ampere+）使用 bfloat16，否则改用 float16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # 不支持 BF16 的 GPU 请改为 torch.float16
    device_map="auto"
)

messages = [
    {"role": "system", "content": "你是一个友好的助手。"},
    {"role": "user", "content": "你好！"}
]

# 构造输入
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# 生成
with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **本地运行配置要求**：加载 `Qwen/Qwen2.5-0.5B-Instruct` 模型使用 bfloat16 时约需 1GB 显存。若显存不足，可换用 CPU 推理（7.1 节代码），或使用 `torch.float32` 在 CPU 上运行。

## 八、常见问题与解决（①+②）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 模型输出混乱/不遵循指令 | 使用了错误的 chat template | 确认模型对应的模板格式，勿混用 |
| 生成时重复用户消息 | 忘记 `add_generation_prompt=True` | 推理时通常需要添加 |
| 微调损失不下降 | 重复套模板或 `add_generation_prompt` 设置不当 | 检查数据整理流程，确保不重复添加 generation prompt |
| 找不到 `chat_template` | 老模型未定义模板 | 手动设置 `tokenizer.chat_template` |

## 九、总结与后续学习路线（①+②）

恭喜你完成了 ChatTemplate 对话模板格式的学习！你现在应该掌握了：

- ✅ ChatTemplate 的作用：将结构化消息转换为带控制 token 的模型输入
- ✅ `apply_chat_template` 的核心参数：`tokenize`、`add_generation_prompt`、`return_tensors`
- ✅ 推理时通常 `add_generation_prompt=True`，微调时需根据数据整理流程选择
- ✅ 不同模型（`Qwen/Qwen2.5-0.5B-Instruct`、`meta-llama/Meta-Llama-3-8B-Instruct` 等）的格式差异
- ✅ 查看和自定义 `chat_template`

> **下一步预告**：从下一节开始，我们进入 **"Transformer 架构（Decoder-Only）"** 板块。你将深入理解 Transformer 的 Encoder-Decoder 结构，重点学习 Decoder-Only 架构（GPT/Llama/Qwen）的内部机制，包括 RMSNorm、SwiGLU、RoPE、KV Cache 等核心组件。这是理解大模型"黑盒"内部工作原理的关键章节。现在，试着用 `apply_chat_template` 处理一段多轮对话，分别观察 `add_generation_prompt=True` 和 `False` 的输出差异吧！
