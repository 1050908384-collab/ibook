# 了解 ChatTemplate 对话模板格式

权威文章：《Chat templates》\
　　网址：https://huggingface.co/docs/transformers/main/en/chat_templating

① 底稿：《tokenizer.apply_chat_template()》\
　　网址：https://blog.csdn.net/KIKI3666/article/details/156544210

② 参考文章：《tokenizer的apply_chat_template》\
　　网址：https://blog.csdn.net/l963852k/article/details/148840510

> **适合人群**：已完成"熟悉 padding、truncation、max_length 等关键参数"章节，能够使用 `AutoTokenizer` 进行编码的初学者。本节时长约 0.5 小时，目标是理解 ChatTemplate 的作用，掌握 `apply_chat_template` 的用法，能够区分不同模型的对话格式，并在推理和微调场景中正确应用。（①+②）
>
> **前置说明**：上节我们学会了如何将文本编码为模型输入。但在使用现代对话模型（如 Qwen、Llama、ChatGLM）时，模型期望的输入不是原始文本，而是**带有特殊控制 token 的对话格式**。ChatTemplate 就是用来完成这个转换的。所有代码请在已安装 `transformers` 的 `pytorch` conda 环境中运行。

## 一、为什么需要 ChatTemplate？（①+②）

现代大语言模型（尤其是 Chat 模型）在微调时会使用特定的对话格式训练。不同模型使用不同的**控制 token**来区分说话人角色（system/user/assistant）和消息边界。（①）

例如：
- **Qwen** 使用 `<|im_start|>user\n...<|im_end|>\n`
- **Llama-3** 使用 `<|start_header_id|>user<|end_header_id|>\n...<|eot_id|>`
- **Zephyr** 使用 `<|user|>\n...<|assistant|>\n`

> **关键认知**：这些控制 token 不是装饰，而是模型训练时的"语法"。用错格式，模型性能会急剧下降。（②）

> **本地运行配置要求**：本节代码均为纯 Python 操作，无需 GPU，CPU 即可运行，内存建议 ≥ 4GB。首次加载模型时需要联网下载 tokenizer 配置文件（约 1~5MB）。

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

**输出（Qwen 格式）**：（①）

```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
你好！<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮你的吗？<|im_end|>
<|im_start|>user
你能解释一下量子计算吗？<|im_end|>
```

## 三、关键参数详解（①+②）

| 参数 | 类型 | 说明 |
|---|---|---|
| `conversation` | `List[Dict]` | 消息列表，每个 dict 含 `"role"`（system/user/assistant）和 `"content"` |
| `tokenize` | `bool` | `False` → 返回字符串；`True` → 返回 token IDs 列表 |
| `add_generation_prompt` | `bool` | `True` 时在末尾添加 assistant 开始标记，提示模型生成 |
| `return_tensors` | `str` | 当 `tokenize=True` 时，返回 `"pt"`/`"np"` 张量 |
| `continue_final_message` | `bool`/`str` | 让模型继续最后一条消息（用于 prefill），不能与 `add_generation_prompt` 同用 |

## 四、推理 vs 微调：add_generation_prompt 的区别（权威+①+②）

### 4.1 推理时：add_generation_prompt=True（①）

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
| `add_generation_prompt=False` | `...<|im_end|>`（结束） |
| `add_generation_prompt=True` | `...<|im_end|>\n<|im_start|>assistant\n`（准备生成） |

### 4.2 微调时：add_generation_prompt=False（权威+①）

微调时训练数据包含完整的 user + assistant 对话，不需要额外的 generation prompt：

```python
def format_example(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False  # 微调时不加
        )
    }

dataset = raw_dataset.map(format_example)
```

> **权威提醒**：训练时设置 `add_generation_prompt=False`，因为 assistant 的回复已经在训练数据中了。（权威）

## 五、不同模型的模板格式对比（②）

| 模型 | 格式名称 | 控制 token 示例 |
|---|---|---|
| Qwen/Qwen2.5 | ChatML | `<|im_start|>user\n...<|im_end|>\n` |
| Meta-Llama-3 | Llama-3 | `<|start_header_id|>user<|end_header_id|>\n...<|eot_id|>` |
| Zephyr-7B | Zephyr | `<|user|>\n...<|assistant|>\n` |
| Mistral-7B | ChatML | `<s>[INST] ... [/INST]` |

> **重要**：不同模型模板**不能混用**！用 Qwen 的模板去调 Llama 模型，效果会很差。（②）

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

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
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
).to(model.device)

# 生成
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **本地运行配置要求**：加载 `Qwen2.5-0.5B` 模型需要约 1GB 显存（GPU）或 4GB 内存（CPU）。若显存不足，可换用更小的模型如 `Qwen/Qwen2.5-0.5B-Instruct` 或仅运行 `tokenize=False` 的代码。

## 八、常见问题与解决（①+②）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 模型输出混乱/不遵循指令 | 使用了错误的 chat template | 确认模型对应的模板格式，勿混用 |
| 生成时重复用户消息 | 忘记 `add_generation_prompt=True` | 推理时务必添加 |
| 微调损失不下降 | `add_generation_prompt=True` 导致标签错位 | 微调时设为 `False` |
| 找不到 `chat_template` | 老模型未定义模板 | 手动设置 `tokenizer.chat_template` |

## 九、总结与后续学习路线（①+②）

恭喜你完成了 ChatTemplate 对话模板格式的学习！你现在应该掌握了：

- ✅ ChatTemplate 的作用：将结构化消息转换为带控制 token 的模型输入
- ✅ `apply_chat_template` 的核心参数：`tokenize`、`add_generation_prompt`、`return_tensors`
- ✅ 推理时 `add_generation_prompt=True`，微调时 `add_generation_prompt=False`
- ✅ 不同模型（Qwen、Llama、Zephyr）的格式差异
- ✅ 查看和自定义 `chat_template`

> **下一步预告**：从下一节开始，我们进入 **"Transformer 架构（Decoder-Only）"** 板块。你将深入理解 Transformer 的 Encoder-Decoder 结构，重点学习 Decoder-Only 架构（GPT/Llama/Qwen）的内部机制，包括 RMSNorm、SwiGLU、RoPE、KV Cache 等核心组件。这是理解大模型"黑盒"内部工作原理的关键章节。现在，试着用 `apply_chat_template` 处理一段多轮对话，分别观察 `add_generation_prompt=True` 和 `False` 的输出差异吧！