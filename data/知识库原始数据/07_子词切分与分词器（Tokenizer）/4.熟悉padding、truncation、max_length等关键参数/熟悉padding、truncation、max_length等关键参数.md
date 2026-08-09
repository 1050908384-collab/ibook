# 熟悉 padding、truncation、max_length 等关键参数

权威文章：《Tokenizers - Hugging Face NLP Course》\
　　网址：https://huggingface.co/learn/nlp-course/chapter2/4

权威文章：《Tokenizer API Reference》\
　　网址：https://huggingface.co/docs/transformers/main_classes/tokenizer

① 底稿：《Hugging Face transformers 库中 tokenizer 常用的方法和属性》\
　　网址：https://blog.csdn.net/u013172930/article/details/146155496

② 参考文章：《Hugging Face Tokenizer的padding、truncation参数详解：如何让你的BERT/RoBERTa输入不出错？》\
　　网址：https://blog.csdn.net/weixin_42537499/article/details/161131794

③ 参考文章：《HuggingFace Transformers库实战：Tokenizer与Model的高效应用指南》\
　　网址：https://blog.csdn.net/p8q9r0/article/details/151163743

> **适合人群**：已完成"掌握 HuggingFace Tokenizer 的加载与使用"章节，能够使用 `AutoTokenizer.from_pretrained()` 加载分词器并执行基础编码解码的初学者。本节时长约 0.5 小时，目标是深入理解 `padding`、`truncation`、`max_length` 等关键参数的行为差异，能够根据任务场景选择正确的配置，避免"维度不匹配"等常见错误。（①+②）
>
> **前置说明**：上节我们学会了加载分词器和基础编码。但在实际训练中，一个 batch 里的句子长度往往参差不齐——有的 10 个 token，有的 500 个 token。神经网络要求输入是固定维度的张量，因此需要使用 `pad_token_id` 补齐短句（padding）和截断长句（truncation）。本节将详细讲解这些参数的含义、组合方式和常用配置。所有代码请在已安装 `transformers` 的 `pytorch` conda 环境中运行。

## 一、为什么需要 padding 和 truncation？（②）

神经网络（尤其是 Transformer）的输入必须是**固定形状的张量。**

- **padding**：使用 `pad_token_id` 将短句补齐到统一长度，并通过 `attention_mask` 标识补齐位置，使 batch 内所有样本维度一致，才能堆叠成矩阵。（②）
- **truncation**：将长句截断到最大长度，避免超出模型的处理能力（如 `bert-base-chinese` 通常限制在 512 token 以内）。（②）

> **常见错误**：如果不设置 `padding=True`，batch 内各样本长度不一致，PyTorch `DataLoader` 会报错 `RuntimeError: stack expects each tensor to be equal size`。（②）

> **本地运行配置要求**：本节代码均为纯 Python 操作，无需 GPU，CPU 即可运行，内存建议 ≥ 4GB。

## 二、padding：填充短句（①+②）

`padding` 参数控制是否以及如何将短句填充到统一长度。

### 2.1 padding 的三种模式（权威②）

| 取值 | 行为 | 适用场景 |
|---|---|---|
| `True` 或 `'longest'` | 填充到当前 batch 中最长序列的长度（含特殊 token） | 训练时常用，动态 padding，通常更节省计算 |
| `'max_length'` | 填充到 `max_length` 指定的长度（或 tokenizer 的默认最大长度） | 需要固定输入尺寸的场景 |
| `False` 或 `'do_not_pad'`（默认） | 不填充，返回原始长度 | 单条推理或自定义 collator |

### 2.2 代码演示（①+②）

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

texts = ["我喜欢自然语言处理", "深度学习很有趣", "你好"]

# padding='longest'：填充到当前 batch 中最长序列的长度
encoded = tokenizer(texts, padding=True, return_tensors="pt")
print("padding=True 形状:", encoded["input_ids"].shape)

# padding='max_length' + max_length：填充到指定长度
encoded = tokenizer(texts, padding="max_length", max_length=12, return_tensors="pt")
print("padding=max_length 形状:", encoded["input_ids"].shape)

# padding=False：不填充
encoded = tokenizer(texts, padding=False)
print("padding=False 长度:", [len(ids) for ids in encoded["input_ids"]])
# 注意：上述长度为模型输入长度（含 [CLS]、[SEP] 等特殊 token），与纯 tokenize 长度不同
```

> **注意**：`padding=True` 是训练时较为常用的做法，它只填充到 batch 内最长序列的长度（含特殊 token），避免不必要的计算。（②）

### 2.3 padding 方向：left vs right（权威②）

默认情况下，BERT/GPT 等模型使用 **right padding**（右侧补齐）。某些生成模型（如 GPT）在推理时可能需要 **left padding**，让生成 token 出现在右侧。

```python
# 设置左侧填充（少数场景需要）
tokenizer.padding_side = "left"
encoded = tokenizer(texts, padding=True, return_tensors="pt")
```

## 三、truncation：截断长句（①+②）

`truncation` 控制是否以及如何将超长序列截断。

### 3.1 truncation 的四种模式（权威②）

| 取值 | 行为 | 适用场景 |
|---|---|---|
| `True` 或 `'longest_first'` | 对于单个序列从右侧截断；对于句子对，同时从两个序列中移除 token，直到总长度满足要求 | 通用场景 |
| `'only_first'` | 只截断第一个序列（句子对中的第一个参数） | 问答任务中截断过长的 context，保留 question |
| `'only_second'` | 只截断第二个序列（句子对中的第二个参数） | 保留 context，截断 question |
| `False` 或 `'do_not_truncate'`（默认） | 不截断，超长时可能报错 | 确保不丢失信息 |

### 3.2 代码演示（②）

```python
long_text = "自然语言处理" * 200  # 构造一个超长文本

# truncation=True：截断到 max_length 指定的长度（或 tokenizer 默认最大长度）
encoded = tokenizer(long_text, truncation=True, max_length=512, return_tensors="pt")
print("截断后长度:", encoded["input_ids"].shape[1])

# truncation=True + max_length=10：截断到指定长度
encoded = tokenizer(long_text, truncation=True, max_length=10, return_tensors="pt")
print("截断到 10:", encoded["input_ids"].shape[1])

# 句子对截断策略：第一个参数为 context（长文本），第二个参数为 question
encoded = tokenizer(long_text, "这是一个问题", truncation="only_first", max_length=20, return_tensors="pt")
# truncation="only_first"：只截断第一个参数（context），保留第二个参数（question）完整
```

### 3.3 关键规则：max_length 的作用（权威②）

- 如果设置了 `max_length`，truncation 和 padding 都以此为上限。
- 如果未设置 `max_length`，通常使用 tokenizer 的 `model_max_length`（如 `bert-base-chinese` 通常为 512）。
- 如果 tokenizer 未定义 `model_max_length` 且未显式设置 `max_length`，`truncation=True` 的行为可能因 transformers 版本而异。建议始终显式设置 `max_length`，或在不确定时查阅当前版本的官方文档。

## 四、完整参数组合实战（①+②+③）

实际训练中较为常用的组合：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

texts = ["这是一个测试句子", "这是另一个句子，稍微长一点"]

# 训练时的标准写法
encoded = tokenizer(
    texts,
    padding=True,           # 填充到 batch 最长
    truncation=True,       # 超过 max_length 就截断
    max_length=512,         # 明确上限
    return_tensors="pt"   # 返回 PyTorch 张量
)

print(encoded.keys())
# 预期输出格式类似：dict_keys(['input_ids', 'attention_mask', 'token_type_ids'])
# 实际以运行结果为准

print(encoded["input_ids"].shape)
# 形状由当前 batch 中最长序列决定，实际以运行结果为准
```

> **本地运行配置要求**：此代码纯 CPU 运行即可，内存 ≥ 4GB。

## 五、在 DataLoader 中使用 DataCollatorWithPadding（③）

当使用 PyTorch `DataLoader` 时，可以用 `DataCollatorWithPadding` 自动处理 padding：

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)

# 假设 raw_texts 是原始文本列表
# DataCollator 会在每个 batch 内动态 padding 到该 batch 的最长长度
```

> 这比预先对所有数据做 `padding='max_length'` 通常更高效，因为不同 batch 的最长长度可能不同。（③）

## 六、在 Trainer 中使用（③）

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    data_collator=DataCollatorWithPadding(tokenizer)  # 自动处理 padding
)
```

## 七、常见问题与解决（①+②）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `RuntimeError: stack expects each tensor...` | batch 内长度不一致且未设置 padding | 添加 `padding=True` 或 `padding='max_length'` |
| 截断后丢失关键信息 | truncation 策略不当 | 使用 `only_first`/`only_second` 精准控制 |
| padding 后 attention_mask 全为 1 | 未正确设置 padding | 检查 `attention_mask` 是否为 0/1 混合 |
| 生成模型输出异常 | GPT 等模型用了 right padding | 设置 `padding_side='left'` |
| 显存占用过高 | `max_length` 设置过大 | 根据实际数据分布设置合理的 `max_length` |

## 八、总结与后续学习路线（①+②+③）

恭喜你完成了 padding、truncation、max_length 关键参数的学习！你现在应该掌握了：

- ✅ padding 的三种模式：`True`/`longest`、`max_length`、`False`
- ✅ truncation 的四种模式：`True`/`longest_first`、`only_first`、`only_second`、`False`
- ✅ `max_length` 的优先级规则
- ✅ `DataCollatorWithPadding` 在 DataLoader 中的高效使用
- ✅ Trainer 集成方式

> **下一步预告**：在下一节 **"了解 ChatTemplate 对话模板格式"** 中，我们将学习现代大模型（如 Llama、Qwen、ChatGLM）的对话格式处理。你会理解 `apply_chat_template` 的作用，学会将多轮对话转换为模型可接受的输入格式，这是使用指令微调模型和构建聊天机器人的必备技能。现在，试着用不同的 padding 和 truncation 组合处理一批长度差异较大的文本，观察 `attention_mask` 的变化吧！