# 掌握 HuggingFace Tokenizer 的加载与使用

权威文章：《Tokenizers - Hugging Face NLP Course》  
　　网址：https://huggingface.co/learn/nlp-course/chapter2/4

权威文章：《Tokenizer API Reference》  
　　网址：https://huggingface.co/docs/tokenizers/api/tokenizer

① 底稿：《Hugging Face transformers 库中的 from_pretrained 和 save_pretrained：加载和保存模型及分词器》  
　　网址：https://blog.csdn.net/u013172930/article/details/146151747

② 参考文章：《深入解析 AutoTokenizer.from_pretrained 参数及其应用场景》  
　　网址：https://blog.csdn.net/weixin_29201313/article/details/158904800

③ 参考文章：《（详细讲解）transformers 中的 AutoTokenizer 的用法》  
　　网址：https://blog.csdn.net/qq_43391414/article/details/149350893

④ 参考文章：《HuggingFace 学习笔记--Tokenizer 的使用》  
　　网址：https://blog.csdn.net/weixin_43863869/article/details/134631725

> **适合人群**：已完成"理解 BPE、WordPiece、SentencePiece 算法原理"章节，理解子词分词核心思想的初学者。本节时长约 0.5 小时，目标是掌握使用 HuggingFace `AutoTokenizer` 加载预训练分词器、执行编码解码、批量处理和保存加载，能够独立完成文本到模型输入的转换。（①+③）
>
> **前置说明**：上节我们学习了 BPE、WordPiece、SentencePiece 的算法原理。本节将把这些理论付诸实践——使用 HuggingFace 提供的 `transformers` 库，加载真实预训练模型对应的分词器，观察它们对文本的实际切分效果。所有代码请在已安装 `transformers` 的 `pytorch` conda 环境中运行。

## 一、为什么使用 AutoTokenizer？（①+③）

HuggingFace Hub 上有大量预训练模型，每个模型可能使用不同的分词算法（BPE、WordPiece、Unigram）和不同的词表。`AutoTokenizer` 是一个**智能识别器**：你只需提供模型名称，它会自动判断该用哪种分词器类（`BertTokenizer`、`GPT2Tokenizer`、`LlamaTokenizer` 等），无需手动选择。（①+③）

```python
from transformers import AutoTokenizer

# 一行代码，自动加载 BERT 对应的分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
```

> **本地运行配置要求**：首次执行 `from_pretrained()` 时，需要从 HuggingFace Hub 下载分词器配置文件（体积通常在数 MB 以内）。需要**联网**，对电脑配置无特殊要求，CPU 即可，内存建议 ≥ 4GB。

## 二、加载 Tokenizer：from_pretrained（①+②）

### 2.1 从 HuggingFace Hub 加载（①）

```python
from transformers import AutoTokenizer

# 加载 BERT 中文分词器（WordPiece 算法）
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-chinese")

# 加载 GPT-2 分词器（Byte-level BPE 算法）
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
```

`from_pretrained` 会自动下载并缓存分词器所需的全部文件（`tokenizer_config.json`、`vocab.txt` 或 `vocab.json` / `merges.txt` 等）。（①）

> **本地运行配置要求**：需要联网下载配置文件（体积通常在数 MB 以内），CPU 即可执行。若网络较慢，可参考下方镜像源或缓存配置。

### 2.2 关键参数详解（②）

| 参数 | 作用 | 示例 |
|---|---|---|
| `pretrained_model_name_or_path` | 模型 ID 或本地路径 | `"bert-base-chinese"` 或 `"./local_path"` |
| `cache_dir` | 指定缓存目录 | `cache_dir="./hf_cache"` |
| `local_files_only=True` | 只使用本地缓存，不联网 | 离线环境必备 |
| `use_fast=True` | 使用 Rust 实现的快速分词器（默认） | 批量处理时速度通常更快 |
| `trust_remote_code=True` | 允许执行远程自定义代码 | 部分社区模型需要 |

> `use_fast=True`（默认）会尝试加载 `tokenizers` 库提供的 Rust 实现分词器。在批量处理场景下，其速度通常比纯 Python 实现快数倍至数十倍（具体取决于文本长度和 batch size）。如果遇到分词结果异常，可设为 `False` 回退到慢速版本。（②）

### 2.3 从本地路径加载（①）

如果你已经下载了模型文件，或在内网环境使用：

```python
# 指向包含 tokenizer 配置文件的本地目录
tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese-local")
```

> **本地运行配置要求**：本地目录应包含该 tokenizer 所需的词表、配置和特殊 token 文件；缺少关键文件时会加载失败。使用 `save_pretrained()` 保存的完整目录是可靠的加载来源。

## 三、编码：从文本到数字（①+③+④）

分词器的核心任务是将人类可读的文本转换为模型可计算的数字。这个过程在 HuggingFace 中称为 **Encoding**，分为两步：分词（Tokenization）→ 转 ID（Conversion to input IDs）。（①）

### 3.1 统一接口：直接调用 tokenizer（③+④）

推荐的方式是直接调用 tokenizer 对象，它会一次性完成分词、加特殊 token、转 ID、生成 attention_mask 等全部操作：（③）

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "我喜欢自然语言处理"
encoded = tokenizer(text)
print(encoded)
```

> 输出结果会随 transformers 版本和词表版本变化，以下为示意格式，实际以运行结果为准：（④）

返回的字典包含三个关键字段：（③）

- `input_ids`：每个 token 对应的整数 ID
- `attention_mask`：标记哪些位置是有效 token（1），哪些是 padding（0）
- `token_type_ids`：标记 token 属于第一个句子（0）还是第二个句子（1），用于句子对任务

> **本地运行配置要求**：编码操作纯 CPU 计算，无额外配置要求，内存 ≥ 4GB 即可。

### 3.2 返回 PyTorch 张量（③）

```python
encoded = tokenizer(text, return_tensors="pt")
print(encoded["input_ids"].shape)
```

> `return_tensors="pt"` 返回 PyTorch 张量，`"np"` 返回 NumPy 数组，`"tf"` 返回 TensorFlow 张量。

### 3.3 分步观察：tokenize + convert_tokens_to_ids（①）

如果你想观察中间过程，可以分步执行：（①）

```python
# 第 1 步：分词，得到 token 字符串列表
tokens = tokenizer.tokenize("我喜欢自然语言处理")
print("Tokens:", tokens)
print("Token数量:", len(tokens))

# 第 2 步：将 token 转换为 ID
ids = tokenizer.convert_tokens_to_ids(tokens)
print("IDs:", ids)
print("ID数量:", len(ids))
```

> 注意：分步执行不会自动添加 `[CLS]`、`[SEP]` 等特殊 token。直接调用 `tokenizer(text)` 会自动添加。上述输出结果会随 transformers 版本和词表版本变化，实际以运行结果为准。

## 四、解码：从数字到文本（①+③）

`decode()` 方法将 `input_ids` 还原为可读文本，同时会把属于同一个词的子词重新拼接起来。（①）

```python
# 使用 3.1 节得到的 encoded 结果
ids = encoded["input_ids"]

# 默认保留特殊 token
print(tokenizer.decode(ids))

# 跳过特殊 token
print(tokenizer.decode(ids, skip_special_tokens=True))
```

> 输出结果会随版本和词表变化，实际以运行结果为准。

> **本地运行配置要求**：解码操作纯 CPU 计算，无额外配置要求。

## 五、批量处理（③+④）

实际训练时，你需要一次性处理多个句子（一个 batch）。tokenizer 支持直接传入列表，自动进行批量编码。（③+④）

```python
texts = ["我喜欢自然语言处理", "深度学习很有趣"]

# 批量编码，padding 到最长句子，截断到 max_length
encoded = tokenizer(
    texts,
    padding=True,           # 自动填充
    truncation=True,        # 自动截断
    max_length=512,       # 最大长度
    return_tensors="pt"
)

print(encoded["input_ids"].shape)
print(encoded["attention_mask"])
```

> 第二个句子更短，所以 `attention_mask` 中对应位置可能为 0，表示 padding 位置。（③）

### 5.1 使用 batch_encode_plus（④）

```python
encoded = tokenizer.batch_encode_plus(
    texts,
    padding='max_length',
    max_length=12,
    truncation=True,
    return_tensors='pt',
    return_token_type_ids=True,
    return_attention_mask=True
)
```

> `padding='max_length'` 会将所有句子填充到 `max_length` 指定的长度；`padding='longest'` 则只填充到 batch 内最长句子的长度。（④）

> **本地运行配置要求**：批量处理时内存占用与 batch 大小和序列长度成正比。示例中 batch=2、max_length=12，几乎无压力。实际训练时若 batch_size=64、max_length=512，建议内存 ≥ 8GB。

## 六、不同模型的 Tokenizer 差异（③）

不同预训练模型的 tokenizer 在特殊 token 和分词行为上有显著差异，使用时需要特别注意：（③）

以下列举了一些常见模型 checkpoint 的分词方案，具体行为以该 checkpoint 的配置为准：

| 模型 checkpoint | 分词算法 | 特殊 token | 注意事项 |
|---|---|---|---|
| **`bert-base-chinese`** | WordPiece | `[CLS]`, `[SEP]`, `[PAD]` | 双句分类任务用 `[SEP]` 分隔；输入长度通常限制在 512 以内 |
| **`gpt2`** | Byte-level BPE | `<\|endoftext\|>` | 默认没有 `pad_token`，通常需要手动设置；输入长度通常限制在 1024 以内 |
| **`meta-llama/Llama-2-7b-hf`** | SentencePiece BPE | `<s>`, `</s>` | 需严格设置 `pad_token`；原始上下文长度 4096 |
| **`roberta-base`** | BPE | `<s>`, `</s>` | 使用 `</s>` 替代 `[SEP]`；支持句子对任务 |

### 6.1 GPT-2 的 pad_token 缺失（③）

GPT-2 是单向自回归模型（生成任务），默认没有 `pad_token`。训练时通常需要手动设置：

```python
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token  # 用 eos_token 作为 pad_token
```

## 七、保存与加载本地分词器（①）

### 7.1 保存分词器（①）

```python
# 保存分词器到本地目录
tokenizer.save_pretrained("./my_tokenizer")

# 保存的内容通常包括（具体文件因 tokenizer 类型而异）：
# - tokenizer_config.json
# - vocab.txt（WordPiece 类）或 vocab.json + merges.txt（BPE 类）
# - special_tokens_map.json
```

### 7.2 加载本地分词器（①）

```python
# 从本地目录加载
tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")
```

> **本地运行配置要求**：保存操作生成约数 MB 的文件，磁盘预留相应空间即可。

## 八、词表操作与自定义词汇（③+④）

### 8.1 查看词表信息（③）

```python
# 查看词汇表大小
print(len(tokenizer))  # 或 tokenizer.vocab_size

# 查看最大序列长度
print(tokenizer.model_max_length)  # 例如 bert-base-chinese 通常为 512
```

### 8.2 添加自定义词汇（③+④）

如果词表中没有某些专业词汇（如医学术语），可以手动添加：

```python
# 添加新词
num_added = tokenizer.add_tokens(["新词1", "新词2"])
print(f"添加了 {num_added} 个新词")

# 添加特殊 token
tokenizer.add_special_tokens({"cls_token": "<my_cls>"})

# 如果配合模型使用，需要同步更新模型的 embedding 层
# model.resize_token_embeddings(len(tokenizer))
```

> 添加新词后，该词通常会被当作一个整体 token 处理，不会再被拆分成子词。（④）

> **本地运行配置要求**：词表操作纯 CPU 计算，无额外配置要求。

## 九、常见问题与解决（①+②+③+④）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `OSError: Can't load tokenizer` | 本地目录缺少必要的配置文件 | 确保本地目录包含完整的分词器文件；推荐使用 `save_pretrained()` 保存的目录 |
| 分词结果出现预期外的空格 | `use_fast=True` 的兼容性问题 | 设置 `use_fast=False` 回退到慢速分词器 |
| GPT-2 padding 报错 | 默认没有 `pad_token` | 手动设置 `tokenizer.pad_token = tokenizer.eos_token` |
| 解码后中文被空格分隔 | decode 默认按 token 拼接 | 正常现象，设置 `skip_special_tokens=True` 即可 |
| 添加新词后模型输出维度不匹配 | 未更新模型 embedding 层 | 执行 `model.resize_token_embeddings(len(tokenizer))` |
| 加载速度极慢 | 网络问题或缓存目录问题 | 使用 `cache_dir` 指定本地缓存，或提前下载 |

## 十、总结与后续学习路线（①+③）

恭喜你完成了 HuggingFace Tokenizer 加载与使用的学习！你现在应该掌握了：

- ✅ `AutoTokenizer` 的作用：智能识别并加载对应分词器
- ✅ `from_pretrained()`：从 Hub 或本地加载，理解 `use_fast`、`cache_dir`、`local_files_only` 等参数
- ✅ 编码流程：`tokenizer()` 统一接口、返回字段（`input_ids`、`attention_mask`、`token_type_ids`）
- ✅ 解码方法：`decode()` 与 `skip_special_tokens`
- ✅ 批量处理：传入列表自动 batch 编码
- ✅ 不同模型差异：BERT 与 GPT-2 等模型的特殊 token 区别
- ✅ 保存加载：`save_pretrained()` 与本地加载
- ✅ 词表扩展：`add_tokens()` 与 `add_special_tokens()`

> **下一步预告**：在下一节 **"熟悉 padding、truncation、max_length 等关键参数"** 中，我们将深入讲解这三个核心参数的机制与最佳实践。你会理解 `padding='longest'` 与 `'max_length'` 的区别、`truncation` 的多种策略、以及如何在实际训练循环中正确配置这些参数，让不同长度的文本能够整齐地组成 batch。现在，试着加载 `bert-base-chinese` 和 `gpt2` 两个分词器，对同一段文本观察它们的切分差异吧！