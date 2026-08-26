# 学习包

## 块 1

**本块学习检查**：阅读完本节后，请暂停并回答——原始文本数据（如`{"text":"人工智能正在改变世界"}`）能否直接输入大模型？如果不能，模型实际需要的数据格式是什么？

---

## 一、为什么需要Datasets库？

### 1.1 大模型训练为什么需要数据处理？

**核心一句话：原始数据不能直接喂给模型，必须先“翻译”成模型能读的格式。**

大语言模型的训练流程是这样的：

```
原始文本数据 → 数据清洗 → 格式转换 → Tokenizer编码 → Dataset → 模型训练
```

每一步都在为最终训练做准备。你现在的目标是 **LoRA 微调**，而微调的第一步就是准备好训练数据——这正是本节要解决的问题。

**举个具体例子：**

原始数据长这样（人类能读懂的文本）：

```json
{
  "text": "人工智能正在改变世界"
}
```

但模型实际需要的，是经过 Tokenizer 编码后的数值（模型只认数字，不认文字）：

```python
{
  "input_ids": [..., ..., ...],
  "attention_mask": [1, 1, ...]
}
```

> 注意：上面 `input_ids` 里的 `...` 只是**示意**，实际数值取决于你所用模型的分词器，请以实际运行输出为准。

**为什么不能直接用原始文本？** 因为模型内部的计算全部基于向量和矩阵运算，必须先把文本拆分成 token（词元），再映射为对应的 ID 数字，同时用 `attention_mask` 标记哪些位置是有效内容。

**这跟你学 LoRA 有什么关系？** LoRA 微调的本质是在冻结原模型参数的基础上，用少量可训练参数适配新任务。但无论参数怎么调，**输入数据的格式必须和预训练时一致**——这就是为什么你需要掌握 Datasets 库，用它来完成数据清洗、格式转换和 Tokenizer 编码。

---

**动手验证（可选）

## 块 2

### 1.2 Hugging Face Datasets是什么？（①）

`datasets` 是 Hugging Face 提供的数据处理库，专为高效处理大规模数据集而设计。它的核心功能包括：

- **加载数据集**：从Hub或本地文件加载数据。
- **数据转换**：修改字段或进行Tokenizer编码。
- **数据过滤**：删除无效样本。
- **数据随机化**：通过`shuffle`打乱数据顺序。
- **高效处理**：支持大规模数据集，避免内存溢出。

**安装与导入**：

```bash
pip install datasets
```

```python
from datasets import Dataset
```

**动手试一试**：安装库后，运行`from datasets import Dataset`，确认无报错即可。这是你后续

## 块 3

### 2.1 Dataset是什么？

把 Dataset 想成一张**专门给机器学习用的数据表**，就像 Python 的 `list` 或 Pandas 的 `DataFrame`。

**动手试一下**（打开你的 Python 环境，跟着敲）：

```python
from datasets import Dataset

dataset = Dataset.from_dict(
    {
        "text":[
            "你好",
            "欢迎学习LLM"
        ]
    }
)
print(dataset)
```

你会看到输出：

```
Dataset({
features:['text'],
num_rows:2
})
```

这表示：这张表有 1 个字段（`text`），2 行数据。

**查看内容**：

```python
print(dataset[0])          # 看第一条
print(dataset.column_names)  # 看有哪些字段
```

输出分别是：

```
{"text":"你好"}
['text']
```

---

**本块学习检查**：  
用 `Dataset.from_dict` 创建一个包含 3 条文本的数据集，然后

## 块 4

### 3.1 map()是什么？

**一句话理解**：`map()` 就是给Dataset里的**每一条数据**执行同一个处理函数，像流水线一样逐个处理。

**两种模式（写函数前先确认）**：
- **默认模式**（`batched=False`）：函数每次收到**单条样本**，形如 `{"text": "..."}`。
- **批量模式**（`batched=True`）：函数收到**一批样本按列组织的字典**，形如 `{"text": ["...", "...", ...]}`，此时要处理的是**列表**。

**常见用途**：
- Tokenizer编码（为LoRA微调准备输入）
- 添加新字段
- 修改数据格式
- 数据清洗

**类比**：就像Python内置的 `map(func, list)` 对列表每个元素套用函数（注意：Python的 `list` 本身没有 `.map()` 方法）。

---

**实训说明**：本章是纯CPU数据预处理，不涉及模型训练。最低配置双核CPU+8GB内存+10GB磁盘即可，推荐4核+16GB内存，**不需要GPU**。

---

**动手验证**

## 块 5

**动手做：给数据加个“长度”字段**

你正在为LoRA微调准备数据，先掌握最常用的`map`操作。它就像给每条数据“贴标签”——批量处理每一行。

**第一步：造一个迷你数据集**

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": ["Hello", "Hugging Face"]
})
```

**第二步：写一个处理函数**

这个函数接收一条样本（`example`），计算`text`的长度，并作为新字段`length`加回去：

```python
def add_length(example):
    example["length"] = len(example["text"])
    return example
```

**第三步：用`map`执行**

```python
dataset = dataset.map(add_length)
```

**第四步：验证结果**

```python
print(dataset)
```

你会看到每条数据都多了`length`字段，比如`"Hello"`对应

## 块 6

**加载Tokenizer（动手做）**

LoRA微调前，先准备分词器：

```python
from transformers import AutoTokenizer
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**编写Tokenize函数**

定义处理函数供`map`使用：

```python
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
```

- `truncation=True`：超长截断
- `padding="max_length"`：统一填充至128
- `max_length=128`

## 块 7

## 5.3 使用map批量处理

你的目标是LoRA微调，而微调前必须把文本转成模型能读的格式。`map`就是帮你**一次性处理整个数据集**的工具——不用写for循环，一行代码搞定。

```python
tokenized_dataset = dataset.map(tokenize)
```

这行代码会把`tokenize`函数应用到数据集里的**每一条样本**上，结果自动存成新数据集。

**看结果**：

```python
print(tokenized_dataset[0])
```

输出类似（以`bert-base-uncased`为例）：

```python
{
"text":"Hello",
"input_ids":[101,7592,102],
"attention_mask":[1,1,1]
}
```

**关键点**：
- `101`和`102`是BERT的`[CLS]`和`[SEP]`特殊token，不用管它们，模型自己会用
- `input_ids`是文本转成的数字编号，模型通过这些编号查找词嵌入。

## 块 8

## 6.1 为什么需要batched？——从“逐个处理”到“批量处理”

默认情况下，`map()` 是一条一条处理数据的：

```
example1
example2
example3
```

每条数据单独走一遍处理流程。数据量小的时候没问题，但当你做 LoRA 微调、面对上万条训练样本时，这种“逐个处理”的方式效率就很低了。

开启 `batched=True` 后，`map()` 会**一次处理一批数据**：

```
[
example1,
example2,
example3
]
```

把多条数据打包成一个列表，一次性交给处理函数。这就是批量处理的核心思路：**减少重复开销，充分利用硬件和库的批处理能力**。

---

## 6.2 示例：批量 Tokenize

```python
from transformers import AutoTokenizer
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True
    )

dataset = dataset.map(
    tokenize_batch,
    batched=True
)
```

注意：当 `batched=True` 时，这里的 `batch["text"]` 是一个**列表**（包含多条文本），而不是单条字符串。当 `batched=False` 时，`batch` 是单个样本的字典，`batch["text"]` 是单个字符串。Tokenizer 会一次性处理整个列表。

**优势：**

- **更快**：批量操作减少了函数调用和内存拷贝的开销；
- **更适合大规模数据**：LoRA 微调通常需要处理数万条样本，批量处理是标配；
- **充分利用 Tokenizer 批处理能力**：Hugging Face Tokenizer 底层针对批量输入做了优化。

> ⚠️ **重要提醒**：这里的 `padding=True` 只会把**当前这个 map batch 内**的样本补齐到该 batch 的最长长度。**不同 map batch 之间的序列长度仍可能不同**。如果你的下游训练要求统一长度，应该在训练阶段配合**动态填充的 DataCollator**（如 `DataCollatorWithPadding`，按训练 batch 动态补齐），或者在这里直接用固定的 `padding="max_length", max_length=...`。

---

## 七、remove_columns：删除无用字段

Tokenizer 之后，你的数据集可能长这样：

```python
{
    "text": "Hello",
    "input_ids": [...]
}
```

训练时，模型通常只需要：

```python
input_ids
attention_mask
```

`text` 字段已经完成使命，留着只会占内存。删除它：

```python
dataset = dataset.remove_columns(["text"])
```

查看确认：

```python
print(dataset.column_names)
```

结果：

```
["input_ids", "attention_mask"]
```

> 💡 **补充说明**：`input_ids` / `attention_mask` 是**输入**，训练还需要**标签**：
> - 文本**分类**：需要 `labels

## 块 9

**filter() 是什么？**
一句话：按条件删数据。比如删掉空文本、过短文本、垃圾数据。

**怎么用？** 三步走：

1. **准备数据**：先有一个数据集，里面装着文本。
2. **写规则**：定义一个函数，判断每条数据是否“合格”。比如，`check_length` 检查文本长度是否大于 2。
3. **执行过滤**：调用 `dataset.filter(check_length)`，不合格的（长度≤2的）自动被删掉。

**看个例子**：数据是 `["hi", "hello world", ""]`，规则是“长度>2”。过滤后，只剩 `"hello world"`，`"hi"` 和空文本都被删了。

**动手试一下**：把

## 块 10

## 9.1 为什么需要shuffle？

你正在为LoRA微调准备数据。先想一个问题：如果训练数据顺序固定，比如前100条全是类别A，接着100条全是类别B，模型会怎样？

它会“偷懒”——学到数据顺序的规律，而不是真正理解内容。比如它可能学会“看到A之后猜A”，而不是学会区分A和B。

`shuffle()` 就是解决这个问题的：**随机打乱训练样本的顺序**，让模型无法依赖顺序，只能老老实实学内容本身。

---

## 9.2 基础使用

一行代码搞定：

```python
dataset = dataset.shuffle()
```

---

## 9.3 设置随机种子

打乱是随机的，但实验需要可复现。用 `seed` 固定随机结果：

```python
dataset = dataset.shuffle(seed=42)
```

相同 `seed`，每次打乱结果一致。这样你跑两次实验，数据顺序相同，结果差异就只来自模型本身。

---

## 十、数据预处理流程示例

> 注意：下面演示 `filter → shuffle → map(tokenize)` 这条核心链路，**只是概念示意**，不是LoRA/QLoRA训练的完整流程。真实训练前还需要划分训练/验证集、删除多余列、构造标签（如Causal LM的`labels`，padding掩码设为`-100`）等。

假设原始数据：

```python
dataset = Dataset.from_dict(
{
"text":[
"Hello AI",
"H

## 块 11

```python
dataset = dataset.filter(lambda x: len(x["text"]) > 5)
```

**第二步：随机打乱**  
用`shuffle`打乱顺序，`seed=42`保证每次运行结果一致，方便复现：

```python
dataset = dataset.shuffle(seed=42)
```

**第三步：Tokenizer编码**  
定义`preprocess`函数，把文本转成模型能读的token，`truncation=True`截断超长文本，`max_length=128`限制长度。然后用`map`应用到整个数据集：

```python
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

dataset = dataset.map(preprocess)
```

## 块 12

## 十.1 面向 Causal LM 的较完整数据准备示例（含划分/标签/动态填充）

前面你已经掌握了“清洗 + 编码”的核心链路，但那是单次处理。真实训练中，你还需要**划分数据集**、**构造标签**，以及**在训练时动态填充**。这一节用一个完整示例，把这几步串起来。

### 为什么需要这套流程？

你要做 LoRA 微调，数据准备的质量直接决定微调效果。Causal LM（自回归语言模型）的训练数据需要满足两个特殊要求：

1. **文本要拼接成完整段落**——把 prompt 和 response 连在一起，模型才能学会“看到上文，预测下文”。
2. **标签就是输入本身**——对 Causal LM 来说，`labels` 直接等于 `input_ids`，模型学习的是预测下一个 token。

### 完整代码示例

```python
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# 1) 准备原始数据（这里用少量样本演示流程）
raw = Dataset.from_dict({
    "prompt": ["翻译成英文：你好", "续写：今天天气"],
    "response": ["Hello", "很好，适合出门。"],
})

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LEN = 128

# 2) 文本拼接 + 截断；Causal LM 通常让 labels = input_ids
def build_example(example):
    # 把 prompt 与 response 拼成一段训练文本（真实项目可用 chat template）
    text = example["prompt"] + "\n" + example["response"] + tokenizer.eos_token
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN)
    enc["labels"] = enc["input_ids"].copy()   # 语言模型标签
    return enc

# 先划分再各自处理，避免验证集信息在处理时"泄漏"进训练集统计
split = raw.train_test_split(test_size=0.5, seed=42)   # 得到 DatasetDict(train/test)

tokenized_train = split["train"].map(
    build_example,
    remove_columns=split["train"].column_names,   # 去掉原始文本列，只留 input_ids/attention_mask/labels
)
tokenized_test = split["test"].map(
    build_example,
    remove_columns=split["test"].column_names,
)

# 3) 训练阶段动态填充：按每个训练 batch 的最长长度补齐（比在 map 里固定 max_length 更省算力）
#    Causal LM 用 DataCollatorForLanguageModeling(mlm=False)；分类任务则用 DataCollatorWithPadding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# tokenized_train / tokenized_test 即可分别作为 train_dataset / eval_dataset
# data_collator 传给 Trainer(..., data_collator=data_collator)（完整 Trainer 用法见 11.4）
```

### 逐步拆解：每一步在做什么？

**第 1 步：准备原始数据。** 用 `Dataset.from_dict()` 创建一个包含 `prompt` 和 `response` 两

## 块 13

## 十.1 面向 Causal LM 的较完整数据准备示例

### 核心流程：先划分，再处理

**为什么先划分？** 使用 `datasets` 库的 `train_test_split()` 方法把数据分成 `train` 和 `test` 两部分，返回一个 `DatasetDict`。设置 `seed` 参数保证每次划分结果一致。**关键原则**：先划分再处理——如果先做拼接或统计再划分，验证集会“偷看”训练集的信息，造成**数据泄漏**，让评估结果虚高。

### 标签怎么设？

- **Causal LM（自回归语言模型）**：最常见的做法是让 `labels = input_ids`，即标签就是输入本身。如果只想对 response 部分计算损失，可以把 prompt 和 padding 位置的 token 设为 `-100`（PyTorch 交叉熵损失会自动忽略这个值）。
- **分类任务**：需要保留或构造 `labels`（类别 id），并且改用 `DataCollatorWithPadding` 来处理。

### 拼接与截断

用 `truncation=True` 配合 `max_length=...` 控制序列长度。真实对话数据建议用 `tokenizer.apply_chat_template` 来拼接多轮对话，而不是手动拼字符串。

### 动态填充：省显存的关键

**不要在 `map` 里固定 padding**。正确做法是用 DataCollator 在训练时按 batch 动态补齐——每个 batch 只填充到该 batch 的最长长度，而不是全局最大长度，通常更省显存和算力。如果数据量小或特殊需求，也可以在 `map` 里固定 `padding="max_length"`。

> ⚠️ 注意：示例中 `test_size=

## 块 14

## 十.1 面向 Causal LM 的较完整数据准备示例（含划分/标签/动态填充）

你的目标是LoRA微调，而微调的第一步是准备好数据。下面这段代码演示了从原始文本到可训练Dataset的完整链路，我们一步步拆开看。

```python
from datasets import Dataset

dataset = Dataset.from_dict(
    {
        "text": [
            "hello",
            "machine learning",
            "large language model"
        ]
    }
)

dataset = dataset.shuffle(seed=42)      # 随机打乱顺序，固定种子保证可复现
dataset = dataset.filter(
    lambda x: len(x["text"]) > 5        # 只保留长度大于5的样本
)

print(dataset)
```

运行 `python test_dataset.py`，成功输出：

```
Dataset({
features:['text'],
num_rows:2
})
```

**发生了什么？**

1. **创建**：`Dataset.from_dict` 把Python字典包装成Dataset对象，3条文本进入。
2. **shuffle**：`seed=42` 让随机化可复现——每次运行打乱结果一致。
3. **filter**：lambda函数逐条检查，`"hello"`长度为5，不满足大于5的条件，被过滤掉；`"machine learning"`长度为17，`"large language model"`长度为20，均满足条件，剩下2条。

**为什么这对LoRA微调重要？** 微调前你需要：清洗低质量数据（filter）、

## 测评题目

### Q-KB11-ST1102-003
为了在每个训练轮次打乱样本顺序，通常使用？

### Q-KB11-ST1102-001
datasets 的 `map()` 常用于对数据集样本执行批量________。

### Q-KB1101-003
请写代码：使用 GPT-2 模型和分词器，对输入文本 `"Hello, I am"` 生成接下来的 20 个词（设置 `max_new_tokens=20`），并解码输出。

### Q-KB11-ST1100-003
以下哪个库最直接用于训练配置中的分布式加速？

### Q-KB1103-001
TRL（Transformer Reinforcement Learning）库中，专门用于监督微调（SFT）的Trainer类是？

### Q-KB1103-004
在 `SFTConfig` 中，`max_seq_length` 参数的作用是？

### Q-KB1103-003
请写出使用 `SFTTrainer` 进行训练的基本代码框架（包含 `model`, `train_dataset`, `args`, `data_collator`）。

### Q-KB1101-004
`AutoModelForCausalLM` 和 `AutoModelForMaskedLM` 的区别是？
