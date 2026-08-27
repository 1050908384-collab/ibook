# 学习包

## 块 1

## 一、为什么需要Datasets库？

### 1.1 大模型训练为什么需要数据处理？

你要做LoRA微调，第一步不是写模型代码，而是先把数据准备好。看下面这个流程，数据准备是整个训练链路的起点：

```
原始文本数据 → 数据清洗 → 格式转换 → Tokenizer编码 → Dataset → 模型训练
```

**关键认知：原始数据不能直接输入模型。**

举个例子。你手头有一条原始数据：

```json
{
  "text": "人工智能正在改变世界"
}
```

但模型要的不是这段文字，而是这样的结构（ID仅为示意，实际数值取决于所用模型的分词器，请以实际运行输出为准）：

```python
{
  "input_ids": [..., ..., ...],
  "attention_mask": [1, 1, ...]
}
```

也就是说，模型吃的是**数字**，不是**文字**。从 `"人工智能正在改变世界"` 到 `input_ids`，中间需要经过分词、编码、填充、截断等一系列处理。

**这跟你做LoRA微调有什么关系？**

LoRA微调的本质是让模型在特定任务上继续学习，而学习素材就是经过处理的输入输出对。如果你的数据没处理好——比如文本没清洗、长度没统一、格式不对——模型训练时就会报错，或者学出来的效果很差。

所以，你需要一个专门的数据处理工具，把"原始文本"变成"模型能吃的格式"。这就是Hugging Face Datasets库存在的意义。

接下来的内容会带你掌握三个核心操作：`map()`（数据转换）、`filter()`（数据筛选）、`shuffle()`（数据打乱）。完成这一章后，你就能独立完成训练数据的清洗、Token化处理和预处理，为后面的LoRA微调打好地基。

---

**本块学习检查**：用自己的话说一遍——为什么原始文本不能直接喂给模型

## 块 2

**本块学习检查**：安装并导入 `datasets` 库后，运行 `from datasets import Dataset`，确认无报错即通过。

---

`datasets` 是 Hugging Face 的数据处理库，专为高效处理大规模数据而生。它的核心能力围绕四个操作：**加载**（从Hub或本地文件）、**转换**（改字段、Tokenizer编码）、**过滤**（删无效样本）、**随机化**（shuffle打乱）。

**上手三步：**

1. 安装：`pip install datasets`
2. 导入：`from datasets import Dataset`
3. 开始处理数据

**针对你的LoRA微调目标**：微调前，你需要用 `map` 对数据集做Tokenizer编码（如

## 块 3

**本块学习检查**：用 `Dataset.from_dict` 创建含两行文本的数据集，打印 `dataset` 观察输出格式，再分别用 `dataset[0]` 和 `dataset.column_names` 查看首行与字段名，确认与下方示例一致。

---

**Dataset 是什么？**  
把它想成一张**专为机器学习训练准备的数据表**，类似 Python 的 `list` 或 Pandas 的 `DataFrame`。  

**创建示例**（直接复制运行）：  
```python
from datasets import Dataset
dataset = Dataset.from_dict({
    "text": ["你好", "欢迎学习LLM"]
})
print(dataset)
```
输出会显示：  
```
Dataset({
features: ['text'],
num_rows: 2
})
```
这表示：数据集有 1 个字段 `text`，共 2 行数据。  

**查看内容**：  
- 取第一行：`print(dataset[0])`

## 块 4

`map()` 是 Hugging Face Datasets 里做数据转换的核心工具。它的作用很简单：**对数据集中的每一条数据，执行一个你写好的处理函数**。

你可以把它想象成 Python 内置的 `map(func, list)`——对列表里每个元素依次套用同一个函数。注意，Python 的 `list` 本身没有 `.map()` 方法，这是 Datasets 特有的。

**关键点：两种模式，别搞混**

- 默认（`batched=False`）：你的函数每次收到**单条样本**，比如 `{"text": "你好"}`。
- 设置 `batched=True`：函数收到的是**一批样本按列组织的字典**，比如 `{"text": ["你好", "世界", "..."]}`，这时你要对**列表**做批量处理。

写函数前，先确认自己处于哪种模式，否则容易出错。

**常见用途**：Tokenizer 编码、添加新字段、修改数据格式、数据清洗。

**实训提示**：本章只做数据处理，不涉及模型训练。CPU 双

## 块 5

### 基础map使用：给数据加个“长度”字段

你正在为LoRA微调准备数据，`map`是处理数据集的核心工具。它的作用很简单：**对数据集里的每一条样本，执行你定义的函数，然后返回更新后的数据集**。

**第一步：创建数据**

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": ["Hello", "Hugging Face"]
})
```

**第二步：定义处理函数**

这个函数接收一条样本（一个字典），给它加一个新字段，然后返回：

```python
def add_length(example):
    example["length"] = len(example["text"])
    return example
```

**第三步：执行map**

```python
dataset = dataset.map(add_length)
```

**第四步：查看结果**

```python
print(dataset)
```

你会看到每条样本都多了`length`字段，比如`"Hello"`变成了`{"text

## 块 6

### 5.1 加载Tokenizer

先加载一个预训练好的分词器，它是把文本转成模型能读的数字ID的工具：

```python
from transformers import AutoTokenizer

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### 5.2 编写Tokenize函数

接着写一个函数，对数据集里的每条样本做编码。`truncation=True`截断超长文本，`padding="max_length"`把短文本补到固定长度，`max_length=128`设定统一长度：

```python
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
```

**本块学习

## 块 7

```markdown
你的目标是用 LoRA 微调模型，而这一步正是微调前的“数据预处理”。`map` 方法能让你用一行代码，把 `tokenize` 函数应用到数据集里的每一条样本上，批量完成编码。

```python
tokenized_dataset = dataset.map(
    tokenize
)
```

运行后，`tokenized_dataset` 里每一条数据都会多出两个字段：`input_ids`（token 编号）和 `attention_mask`（注意力掩码）。你可以打印第一条看看效果：

```python
print(tokenized_dataset[0])
```

输出大致是这样（以 `bert-base-uncased` 为例，`101`/`102` 是 BERT 的特殊 token `[CLS]`/`[SEP]`，中间是词的编号；实际数值以运行为准）：

```python
{
"text":"Hello",
"input_ids":[101,7592,102],
"attention_mask":[1,1,1]
}
```
```

## 块 8

## 为什么需要 batched？

先看默认行为：`map()` 是一条一条处理数据的，就像排队过安检，一次只过一个人：

```
example1 → 处理 → 输出
example2 → 处理 → 输出
example3 → 处理 → 输出
```

数据量小的时候没问题，但当你处理几万、几十万条样本时，这种串行方式会非常慢。

**开启 `batched=True` 后**，`map()` 会一次性接收一个批次的数据，该批次是一个字典，其中每个键对应一个字段，值为该批次内所有样本该字段的列表，批量处理：

```python
{
  "text": [example1, example2, example3]
} → 一次处理 → 批量输出
```

---

### 实际示例：批量tokenize

```python
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

注意这里的 `batch["text"]` 是一个**列表**（该批次内所有样本的 text 字段），而不是单个字符串。Tokenizer 本身支持批量输入，`batched=True` 正好把这个能力用上。

**优势：**
- **更快**：减少 Python 层面的循环开销，充分利用 Tokenizer 的批处理能力；
- **更适合大规模数据**：数据量越大，收益越明显。

---

### ⚠️ 一个容易踩的坑

`padding=True` 只会在**当前这个 map batch 内部**把样本补齐到该 batch 的最长长度。也就是说：

- batch 1 的最长样本是 100 token → batch 1 全部补到 100；
- batch 2 的最长样本是 80 token → batch 2 全部补到 80。

**不同 map batch 之间的序列长度仍然可能不同。** 如果你的下游训练要求统一长度，有两种做法：

1. **在训练阶段**用动态填充的 `DataCollator`（如 `DataCollatorWithPadding`），按训练 batch 动态补齐；
2. **在这里**直接用固定长度：`padding="max_length", max_length=某个值`。

---

### 本块学习检查

打开你的数据集，对比以下两种方式的速度差异（可以用小数据集先试）：

```python
# 方式A：不开启 batched
dataset.map(lambda x: tokenizer(x["text"], truncation=True))

# 方式B：开启 batched
dataset.map(lambda batch: tokenizer(batch["text"], truncation=True), batched=True)
```

观察：方式B的代码里 `batch["text"]` 是什么类型？为什么 `tokenizer` 能直接处理它？

## 块 9

**filter()：给数据“体检”**

`filter()` 就是按条件删除不符合要求的数据。比如删掉空文本、过短文本或垃圾数据。

**动手试一下**

先造点数据：
```python
dataset = Dataset.from_dict({"text": ["hi", "hello world", ""]})
```

定义规则（保留长度大于2的文本）：
```python
def check_length(example):
    return len(example["text"]) > 2
```

过滤：
```python
dataset = dataset.filter(check_length)
```

**结果**：只保留 `"hello world"`，`"hi"` 和空文本都被删掉了。

**本块学习检查**：把规则改成保留长度大于5的文本，预测过滤后剩什么，再运行验证。

## 块 10

## 9.1 为什么需要shuffle？

你要做 LoRA 微调，数据顺序是个隐藏陷阱。如果数据固定排列成：

```
类别A
类别A
类别A
类别B
类别B
```

模型会偷懒——它可能不是在学"如何分类"，而是在学"前三个是A，后两个是B"这种顺序规律。这会让微调效果失真。

**shuffle 的作用**：把训练样本顺序随机打乱，强迫模型真正学习数据本身的特征，而不是顺序套路。

---

## 9.2 基础使用

一行代码搞定：

```python
dataset = dataset.shuffle()
```

---

## 9.3 设置随机种子

LoRA 微调需要可复现的实验结果。加个 `seed` 参数：

```python
dataset = dataset.shuffle(seed=42)
```

**关键点**：相同 `seed`，每次打乱结果完全一致。这样你调整其他参数时，能确定效果差异来自参数本身，而不是数据顺序变了。

---

## 十、数据预处理流程示例

> 注意：下面这条链路是**概念示意**，帮你理解步骤关系。真实 LoRA 训练前还需要划分训练/验证集、构造标签、处理 padding 等，这里先聚焦核心流程。

假设原始数据：

```python
dataset = Dataset.from_dict(
{
"text":[
"Hello AI",
"Hugging Face",
"Large Language Model"
]
}
)
```

**核心链路**：`filter → shuffle → map(tokenize)

## 块 11

**本块学习检查**：运行下方代码后，打印 `dataset` 的长度，确认清洗后每条样本的 `text` 长度都大于5。

---

你的LoRA微调数据管线，从清洗开始。这段代码用 `filter` 丢掉过短的文本——`len(x["text"])>5` 表示只保留字符数超过5的样本。**动手试**：把阈值改成10，观察数据集长度变化。

接着用 `shuffle(seed=42)` 打乱顺序。固定 `seed=42` 保证每次运行结果一致，方便复现实验。

最后用 `map` 对每条样本执行 `preprocess`：调用 `tokenizer` 将 `text` 编码，`truncation=True` 截断超长文本

## 块 12

## 十.1 面向 Causal LM 的较完整数据准备示例（含划分/标签/动态填充）

前面我们只做到“清洗 + 编码”，现在补一个更接近实战的 **Causal LM 数据准备**流程。你要做 LoRA 微调，数据准备是第一步，这里覆盖四个关键动作：**划分训练/验证集**、**文本拼接与截断**、**构造 `labels`**、**动态填充 DataCollator**。

### 完整代码示例

```python
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# 1) 准备原始数据（少量样本演示流程）
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

tokenized = split.map(
    build_example,
    remove_columns=split["train"].column_names,   # 去掉原始文本列，只留 input_ids/attention_mask/labels
)

# 3) 训练阶段动态填充：按每个训练 batch 的最长长度补齐（比在 map 里固定 max_length 更省算力）
#    Causal LM 用 DataCollatorForLanguageModeling(mlm=False)；分类任务则用 DataCollatorWithPadding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# tokenized["train"] / tokenized["test"] 即可分别作为 train_dataset / eval_dataset
# data_collator 传给 Trainer(..., data_collator=data_collator)（完整 Trainer 用法见 11.4）
```

### 四个关键点逐一说清

**① 先划分，再处理。** 代码里用 `train_test_split(test_size=0.5, seed=42)` 先把原始数据切成训练集和验证集，得到 `DatasetDict`。顺序很重要——如果先做文本处理再划分，验证集的信息可能“泄漏”进训练集的统计中，影响评估的可靠性。

**② 文本拼接与截断。** `build_example` 函数把 `prompt` 和 `response` 拼成一段完整文本，末尾加上 `eos_token`（结束符

## 块 13

## 十.1 面向 Causal LM 的较完整数据准备示例

你的目标是 LoRA 微调，而 LoRA 微调通常作用于 Causal LM（因果语言模型）。所以这一节的数据准备方式，就是你后续微调时最常遇到的场景。我们直接上手，把流程走通。

### 第一步：划分数据集（先划分，再处理）

```python
from datasets import Dataset

dataset = Dataset.from_dict({"text": [...]})  # 你的原始数据

# 先划分，再处理
split = dataset.train_test_split(test_size=0.5, seed=42)
# split 是一个 DatasetDict，包含 train 和 test 两个子集
train_dataset = split["train"]
eval_dataset = split["test"]
```

**为什么必须先划分再处理？** 如果先做统计或拼接再划分，验证集的数据会“看过”训练集的统计信息（比如词频、长度分布），导致评估结果虚高——这就是**数据泄漏**。`seed` 参数保证每次运行划分结果一致，可复现。

> ⚠️ 注意：这里 `test_size=0.5` 只是演示流程用，样本量极小，**不能反映真实训练效果**。实际项目中通常用 0.1~0.2。

### 第二步：构造标签（Causal LM 的核心）

Causal LM 的训练目标是“根据前文预测下一个 token”，所以最常见的做法是：

```python
def preprocess_function(examples):
    # tokenizer 编码文本
    inputs = tokenizer(examples["text"], truncation=True, max_length=512)
    # Causal LM：标签就是输入本身
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs
```

**关键点**：
- `labels = input_ids`：让模型学习预测每一个位置的下一 token。
- 如果你只想对 response 部分计算损失（比如对话场景），可以把 prompt 和 padding 位置的值设为 `-100`，这样损失函数会自动忽略这些位置。
- 真实对话数据建议用 `tokenizer.apply_chat_template` 拼接，而不是手动拼字符串。

### 第三步：动态填充（省显存的关键）

**不要在 `map` 里固定 padding**。原因很简单：一个 batch 里样本长度不一，如果都填充到最长长度，短样本的 padding token 也会参与计算，浪费显存和算力。

正确做法：在训练时用 DataCollator 按 batch 动态补齐。

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM 不需要 masked language modeling
)
```

这个 collator 会在每个 batch 内部自动把样本填充到该 batch 的最长长度，而不是全局最长。

### 第四步：连接 Trainer

Dataset 是数据处理和 Trainer 之间的桥梁，流程是：

```
数据文件 → datasets处理 → Trainer → 模型训练
```

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,      # 验证集必须有
    data_collator=data_collator,    # 动态填充
    # 还需要 TrainingArguments（输出目录、评估策略等）
)
```

> ⚠️ 上面只是**结构示意**，不能直接复制运行。完整可运行代码见后续章节。

### 常见坑位速查

## 块 14

## 十.1 面向 Causal LM 的数据准备示例（动手版）

你的目标是 LoRA 微调，而微调前的数据准备是必经之路。下面这段代码演示了从原始文本到可用 Dataset 的完整流程，请跟着敲一遍。

```python
from datasets import Dataset

# 1. 创建 Dataset：把字典转成数据集
dataset = Dataset.from_dict({
    "text": ["hello", "machine learning", "large language model"]
})

# 2. shuffle：打乱顺序，固定 seed=42 保证可复现
dataset = dataset.shuffle(seed=42)

# 3. filter：过滤掉长度 <= 5 的样本
dataset = dataset.filter(lambda x: len(x["text"]) > 5)

print(dataset)
```

保存为 `test_dataset.py`，运行：

```bash
python test_dataset.py
```

成功输出：

```
Dataset({
features:['text'],
num_rows:2
})
```

三个 ✅ 分别对应：Dataset 创建成功、shuffle 运行成功、filter 运行成功。

**关键理解**：原始 3 条数据，shuffle 打乱顺序后，filter 用 `len(x["text"]) > 5` 筛掉了 `"hello"`（长度 5，不满足 `>5`），最终剩 2 条。

**本块学习检查**：把 `seed=42` 改成 `seed=0`，重新运行，

## 测评题目

### Q-KB11-ST1102-002
datasets 的 `filter()` 会保留使筛选函数返回________的样本。

### Q-KB11-ST1102-003
为了在每个训练轮次打乱样本顺序，通常使用？

### Q-KB11-ST1102-001
datasets 的 `map()` 常用于对数据集样本执行批量________。

### Q-KB1103-001
TRL（Transformer Reinforcement Learning）库中，专门用于监督微调（SFT）的Trainer类是？

### Q-KB1101-002
在进行模型推理时，为了防止计算梯度占用显存，通常使用 `with torch.________():` 上下文管理器。

### Q-KB11-ST1100-003
以下哪个库最直接用于训练配置中的分布式加速？

### Q-KB1101-005
请写代码：将加载的模型移动到 GPU 上（如果可用），并使用 `torch.float16` 半精度加载以节省显存。

### Q-KB1103-005
请写代码：在 `SFTTrainer` 训练完成后，保存微调后的 LoRA 适配器权重到 `./final_lora_adapter` 目录。
