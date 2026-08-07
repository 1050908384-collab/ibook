# Hugging Face Datasets数据处理：掌握 map / filter / shuffle 核心操作（最终版）

## 参考资料

① 权威：《Hugging Face 大语言模型课程：数据集处理基础》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter5/1

② 权威：《Hugging Face 大语言模型课程：Datasets数据集加载与探索》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter5/2

③ 权威：《Hugging Face 大语言模型课程：使用map处理数据集》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter5/3

④ 权威：《Hugging Face 大语言模型课程：数据过滤与处理》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter5/4

⑤ 权威：《Hugging Face 大语言模型课程：大规模数据集处理》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter5/5

⑥ 参考文章：《HuggingFace datasets库使用详解》
　　网址：https://blog.csdn.net/The_Thieves/article/details/148304033

⑦ 参考文章：《HuggingFace Datasets数据处理方法总结》
　　网址：https://blog.csdn.net/weixin_38252409/article/details/143227531

---

**适合人群**：已经完成 Hugging Face 环境准备、掌握 AutoModel / AutoTokenizer 基础加载方式，希望进一步学习大模型训练数据准备的学习者。本文会详细讲解 Hugging Face Datasets 库中的核心数据处理方法，包括 `map()` 数据转换、`filter()` 数据筛选、`shuffle()` 数据随机打乱，以及这些操作在大模型微调流程中的实际应用。完成本章后，你将能够独立完成训练数据清洗、Token化处理和数据集预处理，为后续 Trainer、LoRA 微调打下基础。

---

# 一、为什么需要Datasets库？（①②）

## 1.1 大模型训练为什么需要数据处理？

大语言模型训练流程：

```
原始文本数据

↓

数据清洗

↓

格式转换

↓

Tokenizer编码

↓

Dataset

↓

模型训练
```

原始数据通常不能直接输入模型。

例如：

原始数据：

```json
{
"text":"人工智能正在改变世界"
}
```

模型需要（下面的 ID 仅为**示意**，实际数值取决于所用模型的分词器，请以实际运行输出为准）：

```python
{
"input_ids":[..., ..., ...],
"attention_mask":[1,1,...]
}
```

因此需要一个专门的数据处理工具。

---

## 1.2 Hugging Face Datasets是什么？（①）

`datasets` 是 Hugging Face 提供的数据处理库。

主要功能：

| 功能    | 说明               |
| ----- | ---------------- |
| 加载数据集 | 从Hub、本地文件加载      |
| 数据转换  | 修改字段、Tokenizer编码 |
| 数据过滤  | 删除无效样本           |
| 数据随机化 | shuffle打乱数据      |
| 高效处理  | 支持大规模数据集         |

安装：

```bash
pip install datasets
```

导入：

```python
from datasets import Dataset
```

---

# 二、Dataset基础概念（①②）

## 2.1 Dataset是什么？

Dataset可以理解为：

> 一个专门用于机器学习训练的数据表。

类似：

Python：

```python
list
```

或者：

Pandas：

```python
DataFrame
```

例如：

```python
dataset = Dataset.from_dict(
    {
        "text":[
            "你好",
            "欢迎学习LLM"
        ]
    }
)
```

查看：

```python
print(dataset)
```

输出：

```
Dataset({
features:['text'],
num_rows:2
})
```

---

## 2.2 查看数据内容

查看第一条：

```python
print(dataset[0])
```

输出：

```python
{
"text":"你好"
}
```

查看字段：

```python
print(dataset.column_names)
```

输出：

```
['text']
```

---

# 三、map()：数据转换核心方法（参考文章⑥⑦）

## 3.1 map()是什么？

`map()` 用于：

> 对Dataset中的每一条数据执行一个处理函数。

> 注意：默认（`batched=False`）时处理函数每次收到**单条样本**（形如 `{"text": "..."}`）；当设置 `batched=True` 时，函数收到的是**一批样本按列组织的字典**（形如 `{"text": ["...", "...", ...]}`），此时应对**列表**做批量处理。写函数前先确认自己处于哪种模式（详见后文第六章）。

常见用途：

* Tokenizer编码；
* 添加新字段；
* 修改数据格式；
* 数据清洗。

类似于对一个列表里的每个元素依次套用同一个函数（例如 Python 的内置函数 `map(func, list)`，注意 Python 的 `list` 本身并没有 `.map()` 方法）。

---

## 电脑配置批注

本章节主要进行数据处理，不涉及模型训练。

最低配置：

* CPU：

  * 双核以上
* 内存：

  * 8GB
* 磁盘：

  * 10GB以上

推荐配置：

* CPU：

  * 4核以上
* 内存：

  * 16GB

GPU：

* 不需要。

说明：

`datasets.map()`主要是CPU数据预处理过程。

---

# 四、使用map进行数据修改（参考文章⑥）

## 4.1 基础map使用

创建数据：

```python
from datasets import Dataset


dataset = Dataset.from_dict(
    {
        "text":[
            "Hello",
            "Hugging Face"
        ]
    }
)
```

定义处理函数：

```python
def add_length(example):

    example["length"] = len(
        example["text"]
    )

    return example
```

执行：

```python
dataset = dataset.map(
    add_length
)
```

查看：

```python
print(dataset)
```

结果：

```
{
text:"Hello",
length:5
}
```

---

# 五、map进行Tokenizer编码（重点）（参考文章⑥⑦）

大模型训练最常见：

```
文本

↓

Tokenizer

↓

input_ids

↓

模型
```

---

## 5.1 加载Tokenizer

例如：

```python
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

---

## 5.2 编写Tokenize函数

```python
def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
```

---

## 5.3 使用map批量处理

```python
tokenized_dataset = dataset.map(
    tokenize
)
```

查看：

```python
print(tokenized_dataset[0])
```

输出类似（以 `bert-base-uncased` 为例；`101`/`102` 分别是 BERT 的 `[CLS]`/`[SEP]` 特殊 token，中间为该词的编号，具体数值以实际运行为准。注意：若沿用 5.2 中 `padding="max_length", max_length=128` 的写法，`input_ids`/`attention_mask` 会被补齐到 128 长度，这里为便于展示只截取了有效部分）：

```python
{
"text":"Hello",
"input_ids":[101,7592,102],
"attention_mask":[1,1,1]
}
```

---

# 六、batched参数：批量处理优化（参考文章⑥）

## 6.1 为什么需要batched？

默认：

一次处理一条数据：

```
example1
example2
example3
```

数据量大时效率低。

开启：

```python
batched=True
```

一次处理：

```
[
example1,
example2,
example3
]
```

---

## 6.2 示例

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

优势：

* 更快；
* 更适合大规模数据；
* 充分利用Tokenizer批处理能力。

> 注意：这里的 `padding=True` 只会把**当前这个 map batch 内**的样本补齐到该 batch 的最长长度，**不同 map batch 之间的序列长度仍可能不同**。若下游训练要求统一长度，应在训练阶段配合**动态填充的 DataCollator**（如 `DataCollatorWithPadding`，按训练 batch 动态补齐），或在这里直接用固定的 `padding="max_length", max_length=...`。

---

# 七、remove_columns删除无用字段（参考文章⑦）

Tokenizer之后：

原始：

```python
{
"text":"Hello",
"input_ids":[...]
}
```

训练通常至少需要：

```python
input_ids
attention_mask
```

> 补充：`input_ids` / `attention_mask` 是**输入**，训练还需要**标签**：
> - 文本**分类**：需要 `labels`（类别 id）；
> - **Causal LM**（自回归生成）：通常由 `input_ids` 构造 `labels`（常见做法是让 `labels = input_ids`，并把 prompt/padding 位置置为 `-100` 以忽略损失）；
> - **Seq2Seq**（翻译/摘要）：除 encoder 输入外，还可能需要 decoder 侧的标签字段。
>
> 因此"只保留 `input_ids` 和 `attention_mask`"仅适用于**不需要额外标签列**的场景；删除列前请确认标签已正确构造。

删除：

```python
dataset = dataset.remove_columns(
    ["text"]
)
```

查看：

```python
print(dataset.column_names)
```

结果：

```
[
"input_ids",
"attention_mask"
]
```

---

# 八、filter()：过滤数据（参考文章⑥⑦）

## 8.1 filter是什么？

`filter()`：

> 根据条件删除不符合要求的数据。

例如：

删除：

* 空文本；
* 过短文本；
* 垃圾数据。

---

## 8.2 基础使用

数据：

```python
dataset = Dataset.from_dict(
{
"text":[
"hi",
"hello world",
""
]
}
)
```

定义规则：

```python
def check_length(example):

    return len(
        example["text"]
    ) > 2
```

过滤：

```python
dataset = dataset.filter(
    check_length
)
```

结果：

保留：

```
hello world
```

删除：

```
hi
空文本
```

---

# 九、shuffle()：随机打乱数据（参考文章⑥）

## 9.1 为什么需要shuffle？

训练模型时：

如果数据顺序固定：

```
类别A
类别A
类别A
类别B
类别B
```

模型可能学习到数据顺序规律。

shuffle：

随机化训练样本顺序。

---

## 9.2 基础使用

```python
dataset = dataset.shuffle()
```

---

## 9.3 设置随机种子

为了复现实验：

```python
dataset = dataset.shuffle(
    seed=42
)
```

相同seed：

每次随机结果一致。

---

# 十、数据预处理流程示例（参考文章⑥⑦）

> 说明：本节演示 `filter → shuffle → map(tokenize)` 这条**核心链路**，属于**概念片段**，用于理解顺序与各步骤作用。它**不是**可直接用于 LoRA/QLoRA 训练的完整数据准备流程——真实训练前通常还需要：`train_test_split()` / `DatasetDict` 划分训练验证集、删除多余列（`remove_columns`）、**构造训练标签**（分类的 `labels`、Causal LM 的 `labels` 及 prompt/padding 掩码为 `-100`）、以及按需要做样本拼接/截断与固定/动态 padding 等。请把本节当作"讲清楚步骤关系"的示意。

假设：

原始数据：

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

---

## 第一步：清洗数据

```python
dataset = dataset.filter(
    lambda x:
    len(x["text"])>5
)
```

---

## 第二步：随机打乱

```python
dataset = dataset.shuffle(
    seed=42
)
```

---

## 第三步：Tokenizer编码

```python
def preprocess(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=128
    )


dataset = dataset.map(
    preprocess
)
```

---

最终：

```
原始文本

↓

filter清洗

↓

shuffle打乱

↓

map Tokenizer

↓

模型训练
```

---

# 十一、Dataset与Trainer训练流程连接（①⑤）

后续使用Trainer训练：

```python
trainer = Trainer(
    model=model,
    train_dataset=dataset
)
```

因此：

Dataset是：

```
数据文件

↓

datasets处理

↓

Trainer

↓

模型训练
```

之间的桥梁。

---

# 十二、常见问题排查

| 问题          | 常见原因（可能不止一种）        | 解决             |
| ----------- | --------- | -------------- |
| map速度慢      | 逐条处理只是常见原因之一；还可能是未开启缓存/缓存失效导致重复计算、磁盘 I/O 瓶颈、tokenizer 本身较慢、未用 `num_proc` 多进程、返回数据结构不利于 Arrow 列式存储等 | 优先 `batched=True`；按需开启缓存、设置 `num_proc`、优化处理函数，必要时用 `streaming` |
| 内存占用高       | 一次加载大量数据只是原因之一；也可能是 map 返回超大列（如长序列/大数组）、`batched` 批过大、`writer_batch_size` 过大、未删除多余列使 Arrow 缓存膨胀等 | 需要时用 `streaming` 流式加载；同时可减小批大小、及时 `remove_columns`、控制返回列大小 |
| 字段冲突        | map返回字段重复 | 检查返回内容         |
| Tokenizer报错 | 文本字段错误    | 确认字段名称         |
| shuffle结果不同 | 没有固定seed  | 设置seed         |

---

# 十三、环境验证实验

创建：

```bash
test_dataset.py
```

代码：

```python
from datasets import Dataset


dataset = Dataset.from_dict(
{
"text":[
"hello",
"machine learning",
"large language model"
]
}
)


dataset = dataset.shuffle(
seed=42
)


dataset = dataset.filter(
lambda x:
len(x["text"])>5
)


print(dataset)
```

运行：

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

说明：

✅ Dataset创建成功
✅ shuffle运行成功
✅ filter运行成功

---

# 十四、总结与后续学习路线

本章完成后，你已经掌握：

✅ Dataset基本结构
✅ datasets库加载方式
✅ map数据转换
✅ batched批量处理
✅ Tokenizer结合map
✅ filter数据过滤
✅ shuffle数据随机化
✅ 大模型训练前数据预处理流程

完整大模型数据流程：

```
原始数据
    ↓
datasets加载
    ↓
filter清洗
    ↓
shuffle随机化
    ↓
map Tokenizer
    ↓
Dataset
    ↓
Trainer训练
    ↓
LoRA微调
```

下一阶段：

**Trainer训练器与TrainingArguments参数配置**

将学习如何把处理好的Dataset真正用于大语言模型微调。
