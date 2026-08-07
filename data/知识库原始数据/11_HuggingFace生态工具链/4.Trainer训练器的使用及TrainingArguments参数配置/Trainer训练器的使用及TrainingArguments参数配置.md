# Hugging Face Trainer训练器详解：掌握Trainer使用方法与TrainingArguments参数配置（最终版）

## 参考资料

① 权威：《Hugging Face 大语言模型课程：微调预训练模型》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter3/2

② 权威：《Hugging Face 大语言模型课程：训练过程与Trainer》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter3/3

③ 权威：《Hugging Face 大语言模型课程：Trainer高级配置》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter3/4

④ 权威：《Hugging Face 大语言模型课程：大模型训练实践》
　　网址：https://huggingface.co/learn/llm-course/zh-CN/chapter10/1

⑤ 参考文章：《Hugging Face Trainer详细使用教程》
　　网址：https://blog.csdn.net/u013172930/article/details/146138218

⑥ 参考文章：《TrainingArguments参数详解》
　　网址：https://blog.csdn.net/weixin_38252409/article/details/139511319

---

**适合人群**：已经掌握 Transformers 模型加载方式和 datasets 数据处理流程，希望进入大模型训练阶段的学习者。本文将详细拆解 Hugging Face Trainer 的工作流程，以及 TrainingArguments 中常用训练参数的含义。完成本章后，你将能够使用 Trainer 将模型、数据集、训练策略组合起来，完成一个完整的大模型微调训练流程，为后续 LoRA、QLoRA、DeepSpeed 等高级训练技术打下基础。

---

# 一、为什么需要Trainer训练器？（①②）

## 1.1 大模型训练流程回顾

前面章节已经学习：

```text
文本数据

↓

datasets处理

↓

Tokenizer编码

↓

Dataset

↓

模型
```

但是模型训练还需要：

* 前向传播；
* 计算Loss；
* 反向传播；
* 更新参数；
* 保存checkpoint；
* 评估模型。

传统PyTorch需要手写：

```python
for epoch in range(num_epochs):

    for batch in dataloader:

        outputs = model(batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
```

对于大模型：

还需要处理：

* GPU设备；
* 混合精度；
* 梯度累积；
* 分布式训练；
* 模型保存。

因此 Hugging Face 提供：

```python
Trainer
```

帮助自动完成训练流程。

---

# 二、Trainer是什么？（①②）

## 2.1 Trainer核心作用

Trainer 是 Transformers 提供的高级训练封装。

它负责：

| 功能    | 作用              |
| ----- | --------------- |
| 训练循环  | 自动执行epoch和batch |
| 优化器   | 自动创建optimizer   |
| 学习率调度 | 自动管理scheduler   |
| 保存模型  | 保存checkpoint    |
| 评估    | 计算指标            |
| 日志记录  | 输出训练状态          |

简单理解：

```text
自己写训练代码：

模型
+
DataLoader
+
Optimizer
+
训练循环
+
保存逻辑


Trainer：

模型
+
数据集
+
参数配置

↓

自动训练
```

---

# 三、Trainer训练需要哪些组件？（参考文章⑤）

完整Trainer需要：

```python
Trainer(
    model,
    args,
    train_dataset,
    eval_dataset,
    tokenizer
)
```

对应关系：

| 参数            | 作用      |
| ------------- | ------- |
| model         | 需要训练的模型 |
| args          | 训练参数    |
| train_dataset | 训练数据    |
| eval_dataset  | 验证数据    |
| tokenizer     | 文本处理器   |

---

# 四、环境准备

## 4.1 安装依赖

安装：

```bash
pip install transformers datasets accelerate
```

检查：

```python
import transformers
import datasets
```

无报错即可。

---

## 电脑配置批注

本章节涉及真实模型训练。

最低配置：

* CPU：

  * 4核以上
* 内存：

  * 16GB
* 磁盘：

  * 50GB以上

推荐配置：

* CPU：

  * 8核以上
* 内存：

  * 32GB+
* GPU：

  * NVIDIA CUDA GPU
* 显存：

  * 小模型：

    * 8GB+
  * 7B模型微调：

    * 16GB+

说明：

* Trainer本身支持CPU运行；
* 但大模型训练CPU速度非常慢；
* 实际LLM训练建议使用GPU。

---

# 五、准备训练模型和数据（参考文章⑤）

## 5.1 加载模型

示例：

```python
from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

这里：

```python
num_labels=2
```

表示二分类任务。

---

## 5.2 准备Dataset

例如：

```python
from datasets import Dataset


dataset = Dataset.from_dict(
{
"text":[
"I love this",
"I hate this"
],

"label":[
1,
0
]
}
)
```

---

## 5.3 Tokenizer处理

```python
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


def preprocess(example):

    return tokenizer(
        example["text"],
        truncation=True
    )


dataset = dataset.map(
    preprocess
)
```

> 说明：这里 `preprocess` 只做了 `truncation=True`，**没有做固定 padding**，因此不同样本长度不一。用 `Trainer` 组 batch 时，需要一个**动态填充的数据整理器**（`DataCollatorWithPadding`，见第十一章）在训练阶段按 batch 补齐，否则长度不一的样本拼成一个 batch 时可能报错。
>
> 另外，下面第八章、第十二章的示例会用到 `train_dataset` 和 `eval_dataset`。它们需要你从上面的 `dataset` 划分得到，例如：
>
> ```python
> split = dataset.train_test_split(test_size=0.2, seed=42)
> train_dataset = split["train"]
> eval_dataset = split["test"]
> ```
>
> 本章示例数据量极小（仅两条），仅用于跑通流程，真实训练需要足够规模的数据。

---

# 六、TrainingArguments详解（参考文章⑥）

## 6.1 TrainingArguments是什么？

TrainingArguments：

> 用于集中管理训练配置。

例如：

* 学习率；
* batch大小；
* epoch数量；
* 保存策略；
* 日志策略。

创建：

```python
from transformers import TrainingArguments


training_args = TrainingArguments(
    output_dir="./result"
)
```

---

# 七、常用TrainingArguments参数（参考文章⑥）

## 7.1 output_dir

保存训练结果目录：

```python
output_dir="./checkpoint"
```

保存：

```text
checkpoint/
├── checkpoint-500/
├── checkpoint-1000/
└── ...
```

> 说明：上面只是**示意**。`output_dir` 下实际生成的文件与目录结构会随保存方式（`save_model`/`save_steps`/`save_strategy`）、Transformers 版本与训练状态而变化，常见会看到若干 `checkpoint-<步数>/` 子目录，每个里面包含权重（如 `model.safetensors`）、`config.json`、优化器/调度器状态、`trainer_state.json` 等。不要把它记成固定的三个文件，具体以实际运行结果为准。

---

## 7.2 num_train_epochs

训练轮数：

```python
num_train_epochs=3
```

含义：

完整遍历训练数据3次。

---

## 7.3 per_device_train_batch_size

单GPU batch大小：

```python
per_device_train_batch_size=8
```

例如：

GPU数量：

```text
1张GPU
```

batch：

```text
8
```

每次输入8条数据。

---

## 7.4 learning_rate

学习率：

```python
learning_rate=2e-5
```

控制：

模型参数更新速度。

常见（下面仅为**常见起始范围**，不是固定规则）：

| 任务     | 学习率（经验起点）       |
| ------ | --------- |
| 全参数微调  | 约 1e-5~5e-5 |
| LoRA微调 | 约 1e-4 量级（常比全参更大） |

> 说明：合适的学习率与模型规模、数据量、任务类型、有效 batch size（`per_device_train_batch_size` × 梯度累积 × 设备数）、优化器与调度策略都相关，差异可能很大。上表只作为**起始参考**，实际应结合验证集表现（loss/指标是否稳定下降、是否震荡或发散）来调整。
>
> **学习率与有效 batch size / warmup / scheduler 的关系（需一起调，而不是孤立看一个数）**：
> - **有效 batch size**：真正影响梯度噪声的是「有效 batch size = `per_device_train_batch_size` × `gradient_accumulation_steps` × 设备数」。有效 batch 增大时，常需要**相应上调学习率**（一种常见经验起点是按比例或按平方根缩放），但这只是**起点**，仍要看验证集是否稳定。
> - **warmup**：训练初期直接用目标学习率容易震荡/发散，常用 `warmup_ratio`（如 0.03~0.1）或 `warmup_steps` 让学习率从 0 线性升到目标值再进入衰减。学习率越大，warmup 往往越需要。
> - **scheduler（学习率调度）**：`lr_scheduler_type` 常用 `linear` / `cosine` 等，决定 warmup 之后如何衰减。LoRA/QLoRA 微调常见组合是「较大的峰值学习率（1e-4 量级）+ 一定比例 warmup + cosine 衰减」，但同样**因模型/数据而异**，请以验证集表现为准。
> - **LoRA 学习率为何常更大**：LoRA 只训练注入的低秩增量参数（`A`/`B`），可训练参数量小、初始扰动小，因此常能承受比全参微调更大的学习率；但「更大」不等于「越大越好」，仍需结合上面几项一起在验证集上验证。

---

## 7.5 logging_steps

日志输出频率：

```python
logging_steps=10
```

每10步打印一次。

---

## 7.6 save_steps

保存checkpoint频率：

```python
save_steps=500
```

每500步保存。

---

## 7.7 evaluation_strategy

评估策略：

```python
evaluation_strategy="steps"
```

常见：

| 值     | 含义       |
| ----- | -------- |
| no    | 不评估      |
| steps | 固定step评估 |
| epoch | 每轮评估     |

> 版本提示：该参数名在较新版本的 Transformers 中已改为 `eval_strategy`，`evaluation_strategy` 属于逐步弃用的旧写法。请以你所安装版本的官方文档为准；若运行报参数名相关的警告或错误，改用 `eval_strategy` 即可。
>
> 相关迁移：构造 `Trainer` 时用来传分词器/处理器的参数，在较新版本中也由 `tokenizer=` 迁移为 `processing_class=`（`tokenizer=` 仍可用但会告警并逐步弃用）。本章后续示例沿用了常见教程里的 `tokenizer=` 写法，如你的版本提示弃用，替换为 `processing_class=tokenizer` 即可。

---

# 八、创建Trainer（重点）（参考文章⑤）

> 说明：下面这段是**结构片段**，用来说明 `Trainer(...)` 需要哪些组件，它**依赖前文的变量**（`model`、`training_args`、`train_dataset`、`eval_dataset`、`tokenizer`）。要真正跑通，请以第十二章的**完整最小骨架**为准。

结构片段：

```python
from transformers import Trainer


trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer   # 较新版本写法；旧版本用 tokenizer=tokenizer（见 7.7 迁移说明）
)
```

Trainer内部建立：

```text
模型

↓

训练参数

↓

Dataset

↓

训练循环

↓

优化器

↓

保存模型
```

---

# 九、启动训练（参考文章⑤）

执行：

```python
trainer.train()
```

训练过程：

输出（下面的 loss 数值仅为**示意**，实际取决于模型、数据与超参，请以真实训练日志为准）：

```text
Epoch 1/3

loss:0.45

Epoch 2/3

loss:0.21
```

---

# 十、保存训练结果（参考文章⑤）

训练完成：

```python
trainer.save_model(
    "./my_model"
)
```

保存：

```text
my_model

├── config.json

├── model.safetensors

├── tokenizer.json
```

之后（重新加载时应使用**与训练时一致的任务类**，本例训练用的是 `AutoModelForSequenceClassification`，重载也用它，才能带回分类头；若改用 `AutoModel` 只会得到不含分类头的主干）：

```python
from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained(
    "./my_model"
)
```

即可加载。

---

# 十一、DataCollator数据整理器（参考文章⑤）

## 11.1 为什么需要DataCollator？

不同文本长度：

例如：

```text
Hello

Hello World

Large Language Model
```

Token数量不同。

需要：

```text
padding
```

统一长度。

---

## 11.2 使用 DataCollatorWithPadding

`DataCollatorWithPadding` 是一个**具体的数据整理器**（并不是 Trainer 的"默认整理器"）：它在组 batch 时，按该 batch 内最长样本对 `input_ids`/`attention_mask` 等做**动态填充**，配合前面 `preprocess` 里只做 `truncation` 而不固定 padding 的做法。

```python
from transformers import DataCollatorWithPadding


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

传入Trainer（作为 `data_collator` 参数，与 `model`/`args`/`train_dataset` 等一起传入，此处仅示意该参数位置）：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)
```

---

# 十二、完整Trainer训练案例（参考文章⑤⑥）

> 说明：下面是把前面各步骤串起来的**结构示例**。要真正跑通，还需在此之前完成：准备并 `map` 编码数据、划分 `train_dataset`/`eval_dataset`（见 5.3 的 `train_test_split`）、以及创建 `DataCollatorWithPadding`（见 11.2）。这里把这些依赖一并写出，形成一个更接近可运行的最小骨架：
>
> **本案例的定位（重要）**：这是一个 **BERT 文本分类**案例，目的只是把「模型 + 数据 + `TrainingArguments` + `Trainer`」这条 **Trainer API 主流程**跑通，**不是** Causal LM / 大模型 / LoRA 微调的范例。真正的 LLM 指令微调在数据格式（对话拼接、`labels` 及 prompt/padding 掩码为 `-100`）、模型类（`AutoModelForCausalLM`）、以及是否叠加 PEFT/LoRA 上都不同，请以后续第 12 板块（LoRA/QLoRA）为准。
>
> 另外，本例只有**两条样本**、`test_size=0.5`，仅用于**演示流程能不能走通**，不能反映真实训练/评估效果；真实训练需要足够规模的数据与合理的划分比例（如 8:2 / 9:1）。

代码结构：

```python
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


model_name="bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


# 1) 准备数据（真实场景请替换为足够规模的数据）
dataset = Dataset.from_dict(
    {
        "text": ["I love this", "I hate this"],
        "label": [1, 0]
    }
)


def preprocess(example):
    return tokenizer(example["text"], truncation=True)


dataset = dataset.map(preprocess)


# 2) 划分训练集/验证集
#    注意：这里只有两条样本、test_size=0.5，仅为演示流程；真实数据请用更大规模、更常见的划分比例（如 8:2）
split = dataset.train_test_split(test_size=0.5, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


# 3) 动态填充整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 4) 评估指标：传了 eval_dataset 不代表会自动评估，
#    还需要 eval_strategy 触发评估、compute_metrics 计算指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    # 简单二分类 F1（macro 可按需替换为 sklearn.metrics.f1_score）
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": float(accuracy), "f1": float(f1)}


training_args = TrainingArguments(
    output_dir="./result",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    learning_rate=2e-5,

    logging_steps=10,

    # 传入 eval_dataset 只是提供了验证数据，
    # 是否在训练过程中自动评估由 eval_strategy 决定（旧版本参数名为 evaluation_strategy，见 7.7）
    eval_strategy="epoch",

    # 保存策略：两条样本时 save_steps=500 永远触发不到，改用按 epoch 保存更直观
    save_strategy="epoch"
)


trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    # 较新版本写法；旧版本用 tokenizer=tokenizer（见 7.7 迁移说明）
    processing_class=tokenizer
)


trainer.train()
```

> 补充说明：
> - **`eval_strategy`**：只有设置了 `eval_strategy="epoch"`（或 `"steps"` 并配合 `eval_steps`），Trainer 才会在训练过程中调用评估；**仅传 `eval_dataset` 不会自动触发评估**。旧版本参数名是 `evaluation_strategy`（见 7.7）。
> - **`compute_metrics`**：不传 `compute_metrics` 时，评估阶段只会给出 eval loss；要看 accuracy / F1 等指标必须自己提供该函数（上例给出了最小实现）。
> - **`save_strategy` vs `save_steps`**：`save_steps=500` 表示每 500 步保存一次；本例只有两条样本，一个 epoch 只走极少的 step，`save_steps=500` 实际永远触发不到，因此改用 `save_strategy="epoch"`。真实训练可按数据规模在两者间选择。
> - **`processing_class` vs `tokenizer`**：较新版本用 `processing_class=`，旧版本用 `tokenizer=`（后者仍可用但会逐步弃用，见 7.7）。

---

# 十三、常见问题排查

| 问题                 | 原因          | 解决              |
| ------------------ | ----------- | --------------- |
| CUDA out of memory | batch太大     | 降低batch         |
| 训练速度慢              | 没有GPU       | 检查CUDA          |
| loss不下降            | 学习率问题       | 调整learning_rate |
| checkpoint太多       | 保存频率过高      | 增加save_steps    |
| 数据格式错误             | Dataset字段问题 | 检查column_names  |

---

# 十四、Trainer与后续LoRA微调关系

实际大模型微调：

```text
Dataset

↓

Trainer

↓

PEFT LoRA

↓

训练Adapter参数

↓

保存LoRA权重
```

Trainer并不局限于全参数训练。

后续：

```text
Trainer
+
LoraConfig
+
PEFT
```

即可完成参数高效微调。

> 衔接说明：第十二章的完整案例是 **BERT 文本分类**，只覆盖 Trainer API 的通用主流程；**Causal LM + PEFT/LoRA** 的完整可运行案例（`AutoModelForCausalLM` + `LoraConfig` + `get_peft_model` + 对话数据的 `labels`/`-100` 掩码）请见第 12 板块（LoRA 与 QLoRA）。届时 `TrainingArguments`/`Trainer` 的用法与本章一致，主要差异在**模型类、数据标签构造与 PEFT 注入**。

---

# 十五、总结与后续学习路线

本章完成后，你已经掌握：

✅ Trainer作用
✅ Trainer核心组件
✅ TrainingArguments配置方式
✅ batch、epoch、learning rate设置
✅ checkpoint保存
✅ Dataset接入Trainer
✅ 模型训练启动流程

完整Hugging Face训练流程：

```text
AutoTokenizer

↓

datasets处理数据

↓

Dataset

↓

TrainingArguments

↓

Trainer

↓

模型训练

↓

LoRA微调

↓

部署推理
```

下一阶段：

**参数高效微调（LoRA与QLoRA）**

将学习如何利用Trainer训练大语言模型，同时大幅降低GPU显存需求。
