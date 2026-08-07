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
├── model
├── tokenizer
└── trainer_state.json
```

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

常见：

| 任务     | 学习率       |
| ------ | --------- |
| 全参数微调  | 1e-5~5e-5 |
| LoRA微调 | 1e-4左右    |

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

---

# 八、创建Trainer（重点）（参考文章⑤）

完整代码：

```python
from transformers import Trainer


trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    tokenizer=tokenizer
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

之后：

```python
model = AutoModel.from_pretrained(
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

## 11.2 使用默认DataCollator

```python
from transformers import DataCollatorWithPadding


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

传入Trainer：

```python
trainer = Trainer(
    data_collator=data_collator
)
```

---

# 十二、完整Trainer训练案例（参考文章⑤⑥）

代码结构：

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)


model_name="bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)


training_args = TrainingArguments(
    output_dir="./result",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    learning_rate=2e-5,

    logging_steps=10,

    save_steps=500
)


trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    tokenizer=tokenizer
)


trainer.train()
```

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

```python
Trainer
+
LoraConfig
+
PEFT
```

即可完成参数高效微调。

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
