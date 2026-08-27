# 学习包

## 块 1

## 一、为什么需要Trainer训练器？

### 1.1 大模型训练流程回顾

你已经掌握了前面的流程：**文本数据 → datasets处理 → Tokenizer编码 → Dataset → 模型**。

但模型训练本身还需要完成这些步骤：

- 前向传播
- 计算Loss
- 反向传播
- 更新参数
- 保存checkpoint
- 评估模型

如果用传统PyTorch，你得手写训练循环：

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**问题来了**：这只是一个基础框架。到了大模型场景，你还要额外处理：

- GPU设备管理
- 混合精度
- 梯度累积
- 分布式训练
- 模型保存

这些如果全部手写，代码量会急剧膨胀，而且极易出错。

### 1.2 Trainer的定位

Hugging Face 提供的 `Trainer` 就是来解决这个痛点的——它把这些训练流程**自动封装**起来，你只需要告诉它"用什么模型、什么数据、什么训练策略"，剩下的训练循环、设备调度、checkpoint保存等脏活累活，它帮你完成。

---

## 二、Trainer是什么？

简单说，`Trainer` 是 Hugging Face 提供的一个**高层训练接口**，它把训练流程标准化、自动化。你不需要再手写 `for epoch` 循环，而是通过配置参数来声明训练方式。

结合你的目标（**DeepSpeed分布式训练**），这里先建立一个关键认知：`Trainer` 是后续学习 DeepSpeed 的**前置基础**——DeepSpeed 的分布式策略正是通过 `Trainer` 的配置参数来接入的。所以你现在掌握 `Trainer` 的用法，就是在为分布式训练铺路。

---

## 本块学习检查

**动手验证**：打开你之前写过的任意一个 PyTorch 训练脚本，对照上面的"传统手写循环"代码，逐行指出：

1. 哪些代码对应"前向传播"？
2. 哪些代码对应"反向传播"？
3. 哪些代码对应"更新参数"？

然后思考：如果改用 `Trainer`，这三步分别会被哪个配置项或内部机制替代？（提示：`Trainer` 内部自动完成这些步骤，你只需关注数据、模型和 `TrainingArguments` 配置。）

## 块 2

### 2.1 Trainer核心作用

Trainer是Transformers提供的高级训练封装，帮你把训练流程自动化。对照你写代码的习惯，它把下面这些活儿全包了：

| 功能 | 作用 |
| --- | --- |
| 训练循环 | 自动执行epoch和batch |
| 优化器 | 自动创建optimizer |
| 学习率调度 | 自动管理scheduler |
| 保存模型 | 保存checkpoint |
| 评估 | 计算指标 |
| 日志记录 | 输出训练状态 |

**直观对比：**

自己写训练代码，需要手动拼装：模型 + DataLoader + Optimizer + 训练循环 + 保存逻辑。

用Trainer，你只需提供：模型 + 数据集 + 参数配置，然后自动训练。

---

### 三、Trainer训练需要哪些组件？

完整Trainer需要传入以下参数：

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

| 参数 | 作用 |
| --- | --- |
| model | 需要训练的

## 块 3

**安装依赖（动手版）**

你的目标是DeepSpeed分布式训练，但先别急着碰它——基础库装好，后面才能跑起来。直接执行：

```bash
pip install transformers datasets accelerate
```

装完别停，立刻验证：

```python
try:
    import transformers
    import datasets
except ImportError as e:
    print('依赖安装失败，请检查pip源或Python版本')
    raise
```

无报错即通过。**这一步是硬门槛**，报错就回头检查pip源或Python版本。

**配置提醒**：本章是真实训练，最低4核CPU+16GB内存+50GB磁盘。你有GPU最好，没有也能跑，但大模型CPU会慢到怀疑人生——建议后续用GPU

## 块 4

### 5.1 加载模型（动手实操）

直接上代码，跟着敲：

```python
from transformers import AutoModelForSequenceClassification

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
except OSError as e:
    print(f"模型加载失败：{e}")
    raise
```

**关键点**：`num_labels=2` 表示二分类任务（比如“好评/差评”）。你之后做DeepSpeed分布式训练时

## 块 5

## 5.3 Tokenizer处理：为DeepSpeed训练铺路

你的目标是DeepSpeed分布式训练，但任何分布式训练都从“把数据喂给模型”开始。这里我们聚焦Tokenizer——它把文本转成模型能读的数字。

**核心操作**：用`AutoTokenizer`加载`bert-base-uncased`，然后定义`preprocess`函数对每条样本做`truncation=True`（截断超长文本），最后用`dataset.map(preprocess)`批量处理。

**关键点**：这里**故意不做固定padding**，因为不同样本长度不同。等用`Trainer`组batch时，需要一个`DataCollatorWithPadding`（第十一章）在训练时按batch动态补齐——否则长度不一的样本拼在一起会报错。这个设计对DeepSpeed尤其重要：动态padding能减少无效计算，提升显存利用率。

**数据划分**：后续章节需要`train_dataset`和`eval_dataset`，从`dataset`划分得到：
```python
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
```

> ⚠️ 注意：当前示例数据量极小（仅两条），只用于跑通流程。真实训练（尤其DeepSpeed多卡）需要足够规模的数据。

---

**本块学习检查**：请手动执行一次`dataset.train_test_split(test_size=0.2, seed=42)`，确认返回的`train_dataset`和`eval_dataset`样本数量是否符合预期（两条数据时各为多少？）。这能帮助你验证数据划分是否正确。

## 块 6

`output_dir` 指定模型训练结果的保存目录。你的目标是 DeepSpeed 分布式训练，但无论单机还是多卡，这个参数都是训练流程的起点——它决定了 checkpoint 落在哪里。

```python
output_dir="./checkpoint"
```

运行后，该目录下会按保存策略生成若干 `checkpoint-<步数>/` 子目录（如 `checkpoint-500`、`checkpoint-1000`）。每个子目录内通常包含权重文件（如 `model.safetensors`）、`config.json`、优化器/调度器状态和 `trainer_state.json`。

注意：这只是示意结构。实际文件与目录会随 `save_strategy`、Transformers 版本和训练状态变化，不要死记固定模板，以实际运行为准。

**本块学习检查**：运行一次含 `output_dir` 的 Trainer

## 块 7

### 7.2 num_train_epochs：训练轮数

`num_train_epochs` 控制模型**完整遍历训练数据的次数**。设置为 3，意味着模型将把整个数据集从头到尾学习 3 遍。

```python
num_train_epochs=3
```

**怎么理解？** 假设你有 1000 条训练数据，`num_train_epochs=3` 就表示模型会分 3 轮，每轮看完这 1000 条数据，总共看 3000 条次。

**动手试一下**：在 Trainer 配置中分别设置 `num_train_epochs=1` 和 `num_train_epochs=3`，观察训练日志中 `epoch` 字段的变化——前者只出现 `1.0`，后者会递增到 `3.0`。

---

### 7.3 per_device_train_batch_size：单GPU batch大小

这个参数决定**每张 GPU 一次输入多少条数据**。

```python
per_device_train_batch_size=8
```

**举例**：如果你有 1 张 GPU，设置 batch size 为 8，那么模型每步训练会同时处理 8 条数据，然后更新一次参数。

---

### 7.4 learning_rate：学习率

学习率控制**模型参数每次更新的步长**。设置太大，模型可能震荡不收敛；设置太小，训练会非常缓慢。

```python
learning_rate=2e-5
```

**常见起始范围**（仅作参考，非固定规则）：

| 任务     | 学习率（经验起点）       |
| ------ | --------- |
| 全参数微调  | 约 1e-5~5e-5 |
| LoRA微调 | 约 1e-4 量级（常比全参更大） |

> **关键提醒**：学习率不是孤立看的，它和**有效 batch size**、**warmup**、**scheduler** 需要一起调整：
> - **有效 batch size** = `per_device_train_b

## 块 8

## 7.5 logging_steps：控制训练日志的“节拍器”

在DeepSpeed分布式训练中，日志输出频率直接影响你观察训练进程的效率。`logging_steps`参数控制模型每训练多少步打印一次日志：

```python
logging_steps=10
```

上述配置表示**每10步打印一次**训练信息（如loss值）。对于你的分布式训练场景，建议根据训练总步数合理设置——步数多则适当增大该值，避免日志刷屏；步数少则减小该值，确保能及时观察loss变化趋势。

---

## 7.6 save_steps：checkpoint保存频率

训练过程中定期保存模型权重至关重要，尤其是长时间运行的分布式训练任务：

```python
save_steps=500
```

上述配置表示**每500步保存一次checkpoint**。保存的checkpoint包含模型权重、优化器状态等信息，可用于断点续训或后续评估。在DeepSpeed训练中，建议结合你的GPU资源和训练时长，设置一个既不会过于频繁占用I/O、又能保证安全性的保存间隔。

---

## 7.7 evaluation_strategy：评估策略

该参数决定何时在验证集上评估模型性能：

```python
eval_strategy="steps"
```

常见取值如下：

| 值     | 含义           |
| ------ | -------------- |
| `no`   | 不进行评估     |
| `steps`| 每固定步数评估 |
| `epoch`| 每个epoch评估  |

> **版本提示**：在较新版本的Transformers中，该参数已更名为`eval_strategy`，`evaluation_strategy`属于逐步弃用的旧写法。若运行时报参数名相关警告或错误，请改用`eval_strategy`。同理，构造`Trainer`时传分词器的参数`tokenizer=`在新版本中迁移为`processing_class=`（旧写法仍可用但会告警）。请以你安装版本的官方文档为准。

---

## 八、创建Trainer（重点）

`Trainer`是Hugging Face训练流程的核心封装。以下为结构片段，**依赖前文定义的变量**（`model`、`training_args`、`train_dataset`、`eval_dataset`、`tokenizer`）：

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer  # 新版本写法；旧版本用 tokenizer=tokenizer
)
```

`Trainer`内部自动建立完整训练流水线：

```text
模型 → 训练参数 → Dataset → 训练循环 → 优化器 → 保存模型
```

---

## 九、启动训练

执行一行命令即可开始训练：

```python
trainer.train()
```

训练过程会输出类似以下日志（**loss数值仅为示意**，实际取决于模型、数据与超参）：

```text
Epoch 1/3
loss:0.45

Epoch 2/3
loss:0.21
```

---

## 十、保存训练结果

训练完成后保存模型：

```python
trainer.save_model("./my_model")
```

保存的文件结构：

```text
my_model
├── config.json
├── model.safetensors
└── tokenizer.json
```

重新加载时，**必须使用与训练时一致的任务类**。例如训练用的是`AutoModelForSequenceClassification`，重载也用它，才能带回分类头：

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./my_model")
```

若改用`AutoModel`，只会得到不含分类头的主干模型。

---

## 部署说明

部署模型时需注意以下事项：

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
```

## 块 9

## 11.1 为什么需要DataCollator？

在自然语言处理中，不同句子的长度往往不同。比如：

```text
Hello
Hello World
Large Language Model
```

这三句话的Token数量分别是1、2、3，长短不一。而模型训练时，通常要求一个batch内的样本长度一致，才能高效地组成张量进行并行计算。

**DataCollator的核心作用**就是解决这个问题——它负责在组装batch时，把长度不一的样本**统一到相同长度**，让训练流程顺畅进行。

---

## 11.2 使用 DataCollatorWithPadding

`DataCollatorWithPadding` 是一个**具体的数据整理器**。它的工作方式是：在组装batch时，**按该batch内最长样本的长度**，对较短的样本进行**动态填充**（padding），使所有样本对齐。

这里强调"动态"是因为：它只针对当前batch内的最长样本做填充，而不是固定填充到某个预设长度。这比固定长度填充更节省计算资源。

使用方式很简单：

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)
```

然后将它作为参数传入Trainer：

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

**注意**：`DataCollatorWithPadding` 并不是Trainer的默认整理器，需要显式创建并传入。它通常配合数据预处理时只做截断（truncation）、不固定padding的做法一起使用——预处理阶段先不管长度，等到组装batch时再由DataCollator动态补齐。

---

**本块学习检查**：请回答——`DataCollatorWithPadding` 的填充长度是如何决定的？如果batch中有一条长度为10的样本和一条长度为5的样本，填充后两条样本的长度分别是多少？

## 块 10

## 11.1 为什么需要DataCollator？——从“零散样本”到“整齐批次”

你正在为DeepSpeed分布式训练打基础，而DataCollator正是让数据在**多GPU并行**下高效流转的关键一环。先记住一句话：**Trainer不会自动帮你整理数据，DataCollator就是那个“整理员”**。

### 问题场景：你的数据长短不一

看下面这段代码，你准备了两个句子：

```python
dataset = Dataset.from_dict(
    {
        "text": ["I love this", "I hate this movie"],
        "label": [1, 0]
    }
)
```

经过tokenizer处理后，这两个样本的**长度不同**——`"I love this"`包含3个单词，而`"I hate this movie"`包含4个单词，在BERT的tokenizer下，每个词通常对应一个token，因此两个样本的token数量不同。这就导致：

- 样本A长度：3个token
- 样本B长度：4个token

**问题来了**：深度学习框架（PyTorch）要求一个batch内的所有样本**形状一致**，否则无法组成张量矩阵。你不能把`[3, 768]`和`[4, 768]`的两个向量直接堆叠。

### DataCollator的解决方案：动态填充

代码中的这一行就是核心：

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

`DataCollatorWithPadding`做的事情是：**在每个batch内部，找到该batch中最长的样本，然后把其他样本填充到同样长度**。

关键点在于“**动态**”二字——它不是把所有样本都填充到整个数据集的最大长度，而是**每个batch单独计算**。这样做的好处：

1. **节省计算资源**：如果整个数据集最大长度是512，但大多数样本只有20个token，全部填充到512会浪费大量显存
2. **加速训练**：更短的序列意味着更少的计算量
3. **对DeepSpeed尤其重要**：分布式训练中，显存是稀缺资源，动态填充能让你在同样的显存下塞进更大的batch size

### 为什么不能手动填充？

你可能会想：“我提前把所有样本都填充到统一长度不就行了？”技术上可行，但：

- **浪费**：短样本被迫“加长”，计算量虚增
- **不灵活**：如果新数据来了更长样本，你得重新处理整个数据集

DataCollator是在**训练过程中、每个batch生成时**实时处理的，它只关心当前batch内部的对齐。

### 完整流程回顾

在你的代码中，DataCollator处于这个位置：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # ← 就是这里
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)
```

Trainer在每次取batch时，会调用`data_collator`，把原始样本列表整理成模型能吃的张量。**没有它，训练直接报错**——因为形状不匹配。

### 动手验证（2分钟）

把上面的代码完整跑一遍（用两条样本即可），然后做这个实验：

1. 在`trainer.train()`之前，手动调用一次data_collator：
   ```python
   batch = data_collator([train_dataset[i] for i in range(len(train_dataset))])
   print(batch["input_ids"].shape)
   ```
2. 观察输出的形状——两条样本的`input_ids`形状应该完全一致（都是`[2, 序列长度]`）
3. 再打印`batch["attention_mask"]`，看看哪些位置是填充的（mask值为0的位置）

这个实验能让你直观看到“动态填充”的效果：两条样本被整理成了整齐的矩阵。

---

**小结**：DataCollator是Trainer的数据整理员，负责把长度不一的样本动态填充成统一形状的batch。它不改变数据内容，只做形状对齐——这是分布式训练（包括DeepSpeed）中显存效率的关键一环。

## 块 11

## 为什么需要DataCollator？

在训练开始前，你手里的数据是**长度不一**的——有的句子长，有的句子短。但GPU喜欢“整齐划一”的输入，这就要求我们把一批样本**填充（padding）到相同长度**，或者**截断（truncating）**，再打包成张量喂给模型。

**DataCollator就是干这个的**：它负责把Dataset里的一条条样本，整理成一个batch的标准输入格式（input_ids、attention_mask、labels等），确保Trainer拿到的数据形状一致、可直接计算。

**一个关键提醒**：如果未设置`eval_strategy`，Trainer默认不会在训练过程中定期评估，但会在训练结束后评估一次。若需周期性评估，请设置`eval_strategy="epoch"`或`"steps"`。旧版本参数名是`evaluation_strategy`，新版本用`eval_strategy`。

**关于评估指标**：不传`compute_metrics`时，评估阶段只输出eval loss；想看accuracy/F1，必须自己写个函数传进去。

**保存策略**：`save_steps=500`表示每500步存一次checkpoint。但如果你的数据集很小（比如只有两条样本），一个epoch走不了几步，`save_steps=500`永远触发不到——这时改用`save_strategy="epoch"`更合理。真实训练按数据规模选。

**版本差异**：Trainer的`tokenizer`参数用于传递分词器，新版本中部分组件（如DataCollator）可能使用`processing_class`，但Trainer本身仍使用`tokenizer`参数。

---

### 常见问题速查

| 问题 | 原因 | 解决 |
|------|------|------|
| CUDA out of memory | batch太大 | 降低batch |
| 训练速度慢 | 没有GPU | 检查CUDA |
| loss不下降 | 学习率问题 | 调整learning_rate |
| checkpoint太多 | 保存频率过高 | 增加save_steps |
| 数据格式错误 | Dataset字段问题 | 检查column_names |

---

### Trainer与LoRA的关系

Trainer不局限于全参数训练。实际大模型微调流程：

```
Dataset → Trainer → PEFT LoRA → 训练Adapter参数 → 保存LoRA权重
```

也就是：`Trainer + LoraConfig + PEFT` 即可完成参数高效微调。Causal LM + LoRA的完整案例见第12板块，主要差异在**模型类、数据标签构造与PEFT注入**，Trainer本身用法一致。

---

### 本章总结

你已经掌握：Trainer作用、核心组件、TrainingArguments配置、batch/epoch/learning rate设置、checkpoint保存、Dataset接入、训练启动流程。

完整流程：

```
AutoTokenizer → datasets处理数据 → Dataset → TrainingArguments → Trainer → 模型训练 → LoRA微调 → 部署推理
```

**下一阶段**：参数高效微调（LoRA与QLoRA），学习如何利用Trainer训练大语言模型，同时大幅降低GPU显存需求——这正是你DeepSpeed分布式训练目标的前置基础。

---

**本块学习检查**：打开你的训练脚本，确认三件事——①是否设置了`eval_strategy`（而不是只传`eval_dataset`）；②如果数据集很小，`save_strategy`是否改成了`"epoch"`；③`compute_metrics`是否已传入（否则评估只有loss）。逐项核对后，记录你当前脚本的这三处配置

## 测评题目

### Q-KB1103-003
请写出使用 `SFTTrainer` 进行训练的基本代码框架（包含 `model`, `train_dataset`, `args`, `data_collator`）。

### Q-KB1103-005
请写代码：在 `SFTTrainer` 训练完成后，保存微调后的 LoRA 适配器权重到 `./final_lora_adapter` 目录。

### Q-KB1103-004
在 `SFTConfig` 中，`max_seq_length` 参数的作用是？

### Q-KB1103-001
TRL（Transformer Reinforcement Learning）库中，专门用于监督微调（SFT）的Trainer类是？

### Q-KB1101-002
在进行模型推理时，为了防止计算梯度占用显存，通常使用 `with torch.________():` 上下文管理器。

### Q-KB11-ST1102-002
datasets 的 `filter()` 会保留使筛选函数返回________的样本。

### Q-KB11-ST1100-002
使用私有 Hugging Face 仓库时，通常需要配置访问________。

### Q-KB11-ST1102-003
为了在每个训练轮次打乱样本顺序，通常使用？
