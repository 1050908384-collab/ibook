# Transformers模型加载基础：掌握 AutoModel / AutoTokenizer 加载方式（最终版）

## 参考资料

① 权威：《Hugging Face 大语言模型课程：Transformers介绍》
　　网址：[Hugging Face LLM Course Chapter 1](https://huggingface.co/learn/llm-course/zh-CN/chapter1/1?utm_source=chatgpt.com)

② 权威：《Hugging Face LLM Course：管道背后的机制》
　　网址：[Hugging Face LLM Course Chapter 2-2](https://huggingface.co/learn/llm-course/zh-CN/chapter2/2?utm_source=chatgpt.com)

③ 权威：《Hugging Face LLM Course：模型与Tokenizer加载》
　　网址：[Hugging Face LLM Course Chapter 2-3](https://huggingface.co/learn/llm-course/zh-CN/chapter2/3?utm_source=chatgpt.com)

④ 权威：《Hugging Face LLM Course：使用预训练模型》
　　网址：[Hugging Face LLM Course Chapter 2-4](https://huggingface.co/learn/llm-course/zh-CN/chapter2/4?utm_source=chatgpt.com)

⑤ 权威：《Hugging Face LLM Course：模型加载与推理》
　　网址：[Hugging Face LLM Course Chapter 2-6](https://huggingface.co/learn/llm-course/zh-CN/chapter2/6?utm_source=chatgpt.com)

⑥ 参考文章：《Transformers中AutoModel与AutoTokenizer详解》
　　网址：[CSDN文章1](https://blog.csdn.net/luckyuxuan/article/details/134024915?utm_source=chatgpt.com)

⑦ 参考文章：《HuggingFace Transformers模型加载详解》
　　网址：[CSDN文章2](https://blog.csdn.net/u013172930/article/details/146151747?utm_source=chatgpt.com)

⑧ 参考文章：《AutoModel与AutoTokenizer实际使用案例》
　　网址：[CSDN文章3](https://blog.csdn.net/u013172930/article/details/146155712?utm_source=chatgpt.com)

---

**适合人群**：已经完成 Hugging Face 环境准备、第一次接触大语言模型加载的学习者。本文会拆解“模型为什么需要 Tokenizer”“AutoModel 如何自动识别模型结构”“from_pretrained()到底做了什么”等核心概念，并通过完整代码演示加载模型、处理输入、执行推理全过程。完成本章后，你将能够独立使用 Transformers 加载 BERT、GPT、Llama、Qwen 等不同类型的大语言模型，为后续学习 Trainer、LoRA 微调和模型部署打下基础。

---

# 一、为什么需要 AutoModel 和 AutoTokenizer？（①②）

## 1.1 Transformers加载模型的基本流程（①）

大语言模型不能直接理解文本。

例如：

```text
你好，大模型
```

模型实际接收的是数字（下面的具体 ID 仅为**示意**，实际数值取决于所用模型的分词器，请以实际运行输出为准）：

```text
[..., ..., ...]   # 每个 token 对应一个整数 ID，不同分词器结果不同
```

因此完整流程：

```
文本
 ↓
Tokenizer分词
 ↓
Token ID数字
 ↓
Transformer模型计算
 ↓
输出结果
 ↓
Tokenizer解码
 ↓
文本
```

其中：

* Tokenizer负责“文字 ↔ 数字”的转换；
* Model负责“数字 → 神经网络计算”。

Hugging Face Transformers提供了统一接口，让不同模型都可以使用类似代码加载。

---

## 1.2 为什么不用具体模型类？

传统方式：

加载BERT：

```python
from transformers import BertModel

model = BertModel.from_pretrained(
    "bert-base-uncased"
)
```

加载GPT：

```python
from transformers import GPT2Model

model = GPT2Model.from_pretrained(
    "gpt2"
)
```

问题：

不同模型需要记忆不同类名：

| 模型      | 对应类          |
| ------- | ------------ |
| BERT    | BertModel    |
| GPT-2   | GPT2Model    |
| RoBERTa | RobertaModel |

如果项目需要切换模型，代码需要大量修改。

因此 Hugging Face 提供：

```
AutoModel
AutoTokenizer
```

自动根据模型配置选择正确类型。

---

# 二、AutoTokenizer：自动加载分词器（参考文章⑥⑧）

## 2.1 Tokenizer是什么？

Tokenizer负责：

1. 文本切分；
2. Token映射编号；
3. 添加特殊Token；
4. 生成模型输入格式。

例如：

输入：

```text
Hello AI
```

经过Tokenizer（下面的 ID 仅为示意，且不同模型的分词器结果不同；例如 GPT-2 与 BERT 对同一句话的切分和编号都不一样）：

```python
[
  ...,
  ...
]
```

模型只能处理这些数字。

---

## 2.2 使用AutoTokenizer加载模型（①⑥）

基础代码：

```python
from transformers import AutoTokenizer


checkpoint = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    checkpoint
)
```

这里：

```python
checkpoint = "bert-base-uncased"
```

表示模型名称。

Transformers会自动：

1. 访问 Hugging Face Hub；
2. 下载 tokenizer 配置；
3. 加载词表文件；
4. 创建对应Tokenizer对象。

---

## 2.3 使用Tokenizer处理文本（②）

示例：

```python
text = "Hello, Hugging Face"


inputs = tokenizer(text)


print(inputs)
```

输出类似（以 `bert-base-uncased` 为例；`101`/`102` 分别是 BERT 的 `[CLS]`/`[SEP]` 特殊 token，中间的 ID 为该分词器的编号，具体数值请以实际运行为准）：

```python
{
 'input_ids': [
    101,
    7592,
    1010,
    17662,
    2227,
    102
 ],
 'attention_mask': [
    1,
    1,
    1,
    1,
    1,
    1
 ]
}
```

解释：

| 字段             | 作用         |
| -------------- | ---------- |
| input_ids      | 文本转换后的数字   |
| attention_mask | 告诉模型哪些位置有效 |

---

# 三、AutoModel：自动加载模型（参考文章⑦）

## 3.1 AutoModel是什么？

`AutoModel` 是 Transformers 提供的自动模型加载接口。

作用：

根据：

```
模型名称
+
配置文件
```

自动选择对应模型结构。

例如：

```python
from transformers import AutoModel


model = AutoModel.from_pretrained(
    "bert-base-uncased"
)
```

内部自动识别：

```
config.json

model_type = bert

↓

创建 BertModel
```

Auto Classes会根据模型名称或路径推断对应架构。

---

## 电脑配置批注

本节包含本地模型下载与推理测试。

最低配置：

* CPU：

  * Intel i5 / AMD同级别以上
* 内存：

  * 8GB
* 磁盘：

  * 至少20GB空闲空间

推荐配置：

* 内存：

  * 16GB+
* GPU：

  * NVIDIA GPU
  * CUDA支持
* 显存：

  * 小模型测试：4GB+
  * 7B模型加载：12GB+

说明：

* AutoModel加载API本身CPU即可运行；
* 大模型推理速度主要取决于GPU显存。

---

# 四、AutoModel + AutoTokenizer完整使用流程（参考文章⑥⑧）

## 4.1 加载模型

完整代码：

```python
from transformers import (
    AutoTokenizer,
    AutoModel
)


checkpoint = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    checkpoint
)


model = AutoModel.from_pretrained(
    checkpoint
)
```

此时：

```
tokenizer

负责文本处理


model

负责神经网络计算
```

---

## 4.2 输入文本

```python
text = "Hugging Face is amazing"


inputs = tokenizer(
    text,
    return_tensors="pt"
)


print(inputs)
```

参数：

```python
return_tensors="pt"
```

表示：

返回 PyTorch Tensor。

输出：

```python
{
input_ids: tensor(...),
attention_mask: tensor(...)
}
```

---

## 4.3 输入模型

```python
outputs = model(**inputs)
```

这里：

```text
**
```

表示：

把字典拆开传入（Python 的字典解包语法）。

等价：

```python
model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"]
)
```

---

## 4.4 查看模型输出

```python
print(outputs)
```

输出：

```text
BaseModelOutput
```

包含：

```
last_hidden_state
pooler_output
```

---

# 五、AutoModelFor系列：实际大模型更常用（参考文章⑦）

## 5.1 为什么不用AutoModel？

`AutoModel`返回：

```
隐藏状态(hidden states)
```

适合：

* 特征提取；
* 下游任务开发。

但是聊天模型通常需要：

```
生成文本
```

因此使用：

```
AutoModelForCausalLM
```

---

## 5.2 加载GPT类模型

示例：

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


checkpoint = "gpt2"


tokenizer = AutoTokenizer.from_pretrained(
    checkpoint
)


model = AutoModelForCausalLM.from_pretrained(
    checkpoint
)
```

区别：

| 类                                  | 用途            |
| ---------------------------------- | ------------- |
| AutoModel                          | 基础Transformer |
| AutoModelForSequenceClassification | 分类            |
| AutoModelForCausalLM               | 文本生成          |
| AutoModelForSeq2SeqLM              | 翻译/摘要         |

> **何时用 `AutoModel`、何时用带任务头的 Auto 类？**
> - `AutoModel`：只加载主干、输出隐藏状态（`last_hidden_state` 等），适合做**特征提取**或自定义下游结构。它**不包含任务头**，因此保存后再用 `AutoModel` 加载不会带回分类/生成头。
> - 带任务头的 `AutoModelFor*`：在主干上附加对应任务头（分类头、语言模型头等）。做分类、生成、摘要等任务，以及**保存/重新加载后仍需要该任务头**时，必须使用与训练时一致的任务类，否则任务头语义会丢失。
>
> **关于本章示例范围**：正文主要以 BERT / GPT-2 演示"加载—编码—前向"的通用流程；对于 Llama、Qwen 等 **decoder-only 对话模型**，实际推理还涉及 `AutoModelForCausalLM`、chat template（`tokenizer.apply_chat_template`）、`pad_token` 设置、gated/授权模型的许可，以及 `trust_remote_code=True` 会执行仓库自带代码（需自行确认来源可信）等。这些属于后续章节内容，本章不展开，此处仅作提示，避免读者以为"会加载 BERT 就等同于会正确对话推理任意 LLM"。

---

# 六、模型加载参数详解（参考文章⑦）

## 6.1 device_map

指定设备：

```python
model = AutoModel.from_pretrained(
    checkpoint,
    device_map="auto"
)
```

作用：

在**加载模型时**自动把不同层映射到可用设备（如多张 GPU、CPU，必要时借助内存/磁盘 offload）：

```
GPU
CPU
```

> 说明：`device_map="auto"` 解决的是"**推理/加载时**模型如何放置到设备"的问题，主要面向单卡放不下、需要跨设备切分的场景。它**不等同于训练时的标准设备/并行配置**——真正的训练（尤其分布式/多卡训练）通常由 `accelerate`、`Trainer`、DeepSpeed 等负责设备与并行策略，不要把 `device_map="auto"` 直接当作训练设备方案。

---

## 6.2 torch_dtype

指定精度：

```python
model = AutoModel.from_pretrained(
    checkpoint,
    torch_dtype="auto"
)
```

作用：

按**模型 checkpoint 配置里记录的 dtype**（`config.torch_dtype`）来加载权重，可能是：

```
float32
float16
bfloat16
```

而不是"按显卡型号自动挑选精度"。当 checkpoint 保存为半精度（fp16/bf16）时，用 `"auto"` 加载相比默认的 float32 能降低显存占用；如需强制指定，可直接传 `torch.bfloat16` / `torch.float16`。

---

## 6.3 local_files_only

只读取本地缓存：

```python
model = AutoModel.from_pretrained(
    checkpoint,
    local_files_only=True
)
```

适合：

* 离线服务器；
* 无网络环境。

---

# 七、从本地路径加载模型（参考文章⑧）

除了：

```python
"bert-base-uncased"
```

也可以：

```python
"./my_model"
```

例如：

目录：

```
my_model
├── config.json
├── model.safetensors
├── tokenizer.json
```

> 说明：上面是**小模型**的典型示例。较大的模型权重常被**分片保存**，目录里会是多个分片文件加一个索引，例如：
>
> ```
> my_model
> ├── config.json
> ├── model-00001-of-00003.safetensors
> ├── model-00002-of-00003.safetensors
> ├── model-00003-of-00003.safetensors
> ├── model.safetensors.index.json   # 分片索引，from_pretrained 会据此拼装
> ├── tokenizer.json
> └── tokenizer_config.json
> ```
>
> 两种情况都用同样的 `from_pretrained("./my_model")` 加载即可，无需手动合并分片。

加载：

```python
tokenizer = AutoTokenizer.from_pretrained(
    "./my_model"
)


model = AutoModel.from_pretrained(
    "./my_model"
)
```

---

# 八、常见错误排查

| 错误                   | 原因              | 解决                |
| -------------------- | --------------- | ----------------- |
| Model name not found | 模型名称错误          | 检查Hub名称           |
| 401 Unauthorized     | 模型需要权限          | 先 `hf auth login`（旧写法 `huggingface-cli login`），并确认已在模型页接受授权 |
| CUDA out of memory   | 显存不足            | 降低batch或使用量化      |
| Tokenizer mismatch   | 模型和Tokenizer不匹配 | 使用同checkpoint     |
| 下载速度慢                | 网络问题            | 配置缓存或镜像           |

---

# 九、环境验证实验

创建：

```
test_model.py
```

代码：

```python
from transformers import AutoTokenizer, AutoModel


checkpoint = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    checkpoint
)

model = AutoModel.from_pretrained(
    checkpoint
)


text = "Hello Transformers"


inputs = tokenizer(
    text,
    return_tensors="pt"
)


outputs = model(**inputs)


print(outputs.last_hidden_state.shape)
```

运行：

```bash
python test_model.py
```

成功输出：

类似（形状为 `[batch, seq_len, hidden_size]`；`bert-base-uncased` 的 `hidden_size=768`，`seq_len` 取决于该句被切分成多少个 token，包含 `[CLS]`/`[SEP]`，实际以运行结果为准）：

```
torch.Size([1, seq_len, 768])
```

说明：

✅ Tokenizer加载成功
✅ Model加载成功
✅ 文本转换成功
✅ Transformer前向计算成功

---

# 十、总结与后续学习路线

本章完成后，你已经掌握：

✅ AutoTokenizer加载方式
✅ Tokenizer文本编码流程
✅ AutoModel自动加载机制
✅ from_pretrained()使用方式
✅ 模型输入输出结构
✅ AutoModelForCausalLM生成模型加载方式
✅ 本地模型加载方法

后续学习：

```
AutoModel
    ↓
datasets数据处理
    ↓
Trainer训练器
    ↓
LoRA微调
    ↓
QLoRA量化训练
    ↓
DeepSpeed
    ↓
vLLM部署
```

掌握 AutoModel / AutoTokenizer，是进入 Hugging Face 大模型训练和微调流程的核心基础。
