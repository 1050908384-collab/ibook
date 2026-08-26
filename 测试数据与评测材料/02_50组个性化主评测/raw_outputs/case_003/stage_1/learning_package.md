# 学习包

## 块 1

## 一、先理解 Hugging Face 生态工具链

Hugging Face 是目前大语言模型开发中最常用的开源生态之一。它主要包含四个核心工具：

| 工具 | 作用 |
| --- | --- |
| Transformers | 加载、训练、推理Transformer模型 |
| Datasets | 加载和处理训练数据 |
| Accelerate | 简化单卡、多卡、混合精度训练 |
| Hugging Face Hub | 模型、数据集和代码共享平台 |

**整体学习流程**：Tokenizer → Transformers加载模型 → Datasets处理数据 → Accelerate优化训练 → Trainer完成微调 → LoRA / QLoRA → 部署推理

---

## 二、环境方案选择

Hugging Face学习环境主要有三种：Google Colab（推荐新手）、Windows本地、Linux本地。考虑到你的实训条件，我们重点介绍**Google Colab方案**，它无需本地GPU，开箱即用，最适合快速上手。

---

## 三、动手搭建环境（Google Colab方案）

### 第1步：安装核心库

在Colab中新建代码单元格，依次执行以下安装命令：

```python
!pip install transformers
!pip install datasets
!pip install accelerate
```

这三个库分别对应：加载模型、处理数据、加速训练。安装完成后，建议重启运行时确保生效。

### 第2步：配置Hugging Face Hub

**① 获取Token**：登录Hugging Face官网，进入Settings → Access Tokens，点击New token创建，复制生成的token字符串。

**② 登录Hub**：在Colab中运行：

```python
from huggingface_hub import notebook_login
notebook_login()
```

在弹出的窗口中粘贴你的token，点击Login即可。

**③ 设置缓存路径**（可选但推荐）：默认模型会下载到用户目录，可自定义路径避免占用系统盘：

```python
import os
os.environ['HF_HOME'] = '/content/hf_cache'
```

---

## 四、验证环境是否就绪

运行以下代码，确认三个库都能正常导入：

```python
import transformers, datasets, accelerate
print("Transformers版本:", transformers.__version__)
print("Datasets版本:", datasets.__version__)
print("Accelerate版本:", accelerate.__version__)
```

如果能看到版本号输出，说明环境搭建成功。

---

**本块学习检查**

先用一个最小示例回顾本节内容：在Colab中依次执行安装命令、登录Hub、导入三个库并打印版本号。确认你能解释以下关键写法后再继续本块：

- `!pip install` 的作用是什么？
- `notebook_login()` 为什么需要token？
- 三个库各自负责什么任务？

完成上述操作并理解后，即可进入下一部分学习。

## 块 2

**本块学习检查**：读完本块后，请你在Colab中新建一个笔记本，运行下面两段代码，确认输出无报错。

---

### 2.1 什么是Google Colab？

Colab是Google提供的云端Jupyter Notebook环境。你不需要在本地配置Python，打开浏览器就能直接运行Python代码，还可以申请GPU资源。对于第一次学Transformers、想快速验证模型代码的你来说，这是最省事的起点——尤其适合没有NVIDIA显卡的初学者。

### 2.2 Colab安装环境

打开Colab后，在代码单元格里依次运行：

```python
!pip install "transformers[sentencepiece]"
!pip install datasets accelerate huggingface_hub
```

然后验证安装是否成功：

```python
import transformers
import datasets
import accelerate

print(transformers.__version__)
```

看到版本号输出，说明环境就绪。接下来就能开始你的LoRA微调之旅了。

> **提示**：如果安装或导入时报错，请先检查网络连接是否正常，并确认依赖包之间没有版本冲突。

## 块 3

**Windows兼容性提醒（面向LoRA微调）**

你在Windows上做LoRA微调，先记住三点：

**1. 路径格式别搞混**
- Windows用反斜杠：`D:\huggingface_cache`
- Linux用正斜杠：`/data/huggingface_cache`
- 设置环境变量时，两种系统格式不同，别直接复制。

**2. 用GPU前先查CUDA**
- 确认NVIDIA驱动已装
- 确认CUDA版本
- 确认PyTorch CUDA版本匹配
- 检查命令：`nvidia-smi`

**3. 强烈推荐WSL2**
大模型开发建议走这条路：

```
Windows → WSL2 Ubuntu → Python

## 块 4

## 电脑配置批注：为LoRA微调做准备

你的目标是LoRA微调，所以配置规划要围绕“能跑起来、留有余量”来想。下面分三块讲。

**最低配置（能装环境、能跑小模型）**
- CPU：双核以上
- 内存：8GB
- 磁盘：20GB以上

**推荐配置（跑得更顺，建议按这个来）**
- CPU：四核以上
- 内存：16GB+
- 磁盘：50GB+

**GPU：装基础工具链不需要，但后续跑模型和微调就需要了。**

关于显存，给你一个量级参考（不是硬性门槛，实际取决于模型精度、上下文长度、batch、是否量化、KV Cache、是否训练等，以实际运行为准）：

| 任务 | 经验参考显存 | 主要前提条件（示例） |
|------|------------|-------------------|
| 小模型测试 | 约4GB+ | 数亿~1B级模型、FP16/BF16、短上下文、batch=1、

## 块 5

**本块学习检查**：运行 `python -m venv hf-env` 后，确认目录中出现 `hf-env` 文件夹，即创建成功。

---

**4.1 官方推荐：venv方式**

你的LoRA微调之旅，从这里打好地基。用Python自带的venv创建独立环境，避免依赖冲突。

**创建环境**（终端执行）：

```bash
python -m venv hf-env
```

**激活环境**（每次开始工作前都要执行）：

- **Linux**：`source hf-env/bin/activate`
- **Windows**：`hf-env\Scripts\activate`

激活后，命令行前缀会出现 `(hf-env)`，说明你已进入虚拟环境，后续

## 块 6

**本块学习检查**：运行下方验证代码，确认输出版本号无报错。

你的目标是LoRA微调，所以先装好Transformers。执行：

```bash
pip install "transformers[sentencepiece]"
```

带上`sentencepiece`是因为T5、LLaMA等模型的分词器需要它。装完验证：

```python
import transformers
print(transformers.__version__)
```

看到版本号即成功。

## 块 7

**安装Accelerate（加速你的LoRA训练）**

你的目标是LoRA微调，而微调通常需要大量计算。`accelerate` 这个库就是帮你管理计算资源的工具，它支持：

- **CPU训练**（没有GPU时也能跑）
- **单GPU训练**
- **多GPU训练**（多张卡并行加速）
- **混合精度训练**（在部分GPU上提升速度、减少显存占用）

安装命令很简单：

```bash
pip install accelerate
```

安装后，验证是否成功：

```bash
accelerate --version
```

看到版本号输出，就说明装好了。

---

**提示**：如果你还没装齐其他工具，可以一次性安装完整环境（包含Transformers、Datasets、Acceler

## 块 8

**本块学习检查**：请先按下方“版本矩阵”表格，把你当前环境的每个组件版本填进去（不确定就运行 `pip freeze > requirements.txt` 查看），并确认 Transformers 和 PEFT 已安装。

---

### 6.1 固定一套“已验证版本矩阵”（重要）

后文（第 11、12 板块）的安装命令**默认不锁定版本**，方便快速上手。但 Transformers / PEFT / Accelerate / Datasets / bitsandbytes 迭代较快，`evaluation_strategy` → `eval_strategy`、`tokenizer=` → `processing_class=`、4-bit 量化加载、`merge_and_unload` 等行为**会随版本变化**（出现警告、报错或行为差异）。

因此，在正式实验/教学前，**建议实际跑通一次并记录一套版本矩阵**，之后统一用它安装，避免“文档能跑、学员环境报错”。至少记录：

| 组件 | 版本 | 备注 |
| ---- | ---- | ---- |
| 操作系统 | 例：Ubuntu 22.04 / Windows 11 | 注明是否 WSL2 |
| GPU / 驱动 | 例：RTX 4090 / 驱动 5xx | 无 GPU 则填 CPU |
| Python | 例：3.10 |  |
| PyTorch | 例：2.x | 注明 CPU / CUDA(cuXXX) / ROCm |
| CUDA / ROCm | 例：cu121 | 与 PyTorch 对应 |
| Transformers | 例：4.xx |  |
| Datasets | 例：2.xx / 3.xx |  |
| Accelerate | 例：0.xx / 1.xx |  |
| PEFT | 例：0.xx | 第 12 板块用 |
| bitsandbytes | 例：0.xx | QLoRA 用，注明后端支持 |

> 建议同时记录**测试日期**和**当次使用的安装命令**（可直接 `pip freeze

## 块 9

你的目标是LoRA微调，而Token是访问HF生态的“钥匙”。本块用最直接的方式带你搞定它。

**为什么需要Token？**
简单说：部分模型（如私有模型、需授权模型）和上传模型时，HF需要确认“你是谁”。Token就是你的身份凭证。

**获取Token（2分钟）**
1. 打开Hugging Face官网 → 右上角头像 → `Settings`
2. 左侧选 `Access Tokens` → `Create new token`
3. 权限选择：
   - 只下载模型：选 `Read`
   - 需要上传模型：选 `Write`

**登录（命令行）**
执行以下命令（`huggingface-cli login`已弃用，请使用新版CLI）：
```bash
hf auth login
```
按提示粘贴Token，看到`Login successful`即成功。

**验证登录**
```bash
hf auth whoami
```
能显示你的用户名，说明一切正常。

---

**本块学习检查**（动手验证，不引入新知识）：
1. 在HF官网创建一个`Read`权限的Token。
2. 在终端执行`hf auth login`，用该Token登录。
3. 执行`hf auth whoami`，确认输出你的用户名。

完成这三步，你的HF环境就绪，可以

## 块 10

你的目标是 **LoRA 微调**，而 LoRA 微调通常需要加载一个预训练大模型（比如 7B 模型）。这类模型动辄十几GB，如果下载到默认位置，很容易把系统盘塞满。所以，**提前修改缓存路径**是动手实操前的关键一步。

### 为什么默认路径不行？

Hugging Face 的库默认把下载的模型放在：

- **Linux**: `~/.cache/huggingface`
- **Windows**: `C:\Users\用户名\.cache\huggingface`

问题在于，模型文件非常大：

| 模型 | 空间 |
| ---- | ---- |
| GPT-2 | 几百MB |
| 7B模型 | 十几GB |
| 70B模型 | 百GB |

如果你要微调 7B 模型，十几GB 直接写进系统盘（通常是 C 盘），很快就会爆掉。所以，**强烈建议**在动手前就改好路径。

### 改哪个变量？

主要认识两个环境变量：

- **`HF_HOME`**：这是总根目录，模型、数据集缓存、token 等所有 Hugging Face 相关文件都会放这里。**改它通常就够用了**。
- **`HF_HUB_CACHE`**：仅控制模型缓存目录，数据集缓存由 `HF_DATASETS_CACHE` 控制。

> 注意：如果两个都设置了，`HF_HUB_CACHE` 会覆盖 `HF_HOME` 推导出的对应子目录。新手建议直接设置 `HF_HOME`，简单省事。

---

### 动手配置（Linux）

打开终端，依次执行：

```bash
# 1. 创建缓存目录（比如放在 /data 下）
mkdir -p /data/huggingface_cache

# 2. 临时设置环境变量（仅当前终端有效）
export HF_HOME=/data/huggingface_cache

# 3. 永久生效（写入配置文件）
echo 'export HF_HOME=/data/huggingface_cache' >> ~/.bashrc
source ~/.bashrc

# 4. 验证是否设置成功
echo $HF_HOME
```

看到输出 `/data/huggingface_cache` 就成功了。

---

### 动手配置（Windows）

打开 **PowerShell**，执行：

```powershell
setx HF_HOME D:\huggingface_cache
```

**重要提醒**：`setx` 写入的是持久变量，但**当前窗口不会立即生效**。你需要**关闭并重新打开终端**（如果是 IDE 内置终端，需要重启 IDE）再检查。

新终端里验证：

```powershell
echo $env:HF_HOME
```

确认输出为 `D:\huggingface_cache` 即设置成功。

---

### 首次加载测试

路径配好后，先装 PyTorch。**不要直接 `pip install torch`**，那通常会装成 CPU 版。请先到 [

## 块 11

**本块学习检查**：按下方步骤完成首次加载测试，确认环境配置成功。

---

### 电脑配置批注（针对你的LoRA微调目标）

你的目标是LoRA微调，这意味着你至少需要能**加载并运行一个小模型**。根据你的实训条件，请对照以下配置要求：

- **最低配置**：CPU即可运行小模型，内存建议8GB以上。如果你的电脑只有这个水平，也能完成本块测试，但后续LoRA微调会较慢。
- **推荐配置**：内存16GB以上，最好有NVIDIA CUDA GPU且显存8GB以上。如果你有GPU，请确保PyTorch的CUDA版本匹配，否则会报“CUDA不可用”。

---

### 首次加载模型测试（动手操作）

**第一步**：创建测试文件 `test_hf.py`，输入以下代码：

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

result = generator(
    "Artificial intelligence is",
    max_length=50
)

print(result)
```

**第二步**：在终端运行：

```bash
python test_hf.py
```

**第三步**：观察输出。如果成功输出一段文本，说明三件事全部完成：
- Transformers安装成功
- 模型下载成功
- 缓存目录配置成功

> **模型许可证**：本测试使用GPT-2模型，遵循MIT许可证，可自由使用。

---

### 常见问题速查（遇到报错时对照）

| 问题 | 常见原因 | 解决 |
|------|----------|------|
| Tokenizer报错 | 缺依赖或模型文件问题 | 先补 `transformers[s

## 块 12

## 环境验证清单：Transformers

你的目标是 LoRA 微调，而这一切都建立在环境能跑通的基础上。这一块就是“开机自检”——用最短的时间确认工具链是否就绪。

### 1. 核心三件套验证

打开你的 Python 环境，依次执行：

```python
import transformers
import datasets
```

没有报错，就是 ✅ 正常。

再验证 Accelerate：

```bash
accelerate --version
```

能输出版本号，就是 ✅ 正常。

> **hands-on 提示**：把这三条命令写进一个 `.py` 文件或 notebook 单元格里，一次跑完。以后换机器、换环境，先跑这个脚本再干活。

### 2. Hugging Face 登录验证

```bash
hf auth whoami
```

能显示你的用户名，说明 Hub 配置完成。旧命令 `huggingface-cli whoami` 已弃用，仅作兼容性保留，推荐使用新命令。

### 3. GPU 验证（关键！）

**只验证上面三个库能导入，不代表 GPU 能用。** 你后面做 LoRA 微调，GPU 基本是刚需，所以这一步必须确认到位：

```python
import torch
from transformers import AutoModel

print(torch.__version__)
print(torch.cuda.is_available())        # 期望 True（有可用 GPU 时）
print(torch.version.cuda)               # torch 编译时的 CUDA 版本
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# 确认模型确实加载到了 GPU
# 注意：模型加载需关注显存要求，例如 Llama-2-7b FP16 需要至少14GB显存
model = AutoModel.from_pretrained("bert-base-uncased")
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(next(model.parameters()).device)  # 期望 cuda:0（使用 GPU 时）
```

**三个检查点**：
- `is_available()` 为 `True`
- `version.cuda` 与你的环境匹配
- 参数 device 为 `cuda:0`

> ⚠️ 注意：`torch.cuda.is_available()=True` 只说明 PyTorch 能用 GPU，**不等于** bitsandbytes 4-bit 量化等更上层功能一定可用——那是你后面学 QLoRA 时要验证的。

---

## 总结与后续路线

到这里，环境准备全部完成：

✅ Transformers / Datasets / Accelerate 环境完成  
✅ Hugging Face Hub 配置完成  
✅ Token 配置完成  
✅ 缓存路径优化完成  
✅ Windows/Linux/Colab 方案了解完成  

后续学习路线：

```
AutoTokenizer / AutoModel
        ↓
datasets数据处理

## 块 13

好的，我们直接进入正题。既然你的目标是 LoRA 微调，那么理解模型是如何被“装”进来的，就是万里长征第一步。这一块，我们聚焦于最核心的加载流程，用动手的方式把它彻底搞懂。

**1.1 Transformers加载模型的基本流程**

想象一下，你要用一套乐高积木（预训练模型）来搭建一个专属作品（微调后的模型）。`AutoModel` 和 `AutoTokenizer` 就是你最得力的两个助手。

**第一步：认识你的“翻译官”——AutoTokenizer**

大模型不认识我们说的“人话”，它只认识数字（确切地说是token ID）。`AutoTokenizer` 就是那个“翻译官”，它的核心职责是：

1.  **分词**：把一句话拆成模型能理解的最小单元（token）。
2.  **编码**：把这些token转换成一串数字ID。
3.  **对齐**：确保输入给模型的数字序列，和模型预训练时看到的格式完全一致。

**第二步：认识你的“模型加载器”——AutoModel**

`AutoModel` 则是一个“智能加载器”。你只需要告诉它模型的名字（比如 `bert-base-uncased` 或 `Qwen/Qwen2-7B`），它就能自动识别出这个模型的“骨架”（架构），并为你加载好对应的权重。你不需要手动去写一个BERT类或Llama类，`AutoModel` 会帮你搞定一切。

**第三步：核心动作——`from_pretrained()`**

这是整个流程中最关键的一行代码。无论是 `AutoModel` 还是 `AutoTokenizer`，它们的核心方法都是 `from_pretrained()`。

- 对于 `AutoTokenizer`：`tokenizer = AutoTokenizer.from_pretrained("模型名字")`
- 对于 `AutoModel`：`model = AutoModel.from_pretrained("模型名字")`

这个方法会做两件事：
1.  **下载**：如果本地没有缓存，它会从 Hugging Face Hub 下载模型的配置文件、权重文件和词表文件。
2.  **初始化**：根据配置文件，构建出模型和分词器的实例。

**第四步：串联起来，跑通一个最小流程**

现在，我们把它们串起来，用代码走一遍“加载 -> 处理 -> 推理”的完整流程。这里我们用一个较小的模型 `bert-base-uncased` 来演示，确保你本地能跑通。

```python
from transformers import AutoTokenizer, AutoModel

# 1. 加载分词器和模型
model_name = "bert-base-uncased"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
except OSError as e:
    print(f"模型加载失败，请检查网络连接或本地缓存：{e}")
    raise

# 2. 处理输入文本
text = "Hello, I want to learn LoRA."
inputs = tokenizer(text, return_tensors="pt")

# 3. 执行推理（前向传播）
outputs = model(**inputs)

# 4. 查看输出结果
# outputs.last_hidden_state 是模型最后一层的输出，包含了每个token的语义表示
print(outputs.last_hidden_state.shape)
```

**本块学习检查**

请你在本地环境运行上面的代码，并观察输出结果。思考一下：`outputs.last_hidden_state.shape` 打印出来的形状，它的三个维度分别代表什么含义？这和 `tokenizer` 处理后的 `inputs` 有什么关系？想清楚这个问题，你就理解了模型输入输出的基本维度逻辑，这对后续微调工作至关重要。

## 块 14

## 一、为什么需要 AutoModel 和 AutoTokenizer？

### 1.1 Transformers加载模型的基本流程

大语言模型**不能直接理解文本**。你输入“你好，大模型”，模型实际接收的是一串数字——每个 token 对应一个整数 ID（具体数值取决于分词器，请以实际运行为准）。

完整流程如下：

```
文本 → Tokenizer分词 → Token ID数字 → Transformer模型计算 → 输出结果 → Tokenizer解码 → 文本
```

两个核心角色的分工：

- **Tokenizer**：负责“文字 ↔ 数字”的转换
- **Model**：负责“数字 → 神经网络计算”

Hugging Face Transformers 提供了**统一接口**，让不同模型都能用类似代码加载。

---

**关键认知**：`AutoModel` / `AutoTokenizer` 是**统一的加载入口**——用相似的代码就能把不同模型的权重和分词器读进来。但“能用同一套代码加载”**不等于**“能用同一套推理参数和对话方式使用”。不同模型（尤其 Llama、Qwen 等 decoder-only 对话模型）还涉及各自的 **chat template**、`pad_token` / `eos_token` 设置、是否为 **gated/授权模型**、以及 `trust_remote_code=True` 会执行仓库自带代码（需确认来源可信）等差异。此外，加载模型时需注意显存要求，例如：Llama-2-7b FP16 需要至少14GB显存。

本章主要用 BERT / GPT-2 演示通用的“加载—编码—前向/生成”流程，对话模型的完整用法见后续章节。

---

**本块学习检查**：用自己的话简述“文本

## 块 15

### 为什么不用具体模型类？

想象你要在LoRA微调时快速切换BERT、GPT-2或RoBERTa。传统做法是：

```python
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-uncased")
```

换GPT-2就得改成：

```python
from transformers import GPT2Model
model = GPT2Model.from_pretrained("gpt2")
```

问题很明显：每个模型都要记不同的类名（`BertModel`、`GPT2Model`、`RobertaModel`），切换模型时代码要大量改动。

**AutoModel 的解法**：它自动根据模型配置选择正确类型，你只需写一次：

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
```

换模型时只改字符串，代码结构不变。

---

**动手验证**：打开终端，分别用 `BertModel` 和 `AutoModel` 加载 `"bert-base-uncased"`。

# 模型许可证：BERT 使用 Apache 2.0 许可证。
# 注意：BERT-base FP16 需要至少 4GB 显存。

## 块 16

**本块学习检查**：读完本节后，请用一句话回答：Tokenizer 把文本变成什么，模型才能处理？

---

### 2.1 Tokenizer 是什么？

你要做 LoRA 微调，第一步就是让模型“看懂”文字。Tokenizer 就是翻译官，负责四件事：

1. **切分文本**：把句子拆成小块；
2. **映射编号**：每块对应一个数字 ID；
3. **加特殊标记**：如开头、结尾符号；
4. **生成输入格式**：整理成模型要的样式。

比如输入 `Hello AI`，Tokenizer 会把它变成一串数字（ID 仅为示意）。**注意**：不同模型（如 GPT-2 与 BERT）对同一句话的切分和编号都不一样。

**关键点**：模型只认数字，不认文字。

---

### 2.2 用 AutoTokenizer 加载

基础代码：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

`checkpoint` 就是模型名称。执行后，Transformers

## 块 17

**动手试试：把文本变成数字**

直接调用 `tokenizer`，传入一句话，就能得到模型能读懂的“数字版”文本：

```python
text = "Hello, Hugging Face"
inputs = tokenizer(text)
print(inputs)
```

输出会包含两个关键字段（以 `bert-base-uncased` 为例，具体数值以实际运行为准）：

```python
{
 'input_ids': [101, 7592, 1010, 17662, 2227, 102],
 'attention_mask': [1, 1, 1, 1, 1, 1]
}
```

**这两个字段分别是什么？**

| 字段 | 作用 |
|---|---|
| `input_ids` | 文本转换后的数字编号，模型靠它理解内容 |
| `attention_mask` | 标记哪些位置是有效文本（1有效，0无效） |

**小练习**：换一句你自己的

## 块 18

**本块学习检查**：运行下面的代码，观察输出，确认 `AutoModel` 自动创建了 `BertModel` 实例。

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
print(type(model).__name__)  # 应输出 BertModel
```

---

`AutoModel` 是 Transformers 的自动加载接口。你只需提供**模型名称**（如 `"bert-base-uncased"`）和**配置文件**，它就会自动识别架构。

内部流程：读取 `config.json` 中的 `model_type` 字段（如 `bert`），据此创建对应的模型类（如 `BertModel`）。无需手动指定架构，适合快速加载模型。

# 模型许可证：BERT 采用 Apache 2.0 许可证（商用无需授权）。
# 注意：bert-base-uncased 模型加载需约 1.2GB 显存。

## 块 19

**AutoModel加载与你的LoRA微调目标**

你接下来要做LoRA微调，得先跑通模型加载。本块只讲“电脑配置”这个前提，不涉及代码细节。

**最低配置（能跑，但慢）**
- CPU：Intel i5或AMD同级以上
- 内存：8GB
- 磁盘：至少20GB空闲

**推荐配置（做微调更舒服）**
- 内存：16GB以上
- GPU：NVIDIA显卡，支持CUDA
- 显存：小模型测试4GB以上；加载7B模型需12GB以上

**关键认知（结合你的实操目标）**
- AutoModel加载API本身用CPU就能运行，所以“跑通代码”门槛

## 块 20

**本块学习检查**：运行上方代码后，在终端打印 `tokenizer` 和 `model` 的类型，确认两者分别被成功加载。

---

**动手加载模型（LoRA微调第一步）**

你的目标是LoRA微调，所以先要拿到一个预训练模型。这里用`bert-base-uncased`作为起点。

**两步走：**

1. **加载分词器**：`tokenizer = AutoTokenizer.from_pretrained(checkpoint)`  
   它负责把文本变成模型能读的数字。

2. **加载模型**：`model = AutoModel.from_pretrained(checkpoint)`  
   它负责神经网络计算。

**关键点**：`checkpoint`变量指定了模型名称，两行代码都从同一个

# 模型许可证

## 块 21

## 4.2 输入文本：把文字变成模型能读的格式

你手上有了一段文字，但模型不认识纯文本，它只认数字。所以我们要用 `tokenizer` 把文字“翻译”成数字 ID：

```python
text = "Hugging Face is amazing"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
```

关键参数 `return_tensors="pt"` 的意思是：**返回 PyTorch 的 Tensor 格式**。这样输出才能直接喂给 PyTorch 模型。

打印后你会看到两个键：

- `input_ids`：每个 token 对应的数字 ID
- `attention_mask`：哪些位置是真实文本（1），哪些是填充（0）

## 4.3 输入模型：把字典“拆开”传进去

```python
outputs = model(**inputs)
```

这里的 `**` 是 Python 的**字典解包语法**——把 `inputs` 这个字典里的每个键值对，拆成独立的关键字参数传进去。上面这行等价于：

```python
model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
```

## 4.4 查看输出

```python
print(outputs)
```

输出是一个包含模型预测结果的 `ModelOutput` 对象。你可以通过 `output

## 块 22

## 5.1 为什么不用AutoModel？

你学LoRA微调，最终要微调的是**能生成文本的对话模型**（比如GPT类）。`AutoModel`加载的是Transformer主干，默认输出最后一层隐藏状态（last_hidden_state），适合特征提取或自定义下游任务；而`AutoModelForCausalLM`在主干之上附加语言模型头，输出logits用于文本生成。

但聊天模型要的是**生成文本**，所以得用带“语言模型头”的加载类：`AutoModelForCausalLM`。

## 5.2 加载GPT类模型

直接看代码，这就是你后面做LoRA微调的第一步：

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# 注意：Llama-2-7b FP16 需要至少14GB显存。
```

**关键区别**——不同任务用不同的Auto类：

| 类 | 用途 |
|---|---|
| AutoModel | 基础Transformer（无任务头） |
| AutoModelForSequenceClassification | 分类 |
| AutoModelForCausalLM | 文本生成 |
| AutoModelForSeq2SeqLM | 序列到序列生成任务（如翻译、摘要） |

**什么时候用哪个？**

- `AutoModel`：只加载主干，输出隐藏状态，适合**特征提取**或自定义下游结构。它**不含任务头**，用`AutoModel`加载时，只会加载主干权重，不会加载任务头，因此无法直接用于分类/生成任务。
- `AutoModelFor*`：在主干上加对应任务头。做分类、生成、摘要等任务，以及**保存/重新加载后仍需要该任务头**时，必须用与训练时一致的任务类，否则任务头语义会丢失。

> **提示**：本章用BERT/GPT-2演示通用流程。Llama、Qwen等对话模型还涉及`AutoModelForCausalLM`、chat template、`pad_token`设置、gated模型授权、`trust_remote_code=True`（会执行仓库代码，需确认来源可信）等，后续章节展开。别以为“会加载BERT”就等于“会正确推理任意LLM”。

---

**本块学习检查**：打开Python环境，用`gpt2`跑一遍上面的加载代码，然后打印`model`对象，观察它的结构里是否包含语言模型头（`lm_head`）。

## 块 23

## 5.3 decoder-only 小模型完整生成示例（GPT-2）

前面 BERT 示例只做到“前向计算得到隐藏状态”，并没有真正**生成文本**。这里用小巧的 decoder-only 模型 GPT-2 演示一个**可运行的完整生成流程**，为后续 LoRA 微调打好基础——因为你要微调的 Llama/Qwen 等模型正是这种 decoder-only 架构，生成流程完全一致，只是多了对话模板处理（后续章节展开）。

**完整代码：**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# GPT-2 默认没有 pad_token，批量/生成时需要指定，常用 eos_token 兼作 pad
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)
```

**要点说明（结合你的 LoRA 微调目标）：**

* **模型类选择**：生成模型用 `AutoModelForCausalLM`（带语言模型头），不是 `AutoModel`。你之后做 LoRA 微调时，加载的也是这个类，只是会额外套上 `PeftModel` 包装。
* **pad_token 处理**：decoder-only 模型常需显式设置 `pad_token`（GPT-2 默认没有），用 `eos_token` 兼任即可。这个坑在微调数据整理时同样会遇到。
* **生成方式**：用 `model.generate(...)` 而非直接前向，`max_new_tokens` 控制新生成长度。微调完成后，你也是用同样的方式让模型产出文本。
* **采样参数**：`do_sample` / `temperature` / `top_p` 影响输出风格，不是“越大越好”，需按任务调。LoRA 微调后评估效果时，这些参数也会影响你看到的生成质量。
* **显存要求**：加载模型前需确认显存是否充足，例如 Llama-2-7b FP16 需要至少 14GB 显存。
* **对话模型差异**：Llama/Qwen 等真实对话模型在此基础上，还需先用 `tokenizer.apply_chat_template(messages, ...)` 把多轮消息拼成模型期望的格式，再交给 `generate`——这属于后续章节内容。

> 提示：GPT-2 是英文小模型，输出质量有限，这里只用于演示“decoder-only 生成”的最小可运行链路，不代表生产级效果。

## 块 24

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
`device_map` 解决的是“模型加载后放哪”的问题。你之后做 LoRA 微调，模型通常能放进单卡，所以这个参数主要用于**推理**或**模型太大单卡放不下**的场景。

它的写法很简单：

```python
model = AutoModel.from_pretrained(
    checkpoint,
    device_map="auto"
)
```

`"auto"` 的意思是：加载时自动把模型的不同层分配到可用设备上——比如有多张 GPU 就分到多张 GPU，不够就放 CPU，再不够就用内存/磁盘临时存放（offload）。

**注意**：

## 块 25

### 6.2 torch_dtype：用对精度，省显存

加载模型时，`torch_dtype` 控制权重精度。直接传 `"auto"`：

```python
model = AutoModel.from_pretrained(checkpoint, torch_dtype="auto")
```

它的含义是：**读取模型 checkpoint 配置里记录的 dtype**（`config.torch_dtype`），按此加载权重。可能值是 `float32`、`float16` 或 `bfloat16`——**不是**根据你的显卡自动选。

**关键点**：如果 checkpoint 保存为半精度（fp16/bf16），用 `"auto"` 比默认的 float32 更省显存。想强制指定，直接传：

```python
# 模型许可证

## 块 26

## 6.3 local_files_only：断网也能加载模型

你要做LoRA微调，大概率会在服务器上跑。很多服务器是**离线环境**，没法访问HuggingFace Hub。这时候`local_files_only=True`就是你的救星——它会阻止`from_pretrained`联网下载，仅从本地缓存加载；若本地缓存中不存在该模型，则会抛出异常。

```python
model = AutoModel.from_pretrained(
    checkpoint,
    local_files_only=True
)
```

适用场景：
* 离线服务器
* 无网络环境

> **实操提示**：第一次加载模型时先联网（不设这个参数），让模型下载到本地缓存。之后在离线服务器上就能用`local_files_only=True`直接加载了。

---

## 七、从本地路径加载模型

除了传Hub上的模型名（如`"bert-base-uncased"`），你也可以直接传**本地目录路径**：

```python
"./my_model"
```

目录结构长这样（小模型典型示例）：

```
my_model
├── config.json
├── model.safetensors
├── tokenizer.json
```

**大模型注意**：权重文件太大时会**分片保存**，目录里会有多个分片文件加一个索引：

```
my_model
├── config.json
├── model-00001-of-00003.safetensors
├── model-00002-of-00003.safetensors
├── model-00003-of-00003.safetensors
├── model.safetensors.index.json   # 分片索引，记录了参数与分片的映射，from_pretrained 会自动加载所有分片
├── tokenizer.json
└── tokenizer_config.json
```

两种情况都用**同一行代码**加载，**无需手动合并分片**：

```python
tokenizer = AutoTokenizer.from_pretrained("./my_model")
model = AutoModel.from_pretrained("./my_model")
```

> **模型许可证**：部署引用模型时需确认其许可证类型。例如，Llama 2 Community License（商用月活>7亿需Meta授权）。

---

## 八、常见错误排查

| 错误 | 原因 | 解决 |
|---|---|---|
| Model name not found | 模型名称错误 | 检查Hub名称 |
| 401 Unauthorized | 需要身份认证/权限：未登录、Token权限不足、gated模型未授权、无私有仓库访问权限 | 先`hf auth login`；到模型页**接受授权协议**；确认账号有访问权限；检查Token的**scope/权限** |
| CUDA out of memory | 显存不足 | 降低batch或使用量化 |
| Tokenizer mismatch | 模型和Tokenizer不匹配 | 使用同checkpoint |
| 下载速度慢 | 网络问题 | 配置缓存或镜像 |

---

## 九、环境验证实验

创建`test_model.py`，验证你本地缓存或本地目录里的模型能否正常加载。**建议先跑通这一步，再做LoRA微调**——如果基础

## 块 27

## 6.3 `local_files_only`：让加载过程“断网也能跑”

你已经掌握了`from_pretrained()`的基本用法——它默认会先查本地缓存，找不到就联网下载。但如果你在训练或推理时希望**强制只用本地文件**，就需要用到`local_files_only`参数。

**它解决什么问题？**

- 默认行为：`AutoModel.from_pretrained(checkpoint)`会尝试联网获取模型文件（如果本地没有）。
- 加了这个参数：`AutoModel.from_pretrained(checkpoint, local_files_only=True)`，加载过程**完全离线**，只检查本地缓存，找不到就直接报错，不会发起网络请求。

**为什么这对LoRA微调有用？**

你后续做LoRA微调时，通常会把基础模型（比如`bert-base-uncased`）提前下载到本地。之后每次运行训练脚本，加上`local_files_only=True`可以：

1. **避免意外联网**：防止训练过程中因网络波动或权限问题卡住。
2. **保证可复现**：确保每次加载的都是同一份本地文件，版本一致。
3. **加速启动**：跳过网络检查，直接读本地。

**用法示例（在原始代码基础上修改）：**

```python
from transformers import AutoTokenizer, AutoModel

checkpoint = "bert-base-uncased"

# 强制只用本地文件
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModel.from_pretrained(checkpoint, local_files_only=True)
```

**注意**：如果本地缓存里没有`bert-base-uncased`，运行会直接报错。所以第一次使用时，建议先不加这个参数正常加载一次（下载到缓存），之后再加。

---

**本块学习检查**：请修改你刚才运行的`test_model.py`，给两个`from_pretrained()`调用都加上`local_files_only=True`，重新运行。观察结果——如果之前已成功加载过，

## 测评题目

### Q-KB11-ST1100-001
Hugging Face 数据集加载和处理常使用的库名称是________。

### Q-KB11-ST1100-003
以下哪个库最直接用于训练配置中的分布式加速？

### Q-KB1101-001
请写出使用 `AutoModelForCausalLM` 加载 `gpt2` 模型的代码。

### Q-KB1101-004
`AutoModelForCausalLM` 和 `AutoModelForMaskedLM` 的区别是？

### Q-KB11-ST1102-001
datasets 的 `map()` 常用于对数据集样本执行批量________。

### Q-KB1103-004
在 `SFTConfig` 中，`max_seq_length` 参数的作用是？

### Q-KB1101-003
请写代码：使用 GPT-2 模型和分词器，对输入文本 `"Hello, I am"` 生成接下来的 20 个词（设置 `max_new_tokens=20`），并解码输出。

### Q-KB1103-002
`SFTTrainer` 继承自 HuggingFace 的 `Trainer`，它额外支持直接传入________格式的原始数据集，并自动完成分词和格式化。
