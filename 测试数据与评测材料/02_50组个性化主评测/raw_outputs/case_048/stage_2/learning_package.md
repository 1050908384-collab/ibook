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

整体学习流程为：Tokenizer → Transformers加载模型 → Datasets处理数据 → Accelerate优化训练 → Trainer完成微调 → LoRA/QLoRA → 部署推理。

---

## 二、环境方案选择

Hugging Face学习环境主要有三种：Google Colab（推荐新手）、Windows、Linux。你当前未提供硬件信息，因此本块聚焦于**环境准备的核心操作**，不涉及具体硬件适配细节。

---

## 三、核心安装与配置步骤

### 1. 安装 Transformers、Datasets、Accelerate

使用 pip 安装三个核心库：

```bash
pip install transformers datasets accelerate
```

- **Transformers**：负责模型的加载、训练与推理；
- **Datasets**：负责训练数据的加载与预处理；
- **Accelerate**：负责简化单卡/多卡训练及混合精度配置。

### 2. 配置 Hugging Face Hub 与 Token

Hugging Face Hub 是模型与数据集的共享平台。要下载或上传模型，需要：

1. 在 Hugging Face 官网注册账号并获取 **Access Token**；
2. 使用 `huggingface_hub` 库登录：

```python
from huggingface_hub import login
try:
    login(token="你的Token")
except Exception as e:
    print(f"登录失败: {e}")
```

### 3. 设置模型缓存路径

Hugging Face 默认将下载的模型缓存到用户目录。建议手动指定缓存路径，便于管理：

```python
import os
# 注意：路径不存在时可能引发异常，请确保目录已创建
os.environ["HF_HOME"] = "/your/cache/path"
```

### 4. 部署前检查

部署文档引用模型时，需确认模型许可证与显存要求。例如：

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
```

---

## 本块学习检查

请完成以下最小示例，回顾本块核心内容：

1

## 块 2

**本块学习检查**：读完本块后，请合上资料，用自己的话回答：Google Colab 是什么？它适合哪两类人？在 Colab 中安装 Hugging Face 工具链需要运行哪两条命令？

---

### 2.1 什么是Google Colab？

Google Colab 是 Google 提供的**云端 Jupyter Notebook 环境**。它的核心价值在于：你不需要在本地配置 Python，打开浏览器就能直接运行 Python 代码，还可以申请 GPU 资源。

**为什么适合你？** 你的目标是 LoRA 微调，且学习节奏偏快。Colab 能让你跳过本地环境配置的繁琐步骤，直接开始验证 Transformers 代码——这正是你当前阶段最需要的“快速上手”路径。

**适合人群**：初学者、没有 NVIDIA 显卡的用户、需要快速验证模型代码的人。

---

### 2.2 Colab安装环境

打开 Colab 后，依次运行以下两条命令安装工具链：

```python
!pip install "transformers[sentencepiece]"
!pip install datasets accelerate huggingface_hub
```

安装完成后，运行以下代码验证是否成功：

```python
import transformers
import datasets
import accelerate

print(transformers.__version__)
```

## 块 3

### Windows兼容性提醒（面向LoRA微调准备）

你在Windows上配置HuggingFace环境时，需注意三点：

**1. 路径格式**  
Windows用反斜杠，如`D:\huggingface_cache`；Linux用正斜杠，如`/data/huggingface_cache`。设置环境变量时格式不同，请按系统调整。

**2. CUDA环境（若用GPU）**  
需确认NVIDIA驱动、CUDA版本与PyTorch CUDA版本匹配。用`nvidia-smi`命令检查驱动状态。

**3. 推荐WSL2**  
大模型开发建议用WSL2 Ubuntu，链路为：Windows → WSL2 Ubuntu → Python环境 → Transformers。相比Windows原生，Linux兼容性更好

## 块 4

**本块学习检查**：阅读完配置要求后，请对照你手头的电脑，判断它属于“最低配置”还是“推荐配置”，并记录下你的结论。

---

你的目标是LoRA微调，所以请重点关注“模型微调”那一行的经验参考显存。先明确一点：**安装基础工具链（Transformers、Datasets、Accelerate）不需要GPU**，CPU双核、8GB内存、20GB磁盘就能起步。

但后续要跑模型，就得看配置了。最低配置（双核CPU、8GB内存）只能做环境安装和基础测试；推荐配置（四核CPU、16GB内存、50GB磁盘）会更从容。

关于GPU显存，记住一个核心原则：**实际占用取决于模型精度、上下文长度、batch、是否量化、KV Cache、是否训练等**，下表只是量级参考：

| 任务 | 经验参考显存 | 主要前提条件（示例） |
|------|------------|-------------------|
| 小模型测试 | 约4GB+ | 数亿~1B级模型、FP16/BF16、短上下文、batch=1、纯推理 |
| 7B模型推理 | 约12GB+ |

## 块 5

### 4.1 官方推荐：venv方式

创建虚拟环境：

```bash
python -m venv hf-env
```

激活环境：

- **Linux**：`source hf-env/bin/activate`
- **Windows**：`hf-env\Scripts\activate`

---

**本块学习检查**：请在你的终端中执行上述创建命令，然后激活环境。激活成功后，命令行提示符前应出现 `(hf-env)` 前缀。若未出现，请检查路径是否正确，并重试。

## 块 6

### 安装Transformers：为LoRA微调铺路

你的目标是LoRA微调，而Transformers库是这一切的入口。安装时，请使用：

```bash
pip install "transformers[sentencepiece]"
```

注意这个额外的`sentencepiece`依赖——它不是可选项。因为T5、LLaMA系列以及部分中文模型的Tokenizer，都依赖SentencePiece分词库来正确切分文本。跳过它，后续加载这些模型时会直接报错。

安装完成后，用一行代码验证：

```python
import transformers
print(transformers.__version__)
```

看到版本号输出，说明核心库已就绪。

**本块学习检查**：在终端执行上述安装命令，然后运行验证代码，确认输出版本号。若报错，记录错误信息并检查pip源或网络。注意：Llama-2-7b FP16 需要至少14

## 块 7

**安装Accelerate（加速库）**

Accelerate是Hugging Face生态中用于简化训练流程的工具，尤其适合你后续的LoRA微调目标。它让你用更少的代码切换不同训练环境。

**安装命令**：
```bash
pip install accelerate
```

**它能做什么**：
- 支持CPU训练（无GPU时的备选方案）
- 支持单GPU训练
- 支持多GPU训练（分布式）
- 支持混合精度训练（提升速度、节省显存）

**验证安装**：
```bash
accelerate --version
```
能输出版本号即安装成功。

---

**提示**：若想一次性装齐后续微调所需工具，可运行：
```bash
pip install "transformers[s

## 块 8

## 版本固定：你的“防翻车”第一课

你正在为 LoRA 微调打基础，而版本问题往往是新手第一个“隐形坑”。本块的核心就一句话：**别用最新，用“验证过”的**。

### 为什么必须固定版本？

Transformers、PEFT、Accelerate 这些库迭代极快。后文你会遇到的 `eval_strategy`、`processing_class=`、4-bit 量化加载等写法，**会随版本变化**——可能只是警告，也可能直接报错。文档能跑通、你环境报错，十有八九是版本不一致。

### 你的行动清单（正式实验前必做）

1. **跑通一次**：按后续章节装完环境后，实际运行一个最小示例。
2. **记录版本矩阵**：把下表填完，存进项目里。

| 组件 | 你的版本 | 备注 |
| ---- | ---- | ---- |
| 操作系统 | 例：Ubuntu 22.04 | 注明是否 WSL2 |
| GPU / 驱动 | 例：RTX 4090 | 无 GPU 填 CPU |
| Python | 例：3.10 | |
| PyTorch | 例：2.x | 注明 CPU / CUDA(cuXXX) |
| CUDA | 例：cu121 | 与 PyTorch 对应 |
| Transformers | 例：4.xx | |
| Datasets | 例：2.xx | |
| Accelerate | 例：0.xx | |
| PEFT | 例：0.xx | LoRA 用 |
| bitsandbytes | 例：0.xx | QLoRA 用 |

3. **固化现场**：跑通后执行 `pip freeze > requirements.txt`，附在项目里。以后换机器或重装，直接按这个文件装。

> 注意：上表版本号只是占位示例，**不代表推荐组合**。以你实际验证通过的为准。同时记录**测试日期**。

### 配套工具：huggingface_hub

## 块 9

你的目标是快速掌握LoRA微调，而Hugging Face的Token正是你访问模型库的“钥匙”。本块聚焦于“为什么”和“怎么配”，不涉及模型细节。

**为什么需要Token？**  
并非所有模型都公开可下载。以下场景必须身份认证：  
- **私有模型**：仅限特定账号访问。  
- **需授权协议模型**：如某些商用或受限模型，需同意条款。  
- **上传模型**：向Hub发布你的模型时，需验证身份。

**获取Token（三步）**  
1. 登录Hugging Face官网，进入`Settings`。  
2. 选择`Access Tokens`，点击`Create new token`。  
3. 按用途选权限：  
   - 仅下载模型：选`Read`。  
   - 需上传模型：选`Write`（包含读权限）。

**登录与验证**  
在终端执行（推荐新命令）：  
```bash
hf auth login
```
输入Token后，看到`Login successful`即成功。  
验证身份（推荐）：  
```bash
hf auth whoami
```
> 注意：旧命令`huggingface-cli login`在huggingface

## 块 10

## 9.1 为什么修改缓存？

你的目标是 LoRA 微调，而微调的前提是能顺利加载预训练模型。模型文件会先下载到本地缓存，默认位置在：

- **Linux**：`~/.cache/huggingface`
- **Windows**：`C:\Users\用户名\.cache\huggingface`

问题在于：**模型体积远超你的想象**。参考下表：

| 模型 | 占用空间 |
|------|---------|
| GPT-2 | 几百MB |
| 7B模型 | 十几GB |
| 70B模型 | 百GB |

默认缓存目录通常位于系统盘（C盘或根分区），空间有限。一旦磁盘写满，模型加载会直接失败。**建议提前修改缓存路径**，指向空间充裕的磁盘。

### 关键变量说明

- **`HF_HOME`**：Hugging Face 相关文件的**总根目录**，涵盖模型/数据集下载缓存、token 等。作用范围最广，**改它通常就够用**。
- **`HF_HUB_CACHE`**：仅影响从 Hub 下载的模型/数据集**缓存**目录，粒度更细。
- `TRANSFORMERS_CACHE` 是旧的环境变量，仍可使用，但推荐使用 `HF_HOME` 或 `HF_HUB_CACHE` 以获得更全面的控制。

> 若同时设置多个变量，**细粒度变量（如 `HF_HUB_CACHE`）会覆盖 `HF_HOME` 推导出的对应子目录**。

---

## 十、Linux 配置缓存

**创建目录**：

```bash
mkdir -p /data/huggingface_cache
```

**临时设置**（仅当前终端有效）：

```bash
export HF_HOME=/data/huggingface_cache
```

**永久生效**（写入 shell 配置文件）：

```bash
echo 'export HF_HOME=/data/huggingface_cache' >> ~/.bashrc
source ~/.bashrc
```

**验证**：

```bash
echo $HF_HOME
```

输出应为 `/data/huggingface_cache`。

---

## 十一、Windows 配置缓存

使用 PowerShell 设置**用户级持久变量**：

```powershell
setx HF_HOME D:\huggingface_cache
```

> ⚠️ **重要**：`setx` 写入的是持久变量，**只对之后新开的终端生效**，当前窗口不会更新。请**关闭并重新打开终端**（IDE 内置终端需重启 IDE）后再验证。

**验证**：

```powershell
echo $env:HF_HOME
```

---

## 十二、首次加载模型测试

配置好缓存后，需要安装 PyTorch 才能加载模型。

> ⚠️ **注意**：直接 `pip install torch` 通常只装 CPU 版（或不一定匹配你的 CUDA）。请以 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为准。

**仅 CPU**：

```bash
pip install torch
```

**NVIDIA CUDA**（示例，具体命令以官方选择器为准）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install

## 块 11

## 电脑配置批注：运行模型前，先看看你的机器

你的目标是LoRA微调，但一切要从“能跑起来”开始。本块帮你判断：当前电脑能否完成模型加载测试。

### 最低配置（能跑，但仅限小模型）

- **CPU**：可以运行小模型，但速度较慢；
- **内存**：8GB以上。

### 推荐配置（流畅体验）

- **内存**：16GB以上；
- **GPU**：NVIDIA CUDA GPU；
- **显存**：8GB以上。

> 你的实训条件未提供硬件信息，建议先确认内存是否达到8GB。若无NVIDIA GPU，仍可用CPU完成本测试，只是速度会慢。

> 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

### 验证安装：首次加载模型

创建文件 `test_hf.py`，写入以下代码：

```python
from transformers import pipeline

try:
    generator = pipeline(
        "text-generation",
        model="gpt2"
    )

    result = generator(
        "Artificial intelligence is",
        max_length=50
    )

    print(result)
except Exception as e:
    print(f"模型加载或生成失败：{e}")
    print("请检查网络连接、模型文件完整性及依赖安装。")
```

运行：

```bash
python test_hf.py
```

**成功标志**：程序输出一段文本。这说明三件事全部就绪：

1. Transformers安装成功；
2. 模型下载成功；
3. 缓存目录配置成功。

### 常见问题速查

| 问题

## 块 12

### Transformers 环境验证（面向 LoRA 微调）

既然你的目标是 **LoRA 微调**，那么环境验证的重点就非常明确了：**确保 Transformers 库能正常导入**，这是后续加载预训练模型、执行微调的基础。

#### 第一步：验证 Transformers 库

在你的 Python 环境中，运行以下代码：

```python
import transformers
```

如果没有任何报错，说明 Transformers 已正确安装。这是最基础的一步，也是后续所有微调工作的前提。

#### 第二步：验证 Datasets 库

LoRA 微调通常需要处理数据集，因此 Datasets 库同样关键：

```python
import datasets
```

同样，没有报错即表示安装成功。

#### 第三步：验证 Accelerate 库

Accelerate 用于简化多 GPU/CPU 训练，在微调时也会用到。在终端中运行：

```bash
accelerate --version
```

能正常输出版本号即为正常。

#### 第四步：验证 Hugging Face 登录

LoRA 微调经常需要从 Hugging Face Hub 下载预训练模型。确认你的登录状态：

```bash
hf auth whoami
```

如果显示你的用户名，说明已登录。

---

### 关于 GPU 的关键提醒（重要）

以上四个库能正常导入，**只能说明软件环境就绪，并不代表 GPU 可用**。对于 LoRA 微调，GPU 通常是必需的。因此，请务必验证 PyTorch 的 GPU 状态：

```python
import torch
from transformers import AutoModel

print(torch.__version__)
print(torch.cuda.is_available())        # 期望 True（有可用 GPU 时）
print(torch.version.cuda)               # torch 编译时的 CUDA 版本
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# 确认模型确实加载到了 GPU
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
try:
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(next(model.parameters()).device)  # 期望 cuda:0（使用 GPU 时）
except OSError as e:
    print(f"模型加载失败：{e}")
```

**验证标准**：`is_available()` 为 `True`、`version.cuda` 与你的环境匹配、参数 device 为 `cuda:0`。

> 注意：`torch.cuda.is_available()=True` 只说明 PyTorch 能用 GPU，**不等于** bitsandbytes 4-bit 量化等更上层功能一定可用（后者在后续 LoRA 进阶内容中会涉及）。

---

### 本章环境验证总结

完成以上步骤

## 块 13

## 一、为什么需要 AutoModel 和 AutoTokenizer？

### 1.1 Transformers 加载模型的基本流程

想象一下，你要使用一个预训练好的大语言模型（比如 BERT、GPT 或 Llama）。模型本身只是一堆数字参数，它无法直接“读懂”你输入的文本。要让模型理解你的话，必须经过一个完整的流水线：**文本 → Tokenizer → 模型 → 输出**。

**Tokenizer 是第一步，也是必经之路。** 它的职责是把人类语言（一串字符）切分成模型能处理的“词元”（token），再把这些 token 映射成数字 ID。没有 Tokenizer，模型面对的就是一堆无法解析的原始字符串。

**AutoModel 则是“万能入口”。** 不同架构的模型（BERT 是编码器结构，GPT 是解码器结构，Llama、Qwen 各有差异）在代码层面有不同的类来对应。如果你每次都要记住“这个模型该用哪个类”，不仅繁琐，而且极易出错。AutoModel 的价值在于：**你只需要告诉它模型名称，它会自动识别模型结构并加载对应的类**。

**`from_pretrained()` 是加载动作的核心。** 无论是 AutoModel 还是 AutoTokenizer，都通过这个方法从 Hugging Face 模型库中下载并加载预训练权重和配置文件。它做了三件关键的事：

1. **下载或读取模型文件**——如果本地没有缓存，它会从模型库拉取；
2. **读取配置文件**——确定模型的层数、隐藏维度、注意力头数等结构参数；
3. **实例化模型对象**——根据配置自动搭建网络结构，并载入预训练权重。

**完整的加载流程可以概括为三步：**

```python
from transformers import AutoTokenizer, AutoModel

# 第一步：加载 tokenizer，负责文本 → token ID
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 第二步：加载模型，负责 token ID → 预测结果
model = AutoModel.from_pretrained("bert-base-uncased")

# 第三步：处理输入并推理
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```

这段代码展示了统一入口的威力：**换一个模型名称，其他代码几乎不用改**。比如把 `"bert-base-uncased"` 换成 `"gpt2"` 或 `"Qwen/Qwen2-7B"`，加载流程完全一致。这正是后续学习 LoRA 微调的基础——你只需要知道模型的名字，就能把它加载进来，然后在其上附加可训练的低秩适配器。

**注意：加载大模型时需关注显存占用。** 例如，Llama-2-7b 以 FP16 精度加载至少需要 14GB 显存，部署前请确认硬件条件。

**理解这个流程的关键在于：Tokenizer 负责“翻译”语言，AutoModel 负责“识别”架构，`from_pretrained()` 负责“搬运”权重。三者配合，你就能用统一的代码风格驾驭任何主流模型。**

---

**本块学习检查**（基于参考资料①③，不引入新事实）：

请用自己的话回答两个问题，检验你是否理解了加载流程的核心逻辑：

1. 为什么加载模型时必须同时加载 Tokenizer？如果只加载模型而不加载 Tokenizer，输入文本会怎样？
2. 使用 `AutoModel.from_pretrained("bert-base-uncased")` 时，AutoModel 是如何知道该用哪种具体模型类的？`from_pret

## 块 14

## 一、为什么需要 AutoModel 和 AutoTokenizer？

### 1.1 Transformers加载模型的基本流程

大语言模型**不能直接理解文本**。你输入“你好，大模型”，模型实际接收的是一串数字——每个 token 对应一个整数 ID（具体数值取决于分词器，不同模型结果不同）。

完整流程如下：

```
文本 → Tokenizer分词 → Token ID数字 → Transformer模型计算 → 输出结果 → Tokenizer解码 → 文本
```

两个核心组件的分工：

- **Tokenizer**：负责“文字 ↔ 数字”的双向转换；
- **Model**：负责“数字 → 神经网络计算”。

Hugging Face Transformers 提供了**统一加载入口**——`AutoModel` / `AutoTokenizer`，让你用相似的代码就能加载不同模型的权重和分词器。

**但请注意**：“能用同一套代码加载”**不等于**“能用同一套推理参数和对话方式使用”。不同模型（尤其 Llama、Qwen 等 decoder-only 对话模型）还涉及各自的 **chat template**（`tokenizer.apply_chat_template`）、`pad_token` / `eos_token` 设置、是否为 **gated/授权模型**（需在模型页接受许可）、以及 `trust_remote_code=True` 会执行仓库自带代码（需确认来源可信）等差异。本章主要用 BERT / GPT-2 演示通用的“加载—编码—前向/生成”流程，对话模型的完整用法见后续章节。

> **显存提示**：加载大模型时需注意显存占用，例如 Llama-2-7b FP16 需要至少 14GB 显存。

---

**本块学习检查**：

## 块 15

## 为什么不用具体模型类？

直接加载模型时，不同架构要记不同的类名：

| 模型      | 对应类          |
| ------- | ------------ |
| BERT    | BertModel    |
| GPT-2   | GPT2Model    |
| RoBERTa | RobertaModel |

比如加载BERT要写 `BertModel.from_pretrained("bert-base-uncased")`，换成GPT-2就得改成 `GPT2Model.from_pretrained("gpt2")`。类名记不住不说，项目里想切换模型，所有加载代码都得跟着改，非常麻烦。

Hugging Face 为此提供了 **AutoModel** 和 **AutoTokenizer**：它们会自动根据模型配置选择正确的类型，你只需要指定模型名称，不用关心具体类名。

# 模型许可证：以 Llama 2 Community License 为例（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16

## 块 16

**Tokenizer是什么？**

Tokenizer 是文本与模型之间的“翻译官”。它负责四件事：把文本切分成片段、给每个片段映射一个编号、添加特殊标记（如句首句尾符）、最终生成模型能直接读取的输入格式。

举个例子，输入 `Hello AI`，经过 BERT 的 Tokenizer 处理后，会变成一串数字 ID（不同模型如 GPT-2 和 BERT 的切分与编号都不同）：

```python
[101, 7592, 9932, 102]   # BERT 示例：[CLS], 'hello', 'ai', [SEP]
```

因为模型只能处理数字，所以 Tokenizer 是任何 NLP 任务的第一步。

**如何加载 Tokenizer？**

使用 `AutoTokenizer` 一行代码即可：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 模型许可证：Apache 2.0
# 注意：BERT-base 约需 1.2GB 显存（FP32）
```

这里的 `checkpoint` 指定模型名称。Transformers 库会自动完成：访问 Hugging Face Hub → 下载分词器配置 →

## 块 17

**目标：** 快速掌握用 `tokenizer` 将文本转为模型可读的数字格式。

**核心操作：** 直接调用分词器处理文本，它会自动完成切词和转ID。

```python
text = "Hello, Hugging Face"
inputs = tokenizer(text)
print(inputs)
```

**输出解读（以 `bert-base-uncased` 为例）：**
输出是一个字典，包含两个关键字段：

- **`input_ids`**：文本转换后的数字列表。其中 `101` 和 `102` 是 BERT 特有的 `[CLS]` 和 `[SEP]` 标记，中间的数字是每个词的编号。
- **`attention_mask`**：全是 `1`，告诉模型这些位置都是有效内容，需要关注。

**学习检查：** 请观察输出，确认 `input_ids` 的长度是否与 `

## 块 18

**本块学习检查**：请阅读下方代码，指出 `AutoModel` 是根据什么信息来自动选择模型结构的？

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
```

---

`AutoModel` 是 Transformers 提供的自动模型加载接口。它的核心作用是：根据你提供的**模型名称**和**配置文件**，自动选择对应的模型结构。

具体来说，当你调用 `AutoModel.from_pretrained("bert-base-uncased")` 时，内部会自动读取该模型的 `config.json` 文件，识别其中的 `model_type` 字段（例如 `bert`），然后据此创建相应的模型实例（如 `BertModel`）。这一机制被称为 Auto Classes，它能根据模型名称和配置自动选择对应的模型类。

# 模型许可证：BERT 模型采用 Apache 2.0 许可证。
# 注意：BERT-base 模型 FP16 推理需要至少 2GB 显存。

## 块 19

**本块学习检查**：对照你的电脑配置，判断能否运行7B模型推理，并说明依据。

你的目标是LoRA微调，先确认硬件门槛。本块只讲配置要求，不涉及代码。

**模型许可证**：Llama 2 Community License（商用月活>7亿需Meta授权）。

**最低配置**（能跑通流程）：CPU为Intel i5或AMD同级以上，内存8GB，磁盘空闲20GB。AutoModel加载API本身用CPU就能运行。

**推荐配置**（流畅微调）：内存16GB+，NVIDIA GPU且支持CUDA。显存方面：小模型测试需4GB+，加载7B模型需12GB+。

**关键结论**：大模型推理速度

## 块 20

**4.1 加载模型**

针对LoRA微调目标，先加载预训练模型及其分词器。完整代码如下：

```python
from transformers import AutoTokenizer, AutoModel

checkpoint = "bert-base-uncased"
# 模型许可证：Apache 2.0
# 注意：BERT-base FP16 需要至少 1.2GB 显存

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
```

**分工明确**：`tokenizer`负责将文本转为模型可读的输入格式；`model`负责神经网络计算。两者通过同一`checkpoint`保持配置一致。

**学习检查**：请解释为何`tokenizer`和`model`必须使用相同的`checkpoint`？

## 块 21

## 4.2 输入文本：把文字变成模型能读的格式

你已经掌握了AutoModel和AutoTokenizer的基本加载方式，现在进入完整流程的核心环节——**喂数据**。因为你的目标是LoRA微调，这一步是后续所有操作的地基。

```python
text = "Hugging Face is amazing"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
```

关键参数就一个：`return_tensors="pt"`，它的作用是让tokenizer返回**PyTorch张量**格式（而不是Python列表）。输出结果包含两个键：

- `input_ids`：文本转换后的token编号序列
- `attention_mask`：注意力掩码，标记哪些位置是真实内容

## 4.3 输入模型：一行代码搞定

```python
outputs = model(**inputs)
```

这里的`**`是Python的**字典解包语法**——把`inputs`这个字典的键值对拆开，作为独立参数传给模型。上面这行代码等价于：

```python
model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
```

两种写法效果完全一样，但`**inputs`更简洁，实际项目中几乎都用这种写法。

## 4.4 查看模型输出

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存。
```

## 块 22

## 5.1 为什么不用 AutoModel？

你已经掌握 PyTorch 基础，这里用一句话帮你理清关键区别：

**`AutoModel` 返回的是隐藏状态（hidden states），而不是文本。**

它适合两类场景：

- 特征提取（比如把句子转成向量）
- 下游任务开发（你自己在主干上接自定义结构）

但聊天模型要的是**生成文本**，不是向量。所以实际微调 LoRA 时，你面对的是 GPT、Llama 这类生成模型，必须用带“因果语言建模”任务头的类：

```
AutoModelForCausalLM
```

---

## 5.2 加载 GPT 类模型

看一个最小示例，加载 GPT-2：

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
```

注意：这里 tokenizer 和 model 用的是**同一个 checkpoint 名称**，这是 HuggingFace 生态的标准做法。

---

## 各类 Auto 类怎么选？

| 类 | 用途 |
|---|---|
| `AutoModel` | 基础 Transformer，输出隐藏状态 |
| `AutoModelForSequenceClassification` | 分类 |
| `AutoModelForCausalLM` | 文本生成 |
| `AutoModelForSeq2SeqLM` | 翻译/摘要 |

**核心判断标准：**

- `AutoModel` 只加载主干，不包含特定任务头。如果你用 `AutoModelForSequenceClassification` 保存模型，再用 `AutoModel` 加载，分类头不会加载，因为类不匹配。
- `AutoModelFor*` 在主干上附加了对应任务头。做分类、生成、摘要等任务时，**必须用与训练时一致的任务类**，否则任务头语义丢失。

对你做 LoRA 微调而言：目标模型是生成式对话模型，所以 `AutoModelForCausalLM` 是正确入口。

---

## 一个提醒

本章示例用 BERT / GPT-2 演示“加载—编码—前向”的通用流程。但 Llama、Qwen 等 decoder-only 对话模型，实际推理还涉及 chat template、`pad_token` 设置、gated 模型许可、`trust_remote_code=True` 等，这些属于后续章节，这里不展开。**会加载 BERT 不等于会正确推理任意生成式大模型**，记住这一点即可。

## 块 23

## 5.3 decoder-only 小模型完整生成示例（GPT-2）

前面 BERT 示例只做到“前向计算得到隐藏状态”，并没有真正**生成文本**。这里用小巧的 decoder-only 模型 GPT-2 演示一个**可运行的完整生成流程**，为后续 LoRA 微调路线做好衔接。

### 完整代码

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

### 要点说明

**1. 模型类选择**：生成模型用 `AutoModelForCausalLM`（带语言模型头），不是 `AutoModel`。这是因果语言模型与普通编码器模型的本质区别。

**2. pad_token 设置**：decoder-only 模型常需显式设置 `pad_token`（GPT-2 默认没有），这里用 `eos_token` 兼作 `pad_token`。这是新手最容易遗漏的坑。

**3. 生成方式**：用 `model.generate(...)` 而非直接前向传播。`max_new_tokens=30` 控制新生成的最大长度，注意不是总长度。

**4. 采样参数**：`do_sample=True` 开启随机采样；`temperature=0.8` 控制随机性（越低越确定）；`top_p=0.9` 做核采样截断。这些参数不是“越大越好”，需按任务调。

**5. 对话模型差异**：真实对话模型（Llama/Qwen 等）在此基础上，还需先用 `tokenizer.apply_chat_template(messages, ...)` 把多轮消息拼成模型期望的格式，再交给 `generate`——这属于后续章节内容。

> **提示**：GPT-2 是英文小模型，输出质量有限，这里只用于演示“decoder-only 生成”的最小可运行链路，不代表真实对话效果。

### 本块学习检查

请回答：为什么生成任务必须用 `AutoModelForCausalLM` 而不是 `AutoModel`？如果 GPT-2 不设置 `pad_token

## 块 24

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModel.from_pretrained(checkpoint, device_map="auto")
```

**关键区分**：这针对的是**推理/加载时**的跨设备切分，尤其适合单卡放不下的场景。它**不是**训练时的标准设备方案——真正的训练（尤其多卡）由 `accelerate`、`Trainer`、DeepSpeed 负责设备与并行策略，别把 `device_map="auto"` 直接当训练配置。

**本块学习检查**：对照你的 LoRA 微调目标，回答：`device_map="auto"` 能否替代 `Trainer`

## 块 25

**本块学习检查**：阅读下面代码，回答 `torch_dtype="auto"` 的作用是什么？

```python
model = AutoModel.from_pretrained(checkpoint, torch_dtype="auto")
```

**正文**：

`torch_dtype` 控制模型加载时的精度。传 `"auto"`，会按 checkpoint 配置里记录的 dtype（`config.torch_dtype`）加载权重，可能是 `float32`、`float16` 或 `bfloat16`。注意：它**不是**按显卡自动选精度。

若 checkpoint 保存为半精度（fp16/bf16），用 `"auto"` 比默认 float32 更省显存。想强制指定，可直接传 `torch

# 模型许可证：Ll

## 块 26

## 6.3 local_files_only：离线加载的“保险开关”

当你准备做 LoRA 微调时，模型加载是第一步。`local_files_only=True` 这个参数的作用很纯粹：**只读取本地缓存，绝不联网**。

```python
model = AutoModel.from_pretrained(
    checkpoint,
    local_files_only=True
)
```

**什么时候用？** 两种典型场景：

- **离线服务器**：机器本身不连外网，无法访问 HuggingFace Hub；
- **无网络环境**：比如内网开发机、临时断网的情况。

**对你意味着什么？** 如果你打算在本地做 LoRA 微调，建议提前把需要的模型下载好（联网时先跑一次 `from_pretrained` 让模型进入缓存），之后训练阶段加上 `local_files_only=True`，可以避免训练中途因网络波动导致加载失败。

---

## 七、从本地路径加载模型：不止是“名字”，也可以是“路径”

前面用的 `"bert-base-uncased"` 是 HuggingFace Hub 上的模型名称。但 `from_pretrained` 同样接受**本地目录路径**：

```python
"./my_model"
```

一个典型的本地模型目录长这样：

```
my_model
├── config.json
├── model.safetensors
├── tokenizer.json
```

> **注意**：上面是**小模型**的典型示例。较大的模型权重常被**分片保存**，目录里会是多个分片文件加一个索引，例如：
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

加载代码：

```python
tokenizer = AutoTokenizer.from_pretrained(
    "./my_model"
)

model = AutoModel.from_pretrained(
    "./my_model"
)
```

**对你意味着什么？** 做 LoRA 微调时，你往往需要先下载一个基础模型（比如某个 7B 模型），然后基于它做参数高效微调。这个基础模型就可以放在本地目录里，用路径方式加载——既稳定又可控。

> **模型许可证**：部署或微调前请确认模型许可证类型。例如，Llama 2 Community License 允许商用，但月活用户超过 7 亿时需 Meta 授权。
>
> **显存要求**：加载大模型前请确认显存是否充足。例如，Ll

## 块 27

## 6.3 `local_files_only`：让加载过程“断网”也能跑

你已经掌握了`from_pretrained()`的基本用法——它会自动下载模型权重。但有个实际问题：如果网络不稳定，或者你已经下载过模型，想完全离线使用，该怎么办？

这时就需要`local_files_only`参数。它的作用很简单：**只从本地加载文件，不发起任何网络请求**。

### 用法示例

```python
from transformers import AutoTokenizer, AutoModel

checkpoint = "bert-base-uncased"

# 只从本地加载，不联网
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    checkpoint,
    local_files_only=True
)
```

### 关键理解

- 设置`local_files_only=True`后，Hugging Face会**强制**只检查本地缓存目录。
- 如果本地**没有**对应的模型文件，程序会**直接报错**，而不是尝试联网下载。
- 如果本地**已有**该模型（比如之前运行过不带此参数的代码），加载会正常完成，效果与联网加载完全一致。

### 典型使用场景

1. **离线环境**：服务器无法访问外网，但模型已提前下载到本地。
2. **避免意外联网**：确保代码运行时不触发下载，节省流量、提高速度。
3. **调试问题**：排查“加载的到底是本地还是远程文件”这类疑惑。

### 学习检查

请思考：如果你第一次运行`local_files_only=True`加载一个**从未下载过**的模型，会发生什么？为什么？

（答案：会报错。因为该参数禁止联网下载，而本地又没有缓存文件，加载无法完成。）

> **部署说明**：本示例使用`bert-base-uncased`模型，其许可证为Apache 2.0。该模型体积较小（约440MB），在CPU或GPU上均可运行，显存需求较低（FP

## 测评题目

### Q-KB11-ST1100-001
Hugging Face 数据集加载和处理常使用的库名称是________。

### Q-KB11-ST1100-003
以下哪个库最直接用于训练配置中的分布式加速？

### Q-KB1101-003
请写代码：使用 GPT-2 模型和分词器，对输入文本 `"Hello, I am"` 生成接下来的 20 个词（设置 `max_new_tokens=20`），并解码输出。

### Q-KB1101-001
请写出使用 `AutoModelForCausalLM` 加载 `gpt2` 模型的代码。

### Q-KB11-ST1102-001
datasets 的 `map()` 常用于对数据集样本执行批量________。

### Q-KB11-ST1100-002
使用私有 Hugging Face 仓库时，通常需要配置访问________。

### Q-KB11-ST1102-002
datasets 的 `filter()` 会保留使筛选函数返回________的样本。

### Q-KB11-ST1102-003
为了在每个训练轮次打乱样本顺序，通常使用？
