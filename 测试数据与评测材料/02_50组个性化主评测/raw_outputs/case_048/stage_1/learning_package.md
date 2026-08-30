# 学习包

## 块 1

## 环境准备：在 Python 环境中安装 transformers、tokenizers 库

> **适合人群**：你已完成"深度学习框架PyTorch基础"板块，在 Ubuntu 24.04 上成功配置过 `pytorch` conda 环境并安装 PyTorch 2.5.1。本节目标是在现有 PyTorch 环境中成功安装 `transformers` 和 `tokenizers` 库，并验证能正常加载预训练模型，为后续学习 BPE/WordPiece 算法和 HuggingFace Tokenizer 使用做好准备。
>
> **前置说明**：`transformers` 是 HuggingFace 提供的核心库，封装了大量预训练模型（如 BERT、GPT-2、Llama、Qwen 等）；`tokenizers` 是其底层的高性能分词库，基于 Rust 实现。本节所有命令请在前面配置好的 `pytorch` conda 环境中执行。

### 本块学习检查

在继续之前，请先用一个最小示例快速回顾前置知识：确认你能在终端中完成 **conda 虚拟环境的创建与激活**（例如 `conda create -n pytorch python=3.10` 后 `conda activate pytorch`），并确认 `pip` 能正常使用（可尝试 `pip --version`）。如果你已经配置过 pip 国内镜像源（如清华源），请确认其生效。能解释这些关键写法后再继续本块。

---

## 一、前置检查：确认 PyTorch 环境就绪

以下步骤在前面板块已经详细学过，本节只做**快速确认**。如果某一步报错，请回到"环境准备：根据 CUDA 版本安装 PyTorch"章节排查。

### 1.1 确认 conda 环境和 PyTorch 版本

```bash
conda activate pytorch

python -c "import torch; print('PyTorch:', torch.__version__)"
# 预期输出类似如下，实际以版本为准：PyTorch: 2.5.1
```

> **版本要求**：本课程在 Python 3.10 + PyTorch 2.5.1 环境下验证通过。不同版本的 transformers 对 Python 和 PyTorch 版本要求可能不同，建议以安装时的官方文档为准。以下是本课程验证过的依赖版本示例（`requirements.txt`）：
> ```text
> torch==2.5.1
> transformers>=4.45.0
> tokenizers>=0.20.0
> ```
> 如果你按照前面板块完成了环境搭建，当前环境已满足上述要求。

### 1.2 确认 pip 可用

```bash
pip --version
# 预期输出类似如下，实际以版本为准：pip 24.x.x
```

---

## 二、安装 transformers 库

确认 PyTorch 环境就绪后，接下来安装 `transformers` 库。安装命令如下：

```bash
pip install transformers
```

安装完成后，可以验证安装是否成功。以下提供两种验证方式：

**方式一（在线验证）**：以下命令会下载模型，需确保网络畅通：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('We love you'))"
```

如果输出包含正面情感分析结果（如 `[{'label': 'POSITIVE', 'score': 0.999...}]`），说明 `transformers` 已正确安装并成功加载了预训练模型。

**方式二（离线验证）**：若网络受限，可改用以下命令验证本地加载：

```bash
python -c "from transformers import AutoModel; print('

## 块 2

### 标准安装（推荐）

既然你已具备 PyTorch 基础，安装 `transformers` 只需一条命令。打开终端，确保当前处于你配置好的 PyTorch 环境中，执行：

```bash
pip install transformers
```

该命令会联网下载安装包（体积通常在数百 MB 量级），对电脑配置无特殊要求，CPU 即可完成。若网络较慢，可改用清华镜像源加速：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**本块学习检查**：请确认你当前终端激活的是 PyTorch 环境，然后执行上述标准安装命令，观察安装过程是否顺利完成。若遇到网络超时，改用镜像源命令重

## 块 3

### 带 PyTorch 依赖的安装（可选）

考虑到你的目标是 LoRA 微调，而 PyTorch 是运行 transformers 模型的基础，这里解释两种安装方式。

**方式一：pip 安装（推荐）**

如果你在一个**全新的空环境**中（即尚未安装 PyTorch），可以用一条命令同时安装 transformers 及其 PyTorch 适配依赖：

```bash
pip install 'transformers[torch]'
```

> **本课程情况**：PyTorch 已提前装好，因此你只需直接执行 `pip install transformers` 即可，**无需**重复安装 torch，避免版本冲突。

**方式二：conda 安装（可选）**

若你习惯使用 conda，可从 conda-forge 通道安装：

```bash
conda install conda-forge::transformers
```

> **注意**：conda-forge 通道的 transformers 版本更新可能稍慢于

## 块 4

**本块学习检查**：运行下面的命令，确认 `tokenizers` 是否已随 `transformers` 自动安装成功。

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
```

如果看到类似 `tokenizers: 0.x.x` 的输出（版本号以实际为准），说明安装正常，无需额外操作。

---

### 3.1 是否需要单独安装？

**通常不需要。** `tokenizers` 是 `transformers` 的依赖项，安装 `transformers` 时会自动装上兼容版本。你只需通过上面的命令验证即可。

### 3.2 什么情况下需要手动安装？

只有两种场景才需要手动安装：

1. **独立使用** `tokenizers`（例如自定义训练词表）；
2. 自动安装的版本不满足你的需求。

此时执行：

```bash
pip install tokenizers
```

> **踩坑提醒**：某些旧版本在特定 Python 环境下可能需从源码编译，依赖 Rust 工具链。但本课程使用 Python 3.10 + 较新版 `tokenizers`，通常有预编译 wheel 包，直接安装即可，无需 Rust。

## 块 5

## 4.1 快速验证脚本（①）

既然你的目标是 LoRA 微调，第一步是确认 HuggingFace 工具链能跑通。在终端执行以下命令：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

这条命令做了三件事：加载 `pipeline`、调用情感分析任务、对输入文本 `'we love you'` 给出判断。预期输出类似（实际以版本为准）：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

**本地运行配置要求**：首次运行会自动从 HuggingFace Hub 下载预训练模型（如 `distilbert-base-uncased-finetuned-sst-2-english`，体积通常在数百 MB 量级）到本地缓存。需要**联网**，磁盘建议预留 1GB 以上空间。

## 块 6

**查看安装版本：确认你的环境就绪**

在完成 transformers 和 tokenizers 的安装后，第一步是验证它们是否被 Python 正确识别。运行以下命令，它会打印出两个库的版本号：

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

**如何理解输出**：如果命令成功执行并显示类似 `transformers 版本: 4.x.x` 和 `tokenizers 版本: 0.x.x` 的数字，说明安装无误，可以继续后续的模型加载操作。如果出现 `ModuleNotFoundError`，则需回到安装步骤排查。

**本块学习检查

## 块 7

## 5.1 通过环境变量配置缓存目录

在开始LoRA微调之前，先确保模型文件有合适的存放位置。模型体积可达数GB至数十GB，若使用云服务器，建议将缓存设置到数据盘（如`/mnt/data`），避免系统盘被占满。

**临时生效**（仅当前终端会话有效）：

```bash
export HF_HOME="/mnt/data/huggingface"
```

**永久生效**（写入配置文件，每次登录自动加载）：

```bash
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

**优先级顺序**（由高到低）：`HF_HUB_CACHE` > `HF_HOME` > `XDG_CACHE_HOME` + `/huggingface`。即若设置了`HF_HUB_CACHE`，则优先使用它指定的路径。

## 5.2 查看当前缓存路径

在Python中运行以下代码，确认实际生效的缓存位置：

```python
from transformers import utils
print(utils.default_cache_path)
```

**本块学习检查**：执行上述Python代码，观察输出路径是否与你设置的`HF_HOME`一致。若不一致，检查环境

## 块 8

### 6.1 预先下载模型（①）

在开始LoRA微调之前，你需要先确保模型文件已经完整地保存在本地。这样后续的微调过程就不会因为网络问题而中断。

**操作步骤**：使用`snapshot_download`函数，将整个模型仓库下载到当前目录下的`bert-base-chinese`文件夹中。

```python
from huggingface_hub import snapshot_download

try:
    # 下载整个模型仓库到指定目录
    snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
except Exception as e:
    print(f"模型下载失败，请检查网络连接后重试。错误详情：{e}")
```

> **理解要点**：`repo_id`是模型在HuggingFace Hub上的唯一标识，`local_dir`指定了本地存储路径。下载完成后，该目录下会包含模型权重、配置文件等所有必要文件。**模型许可证**：bert-base-chinese 采用 Apache 2.0 许可证，可自由商用。

---

### 6.2 离线加载（①）

模型下载到本地后，你可以通过以下两种方式确保代码**只使用本地文件**，完全脱离网络。

**方式一：设置环境变量（推荐用于脚本运行）**

在命令行中，运行Python脚本前设置`HF_HUB_OFFLINE=1`，这会阻止所有来自HuggingFace Hub的HTTP请求。

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

**方式二：代码内指定（推荐用于交互式开发）**

在`from_pretrained`方法中，将第一个参数改为本地路径，并设置`local_files_only=True`，强制只加载本地文件。

```python
from transformers import AutoModel

try:
    model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
except OSError as e:
    print(f"本地模型加载失败，请检查模型文件是否完整。错误详情：{e}")
```

> **两者的区别**：方式一是全局性的，对整个脚本生效；方式二更精细，只对当前加载操作生效。在你的LoRA微调脚本中，建议使用方式二，这样更灵活。**显存要求**：bert-base-chinese 模型 FP16 精度下运行约需 2GB 显存。

---

### 常见问题排查（①+②+③+④）

在环境准备过程中，你可能会遇到以下问题。请对照表格快速定位并解决：

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `pip install transformers` 速度极慢或超时 | 默认源在国外 | 使用清华镜像源：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found` | Python 版本过低或过高 | 确保 Python 3.10+（本课程使用 3.10） |
| 导入 transformers 时报 `signal` 模块错误 | transformers 版本与 Python 不兼容 | 升级 transformers 到最新版：`pip install -U transformers` |
| `tokenizers` 安装时提示需要 Rust 编译器 | 该版本无预编译 wheel，需从源码构建 | 换用 conda 安装：`conda install -c huggingface tokenizers`；或升级 pip 以获取预编译版本 |

## 块 9

**下一步预告**：下一节将深入讲解子词切分算法（BPE、WordPiece、SentencePiece），理解为何 `"unhappiness"` 被切为 `["un", "happiness"]` 而非按单词或字母切分，以及算法如何从语料库统计学习最优词表。随后，我们将用已装好的 `transformers` 库加载真实预训练分词器（如 GPT-2 的 BPE），亲手观察切分结果。

**本块学习检查**：确认你的 Python 环境中 `transformers` 和 `tokenizers` 已成功安装（可尝试导入库，无报错即就绪）。若未完成，请先返回安装步骤。

现在，确保环境就绪，

## 测评题目

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB07-ST0700-001
Hugging Face 模型与分词器常用的 Python 库名称是________。

### Q-KB0702-004
在 Hugging Face Tokenizer 的输出中，`attention_mask` 的作用是什么？

### Q-KB0701-001
Byte Pair Encoding (BPE) 算法的核心思想是？

### Q-KB0701-003
SentencePiece 模型将输入文本视为原始的字符串序列，无需预先进行空格分词，可以直接从原始句子中学习子词单元。（ ）

### Q-KB0703-001
当设置 `truncation=True` 时，如果输入序列长度超过模型的最大长度，分词器会如何处理？

### Q-KB0703-002
`padding='max_length'` 参数会将所有序列填充或截断到由________参数指定的长度。

### Q-KB0701-004
以下关于子词切分（Subword Tokenization）的优点，描述**错误**的是？
