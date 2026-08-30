# 学习包

## 块 1

## 环境准备：安装 transformers 与 tokenizers

**本块定位**：你已完成 PyTorch 环境搭建，现在要做的就是在同一个 conda 环境里装上 HuggingFace 的两件套——`transformers`（模型库）和 `tokenizers`（分词库）。这是你后续做 LoRA 微调的基础设施，没有它们，微调无从谈起。

### 一、前置检查：确认环境就绪

你的 RTX 4060 8GB 显存跑 LoRA 微调是够用的，但前提是 PyTorch 环境必须正常。先花 1 分钟快速确认：

```bash
conda activate pytorch

python -c "import torch; print('PyTorch:', torch.__version__)"
# 预期输出类似：PyTorch: 2.5.x（以实际版本为准）
```

**版本要求**：本课程在 Python 3.10 + PyTorch 2.5.1 下验证通过，对应验证版本为 `transformers==4.45.0`、`tokenizers==0.20.0`。不同版本的 transformers 对 Python 和 PyTorch 版本要求不同，以官方文档为准。本课程验证过的依赖版本：

```
torch==2.5.1
transformers==4.45.0
tokenizers==0.20.0
```

如果你按前面板块完成了搭建，当前环境已满足要求。

再确认 pip 可用：

```bash
pip --version
# 预期输出类似：pip 24.x.x
```

### 二、安装 transformers 库

确认无误后，直接安装：

```bash
pip install transformers
```

安装完成后，验证是否能正常加载预训练模型（这是本块的关键验证步骤）：

```bash
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('bert-base-uncased'); print('模型加载成功')"
```

> **注意**：首次加载会下载模型文件到本地缓存，需要网络连接。如果这一步成功，说明 transformers 安装正确且能正常调用 PyTorch 后端。
>
> **模型许可证**：`bert-base-uncased` 采用 Apache 2.0 许可证，可自由使用（含商用）。若后续替换为其他模型（如 Llama 系列），需另行确认对应许可证条款（如 Llama 2 Community License，商用月活>7亿需 Meta 授权）。

### 三、安装 tokenizers 库

`tokenizers` 是 transformers 的底层分词库，基于 Rust 实现，通常随 transformers 自动安装。单独确认一下：

```bash
pip install tokenizers

python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
```

### 常见问题速查

- **版本冲突**：如果安装时报依赖冲突，优先检查 PyTorch 版本是否满足 transformers 的要求，必要时按官方文档调整版本。
- **Python 版本兼容性**：不同 Python 版本对 transformers 的支持不同，本课程在 Python 3.10 下验证

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的PyTorch环境已就绪，现在安装`transformers`库——这是后续加载模型、执行LoRA微调的核心工具。在终端运行：

```bash
pip install transformers
```

该命令会联网下载安装包（数百MB），你的RTX 4060 8GB完全满足要求，CPU也能执行。若网络慢，用清华镜像加速：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若安装失败，请检查网络连接或使用镜像源。

安装完成后，即可在Python中加载模型

## 块 3

### 带 PyTorch 依赖的安装（可选）

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

若你在全新空环境中安装，可用一条命令同时装好 transformers 和 PyTorch 适配依赖：

```bash
pip install "transformers[torch]"
```

**注意**：本课程 PyTorch 已提前装好，直接 `pip install transformers` 即可，无需重复安装 torch。

**实践任务**：在终端输入 `pip list | grep transformers`，确认 transformers 已安装且版本存在。若未安装，执行上述命令完成安装。若使用 Windows PowerShell，请

## 块 4

## 3.1 是否需要单独安装？

`tokenizers` 是 `transformers` 的**依赖项**，安装 `transformers` 时会**自动安装**兼容版本，一般无需手动单独安装。你可以运行以下命令验证：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
# 预期输出类似：tokenizers: 0.x.x（实际以版本为准）
```

## 3.2 什么情况下需要手动安装？

如果你需要**独立使用** `tokenizers`（例如自定义训练词表），或自动安装的版本不满足需求，可手动指定版本：

```bash
pip install tokenizers
```

> **踩坑提醒**：某些旧版本 tokenizers 在特定 Python 环境下可能需要从源码编译，依赖 Rust 工具链。但本课程使用 Python 3.10 + 较新版 tokenizers 通常有预编译 wheel 包，可直接安装，无需 Rust。

---

## 四、验证安装：运行你的第一个 HuggingFace 模型

安装完成后必须验证 `transformers` 能否正常工作。最简单的方法是使用 `pipeline` 加载一个情感分析模型进行推理。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

---

**本块学习检查

## 块 5

**动手验证：跑通你的第一个 HuggingFace 模型**

在终端里直接执行下面这行命令，它会自动加载一个情感分析模型，并判断“we love you”这句话的情绪：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

预期会看到类似这样的输出（具体数值因版本略有差异）：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

**运行前须知**：首次执行时，系统会自动联网下载预训练模型（如 `distilbert-base-uncased-finetuned-sst-2-english`，约数百 MB）到本地缓存。请确保网络畅通，磁盘预留 1GB 以上空间。**CPU 即可运行，无需 GPU**，内存建议 ≥ 4GB。模型许可证：Apache 2.0。

**本块学习检查**：完成本块

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

运行以下命令，确认环境已就绪：

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存

看到两个版本号输出，说明安装成功。这为后续加载模型（如LoRA微调所需的预训练模型）打好基础

## 块 7

## 5.1 通过环境变量配置缓存路径

你的 LoRA 微调目标中，模型文件动辄数 GB，而你的 RTX 4060 只有 8GB 显存——**显存装不下模型，但硬盘可以**。所以先学会把缓存路径指到合适位置。

**临时生效**（仅当前终端）：
```bash
export HF_HOME="/mnt/data/huggingface"
```

**永久生效**（写入配置，重启终端仍有效）：
```bash
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

**优先级**（同时设置多个时，高优先级覆盖低优先级）：
1. `HF_HUB_CACHE`（最优先）
2. `HF_HOME`
3. `XDG_CACHE_HOME` + `/huggingface`

> **建议**：若用云服务器，把 `HF_HOME` 指向数据盘（如 `/mnt/data`），避免系统盘被模型占满——后续微调和部署时模型可达数 GB 至数十 GB。

## 5.2 查看当前缓存路径

```python
from transformers import utils
print(utils.default_cache_path)
```

**本块学习检查**：完成本块后，

## 块 8

## 6.1 预先下载模型：为离线环境做准备

你的目标是 LoRA 微调，这意味着后续会反复加载同一个预训练模型。如果每次都从 HuggingFace Hub 在线拉取，不仅慢，而且一旦网络波动就会中断。更稳妥的做法是：**先把模型完整下载到本地，之后所有操作都在本地进行**。

使用 `snapshot_download` 可以一次性下载整个模型仓库（包括配置文件、权重、词表等）：

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
```

执行后，`bert-base-chinese` 的全部文件会保存到你当前目录下的 `./bert-base-chinese` 文件夹中。这一步只需做一次，后续微调时直接指向这个本地路径即可。

> **模型许可证**：使用前请确认所选模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需 Meta 授权），确保符合使用场景。

## 6.2 离线加载：让代码只读本地文件

模型下载到本地后，有两种方式确保代码不再发起网络请求：

**方式一：环境变量全局拦截**

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

设置 `HF_HUB_OFFLINE=1` 后，HuggingFace 生态的所有组件（包括 `transformers`）都会默认禁止访问网络，只从本地缓存或指定路径读取文件。

**方式二：代码内指定本地模式**

```python
from transformers import AutoModel

# 注意：加载模型前请确认本地路径存在，且显存满足模型需求（如 Llama-2-7b FP16 需至少 14GB 显存）
model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
```

`local_files_only=True` 明确告诉 `from_pretrained`：只加载本地文件，不要尝试联网下载。如果本地路径不存在或文件不完整，会直接报错而不是卡住等待网络超时。

> **实践建议**：在 LoRA 微调脚本中，优先使用方式二。它更精确——只对模型加载生效，不影响其他可能需要的网络操作。

---

## 本块学习检查

完成本块后，请用自己的话说明：**预先下载模型与离线加载，和你当前“LoRA微调”的学习目标之间是什么关系？** 具体想清楚两个问题：

1. 如果跳过“预先下载”这一步，直接在线加载模型进行微调，会面临哪些风险？

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

环境已就绪，我们即将进入下一节，深入讲解现代大语言模型使用的子词切分算法（如BPE、WordPiece、SentencePiece）。你会了解为什么 "unhappiness" 会被切分成 `["un", "happiness"]` 而不是按单词或字母切分，以及这些算法如何用统计方法从语料库中"学习"出最优的词表。

随后，我们将用今天装好的 `transformers` 库加载真实的预训练分词器（如 GPT-2 的 BPE 分词器），亲手观察切分

## 测评题目

### Q-KB07-ST0700-003
安装 Hugging Face Transformers 的常用命令是？

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB0703-002
`padding='max_length'` 参数会将所有序列填充或截断到由________参数指定的长度。

### Q-KB0703-001
当设置 `truncation=True` 时，如果输入序列长度超过模型的最大长度，分词器会如何处理？

### Q-KB0701-001
Byte Pair Encoding (BPE) 算法的核心思想是？

### Q-KB0702-001
请写出使用 Hugging Face `transformers` 库加载名为 `bert-base-uncased` 的分词器的代码。

### Q-KB0701-003
SentencePiece 模型将输入文本视为原始的字符串序列，无需预先进行空格分词，可以直接从原始句子中学习子词单元。（ ）

### Q-KB0702-005
请写代码：加载 `gpt2` 分词器，并对文本 `"Hello, world!"` 进行分词，要求返回 PyTorch 张量，并打印出 `input_ids` 和 `attention_mask`。
