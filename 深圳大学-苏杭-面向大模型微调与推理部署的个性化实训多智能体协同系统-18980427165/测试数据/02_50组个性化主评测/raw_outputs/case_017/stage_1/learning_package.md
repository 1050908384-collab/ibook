# 学习包

## 块 1

## 本块学习检查

在开始本块之前，请先完成一个最小示例回顾，确认环境准备的基础已牢固：

1. 确认你已安装 Python 3.10 或更高版本（可在终端运行 `python --version` 查看）。
2. 确认你已创建并激活 conda 虚拟环境（例如名为 `pytorch` 的环境），并已安装 IDE（如 PyCharm 或 VS Code）。
3. 确认你已配置 pip 国内镜像源（如清华源或阿里源），以加速后续依赖下载。

如果你能顺利完成以上三步并解释每一步的目的，再继续本块内容；若某一步卡住，请先回到环境准备章节补齐基础。

---

## 一、前置检查：确认 PyTorch 环境就绪

你已经学过 PyTorch 环境的搭建，这里只做快速确认。如果某一步报错，请回到"环境准备：根据 CUDA 版本安装 PyTorch"章节排查。

### 1.1 确认 conda 环境和 PyTorch 版本

```bash
conda activate pytorch

python -c "import torch; print('PyTorch:', torch.__version__)"
# 预期输出类似如下，实际以版本为准：PyTorch: 2.5.x
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

确认前置环境就绪后，在 `pytorch` conda 环境中执行安装命令。`transformers` 是 HuggingFace 提供的核心库，封装了大量预训练模型（如 BERT、GPT-2、Llama、Qwen 等）；`tokenizers` 是其底层的高性能分词库，基于 Rust 实现。

安装命令如下（在已激活的 `pytorch` 环境中执行）：

```bash
pip install transformers
```

安装完成后，建议同时安装 `tokenizers` 库：

```bash
pip install tokenizers
```

> **提示**：由于你已配置 pip 国内镜像源，下载速度会明显提升。若安装过程中遇到版本冲突，请参考官方文档（https://huggingface.co/docs/transformers/installation）确认兼容性。

---

## 三、验证安装是否成功

安装完成后，运行以下命令验证两个库能否正常导入：

```bash
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
```

如果都能打印出版本号，说明环境准备完成，可以进入后续的 BPE/WordPiece 算法和 HuggingFace Tokenizer 使用学习。

---

**操作验证任务**：请在你的 `pytorch` 环境中依次执行上述三条命令（确认 PyTorch 版本、安装 transformers 和 tokenizers、验证导入），并将输出结果记录下来。如果遇到任何报错，请记录错误信息并尝试根据官方文档排查。

## 块 2

## 标准安装 transformers（推荐）

你已有 PyTorch 基础，安装 transformers 只需一条命令。在已装好 PyTorch 的 Python 环境中执行：

```bash
pip install transformers
```

此命令会联网下载安装包（通常数百 MB），**CPU 即可完成**，无需独立 GPU——这正好匹配你当前的硬件条件，放心操作。

若网络较慢，用清华镜像源加速：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**本块学习检查**：完成安装后，用自己的话说明“安装 transformers”与你 LoRA 微调目标的关系——它是你后续加载预训练模型、执行微调的基础工具库。若遇到无法解释

# 模型许可证：Llama 2 Community License（

# 注意：Llama-2-7b FP16

## 块 3

## 2.2 带 PyTorch 依赖的安装（可选）

你的目标是LoRA微调，而LoRA微调离不开PyTorch。这里有个省事的安装方式：如果你在一个**全新的空环境**中，可以用一条命令把transformers和PyTorch一起装好：

```bash
pip install 'transformers[torch]'
```

**但注意**：本课程环境里PyTorch已经提前装好了，你直接 `pip install transformers` 即可，**不要**重复安装torch，避免版本冲突。

## 2.3 使用 conda 安装（可选）

如果你习惯用conda，也可以从conda-forge通道安装：

```bash
conda install conda-forge::transformers
```

> 提醒：conda-forge的版本可能比pip稍旧，建议优先用pip装最新版。

---

##

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 4

### 3.1 是否需要单独安装？

`tokenizers` 通常是 `transformers` 的**依赖项**，安装 `transformers` 时会**自动安装**兼容版本，一般无需手动单独安装。

你可以验证是否已自动安装：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
# 预期输出类似：tokenizers: 0.x.x（实际以版本为准）
```

### 3.2 什么情况下需要手动安装？

如果你需要**独立使用** `tokenizers`（例如自定义训练词表），或自动安装的版本不符合需求，可手动指定版本：

```bash
pip install tokenizers
```

> **踩坑提醒**：某些旧版本 tokenizers 在特定 Python 环境下可能需从源码编译，依赖 Rust 工具链。但本课程使用 Python 3.10 + 较新版 tokenizers 通常有预编译 wheel 包，可直接安装，无需 Rust。

---

## 四、验证安装：运行你的第一个 HuggingFace 模型

安装后必须验证 transformers 能否正常工作。最简单的方法是使用 `pipeline` 加载一个情感分析模型进行推理。你可以在 CPU 上运行此验证，无需 GPU。

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this course!")
print(result)
```

> **模型许可证**：示例使用

> **显存提示**：本示例模型较小，CPU 即可运行；若加载 Llama-2-7b 等大模型，FP16 精度下需至少 14GB 显存。

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

在终端运行：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

预期输出类似（版本不同可能略有差异）：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

**运行要点**：首次执行会自动下载预训练模型（如 `distilbert-base-uncased-finetuned-sst-2-english`，约数百MB）到本地缓存。需联网，磁盘预留1GB以上，CPU即可运行，无需GPU，内存建议≥4GB。

**操作验证**：运行成功后，将句子换成 `'I hate this'`，观察输出标签是否变为 `N

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

运行下面命令，确认两个库已装好：

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

看到版本号即安装成功。这为后续加载模型、做LoRA微调打好基础。

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 5.1 通过环境变量配置缓存路径

你的电脑没有独立GPU，模型文件会占用大量磁盘空间。建议先规划好存放位置，再开始微调。

**临时生效**（仅当前终端）：
```bash
export HF_HOME="/mnt/data/huggingface"
```

**永久生效**（写入配置文件）：
```bash
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

**优先级**（高→低）：
1. `HF_HUB_CACHE`（最优先）
2. `HF_HOME`
3. `XDG_CACHE_HOME` + `/huggingface`

> **建议**：若用云服务器，将 `HF_HOME` 设到数据盘（如 `/mnt/data`），避免系统盘被模型占满。后续微调加载的模型可达数GB至数十GB。

### 5

## 块 8

## 6.1 预先下载模型（①）

考虑到你目前没有独立GPU，所有操作都依赖CPU完成，提前把模型下载到本地会非常实用——这样后续做LoRA微调时，模型加载不会因为网络波动而中断。

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
# 模型许可证：Apache 2.0（bert-base-chinese 采用此许可证，商用需遵守其条款）
snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
```

这段代码会把 `bert-base-chinese` 模型的全部文件保存到当前目录下的 `bert-base-chinese` 文件夹中。下载完成后，这个文件夹就是你的“本地模型库”。

## 6.2 离线加载（①）

模型下载到本地后，有两种方式让它完全离线运行：

**方式一：设置环境变量，阻止所有HTTP请求**

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

运行脚本前加上这个环境变量，transformers 就会完全跳过网络检查，直接读取本地缓存。

**方式二：在代码中指定只加载本地文件**

```python
from transformers import AutoModel

# 注意：CPU 环境下运行 bert-base-chinese（约 420MB 参数），
# 内存占用约 2-4GB，请确保系统有足够可用内存
model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
```

`local_files_only=True` 告诉 transformers：只从本地路径加载，不要尝试联网。对于你后续的LoRA微调，推荐用这种方式，因为微调脚本中需要明确指定模型路径。

---

## 七、常见问题与解决

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `pip install transformers` 速度极慢或超时 | 默认源在国外 | 使用清华镜像源：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found` | Python 版本过低或过高 | 建议使用 Python 3.10 或更高版本 |
| 导入 transformers 时报 `signal` 模块错误 | transformers 版本与 Python 不兼容 | 升级 transformers 到最新版：`pip install -U transformers` |
| `tokenizers` 安装时提示需要 Rust 编译器 | 该版本无预编译 wheel，需从源码构建 | 换用 conda 安装：`conda install -c huggingface tokenizers`；或升级 pip 以获取新版预编译包 |
| `pipeline()` 首次运行卡住不动 | 从 HuggingFace Hub 下载模型，网络不通 | 检查网络；或预先使用 `snapshot_download()` 下载到本地 |
| 磁盘空间不足 | 缓存目录在系统盘，模型文件积累过多 | 设置 `HF_HOME` 到数据盘；定期清理 `~/.cache/huggingface/hub` |
| 验证脚本输出结果与预期不符 | 模型下载不完整 | 删除缓存重新下载，或设置 `local_files_only=False` 强制校验 |

> **最重要的排查口诀**：先确认 `python --version` ≥ 3.10 → 再确认 `torch.__version__` 版本与预期一致 → 最后确认 `transformers.__version__` 为较新稳定版。

**本块学习检查**：完成本块后，用自己的话说明本块

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的环境已就绪。下一节将讲解子词切分算法（如BPE），你会明白"unhappiness"为何被切成`["un", "happiness"]`。随后用今天装好的`transformers`库加载GPT-2的BPE分词器，亲手观察切分结果。这为LoRA微调打下基础——微调前需用分词器处理数据。现在，确认环境可用，我们继续。

## 测评题目

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB07-ST0700-001
Hugging Face 模型与分词器常用的 Python 库名称是________。

### Q-KB0704-005
请写代码：加载 `meta-llama/Llama-2-7b-chat-hf` 分词器，使用 `apply_chat_template` 生成一个包含系统提示（"You are a helpful assistant."）、用户提问（"What is AI?"）的输入字符串，要求包含生成提示（generation prompt），且不进行分词。

### Q-KB0701-002
WordPiece 算法与 BPE 的主要区别在于合并标准不同。WordPiece 选择合并哪一对符号？

### Q-KB0703-004
`padding=True` 和 `padding='longest'` 的效果是完全一样的。（ ）

### Q-KB0701-003
SentencePiece 模型将输入文本视为原始的字符串序列，无需预先进行空格分词，可以直接从原始句子中学习子词单元。（ ）

### Q-KB0703-003
对于一个最大序列长度为 512 的模型，执行 `tokenizer(text, max_length=100, padding='max_length', truncation=True)` 后，输出序列的长度是多少？

### Q-KB0702-001
请写出使用 Hugging Face `transformers` 库加载名为 `bert-base-uncased` 的分词器的代码。
