# 学习包

## 块 1

## 一、前置检查：确认 PyTorch 环境就绪

你之前已经完成了 PyTorch 环境的搭建，这里只做**快速确认**，确保后续安装不会出错。如果某一步报错，请回到“环境准备：根据 CUDA 版本安装 PyTorch”章节排查。

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
> **兼容性说明**：根据 transformers 官方文档，transformers 4.45.0 要求 PyTorch >= 2.1.0，因此 torch 2.5.1 满足该要求。如果你按照前面板块完成了环境搭建，当前环境已满足上述要求。

### 1.2 确认 pip 可用

```bash
pip --version
# 预期输出类似如下，实际以版本为准：pip 24.x.x
```

---

## 二、安装 transformers 库（①）

**本块学习检查**：在继续安装之前，请先用一个最小示例回顾前置知识：确认你已安装 Python 3.10+、配置好 conda 虚拟环境、安装了 IDE（如 PyCharm 或 VS Code），并配置了 pip 国内镜像源以加速下载。确认你能解释这些关键写法后再继续本块。

接下来，在已激活的 `pytorch` conda 环境中，使用 pip 安装 `transformers` 库：

```bash
pip install transformers
if [ $? -ne 0 ]; then echo '安装失败，请检查网络或依赖'; exit 1; fi
```

安装完成后，可以快速验证是否成功导入：

```bash
python -c "import transformers; print('transformers:', transformers.__version__)"
if [ $? -ne 0 ]; then echo '导入失败，请检查安装是否正确'; exit 1; fi
```

> **提示**：`transformers` 是 HuggingFace 提供的核心库，封装了大量预训练模型（如 BERT、GPT-2、Llama、Qwen 等）。安装时会自动拉取依赖的 `tokenizers` 库（基于 Rust 实现的高性能分词库），无需单独安装。

---

## 三、安装 tokenizers 库（可选，通常随 transformers 自动安装）

如果你需要单独安装或确认 `tokenizers` 版本，可执行：

```bash
pip install tokenizers
if [ $? -ne 0 ]; then echo '安装失败，请检查网络或依赖'; exit 1; fi
```

验证导入：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
if [ $? -ne 0 ]; then echo '导入失败，请检查安装是否正确'; exit 1; fi
```

> **说明**：`tokenizers` 是 `transformers` 的底层依赖，一般随 `transformers` 自动安装。单独安装用于确保版本满足要求（如 `>=0.20.0`）。

---

## 四、验证安装成功

运行以下命令，确认两个库都能正常导入且版本符合要求：

```bash
python -c "import torch,

## 块 2

### 2.1 标准安装（推荐）

你已经装好了 PyTorch，现在只需一条命令就能安装 `transformers` 库。打开终端，执行：

```bash
pip install transformers
```

这个命令会联网下载安装包（通常几百 MB），**对电脑配置没有特殊要求，CPU 就能完成**。如果你网络较慢，可以用清华镜像源加速：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**本块学习检查**：确认你的终端已处于之前配置好的 PyTorch 环境中，然后运行上述命令。安装成功后，在 Python 中执行 `import transformers`，若无报错，说明安装完成。

## 块 3

### 带 PyTorch 依赖的安装（可选）

如果你是在**全新的空环境**中安装，可以用一条命令同时装好 transformers 和 PyTorch 适配依赖：

```bash
pip install 'transformers[torch]'
```

**但请注意**：本课程中 PyTorch 已提前装好，你只需直接 `pip install transformers` 即可，**无需重复安装 torch**。这样既节省时间，也避免版本冲突。

> **小提示**：这条命令中的 `[torch]` 是 extras 语法，表示“额外附带 torch 相关依赖”。理解这一点，比死记命令更重要——它解释了为什么有时安装一个库会“顺带”装好另一个库。

---

**本块学习检查**：请用自己的话回答——为什么本课程中不需要执行 `pip install 'transformers[torch]'`？

## 块 4

### 3.1 是否需要单独安装？

`tokenizers` 通常是 `transformers` 的**依赖项**，安装 `transformers` 时会**自动安装**兼容版本的 `tokenizers`，一般不需要手动单独安装。

你可以验证是否已自动安装：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
# 预期输出类似如下，实际以版本为准：tokenizers: 0.x.x
```

### 3.2 什么情况下需要手动安装？

如果你需要**独立使用** `tokenizers` 库（例如自定义训练一个词表），或者自动安装的版本不符合需求，可以手动指定版本：

```bash
pip install tokenizers
```

> **踩坑提醒**：参考文章记录，某些旧版本 tokenizers 在特定 Python 环境下可能需要从源码编译，进而依赖 Rust 工具链。但本课程使用的 Python 3.10 + 较新版 tokenizers 通常有预编译的 wheel 包，可直接安装，无需 Rust。

---

## 四、验证安装：运行你的第一个 HuggingFace 模型

安装完成后，必须验证 transformers 是否能正常工作。最简单的方法是使用 `pipeline` 加载一个情感分析模型进行推理。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权

# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 5

**4.1 快速验证脚本：跑通你的第一个模型**

既然你的目标是理解Transformer架构，且每天只有半小时，我们不必纠结代码细节，而是把这一步当作“开箱检查”——确认环境装好了，并亲眼看到模型输出，为后续理解架构建立直观印象。

**操作步骤**：在终端粘贴并运行下面这行命令：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

**你会看到什么**：命令会输出一个类似下面的结果（具体数值因版本而异）：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

这表示模型把“we love you”判断为**正面情绪**（POSITIVE），置信度接近1。这个输出就是Transformer模型“理解”语言后给出的结果。

**运行前须知**（针对你的无GPU

## 块 6

**本块学习检查**：运行下面的命令，确认你已成功安装这两个库，并记录输出的版本号。

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

**理解要点**：这条命令用 Python 的 `-c` 参数直接执行代码。`import` 导入库，`__version__` 是库自带的版本属性。看到版本号，说明安装成功，且环境能正常调用它们——这是你后续加载模型的第一步。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 7

## 5.1 通过环境变量配置缓存目录

你正在学习 Transformer，但先别急着看模型结构——**模型文件动辄数 GB**，如果下载到系统盘，很快会占满空间。这一节教你用环境变量指定缓存位置，属于“地基”工作。

**临时生效**（仅当前终端有效）：

```bash
export HF_HOME="/mnt/data/huggingface"
```

**永久生效**（写入配置文件，每次登录自动加载）：

```bash
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

**优先级**（高→低）：`HF_HUB_CACHE` > `HF_HOME` > `XDG_CACHE_HOME` + `/huggingface`。即：若设置了 `HF_HUB_CACHE`，它说了算；否则看 `HF_HOME`；再否则用系统默认的 `XDG_CACHE_HOME` 下的 `huggingface` 文件夹。

> **给你的建议**：若用云服务器，把 `HF_HOME` 指向数据盘（如 `/mnt/data`），避免系统盘被撑爆。后续微调和部署时，模型体积可达数 GB

## 块 8

### 6.1 预先下载模型（①）

既然你每天只有半小时，且目标是理解Transformer，那么“离线模式”的核心价值在于：**把下载这个耗时步骤提前做完，之后每次运行代码都不再依赖网络**。这能让你把宝贵时间花在理解模型结构上，而不是等待下载。

操作上，只需运行下面这段代码，它会将整个模型仓库（包括配置文件、权重等）下载到你当前目录下的 `./bert-base-chinese` 文件夹中：

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
# 模型许可证：Apache 2.0（bert-base-chinese）
try:
    snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
except OSError as e:
    print(f"下载失败，请检查网络连接：{e}")
```

### 6.2 离线加载（①）

下载完成后，有两种方式让代码“只认本地文件”，不再尝试联网：

**方式一：环境变量（全局生效）**
在终端运行脚本前加上 `HF_HUB_OFFLINE=1`，表示本次运行中禁止所有HTTP请求：

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

**方式二：代码参数（局部生效）**
在加载模型时，显式指定 `local_files_only=True`，告诉Transformers只从本地路径读取：

```python
from transformers import AutoModel

# 注意：加载 bert-base-chinese 模型约需 400MB 显存
try:
    model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
except OSError as e:
    print(f"模型加载失败，请检查本地文件完整性：{e}")
```

**理解要点**：两种方式本质相同——切断网络请求，强制使用本地缓存。方式一适合整个脚本都离线，方式二适合只在加载模型时离线。

---

## 七、常见问题与解决（①+②+③+④）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `pip install transformers` 速度极慢或超时 | 默认源在国外 | 使用清华镜像源：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found` | Python 版本过低或过高 | 确保 Python 3.10+（本课程使用 3.10） |
| 导入 transformers 时报 `signal` 模块错误 | transformers 版本与 Python 不兼容 | 升级 transformers 到最新版：`pip install -U transformers` |
| `tokenizers` 安装时提示需要 Rust 编译器 | 该版本无预编译 wheel，需从源码构建 | 换用 conda 安装：`conda install -c huggingface tokenizers`；或升级 pip 以获取新版预编译包 |
| `pipeline()` 首次运行卡住不动 | 从 HuggingFace Hub 下载模型，网络不通 | 检查网络；或预先使用 `snapshot_download()` 下载到本地 |
| 磁盘空间不足 | 缓存目录在系统盘，模型文件积累过多 | 设置 `HF_HOME` 到数据盘；定期清理 `~/.cache/huggingface/hub` |
| 验证脚本

## 块 9

**下一步预告：理解子词切分算法**

下一节将讲解现代大语言模型使用的子词切分算法，例如 BPE、WordPiece、SentencePiece。你会明白为什么 "unhappiness" 会被切分成 `["un", "happiness"]`，而不是按单词或字母切分——这些算法通过统计方法从语料库中"学习"出最优词表。

随后，我们将用已装好的 `transformers` 库加载真实预训练分词器（如 GPT-2 的 BPE 分词器），亲手观察切分结果。

**本块学习检查**：确认你的 Python 环境中 `transformers` 库已安装成功，并能正常导入（例如执行 `import transformers` 无报错）。若

## 测评题目

### Q-KB07-ST0700-001
Hugging Face 模型与分词器常用的 Python 库名称是________。

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB0704-005
请写代码：加载 `meta-llama/Llama-2-7b-chat-hf` 分词器，使用 `apply_chat_template` 生成一个包含系统提示（"You are a helpful assistant."）、用户提问（"What is AI?"）的输入字符串，要求包含生成提示（generation prompt），且不进行分词。

### Q-KB0701-005
请简述 BPE 算法在解码（Decoding）时可能遇到的一个主要问题，以及 SentencePiece 是如何解决这个问题的。

### Q-KB0701-003
SentencePiece 模型将输入文本视为原始的字符串序列，无需预先进行空格分词，可以直接从原始句子中学习子词单元。（ ）

### Q-KB0702-001
请写出使用 Hugging Face `transformers` 库加载名为 `bert-base-uncased` 的分词器的代码。

### Q-KB0701-004
以下关于子词切分（Subword Tokenization）的优点，描述**错误**的是？

### Q-KB0701-001
Byte Pair Encoding (BPE) 算法的核心思想是？
