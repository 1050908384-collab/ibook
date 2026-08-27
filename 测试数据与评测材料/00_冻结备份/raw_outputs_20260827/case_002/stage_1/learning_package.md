# 学习包

## 块 1

## 一、前置检查：确认 PyTorch 环境就绪

你的目标是 LoRA 微调，而 LoRA 微调的第一步，就是确保你脚下的地基——PyTorch 环境——是稳固的。前面你已经学过环境搭建，这里我们只做**快速确认**，不重复造轮子。如果下面任何一步报错，请回到"环境准备：根据 CUDA 版本安装 PyTorch"章节排查。

### 1.1 确认 conda 环境和 PyTorch 版本

打开终端，逐行执行以下命令：

```bash
conda activate pytorch

python -c "import torch; print('PyTorch:', torch.__version__)"
```

**预期输出**类似（实际以你的版本为准）：

```
PyTorch: 2.5.x
```

**版本要求**：本课程在 Python 3.10 + PyTorch 2.5.1 环境下验证通过。不同版本的 `transformers` 对 Python 和 PyTorch 版本要求可能不同，建议以安装时的官方文档为准。以下是本课程验证过的依赖版本示例（`requirements.txt`）：

```text
torch==2.5.1
transformers>=4.45.0
tokenizers>=0.20.0
```

**兼容性说明**：上述版本组合（torch 2.5.1 + transformers>=4.45.0 + tokenizers>=0.20.0）为本课程实际验证通过的组合，可正常协同工作。若你选择其他版本组合，请以各库官方文档的兼容性声明为准。

如果你按照前面板块完成了环境搭建，当前环境已满足上述要求。

> **动手确认**：执行完上面的命令后，请**自己核对**输出的 PyTorch 版本号是否满足 `2.5.x` 的要求。如果版本过低或过高，先别急着继续，回到环境搭建章节把版本对齐。

### 1.2 确认 pip 可用

```bash
pip --version

## 块 2

**本块学习检查**：运行安装命令后，输入 `pip show transformers`，确认输出中出现“Name: transformers”即安装成功。

---

你已装好 PyTorch，现在只需一条命令就能装 `transformers`。打开终端，执行：

```bash
pip install transformers
```

它会联网下载安装包（约几百 MB），CPU 电脑也能跑，无需特殊配置。

**网络慢？** 用清华镜像加速，把安装源指到国内服务器：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

装完顺手验证一下：`pip show transformers`，看到版本信息就说明环境就绪，可以开始 LoRA 微调了。

## 块 3

### 带 PyTorch 依赖的安装（可选）

你的目标是 LoRA 微调，这通常需要 PyTorch 作为后端。如果你在一个**全新的空环境**中安装，可以用一条命令同时装好 transformers 和 PyTorch 适配依赖：

```bash
pip install 'transformers[torch]'
```

**注意**：本课程中 PyTorch 已提前装好，直接 `pip install transformers` 即可，**无需重复安装 torch**。

---

**动手验证**：打开终端，运行 `pip show transformers` 查看已安装版本。如果显示版本信息，说明安装成功，可以继续下一步。

## 块 4

### 3.1 是否需要单独安装？

好消息：`tokenizers` 是**独立库**，需要单独安装，但通常与 `transformers` 一起使用。你安装 `transformers` 时，它会**自动安装**一个兼容版本，所以一般不用手动装。

**动手验证一下**（打开终端，运行下面命令）：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
```

如果看到类似 `tokenizers: 0.x.x` 的输出（版本号可能不同），就说明已经装好了。

### 3.2 什么情况下需要手动安装？

只有两种情况才需要手动装：
1. 你要**独立使用** `tokenizers`（比如自己训练一个词表）；
2. 自动装的版本不符合你的需求。

这时运行：

```bash
pip install tokenizers
```

> **踩坑提醒**：有些旧版本 `tokenizers` 在特定 Python 环境下需要从源码编译，依赖 Rust 工具链。但本课程用 Python 3.10 + 较新版，通常有预编译的 wheel 包，直接装就行，不用碰 Rust。

---

## 四、验证安装：跑通你的第一个模型

装完必须验证 `transformers` 能用。最简单的方式是用 `pipeline` 加载一个

## 块 5

**动手验证：跑通你的第一个 HuggingFace 模型**

既然目标是 LoRA 微调，第一步先确认环境能正常调用模型。在终端直接执行下面这条命令：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

这条命令会加载一个情感分析模型，对 `'we love you'` 这句话做预测。预期输出类似（实际数值随版本略有浮动）：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

**运行前注意**：首次执行会自动下载预训练模型（如 `distilbert-base-uncased-finetuned-sst-2-english`，体积数百 MB）到本地缓存。需要联网，磁盘预留 1GB 以上，CPU 即可运行，内存建议 ≥ 4GB。模型许可证：Apache 2.0。显存要求：CPU 运行无需显存

## 块 6

**本块学习检查**：运行命令确认已安装 transformers 和 tokenizers，记下版本号。

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

看到两行版本号即环境就绪，这是 LoRA 微调第一步。报错则检查安装。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需至少14GB显存。

## 块 7

### 5.1 通过环境变量配置缓存目录

为了给 LoRA 微调做准备，你需要把模型文件存到合适的位置。模型动辄几个 GB，别让它们占满系统盘。

**临时设置**（只对当前终端有效）：

```bash
export HF_HOME="/mnt/data/huggingface"
```

**永久设置**（写入配置文件，每次登录自动生效）：

```bash
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

**优先级**：系统按以下顺序查找缓存位置，找到即用：

1. `HF_HUB_CACHE`（最高，默认基于 `HF_HOME` 生成）
2. `HF_HOME`
3. `XDG_CACHE_HOME` + `/huggingface`

> **建议**：如果你用云服务器，把 `HF_HOME` 指向数据盘（如 `/mnt/data`），避免系统盘被模型文件占满。

### 5.2 查看当前缓存路径

在 Python 中快速确认实际生效的路径（返回的是 `HF_HOME` 下的路径）：

```python
from transformers import file_utils
print(file_utils.default_cache_path)
```

---

**

## 块 8

## 6.1 预先下载模型：为离线环境做准备

你的目标是做 LoRA 微调，这意味着你后面会反复加载同一个模型。如果每次都从网上下载，不仅慢，还容易因为网络问题卡住。所以，**提前把模型下载到本地**，是省时省力的关键一步。

做法很简单，用 `huggingface_hub` 库的 `snapshot_download` 函数，把整个模型仓库一次性拉到本地指定目录：

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
```

执行后，`bert-base-chinese` 这个模型就会完整保存在你当前目录下的 `./bert-base-chinese` 文件夹里。**注意**：这一步需要联网，下载完成后，后续操作就可以完全离线了。

## 6.2 离线加载：让代码只认本地文件

模型下载到本地后，有两种方式让代码“断网”运行：

**方式一：环境变量一键断网**

在终端运行脚本时，加上 `HF_HUB_OFFLINE=1` 这个环境变量，HuggingFace 就会阻止所有 HTTP 请求，强制走本地缓存：

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

**方式二：代码里指定只加载本地文件**

在代码中，加载模型时加上 `local_files_only=True` 参数，明确告诉 Transformers：只从本地路径读取，不要联网：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
```

**实操建议**：你在做 LoRA 微调时，推荐用**方式二**，因为它的意图更明确，不容易误触发网络请求。

---

## 常见问题速查（重点看前两条）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `pip install transformers` 速度极慢或超时 | 默认源在国外 | 使用清华镜像源：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found` | Python 版本过低或过高 | 确保 Python 3.10+（本课程使用 3.10） |
| 导入 transformers 时报 `signal` 模块错误 | transformers 版本与 Python 不兼容 | 升级 transformers 到最新版：`pip install -U transformers` |
| `tokenizers` 安装时提示需要 Rust 编译器 | 该版本无预编译 wheel，需从源码构建 | 换用 conda 安装：`conda install -c huggingface tokenizers`；或升级 pip 以获取新版预编译包 |
| `pipeline()` 首次运行卡住不动 | 从 HuggingFace Hub 下载模型，网络不通 | 检查网络；或预先使用 `snapshot_download()` 下载到本地 |
| 磁盘空间不足 | 缓存目录在

## 块 9

**下一步预告：理解子词切分，为LoRA微调打基础**

你即将进入LoRA微调实战，而分词器是模型理解文本的第一道关卡。下一节我们将拆解BPE、WordPiece、SentencePiece三种算法，搞懂为什么`"unhappiness"`会被切成`["un", "happiness"]`——不是按单词，也不是按字母，而是靠统计从语料中"学"出最优词表。

紧接着，你会用刚装好的`transformers`库加载GPT-2的BPE分词器，亲手观察真实切分结果，把抽象概念变成看得见的输出。

**动手验证**：确认你的Python环境已成功导入`transformers`和`tokenizers`，无报

## 测评题目

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB07-ST0700-001
Hugging Face 模型与分词器常用的 Python 库名称是________。

### Q-KB0704-003
使用 `tokenizer.apply_chat_template` 方法，将以下对话历史转换为模型输入文本，但不进行分词（即 `tokenize=False`）。
```python
messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"}
]
```

### Q-KB0703-002
`padding='max_length'` 参数会将所有序列填充或截断到由________参数指定的长度。

### Q-KB0703-005
请写代码：对一批文本进行分词，要求：
1.  填充到 128 的长度。
2.  如果超过 128 则进行截断。
3.  返回 TensorFlow 张量。

### Q-KB0704-004
`apply_chat_template` 方法中，`add_generation_prompt=True` 参数的作用是？

### Q-KB0703-001
当设置 `truncation=True` 时，如果输入序列长度超过模型的最大长度，分词器会如何处理？

### Q-KB0703-004
`padding=True` 和 `padding='longest'` 的效果是完全一样的。（ ）
