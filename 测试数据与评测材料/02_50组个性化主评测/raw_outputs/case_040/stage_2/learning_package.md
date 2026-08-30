# 学习包

## 块 1

## 一、前置检查：确认 PyTorch 环境就绪

你已经完成了 PyTorch 基础板块，这里只做**快速确认**，不重复讲解。如果某一步报错，请回到"环境准备：根据 CUDA 版本安装 PyTorch"章节排查。

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

在确认 PyTorch 就绪后，接下来安装 `transformers` 库。`transformers` 是 HuggingFace 的核心库，封装了大量预训练模型（如 BERT、GPT-2、Llama、Qwen 等），是你后续进行 LoRA 微调时直接打交道的工具。

```bash
pip install transformers
```

安装完成后，可以快速验证是否成功：

```bash
python -c "import transformers; print('transformers:', transformers.__version__)"
```

---

## 三、安装 tokenizers 库

`tokenizers` 是 `transformers` 底层的高性能分词库，基于 Rust 实现。它负责将文本切分为模型可理解的 token 序列——这是你后续理解 BPE/WordPiece 算法的基础。

```bash
pip install tokenizers
```

验证安装：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
```

---

## 四、验证能加载预训练模型

安装完成后，做一次端到端验证——尝试加载一个预训练模型。这一步能确认 `transformers` 和 `tokenizers` 在你的 RTX 4060 8GB 环境下能正常工作：

```bash
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('bert-base-uncased'); print('模型加载成功')"
# 注意：bert-base-uncased 约 1.1 亿参数，FP16 显存占用约 2-3GB，适合你的 8GB 显存环境。
# 若后续尝试加载更大模型（如 Llama-2-7b），FP16 需要至少 14GB 显存，请先确认显存是否足够。
```

> 首次加载会下载模型文件，需要联网。如果网络受限，请参考官方文档配置镜像源。

---

## 本块学习检查

完成本块后，请用自己的话说明：**本块内容（安装 transformers 和 tokenizers 并验证模型加载）的核心步骤和目的**。

## 块 2

**本块学习检查**：安装 `transformers` 库是 LoRA 微调的前提——后续微调代码都要调用它。若遇到“pip 未找到”等术语问题，先记下再提问。

你的 PyTorch 环境已就绪，直接运行：

```bash
pip install transformers
```

此命令联网下载（约数百 MB），CPU 即可执行。若网络慢，用清华镜像加速：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装后，请用自己的话说明：为什么装这个库是 LoRA 微调的第一步？

## 块 3

### 带 PyTorch 依赖的安装（可选）

如果你在**全新的空环境**中安装，可用 extras 方式一次性装好 transformers 及其 PyTorch 适配依赖：

```bash
pip install 'transformers[torch]'
```

> 本课程中 PyTorch 已提前装好，直接 `pip install transformers` 即可，无需重复安装 torch。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

## 块 4

## 3.1 需要单独装 tokenizers 吗？

**通常不用。** `tokenizers` 是 `transformers` 的依赖项，装 `transformers` 时会自动带上兼容版本。

验证一下是否已装好：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
# 预期输出：tokenizers: 0.x.x（具体版本号以实际为准）
```

## 3.2 什么时候才手动装？

只有两种情况：**独立使用** tokenizers（比如自定义训练词表），或**自动版本不满足需求**。此时执行：

```bash
pip install tokenizers
```

> **踩坑提醒**：旧版 tokenizers 在某些 Python 环境下可能要从源码编译，依赖 Rust 工具链。但本课程用 Python 3.10 + 较新版 tokenizers，通常有预编译 wheel 包，直接装即可，**无需 Rust**。

---

## 四、验证安装：跑通第一个 HuggingFace 模型

装完必须验证 transformers 能用。最简单的方式：用 `pipeline` 加载情感分析模型做一次推理。

> **模型许可证**：本课程示例使用 Llama 2 模型，遵循 Llama 2 Community License（商用月活>7亿需Meta授权）。
> **显存注意**：Llama-2-7b FP

## 块 5

## 4.1 快速验证脚本（①）

在终端执行下面这条命令，验证 `transformers` 是否装好：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

预期输出类似（实际以版本为准）：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

**本地运行配置要求**：首次运行会自动从 HuggingFace Hub 下载预训练模型（如 `distilbert-base-uncased-finetuned-sst-2-english`，体积通常在数百 MB 量级）到本地缓存。需要**联网**，磁盘预留 1GB 以上空间，**CPU 即可运行**，内存建议 ≥ 4GB。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（

## 块 6

### 4.2 查看安装版本

安装完成后，先确认版本无误，再进入后续微调流程。在终端运行：

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

这会分别打印两个库的版本号。看到输出即说明环境就绪，可以继续。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

## 块 7

## 5.1 通过环境变量配置缓存路径

你的 LoRA 微调目标中，模型文件动辄数 GB。RTX 4060 8GB 显存下，更要把模型缓存放到合适位置，避免系统盘被占满。

**临时生效**（仅当前终端）：

```bash
export HF_HOME="/mnt/data/huggingface"
```

**永久生效**（写入配置文件）：

```bash
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

**优先级**（高→低）：`HF_HUB_CACHE` > `HF_HOME` > `XDG_CACHE_HOME` + `/huggingface`

> **建议**：云服务器请将 `HF_HOME` 指向数据盘（如 `/mnt/data`），防止系统盘被模型文件占满。后续 LoRA 微调与推理部署加载的模型可达数 GB 至数十 GB。

## 5.2 查看当前缓存路径

```python
from transformers import utils
print(utils.default_cache_path)
```

---

## 六、离线模式配置（拓展）

内网或防火墙环境中，transformers 可完全离线运行，前提是设置环境变量 `HF_HUB_OFFLINE=

## 块 8

## 6.1 预先下载模型：为离线环境做准备

你的目标是 LoRA 微调，这意味着后续训练脚本会反复加载同一个预训练模型。如果每次都从 HuggingFace Hub 在线拉取，不仅慢，而且一旦网络波动就会中断训练。所以，**先把模型完整下载到本地磁盘**，是离线训练的第一步。

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
# 模型许可证：请查阅模型仓库页面的 License 声明（如 bert-base-chinese 为 Apache-2.0）
snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
```

执行后，`bert-base-chinese` 的模型权重、配置文件、词表等全部文件都会保存在当前目录下的 `./bert-base-chinese` 文件夹中。你只需要执行一次，后续训练全部走本地路径。

## 6.2 离线加载：切断网络依赖

模型下载到本地后，有两种方式确保代码只读本地文件、不发起任何网络请求：

**方式一：设置环境变量（全局拦截）**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python your_script.py
```

这会让 HuggingFace 生态的组件在本次运行中禁用网络访问。注意：`HF_HUB_OFFLINE=1` 仅影响 `huggingface_hub` 库，需同时设置 `TRANSFORMERS_OFFLINE=1` 才能覆盖 transformers 等库的网络请求。任何缺失的本地文件都会直接报错而不是尝试下载——这正好帮你暴露“哪些文件没下全”。

**方式二：代码中指定 `local_files_only=True`（局部强制）**

```python
from transformers import AutoModel

# 注意：bert-base-chinese 模型 FP16 推理约需 0.4GB 显存，微调训练显存需求更高，请根据实际 batch size 评估
model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
```

这个参数的含义是：**只从本地路径加载，绝不联网**。推荐在训练脚本中始终加上它，避免意外触发网络请求导致卡死。

## 常见问题速查（重点看前三条）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `pip install transformers` 极慢或超时 | 默认源在国外 | 清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found` | Python 版本不兼容 | 确保 Python 3.10+（本课程用 3.10） |
| `pipeline()` 首次运行卡住 | 正在从 Hub 下载模型，网络不通 | 预先用 `snapshot_download()` 下载到本地；使用 pipeline 时需设置 `local_files_only=True` 或设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` |
| 磁盘空间不足 | 缓存堆积在系统盘 | 设置 `HF_HOME` 到数据盘；清理 `~/.cache/huggingface/hub` |

**排查口诀**：先 `python --version` ≥ 3.10 → 再 `torch.__version__`

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

环境已就绪，下一步将深入子词切分算法（如BPE），理解"unhappiness"为何切分为`["un", "happiness"]`。随后用`transformers`加载GPT-2的BPE分词器，亲手观察切分结果。这为LoRA微调打下分词基础——模型输入依赖分词器，掌握其机制才能正确预处理数据。现在确认环境可用，我们继续。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-

## 测评题目

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB07-ST0700-003
安装 Hugging Face Transformers 的常用命令是？

### Q-KB0701-003
SentencePiece 模型将输入文本视为原始的字符串序列，无需预先进行空格分词，可以直接从原始句子中学习子词单元。（ ）

### Q-KB0704-001
Hugging Face 中的 `chat_template` 主要用于什么目的？

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

### Q-KB0701-004
以下关于子词切分（Subword Tokenization）的优点，描述**错误**的是？

### Q-KB0702-004
在 Hugging Face Tokenizer 的输出中，`attention_mask` 的作用是什么？
