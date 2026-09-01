# 学习包

## 块 1

## 本块学习检查

在开始之前，先用一个最小示例快速回顾前置知识：确认你能在终端中激活 conda 虚拟环境（`conda activate pytorch`），并解释为什么要配置 pip 国内镜像源（加速下载）。如果你能流畅完成并解释，再继续本块内容。

---

## 环境准备：安装 transformers 与 tokenizers 库

> **适配你的环境**：你使用的是 MacBook M2（16GB 统一内存），目标是用 MLX 做 LoRA 微调。本块只负责把 `transformers` 和 `tokenizers` 装好并验证可用，为后续微调打基础。注意：本课程验证环境是 Python 3.10 + PyTorch 2.4.x，你的 M2 上需安装 PyTorch 2.4.x 版本（Apple Silicon 官方支持版本）。

### 一、前置检查：确认 PyTorch 环境就绪

以下步骤前面已详细学过，这里只做**快速确认**。如果某一步报错，请回到"环境准备：根据 CUDA 版本安装 PyTorch"章节排查（注意：M2 上无 CUDA，用 CPU 版即可）。

#### 1.1 确认 conda 环境和 PyTorch 版本

```bash
conda activate pytorch

python -c "import torch; print('PyTorch:', torch.__version__)"
# 预期输出类似如下，实际以版本为准：PyTorch: 2.4.x
```

**版本要求**：本课程在 Python 3.10 + PyTorch 2.4.x 环境下验证通过。不同版本的 `transformers` 对 Python 和 PyTorch 版本要求可能不同，建议以安装时的官方文档为准。以下是本课程验证过的依赖版本示例（`requirements.txt`）：

```text
torch==2.4.1
transformers>=4.45.0
tokenizers>=0.20.0
```

如果你按前面板块完成了环境搭建，当前环境已满足上述要求。

#### 1.2 确认 pip 可用

```bash
pip --version
# 预期输出类似如下，实际以版本为准：pip 24.x.x
```

---

### 二、安装 transformers 库

确认 PyTorch 就绪后，在同一个 `pytorch` conda 环境中安装 `transformers` 库。`transformers` 是 HuggingFace 的核心库，封装了大量预训练模型（如 BERT、GPT-2、Llama、Qwen 等）；`tokenizers` 是其底层的高性能分词库，基于 Rust 实现。

安装命令（在 `pytorch` 环境中执行）：

```bash
pip install transformers
```

安装完成后，验证是否能正常导入：

```bash
python -c "import transformers; print('transformers:', transformers.__version__)"
```

同时确认 `tokenizers` 已随依赖安装：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
```

**验证任务**：确认两个库都能成功导入并打印版本号。如果导入报错，优先检查是否仍在 `pytorch` 环境中，以及 pip 是否使用了正确的 Python 解释器（可用 `which python` 确认）。

---

**本块学习检查**：完成上述安装与导入验证后，记录你实际安装的 `transformers` 和 `tokenizers` 版本号，并确认它们满足 `transformers>=4.

## 块 2

### 本块学习检查

完成本块后，请用自己的话说明：安装 `transformers` 库与你的 LoRA 微调目标有什么关系？如果遇到无法解释的术语（如“pip”“镜像源”），先记录下来，稍后提问。

---

**标准安装（推荐）**

你已经装好了 PyTorch，现在只需一条命令就能安装 `transformers` 库：

```bash
pip install transformers
```

这个命令会联网下载安装包（通常数百 MB），你的 MacBook M2（16GB 内存）CPU 完全够用，无需特殊配置。

**如果网络较慢**，用清华镜像源加速：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

## 块 3

### 2.2 带 PyTorch 依赖的安装（可选）

你在 Mac M2 上已装好 PyTorch，所以本块内容可跳过——直接 `pip install transformers` 即可，无需重复安装 torch。

若未来在全新空环境（非本课程）中，可用 extras 方式一次性装齐 transformers 及其 PyTorch 适配依赖：

```bash
pip install 'transformers[torch]'
```

> 本课程场景：PyTorch 已提前装好，直接安装 transformers 即可。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

## 块 4

### 3.1 是否需要单独安装？

`tokenizers` 是 `transformers` 的依赖项，安装 `transformers` 时会自动装上兼容版本，一般无需手动单独安装。你可以用下面命令验证：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
# 预期输出类似：tokenizers: 0.x.x（实际以版本为准）
```

### 3.2 什么情况下需要手动安装？

如果你要独立使用 `tokenizers`（比如自定义训练词表），或自动版本不满足需求，可手动指定版本：

```bash
pip install tokenizers
```

> **踩坑提醒**：旧版 tokenizers 在特定 Python 环境下可能需从源码编译，依赖 Rust 工具链。但本课程用 Python 3.10 + 较新版 tokenizers，通常有预编译 wheel 包，直接安装即可，无需 Rust。

---

## 四、验证安装：运行你的第一个 HuggingFace 模型

装完必须验证 `transformers` 能否工作。最简单是用 `pipeline` 加载情感分析模型做推理。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**本块学习检查**

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

在你的Mac M2上，打开终端，直接复制运行下面这行命令：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

这是验证transformers库是否装好的最快方式——它会自动加载一个情感分析模型，判断"we love you"是正面还是负面。首次运行会联网下载模型（约几百MB），请确保磁盘有1GB以上空间，CPU就能跑，内存4GB以上即可。预期输出类似：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

看到`POSITIVE`就说明环境OK。这一步跑通后，你后续做LoRA微调

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

在Mac M2上，运行以下命令验证transformers和tokenizers已正确安装：

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

看到版本号输出即安装成功。这是LoRA微调前的环境确认步骤——确保HuggingFace工具链可用，后续加载模型和微调才能顺利进行。若报错，检查安装步骤是否完整。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 7

## 5.1 通过环境变量配置缓存路径

在Mac M2上做LoRA微调，模型文件动辄数GB，默认缓存位置可能挤占系统盘。用环境变量把缓存指到你的数据盘或外置存储更稳妥。

**临时生效**（仅当前终端）：
```bash
export HF_HOME="/mnt/data/huggingface"
```

**永久生效**（写入shell配置）：
```bash
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

**优先级**（高→低）：
1. `HF_HUB_CACHE`（最优先）
2. `HF_HOME`
3. `XDG_CACHE_HOME` + `/huggingface`

> 建议：云服务器用户把`HF_HOME`设到数据盘（如`/mnt/data`），避免系统盘被模型占满。后续微调和推理加载的模型可达数GB至数十GB。

## 5.2 查看当前缓存路径

```python
from transformers import utils
print(utils.default_cache_path)
```

## 六、离线模式（拓展

## 块 8

## 6.1 预先下载模型（①）

考虑到你后续要做 LoRA 微调，模型文件往往较大，且训练过程中需要反复加载。提前把模型下载到本地，能避免每次运行时都去 HuggingFace Hub 拉取，既省时间也避免网络波动中断训练。

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
# 模型许可证：bert-base-chinese 采用 Apache 2.0 许可证，可自由商用
snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
```

这条命令会把 `bert-base-chinese` 的完整仓库（含配置文件、权重等）下载到你当前目录下的 `./bert-base-chinese` 文件夹中。

## 6.2 离线加载（①）

模型下载到本地后，有两种方式让代码只读本地文件、不发起网络请求：

**方式一：设置环境变量阻止所有 HTTP 请求**

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

适合整个脚本都不需要联网的场景，比如你跑 LoRA 训练脚本时，直接加上这个前缀即可。

**方式二：在代码中指定只加载本地文件**

```python
from transformers import AutoModel

# 注意：bert-base-chinese FP16 推理需要至少 2GB 显存，训练需根据 batch size 调整
model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
```

`local_files_only=True` 明确告诉 transformers：只从本地路径加载，不要尝试联网下载。这种方式更灵活，适合脚本中部分模型用本地、部分仍需联网的情况。

---

## 七、常见问题与解决（①+②+③+④）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `pip install transformers` 速度极慢或超时 | 默认源在国外 | 使用清华镜像源：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found` | 包名拼写错误、源配置不当或 Python 版本不兼容 | 检查包名拼写；更换镜像源；确认 Python 3.10+（本课程使用 3.10） |
| 导入 transformers 时报 `signal` 模块错误 | Python 环境配置异常或版本不兼容 | 检查 Python 环境完整性（如重新创建虚拟环境）；提供完整错误信息以便进一步诊断 |
| `tokenizers` 安装时提示需要 Rust 编译器 | 该版本无预编译 wheel，需从源码构建 | 安装 Rust 工具链（`rustup`）；或使用 conda 安装预编译包（需确认该版本可用）；或升级 pip 获取新版预编译包 |
| `pipeline()` 首次运行卡住不动 | 从 HuggingFace Hub 下载模型，网络不通 | 检查网络；或预先使用 `snapshot_download()` 下载到本地 |
| 磁盘空间不足 | 缓存目录在系统盘，模型文件积累过多 | 设置 `HF_HOME` 到数据盘；定期清理 `~/.cache/huggingface/hub` |
| 验证脚本输出结果与预期不符 | 模型下载不完整或缓存损坏 | 删除缓存重新下载，并检查模型文件完整性（如对比文件大小或校验和） |

> **最重要的排查口诀**：先确认 `python

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的环境已就绪。下一节将深入讲解现代大语言模型的子词切分算法（如BPE、WordPiece、SentencePiece），你会了解为何"unhappiness"被切分为`["un", "happiness"]`而非按单词或字母切分，以及这些算法如何用统计方法从语料库中"学习"最优词表。随后，我们将用今天装好的`transformers`库加载真实预训练分词器（如GPT-2的BPE分词器），亲手观察切分结果。

**操作验证**：确认`

## 测评题目

### Q-KB07-ST0700-003
安装 Hugging Face Transformers 的常用命令是？

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB0702-005
请写代码：加载 `gpt2` 分词器，并对文本 `"Hello, world!"` 进行分词，要求返回 PyTorch 张量，并打印出 `input_ids` 和 `attention_mask`。

### Q-KB0703-002
`padding='max_length'` 参数会将所有序列填充或截断到由________参数指定的长度。

### Q-KB0704-002
如何检查一个已加载的 `tokenizer` 是否内置了 `chat_template`？

### Q-KB0701-005
请简述 BPE 算法在解码（Decoding）时可能遇到的一个主要问题，以及 SentencePiece 是如何解决这个问题的。

### Q-KB0704-003
使用 `tokenizer.apply_chat_template` 方法，将以下对话历史转换为模型输入文本，但不进行分词（即 `tokenize=False`）。
```python
messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"}
]
```

### Q-KB0701-002
WordPiece 算法与 BPE 的主要区别在于合并标准不同。WordPiece 选择合并哪一对符号？
