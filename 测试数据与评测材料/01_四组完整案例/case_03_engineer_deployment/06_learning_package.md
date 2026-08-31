# 学习包

## 块 1

### 一、前置检查：确认 PyTorch 环境就绪

你已经掌握 Python 编程基础，并且在前面的板块中完成了 PyTorch 的安装配置。这里我们只做快速确认，不重复展开细节。

**确认步骤：**

1. 打开终端，激活你的 `pytorch` conda 环境：
   ```bash
   conda activate pytorch
   ```

2. 进入 Python 交互环境，确认 PyTorch 能正常导入且版本正确：
   ```bash
   python
   ```
   ```python
   import torch
   print(torch.__version__)
   ```
   你应该能看到 `2.5.1` 或类似版本号输出。

3. 确认 CUDA 可用（如果你之前配置了 GPU 版本）：
   ```python
   print(torch.cuda.is_available())
   ```
   返回 `True` 说明 GPU 环境正常。

**验证方式：** 如果上述命令都能正常执行且输出符合预期，说明 PyTorch 环境就绪。如果某一步报错，回到"环境准备：根据 CUDA 版本安装 PyTorch"章节排查。

**与学习目标的关系：**

本节是环境准备的第一环。你的目标是完成 LoRA 微调 + 权重合并 + vLLM 部署，而 `transformers` 库正是加载预训练模型、执行 LoRA 微调的核心工具；`tokenizers` 负责将文本转换为模型可接受的输入格式。环境就绪是后续所有操作的前提。

**记录待提问术语：** 如果遇到无法解释的术语（如"分词"的具体含义），先记录下来，在后续学习中带着问题去理解。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

**部署显存提示**：后续使用 vLLM 部署模型时，需注意模型加载的显存占用。例如，Llama-2-7b 以 FP16 精度加载至少需要 14GB 显存，请根据你的 GPU 配置提前规划。

## 块 2

### 1.1 确认 conda 环境和 PyTorch 版本

先激活你的 conda 环境，再确认 PyTorch 是否就绪：

```bash
conda activate pytorch

python -c "import torch; print('PyTorch:', torch.__version__)"
# 预期输出类似如下，实际以版本为准：PyTorch: 2.5.x
```

执行完，终端里会打印出 `PyTorch:` 加版本号。看到类似 `2.5.x` 就说明环境没问题。

**版本要求**：本课程在 Python 3.10 + PyTorch 2.5.1 环境下验证通过。不同版本的 transformers 对 Python 和 PyTorch 版本要求可能不同，建议以安装时的官方文档为准。以下是本课程验证过的依赖版本示例（`requirements.txt`）：

```text
torch==2.5.1
transformers>=4.45.0
tokenizers>=0.20.0
```

如果你按照前面板块完成了环境搭建，当前环境已满足上述要求。

### 1.2 确认 pip 可用

```bash
pip --version
# 预期输出类似如下，实际以版本为准：pip 24.x.x
```

执行完，终端会显示 pip 的版本号，确认能正常调用即可。


**与学习目标的关系**：这一步是 LoRA 微调 + vLLM 部署的前置检查。你的目标是跑通 LoRA 微调并合并权重，再用 vLLM 部署——这些操作都依赖 transformers 和 PyTorch 正确安装。确认好环境，后面跑训练脚本时才不会卡在导入报错上。

**记录任务**：如果执行时遇到任何不认识的报错或术语（比如 `ModuleNotFoundError`），先记下来，稍后一起提问。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 3

# 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） · 本块学习卡
## 立即动手 · （环境准备：在Python环境中安装transformers、tokenizers库） 最简操作
### 本节关键命令 · 你自己的运行记录（请先执行，再读正文）
| 序号 | 命令（原文给出） | 你预期的输出（动手前猜测） | 实际终端输出（执行后粘贴） | 你当下是否看懂每个输出字段？ |
|---|---|---|---|---|
| 2 | *(下一段代码，读完后自行粘贴)* |  |  | 待填 |
| 3 | *(后续命令)* |  |  | 待填 |


> 🎯 **你的画像 = 动手型 learner**：本块**强制阅读顺序与节奏**（不要跳步）：
> 1. 立即从上面表格第一条命令开始 → 粘贴终端 → 实际运行 → 把输出填回表格；
> 2. 遇到下面的『报错排查列表段』时，先停下来，用你机器的输出逐条对；
> 3. 概念说明段**只在你「动手卡住 / 看不懂输出字段」时再回溯阅读**；
> 4. 全部命令执行完后，再回到文末『画像专属 · 本块学习检查』完成验证。

> ⚙️ 实训条件：RTX 3060 12GB；本块所属知识板块：『子词切分与分词器（Tokenizer）』。


## 二、安装 transformers 库（①）


## 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） 本块学习检查
### 动手验证（全部在你本机上独立完成，不会写的地方再回正文找）
1. **执行与输出核对**：本块原文中没有独立代码块，请按文中说明的步骤顺序实际执行，并记录每一步的输出结果。

2. **异常观察**：若执行过程中出现非预期输出，请自行记录报错现象、复现步骤和已尝试的修复动作，再带着这些信息在答疑区向导师提问。

3. **字段自查**：选你执行输出里出现的任意 3 个关键字段，回到本块正文中找到它们的解释，再用 1 句话写下你自己的理解。

4. **流程与顺序自检**：如果本块涉及多条命令，说明你实际执行的顺序是否与文中描述一致？如果不一致，会导致什么后果？


### 实训条件提示 · 你的设备
已记录你的实训条件：RTX 3060 12GB。示例会按本地 NVIDIA GPU 可执行的参数和验证方式组织，并保留与本知识点直接相关的最小步骤。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 4

### 2.1 标准安装（推荐）

`transformers` 库直接用 pip 装就行。你已经有 PyTorch 环境了，所以只需一条命令：

```bash
pip install transformers
```

装完可以在 Python 里跑 `import transformers` 验证一下，不报错就说明装好了。

> **本地运行配置要求**：这步要联网下载安装包（体积通常几百 MB），对电脑配置没要求，CPU 就能跑。网络慢的话用清华镜像加速：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

装完同样用 `import transformers` 确认。

**与学习目标的关系**：这一步装的是 Hugging Face 的 `transformers` 库，后面做 LoRA 微调、权重合并和 vLLM 部署时，模型加载、分词、推理都要靠它。先把环境备好，后面才跑得动。

**记录**：目前没有遇到无法解释的术语，继续往下走。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 5

### 2.2 带 PyTorch 依赖的安装（可选）

如果你是在一个**全新的空环境**中安装（非本课程场景），可以用 extras 方式一次性安装 transformers 及其 PyTorch 适配依赖：

```bash
pip install 'transformers[torch]'
```

执行完，pip 会同时解析并安装 transformers 与 PyTorch 相关依赖包，终端会显示安装成功的版本号列表，你可以用 `pip list | grep torch` 确认 torch 已就位。

> 本课程中 PyTorch 已提前装好，直接 `pip install transformers` 即可，无需重复安装 torch。

### 2.3 使用 conda 安装（可选）

如果你更习惯 conda，也可以从 conda-forge 通道安装：

```bash
conda install conda-forge::transformers
```

执行完，conda 会显示将要安装的包及版本，确认后即可完成安装。你可以用 `conda list transformers` 验证安装结果。

> **注意**：conda-forge 中的版本可能略滞后于 pip，建议优先使用 pip 安装以获得最新版本。


**与当前学习目标的关系**：本块解决的是环境准备问题——你后续要做 LoRA 微调、权重合并和 vLLM 部署，第一步就是确保 transformers 库可用。本课程场景下 PyTorch 已就绪，直接 `pip install transformers` 即可，不需要走 extras 或 conda 路径；这两条是备选方案，了解即可。如果你在安装过程中遇到任何报错或看不懂的术语（比如 extras、conda-forge 通道），先记下来，下一步提问时一起处理。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 6

# 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） · 本块学习卡
## 立即动手 · （环境准备：在Python环境中安装transformers、tokenizers库） 最简操作
### 本节关键命令 · 你自己的运行记录（请先执行，再读正文）
| 序号 | 命令（原文给出） | 你预期的输出（动手前猜测） | 实际终端输出（执行后粘贴） | 你当下是否看懂每个输出字段？ |
|---|---|---|---|---|
| 2 | *(下一段代码，读完后自行粘贴)* |  |  | 待填 |
| 3 | *(后续命令)* |  |  | 待填 |


> 🎯 **你的画像 = 动手型 learner**：本块**强制阅读顺序与节奏**（不要跳步）：
> 1. 立即从上面表格第一条命令开始 → 粘贴终端 → 实际运行 → 把输出填回表格；
> 2. 遇到下面的『报错排查列表段』时，先停下来，用你机器的输出逐条对；
> 3. 概念说明段**只在你「动手卡住 / 看不懂输出字段」时再回溯阅读**；
> 4. 全部命令执行完后，再回到文末『画像专属 · 本块学习检查』完成验证。

> ⚙️ 实训条件：RTX 3060 12GB；本块所属知识板块：『子词切分与分词器（Tokenizer）』。


## 三、安装 tokenizers 库（①+②）


## 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） 本块学习检查
### 动手验证（全部在你本机上独立完成，不会写的地方再回正文找）
1. **执行与输出核对**：本块原文中没有独立代码块，请按文中说明的步骤顺序实际执行，并记录每一步的输出结果。

2. **异常观察**：若执行过程中出现非预期输出，请自行记录报错现象、复现步骤和已尝试的修复动作，再带着这些信息在答疑区向导师提问。

3. **字段自查**：选你执行输出里出现的任意 3 个关键字段，回到本块正文中找到它们的解释，再用 1 句话写下你自己的理解。

4. **流程与顺序自检**：如果本块涉及多条命令，说明你实际执行的顺序是否与文中描述一致？如果不一致，会导致什么后果？


### 实训条件提示 · 你的设备
已记录你的实训条件：RTX 3060 12GB。示例会按本地 NVIDIA GPU 可执行的参数和验证方式组织，并保留与本知识点直接相关的最小步骤。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 7

### 3.1 是否需要单独安装？

`tokenizers` 是 `transformers` 的**依赖项**——也就是说，你装 `transformers` 时，它会被**自动带上**，一般不用手动单独装。

验证一下是否已经装好，直接跑：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
# 预期输出类似：tokenizers: 0.x.x（实际版本号以你环境为准）
```

能打印出版本号，就说明自动安装成功了。

### 3.2 什么情况下需要手动安装？

两种场景你可能需要手动装：一是你要**独立使用** `tokenizers`（比如自己训练一个词表），二是自动装的版本不满足你的需求。这时指定安装即可：

```bash
pip install tokenizers
```

> **踩坑提醒**：某些旧版本 tokenizers 在特定 Python 环境下可能需要从源码编译，进而依赖 Rust 工具链。但本课程用的 Python 3.10 + 较新版 tokenizers 通常有预编译的 wheel 包，直接装就行，不用碰 Rust。

**本块与当前学习目标的关系**：你后续做 LoRA 微调和 vLLM 部署时，模型加载和推理都依赖 tokenizer 来把文本转成模型能读的 token 序列。确认 tokenizers 已随 transformers 装好，等于提前排掉一个环境隐患，后面跑微调和部署时少一个报错点。如果刚才验证命令没打印出版本号，或者报错，先记下来，下一步排查时用。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

**部署补充说明**（供后续微调与部署时参考）：

- 模型许可证：若使用 Llama 2 系列模型，需遵守 Llama 2 Community License；商用且月活用户超 7 亿时需获取 Meta 授权。
- 显存要求：以 Llama-2-7b 为例，FP16 精度加载至少需要 14GB 显存，部署前请确认硬件条件。

## 块 8

安装后须验证transformers能否工作。最直接方式是用`pipeline`加载情感分析模型跑推理，同时确认库和底层依赖装对。

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this course!")
print(result)
```

输出类似`[{'label': 'POSITIVE', 'score': 0.99...}]`即正常。

**小任务**：结合目标（LoRA微调+权重合并+vLLM部署），说明为何验证安装是后续前提。不懂术语（如`pipeline`内部机制）先记下，稍后提问。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 9

### 4.1 快速验证脚本（①）

先跑通一个最小用例，确认 `transformers` 能正常加载并执行推理。在终端里运行：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

预期输出类似下面这样（具体数值随版本浮动）：

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

看到 `POSITIVE` 且分数接近 1，说明安装成功、模型能跑通。这个 `pipeline` 是 `transformers` 的极简入口，一句话完成"加载模型 + 预处理 + 推理 + 后处理"，后面做 LoRA 微调时你会频繁用到它来快速验证模型行为。

**本地运行配置要求**：首次执行时，`transformers` 会自动从 HuggingFace Hub 下载预训练模型（比如 `distilbert-base-uncased-finetuned-sst-2-english`，体积因版本而异，通常几百 MB）到本地缓存。需要**联网**，磁盘预留 1GB 以上空间，**CPU 即可运行**，无需 GPU，内存建议 ≥ 4GB。你的 RTX 3060 在这里用不上，但后面微调和部署阶段它会派上用场。

**模型许可证说明**：本块使用的 `distilbert-base-uncased-finetuned-sst-2-english` 为 Apache 2.0 许可证，可自由使用（包括商用），无月活用户数限制。

**显存要求说明**：本块仅做 CPU 推理验证，不涉及 GPU 加载，因此无显存要求。后续微调或部署阶段涉及 GPU 加载时，需按具体模型另行评估显存占用（例如 Llama-2-7b FP16 需要至少 14GB 显存）。

### 4.2 查看安装版本

确认两个核心库的版本号，方便后续排查问题：

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

执行后终端会打印两行版本号。记下它们，后面装 vLLM 或做 LoRA 微调时如果遇到兼容性问题，版本号是排查的第一步。

**本块与学习目标的关系**：你现在验证的是 `transformers` 的基础链路——模型下载、加载、推理。这是后续 LoRA 微调（`KB-12`）和 vLLM 部署（`KB-15`）的地基：微调要在这个库上加载预训练模型并注入 LoRA 适配器，部署则要把微调后的模型交给 vLLM 托管。如果这里跑不通，后面两步无从谈起。

**记录任务**：执行过程中如果遇到报错或看不懂的术语（比如 `pipeline` 内部机制、`distilbert` 是什么），先记下来，不要卡住。本块只需确认"能跑通 + 版本号可见"，细节留到后续块再深入。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 10

## 五、配置模型缓存目录（①）

用 `from_pretrained()` 加载模型时，transformers 会自动从 HuggingFace Hub 下载文件并缓存到本地，默认路径是 `~/.cache/huggingface/hub`。如果你的系统盘空间紧张，建议改掉这个默认位置。

**怎么改**：在代码里设置环境变量 `HF_HOME`（或 `TRANSFORMERS_CACHE`），指向你希望存放模型文件的目录，再执行加载即可。比如：

```python
import os
os.environ["HF_HOME"] = "/你的目标路径"
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
```

**验证方式**：设置后重新运行一次 `from_pretrained()`，观察终端输出中下载路径是否已指向新目录；或者直接去新路径下确认是否生成了 `hub` 文件夹。

**和你当前目标的关系**：你接下来要做 LoRA 微调和 vLLM 部署，都会反复加载基座模型。提前把缓存指到空间充足的盘（比如你的 RTX 3060 机器上非系统盘），能避免后续下载大模型时系统盘被占满。如果遇到 `HF_HOME` 相关报错或路径不生效，先记下来，下一步配置时再排查。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 11

### 5.1 通过环境变量配置（①）

先直接看操作，两条命令解决临时和永久两种场景：

```bash
# 临时生效（当前终端会话）
export HF_HOME="/mnt/data/huggingface"

# 永久生效（写入 ~/.bashrc）
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

执行完临时生效那条后，可以用 `echo $HF_HOME` 确认变量已设置；永久生效那条执行完 `source` 后，新开的终端里同样能读到这个值。

**配置优先级**（从高到低）：

1. `HF_HUB_CACHE`（最优先）
2. `HF_HOME`
3. `XDG_CACHE_HOME` + `/huggingface`

> **建议**：如果你用云服务器，把 `HF_HOME` 指到数据盘（如 `/mnt/data`），避免系统盘被模型文件占满。后面"参数高效微调"和"模型推理部署"要加载的模型，体积从数 GB 到数十 GB 不等——你后续做 LoRA 微调和 vLLM 部署时，这一步就是给它们腾地方。

### 5.2 在 Python 代码中查看当前缓存路径（①）

```python
from transformers import utils
print(utils.default_cache_path)
```

跑完这段，终端会打印出当前实际的缓存目录，用来验证上面环境变量是否生效。

**模型许可证与显存提示**：后续部署涉及模型加载，需注意两点。一是模型许可证：以 Llama 2 系列为例，采用 Llama 2 Community License，商用场景下月活用户超过 7 亿需获得 Meta 授权。二是显存要求：以 Llama-2-7b FP16 为例，加载至少需要 14GB 显存，部署前请确认 GPU 资源充足。

**本块与学习目标的关系**：你最终要做 LoRA 微调 + 权重合并 + vLLM 部署，这几个环节都会从 Hugging Face 下载模型权重。提前把缓存路径指到数据盘，相当于给后续所有下载操作定好存放位置，避免中途发现系统盘不够用再返工。如果对 `HF_HUB_CACHE` 和 `HF_HOME` 的具体分工还有疑问，先记下来，后面实操时再对照验证。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 12

### 六、离线模式配置（拓展）（①）

后续在内网或防火墙环境里用预训练模型时，transformers 支持完全离线运行，前提是**提前下载好所需文件**。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**验证方式**：配置好后，在断网环境下运行加载模型的代码，能正常完成加载即说明离线配置生效。

**与学习目标的关系**：你后续做 LoRA 微调和 vLLM 部署时，模型文件可能需要在隔离环境中准备，这一步确保你在无网条件下也能顺利加载模型，是微调和部署链路的基础保障。

遇到无法解释的术语（如"防火墙环境"的具体限制）时，先记录下来，稍后提问。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 13

### 6.1 预先下载模型（①）

在联网环境下，先把模型仓库完整拉到本地，后面离线时直接用本地文件，不再发起任何网络请求。

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
# 模型许可证：bert-base-chinese 采用 Apache 2.0 许可证，可商用
snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
```

执行完，你会在当前目录下看到 `bert-base-chinese` 文件夹，里面包含模型权重、配置文件等完整仓库内容。可以用 `ls ./bert-base-chinese` 确认文件是否齐全。

### 6.2 离线加载（①）

离线场景下有两种加载方式，任选其一：

**方式一：设置环境变量阻止所有 HTTP 请求**

```bash
HF_HUB_OFFLINE=1 python your_script.py
```

运行脚本时加上这个环境变量，transformers 会完全跳过网络检查，直接走本地缓存。验证方式：脚本启动时不会出现任何下载进度条或网络连接提示。

**方式二：在代码中指定只加载本地文件**

```python
from transformers import AutoModel

# 注意：bert-base-chinese FP16 加载约需 400MB 显存
model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
```

`local_files_only=True` 强制只从本地路径加载，如果本地没有对应文件会直接报错而不是尝试联网下载。验证方式：断网状态下运行这段代码，模型能正常加载即说明离线配置生效。

**与当前学习目标的关系**：你的目标是 LoRA 微调 + 权重合并 + vLLM 部署，整个流程需要多次加载同一个基础模型（微调前加载、合并权重后加载、部署时再加载）。如果每次都联网下载，既慢又容易中断。提前用 `snapshot_download` 把模型拉到本地，后续所有步骤都用 `local_files_only=True` 或 `HF_HUB_OFFLINE=1` 离线加载，保证流程稳定可复现。

**待记录术语**：如果对 `repo_id` 的命名规则（如 `bert-base-chinese` 中的组织名/模型名结构）或 `snapshot_download` 的增量更新机制有疑问，先记下来，后续遇到具体报错时再针对性查证。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

### 七、常见问题与解决（①+②+③+④）

在安装 `transformers` 和 `tokenizers` 的过程中，你可能会遇到下面这些典型问题。按表格对照排查，能省下不少时间。

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `pip install transformers` 速度极慢或超时 | 默认源在国外 | 使用清华镜像源：`-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found` | Python 版本过低或过高 | 确保 Python 3.10+（本课程使用 3.10） |
| 导入 transformers 时报 `signal` 模块错误 | transformers 版本与 Python 不兼容 | 升级 transformers 到最新版：`pip install -U transformers` |
| `tokenizers` 安装时提示需要 Rust 编译器 | 该版本无预编译 wheel，需从源码构建 | 换用 conda 安装：`conda install -c huggingface tokenizers`；或升级 pip 以获取新版预编译包 |
| `pipeline()` 首次运行卡住不动 | 从 HuggingFace Hub 下载模型，网络不通 | 检查网络；或预先使用 `snapshot_download()` 下载到本地 |
| 磁盘空间不足 | 缓存目录在系统盘，模型文件积累过多 | 设置 `HF_HOME` 到数据盘；定期清理 `~/.cache/huggingface/hub` |
| 验证脚本输出结果与预期不符 | 模型下载不完整 | 删除缓存重新下载，或设置 `local_files_only=False` 强制校验 |

**遇到问题时，按这个顺序排查**：先确认 `python --version` ≥ 3.10 → 再确认 `torch.__version__` 版本与预期一致 → 最后确认 `transformers.__version__` 为较新稳定版。


**实操建议**：执行完安装命令后，直接在终端里运行 `python -c "import transformers; print(transformers.__version__)"`，能看到版本号就说明安装成功。如果报错，对照上面的表格找对应解法。

**与学习目标的关系**：你接下来的目标是 LoRA 微调 + 权重合并 + vLLM 部署，整个流程都建立在 `transformers` 和 `tokenizers` 能正常工作的基础上。这一节排查的是环境层面的坑——如果安装或导入阶段就卡住，后面的微调和部署都没法跑起来。所以先花几分钟把环境弄干净，后面会更顺。

**记录一下**：如果遇到表格里没覆盖的报错，把完整的错误信息记下来，后面提问时带上它，能更快定位问题。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 15

## 八、总结与后续学习路线（①）

环境准备这一步已经收尾了。按你的目标——**LoRA 微调 + 权重合并 + vLLM 部署**——今天装好的这两个库是整个流程的地基，后面每一步都会踩在它们上面。

你现在应该已经确认过这几件事：

- ✅ 在 `pytorch` conda 环境中成功安装了 `transformers` 和 `tokenizers`
- ✅ 验证了 `pipeline('sentiment-analysis')` 能正常运行并输出结果
- ✅ 理解了模型缓存机制，能配置 `HF_HOME` 自定义缓存路径
- ✅ 了解了离线模式的基本配置方法

如果你已经逐项勾掉，说明环境是通的。后面做 LoRA 微调时，`transformers` 负责加载模型和训练循环，`tokenizers` 负责把文本切成模型能吃的 token——这两件事今天已经验证过能跑，后面就不会卡在环境上。

**下一步预告**：在下一节 **"理解 BPE、WordPiece、SentencePiece 算法原理"** 中，我们将深入讲解现代大语言模型使用的子词切分算法。你会了解为什么 "unhappiness" 会被切分成 `["un", "happiness"]` 而不是按单词或字母切分，以及这些算法如何用统计方法从语料库中"学习"出最优的词表。随后进入 **"掌握 HuggingFace Tokenizer 的加载与使用"**，我们将用今天装好的 `transformers` 库加载真实的预训练分词器（如 GPT-2 的 BPE 分词器），亲手观察切分结果。

**这个环节和你的目标有什么关系**：微调 LoRA 时，你加载的模型自带分词器，文本进出都要经过它；部署到 vLLM 时，分词器同样参与请求的解析和响应的生成。所以理解 tokenizer 怎么切词，能帮你判断"为什么这个 prompt 被处理成那样"以及"为什么某些词切分后占的 token 数不一样"——这些在调试微调效果和推理输出时都会遇到。

**部署提醒**：后续部署到 vLLM 时，注意你加载的模型（如 Llama 系列）有对应的许可证要求，例如 Llama 2 Community License 规定商用月活超 700 万用户需获得 Meta 授权；同时模型加载对显存有要求，例如 Llama-2-7b 以 FP16 精度加载至少需要 14GB 显存，部署前请确认硬件配置。

现在，确保你的环境已就绪，我们马上开始！


**本块任务**：用自己的话写一两句，说明"环境准备"和你最终目标（LoRA 微调 + 权重合并 + vLLM 部署）之间的关系。如果遇到解释不了的术语（比如 BPE、WordPiece），先记下来，下一节讲算法原理时带着问题去听。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 测评题目

### Q-KB07-ST0700-003
安装 Hugging Face Transformers 的常用命令是？

### Q-KB07-ST0700-001
Hugging Face 模型与分词器常用的 Python 库名称是________。

### Q-KB0701-001
Byte Pair Encoding (BPE) 算法的核心思想是？

### Q-KB0703-001
当设置 `truncation=True` 时，如果输入序列长度超过模型的最大长度，分词器会如何处理？

### Q-KB0703-004
`padding=True` 和 `padding='longest'` 的效果是完全一样的。（ ）

### Q-KB0703-003
对于一个最大序列长度为 512 的模型，执行 `tokenizer(text, max_length=100, padding='max_length', truncation=True)` 后，输出序列的长度是多少？

### Q-KB0701-002
WordPiece 算法与 BPE 的主要区别在于合并标准不同。WordPiece 选择合并哪一对符号？

### Q-KB0702-002
使用已加载的 `tokenizer` 对象，对字符串 `"Hello, world!"` 进行分词，并打印出 token ID 列表。
