# 学习包

## 块 1

## 一、前置检查：确认 PyTorch 环境就绪

在进入 `transformers` 与 `tokenizers` 的安装之前，先花一分钟确认你的 PyTorch 环境仍然可用。这一步不是重复学习，而是为了把"环境变量"和"Python 包路径"这两个概念对齐——因为后续所有安装操作都发生在同一个 conda 环境里，如果环境错乱，装再多库也无法被正确导入。

**最小回顾**：你之前已经完成过 Python 3.10+ 的安装、conda 虚拟环境的创建与激活、IDE（PyCharm 或 VS Code）的配置，以及 pip 国内镜像源的加速设置。这些前置工作的核心目的只有一个：让 `pip install` 安装的包落到**当前激活的 conda 环境**对应的 site-packages 目录中，而不是系统全局 Python 里。如果你在 IDE 终端里执行 `python` 后看到的路径与你 conda 环境名不一致，说明环境没有激活，这是最常见的"装完却 import 不到"的根源。

**快速确认命令**（在 `pytorch` conda 环境中执行）：

```bash
python -c "import torch; print(torch.__version__)"
```

如果输出类似 `2.5.1` 的版本号，说明 PyTorch 就绪。如果报错 `ModuleNotFoundError`，请回到"环境准备：根据 CUDA 版本安装 PyTorch"章节排查——大概率是 conda 环境未激活或 PyTorch 安装时 CUDA 版本与驱动不匹配。

**原理说明**：`import torch` 之所以能成功，是因为 Python 解释器在启动时会按照 `sys.path` 中的路径顺序搜索模块，而 `sys.path` 的首项通常指向当前 Python 解释器所在环境的 site-packages 目录。conda 环境激活的本质是修改了 `PATH` 环境变量，使 `python` 命令指向该环境专属的解释器，从而让 `sys.path` 自动指向对应的 site-packages。这正是"环境变量决定包路径"这一机制的核心——`pip install` 将包写入的目录，与 `import` 时搜索的目录，必须经由同一个解释器路径才能对齐。若存在多个 Python 解释器（如系统自带 Python 与 conda Python 并存），`PATH` 的解析顺序一旦错位，就会出现 `pip list` 有 torch 但 `import` 失败的路径错位现象。

确认 PyTorch 可用后，就可以继续安装 `transformers` 和 `tokenizers` 了。这里需要理解两个库的分工：`transformers` 是 HuggingFace 的核心库，封装了大量预训练模型（如 BERT、GPT-2、Llama、Qwen 等），提供统一的加载与推理接口；`tokenizers` 是其底层的高性能分词库，基于 Rust 实现，负责将文本拆分为 token 序列——这是后续学习 BPE/WordPiece 算法时直接打交道的组件。两者是"上层接口 + 底层引擎"的关系，安装时 `transformers` 会自动依赖 `tokenizers`，但显式安装可以确保版本匹配。

**本块学习检查**：先用一个最小示例回顾（环境准备：安装Python 3.10+，配置conda虚拟环境，安装IDE如PyCharm或VS Code，配置pip国内镜像源加速下载）；确认能解释关键写法后再继续本块。

## 块 2

### 1.1 确认 conda 环境和 PyTorch 版本

```bash
conda activate pytorch

python -c "import torch; print('PyTorch:', torch.__version__)"
# 预期输出类似如下，实际以版本为准：PyTorch: 2.5.x
```

**版本要求**：本课程在 Python 3.10 + PyTorch 2.5.1 环境下验证通过。不同版本的 transformers 对 Python 和 PyTorch 版本要求可能不同，建议以安装时的官方文档为准。以下是本课程验证过的依赖版本示例（`requirements.txt`）：

```text
torch==2.5.1
transformers>=4.45.0
tokenizers>=0.20.0
```

如果你按照前面板块完成了环境搭建，当前环境已满足上述要求。

**原理说明**：LoRA 微调本质上是基于 PyTorch 的模型参数操作，transformers 库在导入时会动态检测底层 PyTorch 的 API 兼容性。版本不匹配（如 PyTorch 过旧或过新）会导致 transformers 内部某些算子调用失败，这种错误往往在训练中途才暴露，排查成本远高于事前确认。因此，先锁定 conda 环境再验证 torch 版本，是为了把"环境问题"和"代码问题"两个变量分离。

**概念拆解**：`conda activate pytorch` 切换的是虚拟环境，而 `python -c "import torch"` 验证的是该环境内实际可用的 torch 版本——前者决定"你在哪"，后者确认"那里有什么"。

**思考题**：为什么本课程要求先激活 conda 环境再执行版本检查？（提示：不激活环境直接运行 python，可能导入的是系统全局的 torch，而非课程验证过的版本。）

### 1.2 确认 pip 可用

```bash
pip --version
# 预期输出类似如下，实际以版本为准：pip 24.x.x
```

**原理说明**：pip 是后续安装 transformers 和 tokenizers 的唯一入口。确认 pip 可用，本质上是确认"安装通道"通畅——如果 pip 本身损坏或指向了错误的 Python 解释器，后续 `pip install` 会静默安装到错误位置，导致 import 时找不到包。这一步与 1.1 配合，构成了"环境 + 安装工具"的双重前置校验。

**思考题**：如果 `pip --version` 输出的路径与当前 conda 环境的 Python 路径不一致，可能引发什么问题？（提示：安装的包会进入哪个 site-packages？）

## 块 3

# 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） · 本块学习卡
## 概念与设计动机：为什么需要（环境准备：在Python环境中安装transformers、tokenizers库）
### 本块必读锚点 · 概念卡片（阅读正文前先浏览，正文中遇到就填）
| 序号 | 原文锚点（术语 / 字段 / 加粗词） | 在正文中第几次出现时被你抓到？ | 你用 ≤30 字自证其含义（不得查外部资料） | 本锚点和其他锚点的关系？ |
|---|---|---|---|---|
| 1 | （后续在正文里自行补充） |  |  |  |
| 2 | （后续在正文里自行补充） |  |  |  |
| 3 | （后续在正文里自行补充） |  |  |  |
| 4 | （后续在正文里自行补充） |  |  |  |


> 🎯 **你的画像 = 理论型 learner**：本块**强制阅读顺序与节奏**（不要跳步）：
> 1. 先完整通读『概念说明段』和所有『列表 / 定义 / 对比条目』至少 2 遍，**不先看代码块**；
> 2. 列表条目部分标记为『要点卡片』，在上面的概念卡片里逐行填笔记；
> 3. 代码块仅做『对照原理阅读 · 句法分析』，不必立即在你机器上运行；
> 4. 全部通读完后，完成文末『画像专属 · 本块学习检查』和下面的『思考提示』。

### 思考提示（无需查资料，仅用本块正文作答；写不出来就回上文找原句）
> *说明：这里只列『思考题答题指引 + 引用原文锚点』，不给答案，答案由你读完正文后自己填写。*
1. **思考题 1 · 概念边界**：基于正文找出 2 条使用本知识点时的限制或前置条件（必须来自原文，不得脑补）。
2. **思考题 2 · 术语对照**：从上面「本块必读锚点」概念卡片中任选 2 个词，写出它们若被混淆会导致的后果（以原文中明确提到的差异为准）。
3. **思考题 3 · 流程/句法分析**：若本块有代码，回到代码块做句法分析；若无代码，则回到正文第一条列表条目做结构重写（不抄原句）。

> ⚙️ 实训条件：RTX 3060 12GB；本块所属知识板块：『子词切分与分词器（Tokenizer）』。


## 二、安装 transformers 库（①）


## 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） 本块学习检查
### 概念自证 · 思考题作答（答案必须能用正文原句回证）
1. **概念概括**：用你自己的话概括『（环境准备：在Python环境中安装transformers、tokenizers库）』在本块中被引入的核心原因，最多写 3 个要点，并分别回正文标出对应原句。

2. **适用边界说明**：根据本块正文，写出 2 条**不适合使用本块介绍的方法 / 命令**的场景（场景必须来自正文明确提到的限制或反例，不得凭空脑补）。

3. **流程拆解**：回到正文中出现的命令 / 流程段，用编号步骤写出『执行后系统内部依次做了什么』的最小流程；每一步必须能在正文里找到依据（用自己的话写，不要摘抄）。

4. **列表类条目结构默写**：回到正文里第一条『定义 / 对比 / 枚举』型列表，用你自己的话写出条目维度和关键限定词（不要摘抄）。


### 实训条件提示 · 你的设备
已记录你的实训条件：RTX 3060 12GB。示例会按本地 NVIDIA GPU 可执行的参数和验证方式组织，并保留与本知识点直接相关的最小步骤。

## 块 4

### 2.1 标准安装（推荐）

`transformers` 库通过 pip 直接安装。由于你已在 PyTorch 环境中完成基础配置，只需一条命令即可拉取库及其依赖：

```bash
pip install transformers
```

**安装原理**：pip 是 Python 生态的标准包管理器，`transformers` 以 wheel 形式分发，安装过程会自动解析并拉取依赖（如 tokenizers、regex 等）。相比源码编译，pip 安装免去手动处理依赖树的成本，且版本冲突由 pip 的解析器统一协调——这是它在多数场景下成为首选路径的原因。

> **本地运行配置要求**：此步骤需联网下载安装包（体积因版本而异，通常在数百 MB 量级），对电脑配置无特殊要求，CPU 即可执行。若网络较慢，可参考下方镜像源加速。

**如果网络较慢，可使用清华镜像源加速**：

```bash
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**镜像源原理**：默认 PyPI 源与清华镜像源安装结果完全一致，区别仅在下载服务器位置；镜像源通过国内 CDN 缩短传输路径，适合网络延迟高的场景，但不改变包版本或依赖解析逻辑。

## 块 5

### 2.2 带 PyTorch 依赖的安装（可选）

如果你是在一个**全新的空环境**中安装（非本课程场景），可以用 extras 方式一次性安装 transformers 及其 PyTorch 适配依赖。这里的核心设计动机是：`transformers` 本身只是一个模型加载与推理的框架层，真正执行张量运算的底层引擎（如 PyTorch）需要单独安装。从依赖关系上看，`transformers` 在运行时需要调用 PyTorch 提供的张量操作接口来完成模型的前向传播与反向传播，因此两者之间存在明确的“框架—引擎”依赖层级。extras 语法 `[torch]` 的作用，是让 pip 在安装 transformers 的同时，自动解析并拉取与之兼容的 PyTorch 版本，省去你手动对齐两者版本匹配关系的步骤——这在深度学习生态中是一个常见的依赖管理痛点，因为框架与底层库的版本不匹配往往会导致运行时莫名其妙的报错。

```bash
pip install 'transformers[torch]'
```

> 本课程中 PyTorch 已提前装好，直接 `pip install transformers` 即可，无需重复安装 torch。

**概念对比**：`pip install transformers` 与 `pip install 'transformers[torch]'` 的区别在于依赖解析的粒度——前者只装 transformers 本体，后者连 PyTorch 一起装；若环境中已有 torch，前者即可，后者反而可能触发重复安装。

**思考题**：为什么在全新空环境中推荐使用 `'transformers[torch]'` 而不是分别安装两个库？答案要点：extras 方式能一次性解决 transformers 与 PyTorch 的版本兼容匹配问题，避免手动对齐的麻烦。

### 2.3 使用 conda 安装（可选）

如果你更习惯 conda，也可以从 conda-forge 通道安装：

```bash
conda install conda-forge::transformers
```

> **注意**：conda-forge 中的版本可能略滞后于 pip，建议优先使用 pip 安装以获得最新版本。

**思考题**：conda-forge 通道与 pip 在版本更新上的差异是什么？答案要点：conda-forge 中的版本可能略滞后于 pip，因此建议优先使用 pip 安装以获得最新版本。

**模型许可证与显存说明**：若后续部署 Llama 2 系列模型，需注意其许可证为 Llama 2 Community License（商用月活>7亿需Meta授权）；同时，加载 Llama-2-7b FP16 权重至少需要14GB显存，请提前确认硬件条件。

## 块 6

# 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） · 本块学习卡
## 概念与设计动机：为什么需要（环境准备：在Python环境中安装transformers、tokenizers库）
### 本块必读锚点 · 概念卡片（阅读正文前先浏览，正文中遇到就填）
| 序号 | 原文锚点（术语 / 字段 / 加粗词） | 在正文中第几次出现时被你抓到？ | 你用 ≤30 字自证其含义（不得查外部资料） | 本锚点和其他锚点的关系？ |
|---|---|---|---|---|
| 1 | （后续在正文里自行补充） |  |  |  |
| 2 | （后续在正文里自行补充） |  |  |  |
| 3 | （后续在正文里自行补充） |  |  |  |
| 4 | （后续在正文里自行补充） |  |  |  |


> 🎯 **你的画像 = 理论型 learner**：本块**强制阅读顺序与节奏**（不要跳步）：
> 1. 先完整通读『概念说明段』和所有『列表 / 定义 / 对比条目』至少 2 遍，**不先看代码块**；
> 2. 列表条目部分标记为『要点卡片』，在上面的概念卡片里逐行填笔记；
> 3. 代码块仅做『对照原理阅读 · 句法分析』，不必立即在你机器上运行；
> 4. 全部通读完后，完成文末『画像专属 · 本块学习检查』和下面的『思考提示』。

### 思考提示（无需查资料，仅用本块正文作答；写不出来就回上文找原句）
> *说明：这里只列『思考题答题指引 + 引用原文锚点』，不给答案，答案由你读完正文后自己填写。*
1. **思考题 1 · 概念边界**：基于正文找出 2 条使用本知识点时的限制或前置条件（必须来自原文，不得脑补）。
2. **思考题 2 · 术语对照**：从上面「本块必读锚点」概念卡片中任选 2 个词，写出它们若被混淆会导致的后果（以原文中明确提到的差异为准）。
3. **思考题 3 · 流程/句法分析**：若本块有代码，回到代码块做句法分析；若无代码，则回到正文第一条列表条目做结构重写（不抄原句）。

> ⚙️ 实训条件：RTX 3060 12GB；本块所属知识板块：『子词切分与分词器（Tokenizer）』。


## 三、安装 tokenizers 库（①+②）


## 画像专属 · （环境准备：在Python环境中安装transformers、tokenizers库） 本块学习检查
### 概念自证 · 思考题作答（答案必须能用正文原句回证）
1. **概念概括**：用你自己的话概括『（环境准备：在Python环境中安装transformers、tokenizers库）』在本块中被引入的核心原因，最多写 3 个要点，并分别回正文标出对应原句。

2. **适用边界说明**：根据本块正文，写出 2 条**不适合使用本块介绍的方法 / 命令**的场景（场景必须来自正文明确提到的限制或反例，不得凭空脑补）。

3. **流程拆解**：回到正文中出现的命令 / 流程段，用编号步骤写出『执行后系统内部依次做了什么』的最小流程；每一步必须能在正文里找到依据（用自己的话写，不要摘抄）。

4. **列表类条目结构默写**：回到正文里第一条『定义 / 对比 / 枚举』型列表，用你自己的话写出条目维度和关键限定词（不要摘抄）。


### 实训条件提示 · 你的设备
已记录你的实训条件：RTX 3060 12GB。示例会按本地 NVIDIA GPU 可执行的参数和验证方式组织，并保留与本知识点直接相关的最小步骤。

## 块 7

### 3.1 是否需要单独安装？

`tokenizers` 通常是 `transformers` 的**依赖项**，安装 `transformers` 时会**自动安装**兼容版本的 `tokenizers`，一般不需要手动单独安装。

你可以验证是否已自动安装：

```bash
python -c "import tokenizers; print('tokenizers:', tokenizers.__version__)"
# 预期输出类似如下，实际以版本为准：tokenizers: 0.x.x
```

**原理说明**：这一自动安装机制源于 Python 包管理中的依赖解析规则——`transformers` 在其包元数据（`install_requires`）中声明了对 `tokenizers` 的版本约束，包管理器据此自动拉取满足约束的版本。其设计目的是保证 `transformers` 运行时依赖的 tokenizer 版本与自身 API 兼容，避免因版本不匹配导致分词行为异常。你只需关注 `transformers` 的安装，依赖关系由包管理器统一解析。

**概念拆解**：`transformers` 与 `tokenizers` 的关系是“主库 + 依赖库”，前者负责模型加载与推理流程，后者专职文本到 token ID 的转换；两者版本需配套，但你不必手动维护后者。

### 3.2 什么情况下需要手动安装？

如果你需要**独立使用** `tokenizers` 库（例如自定义训练一个词表），或者自动安装的版本不符合需求，可以手动指定版本：

```bash
pip install tokenizers
```

> **踩坑提醒**：参考文章记录，某些旧版本 tokenizers 在特定 Python 环境下可能需要从源码编译，进而依赖 Rust 工具链。但本课程使用的 Python 3.10 + 较新版 tokenizers 通常有预编译的 wheel 包，可直接安装，无需 Rust。

**原理说明**：手动安装的适用场景可从依赖管理机制来理解——自动安装只保证“存在一个可用版本”，但当你的使用场景超出 `transformers` 的默认依赖范围（如直接调用 tokenizer 训练 API），或项目约束要求锁定特定版本时，就需要显式声明安装。这本质上是包管理器中“隐式依赖”与“显式依赖”两种声明方式的区别：前者由主库代管，后者由你直接控制。

**概念拆解**：自动安装 vs 手动安装——触发时机不同：前者随 `transformers` 安装隐式完成，后者由你显式执行 `pip install`；控制粒度也不同：前者只保证兼容版本存在，后者可精确指定版本号。

## 块 8

### 四、验证安装：运行你的第一个 HuggingFace 模型（①）

安装后需验证 transformers 是否正常。最简单用 `pipeline` 加载情感分析模型推理。核心动机：`pipeline` 封装分词、模型前向传播、输出后处理全流程，无需手动管理加载和推理细节，快速确认环境依赖（如 tokenizers、深度学习框架）正确衔接。

**概念对比**：`pipeline` 返回可读标签和置信度；直接调用模型类（如 `AutoModelForSequenceClassification`）返回原始 logits，需额外处理。

**思考题**：
1. 为何 `pipeline` 适合作首个验证工具？
2. 用 `pipeline` 做情感分析时，内部自动完成哪三个主要环节？

## 块 9

### 4.1 快速验证脚本（①）

在终端执行下面这一行命令，它同时完成了“加载模型”和“跑一次推理”两件事：

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

拆开看这行命令的设计动机：`pipeline` 是 HuggingFace 提供的高层封装，它把“下载模型 → 加载权重 → 预处理文本 → 推理 → 后处理输出”整条链路压缩成一次函数调用。你不需要手动指定模型结构或权重路径，`pipeline('sentiment-analysis')` 会自动选择一个适合情感分析任务的预训练模型并完成加载。对刚完成安装的环境来说，这是验证“库是否真正可用”的最短路径——如果这行能跑通，说明 `transformers` 及其依赖（如 `torch`、`tokenizers`）都已正确就位。

**预期输出类似如下，实际以版本为准**：（①）

```
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

> **示例输出，实际值可能略有不同**。输出是一个列表，里面是一个字典，包含两个字段：`label`（分类结果）和 `score`（置信度）。注意 `score` 接近 1.0，说明模型对“we love you”判定为正面情绪非常自信。

> **本地运行配置要求**：首次运行此命令时，`transformers` 会自动从 HuggingFace Hub 下载预训练模型（如 `distilbert-base-uncased-finetuned-sst-2-english`，体积因版本而异，通常在数百 MB 量级）到本地缓存。需要**联网**，磁盘建议预留 1GB 以上空间，**CPU 即可运行**，无需 GPU，内存建议 ≥ 4GB。

> **模型许可证**：本示例使用的 `distilbert-base-uncased-finetuned-sst-2-english` 模型遵循 Apache 2.0 许可证，允许商用及修改，无需额外授权。

这里有个值得留意的 trade-off：`pipeline` 的“零配置”便利性，代价是首次运行时的模型下载时间和磁盘占用。模型文件会缓存在本地，后续再跑同一任务就不会重复下载。你手头的 RTX 3060 12GB 在这个环节暂时用不上——情感分析这个小模型在 CPU 上也就几秒的事，GPU 的显存优势要到后面微调大模型时才会体现。

**概念对比**：`pipeline('sentiment-analysis')` 返回的是**结构化字典列表**（含 `label` 和 `score`），而不是裸的字符串或数字——这种返回粒度让你可以直接用 `result[0]['score']` 提取置信度做后续判断。

思考题（答案都能从上面原文回证）：
1. 为什么首次运行这行命令需要联网？
2. 输出结果中 `score` 字段的含义是什么？它的值接近 1.0 说明什么？

### 4.2 查看安装版本

```bash
python -c "
import transformers
import tokenizers
print('transformers 版本:', transformers.__version__)
print('tokenizers 版本:', tokenizers.__version__)
"
```

这段脚本的作用是确认两个核心库的版本号。`__version__` 是 Python 包的标准属性，直接打印即可。为什么要单独验证 `tokenizers`？因为它是 `transformers` 的底层依赖，负责把文本切分成 token（模型能理解的输入单位）。如果 `transformers` 能正常导入但 `tokenizers` 版本不兼容，后续跑模型时会出现难以排查的报错——所以安装后先确认两者版本，是成本最低的排障手段。

**概念对比**：4.1 的 `pipeline` 验证的是“端到端功能是否可用”，4.2 的版本打印验证的是“依赖版本是否匹配”——前者测功能，后者测环境，两者互补，缺一不可。

思考题：为什么在 4.1 已经能跑通模型的情况下，还要单独查看 `tokenizers` 的版本？

## 块 10

### 五、配置模型缓存目录（①）

调用 `from_pretrained()` 时，transformers 从 HuggingFace Hub 下载并缓存至 `~/.cache/huggingface/hub`。系统盘紧张时可改路径。

**动机**：缓存避免重复下载，用磁盘换带宽。默认路径无需配置，但模型数 GB 会挤占空间。

**思考题**：默认缓存路径？为何修改？

**许可证**：Llama 2 Community（商用月活>7亿需授权）。

**显存**：Llama-2-7b FP16 需≥14GB。

## 块 11

### 5.1 通过环境变量配置（①）

Hugging Face 生态在下载模型时会默认写入用户主目录下的缓存文件夹。你可以通过设置环境变量来改变这个默认位置，核心变量是 `HF_HOME`，其默认值为 `~/.cache/huggingface`。下面两条命令分别对应“临时生效”和“永久生效”两种场景：

```bash
# 临时生效（当前终端会话）
export HF_HOME="/mnt/data/huggingface"

# 永久生效（写入 ~/.bashrc）
echo 'export HF_HOME="/mnt/data/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

临时方式只对当前终端窗口有效，关闭后失效；永久方式把配置写入 shell 启动文件，每次打开终端都会自动加载。

当多个相关变量同时存在时，实际生效的路径按以下优先级从高到低决定：

1. `HF_HUB_CACHE`（最优先）
2. `HF_HOME`
3. `XDG_CACHE_HOME` + `/huggingface`

也就是说，如果显式设置了 `HF_HUB_CACHE`，它会覆盖 `HF_HOME` 的设置。`XDG_CACHE_HOME` 是 freedesktop 规范定义的缓存根目录，Hugging Face 会在其下追加 `/huggingface` 子目录；该变量仅在 `HF_HOME` 未设置时生效。

> **建议**：如果你使用云服务器，建议将 `HF_HOME` 设置到数据盘（如 `/mnt/data`），避免系统盘被模型文件占满。后续“参数高效微调”和“模型推理部署”章节加载的模型体积可达数 GB 至数十 GB。例如，Llama-2-7b FP16 需要至少 14GB 显存，模型文件本身也需相应磁盘空间。

模型文件动辄数 GB，系统盘通常空间有限且价格更贵，数据盘更适合存放这类大体积文件。提前规划缓存路径，可以避免后续训练或部署时磁盘空间不足。

> **模型许可证**：部署文档引用的 Llama 2 模型遵循 Llama 2 Community License，商用月活用户超过 7 亿时需获得 Meta 授权。

**思考题**：当 `HF_HUB_CACHE` 和 `HF_HOME` 同时被设置时，哪个变量最终决定缓存位置？

### 5.2 在 Python 代码中查看当前缓存路径（①）

配置完成后，你可以在 Python 中直接查看当前生效的缓存路径：

```python
from huggingface_hub import scan_cache_dir
print(scan_cache_dir().cache_dir)
```

该命令会返回当前环境实际使用的缓存目录字符串。你也可以通过 `print(os.environ.get('HF_HOME'))` 直接查看环境变量值，用于验证配置是否生效。

**概念对比**：`HF_HUB_CACHE` 是更细粒度的缓存目录变量，优先级更高；`HF_HOME` 是更通用的 Hugging Face 根目录，影响面更广。两者都未设置时，才回退到 `XDG_CACHE_HOME` 派生路径。

**思考题**：`scan_cache_dir().cache_dir` 返回的值，对应的是优先级列表中的哪一个变量（或变量组合）的解析结果？

## 块 12

### 六、离线模式配置（拓展）

后续在内网或防火墙环境中使用预训练模型时，transformers 支持完全离线运行，其核心前提是**提前下载好所需文件**。这一设计动机在于：离线环境无法实时访问 HuggingFace Hub，因此必须将模型、分词器、配置文件等一次性缓存到本地，之后加载时不再触发网络请求。从机制上看，在线模式每次加载可能检查远程更新，而离线模式固定使用本地缓存，二者在触发时机上存在本质差异；从权衡角度看，离线配置牺牲了首次获取的便利性，换取了运行时的稳定与隔离。

**思考题**：transformers 完全离线运行的前提条件是什么？为什么离线模式更适合内网或防火墙环境？

**概念对比**：在线模式 vs 离线模式——前者每次加载可能检查远程更新，后者固定使用本地缓存，触发时机不同。

## 块 13

### 6.1 预先下载模型（①）

离线模式的核心思路是：在联网环境下把模型文件完整拉到本地，之后所有加载操作都指向本地目录，彻底切断对网络的依赖。这一设计的原理在于：`from_pretrained` 在加载模型时，需要同时读取配置文件、分词器词表、权重文件等多个组件，这些组件在 HuggingFace 仓库中以特定目录结构组织。`snapshot_download` 的作用不是下载单个权重文件，而是把整个模型仓库（包括配置文件、分词器、权重等）按仓库结构镜像到本地，从而保证后续 `from_pretrained` 能按相同路径解析到完整的依赖文件。

```python
from huggingface_hub import snapshot_download

# 下载整个模型仓库到指定目录
snapshot_download(repo_id="bert-base-chinese", local_dir="./bert-base-chinese")
```

这里 `local_dir` 指定了本地存放路径，下载完成后该目录下的文件结构与 HuggingFace 仓库一致，这是离线加载能无缝衔接的前提。

### 6.2 离线加载（①）

下载完成后，有两种方式让代码只走本地、不发任何 HTTP 请求。

**方式一**：通过环境变量 `HF_HUB_OFFLINE=1` 全局拦截。设置后，HuggingFace 生态内所有库（transformers、tokenizers 等）的 Hub 访问都会被强制禁用，任何尝试联网的操作都会直接失败，而不是超时重试。

```bash
# 方式一：设置环境变量阻止所有 HTTP 请求
HF_HUB_OFFLINE=1 python your_script.py
```

**方式二**：在代码层面用 `local_files_only=True` 精确控制单次加载行为。这个参数告诉 `from_pretrained` 只检查本地路径，不发起任何网络请求；如果本地找不到对应文件，直接报错而不是尝试下载。

```python
# 方式二：在代码中指定只加载本地文件
from transformers import AutoModel

# 注意：模型许可证请查阅模型卡（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：显存要求因模型而异（如 Llama-2-7b FP16 需要至少14GB显存）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("./bert-base-chinese", local_files_only=True)
model = model.to(device)
```

**概念对比**：`HF_HUB_OFFLINE=1` 是进程级全局开关，影响整个脚本内所有 HuggingFace 调用；`local_files_only=True` 是单次加载的局部约束，只作用于当前 `from_pretrained` 调用。

## 块 14

## 七、常见问题与解决（①+②+③+④）

安装 transformers 和 tokenizers 的过程中，你可能会遇到一些典型问题。这些问题大多集中在网络源、Python 版本兼容性、预编译包缺失和模型下载这几个环节。下面按症状、原因、解决方案的对应关系逐一梳理。

**安装速度慢或超时**：`pip install transformers` 速度极慢或直接超时，通常是因为默认的 PyPI 源在国外，网络延迟高。解决方案是改用清华镜像源，在 pip 命令后追加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 即可。

**找不到匹配版本**：报 `No matching distribution found` 时，说明当前 Python 版本不在 transformers 支持的范围内，过高或过低都会触发。本课程使用 Python 3.10，确保你的 Python 版本在 transformers 支持的范围内，例如 Python 3.8 至 3.11，具体请查阅 transformers 官方文档。

**导入时报 signal 模块错误**：导入 transformers 时出现 `signal` 模块相关报错，本质是 transformers 版本与当前 Python 版本不兼容。处理方式是升级 transformers 到最新版：`pip install -U transformers`。

**tokenizers 需要 Rust 编译器**：安装 tokenizers 时提示需要 Rust 编译器，说明当前版本没有提供预编译的 wheel 包，pip 只能尝试从源码构建，而源码构建依赖 Rust 工具链。两个解决路径：一是换用 conda 安装 `conda install -c huggingface tokenizers`，conda 会直接拉取预编译包；二是升级 pip 到最新版本，并确保 Python 版本和平台受支持，可能有助于获取预编译 wheel，但并非绝对。

**pipeline() 首次运行卡住**：首次调用 `pipeline()` 时卡住不动，是因为它需要从 HuggingFace Hub 下载模型权重，而网络不通导致下载挂起。先检查网络连通性；也可以预先用 `snapshot_download()` 将模型下载到本地，后续运行直接读取本地文件。

**磁盘空间不足**：模型文件默认缓存在系统盘，多次下载不同模型后缓存会持续膨胀。通过设置环境变量 `HF_HOME` 将缓存目录指向数据盘，并定期清理 `~/.cache/huggingface/hub` 下的旧文件。

**验证脚本输出与预期不符**：模型文件下载不完整会导致推理结果异常。删除缓存后重新下载，或使用 huggingface_hub 的 `snapshot_download` 并设置 `force_download=True` 以确保文件完整。

> **最重要的排查口诀**：先确认 `python --version` ≥ 3.10 → 再确认 `torch.__version__` 版本与预期一致 → 最后确认 `transformers.__version__` 为较新稳定版。（①+③）

**概念对比**：`pip install -U transformers` 与 `conda install -c huggingface tokenizers` 的适用场景不同——前者解决的是 transformers 自身版本与 Python 的兼容问题，后者解决的是 tokenizers 缺少预编译 wheel 时的构建依赖问题。

**思考题**：
1. 为什么 `No matching distribution found` 通常与 Python 版本直接相关，而不是网络问题？
2. 当 tokenizers 提示需要 Rust 编译器时，升级 pip 为什么能解决问题？

## 块 15

## 八、总结与后续学习路线（①）

环境准备这一关你已经顺利通过了。现在你拥有的，是后续一切微调工作的物理前提：`pytorch` conda 环境里装好了 `transformers` 和 `tokenizers`，`pipeline('sentiment-analysis')` 能跑通并返回结果，模型缓存机制（`HF_HOME` 自定义路径）和离线模式的基本配置也已掌握。

从你的目标（LoRA 微调）来看，这一节的意义在于：**所有参数高效微调方法都建立在"能加载预训练模型、能操作分词器"的基础之上**。你刚验证的 `pipeline` 走的是封装好的高层接口，而 LoRA 微调需要你直接操作模型和分词器的底层 API——这正是下一步要补的衔接。

**下一步预告**：下一节进入 **"理解 BPE、WordPiece、SentencePiece 算法原理"**，讲解现代大语言模型的子词切分算法。你会明白为什么 `"unhappiness"` 被切成 `["un", "happiness"]` 而不是按整词或单字母切分——这背后是统计方法从语料库中"学"出最优词表的过程。理解这一点对 LoRA 微调很重要：微调时你处理的是 token 序列，而 token 的切分粒度直接决定模型看到的输入表示。随后进入 **"掌握 HuggingFace Tokenizer 的加载与使用"**，用刚装好的 `transformers` 加载真实预训练分词器（如 GPT-2 的 BPE 分词器），亲手观察切分结果。这一步是 LoRA 微调数据预处理的前置技能——你后续构造微调数据集时，每一步都离不开 tokenizer 的调用。

**概念对比**：`pipeline` 高层封装 vs 直接调用 tokenizer + model：前者适合快速验证环境，后者才是微调脚本里实际使用的粒度。

**思考题**（答案均可在原文中找到依据）：
1. 为什么 `"unhappiness"` 会被切分成 `["un", "happiness"]` 而不是按单词或字母切分？原文给出的理由是什么？
2. 下一步将用 `transformers` 库加载哪种类型的真实分词器来观察切分结果？

**模型加载注意事项**：
- 模型许可证：加载预训练模型前需确认其许可证类型，不同模型（如 Llama 系列）有各自的社区许可协议，商用场景下需遵守相应授权条款。
- 显存要求：加载 7B 量级模型（如 Llama-2-7b）进行 FP16 推理时，至少需要约 14GB 显存；实际微调场景下显存需求会更高，需根据模型规模和精度提前规划硬件资源。

## 测评题目

### Q-KB07-ST0700-001
Hugging Face 模型与分词器常用的 Python 库名称是________。

### Q-KB07-ST0700-003
安装 Hugging Face Transformers 的常用命令是？

### Q-KB0704-003
使用 `tokenizer.apply_chat_template` 方法，将以下对话历史转换为模型输入文本，但不进行分词（即 `tokenize=False`）。
```python
messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"}
]
```

### Q-KB0701-001
Byte Pair Encoding (BPE) 算法的核心思想是？

### Q-KB0702-001
请写出使用 Hugging Face `transformers` 库加载名为 `bert-base-uncased` 的分词器的代码。

### Q-KB0702-002
使用已加载的 `tokenizer` 对象，对字符串 `"Hello, world!"` 进行分词，并打印出 token ID 列表。

### Q-KB07-ST0700-002
将文本转换为 token ID 序列的组件通常称为________。

### Q-KB0704-005
请写代码：加载 `meta-llama/Llama-2-7b-chat-hf` 分词器，使用 `apply_chat_template` 生成一个包含系统提示（"You are a helpful assistant."）、用户提问（"What is AI?"）的输入字符串，要求包含生成提示（generation prompt），且不进行分词。
