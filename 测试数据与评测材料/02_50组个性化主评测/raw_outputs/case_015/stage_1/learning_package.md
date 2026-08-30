# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你熟悉 HuggingFace Trainer 和 PyTorch 训练流程，但 LoRA 微调的第一步，恰恰是回到最底层：**你写的 Python 代码，凭什么能被 GPU 执行？**

答案就是 **Python 解释器**——程序的"翻译官"。

你写的 `.py` 文件是人类可读的文本，但 RTX 4080 只认 0 和 1。解释器负责把你的代码逐行翻译成机器指令，电脑才能运行。没有它，你的 LoRA 训练脚本只是一堆无法执行的文字。

**为什么这对你很重要？**

你在 Trainer 里写的 `TrainingArguments`、`PeftModel`，最终都要靠解释器翻译成 GPU 能跑的指令。解释器版本、环境是否干净，直接决定你的 LoRA 脚本能否稳定跑起来——尤其是 16GB 显存下做 QLoRA 时，环境问题往往比模型问题更先出现。

**动手验证（约 2 分钟）**：

打开终端，输入 `python --version`。如果看到版本号，说明解释器已就位；如果报错，说明环境还没配好——这正是你下一步要解决的。

## 块 2

## 1.2 代码编辑器（IDE）：你的“工作台”

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 3

## 2.1 Windows 用户：为 LoRA 微调铺好路

你已熟悉 HuggingFace Trainer，接下来要跑 LoRA 微调，Python 环境是第一步。下面三步帮你快速搞定。

**第一步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标移到顶部 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   不勾选的话，命令行输入 `python` 会提示"不是内部或外部命令"，后续 pip 装包、跑训练脚本都会卡住。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。想改路径的话，确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python 3.13.x`（或你下载的版本号），说明安装成功！
4. 再输入：

```bash
pip --version
```

如果也

## 块 4

你的目标是 LoRA 微调，而 LoRA 依赖 PyTorch 等深度学习框架，这些框架又依赖 Python 环境。本块帮你把 macOS 上的 Python 装好，这是后续搭建训练环境的第一步。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认不提供 `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA 微调）的关系；遇到无法解释的术语时记录下来再提问。

## 块 5

## 2.3 Linux 用户（Ubuntu/Debian）：为 LoRA 微调铺路

你的 RTX 4080 已就位，现在确保系统 Python 环境能支撑后续 PyTorch 与 HuggingFace Trainer 的安装。Linux 常预装 Python，但版本可能偏老，直接装好 3.10+ 与虚拟环境工具，避免微调脚本跑起来才发现环境缺东西。

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**为什么这样写**：`-y` 自动确认安装，

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.4 操作系统与实训环境该怎么选

你的目标是LoRA微调，这属于GPU实训。选对环境，能直接决定你后续跑训练时是否顺畅。

**日常写Python、练命令（Windows用户）**：启用 **WSL2 + Ubuntu LTS**（如Ubuntu 22.04/24.04 LTS）。它提供接近真实生产的Linux环境，命令、路径、包管理都和后续服务器一致。安装只需管理员PowerShell执行 `wsl --install`。

**日常写Python（macOS/Linux用户）**：直接用本机终端即可，本身就是类Unix环境。

**LoRA/QLoRA微调等GPU实训**：应使用**装有Ubuntu LTS的NVIDIA GPU云服务器，或本地实体Linux（Ubuntu LTS）机器**。这类任务依赖NVIDIA驱动、CUDA与显卡直通，Ubuntu LTS对深度学习生态支持最成熟。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox虚拟机"当作最佳方案**。CentOS 8已停止维护；VirtualBox默认**无法直通NVIDIA GPU**，跑不了真正的GPU训练，只适合纯CPU隔离小实验。

**

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你在用HuggingFace Trainer跑LoRA微调时，`pip install` 装 `peft`、`transformers` 等库是必经步骤。如果官方源下载慢，别急着改全局配置——**临时指定镜像**最干净：

```bash
pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**关键点**：官方PyPI（`pypi.org`）始终是首选默认源，镜像只是网络不佳时的临时加速手段。临时指定只影响这一次安装，不污染你的日常环境，网络恢复或镜像出问题时无需清理配置。

**动手验证**：用上述命令安装任意库，观察下载速度；再运行 `pip install numpy`（不带`-i`），确认

## 块 8

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置已生效，pip 会默认从清华镜像下载包。若输出为空或未显示该行，请检查配置文件路径和内容是否正确。

## 块 9

你的目标是用 LoRA 微调模型，而微调前需要跑训练脚本、看日志、改配置。VS Code 就是干这个的“工作台”——它轻量、免费，启动快，插件丰富，后续写 Linux 命令或改脚本时也能复用。

**安装三步走：**

1. 打开 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路“下一步”即可。
3. 装完打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装两个**必装插件**：
   - **Python**（微软官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

装完插件后，VS Code 才能识别 `.py` 文件，给你代码补全、错误提示和一键运行功能——这对你调试 LoRA 训练脚本很关键。

**配置 Python 解释器：**

1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

PyCharm Community 是 JetBrains 出品的免费 Python 专用 IDE，内置代码分析、版本控制、数据库工具，适合大型项目开发——比如你后续要进行的 LoRA 微调实验。

**安装步骤**：
1. 访问官网下载页，选择 **Community Edition**（社区版，免费）。
2. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行可直接启动）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`。
2. `Location` 处选择项目路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured interpreter`，再选系统已装的 Python 3.10+。

---

**练习任务**：按上述步骤创建一个名为 `lora_test` 的新项目，

## 块 11

### 5.1 为什么用 Conda

你已经熟悉 HuggingFace Trainer 和 PyTorch 训练流程，接下来要学 LoRA 微调。LoRA 微调通常需要特定版本的 PyTorch 和 CUDA 环境，如果直接在系统 Python 里装，很容易和已有项目冲突。**Conda** 就是解决这个问题的工具——它为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习这种依赖复杂的场景，Conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——你的目标是 LoRA 微调，环境越干净越好，需要什么装什么。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 conda 通过初始化脚本正确接管，而不是把目录硬加进 PATH。
4. **验证安装**：打开**新的**终端（Windows 用 "Anaconda Prompt

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

创建虚拟环境是为LoRA微调做准备的关键一步。你的RTX 4080跑LoRA时，不同项目可能需要不同版本的PyTorch和CUDA，现在学会隔离环境，以后直接复用这套流程。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫`llm-learn`（`n`是`name`的缩写）
- `python=3.10`：指定环境里安装Python

## 块 13

## 5.4 激活与退出环境

你已经熟悉 HuggingFace Trainer 了，但 LoRA 微调会涉及多个依赖库（如 `peft`、`bitsandbytes`）。用 Conda 环境隔离项目，能避免版本冲突——这正是你接下来要做的。

**激活环境**（进入这个"独立房间"）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**查看所有已创建的环境**：

```bash
conda env list
```

带 `*` 的就是

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，这意味着你马上要跑训练脚本、调超参数。如果 IDE 用的 Python 解释器不对，`import torch` 都会报错——所以这一步值得认真做。

**操作步骤：**

1. 终端里先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板
3. 输入并选择 `Python: Select Interpreter`
4. 在列表里选带有 `conda` 字样的项，形如 `Python 3.10.x ('llm-learn': conda)`

**为什么这样做**：VS Code 会自动识别 conda 环境。选中后，你在当前工作区写的所有代码都会使用 `llm-learn` 里的 Python 和已装好的包——包括你后续 LoRA 微调要用的 transformers、peft 等。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程略有不同：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`
3. 点齿轮图标 → `Add...`
4. 左侧选 `Conda Environment` → `Existing environment`
5. 在 `Interpreter` 路径中选择 conda 可执行文件（如 `~/

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你在命令行验证环境，是为LoRA微调铺路——训练脚本必须跑在正确的环境里。先激活环境，再输入`python`进入交互模式，看到`>>>`后执行：

```python
print("Hello, Python! 我的环境准备好了！")
```

打印成功即环境就绪，用`exit()`退出。接着在VS Code中建`my-first-project`文件夹，新建`hello.py`，写入`print("Hello from VS Code!")`，点右上角运行按钮或按`Ctrl + F5`，终端输出即

## 块 16

## 7.3 在 PyCharm 中验证

你已经熟悉 HuggingFace Trainer 和 PyTorch 进阶训练，环境验证对你来说更多是确认“工作台顺手”，而不是学习新概念。按下面三步走，30 秒内完成：

1. **新建项目**：打开 PyCharm，`File → New Project`，创建后右键项目根目录 → `New → Python File`，命名为 `hello.py`。
2. **写入代码**：在 `hello.py` 中输入 `print("Hello, LoRA!")`（这里用你自己的目标字符串即可，不必拘泥于原文示例）。
3. **运行**：右键编辑器空白处 → `Run 'hello'`，底部 **Run 窗口**会输出你打印的内容。

> 小提示：你的 RTX 4080 16GB 在后续 LoRA 微调中会频繁用到 GPU 资源监控，PyCharm 的 Run 窗口同样会显示训练日志，确认输出无报错是每次跑脚本的第一步。

## 测评题目

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0205-008
写一个类 `Stack`（栈），实现 `push(item)`、`pop()`、`peek()`、`is_empty()` 方法。`pop()` 在栈空时抛出 `IndexError`。

### Q-KB0205-001
以下哪个是装饰器的正确语法？

### Q-KB0203-002
Python中，`*args` 接收的是________类型；`**kwargs` 接收的是________类型。

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

### Q-KB0204-005
请写一个自定义异常类 `ModelError`，继承自 `Exception`，并在模型加载失败时抛出该异常，附带错误信息。

### Q-KB0205-004
请写一个生成器函数 `fibonacci(n)`，生成前 `n` 个斐波那契数。
