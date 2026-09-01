# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已经有 NLP 和注意力机制的基础，目标是用 LoRA 微调模型。但在动手之前，先花 2 分钟搞清楚一个底层概念——**Python 解释器**。

你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。**Python 解释器**的作用就是把你的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本。

这和 LoRA 微调有什么关系？你之后跑微调脚本时，用的就是 Python 解释器去执行那些 `.py` 文件。它负责把你的训练代码翻译成 GPU（比如你的 RTX 4060 Ti）能执行的指令。理解这一点，你就知道为什么环境搭建是第一步——没有翻译官，你的微调代码根本跑不起来。

**动手验证**：打开终端，输入 `python --version`，看到版本号就说明解释器已就位。如果报错，记下错误信息，这正是你环境配置的起点。

## 块 2

### 1.2 代码编辑器（IDE）：你的"工作台"

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

### 1.3 虚拟环境：项目的"独立房间"

想象你同时在做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。如果都装到同一个 Python 里，就会冲突报错。**虚拟环境**就是为每个项目创建一个独立的 Python 空间，互不影响。本文使用 **conda** 管理虚拟环境，这是常用的虚拟环境管理工具之一。

---

## 第二章：安装 Python 3.10+

**为什么选 3

## 块 3

## 2.1 Windows 用户：为 LoRA 微调铺路

你已掌握 NLP 和注意力机制，接下来要动手做 LoRA 微调。微调离不开 Python 环境，本块就是帮你把“工作台”搭好。

**第一步：下载安装包**

打开浏览器访问 https://www.python.org，鼠标移到顶部 **Downloads**，网站通常会推荐适用于 Windows 的最新稳定版本（如 Python 3.12 或 3.13），点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统“环境变量”。不勾选的话，后续在命令行输入 `python` 会提示“不是内部或外部命令”，所有命令行操作都无法进行——包括你之后跑微调脚本。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。若想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

按 `Win + R`，输入 `cmd` 回车打开命令提示符，依次输入：

```bash
python --version
```

显示 `Python 3.13.x`（或你下载的版本号）即成功。再输入：

```bash
pip --version
```

显示版本号说明

## 块 4

### 2.2 macOS 用户：为 LoRA 微调铺好 Python 地基

你的目标是 LoRA 微调，这意味着很快要跑 PyTorch 和 HuggingFace 脚本。macOS 上 Python 环境若没弄对，后面会频繁踩坑。好消息是，macOS 12.3 起不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用，**千万别直接依赖它**——版本老旧且权限受限，跑深度学习库容易出诡异问题。

**方法一：官方安装包（推荐，最省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 完成。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：现代 macOS 命令行默认**没有** `python` 命令，只有 `python3`。请始终用 `python3` / `pip3`，否则会碰到"命令不存在"或误用系统旧版的问题。后续你安装 PyTorch、transformers 时，也要用 `pip3` 来装。

**方法二：Homebrew（若你已装过）**

终端执行：

```bash
brew install python
```

装完同样用以下命令验证：

```bash
python3 --version
pip

## 块 5

你的目标是跑通 LoRA 微调，而这一步是搭好地基。Linux 上通常自带 Python，但版本可能偏老，直接装新的最省心。

在 Ubuntu/Debian 终端依次执行：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**为什么这样写**：`-y` 参数表示自动确认安装，不需要你手动输入 `y`。`python3-venv` 是创建虚拟环境所需的模块，一起安装避免后续补装。

**本块学习检查**：完成本块后，用自己的话说明本

## 块 6

## 2.4 操作系统与实训环境该怎么选

你已经有 NLP 和注意力机制的基础，接下来做 LoRA 微调，环境选型直接决定你能不能跑起来。核心原则：**日常练习和 GPU 实训分开选环境**。

**日常写 Python、练命令行（Windows 用户）**

启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在 Windows 里提供一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**

直接用本机终端，本身就是类 Unix 环境。

**LoRA / QLoRA 微调等 GPU 实训**

用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护（EOL 2021-12-31），CentOS 7 也已于 2024 年停止维护，建议使用 Ubuntu LTS 等活跃维护的发行版；VirtualBox 这类桌面虚拟机**默认无法直通 NVIDIA GPU**，跑

## 块 7

## 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认它从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、最应保留的默认源**。

**只有**当你网络受限、访问官方源明显变慢或经常超时时，才临时切到国内镜像。镜像是“网络不好时的临时加速手段”，**不是必须永久替换的默认设置**——官方源始终是首选。

## 3.2 首选做法：临时指定镜像源（推荐）

某次安装慢时，最干净的做法是**只对这一次安装临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方 PyPI 仍是默认源

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，而微调过程要下载大量预训练权重和依赖包。如果你长期处于网络受限环境，几乎每次都要用镜像，才值得把它写入配置作为默认源。配置后请记得：官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车。这会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

**macOS / Linux 用户：**

1. 打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样的内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 按 `Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。它不只适合 Python，后续写脚本、改配置时也能复用，正好匹配你 LoRA 微调中频繁调整训练脚本和配置的需求。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能——微调脚本调试时这些能力很关键。

**配置 Python 解释器**：
1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Inter

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置了代码分析、版本控制、数据库工具等功能，适合大型项目开发。对你做 LoRA 微调来说，一个专业的 IDE 能帮你管理复杂的训练脚本和调试过程。

**安装步骤：**

1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置：**

1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured interpreter`，然后选择系统安装的 Python

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 11

## 5.1 为什么用 Conda

你已经掌握了 NLP 和注意力机制，接下来要做 LoRA 微调，这意味着你很快就要面对 PyTorch、CUDA 等复杂依赖的安装与版本匹配问题。Conda 的价值正在于此：它不仅能像普通虚拟环境那样为每个项目隔离 Python 空间，还能直接管理不同版本的 Python 解释器本身。对深度学习场景来说，这是近乎必备的工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，对比如下：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装 numpy、pandas、jupyter 等大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**。你的目标是 LoRA 微调，环境越干净越好——需要什么装什么，避免无关包干扰后续依赖管理。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 后按提示输入 `yes` 同意协议、选择路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 conda 通过初始化脚本正确接管，而非硬加 PATH。
4. **验证安装**：打开**新的**终端（Windows 用 "Anaconda Prompt (Miniconda3)"

## 块 12

你已经有 NLP 和注意力机制的基础，现在要迈向 LoRA 微调。微调大模型时，不同项目常需要不同版本的 PyTorch 和 CUDA，用 Conda 虚拟环境隔离项目依赖，能避免版本冲突。这套流程学会后，后续微调项目直接复用。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境安装 Python 3.10

执行后，终端列出待安装包并询问 `Proceed ([y]/n)?`，

## 块 13

### 5.4 激活与退出环境

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前出现 `(llm-learn)` 标志，表示后续 `python` 和 `pip` 命令都作用在此环境，不影响系统或其他项目。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应显示 `Python 3.10.x`。

**退出环境**（回到系统默认）：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**查看所有环境**：

```bash
conda env list
```

带 `*` 的是当前激活的环境。

**删除环境**（清理时用）：

```bash
conda remove -n llm-learn --all
```

---

## 第六章：在 VS Code / PyCharm 中使用 Conda 环境

---

**本块学习检查**：完成本块

## 块 14

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已掌握NLP和注意力机制，接下来要做LoRA微调。微调前必须让代码跑在正确的Python环境里——这就是本块要解决的：把VS Code或PyCharm指向你为微调任务创建的`llm-learn` conda环境。

**VS Code配置（推荐，轻量且适合调试）**

1. 终端先激活环境：`conda activate llm-learn`
2. 打开VS Code，按`Ctrl + Shift + P`打开命令面板
3. 输入并选择`Python: Select Interpreter`
4. 在列表中选带`conda`字样的项（形如`Python 3.10.x ('llm-learn': conda)`）

VS Code会自动识别conda环境。选中后，你在该工作区写的所有代码都会使用`llm-learn`里的Python和已装好的包（如PyTorch、transformers）——这是LoRA微调的基础。

**PyCharm配置（如果你更习惯IDE）**

1. 打开PyCharm，进入`File` → `Settings`（Mac为`PyCharm` → `Preferences`）
2. 左侧选`Project: 你的项目名` → `

## 块 15

### 7.2 在 VS Code 中验证

1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`。
3. 在文件中输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

4. 保存文件，点击右上角运行按钮（▶），在下方终端查看输出。

## 块 16

## 7.3 在 PyCharm 中验证

你已经掌握 NLP 和注意力机制，现在要进入 LoRA 微调实战。在动手微调之前，先确保你的开发环境能正确运行代码——这是所有后续工作的地基。

**操作步骤：**

1. 打开 PyCharm，新建一个项目。
2. 在项目中创建文件 `hello.py`。
3. 在文件中输入一行最简单的代码（例如打印一句话）。
4. 右键点击文件 → 选择 `Run 'hello'`。
5. 观察下方 Run 窗口，确认输出结果正确显示。

如果一切顺利，说明你的 PyCharm 已经正确关联了 Python 解释器，可以开始写真正的代码了。

**本块学习检查**

## 测评题目

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0201-002
Python中，`type(3.14)` 的返回结果是________；`type("3.14")` 的返回结果是________。

### Q-KB0201-006
以下代码输出什么？
```python
x = 10
def foo():
    x = 20
    def bar():
        nonlocal x
        x = 30
    bar()
    print(x)
foo()
print(x)
```

### Q-KB0204-004
以下代码中，`finally` 块在什么情况下执行？
```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("除零错误")
finally:
    print("清理完成")
```

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

### Q-KB0203-002
Python中，`*args` 接收的是________类型；`**kwargs` 接收的是________类型。

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？
