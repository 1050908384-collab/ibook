# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你写下的 Python 代码是给人看的文字，但电脑只懂 0 和 1。**Python 解释器**就是那个“翻译官”——它把你的代码逐句翻译成电脑能执行的指令。没有它，你写的 `.py` 文件就只是一堆普通文本，电脑根本不会理睬。

**为什么这个“翻译官”对你做 QLoRA 微调很重要？**  
因为 QLoRA 微调本质上就是让电脑执行大量数学运算（训练模型）。你写的每一行 Python 代码——无论是加载模型、处理数据，还是调用微调函数——都必须经过解释器翻译，电脑才能“听懂”并跑起来。所以，装好解释器，是你动手微调的第一块基石。

**动手验证一下（1 分钟）**  
打开你的终端（Windows 用 PowerShell，Mac 用“终端”App），输入下面这行命令并回车：

```bash
python --version
```

如果看到类似 `Python 3.x.x` 的版本号，说明解释器已经装好并“在线

## 块 2

好的，我们直接进入正题。你的目标是QLoRA微调，这需要你熟练操作代码和模型文件，一个顺手的“工作台”是第一步。

**1.2 你的“工作台”：代码编辑器（IDE）**

记事本写代码效率太低。**IDE（集成开发环境）** 能提供语法高亮、自动补全、运行调试等功能，就像用Word写文档一样舒服。我们推荐两款：

- **VS Code**：轻量、免费，插件丰富，适合快速上手。
- **PyCharm**：专业、强大，专为Python设计，功能更全面。

**1.3 虚拟环境：项目的“独立房间”**

想象你同时做两个项目：A项目需要`numpy 1.0`，B项目需要`numpy

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 QLoRA 微调，这需要先有一个能跑 Python 的工作台。下面三步，跟着做就行。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`。
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`，这是 3.10+ 的最新版本，满足要求）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，

## 块 4

### macOS 用户：安装 Python 3.10+

现代 macOS（Catalina 10.15 起）不再预装 Python 2，系统自带的 `python3` 仅供内部脚本使用。为了 QLoRA 微调，请安装自己的 Python，避免依赖系统版本。

**方法一：官方安装包（推荐）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install"。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认没有 `python` 命令，系统与官方安装包统一使用 `python3`。始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

在终端执行：

```bash
brew install python
```

安装后同样用 `python3 --version` 验证。

**本块学习检查**：在终端运行 `python3 --version`，确认输出版本号（如 `Python 3.10.x` 或更高）。若提示命令不存在，请重新检查安装步骤。

## 块 5

## 动手装好 Python，为 QLoRA 铺路

你的目标是 QLoRA 微调，但一切从环境开始。Ubuntu/Debian 通常自带 Python，但版本偏老，直接装新的最省心。

**三步搞定：**

```bash
# 1. 更新软件源（让系统知道最新软件包）
sudo apt update

# 2. 安装 Python 3、pip 和 venv（-y 自动确认，不用手动输 y）
sudo apt install python3 python3-pip python3-venv -y

# 3. 验证安装
python3 --version
```

**关键点**：`python3-venv` 是创建虚拟环境的模块，现在一起装

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 QLoRA 微调，所以环境选择直接决定后面能不能跑通。记住一条主线：**命令练习用 WSL2，GPU 实训用 Ubuntu LTS**。

**如果你用 Windows：**
日常写 Python、练命令行，推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得一个接近真实服务器的 Linux 环境，命令、路径、包管理都和后续云端一致。安装很简单：管理员 PowerShell 执行 `wsl --install`。

**如果你用 macOS / Linux：**
直接使用本机终端即可，本身就是类 Unix 环境，不用额外配置。

**做 QLoRA 微调这类 GPU 实训时：**
必须用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。因为这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 避坑提醒：**
不要用 **"CentOS + VirtualBox 虚拟机"** 当主力方案。CentOS 8 已停止维护（2021 年 EOL），其继任者 CentOS Stream 8 也已于 2024 年停止支持；VirtualBox

## 块 7

**动手场景：装包卡住时，用镜像源“救急”**

你正在为 QLoRA 微调装 `numpy` 这类基础库，结果 `pip install` 转圈半天或直接超时。这时才需要镜像源——它只是“网络不好时的临时加速手段”，不是永久设置。官方 PyPI（`pypi.org`）永远是最权威、最完整、应保留的默认源。

**首选做法：只对本次安装临时指定镜像**

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这样**：它只影响这一次安装，不改动全局配置。官方源仍是默认；网络恢复或镜像出问题时，你无需清理任何设置。

**本块学习检查**：在你的终端里，用临时镜像源安装 `numpy`，确认安装成功后，再运行 `pip config list` 查看全局配置——如果输出中没有

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正在为 QLoRA 微调做准备，这意味着要频繁下载 PyTorch、Transformers 等大体积依赖包。如果网络环境长期受限，每次手动指定镜像太麻烦，可以把镜像写进 pip 配置，一劳永逸。

**但先记住一个原则**：官方 PyPI 永远是最可靠的来源。这个配置只在网络受限期间使用，网络恢复后建议改回或删除。

---

### Windows 用户操作

1. 按 `Win + R`，输入 `%APPDATA%` 回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹
2. 新建一个名为 `pip` 的文件夹
3. 在 `pip` 文件夹内新建文本文件，命名为 `pip.ini`
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：默认从这里下载包。

**⚠️ 关于 `trusted-host` 的警告**：网上很多教程会让你顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**。加上它等于让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用完全不需要。只有企业内网自建源、或确实遇到证书报错且确认来源可信时，才临时加。

---

### macOS / Linux 用户操作

1. 终端执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 按 `Ctrl + O` 保存，`Ctrl + X` 退出。

---

### 验证配置是否生效

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

### 改回官方源

删除刚才创建的 `pip.

## 块 9

**目标对齐**：你要做 QLoRA 微调，核心是高效训练大模型。VS Code 是你写微调脚本、看训练日志的“主战场”——轻量、免费，启动快，插件生态强，后续调 Linux 命令、改配置文件都用得上。

**安装三步走**：
1. 打开 https://code.visualstudio.com，下载你系统的安装包。
2. 运行安装程序，一路“下一步”即可。
3. **装插件**：点左侧活动栏最下方的四个方块图标（扩展），搜索安装：
   - **Python**（微软官方，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

> 为什么必装：不装 Python 插件，VS Code 认不出 `.py` 文件，代码补全、错误提示、一键运行全都没有——写微调脚本会寸步难行。

## 块 10

**本块学习检查**：安装完成后，新建一个项目，确认你能在 PyCharm 中看到 Python 解释器路径，并成功运行 `print("hello")`。

---

**PyCharm Community：专业、开箱即用**

PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。内置代码分析、版本控制、数据库工具，适合大型项目开发——这正是你后续做 QLoRA 微调时需要的工程化环境。

**安装步骤（跟着做）：**

1. 打开 https://www.jetbrains.com/pycharm/download/
2. 下载 **Community Edition**（社区版，免费）
3. 运行安装程序，**务必勾选**：
   - `Add "Open Folder as Project"`（右键快速打开项目）
   - `Add to PATH`（命令行直接启动）

**首次启动配置：**

1. 打开 PyCharm → `New Project`
2. `Location` 选项目路径（**路径中不要含中文**）
3. `Interpreter` 选 `Previously configured interpreter`

## 块 11

好的，让我们开始搭建你的 Python 开发工作台。你的目标是进行 QLoRA 微调，这需要一套干净、可控的深度学习环境。Conda 就是帮你管理这一切的得力工具。

### 5.1 为什么用 Conda

简单说，Conda 能为你**每个项目**创建一个独立的 Python 空间，避免不同项目间的依赖冲突。它不仅能管理 Python 包，还能管理 **Python 解释器本身的版本**。对于你后续要进行的 QLoRA 微调（需要 PyTorch、CUDA 等复杂依赖），Conda 几乎是必备的，它能帮你把这些复杂的依赖关系理得清清楚楚。

### 5.2 安装 Conda：Anaconda vs Miniconda

安装 Conda 有两个主流选择，我们直接看对比：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**我们推荐 Miniconda**。原因很简单：环境更干净，后续需要什么包装什么包，不会有一堆用不上的库拖慢速度、增加混乱。

**安装步骤：**

1.  访问 `https://docs.conda.io/en/latest/miniconda.html`。
2.  下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3.  运行安装程序，这里有个关键区别：
    *   **Windows**：一路下一步，**保持默认，不要勾选 "Add Miniconda3 to my PATH environment variable"**。这个选项容易让 conda 和系统已有的 Python 冲突，导致 `python`、`pip` 指向不明。日常使用请通过开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来操作。
    *   **macOS/Linux

## 块 12

**目标**：为 QLoRA 微调打地基——先学会用 Conda 隔离项目环境，避免不同项目互相干扰。

**动手做**：打开终端，输入下面这条命令，然后回车：

```bash
conda create -n llm-learn python=3.10
```

**这条命令在干什么**？拆开看：
- `conda create`：告诉 Conda“我要新建一个环境”。
- `-n llm-learn`：给这个环境起名 `llm-learn`（`n` 是 `name` 的缩写）。
- `python=3.10`：指定环境内安装 Python 3.10 版本。

**接下来**：终端会列出准备安装的包，并问你是否继续，显示 `Proceed ([y]/n

## 块 13

## 5.4 激活与退出环境——进入你的专属“工作间”

你的目标是跑通 QLoRA 微调，这意味着后面所有操作都要在一个干净、隔离的环境里进行，避免和系统自带的 Python 打架。这一节就是教你如何“进门”和“出门”。

**激活环境（进入工作间）**

在终端里输入：

```bash
conda activate llm-learn
```

**怎么确认自己进来了？** 激活后，命令行最前面会出现 `(llm-learn)` 的标志。看到它，就说明你后续敲的 `python` 和 `pip` 命令都只作用于这个环境，不会干扰系统或其他项目——这对你后面安装 QLoRA 相关的依赖包至关重要。

**验证一下当前环境的 Python 版本**：

```bash
python --version
```

正常会显示 `Python 3.10.x`。记住这个版本号，后面装包时如果报错，先回来确认环境对不对。

**

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 QLoRA 微调，这意味着你马上要跑大量依赖特定包（如 `transformers`、`peft`）的代码。所以，**确保代码跑在正确的环境里**，是第一步，也是最容易踩坑的一步。

跟着做，三步搞定：

1. **激活环境**：在终端运行 `conda activate llm-learn`。这一步确保你后续操作基于 `llm-learn` 环境。
2. **打开命令面板**：在 VS Code 中按 `Ctrl + Shift + P`。
3. **选择解释器**：输入 `Python: Select Interpreter`，在列表中选择带有 `conda` 字样的 Python 路径（形如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这么做**：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的所有代码，都会使用 `llm-learn` 里的 Python 和已安装的包。**如果选错，你可能会遇到 `ModuleNotFoundError`——明明装了的包却找不到，多半是解释器选错了。**

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，路径略有不同：

1. 打开 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`

## 块 15

**本块学习检查**：完成下方两个验证后，在终端输入 `exit()` 退出 Python，并确认 VS Code 终端显示了 `Hello from VS Code!`。

你的目标是 QLoRA 微调，动手验证环境是第一步。跟着做，很快。

**1. 命令行验证（最直接）**

打开终端，先激活环境：

```bash
conda activate llm-learn
```

输入 `python` 回车，看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车。如果正确打印出这句话，说明 Python 能跑。输入 `exit()` 退出。

**2. VS Code 验证（写代码更顺手）**

新建文件夹 `my-first-project

## 块 16

好的，我们直接进入正题。你的目标是QLoRA微调，那环境就是你的“工作台”。这一块，我们只做一件事：**验证你的 PyCharm 能跑代码**。

### 7.3 在 PyCharm 中验证

打开 PyCharm，新建一个项目，然后在项目里新建一个 Python 文件，命名为 `hello.py`。

在文件里随便写点东西，比如 `print("hello")`。然后，在文件编辑区**右键点击**，选择 **`Run 'hello'`**。

看屏幕**下方**的 **Run 窗口**。如果一切正常，你应该能看到程序输出的结果。看到输出，就说明你的“工作台”已经能正常运转了。

---

### 本章小结：你的“工作台”清单

完成本章，你的电脑上应该具备以下内容：

| 组件 | 状态 | 验证命令 |
|------|------|----------|
| Python 3.10+ | ✅
