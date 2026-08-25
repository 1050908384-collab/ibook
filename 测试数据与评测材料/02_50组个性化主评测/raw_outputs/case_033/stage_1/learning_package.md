# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

**别急着装，先搞懂装的是什么。**

你写的 Python 代码是给人看的文字，但电脑只认 0 和 1。**Python 解释器**就是那个"翻译官"——它把你的代码实时翻译成电脑能执行的指令。

**关键点**：没有解释器，你写的 `.py` 文件对电脑来说就是一堆普通文本，毫无意义。

**动手验证（30秒）**：
1. 随便建一个文本文件，改名成 `test.py`
2. 双击打开——你会发现它只是用记事本打开了，什么都没发生
3. 这就是"没有翻译官"的样子：代码在，但没人执行

**记住这个比喻**：你写代码 = 用中文写菜谱；解释器 = 厨师，照着菜谱做菜。没有厨师，菜谱只是一张纸。

**下一步**：我们要安装的，就是这位"厨师"。装好之后，你的 `.py` 文件才能真正"跑"起来。

## 块 2

### 你的“工作台”：从记事本到 IDE

你的目标是 vLLM 部署，这需要频繁调试代码。用记事本写代码效率太低，**IDE（集成开发环境）** 能提供语法高亮、自动补全和运行调试，就像用 Word 写文档一样顺手。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个装上即可。

### 虚拟环境：项目的“独立房间”

部署 vLLM 时会装很多依赖包。假设项目 A 要 `numpy 1.0`，项目 B 要 `numpy 2.0`，全装一起就冲突报错。**虚拟环境**为每个项目创建独立 Python 空间，

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 Python 是这一切的地基。下面三步，跟着做就行。

**第一步：下载安装包**

1.  打开浏览器，访问 `https://www.python.org`。
2.  鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的Windows版本（如 `Python 3.x.x`）。
3.  点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1.  **勾选最下方的 "Add python.exe to PATH"**  
    这是重中之重。不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令操作都无法进行。
2.  点击 **"Install Now"**（默认安装）即可。  
    默认路径通常不含中文和特殊符号，但若你的Windows用户名包含中文或空格，建议自定义安装路径以确保路径安全。若想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1.  按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2.  输入 `python --version` 并回车。
3.  如果显示 `

## 块 4

### macOS 安装 Python：直奔 vLLM 部署

你的目标是 vLLM 部署，但先别急——所有推理框架都跑在 Python 上。macOS 上装错版本会浪费你宝贵的 2.5 小时，所以这一步我们一次做对。

**关键认知**：现代 macOS（12.3 起）**不再预装 Python 2**，系统自带的 `python3` 是给系统脚本用的，版本老旧。**别碰它**——直接装你自己的最新版，避免后续 vLLM 依赖冲突。

---

**方法一：官方安装包（推荐，最快上手）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install"，输密码，完事。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

两条命令都应显示版本号，说明装好了。

**为什么是 `python3` 不是 `python`**：macOS 命令行默认**没有** `python` 命令（早期版本指向 Python 2，已移除）。系统和你装的官方包都统一用 `python3` / `pip3`。**始终

## 块 5

**动手装好 Python，为 vLLM 铺路**

你的目标是部署 vLLM，但先别急——所有推理工具都跑在 Python 上。Ubuntu/Debian 通常自带 Python，但版本偏老，直接装新的最省事。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：第一行刷新软件源，确保能装到最新包；第二行一次装齐 Python 本体、包管理器 pip、虚拟环境工具 venv——`-y` 表示自动确认，不用你手动敲 `y`；第三行验证版本，看到类似 `Python

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，这是 GPU 密集型任务，环境选错会直接卡住。按你的实操风格，直接给结论：

**日常练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。管理员 PowerShell 跑一句 `wsl --install` 即可。它给你一个贴近生产服务器的 Linux 环境，命令、路径、包管理都和后续一致，别在 CMD 里浪费时间。

**macOS / Linux 用户**：直接用本机终端，本身就是类 Unix，无需额外配置。

**vLLM 部署（你的核心场景）**：必须用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 是深度学习生态支持最成熟、文档最全的选择。

**⚠️ 避坑**：别用 **CentOS + VirtualBox 虚拟机**。CentOS 8 已于2024年5月31日停止维护；VirtualBox 默认配置下无法直通 NVIDIA GPU，需额外配置和硬件支持，只适合纯 CPU 隔离小实验。

**你的

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**你只需记住一个判断标准**：只有当网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式

## 块 8

### 3.5 改回官方源

当网络恢复或不再需要镜像时，建议改回官方源。操作如下：

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车。
2. 进入 `pip` 文件夹，用记事本打开 `pip.ini`。
3. 删除 `[global]` 下的 `index-url` 行，或将该行注释掉（在行首加 `#`），保存文件。

**macOS / Linux 用户：**

1. 打开终端，执行：

```bash
nano ~/.config/pip/pip.conf
```

2. 删除 `[global]` 下的 `index-url` 行，或将该行注释掉（在行首加 `#`），按 `Ctrl + O` 保存，`Ctrl + X` 退出。

**通用方法（推荐）：**

在命令行直接执行：

```bash
pip config unset global.index-url
```

该命令会移除 `index-url` 配置，使 pip 恢复使用官方 PyPI 源。执行后可用 `pip config list` 确认配置已清除。

## 块 9

**目标导向：** 你的目标是 vLLM 部署，这需要一套可靠的 Python 环境。VS Code 轻量、免费，是搭建这套环境的理想起点，后续写脚本、调配置都用得上。

**安装 VS Code（约5分钟）**

1.  打开官网 https://code.visualstudio.com，下载对应你系统的安装包。
2.  运行安装程序，一路“下一步”完成安装。

**装两个必装插件（关键步骤）**

打开 VS Code，点击左侧活动栏最下方的**四个方块图标**（扩展），搜索并安装：

-   **Python**（Microsoft 官方）：提供语法高亮、调试、智能提示。
-   **Pylance**：让代码提示更智能。

> **为什么必须装？** 不装 Python 插件，VS Code 无法识别 `.py` 文件，也就没有代码补全和错误提示。

**指定 Python 解释器（连接环境）**

1.  按 `Ctrl + Shift + P`（Mac 用 `Cmd +

## 块 10

## 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具等功能，适合大型项目开发——你后续做 vLLM 部署时，项目结构会越来越复杂，这一步值得打好基础。

**安装步骤**（跟着做，别跳）：

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费）
3. 运行安装程序，**务必勾选**：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置**：

1. 打开 PyCharm，选择 `New Project`
2. `Location` 处选择项目存放路径——**路径中不要含中文**，否则后续工具链容易出怪问题
3

## 块 11

### 5.1 为什么用 Conda

你的目标是 vLLM 部署，这属于深度学习推理场景，依赖 PyTorch、CUDA 等复杂库。不同项目对 Python 版本和库版本的要求可能冲突，所以需要**虚拟环境**为每个项目隔离独立的 Python 空间。

**Conda** 是目前最主流的虚拟环境管理工具。它不仅能创建虚拟环境，还能直接管理不同版本的 Python 解释器本身——这对深度学习项目尤其关键，因为某些框架对 Python 版本有硬性要求。对于你接下来的 vLLM 部署，conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境更干净，后续 vLLM 部署需要什么包装什么包，避免预装库干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 后，按提示输入 `yes` 同意协议、选择路径，最后一步选

## 块 12

## 5.3 创建你的第一个虚拟环境

**为什么现在学**：你的目标是 vLLM 部署。后续跑模型时，不同项目常需要不同版本的 PyTorch 和 CUDA。现在花两分钟学会建环境，后面直接复用这套流程，省去反复折腾依赖的麻烦。

**动手做**：打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**拆开看**（就三部分）：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境里装 Python 3.10

## 块 13

## 5.4 激活与退出环境：让你的项目住进“独立房间”

你的目标是 vLLM 部署，而 vLLM 对依赖版本非常敏感。如果所有项目共用一套 Python 包，版本冲突会浪费大量排查时间。Conda 环境就是为每个项目准备的“独立房间”。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志。这意味着后续所有 `python` 和 `pip` 命令都只作用于这个环境，不会污染系统或其他项目——这正是 vLLM 部署前必须做好的隔离。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。确认版本正确，再继续安装 vLLM 相关依赖。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

###

## 块 14

**动手配置：让 VS Code 和 PyCharm 用上你的 Conda 环境**

你的目标是 vLLM 部署，而环境是第一步。下面直接上手，跟着做就行。

**VS Code 配置（推荐先做这个）**

1. 在终端里激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表里选带有 `conda` 字样的项，例如 `Python 3.10.x ('llm-learn': conda)`。

选完后，VS Code 会自动识别这个 conda 环境。之后你在当前工作区写的代码，用的就是 `llm-learn` 里的 Python 和已安装的包。

**PyCharm 配置**

1. 打开 PyCharm，进入 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径里选 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\ll

## 块 15

**目标：** 用命令行快速验证 Python 环境是否可用，为 vLLM 部署打好基础。

**操作步骤（跟着做）：**

1.  **激活环境**：在终端输入 `conda activate llm-learn` 并回车。
2.  **进入交互模式**：输入 `python` 并回车。
3.  **执行测试**：看到 `>>>` 提示符后，输入以下命令并回车：
    ```python
    print("Hello, Python! 我的环境准备好了！")
    ```
4.  **确认与退出**：若屏幕正确打印出上述文字，说明环境正常。输入 `exit()` 退出。

**学习检查：**
请重复步骤 1-3，但将引号内的

## 块 16

### 7.3 在 PyCharm 中验证

你的目标是 vLLM 部署，但别急着跑大模型——先让开发环境“跑通”第一行代码，这是所有后续工作的地基。咱们用最快的方式验证：

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目里新建一个文件，命名为 `hello.py`。
2. **运行它**：在 `hello.py` 文件里随便写点内容（比如 `print("hello")`），然后**右键点击文件** → 选择 **`Run 'hello'`**。
3. **看结果**：PyCharm 底部会弹出 **Run 窗口**，里面应该显示你写的输出结果。

如果 Run 窗口
