# 学习包

## 块 1

**本块学习检查**：读完本节后，请合上资料，用一句话向自己解释“为什么没有解释器，`.py` 文件就只是文本？”说不清就回头重读。

---

### 1.1 Python 解释器：程序的"翻译官"

你的目标是部署 vLLM，但别急着碰模型——先想清楚：你写的 Python 代码是人话，电脑只懂 0 和 1。**解释器就是那个翻译官**，把你的代码逐句翻译成机器指令。没有它，`.py` 文件就是一堆没用的文字。

**动手验证（2 分钟）**：打开记事本，写一行 `print("hello")`，存成 `test.py`。双击它——什么都不会发生。这就是“没有翻译官”的样子。等装好解释器后，再运行同一个文件，你会看到输出。这一对比，你就彻底明白解释器在干什么了。

记住：**装 Python，核心就是装这个翻译官**。后面所有配置（比如 vLLM 依赖的 CUDA、PyTorch）都建立在它之上。现在不懂没关系，先把这个“翻译”的直觉刻在脑子里，下一步装起来你才知道每一步在为什么服务。

## 块 2

**IDE 与虚拟环境：为 vLLM 部署搭好工作台**

你要部署 vLLM，代码不会只写几行，记事本绝对不够用。**IDE（集成开发环境）** 就是你的“工作台”，提供语法高亮、自动补全、运行调试，写代码像用 Word 一样舒服。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个装上即可。

更关键的是**虚拟环境**——项目的“独立房间”。假设你同时做两个项目：一个需要 `numpy 1.0`，另一个需要 `numpy 2.0`，全装进同一个 Python 会冲突报错。虚拟环境为每个项目创建独立 Python 空间，互不干扰。本课程用 **

## 块 3

### 2.1 Windows 用户：安装 Python（为 vLLM 部署打基础）

你正在为 vLLM 部署做准备，Python 是第一步。下面三步走，跟着做就行。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令行操作都无法进行——vLLM 部署全程依赖命令行，这一步不能省。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2

## 块 4

**macOS 安装 Python：直奔 vLLM 部署**

你的目标是部署 vLLM，所以先把地基打牢。macOS 上装 Python 很简单，记住一个核心原则：**别用系统自带的，自己装最新的**。

macOS 12.3 起不再预装 Python 2，系统可能仍保留旧版 `python3` 供系统使用，别碰。自己装，干净又可控。

**方法一：官方安装包（推荐，最省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 就行。
3. 打开终端，输入下面两条命令验证：

```bash
python3 --version
pip3 --version
```

看到版本号就成功了。

**关键提醒：为什么用 `python3` 而不是 `python`？**

macOS 命令行默认**没有** `python` 这个命令。官方安装包也只提供 `python3`。所以，**以后所有命令都用 `python3` 和 `pip3`**，别用 `python`，否则会报"命令不存在"，或者误用系统旧版本，给自己挖坑。

**方法二：Homebrew（如果你已经装了）**

终端执行一条命令：

```bash

## 块 5

**动手装 Python（Ubuntu/Debian）**

你的目标是部署 vLLM，但先得有能跑它的 Python 环境。Linux 通常自带 Python，但版本可能偏老，所以咱们直接装最新的 3.10+。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：

- `sudo apt update`：刷新软件源，让系统知道有哪些新版本可装。
- 第二条命令一口气装三样：`python3`（解释器）、`pip`（装库的工具）、`venv`（虚拟环境模块）。`-y` 表示自动确认，不用你

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，这属于 GPU 实训，环境选错会直接卡住后续所有步骤。按你的学习节奏，直接给结论：

**日常练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**vLLM 部署这类 GPU 实训**：必须用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 别踩坑**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS Linux 8 已停止维护（CentOS Linux 8 于 2021 年 EOL，CentOS Stream 8 于 2024

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（如 `numpy`、`pandas`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、应保留的默认源**。

**关键判断**：只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是“网络不好时的临时加速手段”，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：它不影响你平时的默认行为，官方 PyPI 仍是默认

## 块 8

## 3.3 进阶：把镜像设为默认（长期网络受限才做）

你正在为 vLLM 部署做准备，安装依赖时网络慢会非常耽误时间。如果你长期处于网络受限环境，几乎每次都要用镜像，才建议把它写入配置作为默认源。配置后请记住：**官方 PyPI 仍是最可靠的来源**，网络恢复后建议改回或删除该配置。

### Windows 用户操作

1. 按 `Win + R`，输入 `%APPDATA%`，回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

### macOS / Linux 用户操作

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

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pyp

## 块 9

**动手装好 VS Code，为 vLLM 部署铺路**

你的目标是 vLLM 部署，后面要写 Python 脚本、改配置文件、跑命令，VS Code 就是你的主战场。它免费、启动快，装好后既能写 Python，也能顺手改 Linux 下的配置文件，一举两得。

**安装就三步，跟着做：**

1. 打开浏览器，访问 `https://code.visualstudio.com`，下载你系统对应的安装包。
2. 双击运行，一路点“下一步”装完。
3. 装完先别急着写代码，左侧活动栏最下方有四个方块的图标，点它打开扩展面板，搜索并安装两个**必装插件**：
   - **Python**（微软官方出的，负责语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更聪明）

装完这两个，VS Code 才能认出 `.py` 文件，你写代码

## 块 10

**目标对齐**：你要部署 vLLM，项目通常较大、依赖多。PyCharm Community 免费且开箱即用，适合管理这类大型 Python 项目。

**安装要点**（动手做）：
1. 打开官网下载页，选 **Community Edition**（社区版，免费）。
2. 运行安装程序时，**务必勾选**两项：
   - `Add "Open Folder as Project"`（右键快速打开项目文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**（关键步骤）：
1. 新建项目（`New Project`）。
2. `Location` 选项目路径——**路径中不要含中文**，避免后续 vLLM 相关工具解析出错。
3. `Interpreter` 选 `Previously configured interpreter`，指向系统已装的 **Python 3.10+**。

**动手验证**：安装完成后，用 PyCharm 新建一个空项目，确认能正常打开界面且解释器识别成功。这一步通过，你的 IDE 工作台就绪。

## 块 11

## 5.1 为什么用 Conda

你的目标是 vLLM 部署，这属于深度学习推理场景，依赖复杂（比如 PyTorch、CUDA 等）。如果直接在系统 Python 里装，版本冲突会让人崩溃。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身——这意味着你可以为 vLLM 项目单独建一个环境，指定 Python 版本，互不干扰。对深度学习来说，conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你的目标是部署 vLLM，环境越干净越好，用到什么装什么，避免无关包干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init

## 块 12

**创建你的第一个虚拟环境**

你的目标是部署 vLLM，而不同项目常需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y` 并回车，等待安装完成。

**

## 块 13

## 5.4 激活与退出环境

你的目标是部署 vLLM，所以现在要进入专门为这个项目准备的“独立房间”——`llm-learn` 环境。

**激活环境**（进入房间）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志。这意味着你之后敲的 `python` 和 `pip` 命令都只在这个环境里生效，不会弄乱系统或其他项目——这对 vLLM 部署至关重要，因为它的依赖包版本必须精确隔离。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。确认版本正确，再继续部署。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

## 5.5 查看和管理环境

**查看所有已创建的环境**：

```bash
conda env list
```

带 `*`

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是部署 vLLM，而这一步的关键是让 VS Code 用上你为项目准备的 `llm-learn` 环境——否则你写的代码可能跑在错误的 Python 上，导致包找不到。

**操作步骤：**

1. 先创建并激活环境：`conda create -n llm-learn python=3.10 -y`，然后 `conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表中，选择带有 `conda` 字样的 Python 路径，例如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这样写**：VS Code 能自动识别 conda 环境。选中后，当前工作区所有代码都会使用 `llm-learn` 里的 Python 和已装好的包，避免版本混乱。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程略有不同：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...

## 块 15

**动手验证：让 Python 先“开口说话”**

你的目标是部署 vLLM，但任何部署都从“环境能跑通”开始。这一步，我们只做一件事：**用命令行确认你的 Python 环境是活的**。

**第一步：激活专属环境**
在终端输入并回车：
```bash
conda activate llm-learn
```
看到命令行前面出现 `(llm-learn)` 前缀，说明环境已激活。

**第二步：进入 Python 交互模式**
输入 `python` 并回车，你会看到 `>>>` 提示符。这表示 Python 正在等你下指令。

**第三步：输入你的第一行代码**
在 `>>>` 后输入：
```python
print("Hello, Python! 我的环境准备好了！

## 块 16

### 7.3 在 PyCharm 中验证

你的目标是 vLLM 部署，但任何部署都从“能跑通一行代码”开始。这一步就是你的第一个里程碑。

操作很简单，三步：

1. **新建项目，创建 `hello.py`**  
   打开 PyCharm，新建一个项目，然后在项目里新建一个 Python 文件，命名为 `hello.py`。

2. **右键点击文件 → `Run 'hello'`**  
   在编辑区右键点击 `hello.py`，选择 `Run 'hello'`。PyCharm 会直接运行这个文件。

3. **下方 Run 窗口应输出结果**  
   看屏幕下方的 **Run** 窗口，那里会显示程序运行的结果。如果看到了输出，说明你的开发环境已经打通了。

> **给你的提醒**：这一步不需要理解代码含义，只需要确认“环境能跑”。跑通后，你后续学 vLLM 部署时，所有操作都会基于这个“能跑”的基础。

---

## 第八章：常见问题排查

环境搭建时最容易卡在下面几个问题上，遇到直接对照解决：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 |
