# 学习包

## 块 1

**本块学习检查**：读完这一节后，请合上资料，用自己的话回答："Python 解释器是干什么的？没有它，你写的 .py 文件是什么？" 能说清楚，就说明你理解了。

---

### 1.1 Python 解释器：程序的"翻译官"

你的目标是用 vLLM 部署模型，但先别急着碰那些高级工具。**任何 Python 项目的地基，都是这个"翻译官"**——vLLM 本身也是用 Python 写的，跑它之前，你得先让电脑能"读懂"Python。

**它到底解决什么问题？**

你写的代码是给人看的文字，比如 `print("hello")`。但电脑的 CPU 只认 0 和 1。**Python 解释器**就是那个中间人：它把你写的每一行代码，实时翻译成电脑能执行的机器指令。

**为什么你必须先懂这个？**

很多新手直接下载安装包，一路点"下一步"，装完还是一脸懵。但如果你先花 2 分钟搞懂这个"翻译官"的角色，后面每一步你都知道自己在干什么：你是在请一个翻译官住进你的电脑。

**一个关键认知**：没有解释器，你写的 `.py` 文件**只是一堆文本**，电脑根本不会执行它。你写的代码再漂亮，也只是一封电脑看不懂的信。

**动手验证（不装软件也能做）**：打开你的文件管理器，随便新建一个 `test.py` 文件

## 块 2

### 你的“工作台”：从记事本到 IDE

写代码用记事本也能行，但就像用手写代替 Word，效率天差地别。**IDE（集成开发环境）** 就是你的专业工作台，自带语法高亮、自动补全、运行调试，让你写代码像写文档一样顺手。针对你 **vLLM 部署** 的目标，推荐两款：**VS Code**（轻量免费，上手快）和 **PyCharm**（专业强大，功能全）。选 VS Code 起步即可，够用且不拖累学习节奏。

### 虚拟环境：项目的“独立房间”

部署 vLLM 时，你可能会同时折腾多个项目。比如 A 项目要 `numpy 1

## 块 3

**目标：为 vLLM 部署铺路，先把 Python 环境搞定。**

你学 vLLM 是冲着部署去的，但部署前得先有 Python 这个“地基”。下面三步，跟着做就行。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`。
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（比如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这一步直接决定你后面能不能用命令行。不勾选的话，你打开终端输入 `python`，系统会提示“不是内部或外部命令”，后续所有操作都卡住。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 Python 环境是第一步。现代 macOS（Catalina 10.15 起）已不再预装 Python 2，系统自带的旧版 `python3` 仅供系统脚本使用，**不要依赖它**——直接装最新的官方版本，干净又省心。

#### 方法一：官方安装包（推荐，最快上手）

1. 打开浏览器，访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点 "Continue" → "Install"，输密码完成安装。
3. 打开终端（Terminal），输入下面两条命令验证：

```bash
python3 --version
pip3 --version
```

看到版本号输出就成功了。

**关键提醒**：在 macOS 上，命令行默认**没有** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统与官方安装包统一使用 `python3`。所以**始终用 `python3` / `pip3`**，别用 `python`，否则会碰到"命令不存在"或误用系统旧版的坑。

#### 方法二：Homebrew（你已装过的话）

如果你已经有 Homebrew（macOS 包管理器），终端执行：

```bash
brew install python

## 块 5

**本块学习检查**：运行 `python3 --version`，确认输出为 `Python 3.10+` 格式（如 `Python 3.10.12`），若版本过低，后续 vLLM 部署可能受阻。

---

你的目标是 vLLM 部署，但先得让 Python 就绪。Linux 常自带 Python，但版本偏老，直接装新的最稳。

**三步走（复制即用）**：

```bash
sudo apt update          # 刷新软件源，确保能拿到最新包
sudo apt install python3 python3-pip python3-venv -y   # 一键装齐
python3 --version        # 验证版本
```

**关键点**：`-y`

## 块 6

## 环境选择：别让工具拖慢你的 vLLM 部署

你的目标是部署 vLLM，这意味着你最终要在 Linux + NVIDIA GPU 环境里跑推理。那现在该用什么环境练手？记住一句话：**日常练习贴近生产，GPU 实训直接上生产环境**。

**日常写 Python、练命令（Windows 用户）**

启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。为什么不用 Windows 原生 CMD？因为 WSL2 里的命令、路径、包管理方式，和你后续要操作的服务器完全一致——你现在敲的每一行命令，将来在 GPU 服务器上都能直接复用。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**

直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**vLLM 部署等 GPU 实训**

使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 避坑提醒**

不要用 **"CentOS + VirtualBox 虚拟机"** 当主力方案。两个原因：CentOS 8 已于 202

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**关键点**：只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是"网络不好时的临时加速手段"，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：它不影响你平时的默认行为，

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

**什么时候才需要这一步？**  
只有当你长期处于网络受限环境、几乎每次装包都得靠镜像加速时，才值得把镜像写进配置文件设为默认。注意：官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除该配置。

---

**Windows 用户操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 并回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

---

**macOS / Linux 用户操作步骤：**

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

---

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤：**

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器：**

1. 打开 VS Code，按 `Ctrl + Shift + P

## 块 10

**本块学习检查**：完成安装后，打开 PyCharm 新建项目，确认你能找到“Interpreter”选项并看到系统 Python 3.10+ 的路径。

---

你的目标是 vLLM 部署，这需要处理大型项目。PyCharm Community 是 JetBrains 出品的免费 Python 专用 IDE，内置代码分析、版本控制等功能，适合你这种“开箱即用”的需求。

**安装三步走：**
1. 访问官网下载页（https://www.jetbrains.com/pycharm/download/），选 **Community Edition**（免费版）。
2. 运行安装程序，务必勾选两项：`Add "Open Folder as Project"`（右键快速打开文件夹）和 `Add to PATH`（命令行直接启动）。
3. 首次启动选 `New Project`，在 `Location` 设置项目路径——**路径别带中文**，避免后续编码问题。

**关键一步**：在 `Interpreter` 处选 `Previously configured interpreter`，指向你系统里已装的 Python 3.10+。这确保 vLLM 部署时能正确

## 块 11

### 5.1 为什么用 Conda

你的目标是 vLLM 部署，这属于深度学习推理场景，依赖 PyTorch、CUDA 等复杂库。如果直接在系统 Python 里装，很容易出现版本冲突——比如 A 项目要 PyTorch 2.0，B 项目要 1.13，装来装去系统就乱了。

**Conda** 就是来解决这个问题的。它不只是虚拟环境工具，还能直接管理 Python 解释器版本。比如某个项目需要 Python 3.8，另一个要 3.10，用 conda 可以各自建一个独立环境，互不干扰。对深度学习来说，conda 几乎是标配。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| **Anaconda** | 约 500MB+，预装 numpy、pandas、jupyter 等 | 做数据分析、不想手动装包 |
| **Miniconda** | 约 50MB，只有 conda 和基础 Python | 希望环境干净、按需装包 |

**推荐 Miniconda**。你之后要装的是 vLLM 相关的包，环境越干净越好，避免无关库干扰。

**安装步骤：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS/Linux 选 `.sh`）
3. 运行安装：
   - **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器也标了"不推荐"）。直接加进 PATH 容易和系统已有的 Python 冲突，导致 `python`、`pip` 指向混乱。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来操作 conda。如果不勾选，conda 命令只在 Anaconda Prompt 中可用；若想在普通终端使用，需手动将 conda 的 Scripts 目录加入 PATH。
   - **

## 块 12

**动手做：创建你的第一个虚拟环境**

你的目标是 vLLM 部署，而大模型项目常需要不同版本的 PyTorch 和 CUDA。学会创建环境，以后每个项目都能独立隔离，互不干扰。

打开终端，执行这条命令：

```bash
conda create -n llm-learn python=3.10
```

**拆解一下**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定安装 Python 3.10

执行后，终端会列出待安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y`

## 块 13

**本块学习检查**：在终端依次执行 `conda activate llm-learn`、`python --version`、`conda deactivate`，观察并记录每一步命令行前缀的变化。

---

## 5.4 激活与退出环境：进入你的“独立房间”

你的目标是 vLLM 部署，这意味着你马上要装一堆依赖包。如果直接装在系统 Python 里，项目之间会互相“打架”。Conda 环境就是给每个项目一个独立房间，互不干扰。

**激活环境**（进入房间）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志。看到它，就表示后续所有 `python` 和 `pip` 命令都只作用于这个环境，不会影响系统或其他项目——这是你部署 vLLM 时的安全隔离区。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 自动识别 conda 环境，选中后，工作区代码就会使用 `llm-learn` 中的 Python 和已装包。

## 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选 `Project: 你的项目名` → `Python Interpreter`。点击齿轮图标 → `Add...`，左侧选 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件：

- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境搭好后必须验证，确保每个环节都通。

**本块学习检查**：在 VS Code 中完成环境选择后，打开终端运行：

```python -c "import sys; print(sys.executable)"```

## 块 15

**本块学习检查**：完成下面两步后，在终端输入 `exit()` 退出，并确认你看到了 `Hello, Python! 我的环境准备好了！`。

你的目标是 vLLM 部署，但先别急——环境没验证好，后面跑模型会寸步难行。我们直接上手，两步搞定。

**第一步：命令行验证（最快，30秒）**

1. 打开终端，输入 `conda activate llm-learn` 激活环境。
2. 输入 `python` 回车，看到 `>>>` 提示符。
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 回车。
4. 看到文字打印出来，输入 `exit()` 退出。

**第二步：VS Code 验证（日常开发

## 块 16

## 7.3 在 PyCharm 中验证——跑通你的第一行代码

环境已经搭好，现在用 PyCharm 验证一切是否正常。操作很简单，三步走：

1. **新建项目**，在项目中创建一个新文件，命名为 `hello.py`。
2. **运行文件**：在 `hello.py` 上右键点击，选择 `Run 'hello'`。
3. **查看结果**：PyCharm 下方会弹出 **Run 窗口**，里面会显示程序运行输出的结果。

看到输出内容，说明你的 Python 开发工作台已经真正跑通了。

---

## 第八章：常见问题排查（遇到再看）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，用 `python -m pip install 包名` 安装 |

---

## 本章小结：环境清单自查

| 组件 | 状态 | 验证命令 |
|------|------|----------|
|
