# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。**Python 解释器**的作用，就是把你的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本。

**为什么你要先懂这个？** 因为你后面装环境时，装的就是这个"翻译官"。搞清楚它的角色，你就不会在安装时一脸茫然。

**动手验证一下（30秒）**：打开你的记事本，写一行 `print("hello")`，保存为 `test.py`。先双击运行——大概率没反应或闪退。这就是因为没有"翻译官"来翻译它。等你装好解释器后，再在命令行运行 `python test.py`，就能看到输出了。这个对比，能让你直观感受到解释器的存在价值。

记住：你写代码是给人看的，解释器负责把它变成机器能跑的东西。后面所有配置，都是为了让这个"翻译官"好好工作。

## 块 2

好的，我们直接进入正题。你的目标是理解Transformer架构，但工欲善其事，必先利其器。这一节，我们就把“工作台”搭好。

**1.2 代码编辑器（IDE）：你的“工作台”**

用记事本写代码效率太低。我们使用**IDE（集成开发环境）**，它提供语法高亮、自动补全、运行调试等功能，让你写代码更舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。你可以根据喜好选择。

**1.3 虚拟环境：项目的“独立房间”**

想象你同时做两个项目：A项目需要 `numpy 1.0`，B项目需要 `numpy 2.0`。都装到同一个Python里会

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你正为理解 Transformer 架构做准备，动手搭建环境是第一步。跟着下面三步走，快速搞定。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`。
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --

## 块 4

### 2.2 macOS 用户：装好 Python，马上动手

你的目标是理解 Transformer，而动手跑代码是最好的路径。但第一步，得先把 Python 环境弄干净。macOS 从 Monterey 12.3 起不再预装 Python 2，系统可能带旧版 `python3`，但别直接用——**强烈建议自己装最新版**，避免依赖系统版本。

**方法一：官方安装包（推荐，省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install"。
3. 打开终端（Terminal），输入：

```bash
python3 --version
pip3 --version
```

看到版本号就成功了。

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认**没有** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统统一用 `python3` / `pip3`，所以请始终这么敲，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（如果你已有它）**

在终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：打开终端，输入 `python3 --version` 并截图或记下输出

## 块 5

**动手装：Ubuntu/Debian 上安装 Python 3**

你的系统可能自带 Python，但版本偏旧。直接跑下面三行命令，一次搞定安装和验证：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**命令拆解**：  
- `sudo apt update`：刷新软件源，让系统知道有哪些新版本可装。  
- 第二条命令一口气装三样：`python3`（解释器）、`python3-pip`（装第三方库的工具）、`python3-venv`（建虚拟环境用）。  
- 结尾的 `-y` 是“自动确认”，省得你手动敲

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是理解 Transformer 架构，最终要上手跑 GPU 实训。那环境选择就直接影响你后面能不能顺利跑起来，这里给你一条最省力的路线。

**日常写 Python、练命令行（Windows 用户）**

别在 Windows 原生 CMD 里练命令，那和真实服务器差距太大。推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，练完直接无缝迁移到云端。安装很简单：管理员 PowerShell 里执行 `wsl --install` 即可。

**日常写 Python（macOS / Linux 用户）**

直接用本机终端，本身就是类 Unix 环境，不用额外折腾。

**GPU 实训（LoRA / QLoRA 微调、vLLM 部署）**

这是你后面要碰的核心场景。用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全，遇到问题搜得到答案。

**⚠️ 避坑提醒**

别把 **"CentOS + VirtualBox

## 块 7

**本块学习检查**：在终端执行 `pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`，观察是否成功安装，并确认官方源仍是默认设置。

---

**3.1 什么时候才需要镜像源**

`pip` 是 Python 的包管理工具，用来装第三方库（比如 `numpy`、`pandas`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载，这是最权威、最完整、最应保留的默认源。

**只有**当你处在网络受限环境、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置——官方源始终应作为首选。

**3.2 首选做法：临时指定镜像源（推荐）**

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限）

如果你只是偶尔下载慢，**不要**做这一步。只有当你的网络环境长期受限、几乎每次装包都需要镜像时，才值得把镜像写进配置文件作为默认源。配置后请记住：**官方 PyPI 永远是最可靠的来源**，等网络恢复正常后，建议改回或删除该配置。

### Windows 用户操作

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

如果看到 `global.index-url='https://pypi.tuna

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你正在学 Transformer，后续调试 Python 代码、查看训练脚本时，它都能派上用场。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select

## 块 10

**动手装好 PyCharm，开始你的 Transformer 项目**

你正在学习 Transformer 架构，接下来要写代码验证注意力机制。工欲善其事，必先利其器——我们选用 **PyCharm Community**（社区版，免费）作为开发工具，它专为 Python 设计，内置代码分析、版本控制等功能，适合你后续搭建和调试 Transformer 模型。

**安装三步走：**

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选择 **Community Edition**（社区版）下载
3. 运行安装程序，务必勾选这两项：
   - `Add "Open Folder as Project"`（以后右键文件夹就能用 PyCharm 打开）
   - `Add to PATH`（命令行可直接启动）

**首次启动配置：**

1. 打开 PyCharm，点击 `New Project`
2. 在 `Location` 设置项目路径——**路径中不要包含中文**，避免后续编码问题
3. `Interpreter` 选择 `Previously configured interpreter`，然后点击右侧 `...` 按钮，在弹出的窗口中浏览

## 块 11

### 5.1 为什么用 Conda

你已经知道虚拟环境能为每个项目创建独立的 Python 空间。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于你接下来要学习的 Transformer（需要 PyTorch、CUDA 等复杂依赖），conda 几乎是必备工具——它能帮你把每个项目的依赖“关在各自的房间里”，互不干扰。

### 5.2 安装 Conda：Anaconda vs Miniconda

有两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB 空间），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 50-80MB，安装后占用空间约 400MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要什么装什么——这正好符合你“动手实践”的风格，装包过程本身就是理解依赖关系的好机会。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。
   - **macOS/Linux**：若使用 `.sh` 文件，按提示输入 `yes` 同意协议

## 块 12

**动手创建你的第一个虚拟环境**

你正在为理解 Transformer 架构打基础，而大模型项目常需要不同版本的 PyTorch 和 CUDA。现在学会创建隔离环境，以后就能直接复用这套流程，避免版本冲突。

打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**这条命令在做什么？** 拆开看：

- `conda create`：告诉 Conda 你要新建一个环境
- `-n llm-learn`：给这个环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境内安装 Python 3.10 版本

执行后，终端会列出准备安装的包，

## 块 13

### 5.4 激活与退出环境：进入你的“独立房间”

你正在为理解Transformer架构搭建环境，这个环境就像你的专属工作台。**激活环境**就是走进这个房间：

```bash
conda create -n llm-learn python=3.10
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志——这是你的“门牌号”，表示后续所有 `python` 和 `pip` 命令都只作用于这个环境，不会干扰系统或其他项目。

**快速验证**：运行 `python --version`，应显示 `Python 3.10.x`，确认你已站在正确的“楼层”。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**查看所有环境**：

```bash
conda env list
```

带 `*` 号的就是当前激活的环境。

**

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的路径（如 `Python 3.10.x ('llm-learn': conda)`）。VS Code 会自动识别 conda 环境，选好后，工作区代码就会用 `llm-learn` 里的 Python 和已装包运行。

## 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选 `Project: 你的项目名` → `Python Interpreter`，点齿轮图标 → `Add...`，左侧选 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件：
- Windows：`%USERPROFILE%\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`$HOME/miniconda3/envs/llm-learn/bin/python`

请根据你的实际安装路径调整上述路径。

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：在 VS Code 中新建一个 `.py` 文件，输入 `import torch; print(torch.__version__)`，运行。若输出版本

## 块 15

**动手验证：让 Python 说“你好”**

你的目标是理解 Transformer，但先别急——任何模型都跑在能工作的环境里。咱们用最快的方式确认环境没问题。

**步骤 1：激活环境**
在命令行输入：
```bash
conda activate llm-learn
```

**步骤 2：进入 Python 交互模式**
输入 `python` 并回车，看到 `>>>` 提示符就说明成功了。

**步骤 3：运行你的第一行代码**
在 `>>>` 后输入：
```python
print("Hello, Python! 我的环境准备好了！")
```
回车后，如果看到同样的文字打印出来，环境就验证通过了。最后输入 `exit()` 退出。

**本块学习检查**
在命令行中重复步骤 1-3

## 块 16

## 7.3 在 PyCharm 中验证

你的目标是理解 Transformer 架构，但在此之前，先把“地基”踩实——确保代码能跑起来。这一步不涉及模型原理，只关乎工具链是否通畅。

**操作步骤（跟着做，3分钟搞定）：**

1. 打开 PyCharm，新建一个项目（路径不要包含中文或空格）。
2. 在项目中创建文件 `hello.py`。
3. 在文件中输入一行最简单的代码：`print("hello")`。
4. 右键点击文件 → 选择 `Run 'hello'`。
5. 观察下方弹出的 **Run 窗口**——如果看到输出了 `hello`，说明你的开发环境完全正常，可以继续往下走。

**如果没跑通，先查这张表：**

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选择 “Modify”，勾选 “Add Python to environment variables” |
| PyCharm 提示“未选择 Python 解释器” | 未配置解释器路径 | 进入 `File` → `Settings` → `Project` → `Python Interpreter`，手动选择解释器 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |

**本章完成后，你的电脑应具备：**

| 组件 | 状态 | 验证命令 |
|------|------|----------|
| Python 3.10+ | ✅ 已安装 | `python --version` |
| PyCharm | ✅ 已安装 | 打开软件确认 |
| Miniconda / Anaconda | ✅ 已安装 | `conda --version` |
| Conda 虚拟环境 `llm-learn` | ✅ 已创建 | `conda env list` |

**养成的好习惯（对你后续学 Transformer 尤其重要）：**

1. **一个项目一个 conda 环境**，不要在 `base` 环境里乱装包——Transformer 相关的 PyTorch 等依赖很多，环境隔离能避免版本冲突。
2. 安装第三方库前先 `conda
