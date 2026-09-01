# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你现在的目标是**LoRA微调**，这需要你写代码、跑模型。但在那之前，得先让电脑“听得懂”你的话。

电脑只认0和1，你写的Python代码是给人看的文字。**Python解释器**就是中间的“翻译官”：它把你写的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆普通文本，电脑根本不会理睬。

**打个比方**：你给一个只懂中文的人写英文信，不翻译，对方看不懂。解释器就是这个翻译。

**本块学习检查**：完成本块后，用自己的话说说：Python解释器在你的LoRA微调目标中扮演什么角色？如果没装解释器，你写的微调代码会怎样？遇到不懂的术语（比如“解释器”“指令”）先记下来，课后提问。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

**1.2 代码编辑器（IDE）：你的"工作台"**

记事本也能写代码，但效率极低。**IDE**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

**1.3 虚拟环境：项目的"独立房间"**

想象你同时做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。如果装到同一个 Python 里，就会冲突报错。**虚拟环境**就是为每个项目创建独立的 Python 空间，互不影响。本文使用 **conda** 管理虚拟环境，这是目前

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，而 PyTorch 是它的基础。装好 Python，就是给 PyTorch 搭好“家”。下面三步，跟着做就行。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的“环境变量”中。如果不勾选，你打开命令行输入 `python` 时，系统会提示“不是内部或外部命令”，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 若安装成功，

## 块 4

## macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要 Python 环境。好消息是，macOS 上安装 Python 很简单，跟着下面任一方法做就行。

**先记住一个关键点**：在 macOS 上，请始终使用 `python3` 和 `pip3` 命令，而不是 `python`。macOS 自 Catalina 起不再预装 Python 2，系统可能自带 `/usr/bin/python3`（版本较旧），且默认不提供 `python` 命令，因此推荐使用 `python3` 和 `pip3` 以避免版本混淆。

### 方法一：官方安装包（推荐新手）

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入以下命令验证安装是否成功：

```bash
python3 --version
pip3 --version
```

看到版本号输出，就说明安装成功了。

### 方法二：Homebrew（适合已有 Homebrew 的开发者）

如果你已经装了 Homebrew（macOS 的包管理器），在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

Linux（Ubuntu/Debian）常预装Python，但版本可能偏老。为搭建PyTorch开发环境，先更新软件源并安装Python 3、pip和venv：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

`-y`自动确认安装，免手动输入`y`；`python3-venv`用于创建虚拟环境，一并装好避免后续补装。验证版本号确认安装成功。

## 块 6

## 2.4 操作系统与实训环境该怎么选

选对环境，能让你少走弯路。针对你的 **LoRA 微调**目标，环境选择很简单：

**如果你是 Windows 用户**，日常练 Python 和命令行时，推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`。

**如果你是 macOS / Linux 用户**，直接用本机终端即可，本身就是类 Unix 环境。

**关键点——GPU 实训**：做 LoRA / QLoRA 微调这类任务，必须用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。因为这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

> **⚠️ 别踩坑**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于 2024 年 5 月 31 日停止维护；VirtualBox 这类桌面虚拟机**默认无法直通 NVIDIA GPU**，跑不了真正的 GPU 训练。命令练习用 WSL2/Ubuntu L

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载，这是最权威、最完整、最应保留的默认源。

**只有**当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响你平时的默认行为，官方 PyPI 仍是首选源。

**

## 块 8

### 3.3 进阶：把镜像设为默认（只在长期网络受限时用）

如果你只是偶尔下载慢，**别急着改默认配置**。只有当你长期处于网络受限环境、几乎每次都要用镜像时，才考虑把镜像写入配置作为默认源。配置后请记住：官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

**Windows 用户操作步骤：**

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

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 3

## 块 9

## 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，要写 Python 脚本、改配置文件，它都够用。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**安装 Python**：

1. 访问 https://python.org，下载对应系统的 Python 安装包。
2. 运行安装程序，务必勾选 **Add Python to PATH**，然后点击 Install Now。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

装完插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 打开 VS

## 块 10

### 4.2 方案二：PyCharm Community（免费、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。内置代码分析、版本控制等功能，适合大型项目开发。对你后续做 LoRA 微调实验，管理复杂代码结构会很有帮助。

**安装步骤**：
1. 访问官网下载页。
2. 下载 **Community Edition**（免费版）。
3. 运行安装程序，Windows 用户勾选 `Add to PATH`（方便命令行启动）；macOS/Linux 用户可在终端执行 `ln -s /path/to/pycharm/bin/pycharm.sh /usr/local/bin/pycharm` 或修改 `~/.bashrc`。

**首次启动配置**：
1. 打开 PyCharm，选 `New Project`。
2. `Location` 选项目路径（**不要含中文**）。
3. `Interpreter` 选 `Previously configured interpreter`，点击 `Add Inter

## 块 11

### 5.1 为什么用 Conda

你已经知道虚拟环境能为每个项目创建独立的 Python 空间。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于你接下来的 LoRA 微调学习——需要 PyTorch、CUDA 等复杂依赖——conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

有两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，并在最后一步选择 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 conda 会通过初始化脚本正确接管，而不是靠把目录硬加进 PATH。

4. **验证安装**：打开一个**新的**终端（Windows 用 **"Anaconda Prompt (Miniconda3)"**；mac

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

**创建你的第一个虚拟环境**

**为什么这样做**：后续学习大模型微调（如LoRA）时，不同项目需要不同版本的PyTorch和CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：

## 块 13

## 5.4 激活与退出环境

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

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

带 `*` 的就是当前激活的环境。

**删除环境**（如果以后需要清理）：

```bash
conda remove -n llm-learn --all
```

---

## 第六章：在 VS Code / PyCharm 中使用 Conda

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

## 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选择 `Project: 你的项目名` → `Python Interpreter`，点击齿轮图标 → `Add...`，左侧选择 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件。若不确定路径，可在终端运行 `conda env list` 查看实际安装位置，再定位到对应环境的 `python` 可执行文件（Windows 在 `envs\llm-learn\` 目录下，macOS/Linux 在 `envs/llm-learn/bin/` 目录下）。

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：完成本块后，用自己的

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 7.1 在命令行验证

先激活环境：`conda activate llm-learn`，然后输入 `python` 进入交互模式。看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车后，如果正确打印文字，按 `exit()` 退出。

### 7.2 在 VS Code 中验证

新建文件夹 `my-first-project`，用 VS Code 打开。新建文件 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角

## 块 16

## 7.3 在 PyCharm 中验证——你的第一行代码

你已经完成了环境搭建，现在到了最激动人心的时刻：运行你的第一行 Python 代码。这一步验证的是“工具链是否真的通了”，它也是你后续学习 LoRA 微调的基础——如果连 `hello.py` 都跑不起来，后面加载模型、跑训练脚本会遇到更多麻烦。

**操作步骤（跟着做，大约 2 分钟）**

1. 打开 PyCharm，新建一个项目。
2. 在项目中创建一个新文件，命名为 `hello.py`。
3. 在文件中输入一行代码：`print("Hello, PyCharm!")`
4. 在文件编辑区**右键点击**，选择 `Run 'hello'`。
5. 看 PyCharm 下方的 **Run 窗口**，如果出现了 `Hello, PyCharm!`，说明你的 Python 环境、PyCharm 配置全部正常。

**如果没跑通，先别慌**

下面是环境搭建阶段最常见的几个问题，对照排查：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选择 “Modify”，勾选 “Add Python to environment variables” |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示“未选择 Python 解释器” | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误

## 测评题目

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？

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

### Q-KB0203-002
Python中，`*args` 接收的是________类型；`**kwargs` 接收的是________类型。

### Q-KB0201-001
以下哪个是Python中**正确的**变量赋值？

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

### Q-KB0205-007
写一个函数 `count_words(text)`，接收一段英文字符串，返回一个字典，统计每个单词出现的次数（忽略大小写）。
