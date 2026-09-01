# 学习包

## 块 1

## 1.1 Python 解释器：程序的“翻译官”

你现在的目标是 LoRA 微调，但别急着跳进模型训练——先想清楚一个底层问题：你写的代码，电脑凭什么能听懂？

答案是：**Python 解释器**。

你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。Python 解释器的作用，就是把你的代码“翻译”成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本，什么也干不了。

**为什么这对你特别重要？**

你接下来要做的 LoRA 微调，本质上是一连串 Python 脚本在驱动 PyTorch 和 HuggingFace 工具链干活。如果解释器没装对，你连第一行 `import torch` 都跑不起来——不是代码写错了，而是“翻译官”没上岗。

**一个直观的理解方式：**

把解释器想象成一个同声传译。你用中文（Python 代码）说了一句话，传译员（解释器）把它翻译成英文（机器指令）讲给电脑听。传译员水平不行，意思就会传错；传译员没到场，你说什么都没用。

**本块学习检查**

请用自己的话回答：为什么没有 Python 解释器，`.py` 文件就“只是一堆文本”？如果你在终端里运行一个 Python 脚本时报错“无法识别该命令”，你觉得最可能的原因是什么？

## 块 2

**本块学习检查**：读完本节后，合上资料，口头回答：IDE 和记事本写代码的最大区别是什么？虚拟环境解决了什么问题？

---

### 1.2 代码编辑器（IDE）：你的"工作台"

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。对 LoRA 微调，两者都够用，选顺手的即可。

### 1.3 虚拟环境：项目的"独立房间"

假设你同时做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。都装进同一个 Python 会冲突报错。**虚拟环境**为每个项目创建独立 Python 空间，互不影响。

## 块 3

**为你的 LoRA 微调目标铺路：Windows 安装 Python 3.10+**

你正在为 LoRA 微调搭建环境，而 Python 是一切的基础。下面我们一步步来，每个细节都关系到后续能否顺利运行训练脚本。

**第一步：下载安装包**

打开浏览器，访问 Python 官网 `https://www.python.org`。把鼠标移到顶部菜单的 **Downloads** 上，网站会自动推荐适合你 Windows 系统的最新稳定版本（如 Python 3.13.x 或更高版本）。点击下载 **Windows installer (64-bit)** 这个安装文件。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载好的 `.exe` 文件，看到安装界面后，**有两件事必须做对**：

1. **务必勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 加入系统的"环境变量"。如果不勾选，之后你在命令行里输入 `python`，系统会提示"不是内部或外部命令"，所有后续操作都会卡在这一步。这是初学者最容易踩的坑。

2. 点击 **"Install Now"** 使用默认安装。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，这个路径不含中文和特殊符号，可以避免后续因路径问题报错。如果你以后想改安装路径，务必确保新路径中**没有中文或空格**。

**第三步：验证

## 块 4

## 2.2 macOS 用户：安装 Python 3.10+

**先了解现状**：较老的 macOS 曾预装 Python 2.7，macOS Monterey 12.3 起已不再预装 Python 2；部分系统可能带有供系统脚本使用的旧版 `python3`。无论哪种情况，都**强烈建议自行安装最新版 Python**，避免直接依赖系统自带版本。

### 方法一：官方安装包（推荐新手）

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：在现代 macOS 上，命令行默认**并不提供** `python` 命令（早期版本曾指向系统自带的 Python 2，现已移除），系统与官方安装包统一使用 `python3`。因此请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

### 方法二：Homebrew（适合已有 Homebrew 的开发者）

如果你已经安装了 Homebrew（macOS 的包管理器），在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证

## 块 5

**本块学习检查**：在终端依次执行三条命令后，用 `python3 --version` 看到的输出，就是你的 Python 版本号。若显示以 `3.10` 或更高开头，说明安装成功。

---

Linux 系统通常自带 Python，但版本可能偏旧。按下面三步操作，即可完成安装：

1. **更新软件源**：运行 `sudo apt update`，让系统获取最新软件列表。
2. **安装 Python 3 及配套工具**：运行 `sudo apt install python3 python3-pip python3-venv -y`。这里 `-y` 表示自动确认，无需手动输入 `y`；`python3-venv` 是创建虚拟环境必需的模块。

## 块 6

### 2.4 操作系统与实训环境该怎么选

你正在为 LoRA 微调打基础，环境选对了，后面跑 GPU 实训会顺畅很多。结合你手头的 RTX 3060 12GB，重点看下面第三点。

**第一步：日常练 Python 和命令行**
- 如果你用 **Windows**，强烈建议启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04 或 24.04）。它让你在 Windows 里直接获得一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和以后服务器上的一致。安装很简单，管理员 PowerShell 里执行 `wsl --install` 就行。
- 如果你用 **macOS 或 Linux**，直接开终端用即可，本身就是类 Unix 环境，不用额外配置。

**第二步：理解为什么这么选**
WSL2 比 Windows 原生 CMD 更贴近实战，因为深度学习生态（包括你后续要用的工具链）在 Linux 下支持最成熟。现在养成 Linux 习惯，后面少踩坑。

**第三步：GPU 实训（你的核心场景）**
LoRA 微调这类任务依赖 NVIDIA 驱动、CUDA 和显卡直通。**请使用装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、最应保留的默认源**。

**你只需要记住一个判断标准**：只有当访问官方源明显变慢或经常超时时，才临时换用国内镜像。镜像只是“网络不好时的加速手段”，**不是永久设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响你平时的默认行为，官方 PyPI 仍是默认源；

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

> **先判断，再动手**：如果你只是偶尔下载慢，用 `-i` 参数临时指定镜像就够了。只有**长期处于网络受限环境**、几乎每次都要用镜像时，才考虑把它写入配置作为默认源。配置后请记得：官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

### Windows 用户操作步骤

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

### macOS / Linux 用户操作步骤

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

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 改回官方源

## 块 9

## 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 打开 VS Code，按 `Ctrl + Shift + P

## 块 10

**本块学习检查**：安装完成后，在 PyCharm 中新建一个项目，确认你能找到“解释器”设置，并说出它指向的 Python 版本号。

---

PyCharm Community 是 JetBrains 出品的免费 Python 专用 IDE，内置代码分析、版本控制等功能，适合大型项目。既然你的目标是 LoRA 微调，后续会涉及大量代码调试，选它省心。

**安装要点**：
1. 访问官网下载页，选 **Community Edition**（免费版）。
2. 安装时勾选两项：`Add "Open Folder as Project"`（方便右键打开项目）和 `Add to PATH`（命令行直接启动）。

**首次启动**：
1. 点击 `New Project`。
2. 在 `Location` 设置项目路径——**务必不含中文**，否则后续环境易出乱码。
3. `Interpreter` 选 `Previously configured interpreter`，指向你已装的 Python 3.10+。

**为什么强调解释器**：LoRA 微

## 块 11

### 5.1 为什么用 Conda

你已经知道虚拟环境能为每个项目创建独立的 Python 空间。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于你接下来的目标——**LoRA 微调**，需要安装 PyTorch、CUDA 等复杂依赖，conda 几乎是必备工具。它能帮你把不同项目的依赖彻底隔离开，避免“装了这个库，那个项目就跑不了”的混乱。

### 5.2 安装 Conda：Anaconda vs Miniconda

有两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 70-80MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包。你的 RTX 3060 12GB 显卡在深度学习任务中完全够用，但依赖管理必须干净，否则 CUDA 版本冲突会让你寸步难行。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可

## 块 12

**本块学习检查**：在终端执行 `conda create -n llm-learn python=3.10`，观察输出中的 `Proceed ([y]/n)?`，输入 `y` 回车，确认环境创建成功（即终端回到提示符且无报错）。

---

你连续两次低分，我们放慢节奏，只学一个命令。目标是为 LoRA 微调准备独立环境，避免不同项目互相干扰。

**为什么这样做**：后续微调大模型时，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（三步记忆）：
- `conda create

## 块 13

## 5.4 激活与退出环境

你正在为 LoRA 微调搭建独立工作台。Conda 环境就像一个“独立房间”，让项目依赖互不干扰。

**进入房间**——激活环境：

```bash
conda activate llm-learn
```

激活后，命令行前会出现 `(llm-learn)` 标志。这意味着后续所有 `python` 和 `pip` 命令都只作用于这个环境，不会影响系统或其他项目——这正是微调实验需要的隔离性。

**确认房间配置**——验证 Python 版本：

```bash
python --version
```

应显示 `Python 3.10.x`，确保环境符合预期。

**离开房间**——退出环境：

```bash
conda deactivate
```

回到系统默认环境。

### 5.5 查看和管理环境

**查看所有环境**：

```bash
conda env list
```

带 `*` 的是

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。然后打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 会自动识别 conda 环境。选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

## 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选择 `Project: 你的项目名` → `Python Interpreter`。点击齿轮图标 → `Add...`。左侧选择 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：

- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：在 VS Code 中新建一个 `.py` 文件，输入 `import torch; print(torch.__version__)`，运行后应输出 PyTorch

## 块 15

### 在命令行验证你的 Python 环境

你已经连续两次在测试中遇到困难，我们先把基础打牢。这一步只做一件事：确认你的 Python 能跑起来。

**操作步骤：**

1. **激活环境**：在终端输入 `conda activate llm-learn`，回车。
2. **进入 Python**：输入 `python`，回车。你会看到 `>>>` 提示符，这表示已进入交互模式。
3. **运行第一行代码**：在 `>>>` 后输入 `print("Hello, Python! 我的环境准备好了！")`，回车。
4. **确认输出**：屏幕上应显示同样的文字。看到后，输入 `exit()` 退出。

**本块学习检查**

完成上述步骤后，请回答：你看到的

## 块 16

## 7.3 在 PyCharm 中验证——你的第一行代码

环境已经搭好，现在是时候让代码真正跑起来了。这一步的目标很简单：**确认你的开发工作台能正常运行 Python 程序**。

### 操作步骤（三步完成）

1. **新建项目**：打开 PyCharm，创建一个新项目（项目路径**不要包含中文或空格**，这是本章强调的好习惯）。
2. **创建 `hello.py` 文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3. **运行它**：在编辑区右键点击文件 → 选择 `Run 'hello'`。

### 你应当看到的结果

点击运行后，PyCharm **下方会弹出 Run 窗口**，窗口中应显示你写在 `hello.py` 里的代码输出结果。看到输出，就说明你的 Python 解释器、项目配置和 PyCharm 之间的链路是通的——这是你后续所有代码实验的基础。

> **学习提示**：如果你连续两次测试不理想，不必焦虑。这一章的核心不是写代码，而是**确认工具链通畅**。只要 Run 窗口有输出，你就已经完成了环境验证中最关键的一步。

---

## 常见问题排查（遇到再看）

运行过程中如果出现异常，对照下表快速定位：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源），示例：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>`；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P

## 测评题目

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0201-005
以下代码输出什么？
```python
for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
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

### Q-KB0202-002
字典 `d = {"a": 1, "b": 2}`，获取键 `"a"` 对应值的方法是________；安全获取不存在的键 `"c"`（返回默认值0）的方法是________。

### Q-KB0203-006
关于Python的MRO（方法解析顺序），以下说法**错误**的是？

### Q-KB0201-004
以下代码输出是？
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```

### Q-KB0202-003
以下代码输出是？
```python
s = {1, 2, 3, 3, 2, 1}
print(len(s))
```
