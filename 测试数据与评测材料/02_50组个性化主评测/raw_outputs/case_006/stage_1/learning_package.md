# 学习包

## 块 1

**个性化教学正文**

你好！作为数学专业硕士，你对抽象概念和逻辑结构应该非常熟悉。这一节我们只讲一个核心概念，用你熟悉的"映射"思维来理解，会非常轻松。

**1.1 Python 解释器：程序的"翻译官"**

想象一下：你写的 Python 代码，本质上是给人类阅读的"数学符号语言"；而电脑CPU只认识二进制（0和1）。这两者之间，需要一个"翻译"。

**Python 解释器**就是这位翻译官。它的职责是：**把你写的代码，逐句翻译成电脑能执行的机器指令**。

用你的数学直觉来类比：

- 你写的代码（`.py`文件） = 一个用"高级语言"描述的算法（比如矩阵乘法的步骤）。
- 电脑的CPU = 只能执行最基础加减和逻辑判断的硬件。
- 解释器 = 那个把算法步骤"翻译"成硬件指令的中间层。

**关键点**：没有这个解释器，你写的 `.py` 文件对电脑来说，就是一堆无法理解的普通文本，不会产生任何计算效果。

**本块学习检查**（请先思考，再继续）：

请用一句话向一位文科生解释：为什么我们不能直接用中文或英文给电脑下命令，而必须经过"解释器"这一层？提示：从"电脑只认0和1"这个事实出发。

## 块 2

### 1.2 代码编辑器（IDE）：你的"工作台"

你不需要用记事本写代码——那效率太低。**IDE（集成开发环境）** 把语法高亮、自动补全、运行调试集成在一起，让你写代码像用 Word 写文档一样顺手。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。初学选 VS Code 足够，它启动快、界面简洁，适合你每天 1 小时的专注学习节奏。

## 块 3

## 2.1 Windows 用户：安装 Python 3.10+

你不需要编程经验，只需按顺序完成以下三步。整个过程大约需要 5 分钟。

### 第一步：下载安装包

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

### 第二步：运行安装程序（⚠️ 关键步骤）

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

### 第三步：验证安装

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

## 块 4

## 2.2 macOS 用户：安装 Python 3.10+

你作为数学专业背景的学习者，对“环境”这个概念应该不陌生——就像做数值计算前要确认软件版本一样，Python 环境也需要你主动搭建。macOS 12.3 起不再预装 Python 2，部分系统虽有旧版 `python3` 供系统脚本使用，但**强烈建议自行安装最新版**，避免依赖系统自带版本带来的不确定性。

### 方法一：官方安装包（推荐新手）

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 上，命令行默认**并不提供** `python` 命令（早期版本曾指向系统自带的 Python 2，现已移除），系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到“命令不存在”或误用系统版本的坑。

### 方法二：Homebrew（适合已有 Homebrew 的开发者）

若你已安装 Homebrew（macOS

## 块 5

## 2.3 Linux 用户（Ubuntu/Debian）安装 Python

你的 Ubuntu/Debian 系统很可能已经预装了 Python，但版本可能偏旧。为了确保后续学习 Transformer 时环境一致，我们统一安装 Python 3 及配套工具。

打开终端，依次执行以下三条命令：

```bash
# 第一步：更新软件源，让系统知道最新可用的软件包
sudo apt update

# 第二步：安装 Python 3、pip 和 venv
sudo apt install python3 python3-pip python3-venv -y

# 第三步：验证安装结果
python3 --version
```

**命令解读**（帮你理解而非死记）：

- `sudo`：以管理员权限执行，系统会

## 块 6

## 2.4 操作系统与实训环境该怎么选

作为数学背景的学习者，你不需要纠结“哪个系统最好”，只需按学习阶段选对工具。这里给你一条清晰的分流路线：

**如果你在 Windows 上日常写 Python、练命令行**，推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它相当于在 Windows 内部运行一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致，比在 Windows 原生 CMD 里练习更贴近实战。安装只需在管理员 PowerShell 中执行 `wsl --install` 即可。

**如果你用 macOS 或 Linux**，直接使用本机终端即可，本身就是类 Unix 环境，无需额外配置。

**关于 GPU 实训（如 LoRA/QLoRA 微调、vLLM 部署）**：这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **CentOS + VirtualBox 虚拟机**当作最佳方案。CentOS 8 已停止维护，CentOS 7 也已停止维护，建议使用

## 块 7

**3.1 什么时候才需要镜像源——给数学背景的你**

想象 `pip` 是 Python 的“快递员”，默认从官方仓库 PyPI（`pypi.org`）取货——这是最权威、最完整的源头，应始终作为首选。就像你在数学中优先引用原始文献而非二手转述。

**关键判断**：只有当你身处网络受限环境、访问官方源明显变慢或频繁超时时，才临时改用国内镜像。镜像本质是“网络拥堵时的加速通道”，**不是永久替代品**——网络恢复后应回归官方源。

**3.2 首选做法：临时指定镜像（推荐）**

若某次安装 `numpy` 卡顿，最干净的做法是仅对这一次操作指定镜像，不触碰全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：这好比在计算中临时换一种

## 块 8

## 3.3 进阶：把镜像设为默认（仅长期网络受限时用）

如果你只是偶尔下载慢，**不建议**做这一步。只有当你的网络环境长期受限、几乎每次都需要镜像时，才把镜像写入配置作为默认源。请记住：**官方 PyPI 始终是最可靠的来源**，网络恢复后建议改回或删除该配置。

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

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

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

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你作为数学背景的初学者，后续写 Python 脚本、查看配置文件时都用得上，一次安装长期受益。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。对无编程经验的你来说，这些提示能大幅降低起步难度。

**配置 Python 解释器**：

1

## 块 10

好的，数学硕士，我们开始搭建你的Python开发工作台。既然你熟悉严谨的数学推导，我们可以把IDE理解为一个“高级计算器”——它帮你管理代码、运行结果和项目文件，让你专注于逻辑本身。

本块我们选择 **PyCharm Community**，它免费且功能完整，适合你后续进行深度学习项目。安装时，请务必勾选两个选项：`Add "Open Folder as Project"`（方便你右键直接打开项目文件夹）和 `Add to PATH`（让你能在命令行直接启动它）。

首次启动时，选择 `New Project`。**关键一步**：在 `Location` 处选择项目存放路径，**路径中不要包含中文**，这可以避免很多潜在的编码错误。在 `Interpreter` 处，选择 `Previously configured interpreter`，并指定你系统里已安装的 Python 3.10+ 版本。

**本块学习检查**：请口头复述安装PyCharm Community时，必须勾选的两个选项分别是什么，以及它们各自的作用。

## 块 11

## 5.1 为什么用 Conda

你之前已经了解到，虚拟环境能为每个项目创建独立的 Python 空间。现在要问的是：**用什么工具来管理这些环境？**

答案是 **Conda**——目前最主流的虚拟环境管理工具。它有两个关键能力：

1. **创建虚拟环境**：为每个项目隔离出独立的 Python 空间，互不干扰。
2. **管理 Python 解释器版本**：不同项目可能需要不同版本的 Python，Conda 可以自由切换。

**为什么深度学习离不开 Conda？** 后续学习 Transformer 架构时，你需要安装 PyTorch、CUDA 等复杂依赖。这些库之间的版本匹配关系非常微妙，手动管理极易出错。Conda 能帮你把整套依赖关系"锁"在独立环境里，装坏了就删掉重来，不影响系统其他部分。

---

## 5.2 安装 Conda：Anaconda vs Miniconda

Conda 有两个发行版可选：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB 至 1GB 以上），预装 numpy、pandas、jupyter 等大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50-100MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本课程推荐 Miniconda**。理由很直接：环境干净，需要什么装什么，避免预装一堆用不到的库造成混乱。

---

## 安装步骤

**第一步：下载**

访问 https://docs.conda.io/en/latest/miniconda.html，按系统选择安装包：
- Windows：`.exe` 文件
- macOS：`.pkg` 或 `.sh` 文件
- Linux：`.sh` 文件

**第二步：运行安装程序**

- **Windows**：一路下一步，但**务必保持默认，不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器会标注为"不推荐"）。为什么？把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python 或其他工具冲突

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：后续学习大模型微调时，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，

## 块 13

## 5.4 激活与退出环境

**激活环境**（进入这个"独立房间"）：

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

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境。选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：在 VS

## 块 15

**本块学习检查**：完成下面两步后，在终端输入 `exit()` 退出。

---

你已成功激活 `llm-learn` 环境。现在，在命令行输入 `python` 并回车，看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车后，若屏幕打印出同样文字，说明环境配置成功。这一步验证了 Python 解释器能正常运行，是后续所有代码执行的基础。

接着，在 VS Code 中新建文件夹 `my-first-project`，创建文件 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角小三角或按 `Ctrl + F5`，下方终端应输出 `Hello from VS Code!`。

## 块 16

### 7.3 在 PyCharm 中验证

作为数学背景的学习者，你可能习惯在公式推导后立即验证结果。这里同理：写完代码，运行它，看到输出，才算真正“跑通”。

**操作步骤**（三步走）：

1. **新建项目**：打开 PyCharm，创建一个新项目（Project），然后在项目中新建一个 Python 文件，命名为 `hello.py`。
2. **运行文件**：在 `hello.py` 的编辑区**右键点击**，从弹出菜单中选择 `Run 'hello'`。PyCharm 会启动 Python 解释器执行这个文件。
3. **查看输出**：屏幕**下方会自动打开 Run 窗口**，你的代码执行结果（比如打印的文本）会显示在这里。

**验证要点**：只要 Run 窗口里出现了你预期输出的内容，就说明你的开发环境（Python + PyCharm）已经配置成功，可以开始正式写代码了。

---

### 第八章：常见问题排查（速查表）

如果你在验证过程中遇到问题，先对照下表定位，再按方案解决。**核心原则：先看报错现象，再查原因，最后对症下药。**

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源：`-i https://pypi.tuna.tsinghua.edu.cn/simple`）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户点击开始菜单，在 Anaconda/Miniconda 文件夹下打开 `Anaconda Prompt (Miniconda3)` 后再执行命令；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda

## 测评题目

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0205-004
请写一个生成器函数 `fibonacci(n)`，生成前 `n` 个斐波那契数。

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？
