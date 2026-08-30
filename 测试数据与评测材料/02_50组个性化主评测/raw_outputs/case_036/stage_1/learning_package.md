# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你最终要跑通 QLoRA 微调，而 QLoRA 是基于 Python 代码运行的。所以第一步不是急着写模型，而是先确保你的电脑能“读懂”你写的 Python 代码。

**Python 解释器**就是干这个的：你写的代码是人类语言，电脑只认 0 和 1，解释器负责把代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本，什么也跑不起来。

结合你的环境（RTX 4060 8GB），后面你会用这块显卡跑微调，但显卡再强，也得先有解释器把你的训练脚本“翻译”给系统，才能调动 GPU 干活。

现在你只需要记住：**装 Python 开发环境，第一步就是装解释器**。它是你所有后续操作（包括 QLoRA）的地基。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

**IDE：你的"工作台"**

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

**虚拟环境：项目的"独立房间"**

想象你同时做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。都装到同一个 Python 里会冲突报错

## 块 3

**本块学习检查**：安装 Python 是搭建 QLoRA 微调工作台的第一步。QLoRA 依赖 PyTorch 等工具，而它们都通过 `pip` 安装，所以必须先让 Python 和 `pip` 在 Windows 上跑通。遇到不懂的术语（如 PATH、环境变量），先记下来再提问。

---

### 2.1 Windows 用户

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，在命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

## 块 4

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

macOS 用户请注意：现代 macOS（Catalina 10.15 起）不再预装 Python 2，系统自带 Python 3（版本可能较旧，如 3.8），但为了确保兼容性和最新特性，建议自行安装最新版 Python 3。

**方法一：官方安装包（新手推荐）**

1. 访问 python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install"。
3. 打开终端验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认不提供 `python` 命令，系统与官方安装包统一用 `python3`。始终用 `python3` / `pip3`，避免“命令不存在”或误用系统版本。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：现在，请你用自己的话回答——为什么在 macOS 上必须用 `python3`

## 块 5

**本块学习检查**：完成安装后，请用自己的话说明“Python环境”与“QLoRA微调”的关系；遇到无法解释的术语（如“虚拟环境”）时记录下来再提问。

Ubuntu/Debian 通常自带 Python，但版本可能偏老。执行以下命令安装最新版及配套工具：

```bash
sudo apt update          # 更新软件源
sudo apt install python3 python3-pip python3-venv -y
python3 --version        # 验证安装
```

**要点**：`-y` 自动确认安装，免去手动输入；`python3-venv` 用于创建虚拟环境，提前装好避免后续补装。你的 RTX 4060 上跑

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是跑通 QLoRA 微调，而 QLoRA 依赖 NVIDIA 驱动、CUDA 与显卡直通，所以环境选择要围绕“GPU 能不能真正用上”来决策。

**日常写 Python、练命令行（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它提供接近真实生产的 Linux 环境，命令、路径、包管理与后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**：使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护、生态落后；VirtualBox 默认无法直通 NVIDIA GPU，跑不了真正的 GPU 训练，只适合纯 CPU 隔离小实验。

**结合你的 RTX 4060 8GB**：本地实体 Ubuntu LTS 是理想

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

`pip` 是 Python 的包管理工具，用来安装第三方库（如 `numpy`）。默认从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，应保留为默认。

**什么时候才需要镜像源？** 仅当网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像是“网络不好时的临时加速手段”，不是永久默认。

**首选做法：临时指定镜像源**（推荐）。只对某一次安装临时指定，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐？** 不影响平时默认行为，官方源仍是首选；网络恢复或镜像出问题时

## 块 8

### 3.3 进阶：把镜像设为默认（长期网络受限才用）

如果你只是偶尔下载慢，用 `-i` 参数临时指定镜像就够了。但如果你**长期网络受限**，几乎每次都要用镜像，那就可以把它写进配置，让 pip 默认走镜像源。注意：官方 PyPI 始终是最可靠的，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从哪个地址下载包。

> **⚠️ 别乱加 `trusted-host`**：网上很多教程会让你顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且确认来源可信时，才临时加。

**macOS / Linux 用户：**

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

4. `Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置是否生效

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：完成本块后，

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费，内置代码分析、版本控制等功能，适合大型项目。你已有 PyTorch 和 Transformer 基础，用它管理 QLoRA 微调项目会更顺手。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（免费）
3. 运行安装，勾选：
   - `Add "Open Folder as Project"`（右键快速打开项目）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开 PyCharm，选 `New Project`
2. `Location` 选项目路径（**路径不要含中文**）
3. `Interpreter` 选 `Previously configured interpreter`，再选系统 Python 3.10+

**本块学习检查**：完成本块后，用自己的话说明 PyCharm

## 块 11

## 5.1 为什么用 Conda

前面提到，虚拟环境可以为每个项目创建独立的 Python 空间。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于后续学习深度学习（需要 PyTorch、CUDA 等复杂依赖），conda 几乎是必备工具。

> **和你当前目标的关系**：你要做 QLoRA 微调，这需要 PyTorch + CUDA 环境。Conda 能帮你为这个项目单独建一个干净的环境，避免和系统 Python 或其他项目互相干扰——这正是你接下来要用的工作台。

## 5.2 安装 Conda：Anaconda vs Miniconda

有两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB 空间），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包。你之后要装 PyTorch 等深度学习库，用 Miniconda 按需安装更合适。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。
   - **macOS/Linux**：运行 `.

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你现在要学QLoRA微调，但不同项目需要不同版本的PyTorch和CUDA。学会创建虚拟环境，就是为后续微调项目准备一个干净的“工作台”。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名（`n`是`name`的缩写）
- `python=3.10`：指定安装Python 3.10

执行后终端会询问：

```

## 块 13

### 5.4 激活与退出环境

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

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS

## 块 15

## 7.1 在命令行验证

你的目标是QLoRA微调，而微调的第一步，是确保你的“工作台”能正常运转。这一步就像开机自检——确认Python环境可用，后续才能放心安装和运行微调工具。

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符。
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 并回车。
4. 正确打印后，输入 `exit()` 退出。

## 7.2 在 VS Code 中验证

命令行验证通过后，再用编辑器验证，确保日常写代码的环境也没问题：

1. 新建文件夹 `my-first-project`，

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 16

## 7.3 在 PyCharm 中验证——你的第一行代码

既然你的目标是 QLoRA 微调，那 PyCharm 就是你接下来写微调脚本的主战场。这一步验证的意义在于：**确认你的“代码编辑器”和“Python 解释器”已经打通**——就像确认新买的键盘能敲出字来，再开始练打字。

操作很简单，三步：

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目里新建一个 Python 文件，命名为 `hello.py`。
2. **运行文件**：在 `hello.py` 的编辑区域**右键点击**，在弹出的菜单中选择 `Run 'hello'`。
3. **查看结果**：PyCharm 底部会弹出 **Run 窗口**，里面会显示程序运行后的输出结果。看到输出，就说明你的开发环境已经能正常执行 Python 代码了。

> 小提示：这一步验证的是“环境通不通”，代码本身很简单。但如果你连这一步都卡住，后面跑 QLoRA 脚本时会非常痛苦——所以请务必亲手跑通。

---

## 第八章：常见问题排查（速查表）

环境搭建最容易出问题，这里把最常见的坑和解决办法列成表，遇到问题直接对照查：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选择 “Modify”，勾选 “Add Python to environment variables” |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示“未选择 Python 解释器” | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python

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

### Q-KB0205-003
以下代码输出什么？
```python
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0202-002
字典 `d = {"a": 1, "b": 2}`，获取键 `"a"` 对应值的方法是________；安全获取不存在的键 `"c"`（返回默认值0）的方法是________。

### Q-KB0205-009
写一个装饰器 `timer`，测量并打印被装饰函数的执行时间（秒），同时不改变原函数的返回值。然后用它装饰一个计算前100万个数平方和的函数。

### Q-KB0201-002
Python中，`type(3.14)` 的返回结果是________；`type("3.14")` 的返回结果是________。
