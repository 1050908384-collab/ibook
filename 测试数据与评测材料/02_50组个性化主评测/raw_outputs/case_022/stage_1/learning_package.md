# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已经知道LoRA微调最终要跑PyTorch代码，但先别急着装库。这一节只解决一个问题：**你写的`.py`文件，电脑凭什么能执行？**

答案就是**Python解释器**——程序的"翻译官"。

你写的代码是人类能看懂的文字，但电脑只认0和1。解释器的作用，就是把你的代码翻译成电脑能执行的指令。没有它，你写的`.py`文件只是一堆文本。

对你而言，这个概念的实操意义在于：**你后续所有LoRA训练脚本，最终都是通过解释器来驱动GPU工作的**。你调用的`transformers`、`peft`这些库，本质也是被解释器翻译后，才变成显卡能听懂的指令。

所以，装Python解释器不是"走个形式"，它是你整个LoRA微调工作流的**第一道地基**。地基没打牢，后面跑`train.py`时会遇到各种莫名其妙的报错，而你连排查方向都没有。

现在你只需要记住：**解释器 = 翻译官**，它负责把你的代码变成电脑能执行的动作。至于怎么装、装哪个版本，下一节再讲。

## 块 2

好的，我们直接进入正题。既然你的目标是 LoRA 微调，那这个“工作台”就是你的主战场。

### 1.2 代码编辑器（IDE）：你的“工作台”

记事本写代码效率太低，IDE 提供语法高亮、自动补全和调试功能。针对你的 LoRA 微调任务，推荐使用 **VS Code**（轻量免费），它配合远程开发插件能很好地利用你的 RTX 4060 Ti 16GB 显卡进行本地调试。

---

## 第二章：安装 Python

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，而 PyTorch 是它的基础。本块安装的 Python 和 pip，正是运行 PyTorch 的起点。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这会让你在命令行直接输入 `python` 就能运行。不勾选的话，后续所有命令行操作都会因“不是内部或外部命令”而中断。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。若改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入

## 块 4

你的目标是用 LoRA 微调大模型，而 LoRA 的核心是“只训练一小部分参数”。要动手跑通微调脚本，第一步就是确保本机 Python 环境干净、可用。macOS 系统自带的 Python 版本混乱（可能缺失、可能是旧版），直接用它跑 PyTorch 和 HuggingFace 工具链，很容易踩到“命令不存在”或版本不兼容的坑。所以，本块的核心就是：**装一个独立的、最新的 Python 3**，并学会用 `python3` / `pip3` 命令验证。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：在现代 macOS 上，命令行默认**并不提供** `python` 命令（早期版本曾指向系统自带的 Python 2，现已移除），系统与官方安装包统一使用 `python3`。因此请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的用户）**

1. 在终端执行：`

## 块 5

## 2.3 Linux 用户（Ubuntu/Debian）

你的目标是 LoRA 微调，这意味着很快要跑 PyTorch 训练脚本。Linux 上 Python 环境是否干净，直接决定后面装 `transformers`、`peft` 时会不会踩坑。

Ubuntu/Debian 通常预装 Python，但版本可能偏老。用以下命令安装 Python 3 及配套工具：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**为什么这样写**：`-y` 参数表示自动确认安装，不需要你手动确认每个安装步骤。

**本块学习检查**：完成本块后，用自己的话

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，且手上有 RTX 4060 Ti 16GB，这意味着你**必须**让 GPU 真正跑起来。环境选错，后面 rank/alpha 调参全是空谈。

**日常练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。管理员 PowerShell 执行 `wsl --install` 即可。它给你接近生产服务器的 Linux 环境，命令和路径习惯与后续云端一致，比 Windows CMD 更贴近实战。

**macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**GPU 实训（你的场景）**：用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。LoRA 微调依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别踩坑**：**不要**把 "CentOS + VirtualBox 虚拟机"当最佳方案。CentOS 8 已于 2021 年停止维护，CentOS 7 也于 2024 年停止维护，不建议新项目使用 CentOS 系列；VirtualBox 对 NVIDIA GPU 直通支持有限，配置复杂且

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你在准备LoRA微调时，会频繁用`pip`安装`transformers`、`peft`等库。默认从官方PyPI（`pypi.org`）下载最权威完整，应保留为默认源。只有网络受限、官方源明显变慢或超时时，才临时切国内镜像——**镜像是临时加速手段，不是永久替换**。

**首选做法**：只对某次安装临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样不影响平时默认行为；网络恢复或镜像出问题时，无需清理配置。

**实践提示**：你的RTX 4060 Ti 16GB跑LoRA时，安装`peft

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，而微调过程中会反复用 pip 安装 PyTorch 生态的依赖包。如果每次都手动指定镜像地址，效率太低。本节的“设为默认”正是解决这个痛点的方案——但请记住一个前提：**只有当你长期处于网络受限环境、几乎每次都要用镜像时，才考虑写入配置**。官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除该配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

**macOS / Linux 用户：**

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

写入同样内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置是否生效

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即配置成功。

### 3.5 改回官方源

删除 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 启动快、插件丰富，且完全免费。它不只适合 Python，后续你写训练脚本、调配置文件时都能复用，是 LoRA 微调实验的轻便工作台。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能——这对你调试 LoRA 训练脚本至关重要。

**配置 Python 解释器**：
1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择需要的解释器版本

## 块 10

**本块学习检查**：完成安装后，用自己的话说明“PyCharm Community”与“LoRA微调”的关系；遇到无法解释的术语（如“解释器”）先记录下来再提问。

---

PyCharm Community 是 JetBrains 出品的免费 Python 专用 IDE，内置代码分析、版本控制等功能，适合大型项目开发。安装时访问官网下载社区版，运行安装程序时建议勾选 `Add "Open Folder as Project"`（右键快速打开文件夹）和 `Add to PATH`（命令行直接启动）。

首次启动时选择 `New Project`，在 `Location` 处设置项目路径（**路径中不要含中文**），`Interpreter` 选择 `Previously configured interpreter`，再选系统已装的 Python 3.10+。

**针对你的实训条件**：RTX 4060 Ti 16GB 足够支撑 LoRA 微调实验，但 PyCharm 本身不直接涉及 GPU 配置——它只是你编写、运行微调脚本的“工作台”。后续配置 Conda 虚拟环境时，你会在此 IDE

## 块 11

## 5.1 为什么用 Conda：为 LoRA 微调铺路

你已经掌握 Transformer 架构和 PyTorch 进阶训练，马上要进入 LoRA 微调。但先想一个问题：LoRA 微调需要 PyTorch、CUDA、transformers、peft 等一堆依赖，版本稍有不匹配，训练就会报错。Conda 就是来解决这个问题的。

虚拟环境为每个项目创建独立的 Python 空间，而 **Conda** 是目前最主流的虚拟环境管理工具——它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于深度学习这种依赖复杂（PyTorch、CUDA 等）的场景，conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境更干净，后续需要什么装什么。你的 RTX 4060 Ti 16GB 跑 LoRA 微调，环境越干净越容易排查问题。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 后，

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你正迈向LoRA微调，核心是隔离项目依赖。本块教你用一条命令搭建专属环境：

```bash
conda create -n llm-learn python=3.10
```

拆解来看：`conda create` 是创建指令；`-n llm-learn` 为环境命名（`n` 即 `name`）；`python=3.10` 锁定Python版本。执行后终端会列出待装包并询问 `Proceed ([y]/n)?`，输入 `y` 回车等待完成即可。

**为何与你相关**：后续

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

先激活环境：`conda activate llm-learn`。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。VS Code 会自动识别 conda 环境，选择后，工作区代码将使用 `llm-learn` 中的 Python 和已装包。

## 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点击 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`），左侧选 `Project: 项目名` → `Python Interpreter`，点击齿轮 → `Add...`，选 `Conda Environment` → `Existing environment`，在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：Windows 为 `C:\Users\用户名\miniconda3\envs\llm-learn\python.exe`，macOS/Linux 为 `~/miniconda3/envs/llm-learn/bin/python`。

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你的LoRA微调之旅从验证环境开始。在命令行中，先激活环境：

```bash
conda activate llm-learn
```

输入 `python` 进入交互模式，看到 `>>>` 提示符后，执行：

```python
print("Hello, Python! 我的环境准备好了！")
```

正确打印即环境可用，输入 `exit()` 退出。

接着在VS Code中：新建文件夹 `my-first-project` 并打开，创建 `hello.py`，写入：

```python
print("Hello from VS Code!")
```

点击右上角小三角或按 `Ctrl + F5`，下方终端会

## 块 16

### 7.3 在 PyCharm 中验证

既然你的目标是 LoRA 微调，我们先把“跑通第一行代码”当作一次环境体检——这一步不通过，后面加载模型、调 rank/alpha 都无从谈起。

1. **新建项目**：打开 PyCharm，`File → New Project`，位置选一个**不含中文或空格**的路径（比如 `D:\lora_work`）。
2. **创建 `hello.py`**：在项目根目录右键 → `New → Python File`，命名 `hello`。输入一行：`print("hello")`。
3. **运行**：在 `hello.py` 编辑区右键 → `Run 'hello'`。下方 **Run 窗口**应显示 `hello`。

如果 Run 窗口报错，先对照第八章排查表自查，重点看解释器是否选对（`Ctrl + Shift + P` → `Python: Select Interpreter`，选你 `llm-learn` 环境）。

---

## 第八章：常见问题排查（节选）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限 | 临时加 `-i` 指定国内镜像（如清华源） |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，用 `python -m pip install 包名` 安装 |

---

## 本章小结

完成本章后，你的电脑应具备：

| 组件 | 状态 | 验证命令 |
|------|------|----------|
| Python 3.10+ | ✅ 已安装 | `python --version` |
| pip 国内镜像源 | ⭕ 可选 | `pip config list` |
| PyCharm | ✅ 已安装 | 打开软件确认 |

## 测评题目

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

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0203-004
以下代码输出是？
```python
class A:
    def __init__(self):
        self.x = 1
class B(A):
    def __init__(self):
        super().__init__()
        self.y = 2
b = B()
print(b.x, b.y)
```

### Q-KB0205-007
写一个函数 `count_words(text)`，接收一段英文字符串，返回一个字典，统计每个单词出现的次数（忽略大小写）。

### Q-KB0205-006
以下关于生成器的说法，**错误**的是？

### Q-KB0205-005
请写一个**带参数的装饰器** `retry(times=3)`，使被装饰的函数在抛出异常时自动重试指定次数。
