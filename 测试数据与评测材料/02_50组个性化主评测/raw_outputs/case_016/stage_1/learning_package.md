# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已经熟悉 PyTorch 和 Transformer，但 QLoRA 微调的第一步，是确保代码能被电脑执行。这里只讲一个核心概念：**Python 解释器**。

你写的 `.py` 文件是人类可读的文本，但 GPU（你的 RTX 4060）只认 0 和 1。**解释器就是"翻译官"**，负责把你的代码翻译成电脑能执行的指令。没有它，你写的微调脚本只是一堆文本，无法驱动显卡。

**为什么这和 QLoRA 有关？** 你后续要运行的训练脚本、加载模型的命令，都需要解释器来"翻译"给硬件。装好它，你的工作台才算通电。

**

## 块 2

**IDE 与虚拟环境：为 QLoRA 微调搭好"工作台"**

你已熟悉 PyTorch 和 Transformer，目标是在 8GB 显存上跑 QLoRA 微调。这一步不是学语法，而是确保你的开发环境能支撑后续的模型加载与训练调试。

**1.2 代码编辑器（IDE）**：记事本也能写代码，但调试 QLoRA 脚本时，你需要语法高亮和断点调试来快速定位问题。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个顺手即可。

**1.3 虚拟环境（conda）**：这是关键。QLoRA 微调依赖特定版本的 `transformers`、`peft`、`bitsandbytes` 等库，不同项目间版本冲突会直接导致报错。用 **conda** 为每个项目创建

## 块 3

### Windows 用户：安装 Python 3.10+

你的目标是 QLoRA 微调，而 QLoRA 依赖 PyTorch，PyTorch 又依赖 Python。所以，先把 Python 环境装对，是后面所有工作的地基。下面三步，请跟着做。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
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
python --version
```

**本块学习

## 块 4

**macOS 用户：安装 Python 3.10+**

你的目标是 QLoRA 微调，这依赖 PyTorch 等工具链，而它们都需要一个干净的 Python 环境。macOS 从 Big Sur 11 起不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用，版本老旧。**强烈建议自行安装最新版 Python**，避免依赖系统版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 上，命令行默认**不提供** `python` 命令（早期版本曾指向 Python 2，现已移除），系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

若已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习

## 块 5

**本块学习检查**：完成安装后，请用自己的话说明：为什么 `python3-venv` 对后续 QLoRA 微调环境隔离很重要？遇到无法解释的术语（如“虚拟环境”）时记录下来再提问。

---

Ubuntu/Debian 通常自带 Python，但版本可能偏旧。执行以下命令安装 Python 3 及配套工具：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**要点**：`-y` 自动确认安装；`python3-venv` 用于创建虚拟环境，避免后续为 QLoRA 微调安装依赖时污染系统 Python。装完用 `python3

## 块 6

### 你的环境选择，直接决定 QLoRA 能否跑起来

你的目标是 QLoRA 微调，且手头是 RTX 4060 8GB。这决定了环境选择的核心逻辑：**GPU 实训必须用 Linux，且要能直通显卡**。

#### 日常练习（写 Python、练命令）
- **Windows 用户**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。它提供接近真实生产的 Linux 环境，命令、路径、包管理与服务器一致。管理员 PowerShell 执行 `wsl --install` 即可。
- **macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境。

#### GPU 实训（QLoRA 微调）
必须使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。QLoRA 依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

> **⚠️ 别踩坑**：不要用 **"CentOS + VirtualBox 虚拟机"**。CentOS 8 已停止维护（2021 年 12 月 31 日 EOL），建议使用 CentOS Stream 或 Ubuntu LTS；VirtualBox

## 块 7

你在 8GB 显存上做 QLoRA 微调，安装 `transformers`、`bitsandbytes` 这类库时，`pip` 默认从官方 PyPI（`pypi.org`）下载。这是最权威、最完整的源，正常情况下应保留为默认。

**只有**当你发现下载明显变慢、经常超时，或处于网络受限环境时，才需要临时换用国内镜像。镜像只是“网络不好时的加速手段”，不是永久设置——官方源始终是首选。

**首选做法：临时指定镜像源**（推荐）

只对某一次安装临时指定镜像，不改动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样不会影响你平时的默认行为；哪天网络恢复或镜像出问题，也无需清理配置。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 QLoRA 微调准备环境，而下载预训练模型和依赖包是绕不开的一步。如果你长期处于网络受限环境，几乎每次都要用镜像，才值得把它设为默认。**官方 PyPI 仍是最可靠的来源，网络恢复后建议改回。**

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**。只有企业内网自建源或确实遇到证书报错且来源可信时，才临时加。

**macOS / Linux 用户：**

1. 终端执行：`mkdir -p ~/.config/pip`
2. 创建配置文件：`nano ~/.config/pip/pip.conf`
3. 写入同样内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `pip.conf`，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**

完成本块后，用自己的话说明：本块内容与你当前 QLoRA 微调目标的关系是什么？例如，设置镜像默认源如何影响你后续安装依赖包的效率与安全性？遇到无法解释的术语（如 `index-url`、`trusted-host`）时

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

VS Code是微软出品的免费编辑器，启动快、插件丰富，不仅适合Python，也适合后续写脚本、改配置文件。对你要做的QLoRA微调来说，它是轻量可靠的“工作台”——写训练脚本、看日志、改超参数都在这完成。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python语言服务器，让代码提示更智能）

装完这两个，VS Code才能识别`.py`文件，给你代码补全、错误提示和一键运行——这对调试微调脚本很关键。

**配置Python解释器**：
1. 打开VS Code，按`

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm Community 是 JetBrains 出品的免费 Python IDE，内置代码分析、版本控制等功能，适合大型项目。安装时访问官网下载社区版，建议勾选“Add to PATH”和“Add Open Folder as Project”。首次启动选 `New Project`，项目路径避免中文，解释器选系统 Python 3.10+。

**与你的目标关联**：QLoRA微调涉及大量脚本调试与文件管理，PyCharm 的代码分析和项目视图能帮你高效组织微调代码。你的 RTX 4060 8GB 足够运行微调，IDE 本身不占显存，专注代码编写即可。若已有 VS Code 习惯，此方案可选；但 PyCharm 的“开箱即用”减少配置时间，适合你每日2小时的紧凑节奏。

## 块 11

### 5.1 为什么用 Conda

你已经熟悉 Python 和 PyTorch，但做 QLoRA 微调时，项目依赖会非常“挑剔”：不同项目可能需要不同版本的 PyTorch、CUDA 甚至 Python 解释器。Conda 的核心价值就在于此——它能为每个项目创建**独立的 Python 空间**，并且能直接管理 Python 解释器版本。这意味着你可以为微调实验单独建一个环境，装好特定版本的 PyTorch，而不会影响系统里其他项目的运行环境。对深度学习这种依赖复杂的场景，Conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

安装前先选发行版，两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装 numpy、pandas、jupyter 等大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**。你的目标是 QLoRA 微调，环境越干净越好——需要什么包装什么包，避免无关库干扰依赖解析。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，并在最后一步选择 `yes` 让安装器执行 **`conda init`**（完成

## 块 12

你已熟悉 PyTorch 和 Transformer，但 QLoRA 微调需要独立的 PyTorch/CUDA 版本组合。本块教你用 Conda 隔离项目环境，避免版本冲突。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境
- `-n llm-learn`：环境名为 `llm-learn`（`n` 即 `name`）
- `python=3.10`：指定 Python 3.10

终端会列出待装包并询问 `Proceed ([y]/n)?`，输入 `y` 回车等待完成。

**本块学习检查**：完成本块后，用自己的话说明本块内容

## 块 13

### 5.4 激活与退出环境

你的目标是在8GB显存上跑QLoRA微调，而每个项目依赖（如PyTorch、transformers）版本都可能冲突。Conda环境就是为每个项目准备的“独立房间”，避免互相干扰。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`——这正是你后续安装PyTorch等依赖的版本基准。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**查看所有已创建的环境**：

```bash
conda env list
```

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习

## 块 14

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

> **针对你的实训条件（RTX 4060 8GB）**：后续做 QLoRA 微调时，你需要在 `llm-learn` 环境中安装 PyTorch（CUDA 版）等依赖。选对解释器，才能确保代码调用的是这个环境里的 GPU 版工具链。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你已经熟悉 Python 和 PyTorch，但 QLoRA 微调前，得先确认环境能跑通。本块只做一件事：**验证 Python 能正常运行**。

**命令行验证**（最直接）：

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 回车，看到 `>>>` 提示符
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 回车
4. 正确打印后，输入 `exit()` 退出

**VS Code 验证**（日常开发更常用）：

1. 新建

## 块 16

## 7.3 在 PyCharm 中验证——你的第一行代码

既然你的目标是 QLoRA 微调，那我们先确认工作台能跑通最基本的 Python 执行链路。这一步不涉及任何深度学习内容，但它验证的是：**PyCharm 能否正确调用你配置好的 Python 解释器并执行代码**。如果这一步失败，后续所有微调脚本都无法运行。

**操作步骤：**

1. 打开 PyCharm，新建一个项目（项目路径**不要包含中文或空格**，这是本章强调的好习惯）。
2. 在项目中新建文件 `hello.py`。
3. 在文件中输入一行代码：`print("Hello, QLoRA!")`。
4. 右键点击文件 → 选择 `Run 'hello'`。
5. 观察下方 **Run 窗口**，应能看到输出结果。

**如果 Run 窗口报错**，先对照第八章排查表自查：

- 提示“未选择 Python 解释器”→ 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择你创建的 `llm-learn` 环境。
- 提示 `python` 不是内部或外部命令 → 安装时未勾选 "Add to PATH"，需重新运行安装程序修复。
- 激活 conda 环境后 `python` 版本不对 → 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境。

**本章小结（环境验证清单）：**

| 组件 | 验证命令 |
|------|----------|
| Python 3.10+ | `python --version` |
| Conda 虚拟环境 `llm-learn` | `conda env list` |
| PyCharm | 打开软件确认 |

**养成的好习惯**（与 QLoRA 微调直接相关）：

1. **一个项目一个 conda 环境**——微调时你会安装大量依赖，隔离环境避免冲突。
2. 安装第三方库前先 `conda activate` 进入对应环境。
3. 优先用 `conda install`，找不到的再用 `pip`

## 测评题目

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0203-006
关于Python的MRO（方法解析顺序），以下说法**错误**的是？

### Q-KB0203-001
以下函数定义中，哪个参数是**默认参数**？
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

### Q-KB0205-004
请写一个生成器函数 `fibonacci(n)`，生成前 `n` 个斐波那契数。

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

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

### Q-KB0205-008
写一个类 `Stack`（栈），实现 `push(item)`、`pop()`、`peek()`、`is_empty()` 方法。`pop()` 在栈空时抛出 `IndexError`。
