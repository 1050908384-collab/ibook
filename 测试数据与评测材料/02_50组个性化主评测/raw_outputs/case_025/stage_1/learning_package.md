# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已经有PyTorch和Transformer基础，目标是用云上的T4 16GB做QLoRA微调。在动手前，先花2分钟搞懂一个底层角色。

**Python解释器：程序的“翻译官”**

你写的Python代码是人类能看懂的文字，但电脑只认0和1。**Python解释器**的作用就是把你的代码翻译成电脑能执行的指令。没有它，你写的`.py`文件只是一堆文本。

**这和QLoRA有什么关系？**

你即将运行的微调脚本（比如用HuggingFace加载模型、调用QLoRA库）本质上是`.py`文件。T4显卡

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的目标是QLoRA微调，这需要写Python代码。记事本虽能写，但效率极低。**IDE**提供语法高亮、自动补全、运行调试，让你写代码像用Word一样舒服。推荐**VS Code**（轻量免费）和**PyCharm**（专业强大），选一个即可。

微调时，不同项目可能依赖不同版本的库（如numpy 1.0 vs 2.0），全装一起会冲突。**虚拟环境**为每个项目创建独立Python空间，互不影响。用**conda**管理虚拟环境，这是最主流方案。

安装Python 3.10

## 块 3

**本块学习检查**：完成安装后，请用自己的话说明：Python 和 pip 的安装成功，与你“QLoRA微调”的目标有什么关系？如果遇到无法解释的术语（如“环境变量”“PATH”），记录下来，稍后提问。

---

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

> 你的目标是用 T4 云GPU做QLoRA微调，微调工具（如HuggingFace生态）依赖Python环境，所以这一步是起点。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把Python加入系统“环境变量”。不勾选的话，后续在命令行输入 `python` 会提示“不是内部或外部命令”，所有命令行操作都无法进行——包括你之后安装微调依赖库的命令。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径通常不含中文和特殊符号，但若用户名包含中文或空格，建议自定义安装路径以确保路径安全。如果你想改路径，务必确保路径中**没有

## 块 4

## 2.2 macOS 用户：为 QLoRA 微调铺好 Python 地基

现代 macOS（Catalina 10.15 起）不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用。**强烈建议自行安装最新版 Python**，避免依赖系统版本——这能让你后续安装 PyTorch、transformers 等 QLoRA 依赖时少踩坑。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 命令行默认**不提供** `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，用自己的话说明——

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的目标是QLoRA微调，这需要Python环境。Ubuntu/Debian通常预装Python但版本可能较老，所以用以下命令更新并安装：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

`-y`自动确认安装，`python3-venv`用于创建虚拟环境（微调项目隔离依赖的关键）。安装后运行`python3 --version`验证版本。

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 **QLoRA 微调**，且实训环境是 **Tesla T4 16GB 云服务器**。这一节直接关系到你后续能否顺利跑通训练，选错环境会浪费大量时间。

**日常写 Python、练命令行（Windows 用户）**：推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在 Windows 里提供一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续的服务器一致，比在 Windows 原生 CMD 里练命令更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install` 即可。

**日常写 Python（macOS / Linux 用户）**：直接使用本机终端即可，本身就是类 Unix 环境。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**：应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态的支持最成熟、文档最全。你的 T4 云服务器正是这一路线。

> **⚠️ 不推荐的做法**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护、生态逐渐落后；而 VirtualBox 这类

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（如 `numpy`、`pandas`）。默认从官方 PyPI（`pypi.org`）下载——这是最权威、最完整的源，应始终作为首选。

**只有**当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。镜像只是“网络不好时的加速手段”，不是永久替换项。

### 3.2 首选做法：临时指定镜像源（推荐）

只对某次安装临时用镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方源仍是默认；网络恢复或镜像出问题时，无需清理配置。

---

**本块学习检查**：完成本块后，用自己的话说明：镜像源与你的

## 块 8

## 3.3 进阶：把镜像设为默认（长期网络受限才做）

你现在的目标是 QLoRA 微调，后面会频繁用 pip 安装 PyTorch、transformers、bitsandbytes 等大体积包。如果每次下载都卡在慢速网络上，确实很耽误事。但请记住：**只有当你长期处于网络受限环境、几乎每次都要用镜像时，才把它写入配置作为默认源**。官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除该配置。

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车。这会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 的作用就是告诉 pip：默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

### macOS / Linux 操作步骤

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

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

**4.1 方案一：VS Code（轻量、免费、推荐）**

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。后续写 Python 脚本、调配置文件时都用得上。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 打开 VS Code

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版免费，内置代码分析、版本控制、数据库工具，适合大型项目。你的目标是 QLoRA 微调，这属于大型深度学习项目，用它能更高效地管理代码。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **Community Edition**（免费）
3. 运行安装程序，勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开 PyCharm，选 `New Project`
2. `Location` 选项目路径（**路径中不要含中文**）
3. `Interpreter` 选 `Previously configured interpreter`，再选系统 Python 3.10+

**

## 块 11

### 5.1 为什么用 Conda

你已经掌握了 PyTorch 基础和 Transformer 架构，接下来要挑战 QLoRA 微调。这个任务有个现实问题：微调需要特定版本的 PyTorch、CUDA 等复杂依赖，如果直接装进系统 Python，很容易和已有环境冲突，搞乱你的工作台。

Conda 就是来解决这个问题的。它是最主流的虚拟环境管理工具，能为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备工具——你的云 T4 16GB 环境，后续装 PyTorch 和 CUDA 依赖时，conda 能帮你隔离得干干净净。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB 空间），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 50-80MB，安装后占用空间约 400MB 以上），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你的目标是 QLoRA 微调，环境越干净越好，需要什么装什么，避免无关包干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你已掌握PyTorch基础和Transformer架构，下一步要QLoRA微调。微调不同模型时，常需不同版本的PyTorch和CUDA——用独立环境隔离项目，避免版本冲突。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

拆解：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名（`n`是`name`缩写）
- `python=3.10`：指定安装Python 3.10

执行后终端列出待装包

## 块 13

## 5.4 激活与退出环境：为 QLoRA 微调隔离你的工作台

你的目标是 QLoRA 微调，这意味着你会在 `llm-learn` 环境里安装 PyTorch、Transformers 等依赖。激活环境，就是进入这个专属“独立房间”，避免与系统或其他项目冲突。

**激活环境**：

```bash
conda activate llm-learn
```

激活后，命令行前会出现 `(llm-learn)` 标志。此后所有 `python` 和 `pip` 命令都只作用于这个环境，不会影响系统全局。

**验证环境**：

```bash
python --version
```

应显示 `Python 3.10.x`——这是你后续微调脚本的运行基础。

**退出环境**（回到系统默认）：

```bash
conda deactivate
```

### 5.5 查看与管理环境

**查看所有环境**：

```bash
conda env list
```

带 `*`

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

**本块学习检查**：完成本块后，用自己的话说明VS Code/PyCharm配置Conda环境与“QLoRA微调”目标的关系；遇到无法解释的术语（如“解释器”）时记录下来再提问。

---

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 自动识别 conda 环境后，你写的代码就会使用 `llm-learn` 中的 Python 和已安装的包——这是后续跑 QLoRA 脚本的前提。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   -

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

先激活环境：`conda activate llm-learn`，然后输入 `python` 进入交互模式。看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车，若正确打印文字，按 `exit()` 退出。这是命令行验证。

接着用 VS Code 验证：新建文件夹 `my-first-project` 并用 VS Code 打开，新建 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角小三角或按 `Ctrl + F5`，下方

## 块 16

## 7.3 在 PyCharm 中验证——你的第一行代码

既然你的目标是 QLoRA 微调，环境验证这一步就是给后续所有工作“打地基”。在 PyCharm 里跑通第一行代码，确认工具链能正常工作，后面跑微调脚本时才不会在环境问题上卡壳。

**操作步骤**：

1. 打开 PyCharm，新建一个项目，在项目中创建文件 `hello.py`。
2. 在文件中输入一行最简单的输出语句（例如 `print("Hello")`）。
3. 在文件编辑区**右键点击**，选择 `Run 'hello'`。
4. 观察 PyCharm **下方**的 Run 窗口，确认输出结果正确显示。

跑通这一步，说明你的 Python 解释器、项目配置和 IDE 都正常工作——这是后续运行任何微调脚本的前提。

---

## 常见问题排查（环境验证时可能遇到）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

>

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

### Q-KB0201-004
以下代码输出是？
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```

### Q-KB0204-003
请写代码：读取 `data.txt` 文件，捕获文件不存在的异常，若不存在则打印 "文件未找到"。

### Q-KB0203-001
以下函数定义中，哪个参数是**默认参数**？
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

### Q-KB0201-001
以下哪个是Python中**正确的**变量赋值？

### Q-KB0205-008
写一个类 `Stack`（栈），实现 `push(item)`、`pop()`、`peek()`、`is_empty()` 方法。`pop()` 在栈空时抛出 `IndexError`。
