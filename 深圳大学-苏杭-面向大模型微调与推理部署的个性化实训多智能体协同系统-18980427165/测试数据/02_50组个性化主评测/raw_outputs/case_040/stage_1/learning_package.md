# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

你写下的 Python 代码是给人看的文字，但电脑只认 0 和 1。**Python 解释器**就是那个"翻译官"——它把你写的代码逐行翻译成电脑能执行的指令。没有它，你的 `.py` 文件就只是一堆普通文本，电脑根本不知道你想让它做什么。

**为什么这对你的 LoRA 微调目标很重要？**

你后续要运行的 PyTorch 代码、加载 Transformer 模型、执行 LoRA 微调脚本，全都依赖这个"翻译官"先把 Python 代码转成 GPU 能理解的指令。装好它，你的 RTX 4060 才能开始干活。

**动手验证一下（约 2 分钟）**

打开终端（Windows 搜索"cmd"或"PowerShell"），输入以下命令后按回车：

```
python --version
```

如果看到类似 `Python 3.x.x` 的输出，说明解释器已经装好并正常工作。如果提示"不是内部或外部命令"，说明还没装好，回到上一节检查安装步骤。

---

**本块学习检查**

完成本块后，请

## 块 2

好的，我们开始搭建你的“工作台”。既然你的目标是LoRA微调，那代码编辑器就是你的主战场。记事本写代码效率太低，我们直接用**IDE**，它自带语法高亮、自动补全和调试功能，能让你把精力集中在模型和代码逻辑上。

推荐两款：**VS Code**（轻量免费，插件丰富）和 **PyCharm**（专业强大）。对于PyTorch和Transformer这类项目，两者皆可，你可以根据喜好选择。

接下来是**虚拟环境**，这步非常关键。想象你同时做两个项目，一个需要`numpy 1.0`，另一个需要`numpy 2.0`，装在一起就会冲突。虚拟环境就是为每个项目创建独立的Python空间，互不干扰。我们使用**conda**来管理，这是最主流的方案。

最后，安装 **Python 3.10 或更高版本**。Python 3.10 引入了更清晰的错误提示（如

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.1 Windows 用户：安装 Python 3.10+

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

3

## 块 4

### macOS 用户：安装 Python 3.10+

现代 macOS（Catalina 10.15 起）不再预装 Python 2，部分系统可能带有供系统脚本使用的旧版 `python3`。**强烈建议自行安装最新版 Python**，避免依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 命令行默认**不提供** `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**

完成本块后，用自己的话说明：本块内容（正确安装 Python 环境）与你的学习目标（LoRA 微调）有什么关系？遇到无法解释的术语，

## 块 5

**Linux 用户（Ubuntu/Debian）安装 Python 3.10+**

你的目标是用 PyTorch 做 LoRA 微调，而 RTX 4060 需要较新的 Python 环境。Linux 虽预装 Python，但版本可能偏老，所以手动安装。

**操作步骤**（在终端依次执行）：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**关键点**：`-y` 自动确认安装，免手动输入 `y`；`python3-venv` 是创建虚拟环境的模块，一并

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，这意味着最终要跑 GPU 训练。环境选对了，后面能省大量折腾时间。

**日常练习（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。管理员 PowerShell 执行 `wsl --install` 即可。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和服务器一致，比 Windows 原生 CMD 更贴近实战。

**日常练习（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**GPU 实训（LoRA/QLoRA 微调）**：使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"** 当最佳方案。CentOS 8 已停止维护、生态落后；VirtualBox 默认**无法直通 NVIDIA GPU**，跑不了真正的 GPU 训练，只适合纯 CPU 隔离小实验。

**路线总结**：命令练习用 WSL2/Ubuntu LTS，GPU 实训用云上或实体 Ubuntu LTS。

---

**本块学习检查**：完成本

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认它从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、应保留的默认源**。

**什么时候才需要镜像源？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

**首选做法：临时指定镜像源（推荐）**

如果某次安装慢，最干净的做法是

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正在为 LoRA 微调准备环境，但先停一下——**这个操作只适合长期网络受限的情况**。如果你只是偶尔下载慢，用 `-i` 参数临时指定镜像就够了，别改默认配置。配置后记住：官方 PyPI 最可靠，网络恢复后建议改回。

**Windows 操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开配置文件夹
2. 新建文件夹，命名为 `pip`
3. 在 `pip` 文件夹内新建文本文件，命名为 `pip.ini`
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：以后默认从这里下载包。

**macOS / Linux 操作步骤：**

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

写入同样的内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

**⚠️ 关于 `trusted-host` 的警告：** 网上很多教程会顺手加一行 `trusted-host = ...`，**不要加**。除非镜像确实无法通过 HTTPS 证书验证——加了它等于让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用不需要。只有企业内网自建源、或确认来源可信且遇到证书报错时，才临时加。

### 验证配置

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 就说明成功了。

### 改回官方源

删除 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查：** 用自己的话说说，为什么这个"设为默认镜像"的操作只在长期网络受限时才值得做？它和你当前 LoRA 微调的学习目标有什么关系？（提示：想想你后面要安装哪些深度学习依赖包，以及网络环境对安装效率的影响。）遇到无法解释的术语，记下来再提问。

## 块 9

**VS Code：你的轻量开发工作台**

你的目标是 LoRA 微调，这需要写 Python 代码。VS Code 是微软的免费编辑器，启动快、插件丰富，适合 Python 开发，后续写脚本也能用。

**安装步骤**（约 2 分钟）：

1. 访问 `https://code.visualstudio.com`，下载对应系统安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**（关键一步）：

打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

- **Python**（Microsoft 官方）：提供语法高亮、调试、智能提示
- **Pylance**：让代码提示更智能

装完插件，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行。安装完成后，点击右下角提示的 Reload 或重启 VS Code，插件才会生效。

**配置 Python 解释器**：

1. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择你刚安装的 Python

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm Community是免费开源的Python专用IDE，内置代码分析、版本控制等，适合大型项目。安装时从官网下载社区版，勾选“Open Folder as Project”和“Add to PATH”。首次启动新建项目，路径避免中文，解释器选系统Python 3.10+。

**实践任务**：按步骤安装PyCharm并新建项目，确认解释器为Python 3.10+。这为后续LoRA微调代码编写提供稳定环境——IDE的代码分析能帮你快速发现语法错误，版本控制便于管理微调脚本迭代。

## 块 11

## 5.1 为什么用 Conda

你之前已经掌握了 Python 基础环境的使用，现在进入深度学习领域，会面临一个关键问题：**不同项目需要不同的依赖版本，装在一起会互相冲突**。Conda 就是解决这个问题的核心工具。

Conda 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于你接下来的 LoRA 微调任务——需要 PyTorch、CUDA 等复杂依赖——conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，因为环境更干净，后续需要什么包装什么包——这正好符合你"按需搭建"的实践习惯。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，并在最后一步选择 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是LoRA微调，而微调前需要隔离项目环境。现在动手创建第一个虚拟环境，为后续安装PyTorch做准备。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**拆解**：`conda create`是创建命令；`-n llm-learn`给环境命名（`n`即`name`）；`python=3.10`指定Python版本。

终端会列出待装包并询问`Proceed ([y]/n)?`，输入`y`回车等待完成。

**练习**：执行上述

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

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，这意味着马上要跑 PyTorch 训练脚本。先确保 VS Code 用的是 `llm-learn` 环境，否则代码会报“找不到 torch”。

1. 终端执行：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板
3. 输入并选择 `Python: Select Interpreter`
4. 在列表中选带 `conda` 字样的项，形如 `Python 3.10.x ('llm-learn': conda)`

VS Code 会自动识别 conda 环境。选好后，当前工作区所有代码都运行在 `llm-learn` 中，你安装的 PyTorch 等包才能被 import。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，进入 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`）
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`
3. 点齿轮图标 → `Add...`
4. 左侧选 `Conda Environment` → `Existing environment`
5. 在 `Interpreter` 路径中选择 conda 环境的 Python：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

**本

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

先激活环境：`conda activate llm-learn`，再输入 `python` 进入交互模式。看到 `>>>` 后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车，若正确打印文字，用 `exit()` 退出。

接着在 VS Code 中验证：新建文件夹 `my-first-project` 并用 VS Code 打开，新建 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角小三角或按 `Ctrl + F5`，下方终端应输出 `Hello from VS

## 块 16

## 7.3 在 PyCharm 中验证

你的目标是用 LoRA 微调大模型，而这一切的起点，是确保你的开发环境能跑通最基础的代码。PyCharm 是你接下来写训练脚本的主战场，先在这里完成一次"冒烟测试"。

1. **新建项目**：打开 PyCharm，点击 `File → New Project`，创建一个新项目。项目路径**不要包含中文或空格**，例如 `D:\lora-finetune`。
2. **创建 `hello.py`**：在项目目录上右键 → `New → Python File`，命名为 `hello.py`。输入一行代码：`print("Hello, LoRA!")`。
3. **运行脚本**：在 `hello.py` 编辑区右键 → `Run 'hello'`。此时下方会弹出 **Run 窗口**，你应该能看到输出结果 `Hello, LoRA!`。

如果 Run 窗口正常输出，说明你的 PyCharm 已正确关联 Python 解释器，可以开始后续的 LoRA 微调实战了。

---

## 第八章：常见问题排查

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import

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

### Q-KB0204-005
请写一个自定义异常类 `ModelError`，继承自 `Exception`，并在模型加载失败时抛出该异常，附带错误信息。

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

### Q-KB0205-004
请写一个生成器函数 `fibonacci(n)`，生成前 `n` 个斐波那契数。

### Q-KB0205-005
请写一个**带参数的装饰器** `retry(times=3)`，使被装饰的函数在抛出异常时自动重试指定次数。

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

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
