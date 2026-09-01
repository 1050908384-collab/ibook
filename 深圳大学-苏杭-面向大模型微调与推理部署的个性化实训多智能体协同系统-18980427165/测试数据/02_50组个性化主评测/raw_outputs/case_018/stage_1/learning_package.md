# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 1.1 Python 解释器：程序的"翻译官"

你已经有PyTorch和Transformer基础，但环境准备是微调的地基。先别急着装包，理解这个核心概念，后面每一步你都知道自己在做什么。

你写的Python代码是人类能看懂的文字，但电脑只认0和1。**Python解释器**的作用就是把你的代码翻译成电脑能执行的指令。没有它，你写的`.py`文件只是一堆文本。

对你而言，这个"翻译官"直接决定了你能否在RTX 3090上跑LoRA微调脚本。你写的训练代码（比如调用HuggingFace的`Trainer`）最终都要靠它翻译成GPU能执行的指令。解释器版本不对，你的`transformers`库可能直接报错，更别提微调了。

**实践要点**：装好解释器后，打开终端输入`python --version`，确认版本号正常显示。这一步能验证你的"翻译官"是否就位——它是一切微调实验的起点。

## 块 2

你的目标是 LoRA 微调，这需要一套能跑通 PyTorch 与 HuggingFace 工具链的稳定环境。本块先解决“工作台”和“独立房间”两个基础问题。

**IDE 选择**：记事本写代码效率极低。IDE 提供语法高亮、自动补全、运行调试，让你写代码像用 Word 一样舒服。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大）。两者都能胜任后续微调脚本的编写与调试，选顺手即可。

**虚拟环境**：你的 24GB 显存适合跑 LoRA，但不同项目可能依赖不同版本的库（如 numpy 1.x vs 2.x），同装一个 Python 会冲突。**虚拟环境**为每个项目创建独立的包依赖空间，避免不同项目间的库版本冲突。本教程使用 **conda** 管理，这是

## 块 3

**本块学习检查**：完成安装后，请用自己的话说明：为什么“Add python.exe to PATH”对后续 LoRA 微调环境搭建至关重要？遇到无法解释的术语（如 PATH、pip）时记录下来再提问。

---

### 2.1 Windows 用户：为 LoRA 微调铺路

你的目标是用 24GB 显存做 LoRA 微调，而 Python 是所有 PyTorch 工作的地基。本块只做一件事：装好 Python 并确保命令行能调用它。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这会把 Python 加入系统“环境变量”。不勾选的话，命令行输入 `python` 会提示“不是内部或外部命令”，后续所有 pip 安装 PyTorch、transformers 等操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。若改路径，务必确保**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，

## 块 4

### 2.2 macOS 用户：为 LoRA 微调铺好 Python 地基

你已有 24GB 显存的 RTX 3090，目标是用 LoRA 做全参微调。在动手写训练脚本前，先把 Python 环境弄干净——这是所有 PyTorch 和 HuggingFace 代码的运行基础。

**关键认知**：现代 macOS（Catalina 10.15 起）不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用，版本可能老旧。**务必自行安装最新版 Python**，避免后续装 PyTorch 时出现版本不兼容的怪问题。

**方法一：官方安装包（推荐）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 完成。
3. 终端验证：

```bash
python3 --version
pip3 --version
```

**避坑提醒**：macOS 命令行默认**没有** `python` 命令，统一用 `python3` / `pip3`。写 LoRA 训练脚本时，记得用 `python3 train.py` 启动，别敲成 `python`。

**方法二：Homebrew（已装则用）**

```bash
brew install python
```

装完同样用 `python3 --version` 验证

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 5

你的目标是用 24GB 显存做 LoRA 微调，Linux 环境是第一步。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新环境更稳。

```bash
# 更新软件源
sudo apt update

# 安装 Python 3、pip 和 venv
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

`-y` 自动确认安装，省去手动输入；`python3-venv` 是建虚拟环境的模块，提前装好，避免后面补。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法

## 块 6

### 2.4 操作系统与实训环境该怎么选

你已有24GB显存的RTX 3090，目标直指LoRA微调，环境选择直接影响你能否顺利跑通训练。核心原则：**命令练习与GPU实训分开选环境**。

**日常写Python、练命令行（Windows用户）**：启用 **WSL2 + Ubuntu LTS**（如Ubuntu 22.04/24.04 LTS）。它提供接近真实生产的Linux环境，命令、路径、包管理与后续服务器一致，比Windows原生CMD更贴近实战。安装只需管理员PowerShell执行 `wsl --install`。

**日常写Python（macOS/Linux用户）**：直接用本机终端，本身就是类Unix环境。

**LoRA/QLoRA微调等GPU实训**：使用**装有Ubuntu LTS的NVIDIA GPU云服务器，或本地实体Linux（Ubuntu LTS）机器**。这类任务依赖NVIDIA驱动、CUDA与显卡直通，Ubuntu LTS对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox虚拟机"当最佳方案**。CentOS 8已停止维护、生态落后；VirtualBox这类桌面虚拟机**默认无法直通NVIDIA GPU**，跑不了真正的GPU训练，只适合纯CPU隔离小实验。命令练习用WSL2/Ubuntu LTS，GPU实训用云上或实体Ubuntu LTS，才是稳妥路线。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 7

## 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如你后面微调要用的 `transformers`、`peft`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**关键判断**：只有当你访问官方源明显变慢、经常超时，或处于网络受限环境时，才临时切到国内镜像。**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终是首选。

## 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你在 RTX 3090 上做 LoRA 微调，核心流程是：`pip install` 装好 HuggingFace 生态工具链（transformers、peft、bitsandbytes 等），然后跑训练脚本。如果每次装包都卡在下载上，那确实值得把镜像写进配置。但记住：**只有长期网络受限才这么做**，官方 PyPI 始终是最可靠来源，网络恢复后建议改回。

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建 `pip.ini` 文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从清华镜像下载包。

### macOS / Linux 操作步骤

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

写入同样的内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

### ⚠️ 关于 `trusted-host` 的坑

网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常用不需要。只有企业内网自建源、或确认来源可信且确实报证书错误时，才临时加。

### 验证配置是否生效

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

**本块学习检查**：完成本块后，用自己的话说明——这个镜像配置操作，和你接下来要做的 LoRA 微调有什么关系？具体来说，它解决了你安装 PyTorch、transformers、peft

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

VS Code 是微软出品的免费编辑器，启动快、插件丰富，适合 Python 及后续 Linux 命令行、脚本编写。安装步骤：访问 https://code.visualstudio.com 下载安装包，运行后一路下一步。**必装插件**：点击左侧活动栏最下方的四个方块图标（扩展），搜索安装 **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）和 **Pylance**（Python 语言服务器，让代码提示更智能）。安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`），输入 `Python: Select Interpreter`，选择你刚安装的 Python 版本（如 `Python 3.13.x`）。

---

**操作验证**：完成安装后，新建一个 `.py

## 块 10

### 4.2 方案二：PyCharm Community（免费、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置了代码分析、版本控制等功能，适合大型项目开发。

**安装步骤：**

1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置：**

1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured interpreter`，然后选择系统安装的 Python 3.10+。

---

**本块学习检查**：完成本块后，用自己的话说明本

## 块 11

## 5.1 为什么用 Conda

你的目标是 LoRA 微调，这意味着你马上要面对 PyTorch、CUDA、transformers 这些复杂依赖。Conda 不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身——这对深度学习项目几乎是必备工具。简单说：**Conda 让你的每个项目拥有独立的 Python 空间，互不干扰。**

## 5.2 安装 Conda：Anaconda vs Miniconda

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 1GB+），预装大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境更干净，后续 LoRA 微调需要什么包装什么包，避免无关库干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python 冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **Linux**：运行 `.sh` 后按提示输入 `yes` 同意协议、选择路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 conda 通过初始化脚本正确接管，而不是硬加 PATH。安装完成后执行 `source ~/.bashrc` 或重新登录终端，确保 conda 命令可用。
4. **验证安装**：打开**新的**终端（Windows 用 "Anaconda Prompt (Miniconda3)"；Linux 关闭并重开终端

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的RTX 3090有24GB显存，做LoRA微调时，不同项目常需不同版本的PyTorch和CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的

## 块 13

### 5.4 激活与退出环境

你的目标是LoRA微调，而微调通常需要独立的项目环境，避免依赖冲突。现在进入这个“独立房间”：

```bash
conda activate llm-learn
```

激活后，命令行前出现 `(llm-learn)` 标志，表示后续 `python` 和 `pip` 命令都作用于此环境，不影响系统或其他项目。

验证Python版本：

```bash
python --version
```

应显示 `Python 3.10.x`。

退出环境：

```bash
conda deactivate
```

### 5.5 查看和管理环境

查看所有环境：

```bash
conda env list
```

带 `*` 的是当前激活环境。删除环境（如需清理）：

```bash
conda remove -n llm-learn --all
```

---

## 第六章：在 VS Code / PyCharm 中使用 Conda 环境

在IDE中选择 `llm-learn` 作为解释

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

### 6.1 VS Code 配置 Conda 环境

在开始 LoRA 微调前，先确保你的代码跑在正确的环境里。操作如下：

1. 终端激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表中选择带 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样做**：VS Code 会自动识别 conda 环境。选中后，你在此工作区写的所有代码都会使用 `llm-learn` 里的 Python 和已装好的包——这对后续加载模型、跑 LoRA 至关重要。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/env

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是LoRA微调，但任何微调都始于一个能跑通代码的环境。本块就是验证工作台是否就绪——通过命令行和VS Code两种方式运行`print`，确认Python解释器正常。

**操作步骤（命令行）**：
1. 激活环境：`conda activate llm-learn`
2. 输入`python`进入交互模式，看到`>>>`提示符
3. 输入`print("Hello, Python! 我的环境准备好了！")`并回车
4. 正确打印后，输入`exit()`退出

**操作步骤

## 块 16

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

## 7.3 在 PyCharm 中验证

你的环境已经装好，现在用 PyCharm 跑通第一行代码，确认整个工具链能正常工作。这是后续做 LoRA 微调前必须通过的“点火测试”。

**操作步骤**：

1. 打开 PyCharm，新建项目。在 `New Project` 对话框中选择 `Previously configured interpreter`，指定 Python 3.10+ 的路径，创建 `hello.py` 文件。
2. 在文件中输入 `print("Hello")`。
3. 右键点击文件 → 选择 `Run 'hello'`。
4. 观察下方 Run 窗口，看到输出结果即验证成功。

**注意**：建议项目路径避免包含中文或空格，以减少某些库或工具在加载模型和数据集时可能出现的兼容性问题。

---

## 第八章：常见问题排查

你在配置环境时可能遇到以下问题，对照处理：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `

## 测评题目

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

### Q-KB0205-001
以下哪个是装饰器的正确语法？

### Q-KB0203-001
以下函数定义中，哪个参数是**默认参数**？
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

### Q-KB0201-004
以下代码输出是？
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```

### Q-KB0204-005
请写一个自定义异常类 `ModelError`，继承自 `Exception`，并在模型加载失败时抛出该异常，附带错误信息。
