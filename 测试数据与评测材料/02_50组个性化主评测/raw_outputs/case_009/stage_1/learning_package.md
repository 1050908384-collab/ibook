# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

你已经有 Transformer 和 HuggingFace 的基础，但环境这块我们快速对齐一下。

**核心概念**：你写的 Python 代码是人类可读的文字，但电脑只认 0 和 1。**Python 解释器**就是把你写的代码翻译成电脑能执行的指令的"翻译官"。没有它，你写的 `.py` 文件只是一堆文本，什么也干不了。

**和你目标的关系**：你要做 LoRA 微调，核心操作是写 Python 脚本调用 HuggingFace 的 PEFT 库。如果解释器没装好，你的微调脚本连跑都跑不起来——不是模型的问题，是"翻译官"没上岗。

**动手验证**：打开终端，输入 `python --version`，看到版本号输出就说明解释器已就绪。如果提示"不是内部或外部命令"，说明解释器还没装或没加入环境变量。

**本块学习检查**：用自己的话说说——为什么没有 Python 解释器，你的微调脚本就无法执行？如果遇到解释不清的术语（比如"环境变量"），先记下来，后面章节会讲到。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是LoRA微调，这需要写代码、跑脚本。记事本也能写，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

微调时，不同项目依赖版本可能冲突。比如A项目要`numpy 1.0`，B项目要`numpy 2.0`，装一起会报错。**虚拟环境**

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要先有能运行 PyTorch 的 Python 环境。本块带你装好 Python 本身。

**第一步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标移到顶部 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   不勾选的话，命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令都跑不了。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续报错。想改路径的话，确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入并回车：

```bash
python --version
```

3. 显示 `Python 3.13.x`（或你下载的版本号）即

## 块 4

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

macOS 12.3 起不再预装 Python 2，系统可能带旧版 `python3`。**强烈建议自行安装最新版 Python**，别依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 完成。
3. 打开终端，验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认不提供 `python` 命令（早期曾指向 Python 2，现已移除），系统与官方安装包统一用 `python3`。请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

已装 Homebrew 的话，终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

**练习任务**：安装完成后，在终端依次运行 `python3 --version` 和 `pip3 --version

## 块 5

**本块学习检查**：完成安装后，请用自己的话说明“Python 3.10+ 环境”与“LoRA微调”的关系；若遇到无法解释的术语（如`venv`），记录下来再提问。

---

Linux（Ubuntu/Debian）通常自带Python，但版本可能偏老。执行以下命令安装：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3、pip 和虚拟环境模块
sudo apt install python3 python3-pip python3-venv -y

# 验证版本
python3 --version
```

**要点**：`-y` 自动确认安装，免手动输入`y`；`python3-venv` 用于创建

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是跑通 LoRA 微调，而微调依赖 GPU 环境。选错操作系统，后面会卡在驱动和 CUDA 上。

**日常写 Python、练命令行（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它提供接近真实生产的 Linux 环境，命令、路径、包管理与服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护；VirtualBox 默认无法直通 NVIDIA GPU，跑不了真正的 GPU 训练，只适合纯 CPU 隔离小实验。

## 块 7

你在用 HuggingFace 做文本分类时，应该已经装过不少库。如果某次 `pip install transformers` 卡住不动，或频繁超时，这就是需要镜像源的信号。

**核心原则**：官方 PyPI（`pypi.org`）永远是最权威、最完整的默认源，应保留。镜像只是网络不畅时的**临时加速手段**，不是永久替换。

**推荐做法**：只对某一次安装临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样官方源仍是默认，网络恢复或镜像出问题时，无需清理任何配置。

**本块学习检查**：完成本块后，用自己的话说明：本块内容与你当前 LoRA 微调目标（安装相关依赖库）有什么关系？遇到无法解释的术语（如“镜像”“PyPI”）时记录下来，稍后提问。

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你已经熟悉 HuggingFace 生态，对 `pip install` 应该不陌生。这里要提醒你：**只有当长期处于网络受限环境、几乎每次都要用镜像时**，才把镜像写入配置作为默认源。配置后请记住——官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

### Windows 用户操作

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

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

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 改回官方源

删除上面创建的 `pip.ini` / `pip.conf` 文件即可，

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 打开 VS Code，按 `Ctrl + Shift + P`

## 块 10

**本块学习检查**：完成安装后，请用自己的话说明：PyCharm Community 的安装与你的 LoRA 微调目标有什么关系？若遇到“解释器”“虚拟环境”等术语无法解释，请记录下来再提问。

---

**目标**：为 LoRA 微调准备一个专业、开箱即用的代码编辑环境。

**为什么选它**：PyCharm 社区版免费，内置代码分析、版本控制等功能，适合管理微调脚本这类稍复杂的项目。

**安装步骤**：
1. 确保你的操作系统满足 PyCharm Community Edition 的最低要求（Windows 10/11、macOS 10.14+、Linux），并建议安装 Python 3.8 及以上版本。
2. 打开 https://www.jetbrains.com/pycharm/download/
3. 下载 **Community Edition**（社区版）
4. 运行安装程序，务必勾选：
   - `Add "Open Folder as Project"`（方便右键打开项目文件夹）
   - `Add to PATH`（命令行可直接启动）

**首次启动配置**：

## 块 11

### 5.1 为什么用 Conda

你已经熟悉 Transformer 和 HuggingFace，做文本分类时肯定装过不少包。但当你准备用 LoRA 微调模型时，依赖会更复杂——PyTorch、CUDA、transformers 等版本必须匹配。Conda 的价值就在这里：它为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。这意味着你可以为 LoRA 微调项目单独建一个环境，装好配套依赖，不污染你现有的其他项目。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装 numpy、pandas、jupyter 等 | 做数据分析、不想手动装包 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发 |

**推荐 Miniconda**——你的目标是 LoRA 微调，环境越干净越好，需要什么装什么。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进 PATH 容易和已有 Python 冲突，导致 `python`、`pip` 指向不明。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 即可。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选路径，最后一步选 `yes` 让安装器执行 **`conda init`**。这样终端会通过初始化脚本正确接管 conda，而不是硬加 PATH。
4. **验证安装**：打开**新的**终端（Windows 用 "Anaconda Prompt (Miniconda3)"；macOS/Linux 重开终端），输入：

```bash
conda --version

## 块 12

### 创建你的第一个虚拟环境

你已经熟悉 Transformer 和 HuggingFace，但 LoRA 微调需要独立的 PyTorch/CUDA 环境。这一步就是为后续微调项目打好隔离基础。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**拆解**：`conda create` 创建环境；`-n llm-learn` 命名环境（`n` 即 `name`）；`python=3.10` 指定 Python 版本。

终端会列出待装包并询问 `Proceed ([y]/n)?`，输入 `y` 回车等待完成。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA 微调

## 块 13

### 5.4 激活与退出环境

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

既然你已熟悉 HuggingFace 工具链，这里的关键是确保 VS Code 指向正确的解释器。先激活环境：`conda activate llm-learn`。然后打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的路径（如 `Python 3.10.x ('llm-learn': conda)`）。这样，你写的所有代码都会使用该环境中的 Python 和已装包——包括你后续做 LoRA 微调时需要的 transformers、peft 等。

## 6.2 PyCharm 配置 Conda 环境

若你用 PyCharm：打开 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`），左侧选 `Project: 你的项目名` → `Python Interpreter`，点击齿轮 → `Add...`，选 `Conda Environment` → `Existing environment`，在 `Interpreter` 路径中指定 conda 环境的 Python 可执行文件：
- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

配置完成后，务必验证：在 VS Code 或 PyCharm 中新建一个 Python 文件，输入 `import torch`，然后输入 `print(torch.__version__)` 并运行。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语

## 块 15

## 7.1 在命令行验证

你已经熟悉Transformer和HuggingFace，但微调LoRA前，先确认环境能跑通。打开终端，激活环境：

```bash
conda activate llm-learn
```

输入 `python` 进入交互模式，看到 `>>>` 提示符后，执行：

```python
print("Hello, Python! 我的环境准备好了！")
```

正确打印文字后，输入 `exit()` 退出。

## 7.2 在 VS Code 中验证

新建文件夹 `my-first-project`，用 VS Code 打开。新建 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角小三角或按 `Ctrl + F5`，下方终端应输出：

```
Hello from VS Code!
```

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释

## 块 16

### 7.3 在 PyCharm 中验证

你已经熟悉 Transformer 和 HuggingFace，对 Python 环境应该不陌生。这里快速过一遍 PyCharm 的验证流程，确保工具链畅通，为后续 LoRA 微调实验打好基础。

**操作步骤：**

1. 打开 PyCharm，新建一个项目（路径不要包含中文或空格）。
2. 在项目中新建 Python 文件，命名为 `hello.py`。
3. 在文件中输入一行代码：`print("Hello, LoRA!")` —— 内容随意，关键是验证环境能跑通。
4. 在编辑区右键点击文件 → 选择 `Run 'hello'`。
5. 查看下方 **Run 窗口**，如果输出了你打印的内容，说明 PyCharm 配置正确，可以正常执行 Python 代码。

**小提示**：如果你之前用 VS Code 或命令行，这里只是换了个入口，本质都是调用同一个 Python 解释器。确保 PyCharm 右下角显示的解释器是你创建的 `llm-learn` 环境，而不是 base 环境。

---

### 常见问题速查

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里

## 测评题目

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

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

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0201-001
以下哪个是Python中**正确的**变量赋值？
