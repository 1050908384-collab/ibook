# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你正为 QLoRA 微调做准备，但先别急着碰模型。QLoRA 微调的本质，是让电脑执行你写的 Python 代码来训练模型。而电脑只认 0 和 1，看不懂你写的代码——这就是为什么你需要 **Python 解释器**。

把它想成一位“翻译官”：你写的是人类语言（Python 代码），电脑执行的是机器指令（0 和 1）。解释器负责把前者翻译成后者。没有它，你写的 `.py` 文件只是一堆普通文本，电脑根本不会理睬。

对你而言，这意味着：**装好解释器，是跑通 QLoRA 微调脚本的第一道门槛**。后续你运行任何微调代码，背后都是这位“翻译官”在默默工作。

**动手验证**：打开你的终端（命令行），输入 `python --version` 并回车。如果看到类似 `Python 3.x.x` 的版本号，说明解释器已就位；如果提示“找不到命令”，说明还没装好，记下这个报错，我们下一步解决。

## 块 2

你的目标是 QLoRA 微调，这意味着你很快会接触 HuggingFace 生态和大量脚本。工欲善其事，必先利其器——本块就是帮你把“工作台”搭对，避免后续在环境问题上卡壳。

**IDE 选型**：你有两个主流选择。**VS Code** 轻量免费，插件丰富，适合快速改脚本；**PyCharm** 专业强大，调试功能更顺手。两者都能满足你的需求，建议先用 VS Code 上手，它更贴近深度学习社区的主流工作流。

**虚拟环境（重点）**：QLoRA 微调会依赖特定版本的 PyTorch、transformers 等库。不同项目对库版本要求可能冲突（比如项目 A 要 numpy 1.0，项目 B 要 2.0），全装一起会报错。**虚拟环境**就是给每个项目一个独立的 Python 空间，互不

## 块 3

## 2.1 Windows 用户：为 QLoRA 微调铺路

你已经有 PyTorch 和 Transformer 基础，目标是用 12GB 显存跑 QLoRA。在动手微调前，先把 Python 环境搭好——这是所有后续操作的地基。

**第一步：下载安装包**

打开浏览器访问 https://www.python.org，鼠标移到顶部 **Downloads** 菜单，网站会自动推荐适合你系统的 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，在命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令操作都会卡住——包括你之后要用的 `pip install` 安装 QLoRA 相关库。

2. 点击 **"Install Now"**（默认安装）  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

按 `Win + R`，输入 `cmd` 回车打开命令提示符，依次输入：

```bash
python --version
```

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 4

### macOS 用户：安装 Python 3.10+

你的目标是 QLoRA 微调，这需要稳定的 Python 环境。macOS 自带的 Python 版本可能较旧或缺失，且系统 Python 可能被系统工具依赖，不建议直接用于开发环境。**强烈建议自行安装最新版**，避免后续环境问题。

**方法一：官方安装包（推荐）**

1. 访问 [python.org](https://www.python.org)，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**注意**：现代 macOS 默认**不提供** `python` 命令，统一使用 `python3` / `pip3`。请始终用这两个命令，避免"命令不存在"或误用系统旧版本的坑。如果 `python3 --version` 显示的版本不是刚安装的版本，请检查 PATH 环境变量，确保新安装的 Python 在 PATH 中优先。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块

## 块 5

**本块学习检查**：完成安装后，用自己的话说明“Python 环境”与“QLoRA微调”的关系；遇到不懂的术语（如虚拟环境）先记下再提问。

Linux 常自带 Python，但版本可能偏老。Ubuntu/Debian 执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

`-y` 自动确认安装，免手动输入；`python3-venv` 用于建虚拟环境，一并装好省得补。

**练习**：运行上述命令，确认 `python3 --version` 输出版本号。你的 RTX 3060 后续跑 QLoRA

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 QLoRA 微调，且手头是 RTX 3060 12GB 显卡。这意味着你最终要跑 GPU 训练，所以环境选择直接决定你能否顺利跑通。

**日常练习阶段**：如果你是 Windows 用户，强烈建议启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。macOS/Linux 用户直接用本机终端即可。

**GPU 实训阶段**：QLoRA 微调这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 "CentOS + VirtualBox 虚拟机" 当最佳方案。CentOS 8 已停止维护；VirtualBox 默认无法直通 NVIDIA GPU，跑不了真正的 GPU 训练，只适合纯 CPU 隔离小实验。

**路线总结**：命令练习用 WSL2/Ubuntu LTS，GPU 实训用云上或实体的 Ubuntu LTS。

---

**

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 7

你的目标是用 QLoRA 微调模型，而 QLoRA 依赖 PyTorch 等库。安装这些库时，`pip` 默认从官方源（`pypi.org`）下载——这是最权威、最完整的源，应始终保留为默认。

**只有**当你访问官方源明显变慢或经常超时时，才需要临时切换。镜像只是“网络不好时的加速手段”，不是永久设置。

**首选做法**：只对某次安装临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样官方源仍是默认，网络恢复或镜像出问题时无需清理配置。

**本块学习检查**：用自己的话说明“临时指定镜像”与“永久换源”的区别，以及为什么前者更适合你后续安装 QLoRA 依赖的场景。遇到不懂的术语（如“全局配置”）先记下来再提问。

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 QLoRA 微调准备环境，而 pip 下载速度直接影响你安装 `transformers`、`bitsandbytes` 等依赖的效率。如果你长期处于网络受限环境，几乎每次都要手动指定镜像源，那才值得把它设为默认。**注意**：官方 PyPI 始终是最可靠的，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

`index-url` 告诉 pip 默认从这个地址下载包。`trusted-host` 用于跳过该镜像的 TLS 证书校验，仅在网络受限或证书验证失败时使用；若你的网络环境正常，可删除此行以保留传输安全。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**它。只有企业内网自建源或确认来源可信时才临时加。

**macOS / Linux 用户：**

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
trusted-host = pypi.tuna.tsinghua.edu.cn
```

4. 按 `Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：
1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，从列表中选择已

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

你已经有 PyTorch 和 Transformer 的基础，接下来要学 QLoRA 微调，这意味着你的项目会依赖 PyTorch、CUDA 等一堆复杂包。如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 能帮每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身——这对深度学习几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，环境更干净，需要什么装什么。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python 冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"**。
   - **macOS/Linux**：运行 `.sh` 后按提示输入 `yes` 同意协议、选择路径，最后一步选 `yes` 让安装器执行 **`conda init`**。这样终端打开时 conda 会正确接管，而非硬加 PATH。
4. **验证安装**：打开**新的**终端（Windows 用 "Anaconda Prompt (Miniconda3)"；macOS/Linux 重开终端），输入：

```bash
conda --version
```

显示类似 `conda 24.x.x` 即成功。

---

**本块学习检查**：完成本块后，用自己的话说明：Conda

## 块 12

你的目标是用 12GB 显存的 RTX 3060 跑 QLoRA 微调，而 QLoRA 依赖特定版本的 PyTorch 和 CUDA。为了避免不同项目之间互相干扰，现在先学会用 Conda 建一个独立环境，以后每次微调都能直接复用这套流程。

打开终端，执行：

```bash
conda create -n llm-learn python=3.10
```

这条命令拆开看：
- `conda create`：创建环境的固定开头。
- `-n llm-learn`：给环境命名为 `llm-learn`（`n` 就是 `name` 的缩写）。
- `python=3.10`：指定这个环境里安装 Python 3.10。

执行后，终端会列出准备安装的包

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 5.4 激活与退出环境

**激活环境**（进入这个"独立房间"）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

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

**删除环境**（如果

## 块 14

### 6.1 VS Code 配置 Conda 环境

你已经创建了 `llm-learn` 环境，现在让 VS Code 用它跑代码：

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境。选好后，你写的代码就会用 `llm-learn` 里的 Python 和已装好的包——包括后面做 QLoRA 要用的 PyTorch 等。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

**本块学习检查**：完成本块后，

## 块 15

## 7.1 在命令行验证

激活环境后，输入 `python` 进入交互模式，看到 `>>>` 提示符，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车后正确打印文字，按 `exit()` 退出。

## 7.2 在 VS Code 中验证

新建文件夹 `my-first-project` 并用 VS Code 打开，新建 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角运行按钮或按 `Ctrl + F5`，下方终端应输出 `Hello from VS Code!`。

**本块学习检查**：完成本块后，用自己的话说明本块内容与 QLoRA 微调目标的关系——环境验证是后续

## 块 16

## 7.3 在 PyCharm 中验证

你的目标是用 QLoRA 微调大模型，而这一切的前提是代码能跑起来。现在，我们就在 PyCharm 里跑通第一行代码，确认整条工具链是通的。

**操作步骤（跟着做）：**

1. 打开 PyCharm，新建一个项目。
2. 在项目中创建一个新文件，命名为 `hello.py`。
3. 在文件里输入：`print("Hello, QLoRA!")` —— 这是你与模型微调之路的第一次对话。
4. 在文件编辑区**右键点击**，选择 `Run 'hello'`。
5. 看 PyCharm **下方**的 Run 窗口，如果输出了 `Hello, QLoRA!`，说明环境验证成功。

> 小提示：如果 Run 窗口没有出现，检查一下是不是文件没保存（文件名旁有星号表示未保存），按 `Ctrl+S` 保存后再运行。

---

## 常见问题排查（精简版）

跑代码时遇到报错很正常，下面是几个高频问题，直接对照解决：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源） |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

---

## 本章小结

现在你的电脑应该具备：

| 组件 | 状态 |

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

### Q-KB0202-003
以下代码输出是？
```python
s = {1, 2, 3, 3, 2, 1}
print(len(s))
```

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0205-009
写一个装饰器 `timer`，测量并打印被装饰函数的执行时间（秒），同时不改变原函数的返回值。然后用它装饰一个计算前100万个数平方和的函数。

### Q-KB0204-001
以下哪种文件打开模式表示**追加写入**（不覆盖原内容）？

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

### Q-KB0205-005
请写一个**带参数的装饰器** `retry(times=3)`，使被装饰的函数在抛出异常时自动重试指定次数。
