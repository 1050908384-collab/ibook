# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你正打算用10GB显存的RTX 3080跑QLoRA微调，这很实际——但先别急着碰模型。QLoRA微调的本质是运行Python代码来加载、训练和保存模型，而这一切的第一步，是让电脑能“读懂”你的代码。

**Python解释器就是那个“翻译官”**。你写的代码是人类语言，电脑只认0和1。解释器负责把你的`.py`文件翻译成机器指令。没有它，你写的代码只是一堆文本，模型训练根本无从谈起。

你已有PyTorch和Transformer基础，所以这里不啰嗦基础语法。但请记住一个关键点：**你之后所有QLoRA脚本（比如加载模型、配置LoRA、跑训练循环）都要靠解释器来执行**。装好它，你的工作台才算真正“通电”。

现在，请检查你的电脑上是否已安装Python解释器。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是QLoRA微调，这需要写代码、跑脚本。记事本写代码效率极低，所以你需要一个“工作台”——**IDE**。它提供语法高亮、自动补全、运行调试，让你写代码像用Word写文档一样舒服。推荐两款：**VS Code**（轻量免费）和**PyCharm**（专业强大）。你的RTX 3080跑微调时，IDE本身不占显存，选顺手的即可。

更关键的是**虚拟环境**。你只有10GB显存，想用小模型做QLoRA，

## 块 3

好的，我们直接进入正题。既然你的目标是QLoRA微调，那Python环境就是一切的起点。我们跳过基础概念，直接搞定安装。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。鼠标移到顶部菜单 **Downloads**，在Downloads页面中找到“Windows”部分，选择适合你系统的安装包（64位或32位）。点击下载 **Windows installer (64-bit)** 这个文件。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件。弹出安装界面后，**务必做两件事**：

1.  **勾选最下方的 "Add python.exe to PATH"**。这一步至关重要。如果不勾选，你之后在命令行里输入 `python` 会提示“不是内部或外部命令”，后续所有操作都无法进行。
2.  点击 **"Install Now"** 使用默认安装即可。默认路径不含中文和特殊符号，能避免后续很多路径报错。

**第三步：验证安装**

1.  按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2.  输入 `python --version` 并回车。如果显示 `Python 3.13.x` 之类的版本号，说明安装成功。
3.  再输入 `pip --version` 并回车。如果也显示版本号，说明包管理工具 `pip` 也一并装好了。

**本块学习检查**

## 块 4

**macOS 用户：安装 Python 3.10+**

macOS 12.3 起不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用。为了你的 QLoRA 微调项目，**务必自行安装最新版 Python**，避免依赖系统版本。

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

已安装 Homebrew 的话，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**

完成本块后，用自己的话说明：为什么安装 Python 时要避开系统自带版本？这与后续用 RTX 3080 做 QLo

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的Linux可能自带Python，但版本偏老。QLoRA微调需要干净、可用的Python环境，所以先更新软件源并安装基础工具：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

`-y` 自动确认安装，免去手动输入`y`。`python3-venv` 用于创建虚拟环境，避免污染系统Python——

## 块 6

你的目标是QLoRA微调，而QLoRA这类GPU实训，对环境的要求和日常写Python完全不同。选错环境，后面跑模型时会寸步难行。

**核心结论**：日常练习用WSL2，GPU实训用Ubuntu LTS。

- **如果你用Windows**：日常写Python、练命令行，推荐启用**WSL2 + Ubuntu LTS**（如Ubuntu 22.04/24.04 LTS）。它让你在Windows里获得接近真实生产的Linux环境，命令、路径、包管理都和后续服务器一致。安装只需管理员PowerShell执行`wsl --install`。
- **如果你用macOS/Linux**：直接使用本机终端即可，本身就是类Unix环境。
- **做LoRA/QLoRA微调或vLLM部署**：必须用**装有Ubuntu LTS的NVIDIA GPU云服务器，或本地实体Linux（Ubuntu LTS）机器**。这类任务依赖NVIDIA驱动、CUDA与显卡直通，Ubuntu LTS对深度学习生态支持最成熟。

**⚠️ 避坑提醒**：不要用"CentOS + VirtualBox虚拟机"当最佳方案。CentOS 8已于2024年5月31日停止维护；VirtualBox默认无法直通NVIDIA GPU，跑不了真正的GPU训练，只适合纯CPU隔离小实验。

---

**本块学习检查**：完成本块后，用自己的话说明：为什么

## 块 7

你在用 10GB 显存的 RTX 3080 跑 QLoRA，安装 PyTorch 这类库时，`pip` 默认从官方源 `pypi.org` 下载——这是最权威、最完整的源，应始终保留为默认。

**什么时候才需要镜像源？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像是“网络不好时的临时加速手段”，不是永久替换项。

**首选做法：临时指定镜像源（推荐）**。只对某一次安装临时指定，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样不影响平时默认行为，网络恢复或镜像出问题时也无需清理配置。

**本块学习检查**：完成本块后，用自己的话说明：本块内容与你“用 RTX 3080 做 QLoRA 微调”这个

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正为 QLoRA 微调准备环境，10GB 显存意味着要下载不少预训练模型和依赖包。如果网络长期不稳，把清华镜像设为默认能省下大量等待时间。但请记住：**官方 PyPI 始终是最可靠来源**，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建 `pip.ini` 文本文件。
4. 用记事本写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**此项。只有企业内网自建源或确认来源可信时才临时加。

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

删除 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：完成本块后，用自己的话说明：为什么 QLoRA 微调准备阶段值得配置默认镜像？配置时为什么要避免添加 `trusted-host`？遇到无法解释的术语（如 TLS）时记录下来，稍后提问。

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 QLoRA 微调时，要写 Python 脚本、改配置文件，它都能胜任。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能

## 块 10

## 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具等功能，适合大型项目开发——比如你后续要做的 QLoRA 微调项目，代码量会明显增长，一个好用的 IDE 能帮你省下不少排查时间。

**安装步骤**：

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费）
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置**：

1. 打开 PyCharm，选择 `New Project`
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）
3. `Interpreter` 选择 `Previously configured interpreter`，然后点击右侧的 `...` 按钮，选择已安装的 Python 解释器（如 conda 环境或系统 Python），

## 块 11

### 5.1 为什么用 Conda

你已经有 PyTorch 和 Transformer 的基础，接下来要跑 QLoRA 微调，这需要管理 PyTorch、CUDA 等一堆复杂依赖。Conda 能帮你为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身——这正是深度学习项目隔离环境的关键工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB），预装大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 70-80MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，环境更干净，后续需要什么装什么，适合你这种目标明确的微调项目。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。macOS 用户需根据芯片架构选择：Apple Silicon 建议使用 `.sh` 安装包。
3. 运行安装程序：
   - **Windows**：一路下一步，**默认不勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"），避免和系统已有 Python 冲突。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。若希望在系统终端直接使用 conda，可勾选该选项，但需注意与已有 Python 的冲突风险。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化），这样终端能正确接管 conda。
4. **验证安装**：打开**新的**终端

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（QLoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：后续做 QLoRA 微调时，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给

## 块 13

### 5.4 激活与退出环境：为 QLoRA 微调锁定“工作台”

你的目标是用 10GB 显存跑 QLoRA，这意味着环境必须干净、可控。激活环境就是进入一个专属“独立房间”，避免依赖冲突。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。这对 QLoRA 至关重要——你之后安装的 `transformers`、`peft` 等库都会隔离在此，不会污染系统 Python。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。确认版本，避免后续依赖不兼容。

**退出环境**（回到系统默认环境）：

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 QLoRA 微调，这需要在一个干净、依赖齐全的环境里跑代码。下面把 VS Code 指向你刚建好的 `llm-learn` 环境：

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。这样你后续跑 QLoRA 脚本时，import 的 PyTorch、Transformers 等库都来自这个环境，不会和系统 Python 混淆。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，步骤类似：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径下拉框中选择或浏览到 conda 环境的

## 块 15

## 在命令行验证你的 Python 环境

既然你的目标是QLoRA微调，那第一步就是确保环境能跑起来。我们先用命令行做最基础的验证。

**操作步骤：**

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符。
3. 输入：`print("Hello, Python! 我的环境准备好了！")`
4. 回车后，如果正确打印出文字，说明Python环境正常。输入 `exit()` 退出。

## 在 VS Code 中验证

命令行验证通过后，再用编辑器确认开发环境：

1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`，输入 `print("Hello, Python!")`。
3.

## 块 16

## 7.3 在 PyCharm 中验证

你的目标是 QLoRA 微调，这需要你熟练操作 Python 开发环境。现在，我们来验证 PyCharm 是否已准备好迎接你的第一个项目。

**操作步骤：**

1. **新建项目**：打开 PyCharm，点击 `File` → `New Project`，创建一个新项目。
2. **创建文件**：在项目目录上右键 → `New` → `Python File`，命名为 `hello.py`。
3. **编写代码**：在文件中输入 `print("Hello, QLoRA!")`。
4. **运行**：右键点击 `hello.py` 文件 → 选择 `Run 'hello'`。
5. **查看输出**：下方会弹出 `Run` 窗口，如果看到 `Hello, QLoRA!` 输出，说明环境完全正常。

> **小提示**：项目路径不要包含中文或空格，这是本章强调的好习惯，能避免后续很多莫名其妙的报错。

---

## 本章小结

完成本章后，你的电脑上应该具备以下内容：

| 组件 | 状态 |
|------|------|
| Python 解释器 | 已安装并可用 |
| PyCharm | 已安装并配置完成 |
|

## 测评题目

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0205-008
写一个类 `Stack`（栈），实现 `push(item)`、`pop()`、`peek()`、`is_empty()` 方法。`pop()` 在栈空时抛出 `IndexError`。

### Q-KB0202-005
以下代码输出是？
```python
d = {}
d.setdefault("key", []).append(1)
d.setdefault("key", []).append(2)
print(d)
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

### Q-KB0204-003
请写代码：读取 `data.txt` 文件，捕获文件不存在的异常，若不存在则打印 "文件未找到"。

### Q-KB0204-001
以下哪种文件打开模式表示**追加写入**（不覆盖原内容）？
