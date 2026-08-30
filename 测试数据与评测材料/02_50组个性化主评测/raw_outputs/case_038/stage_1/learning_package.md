# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

你已经掌握 PyTorch 和 Transformer 基础，目标是用 LoRA 微调模型。但先别急着写训练代码——你写的 Python 代码，电脑并不直接认识。它只懂 0 和 1，而你的 `.py` 文件对电脑来说只是一堆文本。

**Python 解释器**就是那个"翻译官"：它把你写的人类可读代码，逐行翻译成电脑能执行的机器指令。没有它，你的 LoRA 训练脚本根本跑不起来。

打个比方：你写好一份英文菜谱（Python 代码），解释器就是那位厨师（电脑）的翻译，把菜谱翻译成厨师能听懂的操作步骤（机器指令）。翻译官不在，菜谱就只是一张纸。

所以，环境准备的第一步，就是确保这位"翻译官"正确安装并可用。后面我们会一步步验证它是否就位。

---

**本块学习检查**

完成本块后，请用自己的话说明：Python 解释器在你的 LoRA 微调学习目标中扮演什么角色？如果遇到无法解释的术语（如"解释器"本身），先记录下来，稍后提问。

## 块 2

**IDE：你的“工作台”**

记事本也能写代码，但效率极低。IDE提供语法高亮、自动补全、运行调试等功能，让你写代码像用Word写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。你已有PyTorch基础，选VS Code即可，轻量且适合后续LoRA脚本调试。

**虚拟环境：项目的“独立房间”**

想象两个项目：A需`numpy 1.0`，B需`numpy 2.0`，同装一个Python会冲突。虚拟环境为每个项目创建独立Python空间，互不影响。用**conda**管理，这是数据科学领域的主流方案；**venv**和**virtualenv**也是常见选择。

**Python 3.10+**

3.10是较新的稳定版本，引入更友好错误提示、模式匹配等新特性，主流库均已

## 块 3

**本块学习检查**：完成安装后，请用自己的话说明“把 Python 添加到 PATH”这一步，与你最终用 LoRA 微调模型的目标有什么关系？遇到任何看不懂的术语（比如“环境变量”），先记下来，稍后提问。

---

### 2.1 Windows 用户：三步装好 Python

你的目标是跑 LoRA 微调，而 PyTorch 等工具都依赖 Python 环境。下面三步帮你把“地基”打牢。

**第一步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标移到顶部 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统“环境变量”。不勾选的话，后续在命令行敲 `python` 会提示“不是内部或外部命令”，所有命令操作都无法进行——包括你之后装 PyTorch 和跑微调脚本。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。若想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd

## 块 4

### 2.2 macOS 用户：为 LoRA 微调铺好 Python 地基

现代 macOS（Catalina 10.15 起）不再预装 Python 2，部分系统可能带有供系统脚本使用的旧版 `python3`。**强烈建议自行安装最新版 Python**，避免依赖系统自带版本——这能确保你后续安装 PyTorch 等深度学习库时，不会因版本混乱而卡壳。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 命令行默认**并不提供** `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

在 Ubuntu/Debian 上安装 Python 3 和 pip，为后续搭建 LoRA 微调环境打基础。执行以下命令：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

`-y` 自动确认安装，`python3-venv` 用于创建虚拟环境，避免后续补装。安装后，你将拥有运行 Python 代码和安装依赖包的基础工具，这是进行 LoRA 微调实验的第一步。

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，且手上有 RTX 3060 12GB 显卡。这意味着你**最终必须跑在 Linux + NVIDIA GPU 环境**上，因为微调任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

**当前阶段怎么选？**

- **如果你是 Windows 用户**：日常练 Python、命令行时，启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它提供接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`。
- **如果你是 macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境。
- **做 LoRA 微调实训时**：用装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器。

**⚠️ 别踩的坑**：不要把 **CentOS + VirtualBox 虚拟机**当最佳方案。CentOS 8 已停止维护，CentOS Stream 8 也已于 2024 年 5 月 31 日 EOL；VirtualBox 默认无法直通 NVIDIA GPU，跑不了真正的 GPU 训练，只

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

`pip` 是 Python 的包管理工具，用来安装第三方库（如 `numpy`）。默认从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的默认源。

**什么时候用镜像源？** 仅当网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像是“网络不好时的临时加速手段”，不是永久替换项——官方源始终是首选。

**推荐做法：临时指定镜像源**（只对单次安装生效，不改全局配置）：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐？** 不影响平时默认行为，官方 PyPI 仍是默认源；网络恢复或镜像出问题时

## 块 8

## 3.3 进阶：把镜像设为默认——仅限长期网络受限时

你正在为 LoRA 微调做准备，而微调离不开快速安装 PyTorch 等大型依赖包。如果你**长期处于网络受限环境**，几乎每次安装包都卡在下载上，才值得把镜像写进配置。注意：官方 PyPI 始终是最可靠的，网络恢复后记得改回。

**Windows 操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**它。只有企业内网自建源或确认来源可信时才临时加。

**macOS / Linux 操作步骤：**

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

写入同样内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

### 3.5 改回官方源

删除 `pip.ini` / `pip.conf`，或执行：

```bash
pip config unset global.index-url
```

**本块学习检查**：用自己的话说说，为什么只在长期网络受限时才把镜像设为默认？这个配置和你的 LoRA 微调目标有什么关系？遇到不懂的术语记下来再提问。

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，要频繁修改 Python 脚本、查看训练日志，它都能胜任；以后学 Linux 命令行、写配置文件也用得上。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能——这对你调试 LoRA 训练脚本很关键。

**配置 Python 解释器**：

1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。

## 块 10

PyCharm Community 是 JetBrains 出品的免费 Python 专用 IDE，内置代码分析、版本控制等功能，适合大型项目。安装时访问官网下载社区版，运行安装程序并勾选 `Add "Open Folder as Project"`；`Add to PATH` 为可选，用于命令行启动。

首次启动：新建项目，`Location` 选路径（**不含中文**），`Interpreter` 选 `Previously configured interpreter`，指向系统 Python 3.10+。

**本块学习检查**：完成安装后，用自己的话说明 PyCharm 与你的 LoRA 微调目标有何关系（提示：它帮你管理代码和项目文件）。若遇到无法解释的术语（如 Interpreter），记录下来再提问。

## 块 11

## 5.1 为什么用 Conda

你之前已经掌握了 Python 基础环境（KB-04、KB-08），现在要进入 LoRA 微调，第一步就是搭好“项目隔离”的工作台。Conda 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于后续学习深度学习（需要 PyTorch、CUDA 等复杂依赖），conda 几乎是必备工具——你的 RTX 3060 12GB 显卡要跑 LoRA，PyTorch 和 CUDA 的版本搭配必须精确，Conda 能帮你把每个项目的依赖“锁”在独立空间里，互不干扰。

## 5.2 安装 Conda：Anaconda vs Miniconda

有两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包——LoRA 微调只需要 PyTorch、transformers 等少数几个包，用 Anaconda 会白白占用磁盘空间。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。

## 块 12

**创建你的第一个虚拟环境**

你正在为 LoRA 微调打基础——不同项目需要不同版本的 PyTorch 和 CUDA，现在学会隔离环境，之后直接复用这套流程。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**拆解一下**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定装 Python 3.10

执行后终端会列出待装包并询问 `Proceed ([y]/n)?`，输入 `y` 回车，等待完成。

**本块学习检查**：用自己的话说说

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

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

---

**动手验证**：激活环境后，运行 `python --version`，确认版本号以 `3.10` 开头；再运行 `conda deactivate`

## 块 14

## 6.1 VS Code 配置 Conda 环境

你正为 LoRA 微调搭建工作台，这一步是让 VS Code 用上你已建好的 `llm-learn` 环境。

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境，选好后，你在这个工作区写的代码就会使用 `llm-learn` 中的 Python 和已装好的包——也就是你为 LoRA 微调准备的 PyTorch 等依赖所在的环境。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`

**本块学习检查**：

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是LoRA微调，但任何微调都要先跑通代码环境。本块就是验证环境是否可用。

**在命令行验证**（适合快速测试）：
1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 并回车
4. 正确打印后，输入 `exit()` 退出

**在VS Code中验证**（适合写正式脚本）：
1. 新建文件夹

## 块 16

## 7.3 在 PyCharm 中验证

既然你的目标是 LoRA 微调，后面会频繁在 PyCharm 里跑训练脚本，所以先在这里把“运行代码”这个动作练熟。

操作步骤：

1. 新建项目，创建 `hello.py` 文件。
2. 在编辑区右键点击文件 → 选择 `Run 'hello'`。
3. 下方 Run 窗口会输出程序结果。

跑通这一步，说明你的开发工作台已经能正常执行 Python 代码了——这是后续运行 LoRA 训练脚本的基础。

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

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？

### Q-KB0201-005
以下代码输出什么？
```python
for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
```

### Q-KB0204-005
请写一个自定义异常类 `ModelError`，继承自 `Exception`，并在模型加载失败时抛出该异常，附带错误信息。

### Q-KB0202-005
以下代码输出是？
```python
d = {}
d.setdefault("key", []).append(1)
d.setdefault("key", []).append(2)
print(d)
```

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。
