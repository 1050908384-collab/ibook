# 学习包

## 块 1

**1.1 Python 解释器：程序的"翻译官"**

你已经熟悉 Tokenizer 如何把文本切成子词，现在换个视角：Python 解释器做的事，本质上也是一种"翻译"——不过它翻译的对象不是文本，而是你写的代码。

**核心类比**：你写的 `.py` 文件是人类可读的文字，但电脑 CPU 只认 0 和 1。解释器就是那个"翻译官"，把代码逐行翻译成机器指令。没有它，`.py` 文件只是一堆文本，什么也做不了。

**为什么这对你重要**：你在 Transformer 中处理的 token 序列，最终都要通过 Python 代码来驱动模型训练和推理。理解解释器这一层，你就能明白：为什么同样的代码在不同环境（比如你的 RTX 3060 上）表现不同——因为解释器负责把你的高级指令翻译成硬件能执行的底层操作。

**一个值得记住的点**：解释器是"逐行翻译并立即执行"，而不是像编译器那样"全部翻译完再执行"。这意味着你写的每一行 Python 代码，在解释器眼里都是"待翻译的指令流"——就像 BPE 把文本切成 token 流一样，解释器把代码切成一条条可执行的指令。

**学习检查**：用自己的话回答——如果没有 Python 解释器，你写好的 `model.py` 文件在电脑上运行时会怎样？

## 块 2

### 1.2 代码编辑器（IDE）：你的"工作台"

你已熟悉Tokenizer和BPE，接下来要深入Transformer，代码量会骤增。记事本写代码效率极低，**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试，让你写代码像用Word写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。建议选VS Code，轻量且对PyTorch项目友好。

### 1.3 虚拟环境：项目的"独立房间"

你即将用PyTorch搭建Transformer，不同项目依赖版本可能冲突。比如A项目要`numpy 1.0`，B项目要`numpy 2.0`，装一起会报错。**虚拟环境**为每个项目创建独立Python空间，互不影响。用 **conda** 管理，这是最主流方案。

---

## 第二章：安装 Python

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你已经熟悉 Tokenizer 和 BPE，接下来要深入 Transformer 架构。在动手写代码之前，先把 Python 环境搭好——这是运行一切深度学习实验的基础。

**第一步：下载安装包**

打开浏览器，访问 https://www.python.org。鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

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

3. 如果

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

你已熟悉 Tokenizer 与 BPE，接下来要深入 Transformer 架构。在动手实现之前，先把 Python 环境搭好——这是所有后续代码的基础。

**关键背景**：现代 macOS（Catalina 10.15 起）不再预装 Python 2，系统自带的 `python3` 版本可能较旧，建议自行安装最新版 Python 以避免兼容性问题。

---

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 上，命令行默认**不提供** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本的坑。

---

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

已安装 Homebrew 的话，在终端执行：

```bash
brew install python
```

安装

## 块 5

你的目标不是装好 Python 本身，而是为后续跑通 Transformer 模型铺路。在 Ubuntu/Debian 上，系统通常自带 Python，但版本可能偏老，直接用来训练模型容易踩坑。

打开终端，依次执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

第一行更新软件源，确保能装到最新包；第二行安装 Python 3、包管理器 pip 和虚拟环境模块 venv。`-y` 让系统自动确认，省去手动输入 `y` 的步骤。`python3-venv` 是创建隔离环境的关键，提前装好，避免后续为每个项目单独补充

## 块 6

### 2.4 操作系统与实训环境该怎么选

你已经熟悉 Tokenizer 和 BPE，下一步深入 Transformer 时，环境选择会直接影响你跑模型、做微调的效率。这里按你的实际场景来拆解：

- **如果你在 Windows 上写 Python、练命令**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比在 Windows 原生 CMD 里练更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。
- **如果你用 macOS 或 Linux**：直接用本机终端即可，本身就是类 Unix 环境，无需额外配置。
- **如果你要做 GPU 实训（比如 LoRA/QLoRA 微调、vLLM 部署）**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这些任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已于2021年12月31日停止维护，但 CentOS Stream

## 块 7

你已熟悉Tokenizer和BPE，现在要深入Transformer架构。在动手搭建模型前，先确保环境能顺畅安装PyTorch等依赖库。

`pip`是Python的包管理工具，默认从官方PyPI服务器（`pypi.org`）下载库，这是最权威、最完整的源，应始终作为首选。

**什么时候才需要镜像源？** 仅当网络受限、访问官方源明显变慢或频繁超时时，才临时切换。镜像只是“网络不佳时的加速手段”，不是永久替代品。

**推荐做法：临时指定镜像源**（只对单次安装生效，不改全局配置）：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样官方源仍是默认，网络恢复或镜像异常时无需清理配置。你的RTX 3060环境后续安装PyTorch时，若遇下载缓慢，即可用此方式临时加速。

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你已熟悉 Tokenizer 与 BPE 的底层逻辑，现在进入 Transformer 架构学习前，先把环境彻底备好。若你**长期处于网络受限环境**、几乎每次安装包都要用镜像，才建议把镜像写入配置作为默认源。配置后请牢记：**官方 PyPI 仍是最可靠来源**，网络恢复后建议改回或删除该配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

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
```

4. 按 `Ctrl + O` 保存，`Ctrl + X` 退出。

配置完成后，可通过 `pip config list` 查看当前配置；若需改回官方

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：你已熟悉Tokenizer和BPE，接下来要深入Transformer架构，写代码、调试模型是常态。VS Code启动快、插件丰富，免费且跨平台，后续写脚本、改配置都能复用，是性价比最高的起步工具。

**安装步骤**：
1. 访问 `https://code.visualstudio.com`，下载对应系统的安装包。
2. 运行安装程序，一路“下一步”即可。

**必装插件**：打开VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（微软官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python语言服务器，让代码提示更智能）

安装后，VS Code才能识别`.py`文件，提供代码补全、错误提示和一键运行功能——这对你后续调试Transformer代码至关重要。

**配置Python解释器**：
1. 按`Ctrl + Shift + P`（Mac是`Cmd + Shift + P`）。

## 块 10

### 4.2 方案二：PyCharm Community（免费、轻量）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置了代码分析、版本控制等功能，适合大型项目开发。需注意，社区版功能较专业版精简，数据库工具等高级功能仅专业版提供。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add launchers to the PATH`（命令行可直接启动 PyCharm，该选项可能因版本而异）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）。
3. `Interpreter` 选择系统安装的 Python 3.10+

## 块 11

### 5.1 为什么用 Conda

你已经熟悉了 Tokenizer 和 BPE，下一步要深入 Transformer，这意味着你很快会接触 PyTorch、CUDA 这类复杂依赖。Conda 能做的，不只是为每个项目创建独立的 Python 空间，它还能直接管理 Python 解释器本身的版本——这正是深度学习环境最需要的。想象一下，不同项目可能需要不同版本的 Python 和 CUDA，Conda 让这一切互不干扰，所以它几乎是深度学习必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，核心区别在“体积”和“预装”：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装 numpy、pandas、jupyter 等大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你的目标是 Transformer，环境越干净越好，需要什么装什么，避免预装库干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认，不要勾选 "Add Miniconda3 to my PATH environment variable"**，以避免与已有 Python 环境冲突。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端，conda 通过初始化脚本正确接管，而非硬加 PATH。

4. **验证安装**：打开一个**新的**终端（Windows 用 **"

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：你已熟悉 Tokenizer 和 BPE，下一步深入 Transformer 时，不同项目常需不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，

## 块 13

**5.4 激活与退出环境：为你的 Transformer 项目隔离“工作台”**

你已熟悉 Tokenizer 与 BPE，接下来要深入 Transformer 架构。在动手写代码前，先确保所有依赖都装在一个“独立房间”里，避免污染系统环境。

**进入环境**（相当于打开这个房间的门）：

```bash
conda activate llm-learn
```

激活后，命令行前会出现 `(llm-learn)` 标志。此后你执行的 `python` 和 `pip` 命令，都只作用于这个环境，不会影响系统或其他项目——这对后续调试 Transformer 模型至关重要。

**验证当前 Python 版本**：

```bash
python --version
```

应显示 `Python 3.10.x`。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

---

**本块学习检查**：执行 `conda activate llm-learn` 后，观察命令行提示符的变化；再

## 块 14

## 6.1 VS Code 配置 Conda 环境

既然你已熟悉 Tokenizer 和 BPE，接下来要深入 Transformer 架构，那代码环境必须指向 `llm-learn` 这个 conda 环境，否则你装的 PyTorch 等依赖会“找不到”。

**操作步骤**：

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入 `Python: Select Interpreter` 并回车。
4. 在列表中选择带有 `conda` 字样的 Python 路径（形如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的所有代码都会使用 `llm-learn` 环境中的 Python 解释器和已安装的包——包括你后续要用的深度学习框架。

## 6.2 PyCharm 配置 Conda 环境

如果你更习惯 PyCharm，流程同样直接：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 上是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `

## 块 15

### 在命令行验证

你已经熟悉了Tokenizer和BPE，现在进入Transformer前，先确保环境就绪。这一步是基础，但很关键。

1. **激活环境**：输入 `conda activate llm-learn`，回车。
2. **进入交互模式**：输入 `python`，回车。看到 `>>>` 提示符，说明Python已启动。
3. **运行第一行代码**：在 `>>>` 后输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
   回车后，若正确打印出这句话，说明环境正常。
4. **退出**：输入 `exit()` 回车。

**本块学习检查**：在命令行完成上述步骤，确认输出与预期一致。若打印成功，

## 块 16

**本块学习检查**：完成下方步骤后，在 PyCharm 的 Run 窗口看到输出结果，即代表你的开发工作台已成功跑通。

---

### 7.3 在 PyCharm 中验证

你已经熟悉 Tokenizer 和 BPE 算法，现在要进入 Transformer 架构的学习。在动手理解模型代码之前，先确保你的“实验台”能正常运转——这一步验证的就是环境是否真正打通。

操作很简单，三步：

1. **新建项目**：打开 PyCharm，创建一个新项目（注意：项目路径不要包含中文或空格）。
2. **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3. **运行**：在 `hello.py` 的编辑区右键点击，选择 `Run 'hello'`。

如果一切正常，PyCharm 下方的 **Run 窗口**会输出你写在文件里的内容（比如 `print("

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

### Q-KB0204-003
请写代码：读取 `data.txt` 文件，捕获文件不存在的异常，若不存在则打印 "文件未找到"。

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0205-004
请写一个生成器函数 `fibonacci(n)`，生成前 `n` 个斐波那契数。

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0204-001
以下哪种文件打开模式表示**追加写入**（不覆盖原内容）？

### Q-KB0201-002
Python中，`type(3.14)` 的返回结果是________；`type("3.14")` 的返回结果是________。
