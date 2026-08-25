# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

你的目标是 LoRA 微调，但先别急着碰模型——所有微调代码最终都要靠 Python 解释器来跑。理解它，你才知道自己每一步在干什么。

**核心问题**：你写的代码是给人看的文字，电脑只认 0 和 1。中间缺一个"翻译"。

**Python 解释器就是那个翻译官**：它把你写的 `.py` 文件里的代码，逐行翻译成电脑能执行的机器指令。没有它，你的代码文件只是一堆普通文本，电脑完全看不懂。

**为什么这对你很重要**：你之后跑 LoRA 微调脚本时，命令行里敲的 `python train.py` 就是在调用解释器去"翻译并执行"你的训练代码。如果解释器没装好，你连第一行代码都跑不起来。

**怎么验证你理解了**：打开命令行，输入 `python --version`，如果能看到版本号，说明解释器已经就位，可以开始下一步。如果提示"不是内部或外部命令"，说明解释器还没装或没配置好——这正是下一节要解决的问题。

## 块 2

**本块学习检查**：读完本节后，请合上资料，用一句话分别说出 IDE 和虚拟环境各自解决什么问题，再对照原文检查。

---

你的目标是 LoRA 微调，这需要写代码、跑实验。记事本也能写，但你会疯掉——**IDE（集成开发环境）**就是你的“工作台”，提供语法高亮、自动补全、运行调试，写代码像用 Word 一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。先装 VS Code，够用且上手快。

接下来是**虚拟环境**，这是你搞 LoRA 必须养成的习惯。想象两个项目：A 要 `numpy 1.0`，B 要 `numpy 2.0`，装一起就冲突报错。虚拟环境给每个项目一个独立 Python 空间，互不干扰。用 **conda** 管理，这是主流方案

## 块 3

### 2.1 Windows 用户：为 LoRA 微调铺路

你的目标是 LoRA 微调，这意味着后续要在命令行里跑训练脚本。所以，这一步的核心不是“装个软件”，而是**让系统认得出 Python**。下面三步，每一步都直接服务于这个目标。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`。
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这是全流程最关键的一步。这个选项会把 Python 加入系统“环境变量”。如果不勾选，你打开命令行输入 `python` 时，系统会提示“不是内部或外部命令”，后续所有命令行操作（包括跑 LoRA 脚本）都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或

## 块 4

你的目标是 LoRA 微调，这需要你亲手操作模型和代码。在开始之前，先把地基打牢——装好 Python 环境。macOS 用户看这里，我们直接进入正题。

macOS 系统自带 Python 3，但未预装 Python 2，系统自带的 `python3` 是给系统脚本用的，别直接依赖它。**强烈建议自己装最新版**，避免后续微调时版本冲突。

**方法一：官方安装包（推荐，最省心）**

1.  打开浏览器，访问 `https://www.python.org`，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2.  双击下载的文件，一路点 "Continue" → "Install" 完成安装。
3.  打开终端（Terminal），输入下面两行命令验证是否成功：

```bash
python3 --version
pip3 --version
```

看到版本号输出就对了。

**关键提醒**：在 macOS 上，命令行里**没有** `python` 命令，只有 `python3`。以后你运行任何代码、安装任何库（比如后面微调要用的 `torch`），都请用 `python3` 和 `pip3`，别用 `python`，否则会报“命令不存在”。

**方法二：

## 块 5

> 目标：为 LoRA 微调铺路，先确保 Python 环境就绪。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省事。

打开终端，依次执行：

```bash
sudo apt update          # 刷新软件源
sudo apt install python3 python3-pip python3-venv -y
python3 --version        # 确认安装成功
```

**关键点**：`-y` 让系统自动确认，免去手动输入 `y`；`python3-venv` 是虚拟环境模块，提前装好，后面跑 LoRA 项目隔离依赖时直接用。

**本块学习检查**：执行完上述命令后，运行 `python3

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，而且喜欢动手实操，那环境选对，后面能省一半力气。记住一条主线：**日常练命令用 WSL2，GPU 实训用 Ubuntu LTS**。

**如果你是 Windows 用户，日常写 Python、练命令行**：直接启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里拥有一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致，比在 Windows 原生 CMD 里练更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install` 即可。

**如果你是 macOS 或 Linux 用户**：直接用本机终端就行，本身就是类 Unix 环境，不用额外折腾。

**到了 LoRA / QLoRA 微调、vLLM 部署这类 GPU 实训**：必须用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 避坑提醒**：别把 **"CentOS + VirtualBox 虚拟机"** 当最佳方案

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

只有当你**网络受限、访问官方源明显变慢或经常超时**时，才考虑临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，如果网络长期受限、每次装包都卡在下载上，才考虑把镜像写进配置当默认源。**注意**：官方 PyPI 始终最可靠，网络恢复后建议改回或删除配置。

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

> **⚠️ 关于 `trusted-host`**：网上教程常顺手加一行 `trusted-host = ...`，**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用无需它。只有企业内网自建源或确认来源可信时才临时加。

**macOS / Linux 用户：**
1. 打开终端，执行：
```bash
mkdir -p ~/.config/pip
```
2. 创建配置文件：
```bash
nano ~/.config/pip/pip.conf
```
3. 写入同样内容：
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```
4. 按 `Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置是否生效
命令行输入：
```bash
pip config list
```
看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即配置成功。

### 3.5 改回官方源
删除上面创建的 `pip.ini` / `pip.conf` 文件，或执行：
```bash
pip config unset global.index-url
```

---

**本块学习检查**：在你自己的终端里执行 `pip config list`，确认输出中包含清华镜像地址。若没有，检查文件路径和内容是否与上文

## 块 9

**VS Code：你的轻量级 Python 工作台**

你的目标是 LoRA 微调，这需要大量编写和调试 Python 代码。VS Code 正是为此而生的免费工具——启动快、插件丰富，后续写脚本、改配置都能用上。

**三步装好：**

1. 打开 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路“下一步”即可。
3. **装两个必装插件**：点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（微软官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更智能）

装完 Python 插件后，VS Code 才能识别 `.py` 文件，给你代码补全、错误提示和一键运行功能。

**最后一步——指定解释器：**

1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。内置代码分析、版本控制、数据库工具，适合大型项目开发——你的 LoRA 微调项目会涉及多文件与依赖管理，它能让代码结构一目了然。

**安装步骤**（跟着做，5分钟搞定）：
1. 打开 https://www.jetbrains.com/pycharm/download/
2. 下载 **Community Edition**（社区版，免费）
3. 运行安装程序，务必勾选：
   - `Add "Open Folder as Project"`（右键文件夹即可打开项目）
   - `Add to PATH`（命令行直接敲 `pycharm` 启动）

**首次启动配置**（关键一步）：
1. 打开 PyCharm，选 `New Project`
2. `Location` 选项目路径——**路径中不要含

## 块 11

### 5.1 为什么用 Conda

你的目标是 LoRA 微调，这需要 PyTorch 等深度学习依赖。Conda 能为你每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器。对深度学习来说，它是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用空间通常超过 3GB），预装大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 50-100MB，安装后占用空间约 400MB 以上），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，环境更干净，后续需要什么装什么。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，默认不勾选 "Add Miniconda3 to my PATH"（若希望在任何终端使用 conda，可勾选；否则使用 Anaconda Prompt）。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择路径，最后一步选 `yes` 执行 **`conda init`**，让 conda 通过初始化脚本正确接管。
4. **验证安装**：打开**新的**终端（Windows 用 "Anaconda Prompt (Miniconda3)"；macOS/Linux 重开终端），输入：

```bash
conda --version
```

显示类似 `conda 24.x.x` 即成功。

---

**本块学习检查**：打开终端，运行 `conda --version`，

## 块 12

**动手做：创建你的第一个虚拟环境**

你的目标是 LoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA，所以现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y` 并回车，等待安装

## 块 13

### 5.4 激活与退出环境

**激活环境**——相当于走进你的专属“独立房间”，避免干扰系统或其他项目：

```bash
conda activate llm-learn
```

激活后，命令行前会出现 `(llm-learn)` 标志，表示后续 `python` 和 `pip` 命令都作用于这个环境。

**验证当前 Python 版本**：

```bash
python --version
```

应显示 `Python 3.10.x`。

**退出环境**，回到系统默认：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**列出所有环境**：

```bash
conda env list
```

带 `*` 的是当前激活的环境。

**删除环境**（如需清理）：

```bash
conda remove -n llm-learn --all
```

---

## 第六章：在 VS Code / PyCharm 中使用 Conda 环境

> **本块学习检查**：在终端依次

## 块 14

### 6.1 VS Code 配置 Conda 环境（动手做）

你的目标是 LoRA 微调，环境是第一步。跟着做，别跳步：

1. **激活环境**：在终端输入 `conda activate llm-learn`。看到命令行前缀变成 `(llm-learn)` 就对了。
2. **打开 VS Code**，按 `Ctrl + Shift + P` 打开命令面板。
3. **输入并选择** `Python: Select Interpreter`。
4. **选对解释器**：在列表里找带有 `conda` 字样的项，例如 `Python 3.10.x ('llm-learn': conda)`。**别选错**——选错的话，你写的代码会用错 Python 和包。

**为什么**：VS Code 能自动识别 conda 环境。选好后，这个工作区里的代码就跑在 `llm-learn` 环境里，你为微调装的包（比如 PyTorch）才能被 import 到。

---

### 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程稍长但一样直接：

1. 打开 PyCharm，点 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧找 `Project:

## 块 15

### 在命令行验证你的 Python 环境

既然你的目标是 LoRA 微调，第一步就是确保 Python 环境能跑起来。这里用最直接的方式验证。

1. **激活环境**：在终端输入 `conda activate llm-learn`，回车。
2. **进入 Python**：输入 `python`，回车。看到 `>>>` 提示符，说明进入了交互模式。
3. **运行代码**：在 `>>>` 后输入 `print("Hello, Python! 我的环境准备好了！")`，回车。
4. **确认输出**：屏幕打印出这句话，说明环境正常。输入 `exit()` 退出。

**本块学习检查**：在 `>>>` 后输入 `print(2+3)`，确认输出 `5`。

## 块 16

**本块学习检查**：在 PyCharm 中完成 `hello.py` 的运行验证，并对照第八章排查表，记录你遇到的一个问题及对应解决方案（若未遇到问题，则记录排查表中你最可能遇到的一项）。

---

### 7.3 在 PyCharm 中验证

你的目标是 LoRA 微调，这意味着你马上要面对真实的模型训练代码。在跑那些复杂脚本之前，先用一个最简单的文件确认你的“工作台”完全打通——这一步能省下你后面排查环境问题的无数时间。

**操作步骤（跟着做，2分钟搞定）：**

1. **新建项目**：打开 PyCharm，创建一个新项目（项目路径**不要包含中文或空格**，这是本章小结里的好习惯）。
2. **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3. **写入代码**：在文件中输入 `print("Hello, PyCharm!")`（这是验证环境的标准动作，代码本身很简单，重点看运行流程）。
4. **运行**：在 `hello.py` 文件内**右键点击**，选择 **`Run 'hello'`**。
5. **查看结果**：观察 PyCharm 下方的 **Run 窗口**，你应该能看到输出结果。

**关键点**：如果 Run 窗口没有输出，或者报错，**不要慌**——直接看下面的排查表，90% 的环境问题都能在这里找到答案。

---

### 第八章：常见问题排查（速查表）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows：确保已安装 Anaconda 或 Miniconda，并在安装时勾选 Add to PATH；若未勾选，可手动将 conda 的 Scripts
