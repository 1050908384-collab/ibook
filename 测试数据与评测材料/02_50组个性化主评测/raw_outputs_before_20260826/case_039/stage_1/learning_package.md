# 学习包

## 块 1

**本块学习检查**：读完本节后，请合上资料，用自己的话向自己解释“为什么没有解释器，`.py` 文件就只是一堆文本？”能说清楚，再继续往下装。

---

## 1.1 Python 解释器：程序的“翻译官”

你马上要动手装东西了，但先别急着点“下一步”。花 2 分钟搞懂你装的到底是什么，后面每一步你都会心里有数。

**核心问题**：你写的 Python 代码是给人看的文字，但电脑只认 0 和 1。这两者之间怎么打通？

**答案就是解释器**。它的角色是“翻译官”：把你写的代码逐句翻译成电脑能直接执行的指令。

**关键认知**：没有这个解释器，你辛辛苦苦写的 `.py` 文件，在电脑眼里就是一堆普通的文本，毫无执行能力。

**对你意味着什么**：你后面做 LoRA 微调，写的每一行训练脚本、数据处理代码，最终都要靠这个“翻译官”去驱动电脑真正跑起来。所以，装好它，是你整个动手实操的**第一步地基**。

现在，带着“我在装翻译官”这个清晰认知，再去进行下一步安装操作，你就不是在机械点鼠标，而是在搭建自己的开发工作台了。

## 块 2

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，这需要一套趁手的工具。别在环境搭建上浪费太多时间，我们快速过一遍核心。

**1.2 代码编辑器（IDE）：你的"工作台"**

记事本也能写代码，但效率极低。**IDE（集成开发环境）** 提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。我们推荐 **VS Code**，它是常用的轻量级编辑器，功能丰富，适合大多数开发场景。

**1.3 虚拟环境：项目的"独立房间"**

这是关键。想象你同时在做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `

## 块 3

## 2.1 Windows 用户：安装 Python 工作台

你的目标是 LoRA 微调，这需要 Python 环境作为地基。下面三步走，每一步都配有“为什么”，让你不仅会装，更懂装。

### 第一步：下载安装包

1. 打开浏览器，访问 `https://www.python.org`
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

### 第二步：运行安装程序（⚠️ 关键步骤）

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   **为什么**：这个选项把 Python 加入系统“环境变量”。不勾选的话，打开命令行输入 `python`，系统会提示“不是内部或外部命令”，后续所有命令行操作都无法进行——LoRA 微调中你要用命令行装库、跑脚本，这一步省不得。

2. 点击 **"Install Now"**（默认安装）即可。  
   **为什么**：默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python313`（对应 Python 3.13），不含中文和特殊符号，避免后续路径报错。进阶用户想改路径，务必

## 块 4

## 2.2 macOS 用户：装好 Python，为 LoRA 微调铺路

你的目标是 LoRA 微调，这需要稳定的 Python 环境。macOS 上别依赖系统自带版本——macOS 12.3 起不再预装 Python 2，且系统自带的 `python3` 仅供系统脚本使用。**强烈建议自行安装最新版**，避免后续装 PyTorch 等库时踩坑。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 完成。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：现代 macOS 命令行默认**不提供** `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

已装 Homebrew 的话，终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

**本块学习检查**：打开终端，依次运行 `

## 块 5

**动手装好 Python，为 LoRA 微调铺路**  
你的目标是跑 LoRA 微调，但先别急着碰模型——Linux 上自带的 Python 可能太老，装不对后面全卡壳。跟着敲，三步搞定：

```bash
sudo apt update          # 刷新软件源，让系统知道最新版本
sudo apt install python3 python3-pip python3-venv -y
python3 --version        # 确认装好了
```

**关键点**：`-y` 是自动确认，不用你手动按 `y`；`python3-venv` 顺手装上，因为后面建虚拟环境（隔离项目依赖）必须用它，省得再补。

**本块学习检查

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选择直接决定你后面会不会卡壳。按阶段分，就两条路：

**日常写 Python、练命令行**：如果你是 Windows 用户，**别在 CMD 里硬练**，直接启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实服务器的 Linux 环境，命令、路径、包管理都和后续云端操作一致。安装就一步：管理员 PowerShell 执行 `wsl --install`。macOS 或 Linux 用户不用折腾，本机终端就是类 Unix 环境，直接用。

**LoRA / QLoRA 微调、vLLM 部署这类 GPU 实训**：必须用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。原因很直接：这些任务依赖 NVIDIA 驱动、CUDA 和显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别踩坑**：**别把 "CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已停止维护（CentOS Stream 8 维护至202

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、最应保留的默认源**。

**什么时候才需要镜像源？** 只有当你处在网络受限环境、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。记住：**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么

## 块 8

## 3.3 进阶：把镜像设为默认——仅限长期网络受限时

这一步**不是必做项**。只有当你长期处在网络受限环境、几乎每次安装包都要靠镜像才能下载时，才值得把镜像写进配置文件作为默认源。配置前先记住一句话：**官方 PyPI 永远是最可靠的来源**，网络恢复正常后，建议改回或删除这个配置。

---

### Windows 用户操作步骤

1. 按 `Win + R`，输入 `%APPDATA%` 并回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在这个文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host` 的警告**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

---

### macOS / Linux 用户操作步骤

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

---

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤：**

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器：**

1. 打开 VS Code，按 `Ctrl + Shift + P

## 块 10

**本块学习检查**：安装完成后，新建一个项目，确认右下角显示你选择的 Python 3.10+ 解释器，并在终端输入 `python --version` 验证版本。

---

PyCharm Community 是 JetBrains 出品的免费 Python 专用 IDE，内置代码分析、版本控制等功能，适合大型项目开发。它采用 Apache 2.0 许可证，允许商业使用，但需遵守 JetBrains 商标政策。

**安装步骤：**

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **Community Edition**（社区版，免费）
3. 运行安装程序，务必勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置：**

1. 打开 PyCharm，选择 `New Project`
2. 在 `Location` 设置项目路径（**路径不要含中文**）
3. `Interpreter` 选 `Previously configured interpreter`，然后选择系统已装的 Python 3.10+

**动手验证**：

## 块 11

## 5.1 为什么用 Conda

你要做 LoRA 微调，第一步不是写代码，而是先把“工地”搭好。前面说过，虚拟环境能让每个项目拥有独立的 Python 空间——这就像给每个项目一间单独的房间，互不干扰。而 **Conda** 是目前最主流的虚拟环境管理工具，它的厉害之处在于：不仅能创建虚拟环境，还能直接管理不同版本的 Python 解释器本身。

为什么这对你特别重要？因为深度学习（PyTorch、CUDA 这些复杂依赖）对环境的要求非常苛刻——版本不对、路径冲突，一折腾就是半天。Conda 几乎是这个领域的必备工具，它能帮你把这些坑提前填平。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，直接看表：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 800MB 以上，安装后占用空间更大），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐你选 Miniconda**。理由很直接：你要做 LoRA 微调，环境越干净越好，需要什么装什么，不给自己留一堆用不上的“行李”。

**安装步骤（跟着做就行）：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包：Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`
3. 运行安装程序，注意区分系统：

   - **Windows**：一路下一步，但**关键一步**——不要勾选 "Add Miniconda3 to my PATH environment variable"（安装器也标注了“不推荐”）。为什么？把 Miniconda 硬塞进系统 PATH 容易和已有的 Python/工具冲突，导致 `python`

## 块 12

**目标**：为 LoRA 微调项目打好隔离环境，避免依赖冲突。

**动手做**：在终端输入下面这行命令，创建名为 `llm-learn` 的环境，并指定 Python 3.10：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（记住这个套路，以后创建任何环境都通用）：
- `conda create`：创建环境的固定开头
- `-n llm-learn`：给环境起名，`-n` 就是 `name` 的缩写
- `python=3.10`：指定环境内的 Python 版本

**执行后**：终端会列出待安装的包，并提示 `Proceed ([y]/n)?`，输入 `y

## 块 13

## 5.4 激活与退出环境——为 LoRA 微调开好“独立房间”

你的目标是 LoRA 微调，这意味着你会在 `llm-learn` 这个环境里安装 PyTorch、Transformers 等依赖。现在，我们把这个环境“激活”，让所有命令都只作用在它里面，不污染系统。

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

**查看所有已

## 块 14

**目标导向**：你要做 LoRA 微调，但先别急着碰模型——环境不通，后面全是坑。这一章就是让你 10 分钟内把 VS Code 和 PyCharm 都指向 `llm-learn` 环境，并验证能跑通。

**VS Code 配置（推荐你先做这个）**

1. 打开终端，先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板
3. 输入并选择 `Python: Select Interpreter`
4. 在列表里选带 `conda` 字样的项，形如：`Python 3.10.x ('llm-learn': conda)`

选完后，VS Code 会自动记住：这个工作区里所有 Python 代码，都用 `llm-learn` 里的解释器和已装好的包。你之后写 LoRA 脚本时，`import torch` 才不会报错。

**PyCharm 配置（备用方案）**

1. 打开 PyCharm，点 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）
2. 左侧展开 `Project: 你的项目名`，点 `Python Interpreter`
3. 点齿轮图标 → `Add...`
4. 左侧选 `Conda

## 块 15

### 7.1 在命令行验证

既然你的目标是LoRA微调，环境验证是第一步。跟着做，快速确认一切就绪：

1. **激活环境**：输入 `conda activate llm-learn`，回车。
2. **进入Python**：输入 `python`，回车，看到 `>>>` 提示符就对了。
3. **测试输出**：输入 `print("Hello, Python! 我的环境准备好了！")`，回车。
4. **退出**：看到文字打印后，输入 `exit()` 回车。

**本块学习检查**：完成上述步骤后，在命令行再输入一次 `python`，确认能再次进入 `>>>`，然后退出。这确保环境可重复使用。

## 块 16

好的，让我们直接开始动手。你的目标是 LoRA 微调，那第一步就是确保你的“工作台”能跑起来。这一块，我们就在 PyCharm 里验证环境是否畅通。

### 7.3 在 PyCharm 中验证

1.  **新建项目**：打开 PyCharm，新建一个项目。在项目里创建一个新的 Python 文件，命名为 `hello.py`。
2.  **运行代码**：在 `hello.py` 里输入 `print("Hello, LoRA!")`。然后，在文件编辑区**右键点击**，选择 **`Run 'hello'`**。
3.  **查看结果**：PyCharm 下方会弹出 **Run** 窗口，你应该能看到程序输出的结果。看到输出，就说明你的开发环境已经
