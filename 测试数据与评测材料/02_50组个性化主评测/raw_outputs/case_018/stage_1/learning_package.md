# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你写下的 Python 代码，是给人看的“文字”；但电脑只听得懂 0 和 1 的机器指令。**Python 解释器**，就是夹在中间的“翻译官”——它把你写的代码逐行翻译成电脑能执行的指令。

**关键认知**：没有解释器，你写的 `.py` 文件就只是一堆普通文本，电脑根本不会“运行”它。

**动手验证（30秒）**：在桌面上新建一个文本文件，重命名为 `test.py`，用记事本输入 `print("hello")` 并保存。双击它——什么都不会发生。这就是“没有翻译官”的状态。等本章环境搭好后，我们再回来让这行字真正“跑”起来。

---

**本块学习检查**：用自己的话回答——Python 解释器在“代码”和“电脑”之间扮演什么角色？如果电脑里没有它，你写的 `.py` 文件会怎样？

## 块 2

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，这需要你亲手操作大量代码。所以，你的“工作台”必须顺手。

**1.2 代码编辑器（IDE）：你的“工作台”**

记事本也能写代码，但效率极低。**IDE（集成开发环境）** 提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐 **VS Code**（轻量免费），它足够强大且启动快，非常适合你这种需要快速迭代的微调实验。

**1.3 虚拟环境：项目的“独立房间”**

想象你同时在做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要 Python 环境。Windows 上安装很简单，跟着做，10 分钟搞定。

> **说明**：3.10+ 表示 3.10 及以上版本，本教程以 3.13 为例，所有操作同样适用于 3.10 及更高版本。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**

## 块 4

## 2.2 macOS 用户：装好 Python，为 LoRA 微调铺路

你的目标是 LoRA 微调，这需要 Python 环境。macOS 上有个坑：**macOS 12.3 起系统不再预装 Python 2**，且系统自带的 `python3` 是给系统脚本用的。**强烈建议自己装最新版**，别依赖系统自带版本。

**方法一：官方安装包（推荐新手，最省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 即可。
3. 打开终端（Terminal），输入下面两条命令验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`？** 现代 macOS 默认**没有** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统与官方安装包统一用 `python3`，所以请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已装 Homebrew（macOS 包管理器），终端执行：

```bash
brew

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是 LoRA 微调，第一步是把环境弄稳。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省心。

打开终端，依次执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：`sudo apt update` 先刷新软件源，确保能装到最新版。第二条命令一口气装好 Python 3、包管理器 pip 和虚拟环境模块 venv——`-y` 表示自动确认，不用你手动按 `y`。装 venv 是为了后面隔离项目依赖，避免

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，那就直接奔着 GPU 实训去。选环境分两步走：

**第一步：日常练手（写 Python、敲命令）**
- **Windows 用户**：别用 CMD 练命令，直接启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实服务器的 Linux 环境，命令、路径、包管理都和后续云端一致。管理员 PowerShell 里跑一句 `wsl --install` 就装好了。
- **macOS / Linux 用户**：不用折腾，本机终端就是类 Unix 环境，直接用。

**第二步：LoRA / QLoRA 微调实训**
- 用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器**，或**本地实体 Ubuntu LTS 机器**。这类任务依赖 NVIDIA 驱动、CUDA 和显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 别踩坑**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于2021年停止维护，CentOS 7 也于2024年停止维护，建议使用

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、应保留的默认源**。

**什么时候才用镜像？** 只有当你处在网络受限环境、访问官方源明显变慢或经常超时时，才临时切换。记住：**镜像是"网络不好时的临时加速手段"，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正在为 LoRA 微调准备环境，而微调过程中要反复下载模型和依赖包。如果你**长期处于网络受限环境**、几乎每次都要用镜像，才考虑把它写入配置作为默认源。配置后请记住：**官方 PyPI 仍是最可靠的来源**，网络恢复后建议改回或删除该配置。

### Windows 用户操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车。这会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

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

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，要频繁写 Python 脚本、改配置文件，它都能胜任。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

装完这两个插件，VS Code 才能识别 `.py` 文件，给你代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择你刚安装的 Python 版本

## 块 10

**本块学习检查**：完成安装后，打开 PyCharm 并新建一个项目，确认你能看到代码编辑区，且右下角显示你选择的 Python 3.10+ 解释器。

---

**4.2 方案二：PyCharm Community（专业、开箱即用）**

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。内置代码分析、版本控制、数据库工具，适合大型项目开发——这正是你后续做 LoRA 微调时需要的。

**安装步骤**（跟着做，很快）：

1. 打开 https://www.jetbrains.com/pycharm/download/
2. 下载 **Community Edition**（社区版，免费）
3. 运行安装程序，务必勾选：
   - `Add "Open Folder as Project"`（右键快速打开项目）
   - `Add to PATH`（勾选后可在命令行中直接使用 pycharm 命令启动，若不勾选也可正常使用

## 块 11

好的，收到你的学习需求。为了让你能顺利开始 LoRA 微调，我们首先要搭建一个干净、可控的“工作台”。Conda 就是管理这个工作台的核心工具，它能确保不同项目互不干扰，这是后续安装 PyTorch 等深度学习库的基石。

## 5.1 为什么用 Conda

简单来说，虚拟环境能为每个项目创建独立的 Python 空间，而 **Conda** 是目前最主流的虚拟环境管理工具。它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于你接下来要进行的 LoRA 微调（需要 PyTorch、CUDA 等复杂依赖），Conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

安装 Conda 有两个选择，我们推荐 **Miniconda**。

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包。

**安装步骤：**

1.  访问 <https://docs.conda.io/en/latest/miniconda.html>。
2.  下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3.  运行安装程序，**注意系统差异**：
    *   **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (

## 块 12

**本块学习检查**：运行创建命令后，在终端输入 `y` 并回车，观察安装过程是否正常结束。

---

你的目标是 LoRA 微调，而微调不同模型时，常需要不同版本的 PyTorch 和 CUDA。现在学会创建虚拟环境，以后就能直接复用这套流程，避免环境冲突。

在终端中执行以下命令，创建一个名为 `llm-learn` 的环境，并指定 Python 3.10：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（帮你快速记住）：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写

## 块 13

### 5.4 激活与退出环境（你的“项目隔离”开关）

你马上要做 LoRA 微调，最怕的就是把不同项目的 Python 包搞混。Conda 环境就是你的“独立房间”，激活它，所有操作都在里面进行。

**进入房间**（激活环境）：

```bash
conda activate llm-learn
```

**怎么确认进来了**：命令行前面出现 `(llm-learn)` 标志。之后你敲的 `python` 和 `pip` 命令，都只作用于这个环境，不会污染系统或其他项目。

**快速验证**（确保版本正确）：

```bash
python --version
```

应显示 `Python 3.10.x`。

**离开房间**（回到系统默认环境）：

```bash
conda deactivate
```

---

### 5.5 查看和管理环境

**查看所有环境**（带 `*` 的是当前激活的）：

```bash
conda env list
```

**

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，动手跑代码前，先把工作台对准 `llm-learn` 这个环境。操作很简单：

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`
3. 输入 `Python: Select Interpreter`
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）

选好后，VS Code 会自动记住这个工作区用的解释器，你写的代码就会调用 `llm-learn` 里的 Python 和已装好的包，不会跑错环境。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程也很快：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`
3. 点击齿轮图标 → `Add...`
4. 左侧选择 `Conda Environment` → `Existing environment`
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件，可用 `$CONDA_PREFIX/envs/llm-learn/bin/python` 定位（Windows 对应路径为 `...\envs\llm-learn

## 块 15

**本块学习检查**：完成下方两步操作后，在终端看到输出即通过。

---

你正在为 LoRA 微调搭环境，先确认 Python 能跑起来。跟着做，别跳过。

**第一步：命令行验证**

1. 打开终端，输入 `conda activate llm-learn` 并回车，激活你的环境。
2. 输入 `python` 回车，你会看到 `>>>` 提示符。
3. 在 `>>>` 后输入 `print("Hello, Python! 我的环境准备好了！")`，回车。
4. 看到这行文字打印出来，就输入 `exit()` 退出。

**第二步：VS Code 验证**

1. 新建文件夹 `my-first-project`，用 VS

## 块 16

## 7.3 在 PyCharm 中验证——跑通你的第一行代码

你的目标是 LoRA 微调，而微调的第一步，是确保你的“工作台”能跑起来。这一节我们用最直接的方式验证：**写一行代码，让它运行，看到输出**。

**操作步骤（跟着做，3 分钟搞定）：**

1. 打开 PyCharm，新建一个项目（Project）。
2. 在项目中创建一个新文件，命名为 `hello.py`。
3. 在文件里输入一行代码：`print("Hello, LoRA!")`（这是你微调之路的“开机自检”）。
4. 在文件编辑区**右键点击**，选择 `Run 'hello'`。
5. 看 PyCharm **下方**的 Run 窗口——如果出现了 `Hello, LoRA!`，恭喜，你的开发环境完全打通了。

**为什么这一步重要？** 微调 LoRA 时，你会反复执行“写代码 → 运行 → 看输出”这个循环。现在跑通最小闭环，后面所有操作都建立在这个基础上。

---

## 第八章：常见问题排查（遇到再看，没遇到跳过）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python
