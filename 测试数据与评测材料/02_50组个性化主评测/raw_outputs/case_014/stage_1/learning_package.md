# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

**别急着下载，先花2分钟搞懂这个核心概念。**

你的目标是跑通DeepSpeed分布式训练，但无论多高级的框架，第一步都是让电脑能读懂你写的Python代码。你写的代码是人类能看懂的文字，但电脑只认0和1。

**Python解释器**就是那个"翻译官"——它把你的代码翻译成电脑能执行的指令。没有它，你写的`.py`文件只是一堆普通文本，电脑根本不知道你想干什么。

**为什么这对你特别重要？**

DeepSpeed的训练脚本本质上是`.py`文件。如果你不理解解释器的作用，后面遇到"为什么我的代码跑不起来"时，你会分不清是代码问题、环境问题，还是解释器配置问题。现在花2分钟理解这个角色，后面每一步你都知道自己在做什么。

**动手验证一下（30秒）：**

打开你的终端（命令行窗口），输入：

```bash
python --version
```

如果能看到类似`Python 3.x.x`的版本号，说明你的电脑里已经有"翻译官"待命了。如果提示找不到命令，别慌——这正是下一节要解决的问题。

**记住这个比喻：** 解释器是翻译官，你的代码是演讲稿，电脑是听众。没有翻译官，演讲稿再精彩，听众也听不懂。

## 块 2

**本块学习检查**：打开 VS Code 或 PyCharm，新建一个 Python 文件，输入 `print("hello")`，观察语法高亮是否出现；再尝试删除一个括号，看编辑器是否给出错误提示。

---

你的目标是 DeepSpeed 分布式训练，这需要你熟练操作代码和调试环境。记事本写代码效率太低，**IDE（集成开发环境）** 才是你的"工作台"——它提供语法高亮、自动补全、运行调试，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费，适合快速上手）和 **PyCharm**（专业强大，适合复杂项目）。

装好 IDE 后，还要解决"环境冲突"问题。想象你同时做两个项目：A 需要 `numpy 1.0`，B 需要 `numpy 2.0`，都装进同一个 Python 会报错。**虚拟环境**就是为

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是跑 DeepSpeed 分布式训练，但一切从“能运行 Python”开始。下面三步，照着做就行。

**第一步：下载安装包**

打开浏览器，访问 https://www.python.org，鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的 Windows 版本（通常为最新稳定版，如 Python 3.12 或 3.13）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这是重中之重。不勾选的话，你打开命令行输入 `python`，系统会提示“不是内部或外部命令”，后续所有命令行操作（包括装 DeepSpeed）都做不了。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入 `python --version` 并回车。如果显示 `Python 3

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

现代 macOS（12.3 起）不再预装 Python 2，系统自带的 `python3` 仅供内部脚本使用。为了你的 DeepSpeed 分布式训练目标，**务必自行安装最新版 Python**，避免依赖系统版本。

**方法一：官方安装包（推荐，新手友好）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端，验证安装：

```bash
python3 --version
pip3 --version
```

**关键提醒**：官方安装包默认安装 `python3`，不创建 `python` 符号链接，因此请始终用 `python3` / `pip3`，避免“命令不存在”或误用系统版本。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

终端执行：

```bash
brew install python
```

安装后同样用 `python3 --version` 验证。

**本块学习检查**：在终端运行 `python3 --version`，确认输出为 3.10 或更高版本号。若低于此版本，请重新下载官方安装包覆盖安装。

## 块 5

**动手装 Python：Ubuntu/Debian 三步走**

你的目标是跑 DeepSpeed，但先得有 Python 环境。Linux 通常自带 Python，但版本可能太老，所以咱们直接装新的。

```bash
# 1. 更新软件源，让系统知道最新软件包
sudo apt update

# 2. 安装 Python 3.10、pip 和虚拟环境模块
sudo apt install python3.10 python3.10-venv python3-pip -y

# 3. 验证安装成功
python3 --version
```

**关键点**：`-y` 是自动确认，不用手动按 `y`；`python3.10-venv` 是建虚拟环境用的，

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是跑 DeepSpeed 分布式训练，那环境选错一步，后面全是坑。按你的动手习惯，直接照着下面两条路走：

**1. 日常练命令、写 Python（Windows 用户）**

别在 CMD 里练了，直接上 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实服务器的 Linux 环境，命令、路径、包管理方式都和以后上服务器时一模一样。装起来也快，管理员 PowerShell 里跑一句 `wsl --install` 就行。

**2. 跑 GPU 实训（LoRA/QLoRA 微调、vLLM 部署、DeepSpeed 训练）**

这类任务必须用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。原因很直接：DeepSpeed 这类分布式训练依赖 NVIDIA 驱动、CUDA 和显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全，遇到问题搜得到答案。

**⚠️ 别踩的坑：** 别把 **"CentOS + VirtualBox 虚拟机"** 当主力方案。CentOS 8 已于202

## 块 7

**本块学习检查**：看完后，请你在终端执行一次临时镜像安装（比如 `pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`），观察输出中是否出现 `tuna` 地址，确认只影响这一次安装。

---

`pip` 是 Python 装第三方库（如 `numpy`）的工具。默认从官方 PyPI（`pypi.org`）下载——这是**最权威、最完整、应保留的默认源**。

**什么时候才需要镜像源？** 仅当网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像只是“网络不好时的加速手段”，**不是永久替换**，官方源始终首选。

**首选做法：临时指定镜像（推荐）**。只对某一次安装临时指定，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

**先判断一下**：你是不是几乎每次装包都得手动加 `-i` 镜像参数？如果是，才值得做这一步。如果只是偶尔慢，跳过本节，别给自己添麻烦。

**重要提醒**：配置写好后，pip 会默认走镜像源。但官方 PyPI 始终是最可靠的，等网络恢复正常，记得改回官方源（见 3.5）。

---

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%` 回车，会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在这个文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**这行配置的作用**：`index-url` 告诉 pip 默认从这个地址下载包，以后不用每次手动指定。

---

### macOS / Linux 操作步骤

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

### ⚠️ 关于 `trusted-host`，别乱加

网上很多教程会顺手让你加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——加上它，pip 会跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**不需要**。只有企业内网自建源、或确实遇到证书报错且确认来源可信时，才临时加。

---

## 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

## 块 9

**目标对齐**：你正在为 DeepSpeed 分布式训练铺路。VS Code 是你写训练脚本、改配置文件的“主战场”——它轻量、免费，启动快，插件生态强，后续调 Linux 命令、看日志都用得上。

**动手装（3 步）**：
1. 打开 https://code.visualstudio.com，下载你系统的安装包。
2. 运行安装程序，一路“下一步”到底。
3. **装插件**：点左侧最下方四个方块图标（扩展），搜并装这两个：
   - **Python**（微软官方，语法高亮、调试、智能提示）
   - **Pylance**（语言服务器，让代码提示更聪明）

装完 Python 插件，VS Code 才认 `.py` 文件，才能给你补全、报错、一键运行。

**选解释器（关键一步）**：
1. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选你刚装的 Python 版本（如 `Python 3.13.x`）。

**本

## 块 10

**本块学习检查**：完成安装后，打开 PyCharm 并新建一个项目，确认你能找到“Interpreter”设置项并看到系统 Python 3.10+ 的选项。

---

你的目标是 DeepSpeed 分布式训练，这意味着项目结构会越来越复杂。PyCharm Community 正是为此设计的——它免费、专业，内置代码分析和版本控制，能帮你管理大型项目。

**安装要点**（3 步）：
1. 访问 JetBrains 官网下载页，选 **Community Edition**（社区版）。
2. 运行安装程序，务必勾选两项：`Add "Open Folder as Project"`（右键快速打开项目）和 `Add to PATH`（命令行直接启动）。
3. 首次启动选 `New Project`，**路径不要含中文**，`Interpreter` 选 `Previously configured interpreter`，指向系统 Python 3.10+。

**实操提示**：按你 fast 的学习节奏，直接动手装，装完立刻建个项目验证。路径含中文会导致后续 DeepSpeed 环境报错，这是最常见的坑，一次避开。

## 块 11

### 5.1 为什么用 Conda

你的目标是 DeepSpeed 分布式训练，这意味着你马上要面对 PyTorch、CUDA 这一大堆复杂依赖。Conda 就是为这个场景准备的：它不仅能创建虚拟环境，还能直接管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，看表：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 1GB+），预装数百个常用科学计算包（如 numpy、pandas、jupyter、scipy、matplotlib 等） | 做数据分析、不想手动装包 |
| **Miniconda** | 体积小（约 50-100MB），只含 conda 和基础 Python | 希望环境干净、按需装包 |

**推荐 Miniconda**。你后面要装的是 PyTorch、CUDA 这类特定包，环境越干净越好，避免无关库干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认，不要勾选 "Add Miniconda3 to my PATH environment variable"**。不勾选 PATH 是为了避免与已有 Python 冲突，导致 `python`、`pip` 指向混乱；但若希望在任何终端使用 conda，可勾选该选项，或手动将 conda 添加到 PATH。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 操作 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选路径，最后一步输入 `yes` 执行 **`conda init`**。这样终端每次打开时 conda 通过初始化脚本接管，而不是硬加 PATH。若 macOS 使用 `.pkg` 安装

## 块 12

**本块学习检查**：在终端执行 `conda create -n llm-learn python=3.10`，看到 `Proceed ([y]/n)?` 后输入 `y` 回车，确认环境创建成功。

---

你的目标是 DeepSpeed 分布式训练，但先别急着碰它——DeepSpeed 依赖特定版本的 PyTorch 和 CUDA，不同项目要求不同。现在学会用 Conda 隔离环境，以后每个项目都能“各用各的”，互不干扰。

**动手做**：打开终端，逐字输入下面这行命令：

```bash
conda create -n llm-learn python=3.10
```

**拆开看**：
- `conda create`：告诉 Conda“我要新建一个环境”
- `-n ll

## 块 13

**本块学习检查**：打开终端，依次执行 `conda activate llm-learn`、`python --version`、`conda deactivate`，观察每一步命令行前缀的变化，并确认 Python 版本显示为 3.10.x。

---

你的目标是 DeepSpeed 分布式训练，但先别急着碰模型——训练脚本跑在哪个环境里，决定了你能不能顺利调起多卡。Conda 环境就是你的“独立工作间”，激活它，你敲的所有 `python` 和 `pip` 命令都只在这个房间里生效，不会弄乱系统或其他项目。

**进入工作间**（激活环境）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志，这就是你“在场”的证明。**立刻验证**：

```bash
python --version
```

应该显示 `Python 3.10.x`。这个版本是后续 PyTorch 和 DeepSpeed 安装的

## 块 14

**本块学习检查**：完成配置后，在 VS Code 中新建文件输入 `print("hello")` 并运行，确认输出结果来自 `llm-learn` 环境。

---

## 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS

## 块 15

**个人化教学（动作：explain）**

你的目标是 DeepSpeed 分布式训练，但任何分布式任务都从“环境能跑通”开始。本块用最快路径验证你的 Python 工作台是否就绪，为后续多卡训练铺路。

**命令行验证（最快路径）**
1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 回车，看到 `>>>` 即进入交互模式。
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 并回车。
4. 看到文字打印即成功，输入 `exit()` 退出。

**VS Code 验证（日常开发用）**
1. 新建文件夹 `my-first-project` 并用 VS

## 块 16

### 7.3 在 PyCharm 中验证——动手跑通第一行代码

你的目标是 DeepSpeed 分布式训练，但一切大型模型训练都始于“代码能跑起来”这个地基。这一节我们用最快的方式，在 PyCharm 里验证你的环境是否真正打通。

**操作步骤（跟着做，2分钟搞定）：**

1. **新建项目**：打开 PyCharm，创建一个新项目（项目路径不要包含中文或空格）。
2. **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3. **写入代码**：在文件中输入 `print("Hello, DeepSpeed!")`（这是验证环境用的简单输出，不涉及分布式训练本身）。
4. **运行**：在 `hello.py` 文件上**右键点击** → 选择 **`Run 'hello'`**。
5. **查看结果**：看 PyCharm 下方的 **Run 窗口**，如果输出了 `Hello, DeepSpeed!`，说明你的 Python 解释器、项目配置、运行链路全部正常。

> **为什么这一步对你重要**：DeepSpeed 是在 PyTorch 之上的分布式训练框架，如果连最基本的“运行一个文件”都报错，后续配置分布式环境时会更加混乱。现在花2分钟确认地基，后面省2小时。

---

### 第八章：常见问题排查（遇到报错再来看）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
