# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你写下的 Python 代码是人类语言，但电脑只认 0 和 1。**Python 解释器**就是那个“翻译官”——它把你写的代码逐句翻译成电脑能执行的指令。没有它，你保存的 `.py` 文件就只是一堆普通文本，电脑根本看不懂。

**为什么你现在就要懂这个？** 因为你做 LoRA 微调时，要跑的训练脚本（比如用 PyTorch 写的那一套）本质上都是 `.py` 文件。解释器装对了，这些脚本才能被“翻译”成 GPU 能算的指令；装不对，后面寸步难行。

**动手验证一下（30秒）**：打开终端（Mac 叫“终端”，Windows 叫“命令提示符”），输入 `python --version` 并回车。如果看到类似 `Python 3.x.x` 的版本号，说明解释器已经在工作了；如果提示“找不到命令”，说明还没装好，继续往下看安装步骤即可。

> 记住这个比喻：解释器 = 翻译官。你写代码，它负责让电脑听懂。

## 块 2

### 1.2 代码编辑器（IDE）：你的"工作台"

写代码用记事本也能行，但效率太低。**IDE（集成开发环境）**就是给你的代码加buff：语法高亮、自动补全、运行调试，用起来像写Word一样顺手。你目标LoRA微调，后面要反复改模型代码，选对工具能省一半时间。推荐两款：**VS Code**（轻量免费，上手快）和 **PyCharm**（专业强大，适合深度调试）。先装VS Code，够用且不折腾。

### 1.3 虚拟环境：项目的"独立房间"

想象你同时做两个项目：A需要`numpy 1.0`，B需要`numpy 2.0`，

## 块 3

**本块目标**：在 Windows 上装好 Python 3.10+，为 LoRA 微调铺路。你偏好动手实操，我们直接按步骤来。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。把鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。Python 3.10+ 包括 3.10、3.11、3.12 等版本，推荐下载最新稳定版。点击下载 **Windows installer (64-bit)**。

> 动手点：确认下载的是 64 位安装包，别下成 32 位。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，之后在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令行操作都会卡住——LoRA 微调全程依赖命令行，这一步不能省。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，

## 块 4

## macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要你亲手搭建一个可用的 Python 环境。好消息是，macOS 上这一步并不复杂——跟着下面的操作走，10 分钟内就能搞定。

> **先记住一个关键点**：在 macOS 上，请始终使用 `python3` 和 `pip3` 命令，而不是 `python` 和 `pip`。macOS 12.3 及更高版本不再预装 Python 2，系统不提供 `python` 命令；在较早版本（如 Catalina 10.15 至 Monterey 12.2）中，`python` 命令可能指向 Python 2.7，建议始终使用 `python3` 以避免误用。

### 方法一：官方安装包（推荐新手，操作最简单）

1. 打开浏览器，访问 https://www.python.org
2. 下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）
3. 双击下载的文件，一路点击 **"Continue"** → **"Install"**，等待安装完成
4. 打开终端（Terminal），输入以下两条命令验证安装：

```bash
python3 --version
pip3 --version
```

如果两条命令都正常输出版本号，说明安装成功。

### 方法二：Homebrew（适合已装

## 块 5

**动手装好 Python，马上开跑 LoRA**

你的目标是 LoRA 微调，但先别急着碰模型——环境没搭好，后面全是坑。这一节我们只干一件事：在 Ubuntu/Debian 上把 Python 3 装到能用。

**三步走，复制粘贴即可**

打开终端，依次执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**每行在干嘛，心里要有数**

- 第一行：刷新软件列表，让系统知道有哪些新版本可装。
- 第二行：一次装齐三样东西——`python3`（解释器）、`python3-pip`（装第三方

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，那环境选型直接决定你能不能跑起来。记住一条主线：**日常练习和 GPU 实训分开选**。

**日常写 Python、练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境，不用额外配置。

**LoRA / QLoRA 微调等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别踩坑**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于 2021 年停止维护，CentOS 7 也于 2024 年停止维护，不建议

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认它从官方 PyPI（`pypi.org`）下载——这是**最权威、最完整、应保留的默认源**。

只有当网络受限、访问官方源明显变慢或经常超时时，才临时切到国内镜像。**镜像只是"网络不好时的加速手段"，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

某次安装慢时，最干净的做法是**只对这一次临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方 PyPI 仍是默认源；网络恢复或镜像出问题时，无需清理配置。

**本块学习检查

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你现在要跑 LoRA 微调，后面会频繁 `pip install` 各种依赖包。如果你长期处在网络受限环境、几乎每次都要用镜像，才值得把它设为默认。**注意前提**：官方 PyPI 永远是最可靠的，网络恢复后建议改回或删除配置。

---

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%` 回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

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

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明

## 块 9

**目标对齐**：你正在为 LoRA 微调搭建环境，VS Code 就是你的“操作台”。它轻量、免费，启动快，插件丰富，后续写训练脚本、改配置都能用。

**安装（3 步搞定）**
1. 打开 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路“下一步”即可。
3. 装完打开 VS Code，点击左侧最下方的四个方块图标（扩展），搜索并安装两个**必装插件**：
   - **Python**（微软官方，提供语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更智能）

> 安装 Python 插件可获得更完善的语法高亮、调试和智能提示等功能，但 VS Code 本身也能识别 `.py` 文件。

**配置 Python 解释器**
1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。

## 块 10

**本块学习检查**：安装完成后，打开 PyCharm 新建项目，确认你能找到“Interpreter”设置项并看到系统 Python 3.10+ 的选项。

---

你的目标是 LoRA 微调，这需要写 Python 脚本、调试代码。PyCharm Community 是免费且开箱即用的专业 IDE，正好适合你边学边练。

**安装要点**（跟着做，5分钟搞定）：

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费那个）
3. 安装时务必勾选两项：
   - `Add "Open Folder as Project"` —— 以后右键文件夹就能用 PyCharm 打开
   - `Add to PATH` —— 命令行直接敲 `pycharm` 启动

**首次启动配置**（关键一步）：

1. 点 `New Project`
2. `Location` 选项目路径——**别用

## 块 11

**本块学习检查**：安装完成后，打开终端输入 `conda --version`，确认输出类似 `conda 24.x.x` 的版本号，再继续往下学。

---

### 5.1 为什么用 Conda

你马上要动手做 **LoRA 微调**，这需要 PyTorch 等一堆深度学习依赖。如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 就是来解决这个问题的：它为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，看表：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装 numpy、pandas、jupyter 等 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，环境干净，后面需要什么装什么。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**勾选 "Add Miniconda3 to my PATH environment variable"**，以便在命令行直接使用 conda；若担心冲突，可后续通过 `conda init` 调整。日常也可用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步输入 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端 conda 会通过初始化脚本正确

## 块 12

**目标**：为 LoRA 微调铺路——不同项目需要不同 PyTorch/CUDA 版本，环境隔离是第一步。

**动手做**：在终端执行下面这条命令，创建名为 `llm-learn` 的独立环境：

```bash
conda create -n llm-learn python=3.10
```

**拆解**（记住这个套路，以后复用）：
- `conda create`：创建环境的命令
- `-n llm-learn`：环境名，`n` 就是 `name`
- `python=3.10`：指定装 Python 3.10

**执行后**：终端会列出待装包并问 `Proceed ([y]/n)?`，输入 `y` 回车，等它装完。

## 块 13

**本块学习检查**：完成下面三步操作，确认你已掌握环境切换的核心动作：

1. 在终端输入 `conda activate llm-learn`，观察命令行提示符是否出现 `(llm-learn)` 前缀。
2. 输入 `python --version`，确认输出为 `Python 3.10.x`。
3. 输入 `conda deactivate`，确认前缀消失，回到系统默认环境。

---

**5.4 激活与退出环境**

激活环境就是“进入这个独立房间”。执行：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

验证当前环境的 Python 版本：

```bash
python --version
```

应该显示 `Python 3.10.x`。

退出环境（回到系统默认环境）：

```bash

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，这意味着代码必须跑在正确的环境里。跟着做：

1. **先激活环境**：在终端执行 `conda activate llm-learn`。
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入 `Python: Select Interpreter` 并回车。
4. 在列表中选择带有 `conda` 字样的 Python 路径（形如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样做**：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的所有代码都会使用 `llm-learn` 环境里的 Python 和已安装的包——包括你后续微调要用的 PyTorch 等依赖。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程略有不同：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行

## 块 15

**本块学习检查**：完成下方两步验证后，在终端输入 `exit()` 退出 Python。

你的目标是 LoRA 微调，动手验证环境是第一步。跟着做，很快：

1. **激活环境**：在命令行输入 `conda activate llm-learn`，回车。
2. **进入 Python**：输入 `python`，回车，看到 `>>>` 提示符。
3. **运行代码**：在 `>>>` 后输入 `print("Hello, Python! 我的环境准备好了！")`，回车。
4. **确认输出**：屏幕打印出这句话，说明环境正常。输入 `exit()` 退出。

**VS Code 验证**（更贴近后续写代码场景）：

1. 新建文件夹 `my

## 块 16

好的，让我们直接开始动手。你的目标是掌握 LoRA 微调，这需要扎实的 Python 基础。所以，我们第一步就是确保你的“工作台”完全就绪。

### 7.3 在 PyCharm 中验证

既然你偏好动手实践，我们直接操作。请按以下步骤，在 PyCharm 中运行你的第一行代码：

1.  **新建项目**：打开 PyCharm，创建一个新项目。
2.  **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3.  **编写代码**：在 `hello.py` 中输入 `print("Hello, World!")`。
4.  **运行文件**：在编辑区**右键点击** `hello.py` 文件，选择 **`Run 'hello'`**。
5.  **验证结果**：查看 PyCharm 底部的 **Run** 窗口。如果一切正常，你会看到输出了 `Hello, World!`。

看到这行输出，就代表你的开发环境已经可以正常工作了。

---

### 第八章：常见问题排查

在环境搭建过程中，你可能会遇到一些小状况。这里有一份“避坑指南”，遇到问题时可以对照排查：

| 问题现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选择 “Modify”，勾选 “Add Python to environment variables”。 |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源。 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端。 |
| VS Code 提示“未选择 Python 解释器” | 未配置解释器路径 | 按 `Ctrl + Shift +
