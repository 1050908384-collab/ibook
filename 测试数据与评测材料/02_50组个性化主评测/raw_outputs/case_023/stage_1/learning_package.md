# 学习包

## 块 1

**本块学习检查**：读完 1.1 节后，请用自己的话说说：为什么没有 Python 解释器，你写的 `.py` 文件就“只是一堆文本”？然后打开电脑，找到你准备安装 Python 的位置，确认你清楚自己接下来要装的是什么。

---

**1.1 Python 解释器：程序的“翻译官”**

你马上要动手做 QLoRA 微调，但先别急着敲代码——你写的 Python 代码是给人看的文字，电脑只认 0 和 1。**Python 解释器**就是那个“翻译官”，它把你的代码翻译成电脑能执行的指令。

没有它，你写的 `.py` 文件就只是一堆文本，电脑根本看不懂。

**为什么这对你很重要？** 因为 QLoRA 微调要跑大量模型训练代码，每一步都依赖解释器正确“翻译”。装对了，后面才能顺利跑通。

**你现在要做的**：记住这个核心概念——解释器 = 翻译官。接下来安装时，你装的正是这个“翻译官”。装好后，你的电脑才能“听懂”你写的 Python 指令。

## 块 2

**你的目标：QLoRA 微调。** 在动手前，先把“工作台”搭对，否则后面每一步都会卡壳。

**1.2 代码编辑器（IDE）：你的“工作台”**
记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐选 **VS Code**（轻量免费），适合你快速上手、边学边改。

## 块 3

## 2.1 Windows 用户：安装 Python 3.10+

你的目标是 QLoRA 微调，而 HuggingFace 生态工具链（transformers、peft、bitsandbytes 等）都依赖 Python 环境。所以，这一步不是“装个软件”，而是给你的微调工作台打地基。咱们直接上手，按步骤来。

### 第一步：下载安装包

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

> 不用纠结版本号，只要是 3.10 以上就行。QLoRA 相关库对 3.10+ 支持良好，推荐直接下最新稳定版。

### 第二步：运行安装程序（⚠️ 关键步骤）

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这一步是新手最容易踩的坑。不勾选的话，你后面打开命令行输入 `python`，系统会提示“不是内部或外部命令”，所有后续操作全部卡死。勾上它，等于告诉系统“python 命令我随时要用”。

2. 点击 **"Install Now"**（默认安装）即可。

## 块 4

### macOS 安装 Python：两条路，选你顺手的

你的目标是 QLoRA 微调，这需要 Python 环境。macOS 上别用系统自带的 Python——它可能很旧，甚至没有。**自己装一个干净的最新版**，后面跑代码才不踩坑。

**方法一：官方安装包（新手首选，点鼠标就行）**

1. 打开浏览器，访问 https://www.python.org
2. 下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）
3. 双击下载的文件，一路点 **Continue** → **Install**，输密码完成安装
4. 打开 **终端**（按 `Command + 空格`，输入 Terminal 回车），输入下面两行验证：

```bash
python3 --version
pip3 --version
```

看到版本号就成功了。

**注意：为什么用 `python3` 而不是 `python`？**

macOS 上通常使用 `python3` 命令，因为系统自带的 `python` 可能指向旧版本或不存在，建议使用 `python3` 以确保使用最新版本。所以以后你敲命令，**永远用 `python3` 和 `pip3`**，别偷懒少打那个 `3`，否则可能报"命令不存在"。

**方法二：Homebrew（如果你已经装了它

## 块 5

**动手装好 Python，为 QLoRA 微调铺路**

你正为 QLoRA 微调做准备，环境是第一步。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最稳妥。

打开终端，逐行执行：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**关键点**：`-y` 让安装自动确认，省去手动输入 `y`；`python3-venv` 是建虚拟环境的模块，一次装好，后面微调时隔离依赖不冲突。

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 QLoRA 微调，那环境选型就一句话：**别在 Windows 原生环境里硬扛，直接奔着 Linux 去。**

**日常练手（写 Python、敲命令行）**

- **Windows 用户**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装需 Windows 10 2004+ 或 Windows 11，管理员 PowerShell 执行 `wsl --install`，安装后需重启，并可用 `wsl --set-default-version 2` 设置 WSL2。
- **macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境，不用额外折腾。

**GPU 实训（LoRA / QLoRA 微调）**

必须用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 避坑**：别把 **"CentOS + VirtualBox 虚拟机"

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 装第三方库（如 `numpy`、`pandas`）的工具。默认从官方 PyPI（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**只有**当你网络受限、访问官方源明显变慢或经常超时时，才临时切到国内镜像。记住：**镜像是"网络不好时的临时加速手段"，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

某次安装慢时，最干净的做法是**只对这一次临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方 PyPI 仍是默认源；网络恢复或镜像出问题时，无需清理

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

如果你只是偶尔下载慢，**不建议**做这一步。只有当长期网络受限、几乎每次都要用镜像时，才把它写进配置当默认源。配置后记住：官方 PyPI 仍是最可靠的，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——加了会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加。

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

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

###

## 块 9

**你的目标：QLoRA 微调，但先别急——工欲善其事，必先利其器。** 你只需要一个轻量、免费、启动快的编辑器：**VS Code**。

**为什么选它**：微软出品，免费。不仅写 Python 顺手，以后调 Linux 命令、改配置文件都用得上，一次投资，长期受益。

**安装三步走**：
1. 打开 https://code.visualstudio.com，下载你系统的安装包。
2. 运行安装，一路“下一步”即可。
3. **装插件（关键）**：点左侧最下方四个方块图标（扩展），搜索安装：
   - **Python**（官方出品）：语法高亮、调试、智能提示。
   - **Pylance**：让代码提示更聪明。

装完插件，VS Code 才能识别 `.py` 文件，给你代码补全和错误提示。

**配置解释器（让 VS Code 找到你的 Python）**：
1. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选你

## 块 10

**本块学习检查**：安装完成后，打开 PyCharm 新建一个项目，确认你能找到“Interpreter”设置并看到你系统里的 Python 3.10+ 版本。如果找不到，回头检查安装时是否勾选了“Add to PATH”。

---

**4.2 方案二：PyCharm Community（免费、开源、适合入门）**

你的目标是 QLoRA 微调，这涉及大型项目开发，PyCharm 社区版正好适合——它免费、开源，内置代码分析和版本控制，能帮你管理复杂的微调脚本。注意：社区版功能比专业版精简，但足以满足本教程需求。

**安装步骤（跟着做）：**

1. 打开官网下载页：https://www.jetbrains.com/pycharm/download/
2. 选 **Community Edition**（社区版，免费那个）
3. 运行安装程序，**务必勾选**：
   - `Add "Open Folder as Project"`（以后右键文件夹就能打开）
   - `Add to PATH`（命令行直接

## 块 11

## 5.1 为什么用 Conda

你的目标是做 **QLoRA 微调**，这意味着后面要装 PyTorch、CUDA 这类“牵一发动全身”的复杂依赖。如果直接在系统 Python 里装，很容易把环境搞乱，项目之间互相打架。

**Conda** 就是来解决这个问题的。它不只是创建虚拟环境，还能直接管理 **Python 解释器版本**——比如这个项目用 Python 3.10，那个项目用 3.8，Conda 都能给你隔离好。对深度学习来说，这几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，直接看表：

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| **Anaconda** | 体积大（约 500MB 至 1GB），预装超过 250 个常用数据科学包 | 做数据分析、不想手动装包的人 |
| **Miniconda** | 体积小（约 50-100MB），仅包含 conda 和 Python 及其必要依赖，按需安装 | 希望环境干净、做项目开发的人 |

**推荐 Miniconda**——你要做微调，环境越干净越好，需要什么装什么。

### 安装步骤

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包：Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`
3. 运行安装程序：

- **Windows**：一路下一步，**千万别勾** "Add Miniconda3 to my PATH environment variable"（安装器自己都标了“不推荐”）。勾了容易和系统已有的 Python 冲突，导致 `python`、`pip` 指向不明。日常使用请打开开始菜单里的 **"Anaconda Prompt (Miniconda3)"**，即可直接使用 conda 命令。若需在普通终端使用，可手动运行 `conda init` 或添加 PATH。
- **macOS/Linux**：运行 `.sh` 文件，按提示输入

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：你正在为 QLoRA 微调打基础。不同项目需要不同版本的 PyTorch 和 CUDA，学会创建环境后，这套流程可以直接复用到微调项目里。

**动手操作**：打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（记住这三个部分就行）：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境里装 Python 3.10

执行后，终端会列出将要安装

## 块 13

## 5.4 激活与退出环境：走进你的“独立房间”

你的目标是 QLoRA 微调，这意味着后面要装很多专门的库。为了避免这些库把系统环境搞乱，我们用 Conda 给项目建一个“独立房间”。

**第一步：进入房间（激活环境）**

在终端输入：

```bash
conda activate llm-learn
```

**怎么确认进来了？** 看命令行最前面，出现了 `(llm-learn)` 标志，就说明你已经在房间里了。为了更可靠，可以再运行 `conda info --envs`，当前环境前会有 `*` 标记。此时所有 `python` 和 `pip` 命令都只作用于这个环境，不会影响系统或其他项目。

**第二步：检查房间配置（验证版本）**

```bash
python --version
```

应该显示 `Python 3.10.x`。如果不是，说明环境

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 QLoRA 微调，而 QLoRA 依赖 HuggingFace 生态工具链，这些工具需要装在干净的环境里。下面动手配置：

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境。选好后，你在这个工作区写的代码就会用 `llm-learn` 里的 Python 和已装好的包——包括后续微调要用的库。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\

## 块 15

**本块学习检查**：完成下方步骤后，在终端输入 `exit()` 退出 Python。

---

你的目标是 QLoRA 微调，但先别急——任何微调都要从“环境能跑通”开始。本块只做一件事：**用命令行确认你的 Python 环境已激活且可用**。

**操作步骤（跟着做，约2分钟）：**

1. **激活环境**：在终端输入 `conda activate llm-learn`，回车。这是你后续所有微调工作的“工作台”。
2. **进入 Python**：输入 `python`，回车。看到 `>>>` 提示符，说明已进入 Python 交互模式。
3. **运行第一行代码**：在 `>>>` 后输入：
   ```python

## 块 16

### 7.3 在 PyCharm 中验证

跟着做，三步搞定：

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目里新建一个文件，命名为 `hello.py`。
2. **运行文件**：在 `hello.py` 文件上**右键点击**，选择菜单里的 `Run 'hello'`。
3. **看结果**：PyCharm 下方会弹出 **Run 窗口**，你的代码输出结果就会显示在这里。

> **动手验证**：在 `hello.py` 里输入 `print("Hello, QLoRA!")`，然后运行。看到 Run 窗口输出这行字，说明你的开发工作台已经全部打通了。

---

### 第八章：常见问题排查

运行或安装时遇到报错？先对照下表找原因：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

---

### 本章小结：你的工作台清单

完成本章后，你的电脑上应该具备以下内容：

| 组件 | 状态 | 验证命令 |
|------|----------|
