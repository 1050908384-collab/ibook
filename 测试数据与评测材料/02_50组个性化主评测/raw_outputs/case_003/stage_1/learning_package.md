# 学习包

## 块 1

## 1.1 Python 解释器：程序的“翻译官”

你写下的 Python 代码是人类语言，但电脑只认 0 和 1。**Python 解释器**就是中间的“翻译官”——它把你写的代码逐句翻译成电脑能执行的指令。没有它，你保存的 `.py` 文件只是一堆普通文本，电脑根本不知道你想干什么。

**为什么这对你的 LoRA 微调目标很重要？**  
你之后要运行的微调脚本（比如训练模型、加载数据集）都是 `.py` 文件。解释器就是让这些脚本真正“跑起来”的第一步。

**动手验证一下（30秒）：**  
打开你的终端（Mac 叫“终端”，Windows 叫“命令提示符”），输入 `python --version` 并回车。如果看到类似 `Python 3.x.x` 的版本号，说明解释器已经装好并可以正常调用了；如果提示“找不到命令”，说明还没装好，继续往下学即可。

> 记住这个感觉：**你写代码 → 解释器翻译 → 电脑执行**。后面所有环境配置，都是为了让这条链路更顺畅。

## 块 2

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，这需要你亲手写代码、跑模型。所以，你的“工作台”必须顺手。

**1.2 代码编辑器（IDE）：你的“工作台”**

别用记事本写代码，那效率太低。IDE 就是你的“智能工作台”，自带语法高亮、自动补全和调试功能，就像用 Word 写文档一样舒服。我们推荐你使用 **VS Code**，它轻量、免费，而且对 Python 和后续的 AI 开发支持极好，完全够用。

**1.3 虚拟环境：项目的“独立房间”**

想象一下，你同时做两个项目：项目 A 需要 `numpy 1.0

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要先有能跑 Python 的工作台。下面三步，跟着做就行。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站通常会推荐适用于你系统的最新稳定版本（如 `Python 3.13.x`），但请确保版本号不低于 3.10。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这一步不勾，后面在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令都跑不了。勾上它，系统才能找到 Python。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续报错。想改路径的话，确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3.

## 块 4

## macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这意味着你需要一个干净、可控的 Python 环境。macOS 上有个常见坑：**系统自带的 Python 别直接用**。macOS 自 Catalina 10.15 起不再预装 Python 2，但系统自带的 `python3` 可能版本较旧，且不推荐用于开发。直接依赖它，版本和权限都可能出问题。

**推荐方法：官方安装包（新手最稳）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 完成。
3. 打开终端（Terminal），输入以下命令验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：在 macOS 上，请**始终使用 `python3` 和 `pip3`**，而不是 `python` 和 `pip`。macOS 默认不提供 `python` 命令，但你可能通过安装其他软件（如 Homebrew）获得它。用 `python` 可能遇到"命令不存在"或误用系统旧版本的问题。

**备选方法：Homebrew（如果你已装）**

在终端执行：

```bash
brew install python
```

装完同样用 `python

## 块 5

**本块学习检查**：运行 `python3 --version`，确认输出类似 `Python 3.10.x` 或更高版本号；若命令报错，重跑安装命令并观察输出。

---

你的目标是 LoRA 微调，但先得让电脑能跑 Python。Ubuntu/Debian 通常自带 Python，但版本偏老，可能不够用。按下面三步走：

1. **更新软件源**：`sudo apt update`——让系统知道最新软件包。
2. **安装三件套**：`sudo apt install python3 python3-pip python3-venv -y`。`-y` 是自动确认，省得你手动敲 `y`；`python3-venv

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选择就直接决定你后面能不能顺利用上 GPU。记住一条主线：**日常练手用 WSL2，真正跑微调用 Ubuntu LTS 的 GPU 机器**。

**如果你是 Windows 用户**，日常写 Python、练命令行时，强烈建议启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04 或 24.04 LTS）。它相当于在 Windows 里开了一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和以后要用的服务器一致。安装很简单，管理员身份打开 PowerShell，执行 `wsl --install` 就行。

**如果你是 macOS 或 Linux 用户**，不用额外折腾，直接用本机终端就行，它本身就是类 Unix 环境。

**到了 LoRA / QLoRA 微调、vLLM 部署这类 GPU 实训环节**，就要切换到装有 **Ubuntu LTS 的 NVIDIA GPU 云服务器**，或者本地实体 Linux（Ubuntu LTS）机器。原因很直接：这些任务依赖 NVIDIA 驱动、CUDA 和显卡直通，而 Ubuntu LTS 对深度学习生态的支持最成熟、文档最全。

> **⚠️ 别踩坑**：不要用 **"CentOS + VirtualBox 虚拟机

## 块 7

**本块学习检查**：打开终端，输入 `pip install numpy`，观察是否卡住或超时。如果明显变慢，再用临时镜像重装一次，对比两次速度差异。

---

`pip` 是 Python 的包管理工具，用来装第三方库（比如 `numpy`）。默认它从官方 PyPI（`pypi.org`）下载——这是**最权威、最完整、最该保留的默认源**。

**什么时候才需要镜像源？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切到国内镜像。记住：**镜像是"网络不好时的临时加速手段"，不是永久默认设置**——官方源始终是首选。

**首选做法：临时指定镜像（推荐）**。如果只是某次安装慢，最干净的方式是只

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

只有当你的网络长期受限、几乎每次安装包都要靠镜像时，才值得把它设为默认。配置后请记住：官方 PyPI 依然是最可靠的来源，网络恢复后建议改回或删除配置。

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

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**它。只有企业内网自建源或确认来源可信时才临时加。

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

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：动手做一遍——按你的系统完成配置，然后运行 `pip config list` 确认输出包含 `global.index-url='https://pypi.tuna.tsinghua

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

1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Inter

## 块 10

**PyCharm Community：为 LoRA 微调准备的“专业工作台”**

你正朝着 LoRA 微调前进，而 PyCharm Community 就是那个“开箱即用”的基地——它免费、专为 Python 设计，内置代码分析和版本控制，能帮你管理微调脚本和数据集，省心省力。

**安装三步走（跟着点就行）：**

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费那个）。
3. 运行安装包，**务必勾选**这两项：
   - `Add "Open Folder as Project"`：以后右键文件夹就能直接打开项目。
   - `Add to PATH`：方便在命令行里直接敲 `pycharm` 启动。

**首次启动，两个关键设置：**

- 点 `New Project`，在 `Location` 选项目存放路径——**路径千万别带中文**，否则后续

## 块 11

### 5.1 为什么用 Conda

你已经知道虚拟环境能为每个项目创建独立的 Python 空间。**Conda** 是当前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能直接管理不同版本的 Python 解释器本身。这对你接下来的 **LoRA 微调** 学习尤其关键——因为深度学习依赖 PyTorch、CUDA 等复杂依赖，用 conda 隔离环境能避免版本冲突，几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，按需取用：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境更干净，后续做 LoRA 微调时需要什么包装什么包，不拖泥带水。

**安装步骤（跟着做）：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成

## 块 12

**动手创建你的第一个虚拟环境**

你的目标是 LoRA 微调，而微调不同模型时，往往需要不同版本的 PyTorch 和 CUDA。如果全装在一起，很容易冲突。现在学会创建独立环境，以后每个项目各用各的环境，互不干扰。

打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**这条命令在做什么？** 拆开看：

- `conda create`：告诉 Conda 你要新建一个环境
- `-n llm-learn`：给这个环境起名叫 `llm-learn`（`n` 就是 `name` 的缩写）
- `python=3.10`：指定该环境使用 Python 3.10 版本

按下

## 块 13

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

### 5.5 查看和管理环境

**查看所有已创建的环境**：

```bash
conda env list
```

带 `*` 的就是当前激活的环境。

**删除环境**（如果以后需要清理）：

```bash
conda remove -n llm-learn --all
```

---

## 第六章：在 VS Code / PyCharm 中使用 Conda

## 块 14

## 6.1 VS Code 配置 Conda 环境

你正在为 LoRA 微调搭建工作台，核心是让 VS Code 用上 `llm-learn` 环境。跟着做：

1. **激活环境**：在终端运行 `conda activate llm-learn`
2. **打开命令面板**：在 VS Code 中按 `Ctrl + Shift + P`
3. **选择解释器**：输入 `Python: Select Interpreter`
4. **挑选 conda 路径**：选择带 `conda` 字样的项（如 `Python 3.10.x ('llm-learn': conda)`）

**为什么**：VS Code 自动识别 conda 环境，选中后，你写的代码就会用 `llm-learn` 里的 Python 和已装好的包——这正是 LoRA 微调所需的依赖环境。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`
3. 点齿轮图标 → `Add...`
4. 左侧选 `Conda Environment` → `Existing environment`
5. 在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件（可用 `conda

## 块 15

**本块学习检查**：完成下面两步，确认你的“工作台”能跑通代码。

**1. 命令行验证（最快路径）**
- 先激活环境：`conda activate llm-learn`
- 输入 `python` 回车，看到 `>>>` 提示符后，输入：
  ```python
  print("Hello, Python! 我的环境准备好了！")
  ```
- 正确打印文字后，输入 `exit()` 退出。

**2. VS Code 验证（日常写代码用）**
- 新建文件夹 `my-first-project` 并用 VS Code 打开
- 新建文件 `hello.py`，输入：
  ```python
  print("Hello from VS Code!")
  ```
- 点右上角

## 块 16

好的，让我们直接开始。你的目标是LoRA微调，这意味着你最终要训练自己的大模型。在那之前，把“地基”打牢至关重要。这一章就是验证你的“地基”是否稳固。

**7.3 在 PyCharm 中验证**

既然你喜欢动手，我们就直接来操作。这一步是确认你的“工作台”能跑起来。

1.  **新建项目**：打开 PyCharm，创建一个新项目。
2.  **创建文件**：在项目中，新建一个 Python 文件，命名为 `hello.py`。
3.  **编写代码**：在 `hello.py` 中输入 `print("Hello, World!")`。
4.  **运行**：在代码编辑区，**右键点击**鼠标，在弹出的菜单中选择 `Run 'hello'`。
5.  **查看结果**：看 PyCharm 下方的 **Run** 窗口。如果一切正常，你会看到程序输出的结果。

---

**第八章：常见问题排查**

动手过程中，遇到报错是家常便饭。这里有一份“避坑指南”，遇到问题先来这里对照一下。

| 问题现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选择 “Modify”，勾选 “Add Python to environment variables”。 |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源。 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端。 |
| VS Code 提示“未选择 Python 解释器” | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择。 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境。 |
| pip 安装后，Python 里 `
