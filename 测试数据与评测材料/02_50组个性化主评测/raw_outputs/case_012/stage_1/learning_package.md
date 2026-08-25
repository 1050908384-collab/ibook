# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。**Python 解释器**的作用，就是把你的代码“翻译”成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本。

**动手验证一下（约 2 分钟）：**

1. 随便新建一个文本文件，改名为 `test.py`，在里面写一行：`print("hello")`。
2. 双击运行它——你会发现它只是闪了一下（或根本没反应），因为电脑读不懂这段文字。
3. 等装好解释器后，再运行同一个文件，你就能看到 `hello` 被打印出来。

这个对比，就是解释器存在的意义：**它让你的文字变成电脑的行动**。记住这个感觉，后面每一步你都会知道自己在做什么。

## 块 2

好的，我们直接进入正题。既然你的目标是理解 Transformer 架构，并且喜欢动手实践，那我们就别把时间浪费在“记事本写代码”这种低效方式上。你需要的是一套能让你专注算法逻辑、快速验证想法的开发环境。

**1.2 你的“工作台”：代码编辑器（IDE）**

你马上要写的 Transformer 代码，会涉及大量的缩进、函数调用和调试。记事本绝对不行。你需要一个 **IDE（集成开发环境）**，它能提供**语法高亮**（让代码结构一目了然）和**自动补全**（减少拼写错误），让你写代码像用 Word 写文档一样舒服。

对于你的学习节奏，推荐 **VS Code**。它**轻量免费**，启动快，非常适合你这种需要快速迭代、频繁验证想法的学习场景

## 块 3

### 2.1 Windows 用户：动手装好你的 Python 工作台

你的目标是理解 Transformer 架构，而动手实践是掌握它的最佳路径。装好 Python 就是第一步，跟着下面三步走，很快就能跑起来。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车

## 块 4

### macOS 用户：安装 Python 3.10+

现代 macOS（12.3 起）不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用。**强烈建议自行安装最新版 Python**，避免依赖系统版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认不提供 `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：在终端运行 `python3 --version`，确认输出为 Python 3.10 或更高版本；再运行 `pip3 --version`，确认 pip 可用。若命令报错，请检查安装步骤或重新

## 块 5

**动手装好 Python 3：Ubuntu/Debian 实操**

你的 Linux 系统通常自带 Python，但版本可能偏旧。别急，跟着下面三步，装好最新版 Python 3 和配套工具：

```bash
# 1. 更新软件源，确保能拿到最新安装包
sudo apt update

# 2. 安装 Python 3、pip 和虚拟环境模块（-y 自动确认，不用手动敲 y）
sudo apt install python3 python3-pip python3-venv -y

# 3. 验证安装是否成功
python3 --version
```

**关键点**：`python3-venv` 是创建虚拟环境的核心模块，现在一起装好，省得

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是理解 Transformer 架构，最终要动手跑 LoRA/QLoRA 微调和 vLLM 部署这类 GPU 实训。选对环境，能让你少走弯路，更快进入实战。

**日常写 Python、练命令行（Windows 用户）**

推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在你 Windows 里提供一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续的服务器一致，比在 Windows 原生 CMD 里练命令更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install` 即可。

**日常写 Python（macOS / Linux 用户）**

直接使用本机终端即可，本身就是类 Unix 环境，无需额外配置。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**

应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态的支持最成熟、文档最全。

> **⚠️ 不推荐的做法**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护（202

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是最权威、最完整、最应保留的默认源。

**只有**当你身处网络受限环境、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响你平时的

## 块 8

## 动手实操：把镜像设为默认（长期网络受限才做）

你现在的目标是快速理解 Transformer，而装环境只是“工具准备”。如果你只是偶尔下载慢，**别急着改默认配置**——用 `-i` 参数临时指定镜像就够了。只有当你的网络长期受限、每次都要敲镜像地址时，才值得把它写进配置文件。

**为什么这样做**：写配置相当于告诉 pip“以后默认从清华源下载”，省去每次手动指定。但官方 PyPI 始终是最可靠的，网络恢复后建议改回。

### Windows 实操步骤

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开配置目录（类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 检查是否已有 `pip` 文件夹，若没有则新建，命名为 `pip`。
3. 进入 `pip` 文件夹，新建文本文件 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从哪个地址下载包。

### macOS / Linux 实操步骤

1. 打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建并编辑配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样内容，按 `Ctrl + O` 保存，`Ctrl + X` 退出。

### 安全提醒：别乱加 trusted-host

网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——加上它等于让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用不需要。只有企业内网自建源或确实遇到证书报错且确认来源可信时，才临时加。另外，如果镜像源是 HTTP 而非 HTTPS，则需要添加 `trusted-host` 才能正常使用。

### 验证配置是否生效

命令行执行：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 就说明成功了。如果

## 块 9

**动手搭好你的 Python 工作台：VS Code 安装与配置**

你的目标是理解 Transformer 架构，而写代码跑模型是理解它的最快方式。VS Code 就是帮你把代码写顺手的工具——它轻量、免费，启动快，插件丰富，以后学 Linux、写脚本也能复用。

**安装步骤（约5分钟）**

1. 打开浏览器，访问 `https://code.visualstudio.com`，下载对应你系统的安装包。
2. 运行安装程序，一路点“下一步”完成安装。

**装两个必装插件（关键）**

打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

- **Python**（Microsoft 官方出品）：提供语法高亮、调试、智能提示。
- **Pylance**：Python 语言服务器，让代码提示更智能。

> 为什么必须装？不装 Python 插件，VS Code 无法识别 `.py` 文件，也就没有代码补全、错误提示和一键运行功能。

**配置

## 块 10

**本块学习检查**：安装完成后，在 PyCharm 中新建项目并选择系统 Python 3.10+，确认能成功运行一段 `print("Hello")` 代码，即可验证环境可用。

---

**4.2 方案二：PyCharm Community（专业、开箱即用）**

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。内置代码分析、版本控制、数据库工具，适合大型项目开发。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费）
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`
2. `Location` 处选择项目路径（**路径中不要含中文**）
3. `Interpreter`

## 块 11

## 5.1 为什么用 Conda

你马上要开始搭建 Transformer 的学习环境了。这个项目需要 PyTorch、CUDA 等一堆复杂依赖，如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 就是来解决这个问题的——它能为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习来说，Conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，直接看表：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用空间通常超过 3GB），预装 numpy、pandas、jupyter 等大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 50-100MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境干净，后面需要什么装什么，不浪费空间。

## 安装步骤

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包：Windows 选 `.exe`，macOS 选 `.pkg`（图形化安装）或 `.sh`（命令行安装），Linux 选 `.sh`
3. 运行安装程序，注意系统差异：

- **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器自己都标了"不推荐"）。直接塞进 PATH 容易和系统已有的 Python 冲突，导致 `python`、`pip` 指向不明。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来操作 conda。若需在 cmd 或 PowerShell 中使用 conda，可手动添加 PATH，但需注意与系统 Python 的冲突风险。
- **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选安装路径，最后一步**必须选 `

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么先学这个**：你后面要学 Transformer 微调，不同项目需要不同版本的 PyTorch 和 CUDA。现在花两分钟学会建环境，以后直接复用这套流程，省去反复折腾依赖的麻烦。

**动手做**：打开终端，输入下面这行命令，然后回车：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（记住这三个部分就够了）：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`

## 块 13

### 5.4 激活与退出环境

**激活环境**（进入这个"独立房间"）：

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

先激活环境：`conda activate llm-learn`。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境。选择后，当前工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

## 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选择 `Project: 你的项目名` → `Python Interpreter`。点击齿轮图标 → `Add...`。左侧选择 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：在 VS Code 或 PyCharm 中，新建一个 Python 文件，输入 `import sys` 并运行，确认无报错即表示

## 块 15

**动手验证：让环境“开口说话”**

既然你喜欢动手，我们直接来。打开终端（命令行），先激活环境：

```bash
conda activate llm-learn
```

接着输入 `python` 并回车，你会看到 `>>>` 提示符——这就是 Python 在等你下指令。

在 `>>>` 后输入下面这行，再按回车：

```python
print("Hello, Python! 我的环境准备好了！")
```

屏幕打印出这句话，说明环境没问题。输入 `exit()` 退出。

**换到 VS Code 再试一次**：新建文件夹 `my-first-project`，用 VS Code 打开；新建 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点右上角小三角或按 `Ctrl + F5

## 块 16

好的，让我们直接进入正题，动手验证你的开发环境。

## 7.3 在 PyCharm 中验证

既然你偏好动手实践，我们直接开始。请打开 PyCharm，按照下面的步骤操作：

1.  **新建项目**：点击 `File` → `New Project`，创建一个新的 Python 项目。
2.  **创建文件**：在项目目录上右键，选择 `New` → `Python File`，并将其命名为 `hello.py`。
3.  **编写代码**：在打开的 `hello.py` 文件中，输入 `print("Hello, World!")`。
4.  **运行脚本**：在 `hello.py` 文件内，点击鼠标右键，选择 `Run 'hello'`。
5.  **查看结果**：此时，PyCharm 界面下方会弹出 `Run` 窗口，你应该能在其中看到程序输出的 `Hello, World!`。

看到这行输出，就说明你的 Python 开发工作台已经成功搭建并运行起来了。

---

## 第八章：常见问题排查

在搭建过程中，你可能会遇到一些常见问题。这里有一份速查表，遇到问题时可以对照排查：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）
