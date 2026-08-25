# 学习包

## 块 1

**1.1 Python 解释器：程序的"翻译官"**

你写代码是为了让电脑干活，但电脑只懂 0 和 1，不懂你的文字。**Python 解释器**就是中间的"翻译官"：它把你写的 Python 代码翻译成电脑能执行的指令。

没有它，你写的 `.py` 文件只是一堆普通文本，电脑根本不会理你。

**一句话记住**：解释器 = 翻译官，负责把你的话变成电脑能懂的命令。

---

**本块学习检查**（动手验证，约 1 分钟）：

请用自己的话说：如果电脑里没有 Python 解释器，你写好的代码文件会怎样？想好后，在终端输入 `python --version`，看是否出现版本号——出现即说明解释器已就位，没出现则先别急，下一节带你装好它。

## 块 2

### 1.2 代码编辑器（IDE）：你的“工作台”

写代码不一定要用记事本——虽然它也能写，但效率太低。**IDE（集成开发环境）**把语法高亮、自动补全、运行调试这些功能打包在一起，让你写代码像用 Word 写文档一样顺手。对你做 LoRA 微调来说，推荐两款：**VS Code**（轻量免费，上手快）和 **PyCharm**（专业强大，功能全）。选 VS Code 起步就够用。

### 1.3 虚拟环境：项目的“独立房间”

想象你同时在跑两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。如果都装进同一个 Python，就会冲突报错。**虚拟环境**就是给每个项目一个独立的 Python 空间，互不干扰。做 LoRA 微调时，不同实验常依赖不同版本库，

## 块 3

**目标：LoRA微调**。在动手微调前，先把“工作台”搭好——Python 是运行一切微调脚本的基础。你每天只有1小时，所以直接上干货，按步骤来，10分钟内搞定。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。鼠标移到顶部菜单 **Downloads**，网站会根据你的操作系统显示推荐版本，通常为最新稳定版，如 Python 3.12 或更高版本。点击下载 **Windows installer (64-bit)**，并确认下载的是 64 位安装包。

**第二步：运行安装程序（⚠️ 关键步骤，别跳过）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**。  
   这是最容易踩的坑。不勾选的话，后面你在命令行输入 `python`，系统会提示“不是内部或外部命令”，LoRA 相关的所有命令行操作（安装依赖、跑脚本）全都会卡死。

2. 点击 **"Install Now"**（默认安装）。  
   默认路径不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

按 `Win + R`，输入 `cmd`，回车打开命令提示符。输入以下命令并

## 块 4

### macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要你亲手搭建一个可靠的 Python 环境。好消息是，macOS 上安装 Python 很简单，跟着下面的步骤走就行。

**先了解现状**：macOS 12.3 起不再预装 Python 2，但系统可能仍带有供系统脚本使用的旧版 `python3`。**强烈建议你自行安装最新版 Python**，别依赖系统自带的版本，免得后面微调时出幺蛾子。

---

**方法一：官方安装包（推荐新手，最省心）**

1. 打开浏览器，访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入下面两行命令验证是否装好：

```bash
python3 --version
pip3 --version
```

看到版本号输出就说明成功了。

**关键提醒**：在现代 macOS 上，命令行默认**并不提供** `python` 命令。系统与官方安装包统一使用 `python3`。所以请**始终用 `python3` / `pip3`**，别用 `python`，否则会踩到"命令不存在"或误

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是跑 LoRA 微调，Linux 上先得有顺手的 Python。Ubuntu/Debian 通常自带 Python，但版本可能偏老，我们直接装新的。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：第一条更新软件源，确保能装到最新包；第二条一次装齐 Python 3、包管理器 pip 和虚拟环境模块 venv，`-y` 表示自动确认，不用你手动敲 `y`；第三条验证版本，看到输出就成功了。

**本块学习检查**

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，而且喜欢动手实操。环境选对了，后面跑代码能省一半力气。记住一条主线：**日常练手和 GPU 实训分开选环境**。

**日常写 Python、练命令行（Windows 用户）**：直接启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04 或 24.04）。它让你在 Windows 里拥有一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致。安装很简单：管理员 PowerShell 里执行 `wsl --install` 即可。别在 Windows 原生 CMD 里练，路径和命令习惯和服务器差太多，容易白练。

**日常写 Python（macOS / Linux 用户）**：不用额外折腾，直接用本机终端。macOS 和 Linux 本身就是类 Unix 环境，和服务器天然一致。

**LoRA / QLoRA 微调这类 GPU 实训**：必须用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 和显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全，遇到问题搜得到答案

## 块 7

你正在为 LoRA 微调做准备，安装依赖库（如 `transformers`、`peft`）时，`pip` 默认从官方 PyPI（`pypi.org`）下载。这是最权威、最完整的源，**应始终保留为默认**。

**什么时候才需要换镜像？** 仅当网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像只是“网络不好时的加速手段”，不是永久设置。

**首选做法：临时指定镜像（推荐）**  
只对某一次安装临时指定国内镜像，不改动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样官方源仍是默认，网络恢复或镜像出问题时，无需清理配置。

**本块学习检查**：  
打开终端，用上述命令临时安装 `numpy`，确认安装成功且命令中包含了 `-i` 参数

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，如果网络长期受限，每次用镜像下载依赖包会很麻烦。本块教你**一次性配置**，让 pip 默认走清华镜像。

> **重要**：仅当长期网络受限才这样做。官方 PyPI 最可靠，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车（打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹）。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**。只有企业内网自建源或确认来源可信时才临时加。

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

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：你要做 LoRA 微调，后续会频繁接触 Python 脚本、配置文件甚至 Linux 命令行。VS Code 启动快、插件丰富，一个编辑器全搞定，省心。

**安装步骤**（跟着点就行，约5分钟）：

1. 打开 https://code.visualstudio.com，下载你系统的安装包。
2. **Windows**：运行安装程序，一路“下一步”直到完成。**macOS**：打开下载的 `.zip` 文件，将 `Visual Studio Code.app` 拖入“应用程序”文件夹；或用终端执行 `brew install --cask visual-studio-code`。**Linux**：下载 `.deb` 或 `.rpm` 包，用系统包管理器安装（如 `sudo dpkg -i <包名>.deb`）。安装后若系统提示安全警告，按提示允许运行；如遇异常，重启 VS Code

## 块 10

**动手装 PyCharm，为 LoRA 微调搭好“工作台”**

你目标是 LoRA 微调，后面要读、改大量 Python 代码。PyCharm Community 免费、专业，正好当你的主力 IDE。

**安装要点（跟着点就行）**
1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费那个）。
3. 安装时勾上这一项，后面省事：
   - `Add "Open Folder as Project"`（右键文件夹就能用 PyCharm 打开）

**首次启动，两步配置**
1. 打开后选 `New Project`。
2. **Location** 选项目路径——**千万别含中文**，否则后面跑 LoRA 脚本容易出怪问题。
3. **Interpreter** 选 `Previously configured interpreter`，然后指定你系统里装好的 Python 3.10+。

**本块学习检查

## 块 11

### 5.1 为什么用 Conda

你之前接触过虚拟环境的概念——每个项目拥有独立的 Python 空间，互不干扰。**Conda** 是目前最主流的虚拟环境管理工具，它的特别之处在于：不仅能创建虚拟环境，还能直接管理不同版本的 Python 解释器本身。

这对你的 **LoRA 微调** 目标特别重要。微调大模型时，你需要安装 PyTorch、CUDA 等复杂依赖，这些库对 Python 版本和底层环境非常敏感。用 Conda 可以随时创建“干净”的环境，装错或搞乱了直接删掉重建，不会影响你系统里其他项目。可以说，Conda 是深度学习开发的“必备工具”。

### 5.2 安装 Conda：Anaconda vs Miniconda

安装前先选版本，两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**。原因很直接：你后续做 LoRA 微调，需要什么包装什么包，环境越干净越不容易出问题。Anaconda 预装的大量库对你来说反而是负担。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器也标为“不推荐”）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `

## 块 12

**创建你的第一个虚拟环境**

你即将开始 LoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

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

输入 `y` 并回车，等待安装完成。

**

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

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件（请根据实际安装路径替换）：
   - Windows：`${CONDA_PREFIX}\envs\llm-learn\python.exe`
   - macOS/Linux：`${CONDA_PREFIX}/envs/llm-learn/bin/python`

---

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证

## 块 15

**在命令行验证（动手做）**

你的目标是LoRA微调，但任何微调都从“环境能跑通”开始。这一步就是快速确认你的Python工作台没装错。

1. **激活环境**：在终端输入 `conda activate llm-learn`，回车。
2. **进入Python**：输入 `python`，回车。看到 `>>>` 提示符，说明进入交互模式。
3. **测试输出**：在 `>>>` 后输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
   回车。如果看到这行文字打印出来，说明环境正常。
4. **退出**：输入 `exit()` 回车。

**本块学习检查**：按上述步骤操作，确认终端

## 块 16

### 7.3 在 PyCharm 中验证

打开 PyCharm，新建一个项目，然后在项目里创建一个新文件，命名为 `hello.py`。在文件里输入一行最简单的代码（比如打印一句话）。写完后，在文件编辑区**右键点击**，选择菜单里的 **`Run 'hello'`**。PyCharm 会在窗口**下方**打开一个 **Run** 面板，你的代码运行结果就会显示在那里。看到输出，就说明你的开发工作台已经能跑代码了。

---

### 第八章：常见问题排查（速查表）

运行环境出问题时，先对照这张表找原因：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 用户先确保 conda 已正确安装，运行 `conda init` 后重新打开终端或执行 `source ~/.bashrc`（或对应 shell 配置文件）使配置生效 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

---

### 本章小结：环境清单

完成本章后，你的电脑上
