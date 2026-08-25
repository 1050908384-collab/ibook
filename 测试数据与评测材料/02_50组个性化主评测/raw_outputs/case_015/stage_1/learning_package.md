# 学习包

## 块 1

**1.1 Python 解释器：程序的"翻译官"**

你写代码时用的是人类语言，但电脑只懂0和1。**Python解释器**就是中间的"翻译官"——它把你写的代码翻译成电脑能执行的指令。没有它，你保存的`.py`文件就只是一堆普通文本。

**为什么你现在必须理解它？** 因为你最终要跑LoRA微调，而微调脚本正是通过解释器来运行的。搞懂这一层，后面配置环境时你就不会盲目点"下一步"。

**动手验证一下**：打开终端（Mac/Linux）或命令提示符（Windows），输入`python`并回车。如果看到类似`Python 3.x.x`的版本信息，说明解释器已就绪；如果提示"未找到命令"，说明还没装好——这正是下一节要解决的。

> 记住：解释器是地基，地基稳了，LoRA微调才能跑得顺。

## 块 2

### 你的“工作台”与“独立房间”

你的目标是 LoRA 微调，这需要你写代码、跑实验。记事本也能写，但效率极低。**IDE（集成开发环境）** 提供语法高亮、自动补全、运行调试，让你写代码像用 Word 写文档一样舒服。推荐 **VS Code**（轻量免费）和 **PyCharm**（专业强大），选一个上手即可。

接着是**虚拟环境**——项目的“独立房间”。想象你同时做两个项目：A 需要 `numpy 1.0`，B 需要 `numpy 2.0`，都装进同一个 Python 会冲突报错。虚拟环境为每个项目创建独立 Python 空间，互不影响

## 块 3

### Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要先有一个能跑 Python 的工作台。下面三步，跟着做就行。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`。
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（比如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**。  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，之后在命令行输入 `python`，系统会提示"不是内部或外部命令"，后续所有命令行操作都做不了。

2. 点击 **"Install Now"**（默认安装）。  
   默认安装路径通常不含中文和特殊符号，但若你的 Windows 用户名包含中文或空格，建议自定义安装路径，确保路径中不含中文和空格。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python 3.13.x`

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要你亲手搭建一个干净、可控的 Python 环境。macOS 系统自带的 Python 版本可能不满足最新依赖要求，**强烈建议**自行安装最新版 Python 3.10+，避免后续微调时出现依赖冲突。

**为什么用 `python3` 而不是 `python`？**
现代 macOS（Catalina 10.15 起）已不预装 Python 2，命令行默认**没有** `python` 命令。系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到“命令不存在”的坑。

---

**方法一：官方安装包（推荐新手，最省心）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

看到版本号输出，即安装成功。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是跑 LoRA 微调，得先把 Python 环境弄利索。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省事。

打开终端，依次执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

第一行刷新软件源，第二行装 Python 3、pip 和 venv。`-y` 是自动确认，不用手动敲 `y`。`python3-venv` 是建虚拟环境用的，一起装好，免得后面缺模块再折腾。

装完跑第三行，看到版本

## 块 6

## 你的环境选择路线图（目标：LoRA 微调）

你的目标是跑通 LoRA 微调，且偏好动手实操。环境选对了，后面省下的时间远超今天花掉的十分钟。按你的场景，直接看两条路：

**路线 A：日常练 Python 和命令行（Windows 用户）**
启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。为什么？它在你 Windows 里开出一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致——你在本地敲的每一行命令，换到云服务器上都能直接复用，不用学两套。安装很简单：管理员 PowerShell 里执行 `wsl --install` 即可。

**路线 B：LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**
用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全，遇到问题搜得到答案。

**⚠️ 避坑提醒**
别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于

## 块 7

**本块学习检查**：运行 `pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`，观察安装速度是否明显快于默认源。

---

你要做 LoRA 微调，第一步是装好 PyTorch 等库。`pip` 默认从官方 PyPI（`pypi.org`）下载——这是最权威、最完整的源，**永远保留为默认**。

**什么时候用镜像？** 仅当网络受限、官方源明显变慢或频繁超时。镜像只是“临时加速手段”，不是永久设置。

**首选做法：临时指定镜像**，只对某一次安装生效，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐？** 不干扰日常默认行为；网络恢复或镜像出问题时，无需清理配置。

**动手验证**：先正常安装一次 `numpy

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为 LoRA 微调准备环境，下载依赖包是绕不开的一步。如果你**长期处于网络受限环境**，几乎每次装包都要手动加镜像地址，那才值得把它写进配置，让 pip 默认走镜像。但请记住：**官方 PyPI 始终是最可靠的来源**，网络恢复后建议改回或删除该配置。

### Windows 用户操作步骤

1. 按 `Win + R`，输入 `%APPDATA%` 回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包；`trusted-host` 用于避免该镜像的 HTTPS 证书验证失败。

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
trust

## 块 9

好的，我们直接进入正题。你的目标是 LoRA 微调，这意味着你很快就要和深度学习框架打交道了。工欲善其事，必先利其器，一个顺手的代码编辑器是第一步。VS Code 就是那个“轻量、免费、推荐”的方案。

**为什么选它**：VS Code 启动快，插件丰富，不仅写 Python 好用，以后你调试 LoRA 脚本、查看配置文件时都用得上。

**安装步骤**：

1.  访问 `https://code.visualstudio.com`，下载对应系统的安装包。
2.  运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

-   **Python**（微软官方出品，提供语法高亮、调试、智能提示）
-   **Pylance**（让代码提示更智能）

装好这两个插件，VS Code 才能识别 `.py` 文件，给你代码补

## 块 10

**动手装 PyCharm，为 LoRA 微调搭好“工作台”**

你的目标是 LoRA 微调，这需要写、改、调试大量 Python 代码。PyCharm Community（免费版）就是为此准备的“专业工作台”——它内置代码分析、版本控制等工具，适合管理微调项目这种稍复杂的工程。

**安装三步走：**

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费）下载。
3. 运行安装程序，**务必勾选** `Add launchers dir to the PATH`（将启动器目录添加到 PATH），方便在命令行直接启动。

**首次启动配置（关键两步）：**

1. 打开后选 `New Project`。
2. 在 `Location` 设置项目路径——**路径中不要含中文**，避免后续工具链出怪问题。`Interpreter` 选 `Previously configured interpreter`

**补充说明：** PyCharm 的 `

## 块 11

### 5.1 为什么用 Conda

你的目标是 LoRA 微调，这意味着要碰 PyTorch、CUDA 这类复杂依赖。如果直接在系统 Python 里装，很容易装乱、装冲突。Conda 就是来解决这个问题的——它给每个项目建一个独立的 Python 空间，还能管不同版本的 Python 解释器本身。对深度学习来说，这几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，直接看表：

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| **Anaconda** | 大（500MB+），预装 numpy、pandas、jupyter 等 | 做数据分析、不想手动装包 |
| **Miniconda** | 小（约50MB），只带 conda 和基础 Python | 项目开发、想要干净环境 |

**推荐 Miniconda**。你后面要装 PyTorch 这些，环境越干净越好，用什么装什么，不拖泥带水。

**安装步骤（跟着做）：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包：Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`
3. 运行安装程序：
   - **Windows**：一路下一步，**别勾** "Add Miniconda3 to my PATH environment variable"（安装器自己都标了"不推荐"）。直接塞进 PATH 容易和系统已有的 Python 冲突，到时候 `python`、`pip` 指向谁都不知道。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来敲 conda 命令就行。
   - **macOS/Linux**：运行 `.sh` 文件，输入 `yes` 同意协议、选安装路径，最后一步也输入 `yes` 让安装器执行 **`conda init`**。这样终端每次打开，conda 会自动接管，不用手动改 PATH。
4. **验证安装**：开一个**新**终端（Windows 用 "Anaconda Prompt (Min

## 块 12

**动手创建你的第一个环境**

你的目标是 LoRA 微调，而微调不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**拆解一下**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y` 并回车，等待安装完成

## 块 13

**本块学习检查**：激活环境后，运行 `python --version`，确认输出为 `Python 3.10.x`，再退出环境，观察命令行提示符变化。

---

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

**

## 块 14

## 6.1 VS Code 配置 Conda 环境

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
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件。可先运行 `conda env list` 查看环境路径，再定位到 `llm-learn` 环境对应的 `python` 可执行文件。

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都

## 块 15

你的目标是 LoRA 微调，但一切始于环境能跑通。本块用最快路径验证你的“工作台”。

**动手验证（命令行）**
1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 回车，看到 `>>>` 提示符。
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 并回车。
4. 看到文字打印，输入 `exit()` 退出。

**换条路再试（VS Code）**
1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建 `hello.py`，输入 `print("Hello from VS Code!")`。
3. 点右上角小三角或

## 块 16

### 7.3 在 PyCharm 中验证

你的目标是 LoRA 微调，这意味着你很快就要在真实项目中跑代码了。所以，**先确认你的开发工具能顺利运行代码**，这是动手的第一步。

在 PyCharm 里做一次快速验证，三步搞定：

1. **新建项目**，在项目中创建一个新文件，命名为 `hello.py`。
2. **运行它**：在 `hello.py` 文件上**右键点击**，选择 `Run 'hello'`。
3. **看结果**：PyCharm 底部会弹出 **Run 窗口**，里面应该显示你写的输出内容。

> **动手建议**：别只看，立刻操作一遍。如果 Run 窗口里出现了你预期的文字，说明你的“开发工作台”已经能正常运转了，可以放心进入下一步。

---

### 第八章：常见问题排查（动手时必看）

你动手时如果卡住，大概率是下面几种情况。对照表格快速解决：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选择 “Modify”，勾选 “Add Python to environment variables” |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示“未选择 Python 解释器” | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda
