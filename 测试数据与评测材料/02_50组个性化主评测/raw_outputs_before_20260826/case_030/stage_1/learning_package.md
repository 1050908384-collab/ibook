# 学习包

## 块 1

**1.1 Python 解释器：程序的"翻译官"**

你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。**Python 解释器**的作用就是把你的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本。

**为什么你现在必须搞懂它？** 因为你的目标是部署 vLLM，而 vLLM 是跑在 Python 里的。解释器装不对，后面所有模型推理的代码都跑不起来。

**动手验证一下（30秒）：**
打开终端（Mac/Linux）或命令提示符（Windows），输入：
```bash
python --version
```
如果看到类似 `Python 3.x.x` 的输出，说明解释器已就位；如果提示"找不到命令"，说明还没装好，继续往下学即可。

记住：**解释器 = 翻译官**，你的代码靠它变成电脑能执行的指令。后面每一步安装，都是在给这位"翻译官"配齐工具。

## 块 2

**本块学习检查**：读完本节后，请合上资料，用一句话说出“为什么装完 Python 还要装 IDE 和虚拟环境”。说不出来就回头再看一遍。

---

你的目标是部署 vLLM，这需要你亲手操作环境。记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。选 VS Code 上手更快，适合你每天 2.5 小时的快节奏。

装完 IDE，下一个关键步骤是**虚拟环境**——这是你部署 vLLM 时避免依赖冲突的保命工具。想象你同时在做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。如果都装到同一个 Python

## 块 3

## 2.1 Windows 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 vLLM 依赖 Python 环境。下面三步，咱们直接上手装好。

**第一步：确认系统要求**

确保你的 Windows 是 **64 位**且版本不低于 **Windows 8.1**，否则可能无法安装或运行 Python 3.10+。

**第二步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标移到顶部 **Downloads** 菜单，网站会自动推荐 Windows 版本。
3. 点击下载 **Windows installer (64-bit)**。建议选择 **Python 3.10 或 3.11** 版本（vLLM 官方支持 Python 3.8–3.12，对 3.13 的支持尚不明确）。

**第三步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   不勾选的话，命令行里输入 `python` 会提示"不是内部或外部命令"，后续所有命令操作都会卡住。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续报错。想改路径

## 块 4

## 2.2 macOS 用户：装好 Python，为 vLLM 铺路

你的目标是 vLLM 部署，第一步是拿到干净的 Python 环境。macOS 上别依赖系统自带的 Python——现代 macOS（12.3 起）已不预装 Python 2，部分系统虽有旧版 `python3`，但那是给系统脚本用的，**强烈建议自己装最新版**。

**方法一：官方安装包（推荐，最快上手）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install"。
3. 打开终端（Terminal），输入：

```bash
python3 --version
pip3 --version
```

看到版本号输出，说明装好了。

**关键提醒：用 `python3` 而不是 `python`**。现代 macOS 命令行默认不提供 `python` 命令，系统与官方安装包统一用 `python3`。始终用 `python3` / `pip3`，避免"命令不存在"或误用系统旧版的坑。

**方法二：Homebrew（已装过就用这个）**

如果你已有 Homebrew，终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

**本块学习检查

## 块 5

**动手装 Python：Ubuntu/Debian 三步走**

你的目标是部署 vLLM，而 Python 是它的地基。别担心，跟着敲就行。

1. **更新软件源**（让系统知道最新软件在哪）：
   ```bash
   sudo apt update
   ```

2. **安装 Python 3、pip 和 venv**（一条命令搞定）：
   ```bash
   sudo apt install python3 python3-pip python3-venv -y
   ```
   `-y` 是自动确认，不用手动按 `y`。`python3-venv` 是建虚拟环境用的，提前装好，省得后面补。

3. **验证安装**：
   ```bash
   python3

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，这意味着环境选择直接决定后续能否顺利跑通 GPU 实训。这里给你三条路线，按需取用：

**日常练手（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得接近真实生产的 Linux 环境，命令、路径、包管理与服务器一致，比 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常练手（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**GPU 实训（vLLM 部署的关键）**：使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。vLLM 依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 避坑提醒**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已停止维护、生态落后；VirtualBox 默认**无法直通 NVIDIA GPU**，跑不了真正的

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来装第三方库（比如 `numpy`、`pandas`）。默认它从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，**应始终作为首选**。

只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切到国内镜像。**镜像只是"网络不好时的加速手段"，不是永久替换**——官方源才是默认。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方 PyPI 仍是默认源；网络恢复或镜像出问题时，无需清理

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你的目标是 vLLM 部署，这意味着你会频繁安装大型依赖包。如果网络长期受限，每次手动指定镜像太浪费时间，可以把它设为默认。**但请记住：官方 PyPI 最可靠，网络恢复后建议改回。**

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开配置文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从清华镜像下载包。

### macOS / Linux 操作步骤

1. 打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样的内容，保存退出（`Ctrl + O` 保存，`Ctrl + X` 退出）。

### ⚠️ 关于 `trusted-host` 的警告

网上很多教程会让你加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**此项。只有企业内网自建源或确认来源可信时才临时加。

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即配置成功。

### 改回官方源

删除 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**动手验证**：配置完成后，运行 `pip config list` 确认输出。然后尝试安装一个小包（如 `pip install requests`），观察下载速度是否明显提升。如果遇到证书报错，先检查镜像地址是否拼写正确，再考虑是否真的需要 `trusted-host`。

## 块 9

**目标导向**：你正在为 vLLM 部署铺路，VS Code 是写推理脚本、调配置文件的趁手工具。它免费、启动快，装完插件后能直接识别 `.py` 文件，给你代码补全和错误提示——这些在你调试 vLLM 调用代码时会省下大量时间。

**动手装（约10分钟）**：

1. 打开 https://code.visualstudio.com，下载你系统的安装包，运行后一路“下一步”即可。
2. 装完后，点左侧活动栏最下方的**四个方块图标**（扩展），搜索并安装这两个：
   - **Python**（微软官方，提供语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更聪明）
3. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`），输入 `Python: Select Interpreter`，选你刚装的 Python 版本（如 `Python 3.13.x`）。

**本块学习检查**：装完插件后，新建一个 `test.py` 文件，输入 `print("ok")`，

## 块 10

**本块学习检查**：安装完成后，用 PyCharm 新建一个项目，确认你能在右下角看到 Python 3.10+ 的解释器路径。

---

**目标**：为 vLLM 部署搭好“工作台”——装一个专业又免费的代码编辑器。

**为什么选 PyCharm Community**：它是 Python 专用工具，免费、开箱即用，内置代码分析和版本控制，适合你后面写 vLLM 推理脚本这种稍大的项目。

**安装（3步）**：
1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费那个）下载。
3. 运行安装包，**务必勾选**这两项：
   - `Add "Open Folder as Project"`——以后右键文件夹就能直接打开。
   - `Add to PATH`——方便在命令行里启动。

**首次启动（3步）**：
1. 打开 PyCharm

## 块 11

## 5.1 为什么用 Conda

你的目标是部署 vLLM，这背后涉及 PyTorch、CUDA 等一堆复杂依赖。如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建独立环境，还能管理不同版本的 Python 解释器本身——这对深度学习项目几乎是必备的。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 安装包约 500MB-1GB，安装后占用空间通常超过 3GB，预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 安装包约 50-80MB，只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境更干净，后续需要什么装什么，正好符合你按需部署 vLLM 的节奏。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 conda 通过初始化脚本正确接管

## 块 12

**目标**：为 vLLM 部署做准备，先学会用 Conda 隔离项目环境。

**为什么现在做**：你之后跑 vLLM 时，不同项目会需要不同版本的 PyTorch 和 CUDA。现在学会建环境，后面直接复用这套流程，避免依赖冲突。

**动手操作**：打开终端，执行下面这行命令：

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

**目标**：让 VS Code 和 PyCharm 都指向你为 vLLM 部署准备的 `llm-learn` 环境。

**VS Code 配置（3 步）**
1. 先激活环境：`conda activate llm-learn`
2. 在 VS Code 中按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`
3. 选择带 `conda` 字样的路径（如 `Python 3.10.x ('llm-learn': conda)`）

选好后，工作区代码就会用 `llm-learn` 里的 Python 和已装包。

**PyCharm 配置（5 步）**
1. `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`）
2. 左侧 `Project: 项目名` → `Python Interpreter`
3. 齿轮图标 → `Add...`
4. 选 `Conda Environment` → `Existing environment`
5. 填 Python 路径：
   - Windows：`%USERPROFILE%\miniconda3\envs\llm-learn\python.exe`（将 `%USERPROFILE%` 替换为你的实际用户目录）
   - macOS/Linux：`$HOME/miniconda3/envs/llm-learn/bin/python`（将 `$HOME` 替换为你的实际主目录）

**本块学习检查**：配置完成后，在 VS Code 终端输入 `conda activate llm-learn`，再运行

## 块 15

**本块学习检查**：完成下方两步操作后，在终端确认输出结果与预期一致。

你的目标是部署 vLLM，但先别急着碰模型——环境没跑通，后面全是坑。咱们用最快方式验证 Python 工作台是否就绪。

**第一步：命令行验证（30秒）**
1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 回车，看到 `>>>` 提示符
3. 粘贴这行，回车：
```python
print("Hello, Python! 我的环境准备好了！")
```
4. 看到文字打印出来，输入 `exit()` 退出

**第二步：VS Code 验证（1分钟）**
1. 新建文件夹 `my-first-project`，用 VS Code

## 块 16

### 7.3 在 PyCharm 中验证

你的目标是 vLLM 部署，但任何部署都从“能跑通一行代码”开始。这一步就是你的第一个实战检查点。

1. **新建项目**：打开 PyCharm，`File` → `New Project`，创建后右键点击项目文件夹，选择 `New` → `Python File`，命名为 `hello.py`。
2. **运行**：在 `hello.py` 中输入 `print("Hello, vLLM!")`，然后**右键点击文件** → 选择 **`Run 'hello'`**。
3. **验证**：下方弹出 **Run 窗口**，看到输出的 `Hello, vLLM!` 即表示环境链路（Python → PyCharm → 解释器）已打通。

> **本块学习检查**：修改 `hello.py` 内容为 `print(1+1)`，再次右键运行，确认 Run 窗口输出 `2`。这能验证代码修改→运行→反馈的完整循环，是后续调试的基础。

---

## 第八章：常见问题排查（重点看前两行）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python |
