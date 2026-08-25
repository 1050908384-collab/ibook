# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

先别急着下载安装包，花 2 分钟搞懂一个概念，后面每一步你都会清楚自己在做什么。

**问题出在哪？** 你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。这两者之间需要一座桥。

**Python 解释器就是这座桥。** 它的作用是把你的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆普通的文本，电脑根本看不懂。

**打个比方：** 你和一个只说外语的人交流，需要一位翻译官。Python 解释器就是你和电脑之间的翻译官——你写一句，它翻译一句，电脑执行一句。

**对你意味着什么？** 你后面要做的 QLoRA 微调，本质上是写 Python 代码让电脑去训练模型。如果解释器没装对，代码写得再好也跑不起来。所以这一步是地基。

**本块学习检查：** 用自己的话说说，为什么没有 Python 解释器，`.py` 文件就"只是一堆文本"？想清楚再继续往下学。

## 块 2

### 你的“工作台”与“独立房间”：为 QLoRA 微调铺路

既然你的目标是 **QLoRA 微调**，那代码量不小，调试也频繁。用记事本写代码，就像徒手搬砖——能搬，但效率极低。**IDE（集成开发环境）** 就是你的“工作台”，它提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费，适合快速上手）和 **PyCharm**（专业强大，适合复杂项目）。选 VS Code 起步即可，够用且不臃肿。

接下来是**虚拟环境**——你的“独立房间

## 块 3

**本块学习检查**：完成安装后，打开命令提示符，依次输入 `python --version` 和 `pip --version`，确认两个命令都能显示版本号。若 `python` 提示“不是内部或外部命令”，请回到第二步，重新运行安装程序，勾选“Add python.exe to PATH”后点击“Install Now”。

---

你的目标是QLoRA微调，这需要扎实的Python环境。本块带你完成Windows上的Python安装，全程动手操作，预计10分钟。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐Windows版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这是最关键的一步。勾选后，系统才能识别 `python` 命令。否则，后续你在命令行输入 `python` 时会提示“不是内部或外部命令”，所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续报错。若你修改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车

## 块 4

## 2.2 macOS 用户：安装 Python 3.10+

你的目标是 QLoRA 微调，这需要 Python 环境。好消息是，macOS 上安装 Python 很简单，跟着做就行。

**先记住一个关键点**：现代 macOS（Catalina 10.15 起）不再预装 Python 2，系统自带的 `python3` 可能是旧版，**别依赖它**。自己装最新版最稳妥。

**方法一：官方安装包（推荐，新手友好）**

1. 打开浏览器，访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点 "Continue" → "Install"，输密码完成安装。
3. 打开终端（Terminal，按 `Cmd + 空格` 搜索 "Terminal"），输入下面两行验证：

```bash
python3 --version
pip3 --version
```

看到版本号就成功了。

**为什么用 `python3` 而不是 `python`？**

这是新手最容易踩的坑。现代 macOS 上，命令行默认**没有** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统统一用 `python3`。所以**始终用 `python3` 和 `pip3`**

## 块 5

**动手装好 Python，为 QLoRA 铺路**

你目标是用 QLoRA 微调模型，但先得有能跑代码的 Python 环境。Ubuntu/Debian 通常自带 Python，但版本可能偏老，所以咱们直接装 Python 3.10 或更高版本。

打开终端，逐行执行以下命令：

```bash
# 1. 更新软件源，确保能拿到最新软件包
sudo apt update

# 2. 安装 Python 3.10、pip 和虚拟环境模块
sudo apt install python3.10 python3.10-pip python3.10-venv -y

# 3. 验证安装结果（确保版本 >= 3.10）
python3 --

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是 **QLoRA 微调**，这意味着你最终要在 GPU 上跑训练。选对环境，能让你从“练命令”到“跑模型”都少踩坑。按你的动手学习风格，我们直接按场景分：

**场景一：日常写 Python、练命令行（Windows 用户）**
启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**场景二：日常写 Python（macOS / Linux 用户）**
直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**场景三：QLoRA 微调、vLLM 部署等 GPU 实训（你的核心目标）**
用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别走弯路**：不要把 **"CentOS + Virtual

## 块 7

**本块学习检查**：读完本节后，请你在终端里实际运行一次 `pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`，观察它是否成功安装且没有改动你的全局配置。

---

你正在为 QLoRA 微调做准备，装 PyTorch 等库时如果卡在下载，多半是网络问题。`pip` 默认从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，**应始终保留为默认**。

**什么时候才用镜像？** 只有当你访问官方源明显变慢或经常超时时，才临时切换。镜像是“网络不好时的临时加速手段”，不是永久替换。

**首选做法：临时指定镜像源（推荐）**。只对某一次安装临时指定，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正在为 QLoRA 微调准备环境，下载依赖包是第一步。如果每次 `pip install` 都卡在“龟速”上，那确实很耽误事。但请记住：**只有当你长期处于网络受限环境、几乎每次都要用镜像时，才这么做**。官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除配置。

**Windows 用户操作：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开配置文件夹（类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

**macOS / Linux 用户操作：**

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

想恢复

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

**动手装 PyCharm，为 QLoRA 微调铺路**

你的目标是跑通 QLoRA 微调，那需要一个趁手的“工作台”。PyCharm Community 版免费且功能足够，正好适合你这种喜欢动手、节奏快的风格。

**安装三步走（跟着点就行）：**

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费的那个）下载。
3. 运行安装包，**务必勾选**这两项：
   - `Add "Open Folder as Project"` —— 以后右键文件夹就能直接打开项目，省事。
   - `Add to PATH` —— 方便在命令行中调用 PyCharm 的启动脚本（具体命令可能因系统而异）。

**首次启动，两个关键设置：**

- 新建项目（`New Project`）时，`Location` 路径建议使用英文，减少潜在兼容性问题。
- `Interpreter` 选 `Previously configured interpreter`，

## 块 11

你的目标是 **QLoRA 微调**，这意味着你马上要面对 PyTorch、CUDA 这些“重量级”依赖。如果直接在系统 Python 里装，很容易把环境搞乱，装错版本、冲突、卸载不干净……这些坑会浪费你大量时间。

**Conda 就是来解决这个问题的**。它不只是创建虚拟环境，还能直接管理 Python 解释器的版本。比如，你的项目 A 需要 Python 3.8，项目 B 需要 3.11，用 Conda 就能轻松切换，互不干扰。对于深度学习这种依赖复杂的场景，Conda 几乎是标配。

---

### 装哪个？Anaconda 还是 Miniconda？

| 选项 | 特点 | 适合谁？ |
|------|------|----------|
| **Anaconda** | 大而全（通常超过 500MB），自带 numpy、pandas 等 | 做数据分析、不想手动装包 |
| **Miniconda** | 小而精（约 50-80MB），只带 conda 和基础 Python | **你**：做项目开发，环境干净，需要什么装什么 |

**推荐 Miniconda**。你的目标是 QLoRA 微调，环境越干净，后面排查问题越容易。

---

### 安装 Miniconda（动手操作）

**第一步：下载**

打开 https://docs.conda.io/en/latest/miniconda.html ，下载对应你系统的安装包：
- Windows 选 `.exe`
- macOS 选 `.pkg` 或 `.sh`
- Linux 选 `.sh`

**第二步：安装（注意区分系统）**

- **Windows**：一路点“下一步”，**关键一步**：不要勾选 “Add Miniconda3 to my PATH environment variable”。这个选项安装器自己都标了“不推荐”。勾了容易和你系统里已有的 Python 冲突，导致 `python` 命令指向不明。装完后，从开始菜单打开 **“Anaconda Prompt (Miniconda3)”** 来用 conda，别用普通的 CMD。

- **macOS/Linux**：运行下载的 `.sh` 文件，按提示输入 `yes` 同意协议、选安装路径。最后一步

## 块 12

**动手创建你的第一个虚拟环境**

你的目标是 QLoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

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

## 块 13

### 5.4 激活与退出环境：进入你的“独立房间”

你的目标是跑通 QLoRA 微调，而 Conda 环境就是给这个项目准备的“独立房间”。激活它，就是走进这个房间，让所有命令都只在这个房间里生效，不干扰系统里的其他项目。

**激活环境**（走进房间）：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志。看到它，就说明后续的 `python` 和 `pip` 命令都作用在这个环境里了，你的 QLoRA 依赖（比如 PyTorch、Transformers）都会从这里加载，不会和系统全局的版本冲突。

**验证当前环境的 Python 版本**（确认房间配置正确）：

```bash
python --version
```

应该显示 `Python 3.10.x`。如果版本不对，说明环境没建好，先回头检查创建步骤。

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。

打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 会自动识别 conda 环境。选好后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包——包括你之后做 QLoRA 微调要用的 PyTorch 和 HuggingFace 工具链。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`%USERPROFILE%\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`$HOME/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

## 块 15

**本块学习检查**：完成下方两个验证步骤后，在终端输入 `exit()` 退出 Python，并确认 VS Code 终端输出了 `Hello from VS Code!`。

---

你的目标是 QLoRA 微调，但一切始于环境跑通。我们直接动手，两步验证你的“开发工作台”是否就绪。

**第一步：命令行验证（最快反馈）**

1. 打开终端，激活环境：`conda activate llm-learn`
2. 输入 `python` 并回车，看到 `>>>` 提示符即进入交互模式。
3. 输入：`print("Hello, Python! 我的环境准备好了！")`
4. 回车后若正确打印文字，输入 `exit()` 退出。注意

## 块 16

## 7.3 在 PyCharm 中验证——跑通你的第一行代码

你现在已经装好了 PyCharm，这一步我们直接动手，把环境跑通。别担心，整个过程就三步，跟着做就行。

**操作步骤：**

1. **新建项目**：打开 PyCharm，点击 `File → New Project`，创建一个新项目。
2. **创建 `hello.py` 文件**：在项目目录上右键 → `New → Python File`，文件名输入 `hello.py`。
3. **运行文件**：在 `hello.py` 中随便写一行代码（比如 `print("Hello")`），然后**右键点击文件** → 选择 **`Run 'hello'`**。

**验证结果**：看 PyCharm 下方的 **Run 窗口**，如果输出了你打印的内容，说明你的开发环境已经全部打通了——Python 装好了、解释器配置对了、PyCharm 能正常调用 Python 执行代码。

---

## 第八章：常见问题排查（遇到再看）

环境搭建过程中，你可能会遇到下面这些坑。**不用提前背**，遇到对应问题时回来查表即可。

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境
