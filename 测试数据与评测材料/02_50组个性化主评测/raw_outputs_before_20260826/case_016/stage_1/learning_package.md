# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

你的目标是 QLoRA 微调，但别急着碰模型——先想清楚：你写的 Python 代码是人类语言，电脑只认 0 和 1。**Python 解释器**就是那个"翻译官"，把你的代码逐行翻译成电脑能执行的指令。

没有它，你写的 `.py` 文件只是一堆普通文本，电脑根本不知道你想干什么。

**为什么这和你有关？** QLoRA 微调本质上是运行大量 Python 脚本去操作模型权重。如果解释器没装对，你后面跑任何训练脚本都会在第一行就报错，而且报错信息往往指向"找不到 Python"——那不是模型的问题，是环境的问题。

**你现在只需要记住三件事：**

1. 解释器 = 翻译官，把代码翻译成机器指令
2. 没有解释器，`.py` 文件 = 纯文本，毫无作用
3. 后面所有安装步骤，都是为了让你拥有一个能正常工作的"翻译官"

**本块学习检查**：用自己的话说一遍"为什么没有 Python 解释器，你写的代码就跑不起来？"（提示：电脑只认识什么？代码是什么形式？）

## 块 2

### 你的“工作台”与“独立房间”：为 QLoRA 微调铺路

你马上要上手 QLoRA 微调，这活儿需要反复改代码、跑实验。记事本写代码？效率太低，调试起来能把你逼疯。所以，先搭好两个基础工具：

**1. 代码编辑器（IDE）——你的“工作台”**

IDE 就是给程序员用的“Word”，自带语法高亮（代码五颜六色，一眼看出拼写错误）、自动补全（少敲很多字）和运行调试（出错时能定位到具体行）。推荐两款：

- **VS Code**：轻量、免费，插件丰富，适合快速上手。
- **PyCharm**：更专业

## 块 3

**目标导向：** 你的目标是QLoRA微调，这意味着你很快就要在命令行里安装各种工具包。本块的任务就是确保你的Windows电脑能顺利运行`python`和`pip`命令，这是所有后续操作的地基。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。把鼠标移到顶部菜单的 **Downloads** 上，网站会自动推荐适合你Windows系统的版本（比如 `Python 3.13.x`，满足3.10+要求）。点击下载 **Windows installer (64-bit)** 这个文件。

**第二步：运行安装程序（关键！）**

双击下载的 `.exe` 文件。弹出安装界面后，**先别急着点安装**，做两件事：

1.  **勾选最下方的 "Add python.exe to PATH"**。这一步是重中之重。如果不勾选，你之后在命令行输入`python`，系统会提示“不是内部或外部命令”，所有后续操作都会卡死。
2.  然后点击 **"Install Now"**（默认安装）即可。默认路径不含中文和特殊符号，能避免后续很多路径报错。如果你非要改路径，务必确保新路径里**没有中文或空格**。

**第三步：验证安装**

1.  按 `Win + R` 键，输入 `cmd`，回车，打开命令提示符。
2.  输入 `python --version` 并回车。

## 块 4

### 2.2 macOS 用户：装好 Python，为 QLoRA 微调铺路

你的目标是跑通 QLoRA 微调，而第一步就是有个干净、可用的 Python 环境。macOS 上有个常见坑：**系统自带的 Python 版本老旧且不可靠**——macOS Monterey 12.3 起不再预装 Python 2，系统可能只有旧版 `python3`。**强烈建议自行安装最新版 Python**，别依赖系统自带版本。

#### 方法一：官方安装包（推荐，最省心）

1. 打开浏览器，访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入下面两条命令验证是否装好：

```bash
python3 --version
pip3 --version
```

看到版本号输出就说明成功了。若提示 pip 版本过旧，可运行 `pip3 install --upgrade pip`；若命令找不到，请检查 PATH 配置。

**关键提醒：为什么用 `python3` 而不是 `python`？** 现代 macOS 命令行默认**不提供** `python` 命令（早期版本曾指向系统自带的 Python

## 块 5

**本块学习检查**：运行 `python3 --version`，确认输出类似 `Python 3.10.x` 或更高版本号。

你的目标是 QLoRA 微调，Linux 环境是基础。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装齐三件套：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

- `update`：刷新软件源，确保能装到最新版。
- `-y`：自动确认，免去手动输入 `y`。
- `python3-venv`：虚拟环境模块，QLoRA 项目隔离依赖时必用，现在装好

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是 QLoRA 微调，这需要 GPU 实训环境。选对环境，能让你从“写代码”顺利过渡到“跑训练”。

**日常写 Python、练命令行（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在 Windows 里提供接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比在 Windows 原生 CMD 里练更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **“CentOS + VirtualBox 虚拟机”当最佳方案**。CentOS 8 已停止维护、生态落后；VirtualBox 桌面虚拟机**默认无法直通 NVIDIA GPU**，跑不了真正的 GPU 训练，只适合纯 CPU 的隔离小实验。

**本块学习检查**：确认你的实训环境属于

## 块 7

**目标**：QLoRA 微调需要装一堆库（如 transformers、peft），网络慢会卡住你。本块教你**只在必要时**用镜像加速，不搞乱默认环境。

**核心判断**：`pip` 默认从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，**永远首选**。只有当你**明显感觉下载慢或经常超时**（典型网络受限环境）时，才临时切镜像。记住：镜像是“临时加速手段”，不是永久设置。

**动手做法（推荐）**：某次安装慢，只对**这一次**临时指定清华镜像，不动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样好**：官方源仍是默认，平时行为不变；哪天网络恢复或镜像出问题，你无需清理任何配置。

**本块学习检查**

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

**这个操作只适合你这种长期网络受限的情况**——比如你准备用 QLoRA 微调模型，需要反复下载依赖包，每次都手动加镜像参数太麻烦。但记住：**官方 PyPI 才是最可靠的来源**，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车。这会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

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

如果你想恢复默认的官方 PyPI，删除

## 块 9

**动手装好 VS Code，为 QLoRA 微调铺路**

你的目标是跑通 QLoRA 微调，这需要你频繁修改 Python 脚本、查看报错、调试代码。VS Code 就是干这个的利器——它免费、启动快，而且插件生态丰富，后续你写训练脚本、改配置文件、看日志都用得上。

**安装三步走（5 分钟搞定）**

1. 打开 https://code.visualstudio.com，下载你系统的安装包。
2. 双击运行，一路“下一步”装完。
3. **关键一步**：装完先别急着写代码，先装两个插件。点击左侧活动栏最下方的**四个方块图标**（扩展商店），搜索并安装：
   - **Python**（微软官方出品）——装了这个，VS Code 才能识别 `.py` 文件，给你语法高亮、调试和智能提示。
   - **Pylance**（Python 语言服务器）

## 块 10

**动手装 PyCharm，为 QLoRA 微调备好“工作台”**

你的目标是跑通 QLoRA 微调，这需要写、改、调试较长的 Python 脚本。PyCharm Community 版免费且“开箱即用”，内置代码分析，能帮你少犯低级错误，适合直接上手。

**安装三步走：**
1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`。
2. 选 **Community Edition**（社区版）下载。
3. 运行安装包，务必勾选这两项：
   - `Add launchers dir to the PATH`：方便在命令行直接启动 PyCharm。
   - `Create Desktop Shortcut`：在桌面创建快捷方式，方便快速打开。

**首次启动配置：**
1. 打开 PyCharm，点 `New Project`。
2. `Location` 选项目存放路径，**路径别带中文**，避免后续工具链出怪

## 块 11

### 5.1 为什么用 Conda

你的目标是 QLoRA 微调，这意味着后面要装 PyTorch、CUDA 这类“重依赖”。Conda 不只是管 Python 包，它连 Python 解释器版本都能管——这在深度学习里几乎是刚需。简单说：**Conda = 每个项目一个独立小房间，互不干扰。**

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，直接给你结论：

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| **Anaconda** | 500MB+，预装 numpy、pandas 等一堆库 | 懒得装包、做数据分析的人 |
| **Miniconda** | 50MB，只有 conda 和基础 Python | **你**——要干净环境、按需装包 |

**选 Miniconda**，后面要啥装啥，不浪费空间。

**安装步骤（跟着做）：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS/Linux 选 `.sh`）
3. 运行安装程序，**关键区别在这**：
   - **Windows**：一路下一步，**建议勾选** "Add Miniconda3 to my PATH"，这样能在任意终端直接用 conda 命令。若担心与系统 Python 冲突，可保持默认不勾选，但需使用开始菜单里的 **"Anaconda Prompt (Miniconda3)"**。
   - **macOS/Linux**：运行 `.sh`，输 `yes` 同意协议、选路径，最后一步**必须输 `yes` 让 `conda init` 执行**。这样终端每次打开 conda 自动接管，不用手动改 PATH。
4. **验证**：**新开一个终端**（Windows 若没勾 PATH 用 Anaconda Prompt；macOS/Linux 关掉重开），输入：

```bash
conda --version
```

看到 `conda 24.x.x` 之类的输出，就成了。

---

**本块学习检查**：按上面

## 块 12

### 创建你的第一个虚拟环境：为 QLoRA 微调铺路

你的目标是 QLoRA 微调，而不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

**动手操作**：在终端执行以下命令：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（跟着敲，边敲边理解）：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包，

## 块 13

### 5.4 激活与退出环境：进入你的“独立房间”

你的目标是 QLoRA 微调，这意味着你会频繁切换不同的项目环境。把 Conda 环境想象成一个个独立的“房间”，每个房间有自己专属的 Python 版本和工具包，互不干扰。

**进入房间（激活环境）**：

```bash
conda activate llm-learn
```

**为什么这样写**：执行后，命令行前面会出现 `(llm-learn)` 标志，就像挂上了“房间门牌”。此时你运行的 `python` 和 `pip` 命令都只作用于这个环境，不会污染系统或其他项目——这对后续微调实验至关重要。

**验证你所在的环境**：

```bash
python --version
```

应显示 `Python 3.10.x`，确认你已进入正确的环境。

**离开房间（退出环境）**：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 QLoRA 微调，这意味着你马上要跑大模型训练脚本。先确保 VS Code 用的是正确的环境，否则代码会报“找不到包”的错误。

**操作步骤：**

1. 打开终端，先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板
3. 输入并选择 `Python: Select Interpreter`
4. 在列表中选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）

**为什么这样做**：VS Code 会自动识别 conda 环境。选好后，你在当前工作区写的代码就会使用 `llm-learn` 环境里的 Python 和已安装的包——包括你后续微调要用的库。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，操作略有不同：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`
3. 点击齿轮图标 → `Add...`
4. 左侧选择 `Conda Environment` → `Existing environment`
5. 在 `Interpreter`

## 块 15

### 在命令行验证你的环境

既然你的目标是QLoRA微调，环境验证是第一步。跟着做，很快就能跑起来。

1. **激活环境**：输入 `conda activate llm-learn`，回车。
2. **进入Python**：输入 `python`，回车。看到 `>>>` 提示符就对了。
3. **测试输出**：在 `>>>` 后输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
   回车后，如果屏幕上打印出这句话，说明环境正常。
4. **退出**：输入 `exit()` 回车。

**动手检查**：如果打印成功，说明你的命令行环境已就绪，可以继续下一步。如果报错，检查环境名是否拼写正确

## 块 16

## 7.3 在 PyCharm 中验证——跑通你的第一行代码

你已经装好了 Python、Conda 和 PyCharm，现在要做的只有一件事：**让代码真正跑起来**。这一步走通，后面 QLoRA 微调的所有实验才有落地的地基。

操作路径很短，跟着做：

1. 打开 PyCharm，**新建项目**，在项目里创建一个新文件，命名为 `hello.py`。
2. 在文件中输入一行最简单的打印语句。
3. **右键点击** `hello.py` 文件 → 选择 **`Run 'hello'`**。
4. 看 PyCharm **下方 Run 窗口**——如果输出了结果，说明你的开发工作台已经完整打通。

> 这一步验证的不只是“代码能运行”，更是“编辑器 → 解释器 → 输出”整条链路是否顺畅。QLoRA 微调后面要跑大量训练脚本，这条链路必须现在确认无误。

---

## 第八章：常见问题排查——卡住了先看这里

跑代码时遇到报错很正常，关键是快速定位。以下是环境搭建阶段最高频的几类问题：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda
