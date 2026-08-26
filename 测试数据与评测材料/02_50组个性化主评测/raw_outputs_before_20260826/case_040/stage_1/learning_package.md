# 学习包

## 块 1

**1.1 Python 解释器：程序的"翻译官"**

你的目标是 LoRA 微调，但别急着碰模型——先把这个"翻译官"装明白。

**它到底干嘛的？**  
你写的 Python 代码是人话，电脑只认 0 和 1。解释器就是中间的翻译官：把你写的 `.py` 文件逐行翻译成电脑能执行的指令。**没有它，你的代码只是一堆文本**，跑不起来。

**对你意味着什么？**  
后面做 LoRA 微调，你要写训练脚本、加载模型、调参数，每一步都靠解释器把你的 Python 代码"翻译"给电脑执行。现在装好它，后面才不卡壳。

**动手验证（2分钟）**  
装完后打开终端（Mac/Linux）或命令提示符（Windows），输入 `python` 回车。如果看到 `>>>` 提示符，说明翻译官已上岗。输入 `

## 块 2

**你的目标：LoRA 微调。** 这意味着你很快要接触 HuggingFace 生态，而这一切都建立在“能顺畅运行 Python 代码”的基础上。本块帮你把工作台搭好，不绕弯路。

**1.2 代码编辑器：选 VS Code 就够了**

写代码用记事本效率极低。IDE（集成开发环境）提供语法高亮、自动补全、运行调试，像用 Word 写文档一样舒服。推荐 **VS Code**：轻量、免费，对 Python 和后续的 LoRA 脚本支持极好。

> **动手做**：下载安装 VS Code，打开后按 `Ctrl+Shift+X`，搜索并安装官方“Python”扩展。装完即用。

**1

## 块 3

### Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，而微调的第一步是让 Python 在你的电脑上跑起来。跟着下面的步骤，动手装好它。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的 Windows 版本（如 `Python 3.x.x`）。
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

你的目标是 LoRA 微调，这需要 Python 环境。macOS 上别用系统自带的 Python——现代 macOS（Catalina 10.15 起）已不预装 Python 2，部分系统自带的旧版 `python3` 仅供系统脚本使用。**强烈建议自行安装最新版**，避免依赖系统版本。

#### 方法一：官方安装包（推荐新手）

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 上，命令行默认**不提供** `python` 命令（早期版本曾指向系统自带的 Python 2，现已移除）。系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

#### 方法二：Homebrew（适合已有 Homebrew 的开发者）

如果你已安装 Homebrew（macOS 包管理器），在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version`

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是跑 LoRA 微调，但先别急着碰模型——得确保 Linux 上有能用的 Python 环境。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省心。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：第一行刷新软件源，让系统知道最新包；第二行一次装齐 Python 本体、pip（装库的工具）和 venv（建虚拟环境用），`-y` 表示自动确认，不用手动敲 `y`；

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，那环境选择直接决定你后面能不能顺利跑通代码。别纠结，按下面两步走：

**第一步：日常练 Python 和命令行**
- **Windows 用户**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。在管理员 PowerShell 里执行 `wsl --install` 即可。这里练的命令、路径、包管理方式，和以后服务器上完全一致，比 Windows 原生 CMD 更贴近实战。
- **macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境，不用额外配置。

**第二步：LoRA / QLoRA 微调实训**
- 用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

> **⚠️ 避坑**：别用 **"CentOS + VirtualBox 虚拟机"**。CentOS 8 已于2024年5月31日停止维护（EOL），生态落后，可考虑 CentOS 9 Stream 或 Rocky Linux 等替代方案；

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认它从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、应保留的默认源**。

只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是"网络不好时的临时加速手段"，不是永久替换默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：它不影响你平时的默认行为，官方 PyPI 仍是默认源；网络

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为 LoRA 微调做准备，后面会频繁用 pip 安装 HuggingFace 生态的库。如果每次下载都卡在网络上，确实很耽误事。但请记住：**只有当你长期处于网络受限环境、几乎每次都要用镜像时，才这样做**。官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除配置。

**Windows 用户操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开配置文件夹（路径类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

**macOS / Linux 用户操作步骤：**

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

###

## 块 9

## 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，要频繁编辑 Python 脚本和配置文件，它都能胜任。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**（这是关键，别跳过）：

打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

> 为什么必须装？安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。没有它，你写微调脚本时会寸步难行。

## 块 10

**本块学习检查**：安装完成后，在 PyCharm 中新建一个项目，确认 `Location` 路径无中文，并检查 `Interpreter` 是否指向你已安装的 Python 3.10+。若成功，说明你的开发工作台已就绪，可继续 LoRA 微调的环境配置。

---

**PyCharm Community：专业、开箱即用**

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具，适合大型项目开发——包括你即将进行的 LoRA 微调实验。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费）
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置**：
1. 打开 PyCharm

## 块 11

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，这需要你亲手搭建一个不“打架”的 Python 环境。Conda 就是干这个用的。

**为什么必须用 Conda？**

简单说，它就是给你的每个项目一个独立的“隔间”。你之后做 LoRA 微调，需要 PyTorch、CUDA 这些依赖，版本很容易冲突。Conda 能帮你管好每个“隔间”里的 Python 版本和库，互不干扰。这是深度学习的必备技能。

**装哪个？选 Miniconda**

有两个选择：Anaconda 和 Miniconda。Anaconda 很大（500MB+），预装了很多库，适合不想动手装包的人。但我们要做项目开发，环境干净最重要，所以**推荐 Miniconda**（约 70-100MB），需要什么装什么。

**动手安装 Miniconda（跟着做）**

1.  打开浏览器，访问 `https://docs.conda.io/en/latest/miniconda.html`。
2.  下载对应你系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3.  运行安装程序，这里有个**关键区别**：
    *   **Windows 用户**：一路下一步，但**千万别勾选** “Add Miniconda3 to my PATH environment variable” 这个选项。勾了容易和你系统里已有的 Python 冲突。以后要用 conda，就打开开始菜单里的 **“Anaconda Prompt (Miniconda3)”** 这个专用终端。
    *   **macOS/Linux 用户**：运行 `.sh` 文件，按提示输入 `yes`，最后一步**必须输入 `yes`** 让安装器执行 `conda init`。这样每次打开终端，conda 就能自动接管，不用手动改配置。
4.  **验证安装**：**新开一个**终端（Windows 用 Anaconda Prompt，macOS/Linux 重开终端），输入：

```bash
conda --version
```

如果看到类似 `conda 24.x.x` 的版本号，就说明装好了。如果提示“command not found

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么先学这个**：你做 LoRA 微调时，不同项目会需要不同版本的 PyTorch 和 CUDA。现在把创建环境的流程练熟，后面直接复用，省去环境冲突的麻烦。

**动手做**：在终端里输入下面这行命令，然后回车：

```bash
conda create -n llm-learn python=3.10 -y
```

**这行命令在干什么**（拆开看）：
- `conda create`：告诉 Conda 你要新建一个环境
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 就是 `name` 的缩写）
- `python

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

既然你的目标是 LoRA 微调，我们得先把“工作台”搭好。这一步很关键——选对了环境，后面跑的代码才不会报“找不到包”的错。

**操作步骤（跟着做就行）：**

1. **先激活环境**：在终端里输入 `conda activate llm-learn`。这一步是告诉系统，我们接下来要用 `llm-learn` 这个环境。
2. **打开 VS Code**，按下 `Ctrl + Shift + P` 打开命令面板。
3. 输入 `Python: Select Interpreter` 并回车。
4. 在列表中选择带有 `conda` 字样的 Python 路径，例如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这样做**：VS Code 能自动识别 conda 环境。选好之后，你在这个工作区里写的所有代码，都会使用 `llm-learn` 环境里的 Python 和已安装的包。这样，你后面做 LoRA 微调时用到的库（比如 HuggingFace 生态的工具）才能被正确调用。

---

## 6.2 PyCharm 配置 Conda 环境

如果你用的是 PyCharm，操作也差不多：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac

## 块 15

**本块学习检查**：完成下方两个验证后，在终端输入 `exit()` 退出 Python，并记录你看到的输出结果。

---

你的目标是 LoRA 微调，但任何微调都从“环境能跑通”开始。本块只做一件事：**用两行代码确认 Python 环境正常**。

**第一步：命令行验证（最快反馈）**

1. 打开终端，激活环境：`conda activate llm-learn`
2. 输入 `python` 回车，看到 `>>>` 提示符即进入交互模式。
3. 输入：`print("Hello, Python! 我的环境准备好了！")`
4. 回车后若正确打印文字，输入 `exit()` 退出。

**第二步：VS Code 验证（日常开发

## 块 16

好的，我们直接进入正题。你的目标是 LoRA 微调，这意味着你最终要操作大模型，所以一个干净、可控的 Python 环境是绝对的地基。这一章就是帮你把这块地基夯实。

你的学习风格是“动手实操”，所以我们不空谈理论，直接带你走一遍流程。请打开 PyCharm，跟着下面的步骤来。

### 7.3 在 PyCharm 中验证：跑通你的第一行代码

这一步是为了确认我们之前安装的 PyCharm 和 Python 解释器能正常协同工作。

1.  **新建项目**：打开 PyCharm，点击 “New Project”。给项目起个名字，比如 `hello-world`。**注意**：项目路径不要包含中文或空格，这是为了避免后续很多奇怪的报错。
2.  **创建 Python 文件**：在项目目录上右键 -> New -> Python File，命名为 `hello.py`。
3.  **输入代码**：在 `hello.py` 中输入 `print("Hello, World!")`。
4.  **运行**：在代码编辑区右键，选择 `Run 'hello'`。或者，点击右上角的绿色三角形按钮。
5.  **查看结果**：看 PyCharm 底部的 “Run” 窗口，如果输出了 `Hello, World!`，恭喜你，你的开发工作台就正式搭建成功了！

---

### 第八章：常见问题排查（你的“避坑”指南）

在动手过程中，你几乎肯定会遇到下面这些问题。别慌，对照表格解决即可。

| 问题现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| 输入 `python` 提示“不是内部或外部命令” | 安装 Python 时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选择 “Modify”，勾选 “Add Python to environment variables” |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`
