# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

你的目标是部署 vLLM，但先别急着碰模型——所有 AI 工具链都跑在 Python 上，而解释器就是这一切的起点。

**它到底是什么？**

你写的 Python 代码是给人看的文字，电脑只认 0 和 1。解释器就是中间的"翻译官"：把你写的代码逐行翻译成电脑能执行的指令。没有它，你保存的 `.py` 文件对电脑来说就是一堆普通文本，什么也干不了。

**为什么这对 vLLM 很重要？**

vLLM 是一个 Python 库，它依赖解释器来运行。解释器装不好，后面所有环境配置都会出问题。所以这一步值得花 2 分钟搞明白，而不是盲目点"下一步"。

**动手验证一下（30秒）**

装好 Python 后，打开命令行（Windows 搜 `cmd`，Mac 打开"终端"），输入：

```
python --version
```

如果看到类似 `Python 3.x.x` 的输出，说明解释器已经就位，翻译官上班了。如果提示"找不到命令"，说明还没装好，回头检查安装步骤。

记住：**先有翻译官，代码才能跑起来**。这是你搭建 vLLM 工作台的第一块地基。

## 块 2

你的目标是部署 vLLM，这需要一套干净、可控的 Python 环境。本块先帮你把“工作台”搭对。

**IDE 选择**：直接装 **VS Code**。它轻量、免费，配合 Python 插件后，能提供语法高亮和自动补全，足够你后续写推理脚本。PyCharm 虽强大，但对快速上手 vLLM 来说偏重，暂不推荐。

**虚拟环境（关键）**：部署 vLLM 时，依赖库版本极易冲突。想象两个项目分别要 `numpy 1.0` 和 `2.0`，装一起就报错。**虚拟环境**就是给每个项目一个独立 Python 空间，互不干扰。

## 块 3

**目标：** 你的最终目标是部署 vLLM，而 Python 是这一切的地基。本块带你用最快、最稳的方式在 Windows 上装好它。

**第一步：下载安装包**

1.  浏览器打开 `https://www.python.org`。
2.  鼠标悬停顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3.  点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1.  **勾选最下方的 "Add python.exe to PATH"**：这会把 Python 加入系统"环境变量"。不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令行操作都无法进行。
2.  点击 **"Install Now"**（默认安装）即可。**注意：请检查默认安装路径是否包含中文（如用户名为中文时），若包含，请手动改为纯英文路径**，避免后续路径报错。进阶用户若想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1.  按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2.  输入 `python --version

## 块 4

### macOS 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 vLLM 依赖 Python 3.10+。macOS 系统自带的 Python 版本老旧且不可靠，所以**必须自己装一个干净的 Python**。

**方法一：官方安装包（推荐，最省心）**

1. 打开浏览器，访问 `https://www.python.org`，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点 “Continue” → “Install”，输密码完成安装。
3. 打开终端（Terminal），输入下面两条命令验证：

```bash
python3 --version
pip3 --version
```

看到版本号就成功了。

**关键提醒：用 `python3` 而不是 `python`**

现代 macOS 默认**没有** `python` 命令，只有 `python3`。如果你敲 `python`，会报“命令不存在”。所以从现在起，所有命令都用 `python3` 和 `pip3`，别混用。

**方法二：Homebrew（如果你已经装了）**

终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：打开终端，依次运行 `python3 --version` 和 `pip3 --version`，确认两个命令都输出版本号（而非报错）。如果 `

## 块 5

**目标：装好 Python 3 + pip，为 vLLM 部署铺路。**

Linux 常自带 Python，但版本偏老。直接跑下面三行，搞定 Ubuntu/Debian：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解**：
- 第一行：刷新软件源，确保能装到最新包。
- 第二行：一次装齐 Python 3、pip（装包工具）、venv（虚拟环境）。`-y` 表示自动确认，不用手动输 `y`。
- 第三行：验证版本，看到 `Python 3.x` 即成功。

**本块学习检查**：

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，这属于 GPU 实训，环境选择直接决定你能不能跑起来。先记住一条主线：**命令练习用 WSL2/Ubuntu LTS，GPU 实训用 Ubuntu LTS 实体机或云服务器**。

**日常写 Python、练命令行（Windows 用户）**

启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**

直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**vLLM 部署等 GPU 实训**

用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"** 当最佳方案。CentOS 8 已于 2021 年停止维护，CentOS 7

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**只有当网络受限、访问官方源明显变慢或经常超时时**，才考虑临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不会

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为 vLLM 部署做准备，后续要拉取大量依赖包。如果当前网络环境长期不稳定，每次都手动加 `-i` 参数确实低效——那就把清华镜像写进 pip 的全局配置，一劳永逸。

**但先记住一条底线**：官方 PyPI 永远是最可靠的来源。配置镜像只是权宜之计，网络恢复后务必改回或删除配置。

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 新建文件夹，命名为 `pip`。
3. 进入 `pip` 文件夹，新建文本文件，命名为 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：默认从这里下载包。

### macOS / Linux 操作步骤

1. 终端执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建并编辑配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

### ⚠️ 关于 `trusted-host` 的警告

网上很多教程会顺手加一行 `trusted-host = ...`，**不要加**。这行配置会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用完全不需要。只有企业内网自建源、或确认来源可信且确实遇到证书报错时，才临时加上。

### 验证配置是否生效

命令行执行：

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

**本块学习检查**：执行 `pip config list`，确认输出中包含 `global.index-url` 指向清华镜像。然后执行 `pip config unset global.index-url`

## 块 9

**本块学习检查**：完成 VS Code 安装后，打开软件，点击左侧活动栏最下方的四个方块图标，确认“扩展”面板能正常加载；再按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`），输入 `Python: Select Interpreter`，看能否看到你刚安装的 Python 版本（如 `Python 3.13.x`）。能完成这两步，说明你的开发台已就绪。

---

**为什么先装 VS Code**：你的目标是 vLLM 部署，后续要写 Python 脚本、改配置文件。VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端和配置文件编辑，学 Linux 命令行时同样能用。

**安装步骤（三步走）**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、

## 块 10

**本块目标**：装好 PyCharm Community，并完成首次启动配置，为后续 vLLM 项目开发准备好“工作台”。

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具，适合大型项目开发——vLLM 部署涉及多文件、多依赖，用它管理更顺手。

**安装步骤（跟着做）**：
1. 打开浏览器，访问 `https://www.jetbrains.com/pycharm/download/`。
2. 在页面上找到 **Community Edition**（社区版，免费），点击下载。
3. 运行下载好的安装程序，在安装选项中，务必勾选这两项：
   - `Add "Open Folder as Project"`：以后在文件夹上右键就能直接用 PyCharm 打开。
   - `Add to PATH`：这样在命令行里也能直接敲 `pycharm` 启动。

**首次启动配置（关键

## 块 11

### 5.1 为什么用 Conda（为 vLLM 部署打地基）

你的目标是部署 vLLM，这背后依赖 PyTorch、CUDA 等一堆复杂组件。如果直接在系统 Python 里乱装，很容易把环境搞成一团浆糊。Conda 就是来解决这个问题的：它能为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对于深度学习这种依赖复杂的场景，Conda 几乎是必备工具。

### 5.2 安装 Conda：选 Miniconda

Conda 有两个主流发行版：

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| **Anaconda** | 体积大（500MB+），预装 numpy、pandas 等一堆库 | 做数据分析、不想手动装包的人 |
| **Miniconda** | 安装包约 50-80MB，安装后占用空间约 400MB 以上，只带 conda 和基础 Python | 希望环境干净、按需装包的人 |

**推荐你选 Miniconda**。理由很直接：环境干净，后面 vLLM 需要什么依赖，就手动装什么，不拖泥带水。

### 安装步骤（跟着做）

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序，注意区分系统：
   - **Windows**：一路下一步，**千万别勾选 "Add Miniconda3 to my PATH environment variable"**（安装器自己都标了"不推荐"）。硬塞进 PATH 容易和系统已有的 Python 冲突，导致 `python`、`pip` 指向混乱。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来敲 conda 命令就行。
   - **macOS/Linux**：运行 `.sh` 文件，输入 `yes` 同意协议、选安装路径，最后一步**必须选 `

## 块 12

**动手创建你的第一个虚拟环境**

你的目标是部署 vLLM，而大模型项目常需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程，避免版本冲突。

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

输入 `y` 并回车

## 块 13

### 激活与退出环境：进入你的专属“工作间”

你的目标是部署 vLLM，而 Conda 环境就是你的专属“工作间”。激活环境，就是走进这个房间，让所有操作都在里面进行。

**激活环境**（进入房间）：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志。这意味着后续所有的 `python` 和 `pip` 命令都只作用于这个环境，不会干扰系统或其他项目——这对隔离 vLLM 的依赖至关重要。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

**本块学习检查**：执行 `conda activate llm-learn`，确认命令行出现 `(llm-learn)` 前缀；再执行 `python --version`

## 块 14

**本块学习检查**：完成下面两步，确认 VS Code 已正确绑定 `llm-learn` 环境。

1. 在 VS Code 中按 `Ctrl + Shift + P`，输入并选择 `Python: Select Interpreter`。
2. 在列表中找到带有 `conda` 字样的选项（如 `Python 3.10.x ('llm-learn': conda)`），选中它。

选中后，你在该工作区写的所有代码都会自动使用 `llm-learn` 环境里的 Python 和已安装的包——这正是你后续跑 vLLM 推理脚本的基础。

---

**动手配置（二选一，按你常用的编辑器来）**

**VS Code 路线**：
- 先打开终端，运行 `conda activate llm-learn` 激活环境。
- 回到 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
- 输入 `Python: Select Interpreter` 并回车。
- 从下拉列表中选择带 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**PyCharm 路线**：
- 打开项目，进入 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
- 左侧选 `Project: 你的项目名` → `Python Interpreter`。
- 点击齿轮图标 → `Add...`。
- 左侧选 `Conda Environment` → `Existing environment`

## 块 15

## 7.1 在命令行验证

你的目标是部署 vLLM，但任何部署都从“环境能跑通”开始。这一步就是你的**启动自检**。

**操作步骤（跟着做）：**

1. **激活环境**：输入 `conda activate llm-learn`，回车。
2. **进入 Python**：输入 `python`，回车。看到 `>>>` 提示符，说明 Python 已就绪。
3. **运行测试**：在 `>>>` 后输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
   回车。
4. **确认输出**：屏幕打印出这句话，说明环境正常。输入 `exit()` 退出。

**为什么这对 vLLM 重要？**

## 块 16

### 7.3 在 PyCharm 中验证

既然你的目标是快速上手 vLLM 部署，那我们先确保工作台能跑通最简单的代码，别让环境问题卡住你。

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目里新建一个 Python 文件，命名为 `hello.py`。
2. **运行文件**：在 `hello.py` 里随便写一行打印语句（比如 `print("hello")`），然后**右键点击该文件**，选择 **`Run 'hello'`**。
3. **查看输出**：PyCharm 底部会弹出 **Run 窗口**，里面应该显示你打印的内容。

如果这一步顺利，说明你的 PyCharm 和 Python 解释器已经正确关联，可以开始写真正的代码了。

---

### 第七章：常见问题排查（重点看前两条）

你之后装 vLLM 时大概率会遇到下面这些问题，提前知道怎么处理能省很多时间：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错
