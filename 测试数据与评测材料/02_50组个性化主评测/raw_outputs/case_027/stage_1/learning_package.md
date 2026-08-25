# 学习包

## 块 1

**本块学习检查**：读完本节后，合上资料，用自己的话说清楚“Python 解释器”是干什么的；如果说不出来，回头重读一遍。

---

### 1.1 Python 解释器：程序的“翻译官”

你的目标是用 vLLM 部署模型，但先别急着碰那些高级工具——**所有 Python 项目的地基都是解释器**。没有它，你写的代码只是一堆无法运行的文本。

**为什么你需要它？**

你写的 Python 代码是人类语言，但电脑只认 0 和 1。解释器就是中间的“翻译官”，负责把你的代码翻译成电脑能执行的指令。

**一个关键认知：**

你写的 `.py` 文件本身**不是程序**，只是文本。只有经过解释器翻译，电脑才能运行它。所以，装好解释器是第一步，也是后续所有操作（包括 vLLM 部署）的前提。

**动手验证（30秒）：**

打开你的终端或命令行，输入 `python --version` 并回车。如果看到版本号，说明解释器已就绪；如果提示“找不到命令”，说明还没装好，继续往下看安装步骤。

## 块 2

**你的目标：vLLM部署。** 装好IDE和虚拟环境，是后面跑模型推理的地基。

**1.2 代码编辑器（IDE）：你的"工作台"**

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。你追求快速上手，选 VS Code 即可。

**1.3 虚拟环境：项目的"独立房间"**

想象你同时在做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。如果

## 块 3

### Windows 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 Python 是这一切的地基。我们直接上手，按步骤来。

**第一步：下载安装包**

1. 浏览器访问 https://www.python.org
2. 鼠标移到顶部 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径通常不含中文和特殊符号，但若用户名包含中文或空格，建议手动修改安装路径，确保路径中不含中文和空格。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 显示 `Python 3.13.x`（或你下载的版本号），说明安装成功！
4

## 块 4

### macOS 用户：安装 Python 3.10+

你的目标是部署 vLLM，所以先把 Python 环境弄干净。macOS 12.3 起不再预装 Python 2，且系统自带的 python3 版本可能较旧，建议自行安装最新版，别依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 完成。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**关键坑**：现代 macOS 默认**没有** `python` 命令，系统统一用 `python3` / `pip3`。始终用这两个，别踩"命令不存在"的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

已装 Homebrew 的话，终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：在终端运行 `python3 --version`，确认输出版本号（如 `Python 3.x.x`）。若提示 `command not found`，说明安装未成功，回头检查安装步骤。

## 块 5

你的目标是部署 vLLM，但这一步的前提是先把 Python 环境弄干净。Ubuntu/Debian 通常自带 Python，但版本偏老，直接装 vLLM 容易踩坑。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

第一行刷新软件源，第二行一次装齐 Python、pip 和虚拟环境模块。`-y` 表示自动确认，不用手动敲 `y`。`python3-venv` 是后面建虚拟环境必需的，现在装好省得回头补。

**本块学习检查**：执行完 `python3 --version`，确认输出以 `Python

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，这是 GPU 密集型任务，环境选错会直接卡住。按阶段分：

- **日常练命令（Windows）**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。管理员 PowerShell 执行 `wsl --install` 即可。它给你接近真实生产的 Linux 环境，命令和路径与服务器一致，比 CMD 更实战。
- **日常写 Python（macOS/Linux）**：直接用本机终端，本身就是类 Unix。
- **vLLM 部署等 GPU 实训**：用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 生态最成熟。

> **⚠️ 别踩坑**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于2021年12月停止维护，CentOS Stream 8 也于2024年5月停止维护；VirtualBox 默认**无法直通 NVIDIA GPU**，跑不了真 GPU 训练，只适合纯 CPU 小实验。

**实操

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是最权威、最完整、最应保留的默认源。

**只有**当你网络受限、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响你

## 块 8

## 3.3 进阶：把镜像设为默认——长期网络受限才这么做

你正在为 vLLM 部署做准备，后面会频繁用 pip 装大模型推理相关的大体积依赖包。如果网络长期不稳定，每次手动加 `-i` 镜像地址太烦，那就直接写进配置文件，一劳永逸。

**但先记住**：官方 PyPI 永远是最可靠的。网络恢复后，记得改回或删掉这个配置。

---

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车，会打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 在里面**新建一个 `pip` 文件夹**。
3. 在 `pip` 文件夹里**新建 `pip.ini` 文本文件**。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：默认从这个地址下载包。

---

### ⚠️ 别乱加 `trusted-host`

网上很多教程会让你顺手加一行 `trusted-host = ...`。**除非镜像确实过不了 HTTPS 证书验证，否则不要加**——加了等于让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里这些主流镜像都有合法 HTTPS 证书，正常用**不需要**。只有企业内网自建源、或确实报证书错误且确认来源可信时，才临时加。

---

### macOS / Linux 操作步骤

1. 终端执行：

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

4. `Ctrl + O` 保存，`Ctrl + X` 退出。

---

### 验证配置是否生效

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 就成功了。

---

### 改回官方源

删除刚才创建的 `pip.ini

## 块 9

**你的目标：** 快速搭好写 Python 代码（尤其是为 vLLM 部署做准备）的工作台。VS Code 轻量、免费，启动快，插件多，后续写脚本、改配置都用得上。

**安装（2分钟搞定）**
1. 打开 https://code.visualstudio.com，下载你系统的安装包。
2. 运行安装程序，一路“下一步”即可。

**装完必做：装两个插件**
点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（微软官方）：让 VS Code 能识别 `.py` 文件，提供语法高亮、调试、一键运行。
- **Pylance**：让代码提示更智能，写代码时少敲错。

**关键一步：选对解释器**
1. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择你刚装的 Python 版本（如 `Python 3.13.x`）。

**本块学习检查**：装完插件后，新建一个 `test.py` 文件

## 块 10

**目标对齐**：你正在为 vLLM 部署做准备，PyCharm 社区版能帮你高效管理大型推理项目代码。本块聚焦“装好、配好、能跑”，全程动手。

**为什么选它**：JetBrains 出品，Python 专用，社区版免费。内置代码分析、版本控制、数据库工具，适合大型项目（如 vLLM 部署脚本）开发。

**安装步骤（动手做）**：
1. 浏览器打开 `https://www.jetbrains.com/pycharm/download/`
2. 下载 **Community Edition**（社区版，免费）
3. 运行安装程序，**务必勾选**：
   - `Add "Open Folder as Project"`（右键快速打开项目文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置（动手做）**：
1. 打开 PyCharm → 点 `New Project`
2. `Location` 选项目路径，**路径中不要含中文**（避免编码坑）
3. `Interpreter` 选 `Previously configured interpreter`

## 块 11

## 5.1 为什么用 Conda

你之前已经知道，虚拟环境能为每个项目创建独立的 Python 空间。而 **Conda** 是目前最主流的虚拟环境管理工具——它不仅能创建虚拟环境，还能直接管理不同版本的 Python 解释器本身。这一点对你接下来的 vLLM 部署特别关键：深度学习项目往往依赖 PyTorch、CUDA 这类复杂依赖，用 Conda 管理环境几乎是必备技能。

## 5.2 安装 Conda：Anaconda vs Miniconda

安装 Conda 有两个选择，先看对比：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐你选 Miniconda**。理由很简单：你的目标是 vLLM 部署，环境越干净越好，需要什么装什么，避免无关包干扰后续的依赖管理。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器本身也标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`

## 块 12

**创建你的第一个虚拟环境**

你要部署 vLLM，后面会频繁切换不同版本的 PyTorch 和 CUDA。现在花两分钟建好环境，以后直接复用这套流程。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**拆解一下**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定装 Python 3.10

执行后，终端会列出待安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y` 回车，等安装完成即可。

**本块学习

## 块 13

**激活环境（进入这个“独立房间”）**  
```bash
conda activate llm-learn
```  
激活后，命令行前会出现 `(llm-learn)` 标志，表示后续 `python` 和 `pip` 命令都作用于此环境，不影响系统或其他项目。

**验证当前环境的 Python 版本**  
```bash
python --version
```  
应显示 `Python 3.10.x`。

**退出环境（回到系统默认环境）**  
```bash
conda deactivate
```

**查看所有已创建的环境**  
```bash
conda env list
```  
带 `*` 的是当前激活的环境。

**删除环境（如后续需清理）**  
```bash
conda remove -n llm-learn --all
```

---

**本块学习检查**  
1. 执行 `conda activate llm-learn`，观察命令行前缀变化。  
2. 运行 `python --version`，确认版本为 3

## 块 14

**本块学习检查**：完成配置后，在 VS Code 或 PyCharm 中新建一个 Python 文件，输入 `import sys; print(sys.executable)` 并运行，观察输出的路径中是否包含 `llm-learn`。若包含，说明当前解释器已正确指向 Conda 环境。

---

**VS Code 配置 Conda 环境（动手操作）**

1. **激活环境**：在终端运行 `conda activate llm-learn`，确保后续操作基于该环境。
2. **打开命令面板**：在 VS Code 中按 `Ctrl + Shift + P`。
3. **选择解释器**：输入并选择 `Python: Select Interpreter`。
4. **确认 Conda 路径**：在列表中选择带有 `conda` 字样的 Python 版本（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样做**：VS Code 能自动识别 Conda 环境。选定后，当前工作区的代码将使用 `llm-learn` 环境中的 Python 及已安装的包（如后续 vLLM 部署所需的依赖）。

---

**PyCharm 配置 Conda 环境**

1. **打开设置**：点击 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`）。
2. **进入解释器设置**：左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. **添加解释器**：点击齿轮

## 块 15

**动手验证：让环境“开口说话”**

你已经装好了工作台，现在来确认它真的能用。这一步很关键，因为后续所有 vLLM 部署操作都建立在这个环境之上。

**操作路径（命令行）**

1. 激活你的专属环境：`conda activate llm-learn`
2. 输入 `python` 并回车，进入交互模式。
3. 看到 `>>>` 提示符后，输入：`print("Hello, Python! 我的环境准备好了！")`
4. 回车，屏幕正确打印出这句话，就说明环境正常。输入 `exit()` 退出。

**操作路径（VS Code）**

1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`

## 块 16

## 7.3 在 PyCharm 中验证你的第一行代码

既然你的目标是 vLLM 部署，那环境验证这一步就是你的“发射前检查”。别小看它——后面跑模型推理时，任何环境问题都会在这里提前暴露。我们直接上手，三步搞定：

1. **新建项目，创建 `hello.py`**  
   打开 PyCharm，新建一个项目，然后在项目目录下新建一个 Python 文件，命名为 `hello.py`，输入以下代码：
   ```python
   print('Hello, vLLM!')
   ```

2. **右键点击文件 → `Run 'hello'`**  
   在 `hello.py` 文件上点右键，选择 `Run 'hello'`。PyCharm 会自动配置运行环境并执行。

3. **下方 Run 窗口应输出结果**  
   看屏幕下方的 **Run** 窗口，你应该能看到 `Hello, vLLM!`。看到输出，说明你的开发工作台已经全线贯通——Python 解释器、IDE、项目配置全部正常。

---

## 第八章：常见问题排查（重点看前两条）

你的目标是 vLLM 部署，后面要装大量依赖包，所以**网络和 pip 的问题是你最可能踩的坑**。先记住这两条：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |

其他问题（conda 找不到、解释器未选择、Python 版本不对、import 报错）先有个印象，遇到时再回来看表。

---

## 本章小结：你的环境清单

完成本章后，你的电脑上应该具备以下内容：

| 组件 | 状态 |
