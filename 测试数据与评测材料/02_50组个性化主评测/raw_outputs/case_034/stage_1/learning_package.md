# 学习包

## 块 1

**本块学习检查**：读完下面内容后，请合上资料，用一句话向自己解释“为什么没有解释器，`.py` 文件就只是一堆文本”。能说清楚，就继续；说不清，回头再看一遍。

---

### 1.1 Python 解释器：程序的“翻译官”

你写的 Python 代码，是给人看的文字；但电脑只认 0 和 1。这两者之间，靠**Python 解释器**来搭桥。

它的角色，就是**翻译官**：把你写的代码，逐句翻译成电脑能执行的指令。没有它，你辛苦写的 `.py` 文件，在电脑眼里就是一堆普通文本，什么也干不了。

所以，装 Python 环境，**核心就是装好这个“翻译官”**。后面你写的每一行代码，都要经过它才能变成电脑的动作。

现在你知道了：我们不是在装一个“软件”，而是在请一位“翻译官”上岗。接下来，我们就去把它请到你的电脑上。

## 块 2

### 1.2 代码编辑器（IDE）：你的“工作台”

你目标明确：理解Transformer架构。但动手前，得先把“工作台”搭好。记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用Word写文档一样舒服。推荐两款：**VS Code**（轻量免费）和**PyCharm**（专业强大）。选VS Code起步，够用且快。

### 1.3 虚拟环境：项目的“独立房间”

想象你同时做两个项目：A项目需要`numpy 1.0`，B项目需要`numpy 2.0`。都装进同一个Python会冲突报错。**虚拟环境**为每个项目创建独立Python空间，互不影响。用**conda**管理，这是目前最主流方案。Transformer项目建议单独建一个环境，避免依赖打架。

---

## 块 3

### 2.1 Windows 用户：装好 Python，开启你的 Transformer 之旅

你想动手理解 Transformer，那得先有把“锤子”——Python。别担心，跟着做，5分钟搞定。

**第一步：下载安装包**

1.  打开浏览器，访问 `https://www.python.org`。
2.  鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3.  点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1.  **勾选最下方的 "Add python.exe to PATH"**。这一步是“命门”。不勾选，之后在命令行敲 `python` 会提示“不是内部或外部命令”，后续所有操作都卡壳。
2.  点击 **"Install Now"**（默认安装）即可。默认安装路径通常为 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python313`，若用户名含中文或空格，建议自定义安装路径为纯英文无空格目录（如 `C:\Python313`）。

**第三步：验证安装**

1.  按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2.  输入以下命令并回车：

```bash
python --version

## 块 4

**macOS 安装 Python：两条路，任选其一**

现代 macOS（Catalina 10.15 起）不再预装 Python 2，部分系统虽有旧版 `python3`，但仅供系统脚本使用。**强烈建议自行安装最新版**，别依赖系统自带版本。

**方法一：官方安装包（新手首选）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 完成。
3. 打开终端，验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：现代 macOS 命令行默认**没有** `python` 命令（早期曾指向 Python 2，现已移除）。请始终用 `python3` / `pip3`，避免“命令不存在”或误用系统版本。

**方法二：Homebrew（已有包管理器者）**

终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：打开终端，依次运行 `python3 --version` 和 `pip3 --version`，确认两行均输出版本号（如 `Python 3.x.x`）。若报错，重走方法一第 2 步。

## 块 5

**动手装 Python：Ubuntu/Debian 三步走**

你的 Linux 可能自带 Python，但版本偏老。直接上命令：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：`update` 刷新软件源；`install` 一次装齐 Python、pip 和虚拟环境模块；`-y` 自动确认，省得手动输 `y`。装完用 `python3 --version` 验证。

**本块学习检查**：运行上述命令后，把终端显示的版本号记下来——这就是你 Python 工作台的起点。

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是理解Transformer架构，而动手跑通代码是关键。选对环境，就是为后续的PyTorch实操铺路。

**日常写Python、练命令（Windows用户）**：直接启用 **WSL2 + Ubuntu LTS**（如22.04/24.04）。它给你一个接近真实服务器的Linux环境，命令和包管理方式都和云端一致，比Windows自带的CMD更贴近实战。安装只需管理员PowerShell执行 `wsl --install` 即可。

**日常写Python（macOS/Linux用户）**：直接用本机终端，本身就是类Unix环境，无需额外配置。

**GPU实训（LoRA/QLoRA微调、vLLM部署）**：这类任务依赖NVIDIA驱动、CUDA和显卡直通，务必使用**装有Ubuntu LTS的NVIDIA GPU云服务器，或本地实体Linux（Ubuntu LTS）机器**。Ubuntu LTS对深度学习生态支持最成熟、文档最全。

> **⚠️ 避坑提醒**：别把 **“CentOS + VirtualBox虚拟机”** 当最佳方案。CentOS 8已于2021年12月31日停止维护（可改用CentOS Stream 8，其维护至2024年5月31日），生态落后；VirtualBox这类桌面虚拟机**默认无法直通NVIDIA GPU**，跑不了真正的GPU训练，

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、应保留的默认源**。

**什么时候才考虑镜像？** 只有当你网络受限、访问官方源明显变慢或经常超时，才临时切换到国内镜像。**镜像是“网络不好时的临时加速手段”，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐？** 它不影响你平时的默认

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

如果你**长期**待在没有外网或网速极慢的环境，每次装包都靠镜像，那才值得把镜像写进 pip 的默认配置。但请记住：**官方 PyPI 永远是最可靠的**，一旦网络恢复，建议立刻改回或删掉这个配置。

**Windows 操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在里面**新建一个 `pip` 文件夹**。
3. 进入 `pip` 文件夹，**新建一个 `pip.ini` 文件**（用记事本创建并保存）。
4. 用记事本打开 `pip.ini`，粘贴下面三行：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

**为什么写这三行？** `index-url` 就是告诉 pip：“以后默认去这个地址下载包”。`trusted-host` 用于标记该镜像为可信主机，避免因证书验证问题导致安装失败。

> **⚠️ 关于 `trusted-host` 的说明**  
> 网上很多教程会让你顺手加一行 `trusted-host = ...`，**除非镜像的 HTTPS 证书真的验证失败，否则不要加**。加了它，pip 就会跳过对该域名的 TLS 安全校验，等于把传输数据“裸奔”。清华、阿里这些主流镜像都有正规 HTTPS 证书，正常用**根本不需要**。只有企业内网自建源、或你确认来源可信且确实报证书错误时，才临时加上。若你确认镜像证书受信任，也可不加此行。

**macOS / Linux 操作步骤：**

1. 打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建并编辑配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样的三行内容：

```ini
[

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：
1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Inter

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具等功能，适合大型项目开发。

**安装步骤**：
1. 访问官网下载页。
2. 下载 **Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured interpreter`，然后选择系统安装的 Python 3.10+。

---

**本块学习检查**：动手操作——按上述步骤创建一个新项目，并确认解释器为 Python

## 块 11

### 5.1 为什么用 Conda

你之前已经知道，虚拟环境能为每个项目创建独立的 Python 空间。现在要学的 **Conda**，是目前最主流的虚拟环境管理工具。它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身——这对你后续学习 Transformer 架构至关重要，因为你需要安装 PyTorch、CUDA 等复杂依赖，Conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，直接看表：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 70MB，安装后占用约 400MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。原因很直接：环境更干净，后续需要什么包装什么包，不浪费空间，也不容易出冲突。

**安装步骤（跟着做）：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（

## 块 12

**动手创建你的第一个虚拟环境**

你正在为理解 Transformer 架构打基础，而大模型微调时，不同项目常需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程，省去大量环境冲突的麻烦。

打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**这条命令在做什么？拆开看：**

- `conda create`：告诉 Conda 你要创建一个新环境
- `-n llm-learn`：给这个环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定该环境内安装 Python 3.10 版本

执行后，终端

## 块 13

### 5.4 激活与退出环境：进入你的“独立房间”

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

## 第六章：在 VS Code / PyCh

## 块 14

**本块学习检查**：跟着下面的步骤亲手配置一次，完成后在 VS Code 终端运行 `conda env list`，确认 `llm-learn` 环境存在且已被选中。

---

### 6.1 VS Code 配置 Conda 环境

你的目标是让 VS Code 使用 `llm-learn` 环境里的 Python。操作很简单：

1. **先激活环境**：在终端输入 `conda activate llm-learn`。这一步确保后续操作基于正确环境。
2. **打开命令面板**：在 VS Code 中按 `Ctrl + Shift + P`。
3. **搜索解释器**：输入 `Python: Select Interpreter` 并回车。
4. **选择 conda 路径**：在列表中找到带有 `conda` 字样的项，例如 `Python 3.10.x ('llm-learn': conda)`，点击它。

**为什么这样写**：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的代码就会使用 `llm-learn` 中的 Python 和已安装的包——这正是你后续学习 Transformer 所需的运行基础。

---

### 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程类似：

1. 打开 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add

## 块 15

### 7.1 在命令行验证

1. 先激活环境：输入 `conda activate llm-learn`，回车。
2. 输入 `python` 并回车，进入交互模式。
3. 看到 `>>>` 提示符后，输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
4. 回车，若正确打印文字，输入 `exit()` 退出。

### 7.2 在 VS Code 中验证

1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`，输入：
   ```python
   print("Hello from VS Code!")
   ```
3. 点击右上角小三角（运行

## 块 16

### 7.3 在 PyCharm 中验证

打开 PyCharm，新建一个项目，然后在项目中创建一个名为 `hello.py` 的 Python 文件。在文件中输入你的第一行代码（比如打印一句问候语）。写好后，在文件编辑区**右键点击**，选择 **`Run 'hello'`**。此时，PyCharm 下方的 **Run 窗口**会自动打开并显示程序运行的结果。看到输出，就说明你的开发工作台已经成功跑通了！

---

## 第八章：常见问题排查

环境搭建过程中，你可能会遇到下面这些情况。别慌，对照表格快速定位解决：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

---

## 本章小结

完成本章后，你的电脑上应该具备以下内容：

| 组件 | 状态 | 验证命令 |
|------|------|----------|
| Python 3.10+ | ✅ 已安装
