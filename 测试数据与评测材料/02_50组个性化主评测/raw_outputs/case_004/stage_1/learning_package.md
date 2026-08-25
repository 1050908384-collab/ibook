# 学习包

## 块 1

好的，我们直接切入正题。你的目标是部署 vLLM，这需要你亲手操控一个强大的 Python 环境。所以，我们不只是“了解”解释器，而是要把它当成你工作台上的核心工具来认识。

**1.1 Python 解释器：程序的“翻译官”**

想象一下，你写的 Python 代码（比如一个用于控制模型推理的脚本）是人类语言，但你的电脑（GPU 服务器）只懂 0 和 1 的机器语言。**Python 解释器**，就是那个在你和电脑之间，把代码实时“翻译”成机器指令的“翻译官”。

没有这个“翻译官”，你精心编写的 `.py` 文件（比如 `deploy_vllm.py`）对电脑来说就是一堆无法理解的普通文本，更别提驱动 vLLM 进行推理了。

**动手验证一下：**
在终端里输入 `python` 并回车，你会看到 `>>>` 提示符，这就是解释器在等你“发号施令”。输入 `print("vLLM ready")` 并回车，如果它立刻输出 `vLLM ready`，就说明你的“翻译官”已经上岗，随时可以开始工作。

## 块 2

**本块学习检查**：读完本节后，请暂停并回答——IDE 和虚拟环境分别解决什么问题？如果你要部署 vLLM，需要用到哪个工具来隔离依赖？

---

你的目标是部署 vLLM，但先别急着碰模型。写代码不能靠记事本，你需要一个“工作台”——**IDE（集成开发环境）**。它提供语法高亮、自动补全、运行调试，让写代码像用 Word 一样顺手。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。选 VS Code 起步即可，够快够用。

更关键的是**虚拟环境**。想象两个项目：A 要 `numpy 1.0`，B 要 `n

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 Python 是这一切的地基。我们直接上手，快速搞定环境。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`，鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**。这一步至关重要：不勾选的话，之后在命令行输入 `python` 会提示"不是内部或外部命令"，所有后续操作都无法进行。
2. 点击 **"Install Now"**（默认安装）即可。默认路径不含中文和特殊符号，能避免后续路径报错。如果你想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入 `python --version` 并回车。如果显示 `Python 3.13.x`（或你下载的版本号），说明安装成功！
3. 再输入 `pip --version`，如果也显示版本号，说明包管理工具

## 块 4

## 2.2 macOS 用户：装好 Python，为 vLLM 铺路

你的目标是部署 vLLM，而 Python 是它的运行地基。macOS 上有个常见坑：**macOS Monterey 12.3 起系统不再预装 Python 2，Catalina 10.15 仍预装 Python 2.7，但建议使用最新版 Python 3**。部分系统可能带旧版 `python3`，但那是给系统脚本用的，**别直接用**。自己装最新版最稳妥。

**方法一：官方安装包（推荐，省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 搞定。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：现代 macOS 上命令行默认**没有** `python` 命令，只有 `python3`。所以**永远用 `python3` / `pip3`**，别敲 `python`，否则会报"命令不存在"或误用系统旧版本。

**方法二：Homebrew（你已装过就用这个）**

终端执行：

```bash
brew install python
```

装完同样

## 块 5

**本块学习检查**：运行 `python3 --version`，确认输出类似 `Python 3.10+` 的版本号，即安装成功。

---

你的目标是 vLLM 部署，而 vLLM 依赖较新的 Python 环境。Linux 预装版本常偏老，直接装新版本最稳妥。

**操作步骤**（Ubuntu/Debian）：

1. **更新软件源**：`sudo apt update`——刷新可用软件列表，避免装到旧包。
2. **安装 Python 3 全家桶**：  
   `sudo apt install python3 python3-pip python3-venv -y`  
   - `python3`：解释器本体。  
   - `python3-pip

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是 **vLLM 部署**，这属于 GPU 实训。选对环境，直接决定你后面能否顺利跑通模型。

**核心原则：命令练习和 GPU 实训，分开选环境。**

- **日常练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和服务器一致。安装只需管理员 PowerShell 跑 `wsl --install`。
- **日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。
- **vLLM 部署（GPU 实训）**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

> **⚠️ 避坑**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已停止维护（2021 年 EOL），CentOS 7 也已于 2024 年 6 月停止维护，建议使用 Ubuntu LTS 等

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来装第三方库（比如 `numpy`、`pandas`）。默认它从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**你的场景**：部署 vLLM 时，你会频繁安装大体积依赖包。如果发现下载卡顿、超时，才考虑换源。记住：**镜像是"网络不好时的临时加速手段"，不是永久设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

某次安装慢？最干净的办法是**只对这一次安装临时指定镜像**，不动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响你平时的默认行为，

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 vLLM 部署准备环境，下载依赖包是绕不开的一步。如果网络长期受限，每次都用 `-i` 参数指定镜像太麻烦，那就把它写进配置文件，让 pip 默认走镜像。

**⚠️ 前提提醒**：官方 PyPI 最可靠。网络恢复后，记得改回或删除配置。

---

**Windows 用户操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 新建文件夹，命名为 `pip`。
3. 在 `pip` 文件夹内新建文本文件，命名为 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

---

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

---

**⚠️ 关于 `trusted-host`**：网上教程常会加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**加。只有企业内网自建源或确认来源可信时才临时加。

---

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

---

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续要部署 vLLM，会频繁接触 Python 脚本和配置文件，它都能胜任。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：
1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python

## 块 10

**动手装 PyCharm，为 vLLM 部署备好工作台**

你的目标是快速上手 vLLM 部署，所以选 PyCharm Community 版最省心——免费且开箱即用，内置代码分析、版本控制等工具，适合管理后续的大型项目。

**安装三步走：**
1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费）下载。
3. 运行安装程序，务必勾选这两项：
   - `Add "Open Folder as Project"`（右键文件夹即可快速打开）
   - `Add to PATH`（命令行直接启动 PyCharm）
   - **安装路径中不要含中文**，避免后续部署报错。

**首次启动配置（关键）：**
1. 打开后选 `New Project`。
2. `Location` 设置项目路径——**路径中不要含中文**，避免后续部署报错。
3. `Interpreter` 选 `Previously configured interpreter`，

## 块 11

**为什么用 Conda（针对 vLLM 部署）**

你的目标是部署 vLLM，这意味着你马上要面对 PyTorch、CUDA 等一堆复杂依赖。Conda 就是为这种情况准备的：它不只创建虚拟环境，还能直接管理不同版本的 Python 解释器本身。对深度学习项目来说，这是近乎必备的工具。

**安装选型：Miniconda（推荐）**

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| Anaconda | 安装包约 1GB 以上，预装 numpy、pandas 等 | 数据分析、不想手动装包 |
| Miniconda | 约 70-80MB，只含 conda 和基础 Python | 项目开发、环境干净 |

你走项目开发路线，选 **Miniconda**——需要什么装什么，不拖泥带水。

**安装步骤（按你的系统操作）**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应安装包：Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`
3. 运行安装：
   - **Windows**：一路下一步，**别勾选** "Add Miniconda3 to my PATH environment variable"（安装器也标了"不推荐"）。直接塞进 PATH 容易和系统 Python 冲突，导致 `python`、`pip` 指向混乱。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 操作 conda。
   - **macOS/Linux**：运行 `.sh`，按提示输入 `yes` 同意协议、选安装路径，最后一步**必须选 `yes` 执行 `conda init`**。这样终端每次打开时 conda 会通过初始化脚本正确接管，而不是硬加 PATH。

4. **验证安装**：打开一个**新的**终端（Windows 必须用 Anaconda Prompt，不能用普通 cmd；macOS/Linux 重开终端），输入：

```bash
conda --version
```

看到类似 `conda 24.x.x` 的输出，就成功了。注意：Windows

## 块 12

## 5.3 创建你的第一个虚拟环境

**为什么现在做**：你后面要部署 vLLM，不同项目对 PyTorch 和 CUDA 版本要求不同。现在把环境创建练熟，后面直接复用。

打开终端，执行：

```bash
conda create -n llm-learn python=3.10
```

**逐段看懂它**：
- `conda create` —— 创建环境的命令
- `-n llm-learn` —— 给环境命名（`n` 就是 `name`）
- `python=3.10` —— 指定环境内安装 Python 3.10

执行后终端会列出待装包并询问：

```bash
Proceed ([y]/n)?
```

输入 `y` 回车，等待完成。

## 块 13

## 5.4 激活与退出环境（动手实操）

你的目标是 vLLM 部署，而 vLLM 对依赖版本非常敏感。Conda 环境就是你的“隔离工作台”——避免不同项目互相干扰。

**第一步：激活环境**

```bash
conda activate llm-learn
```

**验证是否成功**：激活后，命令行最前面会出现 `(llm-learn)` 标志。这意味着后续所有 `python` 和 `pip` 命令都只作用于这个环境，不会污染系统或其他项目。

**第二步：确认 Python 版本**

```bash
python --version
```

应显示 `Python 3.10.x`。这是你后续安装 vLLM 依赖的基础。

**第三步：退出环境**

```bash
conda deactivate
```

退出后，`(llm-learn)` 标志消失，回到系统默认环境。

---

**本块学习检查**：执行 `conda activate llm-learn` 后，观察

## 块 14

### 6.1 VS Code 配置 Conda 环境

你的目标是部署 vLLM，所以先确保 `llm-learn` 环境是激活状态。打开终端，运行：

```bash
conda activate llm-learn
```

接着打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板，输入并选择 `Python: Select Interpreter`。在弹出的列表中，找到带有 `conda` 字样的选项（例如 `Python 3.10.x ('llm-learn': conda)`），选中它。

**为什么这样做**：VS Code 会自动识别 conda 环境。一旦选中，你在当前工作区写的所有代码都会使用 `llm-learn` 环境中的 Python 解释器和已安装的包——包括你后续要用的 vLLM 相关依赖。

### 6.2 PyCharm 配置 Conda 环境

如果你更习惯 PyCharm，操作同样直接：

1. 点击 `File` → `Settings`（Mac 上是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中，选择你的 conda 环境目录下的 `python.exe`（通常位于 `.../envs/llm-learn/python.exe`），请根据实际安装位置调整。

## 块 15

**目标**：用命令行确认 Python 环境可用，为 vLLM 部署打好基础。

**操作步骤**（跟着做，2分钟搞定）：

1. 打开终端，激活环境：
   ```bash
   conda activate llm-learn
   ```
2. 确认终端提示符前出现 `(llm-learn)`，再输入 `python` 回车，看到 `>>>` 提示符。
3. 逐字输入（注意引号是英文的）：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
4. 回车，屏幕打印出这句话 = 环境正常。输入 `exit()` 退出。

**本块学习检查**：重新打开终端，不激活环境直接输入 `

## 块 16

**本块学习检查**：完成下面三步，确认你的 PyCharm 能跑通第一行代码。

---

### 7.3 在 PyCharm 中验证

你的目标是部署 vLLM，但任何部署都从“能跑一个脚本”开始。这一步就是验证你的开发台是否真正通电。

**操作步骤（跟着做，2分钟搞定）：**

1. **新建项目**：打开 PyCharm，点击 `File` → `New Project`，创建一个新项目（路径不要包含中文或空格）。
2. **创建文件**：在项目目录上右键 → `New` → `Python File`，命名为 `hello.py`。
3. **写代码**：在文件中输入 `print("Hello, vLLM!")`（这是你验证环境的第一行代码）。
4. **运行**：在编辑区右键点击文件 → 选择 `Run 'hello'`。
5. **看结果**：下方会弹出 `Run` 窗口，里面应输出 `Hello, vLLM!`。

**如果 Run 窗口没出现或报错**，先别慌，直接对照第八章的排查表——大部分问题都是环境配置的小坑，下面给你挑出最可能遇到的。

---

### 第八章：常见问题排查（重点看前两条）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows
