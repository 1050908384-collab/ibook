# 学习包

## 块 1

**本块学习检查**：读完这一节后，请合上资料，用自己的话回答一个问题——"如果电脑不装 Python 解释器，我写的代码文件会怎样？"答不上来就回头再看一遍。

---

你现在的目标是 LoRA 微调，但先别急着碰模型。任何微调脚本，第一行代码都需要 Python 来解释。所以，咱们先把"翻译官"请进门。

**一句话理解**：你写的代码是给人看的，电脑只懂 0 和 1。Python 解释器就是那个把"人话"翻译成"机器话"的中间人。没有它，你辛苦写的 `.py` 文件，在电脑眼里就是一堆普通文本，啥也干不了。

**动手验证一下**（不用装任何东西，纯理解）：
1. 打开记事本，随便写一行 `print("hello")`，保存为 `test.py`。
2. 双击这个文件——你会发现它只是闪了一下或根本没反应。这就是因为没有"翻译官"来执行它。

**为什么这对你很重要**：你之后跑 LoRA 微调，要执行的是 `train.py` 这类脚本。如果解释器没装对，脚本连跑都跑不起来，更别说调参了。所以，现在花 2 分钟搞懂这个概念，后面每一步你都知道自己在做什么，而不是盲目点"下一步"。

**记住这个比喻**：解释器就是你的私人翻译官，你写代码，它负责让电脑照做。

## 块 2

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，那环境就是你的第一个战场。别怕，我们一步步来。

**1.2 代码编辑器（IDE）：你的“工作台”**

写代码就像做木工，记事本就是一把钝刀，能用但费劲。**IDE（集成开发环境）** 才是你的专业工作台，自带语法高亮、自动补全和调试功能，能让你把精力集中在 LoRA 逻辑本身，而不是和拼写错误搏斗。

推荐两款：
- **VS Code**：轻量、免费、启动快，装个 Python 插件就很好用。
- **PyCharm**：功能更专业、更强大，适合复杂项目。

**1.3 虚拟环境：项目的“独立房间”**

这是新手最容易踩的坑。想象你同时做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`

## 块 3

**动手装好 Python，为 LoRA 微调铺路**

你的目标是 LoRA 微调，这需要先有一个能跑 Python 的环境。别担心，跟着下面三步走，10 分钟就能搞定。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。把鼠标移到顶部菜单的 **Downloads** 上，网站会自动推荐适合你 Windows 的版本（比如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)** 这个文件。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载好的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**。这一步非常关键！如果不勾选，之后在命令行里输入 `python` 会提示"不是内部或外部命令"，后续所有操作都进行不下去。勾选它，系统才能找到 Python。
2. 直接点击 **"Install Now"**（默认安装）。默认安装路径通常不含中文和特殊符号，但若用户名包含中文或空格，建议自定义安装路径，确保路径中不含中文或空格。

**第三步：验证安装**

1. 按 `Win + R` 键，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python

## 块 4

### macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要扎实的 Python 环境。别担心，macOS 上装 Python 比你想象中简单，跟着做就行。

**先记住一个关键点**：macOS 12.3 起不再预装 Python 2，系统自带的 `python3` 可能为旧版，**强烈建议自己装最新版**，别依赖系统版本。

---

**方法一：官方安装包（推荐你，新手友好）**

1. 打开浏览器，访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点 "Continue" → "Install"，输密码完成安装。
3. 打开终端（Terminal），输入下面两行验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`？** 这是 macOS 上最常见的坑：命令行默认**没有** `python` 命令（早期版本指向 Python 2，现已移除）。系统和你装的 Python 都统一用 `python3`。所以**始终用 `python3` / `pip3`**，避免"命令不存在"或误用旧版。

---

**方法二：Homebrew（如果你已经装了它）**

在终端执行：

```bash
brew install python

## 块 5

**动手装好 Python，马上就能跑 LoRA 实验**

你的目标是 LoRA 微调，第一步就是把环境备齐。Ubuntu/Debian 通常自带 Python，但版本可能偏老，我们直接装新的。

打开终端，依次执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**关键点**：`-y` 让系统自动确认安装，不用你手动输入 `y`；`python3-venv` 是建虚拟环境用的，现在一起装好，省得后面缺了再补。

**本块学习检查**：执行完 `python3 --version` 后，把输出的版本号记下来

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选择直接决定你后面能不能顺利用上 GPU。记住一个核心原则：**日常练手和真正跑训练，用两套环境，别混在一起。**

**日常写代码、练命令（如果你是 Windows 用户）**

别在 Windows 自带的 CMD 里练命令了，那和你以后在服务器上操作完全是两码事。直接启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04 或 24.04 LTS）。它相当于在 Windows 里装了一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和云服务器一致。安装很简单，管理员身份打开 PowerShell，执行一条命令：

```bash
wsl --install
```

如果你是 macOS 或 Linux 用户，那更省事，直接用本机终端就行，本身就是类 Unix 环境，不用额外折腾。

**跑 LoRA / QLoRA 微调、vLLM 部署这类 GPU 实训**

这里必须用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或者本地实体 Linux（Ubuntu LTS）机器**。原因很直接：这些任务依赖 NVIDIA 驱动、CUDA 和显卡直通，而 Ubuntu LTS 对深度学习生态的支持最成熟、

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**关键判断**：只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它

## 块 8

### 3.3 进阶：把镜像设为默认——网络受限时的“一劳永逸”

**什么时候才需要这一步？**  
只有当你长期待在网络受限环境（比如公司内网、校园网），几乎每次装包都得手动加镜像地址时，才值得把镜像写进配置文件设为默认。**注意**：官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除配置。

---

**Windows 操作步骤（跟着做就行）：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车。会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在这个文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**这行是什么意思？** `index-url` 就是告诉 pip：以后默认从这个地址下载包。

---

**macOS / Linux 操作步骤：**

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

**⚠️ 关于 `trusted-host` 的提醒**  
网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——加了它，pip 会跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常用**不需要**。只有企业内网自建源、或确实遇到证书报错且确认来源可信时，才临时加。

---

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config

## 块 9

**目标导向：** 你的目标是 LoRA 微调，这需要你熟练操作 Python 代码和训练脚本。VS Code 正是你后续调试和运行这些脚本的“主战场”。它免费、启动快，而且插件生态丰富，能让你写代码、跑训练的过程顺畅不少。

**安装步骤（跟着做就行）：**

1. 打开浏览器，访问 `https://code.visualstudio.com`，下载你电脑对应系统的安装包。
2. 双击运行安装程序，一路点击“下一步”直到完成即可，不用改任何默认选项。

**关键一步：装插件**

打开 VS Code 后，看左侧竖着的活动栏，点击最下方那个像四个方块一样的图标（这就是“扩展”商店）。在搜索框里依次搜索并安装下面两个插件：

- **Python**：微软官方出品。装上它，VS Code 才能“看懂” `.py` 文件，给你提供语法高亮和错误提示。
- **Pylance**：这是 Python

## 块 10

**动手装好 PyCharm，为 LoRA 微调铺路**

你目标是 LoRA 微调，代码会越写越长，一个专业 IDE 能让你少踩坑。PyCharm Community 免费且开箱即用，适合你快速上手。

**安装三步走：**

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费）下载
3. 运行安装程序，**务必勾选** `Add launchers dir to the PATH`（将 PyCharm 启动器添加到 PATH，命令行直接敲 `pycharm` 启动）

**首次启动配置：**

1. 打开 PyCharm，点 `New Project`
2. `Location` 选项目路径——**路径别带中文**，否则后续跑 LoRA 可能出怪问题
3. `Interpreter` 选 `Previously configured interpreter`，再选你系统里的 Python 3.10+

装好后，

## 块 11

### 5.1 为什么用 Conda

你的目标是 LoRA 微调，这需要 PyTorch、CUDA 等一堆复杂依赖。如果直接在系统 Python 里装，很容易把环境搞乱。Conda 就是来解决这个问题的：它为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备工具。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB 空间），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**，环境干净，后续需要什么装什么，适合你这种要跑 LoRA 微调的项目开发场景。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python 冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 conda 会通过初始化脚本正确接管，而不是

## 块 12

**动手做：创建你的第一个虚拟环境**

你马上要学 LoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，后面直接复用这套流程。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定装 Python 3.10

执行后，终端会列出要安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y` 回车，等待完成。

**本块

## 块 13

**本块学习检查**：在终端依次执行 `conda activate llm-learn`、`python --version`、`conda deactivate`，观察并记录每一步命令行提示符的变化，确认 `python --version` 输出为 `Python 3.10.x`。

---

**激活环境**：输入 `conda activate llm-learn`，相当于进入这个项目的“独立房间”。激活后，命令行开头会出现 `(llm-learn)` 标志，这表示之后所有 `python` 和 `pip` 命令都只作用于这个环境，不会干扰系统或其他项目。

**验证环境**：接着运行 `python --version`，应显示 `Python 3.10.x`，确认你用的是环境内的解释器。

**退出环境**：需要回到系统默认环境时，运行 `conda deactivate`。

**查看与管理**：用 `conda env list` 列出所有环境，带 `*` 号的就是当前激活的。以后

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，所以先确保环境里装好了 PyTorch 等依赖。跟着做：

1. 打开终端，先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入 `Python: Select Interpreter` 并回车。
4. 在列表中选择带有 `conda` 字样的 Python 路径（例如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样做**：VS Code 会自动识别 conda 环境。选中后，你在当前工作区写的所有代码都会使用 `llm-learn` 环境里的 Python 和已安装的包——包括你后续做 LoRA 微调要用的 PyTorch。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程如下：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 上是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的

## 块 15

**动手验证：让 Python 先“开口说话”**

你的目标是 LoRA 微调，但一切大模型实验都从环境跑通开始。咱们用最快的方式确认：命令行里 Python 已经能用。

**第一步：激活环境**
在终端输入 `conda activate llm-learn`，确保你在正确的环境里。

**第二步：进入 Python 交互模式**
输入 `python` 并回车。看到 `>>>` 提示符，说明 Python 已就绪。

**第三步：打印第一句话**
在 `>>>` 后输入：
```python
print("Hello, Python! 我的环境准备好了！")
```
回车后，如果屏幕打印出这句话，说明环境完全正常。输入 `exit()` 退出。

**本块学习检查**
在命令行重复上述步骤，

## 块 16

好的，我们来把这一部分内容，用最适合你的方式重新组织一下。

你的目标是 **LoRA 微调**，这意味着你很快就要接触 PyTorch 和模型训练了。一个干净、可控的开发环境是这一切的基础。所以，我们不仅要“跑通”代码，更要理解每一步在做什么，为后续的“大工程”做好准备。

---

### 7.3 在 PyCharm 中验证：跑通你的第一行代码

这一步是检验我们之前所有配置工作的“验收时刻”。请跟着做，确保你的“工作台”完全正常。

1.  **新建项目**：打开 PyCharm，创建一个新项目。项目路径务必**不要包含中文或空格**，这是为了避免后续一些不必要的麻烦。
2.  **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3.  **输入代码**：在 `hello.py` 中输入 `print("Hello, World!")`。
4.  **运行**：在编辑区**右键点击**鼠标，选择 **`Run 'hello'`**。
5.  **验证**：查看 PyCharm 下方的 **Run** 窗口，如果正确输出了 `Hello, World!`，恭喜你，你的开发环境已经可以正常工作了！

---

### 7.4 常见问题排查（你的“急救手册”）

在配置过程中，你可能会遇到一些小状况。别担心，这很正常。这里有一份“急救手册”，遇到问题可以对照着快速解决。

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 使用 `-i` 指定国内镜像（如清华源），例如：`pip install 包名 -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/L
