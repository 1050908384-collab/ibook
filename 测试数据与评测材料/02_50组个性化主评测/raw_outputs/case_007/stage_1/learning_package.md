# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

你写 Python 代码，是为了让电脑干活。但有个问题：你写的是人话，电脑只懂 0 和 1。中间缺个"翻译官"。

**Python 解释器**就是干这个的——它把你写的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件就只是一堆普通文本，电脑根本不理。

**为什么你特别需要搞懂这个？** 因为你接下来要做 LoRA 微调，会用到大量 Python 脚本。如果解释器没装对，后面每一步都会报错，而且你都不知道错在哪。

**动手验证一下（30秒）：**
打开你的终端（Mac 叫"终端"，Windows 叫"命令提示符"），输入：
```
python --version
```
如果看到类似 `Python 3.x.x` 的版本号，说明解释器已经装好了。如果提示"找不到命令"，说明还没装——别急，下一节就带你装。

**记住这个比喻：** 解释器就像你和电脑之间的同声传译。你说话（写代码），它翻译（执行指令），电脑干活（出结果）。没有翻译官，你俩就是鸡同鸭讲。

## 块 2

### 你的“工作台”与“独立房间”

目标是LoRA微调，这意味着你很快要跑深度学习代码。别用记事本硬扛——**IDE（集成开发环境）**就是你的“工作台”，提供语法高亮、自动补全、运行调试，写代码像用Word写文档一样舒服。推荐**VS Code**（轻量免费）和**PyCharm**（专业强大），先装VS Code，够用且上手快。

接着是**虚拟环境**——每个项目的“独立房间”。想象A项目要`numpy 1.0`，B项目要`numpy 2.0`，全装一起就冲突报错。虚拟环境为每个项目创建独立Python空间，互不干扰。我们用**conda**管理，这是目前最

## 块 3

**本块学习检查**：完成安装后，打开命令提示符，依次输入 `python --version` 和 `pip --version`，确认两条命令都显示版本号。若 `python` 提示“不是内部或外部命令”，请回到第二步，重新运行安装程序并勾选“Add python.exe to PATH”。

---

你的目标是 **LoRA 微调**，这需要你在命令行里频繁调用 Python 和 pip。所以，本块的核心不是“装个软件”，而是**确保命令行能认出 Python**。我们直接上手。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`。
2. 把鼠标移到顶部菜单 **Downloads** 上，网站会自动推荐适合 Windows 的版本（比如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件。弹出安装界面后，**只做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**。  
   这是全流程最关键的一步。不勾选的话，你之后在命令行输入 `python`，系统会提示“不是内部或外部命令”，后续所有操作都会卡住。

2. 点击 **"Install Now"**（默认安装）。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，

## 块 4

### macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要扎实的 Python 环境。macOS 上老版本曾预装 Python 2.7，但**macOS 12.3 起已不再预装 Python 2**；现代 macOS 仍可能带有供系统脚本使用的旧版 `python3`。无论哪种情况，都**强烈建议自行安装最新版 Python**，避免直接依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：在现代 macOS 上，命令行默认**并不提供** `python` 命令（早期版本曾指向系统自带的 Python 2，现已移除），系统与官方安装包统一使用 `python3`。因此请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已经安装了 Homebrew（macOS 的包管理器），在

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你目标是跑 LoRA 微调，但先别急着碰模型——Linux 上 Python 版本太老会导致后面装 PyTorch 报错。跟着做，三步搞定：

```bash
# 1. 更新软件源，确保能装到最新包
sudo apt update

# 2. 一次装齐 Python 3、pip 和虚拟环境模块
sudo apt install python3 python3-pip python3-venv -y

# 3. 验证版本，确认安装成功
python3 --version
```

**为什么这么写**：`-y` 让系统自动确认，不用你手动输 `y`；`python3 --

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选择就一句话：**别在 Windows 上硬练，直接拥抱 Linux**。

**日常写 Python、练命令行（Windows 用户）**
启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**
直接用本机终端，本身就是类 Unix 环境，不用额外折腾。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**
用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已停止维护、生态落后；VirtualBox 默认无法直通 NVIDIA GPU，跑不了真正的 GPU 训练，只适合纯 CPU 的隔离小实验。

**本块学习检查**：打开终端，输入 `uname -a` 确认你在 Linux 环境（WSL2 或本机）。若

## 块 7

你的目标是跑通 LoRA 微调，但这一步卡在“装包”上——镜像源就是解决这个卡点的工具。

**什么时候才需要镜像源？**

`pip` 是 Python 的包管理工具，用来安装第三方库（比如后面微调要用的 `torch`、`transformers`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**只有当你**访问官方源明显变慢、经常超时，或处在网络受限环境时，才临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是永久设置**——官方源始终是首选。

**首选做法：临时指定镜像源（推荐）**

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为 LoRA 微调做准备，而微调通常需要下载大量预训练模型和依赖包。如果你长期处于网络受限环境，几乎每次都要手动指定镜像源，那才值得把它设为默认。**注意**：官方 PyPI 始终是最可靠的，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**。只有企业内网自建源、或确实遇到证书报错且确认来源可信时，才临时加。

**macOS / Linux 用户：**

1. 打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样的内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `pip.conf` 文件即可，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：在终端执行 `pip config list`，确认输出中包含 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'

## 块 9

**目标对齐**：你正在为 LoRA 微调搭建环境，VS Code 就是你的“操作台”——它轻量、免费，启动快，插件丰富，后续写 Python 脚本、调配置文件都能用上。

**安装三步走**（跟着做，5分钟搞定）：

1. 打开官网 https://code.visualstudio.com，下载你系统的安装包。
2. 运行安装程序，一路“下一步”直到完成。
3. **装两个必装插件**：点左侧活动栏最下方的“四个方块”图标（扩展），搜索并安装：
   - **Python**（微软官方，提供语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更智能）

> 为什么必须装？不装 Python 插件，VS Code 认不出 `.py` 文件，代码补全、错误提示、一键运行全都没有。

**关键一步：指定 Python 解释器**

1. 按 `Ctrl + Shift + P`

## 块 10

**动手装好 PyCharm，为 LoRA 微调铺路**

你目标是跑 LoRA 微调，代码会越来越多，需要一个专业 IDE。PyCharm Community 免费且开箱即用，正好适合你。

**安装三步走：**

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费）下载
3. 运行安装包，务必勾选这两项：
   - `Add "Open Folder as Project"` —— 以后右键文件夹就能直接打开
   - `Add to PATH` —— 命令行能直接启动 PyCharm

**首次启动配置：**

1. 打开 PyCharm，点 `New Project`
2. `Location` 选项目存放路径，**路径别带中文**，否则后续跑微调容易出怪问题
3. `Interpreter` 选 `Previously configured interpreter`，再选你系统里的 Python 3.10+

装好后，你的工作台就绪，下一步配置 Conda 虚拟环境做项目

## 块 11

### 5.1 为什么用 Conda

你的目标是做 LoRA 微调，这条路绕不开 PyTorch 和 CUDA 这类复杂依赖。Conda 是目前最主流的虚拟环境管理工具，它不仅能像前面说的那样为每个项目创建独立的 Python 空间，还能直接管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备工具——它帮你把项目之间的依赖隔离开，避免"装一个包搞坏另一个项目"的连锁反应。

### 5.2 安装 Conda：Anaconda vs Miniconda

安装前先做个选择，两个选项差别不小：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐你选 Miniconda**。理由很直接：你的目标是 LoRA 微调，环境越干净越好，后面需要什么包装什么包，不背一堆用不上的库。

**安装步骤（跟着做就行）：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包：Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`
3. 运行安装程序，这里有个关键区别：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器自己也标了"不推荐"）。为什么？把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python 或其他工具冲突，导致 `python`、`pip` 指向不明。日常使用就打开开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来操作 conda。
   - **macOS/Linux**：运行

## 块 12

**创建你的第一个虚拟环境**

你的目标是LoRA微调，不同项目需要不同版本的PyTorch和CUDA。现在学会创建环境，以后直接复用这套流程。

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

**本块学习

## 块 13

**本块学习检查**：在终端依次执行 `conda activate llm-learn`、`python --version`、`conda deactivate`，观察并记录每一步命令行前缀的变化。

---

**5.4 激活与退出环境**

你要做 LoRA 微调，项目隔离是第一步。把 Conda 环境想象成你的专属“工作台”——激活它，就是进入这个工作台。

**进入工作台**：

```bash
conda activate llm-learn
```

执行后，命令行前面会出现 `(llm-learn)` 标志。这意味着你后续的 `python` 和 `pip` 命令都只作用于这个环境，不会干扰系统或其他项目。

**确认版本**：

```bash
python --version
```

正常应显示 `Python 3.10.x`。

**离开工作台**：

```bash
conda deactivate
```

执行后，`(llm-learn)` 标志消失，回到系统默认环境。

**5.

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，所以先确保环境就绪。跟着做：

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 会自动识别 conda 环境。选好后，你在这个工作区写的代码就会用 `llm-learn` 环境中的 Python 和已装好的包——LoRA 微调需要的 PyTorch、HuggingFace 库都在里面。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python

## 块 15

**本块学习检查**：完成下方两个验证后，在终端输入 `exit()` 退出，并确认你已看到两段“Hello”输出。

---

你的目标是 LoRA 微调，但所有大模型实验都从“环境能跑通”开始。本块用 5 分钟验证你的工作台是否就绪，全程动手操作。

**第一步：命令行验证（最快反馈）**

1. 打开终端，输入 `conda activate llm-learn` 激活环境。
2. 输入 `python` 回车，看到 `>>>` 提示符即进入交互模式。
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 并回车。
4. 看到正确打印后，输入 `exit()` 退出。

**

## 块 16

### 7.3 在 PyCharm 中验证——跑通你的第一行代码

环境搭好了，现在我们来亲手验证它。你的目标是 LoRA 微调，而微调的第一步，就是确保你的“工作台”能正常运转。这一节我们用最直接的方式——动手跑一段代码，来确认一切就绪。

**操作步骤（跟着做，2分钟搞定）：**

1. 打开 PyCharm，新建一个项目。
2. 在项目中新建一个 Python 文件，命名为 `hello.py`。
3. 在文件里输入一行最简单的代码（比如打印一句问候）。
4. 在文件编辑区**右键点击**，选择 **`Run 'hello'`**。
5. 看下方弹出的 **Run 窗口**——如果能看到你打印的内容，说明你的开发环境完全正常，可以开始后续学习了。

**如果没跑通？别慌，先对照下面这张“问题排查表”：**

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像，如：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名`；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看
