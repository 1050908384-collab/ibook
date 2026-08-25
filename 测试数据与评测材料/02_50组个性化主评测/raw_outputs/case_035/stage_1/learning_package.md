# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

你写下的 Python 代码是给人看的文字，但电脑只懂 0 和 1。**Python 解释器**就是那个"翻译官"——它把你的代码逐句翻译成电脑能执行的指令。没有它，你写的 `.py` 文件就只是一堆普通文本，电脑根本不知道你想干什么。

**动手验证一下（30秒）**：在电脑上随便新建一个文本文件，把内容改成 `print("hello")`，保存为 `test.py`。现在双击它——大概率什么都不会发生，或者一闪而过。这就是因为电脑没有"翻译官"去解读里面的内容。等你装好解释器后，再运行同一个文件，它就会在屏幕上打出 `hello`。

**记住这个比喻**：解释器 = 翻译官，`.py` 文件 = 写满外语的纸条。翻译官不在，纸条就是废纸。

---

**本块学习检查**：用自己的话说，为什么没有 Python 解释器时，`.py` 文件无法运行？请用"翻译官"这个比喻回答。

## 块 2

**本块学习检查**：读完本节后，请暂停并完成两件事：① 用一句话分别说出 IDE 和虚拟环境各自解决什么问题；② 在纸上画出“两个项目 + 各自虚拟环境 + 同一个 Python”的关系草图。画完再继续。

---

## 1.2 代码编辑器（IDE）：你的“工作台”

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。本文推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

**给你的建议**：你的目标是 LoRA 微调，动手实操是主线。建议直接选 **VS Code**——轻量、免费，打开快，适合你每天 1 小时的快节奏学习。PyCharm 虽强，但对新手偏重，等需要大型项目时再换不

## 块 3

**本块学习检查**：完成安装后，请打开命令提示符，依次输入 `python --version` 和 `pip --version`，确认两行都显示版本号（如 `Python 3.13.x`），再继续下一步。

---

### 2.1 Windows 用户：三步装好 Python

你的目标是跑 LoRA 微调，而所有 AI 工具链都建立在 Python 之上。下面三步帮你把地基打牢，全程约 10 分钟。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这是最容易踩的坑。不勾选的话，之后在命令行输入 `python` 会提示"不是内部或外部命令"，所有后续操作都会卡住。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，能避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要 Python 环境。好消息是，现代 macOS（Catalina 10.15 起）不再预装 Python 2，但系统可能带有旧版 `python3`（如 3.8 或 3.9）。**强烈建议自行安装最新版 Python 3.10+**，避免使用系统自带版本，以免后续安装包时遇到权限或版本问题。

**方法一：官方安装包（推荐，省心）**

1. 打开浏览器，访问 https://www.python.org/downloads/macos/ ，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点 "Continue" → "Install"，输密码完成安装。
3. 打开终端（Terminal），输入下面两行验证：

```bash
python3 --version
pip3 --version
```

看到版本号就成功了。

**关键提醒**：命令是 `python3` 和 `pip3`，不是 `python`。现代 macOS 默认**不提供** `python` 命令，用 `python` 会报"命令不存在"。以后装包、跑脚本，统一用 `python3` / `pip3`。

**方法二：Homebrew（如果你已装过）

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是跑 LoRA 微调，但先别急着碰模型——Linux 上没装对 Python，后面全是坑。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省心。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：第一行更新软件源，让系统知道最新包在哪；第二行一次装齐 Python 本体、包管理器 pip、虚拟环境模块 venv；`-y` 是自动确认，省得你手动敲 `y`。第三

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，那环境选型直接决定后面能不能顺利跑起来。按你的情况，分两条线看：

**一、日常练 Python 和命令（每天 1 小时，别浪费在环境折腾上）**

- **Windows 用户**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里直接获得一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`，一条命令搞定。
- **macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**二、LoRA / QLoRA 微调实训（你的核心目标）**

必须使用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 避坑提醒**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于 2021 年停止

## 块 7

**本块学习检查**：运行 `pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`，观察安装速度是否明显快于默认源；若成功，说明镜像源生效。

---

**3.1 什么时候才需要镜像源**

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、最应保留的默认源**。

只有当网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

**3.2 首选做法：临时指定镜像源（推荐）**

如果只是某次安装慢，最干净的做法是**只对这一次安装

## 块 8

## 3.3 进阶：把镜像设为默认——仅限长期网络受限时

> **先判断**：如果你只是偶尔下载慢，跳过本节，继续用 `-i` 临时指定镜像即可。只有**长期**处于网络受限环境、几乎每次都要用镜像时，才把镜像写入配置作为默认源。

**重要提醒**：官方 PyPI 始终是最可靠的来源。配置默认镜像后，一旦网络恢复，建议改回或删除该配置。

---

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车。会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建 `pip` 文件夹**。
3. 在 `pip` 文件夹内**新建 `pip.ini` 文本文件**。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**含义**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 别乱加 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加。

---

### macOS / Linux 操作步骤

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

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url

## 块 9

**你的目标：LoRA 微调，但先别急——工欲善其事，必先利其器。**

你每天只有 1 小时，且喜欢动手。所以，我们直接选**VS Code**——它启动快、免费，而且插件丰富。以后你写 LoRA 微调脚本、改配置文件、敲 Linux 命令，都用得上它。别选那些笨重的 IDE，浪费时间。

**安装三步走（照做就行）：**

1. 打开浏览器，访问 `https://code.visualstudio.com`，下载你系统的安装包。
2. 双击运行，一路点“下一步”，装完就完事。
3. **关键一步**：打开 VS Code，看左侧最下方的**四个方块图标**（那是扩展商店），点进去，搜索并安装这两个插件：
   - **Python**（微软官方出的，给你语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更聪明，写 LoRA 代码时少打错字）

**为什么必须装？** 不装 Python 插件，VS Code 根本认不出 `.py` 文件，代码补全

## 块 10

### 4.2 方案二：PyCharm Community（免费、轻量）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费且开源。它内置了代码分析、版本控制等基础功能，适合大型项目开发。注意：数据库工具等高级功能仅存在于专业版（Professional）中。

**安装步骤**：

1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm；Windows 用户建议勾选，macOS/Linux 用户可通过终端命令启动）

**首次启动配置**：

1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured

## 块 11

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，这绕不开 PyTorch 和复杂的依赖管理。Conda 就是干这个的，它能给你的每个项目一个干净、独立的 Python 空间，避免“装 A 包把 B 包搞坏”的惨剧。

**为什么非它不可？**
简单说，Conda 是当前最主流的虚拟环境管理工具。它不仅能隔离 Python 包，还能直接管理 Python 解释器版本。对于深度学习这种依赖极多的场景，它几乎是必备的。

**选哪个？Anaconda 还是 Miniconda？**
这里直接给你结论：**用 Miniconda**。

- **Anaconda**：体积大（500MB+），预装了很多数据科学库。适合不想手动装包、主要做数据分析的人。
- **Miniconda**：体积小（约50MB），只带 conda 和基础 Python。**环境干净，需要什么装什么**，这正符合我们做项目开发的需求。

**动手安装 Miniconda（跟着做就行）**

1. 打开浏览器，访问 `https://docs.conda.io/en/latest/miniconda.html`。
2. 下载对应你操作系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序，这里有个关键区别：
   - **Windows 用户**：一路点“下一步”，但**千万不要勾选** “Add Miniconda3 to my PATH environment variable” 这个选项（安装器也标注了“不推荐”）。勾选它容易和你系统里已有的 Python 冲突，导致 `python`、`pip` 命令指向混乱。以后要用 conda，就从开始菜单打开 **“Anaconda Prompt (Miniconda3)”**。
   - **macOS/Linux 用户**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选好安装路径。**最后一步一定要输入 `yes`**，让安装器执行 `conda init`。这样终端每次启动时 conda 会自动接管，而不是靠硬改 PATH。
4. **验证安装

## 块 12

**目标**：你的目标是 LoRA 微调，而微调前必须隔离项目环境。本块教你创建第一个虚拟环境，这是后续所有项目的“起跑线”。

**动手做**：打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（记住这个套路，以后每个项目都这么干）：
- `conda create`：创建环境的动作
- `-n llm-learn`：给环境命名，`n` 就是 `name` 的缩写
- `python=3.10`：指定环境内的 Python 版本

**执行后**：终端会列出待装包，并询问 `Proceed ([y]/n)?`。输入 `y` 回车

## 块 13

**本块学习检查**：在终端依次执行 `conda activate llm-learn`、`python --version`、`conda deactivate`，观察并记录每一步命令行前缀的变化。

---

你正在为 LoRA 微调准备环境，核心思路是**把项目依赖关进独立房间**，避免和系统其他 Python 包互相干扰。本块只讲两个动作：**进门**和**出门**。

**进门（激活环境）**：在终端输入：

```bash
conda activate llm-learn
```

输入后，命令行最前面会出现 `(llm-learn)` 标志——这就是“房间门牌”。只要这个标志在，你后续敲的所有 `python` 和 `pip` 命令都只作用于这个环境，不会污染系统或其他项目。

**验证房间里的 Python 版本**：

```bash
python --version
```

正常应显示 `Python 3.10.x`。这一步很重要，确保你用的是 LoRA 微

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，所以先确保环境就绪。跟着做：

1. 打开终端，运行 `conda activate llm-learn` 激活环境。
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表中找到带有 `conda` 字样的选项（如 `Python 3.10.x ('llm-learn': conda)`），点击选中。

**为什么这样做**：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的所有代码都会使用 `llm-learn` 环境里的 Python 和已安装的包——包括后续 LoRA 微调需要的库。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，操作如下：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中，选择 conda 环境的 Python 可执行文件：

## 块 15

**本块学习检查**：完成下面两步，确认环境可用。

1. 打开终端，运行 `conda activate llm-learn` 激活环境。
2. 输入 `python` 回车，看到 `>>>` 后，执行：

```python
print("Hello, Python! 我的环境准备好了！")
```

看到文字输出，输入 `exit()` 退出。接着，在 VS Code 中新建文件夹 `my-first-project`，创建 `hello.py`，写入 `print("Hello from VS Code!")`，按 `Ctrl + F5` 运行，终端显示 `Hello from VS Code!` 即全部通过。

## 块 16

好的，收到你的学习需求。我们即将开始你的LoRA微调之旅，而一个干净、可用的Python开发环境是这一切的基石。既然你偏爱动手实践，我们就直接通过“做”来验证环境是否就绪。

**本块学习检查**：请跟随以下步骤，在PyCharm中创建并运行你的第一个Python文件。如果遇到任何问题，请对照第八章的排查表尝试解决，并记录下你遇到的问题和最终的解决方案。

---

### 7.3 在 PyCharm 中验证

1.  **新建项目**：打开PyCharm，创建一个新的纯Python项目。
2.  **创建文件**：在项目中新建一个Python文件，命名为 `hello.py`。
3.  **编写代码**：在 `hello.py` 中输入 `print("Hello, LoRA!")`。
4.  **运行**：在编辑区右键点击鼠标，选择 `Run 'hello'`。
5.  **验证**：查看PyCharm下方的 `Run` 窗口，如果正确输出了 `Hello, LoRA!`，恭喜你，你的开发工作台已经可以正常工作了！

---

### 第八章：常见问题排查

如果在验证过程中遇到问题，不要着急，这是学习过程中的正常一环。请对照下表进行排查：

| 问题现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行Python安装程序，选择 "Modify"，勾选 "Add Python to environment variables"。 |
| `pip install` 速度极慢或超时 | 网络受限，访问官方PyPI慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源。 |
| `conda` 命令找不到 | 未使用Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux运行 `conda init` 后重开终端。 |
| VS Code提示"未选择Python解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择。 |
| 激活conda环境后，`python
