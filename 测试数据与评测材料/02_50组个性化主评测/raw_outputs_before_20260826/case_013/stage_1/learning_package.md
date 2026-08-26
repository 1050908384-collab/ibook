# 学习包

## 块 1

**本块学习检查**：读完本节能用自己的话说出“为什么没有解释器，`.py` 文件就只是一堆文本？”并写下来。

---

你现在的目标是 **LoRA 微调**，但别急着碰模型——所有微调代码都要靠 Python 跑起来，而**解释器**就是让代码活过来的第一步。

**一句话理解**：你写的 Python 代码是给人看的文字，电脑只懂 0 和 1。**Python 解释器**就是那个“翻译官”，把你的代码翻译成电脑能执行的指令。

**动手验证一下**（不用装任何东西，先建立直觉）：

1. 打开记事本，输入 `print("hello")`，保存为 `test.py`。
2. 双击这个文件——大概率没反应，或者一闪而过。
3. 原因：没有解释器，电脑不知道这堆文字是“程序”，它只是一份普通文本。

**关键点**：解释器不是“编辑器”，编辑器只是写字的工具；解释器才是真正“读懂”并执行代码的引擎。后面你安装 Python，装的核心就是它。

**为什么这对 LoRA 重要**：微调时你会在终端敲 `python train.py`，这个 `python` 命令调用的就是解释器。环境没配好，代码写得再对也跑不起来。

现在你清楚“在装什么”了，下一步就是把它装好。

## 块 2

你正奔着 LoRA 微调去，但先别急着碰模型——**你的"工作台"没搭好，后面会处处卡壳**。这一步的目标很简单：让你写代码时像用 Word 一样顺手，而不是在记事本里受罪。

**IDE 怎么选？** 记住两个名字就够了：
- **VS Code**：轻量、免费，启动快，插件丰富。适合你这种想快速上手、边学边试的风格。
- **PyCharm**：功能更全、更"重"，专业项目常用。但对你当前 LoRA 微调的目标来说，VS Code 完全够用，而且更省心。

**别忽略虚拟环境**——这是你最容易踩的坑。想象你同时做两个项目：一个要 `numpy 1.0`，另一个要 `numpy 2.0`，全装一起必冲突。**虚拟环境**就是给每个项目一个独立 Python 空间，互不

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要 Python 环境。我们直接动手，按三步走。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 添加到系统"环境变量"。不勾选的话，你打开命令行输入 `python`，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径通常为 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python313`，若用户名含中文或空格，建议自定义安装路径为纯英文无空格目录。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python

## 块 4

### macOS 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要扎实的 Python 环境。macOS 上别依赖系统自带的 Python——macOS 12.3 起不再预装 Python 2，系统自带的 `python3` 版本可能过旧，不适合做开发。**强烈建议自己装最新版**。

#### 方法一：官方安装包（推荐，最省心）

1. 打开浏览器，访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入下面两条命令验证是否成功：

```bash
python3 --version
pip3 --version
```

看到版本号输出就说明装好了。

**关键提醒**：在 macOS 上，命令行默认**没有** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统统一使用 `python3`。所以**始终用 `python3` 和 `pip3`**，别用 `python`，否则会报"命令不存在"或误用系统旧版本。

#### 方法二：Homebrew（如果你已装过）

如果你已经有 Homebrew（macOS 的包管理器），在终端执行一条

## 块 5

你的目标是快速上手 LoRA 微调，而动手第一步就是装好 Python。Linux（Ubuntu/Debian）通常自带 Python，但版本可能偏老，不够用。跟着下面三步走，直接搞定：

```bash
# 1. 更新软件源，确保能装到最新包
sudo apt update

# 2. 安装 Python 3、pip 和虚拟环境模块
sudo apt install python3 python3-pip python3-venv -y

# 3. 验证安装成功
python3 --version
```

**关键点**：`-y` 参数让系统自动确认安装，省去手动输入 `y` 的麻烦。`python3-venv` 是创建隔离环境（

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选择直接决定你能不能顺利走到 GPU 训练那一步。按你的情况，分两条线来看：

**日常写 Python、练命令（Windows 用户）**

启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。为什么不用 Windows 自带的 CMD？因为 WSL2 给你的是一个接近真实生产的 Linux 环境，命令、路径、包管理的方式都和后续的服务器一致——你在这里练熟了，后面上云服务器不会懵。安装很简单，管理员 PowerShell 里执行 `wsl --install` 就行。

**macOS / Linux 用户**：直接用自己的终端，本身就是类 Unix 环境，不用额外折腾。

---

**LoRA / QLoRA 微调、vLLM 部署这类 GPU 实训**

用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器**，或者**本地实体 Linux（Ubuntu LTS）机器**。原因很直接：这类任务依赖 NVIDIA 驱动、CUDA 和显卡直通，Ubuntu LTS 对深度学习生态的支持最成熟、文档最全。你后面做 LoRA 微调，跑起来最稳的就是这条路。

---

**⚠️ 一个坑，别踩**：不要把 **"

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、应保留的默认源**。

**什么时候才用镜像？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。记住：**镜像是"网络不好时的临时加速手段"，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响

## 块 8

## 3.3 进阶：把镜像设为默认（长期网络受限才做）

你的目标是 LoRA 微调，后面会频繁用 pip 装 PyTorch、Transformers 等大包。如果网络长期不稳，每次手动加 `-i` 太烦，那就直接写进配置文件，一劳永逸。

**但先记住**：官方 PyPI 永远是最可靠的。网络恢复后，建议改回或删掉配置。

---

### Windows 操作

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 新建文件夹，命名为 `pip`。
3. 进入 `pip` 文件夹，新建文本文件，命名为 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：默认从清华镜像下载。

---

### macOS / Linux 操作

打开终端，依次执行：

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

写入同样内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

按 `Ctrl + O` 保存，`Ctrl + X` 退出。

---

### ⚠️ 别乱加 `trusted-host`

网上很多教程会让你顺手加一行 `trusted-host = ...`。**除非镜像真的报 HTTPS 证书错误，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法证书，正常用**不需要**。只有企业内网自建源、确认可信时才临时加。

---

### 验证配置是否生效

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 就说明成功了。

---

### 改回官方源

删除 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：动手做一遍——按你的系统完成配置，然后运行 `pip config list` 确认输出。再执行 `pip config unset global.index-url` 改回官方源，最后

## 块 9

**本块学习检查**：完成 VS Code 安装与 Python 插件配置后，打开命令面板（`Ctrl + Shift + P` 或 `Cmd + Shift + P`），输入并执行 `Python: Select Interpreter`，确认你能看到并选中已安装的 Python 版本（如 `Python 3.13.x`）。若成功，说明你的开发工作台已就绪，可以开始后续 LoRA 微调实战。

---

**为什么选 VS Code**：你的目标是 LoRA 微调，核心操作是写 Python 脚本、调试代码。VS Code 启动快、插件丰富，且免费轻量，后续写训练脚本、改配置文件时都能用上，是性价比最高的选择。

**安装三步走**：

1. 访问 `https://code.visualstudio.com`，下载对应你系统的安装包。
2. 运行安装程序，一路“下一步”即可，无需额外设置。
3. **关键一步**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并

## 块 10

**动手装 PyCharm，为 LoRA 微调搭好“工作台”**  
你马上要跑 LoRA 微调，代码会越来越长、项目文件也多，PyCharm 正好帮你管住这些。它免费、轻量，适合学习。

**安装三步走**  
1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`  
2. 选 **Community Edition**（社区版，免费）下载。  
3. 运行安装程序，务必勾选这两项：  
   - `Add "Open Folder as Project"`（以后右键文件夹就能直接打开）  
   - Windows 上勾选 `Add to PATH`（命令行也能启动 PyCharm；macOS/Linux 无此选项）

**首次启动，这样配**  
1. 打开 PyCharm，点 `New Project`。  
2. `Location` 选项目存放路径——**路径别带中文**，避免

## 块 11

## 5.1 为什么用 Conda

你要做 LoRA 微调，后面会用到 PyTorch、CUDA 这些依赖，它们之间版本关系很敏感。Conda 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身——这意味着你可以为 LoRA 项目单独建一个环境，装好配套的 Python 和 PyTorch，不污染系统里其他项目。对深度学习来说，conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 安装包约 500MB，安装后占用空间可达数 GB，预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 安装包约 50MB，安装后占用空间约 400MB，只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——你的目标是 LoRA 微调，环境越干净越好，需要什么装什么，避免一堆用不到的库干扰排查。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，最后

## 块 12

**创建你的第一个虚拟环境**

你正在为 LoRA 微调做准备，不同项目需要不同版本的 PyTorch 和 CUDA，所以现在学会隔离环境，以后直接复用这套流程。

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

输入 `y` 并回车，等待安装完成

## 块 13

## 5.4 激活与退出环境

你的目标是 LoRA 微调，而微调项目最怕“污染”——不同项目依赖的包版本互相冲突。Conda 环境就是给每个项目一个“独立房间”。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。这对你后续跑 LoRA 微调脚本至关重要——装错版本的 PyTorch 会导致训练直接报错。

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

**查看所有已创建

## 块 14

**本块学习检查**：完成配置后，在 VS Code 或 PyCharm 中新建一个 Python 文件，输入 `print("hello")` 并运行，观察终端输出是否正常。

---

你的目标是 LoRA 微调，这需要先有一个能跑 PyTorch 的“工作台”。本块教你如何把 VS Code 和 PyCharm 指向你刚创建的 `llm-learn` 环境，确保代码用的是这个环境里的 Python 和包。

**VS Code 配置（推荐，更轻量）**

1. 先激活环境：在终端运行 `conda activate llm-learn`。
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表里选带有 `conda` 字样的项，例如 `Python 3.10.x ('llm-learn': conda)`。

> 为什么这样写：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的所有代码都会使用 `llm-learn` 里的 Python 和已装好的包，不会“跑错环境”。

**PyCharm 配置（如果你用这个）**

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮

## 块 15

**本块学习检查**：完成下方两步验证后，在终端输入 `exit()` 退出。

你的目标是 LoRA 微调，动手前先确认环境能跑通。跟着做：

1. **命令行验证**：激活环境后输入 `python`，看到 `>>>` 提示符，粘贴运行：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
   正确打印即通过。

2. **VS Code 验证**：新建 `my-first-project` 文件夹，用 VS Code 打开，新建 `hello.py`，输入 `print("Hello from VS Code!")`，点右上角小三角或按 `Ctrl + F5`，下方终端输出 `Hello from VS Code!` 即成功。

**检查**

## 块 16

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，而这一切都建立在你的开发环境能正常跑起来的基础上。既然你偏好动手实践，我们就用最快的方式，在 PyCharm 里跑通第一行代码，完成环境验证。

### 7.3 在 PyCharm 中验证

打开 PyCharm，我们直接新建一个项目。在项目里创建一个新的 Python 文件，命名为 `hello.py`。

在 `hello.py` 里，输入一行最简单的代码：

```python
print("Hello, LoRA!")
```

然后，在文件编辑区**点击鼠标右键**，在弹出的菜单里找到并点击 **`Run 'hello'`**。

这时，PyCharm 窗口**下方**会打开一个 **Run** 面板。如果一切正常，你会在这个面板里看到程序运行的结果：`Hello, LoRA!`。

看到这行输出，就说明你的 Python 解释器、项目环境都已经配置好了，可以开始后续的代码学习。

---

### 本章小结

完成本章后，你的电脑上应该具备以下基础：Python 开发环境已安装并配置完毕，能够在 PyCharm 中新建项目、创建 Python 文件并成功运行。通过 `
