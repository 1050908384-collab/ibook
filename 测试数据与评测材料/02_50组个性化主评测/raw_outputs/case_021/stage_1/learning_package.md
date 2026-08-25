# 学习包

## 块 1

**本块学习检查**（动手验证，约2分钟）  
打开你的代码编辑器，新建一个文件，输入一行文字（比如 `print("hello")`），保存为 `.py` 文件。然后思考：如果电脑里没有Python解释器，双击这个文件会发生什么？——答案是：它只是一堆文本，电脑看不懂，什么也不会发生。这个检查帮你确认：**解释器是让代码“活”起来的翻译官**。

---

**正文开始**

你现在的目标是LoRA微调，但别急着碰模型——先确保你的“翻译官”到位。

**1.1 Python解释器：程序的“翻译官”**

你写的Python代码是人类能看懂的文字，但电脑只认0和1。**Python解释器**的作用就是把你的代码翻译成电脑能执行的指令。没有它，你写的`.py`文件只是一堆文本。

**为什么这对你重要？**  
LoRA微调本质上是运行一段Python程序来调整模型参数。如果解释器没装好，你连第一步都迈不出去。

**动手做（hands-on）**：  
1. 打开终端（Mac/Linux）或命令提示符（Windows）。  
2. 输入`python --version`并回车。  
3. 如果看到版本号（如`Python 3.x.x`），说明解释器已就绪；如果提示“未找到命令”，说明还没装好。

**记住**：解释器就是那个把“人类文字”翻译成“电脑指令”的翻译官

## 块 2

**你的目标：LoRA 微调。** 这意味着你很快要跑模型训练代码。别急，先把“工作台”搭好——这就像做实验前先摆好桌子。

**1.2 代码编辑器：你的“工作台”**
记事本也能写代码，但你会疯掉。**IDE（集成开发环境）** 能给你语法高亮、自动补全、运行调试，就像用 Word 写文档一样舒服。推荐两款：
- **VS Code**：轻量、免费，装个 Python 插件就够用，适合快速上手。
- **PyCharm**：更专业、功能更全，但稍重。

**1.3 虚拟环境：项目的“独立房间”**
想象你同时做两个项目：A 需要 `numpy 1.0`，B 需要 `numpy 2.0`。都装进同一个 Python 会冲突报错。**虚拟环境** 就是给每个项目

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，这需要 Python 环境。跟着下面三步走，10 分钟搞定。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令行操作都无法进行——包括你之后微调要用到的命令。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。想改路径的话，确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python

## 块 4

### 2.2 macOS 用户：为 LoRA 微调铺好第一步

你的目标是 LoRA 微调，这意味着后面要跑 Python 代码、装 HuggingFace 工具库。所以，先把 Python 环境弄干净，别让系统自带的旧版本拖后腿。

**关键认知**：现代 macOS（Catalina 10.15 起）不再预装 Python 2，但可能带一个供系统脚本用的旧版 `python3`。**别直接用系统版本**，自己装最新的，避免后续装包时权限或版本冲突。

---

**方法一：官方安装包（推荐，最省心）**

1. 打开浏览器，访问 https://www.python.org
2. 下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）
3. 双击下载的文件，一路点 "Continue" → "Install"，输密码完成安装
4. 打开终端（Terminal），输入下面两行验证：

```bash
python3 --version
pip3 --version
```

看到版本号输出就成功了。

**为什么用 `python3` 而不是 `python`**：现代 macOS 可能仍保留 `python` 命令（通常指向旧版 Python 2），但为了确保使用最新版本，建议统一使用 `python3` 和 `pip3`

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是 LoRA 微调，这需要 Python 环境。Ubuntu/Debian 通常自带 Python，但版本偏老，我们直接装新的。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**关键点**：`-y` 让系统自动确认，不用手动输 `y`；`python3-venv` 是建虚拟环境的模块，现在装好，后面微调时隔离环境不踩坑。

**本块学习检查**：执行完 `python3 --version`，把输出的版本号记下来——如果

## 块 6

**本块学习检查**：学完本块后，请完成以下操作验证（不引入新知识）：  
1. 如果你是 Windows 用户，在 PowerShell 中执行 `wsl --install` 并确认 WSL2 已启用；  
2. 无论你使用哪种系统，在终端中运行 `python --version`，确认 Python 已安装（版本号需为 3.10 或更高）；  
3. 写下你计划用于 LoRA 微调的实训环境（云服务器或本地 Linux），并说明它是否满足“Ubuntu LTS + NVIDIA GPU”这一条件。

---

### 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，这意味着你最终要跑 GPU 训练。选对环境，能让你从现在到微调少走弯路。

**日常写 Python、练命令行（Windows 用户）**  
推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里拥有一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**  
直接用本机

## 块 7

## 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、最应保留的默认源**。

**你只需要记住一个判断标准**：只有当你访问官方源明显变慢、经常超时，或处在网络受限环境时，才临时切换国内镜像。镜像是“网络不好时的临时加速手段”，**不是必须永久替换的默认设置**——官方源始终是首选。

## 3.2 首选做法：临时指定镜像源（推荐）

如果你只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为 LoRA 微调做准备，而微调的第一步就是顺利装好依赖包。如果你**长期处于网络受限环境**，几乎每次安装包都要手动加镜像地址，那才值得把镜像写进配置文件设为默认。**注意**：官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

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

### 3.5

## 块 9

### 你的目标：为 LoRA 微调搭好“代码车间”

既然你要做 LoRA 微调，后面会写大量 Python 脚本、调试模型代码。VS Code 就是你的“工作台”——轻量、免费，启动快，插件生态强，后续写 Linux 命令、改配置文件也全靠它。

**安装三步走（跟着点就行）：**

1. 打开官网 `https://code.visualstudio.com`，下载你系统的安装包。
2. 双击运行，一路“下一步”装完。
3. **装两个必装插件**：打开 VS Code，点左侧活动栏最下方的**四个方块图标**（扩展），搜索并安装：
   - **Python**（微软官方出品，语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更聪明）

> 为什么必须装？不装 Python 插件，VS Code 认不出 `.py` 文件，代码

## 块 10

**动手装好 PyCharm，为 LoRA 微调铺路**

你目标是跑 LoRA 微调，代码量不小，选 PyCharm Community 版最省心：免费、专业、开箱即用，内置代码分析和版本控制，适合管理微调项目。

**安装三步走**

1. 打开官网下载页：https://www.jetbrains.com/pycharm/download/
2. 选 **Community Edition**（社区版，免费那个）
3. 运行安装程序，务必勾选这两项：
   - `Add "Open Folder as Project"` —— 以后右键文件夹就能直接打开项目
   - `Add to PATH` —— 命令行也能启动 PyCharm

**首次启动配置**

1. 打开 PyCharm，点 `New Project`
2. `Location` 选项目存放路径，**务必用英文或拼音命名文件夹，确保路径中不含中文字符**，否则后续跑微调脚本容易出编码问题
3. `Interpreter` 选 `Previously configured interpreter

## 块 11

### 5.1 为什么用 Conda

你的目标是 LoRA 微调，这意味着后面要装 PyTorch、CUDA 这类“重”依赖。如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 能帮你为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身——这正是深度学习项目的“必备工具”。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，按你的情况（希望环境干净、按需装包）**推荐 Miniconda**：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 1GB+），预装 numpy、pandas、jupyter、scipy、matplotlib 等数百个常用包 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50-100MB），只含 conda 和基础 Python | 希望环境干净、做项目开发的用户 |

**安装步骤（跟着做就行）：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**不要勾选 “Add Miniconda3 to my PATH environment variable”**（安装器默认不勾选，并标注为 Not recommended）。直接塞进 PATH 容易和系统已有的 Python 冲突。日常改用开始菜单里的 **“Anaconda Prompt (Miniconda3)”** 来用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（默认选项为 yes）。这样终端会自动接管 conda，不用手动改 PATH。
4. **验证安装**：打开一个**新的**终端（Windows 用 “Anaconda Prompt (Miniconda3)”；macOS/Linux 重开终端），输入

## 块 12

**动手创建你的第一个环境**

你的目标是 LoRA 微调，而微调不同项目往往需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

打开终端，执行：

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

输入 `y` 并回车，等待安装

## 块 13

## 5.4 激活与退出环境：进入你的“独立房间”

你的目标是 LoRA 微调，这意味着你马上要开始动手训练模型了。训练时最怕的就是把项目依赖搞乱，所以 Conda 环境就是你的“独立房间”——每个项目住一间，互不打扰。

**激活环境**（走进房间）：

```bash
conda activate llm-learn
```

**怎么确认你进来了**：激活后，命令行最前面会出现 `(llm-learn)` 标志。看到它，就说明你后续敲的 `python` 和 `pip` 命令都只作用于这个环境，不会污染系统或其他项目。

**验证一下**（确保房间里的 Python 版本正确）：

```bash
python --version
```

应该显示 `Python 3.10.x`。

**退出环境**（走出房间，回到系统默认环境）：

```bash
conda de

## 块 14

**本块学习检查**：完成配置后，在 VS Code 或 PyCharm 中新建一个 Python 文件，输入 `print("hello")` 并运行，观察终端输出是否为 `hello`，确认使用的是 `llm-learn` 环境。

---

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境，选好后，你在这个工作区写的代码就会用 `llm-learn` 环境里的 Python 和已装好的包。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\minic

## 块 15

### 在命令行验证你的 Python 环境

既然你的目标是快速上手 LoRA 微调，先确保工具链能跑通是关键。跟着做，很快。

1. **激活环境**：打开终端，输入 `conda activate llm-learn` 并回车。
2. **进入 Python**：输入 `python`，回车。看到 `>>>` 提示符就对了。
3. **运行测试**：在 `>>>` 后输入 `print("Hello, Python! 我的环境准备好了！")`，回车。
4. **确认输出**：如果屏幕打印出这行文字，说明环境正常。输入 `exit()` 退出。

**本块学习检查**：完成上述步骤后，在终端重新输入 `python`，再执行一次 `print("OK

## 块 16

## 7.3 在 PyCharm 中验证——跑通你的第一行代码

你的目标是 LoRA 微调，而微调的第一步，是确保你的开发环境能“跑起来”。这一节我们用最直接的方式验证：在 PyCharm 里运行一个 Python 文件，看到输出，就算打通了。

**操作步骤（三步走）**

1. **新建项目，创建 `hello.py`**  
   打开 PyCharm，新建一个项目，然后在项目里新建一个 Python 文件，命名为 `hello.py`。这个文件就是你的“试验田”。

2. **右键点击文件 → `Run 'hello'`**  
   在 `hello.py` 的编辑区空白处**右键单击**，菜单里会出现 `Run 'hello'` 选项，点击它。PyCharm 会自动配置运行环境并执行这个文件。

3. **下方 Run 窗口应输出结果**  
   点击运行后，PyCharm 底部会弹出 **Run 窗口**，你的代码输出会显示在这里。看到输出，就说明 PyCharm 已经能正常执行 Python 代码了。

> **为什么这对 LoRA 微调很重要？**  
> 微调涉及大量代码运行和调试，你需要在 PyCharm 里反复运行脚本、查看输出。这一步验证通过，等于确认了“写代码→运行→看结果”这条链路是通的，后面学 LoRA 时才能专注在模型本身，而不是卡在环境上。

---

## 第八章：常见问题排查（先收藏，遇到再查）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/M
