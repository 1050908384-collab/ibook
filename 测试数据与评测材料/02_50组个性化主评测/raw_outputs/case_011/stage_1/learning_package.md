# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

**别急着下载，先搞懂这个核心概念。**

你的学习目标是 **LoRA 微调**，这需要你写 Python 代码来训练模型。但电脑本身**只认 0 和 1**，看不懂你写的英文代码。

**Python 解释器**就是那个"翻译官"：它把你写的、人类能读懂的代码，**实时翻译成电脑能执行的指令**。

**关键点**：没有解释器，你写的 `.py` 文件对电脑来说**只是一堆普通文本**，毫无作用。

**动手验证一下（30秒）**：
1. 随便新建一个文本文件，改名为 `test.py`
2. 用记事本打开，输入 `print("hello")` 并保存
3. 双击运行——你会发现它只是用记事本打开了，什么都没发生

**结论**：这就是因为没有"翻译官"（解释器）来处理它。下一节，我们就来安装这个"翻译官"。

## 块 2

**你的工作台：从记事本到IDE**

写LoRA代码，记事本也能凑合，但你会疯掉。**IDE（集成开发环境）**就是给你配了语法高亮、自动补全、运行调试的“智能工作台”，写代码像用Word一样顺手。推荐两款：**VS Code**（轻量免费，上手快）和**PyCharm**（专业强大，适合深度调试）。你追求速度，先装VS Code，够用且不拖累。

**虚拟环境：项目的“独立房间”**

你马上要跑LoRA微调，会装一堆依赖。假设项目A要`numpy 1.0`，项目B要`numpy 2.0`，全塞进同一个Python里必冲突。**虚拟

## 块 3

### 2.1 Windows 用户：装好 Python，为 LoRA 微调铺路

你接下来的目标是 LoRA 微调，这需要 Python 环境作为地基。跟着下面三步走，10 分钟搞定。

**第一步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标悬停顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令操作都会卡住——这是新手最常见的坑。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --

## 块 4

### macOS 用户：安装 Python 3.10+

macOS 12.3 及更高版本不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用。为了 LoRA 微调，请**自行安装最新版 Python**，别依赖系统版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install"。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：现代 macOS 默认**没有** `python` 命令，统一用 `python3` / `pip3`。始终这样输入，避免“命令不存在”或误用系统旧版。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

终端执行：

```bash
brew install python
```

安装后同样用 `python3 --version` 验证。

---

**本块学习检查**：在终端运行 `python3 --version`，确认输出为 3.10 或更高版本。若版本过低，请重新下载官方安装包覆盖安装

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是 LoRA 微调，第一步得先把 Python 环境弄利索。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省心。

打开终端，逐行执行：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**关键点**：`-y` 让安装过程全自动，不用你手动确认；`python3-venv` 是后面建虚拟环境（隔离项目依赖）的必备模块，现在一起装好

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，那环境选择就一句话：**跟着任务走，别贪多**。

**如果你平时写 Python、练命令行（Windows 用户）**：直接启用 **WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04 或 24.04 LTS）。它相当于在 Windows 里开了一个接近真实服务器的 Linux 环境，命令、路径、包管理方式都和后面要用的服务器一致。安装很简单：用管理员身份打开 PowerShell，执行 `wsl --install` 即可。

**如果你用 macOS 或 Linux**：不用额外装东西，直接用本机终端就行，它本身就是类 Unix 环境，和服务器天然一致。

**到了 LoRA / QLoRA 微调、vLLM 部署这类 GPU 实训**：必须用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。原因很直接：这类任务依赖 NVIDIA 驱动、CUDA 和显卡直通，而 Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别踩这个坑**：不要用 **"CentOS + VirtualBox 虚拟机"** 当主力方案。CentOS 8 已于2021年停止维护，生态

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、应保留的默认源**。

只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：它不影响你平时的默认行为，官方 PyPI

## 块 8

## 3.3 进阶：把镜像设为默认（长期网络受限才用）

你正在为 LoRA 微调做准备，后面会频繁用 pip 安装 PyTorch、transformers 等库。如果网络长期受限，每次手动加镜像地址太麻烦，可以一次性写入配置。

**⚠️ 前提提醒**：只有你**长期处于网络受限环境**、几乎每次都要用镜像时才这样做。官方 PyPI 始终是最可靠来源，网络恢复后记得改回或删除配置。

---

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从清华镜像下载包。

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

3. 写入同样内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

4. `Ctrl + O` 保存，`Ctrl + X` 退出。

---

### ⚠️ 关于 `trusted-host`：别乱加

网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且确认来源可信时，才临时加。

---

### 验证配置是否生效

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`

## 块 9

**动手搭好你的 LoRA 微调工作台：VS Code 安装与配置**

你的目标是 LoRA 微调，而微调离不开写代码。VS Code 轻量、免费、启动快，是微软官方出品，后续写 Python 脚本、改配置文件都能用上，现在装好它，等于给微调实验备好了“主战场”。

**安装步骤（3 步搞定）**

1. 打开浏览器，访问 `https://code.visualstudio.com`，下载对应你系统的安装包。
2. 运行安装程序，一路点“下一步”即可，不用改任何选项。
3. **装两个必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），在搜索框里依次搜并安装：
   - **Python**（微软官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

> 为什么必须装？不装 Python 插件，VS Code 就认不出 `.py` 文件，代码补全、错误提示、一键运行这些功能全都没有，写微

## 块 10

**本块学习检查**：安装完成后，在 PyCharm 中新建项目并选择你系统的 Python 3.10+，确认能成功创建并打开项目窗口。

---

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版免费，内置代码分析、版本控制、数据库工具，适合大型项目——你后续做 LoRA 微调会涉及多文件工程，用它管理更省心。

**安装步骤**（跟着做）：
1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费）下载。
3. 运行安装程序，**务必勾选**：
   - `Add "Open Folder as Project"`（右键快速打开项目文件夹）
   - `Add to PATH`（命令行可直接启动）

**首次启动配置**：
1. 打开 PyCharm，点 `New Project`。
2. `Location`

## 块 11

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，这需要 PyTorch、CUDA 这些“重型”依赖。如果直接在系统 Python 里装，很容易把环境搞乱。Conda 就是用来给每个项目建一个独立、干净的“操作间”，互不干扰。

**为什么选 Conda？**
Conda 不仅能创建虚拟环境，还能直接管理 Python 版本。深度学习项目对 Python 版本和库的版本要求很严格，用 Conda 可以一键切换，非常方便。

**安装哪个？Anaconda 还是 Miniconda？**
- **Anaconda**：体积大（500MB+），预装了很多数据分析库。适合不想手动装包的人。
- **Miniconda**：体积小（约50MB），只包含核心工具。**推荐你选这个**，环境干净，需要什么装什么，更符合项目开发习惯。

**动手安装 Miniconda（三步走）**

1.  **下载**：访问 `https://docs.conda.io/en/latest/miniconda.html`，下载对应你系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
2.  **安装**：
    - **Windows**：一路下一步。**关键一步**：不要勾选 “Add Miniconda3 to my PATH environment variable”（安装器会标记为“不推荐”）。这能避免和系统已有的 Python 冲突。以后用开始菜单里的 **“Anaconda Prompt (Miniconda3)”** 来操作。
    - **macOS/Linux**：运行 `.sh` 文件，输入 `yes` 同意协议，最后一步也输入 `yes` 让安装器执行 `conda init`。这样终端会自动配置好，不用手动改 PATH。
3.  **验证**：**新开一个**终端（Windows 用 Anaconda Prompt），输入 `conda --version`，看到类似 `conda 24.x.x` 就成功了。

**本块学习检查**
打开你的终端（或 Anaconda Prompt），运行 `conda --version`，把输出结果记下来。如果提示找不到命令，回顾一下 Windows 是否用错了终端，或者 macOS/Linux 是否重开了终端。

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：你之后做 LoRA 微调时，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包

## 块 13

## 5.4 激活与退出环境

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

## 第六章：在 VS Code / PyCharm 中使用 Conda

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。

打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境。选好后，你在这个工作区写的代码就会用 `llm-learn` 里的 Python 和已装好的包。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件。先运行 `conda env list` 查看实际路径，再按系统类型选择：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境

## 块 15

**本块学习检查**：完成下面两步，确保你的“工作台”能跑起来。

1. **命令行验证**：打开终端，输入 `conda activate llm-learn` 激活环境，再输入 `python` 进入交互模式。看到 `>>>` 提示符后，输入 `print("Hello, Python! 我的环境准备好了！")` 并回车。若正确打印文字，输入 `exit()` 退出。若 `conda activate` 报错，请检查 conda 是否安装并初始化；若 `python` 命令找不到，请确认 Python 已加入 PATH 或使用 `python3`。
2. **VS Code 验证**：新建文件夹 `my-first-project` 并用 VS Code 打开，

## 块 16

好的，我们直接进入正题。你的目标是 LoRA 微调，这需要你亲手操作大量代码。所以，我们现在就要确保你的“工作台”完全顺手。既然你喜欢动手实践，我们就不空谈理论，直接来验证你的 PyCharm 环境是否一切就绪。

### 7.3 在 PyCharm 中验证：你的第一个“触点”

这一步是建立信心和确认环境通畅的关键。请跟着做：

1.  **新建项目**：打开 PyCharm，点击 `File` → `New Project`。在创建向导中，**务必**展开 `Project Interpreter` 或 `Python Interpreter` 选项，选择我们之前创建好的 `llm-learn` 环境（如果找不到，可以点击设置图标手动添加）。项目位置建议放在一个纯英文路径下，比如 `D:\projects\my_first_project`，避免中文或空格。
2.  **创建文件**：在项目目录上右键 → `New` → `Python File`，命名为 `hello.py`。
3.  **编写代码**：在文件中输入 `print("Hello, LoRA!")`。
4.  **运行验证**：在 `hello.py` 文件内任意位置右键，选择 `Run 'hello'`。此时，PyCharm 底部的 `Run` 窗口会弹出，并显示 `Hello, LoRA!`。

看到这行输出，就代表你的 Python 解释器、项目配置和 IDE 都工作正常，可以开始后续的微调实战了。

### 7.4 常见问题排查（你的“急救包”）

在动手过程中，你可能会遇到一些环境问题。这里有一份快速排查表，遇到问题直接对照解决：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行
