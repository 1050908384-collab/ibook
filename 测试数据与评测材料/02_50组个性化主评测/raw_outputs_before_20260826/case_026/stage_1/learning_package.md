# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

**别急着下载，先搞懂你在装什么。**

你写的 Python 代码是给人看的文字，但电脑只认 0 和 1。**Python 解释器**就是那个"翻译官"——它把你写的代码翻译成电脑能执行的指令。

**关键点**：没有解释器，你写的 `.py` 文件只是一堆文本，电脑根本看不懂。

**为什么这对你重要？**

你的目标是 **LoRA 微调 + 权重合并 + vLLM 部署**。这条路线上，你写的每一行训练脚本、推理代码，最终都要靠 Python 解释器翻译成 GPU 能执行的指令。解释器装不对，后面跑 LoRA 训练时可能直接报错，让你误以为是代码问题，其实是环境问题。

**动手验证一下（1 分钟）**：

打开终端（Mac/Linux）或命令提示符（Windows），输入：

```bash
python --version
```

如果看到类似 `Python 3.x.x` 的输出，说明解释器已经就位；如果提示"找不到命令"，说明还没装好——这正是下一节要解决的。

> **记住**：环境准备阶段多花 2 分钟理解概念，后面跑 LoRA 和 vLLM 时能少踩一半的坑。

## 块 2

**IDE 与虚拟环境：动手前的两块基石**

你的目标是跑通 LoRA 微调和 vLLM 部署，这需要写代码、装依赖。记事本写代码效率太低，**IDE（集成开发环境）**能提供语法高亮、自动补全、运行调试，就像用 Word 写文档一样顺手。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个装上即可。

更关键的是**虚拟环境**。想象你同时做两个项目：A 需要 `numpy 1.0`，B 需要 `numpy 2.0`，都装进同一个 Python 就会冲突报错。虚拟环境就是给每个项目一个独立的 Python 空间，互不干扰。我们使用 **conda** 管理

## 块 3

### Windows 用户：安装 Python 3.10+（动手实操版）

你的目标是跑通 LoRA 微调和 vLLM 部署，所以 Python 环境必须一次装对。跟着下面三步走，全程约 10 分钟。

**第一步：下载安装包**

打开浏览器，访问 https://www.python.org，鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这是最关键的一步。勾选后，Python 会被加入系统"环境变量"。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作（包括装 LoRA 依赖、启动 vLLM）都会卡死。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按

## 块 4

**本块学习检查**：安装完成后，在终端依次运行 `python3 --version` 和 `pip3 --version`，确认两条命令都能输出版本号且无报错。

---

### 2.2 macOS 用户：安装 Python 3.10+

**为什么必须自己装？**

现代 macOS（Catalina 10.15 起）不再预装 Python 2，但系统自带一个旧版 Python 3（`/usr/bin/python3`），版本可能较旧，且不建议直接依赖，**强烈建议自行安装最新版**。

**方法一：官方安装包（推荐，新手友好）**

1. 打开浏览器，访问 `https://www.python.org`，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的 `.pkg` 文件，按提示一路点击 **Continue** → **Install**，等待安装完成。
3. 打开 **终端**（Terminal，按 `Cmd + 空格` 搜索即可），输入以下两条命令验证：

```bash
python3 --version
pip3 --version
```

看到版本号输出，说明安装成功。

**关键提醒：用 `python3` 而不是 `python`**

现代 macOS 上，命令行默认**没有** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统与

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是跑通 LoRA 微调和 vLLM 部署，但这一切都建立在 Python 环境之上。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省心。

打开终端，依次执行：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**命令拆解**：`-y` 让系统自动确认安装，省去手动输入 `y` 的步骤；`python3-venv` 是创建虚拟环境的关键模块

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是跑通 **LoRA 微调 + 权重合并 + vLLM 部署**，所以环境选择直接决定后面能不能顺利动手。记住一条主线：**日常练习贴近生产，GPU 实训必须 Linux**。

**日常写 Python、练命令行（Windows 用户）**
启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比在 Windows 原生 CMD 里练更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**
直接用本机终端，本身就是类 Unix 环境，不用额外配置。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**
用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别踩坑**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于2021年12

## 块 7

你正处在“LoRA微调+权重合并+vLLM部署”这条路上，很快就要安装大量依赖库。这里先解决一个最实际的问题：**什么时候才需要换镜像源？**

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**只有当你**处在网络受限环境、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终应作为首选。

**首选做法：临时指定镜像源（推荐

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调和 vLLM 部署做准备，安装依赖时如果每次都要手动指定镜像源，确实很烦。但请注意：**只有当你长期处于网络受限环境、几乎每次都要用镜像时**，才考虑把它写入配置作为默认源。配置后请记得：官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

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

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

你的目标是快速跑通 LoRA 微调和 vLLM 部署，所以开发环境必须轻、快、不折腾。VS Code 正合适——免费、启动快、插件多，以后写脚本、改配置也能复用。

**安装步骤（3 分钟搞定）：**

1. 打开 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路“下一步”即可，不用改任何选项。

**装完必做：装两个插件**

点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

- **Python**（微软官方出品）——装完后 VS Code 才能识别 `.py` 文件，提供语法高亮、调试和错误提示。
- **Pylance**——Python 语言服务器，让代码补全和提示更智能。

**配置 Python 解释器（关键一步）：**

1. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`）打开命令面板。
2. 输入 `Python: Select Interpreter`，选择

## 块 10

**动手装 PyCharm，为 LoRA 微调项目做准备**

你的目标是跑通 LoRA 微调 + vLLM 部署，代码量不小，需要一个趁手的 IDE。PyCharm Community 免费、专业，内置代码分析和版本控制，适合管理这类大型项目。

**安装要点**

1. 打开 https://www.jetbrains.com/pycharm/download/，下载 **Community Edition**（社区版）。
2. 安装时务必勾选两项：
   - `Add "Open Folder as Project"`——右键文件夹即可打开项目，省事。
   - `Add to PATH`——方便在命令行直接启动。

**首次启动配置**

1. 打开后选 `New Project`。
2. `Location` 选项目路径，**路径别带中文**，避免后续工具链报错。
3. `Interpreter` 选 `Previously configured interpreter`，指向你系统里已装的 Python 3.10+。

**本块学习检查**

装好后，新建一个项目，确认右侧解释器显示的是 Python 3.10+，且项目路径无

## 块 11

你的目标很明确：**LoRA微调 + 权重合并 + vLLM部署**。这条路走通，意味着你要在多个项目环境里反复切换、安装各种深度学习依赖（PyTorch、CUDA 等）。如果不用虚拟环境，这些依赖会互相打架，装一个坏一个。**Conda** 就是来解决这个问题的——它不仅能创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备工具。

**安装选哪个？Miniconda。**

Anaconda 和 Miniconda 都能用，但 Anaconda 体积大（500MB+），预装了一堆你可能用不上的库；Miniconda 体积小（约 80-100MB），只带 conda 和基础 Python，需要什么装什么。你的目标是项目开发，环境越干净越好，所以**选 Miniconda**。

**安装步骤（按你的系统来）：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应安装包：Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`
3. 运行安装程序，注意这几点：
   - **Windows**：一路下一步，**选择 Just Me 安装**，**千万别勾选 "Add Miniconda3 to my PATH environment variable"**（安装器也标了"不推荐"）。直接塞进系统 PATH 容易和已有 Python 冲突，导致 `python`、`pip` 指向不明。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来操作 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选安装路径，最后一步**必须选 `yes` 让安装器执行 `conda init`**。这样终端每次打开时 conda 会自动接管，而不是靠硬加 PATH。
4. **验证安装**：打开一个**新的**终端（Windows 用 "Anaconda Prompt (Miniconda3)"；macOS/Linux 关掉终端重开），输入：

```bash
conda --version

## 块 12

### 创建你的第一个虚拟环境

**为什么现在学这个**：你的目标是LoRA微调+权重合并+vLLM部署。后续不同项目会需要不同版本的PyTorch和CUDA，现在掌握这套流程，后面直接复用。

**动手操作**：打开终端，执行下面这行命令：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（三个部分）：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境命名为`llm-learn`（`n`是`name`的缩写）
- `python=3.10`：指定环境内安装Python 3.10

执行后，终端会列出待安装的包，并提示：

```bash

## 块 13

### 5.4 激活与退出环境：进入你的项目“独立房间”

你的目标是完成 LoRA 微调和 vLLM 部署，这两项任务都依赖干净的 Python 环境。Conda 环境就像为每个项目准备的独立房间，避免依赖冲突。

**激活环境**（进入房间）：

```bash
conda activate llm-learn
```

激活后，命令行前会出现 `(llm-learn)` 标志。这意味着后续所有 `python` 和 `pip` 命令都只作用于这个环境，不会影响系统或其他项目——这正是你后续安装 LoRA 和 vLLM 相关库时需要的隔离性。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应显示 `Python 3.10.x`。确认版本正确，避免后续微调脚本因版本不匹配报错。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

### 5.5 查看

## 块 14

**本块学习检查**：完成配置后，在 VS Code 或 PyCharm 中新建一个 `.py` 文件，输入 `import sys; print(sys.executable)`，运行并确认输出的路径中包含 `llm-learn`。

---

你的目标是 LoRA 微调 + 权重合并 + vLLM 部署，这意味着你很快就要在真实项目中跑代码了。**先别急着写模型代码，把开发环境这把“钥匙”配好，后面才能顺畅地调包、跑训练。**

你偏好动手实操，所以我们直接按步骤来。这里以 VS Code 为例（PyCharm 步骤在最后，原理相同）。

### 6.1 VS Code 配置 Conda 环境

1. **激活环境**：在终端运行 `conda activate llm-learn`。这是前提，确保后续操作基于正确的环境。
2. **打开命令面板**：在 VS Code 中按 `Ctrl + Shift + P`。
3. **选择解释器**：输入并选择 `Python: Select Interpreter`。
4. **指定 conda 环境**：在列表中，选择带有 `conda` 字样的 Python 路径，例如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这么做**：VS Code 会自动识别你创建的 conda 环境。选中后，你在当前工作区写的所有代码，都会使用 `llm-learn` 环境里的 Python 和已安装的包（比如后面要用的 `torch`

## 块 15

### 7.1 在命令行验证

1. 先激活环境：`conda activate llm-learn`
2. 输入 `python` 回车，进入交互模式。
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
3. 点击右上角小三角（运行按钮），或按 `Ctrl + F5`。

## 块 16

## 7.3 在 PyCharm 中验证

你的目标是 LoRA 微调 + 权重合并 + vLLM 部署，这条链路很长，但每一步都建立在“代码能跑起来”这个地基上。现在我们就用 PyCharm 完成第一次运行验证，确认你的开发工作台真正可用。

**操作步骤：**

1. 打开 PyCharm，新建一个项目（项目路径建议不要包含中文或空格，这是本章强调的好习惯）。
2. 在项目中新建一个 Python 文件，命名为 `hello.py`。
3. 在文件中输入一行代码：`print("Hello, World!")`。
4. 在文件编辑区**右键点击**，选择 `Run 'hello'`。
5. 观察 PyCharm 下方弹出的 **Run 窗口**，如果看到输出了 `Hello, World!`，说明你的 Python 解释器配置正确，开发环境已就绪。

**验证要点：** Run 窗口没有报错、且能看到输出结果，即代表验证通过。

---

## 第八章：常见问题排查

在后续安装 LoRA、vLLM 等库时，你大概率会遇到下面这些问题。先记住这张表，遇到时对照处理：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
