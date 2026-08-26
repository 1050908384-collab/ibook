# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

**先别急着下载，花2分钟搞懂这个核心概念。**

你写的Python代码是给人看的文字，但电脑只懂0和1。**Python解释器**就是那个"翻译官"——它把你的代码实时翻译成电脑能执行的指令。

**为什么必须装它？**
没有解释器，你写的 `.py` 文件对电脑来说就是一堆普通文本，双击也不会运行。装了它，你的代码才能被"听懂"并执行。

**记住这个比喻：**
- 你（写代码）→ 说中文
- 电脑（执行）→ 只懂二进制
- 解释器 → 同声传译，把你的话翻给电脑听

**动手验证一下（约1分钟）：**
1. 打开终端（Windows搜索"cmd"，Mac打开"终端"）
2. 输入 `python` 并回车
3. 看到 `>>>` 提示符，说明解释器已就位，正在等你"发话"
4. 输入 `print("hello")` 回车，看到输出 `hello`，就完成了第一次"翻译"

**小提示：** 如果提示"不是内部或外部命令"，说明解释器还没装好或没加入系统路径，回到安装步骤检查即可。

现在你知道了：**装Python，核心就是装这个"翻译官"**。接下来我们去看怎么把它请到电脑里。

## 块 2

**本块学习检查**：读完本节后，请合上资料，用一句话分别说出IDE和虚拟环境各自解决什么问题，再对照原文检查。

---

你的目标是LoRA微调，这意味着很快要跑深度学习代码。记事本写代码效率极低，所以你需要一个IDE（集成开发环境）——它提供语法高亮、自动补全、运行调试，写代码像用Word写文档一样舒服。推荐**VS Code**（轻量免费）或**PyCharm**（专业强大），选VS Code起步即可。

微调LoRA时，不同项目依赖版本常冲突。比如项目A要`numpy 1.0`，项目B要`numpy 2.0`，装一起会报错。**虚拟环境**就是给每个项目一个独立Python空间，互不影响。用

## 块 3

**动手装好 Python，为 LoRA 微调铺路**

你接下来的目标是 LoRA 微调，这需要先有一个能跑 Python 的环境。别担心，Windows 上装 Python 很简单，跟着下面三步走，10 分钟搞定。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示

## 块 4

你的目标是 **LoRA 微调**，而 LoRA 微调需要你亲手操作大模型，所以一个干净、可控的 Python 环境是第一步。macOS 上最省心的做法就是**别用系统自带的 Python**，自己装一个全新的。

**方法一：官方安装包（推荐新手，最省心）**

1.  打开浏览器，访问 `https://www.python.org`，下载 **macOS 64-bit universal2 installer**（一个 `.pkg` 文件）。
2.  双击下载的文件，一路点 **Continue** 和 **Install**，输一下你的电脑密码，装完就行。
3.  打开 **终端**（Terminal，在“启动台”的“其他”文件夹里能找到），输入下面两行命令，每行输完按回车：

```bash
python3 --version
pip3 --version
```

看到版本号输出，就说明装好了。

**为什么用 `python3` 而不是 `python`？**

现代 macOS 上，系统默认不提供 `python` 命令，通常使用 `python3`；若你安装了 Python 3，`python` 可能也可用，但为了兼容性，建议统一使用 `python3`。所以，**以后你所有命令都写 `python3` 和 `pip3`**，能避开很多“

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你正朝着 LoRA 微调前进，而第一步就是让电脑里有一个能用的 Python。Linux 系统通常自带 Python，但版本可能偏旧，咱们直接装最新的稳定版。

打开终端，依次执行下面三行命令：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**命令拆解**：`-y` 让系统自动确认安装，不用你手动敲 `y`；`python3-venv` 是虚拟环境模块，LoRA 微调时

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选错，后面每一步都会卡壳。按你的学习节奏，直接给你结论：

**日常练命令、写 Python**，如果你是 Windows 用户，**启用 WSL2 + Ubuntu LTS**（比如 Ubuntu 22.04/24.04 LTS）。它是在 Windows 里跑一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和之后上服务器一致，比在 Windows 原生 CMD 里练更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。如果你是 macOS 或 Linux 用户，直接用本机终端就行，本身就是类 Unix 环境。

**做 LoRA / QLoRA 微调这类 GPU 实训**，要用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。原因很直接：这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，而 Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 别踩的坑**：不要把 **“CentOS + VirtualBox 虚拟机”当最佳方案**。CentOS 8 已停止维护、生态落后；VirtualBox 这类桌面虚拟机**默认无法直通 NVIDIA GPU**，跑不了真正的

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

**什么时候才用镜像？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为 LoRA 微调做准备，后面会频繁用 pip 安装 PyTorch、Transformers 等大模型相关库。如果每次下载都卡在“墙”外，效率会很低。这一节教你**一次性配置**，让 pip 默认走清华镜像，省下每次敲命令的功夫。

**但请记住前提**：只有当你**长期处于网络受限环境**、几乎每次都要用镜像时才这么做。官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除配置。

---

**Windows 用户（跟着做，5 分钟搞定）：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

---

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

## 块 9

**目标对齐**：你要做 LoRA 微调，后面会频繁写 Python 脚本、改配置文件。VS Code 轻量、免费、启动快，插件生态强，正好覆盖这些场景，所以选它当工作台。

**安装三步走**（照着做，5分钟搞定）：

1. 打开官网 `https://code.visualstudio.com`，下载你系统的安装包。
2. 运行安装程序，一路“下一步”到底，不用改任何选项。
3. 装完打开 VS Code，点左侧活动栏最下方的**四个方块图标**（扩展商店），搜索并安装这两个插件：
   - **Python**（微软官方，提供语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更聪明）

> 为什么必须装？不装 Python 插件，VS Code 认不出 `.py` 文件，没有代码补全、错误提示，也没法一键运行——写微

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制等功能，适合大型项目开发——这正是你后续做 LoRA 微调时管理复杂代码的好帮手。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费）
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`
2. 在 `Location` 处选择项目路径（**路径中不要含中文**）
3. `Interpreter` 选择 `Previously configured interpreter`，然后选系统安装的 Python 3.10+

---

## 第五章：配置 Conda 虚拟环境（

## 块 11

好的，收到你的学习需求。我们直接切入正题，用最直接的方式搞定环境，为你的 LoRA 微调目标铺路。

## 5.1 为什么用 Conda：给你的项目一个干净的家

你马上要开始 LoRA 微调，这需要 PyTorch、CUDA 等一堆复杂的依赖。如果所有项目都混在一个 Python 里，版本冲突会让你寸步难行。**Conda** 就是解决这个问题的利器。它不仅能创建独立的虚拟环境，还能直接管理不同版本的 Python 解释器。对于深度学习，这几乎是必备技能。

## 5.2 安装 Conda：选 Miniconda，轻装上阵

Conda 有两个主流版本，我们直接看对比：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（500MB+），预装了大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约50MB），只包含 conda 和基础 Python | **希望环境干净、做项目开发的用户** |

**我们选 Miniconda**。原因很简单：环境干净，需要什么装什么，不浪费空间，也减少潜在冲突。

**安装步骤（跟着做就行）：**

1.  访问 <https://docs.conda.io/en/latest/miniconda.html>。
2.  下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3.  运行安装程序，**注意系统差异**：
    *   **Windows**：一路下一步，默认不勾选 "Add Miniconda3 to my PATH environment variable"，但若希望在任意终端使用 conda，可勾选；日常使用也可从开始菜单打开 "Anaconda Prompt (Miniconda3)" 来操作。
    *   **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择路径。**最后一步务必选择 `yes` 执行 `conda init`**。这样终端才能正确接管 conda，而不是靠改

## 块 12

**5.3 创建你的第一个虚拟环境**

**为什么这样做**：你之后要学 LoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会建环境，以后直接复用这套流程。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境里装 Python 3.10

执行后，终端会列出要安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y`

## 块 13

## 5.4 激活与退出环境

你正在为 LoRA 微调准备环境，核心思路是：**每个项目一个独立环境，互不干扰**。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志。看到它，就表示后续所有 `python` 和 `pip` 命令都只作用在这个环境里，不会影响系统或其他项目——这对微调实验尤其重要，因为不同项目可能依赖不同版本的 PyTorch。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

---

## 5.5 查看和管理环境

**查看所有已创建的环境**：

```bash
conda env list
```

带 `*` 的就是当前

## 块 14

**本块学习检查**：按下面步骤操作，完成后在终端运行 `python -c "import torch; print(torch.__version__)"`，看到版本号即表示 VS Code 已正确使用 `llm-learn` 环境。

---

你正在为 LoRA 微调做准备，核心前提是让代码跑在正确的环境里。下面用最快路径完成 VS Code 配置。

**VS Code 三步走**：

1. 打开终端，先激活环境：`conda activate llm-learn`
2. 在 VS Code 中按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`
3. 在列表中选择带 `conda` 字样的 Python 路径，例如 `Python 3.10.x ('llm-learn': conda)`

**原理**：VS Code 会自动识别 conda 环境。选中后，当前工作区所有代码都会使用 `llm-learn` 里的 Python 和已安装的包——这确保你后续微调时用的 PyTorch 等库版本正确。

**PyCharm 配置（备用）**：

1. 打开 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`）
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`
3. 点齿轮图标 → `Add...`
4. 选 `Conda Environment` → `Existing environment`
5. 在 `Interpreter` 路径中选 conda

## 块 15

**动手验证：让 Python 回应你**

你的目标是 LoRA 微调，而一切微调都始于环境能跑通。现在，我们用最直接的方式——命令行——来确认你的工作台已就绪。

**第一步：激活环境**
在终端输入 `conda activate llm-learn`，回车。这是进入你专属 Python 环境的“钥匙”。

**第二步：进入交互模式**
输入 `python` 并回车。看到 `>>>` 提示符，说明 Python 正在等你下指令。

**第三步：输入你的第一行代码**
在 `>>>` 后输入：
```python
print("Hello, Python! 我的环境准备好了！")
```
按回车。如果屏幕打印出这句话，恭喜，环境验证成功。

**第四步：退出**
输入 `exit()`

## 块 16

**7.3 在 PyCharm 中验证**

你的目标是用 LoRA 微调模型，但一切得从“跑通第一行代码”开始。这一步就是确认你的开发台能正常运转，别跳过。

操作很简单，三步走：

1. **新建项目**，在里面创建一个 `hello.py` 文件。
2. **右键点击** `hello.py` 文件 → 选择 **`Run 'hello'`**。
3. 看下方弹出的 **Run 窗口**，如果输出了你写的内容，就说明环境没问题。

**动手验证一下**：在 `hello.py` 里写一行 `print("LoRA ready")`，然后运行。看到 `LoRA ready` 输出，就说明你的 PyCharm 能正常执行 Python 代码了。

---

**常见问题排查（遇到再看）**

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源） |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，用 `python -m pip install 包名` 安装 |

---

**本章小结**

完成本章后，你的电脑上应该具备：

| 组件 | 状态 | 验证命令 |
|------|------|----------|
| Python 3.10+ | ✅ 已安装 | `python --version` |
| pip 国内镜像源 | ⭕ 可选（网络受限时） | `pip config list` |
| VS Code 或 PyCharm | ✅ 已安装 | 打开软件确认 |
| Miniconda / Anaconda | ✅ 已安装 | `conda --version` |
| Conda 虚拟环境 `llm-learn` | ✅ 已创建 | `conda env list`
