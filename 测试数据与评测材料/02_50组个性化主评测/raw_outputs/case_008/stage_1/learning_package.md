# 学习包

## 块 1

**本块学习检查**：装好解释器后，打开终端（或命令提示符），输入 `python` 并回车。如果看到类似 `Python 3.x.x` 的版本信息，说明“翻译官”已就位；如果提示“不是内部或外部命令”，说明还没装好或没加入系统路径，回到安装步骤检查。

---

你写代码时，用的是人类能读懂的英文和符号；但电脑的“大脑”只认 0 和 1。这两者之间，需要一个“翻译官”——**Python 解释器**。

它的唯一工作，就是把你写的 `.py` 文件里的代码，逐句翻译成电脑能直接执行的机器指令。没有它，你写的代码就只是一堆普通的文本，电脑根本看不懂，更不会运行。

所以，搭建环境的第一步，就是请来这位“翻译官”。后面你安装的 PyCharm、VS Code 等工具，都只是“编辑器”，负责帮你写代码；真正让代码跑起来的，始终是这个解释器。

记住：**装解释器，才是装 Python 本身。** 其他工具，都是配合它工作的。

## 块 2

## 1.2 代码编辑器（IDE）：你的“工作台”

你要做 LoRA 微调，后面会反复改配置、调参数、看报错。用记事本写代码效率太低，**IDE（集成开发环境）** 能给你语法高亮、自动补全、运行调试，写代码就像用 Word 写文档一样顺手。推荐两款：

- **VS Code**：轻量、免费，插件生态丰富，适合快速上手。
- **PyCharm**：专业、强大，适合复杂项目深度开发。

**建议**：你每天只有 1.5 小时，追求快速上手，选 **VS Code** 就够了，装个 Python 插件就能跑。

## 1.3 虚拟环境：

## 块 3

好的，这是为你准备的个性化教学正文：

你的目标是搞定 LoRA 微调，而这一切都始于一个能跑代码的 Python 环境。别担心，安装很简单，我们一步步来。

**第一步：下载安装包**

打开浏览器，访问 Python 官网 `https://www.python.org`。把鼠标移到顶部菜单的 **Downloads** 上，网站会自动推荐适合你 Windows 系统的版本（比如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)** 这个安装包。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载好的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1.  **勾选最下方的 "Add python.exe to PATH"**。这一步至关重要！如果不勾选，你之后在命令行里输入 `python`，系统会提示“不是内部或外部命令”，后面所有操作都进行不下去。
2.  勾选后，直接点击 **"Install Now"**（默认安装）即可。默认安装路径不含中文和特殊符号，能避免后续很多莫名其妙的报错。

**第三步：验证安装**

1.  按 `Win + R` 键，输入 `cmd`，回车打开命令提示符。
2.  输入以下命令并回车：
    ```bash
    python --version
    ```
3.  如果显示了 `Python 3.13.x`（或你下载的版本号

## 块 4

你的目标是 LoRA 微调，而微调的第一步是让 Python 环境可靠可用。macOS 上最常踩的坑是“系统自带 Python”和“命令不存在”，下面直接给你两条最稳的路。

**方法一：官方安装包（推荐新手，最省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 “Continue” → “Install” 完成。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：macOS 10.15 及后续版本预装 Python 3（通常为 3.8 或更高），但系统自带的 Python 3 可能版本较旧，建议使用官方安装包或 Homebrew 安装最新版 Python 3.10+。命令行默认**不提供** `python` 命令，所以请始终用 `python3` / `pip3`，别用 `python`，否则会报“命令不存在”或误用系统版本。

**方法二：Homebrew（如果你已装过 Homebrew）**

终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

**本块学习检查**：打开终端，依次运行 `

## 块 5

## 动手装好 Python，为 LoRA 微调铺路

你的目标是 LoRA 微调，而跑微调的第一步，就是先把 Python 环境弄利索。在 Ubuntu/Debian 上，系统通常自带 Python，但版本可能偏老，不适合直接开干。按下面三步走，装好 Python 3 和配套工具：

```bash
# 1. 更新软件源，确保能拉到最新安装包
sudo apt update

# 2. 安装 Python 3、pip 和 venv（虚拟环境模块）
sudo apt install python3 python3-pip python3-venv -y

# 3. 验证安装结果
python3 --version
```

**命令小贴士**：

## 块 6

## 2.4 选对环境，少走弯路

你的目标是跑通 LoRA 微调，那环境选择就一句话：**跟着生产环境走，别在 Windows 里硬练。**

**日常写 Python、练命令（Windows 用户）**：直接启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得一个接近真实服务器的 Linux 环境，命令、路径、包管理都和后续云端一致。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**LoRA / QLoRA 微调等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别踩坑**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于2021年12月31日停止维护，CentOS 7 已于2024年6月30日停止维护，建议使用 CentOS Stream 或

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是最权威、最完整、最应保留的默认源。

**关键判断**：只有当你处在网络受限环境、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。镜像是“网络不好时的临时加速手段”，不是必须永久替换的默认设置——官方源始终应作为首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正在为 LoRA 微调做准备，后面会频繁用 pip 安装 PyTorch、Transformers 等依赖包。如果每次下载都卡在网络上，体验会很糟。但请记住：**只有当你长期处于网络受限环境、几乎每次都要用镜像时，才把镜像设为默认**。官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除配置。

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

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

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到

## 块 9

安装 Python 插件后，VS Code 能提供更强大的 Python 开发支持，如智能提示、调试和一键运行等功能。

## 块 10

**本块学习检查**：完成安装后，在 PyCharm 中新建一个项目，确认界面能正常显示代码编辑区，并能在终端运行 `python --version` 看到版本号。

---

**为什么选它**：你要做 LoRA 微调，后面会接触大型项目结构。PyCharm 社区版免费，自带代码分析、版本控制，适合管理复杂代码。

**安装步骤（跟着做）**：
1. 打开下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费那个）下载。
3. 运行安装程序，**务必勾选**这两项：
   - `Add "Open Folder as Project"` —— 以后右键文件夹就能用 PyCharm 打开
   - `Add to PATH` —— 这会修改系统环境变量，安装后可在命令行使用 `pycharm` 命令；若担心影响系统，可取消勾选，改用

## 块 11

好的，我们来把这块内容变成你的专属学习指南。

你的目标是 **LoRA 微调**，这意味着你很快就要和 PyTorch、HuggingFace 这些“重量级选手”打交道。它们依赖关系复杂，版本还特别敏感。如果直接在系统 Python 里装，很容易把环境搞乱，甚至影响电脑上其他程序。Conda 就是帮你把每个项目的依赖“关进独立小房间”的工具，互不干扰，这是搞深度学习的基本功。

考虑到你喜欢动手实践，我们直接进入正题。

### 5.1 为什么用 Conda

简单说，Conda 能创建**独立的虚拟环境**，每个环境里可以有自己专属的 Python 版本和库，项目之间互不影响。比如，你可以在一个环境里用 Python 3.8 跑旧项目，在另一个环境里用 Python 3.11 做 LoRA 微调，完全没问题。

### 5.2 安装 Conda：Anaconda vs Miniconda

安装 Conda 有两个选择，区别很明显：

| 选项 | 特点 | 适合谁？ |
|------|------|----------|
| **Anaconda** | 体积大（500MB+），自带 numpy、pandas 等一大堆库 | 做数据分析、不想手动装包的人 |
| **Miniconda** | 体积小（约50MB），只带核心工具，需要什么装什么 | **你**。追求干净、可控的开发环境 |

**推荐你选 Miniconda**。因为做 LoRA 微调，我们只需要 PyTorch 等特定库，没必要装一堆用不上的东西。

**安装步骤（跟着做就行）：**

1.  打开官网：`https://docs.conda.io/en/latest/miniconda.html`
2.  下载对应你操作系统的版本（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3.  **运行安装程序，这里有个关键区别：**
    *   **Windows 用户**：一路点“下一步”，**务必保持默认，不要勾选** “Add Miniconda3 to my PATH” 选项。这能避免和系统已有的 Python 冲突。以后使用，请从开始菜单打开 **“Anaconda Prompt (Miniconda3)”** 这个专属终端。
    *   **macOS/Linux 用户**

## 块 12

**目标**：为后续 LoRA 微调项目隔离环境，避免不同项目依赖冲突。

**操作**：在终端执行以下命令，创建名为 `llm-learn` 的环境，并指定 Python 3.10：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令。
- `-n llm-learn`：`n` 是 `name` 的缩写，给环境起名 `llm-learn`。
- `python=3.10`：指定该环境安装 Python 3.10。

**执行后**：终端会列出待安装包并询问 `Proceed ([y]/n)?`，输入 `y` 回车，等待完成。

**本

## 块 13

## 5.4 激活与退出环境：进入你的“独立房间”

你马上要开始 LoRA 微调了，每个项目都应该有自己的独立环境，避免依赖互相打架。Conda 就是帮你管理这些“独立房间”的工具。

**激活环境**（走进这个房间）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志。看到它，就表示你后续所有的 `python` 和 `pip` 命令都只作用在这个环境里，不会影响系统或其他项目——这对微调实验特别重要，因为不同项目可能依赖不同版本的 PyTorch。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。确认版本对了，再继续下一步。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**查看所有已

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，动手前先把“工作台”搭对。跟着做：

1. **先激活环境**：在终端运行 `conda activate llm-learn`。
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入 `Python: Select Interpreter` 并回车。
4. 在列表中选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的所有代码都会使用 `llm-learn` 环境里的 Python 和已安装的包——包括之后微调要用的 PyTorch 和 HuggingFace 工具链。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，步骤同样简单：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter`

## 块 15

**本块学习检查**：完成下方两步操作后，在终端输入 `exit()` 退出 Python 交互模式。

---

你的目标是 LoRA 微调，但一切模型训练都从“环境能跑通”开始。本块用最快路径验证你的工作台是否就绪。

**第一步：命令行验证**

1. 打开终端（命令行窗口），输入 `conda activate llm-learn` 并回车，激活你的学习环境。
2. 输入 `python` 并回车，进入 Python 交互模式。你会看到提示符变为 `>>>`。
3. 在 `>>>` 后输入以下内容，然后回车：

```python
print("Hello, Python! 我的环境准备好了！")
```

4. 如果屏幕正确打印出这句话，说明 Python

## 块 16

好的，我们直接进入正题。既然你的目标是 LoRA 微调，并且喜欢动手操作，那我们就用最直接的方式来验证环境。

### 7.3 在 PyCharm 中验证

这一步是确认你的“工作台”能正常运转。跟着做：

1.  **新建项目**：打开 PyCharm，新建一个项目。项目路径**不要包含中文或空格**（这是本章小结里强调的好习惯）。
2.  **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3.  **编写代码**：在 `hello.py` 中输入 `print("Hello, LoRA!")`。
4.  **运行**：在文件编辑区**右键点击**，选择 **`Run 'hello'`**。
5.  **验证**：看 PyCharm 下方的 **Run 窗口**，如果输出了 `Hello, LoRA!`，说明你的 Python 解释器、项目配置和 IDE 都正常工作了。

---

### 第八章：常见问题排查

运行过程中如果遇到问题，对照下表快速定位：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或
