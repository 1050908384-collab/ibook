# 学习包

## 块 1

**本块学习检查**：读完本节后，请合上资料，用自己的话回答：“Python 解释器是干什么的？没有它，你的 `.py` 文件是什么？”答不上来就回头再看一遍。

---

### 1.1 Python 解释器：程序的“翻译官”

你的目标是用 QLoRA 微调大模型，但别急着碰模型——先确保电脑能听懂你的指令。

**为什么先讲这个？** 你写的 Python 代码是人类语言，电脑只认 0 和 1。**解释器**就是中间的“翻译官”，负责把你的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆普通文本，电脑根本不会理你。

**动手验证（2分钟）**：打开你的文件管理器，随便新建一个 `test.py` 文件，用记事本输入 `print("hello")`，保存后双击运行——大概率没反应。这就是因为没有解释器来“翻译”。等后面装好 Python，再运行同一个文件，你就能看到输出了。

**记住这个比喻**：解释器 = 翻译官。你说话（写代码），它翻译给电脑听（执行）。后续装环境、跑微调脚本，每一步都离不开它。现在装好它，后面 QLoRA 的代码才能跑起来。

## 块 2

你的目标是 QLoRA 微调，这意味着你很快要面对真实的大模型项目。本块帮你把“工作台”搭对，避免在环境上浪费宝贵时间。

**IDE 选择**：直接装 **VS Code**。它轻量、免费，配合 Python 插件后，语法高亮、自动补全、调试功能齐全，足够支撑你后续写微调脚本。PyCharm 虽专业，但对追求速度的你来说偏重，暂不推荐。

**虚拟环境（关键）**：QLoRA 依赖的库（如 transformers、peft）版本敏感。用 **conda** 为每个项目建独立环境，就像给项目一个“独立房间”，避免 A 项目要 numpy 1.0、B 项目

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 QLoRA 微调，这需要 Python 环境。跟着做，很快搞定。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

> **说明**：Python 3.10+ 表示 3.10 及以上版本，当前最新稳定版为 3.13.x，下载最新版即可。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 添加到系统"环境变量"。不勾选的话，命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径不含中文和特殊符号，避免后续路径报错。进阶用户想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符

## 块 4

### 你的目标：QLoRA 微调，先搭好 macOS 上的 Python 工作台

macOS 12.3 起不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用。为了 QLoRA 微调，**务必自行安装最新版 Python**，别依赖系统版本。

**方法一：官方安装包（推荐，最快上手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件），该安装包同时支持 Apple Silicon 和 Intel Mac。
2. 双击运行，一路 "Continue" → "Install"。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**关键坑**：命令行默认**没有** `python` 命令，统一用 `python3` / `pip3`。别敲 `python`，否则会报"命令不存在"或误用系统旧版。

**方法二：Homebrew（已装则用）**

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

**本块学习检查**：在终端运行 `python3 --version`

## 块 5

**动手装好 Python，为 QLoRA 铺路**  
你的目标是 QLoRA 微调，但先别急——所有后续实验都跑在 Python 环境里。Ubuntu/Debian 通常自带 Python，但版本偏老，我们直接装最新的 3.10+。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆解一下**：  
- `sudo apt update`：刷新软件源，确保能装到最新包。  
- `python3-pip`：装包工具，后面装 PyTorch、transformers 全靠它。  
- `python3-venv

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 QLoRA 微调，这意味着你最终要在 GPU 上跑训练。环境选对了，后面能省下大量折腾时间。

**日常写 Python、练命令行（Windows 用户）**

启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**

直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**

使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护（2021 年 12 月 31 日 EOL），CentOS Stream

## 块 7

**3.1 什么时候才需要镜像源？**

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载——这是最权威、最完整、最应保留的默认源。

**关键判断**：只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是“网络不好时的临时加速手段”，不是永久替换项**——官方源始终是首选。

**首选做法：临时指定镜像源（推荐）**

如果某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

如果你只是偶尔下载慢，**别用这招**。只有当长期处于网络受限环境、几乎每次都要靠镜像时，才把镜像写入配置作为默认源。配置后记住：官方 PyPI 始终最可靠，网络恢复后建议改回或删除配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：默认从清华镜像下载包。

> **⚠️ 别乱加 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常用**不需要**。只有企业内网自建源、或确认来源可信且遇到证书报错时，才临时加。

**macOS / Linux 用户：**

1. 终端执行：

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

### 3.4 验证配置是否生效

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `pip.conf` 文件，或执行：

```bash

## 块 9

**动手装好 VS Code，为 QLoRA 微调铺路**

你的目标是跑通 QLoRA 微调，这需要写 Python 代码、调试脚本。VS Code 轻量、免费、启动快，是干这活的首选工具，装好它，后面写微调代码时能省不少心。

**安装步骤（跟着点就行）**

1. 打开浏览器，访问 `https://code.visualstudio.com`，下载你系统对应的安装包。
2. 双击运行安装程序，一路点“下一步”直到完成，不用改任何设置。

**装完必做：装两个插件**

打开 VS Code，看左侧竖着的活动栏，点最下面那个四个方块的图标（这就是“扩展”商店）。在搜索框里依次搜这两个，点安装：

- **Python**（微软官方出品）——装上它，VS Code 才能认得出 `.py` 文件，给你语法高亮、调试和智能提示。
- **Pylance**——Python 的语言服务器，让代码提示更聪明，写微调脚本时少打错字。

**关键一步：告诉 VS Code 用哪个

## 块 10

**PyCharm Community：你的 QLoRA 项目工作台**

既然你的目标是 QLoRA 微调，那代码量不小，一个趁手的 IDE 能让你少走弯路。PyCharm 社区版免费且功能足够，适合你这种需要“开箱即用”的场景。注意它比付费的 Professional 版缺少对深度学习框架（如 PyTorch）的专用支持。

**安装要点（2 分钟搞定）**

1. 去官网下载 **Community 版**（认准免费那个）。
2. 安装时务必勾选这两项，后面能省事：
   - `Add launchers dir to the PATH`：命令行里敲 `charm` 就能启动。
   - `Create associations`：关联 Python 文件，双击即可用 PyCharm 打开。

**首次启动，按这个顺序配**

1. 选 `New Project`。
2. **Location** 路径别带中文，否则后面跑模型容易出

## 块 11

## 5.1 为什么用 Conda

你的目标是 QLoRA 微调，这意味着后面要装 PyTorch、CUDA 等一堆复杂依赖。如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身——这对深度学习几乎是必备的。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 1GB 或更大），预装了大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 70-100MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境干净，需要什么装什么，适合你这种目标明确的开发场景。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**（默认不勾选，保持默认即可）。直接塞进 PATH 容易和系统已有的 Python 冲突。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 后，按提示输入 `yes` 同意协议、选择路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样终端会自动接管 conda，而不是靠硬加 PATH。

4. **验证安装**：打开一个**新的**终端（Windows 用 **"Anaconda Prompt (Miniconda3)"**；macOS/Linux 重开终端），输入：

```bash
conda --version

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么现在学**：你马上要搞 QLoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会建环境，后面直接复用这套流程，省去反复踩坑。

**动手做**：打开终端，执行下面这行命令：

```bash
conda create -n llm-learn python=3.10
```

**拆开看**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境里装 Python 3.10

**执行

## 块 13

## 5.4 激活与退出环境

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

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

## 第六章：在 VS Code / PyCharm 中使用 Conda 环境

---

**本块

## 块 14

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件（请根据实际安装路径调整）：
   - 使用环境变量：`${CONDA_PREFIX}/envs/llm-learn/bin/python`
   - 或使用占位符：`<你的conda安装路径>/envs/llm-learn/bin/python`

---

## 第七章：验证——运行你的第一行代码

## 块 15

**动手验证：让环境“开口说话”**

既然你的目标是QLoRA微调，那第一步就是确保工作台能跑起来。别急着敲复杂代码，先用两分钟确认环境通不通。

**方式一：命令行（最直接）**

1. 打开终端，激活你的环境：`conda activate llm-learn`
2. 输入 `python` 回车，看到 `>>>` 提示符就对了。
3. 敲入这行，回车：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
4. 屏幕上打印出这句话，说明解释器工作正常。输入 `exit()` 退出。

**方式二：VS Code（写脚本用）**

1. 新建文件夹 `my-first-project`，用

## 块 16

### 7.3 在 PyCharm 中验证

你的目标是 QLoRA 微调，环境验证是第一步，别跳过——后面跑模型全靠这里打底。跟着做，两步搞定：

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目里新建一个文件，命名为 `hello.py`。
2. **运行脚本**：在 `hello.py` 文件里，右键点击文件内容区域，选择 **`Run 'hello'`**。
3. **看输出**：PyCharm 底部会弹出 **Run** 窗口，里面应该显示你代码的运行结果。

**本块学习检查**：在 `hello.py` 中输入 `print("Hello, QLoRA!")`，然后运行。确认 Run 窗口输出 `Hello, QLoRA!`。如果看到了，说明你的 PyCharm 环境完全可用，可以继续往下走。

---

### 第八章：常见问题排查（速查表）

环境搭建总会踩坑，遇到问题别慌，对照下表解决：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，用 `python -m pip install
