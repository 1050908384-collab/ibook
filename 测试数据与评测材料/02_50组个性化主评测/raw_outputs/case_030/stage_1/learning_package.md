# 学习包

## 块 1

## 一、先搞清楚三个问题（看完再动手，避免常见踩坑）

### 1.1 什么是 Linux？我为什么要装它？

你平时用的 Windows 或 macOS 叫“操作系统”，Linux 是另一个操作系统。全世界绝大多数服务器（网站后台、云服务、AI 训练集群）跑的都是 Linux。学编程到后面，你大概率要和它打交道。

本教程用的是 **Ubuntu 24.04 LTS**（长期支持版），厂家会持续提供多年安全更新，稳定且适合入门。如果你已安装 Ubuntu 22.04，后续步骤同样适用。

> **给你的定位**：你已有 PyTorch 和 Transformer 基础，接下来要做 vLLM 部署。vLLM 这类推理框架的官方支持环境几乎都是 Linux，Windows 原生跑会遇到大量兼容问题。所以这一步不是“学新系统”，而是**给你的模型部署铺路**——装好 Linux（或 WSL2 / 云服务器），确保能在终端执行 Shell 命令，后续安装 vLLM、拉模型、跑推理才能顺畅。

---

**本块学习检查**

先用最小示例回顾环境准备与常用 Shell 命令（cd/ls/mkdir/cp/mv）：

1. 打开终端，用 `cd` 进入你的工作目录（如 `cd ~`）。
2. 用 `ls` 查看当前目录内容。
3. 用 `mkdir test_dir` 新建目录，再用 `ls` 确认它出现。
4. 用 `cp` 复制一个文件到 `test_dir`，用 `mv` 移动或重命名它。
5. 确认你能解释每条命令的作用，再继续本块。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你在本地部署vLLM，核心是跑Linux环境。三种方案，按你的情况直接看结论：

- **虚拟机**：模拟整台电脑，隔离最彻底，但吃内存（建议≥4GB）。若本机内存不大，跑起来会卡。
- **WSL2**：Windows自带的Linux子系统，像装App一样轻量、启动快，和Windows文件互通。**这是首选**，几分钟装好，足够跑vLLM。
- **云服务器**：租远程电脑，有公网IP、不占本地资源，但要付费且依赖网络。你已有

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

三个方案最终效果一致：你得到一台运行 Ubuntu 24.04 的“电脑”（虚拟或远程）。虚拟机/云服务器通过 SSH 从 Windows 远程操作（类似“远程桌面命令行版”）；WSL2 则直接与 Windows 互通，用 VS Code 本地开发。VS Code 装好插件后，写代码有高亮、自动补全、能调试。

**动手验证**：确认你的电脑内存 ≥ 8GB（右键“此电脑”→“属性”→“已安装的内存”）。若不足，直接跳过方案一。

## 块 4

**VirtualBox 是什么？** 它是一款开源虚拟机软件，能在你的 Windows 里“造”出一台假电脑，这台假电脑可以装任何操作系统——包括你后续要用的 Ubuntu 24.04。

**下载与安装步骤：**

1. 浏览器打开：`https://www.virtualbox.org/wiki/Downloads`
2. 找到 **"VirtualBox 7.x.x platform packages"** 下的 **Windows hosts** 版本，点击下载
3. 双击安装包，一路“下一步”保持默认，点“安装”等进度条跑完，最后“完成”

> **常见坑**：弹出“是否允许更改设备”选“是”；杀毒软件拦截时选“允许”或“信任”。

---

**本块学习检查**：完成下载安装后，用自己的

## 块 5

**本块学习检查**：下载 Ubuntu 24.04 镜像文件，是虚拟机安装系统的“原料”。你最终要用 vLLM 部署模型，而 vLLM 跑在 Linux 上——这个镜像就是给你造出 Linux 环境的起点。下载时记住：要 `desktop-amd64.iso`，别拿 server 版（没图形界面，新手难上手）。遇到不懂的术语（如“镜像”），先记下来，后面装系统时会自然理解。

**操作验证**：打开 https://releases.ubuntu.com/24.04/ ，确认你看到的文件名含 `desktop-amd64`，并检查文件大小约 5GB。下载完别双击，放好即可。

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是部署 vLLM，但前提是有一台能跑 Linux 的机器。本块教你用虚拟机在 Windows 上装 Ubuntu 24.04，为后续环境打基础。

**操作步骤（跟着做）：**

1. 打开 VirtualBox，点 **新建**。
2. 填写：名称 `Ubuntu-24.04`，文件夹**别选 C 盘**（选 D/E），类型 `Linux`，版本 `Ubuntu (64-bit)`，下一步。
3. 内存拖到 **4096 MB**，下一步。
4. 选 **现在创建虚拟硬盘**，大小 **60 GB**，类型 **VDI**，存储 **动态分配**，完成。
5. 选中虚拟机 → **设置** → **

## 块 7

**本块学习检查**：完成安装后，请用自己的话说明：为什么在虚拟机里“清除整个磁盘”不会影响你电脑上的真实数据？这与后续用 vLLM 部署模型有什么关系？遇到不懂的术语（如“虚拟盘”）先记下来再提问。

---

**安装 Ubuntu 系统（针对 vLLM 部署准备）**

你的 RTX 4080 已就绪，现在给虚拟机装系统。操作如下：

1. 在 VirtualBox 选中虚拟机，点 **“启动”**。
2. 出现安装界面后：
   - 语言选 **“中文(简体)”**，点“安装 Ubuntu”。
   - 键盘布局默认，点“继续”。
   - 选 **“正常安装”**，勾选“下载更新”和“安装第三方软件”，点“继续”。
   - 安装类型选 **“清除整个磁盘并安装 Ubuntu”**——注意：这只会清空虚拟机的 60G 虚拟磁盘，你的 C/D 盘完全不受影响。点“现在安装”，弹窗点“继续”。
   - 时区输入“Shanghai”，点“继续”。
   - 填用户名、密码，点“继续”。
3. 自动安装

## 块 8

## 2.5 安装 SSH 远程连接工具

**SSH 是什么？** 你可以把它理解为“远程桌面的命令行版”。装上它之后，你可以在 Windows 的终端里直接操控虚拟机里的 Ubuntu，不用来回切换窗口——这对你后续在服务器上部署 vLLM 是必备技能，因为生产环境通常没有图形界面。

### 操作步骤

**1. 打开 Ubuntu 终端**：在桌面空白处点鼠标右键 → 选 **“在终端中打开”**。如果没有该选项，按 `Ctrl + Alt + T` 也能打开。

**2. 依次输入以下命令**（每输完一行按回车）：

```bash
# 第1步：更新软件列表（相当于“刷新应用商店的货架”）
sudo apt update

# 第2步：安装 SSH 服务（让别的电脑能远程连过来）
#     openssh-server 是 SSH 服务端
#     net-tools 提供 ifconfig 命令，用来查看 IP 地址
sudo apt install -y openssh-server net-tools

# 第3步：设置 SSH 开机自动启动
sudo systemctl enable ssh

# 第4步：立刻启动 SSH 服务
sudo systemctl start ssh
```

> **sudo 是什么？** `sudo` = “以管理员身份执行”。Linux 里普通用户不能随便装软件，加 sudo 相当于“给我最高权限”。输入 sudo 命令后会让你输密码，输的时候**屏幕不会有任何显示**（不会出现 `***`），这是正常的安全设计，盲打完按回车就行。

**3. 查看虚拟机的 IP 地址**：

```bash
ifconfig
```

**4. 在输出内容中找到类似 `inet 192.168.xxx.xxx` 的一行**，那一串数字就是 IP 地址，**记下来！** 例如：`192.168.126.128`

**5. 验证 SSH 是否可用**：回到 Windows，按 `Win + R`，输入 `cmd` 回车，打开命令提示符，输入：

```bash
ssh 用户名@IP地址
```

> 请将命令中的 **用户名** 替换为你的 Ubuntu 登录用户名，将 **IP地址** 替换为第 4 步查到的 IP。

**6. 如果提示输入密码**，输入 Ubuntu 里的密码并回车。看到 `Welcome to Ubuntu`

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

安装增强功能后，虚拟机与Windows间可直接复制粘贴文字和文件。若未自动生效，运行：

```bash
sudo apt install -y virtualbox-guest-utils virtualbox-guest-x11
sudo reboot
```

重启后即可跨系统粘贴。方案一至此完成，你已拥有一台完整Ubuntu虚拟机。

**实践提示**：本步服务于后续vLLM部署——在虚拟机内复制部署命令或日志时，无需手动敲入，减少出错。若你更倾向WSL2（Win10 2004+或Win11），可跳过本步，直接进入方案二，其安装更快（几分钟）。

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

**目标**：确认你的Windows能运行WSL2，这是装Linux环境的前置条件，后续才能部署vLLM。

**操作步骤**：
1. 按 `Ctrl + Shift + Esc` 打开任务管理器
2. 点顶部“性能”标签页
3. 右下角找“虚拟化”，看状态
4. 若“已启用” → 直接进入下一步
5. 若“已禁用” → 重启时按品牌对应的BIOS进入键（常见：联想F2、惠普F10或Esc、戴尔F2或F12、华硕F2或Del），具体可搜索“品牌+进入BIOS”确认。进入后找“Configuration”或“Advanced”，将“Intel Virtualization Technology”

## 块 11

### 安装 WSL2（①）：为 vLLM 部署铺平 Linux 环境

你的目标是在 RTX 4080 上跑通 vLLM，而 vLLM 的推理优化依赖 Linux 环境。WSL2 就是 Windows 上最轻量的“Linux 虚拟机”，几分钟装好，后续装 PyTorch、跑模型都在里面进行。

**操作步骤（全程管理员权限）：**

1. 按 `Win` 键，输入 `powershell`，右键“Windows PowerShell” → **“以管理员身份运行”**，弹出提示点“是”。
2. 在蓝色窗口里，**逐行**输入以下两条命令（复制一行，右键粘贴，回车）：

```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

3. 两条都显示“操作成功完成”后，**重启电脑**。
4. 重启后，再次以管理员身份打开 PowerShell，输入：

```bash
wsl --set-default-version 2
```

5. 若提示更新内核，去 `https://aka.ms/wsl2kernel` 下载安装，装完重跑上一条命令：`wsl --set-default-version 2`，然后运行 `wsl --status` 确认 WSL2 为默认

## 块 12

**本块学习检查**：完成安装后，请用自己的话说明：为什么在 Windows 上装 Ubuntu 24.04 是通往 vLLM 部署的第一步？若遇到“搜不到”或“密码不显示”等术语/现象，记录下来再提问。

---

**安装 Ubuntu 24.04（推荐方法一）**

1. 按 `Win` 键，输入 `Microsoft Store`，打开微软应用商店。
2. 搜索框输入 `Ubuntu 24.04`，找到 **“Ubuntu 24.04.x LTS”**（发布者：Canonical Group Limited），点“安装”。
3. 下载约 500MB，装完点“打开”。

**方法二（命令行，备选）**

```bash
wsl --install -d Ubuntu-24.04
```

4. 首次启动显示 “Installing, this may take a few minutes...”，等 2~5 分钟。
5. 设置用户名（全英文小写，如 `zhangsan`）和密码（输入时屏幕无显示，属正常），再输一遍确认。
6. 出现绿色 `用户名@计算机名:~$` 即成功

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

在刚打开的 Ubuntu 终端（绿色提示符窗口）里，逐行执行下面两条命令，让系统保持最新状态：

```bash
# 刷新软件包列表（相当于更新“应用商店的货架”）
sudo apt update

# 升级所有已安装软件到最新版（-y 表示自动确认所有提示）
sudo apt upgrade -y
```

> **注意**：WSL2 与 Windows 紧密集成，**不需要**像虚拟机那样额外安装和配置 SSH，直接用 VS Code 的 Remote-WSL 插件即可开发。

接着验证环境是否正常：

```bash
# 查看系统版本，应显示 Ubuntu 24.04.x
lsb_release -a

# 查看当前用户名，应是你刚才设置的
whoami
```

再测试与 Windows 的互通：

```bash
# 用 Windows 资源管理器打开当前 Linux 目录
explorer.exe .
```

如果弹出资源管理器窗口，说明 WSL2 与 Windows 已无缝衔接，方案二完成。

## 块 14

**本块学习检查**：完成购买服务器后，用自己的话说明“云服务器”与你的vLLM部署目标的关系——即为什么需要一台远程Linux机器来跑推理服务；遇到不懂的术语（如“镜像”）先记下再提问。

**购买步骤（以阿里云为例）**：

1. 打开阿里云官网并登录（支付宝/淘宝账号即可）。
2. 搜索“轻量应用服务器”进入产品页。
3. 学生可先搜“学生优惠”完成认证，价格更低。
4. 点击“立即购买”，按需配置：
   - **地域**：选离你最近的（如华东1杭州）。
   - **镜像类型**：选“系统镜像”。
   - **操作系统**：选

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**为什么不能一直用 root？** `root` 是 Linux 里的“超级管理员”，权限太高，误操作可能删掉系统文件。日常操作应该用普通账户。

购买后，先通过云厂商控制台提供的 **“远程连接”** 或 **VNC** 功能（网页版终端）登录服务器，执行以下操作：

```bash
# 创建新用户（把 yourname 换成你想要的用户名，全英文小写）
adduser yourname
# 会提示设置密码，输入两次即可。其他信息（全名、

## 块 16

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

### 4.3 配置 SSH 密钥登录（安全第二步）

**为什么优先用密钥？** 密码登录容易被暴力破解。配置 SSH 密钥后，既安全又方便——尤其在你后续要反复连接云服务器做 vLLM 部署时，密钥登录能省去每次输密码的麻烦。

**在 Windows 上操作：**

1. 按 `Win + R`，输入 `cmd`，打开命令提示符
2. 生成密钥对：

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

3. 出现 `Enter file in which to save the key` → 直接按 **回车**（默认路径）
4. 出现 `Enter passphrase` → 直接按 **回车**（不设密码短语）
5. 出现 `Enter same passphrase again` → 再按一次 **回车**
6. 看到 randomart image，说明密钥生成成功

**在服务器上配置公钥：**

7. 通过云厂商的 VNC/远程连接登录服务器，执行：

```bash
su - yourname
mkdir -p ~/.ssh
nano ~/.ssh/authorized_keys
```

8. 回到 Windows，查看公钥

## 块 17

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

在阿里云轻量服务器控制台，进入“防火墙”标签页，添加规则：

- **端口**：22
- **协议**：TCP
- **来源**：建议填你的公网IP（浏览器搜索“IP”即可查到），格式如 `123.45.67.89/32`；若IP不固定，可暂填 `0.0.0.0/0`（允许所有IP，安全性较低，仅限临时使用，长期应改为具体IP）
- 点“确定”

**安全提醒**：不要长期用 root

## 块 18

## 从 Windows 连接你的云服务器

你的目标是 vLLM 部署，这意味着你需要一台能跑模型的远程机器。现在你已经有了阿里云服务器，这一步就是打通你本地电脑和它的连接——之后所有部署操作都在这台服务器上完成。

**操作步骤：**

1. 在阿里云控制台找到 **"公网 IP"**（一串数字，如 `47.96.xxx.xxx`），复制下来
2. 按 `Win + R`，输入 `cmd`，打开命令提示符
3. 输入以下命令连接：

```bash
ssh yourname@47.96.xxx.xxx
```

（把 `yourname` 换成你的用户名，IP 换成你复制的公网 IP）

4.

## 块 19

## 5.1 虚拟机 / 云服务器：配置 SSH 免密登录（强烈推荐）

你已经掌握 PyTorch 和 Transformer，接下来要部署 vLLM，远程连接服务器是第一步。如果你用的是**虚拟机或云服务器**，且前面已完成密钥配置，这里直接验证是否生效：

在 Windows 命令提示符中执行：

```bash
ssh 用户名@你的IP地址
```

**能直接连上、不提示密码，就成功了。** 这意味着后续你可以顺畅地在远程环境里跑 vLLM 推理，不用每次输密码打断思路。

> **常见坑预警**：如果提示 `ssh-copy-id 不是内部或外部命令`，说明你的 Windows 版本较老。解决方法：
> - 方法一：手动将公钥内容追加到服务器的

## 块 20

### WSL2 与 VS Code 远程开发：为 vLLM 部署铺路

你已掌握 PyTorch 和 Transformer，现在要迈向 vLLM 部署。本块解决一个关键问题：**在哪里写代码、怎么连上你的运行环境**。你的 RTX 4080 是 GPU 推理的核心，而 VS Code 远程开发让你在 Windows 上写代码、在 Linux 环境（WSL2）里跑 vLLM，两者无缝衔接。

#### 5.2 WSL2：免 SSH 直连

如果你用 **WSL2**，无需配置 SSH。VS Code 通过 Remote-WSL 插件直接与 WSL2 通信，体验更流畅。这省去了 SSH 密钥、端口等繁琐配置，适合你这种以实践为主的学习节奏。

#### 5.3 安装 VS Code 及远程插件

**VS Code**（Visual Studio Code）是微软的免费代码编辑器，轻量、插件多、跨平台，是当前主流选择。

**第一步：安装 VS Code**

1. 浏览器打开 https://code.visualstudio.com/
2. 点 **"Download for Windows"**，下载后双击安装包
3. 安装时注意：
   - 同意协议，路径默认即可
   - **关键一步**：勾选以下选项：
     - ☑ **"将 Code 添加到 PATH 中"**（可在终端直接输 `code` 启动）
     - ☑ **"将 Code 注册为受支持的编辑器"**
     - ☑ **"添加到文件资源管理器上下文菜单"**
   - 一路下一步 → 安装 → 完成后启动 VS Code

**第二步：安装必备插件**

插件是 VS Code 的"外挂功能包"，装完才能高亮、补全、调试。打开方式：左侧竖排图标栏第五个（四个小方块），或按 `Ctrl + Shift + X`。

在

## 块 21

### 连接你的 Linux 环境（①）

你的目标是 vLLM 部署，而 vLLM 跑在 Linux 上。这一步就是让 VS Code 连上你的 Linux，之后才能写代码、跑推理。

**如果你用虚拟机或云服务器：**

1. 打开 VS Code，看左下角绿色图标（像 `><`）。
2. 点击它 → 选 **"Connect to Host…"**。
3. 输入 `用户名@IP地址`（如 `zhangsan@192.168.126.128`）→ 回车。
4. 提示选平台时选 **"Linux"**。
5. 有密钥直接连，没有就输密码。
6. 左下角变成 `SSH: IP地址` 即成功。
7. 点 **"File"** → **"Open Folder…"**，选服务器目录（如 `/home/你的用户名`），点 OK。
8. 现在就能像编辑本地文件一样操作远程文件了。

> **连不上？** 先检查网络（`ping IP地址`）、确认 SSH 服务已启动（`sudo systemctl status ssh`）、密钥权限正确（`chmod 600 ~/.ssh/id_rsa`）。

**如果你用 WSL2：**

1. 点左下角绿色图标 → 选 **"Connect to WSL"**。
2. 自动连上 Ubuntu，左下角变 `WSL: Ubuntu-24.04`。
3. 点 "Open Folder"，选 `/home/

## 块 22

## 6.1 文件与目录操作：最基础也最常用

你的目标是 vLLM 部署，而 vLLM 需要加载模型权重、读取配置文件、管理日志输出——这些全都离不开文件与目录操作。下面这些命令是你进入模型目录、检查权重文件、整理部署脚本的日常工具。

**导航三件套**

- `pwd`：显示当前所在目录的绝对路径。部署前先确认自己在哪，避免把模型文件放错位置。
- `ls`：列出当前目录内容。`ls -l`（Ubuntu 下可简写为 `ll`）显示权限和大小，用来确认模型权重文件是否完整；`ls -a` 显示隐藏文件，比如 `.cache` 或配置文件。
- `cd`：切换目录。`cd ~` 回到主目录，`cd ..` 返回上级，`cd /path/to/dir` 进入指定路径。部署时你会频繁在项目目录和模型目录之间切换。

**创建与查看**

- `mkdir`：创建目录。`mkdir -p test/aaa/bbb` 可递归创建多级目录，适合一次性搭好项目结构。
- `touch`：创建空文件。比如先 `touch config.yaml` 再编辑。
- `cat`：查看小文件内容，`cat -n` 显示行号，适合快速检查配置文件。
- `less`：分页查看大文件（如推理日志），空格翻页、`q` 退出。

**复制、移动与删除**

- `cp`：复制。复制目录必须加 `-r`（递归），例如 `cp -r test/ /home/user/Desktop`。
- `mv`：移动或重命名。`mv old_name new_name` 就是重命名，部署时常用它来整理模型版本。
- `rm`：删除。**谨慎使用**

## 块 23

## 6.2 权限管理：Linux 系统的安全核心

你在部署 vLLM 时，模型文件、日志、脚本都需要正确的权限才能运行。Linux 用「所有者、所属组、其他用户」三类身份控制访问，权限分为读（r=4）、写（w=2）、执行（x=1）。

**常用命令：**

| 命令 | 功能 | 语法 | 示例 |
|---|---|---|---|
| `chmod` | 修改权限 | `chmod [权限值] 文件名` 或 `chmod [用户+/-权限] 文件名` | 数字法：`chmod 744 file1.txt`（7=4+2+1 给所有者，4=4 给其他只读）；符号法：`chmod g+w file1.txt` 给组加写；`chmod a+x test.sh` 给所有人加执行 |
| `chown` | 修改所有者（需 root） | `chown [用户] 文件名` | `sudo chown root file1.txt` |

**sudo 回顾**：前面安装环境

## 块 24

你的目标是vLLM部署，而vLLM跑起来后，最需要关心的就是**显存、内存和进程状态**。本块命令正是为此服务的。

**核心命令速览（结合你的场景）**

- **`free -h`**：查看内存。部署模型前先看内存够不够，避免OOM。
- **`df -h`**：查看磁盘。确认模型下载路径有足够空间。
- **`ps -ef | grep python`**：查看当前运行的Python进程，确认vLLM服务是否在跑。
- **`top`**：实时监控。按`M`按内存排序，观察推理进程的资源占用。

**本块学习检查**

完成本块后，用自己的话说明：这些系统查询命令，如何帮助你判断vLLM部署是否成功、以及排查资源不足的问题？遇到无法解释的术语（如“内核版本”）时，记录下来再提问。

## 块 25

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**6.4 文件编辑：快速修改文件内容**

你的vLLM部署中，经常要改配置文件（如模型路径、显存参数）。这里两个命令够用：

**1. `vim` 编辑器**（改文件内容）

- 打开文件：`vim 文件名`
- 按 `i` 进入编辑模式（左下角出现 `--INSERT--`）
- 改完按 `Esc`，输入 `:wq` 回车保存退出；不想保存就输入 `:q!`
- 找内容：按 `

## 块 26

你的目标是vLLM部署，但先别急着碰模型。部署前，你得能确认这台机器“活着”且“找得到”。本块就练这两个动作。

**1. 测连通性：`ping`**
想知道能否访问外网，用 `ping www.baidu.com`。它会持续发数据包，看到 `time=xx ms` 就说明通了。按 `Ctrl+C` 停止。

**2. 看网卡：`ifconfig`**
部署服务时，你得知道本机IP。运行 `ifconfig`，输出里找 `inet` 字段，那就是IP地址。若提示找不到命令，执行 `sudo apt install net-tools` 安装。

**本块学习检查**
完成练习后，用自己的话说说：这两个命令和“vLLM部署”有什么关系

## 块 27

### 6.6 命令实战场景演练

你已掌握 PyTorch 和 Transformer，但 vLLM 部署的日常操作离不开终端。下面两个场景，直接动手练，把命令变成肌肉记忆。

#### 场景 1：创建并整理文件

**需求**：在主目录下创建 `work` 文件夹，里面再创建 `doc` 和 `code` 两个子文件夹；创建 `note.txt` 文件，写入 "Linux 命令学习笔记"，并移动到 `doc` 文件夹；最后查看 `doc` 文件夹的内容。

```bash
# 1. 进入主目录
cd ~

# 2. 创建 work 及子文件夹
mkdir -p work/doc work/code

# 3. 创建并写入文件
echo "Linux命令学习笔记" > note.txt

# 4. 移动文件
mv note.txt work/doc/

# 5. 查看结果
ls work/doc/
# 预期输出：note.txt
```

**练习要点**：`mkdir -p` 一次建多层目录，`echo >` 写文件，`mv` 移动——这三步是后续管理模型权重和日志的基础操作。

#### 场景 2：查询

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 28

## 6.7 新手避坑指南

你的目标是 vLLM 部署，下面这些坑在部署时几乎必踩，先打好预防针。

**1. 区分大小写**：Linux 命令严格区分大小写。`ls` 正确，`LS` 直接报错。部署时输入命令，注意别开大写锁。

**2. 路径问题**：切换目录、复制文件时，路径写错会提示 "No such file or directory"。先 `ls` 确认路径存在再操作。

**3. rm 命令慎用**：`rm -rf` 是"删库跑路"级别的危险命令。删除前务必 `ls` 确认目录/文件正确，避免误删 `/`（根目录）。

**4. 权限不足**：操作系统文件或其他用户文件时，会提示 "Permission denied"。在命令前加 `sudo` 获取临时管理员权限（需输入当前用户密码）。

**5. 命令记不住？**：输入命令前按 `Tab` 键自动补全（文件/目录名）；忘记语法时，用 `命令 --help`（如 `ls --help`）或 `man 命令`（如 `man cp`）查看

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 29

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的RTX 4080已就绪，但vLLM部署前需先确认Python环境。Ubuntu 24.04自带Python 3.12，直接验证：

```bash
python3 --version   # 预期输出：Python 3.12.x
pip3 --version      # 若提示未找到，执行下方安装
```

若pip缺失，运行：

```bash
sudo apt update
sudo apt install -y python3-pip
```

**为什么现在做这个？** pip是Python的“应用商店”，后续安装PyTorch、

## 块 30

**本块学习检查**：完成安装后，请用自己的话说明：Miniconda 与系统自带 Python 的关系是什么？它对你后续用 vLLM 部署模型（涉及多个依赖包管理）有什么帮助？遇到无法解释的术语（如“虚拟环境”）时记录下来，稍后提问。

---

**动手安装 Miniconda（约10分钟）**

你的 RTX 4080 已就绪，但 vLLM 部署需要干净的 Python 环境。Miniconda 是轻量级环境管理器，后续装 PyTorch、vLLM 时能避免依赖冲突。

**步骤1：下载安装脚本**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

若提示 `wget` 未找到，先执行 `sudo apt install wget`。

**步骤2：运行安装**

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

**步骤3：交互选项**

- 按 `Enter` 阅读协议 → 输入 `yes`
- 安装路径直接回车

## 块 31

## 7.3 运行你的第一个 Python 程序

你的目标是用 vLLM 部署大模型，而 vLLM 本身是 Python 工具链的一部分。所以，在进入部署之前，先确认你的环境能跑 Python——这是所有后续工作的地基。

**操作步骤：**

1. 在终端里执行以下命令，回到主目录并创建第一个 Python 文件：

```bash
cd ~
nano hello.py
```

2. 在打开的 nano 编辑器里，输入这一行代码：

```python
print("Hello, Python on Ubuntu 24.04!")
```

3. 按 `Ctrl + O` 保存文件，回车确认文件名，再按 `Ctrl + X` 退出 nano。

4. 运行你的程序：

```bash
python3 hello.py
```

5. 如果屏幕输出了 `Hello, Python on Ubuntu 24.04!`，说明你的 Linux 环境已经就绪。你刚刚用到了 `cd`（切换目录）、`nano`（编辑文件）、`python3`（运行 Python）三个核心命令——这些正是后续安装 PyTorch、配置 vLLM 时每天都会用的基础操作。

**本块学习检查**

完成上面的操作后，请用自己的话回答：为什么在部署 vLLM 之前，需要先确认 Python 环境能正常运行？这个 `hello.py` 验证过程和你

## 块 32

## 2.1 pwd：打印当前工作目录

在开始部署 vLLM 之前，先解决一个基础问题：**你当前在哪个目录？**

`pwd` = **P**rint **W**orking **D**irectory，作用就是告诉你"我现在在哪个文件夹里"。

```bash
pwd
```

示例输出：
```bash
/home/zhangsan
```

这表示你当前位于 `/home/zhangsan` 目录下。在操作文件之前，养成先敲 `pwd` 的习惯，能避免很多"文件找不到了"的困惑。

**为什么这对 vLLM 部署很重要？** 部署时你需要下载模型权重、编写推理脚本、指定输出路径——如果连当前目录都不清楚，后续 `cd`、`cp`、`mv` 都会迷失方向。每次打开终端，第一件事就是 `pwd`，确认自己站在哪里。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

## 块 33

**本块学习检查**：学完后，请用自己的话说说“绝对路径”和“相对路径”的区别，并说明这对你后续用 vLLM 部署模型时定位文件（如模型权重、配置文件）有什么帮助。遇到不懂的术语先记下来再提问。

---

**2.2 绝对路径 vs 相对路径**

新手最容易混淆，务必理解：

| 路径类型 | 定义 | 示例 |
|---|---|---|
| **绝对路径** | 从根目录 `/` 开始，不依赖当前位置 | `/home/zhangsan/Documents` |
| **相对路径** | 相对于你当前所在的位置 | `./file.txt`、`../backup` |

相对路径中两个重要符号：

- `./`：表示**当前目录**（可省略）
- `../`：表示**上一级目录**
- `~`：表示**当前用户的家目录**（如 `/home/zhangsan`）

> **举例**：假设你在 `/home/zhangsan/Documents` 目录下：
> - `

## 块 34

## ls：列出目录内容——基础用法

你即将用 vLLM 部署模型，第一步是学会“看”文件系统。`ls` 就是你的眼睛。

**核心操作**（在终端输入后回车）：

```bash
ls          # 看当前目录有什么
ls /home/zhangsan   # 看指定路径下有什么
ls -a       # 连隐藏文件一起看（以 . 开头的是隐藏文件）
ls -l       # 详细列表：权限、所有者、大小、修改时间
ls -ahl     # 组合：全部文件 + 详细列表 + 人类可读大小
```

**关键认知**：以 `.` 开头的文件默认隐藏，比如 `.bashrc`、`.ssh`。这些文件很重要——你配置 vLLM 环境变量时会频繁遇到它们，所以 `ls -a` 要养成习惯。

**解读 `ls -l` 输出**：

```
drwxr-xr-x 2 zhangsan zhangsan 4096 Aug  3 10:00 Documents
-rw-r--r--

## 块 35

## ls 其他常用选项（①）

你已经掌握了 `ls` 的基础用法，下面这些选项能帮你更高效地查看文件信息——尤其是部署 vLLM 时，你需要快速确认模型文件的大小、修改时间，以及目录结构。

**常用选项速览：**

| 命令 | 作用 |
|------|------|
| `ls -lh` | 详细列表 + 人类可读大小（KB/MB/GB）。**注意**：`-h` 必须搭配 `-l` 才有效，单独用 `ls -h` 无效果 |
| `ls --full-time` | 显示完整时间格式 |
| `ls -t` | 按最后修改时间排序（最新在前） |
| `ls -S` | 按文件大小排序（从大到小） |
|

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 36

### 4.1 基础用法（①+②）

`cd` 是你在终端里“移动位置”的命令。你正打算用 vLLM 部署模型，而部署时经常需要进入模型目录、切到权重文件所在路径，所以掌握 `cd` 是第一步。

```bash
# 进入指定目录（绝对路径，从根目录 / 开始写全）
cd /home/zhangsan/Documents

# 进入当前目录下的子文件夹（相对路径，./ 可省略）
cd ./Downloads
cd Downloads

# 返回上一级目录
cd ..

# 返回上两级目录
cd ../..

# 回到家目录
cd ~
cd        # 直接 cd 不加参数也是回家目录

# 返回刚才所在的目录（快速来回切换）
cd -
```

### 4.2 路径切换示例（新增②）

假设当前在 `/usr/local` 下，想切换到 `/home/zhangsan`：

```bash
# 方法1：绝对路径（去哪都管用）
cd /home/zhangsan

# 方法2：相对路径（先回根目录，再去 home）
cd ../../home/zhangsan
```

> **建议**：新手优先使用绝对路径，不容易迷路；熟悉后多用相对路径，效率更高。

---

## 五、mkdir：创建目录（①+②）

`mkdir` = **Ma**k**e Dir**ectory，用来创建文件夹。部署 vLLM 时，你通常需要新建项目目录来存放模型和脚本。

```bash
# 在当前目录下创建一个文件夹
mkdir myproject

# 在指定路径下创建文件夹
mkdir /home/zhangsan/myproject

# 递归创建多级目录（父目录不存在时自动创建）
mkdir -p myproject/src/models

# 一次性创建多个同级文件夹
mkdir dir1 dir2 dir3

# 使用花括号批量创建同级文件夹
mkdir {dir1,dir2,dir3}

# 批量创建 100 个文件夹（dir1 ~ dir100）
mkdir dir{1..100}
```

> **注意**：`mkdir` 默认**不能**直接创建多级嵌套目录，必须加 `-p` 参数。这是新手最常踩的坑之一。

---

## 六、touch：创建空文件（①）

`touch` 原本用来“更新文件的时间戳”，但最常用的功能是**快速创建一个空文件**

## 块 37

### 7.1 复制文件（①+②）

`cp` 用于复制文件。基本用法：`cp 源文件 目标位置`。

- **复制到文件夹（保留原名）**：`cp hello.py ./backup/`
- **复制并改名**：`cp hello.py ./backup/hello_backup.py`
- **一次复制多个文件**：`cp file1.txt file2.txt ./backup/`
- **保持文件属性**（如修改时间）：加 `-p` 参数：`cp -p hello.py ./backup/`
- **覆盖前提示**：加 `-i` 参数，若目标已存在会询问确认：`cp -i hello.py ./backup/`

### 7.2 复制目录（①+②）

复制文件夹**必须加 `-r`**（递归），否则报错。

```bash
cp -r myproject ./backup/
```

注意行为差异：
- 若 `backup` 不存在，会新建 `backup` 并把 `myproject` 内容复制进去；
- 若 `backup` 已存在，会创建 `backup/myproject/`，把整个文件夹放进去。

### 7.3 复制软链接（

## 块 38

## 8.1 移动文件/文件夹

在 vLLM 部署时，你常需要把模型权重或推理脚本整理到固定目录。`mv` 就是干这个的：

```bash
# 把文件移动到另一个文件夹
mv hello.py ./backup/

# 把文件夹移动到另一个位置
mv myproject ./archive/

# 用通配符移动多个文件（把所有以 data_ 开头的文件移到 data 文件夹）
mv data_* ./data/
```

## 8.2 重命名

重命名的本质是“在同一目录下移动”：

```bash
# 把 a.txt 重命名为 b.txt
mv a.txt b.txt

# 移动的同时改名（比如把推理脚本备份为带版本号的名称）
mv hello.py ./backup/hello_v2.py
```

## 8.3 安全选项

```bash
# 覆盖前询问（推荐日常使用）
mv -i old.txt new.txt

# 强制覆盖，不提示（慎用）
mv -f old.txt new.txt
```

---

## 九、rm：删除文件与目录

`rm` = **R**e**m**ove，用来删除文件或文件夹。**此命令不可逆，请谨慎使用！**

```bash
# 删除单个文件
rm hello.py

# 删除多个文件
rm file1.txt file2.txt

# 删除文件名以 - 开头的文件（必须用 -- 告诉 rm 后面是文件名而非选项）
rm -- -file.txt

# 删除文件夹（必须加 -r，递归删除其中的所有内容）
rm -r myproject/

# 只删除空文件夹
rm -d empty_dir/

# 强制

## 块 39

## 场景 1：搭建项目目录结构（为 vLLM 部署做准备）

你正在学习 vLLM 部署，动手前先搭好项目骨架，让模型文件、数据和脚本各归其位。跟着做：

```bash
cd ~                    # 回到家目录
mkdir llm-learning      # 创建项目根目录
cd llm-learning         # 进入项目
mkdir data models scripts  # 一次建三个子目录
touch scripts/train.py  # 在 scripts 下建空文件
ls -R                   # 递归查看，验证结构
```

预期看到：

```
llm-learning/
├── data
├── models
├── scripts
│   └── train.py
```

**说明**：`data` 放数据集，`models` 放模型文件，`scripts

## 块 40

## 场景 2：复制与整理实验文件

在准备 vLLM 部署前，你需要把实验数据整理到统一目录。假设 `~/Downloads/` 下已有 `train.csv`、`test.csv`、`dev.csv`，目标是复制到 `~/llm-learning/data/`。

**操作步骤：**

1. **确认文件存在**：`ls ~/Downloads/*.csv` — 用通配符列出所有 CSV 文件，确保源文件都在。
2. **复制文件**：`cp ~/Downloads/train.csv ~/Downloads/test.csv ~/Downloads/dev.csv ~/llm-learning/data/` — `cp` 后跟多个源文件，最后一个参数是目标目录。
3. **验证结果**：`ls ~/llm-learning/data/` — 应看到 `train.csv`、`test.csv`、`

## 块 41

## 场景 3：清理临时文件（动手实操）

你在 `~/llm-learning/` 下误建了一个名为 `temp` 的空文件夹和一个 `old.txt` 文件，现在要清理掉它们。这个场景会用到 `rm` 删除文件、`rm -d` 删除空文件夹，以及删除前用 `ls` 确认目标的好习惯。

```bash
# 1. 先确认要删的内容（好习惯！）
ls ~/llm-learning/temp ~/llm-learning/old.txt

# 2. 删除文件
rm ~/llm-learning/old.txt

# 3. 删除空文件夹
rm -d ~/llm-learning/temp

# 4. 验证
ls ~/llm-learning/
```

**关键点**：`rm` 直接删文件，`rm -d` 只删空文件夹。删除前务必用 `ls` 确认路径和内容，因为 `rm` 没有回收站。

---

## 新手避坑指南（重点看这几条）

**1. 区分文件和文件夹**：`ls -l` 第一列是 `d` 表示文件夹，是 `-` 表示文件。对文件夹操作别忘了 `-r`（如 `cp -r`、`rm -r`）。

**2. 路径写错**：切换目录或复制文件时，路径写错会提示 "No such file or directory"，可先用 `pwd` 和 `ls` 确认当前位置和目录内容。

**3. 路径中包含空格必须用引号包裹**：命令行会把空格当成参数分隔符。务必用双引号包裹：

```bash
# 错误：mkdir my folder 会创建两个文件夹 my 和 folder
# 正确：
mkdir "my folder"
cd "my folder"
touch "my file.txt"
cp "my file.txt" ./backup/
rm "my file.txt"
```

**4. 文件名以 `-` 开头时使用 `--`**：`-` 开头会被误认为选项。用 `--` 告诉命令"后面都是文件名"：

```bash
# 错误：rm -file.txt 会把 -file.txt 当成选项，报错
# 正确：
rm -- -file.txt
touch -- -data.csv
mv -- -old.txt new.txt
```

**5. rm 命令慎用**：`rm -rf` 没有回收站，删除即永久消失。删除前务必用 `ls` 确认目标。

**6. 复制目录忘加 `-r`**：`cp mydir ./backup/` 会报错。记住：**文件夹复制必须加 `-r`**。

**7. 命令记不住？**：输入命令前按 `Tab`

## 块 42

## 环境变量是什么？——先搞懂这张“备忘录”

你已经掌握了 `cd`、`ls`、`mkdir` 这些命令，现在我们要进入 vLLM 部署前的关键一环：**环境变量**。别被名字吓到，它其实很简单。

### 1.1 通俗理解

环境变量就是操作系统或程序运行时使用的**动态键值对**，用来传递配置信息。你可以把它想成操作系统的“备忘录”——系统启动时加载它，所有程序都能读到里面的内容，并据此调整自己的行为。

举个例子：你在终端输入 `python3`，系统怎么知道去哪找它？靠的就是 `PATH` 这个变量。它告诉系统：“去这些目录里找可执行程序。”

**常见环境变量速查表**：

| 变量名 | 作用 |
|---|---|
| `PATH` | 告诉系统去哪里找可执行程序（如 `python3`） |
| `HOME` | 当前用户的家目录路径（如 `/home/zhangsan`） |
| `USER` | 当前登录的用户名 |
| `SHELL` | 当前使用的 Shell 程序路径（如 `/bin/bash`） |
| `LANG` | 系统语言和字符编码（如 `en_US.UTF-8`） |
| `PWD` | 当前工作目录路径（随 `cd` 自动更新） |
| `EDITOR` | 默认文本编辑器（影响 `git commit` 等操作） |

### 1.2 为什么需要环境变量？

1. **控制程序行为**：比如用 `JAVA_HOME` 指定 JDK 路径，让依赖 Java 的程序知道去哪找它。
2. **简化配置**：不用每次运行程序都手动输入长长的路径。
3. **跨进程通信**：

## 块 43

**临时变量：当前终端会话有效**

临时变量只活在当前终端里——关掉窗口或断开 SSH 连接，它就没了。适合快速测试。

```bash
export MY_VAR="hello_linux"   # 设置
echo $MY_VAR                  # 查看，输出 hello_linux
unset MY_VAR                  # 删除
echo $MY_VAR                  # 输出为空
```

`export` 让子进程也能读到这个变量；不加的话只有当前 Shell 自己知道。设置环境变量时习惯加上它。

**本块学习检查**：用自己的话说说临时变量的生命周期，以及它和 vLLM 部署时设置模型路径、显存参数这类操作有什么关系？遇到说不清的术语先记下来再问。

## 块 44

**永久变量：写入配置文件，持久生效**

临时变量只在当前终端有效，关掉就没了。要让变量每次打开终端都在，就得写进**配置文件**。

| 级别 | 配置文件 | 影响范围 | 需要sudo？ |
|---|---|---|---|
| 用户级（推荐） | `~/.bashrc` | 仅当前用户 | 否 |
| 用户级 | `~/.bash_profile` / `~/.profile` | 仅当前用户（登录时生效） | 否 |
| 系统级 | `/etc/profile` 等 | 所有用户 | 是 |

**选哪个？** 日常开发优先用 `~/.bashrc`，只影响自己，改错不伤系统。系统级配置要 `sudo` 且影响所有人，不建议动。

**本块学习检查

## 块 45

你的目标是vLLM部署，但先别急着碰模型。任何部署工具都依赖环境变量来定位路径、设置参数，`~/.bashrc`正是你日常开发中最常用的配置位置——每次打开新的交互式终端时，它都会自动加载。

**操作步骤**（跟着做一遍）：

```bash
# 1. 用 nano 打开配置文件
nano ~/.bashrc

# 2. 滚动到文件末尾，添加一行：
export MY_VAR="hello_permanent"

# 3. 按 Ctrl + O 保存，Ctrl + X 退出

# 4. 让修改立即生效（不重启终端）
source ~/.bashrc

# 5. 验证
echo $MY_VAR
# 输出：hello_permanent
```

**关键理解**：
- `source`相当于“重新执行一遍这个配置文件”，让修改立刻生效，不用关终端。
- 注意限制：`~/.bashrc`只在**交互式终端**加载。SSH会话、IDE内置终端或自动化脚本可能不加载它。若配置未生效，手动`source ~/.bashrc`，或检查所用终端的加载行为。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLL

## 块 46

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

### 3.2 用户级配置：~/.bash_profile 或 ~/.profile（①）

这两个文件在**登录 Shell** 时生效（如SSH远程登录、系统开机登录）。若同时存在，bash优先读取 `~/.bash_profile`。

日常开发中 `~/.bashrc` 更常用，因为每开一个终端都会加载。只有需要“登录时才生效”的特殊场景，才考虑这两个文件。

---

### 四、环境变量的优先级（①）

同一变量多处定义时，优先级从高到低：

1. **临时变量**（当前终端 `export` 的）
2. **用户级配置**（`~/.bashrc` 等）与**系统级

## 块 47

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

### 5.1 PATH：命令的"搜索路径"

`PATH` 是最重要的环境变量，它定义了系统**查找可执行文件**的目录列表。对 vLLM 部署来说，`which python3` 能确认你用的 Python 来自哪个环境，避免装错依赖。

#### 查看 PATH

```bash
echo $PATH
```

输出类似（多个路径用冒号 `:` 分隔）：

```bash
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

输入 `python3` 时，系统**从左到右**在这些目录里找同名文件，找到就执行，否则提示 `command not found`。

#### 查看命令具体位置

```bash
which python3
# 输出：/usr/bin/python3

which conda
# 输出：/home/zhangsan/miniconda3/bin/conda
```

`which` 告诉你命令实际来自 PATH 的哪个目录。

#### 修改 PATH

**追加路径**（末尾，优先级低）：

```bash
export PATH=$PATH:/new/path
```

**前置路径**（开头，优先级高，先搜这里）：

```bash
export PATH=/new/path:$PATH
```

> **安全提示**：不要把 `.`（当前目录）加入 PATH。若某目录有恶意 `ls` 脚本且该目录在 PATH 中，执行 `ls` 可能运行恶意脚本。
>
> **避免重复添加**：多次 `source ~/.bashrc` 或重复写入同一行会导致 PATH 重复。定期检查 `~/.bashrc

## 块 48

你的目标是**用vLLM部署模型**，而vLLM在运行时需要加载CUDA的动态库（`.so`文件）。如果系统找不到这些库，vLLM会直接启动失败。`LD_LIBRARY_PATH`就是告诉系统“去哪些额外目录找这些库”的环境变量。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

先看当前值：
```bash
echo $LD_LIBRARY_PATH
```
大概率是空的，这正常。

临时添加CUDA库路径（仅当前终端有效）：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
这条命令

## 块 49

## 5.3 LIBRARY_PATH：编译时库文件路径

`LIBRARY_PATH` 告诉编译器（如 `gcc`）在**编译阶段**去哪里找库文件，影响静态库（`.a`）和动态库（`.so`）的链接。

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/your/libraries
```

**三个变量的核心区别**：

| 变量名 | 作用阶段 | 查找目标 | 典型用途 |
|---|---|---|---|
| `PATH` | 执行命令时 | 可执行文件 | 让命令随处可运行 |
| `LD_LIBRARY_PATH` | 程序运行时 | 动态库（`.so`） | 运行依赖自定义动态库的程序 |
| `LIBRARY_PATH` | 编译链接时 | 静态库/动态库 | 编译时链接第三方库 |

> **对你的建议**：做 vLLM 部署时，`PATH` 和 `LD_LIBRARY_PATH` 是重点——安装 Miniconda、CUDA、PyTorch 都要正确配置它们。`LIBRARY_PATH` 主要在手动编译依赖时用到。

---

## 六、实战：配置 CUDA 环境变量

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 50

## 6.1 检查 CUDA 是否已在环境变量中

在配置之前，先确认你的系统是否已经能找到 CUDA。打开终端，依次执行：

```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

如果两条命令**都没有输出**，说明 CUDA 尚未加入环境变量，继续下一步。如果已有输出（如 `/usr/local/cuda/bin`），说明已配置好，可跳过后续步骤。

## 6.2 编辑 ~/.bashrc 添加 CUDA 路径

用 nano 打开配置文件：

```bash
nano ~/.bashrc
```

在文件**末尾**添加以下两行：

```bash
# CUDA 环境变量配置
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

> **注意**：`/usr/local/cuda` 是 CUDA 默认安装路径。如果你的 CUDA 装在其他位置，请替换为实际路径。

保存并退出（nano 中：`Ctrl+O` 回车保存，`Ctrl+X` 退出）。

## 6.3 使配置生效并验证

```bash
# 重新加载配置文件
source ~/.bashrc
```

然后验证 CU

## 块 51

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

## 七、调试与验证

### 7.1 查看所有环境变量
```bash
printenv      # 简洁输出所有变量
env           # 更详细的输出
```

### 7.2 查看特定变量
```bash
echo $PATH
printenv PATH
declare -p PATH    # 显示变量定义细节（含是否 export）
```

### 7.3 测试变量是否生效
```bash
# 假设在 ~/.bashrc 里设置了 export MY_VAR="hello"
source ~/.bashrc
echo $MY_VAR
# 应输出：hello
```

### 7.4 追踪变量来源
```bash
# 在所有配置文件中搜索 MY_VAR 的定义

## 块 52

## 8.1 变量未生效（①）——排查与修复

你在配置环境变量后，发现 `vllm` 命令或相关路径没有生效？这通常不是配置本身写错了，而是**修改后没有重新加载**，或者**配置文件里有小错误**。

**先按顺序做这三步排查：**

```bash
# 1. 确认你的变量确实写进了配置文件（检查语法和拼写）
cat ~/.bashrc | grep 'YOUR_VARIABLE_NAME'

# 2. 重新加载配置，让当前终端生效
source ~/.bashrc

# 3. 检查是否有重复定义（比如 PATH 被覆盖了多次）
grep -n "PATH=" ~/.bashrc
```

**常见原因对照：**
- 改了文件但没 `source` 或重启终端 → 执行第 2 步即可
- 配置里少写了 `export` 关键字 → 第 1 步能帮你发现
- 变量在别处被覆盖 → 第 3 步帮你找出重复定义
- 用的终端或 IDE 没加载 `~/.bashrc` → 确认你操作的是同一个终端

**一个实用提醒**

## 块 53

## 9.1 脚本中临时修改变量

在 Shell 脚本里，你可以用 `export` 设置一个变量，但它只在脚本运行期间有效：

```bash
#!/bin/bash
export TEMP_VAR=123
# 脚本结束后，TEMP_VAR 自动消失
```

这对 vLLM 部署很有用：比如临时指定 `CUDA_VISIBLE_DEVICES` 或调整推理参数，又不想污染全局环境。

## 9.2 用 env 指定单次运行变量

如果只想让某条命令用一次变量，用 `env` 前缀：

```bash
env MY_VAR=hello python3 my_script.py
```

`MY_VAR` 只在这条 `python3` 命令执行时存在，跑完就没了。适合在启动 vLLM 服务时临时传入配置。

## 9.3 用 .env 文件分组管理变量

项目需要多个变量时，建一个 `.env` 文件集中管理：

```bash
nano ~/.env
```

内容示例：

```bash
export PROJECT_HOME="$HOME/llm-learning"
export EDITOR="nano"
export TZ="Asia/Shanghai"
```

加载方式：

```bash
set -o allexport
source ~/.env
set +o allexport
```

> **⚠️ 安全警告**：`source .env` 会**执行**文件里的每一行代码。如果 `.env` 来自不可信来源，直接加载可能执行恶意代码。加载前务必检查内容。
>
> **⚠️

## 块 54

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

### 场景 1：创建并验证自定义变量

先设置一个临时变量，并验证它：

```bash
export MY_NAME="zhangsan"
echo $MY_NAME
```

这个变量只在当前终端有效。要让它永久生效，把它写入 `~/.bashrc`，再重新加载：

```bash
echo 'export MY_NAME="zhangsan"' >> ~/.bashrc
source ~/.bashrc
echo $MY_NAME
```

### 场景 2：为自定义脚本目录配置 PATH

假设你把工具脚本放在 `~/scripts`，希望在任何目录直接运行。先创建目录和测试脚本：

```bash
mkdir -p ~/scripts
echo '#!/bin/bash' > ~/scripts/hello_tool.sh
echo 'echo "Hello from my tool!"' >> ~/scripts/hello_tool.sh
chmod +x ~/scripts/hello_tool.sh
```

然后把 `~/scripts` 加入 PATH，写入 `~/.bashrc` 并生效：

```bash
echo 'export PATH=$PATH:$HOME/scripts' >> ~/.bashrc
source ~/.bashrc

## 块 55

**本块学习检查**  
完成本块后，用自己的话说明：CUDA环境变量配置与你的vLLM部署目标有何关系？若遇到无法解释的术语（如`LD_LIBRARY_PATH`），先记录下来再提问。  

---

### 场景 3：模拟 CUDA 环境变量配置

你还没装CUDA？没关系，先练配置方法，为vLLM部署铺路。

**操作步骤**（在终端执行）：  

1. 打开配置文件：  
   ```bash
   nano ~/.bashrc
   ```  

2. 在文件末尾添加两行（假设CUDA装在`/usr/local/cuda`）：  
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH

## 块 56

## 系统级配置文件：公共环境的关键

系统级配置影响所有用户，修改需 `sudo` 权限。日常开发不建议用，但配置公共服务器时必须了解。

### A.1 /etc/environment

**特点**：不需要 `export` 关键字，直接写 `KEY=value`。仅支持简单键值对，不支持 Shell 语法（如 `$PATH` 引用）。

```bash
sudo nano /etc/environment
# 添加：JAVA_HOME=/usr/lib/jvm/java-11-openjdk
# 保存后需重启或重新登录生效
```

### A.2 /etc/profile

**特点**：全局 Bash 配置，需要 `export` 关键字。

```bash
sudo nano /etc/profile
# 添加：export PATH=$PATH:/usr/local/custom/bin
# 保存后 source 当前会话生效，或重启
source /etc/profile
```

**本块学习检查**：完成本块后，用自己的话说明系统级配置与 vLLM 部署的关系（如为何公共服务器需配置

## 块 57

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

先确认你的Shell类型：执行 `echo $SHELL`，输出 `/bin/bash` 即Bash，否则按下方对照表找对应配置文件。

| Shell | 配置文件 | 作用范围 |
|---|---|---|
| Bash | `~/.bashrc` | 非登录Shell（日常终端） |
| Bash | `~/.bash_profile` | 登录Shell（SSH登录） |
| Zsh | `~/.zshrc` | 交互式非登录Shell |
| Zsh | `~/.zprofile` | 登录Shell |
|

## 块 58

## C. 清理 PATH 中的重复项（④）

多次修改 PATH 后，里面可能堆积重复路径。虽然不影响功能，但会让排查问题变得混乱。用下面这条命令一键去重：

```bash
export PATH=$(echo -n $PATH | tr ':' '\n' | awk '!seen[$0]++' | paste -sd: -)
```

**拆解一下**：`echo -n $PATH` 输出当前 PATH（不附加换行），`tr` 将冒号替换为换行符，`awk` 按行去重，`paste` 再把结果拼回冒号分隔的格式。这条命令只影响当前终端会话，想永久生效就把它写进 `~/.bashrc`。

## D. 根据环境加载不同配置（④）

开发和生产环境往往需要不同的环境变量。可以在 `~/.bashrc` 中加一段条件判断：

```bash
if [ "$ENVIRONMENT" = "production" ]; then
    source ~/.env.prod
else
    source ~/.env.dev
fi
```

这样，只要预先设置好 `ENVIRONMENT` 变量，`source ~/.bashrc` 时就会自动加载对应的配置文件。

## E. 脚本调试模式（④）

排查脚本问题时，可以开启追踪模式，让 Shell 打印每一行执行前的变量值：

```bash
bash -x your_script.sh
```

或者在当前 Shell 中临时开启追踪，执行完再关闭：

```bash
set -o xtrace
source ~/.bashrc
set +o xtrace
```

---

## 十一、总结与后续学习路线

恭喜你完成了环境变量配置章节的学习！你现在应该掌握了：

- ✅ 环境变量的概念：键值对，影响系统和程序的运行行为
- ✅ 临时变量（`export` / `unset`）与永久变量（配置文件）的区别
- ✅ 用户级配置（`~/.bashrc`）的选择与限制（

## 测评题目

### Q-KB0100-004
执行以下命令检查系统架构和内核版本：
```bash
arch && uname -r
```

### Q-KB0100-002
执行以下命令验证当前用户和主机名：
```bash
whoami && hostname
```

### Q-KB0101-006
请完成以下操作并截图/粘贴输出：
1. 在 `/tmp` 下创建目录 `practice`
2. 进入该目录，创建文件 `note.txt` 并写入 "Hello Linux"
3. 查看文件内容
4. 复制为 `note_backup.txt`
5. 列出目录下所有文件

### Q-KB0101-004
`cp -r source_dir/ dest_dir/` 中 `-r` 参数的含义是？

### Q-KB0102-004
在 `~/.bashrc` 中写入 `export LD_LIBRARY_PATH=/usr/local/lib` 后，新开一个终端窗口，该变量是否自动生效？为什么？

### Q-KB0102-001
临时设置环境变量 `MY_VAR=hello`，命令为________；若要使其在当前shell及子进程中生效，需使用________命令。

### Q-KB0103-001
创建一个名为 `myenv`、Python版本为3.10的conda虚拟环境，正确命令是？

### Q-KB0102-002
请写出将 `/usr/local/cuda/bin` 永久添加到 `PATH` 环境变量的完整步骤（以bash为例）。
