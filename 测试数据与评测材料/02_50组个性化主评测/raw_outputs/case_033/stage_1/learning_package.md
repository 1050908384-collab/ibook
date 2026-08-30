# 学习包

## 块 1

### 1.1 什么是 Linux？我为什么要装它？

你平时用的 Windows 或 macOS 叫“操作系统”，Linux 是另一个操作系统。全世界绝大多数服务器（网站后台、云服务、AI 训练集群）跑的都是 Linux。你做 vLLM API 服务，最终部署目标几乎必然是 Linux 环境，所以现在装好它，等于提前消除“本地能跑、上服务器就挂”的隐患。

本教程用 **Ubuntu 24.04 LTS**（长期支持版），厂家持续提供多年安全更新，稳定且适合入门。你已有 PyTorch 和 Transformer 基础，但环境操作是硬门槛——先确保能打开终端、执行命令，后面装驱动、跑推理才不会卡壳。

> **给你的实操建议**：你手头是云上的 Tesla T4 16GB，说明你大概率走“云服务器”路线。别纠结虚拟机或 WSL2，直接租一台 Ubuntu 云主机，SSH 登录后就是纯 Linux 终端，和后续部署 vLLM 的环境完全一致。

---

**本块学习检查**

先用最小示例回顾环境准备与常用 Shell 命令，确认能解释关键写法后再继续：

1. 在终端执行 `ls`，观察输出，说出它列出的是什么。
2. 执行 `mkdir test_dir`，再用 `ls` 确认新目录出现。
3. 执行 `cd test_dir` 进入该目录，用 `pwd` 确认当前位置。
4. 在 test_dir 内创建文件 `touch a.txt`，然后 `cp a.txt b.txt`，最后 `mv b.txt ../` 把它移出。每一步后用 `ls` 观察变化。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

你已具备PyTorch和Transformer基础，目标是在云T4上跑vLLM API服务。动手装环境前，先选对“房子”——三种方案各有定位：

- **虚拟机**：在你电脑里模拟一台新电脑，隔离最彻底，但吃内存（建议至少4GB），配置低会卡。适合本机内存≥8GB、想体验完整Linux的人。
- **WSL2**：Windows自带的Linux子系统，像装个App，轻量、启动快、与Windows互通。仅限Win10/11，部分高级功能受限。适合不想折腾、快速上手的人。
- **云服务器**：租阿里云/腾讯云的远程电脑，有公网IP，随时连，不占本地资源，需付费（学生常有优惠，以

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

三个方案最终体验一致：你得到一台运行 Ubuntu 24.04 的“电脑”（虚拟或远程）。虚拟机/云服务器通过 SSH 从 Windows 远程操作；WSL2 则与 Windows 直接互通，用 VS Code 本地开发。SSH 就是“远程桌面命令行版”——在自己电脑敲命令控制另一台电脑。VS Code 装好插件后，写代码有高亮、补全和调试。

对你做 vLLM API 服务而言，这步是地基：先打通“本地写代码 → 远程跑模型”的通道，后续部署推理服务才能顺畅。T4 云服务器走 SSH 连接，正好对应方案三的路径。

## 块 4

**VirtualBox：在你的 Windows 里造一台“假电脑”**

你已有 PyTorch 和 Transformer 基础，目标是用 vLLM 做 API 服务。本步是环境准备：用 VirtualBox 装 Ubuntu，为后续部署 vLLM 提供 Linux 环境。

**操作步骤：**

1. 浏览器打开：`https://www.virtualbox.org/wiki/Downloads`
2. 找到 **"VirtualBox 7.x.x platform packages"** 下的 **Windows hosts**，点击下载
3. 双击安装包，一路点“下一步”，保持默认选项
4. 点“安装”，等进度条跑完，点“完成”

**常见坑预警**：若弹出“是否允许此应用更改你的设备”，点“是”；杀毒软件拦截时选

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 5

**本块学习检查**：完成下载后，用自己的话说说这个镜像文件和你最终要跑的 vLLM API 服务有什么关系；遇到看不懂的术语（如“amd64”）先记下来，稍后提问。

---

**下载 Ubuntu 24.04 镜像（.iso）**

这个文件相当于“虚拟机的安装光盘”，VirtualBox 靠它装系统。

1. 浏览器打开 `https://releases.ubuntu.com/24.04/`
2. 找文件名类似 `ubuntu-24.04.x-desktop-amd64.iso` 的文件（x 是版本号），点击下载
3. 文件约 5GB，下载需几分钟到十几分钟
4. 下载完**别双击打开**，放下载文件夹即可

> **坑**

## 块 6

### 在 VirtualBox 里创建虚拟机（①）

你当前的目标是部署 vLLM API 服务，而这一步是搭建本地实验环境的基础。虽然你最终会在云服务器（如 Tesla T4）上跑推理，但先在虚拟机里熟悉 Ubuntu 24.04 的安装和终端操作，能帮你建立 Linux 环境的肌肉记忆，避免后续在云上踩坑。

**操作步骤（跟着做）：**

1. **打开 VirtualBox**：双击桌面上的 "Oracle VM VirtualBox" 图标。
2. **新建虚拟机**：点击窗口上方的 "新建" 按钮。
3. **填写配置**：
   - 名称填 `Ubuntu-24.04`
   - 文件夹选安装位置（**别放 C 盘**，选 D 或 E 盘）
   - 类型选 `Linux`，版本选 `Ubuntu (64-bit)`，点"下一步"
4. **分配内存**：拖动滑块到 **4096 MB**（4GB），点"下一步"。
5. **创建虚拟硬盘**：
   -

## 块 7

### 安装 Ubuntu 系统（①）

你的目标是跑 vLLM 服务，虚拟机只是临时环境，装系统这步别纠结细节，按流程走完即可。

1. 在 VirtualBox 窗口里，选中虚拟机，点 **“启动”** 按钮。
2. 等几十秒，出现 Ubuntu 安装界面：
   - 语言选 **“中文(简体)”** 或 **“English”** → 点“安装 Ubuntu”。
   - 键盘布局：默认，点“继续”。
   - 更新和其他软件：选 **“正常安装”**，勾上“安装 Ubuntu 时下载更新”和“安装第三方软件” → 点“继续”。
   - 安装类型：选 **“清除整个磁盘并安装 Ubuntu”**（这里的“磁盘”是虚拟机的 60G，不影响你云服务器或本地磁盘）→ 点“现在安装” → 弹窗点“继续”。
   - 时区：输入“Shanghai” → 点“继续”。
   - 用户信息：填名字、计算机名、用户名、密码 → 点“继续”。
3. 自动安装约 10~20 分钟。
4. 提示“安装完成，您需要

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 8

## 2.5 安装 SSH 远程连接工具

**SSH 是什么？** 你可以把它理解为“远程桌面的命令行版”。装好 SSH 后，你就能在 Windows 终端里直接操控虚拟机里的 Ubuntu，不用来回切换窗口——这对你后续在云服务器上部署 vLLM 服务非常关键，因为云服务器通常只有命令行界面，没有桌面。

### 在 Ubuntu 里安装 SSH 服务

1. 在 Ubuntu 桌面空白处**点鼠标右键** → 选 **“在终端中打开”**。如果右键没有，按 `Ctrl + Alt + T` 也能打开终端。

2. 依次输入以下命令（输完一行按回车）：

```bash
# 第1步：更新软件列表（类似“刷新应用商店的货架”）
sudo apt update

# 第2步：安装 SSH 服务（让别的电脑能远程连过来）
sudo apt install -y openssh-server net-tools

# 第3步：设置 SSH 开机自动启动
sudo systemctl enable ssh

# 第4步：立刻启动 SSH 服务
sudo systemctl start ssh
```

> **sudo 是什么？** `sudo` = “以管理员身份执行”。Linux 里普通用户不能随便装软件，加 sudo 相当于“给我最高权限”。输入 sudo 命令后会让你输密码，输的时候**屏幕不会有任何显示**（不会出现 `***`），这是正常的安全设计，盲打完按回车就行。

### 查看 IP 地址并验证连接

3. 查看虚拟机的 IP 地址：

```bash
ifconfig
```

4. 输出内容很多，找到类似 `inet 192.168.xxx.xxx` 的那一行，那一串数字就是 IP 地址。**记下来！** 例如：`192.168.126.128`

5. 回到 Windows，按 `Win + R`，输入 `cmd` 回车，打开命令提示符，输入：

```bash
ssh 你的用户名@192.168.xxx.xxx
```

6. 提示输入密码时，输入 Ubuntu 里的密码并回车。看到 `Welcome to Ubuntu` 之类的欢迎信息，说明成功！

> **常见坑预警**：
> - 提示 `connection refused`：回到 Ubuntu 终端，输入 `sudo systemctl status ssh` 查看 SSH 是否在运行。绿色 `active (running)`

## 块 9

你的目标是用 vLLM 部署模型，但当前阶段先要有个能跑命令的 Linux 环境。虚拟机方案里，装完 Ubuntu 后最关键的一步就是**安装增强功能**，它解决一个很实际的痛点：在 Windows 和虚拟机之间直接复制粘贴文字和文件。

新版 VirtualBox 一般会自动装好。如果没生效，手动执行：

```bash
sudo apt install -y virtualbox-guest-utils virtualbox-guest-x11
sudo reboot
```

重启后，Windows 里 `Ctrl+C` 复制，Ubuntu 里 `Ctrl+V` 粘贴，就能用了。

**方案一完成！** 你已有一台完整的 Ubuntu 虚拟机，后续 Shell 命令和 vLLM 相关操作都可以在这里进行。

**本块学习检查**：用自己的话说说，为什么“能复制粘贴”对后续在终端里执行命令、部署 vLLM 很重要？遇到不懂的术语（比如 `virtualbox-guest-utils`）

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**3.1 确认虚拟化已开启**

虚拟化是CPU的一个功能，让你能在Windows里“套娃”运行另一个系统（WSL2需要它）。大部分电脑默认开启，但部分品牌机（联想、戴尔等）可能关了。

**检查步骤：**
1. 按 `Ctrl + Shift + Esc` 打开**任务管理器**
2. 点顶部**“性能”**标签页
3. 右下角找**“虚拟化”**，看是“已启用”还是“已禁用”
4. “已启用” → 直接跳到下一步；“已禁用” → 进BIOS开启

**BIOS开启方法（禁用时才需要

## 块 11

### 安装 WSL2：为 vLLM 部署铺平 Linux 环境

你已掌握 PyTorch 和 Transformer，下一步要在 T4 云 GPU 上跑 vLLM 服务。vLLM 依赖 Linux 环境，而 WSL2 能让你在 Windows 上直接获得完整 Linux 内核——这正是本块要解决的"环境地基"问题。

**操作步骤（跟着做，约5分钟）**

1. 按 `Win` 键，输入 `powershell`，在搜索结果中**右键**"Windows PowerShell" → 选**"以管理员身份运行"**。
2. 弹出权限确认时点**"是"**。
3. 在蓝色 PowerShell 窗口中，**逐行**复制粘贴以下命令（每行粘贴后按回车）：

```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

4. 两条命令均显示"操作成功完成"后，**重启电脑**。
5. 重启后，再次以管理员身份打开 PowerShell，输入：

```bash
wsl --set-default-version 2
```

6. 若提示需要更新内核，请运行 `wsl --update` 或下载并安装最新的WSL2内核更新包，然后重新执行 `wsl --set

## 块 12

## 安装 Ubuntu 24.04（①）

你的目标是部署 vLLM API 服务，而 WSL2 里的 Ubuntu 就是你的操作台。装好它，你才能在这个 Linux 环境里跑 Shell 命令、装依赖、启动服务。

**推荐方式（图形界面，最简单）**：

1. 按 `Win` 键，输入 `Microsoft Store`，打开微软应用商店
2. 搜索框输入 `Ubuntu 24.04`
3. 找到 **"Ubuntu 24.04.x LTS"**（发布者是 Canonical Group Limited），点"安装"
4. 下载约 500MB，装完后点"打开"

**备选方式（命令行，一条搞定）**：

```bash
wsl --install -d Ubuntu-24.04
```

装完后首次启动会显示 "Installing, this may take a few minutes..."，等 2~5 分钟。接着设置用户名和密码：

- 用户名：全英文小写，如 `zhangsan` → 回车
- 密码：输入时屏幕**没有任何显示**，这是正常的 → 回车
-

## 块 13

## 3.4 更新系统 + 3.5 验证一切正常

你的目标是部署 vLLM 做 API 服务，而 vLLM 对系统环境有要求。所以先把 Ubuntu 系统更新到最新，确保后续安装依赖时不踩坑。

在 Ubuntu 终端（绿色提示符那个窗口）里，逐行执行：

```bash
# 刷新软件包列表（相当于更新"应用商店的货架"）
sudo apt update

# 把所有已安装软件升级到最新版（-y 表示自动确认）
sudo apt upgrade -y
```

> **WSL2 需要 SSH 吗？** 不需要。WSL2 与 Windows 紧密集成，直接用 VS Code 的 Remote-WSL 插件开发即可，不用像虚拟机那样额外配置 SSH。

更新完成后，验证环境是否正常：

1. 执行以下命令，确认系统版本和用户名：

```bash
# 查看系统版本，应显示 Ubuntu 24.04.x
lsb_release -a

# 查看当前用户名
whoami
```

2. 测试与 Windows 互通：

```bash
# 用 Windows 资源管理器打开当前 Linux 目录
explorer.exe .
```

如果弹出资源管理器窗口，说明 WSL2 和 Windows 已无缝衔接。

> **方案二完成！**

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

**本块学习检查**：完成购买后，用自己的话说明“为什么选Ubuntu 24.04系统镜像”与“vLLM部署”的关系；遇到不懂的术语（如“地域”“镜像”）先记下再提问。

---

以阿里云为例（腾讯云、华为云流程类似）：

1. 打开 https://www.aliyun.com/ ，用支付宝或淘宝账号注册/登录。
2. 顶部搜索“轻量应用服务器”并进入；学生先搜“学生优惠”认证，价格更低。
3. 点“立即购买”，按以下配置选：
   - **地域**：选离你最近的（如华东1（杭州））
   - **镜像类型**：系统镜像
   - **操作系统**

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

**为什么不能一直用 root？** `root` 是 Linux 里的"超级管理员"，权限太高，误操作可能删掉系统文件。日常操作应该用普通账户。

购买后，先通过云厂商控制台提供的 **"远程连接"** 或 **VNC** 功能（网页版终端）登录服务器，执行以下操作：

```bash
# 创建新用户（把 yourname 换成你想要的用户名，全英文小写）
adduser yourname
# 会提示设置密码，输入两次即可。其他信息（全名、电话

## 块 16

### 4.3 配置 SSH 密钥登录（安全第二步）

**为什么先做这步？** 你的目标是在云服务器上跑 vLLM API 服务，这意味着服务器会长期暴露在公网。密码登录极易被暴力破解，而密钥登录相当于一把只有你有的“数字钥匙”，既安全又省去每次输密码的麻烦。

**操作步骤（Windows 本地 + 云服务器）**

1. **生成密钥对**：在 Windows 上按 `Win + R`，输入 `cmd` 打开命令提示符，执行：
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```
2. **一路回车**：出现 `Enter file...` 直接回车（用默认路径）；出现 `Enter passphrase` 再回车（不设密码短语）；`Enter same passphrase again` 再回车一次。看到 randomart image 即生成成功。
3. **查看公钥**：在 Windows 上执行 `type %USERPROFILE%\.ssh\id_rsa.pub`，复制全部输出。
4. **服务器上放公钥**：通过云厂商的 VNC/远程连接登录服务器，执行：
   ```bash
   su - yourname
   mkdir -p ~/.ssh
   nano ~/.ssh/authorized_keys
   ```
   将复制的公钥粘贴进 nano，按 `Ctrl+O` 保存，`Ctrl+X` 退出。
5. **设权限**：

## 块 17

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

进入云厂商的轻量应用服务器控制台，点“防火墙”标签页，再点“添加规则”：
- 端口：22
- 协议：TCP
- 来源：建议填你自己的公网IP（浏览器搜“IP”查看），格式如`123.45.67.89/32`；IP不固定可暂填`0.0.0.0/0`（允许所有IP，安全性较低）
- 点“确定”

**安全提醒**：不要长期用root+密码登录，务必用普通用户+SSH密钥登录。

**练习**

## 块 18

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

在阿里云控制台找到**公网IP**（如`47.96.xxx.xxx`）并复制。按`Win + R`输入`cmd`打开命令提示符，执行：

```bash
ssh yourname@47.96.xxx.xxx
```

首次连接提示`Are you sure you want to continue connecting (yes/no)?`时输入`yes`回车。若密钥配置正确，将直接登录成功，无需密码。

**方案三完成**——你已拥有一台24小时在线的Ubuntu服务器，这是后续部署vLLM

## 块 19

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你在云服务器上跑vLLM，每次SSH登录输密码很烦。本块就验证4.3节的密钥是否生效。

**操作**：Windows命令提示符执行：
```bash
ssh 用户名@你的IP地址
```
能直接连上不提示密码，即成功。

**坑**：提示`ssh-copy-id 不是内部或外部命令`，说明Windows较老。解决：装Git for Windows（自带ssh-copy-id），或手动复制（见4.3节）。

## 块 20

### 5.2 WSL2：无需 SSH，直接集成（①）

你用的是云服务器（Tesla T4），所以本小节对你**不是必选**——但了解一下没坏处：如果你以后在本地用 WSL2，VS Code 通过 Remote-WSL 插件直接与 WSL2 通信，**不需要配置 SSH**，体验更流畅。

### 5.3 安装 VS Code 及远程开发插件（①）

**VS Code**（Visual Studio Code）是微软出的免费代码编辑器，轻量、插件多、跨平台。你的目标是 vLLM 部署，后面要写 Python 脚本、看推理日志，VS Code 是合适的操作台。

#### 第一步：安装 VS Code（①）

1. 浏览器打开：https://code.visualstudio.com/
2. 点 **"Download for Windows"** 按钮
3. 下载完成后双击安装包
4. 安装过程：
   - 同意协议 → 下一步
   - 安装路径默认就行 → 下一步
   - **重点！** 这页勾选以下选项：
     - ☑ **"将 Code 添加到"PATH"中"**（重要！这样可以在命令行直接输入 `code` 打开 VS Code）
     - ☑ **"将 Code 注册为受支持的编辑器"**
     - ☑ **"添加到文件资源管理器上下文菜单"**
   - 一路"下一步" → "安装" → 完成后勾上"启动 VS Code" → "完成"

#### 第二步：安装必备插件（①）

**插件（Extension）** 是 VS Code 的"外挂功能包"。VS Code 本身只是编辑器壳子，装了插件才能高亮代码、自动补全、调试。

1. 打开 VS Code
2. 点左侧竖排图标栏里 **第五个图标**（四个小方块组成的正方形），或按 `Ctrl + Shift + X`
3. 在 **扩展商店** 搜索框里依次搜索并安装：

| 搜索关键词 | 插件全名 | 发布者 | 作用 |
|---|---|---|---|
| Python | Python | Microsoft | Python 代码高

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 21

### 连接你的 Linux 环境（①）

你的目标是部署 vLLM，这意味着代码最终要在 Linux 上跑。下面两步帮你把 VS Code 连上远程环境，像编辑本地文件一样操作服务器。

**如果你用云服务器（你的 T4 就在上面）：**

1. 打开 VS Code，看左下角绿色图标（形如 `><`）。
2. 点击它 → 顶部菜单选 **"Connect to Host…"**。
3. 输入 `用户名@你的IP地址`（如 `zhangsan@203.0.113.10`）→ 回车。
4. 若弹出选平台，选 **"Linux"**。
5. 首次连接会提示确认服务器指纹，选 **"Continue"** 即可；配过密钥直接连上；没配则输密码。
6. 连上后左下角变 `SSH: IP地址`，即成功。
7. 点左上角 **"File"** → **"Open Folder…"**，选服务器目录（如 `/home/你的用户名`），点 OK。
8. 现在可以像编辑本地文件一样编辑远程文件了。

**如果你用 WSL2：**

1. 先确保已安装 **Remote-WSL** 扩展（在扩展市场搜索 "Remote - WSL" 并安装）。
2. 点左下角绿色图标 → 选 **"Connect to WSL"**。
3. 自动连上 Ubuntu，左下角变 `WSL: Ubuntu-24.04`。
4. 点 "Open Folder"，选 `/home/你的用户名`。

> **更省事**：在 Ubuntu

## 块 22

## 6.1 文件与目录操作：最基础也最常用

你马上要部署 vLLM 服务，所有模型文件、日志、配置都存放在 Linux 文件系统里。这一节掌握好，后面操作才顺手。

### 先记住三件事

1. **在哪**：`pwd` 显示当前绝对路径，比如 `/home/user/Documents`。
2. **看什么**：`ls` 列出当前目录内容；`ls -l`（Ubuntu 下可简写 `ll`）显示权限和大小；`ls -a` 连隐藏文件一起显示。
3. **去哪**：`cd` 切换目录——`cd ..` 上级、`cd /` 根目录、`cd ~` 回主目录、`cd /具体/路径` 直达。

### 创建与删除

- **建目录**：`mkdir test` 建单个；`mkdir -p test/aaa/bbb` 一次建多级。
- **建文件**：`touch file1.txt file2.log` 一次建多个空文件。
- **删（谨慎！）**：`rm file1.txt` 删文件；`rm -f` 强制不提示；`rm -rf test/` 递归强删目录。**不可恢复，别碰系统文件。**

### 复制与移动

- **复制**：`cp file1.txt /目标路径` 复制文件；复制目录必须加 `-r`：`cp -r test/ /目标路径`。
- **移动/改名**：`mv file1.txt test/` 移动；`mv file1.txt newfile.txt` 改名；`mv test/ /新路径/新名字` 移动并改名。

### 查看文件

- **小文件**：`cat -n file1.txt` 带行号查看。
- **大文件**（如 vLLM 日志）：`less app.log`，空格翻页、回车换行、`q` 退出。

### 操作建议

在云服务器上练习时，先 `mkdir vllm-test` 建个实验目录，在里面反复建、复制、移动、删除，熟悉了再碰真实文件。

### 本块学习检查

完成本块后，用自己的话说明：这些文件操作命令与你“用 vLLM 部署 API 服务”的目标有什么关系？比如部署时你会用到哪些操作场景？遇到无法解释的术语（如权限、递归）先记下来，之后提问。

## 块 23

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是部署vLLM API服务，这需要运行脚本、修改配置、管理模型文件——每一步都涉及文件权限。权限错了，服务可能启动失败或数据泄露。

**核心概念**：每个文件/目录都有三类角色：所有者、所属组、其他用户。权限分三种：读（r=4）、写（w=2）、执行（x=1）。

**常用命令**：

- `chmod` 改权限。数字法推荐：`chmod 744 file1.txt` 表示所有者可读+写+执行（7=4+2+1），其他只读（4）。符号法更直观：`chmod g+w file1.txt` 给所属组加写权限；`chmod a+x test.sh` 给所有用户加执行权限（部署脚本常用）。
- `chown` 改所有者，需root权限：`sudo chown root file1.txt`。

**实践要点**：`sudo` 是临时管理员权限前缀，输密码时屏幕无显示是正常安全设计

## 块 24

**本块学习检查**：学完系统查询命令后，请用自己的话说说这些命令（`uname`、`top`、`ps`、`df`、`free`）如何帮助你为vLLM部署做准备。遇到不懂的术语（如“内核”“进程”），先记下来再提问。

---

**为什么先学这个？** 你要在云T4上跑vLLM API服务，部署前必须能“看懂”机器状态——内存够不够、磁盘满没满、进程是否正常。这5个命令就是你的“仪表盘”。

**逐个上手（跟着敲）**

1. **`uname -a`**：查看系统内核、架构等信息。先跑一次，确认你的云服务器是Linux且架构正常。

2. **`free

## 块 25

本块学习检查：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

**目标关联**：部署vLLM服务时，你需要修改配置文件（如模型路径、端口号）。本块教你用`vim`快速编辑文件，用`echo`写入或追加内容，这是操作云服务器的基础技能。

**核心操作**（T4云服务器终端）：

1. **vim编辑文件**（修改配置首选）
   - 打开：`vim config.py`
   - 按`i`进入编辑模式（左下角显示`--INSERT--`）
   - 修改后按`Esc`，输入`:wq`保存退出
   - 改错了？按`Esc`，输入

## 块 26

## 网络操作：基础网络查询

你的目标是部署 vLLM API 服务，这意味着模型要跑在云服务器上，而你需要通过网络访问它。所以，掌握网络查询命令是排查"连不上"问题的基本功。

**`ping`**：测试网络连通性。语法：`ping [选项] 目标IP/域名`。例如测试能否连外网：`ping www.baidu.com`，按 `Ctrl+C` 停止。

**`ifconfig`**：查看网卡信息，包括 IP 地址、MAC 地址等。直接输入 `ifconfig`，在输出中找到 `eth0` 或 `wlan0` 的 `inet` 字段，那就是你的 IP 地址。部分系统默认用 `ip addr` 命令。

> 注意：`ifconfig

## 块 27

## 6.6 命令实战场景演练

你已经有 PyTorch 和 Transformer 基础，但部署 vLLM 服务时，日常 Linux 操作必须熟练。下面两个场景直接对应你之后的工作流，动手敲一遍。

### 场景 1：创建并整理文件

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

**说明**：`-p` 参数允许一次创建多级目录；`>` 将内容

## 块 28

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

### 6.7 新手避坑指南（针对你的 vLLM 部署目标）

在云服务器上折腾 vLLM 时，终端命令就是你的“方向盘”。下面四个坑，踩中任何一个都会让你卡壳：

**1. 大小写敏感**：Linux 命令严格区分大小写。`ls` 能列出文件，`LS` 直接报错。敲命令时手指别抖。

**2. 路径要确认**：部署时切换目录、复制模型文件，路径写错会提示 `No such file or directory`。先 `ls` 看一眼路径是否存在，再动手。

**3. `rm` 是核弹**：`rm -rf` 能瞬间删光一切。删任何东西前，先用 `ls` 确认你瞄准的是目标文件，而不是根目录 `/`。云服务器上误删，哭都来不及。

**4. 权限不够就加 `sudo`**：访问系统文件或别人的文件时，提示 `Permission denied`，在命令前加 `sudo` 获取临时管理员权限

## 块 29

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的云服务器（Tesla T4 16GB）上，Ubuntu 24.04 已自带 Python 3.12。先验证环境：

```bash
python3 --version   # 预期输出：Python 3.12.x
pip3 --version      # 若提示未找到，继续下面步骤
```

若 `pip3` 缺失，安装它：

```bash
sudo apt update
sudo apt install -y python3-pip
```

**pip 是什么？** Python 的包管理器，类似“应用商店”。后续安装 PyTor

## 块 30

## 7.2 （可选但推荐）安装 Miniconda

你的目标是 vLLM 部署，后续必然涉及多个 Python 包（如 PyTorch、vLLM 本身）的安装与版本管理。系统自带 Python 虽可用，但不同项目依赖容易冲突。Miniconda 能帮你隔离环境，避免“装 A 搞坏 B”的尴尬——这正是你后续实操的刚需。

**安装步骤：**

1. **下载安装脚本**（若提示 `wget` 未找到，先执行 `sudo apt install wget`）：
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **运行安装**：
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. **安装过程交互**：
   - 按 `Enter` 阅读许可协议
   - 输入 `yes` 同意
   - 安装路径直接回车（默认）
   - 提示 "Do you wish to update your shell profile?" 时输入 `yes`

4. **激活配置**：重新打开终端

## 块 31

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

### 7.3 运行你的第一个 Python 程序（①）

你的目标是用 vLLM 部署大模型推理服务，而 vLLM 是 Python 生态的工具。所以，先确认这台 Ubuntu 环境能跑 Python，是后续所有工作的地基。下面三步走，验证环境就绪。

**第一步：创建 Python 文件**

在终端里依次执行：

```bash
cd ~
nano hello.py
```

- `cd ~`：回到当前用户的主目录，后续操作都在这里进行。
- `nano hello.py`：用 nano 编辑器打开（或新建）一个名为 `hello.py` 的文件。

**第二步：输入代码并保存**

在 nano 打开的界面里，输入这一行：

```python
print("Hello, Python on Ubuntu 24.04!")
```

然后按 `Ctrl + O` 保存，按回车确认文件名；再按 `Ctrl + X` 退出 nano。

**第三步：运行程序**

回到终端，执行：

```bash
python3 hello.py
```

如果屏幕输出 `Hello, Python on Ubuntu 24.04!`，说明 Python 环境正常，终端操作也过关了。

---

**为什么这对 vLLM 部署很重要？**

- 如果你使用的是云上的 Tesla T4 16GB 显卡，vLLM 正是跑在这种 GPU 云环境上的推理服务框架。
- 刚才用到的 `cd`（切换目录）、`nano`（编辑文件）、`python3`（执行脚本），是之后写配置、改代码、启动服务的日常操作。
- 这一步验证了 Python 能运行，接下来安装 PyTorch、HuggingFace 工具链、vLLM 才有基础。

**常见问题速查（本块相关）**

| 症状 | 可能原因 | 解决办法 |
|---|---|---|
| `python3` 提示找不到命令 | Python 未安装或未加入 PATH | 回到上一节确认 Python 3.12 与 pip 已就绪 |
| nano 里无法

## 块 32

## 二、你在哪？pwd 与路径的概念

### 2.1 pwd：打印当前工作目录

`pwd` 是 **P**rint **W**orking **D**irectory 的缩写，作用是告诉你“我现在在哪个文件夹里”。

```bash
pwd
```

示例输出：
```bash
/home/zhangsan
```

这表示你当前位于 `/home/zhangsan` 目录下。

**为什么这对你很重要？** 你之后要部署 vLLM 模型服务，需要频繁操作模型文件、配置文件和日志目录。如果在错误目录下执行命令，轻则“文件找不到”，重则覆盖错文件。所以，**操作文件之前先敲 `pwd`**，确认自己在哪里，是避免低级错误的第一道防线。

**实践建议**：打开你的 Ubuntu 终端，先敲 `pwd` 看看当前在哪。然后执行 `cd /tmp` 再敲 `pwd`，观察输出变化。多试几次，直到你形成“先确认位置再动手”的肌肉记忆。

---

**本块学习检查**

完成本块后，请用自己的话回答：`pwd` 命令和你最终要完成的 vLLM 部署目标有什么关系？为什么在操作模型文件前先确认当前目录是必要的？如果遇到无法解释的术语（比如路径中的 `/` 含义），先记录下来，稍后提问。

## 块 33

### 2.2 绝对路径 vs 相对路径

搞清这两个概念，vLLM部署时定位模型文件才不会迷路。

| 路径类型 | 定义 | 示例 |
|---|---|---|
| **绝对路径** | 从根目录`/`开始，不依赖当前位置 | `/home/zhangsan/Documents` |
| **相对路径** | 相对于你当前所在的位置 | `./file.txt`、`../backup` |

两个关键符号：
- `./`：当前目录（可省略）
- `../`：上一级目录
- `~`：当前用户家目录（如`/home/zhangsan`）

**举例**：假设你在`/home/zhangsan/Documents`下：
- `cd /home/zhangsan`（绝对路径，去哪都管用）
- `cd ..`（相对路径，回到上一级）
- `cd ~`（直接回家目录）

---

## 三、ls：列出目录内容

`ls

## 块 34

### 3.1 基础用法

`ls` 是查看目录内容的命令，直接输入 `ls` 列出当前目录下的文件和文件夹（不含隐藏文件）。也可以指定路径，例如 `ls /home/zhangsan` 查看该目录内容。

- `ls -a`：显示所有文件，包括以 `.` 开头的隐藏文件（如 `.bashrc`、`.ssh`）。这些隐藏文件在后续配置环境变量时会频繁遇到，很重要。
- `ls -l`：以详细列表显示，包含权限、所有者、大小、修改时间等信息。
- `ls -ahl`：组合参数，详细列出所有文件，大小以人类可读格式（如 K、M）显示。

### 3.2 解读 ls -l 输出

执行 `ls -l` 后，输出类似：

```bash
drwxr-xr-x 2 zhangsan zhangsan 4096 Aug  3 10:00 Documents
-rw-r--r-- 1 zhangsan zhangsan  220 Aug  3 09:30 hello.py
```

第一列

## 块 35

## 三、ls：其他常用选项（①）

在部署 vLLM 时，你常需要快速定位模型权重文件或日志，以下选项能帮你高效操作：

- **`ls -lh`**：以详细列表+人类可读格式显示文件大小（KB/MB/GB）。注意 `-h` 必须搭配 `-l` 使用，单独 `ls -h` 无效。
- **`ls -t`**：按最后修改时间排序（最新在前），适合找刚下载的模型文件。
- **`ls -S`**：按文件大小排序（从大到小），快速识别大文件。
- **`ls -rt`**：逆转排序（配合 `-t` 或 `-S

## 块 36

## 4.1 cd：切换目录——先学会“走到模型文件所在的地方”

你接下来的目标是部署 vLLM，而 vLLM 的模型权重、配置文件、日志输出都存放在服务器的不同目录里。**能不能快速、准确地“走到”目标目录，是后续所有操作的前提**。这一节我们把 `cd` 的基础用法一次讲透。

### 两种“指路”方式

- **绝对路径**：从根目录 `/` 开始写全，例如 `cd /home/zhangsan/Documents`。无论你现在在哪，这条命令都能直接到达目标。
- **相对路径**：从当前目录出发，例如 `cd Downloads`（进入当前目录下的 Downloads 子文件夹）。`./` 表示当前目录，通常可以省略。

> **给你的建议**：刚开始操作时，优先用绝对路径，避免“迷路”；等熟悉了目录结构，再用相对路径提升效率。

### 常用快捷切换

```bash
# 返回上一级目录
cd ..

# 返回上两级目录
cd ../..

# 回到家目录（两种写法等价）
cd ~
cd

# 返回刚才所在的目录（在两个目录间快速来回切换）
cd -
```

### 路径切换示例

假设当前在 `/usr/local`，想切换到 `/home/zhangsan`：

```bash
# 方法1：绝对路径（推荐新手）
cd /home/zhangsan

# 方法2：相对路径（先回根目录，再去 home）
cd ../../home/zhangsan
```

---

## 五、mkdir：创建目录

`mkdir` = **M**a**k**e **Dir**ectory，创建文件夹。

```bash
# 在当前目录下创建文件夹
mkdir myproject

# 在指定路径下创建文件夹
mkdir /home/zhangsan/myproject

# 递归创建多级目录（父目录不存在时自动创建）
mkdir -p myproject/src/models

# 一次性创建多个同级文件夹
mkdir dir1 dir2 dir3

# 使用花括号批量创建
mkdir {dir1,dir2,dir3}

# 批量创建 100 个文件夹（dir1 ~ dir100）
mkdir dir{1..100}
```

> **注意**：`mkdir` 默认**不能**直接创建多级嵌套目录，必须加 `-p` 参数。这是新手最常踩的坑之一。

---

## 六、touch：创建空文件

`touch` 原本用来更新文件时间戳

## 块 37

## 7.1 复制文件

你的 vLLM 部署工作流里，经常要把模型配置文件、启动脚本复制到备份目录或新环境。`cp` 就是干这个的。

```bash
# 复制文件到另一个文件夹（保留原文件名）
cp hello.py ./backup/

# 复制文件到另一个文件夹，并改名
cp hello.py ./backup/hello_backup.py

# 一次性复制多个文件到目标文件夹
cp file1.txt file2.txt ./backup/

# 复制时保持文件属性不变（如修改时间）
cp -p hello.py ./backup/

# 覆盖前进行提示（如果目标文件已存在）
cp -i hello.py ./backup/
```

## 7.2 复制目录

复制文件夹时，**必须加 `-r` 参数**（recursive，递归），否则系统会拒绝。

```bash
# 复制整个目录到另一个位置
cp -r myproject ./backup/
```

注意：如果 `backup` 文件夹不存在，会新建 `backup` 并把 `myproject` 的内容复制进去；如果 `backup` 已存在，会新建 `backup/myproject/`，把整个文件夹复制进去。

## 7.3 复制软链接

```bash
# 拷贝软链接时保持链接

## 块 38

## 8.1 移动文件/文件夹

在 vLLM 部署场景中，你经常需要把模型权重、配置文件或 Python 脚本移动到指定目录。`mv` 就是干这个的：

```bash
# 把文件移动到另一个文件夹
mv hello.py ./backup/

# 把文件夹移动到另一个位置
mv myproject ./archive/

# 使用通配符移动多个文件（把所有以 data_ 开头的文件移到 data 文件夹）
mv data_* ./data/
```

## 8.2 重命名

```bash
# 把 a.txt 重命名为 b.txt（本质是在同一目录下移动）
mv a.txt b.txt

# 移动的同时改名
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

# 强制删除，不提示确认
rm -f hello.py

# 强制递归删除文件夹及其中所有内容（⚠️ 极度危险，务必确认路径正确！）
rm -rf myproject/
```

> **⚠️ 生命警告**

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 39

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

先搭好项目骨架，后续模型文件、脚本和数据才能各归其位。按顺序执行：

```bash
cd ~
mkdir llm-learning
cd llm-learning
mkdir data models scripts
touch scripts/train.py
ls -R
```

`cd ~` 回用户主目录；`mkdir` 一次建三个子目录；`touch` 生成空文件；`ls -R` 递归查看，确认结构为：

```
llm-learning/
├── data
├── models
├── scripts
│   └── train.py
```

**练习**：手动

## 块 40

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**场景 2：复制与整理实验文件**

你的目标是用vLLM部署模型，这需要先准备好实验数据。本块练习用`cp`命令整理文件，正是为后续加载数据集打基础。

**操作步骤**（在云T4环境终端执行）：

1. **确认文件存在**  
   `ls ~/Downloads/*.csv`  
   查看`Downloads`目录下是否有`train.csv`、`test.csv`、`dev.csv`三个文件。

2. **复制到目标目录**  
   `cp ~/Downloads/t

## 块 41

## 场景 3：清理临时文件——你的 vLLM 部署第一步

在 `~/llm-learning/` 下，你误建了一个 `temp` 空文件夹和一个 `old.txt` 文件。删除它们，保持工作区干净——这是管理模型文件、日志和缓存的基本功。

```bash
# 1. 先确认要删的内容（好习惯！）
ls ~/llm-learning/temp ~/llm-learning/old.txt

# 2. 删除文件
rm ~/llm-learning/old.txt

# 3. 删除空文件夹
rmdir ~/llm-learning/temp

# 4. 验证
ls ~/llm-learning/
```

**关键点**：删除前用 `ls` 确认目标，删除后用 `ls` 验证结果。这套"确认→操作→验证"的流程，和你调试模型时的"检查输入→跑推理→看输出"是同一个思路。

---

## 新手避坑指南（重点看 1、5、6）

**1. 区分文件和文件夹**：`ls -l` 第一列是 `d` 表示文件夹，是 `-` 表示文件。对文件夹操作别忘了 `-r`（如 `cp -r`、`rm -r`）。

**2. 路径写错**：切换目录或复制文件时，路径写错会提示 "No such file or directory"，可先用 `pwd` 和 `ls` 确认当前位置和目录内容。

**3. 路径中包含空格必须用引号包裹**：如果文件夹或文件名包含空格，命令行会把它当成多个参数。务必用双引号包裹：

```bash
# 错误：mkdir my folder 会创建两个文件夹 my 和 folder
# 正确：
mkdir "my folder"
cd "my folder"
touch "my file.txt"
cp "my file.txt" ./backup/
rm "my file.txt"
```

**4. 文件名以 `-` 开头时使用 `--`**：Linux 命令中 `-` 开头通常表示选项。如果文件名本身以 `-` 开头，命令会误把文件名当选项解析。用 `--` 告诉命令"后面都是文件名，不是选项"：

```bash
# 错误：rm -file.txt 会把 -file.txt 当成选项，报

## 块 42

## 环境变量是什么？——为 vLLM 部署打基础

### 1.1 通俗理解

环境变量是操作系统或应用程序运行时使用的**动态键值对**，用于传递配置信息。你可以把它理解为操作系统的“备忘录”——系统启动时加载这些备忘录，所有程序都能读到里面的内容，并据此调整自己的行为。

**常见例子**：

| 变量名 | 作用 |
|---|---|
| `PATH` | 告诉系统去哪里找可执行程序（比如你输入 `python3`，系统就是靠 PATH 找到它的位置） |
| `HOME` | 当前用户的家目录路径（如 `/home/zhangsan`） |
| `USER` | 当前登录的用户名 |
| `SHELL` | 当前使用的 Shell 程序路径（如 `/bin/bash`） |
| `LANG` | 系统语言和字符编码（如 `en_US.UTF-8`） |
| `PWD` | 当前工作目录路径（随 `cd` 命令自动更新） |
| `EDITOR` | 默认文本编辑器（影响 `git commit` 等操作） |

### 1.2 为什么需要环境变量？

1. **控制程序行为**：例如通过 `JAVA_HOME` 指定 JDK 路径，让依赖 Java 的程序知道去哪找它。
2. **简化配置**：避免在每次运行程序时都手动输入长长的路径。
3. **跨进程通信**：子进程会自动继承父进程的环境变量，实现配置的无缝传递。

---

## 二、环境变量的分类：临时 vs 永久

环境变量按生命周期分为两类，理解

## 块 43

## 临时变量：当前终端会话有效

临时变量只活在当前终端里——**关掉窗口或断开 SSH，它就没了**。适合快速测试，比如你调 vLLM 时临时设个 `MODEL_NAME` 试试效果。

```bash
# 设置一个临时变量
export MY_VAR="hello_linux"

# 查看它的值
echo $MY_VAR
# 输出：hello_linux

# 删除这个变量
unset MY_VAR

# 再查看
echo $MY_VAR
# 输出：（空，因为已经删除了）
```

**`export` 是什么？** 不加 `export`，变量只对当前 Shell 有效，子进程（比如你启动的另一个程序）将无法

## 块 44

**永久变量：写入配置文件，持久生效**

临时变量只在当前终端有效，关掉就没了。要让变量每次打开终端都在，就得写进**配置文件**。

| 级别 | 配置文件 | 影响范围 | 是否需要 sudo |
|---|---|---|---|
| **用户级（推荐）** | `~/.bashrc` | 仅当前用户 | 否 |
| **用户级** | `~/.bash_profile` 或 `~/.profile` | 仅当前用户（登录时生效） | 否 |
| **系统级** | `/etc/profile` 等 | 所有用户 | 是 |

日常开发优先用 `~/.bashrc`，只影响你自己，改错也不影响系统。系统级配置要 `sudo` 且影响所有人，不建议动。

**本块学习检查**：完成

## 块 45

## 3.1 用户级配置：修改 ~/.bashrc（推荐）

你要做 vLLM API 服务，部署时经常要设置模型路径、端口号等环境变量。如果每次开终端都重新 export 一遍，太浪费时间了。`~/.bashrc` 就是解决这个问题的——它是 Bash 的配置文件，**每次打开新的交互式终端时自动加载**，日常开发最常用。

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

**`source` 是什么？** 相当于"重新执行一遍这个配置文件"，让刚才的修改在当前终端立刻生效，不用关掉重开。

**注意一个坑**：`~/.bashrc` 不是所有场景都会自动加载。某些 IDE 的内置终端可能只

## 块 46

## 用户级配置与优先级：为 vLLM 部署铺路

你正打算用 vLLM 在云上部署 API 服务，这意味着你会经常通过 SSH 登录服务器、启动服务。理解“登录 Shell”和“普通终端”的区别，能帮你避免“明明配好了环境变量，重启后却失效”的坑。

**用户级配置的两个文件**：`~/.bash_profile` 和 `~/.profile` 仅在**登录 Shell**时生效（`~/.bash_profile` 仅对 Bash 生效，且若存在则优先于 `~/.profile`）。若两者同时存在，Bash 会优先读取 `~/.bash_profile`（若存在），否则读取 `~/.profile`。日常开发中 `~/.bashrc` 更常用，因为每次开终端都会加载；只有需要“登录时才生效”的场景才用这两个文件。

**环境变量优先级**（从高到低）：
1. **临时变量**（当前终端 `export` 的

## 块 47

## 5.1 PATH：命令的"搜索路径"

PATH 是最重要的环境变量，没有之一。它定义了系统**查找可执行文件**的目录列表。

### 查看 PATH

```bash
echo $PATH
```

输出类似这样（多个路径用冒号 `:` 分隔）：

```bash
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

当你在终端输入 `python3` 时，系统会**从左到右**依次在这些目录里找名为 `python3` 的可执行文件，找到了就执行，找不到就提示 `command not found`。

### 查看命令具体位置

```bash
which python3
# 输出：/usr/bin/python3

which conda
# 输出：/home/zhangsan/miniconda3/bin/conda
```

`which` 命令能告诉你某个命令到底是从 PATH 的哪个目录里找到的。

### 修改 PATH

**追加路径**（添加到末尾，优先级较低）：

```bash
export PATH=$PATH:/new/path
```

**前置路径**（添加到开头，优先级更高，系统会先搜这里）：

```bash
export PATH=/new/path:$PATH
```

> **安全提示**：不要把 `.`（当前目录）加入 PATH。如果你在某个目录里下载了一个名为 `ls` 的恶意脚本，而当前目录又在 PATH 里，那么你执行 `ls` 时可能运行的是恶意脚本而非真正的系统命令。
>
> **避免重复添加**：多次 `source ~/.bashrc` 或在 `~/.bashrc` 中重复写入同一行，可能导致 PATH 里出现重复路径。建议定期检查 `~/.bashrc`，确保每条 PATH 修改只出现一次。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（

## 块 48

## 5.2 LD_LIBRARY_PATH：运行时动态库路径

你在做 vLLM 部署时，安装 CUDA 和 PyTorch 后经常要配这个变量。它的作用是告诉系统：**程序运行时**，除了默认目录（`/usr/lib`、`/lib`），还要去哪些额外路径找动态库（`.so` 文件）。

**先看当前值：**

```bash
echo $LD_LIBRARY_PATH
```

**临时添加 CUDA 动态库路径：**

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

**典型场景**：你编译的程序依赖某个 `.so` 文件，但它不在系统默认搜索路径里。把该 `.so` 所在目录加入 `LD_LIBRARY_PATH` 环境变量中。

**

## 块 49

## 5.3 LIBRARY_PATH：编译时库文件路径

`LIBRARY_PATH` 指定编译器（如 `gcc`）在**编译阶段**查找库文件的路径，主要影响静态库（`.a`）和动态库（`.so`）的链接。配置方式：

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/your/libraries
```

**三者区别**：

| 变量名 | 作用阶段 | 查找目标 | 典型用途 |
|---|---|---|---|
| `PATH` | 执行命令时 | 可执行文件 | 让命令在任何位置都能运行 |
| `LD_LIBRARY_PATH` | 程序运行时 | 动态链接库（`.so`） | 运行依赖自定义动态库的程序 |
| `LIBRARY_PATH` | 编译链接时 | 静态库/动态库 | 编译时链接第三方库 |

> **对大模型开发者的建议**：PATH 和 LD_LIBRARY_PATH 是你后续最常打交道的两个变量。安装 Miniconda、CUDA、PyTorch 时，都需要正确配置它们。

---

## 六、实战：配置 CUDA 环境变量

CUDA 是

## 块 50

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

### 6.1 检查 CUDA 是否已在环境变量中

在终端依次运行：

```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

- **有输出**（如 `/usr/local/cuda/bin`）：说明 CUDA 已配置好，直接跳到 6.3 验证。
- **无输出**：说明还没配置，继续 6.2。

> 你的 T4 云实例通常需要手动配置，重点看 6.2。

---

### 6.2 编辑 ~/.bashrc 添加 CUDA 路径

打开配置文件：

```bash
nano ~/.bashrc
```

在文件**末尾**添加两行：

```bash
# CUDA 环境变量配置
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

> **注意**：`/usr/local/cuda` 是默认安装路径。若你的 CUDA 装在其他位置，替换为实际路径。

保存退出（`Ctrl+

## 块 51

## 7.1 查看所有环境变量

先跑 `printenv` 看全部变量，输出简洁；`env` 输出更详细。两者都能让你快速确认当前 shell 里有哪些配置。

## 7.2 查看特定变量

```bash
echo $PATH
printenv PATH
declare -p PATH    # 显示定义细节（含是否 export）
```

第一条最常用，直接看 PATH 值；第三条能确认变量是否已导出给子进程。

## 7.3 测试变量是否生效

假设你在 `~/.bashrc` 里写了 `export MY_VAR="hello"`：

```bash
source ~/.bashrc
echo $MY_VAR
# 应输出：hello
```

**关键点**：改完配置文件必须 `source` 或重开终端，否则不生效。

## 7.4 追踪变量来源

变量值异常时，

## 块 52

## 8.1 变量未生效（①）——vLLM部署前的环境排查

你正在为vLLM API服务配置环境，如果刚改完`~/.bashrc`却发现`vllm`命令找不到，多半是下面几个原因：

**可能原因**：
- 改完配置没执行`source ~/.bashrc`或没重启终端
- 配置文件语法有误（比如漏写了`export`）
- 变量在别处被覆盖
- 当前终端/IDE没有加载`~/.bashrc`

**排查步骤**（按顺序执行）：

```bash
# 1. 确认你的变量确实写进了配置
cat ~/.bashrc | grep "你的变量名"

# 2. 重新加载配置
source ~/.bashrc

# 3. 检查PATH是否有重复定义
grep "PATH=" ~/.bashrc
```

**本块学习检查**：完成本块后，用自己的话说明“变量未

## 块 53

## 9.1 脚本中临时修改变量（①）

在 Shell 脚本里，你可以用 `export` 临时设置一个变量，它只在脚本运行期间生效：

```bash
#!/bin/bash
export TEMP_VAR=123
# 仅在脚本执行期间生效，脚本结束后变量消失
```

这很适合 vLLM 部署脚本：你可以在启动服务前临时设置某个参数，脚本跑完就自动清理，不会污染后续操作。

## 9.2 用 env 命令指定单次运行的变量（①）

如果只想让某个变量对**一次命令**生效，用 `env` 更精准：

```bash
env MY_VAR=hello python3 my_script.py
# MY_VAR 仅在这次 python3 运行时有效
```

部署 vLLM 时，比如你想临时指定一个环境变量来启动 API 服务，又不想改全局配置，用 `env` 一行搞定。

## 9.3 用 .env 文件分组管理变量（④）

项目需要多个变量时，建一个 `.env` 文件集中管理：

```bash
# 创建 ~/.env 文件
nano ~/.env
```

内容示例：

```bash
export PROJECT_HOME="$HOME/llm-learning"
export EDITOR="nano"
export TZ="Asia/Shanghai"
```

加载方法：

```bash
set -o allexport
source ~/.env
set +o allexport
```

> **⚠️ 安全警告**：`source .env` 会**执行**文件中的每一行 Shell 代码。如果

## 块 54

## 场景 1：创建并验证自定义变量

你部署 vLLM 时，经常需要设置模型路径、API 端口等参数。这些值如果每次手动敲，既慢又容易错。本场景教你用环境变量把它们“存”起来，一次设置、随时调用。

**第一步：临时变量（当前终端有效）**

```bash
export MY_NAME="zhangsan"
```

`export` 把变量注入当前 shell 环境。注意：关闭终端后它就消失了，适合临时测试。

**第二步：验证**

```bash
echo $MY_NAME
```

`$` 是取值符号，`echo` 打印结果。看到 `zhangsan` 即成功。

**第三步：永久生效**

```bash
echo 'export MY_NAME="zhangsan"' >> ~/.bashrc
```

`>>` 把这条命令追加到 `~/.bashrc` 文件末尾。每次打开新终端，系统会自动读取该文件，变量就“永久”存在了。

**第四步：重新加载并验证**

```bash
source ~/.bashrc
echo $MY_NAME
```

`source` 让当前终端立即读取刚修改的 `~/.bashrc`，省去重开终端的麻烦。再次 `echo` 确认值还在。

> **实践

## 块 55

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你正为vLLM部署打基础。即使云上T4暂未装CUDA，也可先模拟配置流程，为后续真实环境做准备。

**操作步骤**（在终端执行）：

1. 编辑配置文件：`nano ~/.bashrc`
2. 在文件末尾添加两行（假设CUDA装在`/usr/local/cuda`）：
   ```
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
3. 保存并生效：`source ~

## 块 56

## 系统级配置文件（①）：为 vLLM 服务做准备

你目标是用 vLLM 部署 API 服务，这通常跑在云服务器上。系统级配置影响**所有用户**，需要 `sudo` 权限——日常开发少用，但配置公共服务器时必须懂。

### A.1 /etc/environment（①）

**特点**：不需要 `export` 关键字，直接写 `KEY=value`。但只支持简单键值对，**不支持** Shell 语法（如 `$PATH` 引用）。

```bash
sudo nano /etc/environment
# 添加：
JAVA_HOME=/usr/lib/jvm/java-11-openjdk
# 保存后，重启系统或重新登录才生效
```

### A.2 /etc/profile（①）

**特点**：全局 Bash 配置，**需要** `export` 关键字。

```bash
sudo nano /etc/profile
# 添加：
export PATH=$PATH:/usr/local/custom/bin
# 保存后，source 当前会话生效，或重启系统生效
source /etc/profile
```

**本块学习检查**：完成本块

## 块 57

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你部署vLLM时，终端环境决定配置文件位置。先执行 `echo $SHELL` 确认Shell类型：输出 `/bin/bash` 即Bash。

- **Bash**（最常见）：日常终端用 `~/.bashrc`；SSH登录用 `~/.bash_profile`。
- **Zsh**：用 `~/.zshrc`。
- **Fish**：用 `~/.config/fish/config.fish`。

**实践任务**：在云T4实例上运行 `echo $SHELL`，确认Shell

## 块 58

## 清理 PATH 重复项与配置管理（进阶参考）

你已经掌握了 PATH 的追加与前置修改，但多次修改后 PATH 里可能出现重复路径。重复项不会报错，但会让 `which` 和 `type -a` 的输出变得混乱，排查问题时容易误判。下面这条命令可以一键去重：

```bash
export PATH=$(echo $PATH | awk -v RS=: '!a[$0]++' | paste -sd: -)
```

**拆解理解**：`echo $PATH` 输出当前 PATH，`awk` 按冒号分隔并利用数组 `a` 记录已见过的路径，只保留首次出现的项，最后 `paste` 用冒号重新拼接。这条命令只影响当前 Shell 会话，想永久生效需写入 `~/.bashrc`。

**进阶技巧**：你可以在 `~/.bashrc` 中按环境变量加载不同配置。比如根据 `ENVIRONMENT` 变量的值，决定加载生产还是开发环境的配置：

```bash
if [ "$ENVIRONMENT" = "production" ]; then
    source ~/.env.prod
else
    source ~/.env.dev
fi
```

**调试利器**：当脚本或配置出问题时，可以用追踪模式查看每一行执行前的变量值：

```bash
bash -x your_script.sh
```

或者只对当前 Shell 的某段操作开启追踪：

```bash
set -o xtrace
source ~/.bashrc
set +o xtrace
```

---

### 本章回顾与下一步

至此，环境变量配置章节的核心内容你已经全部过了一遍。回顾一下你已掌握的能力：环境变量是键值对；临时变量用 `export`/`unset`，永久变量写入配置文件；`source` 让配置立即生效；PATH、LD_LIBRARY_PATH、LIBRARY_PATH 三者用途不同；修改 PATH 有追加和前置两种方式；调试用 `echo`、`printenv`、`which`、`type -a`、`grep -r`；`.env` 文件不可 source 不可信

## 测评题目

### Q-KB0100-003
执行以下命令确认系统发行版信息：
```bash
cat /etc/os-release
```

### Q-KB0100-001
请在终端中执行以下命令，验证Linux/WSL是否安装成功，将完整输出粘贴作答：
```bash
uname -a
```

### Q-KB0101-002
命令 `ls -la` 中，`-l` 的作用是________，`-a` 的作用是________。

### Q-KB0101-003
请依次执行以下命令并描述每一步的结果：
```bash
mkdir -p /tmp/test_dir/sub_dir
cd /tmp/test_dir
touch hello.txt
cp hello.txt sub_dir/
mv hello.txt renamed.txt
ls -R
```

### Q-KB0102-003
以下哪个文件是bash shell用户登录时**最先**读取的全局配置文件？

### Q-KB0102-002
请写出将 `/usr/local/cuda/bin` 永久添加到 `PATH` 环境变量的完整步骤（以bash为例）。

### Q-KB0103-005
在conda环境中使用 `pip install` 安装的包，可以用 `conda list` 查看到吗？反过来呢？

### Q-KB0101-005
请写出一条命令，将 `/tmp/test_dir` 下所有 `.txt` 文件复制到 `/tmp/backup/` 目录中（如目录不存在则先创建）。
