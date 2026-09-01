# 学习包

## 块 1

### 1.1 什么是 Linux？我为什么要装它？

你平时用的 Windows 或 macOS 叫“操作系统”，Linux 是另一个操作系统。全世界绝大多数服务器（网站后台、云服务、AI 训练集群）跑的都是 Linux。学编程到后面，你大概率要和它打交道。

本教程用的是 **Ubuntu 24.04 LTS**（长期支持版），厂家会持续提供多年安全更新，稳定且适合入门。如果你已安装 Ubuntu 22.04，后续步骤同样适用。

**为什么这对你（vLLM 部署）很关键**：你之后要跑的 vLLM 推理服务，生产环境几乎都跑在 Linux 服务器上。现在花 3 小时把环境弄熟，后面部署时能少踩一半坑。

**本块学习检查**（先完成再继续）：
1. 打开终端，依次执行 `cd ~`、`ls`、`mkdir test_dir`、`cp -r test_dir test_dir2`、`mv test_dir2 test_dir3`，确认每条命令都能成功执行且无报错。
2. 用一句话解释 `cd` 和 `ls` 各自的作用（基于你刚才的操作观察）。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你有PyTorch和Transformer基础，目标是把vLLM跑起来。动手前先选对“战场”，否则后面全是坑。

**三种方案速览**

| 方案 | 一句话 | 适合你吗？ |
|---|---|---|
| 虚拟机 | 电脑里模拟一台新电脑 | 内存≥8GB才流畅，你每天3小时实操，别把时间耗在卡顿上 |
| **WSL2** | Windows自带的Linux子系统，像装App | **首选**。轻量、启动快、和Windows互通，几分钟装完 |
| 云服务器 | 租

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

三个方案最终效果一致：你得到一台运行 Ubuntu 24.04 的“电脑”（虚拟或远程）。虚拟机/云服务器需通过 SSH 从 Windows 远程操作；WSL2 则与 Windows 互通，用 VS Code 本地开发。三者都支持 VS Code 插件，实现代码高亮、补全和调试。

**实践验证**：对照你的 RTX 4090 环境，确认你计划采用的方案（如云服务器）能否通过 SSH 连接，并检查 VS Code 是否具备远程开发插件。

## 块 4

**VirtualBox 是什么？** 一款开源虚拟机软件，能在你的 Windows 里造出一台“假电脑”，用来装任何操作系统——包括你后续部署 vLLM 所需的 Linux 环境。

**下载步骤：**
1. 浏览器打开：`https://www.virtualbox.org/wiki/Downloads`
2. 找到 **"VirtualBox 7.x.x platform packages"** 下的 **Windows hosts** 版本，点击下载
3. 双击安装包，一路“下一步”保持默认，点“安装”等进度条跑完，最后“完成”

**坑预警：** 弹“是否允许更改设备”选“是”；杀毒软件拦截就选“允许/信任”。

**本块学习检查：** 用自己的话说说，装 VirtualBox 和你最终用 vLL

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**下载 Ubuntu 24.04 镜像文件**

镜像文件（.iso）可理解为“操作系统安装光盘的电子版”，VirtualBox 通过它来安装系统。

**操作步骤：**
1. 浏览器打开 `https://releases.ubuntu.com/24.04/`
2. 找到文件名类似 `ubuntu-24.04.x-desktop-amd64.iso`（x为小版本号），点击下载
3. 文件约5GB，下载需几分钟到十几分钟（看网速）
4. 下载后**不要双击打开**，放在下载文件夹即可

## 块 6

### 在 VirtualBox 里创建虚拟机（①）

你的目标是 vLLM 部署，而 vLLM 依赖 Linux 环境。本块就是为你的 RTX 4090 搭好这个“舞台”——先建一台 Ubuntu 虚拟机，后续所有部署操作都在里面进行。

**操作步骤（跟着做，约 10 分钟）：**

1. **打开 VirtualBox**：双击桌面上的 “Oracle VM VirtualBox” 图标。
2. **新建虚拟机**：点击窗口上方的 **“新建”** 按钮。
3. **填写基本信息**：
   - **名称**：`Ubuntu-24.04`
   - **文件夹**：选 D 盘或 E 盘（**别放 C 盘**，避免拖慢系统）
   - **类型**：`Linux`；**版本**：`Ubuntu (64-bit)`
   - 点“下一步”。
4. **分配内存**：拖动滑块到 **4096 MB**（4GB），点“下一步”。
5. **创建虚拟硬盘**：
   - 选 **“

## 块 7

**目标关联**：你正在为 vLLM 部署准备 Linux 环境，本块完成虚拟机内 Ubuntu 安装，这是后续所有命令操作的基础。

**安装步骤**（跟着做即可）：

1. 在 VirtualBox 中选中虚拟机，点击 **"启动"**。
2. 出现安装界面后：
   - 语言选 **"中文(简体)"**，点"安装 Ubuntu"。
   - 键盘布局默认，点"继续"。
   - 更新选项选 **"正常安装"**，勾选"下载更新"和"安装第三方软件"，点"继续"。
   - 安装类型选 **"清除整个磁盘并安装 Ubuntu"**——注意这里的磁盘是虚拟机的 60G，不会碰你真实的 C/D 盘，点"现在安装"并确认。
   - 时区输入"Shanghai"或地图点中国，点"继续"。
   - 填写用户名、密码等信息，点"继续"。
3. 自动安装约 10~20 分钟。

## 块 8

### 2.5 安装 SSH 远程连接工具

**为什么这一步对你（vLLM 部署）很重要？**  
你的目标是部署 vLLM，而 vLLM 的推理服务通常跑在 Linux 环境里。装好 SSH 后，你就能在 Windows 终端里直接操控虚拟机里的 Ubuntu，后续部署 vLLM、启动服务、查看日志，都不用来回切换窗口，效率高得多。

---

**操作步骤（跟着敲，每行敲完按回车）：**

1. **打开终端**：在 Ubuntu 桌面空白处右键 → 选“在终端中打开”。如果没有，按 `Ctrl + Alt + T` 也行。

2. **依次执行以下命令**：

```bash
# 第1步：更新软件列表（相当于刷新应用商店的货架）
sudo apt update

# 第2步：安装 SSH 服务端和网络工具
sudo apt install -y openssh-server net-tools

# 第3步：设置 SSH 开机自启
sudo systemctl enable ssh

# 第4步：立刻启动 SSH
sudo systemctl start ssh
```

> **sudo 是什么？** 就是“以管理员身份执行”。Linux 里普通用户不能随便装软件，加 sudo 就是临时借管理员权限。输密码时屏幕**不会有任何显示**（不会出现 `***`），这是正常的，盲打后回车即可。

3. **查看虚拟机的 IP 地址**：

```bash
ip addr
```

4. 输出内容很多，找到类似 `inet 192.168.xxx.xxx` 的一行，那一串数字就是 IP 地址。**记下来！** 例如 `192.168.126.128`。

5. **回到 Windows 验证连接**：按 `Win + R`，输入 `cmd` 回车，打开命令提示符，输入：

```bash
ssh ubuntu@192.168.xxx.xxx
```

> 将 `ubuntu` 替换为你的 Ubuntu 实际用户名。

6. 首次连接时，会出现 `Are you sure you want to continue connecting (yes/no)?` 提示，输入 `yes` 回车确认。接着提示输入密码就输入 Ubuntu 的密码，回车。看到 `Welcome to Ubuntu`

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

安装增强功能，是为了让 Ubuntu 虚拟机与 Windows 之间能直接复制粘贴文字和文件。这能帮你把 Windows 里的 vLLM 部署脚本、配置或日志，快速贴进虚拟机执行，省去手动敲命令的麻烦。

新版 VirtualBox 通常会自动装好。若没生效，在 Ubuntu 终端执行：

```bash
sudo apt install -y virtualbox-guest-utils virtualbox-guest-x11
sudo reboot
```

重启后，Windows 里 `Ctrl+C` 复制，Ubuntu 里 `Ctrl+V` 粘贴即可。

**实操验证**：重启后，在 Windows 复制一行命令，粘贴到 Ubuntu 终端并回车，确认能正常执行。

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

**确认虚拟化已开启（①）**

虚拟化是CPU的一项硬件特性，允许在Windows上运行虚拟机或WSL2等隔离环境。大部分电脑默认开启，但部分品牌机（联想、戴尔等）可能关了。

1. 按 `Ctrl + Shift + Esc` 打开**任务管理器**
2. 点顶部**“性能”**标签页
3. 右下角找**“虚拟化”**，看是“已启用”还是“已禁用”
4. 已启用 → 跳到第二步
5. 已禁用 → 进BIOS开启：
   - 重启，开机狂按 `F2` 或 `Del`（联想F2，惠普F10，戴尔F2）
   -

## 块 11

**本块学习检查**：完成安装后，用自己的话说明 WSL2 与“vLLM 部署”这个目标的关系——为什么在 Windows 上跑 Linux 环境是后续推理部署的前提？遇到不懂的术语（如“内核”）先记下来，再提问。

---

你的目标是 vLLM 部署，而 vLLM 依赖 Linux 生态。WSL2 就是 Windows 里跑一个完整 Linux 内核的“虚拟机”，比 WSL1 快得多，且几分钟就能装好。下面按步骤操作：

1. 按 `Win` 键，输入 `powershell`，在搜索结果里 **右键** “Windows PowerShell” → **“以管理员身份运行”**，弹窗点“是”。
2. 在蓝色窗口里，**逐行** 输入以下命令（复制一行，右键粘贴，回车）：

```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

3. 两条命令都显示“操作成功完成”后，**重启电脑**。
4. 重启后，再次以管理员身份打开 PowerShell，输入：

```bash
wsl --set-default-version 2

## 块 12

**本块学习检查**：完成安装后，请用自己的话说说：为什么装好 Ubuntu 24.04 是后续跑 vLLM 部署的第一步？如果安装中遇到任何看不懂的术语（比如“LTS”），先记下来，稍后提问。

---

**安装 Ubuntu 24.04（推荐方法一）**

1. 按 `Win` 键，输入 `Microsoft Store`，打开微软应用商店。
2. 在搜索框输入 `Ubuntu 24.04`，找到 **"Ubuntu 24.04.x LTS"**（发布者是 Canonical Group Limited），点“安装”。
3. 下载约 500MB，完成后点“打开”。

**更快的方法二（命令行）**

在终端里输入下面这一条命令，即可自动下载并安装：

```bash
wsl --install -d Ubuntu-24.04
```

**首次启动设置**

- 等待 2~5 分钟，屏幕显示 "Installing, this may take a few minutes..."。
- 提示设置用户名：输入全英文小写用户名（如 `zhangsan`）→ 回车。
- 设置密码：输入密码（**屏幕无显示是

## 块 13

### 更新系统与验证（WSL2 方案）

既然你的目标是 vLLM 部署，系统更新是确保后续依赖安装顺利的第一步。刚装好的 Ubuntu 可能带着旧软件包，先刷新并升级，避免后面装 PyTorch 或 vLLM 时碰到版本不兼容的坑。

在 Ubuntu 终端（绿色提示符那个窗口）里，逐行执行：

```bash
# 刷新软件包列表（相当于更新“应用商店的货架”）
sudo apt update

# 把所有已安装软件升级到最新版（-y 自动确认所有提示）
sudo apt upgrade -y
```

> **WSL2 需要 SSH 吗？** 不需要。WSL2 与 Windows 紧密集成，直接用 VS Code 的 Remote-WSL 插件开发即可，不用像虚拟机那样额外配 SSH。

接着验证环境是否正常：

```bash
# 查看系统版本，应显示 Ubuntu 24.04.x
lsb_release -a

# 查看当前用户名，应该是你刚设的
whoami
```

再测一下与 Windows 的互通：

```bash
# 用 Windows 资源管理器打开当前 Linux 目录
explorer.exe .
```

如果弹出资源管理器窗口，说明 WSL2 和 Windows 已无缝衔接，方案二完成。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

**本块学习检查**：完成购买后，用自己的话说明“云服务器”与“vLLM部署”的关系；遇到不懂的术语（如“镜像”）先记下再提问。

以阿里云为例（腾讯云、华为云类似）：

1. 打开 https://www.aliyun.com/ ，用支付宝/淘宝账号注册登录。
2. 顶部搜索“轻量应用服务器”进入；学生先搜“学生优惠”认证，价格更低。
3. 点“立即购买”，配置如下：
   - **地域**：选离你最近的（如华东1杭州）。
   - **镜像类型**：系统镜像；**操作系统**：Ubuntu 24.04 LTS。
   - **套餐**：最低2核2

## 块 15

### 创建普通用户（安全第一步）

你的目标是部署 vLLM，但服务器安全是前提。`root` 是超级管理员，权限过高，误操作可能删掉系统文件，所以日常操作要用普通账户。

**操作步骤**（通过云厂商控制台的“远程连接”或 VNC 网页终端登录后执行）：

1. **创建新用户**：把 `yourname` 换成你想要的英文小写用户名，执行 `adduser yourname`，按提示设置密码（输入两次），其余信息（全名、电话等）直接回车跳过。
2. **授予管理员资格**：执行 `usermod -aG sudo yourname`，让该用户能使用 `sudo` 执行管理命令。

**本块

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 16

### 配置 SSH 密钥登录（安全第二步）

你已经有 PyTorch 和 Transformer 的基础，部署 vLLM 时服务器安全是第一步。密码登录容易被暴力破解，密钥登录既安全又方便，而且配置一次后每次连接都不用输密码。

**在 Windows 上生成密钥对：**

1. 按 `Win + R`，输入 `cmd`，打开命令提示符
2. 执行以下命令：

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

3. 出现 `Enter file in which to save the key` → 直接按**回车**（使用默认路径）
4. 出现 `Enter passphrase` → 直接按**回车**（不设密码短语）
5. 出现 `Enter same passphrase again` → 再按一次**回车**
6. 看到 randomart image，说明密钥生成成功

**把公钥放到服务器上：**

7. 通过云厂商的 VNC/远程连接登录服务器，执行：

```bash
su - yourname
mkdir -p ~/.ssh
nano ~/.ssh/authorized_keys
```

8. 回到 Windows 命令提示符，查看公钥内容：

```bash
type %USERPROFILE%\.ssh\id_rsa.pub
```

把输出的内容全部复制，粘贴到服务器 nano 编辑器中，然后 `Ctrl+O` 保存，`Ctrl+X` 退出。

9. 设置权限：

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

## 块 17

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

**目标**：为 vLLM 部署准备一台可 SSH 登录的云服务器。本块只做一件事：在阿里云控制台放行 22 端口，并限制来源。

**操作步骤**（跟着点）：

1. 登录**轻量应用服务器控制台**，点 **“防火墙”** 标签页。
2. 点 **“添加规则”**，按下面填：
   - **端口**：`22`
   - **协议**：`TCP`
   - **来源**：浏览器搜“IP”查你的公网 IP，填成 `

## 块 18

## 从 Windows 连接你的云服务器

你已有一台 24 小时在线的 Ubuntu 服务器，现在只需从 Windows 连上去操作。

**操作步骤：**

1. 在阿里云控制台找到 **"公网 IP"**（一串数字，如 `47.96.xxx.xxx`），复制下来
2. 按 `Win + R`，输入 `cmd`，打开命令提示符
3. 输入命令连接：

```bash
ssh yourname@47.96.xxx.xxx
```

4. 首次连接提示 "Are you sure you want to continue connecting (yes/no)?"，输入 `yes` 回车
5. 若提示输入密码，输入你创建服务器时设置的登录密码即可登录；若已配置密钥且

## 块 19

## 5.1 虚拟机 / 云服务器：配置 SSH 免密登录（强烈推荐）

你已具备 PyTorch 训练基础，接下来要部署 vLLM，频繁 SSH 登录服务器是必经之路。免密登录能省下每次输密码的时间，让 3 小时高效用在模型推理上。

**验证免密是否生效**：在 Windows 命令提示符中执行：

```bash
ssh 用户名@你的IP地址
```

能直接连上且不提示密码，即配置成功。

**常见坑预警**：若提示 `ssh-copy-id 不是内部或外部命令`，说明 Windows 版本较老。解决：
- 方法一（推荐）：安装 Git for Windows（https://git-scm.com/download/win

## 块 20

### 5.2 WSL2：无需 SSH，直接集成（①）

既然你走 WSL2 路线，这里有个好消息：**不用配 SSH**。VS Code 通过 Remote-WSL 插件直接和 WSL2 通信，打开即连，体验比 SSH 更顺滑。你的目标是 vLLM 部署，后面要在 WSL2 里跑推理服务，这一步打通了，后续调试代码就不用切窗口了。

### 5.3 安装 VS Code 及远程开发插件（①）

**VS Code 是什么？** 全称 Visual Studio Code，微软出的免费代码编辑器。轻量、插件多、跨平台，是目前最主流的编辑器之一。你后面写 vLLM 推理脚本、改配置、看日志，全靠它。

#### 第一步：安装 VS Code（①）

1. 浏览器打开：https://code.visualstudio.com/
2. 点 **"Download for Windows"** 按钮
3. 下载完双击安装包
4. 安装过程：
   - 同意协议 → 下一步
   - 路径默认 → 下一步
   - **重点！** 这页勾选：
     - ☑ **"将 Code 添加到"PATH"中"**（重要！这样命令行直接输 `code` 就能打开 VS Code）
     - ☑ **"将 Code 注册为受支持的编辑器"**
     - ☑ **"添加到文件资源管理器上下文菜单"**
   - 一路"下一步" → "安装" → 勾上"启动 VS Code" → "完成"

#### 第二步：安装必备插件（①）

**插件（Extension）是什么？** 就是 VS Code 的"外挂功能包"。VS Code 本身只是个壳子，装了插件才能高亮代码、自动补全、调试

## 块 21

## 5.4 连接你的 Linux 环境（①）

你的目标是 vLLM 部署，而 vLLM 跑在 Linux 上。这一步就是把 VS Code 接到你的 Linux 环境，之后所有代码编辑、命令执行都在这里完成。你有 RTX 4090，如果用的是云服务器（带 GPU 的），走下面的 SSH 流程；如果本机装了 WSL2，直接跳到最后一段。

**虚拟机或云服务器（SSH 连接）：**

1. 打开 VS Code，看左下角那个 **绿色图标**（形状像 `><`）。
2. 点击它 → 顶部弹出菜单，选 **"Connect to Host…"**。
3. 输入 `用户名@IP地址`，例如 `zhangsan@192.168.126.128`，回车。
4. 弹出平台选择时，选 **"Linux"**。
5. 之前配过密钥就直接连上；没配就输密码。
6. 连接成功后，左下角图标变成 `SSH: IP地址`。
7. 点左上角 **"File"** → **"Open Folder…"**，选服务器上的目录（如 `/home/你的用户名`），点 OK。
8. 现在就能像编辑本地文件一样编辑远程文件了。

**WSL2 用户：**

1. 点左下角绿色图标 → 选 **"Connect to WSL"**。
2. 自动连上 Ubuntu，左下角变成 `WSL: Ubuntu-24.04`。
3. 点 "Open Folder"，选 `/home/你的用户名`。

> **更省事的方法**：在 Ubuntu 终端里 `cd` 到项目目录，直接输入 `code .`，VS Code 自动打开并连接 WSL。

---

## 六、终端入门：Linux 基本命令

终端就是那个黑框框，光标闪烁等你

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 22

## 6.1 文件与目录操作：最基础也最常用

你接下来要部署 vLLM，免不了要下载模型权重、编辑配置文件、查看推理日志。这些操作全部建立在文件与目录操作之上。本块把最核心的 10 个命令讲透，每个都配上你马上会用到的场景。

### 先学会“看”：`ls`、`pwd`、`cd`

- **`pwd`**：显示当前所在目录的绝对路径。刚打开终端不知道自己在哪，先敲它。
- **`ls`**：列出当前目录下的内容。常用变体：
  - `ls -l`（Ubuntu 下常简写为 `ll`）：详细列出，含权限、大小、修改时间。
  - `ls -a`：显示隐藏文件（以 `.` 开头的文件，比如 `~/.bashrc`）。
- **`cd`**：切换目录。四个高频用法：
  - `cd ..` 进入上级目录
  - `cd /` 进入根目录
  - `cd ~` 回到当前用户主目录
  - `cd /home/user/Documents` 进入指定绝对路径

### 创建：`mkdir` 和 `touch`

- **`mkdir`**：创建目录。单级用 `mkdir test`；多级嵌套用 `mkdir -p test/aaa/bbb`，`-p` 会自动创建中间层级。
- **`touch`**：创建空文件。可一次创建多个：`touch file1.txt file2.log`。

### 复制、移动、删除：`cp`、`mv`、`rm`

- **`cp`**：复制。文件直接复制：`cp file1.txt /home/user/Desktop`。**复制目录必须加 `-r`**（递归）：`cp -r test/ /home/user/Desktop`。
- **`mv`**：移动或重命名。移动：`mv file1.txt test/`；重命名：`mv file1.txt newfile.txt`；移动并改名：`mv test/ /home/user/newtest`。
- **`rm`**：删除。**谨慎

## 块 23

## 6.2 权限管理：Linux 系统的安全核心

你部署 vLLM 时，模型文件、日志、脚本都需要正确的权限才能运行。Linux 每个文件/目录都有**所有者、所属组、其他用户**三类权限，权限分为**读（r=4）、写（w=2）、执行（x=1）**。

### 核心命令

**`chmod`** — 修改权限，两种用法：

- **数字法（推荐）**：`chmod 744 file1.txt` — 7=4+2+1（所有者可读+写+执行），4=4（其他用户只读）
- **符号法**：`chmod g+w file1.txt` 给所属组加写权限；`chmod a+x test.sh` 给所有用户加执行权限

**`chown`** — 修改所有者（需 root）：`sudo chown root file1.txt`

> **sudo 回顾**：`sudo` 表示"以管理员身份执行"。输入后让你输密码，**屏幕无任何显示**是正常安全设计。

## 块 24

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**6.3 系统查询：了解 Linux 运行状态**

部署vLLM前，先学会“看系统脸色”——下面5个命令帮你掌握机器状态。

**1. 看内核：`uname -a`**
输出内核版本、主机名、系统架构。确认你的RTX 4090驱动与内核匹配。

**2. 实时监控：`top`**
类似任务管理器。按`P`按CPU排序，按`M`按内存排序，`q`退出。跑vLLM时用它盯GPU进程的CPU/内存占用。

**3. 查进程：`ps

## 块 25

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

针对你部署vLLM的目标，修改配置文件（如模型路径、端口）是必经步骤。本块教你两种快速改文件的方法：

**1. vim编辑器**（适合精细修改）
- 打开：`vim 文件名`（如`vim config.py`）
- 按`i`进入编辑模式（左下角显示`--INSERT--`）
- 修改后：按`Esc` → 输入`:wq`回车保存退出；不保存则输入`:q!`
- 查找内容：按`Esc` → 输入`/关键词`回车，`n`下一个，`N`上一个

**2. echo命令**（适合快速写入）
- 覆盖写入：`echo "内容" > 文件名`
- 追加写入：`echo "内容" >> 文件名`

**

## 块 26

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

部署vLLM前，先确认网络与IP。用`ping`测连通性，如`ping www.baidu.com`，按`Ctrl+C`停止。用`ifconfig`查网卡IP（如eth0的inet字段），若提示找不到，可改用`ip addr`，或执行`sudo apt install net-tools`安装。

## 块 27

## 6.6 命令实战场景演练

你的目标是部署 vLLM，这离不开对 Linux 文件操作和进程管理的熟练度。下面两个场景直接对应部署时的常见操作，动手练一遍。

**场景 1：创建并整理文件**

部署 vLLM 时，你会经常创建项目目录、整理配置文件。跟着做：

```bash
# 1. 进入主目录
cd ~

# 2. 创建 work 及子文件夹（doc 存文档，code 存代码）
mkdir -p work/doc work/code

# 3. 创建并写入文件
echo "Linux命令学习笔记" > note.txt

# 4. 移动文件到 doc 文件夹
mv note.txt work/doc/

# 5. 查看结果（预期输出：note.txt）
ls work/doc/
```

**场景 2：查询系统资源并结束进程**

部署 vLLM 后，你需要监控显存/内存，并管理 python 推理进程。跟着做：

```bash
# 1. 查看内存使用情况
free -h

# 2. 查找 python 进程

## 块 28

### 6.7 新手避坑指南（针对 vLLM 部署实操）

你装好环境后，终端操作最容易踩这五个坑，直接关系到你能不能顺利跑起 vLLM：

**1. 命令区分大小写**：Linux 严格区分大小写。`ls` 正确，`LS` 直接报错。敲命令时手指别抖。

**2. 路径写错**：切换目录或复制文件时，路径不对会提示 `No such file or directory`。别急着重敲，先用 `ls` 看看当前目录下到底有什么，确认路径存在再操作。

**3. `rm` 命令慎用**：`rm -rf` 是删库级别的危险命令。删任何东西前，先用 `ls` 确认你指向的是目标文件，千万别手滑删到根目录 `/`。

**4. 权限不足**：操作系统文件或别人的文件时，会提示 `Permission denied`。在命令前加 `sudo` 获取临时管理员权限（需输入当前用户密码）。比如装 Python 包时经常用到

## 块 29

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的RTX 4090已就绪，但vLLM跑在Python里。先确认地基：

```bash
python3 --version   # 预期输出：Python 3.12.x
pip3 --version
```

Ubuntu 24.04自带Python 3.12，通常直接可用。若`pip3`提示未找到，执行：

```bash
sudo apt update
sudo apt install -y python3-pip
```

**pip是什么？** Python的包管理器，相当于“应用商店”。后续装PyTorch、Transformers全靠它

## 块 30

### 7.2 安装 Miniconda：为 vLLM 部署铺路

你的目标是 vLLM 部署，而 vLLM 依赖 Python 环境。系统自带的 Python 虽够用，但后续管理多个项目时，**Miniconda** 能帮你隔离环境，避免依赖冲突。这一步先做最简安装，详细配置后续再讲。

**操作步骤：**

1. **下载安装脚本**（若提示 `wget` 未找到，先执行 `sudo apt install wget`）：
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **运行脚本**：
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. **安装过程**：按 `Enter` 阅读协议 → 输入 `yes` → 默认路径直接回车 → 提示更新 shell profile 时输入 `yes`。

4. **激活环境**：关闭重开终端，或执行 `source ~/.bashrc`。

5. **验证**：
   ```bash
   conda --version
   ```

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 31

## 7.3 运行你的第一个 Python 程序

你的目标是 vLLM 部署，而 vLLM 是一个 Python 工具。所以，在动手部署之前，先确认你的环境能跑 Python，这是最基础的一步。下面我们直接上手。

### 操作步骤

打开你的终端（WSL2 或 SSH 连接云服务器均可），依次执行：

**第一步：创建 Python 文件**

```bash
# 回到主目录
cd ~

# 创建一个叫 hello.py 的文件，用 nano 编辑器打开
nano hello.py
```

**第二步：输入代码**

在 nano 编辑器里输入以下内容：

```python
print("Hello, Python on Ubuntu 24.04!")
```

**第三步：保存并退出**

1. 按 `Ctrl + O` 保存，然后按回车确认文件名
2. 按 `Ctrl + X` 退出 nano

**第四步：运行程序**

```bash
python3 hello.py
```

如果屏幕输出了 `Hello, Python on Ubuntu 24.04!`，恭喜！你的环境已经就绪。你已经成功使用了 `cd`、`nano`、`python3` 三个终端命令，为后续安装 PyTorch 和 vLLM 打下了基础。

### 常见问题速查

| 症状 | 可能原因 | 解决办法 |
|---|---|---|
| `sudo apt update` 很慢 | 默认软件源在国外 | 搜"Ubuntu 24.04 换阿里云源"，按教程换成国内源 |
| 提示 `command not found: ip` | iproute2 未安装 | 执行 `sudo apt install iproute2` |
| 终端里中文显示乱码 | 字符编码问题 | 输入 `export LANG=en_US.UTF-8` 回车 |

### 本块学习检查

完成上面的操作后，请用自己的话回答：**运行第一个 Python 程序这个步骤，和你最终要完成的 vLLM 部署目标之间是什么关系？** 如果你在操作过程中遇到

## 块 32

## 2.1 pwd：打印当前工作目录

`pwd` = **P**rint **W**orking **D**irectory，作用是告诉你“我现在在哪个文件夹里”。

```bash
pwd
```

示例输出：

```bash
/home/zhangsan
```

这表示你当前位于 `/home/zhangsan` 目录下。

**为什么这对你部署 vLLM 很关键？**

你之后要下载模型权重、配置虚拟环境、启动推理服务，每一步都涉及“在哪个目录下执行命令”。如果搞不清当前位置，就会出现“明明下载了模型，却找不到文件”的窘境。所以，**操作前先敲 `pwd`**，确认自己站在正确的位置，是避免路径混乱的第一道防线。

**实操建议（结合你的 RTX 4090 环境）：**

1. 打开终端，先敲 `pwd`，确认初始位置（通常是 `/home/你的用户名`）。
2. 记住这个输出路径——后续创建 vLLM 项目文件夹、存放模型时，你都会以它为起点。
3. 每次切换目录后，如果不确定自己在哪，就再敲一次 `pwd`。

养成这个习惯，能让你在后续的模型部署操作中少走很多弯路。

---

**本块学习检查**

完成本块后，请用自己的话说明：`pwd` 命令的作用是什么？它和你“部署 vLLM”这个目标有什么直接关系？如果遇到无法解释的术语（比如“工作目录”），请记录下来，在提问时一并提出。

## 块 33

### 2.2 绝对路径 vs 相对路径

你部署 vLLM 时，经常要定位模型文件或配置文件。路径搞错，命令就找不到东西。

**绝对路径**：从根目录 `/` 开始写，不依赖你当前在哪。比如 `/home/zhangsan/Documents`，任何时候用都有效。

**相对路径**：相对于你当前所在位置。比如 `./file.txt` 表示当前目录下的文件，`../backup` 表示上一级目录里的 backup。

两个关键符号：
- `./`：当前目录（通常可省略）
- `../`：上一级目录
- `~`：当前用户的家目录（如 `/home/zhangsan`）

**举例**：假设你在 `/home/zhangsan/Documents` 下：
- `cd /home/zhangsan`：绝对路径，从哪执行都行
- `cd ..`：相对路径，回到上一级（即 `/home/zhangsan`）
- `cd ~`：直接跳到家目录

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 34

### 3.1 基础用法

`ls` 是你部署 vLLM 时最常用的“查看”命令。先掌握这几个变体：

```bash
# 列出当前目录内容（不含隐藏文件）
ls

# 列出指定路径，比如你下载的模型目录
ls /home/zhangsan

# 显示所有文件，包括隐藏的（以 . 开头）
ls -a

# 详细列表：权限、所有者、大小、修改时间
ls -l

# 组合：详细 + 所有 + 人类可读大小
ls -ahl
```

**关键点**：以 `.` 开头的文件默认隐藏，如 `.bashrc`、`.ssh`。配置 vLLM 环境变量时你会频繁遇到它们，务必用 `ls -a` 查看。

### 3.2 解读 ls -l 输出

执行 `ls -l` 后，输出类似：

```bash
drwxr-xr-x 2 zhangsan zhangsan 4096 Aug  3 10:00 Documents
-rw-r--r-- 1 zhangsan zhangsan  220 Aug  3 09:30 hello.py
```

**第一列第一个字符**

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 35

## 3.3 其他常用选项（①）

你部署 vLLM 时，经常要确认模型权重文件大小、查看日志文件时间——下面这些 `ls` 选项就是干这个的。

**核心组合：`ls -lh`**
`-h` 必须搭配 `-l` 才有意义，单独用 `ls -h` 没效果。输出格式为：详细列表 + 人类可读大小（KB/MB/GB）。比如你下载完模型后，用这个命令一眼就能看出文件多大。

**按时间/大小排序**
- `ls -t`：按最后修改时间排序，最新的在最前面
- `ls -S`：按文件大小从大到小排
- `ls -rt

## 块 36

### 4.1 基础用法（①+②）

既然你的目标是 vLLM 部署，那路径切换就是基本功——模型文件、代码仓库、虚拟环境，全都要靠 `cd` 定位。记住一个原则：**绝对路径从根目录 `/` 写起，任何位置都能用；相对路径从当前位置出发，省事但容易迷路**。

```bash
# 进入指定目录（绝对路径，部署时最常用）
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
# 方法1：绝对路径（去哪都管用，推荐新手）
cd /home/zhangsan

# 方法2：相对路径（先回根目录，再去 home）
cd ../../home/zhangsan
```

> **建议**：新手优先使用绝对路径，不容易迷路；熟悉后多用相对路径，效率更高。

---

## 五、mkdir：创建目录（①+②）

`mkdir` = **M**a**k**e **Dir**ectory，用来创建文件夹。部署 vLLM 时，你通常需要建项目目录、模型缓存目录、日志目录，这个命令会频繁用到。

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

> **注意**：`mkdir` 默认**不能**直接创建多级嵌套目录，必须加 `-p` 参数。这是新手最常踩的坑

## 块 37

## 7.1 复制文件：把模型脚本备份起来

你在部署 vLLM 时，经常要改 `model.py` 或配置文件。改之前先备份，是工程上的好习惯。

```bash
# 复制文件到 backup 文件夹（文件名不变）
cp model.py ./backup/

# 复制并改名，方便区分版本
cp model.py ./backup/model_v2.py

# 一次复制多个文件
cp config.json tokenizer.json ./backup/

# 保留文件属性（如修改时间），方便对比版本
cp -p model.py ./backup/

# 如果目标已存在，复制前先询问，防止误覆盖
cp -i model.py ./backup/
```

## 7.2 复制目录：整个项目打包备份

复制文件夹**必须加 `-r`**（递归），否则系统拒绝执行。

```bash
# 复制整个项目目录
cp -r myproject ./backup/
```

注意：如果 `backup` 不存在，会新建并复制内容进去；如果已存在，则生成 `backup/myproject/` 嵌套目录。若需保留所有属性（权限、时间戳等），可使用 `cp -a myproject ./backup/`。

## 7.3 复制软链接

```bash
#

## 块 38

## 8.1 移动文件/文件夹

部署 vLLM 时，你常需要把模型权重、配置文件挪到项目目录下。`mv` 就是干这个的。

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

# 移动的同时改名（比如把旧版模型文件改名归档）
mv hello.py ./backup/hello_v2.py
```

## 8.3 安全选项

```bash
# 覆盖前询问（推荐日常使用，防止误覆盖模型文件）
mv -i old.txt new.txt

# 强制覆盖，不提示（慎用）
mv -f old.txt new.txt
```

## 九、rm：删除文件与目录

`rm` = **R**e**m**ove，删除文件或文件夹。**此命令不可逆，请谨慎使用！**

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

#

## 块 39

## 场景 1：搭建一个项目目录结构（为 vLLM 部署做准备）

你正在为 vLLM 部署打基础，这个场景帮你建立项目骨架，后续模型文件、训练脚本都有清晰归属。

**操作步骤**：

```bash
# 1. 回到家目录
cd ~

# 2. 创建项目根目录
mkdir llm-learning

# 3. 进入项目目录
cd llm-learning

# 4. 创建三个子目录
mkdir data models scripts

# 5. 创建空文件
touch scripts/train.py

# 6. 验证结构
ls -R
```

**预期输出**：
```bash
llm-learning/
├── data
├── models
├── scripts
│

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 40

## 场景 2：复制与整理实验文件

你正在准备 vLLM 部署实验，需要先整理数据集。假设 `~/Downloads/` 下已有 `train.csv`、`test.csv`、`dev.csv` 三个文件，目标是把它们复制到 `~/llm-learning/data/` 中。

**操作步骤：**

1. **确认文件存在**：`ls ~/Downloads/*.csv` — 用通配符 `*.csv` 一次列出所有 CSV 文件，确认源文件就位。
2. **复制文件**：`cp ~/Downloads/train.csv ~/Downloads/test.csv ~/Downloads/dev.csv ~/llm-learning/data/` — `cp` 后跟多个源文件，

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 41

## 场景 3：清理临时文件（实战演练）

在 `~/llm-learning/` 下，你误建了一个空文件夹 `temp` 和一个 `old.txt` 文件。现在要把它们清理掉——这正好对应你之后部署 vLLM 时，清理下载的临时模型文件或缓存的操作。

**操作步骤：**

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

第 1 步的 `ls` 是删除前的确认，避免误删。第 3 步的 `rmdir` 专门用于删除空目录，如果目录里有内容会报错——此时需要用 `rm -r`。

---

## 新手避坑指南（重点摘录）

**1. 区分文件和文件夹**：`ls -l` 第一列是 `d` 表示文件夹，是 `-` 表示文件。对文件夹操作别忘了 `-r`（如 `cp -r`、`rm -r`）。

**2. 路径写错**：切换目录或复制文件时，路径写错会提示 "No such file or directory"，可先用 `pwd` 和 `ls` 确认当前位置和目录内容。

**3. 路径中包含空格必须用引号包裹**：命令行会把空格当成多个参数。务必用双引号包裹：

```bash
# 错误：mkdir my folder 会创建两个文件夹 my 和 folder
# 正确：
mkdir "my folder"
cd "my folder"
touch "my file.txt"
cp "my file.txt" ./backup/
rm "my file.txt"
```

**4. 文件名以 `-` 开头时使用 `--`**：`-` 开头通常表示选项，命令会误把文件名当选项解析。用 `--` 告诉命令"后面都是文件名"：

```bash
# 错误：rm -file.txt 会把 -file.txt 当成选项，报错
# 正确：

## 块 42

## 环境变量是什么？——先搞懂这张“系统备忘录”

你学 vLLM 部署，马上会遇到一个坎：装好 CUDA、配好模型路径后，终端却提示 `command not found`。十有八九，问题就出在环境变量上。所以这一节，我们先把“环境变量”这个概念彻底搞懂。

### 1.1 通俗理解

环境变量，就是操作系统或应用程序用于存储配置信息的键值对，可在进程间传递，并影响程序行为。

你可以把它想象成操作系统的**“备忘录”**：系统启动时加载这些备忘录，所有程序都能读到里面的内容，并据此调整自己的行为。

举个例子：你在终端输入 `python3`，系统怎么知道去哪找这个程序？靠的就是 `PATH` 这个变量——它告诉系统“去哪些目录找可执行文件”。

**常见环境变量一览**（这些你以后都会碰到）：

| 变量名 | 作用 |
|---|---|
| `PATH` | 告诉系统去哪里找可执行程序（比如 `python3`、`nvcc` 都靠它定位） |
| `HOME` | 当前用户的家目录路径（如 `/home/zhangsan`） |
| `USER` | 当前登录的用户名 |
| `SHELL` | 当前使用的 Shell 程序路径（如 `/bin/bash`） |
| `LANG` | 系统语言和字符编码（如 `en_US.UTF-8`） |
| `PWD` | 当前工作目录路径，由shell维护，cd命令会改变它 |
| `EDITOR` | 默认文本编辑器（影响 `git commit` 等操作） |

### 1.2 为什么需要环境变量？

1. **控制程序行为**：比如通过 `JAVA_HOME` 指定 JDK 路径，让依赖 Java 的程序知道去哪找它。对你来说，部署 vLLM 时

## 块 43

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

临时变量只在当前终端会话有效——关闭终端或断开SSH连接后即消失，适合快速测试。在vLLM部署中，临时变量常用于临时指定模型路径或测试参数，重启后不残留。

```bash
# 设置临时变量
export MY_VAR="hello_linux"
echo $MY_VAR   # 输出：hello_linux

# 删除变量
unset MY_VAR
echo $MY_VAR   # 输出：（空）
```

**`export` 的作用**：不加`export`，变量仅当前Shell可见

## 块 44

**永久变量：写入配置文件，持久生效**

临时变量只在当前终端有效，关掉就没了。想让每次打开终端都自动生效，就得把变量写进配置文件。

**选哪个文件？**

| 级别 | 配置文件 | 影响范围 | 需要sudo？ |
|---|---|---|---|
| 用户级（推荐） | `~/.bashrc` | 仅当前用户 | 否 |
| 用户级 | `~/.bash_profile` / `~/.profile` | 仅当前用户（登录时生效） | 否 |
| 系统级 | `/etc/profile` | 所有用户 | 是 |

日常开发优先用 `~/.bashrc`，只影响你自己，改错不伤系统。系统级配置要 `sudo` 且影响所有人，不建议动。

**本块学习检查**

## 块 45

## 3.1 用户级配置：修改 `~/.bashrc`（推荐）

`~/.bashrc` 是 Bash Shell 的配置文件，**每次打开新的交互式终端时自动加载**。这是日常开发中最常用的配置位置。

**操作步骤**：

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

> **`source` 是什么？** 相当于"重新执行一遍这个配置文件"，让刚才的修改在当前终端立刻生效，不用关掉重开。

> **`~/.bashrc` 的加载限制**：`~/.bashrc` 通常在打开新的**交互式终端**时加载，但并非所有 SSH 会话、IDE 集成终端或服务进程都会自动加载它。例如，某些 IDE 的内置终端可能只加载 `~/.profile`，非交互式 Shell（如自动化脚本）则通常不会

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 46

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**3.2 用户级配置：~/.bash_profile 或 ~/.profile**

这两个文件在**登录 Shell** 时生效（如 SSH 远程登录、系统开机登录）。若两者同时存在，系统优先读取 `~/.bash_profile`。

日常开发中 `~/.bashrc` 更常用，因为每开一个终端都会加载。只有需要"登录时才生效"的特殊场景，才考虑这两个文件。

---

**四、环境变量的优先级**（从高到低）

1. **临时变量**（当前终端里 `export` 的）
2. **用户级配置**（`~/.bashrc` 等）
3. **系统级配置**（`/etc/profile`

## 块 47

## 5.1 PATH：命令的“搜索路径”

你要部署 vLLM，马上就会遇到一个场景：装完 vLLM 后，终端输入 `vllm serve` 却提示 `command not found`。这背后就是 PATH 在起作用——它是系统查找可执行文件的目录列表，是最重要的环境变量。

**查看当前 PATH：**

```bash
echo $PATH
```

输出类似这样，多个路径用冒号 `:` 分隔：

```bash
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

当你在终端输入 `python3` 时，系统会**从左到右**依次在这些目录里找名为 `python3` 的可执行文件，找到就执行，找不到就提示 `command not found`。

**查看命令具体位置：**

```bash
which python3
# 输出：/usr/bin/python3

which conda
# 输出：/home/zhangsan/miniconda3/bin/conda
```

`which` 能告诉你某个命令到底是从 PATH 的哪个目录里找到的。部署 vLLM 时，如果 `which vllm` 没输出，说明它的安装目录不在 PATH 里。

**修改 PATH 有两种方式：**

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
> **避免重复添加**：多次 `source ~/.bashrc`

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 48

**LD_LIBRARY_PATH：让程序找到“零件”的路径**

你在部署 vLLM 时，会安装 CUDA、PyTorch 等组件，它们运行时需要加载动态库（`.so` 文件）。系统默认去 `/usr/lib`、`/lib` 找，但 CUDA 的库在 `/usr/local/cuda/lib64`，不在默认范围。`LD_LIBRARY_PATH` 就是告诉系统“额外去这些目录找”。

**操作步骤**：

1. 查看当前值：
```bash
echo $LD_LIBRARY_PATH
```

2. 临时添加 CUDA 库路径（仅当前终端有效）：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 49

## 5.3 LIBRARY_PATH：编译时库文件路径

你在部署 vLLM 时，经常需要从源码编译或安装带 CUDA 扩展的库。`LIBRARY_PATH` 就是编译器在**编译阶段**查找库文件的路径，影响静态库（`.a`）和动态库（`.so`）的链接。

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/your/libraries
```

**三个环境变量速查**：

| 变量名 | 作用阶段 | 查找目标 | 典型用途 |
|---|---|---|---|
| `PATH` | 执行命令时 | 可执行文件 | 让命令在任何位置都能运行 |
| `LD_LIBRARY_PATH` | 程序运行时 | 动态链接库（`.so`） | 运行依赖自定义动态库的程序 |
| `LIBRARY_PATH` | 编译链接时 | 静态库/动态库 | 编译时链接第三方库 |

> **实操建议**：你后续安装 CUDA、PyTorch 时，`PATH` 和 `LD_LIBRARY_PATH` 最常用，务必正确配置。

---

## 六、实战：配置 CUDA 环境变量

CUDA

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 50

## 6.1 先确认 CUDA 是否已在环境变量中

在终端执行下面两条命令，检查 CUDA 路径是否已生效：

```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

- **有输出**（例如 `/usr/local/cuda/bin`）：说明已配置好，跳过本节。
- **无输出**：说明还没配置，继续往下做。

## 6.2 把 CUDA 路径写入 ~/.bashrc

打开配置文件：

```bash
nano ~/.bashrc
```

在文件**末尾**追加这两行：

```bash
# CUDA 环境变量配置
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

> **注意**：`/usr/local/cuda` 是 CUDA 默认安装路径。如果你的 CUDA 装在别处，把路径换成实际的。

## 6.3 让配置生效并验证

```bash
# 重新加载配置
source ~/.bashrc

# 再查一次 PATH
echo $PATH | grep cuda

# 检查 nvcc 编译器
nvcc --version
```

能输出 CUDA 版本信息（如 `nvcc: NVIDIA (

## 块 51

## 调试与验证：让环境变量真正为你所用

你正在为 vLLM 部署铺路，环境变量是这条路上的“路标”——配置错了，模型加载就会迷路。这一块教你如何“体检”这些路标。

**查看所有变量**：终端输入 `printenv` 或 `env`，前者简洁，后者更详细。这是快速检查全局配置的起点。

**查看特定变量**：想知道某个变量（比如 `PATH`）的具体值，用 `echo $PATH` 或 `printenv PATH`。想确认它是否被导出（export），用 `declare -p PATH` 查看定义细节。

**测试变量是否生效**：假设你在 `~/.bashrc` 里写了 `export MY_VAR="hello"`，先执行 `source ~/.bashrc` 让配置立即加载，再 `echo $MY_VAR`，应输出 `hello`

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 52

### 变量未生效？三步排查法

你在部署 vLLM 时，如果改了配置却不起作用，别急，按下面三步来：

**第一步：确认语法**
```bash
cat ~/.bashrc | grep "你的变量名"
```
看看输出里有没有 `export` 开头的那行。如果少了 `export`，变量根本不会导出。

**第二步：重新加载**
```bash
source ~/.bashrc
```
改完配置必须重新加载，或者重启终端，否则当前会话不会读到新值。

**第三步：查重复定义**
```bash
grep "PATH=" ~/.bashrc
```
如果发现变量被定义了多次，后面的会覆盖前面的。检查一下有没有冲突。

**注意**：如果你用的是 IDE（比如 PyCharm）或某些终端，它们可能不加载 `~/.bashrc`，这时需要手动 `source` 或重启 IDE。

**模型部署补充**：若加载 Llama-2-7b 模型，请确认其许可证（Llama 2 Community License，商用月

## 块 53

## 脚本中临时修改变量：为 vLLM 部署铺路

你在准备 vLLM 部署，核心是让模型推理跑得稳、跑得快。环境变量是控制运行时行为的关键——比如指定 GPU 显存、模型路径或日志级别。这一块教你三种临时修改变量的方法，避免污染全局环境。

### 方法一：export 临时生效

```bash
#!/bin/bash
export TEMP_VAR=123
# 仅在脚本执行期间生效，脚本结束后变量消失
```

**实操要点**：把 `export` 写在脚本里，变量只在脚本运行时存在。适合在部署脚本中临时设置推理参数，脚本跑完自动清理，不影响你后续的其他操作。

### 方法二：env 单次运行

```bash
env MY_VAR=hello python3 my_script.py
# MY_VAR 仅在这次 python3 运行时有效
```

**实操要点**：不用改脚本内容，直接在命令行前缀加 `env 变量=值`。比如启动 vLLM 服务时，临时指定 `CUDA_VISIBLE_DEVICES=0` 只让程序看到第一张卡。这条命令跑完，变量立刻消失。

### 方法三：.env 文件分组管理

项目变量多了，写进 `.env` 文件统一管理：

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

加载方法

## 块 54

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你正在为 vLLM 部署铺路。部署时经常要指定模型路径、缓存目录等，这些值如果写死在命令里，换环境就得改一遍。本块教你用环境变量把这些值“存”起来，一次设置、处处引用。

**场景 1：创建并验证自定义变量**

先临时设一个变量，`export` 表示“导出到当前终端环境”：

```bash
export MY_NAME="zhangsan"
```

验证一下，`$` 是取值符号：

```bash
echo $MY_NAME
```

但 `export` 只对当前终端有效，关掉就没了。要永久生效，把它追加到 `~/.bashrc`（每次打开终端都会自动加载这个文件）：

```bash
echo 'export MY_NAME="zhangsan"' >> ~/.bashrc
```

最后重新加载配置并验证：

```bash
source ~/.bashrc
echo $MY_NAME
```

**场景 2：为自定义脚本目录配置 PATH**

你打算写脚本辅助 vLLM 部署（比如一键启动服务）。把脚本放在 `~/scripts`，并让系统在任何目录都能直接

## 块 55

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的目标是vLLM部署，而vLLM依赖CUDA环境。即使现在没装CUDA，也可以先练熟配置方法，为后续部署扫清障碍。

**实操步骤**（在你的RTX 4090上同样适用）：

1. 打开终端，编辑配置文件：
   ```bash
   nano ~/.bashrc
   ```

2. 在文件末尾添加两行（假设CUDA装在`/usr/local/cuda`）：
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda

## 块 56

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

系统级配置影响所有用户，改需 `sudo`。日常开发少用，但配公共服务器要懂。

**A.1 /etc/environment**：不用 `export`，直接 `KEY=value`，仅支持简单键值对，不支持在赋值中引用其他变量（如 `PATH=$PATH:/foo`），但支持简单的变量展开。改完重启或重登生效。

```bash
sudo nano /etc/environment
# 加：JAVA_HOME=/usr/lib/jvm/java-11-openjdk
```

**A.2 /etc/profile**：全局 shell 配置文件（适用于所有 POSIX shell），需 `export`。改完 `source` 或重启生效。

```bash
sudo nano /etc/profile
# 加：export PATH=$PATH:/usr/local/custom/bin
source /etc/profile
```

**练习**：在 `/etc/environment` 加 `TEST

## 块 57

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

**B. 不同 Shell 的配置文件对照表（①）**

你平时敲命令的终端，用的可能是 Bash，也可能是 Zsh 或 Fish。它们读的配置文件不一样，改错地方就不生效。

| Shell 类型 | 配置文件 | 作用范围 |
|---|---|---|
| **Bash** | `~/.bashrc` | 非登录 Shell（日常终端） |
| **Bash** | `~/.bash_profile` | 登录 Shell（SSH 登录） |
| **Zsh** | `~/.zshrc` |

## 块 58

## C. 清理 PATH 中的重复项（④）

多次修改 PATH 后，重复路径会拖慢命令查找速度，也可能让不同版本的软件互相干扰。用下面这条命令一键去重：

```bash
export PATH=$(echo $PATH | awk -v RS=':' '!a[$0]++' | paste -sd: -)
```

这条命令的原理：把 PATH 按冒号拆成一行一个路径，用 `awk` 去重（`!a[$0]++` 只保留第一次出现的路径），最后再用 `paste` 把结果拼回冒号分隔的格式。**注意**：这只是临时清理，写入 `~/.bashrc` 才能永久生效。

## D. 根据环境加载不同配置（④）

部署 vLLM 时，开发环境和生产环境的依赖版本、模型路径往往不同。可以在 `~/.bashrc` 里加一段条件判断：

```bash
# 在 ~/.bashrc 中添加
if [ "$ENVIRONMENT" = "production" ]; then
    source ~/.env.prod
else
    source ~/.env.dev
fi
```

这样，通过设置 `ENVIRONMENT` 变量（比如 `export ENVIRONMENT=production`），就能自动加载对应的环境配置，避免手动切换。

## E. 脚本调试模式（④）

配置脚本报错时，用调试模式快速定位问题：

```bash
# 以详细模式运行脚本，会打印每一行执行前的变量值
bash -x your_script.sh

# 或者在当前 Shell 中开启追踪
set -o xtrace
source ~/.bashrc
set +o xtrace
```

`bash -x` 适合调试独立脚本；`set -o xtrace` 适合调试当前 Shell 的配置加载过程，排查 `~/.bashrc` 哪一行出了问题。

---

## 小结与下一步

你已掌握环境变量配置的核心技能：概念、临时/永久配置、`source` 生效、PATH 相关

## 测评题目

### Q-KB0100-001
请在终端中执行以下命令，验证Linux/WSL是否安装成功，将完整输出粘贴作答：
```bash
uname -a
```

### Q-KB0100-002
执行以下命令验证当前用户和主机名：
```bash
whoami && hostname
```

### Q-KB0101-004
`cp -r source_dir/ dest_dir/` 中 `-r` 参数的含义是？

### Q-KB0101-007
请写出一条管道命令，统计当前目录下 `.py` 文件的数量。

### Q-KB0102-001
临时设置环境变量 `MY_VAR=hello`，命令为________；若要使其在当前shell及子进程中生效，需使用________命令。

### Q-KB0102-003
以下哪个文件是bash shell用户登录时**最先**读取的全局配置文件？

### Q-KB0102-004
在 `~/.bashrc` 中写入 `export LD_LIBRARY_PATH=/usr/local/lib` 后，新开一个终端窗口，该变量是否自动生效？为什么？

### Q-KB0101-008
请写出一条命令，查找 `/var/log` 目录下所有大于100MB的文件，并按大小排序显示。
