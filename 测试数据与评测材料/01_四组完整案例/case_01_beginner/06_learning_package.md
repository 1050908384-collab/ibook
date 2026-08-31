# 学习包

## 块 1

# 环境准备：根据 CUDA 版本安装 PyTorch

本节的目标很明确：在你的机器上成功安装 PyTorch（GPU 版或 CPU 版），并验证 `torch.cuda.is_available()` 返回预期结果，为后续张量操作、自动求导和神经网络训练做好准备。

在开始之前，先快速回顾一下前置环境配置。你已经具备 Python 编程基础，且前面章节已经详细学过环境搭建，这里只做**快速确认**，不再重复教学：

- **Python 3.10+**：确认已安装且版本满足要求。
- **conda 虚拟环境**：确认已创建并激活你的工作环境（例如 `conda activate your_env_name`）。
- **IDE**：确认 PyCharm 或 VS Code 已安装并配置好解释器。
- **pip 国内镜像源**：确认已配置，以加速后续依赖包的下载。

如果以上某一步报错，请回到对应章节排查。确认无误后，我们继续。


## 一、前置检查：确认前面章节的配置（不重复教学）

以下步骤在前面板块已经详细学过，本节只做**快速确认**。如果某一步报错，请回到对应章节排查。

**核心检查项**：

1. **NVIDIA 驱动与 CUDA 版本**：在终端运行 `nvidia-smi`，确认驱动已安装且能看到 CUDA 版本号。这个版本号决定了你后续安装 PyTorch 时选择的 CUDA 版本（例如 CUDA 12.1）。
2. **conda 虚拟环境**：确认已创建并激活你的工作环境（例如 `conda activate your_env_name`）。
3. **Python 版本**：在激活的环境中运行 `python --version`，确认是 3.10 或更高版本。

> **概念对比**：`nvidia-smi` 显示的 CUDA 版本是驱动支持的 CUDA 运行时版本，安装 PyTorch 时选择不高于该版本的 CUDA 版本即可，例如驱动支持 CUDA 12.1，就可以安装 `cu121` 版本的 PyTorch。

**设计动机**：为什么需要先确认 CUDA 版本？因为 PyTorch 的 GPU 版是针对特定 CUDA 版本预编译的。如果 PyTorch 的 CUDA 版本高于驱动支持的版本，会导致 `torch.cuda.is_available()` 返回 `False`，GPU 无法被识别。这就是"版本对齐"问题的根源——四者（驱动、CUDA、cuDNN、PyTorch）必须相互兼容。

**思考题**：为什么 `nvidia-smi` 显示的 CUDA 版本高于 PyTorch 要求的 CUDA 版本时，通常可以正常工作？

> 答案线索：PyTorch 的 CUDA 版本是编译时使用的 CUDA 运行时版本，安装时需确保驱动支持的 CUDA 版本不低于 PyTorch 要求的版本，就能正常工作。

**本块学习检查**：先用一个最小示例回顾（环境准备：安装Python 3.10+，配置conda虚拟环境，安装IDE如PyCharm或VS Code，配置pip国内镜像源加速下载）；确认能解释关键写法后再继续本块。

## 块 2

### 1.1 确认 conda 可用（conda/pip 虚拟环境管理 章节已学过）

```bash
conda --version
# 预期输出：conda 24.x.x
```

### 1.2 确认 nvidia-smi 有输出（nvidia-smi 显存监控 章节已学过）

```bash
nvidia-smi
```

- 如果有正常输出（显示显卡型号、驱动版本、CUDA Version），跳到 **第二节**。
- 如果报错 `command not found`，说明当前环境**没有识别到 NVIDIA 驱动**。请先安装驱动（见下方 **1.3 节**），或选择安装 **CPU 版 PyTorch**。

**原理说明**：`nvidia-smi` 是 NVIDIA 驱动自带的命令行工具，其输出中的 CUDA Version 表示当前驱动所支持的最高 CUDA 运行时版本。PyTorch 的 GPU 加速依赖 CUDA 运行时与驱动之间的版本兼容，因此先确认驱动可见性是安装 GPU 版 PyTorch 的前提。若该命令报错，说明驱动未正确安装或未加入 PATH，此时安装 GPU 版 PyTorch 将无法调用 GPU 计算资源，应优先解决驱动问题或改用 CPU 版。

## 块 3

### 1.3 安装 NVIDIA 显卡驱动（如需要）

必要性取决于环境，判断标准：`nvidia-smi` 能否正常输出。

- **云 GPU 实例**：通常已预装，跳过。
- **本地物理机**：若 `nvidia-smi` 报错，需手动安装。
- **WSL2**：Windows 驱动自动穿透，无需单独装；但需安装匹配的 CUDA 工具包，并确保 Windows 驱动支持 WSL2。

**原理：驱动、CUDA 与 `nvidia-smi` 的关系**

三层概念：**显卡驱动**是 OS 与 GPU 的通信层；**CUDA 工具包**是并行计算库，依赖驱动接口；`nvidia-smi` 是驱动自带管理接口，其输出反映驱动加载状态。链条为“驱动为基、CUDA 为用、`nvidia-smi` 为验”——驱动缺失时 CUDA 无法调用 GPU，`nvidia-smi` 报错；驱动就绪后，`nvidia-smi` 正常即底层通信打通。

**Linux 手动安装步骤**：

1. 访问 NVIDIA 官方下载页，选型号和 Linux 64-bit，下载 `.run` 文件。
2. 按官方或发行版指南安装。**关键**：勿直接执行 `.run`；先进入文本模式（如 `init 3`），禁用 nouveau，加 `--no-opengl-files` 参数防冲突。步骤因发行版而异，操作不当可能无法进图形界面，务必查文档。

**替代方案：Ubuntu 附加驱动工具**

```bash
ubuntu-drivers devices          # 查看推荐驱动
sudo ubuntu-drivers autoinstall # 自动安装
```

重启后运行 `nvidia-smi` 验证。

**对比**：`nvidia-smi` 报错=驱动未装或未加载；正常=驱动就绪，可继续 CUDA 与 PyTorch 配置。

## 块 4

## 二、先搞清楚：显卡、驱动、CUDA、PyTorch 四者是什么关系？（②）

这是安装 PyTorch 最容易踩坑的地方。很多人装完后运行 `torch.cuda.is_available()`，发现返回 `False`，然后对着一堆版本号不知道哪里错了。先把四者的依赖关系搞清楚，后面每一步都有据可依。（②）

```
显卡硬件（如 RTX 3060）
    ↓ 决定支持的驱动版本上限
NVIDIA 显卡驱动（Driver）
    ↓ nvidia-smi 右上角显示的 CUDA Version = 驱动支持的最高 CUDA 版本
CUDA Toolkit（nvcc --version，可选）
    ↓ PyTorch 自带 CUDA Runtime，一般不需要单独装
PyTorch（内置 CUDA Runtime）+ cuDNN
```

**最关键的两句话**：（②）

1. **`nvidia-smi` 右上角显示的 `CUDA Version`，是驱动支持的"最高"CUDA 版本，不是你已经安装的 CUDA 版本。** 你安装的 PyTorch 的 CUDA 版本必须 **≤** 这个值。换句话说，驱动是硬件与软件之间的"翻译层"，它决定了上层软件能用到多新的 CUDA 能力；而 PyTorch 自带的 CUDA Runtime 则是在这个上限之内实际运行的版本。两者是"上限"与"实际使用"的关系，不是同一回事。

2. **如果你只是跑 PyTorch 训练或推理，不需要单独安装 CUDA Toolkit。** PyTorch 的 wheel/conda 包里已经自带了 CUDA Runtime，装上就能用 GPU。这是 PyTorch 的一个关键设计选择：把 CUDA Runtime 打包进发行包，让用户免去手动配置底层环境的负担。代价是包体积变大，但换来的是"装上即用"的确定性——你不需要自己去匹配 Toolkit 版本与 PyTorch 版本的兼容性。

> 什么时候才需要单独装 CUDA Toolkit？只有你需要编译 CUDA 扩展（如 `flash-attn`、`bitsandbytes`）或使用 `nvcc` 编译自定义 CUDA 代码时。后续"参数高效微调"和"PyTorch 进阶训练技巧"章节涉及这些库时，我们会再补充安装步骤。（②）

这里有一个值得注意的区分：`nvidia-smi` 显示的 CUDA Version 与 `nvcc --version` 显示的 CUDA 版本，前者是驱动能力的上限，后者是 Toolkit 的安装版本。如果你没有单独装 Toolkit，`nvcc` 命令根本不存在——这并不影响 PyTorch 跑 GPU 训练，因为 PyTorch 用的是自己内置的 Runtime。

对你后续做 LoRA 微调而言，理解这条依赖链的意义在于：当你安装 PyTorch 时，选择 CUDA 版本号只需要看 `nvidia-smi` 右上角的值，确保 PyTorch 的 CUDA 版本 ≤ 那个数字即可。至于 `flash-attn`、`bitsandbytes` 这类需要编译的库，它们才是你将来可能需要单独装 Toolkit 的触发条件——到那一步再处理也不迟。

**思考题**：

1. 为什么说 `nvidia-smi` 右上角的 `CUDA Version` 是"驱动支持的最高版本"而不是"已安装的 CUDA 版本"？请用原文中的一句话回答。
2. 什么情况下才需要单独安装 CUDA Toolkit？请列出原文提到的两种场景。

## 块 5

### 三、查看驱动支持的 CUDA 版本上限

在安装 PyTorch 之前，你需要先确认一个关键约束：**你的 NVIDIA 驱动能支持到哪个 CUDA 版本**。这一步决定了你后续能安装哪个版本的 PyTorch，因为 PyTorch 的 CUDA 版本必须落在驱动支持的范围之内。

执行以下命令查看驱动信息：

```bash
nvidia-smi
```

输出会类似这样（以 RTX 3060 为例）：

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
+=========================================+======================+======================+
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8              15W / 170W|    256MiB / 12288MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

**重点关注右上角 `CUDA Version: 12.2`** ——这表示你的驱动最高支持 CUDA 12.2。那么你安装的 PyTorch 的 CUDA 版本不能超过 12.2。

这里的设计动机是：驱动是硬件与软件之间的桥梁，它决定了上层软件（如 PyTorch）能调用的 CUDA 能力上限。如果你安装的 PyTorch 要求更高的 CUDA 版本，驱动将无法提供对应的运行时支持，导致安装或运行失败。因此，**先查驱动上限，再选 PyTorch 版本**，是一个顺序上的硬约束。

> 如果 `nvidia-smi` 报错或你没有 NVIDIA 显卡，请直接跳到 **第五节（CPU 版安装）**。

## 块 6

### 四、创建 PyTorch 专用 conda 虚拟环境（①+③）

这一步在“conda/pip 虚拟环境管理”章节已详细学过，此处快速执行。

```bash
# 创建名为 pytorch 的虚拟环境，Python 版本 3.10
conda create -n pytorch python=3.10 -y

# 激活环境
conda activate pytorch
```

激活成功后，命令行提示符前面会从 `(base)` 变成 `(pytorch)`。（①+③）

**为什么选 Python 3.10？** PyTorch 2.x 对 Python 3.8~3.12 都有良好支持。3.10 是目前生态兼容性最好的版本，绝大多数深度学习库都已适配。（②）


**思考题**：为什么在创建虚拟环境时指定 Python 3.10，而不是直接使用系统默认版本？（提示：从生态兼容性角度回答）

**概念对比**：`conda create` 与 `conda activate` 的职责不同——前者负责创建独立环境并指定解释器版本，后者负责切换当前 shell 到该环境，二者缺一不可。

## 块 7

# 画像专属 · （环境准备：根据CUDA版本安装PyTorch（如torch 2.0+ with CUDA 11.8/12.1） · 本块学习卡
## 概念与设计动机：为什么需要（环境准备：根据CUDA版本安装PyTorch（如torch 2.0+ with CUDA 11.8/12.1）
### 本块必读锚点 · 概念卡片（阅读正文前先浏览，正文中遇到就填）
| 序号 | 原文锚点（术语 / 字段 / 加粗词） | 在正文中第几次出现时被你抓到？ | 你用 ≤30 字自证其含义（不得查外部资料） | 本锚点和其他锚点的关系？ |
|---|---|---|---|---|
| 1 | （后续在正文里自行补充） |  |  |  |
| 2 | （后续在正文里自行补充） |  |  |  |
| 3 | （后续在正文里自行补充） |  |  |  |
| 4 | （后续在正文里自行补充） |  |  |  |


> 🎯 **你的画像 = 理论型 learner**：本块**强制阅读顺序与节奏**（不要跳步）：
> 1. 先完整通读『概念说明段』和所有『列表 / 定义 / 对比条目』至少 2 遍，**不先看代码块**；
> 2. 列表条目部分标记为『要点卡片』，在上面的概念卡片里逐行填笔记；
> 3. 代码块仅做『对照原理阅读 · 句法分析』，不必立即在你机器上运行；
> 4. 全部通读完后，完成文末『画像专属 · 本块学习检查』和下面的『思考提示』。

### 思考提示（无需查资料，仅用本块正文作答；写不出来就回上文找原句）
> *说明：这里只列『思考题答题指引 + 引用原文锚点』，不给答案，答案由你读完正文后自己填写。*
1. **思考题 1 · 概念边界**：基于正文找出 2 条使用本知识点时的限制或前置条件（必须来自原文，不得脑补）。
2. **思考题 2 · 术语对照**：从上面「本块必读锚点」概念卡片中任选 2 个词，写出它们若被混淆会导致的后果（以原文中明确提到的差异为准）。
3. **思考题 3 · 流程/句法分析**：若本块有代码，回到代码块做句法分析；若无代码，则回到正文第一条列表条目做结构重写（不抄原句）。

> ⚙️ 实训条件：RTX 3060 12GB；本块所属知识板块：『深度学习框架PyTorch基础』。


## 五、安装 PyTorch（①+②+③）


## 画像专属 · （环境准备：根据CUDA版本安装PyTorch（如torch 2.0+ with CUDA 11.8/12.1） 本块学习检查
### 概念自证 · 思考题作答（答案必须能用正文原句回证）
1. **概念概括**：用你自己的话概括『（环境准备：根据CUDA版本安装PyTorch（如torch 2.0+ with CUDA 11.8/12.1）』在本块中被引入的核心原因，最多写 3 个要点，并分别回正文标出对应原句。

2. **适用边界说明**：根据本块正文，写出 2 条**不适合使用本块介绍的方法 / 命令**的场景（场景必须来自正文明确提到的限制或反例，不得凭空脑补）。

3. **流程拆解**：回到正文中出现的命令 / 流程段，用编号步骤写出『执行后系统内部依次做了什么』的最小流程；每一步必须能在正文里找到依据（用自己的话写，不要摘抄）。

4. **列表类条目结构默写**：回到正文里第一条『定义 / 对比 / 枚举』型列表，用你自己的话写出条目维度和关键限定词（不要摘抄）。


### 实训条件提示 · 你的设备
已记录你的实训条件：RTX 3060 12GB。示例会按本地 NVIDIA GPU 可执行的参数和验证方式组织，并保留与本知识点直接相关的最小步骤。

## 块 8

### 5.1 有 NVIDIA GPU：安装 GPU 版 PyTorch（②+③）

**核心原则**：PyTorch 的 CUDA 版本 ≤ `nvidia-smi` 显示的 CUDA Version。（②）

这条原则是整个安装过程的判断基准。`nvidia-smi` 显示的是你的显卡驱动所支持的最高 CUDA 版本，而 PyTorch 安装包内置的是它运行时需要的 CUDA 运行时库。只要前者大于等于后者，驱动就能正常承载 PyTorch 的计算请求。反过来如果 PyTorch 版本高于驱动支持的最高版本，就会出现"驱动太老带不动新库"的兼容性问题。

**推荐做法：去 PyTorch 官网复制安装命令**（最不容易出错）：

1. 打开 PyTorch 官方安装页面：https://pytorch.org/get-started/locally/
2. 按你的实际情况选择：
   - **PyTorch Build**：Stable（推荐）
   - **Your OS**：Linux
   - **Package**：Pip（推荐，版本更灵活）
   - **Language**：Python
   - **Compute Platform**：选择你驱动支持的 CUDA 版本（≤ `nvidia-smi` 的 CUDA Version）
3. 页面会自动生成安装命令，**直接复制粘贴到终端执行**。

这里的设计动机很直接：官网的交互式选择器会根据你选的系统、包管理器和 CUDA 版本，自动拼好一条带正确 `--index-url` 的 pip 命令。你不需要手动记忆 PyTorch 各版本与 CUDA 版本的对应关系，也不需要担心拼错 wheel 包的下载地址——页面生成的就是官方验证过的组合。相比手动改命令，这个方式把"版本匹配"这个最容易出错的环节交给了官方工具。

以 `nvidia-smi` 显示 `CUDA Version: 12.2` 为例，你可以选择 CUDA 12.1 版本的 PyTorch（因为 12.1 ≤ 12.2）：（②）

```bash
# CUDA 12.1 版本（适合驱动支持 CUDA 12.x 的用户）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

如果你的驱动支持 CUDA 11.8（较旧显卡），使用：（②）

```bash
# CUDA 11.8 版本（适合旧显卡或驱动较旧的用户）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

> **如何确定用 cu121 还是 cu118？**
> - `nvidia-smi` 显示 CUDA Version ≥ 12.1 → 用 `cu121`
> - `nvidia-smi` 显示 CUDA Version 在 11.8~12.0 之间 → 用 `cu118`
> - 不确定时，优先选 `cu121`，因为 PyTorch 2.x 对 CUDA 12.x 的支持更完善。（②）

> **需要安装历史版本？** 访问 https://pytorch.org/get-started/previous-versions/ 查看完整对照表。

**也可以用 conda 安装**（如果你更习惯 conda）：（①）

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    pytorch-cuda=12.1 -c pytorch -c nvidia
```

使用 conda 安装前，需先安装 Anaconda 或 Miniconda，并创建虚拟环境（如 `conda create -n pytorch_env python=3.10`），然后激活该环境再执行上述命令。

**概念对比：pip 方式 vs conda 方式**——两者安装的 PyTorch 版本和 CUDA 组件相同，区别在于依赖管理机制：pip 从 PyTorch 官方 wheel 源拉取，版本选择更灵活；conda 通过 `pytorch-cuda=12.1` 显式声明 CUDA 运行时版本，由 conda 统一管理依赖树，适合已经用 conda 管理 Python 环境的场景。选哪种取决于你现有的环境管理习惯，不影响 PyTorch 本身的运行效果。

**思考题**：
1. 为什么 `nvidia-smi` 显示 CUDA Version 12.2 时，可以选择 cu121 的 PyTorch，但不能选 cu130？
2. 判断用 cu121 还是 cu118 的两个判定条件分别是什么？

## 块 9

### 5.2 无 NVIDIA GPU：安装 CPU 版 PyTorch（②+③）

如果你的电脑没有 NVIDIA 显卡（如使用虚拟机、WSL2 无 GPU 穿透、或 Mac），安装 CPU 版即可正常学习 PyTorch 语法，只是训练时速度较慢。（②+③）

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

> **CPU 版能学完全部课程吗？** 可以。PyTorch 的 API（张量操作、自动求导、神经网络搭建）在 CPU 和 GPU 上几乎完全一致。后续涉及大模型微调时，CPU 版会跑得很慢甚至跑不动，但学习概念和调试代码完全没问题。（②）

**原理拆解**：这条命令通过 `--index-url` 指定 PyTorch 官方 CPU 版 wheel 源，避免默认源拉取带 CUDA 依赖的版本。CPU 版与 GPU 版共享同一套 API 设计，因此你后续写 LoRA 微调代码时，张量操作和自动求导的写法完全一致——区别只在于底层计算设备。

**概念对比**：CPU 版 vs GPU 版——训练速度差异显著，但 API 调用方式相同；CPU 版适合语法学习和代码调试，GPU 版适合实际微调。

**思考题**：
1. 为什么 CPU 版 PyTorch 能“学完全部课程”，但大模型微调时“跑得很慢甚至跑不动”？（提示：API 一致性与计算规模的关系）
2. 这条安装命令中 `--index-url` 参数的作用是什么？（提示：指定下载源）

## 块 10

### 5.3 安装过程说明

PyTorch 安装包体积较大（约 2~3GB），下载需要几分钟到十几分钟，取决于网络速度。安装过程中如果看到进度条在走，耐心等待即可。（①）

> **安装太慢？** 如果你在国内网络环境下速度很慢，建议先按本章节第三节的方法配置好全局 pip 国内镜像源，然后仅指定 PyTorch 官方 wheel 仓库即可：
> ```bash
> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
>     --index-url https://download.pytorch.org/whl/cu121
> ```
> 注意：为避免拉到 CPU 版或版本不匹配的包，建议仅使用官方仓库作为唯一源，或确保镜像源已同步对应 CUDA 版本。通用依赖（如 numpy）会自动走你已配置的全局镜像源）。（②）


**设计动机**：这条命令的核心是解决“国内网络下载慢”与“必须拿到 CUDA 版 PyTorch”这对矛盾。全局镜像源解决速度问题，`--index-url` 单独指向官方仓库则保证 PyTorch 本体来自唯一可信来源——因为 PyTorch 的 CUDA 版包只发布在官方仓库，混用多个源会让 pip 的依赖解析“迷路”，可能拉到 CPU 版。

**概念对比**：`--index-url` vs `-i` / `--extra-index-url`：前者是“唯一指定源”，pip 只从这一个源找包；后者是“追加源”，pip 会在多个源之间按优先级挑选，这正是导致版本错乱的根源。

**思考题**：
1. 为什么 CUDA 版 PyTorch 必须用 `--index-url` 指向官方仓库，而不能叠加 `-i` 指定其他镜像源？（提示：pip 在多源下会优先拉到什么版本？）
2. 配置全局镜像源后，通用依赖（如 numpy）和 PyTorch 本体分别从哪个源下载？这种“分流”设计解决了什么问题？

## 块 11

### 六、验证安装（①+②+③）

安装后必须验证PyTorch导入及GPU可用。核心逻辑两步：先确认导入成功，再确认设备可用。前者排除安装损坏或路径冲突，后者排除CUDA与驱动不匹配。

**思考题**：为何两者独立？只做前者可能漏掉GPU不可用但导入正常的问题。

## 块 12

### 6.1 快速验证脚本（②+③）

安装完成后，验证环节的核心目的不是“跑通命令”，而是确认两件事：**PyTorch 是否真的能调用你机器上的 GPU**，以及**它内置的 CUDA 版本与你安装的驱动是否匹配**。这一步直接关系到后续 LoRA 微调能否在 GPU 上加速——如果验证失败，后面所有训练都会默默退回 CPU 模式，速度慢几个数量级。

在终端里逐行执行：

```bash
python -c "
import torch
print('PyTorch 版本:', torch.__version__)
print('CUDA 是否可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('PyTorch 内置 CUDA 版本:', torch.version.cuda)
    print('GPU 数量:', torch.cuda.device_count())
    print('GPU 型号:', torch.cuda.get_device_name(0))
else:
    print('当前使用 CPU 模式')
"
```

这段脚本的逻辑很直白：先打印 PyTorch 版本，再检查 `torch.cuda.is_available()`——这是整个验证的**关键开关**。如果返回 `True`，说明 PyTorch 已经成功识别到你的 NVIDIA 驱动和 CUDA 运行时，此时继续打印内置 CUDA 版本、GPU 数量和型号；如果返回 `False`，则直接提示当前处于 CPU 模式。注意 `torch.version.cuda` 打印的是 **PyTorch 编译时内置的 CUDA 版本**，而不是你系统里单独安装的 CUDA 版本——这两者可能不同，但 PyTorch 只要能通过驱动调用 GPU 即可，不必强求两者完全一致。

**预期输出（GPU 版）**：（③）

```
PyTorch 版本: 2.5.1+cu121
CUDA 是否可用: True
PyTorch 内置 CUDA 版本: 12.1
GPU 数量: 1
GPU 型号: 例如：NVIDIA GeForce RTX 3060（具体型号取决于你的硬件）
```

**预期输出（CPU 版）**：

```
PyTorch 版本: 2.5.1+cpu
CUDA 是否可用: False
当前使用 CPU 模式
```

看到以上输出，说明 PyTorch 安装成功！（③）

**概念对比**：`torch.cuda.is_available()` vs `torch.version.cuda`——前者是**运行时探测**，告诉你当前环境能否实际使用 GPU；后者是**编译期常量**，告诉你 PyTorch 是针对哪个 CUDA 版本构建的。前者为 `True` 才是硬指标。从原理上讲，`torch.cuda.is_available()` 返回 `True` 的前提是 PyTorch 能够成功初始化与驱动匹配的 CUDA 运行时上下文；若驱动版本过旧或缺失，初始化失败，该函数即返回 `False`。而 `torch.version.cuda` 只是一个编译时写入的字符串常量，不反映运行时状态。

## 块 13

### 6.2 运行一个简单的张量运算（①）

这段验证的核心目的，是确认 PyTorch 的**张量（Tensor）**这一核心数据结构在 CPU 和 GPU 上都能正常工作。张量是 PyTorch 中所有运算的基本载体，可以理解为带自动求导能力的多维数组——你之后做 LoRA 微调时，模型参数、输入数据、梯度都是以张量形式存在并参与运算的。

命令本身很直接：先用 `torch.rand(2, 3)` 创建一个 2 行 3 列的随机张量，打印出来确认 CPU 路径没问题；然后通过 `torch.cuda.is_available()` 判断当前环境是否有可用的 GPU，若有则用 `.cuda()` 把张量搬到显存中，再做一次 `x_gpu * 2` 的乘法运算并打印结果。

这里值得注意的设计动机是：**为什么验证要分 CPU 和 GPU 两条路径？** 因为 PyTorch 的安装版本决定了它能否调用 CUDA。如果你装的是 CPU 版，`torch.cuda.is_available()` 会返回 `False`，代码自动走 CPU 分支——这本身就是一种安装正确性的隐式检查。而如果你装的是 GPU 版但驱动或 CUDA 版本不匹配，这一行会直接暴露问题。所以这段代码不只是"跑通一个运算"，它同时验证了**安装类型**与**硬件可用性**两个层面。

对于你的 RTX 3060 12GB 环境，如果输出中能看到 `GPU 张量:` 和 `GPU 张量运算结果:`，说明 PyTorch 已经正确识别了显卡并完成了张量搬运与计算，环境就绪。如果只看到 CPU 分支的输出，则说明当前安装的是 CPU 版本，后续做 LoRA 微调时训练速度会明显受限。

**概念对比**：`torch.rand(2, 3)` 创建的是随机初始化的张量，而 `x.cuda()` 是**显式**将张量从内存搬到显存；两者触发时机不同——前者在创建时决定数据内容，后者在搬运时决定数据位置。

**思考题**：
1. 为什么这段验证代码要用 `torch.cuda.is_available()` 做条件判断，而不是直接调用 `.cuda()`？
2. 如果输出中只出现了 `CPU 张量:` 和 `张量运算结果（CPU）:`，说明 PyTorch 的安装类型是什么？

## 块 14

### 七、常见问题与解决（①+②+③）

这一节把环境准备阶段最常遇到的报错集中梳理成一张排查表。你的目标是 LoRA 微调，最终要跑起来的路径是“数据 → 模型 → GPU 计算”，而 GPU 计算的第一步就是让 PyTorch 真正“看见”显卡。下面这些症状，本质上都是在回答同一个问题：**PyTorch 和 CUDA 之间到底通没通？**

#### 症状一：`torch.cuda.is_available()` 返回 `False`

这是最典型的“不通”信号，可能原因有三个，按出现频率从高到低排列：

1. **装了 CPU 版 PyTorch**。这是最常见的情况——安装时没指定 CUDA 版本，pip 默认给了 CPU 版。验证方法：检查 `torch.__version__` 是否含 `+cpu` 后缀，如果是，重装 GPU 版即可。
2. **没有 NVIDIA 显卡或驱动未装**。先确认 `nvidia-smi` 命令有输出；如果根本没输出，说明系统里没有可用的 NVIDIA 驱动。此时如果机器确实没有 GPU，那就老老实实用 CPU 版，虽然 LoRA 微调在 CPU 上会很慢，但至少能跑通流程。
3. **PyTorch 的 CUDA 版本高于驱动支持上限**。`nvidia-smi` 输出里会显示驱动支持的 CUDA Version（比如 11.8 或 12.1），而 PyTorch 编译时绑定的 CUDA 版本（`+cu118`、`+cu121` 等）必须 ≤ 这个值。如果 PyTorch 要求的更高，驱动带不动，就会返回 `False`。解决办法是重装对应低版本 CUDA 的 PyTorch。

> **概念对比**：`nvidia-smi` 显示的 CUDA Version 是**驱动支持的上限**，而 `torch.__version__` 里的 `+cuXXX` 是 **PyTorch 编译时绑定的 CUDA 版本**。前者是“天花板”，后者是“实际需求”，需求不能超过天花板。

#### 症状二：安装时报 `No matching distribution found`

这通常是 Python 版本和 PyTorch 的兼容范围不匹配。PyTorch 官方对 Python 版本有明确支持区间，过新或过旧都会导致 pip 找不到对应的 wheel 包。确保 Python 版本在 3.8~3.12 之间，推荐使用 3.10，这是当前生态兼容性最好的版本。

#### 症状三：conda 环境相关的问题

两个常见场景：

- **`conda activate pytorch` 无效**：先执行 `conda init` 并重启 shell，确认 conda 初始化已生效；如果仍未解决，再检查当前 shell 是否已加载 conda 配置。
- **多 conda 环境，A 环境能用 GPU，B 环境不行**：每个 conda 环境是独立的 Python 包空间，A 环境装了 GPU 版 PyTorch 不代表 B 环境也有。需要在每个环境里单独检查 `CUDA_VISIBLE_DEVICES` 环境变量是否被设置、`nvidia-smi` 输出是否正常，以及 PyTorch 版本是否含 `+cuXXX` 且与驱动兼容，不能跨环境复用。

#### 症状四：Linux 下提示 `libcudart.so.12: cannot open shared object file`

这是典型的 CUDA 动态库路径问题。PyTorch 运行时需要加载 CUDA 的共享库文件，但系统找不到它。解决方案是手动把 CUDA 库路径加入环境变量：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

注意这只是临时生效，要永久生效需要把这行写入 `~/.bashrc` 文件。

#### 症状五：WSL2 里 `nvidia-smi` 正常但 PyTorch 找不到 GPU

WSL2 的 GPU 透传依赖 Windows 侧的驱动和 WSL2 内核版本。如果 `nvidia-smi` 在 WSL2 里能正常输出，说明驱动透传没问题，但 PyTorch 仍找不到 GPU，先检查 PyTorch 是否为 GPU 版（`torch.__version__` 含 `+cu`），并确认 CUDA 版本与驱动兼容；同时更新 WSL2 内核，在 Windows PowerShell 里执行 `wsl --update` 更新到最新版。


**最重要的排查口诀**（按顺序执行，能解决 90% 的问题）：

1. 先确认 `nvidia-smi` 有输出——证明驱动和硬件没问题；
2. 再确认 PyTorch 版本含 `+cuXXX` 而非 `+cpu`——证明装对了版本；
3. 最后确认 `cuXXX` ≤ `nvidia-smi` 的 CUDA Version——证明版本兼容。


**思考题**（答案都能在原文中找到）：

1. 为什么 `nvidia-smi` 有输出但 PyTorch 仍可能返回 `False`？请列出至少两种可能原因。
2. 排查口诀的三步顺序为什么是“先驱动、再版本、后兼容”？如果第一步就失败，后续步骤还有意义吗？

## 块 15

## 八、总结与后续学习路线（①+②+③）

到这里，PyTorch 环境准备阶段就收尾了。你现在应该已经具备：

- ✅ 一个独立的 `pytorch` conda 虚拟环境
- ✅ 成功安装了 PyTorch 2.5.1（GPU 版或 CPU 版）
- ✅ 理解了显卡、驱动、CUDA、PyTorch 四者的版本对齐关系
- ✅ 验证了 `torch.cuda.is_available()` 返回预期结果
- ✅ 运行了第一个 PyTorch 张量运算

其中，"显卡、驱动、CUDA、PyTorch 四者版本对齐"这条值得再强调一层：你后续做 LoRA 微调时，模型权重的前向传播和反向传播都会落在 GPU 上，而 PyTorch 能否正确调用 GPU 完全取决于这四者的对齐是否成立。`torch.cuda.is_available()` 返回 `True` 只是"环境可用"的底线验证，真正决定训练效率的是 CUDA 版本与 PyTorch 编译时所依赖的 CUDA 版本是否匹配——这也是为什么安装命令里要显式指定 `cu121` 或 `cu118` 这类后缀。

> **下一步预告**：在下一节 **"张量操作"** 中，我们将正式学习 PyTorch 的核心数据结构 —— **Tensor（张量）**。你会发现 PyTorch 的 Tensor 与前面学过的 NumPy ndarray 概念高度相似，今天学的索引、切片、广播等操作在 PyTorch 中几乎一模一样。同时，我们还会学习如何把 Tensor 在 CPU 和 GPU 之间自由移动（`.to('cuda')`）。现在，确保你的 PyTorch 环境已经就绪，我们马上开始！

对你当前的目标（LoRA 微调）来说，Tensor 操作是绕不开的前置基础：LoRA 的核心思路是在冻结原模型权重的前提下，注入低秩矩阵作为可学习参数，而这些低秩矩阵的初始化、更新和合并，本质上都是 Tensor 层面的运算。你已有的 Python 编程基础足以支撑你快速迁移 NumPy 的索引、切片、广播经验到 PyTorch 上，但要注意一个关键差异：NumPy 的数组默认在 CPU 上，而 PyTorch 的 Tensor 可以随时通过 `.to('cuda')` 在 CPU 与 GPU 之间迁移——这个迁移动作的代价（数据拷贝开销）和时机，会直接影响你后续微调脚本的性能表现。

结合你的 RTX 3060 12GB 显存，下一节学习 Tensor 时建议重点关注两个点：一是 Tensor 的形状操作（`reshape`、`permute`、`squeeze` 等），因为 LoRA 注入的低秩矩阵需要与原始权重形状精确对齐；二是设备迁移的语义，因为 12GB 显存对 LoRA 微调是够用的，但如果你同时加载模型和多个中间张量，就需要留意哪些张量该留在 GPU、哪些可以放回 CPU。

**思考题**：
1. 为什么说 `torch.cuda.is_available()` 返回 `True` 只是"环境可用"的底线验证，而不是充分验证？
2. 在环境准备阶段，显卡、驱动、CUDA、PyTorch 四者的版本对齐关系中，哪两个之间的匹配是安装命令中显式指定的？

**概念对比**：NumPy ndarray vs PyTorch Tensor——两者在索引、切片、广播操作上高度相似，但 Tensor 额外支持设备迁移（`.to('cuda')`）和自动求导，后者是 LoRA 微调中梯度计算的基础。

**模型许可证说明**：后续微调所引用的模型需注意其许可证类型，例如 Llama 2 采用 Community License，商用月活超 7 亿需获得 Meta 授权。

## 测评题目

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB04-ST0400-002
在 Python 中可通过 `torch.cuda.is_available()` 检查 PyTorch 是否识别到________。

### Q-KB0405-002
`AdamW` 优化器中，`W` 代表________，它解决了Adam中________的问题。

### Q-KB0402-004
以下哪个操作会**断开**计算图，使后续操作不追踪梯度？

### Q-KB0405-003
请写代码：为模型 `model` 创建 `AdamW` 优化器，学习率 `1e-4`，权重衰减 `1e-2`。

### Q-KB0407-002
加载模型参数时，需要先________模型结构，然后调用________方法加载参数字典。

### Q-KB0407-003
请写代码：保存模型 `model` 的参数到 `checkpoint.pth`，并包含优化器状态和当前epoch。

### Q-KB0403-003
请定义一个简单的神经网络类 `SimpleNet`，包含：
- 一个输入层到隐藏层的线性层（10 -> 20）
- ReLU激活
- 隐藏层到输出层的线性层（20 -> 1）
