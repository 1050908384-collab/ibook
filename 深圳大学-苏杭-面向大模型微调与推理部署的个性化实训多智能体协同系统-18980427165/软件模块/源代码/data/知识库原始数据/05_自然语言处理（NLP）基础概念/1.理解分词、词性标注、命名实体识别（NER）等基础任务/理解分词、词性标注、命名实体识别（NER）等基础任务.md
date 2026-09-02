# 自然语言处理（NLP）基础概念：分词、词性标注与命名实体识别

> **参考文章链接**
> 
> ① 底稿：《NLP核心任务（分词、词性标注、命名实体识别等）》
> 　　网址：https://blog.csdn.net/qq_73899585/article/details/157433742
> 
> ② 参考文章1：《自然语言处理分词词性标注与命名实体识别》
> 　　网址：https://blog.csdn.net/tdfnul_880/article/details/162257397
> 
> ③ 参考文章2：《中文NLP入门必备：全面解析分词、命名实体识别与词性标注（附详细实战案例）》
> 　　网址：https://blog.csdn.net/Kiradzy/article/details/144987768
> 
> ④ 参考文章3：《命名实体识别（NER）标注方法BIO，BMES和BIOES》
> 　　网址：https://blog.csdn.net/shizidushu/article/details/142859605

---

**适合人群**：已掌握 Python 编程基础，初次接触自然语言处理（NLP）的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够理解分词、词性标注、命名实体识别这三大 NLP 基石任务，并能使用 `jieba` 工具完成基础文本分析，为后续"注意力机制""Transformer 架构"打下坚实基础。

**预计学习时长**：约 1.5 小时  
**环境准备**：Python 3.10+，安装 `jieba` 库（`pip install jieba`）

---

## 第一章：NLP 任务概览——从"看见文字"到"理解语义"

自然语言处理（Natural Language Processing，NLP）是人工智能领域最具挑战性的方向之一。它的目标是让计算机能够理解、处理和生成人类语言。

与图像处理不同，语言是高度抽象和结构化的，充满了歧义、隐喻和上下文依赖。NLP 任务按照处理层次可以分为：

```
原始文本
    ↓
【词法分析层】
    ├── 分词（Tokenization）
    ├── 词性标注（POS Tagging）
    └── 命名实体识别（NER）
    ↓
【句法分析层】
    ├── 句法树构建
    └── 依存句法分析
    ↓
【语义分析层】
    ├── 语义角色标注
    └── 语义相似度
    ↓
【应用任务层】
    ├── 文本分类 / 情感分析
    ├── 机器翻译
    ├── 问答系统
    └── 对话系统
```

本章聚焦**词法分析层**的三大核心任务：分词、词性标注、命名实体识别。它们是构建更复杂 NLP 应用的基石。

---

## 第二章：分词（Tokenization）——把句子切成有意义的块

### 2.1 为什么需要分词

对于英文等使用空格分隔单词的语言，分词相对简单，直接用 `split()` 即可：

```python
english_text = "Natural Language Processing is fascinating!"
english_tokens = english_text.split()
print(english_tokens)
# ['Natural', 'Language', 'Processing', 'is', 'fascinating!']
```

但对于**中文**，词与词之间没有明显的分隔符，"字"不等于"词"。在基于"词"的传统 NLP 流程中，不分词直接把句子当成字符序列，计算机会把"自然语言处理"理解为 6 个独立的字，丢失了"自然语言处理"作为一个完整概念的语义。

```python
chinese_text = "自然语言处理是人工智能的重要分支"
# 不分词 = 字符序列
print(list(chinese_text))
# ['自', '然', '语', '言', '处', '理', '是', '人', '工', '智', '能', '的', '重', '要', '分', '支']
```

**为什么这样写**：对于使用 jieba、CRF、HMM 等传统中文 NLP 工具链的任务，**分词是词法分析的第一步**。词是传统"词法→句法→语义"管线中的基础单位，分词质量直接影响后续文本分类、情感分析等任务的效果。注意：**现代大语言模型（BERT、GPT、Llama 等）通常使用子词（Subword）或字符级 Tokenizer，不再走"中文分词→词向量"这条管线**，两者不要混淆。

### 2.2 jieba 分词的三种模式

`jieba` 是 Python 最流行的中文分词库。安装：

```bash
pip install jieba
```

**模式一：精确模式（默认）**

尽量将句子切分为最精确的词，适合文本分析。

```python
import jieba

text = "小明喜欢自然语言处理"

# 精确模式
result = jieba.lcut(text, cut_all=False)
print(result)
# ['小明', '喜欢', '自然语言处理']
```

**为什么这样写**：`jieba.lcut()` 返回列表，比 `jieba.cut()`（返回生成器）更方便直接查看结果。`cut_all=False` 表示精确模式。

**模式二：全模式**

把句子中所有可能的词都切分出来，速度快，但不能消除歧义。

```python
result = jieba.lcut(text, cut_all=True)
print(result)
# ['小明', '喜欢', '自然', '自然语言', '语言', '语言处理', '处理']
```

**为什么这样写**：全模式会输出所有可能的词组合，比如"自然语言处理"会被拆成"自然""自然语言""语言""语言处理""处理"。适合搜索引擎构建索引，但不适合需要精确语义的任务。

**模式三：搜索引擎模式**

在精确模式基础上，对长词再次切分，适合搜索引擎分词。

```python
result = jieba.lcut_for_search(text)
print(result)
# ['小明', '喜欢', '自然', '语言', '自然语言', '处理', '语言处理']
```

**三种模式对比**：

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| 精确模式 | 精确切分，无冗余 | 文本分析、情感分析 |
| 全模式 | 覆盖所有可能词 | 搜索引擎索引 |
| 搜索引擎模式 | 精确+长词再切分 | 搜索引擎、信息检索 |

### 2.3 自定义词典——让分词更懂你的领域

jieba 默认词典对通用词汇覆盖较好，但遇到专业术语、品牌名、人名时可能切分错误。

**创建自定义词典文件** `userdict.txt`：

```
自然语言处理 100 n
ChatGPT 50 nz
特斯拉 50 nt
```

**格式说明**：每行包含 `词语 词频 词性`，词频和词性可选。词频数值越大表示越优先匹配。

**加载并使用**：

```python
import jieba

# 加载自定义词典
jieba.load_userdict('userdict.txt')

text = "小明喜欢自然语言处理，常用ChatGPT辅助学习"
result = jieba.lcut(text)
print(result)
# ['小明', '喜欢', '自然语言处理', '，', '常用', 'ChatGPT', '辅助', '学习']
```

**为什么这样写**：如果不加载自定义词典，"自然语言处理"可能被切成"自然/语言/处理"，"ChatGPT"可能被切成"Chat/GPT"。自定义词典让分词器认识这些专有名词。

### 2.4 分词的两大挑战

**挑战一：歧义**

同样的字符序列，可能有多种切分方式。

```python
ambiguous_texts = [
    "结婚的和尚未结婚的",   # "和尚未" vs "和 尚未"
    "乒乓球拍卖完了",       # "乒乓球拍/卖完了" vs "乒乓球/拍卖/完了"
    "中国人民银行发行货币", # "中国人民银行" vs "中国/人民/银行"
]

for text in ambiguous_texts:
    tokens = jieba.lcut(text)
    print(f"原文: {text}")
    print(f"分词: {' / '.join(tokens)}")
    print()
```

**挑战二：未登录词（OOV）**

词典中没有的新词，如网络流行语、新产品名。

```python
oov_text = "我在用ChatGPT和Claude进行对话"
tokens = jieba.lcut(oov_text)
print(f"原文: {oov_text}")
print(f"分词: {' / '.join(tokens)}")
# 可能输出: ['我', '在', '用', 'ChatGPT', '和', 'Claude', '进行', '对话']
```

**为什么这样写**：未登录词是中文分词的永恒难题。解决方法包括：自定义词典、基于统计的模型（HMM、CRF）、基于深度学习的模型（BERT）。

### 2.5 动手实现：正向最大匹配分词器

为了理解分词的底层原理，我们来实现一个简单的**正向最大匹配（FMM）**分词器。

```python
class MaxMatchSegmenter:
    """
    基于最大匹配的中文分词器
    """
    def __init__(self, dictionary):
        """
        dictionary: 词典，包含所有已知词语
        """
        self.dictionary = set(dictionary)
        self.max_word_length = max(len(word) for word in dictionary)

    def forward_max_match(self, text):
        """
        正向最大匹配：从左到右，每次取最长匹配词
        """
        result = []
        index = 0

        while index < len(text):
            matched = False
            # 从最大长度开始尝试匹配
            for length in range(self.max_word_length, 0, -1):
                if index + length > len(text):
                    continue

                word = text[index:index + length]
                if word in self.dictionary:
                    result.append(word)
                    index += length
                    matched = True
                    break

            # 如果没有匹配到，单字成词
            if not matched:
                result.append(text[index])
                index += 1

        return result


# 构建简单词典
dictionary = [
    "自然", "语言", "处理", "自然语言", "语言处理", "自然语言处理",
    "是", "人工", "智能", "人工智能", "的", "重要", "分支"
]

# 测试
segmenter = MaxMatchSegmenter(dictionary)
text = "自然语言处理是人工智能的重要分支"

print("正向最大匹配:", segmenter.forward_max_match(text))
# ['自然语言处理', '是', '人工智能', '的', '重要', '分支']
```

**为什么这样写**：
- 正向最大匹配是最经典的分词算法之一
- 核心思想：从当前位置开始，优先匹配词典中最长的词
- 如果没有任何词匹配，就把当前字单独作为一个词
- 实际工业级分词器（如 jieba）会结合统计模型和词典，效果远好于纯规则方法

---

## 第三章：词性标注（POS Tagging）——给每个词贴语法标签

### 3.1 什么是词性标注

词性标注（Part-of-Speech Tagging）是为句子中的**每个词标注其语法类别**的任务，如名词、动词、形容词等。

**示例**：

```
我/r 喜欢/v 自然语言处理/n
```

同一个词在不同语境下可能有不同词性。例如"苹果"：
- "吃**苹果**" → 苹果/n（名词）
- "**苹果**公司" → 苹果/nz（专有名词）

**为什么这样写**：词性信息对于理解句子结构和语义至关重要，是句法分析、机器翻译、文本摘要等任务的基础。

### 3.2 jieba 词性标注实战

jieba 通过 `posseg` 模块支持词性标注：

```python
import jieba.posseg as pseg

text = "我喜欢自然语言处理"
result = pseg.lcut(text)

for word, flag in result:
    print(f"{word}: {flag}")
```

**输出**：
```
我: r
喜欢: v
自然语言处理: n
```

### 3.3 常见词性标签速查表

| 标签 | 含义 | 示例 |
|------|------|------|
| `n` | 普通名词 | 书、苹果、电脑 |
| `nr` | 人名 | 李华、鲁迅、马斯克 |
| `ns` | 地名 | 北京、上海、长江 |
| `nt` | 机构团体名 | 清华大学、新华社、腾讯 |
| `nz` | 其他专有名词 | 人工智能、区块链、自然语言处理 |
| `v` | 普通动词 | 跑、吃、学习 |
| `vn` | 动名词 | 写作、旅行、处理 |
| `a` | 形容词 | 美丽、迅速、高兴 |
| `d` | 副词 | 很、非常、极其 |
| `m` | 数量词 | 一个、三只、五本 |
| `q` | 量词 | 个、只、本、条 |
| `r` | 代词 | 我、你、他、它 |
| `p` | 介词 | 在、对、从、向 |
| `c` | 连词 | 和、但是、或者 |
| `u` | 助词 | 的、了、着、过 |
| `t` | 时间词 | 今天、明天、2024年 |
| `f` | 方位词 | 上、下、左、右 |
| `w` | 标点符号 | ，。？！ |
| `x` | 非语素字 / 字符串 | 萄（葡萄中单独出现）、网址串 |

### 3.4 自定义词性

通过自定义词典，可以为特定词语分配自定义词性标签。

创建 `userdict_pos.txt`：

```
自然语言处理 100 nlp
```

加载并使用：

```python
jieba.load_userdict('userdict_pos.txt')
text = "我喜欢自然语言处理"
result = pseg.lcut(text)

for word, flag in result:
    print(f"{word}: {flag}")
# 自然语言处理: nlp
```

---

## 第四章：命名实体识别（NER）——从文本中"抓"出关键信息

### 4.1 什么是命名实体识别

命名实体识别（Named Entity Recognition，NER）是从文本中识别出**具有特定意义的实体**，如人名、地名、机构名、时间、日期等。

**示例**：

```
2024年1月，马斯克在美国加州的特斯拉总部宣布了新的AI计划
    ↓
TIME: 2024年1月
PERSON: 马斯克
LOCATION: 美国、加州
ORGANIZATION: 特斯拉
```

**为什么这样写**：NER 是信息抽取的核心任务，广泛应用于知识图谱构建、舆情监控、金融分析（识别公司名辅助投资决策）、智能问答等领域。

### 4.2 NER 标注体系：BIO、BIOES

NER 通常被建模为**序列标注**任务。每个字（或词）被赋予一个标签，表示它在实体中的位置。

**BIO 标注法**（最常用）：

| 标签 | 含义 | 示例 |
|------|------|------|
| `B-X` | 某类实体的开头（Begin） | `B-PER` 表示人名开始 |
| `I-X` | 某类实体的内部（Inside） | `I-PER` 表示人名内部 |
| `O` | 不属于任何实体（Outside） | |

**示例**：

```
马  B-PER
斯  I-PER
克  I-PER
在  O
特  B-ORG
斯  I-ORG
拉  I-ORG
工  O
作  O
```

**BIOES 标注法**（更精细）：

| 标签 | 含义 | 示例 |
|------|------|------|
| `B-X` | 实体开始（Begin） | |
| `I-X` | 实体内部（Intermediate） | |
| `E-X` | 实体结束（End） | |
| `O` | 非实体 | |
| `S-X` | 单字实体（Single） | |

**为什么这样写**：BIOES 比 BIO 多了 `E`（结束）和 `S`（单字实体），能更精确地标识实体边界，有助于模型学习实体结束位置，提升识别准确率。

### 4.3 用 jieba 实现简单 NER

jieba 的词性标注结果中，`nr`、`ns`、`nt` 分别对应人名、地名、机构名，可以据此做简单的 NER：

```python
import jieba.posseg as pseg

text = "2024年1月，马斯克在美国加州的特斯拉总部宣布了新的AI计划"
words = pseg.lcut(text)

entities = {
    'TIME': [],
    'PERSON': [],
    'LOCATION': [],
    'ORG': []
}

for word, flag in words:
    if flag == 't':
        entities['TIME'].append(word)
    elif flag == 'nr':
        entities['PERSON'].append(word)
    elif flag == 'ns':
        entities['LOCATION'].append(word)
    elif flag == 'nt':
        entities['ORG'].append(word)

for entity_type, entity_list in entities.items():
    if entity_list:
        print(f"{entity_type}: {', '.join(entity_list)}")
```

**可能输出**（jieba 版本、加载词典不同，结果会略有差异）：
```
TIME: 2024年, 1月
PERSON: 马斯克
LOCATION: 美国, 加州
ORG: 特斯拉
```

**⚠️ 局限性**：jieba 基于词典和统计模型，对复杂语境、嵌套实体、新词的效果有限。工业级 NER 通常使用深度学习模型。

### 4.4 深度学习 NER 简介：BiLSTM-CRF

现代 NER 的主流架构是 **BiLSTM-CRF**：

```
输入句子: "马斯克在特斯拉工作"
    ↓
字符/词编码 → 词嵌入层
    ↓
BiLSTM 层（双向长短期记忆网络）
    ← 前向 LSTM: 捕捉从左到右的上下文
    → 后向 LSTM: 捕捉从右到左的上下文
    ↓
特征拼接
    ↓
CRF 层（条件随机场）
    学习标签之间的转移规律
    例如: I-PER 前面必须是 B-PER 或 I-PER（不能直接从 O 跳到 I-PER）；
          B-PER 后面不能跟类型不匹配的 I-ORG / I-LOC，但可以接另一个实体的 B-XXX（相邻实体）或 O
    ↓
输出标签序列: [B-PER, I-PER, I-PER, O, B-ORG, I-ORG, I-ORG, O, O]
    ↓
提取实体: 马斯克(PER), 特斯拉(ORG)
```

**为什么这样写**：BiLSTM 捕捉上下文语义，CRF 约束标签转移的合法性，两者结合使 NER 准确率大幅提升。预训练模型（如 BERT）进一步将 NER 效果推向新高度。

---

## 第五章：NLP 任务的评估指标

如何衡量分词、词性标注、NER 的效果？核心指标有四个：

| 指标 | 含义 | 公式 |
|------|------|------|
| **准确率（Accuracy）** | 预测正确的标签占总标签的比例 | `(TP + TN) / 总数` |
| **精确率（Precision）** | 预测为正的样本中，真正为正的比例 | `TP / (TP + FP)` |
| **召回率（Recall）** | 真正为正的样本中，被正确预测的比例 | `TP / (TP + FN)` |
| **F1 分数** | 精确率和召回率的调和平均 | `2 * P * R / (P + R)` |

**为什么这样写**：对于 NER 等任务，**精确率和召回率往往比准确率更重要**。例如，一个保守的模型只预测它非常确定的实体，精确率会很高但召回率很低；一个激进的模型预测很多实体，召回率高但精确率低。F1 分数平衡了两者。

> **注意层级区分**：NER 的标准评估是**实体级（Entity-Level）F1**：只有当预测出的实体 span（起止位置+类型）与真实实体**完全匹配**才算 TP。下面的代码是一个**简化版 Token-Level** F1 实现（逐 token 比较），用于教学理解四个指标的定义，不等于 NER 的标准实体级评估。实际工程中应使用 `seqeval` 等库提供的实体级评估。

```python
def calculate_metrics_token_level(true_labels, pred_labels):
    """Token-Level 简化 F1：逐 token 比较非 O 标签的对错（仅用于教学，非 NER 标准评估）"""
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == p and t != 'O')
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != p and p != 'O')
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t != p and t != 'O')

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


# 示例
true = ['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'O']
pred = ['B-PER', 'I-PER', 'O', 'B-ORG', 'O', 'O']

metrics = calculate_metrics_token_level(true, pred)
print(f"Precision (token级): {metrics['precision']:.2f}")
print(f"Recall    (token级): {metrics['recall']:.2f}")
print(f"F1        (token级): {metrics['f1']:.2f}")
```

---

## 第六章：综合实战——构建 NLP 处理流水线

把分词、词性标注、NER 串起来，构建一个简单的 NLP 流水线：

```python
import jieba
import jieba.posseg as pseg

class NLPPipeline:
    """简单的 NLP 处理流水线"""

    def __init__(self):
        self.results = {}

    def tokenize(self, text):
        """分词"""
        tokens = jieba.lcut(text)
        self.results['tokens'] = tokens
        return tokens

    def pos_tag(self, tokens=None):
        """词性标注"""
        if tokens is None:
            tokens = self.results.get('tokens', [])
        text = ''.join(tokens)
        pos_tags = [(word, flag) for word, flag in pseg.cut(text)]
        self.results['pos_tags'] = pos_tags
        return pos_tags

    def ner(self, pos_tags=None):
        """命名实体识别"""
        if pos_tags is None:
            pos_tags = self.results.get('pos_tags', [])

        entities = []
        for word, flag in pos_tags:
            if flag == 'nr':
                entities.append((word, 'PERSON'))
            elif flag == 'ns':
                entities.append((word, 'LOCATION'))
            elif flag == 'nt':
                entities.append((word, 'ORGANIZATION'))
            elif flag == 't':
                entities.append((word, 'TIME'))

        self.results['entities'] = entities
        return entities

    def process(self, text):
        """完整处理流程"""
        print(f"📄 原始文本: {text}\n")

        # 分词
        tokens = self.tokenize(text)
        print(f"🔪 分词结果: {' / '.join(tokens)}\n")

        # 词性标注
        pos_tags = self.pos_tag(tokens)
        print("🏷️ 词性标注:")
        for word, flag in pos_tags:
            print(f"  {word}/{flag}", end="  ")
        print("\n")

        # 命名实体识别
        entities = self.ner(pos_tags)
        if entities:
            print("🎯 命名实体:")
            for entity, entity_type in entities:
                print(f"  • {entity} ({entity_type})")
        else:
            print("❌ 未识别到命名实体")

        return self.results


# 使用流水线
pipeline = NLPPipeline()

text = "2024年，马斯克在美国加州的特斯拉总部宣布了新的人工智能计划"
results = pipeline.process(text)

print("\n" + "=" * 50)
print("📊 处理结果汇总")
print("=" * 50)
print(f"分词数量: {len(results['tokens'])}")
print(f"词性标注数量: {len(results['pos_tags'])}")
print(f"实体数量: {len(results['entities'])}")
```

---

## 本章小结与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 中文文本直接用 `split()` 分词 | 中文没有空格分隔，传统词法分析必须使用 jieba 等分词工具；现代 LLM 使用子词 Tokenizer，不走这条管线 |
| 忽略 jieba 的三种分词模式差异 | 文本分析用精确模式，搜索引擎用搜索引擎模式 |
| 遇到专业术语分词错误 | 加载自定义词典 `jieba.load_userdict()` |
| 把"字"当"词"处理 | 传统中文 NLP 管线中语义的基本单位是"词"而非"字" |
| NER 只用规则匹配 | 规则+统计+深度学习结合，工业级用 BiLSTM-CRF 或 BERT |
| BIO 标注中 `I-X` 跟在 `O` 后面 | `I-X` 前面必须是 `B-X` 或 `I-X`，标注时要检查合法性 |
| 评估 NER 只看准确率 | 关注精确率、召回率，重点看**实体级 F1**（用 seqeval 等库，不要只算 token 级） |

| 知识点 | 核心要点 |
|--------|----------|
| **分词** | 传统中文 NLP 的第一步，jieba 提供精确/全/搜索引擎三种模式 |
| **自定义词典** | 解决专业术语、新词的分词问题 |
| **词性标注** | 为每个词标注语法类别（n/v/a/d 等） |
| **NER** | 识别人名、地名、机构名等实体 |
| **BIO 标注** | B-开头、I-内部、O-非实体；B-X 后可接 I-X / O / 另一实体 B-Y |
| **BIOES 标注** | B-开头、I-内部、E-结束、S-单字、O-非实体 |
| **评估指标** | Precision、Recall、F1；NER 用**实体级 F1** 做标准 |

**养成的好习惯**：
1. 任何**传统中文 NLP 管线**任务，第一步先检查分词质量；现代子词 Tokenizer 另当别论
2. 遇到专业领域文本，先准备自定义词典
3. NER 标注数据时，严格遵循 BIO/BIOES 规范
4. 评估 NER 模型时，不要只看 token 级指标，使用 **实体级 F1**（seqeval 等）更可靠

**下一步预告**：掌握了 NLP 基础任务后，你将进入 文本的数值化表示方法 的学习——理解 Bag-of-Words、TF-IDF、Word2Vec 如何将人类语言转换为计算机能理解的数字向量，这是从"文本处理"迈向"深度学习模型"的关键桥梁。在此之后，才是注意力机制与 Transformer 架构。