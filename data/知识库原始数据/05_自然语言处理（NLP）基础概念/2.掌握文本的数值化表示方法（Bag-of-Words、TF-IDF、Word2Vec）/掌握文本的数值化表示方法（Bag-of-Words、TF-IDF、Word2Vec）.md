# 文本的数值化表示：从 Bag-of-Words 到 Word2Vec

> **参考文章链接**
> 
> ① 底稿：《文本数值化表示（词袋模型、TF-IDF、Word2Vec）》
> 　　网址：https://blog.csdn.net/universsky2015/article/details/135244715
> 
> ② 参考文章1：《词向量原理详解：Word2Vec（CBOW/skip-gram）与 GloVe 的差异》
> 　　网址：https://blog.csdn.net/2501_93893835/article/details/154123256
> 
> ③ 参考文章2：《Word2Vec实战：用Skip-gram模型训练中文词向量（附完整代码）》
> 　　网址：https://blog.csdn.net/weixin_29181861/article/details/158131337
> 
> ④ 参考文章3：《机器学习：TF-IDF算法原理及代码实现》
> 　　网址：https://blog.csdn.net/mohanyelong/article/details/141686320


---

**适合人群**：已掌握 Python 基础、NLP 分词任务，希望理解如何将文本转换为计算机可处理的数字向量的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将能够使用 Bag-of-Words、TF-IDF 和 Word2Vec 将文本数值化，为后续"注意力机制""Transformer 架构"打下坚实基础。

**预计学习时长**：约 2 小时  
**环境准备**：Python 3.10+，安装 `jieba`、`gensim`、`scikit-learn`、`numpy`

```bash
pip install jieba gensim scikit-learn numpy
```

---

## 第一章：为什么需要文本数值化表示

计算机只认识数字，不认识文字。要让机器"理解"文本，第一步就是把文字转换成数字向量。这个过程叫做**文本的数值化表示（Text Numerical Representation）**或**词嵌入（Word Embedding）**。

假设你已经用 jieba 把"我喜欢自然语言处理"分成了 `['我', '喜欢', '自然语言处理']`。下一步，这些词怎么输入到机器学习模型中？

**不能直接输入字符串**。神经网络、SVM、逻辑回归等模型都需要数值输入。因此，我们需要一种映射机制：

```
词语  →  数字/向量
"猫"  →  [0, 1, 0, 0, ...]
"狗"  →  [0, 0, 1, 0, ...]
```

本章介绍三种经典的文本数值化方法，按发展顺序从简单到复杂：

| 方法 | 类型 | 维度 | 是否考虑语义 | 适用场景 |
|------|------|------|-------------|----------|
| **Bag-of-Words (BoW)** | 离散表示 | 高维稀疏 | ❌ 否 | 文本分类、主题建模 |
| **TF-IDF** | 离散表示（加权） | 高维稀疏 | ❌ 否 | 信息检索、关键词提取 |
| **Word2Vec** | 分布式表示 | 低维密集 | ✅ 是 | 语义相似度、作为深度学习输入 |

---

## 第二章：词袋模型（Bag-of-Words）——最朴素的文本表示

### 2.1 核心思想

**词袋模型**把文本看作一个"袋子"，只关心里面有哪些词、每个词出现了几次，**完全忽略词语的顺序和语法**。

```
句子A: "猫 喜欢 鱼"
句子B: "狗 喜欢 骨头"

词典: {猫:0, 喜欢:1, 鱼:2, 狗:3, 骨头:4}

句子A的向量: [1, 1, 1, 0, 0]   (猫出现1次, 喜欢1次, 鱼1次)
句子B的向量: [0, 1, 0, 1, 1]   (喜欢1次, 狗1次, 骨头1次)
```

### 2.2 构建词典（Build Dictionary）

词袋模型的第一步是构建**词典（Vocabulary）**：统计语料中所有不重复的词语，每个词分配一个唯一索引。

```python
from collections import Counter

# 假设已有分词后的语料
tokenized_texts = [
    ['我', '喜欢', '猫'],
    ['我', '喜欢', '狗'],
    ['猫', '和', '狗', '都', '可爱']
]

# 构建词典
all_tokens = [word for text in tokenized_texts for word in text]
word_freq = Counter(all_tokens)

# 分配索引（按频率排序，频率高的排在前面）
vocab = {word: idx for idx, (word, _) in enumerate(word_freq.most_common())}
print(vocab)
# {'我': 0, '喜欢': 1, '猫': 2, '狗': 3, '和': 4, '都': 5, '可爱': 6}
```

**为什么这样写**：`Counter.most_common()` 按频率排序，让高频词获得较小的索引，这在某些场景下有利于模型收敛。

### 2.3 One-Hot 编码

One-Hot 是最极端的稀疏表示：每个词用一个向量表示，向量长度等于词典大小，只有对应位置为 1，其余为 0。

```python
import numpy as np

vocab_size = len(vocab)

def one_hot_encode(word, vocab):
    vector = np.zeros(vocab_size)
    if word in vocab:
        vector[vocab[word]] = 1
    return vector

print(one_hot_encode('猫', vocab))
# [0. 0. 1. 0. 0. 0. 0.]  ← 只有索引2的位置是1
```

**为什么这样写**：One-Hot 简单直观，但有两个致命缺陷：
1. **维度灾难**：词典有 10 万个词，每个词就是 10 万维向量
2. **语义鸿沟**："猫"和"狗"的 One-Hot 向量正交（点积为 0），但它们在语义上很相近

### 2.4 词频向量（Count Vectorization）

更实用的 BoW 表示不是给单个词编码，而是给**整个句子/文档**编码：向量每个位置代表对应词在文档中出现的次数。

```python
def text_to_bow(text, vocab):
    vector = np.zeros(len(vocab))
    for word in text:
        if word in vocab:
            vector[vocab[word]] += 1
    return vector

text_a = ['我', '喜欢', '猫']
print(text_to_bow(text_a, vocab))
# [1. 1. 1. 0. 0. 0. 0.]

text_c = ['我', '喜欢', '喜欢', '猫']
print(text_to_bow(text_c, vocab))
# [1. 2. 1. 0. 0. 0. 0.]  ← "喜欢"出现了2次
```

**为什么这样写**：词频向量保留了词语出现次数的信息，比纯 One-Hot 更适合文本分类等任务。

### 2.5 用 sklearn 的 CountVectorizer（简化版）

```python
from sklearn.feature_extraction.text import CountVectorizer

# 原始文本（未分词，CountVectorizer 默认按空格/标点分词，中文需先分词）
texts = ["我 喜欢 猫", "我 喜欢 狗", "猫 和 狗 都 可爱"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
# ['喜欢' '可爱' '和' '狗' '猫' '都' '我']

print(X.toarray())
# [[1 0 0 0 1 0 1]   ← 第一句
#  [1 0 0 1 0 0 1]   ← 第二句
#  [0 1 1 1 1 1 0]]  ← 第三句
```

**⚠️ 避坑提醒**：`CountVectorizer` 默认按空格分词，对中文不友好。**中文文本必须先分词并用空格连接**，否则会把整句当成一个词。

---

## 第三章：TF-IDF——给词语加权

### 3.1 为什么需要 TF-IDF

词袋模型有一个问题：它认为所有词都同等重要。但"的""是""了"这类词在几乎所有文档中都高频出现，对区分文档内容没有帮助。

**TF-IDF** 的核心思想：**一个词对某篇文档的重要性，正比于它在这篇文档中的出现频率，反比于它在整个语料库中的普遍程度**。

### 3.2 TF（词频，Term Frequency）

衡量一个词在**当前文档**中的出现频率。

```
TF = (某个词在文档中的出现次数) / (文档的总词数)
```

```python
def compute_tf(word, document):
    word_count = document.count(word)
    total_words = len(document)
    return word_count / total_words if total_words > 0 else 0

doc = ['猫', '喜欢', '鱼', '猫']
print(compute_tf('猫', doc))   # 2/4 = 0.5
print(compute_tf('鱼', doc)) # 1/4 = 0.25
```

### 3.3 IDF（逆文档频率，Inverse Document Frequency）

衡量一个词在**整个语料库**中的稀有程度。越稀有的词，IDF 越高。

```
IDF = log(语料库的文档总数 / (包含该词的文档数 + 1))
```

**为什么加 1**：防止分母为 0（某个词在所有文档中都没出现）。`log` 用于压缩数值范围。

```python
import math

def compute_idf(word, corpus):
    # corpus 是文档列表的列表
    doc_count = len(corpus)
    docs_containing_word = sum(1 for doc in corpus if word in doc)
    return math.log(doc_count / (docs_containing_word + 1)) + 1

corpus = [
    ['我', '喜欢', '猫'],
    ['我', '喜欢', '狗'],
    ['猫', '和', '狗', '都', '可爱']
]

print(compute_idf('喜欢', corpus))  # 出现2次，IDF较低
print(compute_idf('可爱', corpus))  # 出现1次，IDF较高
```

### 3.4 TF-IDF 计算

```
TF-IDF = TF × IDF
```

```python
def compute_tfidf(word, document, corpus):
    tf = compute_tf(word, document)
    idf = compute_idf(word, corpus)
    return tf * idf

doc = ['我', '喜欢', '猫', '猫']
print(f"猫: {compute_tfidf('猫', doc, corpus):.4f}")
print(f"喜欢: {compute_tfidf('喜欢', doc, corpus):.4f}")
```

**为什么这样写**："猫"在文档中出现 2 次（TF 高），但在 3 篇文档中出现了 2 篇（IDF 中等）。"可爱"只出现 1 次（TF 低），但只在 1 篇文档中出现（IDF 高）。TF-IDF 平衡了这两个因素。

### 3.5 sklearn 实战：TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["我 喜欢 猫 猫", "我 喜欢 狗", "猫 和 狗 都 可爱"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names_out())
print(X.toarray().round(4))
```

**输出解读**：
- 每行是一个文档的 TF-IDF 向量
- "猫"在第一篇文档中 TF-IDF 值最高（出现了两次，且不是所有文档都有）
- "喜欢"在每篇文档中值较低（因为三篇中有两篇包含它，IDF 被稀释）

### 3.6 TF-IDF 的局限性

| 局限 | 说明 |
|------|------|
| 语义缺失 | "猫"和"狗"的 TF-IDF 向量仍然正交，无法捕捉语义相似性 |
| 顺序无关 | 和 BoW 一样，完全忽略词序 |
| 稀疏高维 | 词典大时向量维度极高，大部分为 0 |
| 无法处理未登录词 | 训练时没见过的词，TF-IDF 无法表示 |

---

## 第四章：Word2Vec——让词语拥有"语义坐标"

### 4.1 分布式假设：上下文决定语义

Word2Vec 的核心理念来自**分布式假设（Distributional Hypothesis）**：

> **"一个词的含义，由它周围的词决定。"**

如果两个词经常出现在相似的上下文中，那么它们的语义应该相近。例如：
- "猫"经常出现在"宠物""可爱""抓老鼠"附近
- "狗"经常出现在"宠物""可爱""看门"附近

因此，"猫"和"狗"的向量应该在空间中距离很近。

### 4.2 Word2Vec 的两种架构

Word2Vec 由 Google 于 2013 年提出，包含两种训练模式：

| 模式 | 全称 | 输入 | 预测 | 适用场景 |
|------|------|------|------|----------|
| **CBOW** | Continuous Bag of Words | 上下文词 | 中心词 | 数据集较小、高频词多 |
| **Skip-gram** | - | 中心词 | 上下文词 | 数据集大、低频词多 |

**示例**（窗口大小为 2）：

```
句子: "我 喜欢 自然语言处理 和 深度学习"

窗口大小=2，中心词="自然语言处理":
上下文: ["我", "喜欢", "和", "深度学习"]

CBOW: 输入 ["我", "喜欢", "和", "深度学习"] → 预测 "自然语言处理"
Skip-gram: 输入 "自然语言处理" → 预测 ["我", "喜欢", "和", "深度学习"]
```

### 4.3 CBOW 模型详解

CBOW 通过**上下文词的平均向量**预测中心词。

**网络结构**：
- 输入层：上下文词的 One-Hot 向量（C 个词）
- 投影层：将 One-Hot 映射为稠密向量，取平均
- 输出层：Softmax 预测中心词

**目标函数**：

$$J_{\text{CBOW}} = \frac{1}{T} \sum_{t=1}^{T} \log p(w_t \mid \text{context}_t)$$

$$p(w_t \mid \text{context}_t) = \frac{\exp(\mathbf{v}_{w_t}^T \mathbf{h})}{\sum_{w \in V} \exp(\mathbf{v}_w^T \mathbf{h})}$$

其中 $\mathbf{h}$ 是上下文向量的平均，$V$ 是词汇表。

**为什么这样写**：CBOW 对上下文做平均，训练速度快，但对稀有词效果较差。

### 4.4 Skip-gram 模型详解

Skip-gram 通过**中心词**预测周围的上下文词，是**大语料下的首选**。

**目标函数**：

$$J_{\text{skip-gram}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} \mid w_t)$$

**为什么这样写**：Skip-gram 为每个上下文位置独立计算预测，对低频词更友好，在大数据集上表现更好。

### 4.5 负采样（Negative Sampling）——加速训练

原始 Word2Vec 的 Softmax 需要计算整个词汇表的归一化，当词汇表有几十万词时，计算量巨大。

**负采样**的思想：把多分类问题转化为二分类问题。

- **正样本**：真实的上下文词对（中心词 + 上下文词），标签为 1
- **负样本**：随机采样一些"噪声词"与中心词配对，标签为 0

模型只需要判断"这个词对是否真实共现"，无需计算整个 Softmax。

**为什么这样写**：负采样将计算复杂度从 $O(V)$ 降到 $O(K)$（$K$ 是负样本数，通常 5~20），使大规模训练成为可能。

### 4.6 用 gensim 训练中文词向量

```python
import jieba
from gensim.models import Word2Vec

# 准备语料（实际项目中语料越大越好）
sentences = [
    "自然语言处理是人工智能的重要分支",
    "深度学习在图像识别和自然语言处理中都有应用",
    "机器学习是人工智能的核心技术",
    "神经网络是深度学习的基础模型",
    "词向量是神经网络处理文本的基础",
]

# 分词
tokenized_sentences = [list(jieba.cut(sent)) for sent in sentences]

# 训练 Skip-gram 模型
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # 词向量维度
    window=5,             # 上下文窗口大小
    min_count=1,          # 最小词频（低于此频率的词被忽略）
    sg=1,                 # 1=Skip-gram, 0=CBOW
    negative=5,           # 负采样数
    epochs=100            # 训练轮数
)

# 获取"人工智能"的词向量
vector = model.wv['人工智能']
print(f"向量维度: {vector.shape}")
print(f"前5维: {vector[:5]}")

# 找最相似的词
similar = model.wv.most_similar('人工智能', topn=3)
print(f"与'人工智能'最相似的词: {similar}")
```

**参数详解**：

| 参数 | 含义 | 建议 |
|------|------|------|
| `vector_size` | 词向量维度 | 100~300，大数据集可用 300 |
| `window` | 上下文窗口 | 5~10，短文本用小值 |
| `min_count` | 最小词频 | 5~10，过滤稀有噪声词 |
| `sg` | 训练算法 | 1=Skip-gram（推荐），0=CBOW |
| `negative` | 负采样数 | 5~20，大数据集可用更大值 |
| `epochs` | 迭代次数 | 语料小则增大，如 100~1000 |

### 4.7 词向量的语义运算

训练好的词向量有一个神奇的特性：**语义关系可以通过向量运算表达**。

```python
# 类比推理： king - man + woman ≈ queen
# 中文示例： 北京 - 中国 + 美国 ≈ 华盛顿
try:
    result = model.wv.most_similar(
        positive=['北京', '美国'],
        negative=['中国'],
        topn=1
    )
    print(result)
except KeyError as e:
    print(f"语料太小，无法找到对应词: {e}")
```

**为什么这样写**：Word2Vec 学到的向量空间中，向量差值编码了"关系"。`北京 - 中国` 得到的是"首都关系"的向量，加上 `美国` 就指向 `华盛顿`。

---

## 第五章：三种方法对比总结

| 维度 | Bag-of-Words | TF-IDF | Word2Vec |
|------|-------------|--------|----------|
| **表示类型** | 离散、稀疏 | 离散、稀疏（加权） | 连续、稠密 |
| **维度** | 等于词典大小（万级） | 等于词典大小（万级） | 自定义（通常 100~300） |
| **语义信息** | ❌ 无 | ❌ 无 | ✅ 有 |
| **词序信息** | ❌ 无 | ❌ 无 | ❌ 无（局部窗口） |
| **未登录词** | ❌ 无法处理 | ❌ 无法处理 | ❌ 无法处理 |
| **计算复杂度** | 低 | 低 | 高（需训练） |
| **可解释性** | 高 | 高 | 低（黑盒） |
| **典型应用** | 文本分类、朴素贝叶斯 | 搜索引擎、关键词提取 | 语义相似度、深度学习输入 |

**为什么这样写**：没有最好的方法，只有最适合的方法。BoW 和 TF-IDF 在文本分类、关键词提取中仍然有效；Word2Vec 是深度学习的标准输入表示。

---

## 第六章：综合实战——新闻文本相似度分析

```python
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# 新闻标题
news = [
    "特斯拉发布新款电动汽车",
    "马斯克宣布特斯拉新车计划",
    "苹果发布新款iPhone手机",
    "中国女排夺得世界冠军",
]

# ========== 方法1: TF-IDF 相似度 ==========
print("=" * 40)
print("TF-IDF 相似度")
print("=" * 40)

# 分词并用空格连接
tokenized_news = [' '.join(jieba.cut(title)) for title in news]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(tokenized_news)

# 计算新闻0与其他新闻的相似度
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
for i, sim in enumerate(similarities[0], 1):
    print(f"新闻0 vs 新闻{i}: {sim:.4f}")

# ========== 方法2: Word2Vec 相似度 ==========
print("\n" + "=" * 40)
print("Word2Vec 相似度")
print("=" * 40)

# 训练Word2Vec（实际应用应使用大规模预训练模型）
tokenized = [list(jieba.cut(title)) for title in news]
w2v_model = Word2Vec(sentences=tokenized, vector_size=50, window=3, min_count=1, sg=1, epochs=100)

def get_sentence_vector(words, model):
    """通过平均词向量得到句子向量"""
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 计算句子向量
sentence_vectors = [get_sentence_vector(words, w2v_model) for words in tokenized]

similarities_w2v = cosine_similarity([sentence_vectors[0]], sentence_vectors[1:])
for i, sim in enumerate(similarities_w2v[0], 1):
    print(f"新闻0 vs 新闻{i}: {sim:.4f}")
```

**结果解读**：
- TF-IDF 下，新闻0和新闻1都包含"特斯拉"，相似度较高
- Word2Vec 下，如果语料足够大，"特斯拉"和"马斯克"的向量相近，新闻0和新闻1的相似度会更高
- 新闻0和新闻3（体育）的相似度应该最低

---

## 第七章：常见陷阱与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 中文直接用 `CountVectorizer` | 先用 jieba 分词，空格连接后再用 |
| Word2Vec 语料太小 | 至少几万条句子，或下载预训练词向量 |
| 忽略 `min_count` 参数 | 设置合理的 `min_count`（如 5），过滤噪声 |
| 混淆 CBOW 和 Skip-gram | 大数据集用 Skip-gram（`sg=1`），小数据集用 CBOW |
| 用 One-Hot 输入深度学习模型 | 改用 Word2Vec 或预训练 Embedding，节省内存 |
| 认为 Word2Vec 能处理词序 | Word2Vec 也忽略长距离词序，需用 RNN/Transformer |
| 未登录词直接报错 | 用 `model.wv.get(word, np.zeros(dim))` 安全取值 |
| 把 TF-IDF 向量直接当词向量 | TF-IDF 是文档向量，Word2Vec 是词向量，用途不同 |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **BoW** | 统计词频，高维稀疏，简单有效 |
| **One-Hot** | 最基础的词表示，仅用于理解，实际很少直接用 |
| **TF-IDF** | TF × IDF，降低常见词权重，突出关键词 |
| **Word2Vec** | 分布式表示，语义相似的词向量相近 |
| **CBOW** | 上下文预测中心词，适合小数据 |
| **Skip-gram** | 中心词预测上下文，适合大数据，推荐 |
| **负采样** | 加速训练，将 Softmax 转为二分类 |
| **gensim** | `Word2Vec()` 一行训练，`most_similar()` 查相似词 |

**养成的好习惯**：
1. 中文文本数值化前，**务必先分词**
2. 小数据集用 TF-IDF，大数据集用 Word2Vec
3. Word2Vec 训练时设置合理的 `min_count` 过滤噪声
4. 生产环境优先使用**预训练词向量**（如腾讯 AI Lab、搜狗词向量），而非自己训练
5. 未登录词用 `<UNK>` 或零向量处理，不要直接报错

**下一步预告**：掌握了文本数值化表示后，你将进入 **注意力机制** 的学习——理解 Query、Key、Value 的核心概念，掌握 Self-Attention 与 Multi-Head Attention 的计算原理与公式，这是理解 Transformer 和大语言模型的关键钥匙。继续加油！