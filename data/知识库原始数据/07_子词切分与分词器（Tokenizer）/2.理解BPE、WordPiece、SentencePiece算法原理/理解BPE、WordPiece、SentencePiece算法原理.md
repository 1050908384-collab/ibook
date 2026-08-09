# 理解 BPE、WordPiece、SentencePiece 算法原理

权威文章：《Tokenizers》  
　　网址：https://huggingface.co/docs/tokenizers

权威文章：《Tokenization algorithms》  
　　网址：https://huggingface.co/docs/transformers/en/tokenizer_summary

① 底稿：《文本分词算法：Byte-Pair Encoding (BPE)、WordPiece 和 SentencePiece》  
　　网址：https://blog.csdn.net/shanglianlm/article/details/162464312

② 参考文章：《【大语言模型 22】Tokenization深度技术：BPE、WordPiece、SentencePiece完全指南》  
　　网址：https://blog.csdn.net/maoyu_dual/article/details/150786404

> **适合人群**：已完成"环境准备：在 Python 环境中安装 transformers、tokenizers 库"章节，能够成功导入 `transformers` 和 `tokenizers` 的初学者。本节时长约 1.5 小时，目标是理解现代大语言模型使用的三种核心子词分词算法（BPE、WordPiece、SentencePiece）的原理与区别，能够用手写代码模拟算法流程，为下一节实际加载和使用 HuggingFace Tokenizer 建立理论基础。（①+②）
>
> **前置说明**：在将文本输入大语言模型之前，必须先将其转换为模型能够计算的数字序列，这一过程称为**分词（Tokenization）**。现代大模型通常采用**子词分词（Subword Tokenization）**：高频词保留完整，低频词拆成有意义的子词片段。本节将深入讲解三种主流子词分词算法的原理。所有代码请在已安装 PyTorch 和 transformers 的 `pytorch` conda 环境中运行。

## 一、为什么需要子词分词？（①+②）

在讲算法之前，先看两种极端方案的问题：（①）

- **词级分词（Word-level）**：把每个单词作为一个 token。问题是词汇量巨大（英语就有数十万个单词），而且遇到新词（如"Tokenization"）只能映射到 `<unk>`，模型完全无法理解。
- **字符级分词（Character-level）**：把每个字符作为一个 token。词汇量很小，但序列长度暴增，且单个字符（如"l"）携带的语义信息远少于"love"。

**子词分词**是两者的折中：高频词保持完整（如"agent"），低频词或复合词拆成子词（如"Tokenization" → `["Token", "ization"]`）。这样既能控制词汇表规模，又能通过子词组合理解未登录词。（①+②）

> **本地运行配置要求**：本节代码均为纯 Python 算法演示，无需 GPU，CPU 即可运行。内存建议 ≥ 4GB。

## 二、BPE：从字符开始，贪婪合并（①+②）

### 2.1 核心思想（①+②）

Byte Pair Encoding（BPE）最初是一种数据压缩算法，后被引入 NLP。核心思想非常简单：（①）

1. 从**字符级**开始，将每个词拆成字符序列。
2. 统计所有**相邻字符对**的出现频率。
3. 将**频率最高**的那一对合并成一个新符号。
4. 重复步骤 2-3，直到词表达到预定大小。

### 2.2 算法演示（①）

假设语料库中有以下词频数据：

| 词 | 频率 |
|---|---|
| `low` | 5 |
| `lower` | 2 |
| `newest` | 6 |
| `widest` | 3 |

**初始化**：每个词拆成字符序列，末尾加 `</w>` 标记词边界。

```
l o w </w>    (5次)
l o w e r </w>    (2次)
n e w e s t </w>    (6次)
w i d e s t </w>    (3次)
```

**第 1 轮合并**：统计相邻对，`(e, s)` 出现 6+3=9 次，频率最高，合并为 `es`。

```
l o w </w>
l o w e r </w>
n e w es t </w>
w i d es t </w>
```

**第 2 轮合并**：`(es, t)` 出现 6+3=9 次，合并为 `est`。

```
l o w </w>
l o w e r </w>
n e w est </w>
w i d est </w>
```

**继续迭代**，直到达到预设的合并次数或词表大小。

### 2.3 Python 实现（①）

```python
from collections import Counter

def get_pair_stats(vocab):
    """统计所有相邻符号对的频率"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """将指定符号对在词汇表中合并"""
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def bpe_train(word_freqs, num_merges):
    """训练 BPE，返回合并列表和最终词汇表"""
    # 初始化：拆分为字符序列，末尾加 </w>
    vocab = {' '.join(list(word)) + ' </w>': freq
             for word, freq in word_freqs.items()}
    merges = []

    for i in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

    return merges, vocab

# 使用演示
word_freqs = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges, final_vocab = bpe_train(word_freqs, num_merges=10)
print("合并顺序:", merges)
print("最终词汇表示例:", list(final_vocab.items())[:5])
```

### 2.4 编码新词（①）

训练完成后，对新词分词时，按照学到的**合并顺序**依次应用规则。例如对 `lowest`：

```
初始：l o w e s t </w>
应用 (e, s) → es：l o w es t </w>
应用 (es, t) → est：l o w est </w>
结果：["l", "o", "w", "est"]
```

### 2.5 Byte-level BPE（②）

标准 BPE 以 Unicode 字符为基，但包含所有 Unicode 字符会使基词表过大。GPT-2 使用**Byte-level BPE**：以 256 个字节值为基词表，确保任何文本都能被分词，避免出现 `<unk>`。例如 `openai-community/gpt2` 的词表大小为 50,257（256 字节 token + 50,000 次合并 + 1 个特殊 token）。

## 三、WordPiece：最大化语言模型似然（①+②）

### 3.1 与 BPE 的核心区别（①+②）

WordPiece 与 BPE 非常相似，都是自底向上迭代合并，但**合并准则不同**：（①+②）

- **BPE**：选**频率最高**的字符对。
- **WordPiece**：选**最大化训练数据似然**的字符对。

具体得分公式为（教学简化版，实际实现可能更复杂）：（①）

```
score(x, y) = freq(x, y) / (freq(x) × freq(y))
```

这个分数衡量的是：两个符号"绑定在一起"的频率，相对于它们"各自独立出现"的频率之比。分数越高，说明这对符号越应该被合并。

### 3.2 算法演示（①）

> **以下是一个独立简化示例**，使用另一组假设语料来演示 WordPiece 的得分计算逻辑，与前文 BPE 演示的语料不同。

沿用另一组假设语料，WordPiece 计算各字符对的得分：

| 字符对 | freq(x,y) | freq(x) | freq(y) | score |
|---|---|---|---|---|
| `u` + `g` | 20 | 36 | 20 | 0.028 |
| `g` + `s` | 5 | 20 | 5 | **0.050** |

虽然 `(u, g)` 的共现频率更高，但 `(g, s)` 的 score 更高（因为 `g` 和 `s` 单独出现的频率较低，它们"绑定"的相对强度更大）。因此 WordPiece 会优先合并 `(g, s)`。（①）

### 3.3 编码方式：贪心最长匹配（①）

与 BPE 按合并规则编码不同，WordPiece 在编码时采用**贪心最长匹配**：从左到右，每次匹配词表中最长的子词。（①）

此外，WordPiece 用 `##` 前缀标记非词首的子词，防止跨词边界的歧义合并。例如 `playing` 可能被切分为 `["play", "##ing"]`。

### 3.4 Python 实现（①）

```python
from collections import Counter

def wordpiece_train(word_freqs, num_merges):
    """简化的 WordPiece 训练"""
    # 首字符不加 ##，后续字符加 ##
    vocab = {}
    for word, freq in word_freqs.items():
        chars = word[0]
        for c in word[1:]:
            chars += ' ##' + c
        chars += ' </w>'
        vocab[chars] = freq

    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        symbol_freq = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for s in symbols:
                symbol_freq[s] += freq
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq

        if not pairs:
            break

        # 计算 score，选最大
        best_pair = max(pairs, key=lambda p: pairs[p] /
                        (symbol_freq[p[0]] * symbol_freq[p[1]]))
        merges.append(best_pair)

        bigram = ' '.join(best_pair)
        replacement = ''.join(best_pair)
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        vocab = new_vocab

    return merges, vocab

word_freqs = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges, vocab = wordpiece_train(word_freqs, num_merges=10)
print("WordPiece 合并顺序:", merges)
```

## 四、SentencePiece：语言无关的端到端分词（①+②）

### 4.1 核心创新（①+②）

BPE 和 WordPiece 的传统实现通常依赖**预分词**（如按空格切分），这对中文、日文等无空格语言不友好。SentencePiece 的核心创新是：（①+②）

1. **将空格视为普通字符**：用 `▁`（U+2581，SentencePiece 专用空格标记）表示空格，直接对经过规范化的原始文本流进行操作。
2. **不依赖预分词**：把输入视为经过规范化的原始文本流，无需预先按空格切分。
3. **支持多种算法**：内置 BPE 和 Unigram 两种算法。

> **关于 byte fallback**：SentencePiece 支持将未登录字符回退到字节序列（byte fallback），但这是一种可选配置（如 `--byte_fallback` 标志），并非默认行为。默认情况下，SentencePiece 仍依赖词表覆盖文本中的字符。此外，`▁`（U+2581）是 SentencePiece 的专用空格标记，与普通下划线 `_` 不同；下文手写示例中若使用 `_`，仅作为演示替代符号。

### 4.2 Unigram 算法原理（①+②）

SentencePiece 最具代表性的算法之一是 **Unigram 语言模型**：（①+②）

1. 从一个**很大的候选词表**出发（包含所有可能的子串）。
2. 假设每个子词的出现相互独立，句子的概率为各子词概率的乘积。
3. 通过 **EM 算法**（期望最大化）迭代优化：
   - **E 步**：计算每个句子所有可能分割下各子词的期望出现次数。
   - **M 步**：用期望计数重新估计子词概率。
4. 每轮剪掉概率最低的 10%~20% 子词，直到词表达到目标大小。

编码时，使用**维特比算法**（动态规划）寻找概率最大的分割路径。（①）

### 4.3 多语言示例（②）

| 语言 | 输入 | 分词结果 |
|---|---|---|
| 英语 | `Hello world` | `["▁He", "llo", "▁world"]` |
| 中文 | `你好世界` | `["你", "好", "世界"]` |
| 日文 | `こんにちは世界` | `["こん", "にちは", "世界"]` |
| 混合 | `Hello你好` | `["▁He", "llo", "你", "好"]` |

> 注：表格中的 `▁` 为 SentencePiece 专用空格标记（U+2581）。下文手写代码示例中使用普通下划线 `_` 仅为演示便利，二者不等同。

### 4.4 真实训练：使用 HuggingFace tokenizers（推荐）

以下代码使用 HuggingFace `tokenizers` 库演示真实的 Unigram 训练流程，与生产环境一致：

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 准备语料（实际项目中应为大量文本）
texts = [
    "low lower newest widest",
    "lowering lowers new widely",
    "hello world",
    "natural language processing"
]

# 初始化 Unigram 模型
tokenizer = Tokenizer(models.Unigram())

# 使用 Metaspace 预分词器：将空格替换为 ▁（U+2581）
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# 配置训练器
trainer = trainers.UnigramTrainer(
    vocab_size=50,
    unk_token="<unk>",
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
    show_progress=True,
)

# 从迭代器训练
tokenizer.train_from_iterator(texts, trainer=trainer)

# 测试编码
output = tokenizer.encode("lowest")
print("Tokens:", output.tokens)
print("IDs:", output.ids)
```

> **本地运行配置要求**：需安装 `tokenizers` 库（`pip install tokenizers`），纯 CPU 运行，内存 ≥ 4GB。

### 4.5 手写简化版：理解核心逻辑（教学演示）

> **⚠️ 重要提示**：以下代码是**极度简化的教学演示，不是完整的 EM 实现**。真实的 SentencePiece 包含更复杂的平滑、回退、子词采样和优化策略。代码中使用普通下划线 `_` 代替 SentencePiece 的专用空格标记 `▁`（U+2581），仅为手写示例便利。

```python
from collections import Counter

def init_vocab(texts, max_len=3):
    """收集所有长度 <= max_len 的子串作为初始候选词汇。
    保留所有子串（包括仅出现1次的），避免基础字符被过早剪枝导致词表为空。"""
    vocab = Counter()
    for text in texts:
        # 注意：这里用普通下划线 _ 仅作为手写示例的替代符号，
        # 实际 SentencePiece 使用的是专用空格标记 ▁ (U+2581)
        text = text.replace(' ', '_')
        for i in range(len(text)):
            for j in range(1, max_len+1):
                if i+j <= len(text):
                    vocab[text[i:i+j]] += 1
    return dict(vocab)

def viterbi_segment(text, vocab_probs):
    """用动态规划找概率最大的分割路径。
    若不存在覆盖全句的有效路径，则 fallback 为逐字符切分。"""
    n = len(text)
    dp = [0.0] * (n+1)
    dp[0] = 1.0
    back = [None] * (n+1)

    for i in range(n):
        if dp[i] == 0:
            continue
        for j in range(i+1, n+1):
            sub = text[i:j]
            if sub in vocab_probs:
                prob = dp[i] * vocab_probs[sub]
                if prob > dp[j]:
                    dp[j] = prob
                    back[j] = i

    # 无有效路径时 fallback 到逐字符切分
    if dp[n] == 0:
        return list(text)
    
    segments = []
    idx = n
    while idx > 0:
        prev = back[idx]
        if prev is None:
            return list(text)
        segments.append(text[prev:idx])
        idx = prev
    return segments[::-1]

def train_unigram(texts, vocab_size=100, num_iters=10):
    """简化的 Unigram 训练（教学演示版，非完整 EM 实现）。"""
    raw_vocab = init_vocab(texts, max_len=4)
    vocab = list(raw_vocab.keys())
    base_chars = {w for w in vocab if len(w) == 1}
    total_count = sum(raw_vocab.values())
    probs = {w: raw_vocab[w] / total_count for w in vocab}

    for it in range(num_iters):
        counts = Counter()
        total = 0
        for text in texts:
            text = text.replace(' ', '_')
            seg = viterbi_segment(text, probs)
            for sub in seg:
                counts[sub] += 1
                total += 1

        if total == 0:
            break

        # 为基础字符保留极小的平滑计数，确保它们不会因本轮未被选中而消失
        for char in base_chars:
            if char in vocab and counts[char] == 0:
                counts[char] = 1e-8
        normalizer = sum(counts[w] for w in vocab)
        probs = {w: counts[w] / normalizer for w in vocab if counts[w] > 0}

        # 剪枝：优先删除低概率的长子词，保护单字符不被剪光
        while len(probs) > vocab_size:
            deletable = {w: p for w, p in probs.items() if len(w) > 1}
            if not deletable:
                break
            worst = min(deletable, key=deletable.get)
            del probs[worst]

        vocab = list(probs.keys())
        print(f"Iter {it+1}, vocab size: {len(vocab)}")

    return probs

# 使用演示：使用稍丰富的语料，避免词表过小导致退化
texts = [
    "low lower newest widest",
    "lowering lowers new widely",
    "hello world hello world"
]
probs = train_unigram(texts, vocab_size=30, num_iters=10)
print("最终词汇 (按概率排序):", 
      sorted(probs.keys(), key=lambda x: (-probs[x], x))[:20])
```

## 五、三种算法对比与典型模型示例（①+②）

### 5.1 核心差异速查（①+②）

| 特性 | BPE | WordPiece | SentencePiece (Unigram) |
|---|---|---|---|
| **合并准则** | 频率最高 | 似然最大化 | 概率模型 + EM 算法 |
| **预分词依赖** | 需要（按空格） | 需要（按空格） | **不需要**（经过规范化的原始文本流） |
| **空格处理** | 预分词时处理 | 预分词时处理 | 编码为 `▁` 加入词表 |
| **多语言支持** | 中等 | 中等 | **优秀** |
| **编码方式** | 按合并规则 | 贪心最长匹配 | 维特比算法 |
| **训练速度** | 快 | 中等 | 慢 |

### 5.2 典型模型示例（并非所有版本都适用）（②）

以下列举了一些常见模型的分词方案，但**不同版本或变体可能采用不同策略**，具体以模型仓库配置为准：

- **`openai-community/gpt2`**：使用 **Byte-level BPE**。以 256 字节为基词表，词表大小约 5 万，能处理任何 Unicode 字符包括表情符号。（②）
- **`google-bert/bert-base-uncased`**：使用 **WordPiece**。词表大小 30,522，用 `##` 标记非词首子词。例如 `unaffable` → `["un", "##aff", "##able"]`。（②）
- **`FacebookAI/xlm-roberta-base`**、`**google-t5/t5-base**`：使用 **SentencePiece + Unigram**。词表大小通常在 10 万~25 万量级，同时处理多种语言，无需语言特定的预处理。（②）

### 5.3 如何选择？（②）

| 场景 | 通常推荐的算法 |
|---|---|
| 英语单语言模型 | WordPiece（BERT 风格）或 BPE（GPT 风格） |
| 形态丰富的语言（德语、俄语） | BPE |
| 无空格语言（中文、日文） | **SentencePiece** |
| 多语言模型 | **SentencePiece**（常见选择之一） |
| 追求训练速度 | BPE |

> **注意**：上述建议为一般性参考，实际选型需结合具体模型版本、语料特点和实验结果综合判断。

## 六、词表大小的选择（②）

词表大小是一个关键超参数：（②）

- **太小**（如 < 5,000）：序列过长，信息碎片化，模型效率低。
- **太大**（如 > 200,000）：增加模型参数，embedding 层过大，降低泛化能力。

**一般建议**（在特定条件下，具体需实验验证）：（②）

| 模型类型 | 通常推荐的词表大小范围 |
|---|---|
| 单语言模型 | 1 万 ~ 5 万 |
| 多语言模型 | 5 万 ~ 25 万 |

> 例如 `openai-community/gpt2` 词表 50,257，`google-bert/bert-base-uncased` 词表 30,522，`meta-llama/Llama-2-7b-hf` 词表 32,000。

## 七、常见问题与解决（①+②）

| 问题 | 解答 |
|---|---|
| BPE 和 WordPiece 哪个更好？ | 没有绝对优劣。在特定条件下，BPE 简单高效，WordPiece 子词更具语言学意义。现代生成模型（如 GPT-2、Llama 系列）多用 BPE，理解模型（如 BERT）多用 WordPiece。 |
| 为什么中文模型也用 BPE？ | 中文虽然无空格，但可以通过预分词（如按字或按词）后应用 BPE。更现代的做法是直接使用 SentencePiece。 |
| `<unk>` token 还会出现吗？ | Byte-level tokenizer 或启用 byte fallback 的 tokenizer 通常能覆盖未知字节；普通 SentencePiece 是否产生 `<unk>` 取决于具体配置（如是否启用 byte fallback、词表覆盖范围等）。标准 BPE 和 WordPiece 在基词表覆盖不全时仍可能出现。 |
| 可以自己训练词表吗？ | 可以。HuggingFace `tokenizers` 库提供了 `BpeTrainer`、`WordPieceTrainer`、`UnigramTrainer`。但通常直接加载预训练模型的词表即可。 |

## 八、总结与后续学习路线（①+②）

恭喜你完成了 BPE、WordPiece、SentencePiece 算法原理的学习！你现在应该掌握了：

-   ✅ 子词分词的核心思想：在词汇量与语义表达之间取平衡
-   ✅ **BPE**：自底向上贪婪合并最高频字符对，GPT-2 等模型使用
-   ✅ **WordPiece**：基于似然最大化选择合并对，BERT 等模型使用
-   ✅ **SentencePiece (Unigram)**：语言无关的端到端分词，XLM-R、mT5 等模型使用
-   ✅ 三种算法的核心差异、适用场景和实际应用案例
-   ✅ 词表大小选择的一般建议

> **下一步预告**：在下一节 **"掌握 HuggingFace Tokenizer 的加载与使用"** 中，我们将用今天学到的理论知识，实际操作 HuggingFace 的 `AutoTokenizer`。你会加载 GPT-2 的 BPE 分词器、BERT 的 WordPiece 分词器，观察它们对同一段文本的不同切分结果，并学习 `encode`、`decode`、`batch_encode_plus` 等核心 API。现在，试着把上面的 BPE 实现代码跑一遍，用你自己的语料（如几个英文单词）观察合并过程吧！
