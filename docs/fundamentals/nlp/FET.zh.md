# 文本特征提取技术

!!! info "文档来源"
    本文档改编自 Shantanu Sharma 的 [Natural Language Processing(NLP) Playlist — Chapter 2: Bag of Words, n-gram, TF-IDF](https://medium.com/@shantanu_sharma/natural-language-processing-nlp-playlist-chapter-2-bag-of-words-n-gram-tf-idf-458a9669a746#:~:text=While%20simple%2C%20Bag%20of%20Words,number%20of%20consecutive%20elements%20considered.)。
    
    原作者：[Shantanu Sharma](https://medium.com/@shantanu_sharma)

## 词袋模型（Bag of Words）

词袋模型（BoW）是自然语言处理（NLP）中一种常用的技术，用于将文本文档表示为数值向量。它关注词在文档中的**存在性**，而**忽略了词序和词之间的关系**。

![](https://miro.medium.com/v2/resize:fit:554/1*cNEEuxQs443qpPQvU1Z7iw.png)

将原始文本转换为词袋表示

以下是词袋模型的典型工作流程：

1. **分词**：首先将文本分解成单个词或标记。在这一步骤中，通常会移除标点符号和停用词（如 "and"、"the"、"is" 等常见词）；
2. **计数**：对文本中的每个唯一词进行计数或统计频率。这会生成一个类似字典的结构，其中每个词对应其在文本中出现的次数；
3. **向量化**：最后，每个文档（或文本样本）被表示为一个数值向量，向量中的每个元素对应特定词在文档中的计数。如果某个词在文档中不存在，其计数为零。

> 二元词袋模型（BBoW）是传统词袋模型（BoW）的一种变体。与 BoW 模型计算文档中每个词出现的频率不同，二元词袋模型仅表示一个词在文档中是否出现，即我们只标记 1 或 0 来表示特定词是否存在，而不考虑其频率。

```python
from sklearn.feature_extraction.text import CountVectorizer

# 示例语料库（文档集合）
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# 创建 CountVectorizer 实例
vectorizer = CountVectorizer()

# 将语料库拟合到向量化器并转换为词袋矩阵
bow_matrix = vectorizer.fit_transform(corpus)

# 获取唯一词列表（词汇表）
vocab = vectorizer.get_feature_names_out()

# 将词袋矩阵转换为密集 numpy 数组以便于操作
bow_matrix_dense = bow_matrix.toarray()

# 打印词袋矩阵和对应的词汇表
print("词袋矩阵：")
print(bow_matrix_dense)
print("\n词汇表：")
print(vocab)
```

输出结果：

![](https://miro.medium.com/v2/resize:fit:700/1*4UyehUcarxZm3kKvCvSKhg.png)

虽然简单，但词袋模型在文本分类、情感分析和文档聚类等任务中可以发挥有效作用。然而，当多个词具有相同频率时，我们无法确定哪个词更重要，因为它忽略了词的语义含义和上下文，这限制了它在更复杂的 NLP 任务中的效果。像 TF-IDF（词频-逆文档频率）和词嵌入这样的技术解决了其中的一些限制。

## N-gram 模型

N-gram 是从给定文本或语音样本中提取的 n 个连续项的序列。在自然语言处理（NLP）中，这些项通常是词、字符或符号。N-gram 用于 NLP 中的各种任务，包括语言建模、文本生成和特征提取。N-gram 中的 'n' 值决定了考虑的连续元素数量。

以下是不同类型的 n-gram：

1. **一元文法（1-gram）**：一元文法是单独考虑的单个词。例如，在句子 "I love natural language processing" 中，一元语法为 ['I', 'love', 'natural', 'language', 'processing']；
2. **二文语法（2-gram）**：二元文法由两个相邻词组成的序列。对于同一句子，二元语法为 ['I love', 'love natural', 'natural language', 'language processing']；
3. **三文语法（3-gram）**：三元文法是三个相邻词的序列。继续上面的例子，三元语法为 ['I love natural', 'love natural language', 'natural language processing']；
4. **N 文语法（N-gram）**：N 元文法指的是 N 个相邻元素的序列，这些元素可以是词、字符或符号。例如，如果 'N' 是 4，那么我们就有了 4-gram，也称为四元语法。

N-gram 的价值在于它们能够捕捉比单个词更多的上下文，特别是在词序重要的任务中。它们可以帮助完成语言建模等任务，其中预测句子中的下一个词依赖于前面的词。此外，N-gram 可用于识别文本中的常用短语或表达，有助于情感分析或文档分类等任务。

N-gram 可用于基于前面 n-1 个词或字符预测下一个词或字符来生成文本。这种方法常用于聊天机器人、文本摘要系统和内容生成工具。

然而，使用 N-gram 的主要挑战是随着 'n' 的增加，唯一组合的数量呈指数增长。这可能导致高维度、计算复杂性增加，以及稀疏性问题，特别是在处理大型词汇表或语料库时。

```python
import nltk
from nltk.util import ngrams

# 示例文本
text = "This is a sample sentence for generating n-grams."
# 将文本分词
tokens = nltk.word_tokenize(text)

# 定义 n-gram 的 'n' 值
n = 3  # 三元语法
# 生成 n-gram
ngrams_output = list(ngrams(tokens, n))

# 打印生成的 n-gram
print(ngrams_output)

# 输出结果：
"""
[('This', 'is', 'a'), ('is', 'a', 'sample'), ('a', 'sample', 'sentence'),
('sample', 'sentence', 'for'), ('sentence', 'for', 'generating'),
('for', 'generating', 'n-grams'), ('generating', 'n-grams', '.')]
"""
```

## TF-IDF 模型

TF-IDF（词频-逆文档频率）是一种用于自然语言处理和信息检索的数值统计方法，用于评估一个词在文档集合（语料库）中的某个文档内的重要性。TF-IDF 基于两个主要组成部分计算：词频（TF）和逆文档频率（IDF）。

以下是 TF-IDF 的计算方法：

1. **词频（TF）**：词频衡量一个词在文档中出现的频率。它计算为一个词在文档中出现的次数与文档中总词数的比率。这个想法是给文档中出现频率较高的词更高的权重。TF 的计算公式通常为：

```
TF(t, d) = 词 t 在文档 d 中出现的次数 / 文档 d 中的总词数
```

2. **逆文档频率（IDF）**：逆文档频率衡量一个词在整个文档集合中的独特性或稀有性。它计算为语料库中总文档数与包含该词的文档数之比的对数。IDF 的计算公式通常为：

```
IDF(t, D) = log(语料库中的总文档数 |D| / 包含词 t 的文档数)
```

3. **TF-IDF 计算**：文档 'd' 中词 't' 的 TF-IDF 分数通过将 't' 在 'd' 中的词频（TF）乘以 't' 在整个语料库中的逆文档频率（IDF）来计算。TF-IDF 的计算公式为：

```
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

这里，D 表示整个文档语料库。

TF-IDF 为在文档中**频繁出现**（高 TF）但在整个语料库中**较为罕见**（高 IDF）的词分配更高的权重，表明这些词在区分该文档与其他文档时的重要性。相反，在多个文档中频繁出现的常见词会被分配较低的权重。

TF-IDF 常用于各种文本处理任务，包括：

- 文档检索：基于查询的相关性对文档进行排序；
- 文本分类：提取用于训练机器学习模型的特征；
- 关键词提取：识别文档中的重要词语或短语；
- 信息检索：高效地索引和搜索文本文档。

总的来说，TF-IDF 是一种用于表示和评估文本数据中词语重要性的有用技术，有助于各种 NLP 和信息检索任务。

让我们通过一个简单的例子来理解：

**句子 1**：good boy

**句子 2**：good girl

**句子 3**：boy girl good

首先我们计算词频：**_good: 3, boy: 2, girl: 2_**

我们将计算 TF：

![](https://miro.medium.com/v2/resize:fit:700/1*QcqDIMXOTbvoxbiwg-dapw.png)

然后计算 IDF：

![](https://miro.medium.com/v2/resize:fit:442/1*gPLy_NeBwNPcyu-IcZYXzQ.png)

现在将最后两个表格相乘：

![](https://miro.medium.com/v2/resize:fit:700/1*LkYmwFwkz90mEqz3p8julA.png)

因此我们可以看到，对于句子 1，`boy` 的值比其他词高，所以这里有一些语义含义。

同样，对于句子 2，`girl` 被赋予了重要性，而对于句子 3：`boy` 和 `girl` 都很重要。

TF-IDF 的实现：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例语料库（文档集合）
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# 创建 TfidfVectorizer 实例
vectorizer = TfidfVectorizer()

# 将语料库拟合到向量化器并转换为 TF-IDF 矩阵
tfidf_matrix = vectorizer.fit_transform(corpus)

# 获取唯一词列表（词汇表）
vocab = vectorizer.get_feature_names_out()

# 将 TF-IDF 矩阵转换为密集 numpy 数组以便于操作
tfidf_matrix_dense = tfidf_matrix.toarray()

# 打印 TF-IDF 矩阵和对应的词汇表
print("TF-IDF 矩阵：")
print(tfidf_matrix_dense)
print("\n词汇表：")
print(vocab)
```

输出结果：

![](https://miro.medium.com/v2/resize:fit:700/1*F2IvA1mRBqeoR5i6yaZmvQ.png)

<h2 style="font-size: 1.2em">TF-IDF 的缺点：</h2>

- 由于词序可能不同，语义信息未被存储；
- TF-IDF 对不常见的词给予了重要性；
- 它不考虑文档中词的顺序。

为了克服 TF-IDF 的这些问题，我们可以使用[ word2vec 模型](word_embeddings_zh.md)。
