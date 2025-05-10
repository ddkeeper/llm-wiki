# Text Feature Extraction Techniques

!!! info "Document Source"
    This document is adapted Shantanu Sharma's [Natural Language Processing(NLP) Playlist — Chapter 2: Bag of Words, n-gram, TF-IDF](https://medium.com/@shantanu_sharma/natural-language-processing-nlp-playlist-chapter-2-bag-of-words-n-gram-tf-idf-458a9669a746#:~:text=While%20simple%2C%20Bag%20of%20Words,number%20of%20consecutive%20elements%20considered.).
    
    Original author: [Shantanu Sharma](https://medium.com/@shantanu_sharma)

## Bag of Words

Bag-of-words (BoW) is a common technique in natural language processing (NLP) for representing text documents as numerical vectors. It focuses on the **presence** of words in a document, **ignoring their order or relationships.**

![](https://miro.medium.com/v2/resize:fit:554/1*cNEEuxQs443qpPQvU1Z7iw.png)

Turning raw text into a bag of words representation

Here's how the Bag of Words model typically works:

1. **Tokenization**: The text is first broken down into individual words or tokens. Punctuation and stop words (common words like "and", "the", "is", etc.) are often removed during this step.
2. **Counting**: For each unique word in the text, a count or frequency is kept. This results in a dictionary-like structure where each word corresponds to a count of how many times it appears in the text.
3. **Vectorization**: Finally, each document (or text sample) is represented as a numerical vector, where each element of the vector corresponds to the count of a particular word in the document. If a word is absent from the document, its count is zero.

> Binary Bag of Words (BBoW) is a variation of the traditional Bag of Words (BoW) model. While the BoW model counts the frequency of each word occurrence in a document, the Binary Bag of Words model simply indicates whether a word is present or absent in the document i.e. we just mark 1 or 0 if a particular word is present or not respectively, disregarding its frequency.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus (collection of documents)
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer to the corpus and transform the corpus into a Bag of Words matrix
bow_matrix = vectorizer.fit_transform(corpus)

# Get the list of unique words (vocabulary)
vocab = vectorizer.get_feature_names_out()

# Convert the Bag of Words matrix to a dense numpy array for easier manipulation
bow_matrix_dense = bow_matrix.toarray()

# Print the Bag of Words matrix and the corresponding vocabulary
print("Bag of Words matrix:")
print(bow_matrix_dense)
print("\nVocabulary:")
print(vocab)
```

Output of above:

![](https://miro.medium.com/v2/resize:fit:700/1*4UyehUcarxZm3kKvCvSKhg.png)

While simple, Bag of Words can be effective for certain tasks such as text classification, sentiment analysis, and document clustering. However, if more than one word has same frequency, then we can not determine which word is more important, so it ignores the semantic meaning and context of words, which can limit its effectiveness for more complex NLP tasks. Techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings address some of these limitations.

## N-grams

N-grams are contiguous sequences of n items from a given sample of text or speech. In the context of natural language processing (NLP), these items are typically words, characters, or symbols. N-grams are used for various tasks in NLP, including language modeling, text generation, and feature extraction. The value of 'n' in n-grams determines the number of consecutive elements considered.

Here's a breakdown of different types of n-grams:

1. **Unigrams (1-grams)**: Unigrams are single words considered individually. For example, in the sentence "I love natural language processing," the unigrams would be ['I', 'love', 'natural', 'language', 'processing'].
2. **Bigrams (2-grams)**: Bigrams consist of sequences of two adjacent words. For the same sentence, the bigrams would be ['I love', 'love natural', 'natural language', 'language processing'].
3. **Trigrams (3-grams)**: Trigrams are sequences of three adjacent words. Continuing with the example sentence, the trigrams would be ['I love natural', 'love natural language', 'natural language processing'].
4. **N-grams (N-grams)**: N-grams refer to sequences of N adjacent elements, which can be words, characters, or symbols. For instance, if 'N' is 4, then we have 4-grams, also known as quadgrams.

N-grams are valuable because they capture more context than individual words, especially for tasks where word order is important. They can help in tasks like language modeling, where predicting the next word in a sentence relies on the preceding words. Additionally, n-grams can be used to identify common phrases or expressions in text, aiding in tasks such as sentiment analysis or document categorization.

N-grams can be used to generate text by predicting the next word or character based on the preceding n-1 words or characters. This approach is commonly used in chatbots, text summarization systems, and content generation tools.

However, the main challenge with using n-grams is the exponential increase in the number of unique combinations as 'n' increases. This can lead to high dimensionality, increased computational complexity, and issues with sparsity, especially when dealing with large vocabularies or corpora.

```python
import nltk
from nltk.util import ngrams

# Sample text
text = "This is a sample sentence for generating n-grams."
# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Define the desired value of 'n' for n-grams
n = 3  # for trigrams
# Generating n-grams
ngrams_output = list(ngrams(tokens, n))

# Print the generated n-grams
print(ngrams_output)

# Output of above:
"""
[('This', 'is', 'a'), ('is', 'a', 'sample'), ('a', 'sample', 'sentence'),
('sample', 'sentence', 'for'), ('sentence', 'for', 'generating'),
('for', 'generating', 'n-grams'), ('generating', 'n-grams', '.')]
"""
```

## TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used in natural language processing and information retrieval to evaluate the importance of a term (word) within a document relative to a collection of documents (corpus). TF-IDF is calculated based on two main components: term frequency (TF) and inverse document frequency (IDF).

Here's a breakdown of how TF-IDF is calculated:

1. **Term Frequency (TF)**: Term frequency measures how frequently a term appears in a document. It is calculated as the ratio of the number of times a term occurs in a document to the total number of terms in the document. The idea is to give higher weight to terms that appear more frequently within a document. The formula for TF is typically:

```
TF(t, d) = Number of times term t appears in document d / Total number of terms in document d
```

2. **Inverse Document Frequency (IDF)**: Inverse document frequency measures how unique or rare a term is across a collection of documents. It is calculated as the logarithm of the ratio of the total number of documents in the corpus to the number of documents containing the term. The formula for IDF is typically:

```
IDF(t, D) = log⁡(Total number of documents in the corpus ∣D∣ / Number of documents containing term t)
```

3. **TF-IDF Calculation**: The TF-IDF score for a term 't' in a document 'd' is calculated by multiplying the term frequency (TF) of 't' in 'd' by the inverse document frequency (IDF) of 't' across the entire corpus. The formula for TF-IDF is:

```
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

Here, D represents the entire corpus of documents.

TF-IDF assigns higher weights to terms that are **frequent within a document** (high TF) but **rare across the entire corpus** (high IDF), indicating their importance in distinguishing that document from others. Conversely, common terms that appear frequently across many documents are assigned lower weights.

TF-IDF is commonly used in various text processing tasks, including:

- Document retrieval: To rank documents based on their relevance to a query.
- Text classification: To extract features for training machine learning models.
- Keyword extraction: To identify important terms or phrases within a document.
- Information retrieval: To index and search text documents efficiently.

Overall, TF-IDF is a useful technique for representing and evaluating the importance of terms in text data, aiding in various NLP and information retrieval tasks.

Let's understand it using a simple example:

**Sentence 1**: good boy

**Sentence 2**: good girl

**Sentence 3**: boy girl good

First we calculate, words Frequency: **_good: 3, boy: 2, girl: 2_**

We will calculate TF:

![](https://miro.medium.com/v2/resize:fit:700/1*QcqDIMXOTbvoxbiwg-dapw.png)

Then we will calculate IDF:

![](https://miro.medium.com/v2/resize:fit:442/1*gPLy_NeBwNPcyu-IcZYXzQ.png)

Now multiply last two tables:

![](https://miro.medium.com/v2/resize:fit:700/1*LkYmwFwkz90mEqz3p8julA.png)

Hence we can see that, for Sentence 1, `boy` value is higher compared to other words. so there is some semantic meaning

And similarly, For Sentence 2 `girl` is given importance and for Sentence 3: `boy` and `girl`

Implementation of TF-IDF:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus (collection of documents)
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

# Create an instance of TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the corpus and transform the corpus into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(corpus)

# Get the list of unique words (vocabulary)
vocab = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense numpy array for easier manipulation
tfidf_matrix_dense = tfidf_matrix.toarray()

# Print the TF-IDF matrix and the corresponding vocabulary
print("TF-IDF matrix:")
print(tfidf_matrix_dense)
print("\nVocabulary:")
print(vocab)
```

Output of above:

![](https://miro.medium.com/v2/resize:fit:700/1*F2IvA1mRBqeoR5i6yaZmvQ.png)

<h2 style="font-size: 1.2em">Disadvantages of TF-IDF: </h2>

- Semantic info is not stored, as order of words may not be same.
- TF-IDF gives importance to uncommon words.
- It doesn't consider the order of words within a document.

To overcome these problems of TF-IDF, we can use [ word2vec ](word_embeddings_en.md) model.
