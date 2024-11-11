# Neural Word Sense Disambiguation and Causal Tracing Project [October - November 2024]

Welcome to my **Neural Word Sense Disambiguation and Causal Tracing** project! This assignment is an exploration into natural language processing techniques, such as word sense disambiguation (WSD) using both traditional algorithms and modern neural models like BERT. Additionally, it delves into understanding how large language models, such as GPT-2, process and store information through causal tracing.

## Project Overview

The project is divided into three main sections, each focusing on different techniques and methods:

### 1. Word Sense Disambiguation Using WordNet and NLTK
- **Goal**: Familiarize with WordNet's lexical database and implement foundational WSD algorithms, such as the Most Frequent Sense (MFS) and the simplified Lesk algorithm.
- **Tasks**:
  - Extract and analyze synsets using WordNet.
  - Implement the simplified Lesk algorithm and its extensions to improve sense disambiguation accuracy.
  - Use tokenization and stop word removal techniques to enhance text processing.

### 2. Word Sense Disambiguation Using Neural Models
- **Goal**: Use word embeddings and contextual models to improve WSD accuracy by leveraging semantic properties of modern word vectors, including word2vec and BERT.
- **Tasks**:
  - Compare cosine similarity scores between context and sense signatures using word2vec vectors.
  - Generate sense vectors using BERT and apply a nearest neighbor approach for disambiguation.
  - Analyze the performance differences between methods, including the impact of lowercasing and word order.

### 3. Understanding Transformers Through Causal Tracing
- **Goal**: Understand how autoregressive Transformer models, like GPT-2, process and generate language by analyzing internal state contributions using causal tracing.
- **Tasks**:
  - Implement causal tracing to identify the impact of individual model components (e.g., hidden states, MLPs, attention layers) on output generation.
  - Compare causal patterns across different GPT-2 model sizes.
  - Investigate prompts that yield different causal tracing outcomes to infer how LLMs store factual knowledge.

## Components and How They Work

### 1. **WordNet and NLTK**
- **Synset Analysis**: Use `nltk` and WordNet to explore hyperonyms, hyponyms, and other semantic relationships.
- **Simplified Lesk Algorithm**: Implement the algorithm using tokenized definitions and examples, calculating overlaps to select the correct word sense.
- **Extended Lesk Variants**: Add information from hyponyms, holonyms, and meronyms to improve performance.

### 2. **Neural WSD with word2vec and BERT**
- **Word Embeddings**: Use pre-trained word2vec embeddings to compute cosine similarity scores for sense disambiguation.
- **Contextual Models**: Implement BERT-based methods to utilize token-specific vectors for more accurate WSD.
- **Nearest Neighbor Search**: Average BERT vectors from a sense-annotated training set to identify the most similar sense for ambiguous words.

### 3. **Causal Tracing with GPT-2**
- **Causal Analysis**: Examine the impact of specific model states on output generation using corrupted and restored hidden states.
- **Impact of Model Size**: Compare how the size of GPT-2 models affects their causal tracing patterns and ability to recover from corruption.
- **Prompt Analysis**: Explore prompts that either highlight or obscure causal tracing patterns, hypothesizing about language model behavior.

## Tools and Libraries

- **Programming Language**: Python
- **Libraries**: 
  - `nltk` for lexical analysis
  - `numpy` for vector operations
  - `transformers` and `torch` for working with BERT and GPT-2 models
- **Hardware**: Running BERT-based models can be resource-intensive, so a GPU is recommended for faster processing.

## How to Run

### Prerequisites
1. **Install Dependencies**:
   ```bash
   pip install nltk torch transformers
   ```
2. **Download WordNet Data**:
   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

### Running the Code
1. **WordNet and Lesk Algorithms**:
   - Execute the implementations in `ntlk.py` and `lesk.py` using Python 3.
   ```bash
   python3 ntlk.py
   python3 lesk.py
   ```

2. **BERT-Based WSD**:
   - Run `bert.py` to generate BERT-based word sense predictions.
   ```bash
   python3 bert.py
   ```
   - For GPU processing, use `gpu-run-bert.sh` on a supported server.

3. **Causal Tracing with GPT-2**:
   - Implement and run causal tracing experiments in `gpt2-causal-tracing.py`.
   ```bash
   python3 gpt2-causal-tracing.py
   ```

### Submission and Evaluation
- **Submit via MarkUs**: Include all required Python files (`ntlk.py`, `lesk.py`, `bert.py`, `gpt2-causal-tracing.py`) and a written report (`a2written.pdf`).

## Key Challenges and Learnings

1. **Understanding Lexical Semantics**: Working with WordNet deepened my understanding of semantic relationships and the complexity of WSD.
2. **Vector Representations**: Implementing cosine similarity and averaging vectors provided insight into the strengths and limitations of different representation techniques.
3. **Analyzing Model Behavior**: Causal tracing offered a unique perspective on how language models process and generate language, revealing the intricate workings of neural architectures.

## Future Work and Extensions

- **Incorporate More Sophisticated Models**: Experiment with models like RoBERTa or GPT-3 for better performance.
- **Explore Cross-Linguistic WSD**: Apply the same techniques to multilingual datasets and analyze performance differences.
- **Optimize for Speed**: Implement caching and parallel processing to improve runtime efficiency.

## Notes and Tips

- **Performance Optimization**: Sorting batches by length before processing speeds up BERT-based computations.
- **Memory Management**: When using a GPU, monitor memory usage to prevent overflow issues, especially with large models.
- **Debugging**: Use smaller datasets and debug-friendly modes for quicker iterations during development.

---

This project has been a deep dive into the world of natural language processing and neural models, providing valuable insights into both traditional and cutting-edge techniques. It’s been a fascinating experience, and I hope you find the methods and results as intriguing as I did!
