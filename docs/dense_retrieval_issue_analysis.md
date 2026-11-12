# Dense Retrieval 检索问题分析与优化方向

## 1. 问题背景

在 PaperPilot-RAG 项目的 Day 4 阶段，我们首次实现了基于 Embedding 和余弦相似度的 Dense Retrieval，并使用 QASPER 数据集中的真实科研论文进行了检索实验。

本次实验选取论文：

**Minimally Supervised Learning of Affective Events Using Discourse Relations**

用户提出的问题：

```text
What is the seed lexicon?
```

使用模型：

```text
sentence-transformers/all-MiniLM-L6-v2
```

检索配置：

- Chunking：Section-aware Chunking
- Chunk Size：500
- Chunk Overlap：0
- Chunk 数量：48
- Embedding 维度：384
- 相似度计算：Cosine Similarity
- 检索策略：Top-K Dense Retrieval

## 2. 实验结果

Embedding 输出形状：

```python
chunk_embeddings.shape == (48, 384)
query_embedding.shape == (384,)
```

Top-3 检索结果：

| 排名 | 相似度 | 章节 | 主要内容 |
|---|---:|---|---|
| Rank 1 | 0.6330 | Automatically Labeled Pairs | Seed Lexicon 在事件对标注中的使用规则 |
| Rank 2 | 0.5820 | Cause Pairs | Cause 类型事件对的极性假设 |
| Rank 3 | 0.5813 | AL, CA, and CO | Seed Lexicon 由 15 个正向词和 15 个负向词组成 |

其中，Rank 3 的内容：

> We constructed our seed lexicon consisting of 15 positive words and 15 negative words.

比 Rank 1 更直接地回答了用户的问题。

但是，Dense Retrieval 将 Rank 1 排在了 Rank 3 前面。

## 3. 核心问题：Semantic Similarity 不等于 Answer Relevance

本次实验暴露了 Dense Retrieval 的一个重要局限：

**语义相似度最高的文本，不一定是最能回答用户问题的文本。**

需要区分两个概念：

### 3.1 Semantic Similarity（语义相似度）

表示 Query 和 Document 在语义空间中的接近程度。

Embedding 模型通过将文本转换为向量，再计算向量之间的余弦相似度，实现语义检索。

### 3.2 Answer Relevance（答案相关性）

表示检索到的文本能否有效回答用户提出的问题。

例如：

用户询问：

```text
What is the seed lexicon?
```

Rank 1 主要介绍 Seed Lexicon 在 Automatically Labeled Pairs 中的使用方式。

Rank 3 则直接说明 Seed Lexicon 的组成。

因此，Rank 3 虽然相似度较低，却具有更高的直接答案相关性。

需要注意，余弦相似度分数并不是答案正确率或概率，不能将 0.6330 理解为系统有 63.3% 的概率回答正确。

## 4. 问题产生的可能原因

结合本次实验，可以从以下方面分析：

1. **关键词与上下文关联**

   Rank 1 包含大量与 Query 相关的概念，例如 seed lexicon、event、score 和 discourse relation，因此可能获得较高的语义相似度。

2. **问题意图识别不足**

   当前 Dense Retrieval 主要衡量文本语义接近程度，并没有显式区分用户是在询问概念定义、使用方法还是实验结果。

3. **Embedding 模型的表达局限**

   不同表达方式可能对应相近的向量表示，但模型不一定能够准确识别哪段文本包含最直接的答案。

4. **Chunk 粒度的影响**

   Chunk 的长度、上下文完整性以及切分位置都会影响 Embedding 的语义表达。

## 5. 后续优化方向

基于本次实验，确定以下三个优化方向。

### 5.1 Hybrid Retrieval（混合检索）

结合 Dense Retrieval 和 Sparse Retrieval。

Dense Retrieval 负责语义匹配，Sparse Retrieval 负责关键词匹配。

例如：

- Dense Retrieval：找到语义相关的段落。
- BM25：找到直接包含 Seed Lexicon 等关键词的段落。
- Fusion：融合两种检索结果。

预期价值：

降低单一向量检索对语义表达方式的依赖，提高候选证据的覆盖率。

### 5.2 Reranker（重排序模型）

在 Dense Retrieval 初步召回 Top-K 候选后，使用 Cross-Encoder 对 Query 和候选 Chunk 进行联合编码，重新计算相关性并排序。

基本流程：

```text
User Query
    |
    v
Dense Retrieval
    |
    v
Top-K Candidates
    |
    v
Cross-Encoder Reranker
    |
    v
Re-ranked Results
```

预期价值：

提高真正包含答案的 Chunk 在最终检索结果中的排名。

### 5.3 Query Expansion（查询扩展）

根据用户问题生成相关的检索表达。

例如：

```text
Original Query:
What is the seed lexicon?

Expanded Queries:
- Definition of seed lexicon
- Positive and negative predicates
- Seed lexicon composition
- Polarity initialization vocabulary
```

预期价值：

增加检索表达的覆盖范围，帮助系统发现不同措辞下的相关证据。

## 6. 后续实验评估方案

为了避免仅凭主观感受判断优化效果，后续需要建立统一的检索评测体系。

建议采用以下指标：

| 指标 | 评估目标 |
|---|---|
| Recall@K | Top-K 中是否召回标准证据 |
| MRR@K | 第一个相关证据的排名 |
| nDCG@K | 多个相关结果的排序质量 |
| Hit Rate@K | Top-K 是否至少包含一个相关结果 |
| Latency | 检索耗时 |

利用 QASPER 数据集中的 Evidence 作为评测依据。

特别注意：Evidence 只用于评估，不允许提前加入检索 Corpus 或作为 Embedding 的输入，否则会产生数据泄漏。

后续应分别比较：

1. Dense Retrieval
2. BM25
3. Hybrid Retrieval
4. Dense Retrieval + Reranker
5. Hybrid Retrieval + Reranker

所有方法应使用相同的数据划分、Chunk 配置和评测标准。

## 7. 当前阶段结论

本次实验成功实现了基于 Embedding 的 Dense Retrieval，并在真实科研论文上完成了第一次语义检索。

实验发现：

**Dense Retrieval 能够找到语义相关的内容，但语义相似度排序与实际答案相关性排序之间仍然存在差异。**

这一发现为后续 Hybrid Retrieval 和 Reranker 的实现提供了明确的优化目标。

当前 Dense Retrieval 保留为项目的基础检索 Baseline，后续所有优化均在此基础上进行对照实验。

---

**项目：** PaperPilot-RAG

**阶段：** Day 4 — Embedding & Dense Retrieval

**实验数据集：** QASPER

**实验性质：** 初步探索性实验，尚未构成完整检索性能评测。