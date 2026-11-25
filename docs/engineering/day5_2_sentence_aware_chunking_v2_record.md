# Day 5 工程记录：Sentence-aware Chunking V2

**项目：** PaperPilot-RAG  
**阶段：** Day 5  
**日期：** 2026-11-25  
**状态：** 已完成  
**最终方案：** V2 Sentence-aware Chunking

---

## 1. 项目背景与设计目标

Day 5 的核心任务是解决科研论文 RAG 系统中的 Chunking 问题。

早期实验中，V0 和 V1 在检索指标上各有表现，但二者均不满足项目对最终 LLM 上下文的句子完整性硬性要求。

因此，V0 和 V1 不再作为有效切分方案候选，仅保留为历史诊断参考，用于观察新方案是否仍有优化空间。

本阶段最终目标是设计一种能够同时满足以下要求的切分策略：

1. Chunk 边界必须尽可能保持在完整句子之间。
2. 一个 Chunk 可以包含多个完整句子。
3. 使用 Token 数量而不是字符数量控制 Chunk 长度。
4. 对超过 Parent Chunk 长度限制的完整句子进行特殊处理，不允许直接截断。
5. 解决 Embedding 模型最大输入长度限制。
6. 检索阶段返回完整 Parent Chunk，而不是被切开的 Embedding Window。
7. 通过真实数据集实验和结构完整性审计验证方案。

最终实现 V2：Sentence-aware Chunking + Embedding Windows。

## 2. V2 核心架构

### 2.1 两层 Chunk 结构

V2 将切分任务拆分成两个独立层次：

- Parent Chunk：负责保存完整句子，作为最终检索结果和上下文组织的基本单位。
- Embedding Window：负责解决 Embedding 模型的输入长度限制，仅用于向量表示和检索。

两者通过 parent_chunk_id 建立对应关系。

整体流程：

原始论文
→ 段落级句子识别
→ Sentence-aware Parent Chunk
→ Token-based Embedding Window
→ Embedding
→ Window-level Dense Retrieval
→ Parent-level Retrieval Result
→ 完整 Parent Chunk

### 2.2 Sentence-aware Parent Chunk

核心实现：

`src/paper_rag/data/sentence_token_chunker.py`

主要设计：

- 使用 sentencesplit 进行句子识别。
- 采用 conservative 模式。
- 保留原文字符偏移。
- 使用 TokenCounter 统计 Token 数量。
- 将多个完整句子组合成 Parent Chunk。
- 不跨 Section 组合 Chunk。
- 同一 Section 内可以跨 Paragraph 组合，但保留段落来源信息。
- 使用 ChunkSpan 保存原文位置与 Chunk 内部位置的映射。

默认最大 Parent Chunk 长度为 512 tokens。

特殊情况：如果一个完整句子超过 512 tokens，则将其独立保存为一个 Oversized Parent Chunk，不截断、不拆分句子。

### 2.3 Embedding Window

核心实现：

`src/paper_rag/data/embedding_window.py`

Embedding 模型：

`sentence-transformers/all-MiniLM-L6-v2`

配置：

| 参数 | 数值 |
|---|---:|
| Embedding Dimension | 384 |
| Maximum Sequence Length | 256 |
| Window Overlap | 32 tokens |
| Parent Chunk Maximum | 512 tokens |

设计过程：

1. 对 Parent Chunk 进行 Tokenization。
2. 使用 Fast Tokenizer 获取 Token ID 和 Character Offset。
3. 根据模型最大输入长度构造窗口。
4. 相邻窗口保留 32 tokens 重叠。
5. 重新检查窗口实际模型输入长度，包含 Special Tokens。
6. 保存 Window 与 Parent Chunk 的映射关系。

这里需要特别区分：

Embedding Window 可以跨越句子边界，但 Parent Chunk 必须保持完整句子。

Embedding Window 只是检索表示，不会直接作为最终 LLM 上下文。

### 2.4 Windowed Dense Retriever

核心实现：

`src/paper_rag/retrieval/windowed_dense_retriever.py`

检索流程：

1. 将 Query 编码为向量。
2. 计算 Query 与全部 Embedding Windows 的相似度。
3. 按相似度降序排序。
4. 对同一个 Parent Chunk，只保留最高相似度的 Window 作为匹配结果。
5. 去除重复 Parent Chunk。
6. 返回完整 Parent Chunk 及其最佳匹配 Window。

这种方式使检索阶段能够利用较小的局部语义窗口，同时保证返回内容具有完整句子上下文。

## 3. 索引构建结果

### 3.1 Train

输入：

`data/processed/train_papers.jsonl`

输出：

`data/indexes/qasper_train_windowed_v2`

| 指标 | 结果 |
|---|---:|
| Papers | 888 |
| Empty Papers | 1 |
| Parent Chunks | 16,857 |
| Embedding Windows | 29,846 |
| Oversized Sentences | 12 |
| Embedding Shape | (29846, 384) |
| Embedding Dtype | float32 |
| Embedding Memory | 43.72 MiB |

空论文记录：1801.07804。

Manifest：

- schema_version: 1
- index_type: windowed
- model_name: sentence-transformers/all-MiniLM-L6-v2
- num_chunks: 16857
- num_windows: 29846
- embedding_dimension: 384
- dtype: float32
- normalization: l2
- max_sequence_length: 256
- overlap_tokens: 32

### 3.2 Validation

输入：

`data/processed/validation_papers.jsonl`

输出：

`data/indexes/qasper_validation_windowed_v2`

| 指标 | 结果 |
|---|---:|
| Papers | 281 |
| Empty Papers | 0 |
| Parent Chunks | 4,974 |
| Embedding Windows | 8,640 |
| Oversized Sentences | 1 |
| Embedding Shape | (8640, 384) |
| Embedding Dtype | float32 |
| Embedding Memory | 12.66 MiB |

Train 和 Validation 使用相同的核心切分参数，构建均完成，且完成持久化后的重新加载验证。

## 4. Token Budget Evaluation

### 4.1 实验目的

本实验不再以 V0、V1 的得分高低决定切分策略。

V0、V1 的角色是 diagnostic_only。

V2 是唯一满足句子完整性硬性要求的有效候选。

实验主要回答：

- V2 在不同上下文预算下能否有效召回证据？
- 增加上下文预算能否提高证据覆盖率？
- Train 和 Validation 是否具有一致的实验趋势？
- V2 是否仍有明显需要优化的问题？

### 4.2 实验配置

| 参数 | 配置 |
|---|---|
| Dataset | QASPER |
| Embedding Model | all-MiniLM-L6-v2 |
| Context Tokenizer | all-MiniLM-L6-v2 |
| Context Budgets | 1024 / 2048 / 4096 |
| Evidence Thresholds | 25% / 50% / 75% |
| Context Separator | 两个换行符 |
| Ranking Scope | 同一论文内全部 Parent Chunks |
| Packing | 按检索排名贪心装箱 |
| Chunk Truncation | 禁止 |

注意：由于最终生成模型尚未确定，当前使用 Embedding Tokenizer 作为上下文 Token 计数的代理。当前预算不包含 Question、System Prompt 和其他指令。

### 4.3 Train 实验结果

有效问题：2,098。

不包含 Body Evidence 的问题：495。

以下数据为 Evidence Threshold = 50%、UNION 模式下的结果。

| Context Budget | Hit@10 | MRR@10 | Context Hit Rate | Mean Tokens | Mean Chunks |
|---|---:|---:|---:|---:|---:|
| 1024 | 56.10% | 0.3927 | 56.10% | 991.3 | 4.47 |
| 2048 | 78.41% | 0.4413 | 78.65% | 1992.1 | 7.80 |
| 4096 | 89.61% | 0.4559 | 94.42% | 3687.2 | 13.39 |

### 4.4 Validation 实验结果

有效问题：888。

不包含 Body Evidence 的问题：117。

以下数据同样为 Evidence Threshold = 50%、UNION 模式。

| Context Budget | Hit@10 | MRR@10 | Context Hit Rate | Mean Tokens | Mean Chunks |
|---|---:|---:|---:|---:|---:|
| 1024 | 62.61% | 0.4519 | 62.61% | 990.2 | 4.36 |
| 2048 | 84.46% | 0.5002 | 84.46% | 1993.9 | 7.72 |
| 4096 | 93.13% | 0.5114 | 97.18% | 3629.2 | 13.28 |

### 4.5 实验分析

从 Train 和 Validation 的结果可以得到以下结论：

1. 随着 Context Budget 增加，V2 的 Hit@10 和 Context Hit Rate 均明显提高。
2. Train 和 Validation 呈现一致的趋势。
3. 在 2048 tokens 预算下，V2 已经取得较好的证据覆盖效果。
4. 4096 tokens 下进一步提高了证据覆盖，但同时需要更多上下文资源。
5. MRR@10 随预算增加而提高，说明完整上下文组织也有助于更早达到证据覆盖阈值。

需要注意：这些指标反映的是当前 Evidence Coverage Evaluation 协议下的检索与上下文覆盖能力，不等于最终 LLM 答案的正确率、忠实度或可读性。

V0 和 V1 不满足句子完整性硬性要求，不能作为有效候选。与它们的指标比较只能用于诊断，不构成 V2 的候选选型依据。

## 5. Sentence Integrity Audit

### 5.1 审计目的

为了避免仅凭少量测试样例判断切分正确性，本阶段实现了独立的完整性审计流程。

核心实现：

`src/paper_rag/evaluation/sentence_integrity_auditor.py`

执行入口：

`scripts/audit_sentence_integrity.py`

审计覆盖：

- 原始句子与 Parent Chunk 的映射。
- 原始文本是否丢失。
- 句子是否重复。
- 原文字符偏移是否正确。
- Chunk 内部字符偏移是否正确。
- Parent Chunk Token 数量是否正确。
- Oversized Chunk 是否独立。
- Embedding Window 是否存在 Token 覆盖缺口。
- Window Character Offset 是否正确。
- Window 是否超过模型输入长度。
- Embedding Shape、Dtype、有限数值与 L2 归一化。

### 5.2 审计结果

Train：

| 指标 | 结果 |
|---|---:|
| Source Sentences | 46,882 |
| Parent Chunks | 16,857 |
| Embedding Windows | 29,846 |
| Errors | 0 |
| Result | PASS |

Validation：

| 指标 | 结果 |
|---|---:|
| Source Sentences | 13,266 |
| Parent Chunks | 4,974 |
| Embedding Windows | 8,640 |
| Errors | 0 |
| Result | PASS |

最终结果：

- Train: PASS
- Validation: PASS
- Overall: PASS

审计报告：

`storage/metadata/sentence_integrity_audit.json`

### 5.3 审计结论与边界

Train 和 Validation 均通过了本次结构性完整性审计。

这意味着当前数据中，句子来源映射、Parent Chunk 结构、Token Window 覆盖和向量结构均符合审计规则。

但需要明确：

1. 当前审计使用与 Chunker 相同的句子识别规则，因此不能独立证明所有语言学句界都绝对正确。
2. 当前验证主要针对英文 QASPER 数据集，尚不能直接推断多语言适用性。
3. 审计检查了 Embedding 向量的结构、数值和归一化，没有重新计算全部向量以逐个核验其与原始 Window 的对应关系。

因此，当前结论是通过结构性审计，而不是证明所有潜在输入场景均不存在问题。

## 6. 工程质量与 Git 状态

本阶段完成：

- Sentence-aware Chunker 实现。
- Embedding Window 实现。
- Windowed Dense Retriever 实现。
- Windowed Index Store 持久化与加载。
- Token Budget Evaluation。
- Sentence Integrity Audit。
- 对应单元测试与评估测试。

最终用户确认 Ruff 检查和测试全部通过。

Train 和 Validation 索引均完成构建、持久化及重新加载验证。

Git 状态：

- 代码已提交。
- SSH Push 已成功。
- 当前开发分支为 feature/chunking-v2（根据本阶段终端记录）。
- 本次记录未保留最终提交 Hash。

## 7. 技术风险与后续观察事项

虽然 V2 已满足当前 Day 5 的验收要求，但仍有值得后续关注的问题。

### 7.1 最终生成模型 Tokenizer 尚未确定

目前上下文预算使用 Embedding Tokenizer 作为代理。

确定最终 LLM 后，需要重新计算真实上下文 Token 数量，并考虑 Prompt、Question 和 System Instructions 的额外开销。

### 7.2 极端超长句的上下文问题

当前 Oversized Parent Chunk 保留完整句子，不进行截断。

但是，当一个完整句子本身就超过 LLM 的上下文预算时，当前完整 Chunk 装箱策略可能跳过它。

后续需要根据实际模型和数据情况讨论专门处理方式。

### 7.3 Window 最大相似度聚合策略

当前采用多个 Window 的最高相似度作为 Parent Chunk 的检索分数。

较长 Parent Chunk 具有更多 Window，可能获得更多匹配机会。

未来可以通过实验研究其他聚合方式，但当前没有证据证明该问题已经导致明显性能退化。

### 7.4 句子识别与多语言泛化

当前主要验证英文 QASPER。

对于包含复杂数学表达式、缩写、特殊标点及其他语言的论文，仍需要进一步验证。

### 7.5 独立 Test 集验证

当前完成的是 Train 和 Validation。

在方案冻结后，应在 Test 集上进行最终独立评估，避免将 Train / Validation 的结果直接作为最终泛化结论。

## 8. Day 5 最终技术决策

经过设计实现、真实数据集构建、Token Budget Evaluation 和 Sentence Integrity Audit，最终确定：

**V2 作为 paperpilot-rag 当前唯一有效的句子完整性切分方案，并作为后续开发的默认切分基线。**

核心方案：

- Sentence-aware Parent Chunk。
- Maximum Parent Chunk Tokens = 512。
- Oversized Sentence 独立保存。
- Embedding Window Maximum Sequence Length = 256。
- Window Overlap = 32。
- Window-level Dense Retrieval。
- Parent-level Retrieval Result。

对于 V0 和 V1：

保留历史索引与实验结果，仅作为诊断参考，不再作为有效切分方案候选。

### Day 5 验收

| 验收项目 | 结果 |
|---|---|
| V2 核心代码 | PASS |
| Train 索引构建 | PASS |
| Validation 索引构建 | PASS |
| Token Budget Evaluation | PASS |
| Train Sentence Integrity Audit | PASS |
| Validation Sentence Integrity Audit | PASS |
| Ruff 检查 | PASS |
| 全量测试 | PASS |
| Git Commit | 完成 |
| SSH Push | 完成 |

**最终结论：Day 5 正式结束，V2 冻结为当前默认切分策略。**

本次明确不启动 Task 2，不自动推进后续开发任务。

---

## 9. 项目阶段性总结

Day 5 最重要的成果不是简单提高了检索分数，而是建立了一套能够兼顾语义检索和上下文完整性的两层 Chunking 架构。

通过 Parent Chunk 和 Embedding Window 的职责分离，解决了传统固定长度切分中句子容易被截断，以及完整上下文与 Embedding 模型输入长度之间的矛盾。

同时，Train / Validation 的真实实验和完整性审计为方案提供了可复现的工程证据。

这使得 V2 不再只是一个理论上的切分设想，而是已经完成实现、实验验证和工程验收的项目组件。