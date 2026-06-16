# Phase 1.5: Graph Semantics, Explainability, and Baseline Validation

## Objective

Phase 1 successfully demonstrated that HDGT can convert heterogeneous PDF documents into connected multi-relational graphs.

However, graph construction alone is not a research contribution.

The objective of Phase 1.5 is to establish:

1. Why the graph representation is meaningful.
2. How graph nodes and edges correspond to document semantics.
3. What differentiates HDGT from existing document parsing systems.
4. Whether graph-based retrieval provides advantages over existing retrieval paradigms.
5. How HDGT compares against current state-of-the-art document retrieval methods.

This phase serves as the bridge between graph construction (Phase 1) and semantic embedding integration (Phase 2).

---

# Research Questions

## RQ1

How are document elements transformed into graph nodes?

## RQ2

What semantic meaning is captured by graph edges?

## RQ3

Why is a graph representation superior to page-level retrieval?

## RQ4

Can graph traversal retrieve evidence more effectively than existing retrieval methods?

## RQ5

How does HDGT compare against current document retrieval baselines?

---

# Task 1: End-to-End Graph Construction Walkthrough

## Goal

Provide a complete visual explanation of how a single PDF page is transformed into a graph.

---

## Input Document

Select one representative page from:

* TFMAdapter
* Open-YOLO3D

---

## Required Visualization

### Step 1

Original PDF Page

Show:

* Title
* Paragraphs
* Figures
* Tables
* Captions

---

### Step 2

Detected Layout Elements

Overlay bounding boxes.

Example:

```text
Section Header
Paragraph 1
Paragraph 2
Figure 1
Caption
```

---

### Step 3

Node Creation

Convert elements into graph nodes.

Example:

```text
Section:
    S1

Paragraph:
    T1
    T2

Figure:
    F1

Table:
    TB1
```

---

### Step 4

Edge Creation

Explicitly show:

```text
S1 -> T1
S1 -> T2

T1 -> T2

T2 -> Figure 1

Page -> T1
Page -> T2
Page -> Figure 1
```

---

### Deliverable

Figure:

```text
PDF
↓
Layout Elements
↓
Graph Nodes
↓
Graph Edges
```

This figure will likely appear in the final paper.

---

# Task 2: Graph Semantic Hierarchy Analysis

## Goal

Define the semantic meaning of each node type.

---

## Current Hierarchy

```text
Document
│
├── Page
│
├── Section
│
├── Text
│
├── Figure
│
├── Table
│
└── Formula
```

---

## Investigation

Answer:

### Why are sentences not nodes?

### Why are words not nodes?

### Why is paragraph-level granularity sufficient?

---

## Expected Outcome

Formal justification for node granularity.

This addresses reviewer concerns regarding novelty.

---

# Task 3: Semantic Relationship Analysis

## Goal

Explain what information each edge type represents.

---

### Spatial Edge

Represents:

```text
layout proximity
```

---

### Reading Order Edge

Represents:

```text
narrative flow
```

---

### Parent-Child Edge

Represents:

```text
document hierarchy
```

---

### Reference Edge

Represents:

```text
evidence linkage
```

Example:

```text
Paragraph
    ↓
Figure
```

---

## Deliverable

Edge semantics table.

| Edge Type     | Meaning                |
| ------------- | ---------------------- |
| Spatial       | Layout proximity       |
| Reading Order | Narrative sequence     |
| Contains      | Page membership        |
| Parent-Child  | Hierarchical structure |
| Reference     | Evidence linkage       |

---

# Task 4: Evidence Graph Verification

## Goal

Demonstrate that graph retrieval can recover supporting evidence.

---

## Query Examples

Example:

```text
What does Figure 3 show?
```

Expected path:

```text
Query
↓
Paragraph
↓
Reference Edge
↓
Figure
↓
Caption
```

---

## Deliverable

5–10 manually verified retrieval examples.

---

# Task 5: Baseline Study

## Goal

Position HDGT relative to existing methods.

---

## Baseline Categories

### Layout Parsing Systems

* LayoutLM
* DocFormer
* DocLayout-YOLO

---

### Visual Retrieval Systems

* ColPali
* VisRAG
* RAG-Anything

---

### Long-Document Systems

* MMLongBench baselines
* InternVL
* DocOwl

---

## Deliverable

Comparison table.

| Method       | Graph   | Layout  | Evidence Links |
| ------------ | ------- | ------- | -------------- |
| ColPali      | No      | Partial | No             |
| VisRAG       | No      | Partial | No             |
| RAG-Anything | Partial | Yes     | Partial        |
| HDGT         | Yes     | Yes     | Yes            |

---

# Task 6: Retrieval Benchmark Design

## Goal

Prepare Phase 2 evaluation.

---

## Dataset

MP-DocVQA

---

## Retrieval Task

Given a question:

```text
Retrieve supporting evidence
```

---

## Metrics

* Recall@1
* Recall@5
* Recall@10
* MRR
* Evidence Localization Accuracy

---

## Output

Benchmark protocol for future experiments.

---

# Success Criteria

Phase 1.5 is complete when:

* End-to-end graph construction is visualized.
* Node hierarchy is formally justified.
* Edge semantics are documented.
* Evidence retrieval examples are demonstrated.
* Baseline comparison table is created.
* Retrieval evaluation protocol is finalized.

---

# Deliverables

1. `graph_construction_walkthrough.pdf`
2. `edge_semantics_report.pdf`
3. `baseline_comparison.md`
4. `retrieval_protocol.md`
5. Updated HDGT technical report