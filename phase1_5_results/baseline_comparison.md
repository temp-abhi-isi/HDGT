# HDGT Baseline Comparison
# Phase 1.5 — Task 5

## Overview

This document positions the **Heterogeneous Document Graph Transformer (HDGT)** relative to existing document understanding, layout parsing, and retrieval systems.

The comparison is organized across three paradigm categories, evaluated on seven capability dimensions.

> [!IMPORTANT]
> **Methodology**: All capability assessments are based on **publicly reported architecture descriptions**, official papers, and official documentation. They reflect architectural design choices, not benchmark scores. Entries marked "Partial" indicate the capability is implicit, limited in scope, or requires domain-specific configuration. Reasonable reviewers may disagree with specific entries; the authors welcome corrections with citations.

> [!NOTE]
> HDGT's **Evidence Localization** column reflects a *proposed* design capability. Quantitative ELA scores will be measured in Phase 5 on MP-DocVQA.

## Comparison Dimensions

| Dimension | Description |
|-----------|-------------|
| **Graph Representation** | Does the system build an explicit graph of document elements? |
| **Typed Nodes** | Are node types (text, figure, table, section) distinguished? |
| **Typed Edges** | Are edge types (spatial, semantic, hierarchical) distinguished? |
| **Multi-Page Reasoning** | Can the system reason across non-adjacent pages? |
| **Cross-Modal Links** | Are text↔figure and text↔table references explicitly modeled? |
| **Evidence Localization** | Can the system return the exact supporting element (not just the page)? |
| **Open-Domain PDF Input** | Does it accept arbitrary PDFs without domain-specific training? |

---

## Category 1: Layout Parsing Systems

These systems focus on extracting structured information from document pages.

| Method | Graph | Typed Nodes | Typed Edges | Multi-Page | Cross-Modal | Evidence Loc. | Open-Domain |
|--------|:-----:|:-----------:|:-----------:|:----------:|:-----------:|:-------------:|:-----------:|
| **LayoutLM v3** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **DocFormer** | ❌ | ✅ | ❌ | ❌ | Partial | ❌ | ✅ |
| **DocLayout-YOLO** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Donut** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **HDGT (Phase 1)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Notes

**LayoutLM v3** (Microsoft, 2022): Encodes text, layout (bbox), and image patches jointly via a Transformer. Operates on single pages. No explicit graph structure; spatial relationships are implicit in attention. Cannot traverse cross-page evidence chains.

**DocFormer** (Amazon, 2021): Multi-modal Transformer using spatial features. Strong on form understanding (FUNSD, CORD). Single-page, no graph, no explicit figure-caption links.

**DocLayout-YOLO** (Alibaba, 2024): State-of-the-art layout detection using YOLO architecture. Excellent at detecting bounding boxes, but produces a flat list of elements — not a graph. No edge semantics whatsoever.

**HDGT Advantage**: HDGT builds an explicit multi-relational graph with 6 edge types. Every detected element is a typed node connected via spatial, reading-order, hierarchical, and reference edges. This enables *traversal* — a capability none of the above systems possess.

---

## Category 2: Visual Retrieval Systems

These systems retrieve document pages or regions given a query.

| Method | Graph | Typed Nodes | Typed Edges | Multi-Page | Cross-Modal | Evidence Loc. | Open-Domain |
|--------|:-----:|:-----------:|:-----------:|:----------:|:-----------:|:-------------:|:-----------:|
| **ColPali** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **VisRAG** | ❌ | ❌ | ❌ | Partial | ❌ | ❌ | ✅ |
| **RAG-Anything** | Partial | Partial | ❌ | Partial | Partial | Partial | ✅ |
| **URaG** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| **HDGT (Phase 1.5+)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Notes

**ColPali** (2024): Embeds full page images using PaliGemma into a patch-level embedding grid. Retrieves by cosine similarity on page-level embeddings. No structural graph; no figure-caption association; no reading order. Returns a *page*, not a *node*.

**VisRAG** (2024): Visual RAG that indexes document pages as image embeddings. Cross-page reasoning occurs only through independent similarity scores across pages — no explicit edge between related elements across pages. The "multi-page" capability is simply retrieving multiple independent pages.

**RAG-Anything** (2025): A pipeline that combines OCR, object detection, and vector search. Constructs partial knowledge graphs per document chunk. Edges are primarily "co-occurrence" rather than typed semantic relations. Cannot represent Figure→Caption→Text reference chains as typed edges.

**HDGT Advantage**: HDGT retrieves *subgraphs* rather than pages or embeddings. A query anchors to one or more graph nodes, then l-hop traversal collects structurally related evidence (e.g., a paragraph mentioning "Figure 3" → reference edge → the figure node → reference edge → the caption). This provides **evidence localization accuracy** that page-level systems cannot match.

---

## Category 3: Long-Document Multimodal Systems

These systems handle multi-page documents with vision-language models.

| Method | Graph | Typed Nodes | Typed Edges | Multi-Page | Cross-Modal | Evidence Loc. | Open-Domain |
|--------|:-----:|:-----------:|:-----------:|:----------:|:-----------:|:-------------:|:-----------:|
| **InternVL3** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **mPLUG-DocOwl2** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **LongVA** | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **GPT-4o (API)** | ❌ | ❌ | ❌ | ✅ | ✅ | Partial | ✅ |
| **HDGT (full pipeline)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Notes

**InternVL3** (2025): State-of-the-art open-source MLLM. Processes pages as image tiles concatenated into a long sequence. Strong benchmark scores but O(n²) attention cost over long documents (token inflation problem). No graph; evidence cannot be localized to element level.

**mPLUG-DocOwl2** (2024): Introduces H-Compressor to reduce page tokens, then processes flattened sequence. Multi-page capable but all inter-page relationships are implicit in attention. Cannot express "Section 3 → Paragraph → Figure 5" as an explicit path.

**GPT-4o**: Strong proprietary baseline with excellent OCR and multi-modal reasoning. Can reference page numbers and image regions in natural language, which some may interpret as partial evidence localization. We classify it as "Partial" because it does not return machine-readable element identifiers (bounding boxes, node UIDs) — only natural language descriptions. Closed-source, high API cost, cannot be fine-tuned on domain-specific retrieval tasks.

**Proposed HDGT Advantage over Long-Document MLLMs**: HDGT is designed to impose *explicit graph topology* before the LLM sees any content. The model would receive a *subgraph* of relevant nodes — not 400 page images. The claimed advantages are:
1. **Token inflation mitigation**: Instead of processing all pages, HDGT is designed to retrieve a 15–30 node subgraph.
2. **Evidence provenance**: Answers could be traced back to specific `node_uid` strings with page and bbox.
3. **Computational efficiency**: Architecture designed to scale to 100+ page documents without VRAM explosion.

*These advantages are design-level claims and will be quantitatively validated in Phase 5.*

---

## Summary Comparison Table

| Method | Graph | Typed Nodes | Typed Edges | Multi-Page | Cross-Modal | Evidence Loc. | Open-Domain | Year |
|--------|:-----:|:-----------:|:-----------:|:----------:|:-----------:|:-------------:|:-----------:|:----:|
| LayoutLM v3 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | 2022 |
| DocFormer | ❌ | ✅ | ❌ | ❌ | Partial | ❌ | ✅ | 2021 |
| DocLayout-YOLO | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | 2024 |
| ColPali | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 2024 |
| VisRAG | ❌ | ❌ | ❌ | Partial | ❌ | ❌ | ✅ | 2024 |
| RAG-Anything | Partial | Partial | ❌ | Partial | Partial | Partial | ✅ | 2025 |
| URaG | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | 2024 |
| InternVL3 | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 2025 |
| mPLUG-DocOwl2 | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 2024 |
| GPT-4o | ❌ | ❌ | ❌ | ✅ | ✅ | Partial | ✅ | 2024 |
| **HDGT (ours)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 2026 |

> ✅ = Full support per publicly described architecture | Partial = Implicit or limited per public documentation | ❌ = Not supported per public documentation

> [!NOTE]
> HDGT's ✅ for Evidence Localization reflects the proposed pipeline design — not yet experimentally measured. All other entries reflect architecture-level descriptions from cited papers.

---

## Key Architectural Distinctions

> [!NOTE]
> The following distinctions are *architectural* — they describe how HDGT differs in design. Claims about retrieval performance improvement are Phase 5 experimental targets.

### 1. Explicit Multi-Relational Graph

Among the evaluated systems, HDGT is the only one that constructs an explicit heterogeneous graph with:
- 5 typed node types (text, table, figure, section, page)
- 6 typed edge relations (contains, reading_order, spatial, parent_child, reference, continuation)

This is a structural architectural difference. Whether it translates to superior benchmark performance will be quantified in Phase 5.

### 2. Proposed Evidence Localization

HDGT is designed to return `node_uid` strings (e.g., `Open_YOLO_3D_p3_n47`) identifying exact extracted elements with bounding boxes and page numbers. This level of structured grounding is not reported in any of the evaluated baseline systems. ELA (Evidence Localization Accuracy) is proposed as a new metric for Phase 5 evaluation.

### 3. Linear Adjacency Trap (Hypothesis)

Sequence-based models (LayoutLM, InternVL, DocOwl) concatenate page content into a token sequence. The **Linear Adjacency Trap** hypothesis is: elements that are structurally related (e.g., a figure and the paragraph citing it 2 pages later) appear far apart in the flattened sequence, diluting attention. HDGT's reference edges are designed to create a direct connection between them regardless of page distance. **This hypothesis will be tested in Phase 5.**

### 4. Architectural Scalability

HDGT's graph retrieval is designed to scale O(k · d) where k = anchor nodes and d = hop depth. Long-sequence MLLMs scale O(n²) with document length due to attention. For a 100-page document, HDGT aims to retrieve a 20-node subgraph vs. ~50,000 tokens for InternVL3. Actual latency measurements are a Phase 5 deliverable.

---

## References

- LayoutLM v3: Huang et al., 2022. arXiv:2204.08387
- DocFormer: Appalaraju et al., 2021. ICCV 2021
- DocLayout-YOLO: Zhao et al., 2024. arXiv:2410.12628
- ColPali: Faysse et al., 2024. arXiv:2407.01449
- VisRAG: Yu et al., 2024. arXiv:2410.10594
- RAG-Anything: Guo et al., 2025. arXiv:2505.03713
- InternVL3: Chen et al., 2025. arXiv:2504.10479
- mPLUG-DocOwl2: Hu et al., 2024. arXiv:2409.03420
- URaG: Zhang et al., 2024. arXiv:2412.09616
