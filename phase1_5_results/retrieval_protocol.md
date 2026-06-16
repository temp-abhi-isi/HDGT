# HDGT Retrieval Benchmark Protocol
# Phase 1.5 — Task 6

## Overview

This document defines the evaluation protocol for assessing HDGT's graph-based evidence retrieval against baseline methods. The protocol is designed to be executed in **Phase 5** using the full HDGT pipeline (Phase 1 graph construction + Phase 2 embeddings + Phase 3 GNN + Phase 4 ALR).

This document serves as the formal protocol specification so that Phase 2–4 development is aligned with the final evaluation.

---

## Primary Dataset: MP-DocVQA

### Dataset Description

**MP-DocVQA** (Multi-Page Document Visual Question Answering) is the primary benchmark for HDGT evaluation.

| Property | Value |
|----------|-------|
| Documents | 5,921 multi-page PDFs |
| Total questions | 46,042 QA pairs |
| Average pages per doc | ~8 pages |
| Max pages per doc | 20 pages |
| Question types | Extractive, abstractive, multi-hop |
| Evidence modality | Text, table, figure |
| Annotation | Page number + evidence string per question |

### Splits

| Split | Documents | Questions |
|-------|-----------|-----------|
| Train | 3,395 | 25,128 |
| Val | 1,258 | 9,673 |
| Test (withheld) | 1,268 | 11,241 |

### Download

```bash
# From official MP-DocVQA repository
# https://rrc.cvc.uab.es/?ch=17
# After obtaining access:
python download_mpdocvqa.py --split train val test --output data/mpdocvqa/
```

---

## Secondary Datasets

### DUDE (Document Understanding Dataset and Evaluation)

| Property | Value |
|----------|-------|
| Documents | 5,153 from diverse domains |
| Questions | 26,712 |
| Doc types | Medical, legal, financial, scientific |
| Pages per doc | 1–50 |

DUDE is used for **domain generalization** evaluation. If HDGT's graph topology generalizes across domains, DUDE scores should be strong without domain-specific fine-tuning.

### SlideVQA

| Property | Value |
|----------|-------|
| Presentations | 2,619 slide decks |
| Questions | 14,500 |
| Task | Multi-image reasoning across slide structure |

SlideVQA validates HDGT's `continuation` and `reading_order` edges in slide-specific layouts.

---

## Retrieval Task Definition

### Task

Given a natural language question $q$ and a multi-page document $D$:

> **Retrieve the minimal set of document elements $E^* \subseteq V_D$ that contain the information necessary to answer $q$.**

Where $V_D$ is the set of all nodes in the HDGT graph for document $D$.

### Input

- Query string $q$
- Pre-built HDGT graph $G_D = (V_D, E_D)$ with Phase 2 node embeddings

### Output

- Ranked list of `DocumentNode` objects: $[n_1, n_2, \ldots, n_k]$
- Each node includes: `node_uid`, `type`, `page`, `bbox`, `content`

---

## Evaluation Metrics

### 1. Recall@K (Primary Metric)

For each question $q$ with ground-truth evidence page $p^*$:

$$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}[p^* \in \text{pages}(\hat{E}_{1:K})]$$

where $\hat{E}_{1:K}$ are the top-K retrieved nodes.

Evaluated at: **K = 1, 5, 10**

> [!NOTE]
> MP-DocVQA provides page-level ground truth. We map nodes to pages via `node.page` for this metric.

### 2. Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(p^*)}$$

where $\text{rank}(p^*)$ is the rank of the first retrieved node on the correct evidence page.

### 3. Evidence Localization Accuracy (ELA) — Proposed Metric

MP-DocVQA provides evidence strings (exact text snippets). ELA measures whether the retrieved node **contains** the evidence string:

$$\text{ELA} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}[\text{evidence\_string}(q) \in \hat{n}_1.\text{content}]$$

> [!NOTE]
> **ELA is a proposed metric for Phase 5 evaluation**, not a current contribution. We introduce it here to formalise what element-level localization means for DocVQA. Flat page-level retrievers cannot compute ELA because they do not return element-level content — if HDGT demonstrates a measurable ELA score in Phase 5, this becomes a novel contribution. No ELA results exist yet.

### 4. Answer Exact Match (EM) and ANLS

After retrieval, the top-K nodes are passed to the generator (Phase 4). Standard DocVQA metrics:

- **Exact Match (EM)**: Binary correctness
- **ANLS** (Average Normalized Levenshtein Similarity): Partial credit for near-correct answers

$$\text{ANLS} = \frac{1}{|Q|} \sum_{q \in Q} \max_{a^* \in A^*} \left(1 - \frac{\text{EditDist}(\hat{a}, a^*)}{\max(|\hat{a}|, |a^*|)}\right)$$

---

## Retrieval Algorithm

### HDGT Graph Retrieval (Proposed)

```python
def hdgt_retrieve(query: str, graph: HeteroData,
                  node_embeddings: Dict[str, Tensor],
                  query_embedding: Tensor,
                  k_anchor: int = 3,
                  l_hops: int = 2) -> List[DocumentNode]:
    """
    1. Compute cosine similarity between query embedding and all node embeddings.
    2. Select k_anchor nodes with highest similarity (anchor nodes).
    3. Perform l-hop BFS traversal from each anchor node.
    4. Collect all traversed nodes, deduplicate.
    5. Re-rank by: anchor_similarity + hop_distance_decay.
    6. Return top-K nodes.
    """
```

**Traversal priority** (edge weights):
- `reference` edges: priority 1.0 (follow all reference edges)
- `parent_child` edges: priority 0.9
- `reading_order` edges: priority 0.7
- `spatial` edges: priority 0.5 (follow only if no other path)
- `contains` edges: priority 0.4 (for page context)

### Baseline Retrieval Methods (for comparison)

#### BM25 (Sparse Text Retrieval)
```python
# Treat each page's concatenated text as a document
# Query → BM25 → ranked pages
```

#### Dense Page Retrieval (DPR-style)
```python
# Encode each page as a single vector (mean of node embeddings on that page)
# Query → cosine similarity → ranked pages
```

#### ColPali-style (Vision Retrieval)
```python
# Encode each page as an image using PaliGemma
# Query → patch-level similarity → ranked pages
```

#### Flat Node Retrieval (ablation: HDGT without graph traversal)
```python
# Encode each node individually, retrieve by cosine similarity only
# No graph traversal — tests whether the graph topology adds value
```

---

## Ablation Studies

The following ablations isolate the contribution of each HDGT component:

| Experiment | Description | Expected Result |
|------------|-------------|-----------------|
| **HDGT-Full** | Full system with all edge types | Highest Recall@K, ELA |
| **HDGT-NoGraph** | Node retrieval without traversal | Lower Recall@K |
| **HDGT-NoRef** | Graph without reference edges | Lower ELA (figure/table questions) |
| **HDGT-NoParent** | Graph without parent_child edges | Lower section-level questions |
| **HDGT-NoCont** | Graph without continuation edges | Lower multi-page table questions |
| **BM25** | Sparse text baseline | Baseline comparison |
| **DPR** | Dense page-level baseline | Baseline comparison |

---

## Subgroup Analysis

### By Question Type

| Question Type | Expected HDGT Advantage |
|---------------|------------------------|
| Single-page extractive | Moderate (graphs add little for simple lookups) |
| Cross-page reasoning | High (graph traversal enables multi-hop paths) |
| Figure-referencing | Very High (reference edges directly solve this) |
| Table-referencing | Very High (reference + caption edges) |
| Section-level | High (parent_child edges) |

### By Document Type

| Document Type | Expected Performance |
|---------------|---------------------|
| Academic papers | Best (structured, consistent layouts) |
| Financial reports | Good (tables, figures with captions) |
| Legal documents | Moderate (dense text, few figures) |
| Slide decks | Good (continuation edges help) |

---

## Implementation Timeline

| Phase | Component | Required For Protocol |
|-------|-----------|----------------------|
| Phase 1 ✅ | Graph construction | ✅ Complete |
| Phase 2 | Qwen2.5-VL node embeddings | ✅ Required for retrieval |
| Phase 3 | HDGT-GNN updated embeddings | Optional (improves re-ranking) |
| Phase 4 | ALR fine-tuned generator | Required for EM/ANLS |
| **Phase 5** | **Full evaluation** | **This protocol** |

---

## Expected Results

> [!IMPORTANT]
> **No quantitative estimates are reported at this stage.**
>
> Phase 5 experiments on MP-DocVQA will determine the actual Recall@1, Recall@5, Recall@10, MRR, ANLS, and ELA scores for HDGT and all baseline methods.
>
> The purpose of this protocol is to define the evaluation methodology — not to predict outcomes. Reporting speculative numbers before the experiments are run would be misleading and would invite criticism that cannot yet be answered with data.

Results will be tabulated here after Phase 5 completes.

---

## Output Format

Each retrieval result should be stored as:

```json
{
  "question_id": "mpdocvqa_train_00001",
  "query": "What is the total revenue in Q3?",
  "retrieved_nodes": [
    {
      "node_uid": "annual_report_p7_n45",
      "type": "table",
      "page": 7,
      "rank": 1,
      "score": 0.923,
      "content": "Q3 Revenue: $4.2B  Q3 Expenses: $3.1B  Net: $1.1B",
      "bbox": [0.12, 0.34, 0.88, 0.67]
    },
    ...
  ],
  "ground_truth_page": 7,
  "evidence_string": "Q3 Revenue: $4.2B",
  "recall_at_1": true,
  "ela": true
}
```

---

## Reproducibility Checklist

- [ ] MP-DocVQA dataset downloaded and verified (MD5 checksums)
- [ ] Graph construction: `python run_batch.py data/mpdocvqa/ --output experiments/mpdocvqa/`
- [ ] Phase 2 embeddings computed for all nodes
- [ ] Query encoder checkpoint saved
- [ ] Retrieval script: `python evaluate_retrieval.py --dataset mpdocvqa --split val`
- [ ] Results saved to `experiments/retrieval_results_val.jsonl`
- [ ] Ablation experiments logged separately per config
