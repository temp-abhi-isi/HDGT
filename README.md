# HDGT — Heterogeneous Document Graph Transformer

> **Phase 1: Multi-Modal Parsing & Graph Extraction**

Converts any PDF into a `torch_geometric.data.HeteroData` heterogeneous graph with typed nodes (text, table, figure, section, page) and typed edges (spatial, reading\_order, reference, parent\_child, contains, continuation).

---

## Project Structure

```
HDGT/
│
├── hdgt/
│   ├── parsers/
│   │   └── docling_parser.py      ← PDF → structured node list via Docling
│   ├── graph/
│   │   ├── schema.py              ← NODE_TYPES, EDGE_TYPES, DocumentNode, DocumentEdge
│   │   ├── node_builder.py        ← Node list → 9-dim feature tensors
│   │   ├── edge_builder.py        ← k-NN spatial, reading_order, reference, parent_child
│   │   └── hetero_graph.py        ← Assemble PyG HeteroData
│   ├── visualization/
│   │   └── graph_viz.py           ← Dark-themed networkx visualization
│   └── configs/
│       └── default.yaml           ← Tunable thresholds
│
├── data/                          ← Drop PDFs here
├── experiments/                   ← Output graphs saved here
├── build_graph.py                 ← CLI entry point
└── requirements.txt
```

---

## Environment Setup

### 1. Create conda environment

```bash
conda create -n hdgt python=3.11
conda activate hdgt
```

### 2. Install PyTorch (CUDA 12.4 — compatible with CUDA 12.x drivers)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:

```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # e.g. NVIDIA GeForce RTX 2050
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Basic
python build_graph.py data/paper.pdf

# With visualization
python build_graph.py data/paper.pdf --visualize

# Custom doc ID and output directory
python build_graph.py data/paper.pdf --doc-id arxiv_2401_00001 --output experiments/

# Full options
python build_graph.py --help
```

### Expected output

```
╔══════════════════════════════════════════════════════╗
║        HDGT — Phase 1: Document Graph Builder        ║
╚══════════════════════════════════════════════════════╝
  Document : paper.pdf
  Doc ID   : paper

  [1/5] Parsing PDF with Docling...
        → 412 elements extracted from 17 page(s)

  NODE SUMMARY
  page         :    17 nodes
  section      :     8 nodes
  text         :   312 nodes  [paragraph:280, caption:19, header:7, footer:6]
  table        :    21 nodes
  figure       :    18 nodes
  TOTAL        :   376 nodes

  EDGE SUMMARY
  contains     :   359 edges
  reading_order:   311 edges
  spatial      :  1241 edges
  reference    :    42 edges
  parent_child :   188 edges
  continuation :     4 edges
  TOTAL        :  2145 edges

  ✓  data.validate() passed
  ✓  Graph saved  : experiments/paper_graph.pt
  ✓  Reload check : OK (5 node types)

╔══════════════════════════════════════════════════════╗
║  Done in 38.2s                                       ║
╚══════════════════════════════════════════════════════╝
```

---

## Graph Schema

### Node Types

| Type    | Role values                                  | Feature dim |
|---------|----------------------------------------------|-------------|
| page    | page                                         | 9           |
| section | section\_header, title                       | 9           |
| text    | paragraph, caption, title, header, footer    | 9           |
| table   | table                                        | 9           |
| figure  | figure                                       | 9           |

**Phase 1 features (9-dim):** `[x1, y1, x2, y2, page_norm, width, height, area, role_id]`
**Phase 2** will concatenate Qwen2.5-VL embeddings.

### Edge Types

| Relation       | Direction              | Construction method                            |
|----------------|------------------------|------------------------------------------------|
| contains       | page → element         | Every element gets a contains from its page    |
| reading\_order | element → element      | (y, x) sort within page, chain consecutive     |
| spatial        | element ↔ element      | k-NN (k=5) on centroids, same page             |
| reference      | figure/table → text    | Directional caption heuristics (below/above)   |
| parent\_child  | section → element      | Section → following nodes until next section   |
| continuation   | table/figure → same    | Horizontal overlap across consecutive pages    |

---

## Roadmap

- **Phase 1** ✅ Multi-Modal Parsing & Graph Extraction ← *you are here*
- **Phase 2** 🔲 Qwen2.5-VL Encoding & Token Compression
- **Phase 3** 🔲 HDGT GNN (Disentangled Spatial-Semantic Attention)
- **Phase 4** 🔲 ALR Data Distillation & Supervised Fine-Tuning
- **Phase 5** 🔲 Hybrid Retrieval & Benchmark Evaluation