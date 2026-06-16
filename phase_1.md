Perfect. Before writing any HDGT code, Phase 1 should focus on building a **robust document graph extraction pipeline**. If we do this correctly, every later phase becomes easier.

# Phase 1 Goal

Input:

```text
PDF
```

Output:

```text
Heterogeneous Document Graph

Nodes:
    TextBlock
    Table
    Figure
    Caption
    Page

Edges:
    spatial
    reading_order
    table_continuation
    figure_reference
    semantic_similarity
```

stored as:

```python
torch_geometric.data.HeteroData
```

---

# Phase 1 Architecture

```text
PDF
 │
 ▼
MinerU / Docling
 │
 ▼
Structured JSON
 │
 ▼
Node Builder
 │
 ▼
Edge Builder
 │
 ▼
PyG HeteroData
 │
 ▼
Graph Visualization
```

---

# Step 1: Environment Setup

I recommend:

```bash
conda create -n hdgt python=3.11
conda activate hdgt
```

Install PyTorch first.

For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

# Step 2: Core Libraries

Install:

```bash
pip install torch-geometric
```

```bash
pip install networkx
```

```bash
pip install matplotlib
```

```bash
pip install pandas
```

```bash
pip install numpy
```

```bash
pip install tqdm
```

```bash
pip install pyyaml
```

```bash
pip install shapely
```

```bash
pip install scikit-learn
```

---

# Step 3: Document Parser

I strongly suggest starting with Docling.

Advantages:

* actively maintained
* easier installation
* structured output
* page coordinates
* table extraction
* image extraction

Install:

```bash
pip install docling
```

Test:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

result = converter.convert("sample.pdf")

print(result.document.export_to_markdown())
```

---

# Step 4: Define Graph Schema

Create:

```text
hdgt/
│
├── data/
├── parsers/
├── graph/
├── visualization/
├── configs/
└── experiments/
```

---

Inside:

```python
# graph/schema.py
```

Define node types.

```python
NODE_TYPES = [
    "text",
    "table",
    "figure",
    "caption",
    "page"
]
```

Define edge types.

```python
EDGE_TYPES = [
    "spatial",
    "reading_order",
    "semantic",
    "contains",
    "continuation",
    "reference"
]
```

---

# Step 5: Node Representation

Each node should contain:

```python
{
    "node_id": 17,
    "page": 4,
    "type": "text",

    "bbox": [x1,y1,x2,y2],

    "content": "...",

    "embedding": None
}
```

Don't compute embeddings yet.

That is Phase 2.

---

# Step 6: Parse PDF into Nodes

Example:

```python
pdf
│
├── Page 1
│   ├── Text Block A
│   ├── Text Block B
│   ├── Figure
│   └── Caption
│
├── Page 2
│   └── Table
```

Produces:

```python
nodes = [
    TextNode(),
    TextNode(),
    FigureNode(),
    CaptionNode(),
    TableNode()
]
```

---

# Step 7: Build Spatial Edges

For every page:

```python
Text A
Text B
Figure
Caption
```

connect nearby blocks.

Example:

```python
distance(A,B) < threshold
```

create:

```python
(A,B)
```

Spatial edge.

---

# Step 8: Reading Order Edges

Sort blocks:

```python
y coordinate
then x coordinate
```

Example:

```python
Block1
Block2
Block3
```

Create:

```python
1 → 2
2 → 3
```

These edges are extremely important.

---

# Step 9: Caption Association

If:

```python
Figure
```

is closest to:

```python
Caption
```

create:

```python
figure → caption
```

edge.

Similarly:

```python
table → caption
```

---

# Step 10: Create PyTorch Geometric Graph

Use:

```python
from torch_geometric.data import HeteroData
```

Example:

```python
data = HeteroData()

data["text"].x = text_features

data["table"].x = table_features

data["figure"].x = figure_features
```

Edges:

```python
data[
    "text",
    "spatial",
    "text"
].edge_index = edge_index
```

---

# Step 11: Graph Visualization

Very important.

Use:

```python
networkx
```

to draw:

```text
Text
 │
 ▼
Figure
 │
 ▼
Caption
```

for debugging.

If graph visualization looks wrong, training later will fail.

---

# Step 12: First Deliverable

By end of Phase 1 we should be able to run:

```bash
python build_graph.py sample.pdf
```

and obtain:

```text
nodes: 412
edges: 1687

node types:
    text: 300
    figure: 45
    table: 18
    caption: 49
```

plus:

```python
graph.pt
```

saved as PyG object.

---

# What I would do this week

### Day 1

* Create repository
* Setup environment
* Install Docling
* Test on 5 PDFs

### Day 2

* Design node schema
* Design edge schema

### Day 3

* Implement parser → nodes

### Day 4

* Implement spatial edges

### Day 5

* Build PyG graph

### Day 6

* Visualize graph

### Day 7

* Run on MP-DocVQA samples

Do **not touch Qwen2.5-VL, ALR, retrieval, or GNN layers yet**. The graph extraction pipeline is the foundation. If Phase 1 produces clean heterogeneous graphs, Phase 2 and Phase 3 become straightforward.
