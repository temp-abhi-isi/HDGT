"""
graph/schema.py

Defines node types, edge types, and the core dataclasses for the HDGT
heterogeneous document graph.

Design decisions (incorporated from research review):
  - 'section' replaces treating headings as generic text blocks.
    Long-document reasoning traverses: Question → Section → Paragraph → Figure
  - 'caption' is NOT a node type. Captions are 'text' nodes with role='caption'.
    This lets Qwen2.5-VL use the same text encoder for all text variants in Phase 2.
  - DocumentEdge has explicit weight + metadata so Phase 3 message-passing
    can condition on edge confidence and type without a schema redesign.
  - document_id + node_uid are mandatory: required for multi-doc datasets
    (MP-DocVQA, DUDE, MMLongBench) where thousands of graphs are batched.
  - image_path is a hook for Phase 2 visual encoding (empty in Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Node type registry
# ---------------------------------------------------------------------------

NODE_TYPES: List[str] = [
    "text",     # All text content: paragraphs, captions, headers, footers
    "table",    # Tabular data region
    "figure",   # Image / chart / diagram
    "section",  # Section heading — hierarchical anchor for parent_child edges
    "page",     # Synthetic page-level node (one per PDF page)
]

# Valid role values per node type
NODE_ROLES: Dict[str, List[str]] = {
    "text":    ["paragraph", "caption", "title", "header", "footer"],
    "section": ["section_header", "title"],
    "table":   ["table"],
    "figure":  ["figure"],
    "page":    ["page"],
}

# Numeric encoding for the role feature dimension
ROLE_IDS: Dict[str, int] = {
    "paragraph":      0,
    "caption":        1,
    "title":          2,
    "header":         3,
    "footer":         4,
    "section_header": 5,
    "table":          6,
    "figure":         7,
    "page":           8,
}


# ---------------------------------------------------------------------------
# Edge type registry
# ---------------------------------------------------------------------------

EDGE_TYPES: List[str] = [
    "spatial",        # Physically close on same page (k-NN, not fixed threshold)
    "reading_order",  # Sequential flow: (y, x) sort within a page
    "semantic",       # Phase 2: embedding cosine similarity
    "contains",       # page → element
    "continuation",   # Table / figure spanning consecutive pages
    "reference",      # figure/table → caption (text node with role='caption')
    "parent_child",   # section → paragraph / figure / table below it
]


# ---------------------------------------------------------------------------
# Docling label → (node_type, role) mapping
# ---------------------------------------------------------------------------

# Captions map to ("text", "caption") — not a separate node type.
# Section headings map to ("section", ...) so the graph has explicit
# hierarchical anchors usable for parent_child edge construction.
DOCLING_LABEL_MAP: Dict[str, Tuple[str, str]] = {
    # Section-level
    "title":           ("section", "title"),
    "section_header":  ("section", "section_header"),
    # Text with roles
    "text":            ("text", "paragraph"),
    "paragraph":       ("text", "paragraph"),
    "list_item":       ("text", "paragraph"),
    "page_header":     ("text", "header"),
    "page_footer":     ("text", "footer"),
    "footnote":        ("text", "paragraph"),
    "formula":         ("text", "paragraph"),
    "code":            ("text", "paragraph"),
    "caption":         ("text", "caption"),       # ← key design decision
    # Tables
    "table":           ("table", "table"),
    # Figures
    "picture":         ("figure", "figure"),
    "figure":          ("figure", "figure"),
    "chart":           ("figure", "figure"),
}


# ---------------------------------------------------------------------------
# DocumentNode
# ---------------------------------------------------------------------------

@dataclass
class DocumentNode:
    """
    A single extracted element from a PDF document.

    Attributes
    ----------
    node_id : int
        Integer index unique within this document. Used as the PyG local index.
    node_uid : str
        Globally unique string: "{document_id}_p{page}_n{node_id}".
        Use when combining graphs across MP-DocVQA / DUDE / MMLongBench.
    document_id : str
        Source document identifier (e.g. PDF filename stem).
    page : int
        0-indexed page number.
    type : str
        One of NODE_TYPES.
    role : str
        Sub-classification within type:
          text   → paragraph | caption | title | header | footer
          section→ section_header | title
          table  → table
          figure → figure
          page   → page
    bbox : List[float]
        [x1, y1, x2, y2] normalised to [0, 1], top-left origin.
    content : str
        Raw text. Empty for pure-image figure nodes.
    image_path : Optional[str]
        Path to extracted image (figure nodes). Phase 2 hook — None in Phase 1.
    embedding : Optional[List[float]]
        Semantic embedding. None in Phase 1; set by Qwen2.5-VL in Phase 2.
    metadata : dict
        Parser-level extras: raw_label, num_rows, num_cols, confidence.
    """

    node_id: int
    node_uid: str
    document_id: str
    page: int
    type: str
    role: str
    bbox: List[float]                    # [x1, y1, x2, y2] in [0, 1]
    content: str = ""
    image_path: Optional[str] = None     # Phase 2 hook
    embedding: Optional[List[float]] = None  # Phase 2 hook
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Geometric helpers
    # ------------------------------------------------------------------

    @property
    def cx(self) -> float:
        """Centroid x (normalised)."""
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        """Centroid y (normalised)."""
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return abs(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return abs(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    def centroid_distance(self, other: "DocumentNode") -> float:
        """Euclidean centroid distance (normalised coordinates)."""
        return ((self.cx - other.cx) ** 2 + (self.cy - other.cy) ** 2) ** 0.5

    def __repr__(self) -> str:
        return (
            f"DocumentNode(uid={self.node_uid!r}, type={self.type!r}, "
            f"role={self.role!r}, "
            f"bbox={[round(v, 3) for v in self.bbox]}, "
            f"content={self.content[:40]!r})"
        )


# ---------------------------------------------------------------------------
# DocumentEdge
# ---------------------------------------------------------------------------

@dataclass
class DocumentEdge:
    """
    A directed edge between two DocumentNode objects.

    Attributes
    ----------
    src_id : int
        Global node_id of the source node.
    dst_id : int
        Global node_id of the destination node.
    src_type : str
        Node type of source (one of NODE_TYPES).
    dst_type : str
        Node type of destination (one of NODE_TYPES).
    relation : str
        One of EDGE_TYPES.
    weight : float
        Edge strength. For spatial: 1/(1+distance). For structural: 1.0.
    metadata : dict
        Extensible edge attributes for Phase 3 conditioning:
          confidence  : float  — heuristic or parser confidence
          source      : str    — "knn" | "sort_yx" | "heuristic_below" | ...
          distance    : float  — raw centroid distance (spatial edges)
    """

    src_id: int
    dst_id: int
    src_type: str
    dst_type: str
    relation: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DocumentEdge({self.src_id}→{self.dst_id}, "
            f"{self.src_type}-[{self.relation}]->{self.dst_type}, "
            f"w={self.weight:.3f})"
        )
