"""
graph/edge_builder.py

Builds all edges of the HDGT heterogeneous document graph.

Edge types constructed here:
  contains       — page → every element on that page
  reading_order  — sequential (y,x)-sorted chain per page
  spatial        — k-NN on centroids (same page), k configurable
  reference      — (a) figure/table → caption via positional heuristics
                   (b) text/section → figure/table via in-text mentions ("Figure 3")
  parent_child   — section → following content nodes until next section
  continuation   — table spanning consecutive pages (x-overlap heuristic)

Design choices:
  - k-NN (default k=5) for spatial edges instead of a fixed distance threshold.
    Different PDF layouts (2-column papers, legal, slides, financial) need
    a topology-adaptive approach.
  - Caption heuristics use directional conventions:
      * caption BELOW figure  (academic papers: most common)
      * caption ABOVE table   (some journals / Word docs)
    This gives higher precision than pure nearest-neighbour.
  - All edges carry weight and metadata for Phase 3 message-passing.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from hdgt.graph.schema import DocumentEdge, DocumentNode
from hdgt.graph.node_builder import NodeBuilder

logger = logging.getLogger(__name__)

# ── Regex patterns for in-text figure/table mentions ─────────────────────────
# Matches: "Figure 3", "Fig. 3", "fig 3", "FIG. 3"
INTEXT_FIG_RE = re.compile(r'\b(?:Figure|Fig\.?)\s*(\d+)', re.IGNORECASE)
# Matches: "Table 3", "TABLE 3"
INTEXT_TBL_RE = re.compile(r'\bTable\s*(\d+)', re.IGNORECASE)
# Matches caption starts: "Figure 3.", "Fig. 3:", "Table 2."
CAPTION_FIG_RE = re.compile(r'^(?:Figure|Fig\.?)\s*(\d+)', re.IGNORECASE)
CAPTION_TBL_RE = re.compile(r'^(?:Table)\s*(\d+)', re.IGNORECASE)


class EdgeBuilder:
    """
    Constructs all edges given a list of DocumentNode objects and a built NodeBuilder.

    Parameters
    ----------
    config : dict
        Loaded from configs/default.yaml.
        Relevant keys:
          k_spatial            (int,   default 5)
          caption_y_threshold  (float, default 0.20)
          continuation_x_overlap (float, default 0.30)
    """

    def __init__(self, config: dict) -> None:
        self.k_spatial            = int(config.get("k_spatial", 5))
        self.caption_y_threshold  = float(config.get("caption_y_threshold", 0.20))
        self.cont_x_overlap       = float(config.get("continuation_x_overlap", 0.30))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        nodes: List[DocumentNode],
        node_builder: NodeBuilder,
    ) -> List[DocumentEdge]:
        """
        Build all edges for the document graph.

        Parameters
        ----------
        nodes : List[DocumentNode]
        node_builder : NodeBuilder
            Must have already called node_builder.build(nodes).

        Returns
        -------
        List[DocumentEdge]
        """
        edges: List[DocumentEdge] = []
        edges.extend(self._build_contains_edges(nodes))
        edges.extend(self._build_reading_order_edges(nodes))
        edges.extend(self._build_spatial_edges(nodes))
        edges.extend(self._build_caption_edges(nodes))
        edges.extend(self._build_intext_reference_edges(nodes))
        edges.extend(self._build_parent_child_edges(nodes))
        edges.extend(self._build_continuation_edges(nodes))

        logger.info(f"EdgeBuilder built {len(edges)} edges total.")
        self._log_edge_summary(edges)
        return edges

    # ------------------------------------------------------------------
    # Individual edge builders
    # ------------------------------------------------------------------

    def _build_contains_edges(self, nodes: List[DocumentNode]) -> List[DocumentEdge]:
        """page → every non-page element on that page."""
        edges: List[DocumentEdge] = []
        page_nodes: Dict[int, DocumentNode] = {
            n.page: n for n in nodes if n.type == "page"
        }
        for node in nodes:
            if node.type == "page":
                continue
            pg = page_nodes.get(node.page)
            if pg is None:
                continue
            edges.append(DocumentEdge(
                src_id=pg.node_id,
                dst_id=node.node_id,
                src_type="page",
                dst_type=node.type,
                relation="contains",
                weight=1.0,
                metadata={"source": "page_containment"},
            ))
        logger.debug(f"  contains: {len(edges)}")
        return edges

    def _build_reading_order_edges(self, nodes: List[DocumentNode]) -> List[DocumentEdge]:
        """
        Sort all non-page nodes on each page by (y1, x1) and chain them.
        This approximates the natural left→right, top→bottom reading order.
        """
        edges: List[DocumentEdge] = []
        pages: Dict[int, List[DocumentNode]] = defaultdict(list)
        for node in nodes:
            if node.type != "page":
                pages[node.page].append(node)

        for page_idx, page_nodes in pages.items():
            sorted_nodes = sorted(page_nodes, key=lambda n: (n.bbox[1], n.bbox[0]))
            for i in range(len(sorted_nodes) - 1):
                src = sorted_nodes[i]
                dst = sorted_nodes[i + 1]
                edges.append(DocumentEdge(
                    src_id=src.node_id,
                    dst_id=dst.node_id,
                    src_type=src.type,
                    dst_type=dst.type,
                    relation="reading_order",
                    weight=1.0,
                    metadata={"order": i, "source": "sort_yx", "page": page_idx},
                ))
        logger.debug(f"  reading_order: {len(edges)}")
        return edges

    def _build_spatial_edges(self, nodes: List[DocumentNode]) -> List[DocumentEdge]:
        """
        k-NN spatial edges per page.

        For each non-page node, connect its k nearest neighbours (by centroid
        Euclidean distance) on the same page. Bidirectional: both src→dst and
        dst→src are added (PyG GNN layers expect symmetric neighbourhood).

        Weight = 1 / (1 + distance), so closer nodes have higher weight.
        """
        edges: List[DocumentEdge] = []
        pages: Dict[int, List[DocumentNode]] = defaultdict(list)
        for node in nodes:
            if node.type != "page":
                pages[node.page].append(node)

        for page_idx, page_nodes in pages.items():
            if len(page_nodes) < 2:
                continue

            centroids = np.array([[n.cx, n.cy] for n in page_nodes], dtype=np.float32)
            k = min(self.k_spatial + 1, len(page_nodes))  # +1 because includes self

            nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean")
            nbrs.fit(centroids)
            distances, indices = nbrs.kneighbors(centroids)

            seen = set()
            for i, (dists, nbr_indices) in enumerate(zip(distances, indices)):
                src_node = page_nodes[i]
                for dist, j in zip(dists[1:], nbr_indices[1:]):   # skip self at index 0
                    dst_node = page_nodes[j]
                    pair = (min(src_node.node_id, dst_node.node_id),
                            max(src_node.node_id, dst_node.node_id))
                    if pair in seen:
                        continue
                    seen.add(pair)

                    weight = 1.0 / (1.0 + float(dist))
                    meta = {"distance": float(dist), "source": "knn", "page": page_idx}

                    # Bidirectional
                    edges.append(DocumentEdge(
                        src_id=src_node.node_id, dst_id=dst_node.node_id,
                        src_type=src_node.type,  dst_type=dst_node.type,
                        relation="spatial", weight=weight, metadata=meta,
                    ))
                    edges.append(DocumentEdge(
                        src_id=dst_node.node_id, dst_id=src_node.node_id,
                        src_type=dst_node.type,  dst_type=src_node.type,
                        relation="spatial", weight=weight, metadata=meta,
                    ))

        logger.debug(f"  spatial: {len(edges)}")
        return edges

    def _build_caption_edges(self, nodes: List[DocumentNode]) -> List[DocumentEdge]:
        """
        Associate captions (text nodes with role='caption') to their figure or table.

        Heuristics (academic PDF conventions):
          - Caption BELOW figure: caption.y1 >= figure.y2 - tolerance
                                  AND horizontal overlap > 0
          - Caption ABOVE table: caption.y2 <= table.y1 + tolerance
                                  AND horizontal overlap > 0

        Falls back to nearest figure or table if directional heuristic finds nothing.
        """
        edges: List[DocumentEdge] = []
        pages: Dict[int, List[DocumentNode]] = defaultdict(list)
        for node in nodes:
            pages[node.page].append(node)

        tol = 0.05  # positional tolerance (normalised coords)

        for page_idx, page_nodes in pages.items():
            captions = [n for n in page_nodes if n.role == "caption"]
            figures  = [n for n in page_nodes if n.type == "figure"]
            tables   = [n for n in page_nodes if n.type == "table"]

            for cap in captions:
                matched = False

                # ── Try figure below caption ───────────────────────────
                best_fig, best_fig_dist = None, float("inf")
                for fig in figures:
                    below = cap.bbox[1] >= fig.bbox[3] - tol
                    x_overlap = min(cap.bbox[2], fig.bbox[2]) - max(cap.bbox[0], fig.bbox[0])
                    if below and x_overlap > 0:
                        dist = abs(cap.cy - fig.cy)
                        if dist < best_fig_dist:
                            best_fig_dist = dist
                            best_fig = fig

                if best_fig is not None and best_fig_dist < self.caption_y_threshold:
                    edges.append(DocumentEdge(
                        src_id=best_fig.node_id,
                        dst_id=cap.node_id,
                        src_type="figure",
                        dst_type="text",
                        relation="reference",
                        weight=1.0 / (1.0 + best_fig_dist),
                        metadata={"confidence": 0.90, "source": "heuristic_below"},
                    ))
                    matched = True
                    continue

                # ── Try table above caption ────────────────────────────
                best_tbl, best_tbl_dist = None, float("inf")
                for tbl in tables:
                    above = cap.bbox[3] <= tbl.bbox[1] + tol
                    x_overlap = min(cap.bbox[2], tbl.bbox[2]) - max(cap.bbox[0], tbl.bbox[0])
                    if above and x_overlap > 0:
                        dist = abs(cap.cy - tbl.cy)
                        if dist < best_tbl_dist:
                            best_tbl_dist = dist
                            best_tbl = tbl

                if best_tbl is not None and best_tbl_dist < self.caption_y_threshold:
                    edges.append(DocumentEdge(
                        src_id=best_tbl.node_id,
                        dst_id=cap.node_id,
                        src_type="table",
                        dst_type="text",
                        relation="reference",
                        weight=1.0 / (1.0 + best_tbl_dist),
                        metadata={"confidence": 0.85, "source": "heuristic_above"},
                    ))
                    matched = True
                    continue

                # ── Fallback: nearest figure or table ──────────────────
                if not matched:
                    candidates = figures + tables
                    if candidates:
                        nearest = min(candidates, key=lambda n: cap.centroid_distance(n))
                        dist    = cap.centroid_distance(nearest)
                        if dist < self.caption_y_threshold * 2:
                            edges.append(DocumentEdge(
                                src_id=nearest.node_id,
                                dst_id=cap.node_id,
                                src_type=nearest.type,
                                dst_type="text",
                                relation="reference",
                                weight=1.0 / (1.0 + dist),
                                metadata={"confidence": 0.60, "source": "nearest_fallback"},
                            ))

        logger.debug(f"  reference (caption): {len(edges)}")
        return edges

    def _build_parent_child_edges(self, nodes: List[DocumentNode]) -> List[DocumentEdge]:
        """
        Build section → content hierarchical edges.

        For each page:
          1. Sort all non-page nodes by reading order (y1, x1).
          2. Track the current active 'section' node.
          3. Every node after a section (and before the next section) gets a
             parent_child edge from that section.

        This creates the Section→Paragraph→Figure traversal path critical
        for long-document reasoning in Phase 3/5.
        """
        edges: List[DocumentEdge] = []
        pages: Dict[int, List[DocumentNode]] = defaultdict(list)
        for node in nodes:
            if node.type != "page":
                pages[node.page].append(node)

        for page_idx, page_nodes in pages.items():
            sorted_nodes = sorted(page_nodes, key=lambda n: (n.bbox[1], n.bbox[0]))
            current_section: DocumentNode | None = None

            for node in sorted_nodes:
                if node.type == "section":
                    current_section = node
                elif current_section is not None:
                    edges.append(DocumentEdge(
                        src_id=current_section.node_id,
                        dst_id=node.node_id,
                        src_type="section",
                        dst_type=node.type,
                        relation="parent_child",
                        weight=1.0,
                        metadata={
                            "source": "section_hierarchy",
                            "section_uid": current_section.node_uid,
                        },
                    ))

        logger.debug(f"  parent_child: {len(edges)}")
        return edges

    def _build_continuation_edges(self, nodes: List[DocumentNode]) -> List[DocumentEdge]:
        """
        Detect tables (and figures) that span consecutive pages.

        Heuristic: two table nodes on consecutive pages with significant
        horizontal overlap (default > 0.30 of normalised page width) are
        likely the same table continued.
        """
        edges: List[DocumentEdge] = []

        for element_type in ("table", "figure"):
            page_elements: Dict[int, List[DocumentNode]] = defaultdict(list)
            for node in nodes:
                if node.type == element_type:
                    page_elements[node.page].append(node)

            sorted_pages = sorted(page_elements.keys())
            for i in range(len(sorted_pages) - 1):
                p1, p2 = sorted_pages[i], sorted_pages[i + 1]
                if p2 - p1 != 1:
                    continue
                for n1 in page_elements[p1]:
                    for n2 in page_elements[p2]:
                        x_overlap = (
                            min(n1.bbox[2], n2.bbox[2]) - max(n1.bbox[0], n2.bbox[0])
                        )
                        if x_overlap >= self.cont_x_overlap:
                            edges.append(DocumentEdge(
                                src_id=n1.node_id,
                                dst_id=n2.node_id,
                                src_type=element_type,
                                dst_type=element_type,
                                relation="continuation",
                                weight=float(x_overlap),
                                metadata={
                                    "source": f"{element_type}_continuation",
                                    "x_overlap": float(x_overlap),
                                },
                            ))

        logger.debug(f"  continuation: {len(edges)}")
        return edges

    def _build_intext_reference_edges(self, nodes: List[DocumentNode]) -> List[DocumentEdge]:
        """
        Build reference edges from in-text figure/table citations.

        Algorithm:
          1. Scan captions to extract figure/table numbers:
               "Figure 3: Architecture overview" → fig_num=3
               "Table 2. Results" → tbl_num=2
          2. Build lookup maps: fig_num→node, tbl_num→node (scoped per document_id).
          3. Scan all text/section nodes for mentions like "Figure 3" or "Table 2".
          4. Create a 'reference' edge: text_node → figure/table_node.

        This is particularly powerful for scientific papers where authors write:
          "As shown in Figure 3, our model..."  →  connects that paragraph to Figure 3.
          "Table 2 summarises results..."        →  connects that section to Table 2.

        Limitations (Phase 1):
          - Figure numbers extracted from captions only; not from Docling metadata.
          - If multiple figures share the same number (different doc sections), the
            first one found wins. This is almost never an issue in scientific papers.
          - Numbered references across documents in a batch are scoped by document_id.
        """
        edges: List[DocumentEdge] = []

        # ── Step 1: Build figure/table number → node maps ──────────────
        # Scoped per document_id to avoid cross-document collisions in batches.
        fig_maps: Dict[str, Dict[int, DocumentNode]] = defaultdict(dict)  # doc_id → {num → node}
        tbl_maps: Dict[str, Dict[int, DocumentNode]] = defaultdict(dict)

        captions = [n for n in nodes if n.role == "caption"]
        for cap in captions:
            text = cap.content.strip()

            fig_match = CAPTION_FIG_RE.match(text)
            if fig_match:
                num = int(fig_match.group(1))
                # Find the nearest figure node to this caption (already linked by _build_caption_edges)
                # As a proxy: pick the figure node on the same page with smallest distance
                same_page_figs = [n for n in nodes if n.type == "figure" and n.page == cap.page]
                if same_page_figs:
                    nearest_fig = min(same_page_figs, key=lambda n: cap.centroid_distance(n))
                    fig_maps[cap.document_id][num] = nearest_fig
                continue

            tbl_match = CAPTION_TBL_RE.match(text)
            if tbl_match:
                num = int(tbl_match.group(1))
                same_page_tbls = [n for n in nodes if n.type == "table" and n.page == cap.page]
                if same_page_tbls:
                    nearest_tbl = min(same_page_tbls, key=lambda n: cap.centroid_distance(n))
                    tbl_maps[cap.document_id][num] = nearest_tbl

        # ── Step 2: Scan text/section nodes for in-text mentions ────────
        mention_nodes = [n for n in nodes if n.type in ("text", "section") and n.content]
        for node in mention_nodes:
            text = node.content
            doc_id = node.document_id

            # Figure mentions
            for match in INTEXT_FIG_RE.finditer(text):
                num = int(match.group(1))
                target = fig_maps[doc_id].get(num)
                if target is None:
                    continue
                # Avoid self-reference (caption shouldn't reference its own figure)
                if target.node_id == node.node_id:
                    continue
                edges.append(DocumentEdge(
                    src_id=node.node_id,
                    dst_id=target.node_id,
                    src_type=node.type,
                    dst_type="figure",
                    relation="reference",
                    weight=0.95,
                    metadata={
                        "confidence": 0.95,
                        "source": "intext_regex",
                        "mention": match.group(0),
                        "figure_num": num,
                    },
                ))

            # Table mentions
            for match in INTEXT_TBL_RE.finditer(text):
                num = int(match.group(1))
                target = tbl_maps[doc_id].get(num)
                if target is None:
                    continue
                if target.node_id == node.node_id:
                    continue
                edges.append(DocumentEdge(
                    src_id=node.node_id,
                    dst_id=target.node_id,
                    src_type=node.type,
                    dst_type="table",
                    relation="reference",
                    weight=0.95,
                    metadata={
                        "confidence": 0.95,
                        "source": "intext_regex",
                        "mention": match.group(0),
                        "table_num": num,
                    },
                ))

        logger.debug(f"  reference (intext): {len(edges)}")
        return edges

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_edge_summary(self, edges: List[DocumentEdge]) -> None:
        counts: Dict[str, int] = defaultdict(int)
        for e in edges:
            counts[e.relation] += 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        logger.info(f"Edge counts — {summary}")

    def print_summary(self, edges: List[DocumentEdge]) -> None:
        counts: Dict[str, int] = defaultdict(int)
        for e in edges:
            counts[e.relation] += 1
        # also sub-categorise reference edges by source
        ref_sources: Dict[str, int] = defaultdict(int)
        for e in edges:
            if e.relation == "reference":
                src = e.metadata.get("source", "unknown")
                ref_sources[src] += 1
        print("\n" + "=" * 54)
        print("  EDGE SUMMARY")
        print("=" * 54)
        for relation, count in sorted(counts.items()):
            print(f"  {relation:<18} : {count:>6} edges")
            if relation == "reference" and ref_sources:
                for src, c in sorted(ref_sources.items()):
                    print(f"    ↳ {src:<20} : {c:>5}")
        print("-" * 54)
        print(f"  {'TOTAL':<18} : {sum(counts.values()):>6} edges")
        print("=" * 54 + "\n")
