"""
visualization/graph_viz.py

Renders a heterogeneous document graph using networkx + matplotlib.

Color scheme (per node type):
  page    → purple   (#8e44ad)
  section → navy     (#2c3e50)
  text    → steel    (#2980b9)
  table   → orange   (#e67e22)
  figure  → green    (#27ae60)

Edge style (per relation):
  contains      → thin solid gray
  reading_order → thin solid blue
  spatial       → dashed light-blue
  reference     → dotted red
  parent_child  → solid green, thick
  continuation  → dashed orange, thick
  semantic      → dotted purple (Phase 2)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# ── Visual constants ──────────────────────────────────────────────────────────

NODE_COLORS: Dict[str, str] = {
    "page":    "#8e44ad",   # purple
    "section": "#2c3e50",   # navy
    "text":    "#2980b9",   # steel blue
    "table":   "#e67e22",   # orange
    "figure":  "#27ae60",   # green
}

EDGE_STYLES: Dict[str, Dict] = {
    "contains":      {"color": "#95a5a6", "style": "solid",  "width": 0.6, "alpha": 0.5},
    "reading_order": {"color": "#3498db", "style": "solid",  "width": 0.8, "alpha": 0.6},
    "spatial":       {"color": "#85c1e9", "style": "dashed", "width": 0.6, "alpha": 0.4},
    "reference":     {"color": "#e74c3c", "style": "dotted", "width": 1.5, "alpha": 0.9},
    "parent_child":  {"color": "#27ae60", "style": "solid",  "width": 1.8, "alpha": 0.8},
    "continuation":  {"color": "#f39c12", "style": "dashed", "width": 1.8, "alpha": 0.8},
    "semantic":      {"color": "#9b59b6", "style": "dotted", "width": 1.2, "alpha": 0.7},
}

DEFAULT_EDGE_STYLE = {"color": "#bdc3c7", "style": "solid", "width": 0.5, "alpha": 0.3}


class GraphVisualizer:
    """
    Converts a PyG HeteroData object into a networkx graph and renders it.

    Usage
    -----
    >>> viz = GraphVisualizer(config)
    >>> viz.render(data, output_path="experiments/paper_graph.png")
    """

    def __init__(self, config: dict) -> None:
        viz_cfg = config.get("visualization", {})
        self.figsize     = tuple(viz_cfg.get("figsize", [20, 14]))
        self.node_size   = int(viz_cfg.get("node_size", 800))
        self.font_size   = int(viz_cfg.get("font_size", 7))
        self.show_labels = bool(viz_cfg.get("show_edge_labels", False))
        self.dpi         = int(viz_cfg.get("dpi", 150))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        data: HeteroData,
        output_path: Optional[str | Path] = None,
        title: str = "HDGT Document Graph",
        max_nodes: int = 300,
    ) -> None:
        """
        Render the graph and optionally save to a PNG file.

        Parameters
        ----------
        data : HeteroData
        output_path : str or Path, optional
            If given, saves PNG here.
        title : str
        max_nodes : int
            Cap node count to keep visualization readable for large docs.
            Randomly samples if the graph exceeds this.
        """
        G, node_colors, node_labels = self._hetero_to_networkx(data, max_nodes)

        if G.number_of_nodes() == 0:
            logger.warning("Empty graph — nothing to visualize.")
            return

        fig, ax = plt.subplots(figsize=self.figsize, facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white", fontsize=14, pad=20, fontweight="bold")

        # Layout
        pos = self._compute_layout(G)

        # ── Draw edges by relation type ───────────────────────────────
        edge_groups = defaultdict(list)
        for u, v, d in G.edges(data=True):
            edge_groups[d.get("relation", "other")].append((u, v))

        for relation, edge_list in edge_groups.items():
            style = EDGE_STYLES.get(relation, DEFAULT_EDGE_STYLE)
            nx.draw_networkx_edges(
                G, pos, edgelist=edge_list, ax=ax,
                edge_color=style["color"],
                style=style["style"],
                width=style["width"],
                alpha=style["alpha"],
                arrows=True,
                arrowsize=8,
                arrowstyle="->",
                connectionstyle="arc3,rad=0.1",
                min_source_margin=10,
                min_target_margin=10,
            )

        # ── Draw nodes ────────────────────────────────────────────────
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=self.node_size,
            alpha=0.92,
        )

        # ── Node labels ───────────────────────────────────────────────
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, ax=ax,
            font_size=self.font_size,
            font_color="white",
            font_weight="bold",
        )

        # ── Legend ───────────────────────────────────────────────────
        node_patches = [
            mpatches.Patch(color=color, label=ntype)
            for ntype, color in NODE_COLORS.items()
        ]
        edge_patches = [
            mpatches.Patch(color=style["color"], label=rel, alpha=0.8)
            for rel, style in EDGE_STYLES.items()
        ]
        legend = ax.legend(
            handles=node_patches + edge_patches,
            loc="upper left",
            fontsize=8,
            framealpha=0.3,
            facecolor="#2c3e50",
            labelcolor="white",
            ncol=2,
        )

        ax.axis("off")
        plt.tight_layout()

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=self.dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info(f"Graph saved to: {out}")
            print(f"  Graph visualization: {out}")

        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hetero_to_networkx(
        self,
        data: HeteroData,
        max_nodes: int,
    ) -> Tuple[nx.DiGraph, List[str], Dict]:
        """Convert HeteroData → networkx DiGraph with colour + label metadata."""
        import random

        G = nx.DiGraph()
        node_colors: List[str] = []
        node_labels: Dict[int, str] = {}

        # ── Nodes ─────────────────────────────────────────────────────
        # Build a global node index: (ntype, local_idx) → G node id
        node_map: Dict[Tuple[str, int], int] = {}
        g_id = 0

        for ntype in data.node_types:
            ndata = data[ntype]
            n = ndata.x.shape[0]
            uids   = getattr(ndata, "node_uids", [f"{ntype}_{i}" for i in range(n)])
            roles  = getattr(ndata, "roles",     [ntype] * n)

            # Sample if too many nodes
            indices = list(range(n))
            if n > max_nodes // max(len(data.node_types), 1):
                indices = random.sample(indices, min(len(indices), max_nodes // max(len(data.node_types), 1)))

            for local_idx in indices:
                node_map[(ntype, local_idx)] = g_id
                G.add_node(g_id)
                node_colors.append(NODE_COLORS.get(ntype, "#bdc3c7"))
                # Short label: type abbreviation + local index
                short = f"{ntype[0].upper()}{local_idx}"
                if roles and local_idx < len(roles) and roles[local_idx] not in (ntype, ""):
                    short = f"{ntype[0].upper()}{local_idx}\n({roles[local_idx][:3]})"
                node_labels[g_id] = short
                g_id += 1

        # ── Edges ─────────────────────────────────────────────────────
        for (src_type, relation, dst_type) in data.edge_types:
            edata = data[src_type, relation, dst_type]
            edge_index = edata.edge_index

            for i in range(edge_index.shape[1]):
                src_local = int(edge_index[0, i])
                dst_local = int(edge_index[1, i])

                src_g = node_map.get((src_type, src_local))
                dst_g = node_map.get((dst_type, dst_local))

                if src_g is None or dst_g is None:
                    continue  # node was sampled out

                G.add_edge(src_g, dst_g, relation=relation)

        return G, node_colors, node_labels

    def _compute_layout(self, G: nx.DiGraph) -> Dict:
        """Choose layout based on graph size."""
        n = G.number_of_nodes()
        if n == 0:
            return {}
        if n <= 50:
            return nx.spring_layout(G, seed=42, k=2.5)
        elif n <= 150:
            return nx.kamada_kawai_layout(G)
        else:
            return nx.spring_layout(G, seed=42, k=1.5, iterations=30)
