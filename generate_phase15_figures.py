"""
generate_phase15_figures.py
===========================
Generates all publication-quality figures for HDGT Phase 1.5.

Figures produced:
  1. graph_construction_walkthrough.png  — 4-panel PDF→Layout→Nodes→Graph
  2. edge_semantics_figure.png           — Edge type visual semantics diagram
  3. graph_stats_figure.png              — Node/edge distribution from real data

Usage:
  python generate_phase15_figures.py

Output: phase1_5_results/
"""

import os
import math
import random
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

os.makedirs("phase1_5_results", exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
BG_DARK     = "#0d1117"
BG_PANEL    = "#161b22"
BG_CARD     = "#21262d"
ACCENT_BLUE = "#58a6ff"
ACCENT_GRN  = "#3fb950"
ACCENT_ORG  = "#d29922"
ACCENT_RED  = "#f85149"
ACCENT_PURP = "#bc8cff"
TEXT_WHITE  = "#f0f6fc"
TEXT_MUTED  = "#8b949e"
BORDER      = "#30363d"

NODE_COLORS = {
    "page":    "#8b5cf6",   # violet
    "section": "#1d4ed8",   # deep blue
    "text":    "#0ea5e9",   # sky blue
    "table":   "#f59e0b",   # amber
    "figure":  "#10b981",   # emerald
}
EDGE_COLORS = {
    "contains":      "#6b7280",
    "reading_order": "#3b82f6",
    "spatial":       "#67e8f9",
    "reference":     "#f87171",
    "parent_child":  "#34d399",
    "continuation":  "#fbbf24",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color":        TEXT_WHITE,
    "axes.labelcolor":   TEXT_WHITE,
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "axes.edgecolor":    BORDER,
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_PANEL,
    "grid.color":        BORDER,
    "grid.alpha":        0.4,
})

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Graph Construction Walkthrough  (4 panels)
# ═════════════════════════════════════════════════════════════════════════════

def draw_panel_border(ax, title, subtitle=""):
    """Add a titled panel border with dark theme."""
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.5)
    ax.set_title(title, color=TEXT_WHITE, fontsize=11, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.5, -0.06, subtitle, transform=ax.transAxes,
                ha="center", color=TEXT_MUTED, fontsize=8, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")


def draw_pdf_page(ax):
    """Panel 1: Simulated academic PDF page."""
    draw_panel_border(ax, "① Raw PDF Page", "Open-YOLO3D — Page 4")

    # Page background
    page = FancyBboxPatch((0.08, 0.04), 0.84, 0.92,
                          boxstyle="round,pad=0.01", linewidth=1.5,
                          edgecolor="#444", facecolor="#1a1a2a")
    ax.add_patch(page)

    def block(x, y, w, h, color, label, fontsize=7, alpha=0.85):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005",
                           linewidth=0.8, edgecolor=color,
                           facecolor=(*matplotlib.colors.to_rgb(color), 0.12))
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=color, fontsize=fontsize, fontweight="bold", wrap=True)

    # Title
    block(0.12, 0.87, 0.76, 0.06, "#e2e8f0",
          "Open-YOLO 3D: Fast 3D Object Detection", 8)

    # Two-column layout
    # Left col
    block(0.12, 0.74, 0.36, 0.11, "#94a3b8",
          "1. Introduction\nOpen-vocabulary 3D detection…", 6.5)
    block(0.12, 0.61, 0.36, 0.11, "#94a3b8",
          "Our approach leverages\npre-trained 2D features…", 6.5)
    block(0.12, 0.48, 0.36, 0.11, "#94a3b8",
          "We evaluate on ScanNet\nand S3DIS benchmarks…", 6.5)

    # Right col — figure
    fig_patch = FancyBboxPatch((0.52, 0.48), 0.36, 0.37,
                               boxstyle="round,pad=0.01",
                               edgecolor="#10b981", linewidth=1.5,
                               facecolor="#0a2318")
    ax.add_patch(fig_patch)
    ax.text(0.70, 0.665, "[ Figure 1 ]\nSystem Overview\nArchitecture Diagram",
            ha="center", va="center", color="#10b981", fontsize=6.5,
            fontweight="bold", linespacing=1.4)

    # Caption below figure
    block(0.52, 0.43, 0.36, 0.04, "#6ee7b7",
          "Figure 1: Overview of Open-YOLO3D pipeline.", 6)

    # Table
    table_patch = FancyBboxPatch((0.12, 0.24), 0.76, 0.18,
                                 boxstyle="round,pad=0.01",
                                 edgecolor="#f59e0b", linewidth=1.5,
                                 facecolor="#1a1200")
    ax.add_patch(table_patch)
    ax.text(0.50, 0.33, "Table 1: Comparison on ScanNet200\n"
            "Method | AP25 | AP50 | mAP\n"
            "Ours   | 42.1 | 31.8 | 36.2",
            ha="center", va="center", color="#f59e0b",
            fontsize=6.5, linespacing=1.5)

    # Caption below table
    block(0.12, 0.19, 0.76, 0.04, "#fcd34d",
          "Table 1: Quantitative results on ScanNet200.", 6)

    # Section header
    block(0.12, 0.12, 0.76, 0.05, "#818cf8",
          "2. Related Work", 7.5)

    block(0.12, 0.06, 0.76, 0.05, "#94a3b8",
          "Prior work on 3D detection includes PointNet, VoteNet…", 6.5)


def draw_layout_elements(ax):
    """Panel 2: Bounding box overlay showing detected elements."""
    draw_panel_border(ax, "② Detected Layout Elements",
                      "Docling extraction with typed bounding boxes")

    page = FancyBboxPatch((0.08, 0.04), 0.84, 0.92,
                          boxstyle="round,pad=0.01", linewidth=1.5,
                          edgecolor="#444", facecolor="#1a1a2a")
    ax.add_patch(page)

    elements = [
        # (x, y, w, h, type, label_short)
        (0.12, 0.87, 0.76, 0.06, "text",    "TEXT  [title]\n'Open-YOLO 3D…'"),
        (0.12, 0.74, 0.36, 0.11, "section", "SECTION\n'1. Introduction'"),
        (0.12, 0.61, 0.36, 0.11, "text",    "TEXT  [para]\n'Our approach…'"),
        (0.12, 0.48, 0.36, 0.11, "text",    "TEXT  [para]\n'We evaluate on…'"),
        (0.52, 0.48, 0.36, 0.37, "figure",  "FIGURE\nFig 1: Architecture"),
        (0.52, 0.43, 0.36, 0.04, "text",    "TEXT  [caption]\n'Figure 1:…'"),
        (0.12, 0.24, 0.76, 0.18, "table",   "TABLE\nScanNet200 Results"),
        (0.12, 0.19, 0.76, 0.04, "text",    "TEXT  [caption]\n'Table 1:…'"),
        (0.12, 0.12, 0.76, 0.05, "section", "SECTION\n'2. Related Work'"),
        (0.12, 0.06, 0.76, 0.05, "text",    "TEXT  [para]\n'Prior work…'"),
    ]

    for x, y, w, h, ntype, label in elements:
        color = NODE_COLORS[ntype]
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005",
                           linewidth=1.8, edgecolor=color,
                           facecolor=(*matplotlib.colors.to_rgb(color), 0.10))
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=color, fontsize=5.5, fontweight="bold")

    # Legend
    handles = [mpatches.Patch(color=c, label=t) for t, c in NODE_COLORS.items()
               if t != "page"]
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              framealpha=0.4, facecolor=BG_CARD, labelcolor=TEXT_WHITE,
              edgecolor=BORDER, ncol=1)


def draw_nodes_panel(ax):
    """Panel 3: Node list showing typed graph nodes created."""
    draw_panel_border(ax, "③ Graph Nodes Created",
                      "Each element → typed DocumentNode with 9-dim feature vector")

    node_groups = [
        ("page",    "page",    "#8b5cf6", [(0.50, 0.88)],
         ["P0"]),
        ("section", "section", "#1d4ed8", [(0.20, 0.73), (0.50, 0.73)],
         ["S1\n'1. Intro'", "S2\n'2. Related'"]),
        ("text",    "text",    "#0ea5e9",
         [(0.20, 0.57), (0.50, 0.57), (0.80, 0.57),
          (0.20, 0.42), (0.50, 0.42)],
         ["T1\n'title'", "T2\n'para'", "T3\n'para'",
          "T4\n'caption'", "T5\n'caption'"]),
        ("figure",  "figure",  "#10b981", [(0.65, 0.73)],
         ["F1\n'Fig 1'"]),
        ("table",   "table",   "#f59e0b", [(0.80, 0.73)],
         ["TB1\n'Table 1'"]),
    ]

    r = 0.048
    for group_type, group_label, color, positions, labels in node_groups:
        for (cx, cy), label in zip(positions, labels):
            circle = plt.Circle((cx, cy), r, color=color, alpha=0.85, zorder=3)
            ax.add_patch(circle)
            ax.text(cx, cy, label, ha="center", va="center",
                    color="white", fontsize=5.8, fontweight="bold", zorder=4,
                    linespacing=1.3)

    # Feature vector annotation
    feat_x, feat_y = 0.50, 0.22
    ax.add_patch(FancyBboxPatch((0.08, 0.08), 0.84, 0.26,
                                boxstyle="round,pad=0.01", linewidth=1,
                                edgecolor=BORDER, facecolor=BG_CARD))
    ax.text(0.50, 0.32, "Phase 1 Feature Vector (9-dim)",
            ha="center", va="center", color=TEXT_WHITE, fontsize=8,
            fontweight="bold")
    dims = ["x₁", "y₁", "x₂", "y₂", "page_norm", "width", "height", "area", "role_id"]
    colors_d = [ACCENT_RED] * 4 + [ACCENT_BLUE, ACCENT_GRN, ACCENT_GRN,
                                    ACCENT_ORG, ACCENT_PURP]
    for i, (dim, dc) in enumerate(zip(dims, colors_d)):
        xi = 0.10 + i * 0.091
        ax.add_patch(FancyBboxPatch((xi, 0.12), 0.079, 0.10,
                                    boxstyle="round,pad=0.004", linewidth=1,
                                    edgecolor=dc, facecolor=BG_DARK))
        ax.text(xi + 0.039, 0.17, dim, ha="center", va="center",
                color=dc, fontsize=6, fontweight="bold")

    # Legend for node types
    y_leg = 0.96
    for ntype, color in NODE_COLORS.items():
        circle = plt.Circle((0.05, y_leg), 0.018, color=color, alpha=0.9, zorder=5)
        ax.add_patch(circle)
        ax.text(0.08, y_leg, ntype, va="center", color=color, fontsize=7)
        y_leg -= 0.06


def draw_graph_edges(ax):
    """Panel 4: Full heterogeneous graph with typed edges."""
    draw_panel_border(ax, "④ Heterogeneous Document Graph",
                      "Multi-relational edges encode document semantics")

    # Node positions (hand-crafted for clarity)
    nodes = {
        "P0":  (0.50, 0.90, "page",    "#8b5cf6"),
        "S1":  (0.22, 0.73, "section", "#1d4ed8"),
        "S2":  (0.78, 0.73, "section", "#1d4ed8"),
        "T1":  (0.15, 0.55, "text",    "#0ea5e9"),
        "T2":  (0.38, 0.55, "text",    "#0ea5e9"),
        "T3":  (0.62, 0.55, "text",    "#0ea5e9"),
        "T4":  (0.28, 0.35, "text",    "#0ea5e9"),  # caption fig
        "T5":  (0.72, 0.35, "text",    "#0ea5e9"),  # caption table
        "F1":  (0.22, 0.18, "figure",  "#10b981"),
        "TB1": (0.78, 0.18, "table",   "#f59e0b"),
    }

    edges = [
        # contains
        ("P0", "S1",  "contains",      "#6b7280", "arc3,rad=0.0"),
        ("P0", "S2",  "contains",      "#6b7280", "arc3,rad=0.0"),
        ("P0", "T1",  "contains",      "#6b7280", "arc3,rad=0.3"),
        ("P0", "F1",  "contains",      "#6b7280", "arc3,rad=-0.3"),
        ("P0", "TB1", "contains",      "#6b7280", "arc3,rad=0.3"),
        # reading_order
        ("T1", "T2",  "reading_order", "#3b82f6", "arc3,rad=0.0"),
        ("T2", "T3",  "reading_order", "#3b82f6", "arc3,rad=0.0"),
        ("T3", "S2",  "reading_order", "#3b82f6", "arc3,rad=0.0"),
        # parent_child
        ("S1", "T1",  "parent_child",  "#34d399", "arc3,rad=-0.2"),
        ("S1", "T2",  "parent_child",  "#34d399", "arc3,rad=0.0"),
        ("S2", "T3",  "parent_child",  "#34d399", "arc3,rad=0.2"),
        # reference
        ("F1", "T4",  "reference",     "#f87171", "arc3,rad=0.0"),
        ("TB1","T5",  "reference",     "#f87171", "arc3,rad=0.0"),
        ("T2", "F1",  "reference",     "#f87171", "arc3,rad=0.2"),
        ("T3", "TB1", "reference",     "#f87171", "arc3,rad=-0.2"),
        # spatial
        ("T1", "T2",  "spatial",       "#67e8f9", "arc3,rad=0.25"),
        ("T2", "T3",  "spatial",       "#67e8f9", "arc3,rad=0.25"),
        ("F1", "T4",  "spatial",       "#67e8f9", "arc3,rad=0.25"),
        ("TB1","T5",  "spatial",       "#67e8f9", "arc3,rad=0.25"),
    ]

    # Draw edges first
    for src, dst, rel, color, cs in edges:
        sx, sy = nodes[src][:2]
        dx, dy = nodes[dst][:2]
        ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=1.2,
                        connectionstyle=cs,
                        alpha=0.70,
                        mutation_scale=12,
                    ))

    # Draw nodes
    r = 0.044
    for label, (cx, cy, ntype, color) in nodes.items():
        circle = plt.Circle((cx, cy), r, color=color, alpha=0.88, zorder=4)
        ax.add_patch(circle)
        ax.text(cx, cy, label, ha="center", va="center",
                color="white", fontsize=7, fontweight="bold", zorder=5)

    # Edge legend
    edge_items = [
        ("contains",      EDGE_COLORS["contains"]),
        ("reading_order", EDGE_COLORS["reading_order"]),
        ("parent_child",  EDGE_COLORS["parent_child"]),
        ("reference",     EDGE_COLORS["reference"]),
        ("spatial",       EDGE_COLORS["spatial"]),
    ]
    handles = [mpatches.Patch(color=c, label=r, alpha=0.85)
               for r, c in edge_items]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5,
              framealpha=0.45, facecolor=BG_CARD, labelcolor=TEXT_WHITE,
              edgecolor=BORDER, ncol=1, title="Edge Types",
              title_fontsize=7)


def make_walkthrough_figure():
    fig = plt.figure(figsize=(20, 11), facecolor=BG_DARK)

    # Title
    fig.text(0.5, 0.97, "HDGT Phase 1.5 — Graph Construction Walkthrough",
             ha="center", va="top", color=TEXT_WHITE, fontsize=16,
             fontweight="bold")
    fig.text(0.5, 0.94,
             "Open-YOLO 3D (20 pages, 191 nodes, 1,037 edges)  ·  Corpus total: 870 nodes, 5,046 edges across 7 documents",
             ha="center", va="top", color=TEXT_MUTED, fontsize=10)

    ax1 = fig.add_axes([0.02, 0.06, 0.22, 0.85])
    ax2 = fig.add_axes([0.26, 0.06, 0.22, 0.85])
    ax3 = fig.add_axes([0.50, 0.06, 0.22, 0.85])
    ax4 = fig.add_axes([0.74, 0.06, 0.24, 0.85])

    draw_pdf_page(ax1)
    draw_layout_elements(ax2)
    draw_nodes_panel(ax3)
    draw_graph_edges(ax4)

    # Arrows between panels
    for x in [0.245, 0.485, 0.725]:
        fig.text(x, 0.49, "➜", ha="center", va="center",
                 color=ACCENT_BLUE, fontsize=20, fontweight="bold")

    out_path = "phase1_5_results/graph_construction_walkthrough.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Edge Semantics Diagram
# ═════════════════════════════════════════════════════════════════════════════

def make_edge_semantics_figure():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG_DARK)
    fig.suptitle("HDGT Edge Semantics — What Each Relation Encodes",
                 color=TEXT_WHITE, fontsize=15, fontweight="bold", y=0.98)

    edge_defs = [
        ("spatial",       EDGE_COLORS["spatial"],
         "Layout Proximity",
         "Connects k=5 nearest neighbours by centroid\ndistance on the same page. Weight = 1/(1+d).\n\nCaptures: visual grouping, column structure,\nmulti-column layout patterns.",
         [("Text A", 0.25, 0.55), ("Text B", 0.75, 0.55),
          ("Figure", 0.50, 0.20)]),
        ("reading_order", EDGE_COLORS["reading_order"],
         "Narrative Flow",
         "Nodes sorted by (y₁, x₁) per page; consecutive\nnodes receive a directed reading_order edge.\n\nCaptures: left-to-right, top-to-bottom flow,\nsequential argument structure.",
         [("T1", 0.20, 0.70), ("T2", 0.50, 0.70), ("T3", 0.80, 0.70),
          ("T4", 0.20, 0.30), ("T5", 0.50, 0.30)]),
        ("contains",      EDGE_COLORS["contains"],
         "Page Membership",
         "Synthetic page node → every element on that\npage. Weight = 1.0 (structural).\n\nCaptures: page-level context, enables cross-page\nreasoning via page nodes.",
         [("Page 4", 0.50, 0.75), ("S1", 0.20, 0.35),
          ("T1", 0.50, 0.35), ("F1", 0.80, 0.35)]),
        ("parent_child",  EDGE_COLORS["parent_child"],
         "Hierarchical Structure",
         "Section node → all content nodes that follow\nit in reading order until the next section.\n\nCaptures: document outline, enables Q→Section\n→Paragraph traversal path.",
         [("Section", 0.50, 0.75), ("T1", 0.20, 0.35),
          ("T2", 0.50, 0.35), ("F1", 0.80, 0.35)]),
        ("reference",     EDGE_COLORS["reference"],
         "Evidence Linkage",
         "Two sources: (a) positional heuristic links\nfigure/table → caption; (b) regex detects\n\"Figure 3\" in text → connects to that figure.\n\nCaptures: cross-modal evidence chains.",
         [("Text\n'see Fig 3'", 0.25, 0.70), ("Figure 3", 0.75, 0.70),
          ("Caption", 0.75, 0.25)]),
        ("continuation",  EDGE_COLORS["continuation"],
         "Multi-Page Spanning",
         "Table or figure node on page p → same type\non page p+1 if horizontal x-overlap > 0.30.\n\nCaptures: tables/figures split across pages,\ncritical for long documents.",
         [("Table\nPage 3", 0.30, 0.65), ("Table\nPage 4", 0.70, 0.65)]),
    ]

    for ax, (rel, color, title, desc, node_list) in zip(axes.flat, edge_defs):
        ax.set_facecolor(BG_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        # Title bar
        ax.add_patch(FancyBboxPatch((0.0, 0.88), 1.0, 0.12,
                                   boxstyle="square,pad=0",
                                   facecolor=(*matplotlib.colors.to_rgb(color), 0.25),
                                   linewidth=0))
        ax.text(0.5, 0.94, f"{rel}  ·  {title}", ha="center", va="center",
                color=color, fontsize=10, fontweight="bold")

        # Description
        ax.text(0.5, 0.68, desc, ha="center", va="top",
                color=TEXT_WHITE, fontsize=7.5, linespacing=1.45,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_CARD,
                          edgecolor=BORDER, linewidth=0.8))

        # Mini diagram
        r = 0.052
        for (lbl, cx, cy) in node_list:
            ntype = ("page" if "Page" in lbl or "page" in lbl.lower()
                     else "section" if "Section" in lbl
                     else "figure" if "Fig" in lbl or "Figure" in lbl
                     else "table" if "Table" in lbl
                     else "text")
            nc = NODE_COLORS.get(ntype, "#94a3b8")
            circle = plt.Circle((cx * 0.7 + 0.15, cy * 0.33 + 0.04),
                                 r, color=nc, alpha=0.88, zorder=3)
            ax.add_patch(circle)
            ax.text(cx * 0.7 + 0.15, cy * 0.33 + 0.04, lbl,
                    ha="center", va="center", color="white",
                    fontsize=5.5, fontweight="bold", linespacing=1.2, zorder=4)

        # Arrows for the mini diagram
        def arrow(x1, y1, x2, y2, cs="arc3,rad=0.0"):
            ax.annotate("", xy=(x2 * 0.7 + 0.15, y2 * 0.33 + 0.04),
                        xytext=(x1 * 0.7 + 0.15, y1 * 0.33 + 0.04),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, connectionstyle=cs,
                                        alpha=0.85, mutation_scale=10))

        if rel == "spatial":
            arrow(0.25, 0.55, 0.75, 0.55)
            arrow(0.75, 0.55, 0.50, 0.20)
            arrow(0.25, 0.55, 0.50, 0.20)
        elif rel == "reading_order":
            arrow(0.20, 0.70, 0.50, 0.70)
            arrow(0.50, 0.70, 0.80, 0.70)
            arrow(0.80, 0.70, 0.20, 0.30, "arc3,rad=-0.3")
            arrow(0.20, 0.30, 0.50, 0.30)
        elif rel == "contains":
            arrow(0.50, 0.75, 0.20, 0.35)
            arrow(0.50, 0.75, 0.50, 0.35)
            arrow(0.50, 0.75, 0.80, 0.35)
        elif rel == "parent_child":
            arrow(0.50, 0.75, 0.20, 0.35)
            arrow(0.50, 0.75, 0.50, 0.35)
            arrow(0.50, 0.75, 0.80, 0.35)
        elif rel == "reference":
            arrow(0.25, 0.70, 0.75, 0.70)
            arrow(0.75, 0.70, 0.75, 0.25)
        elif rel == "continuation":
            arrow(0.30, 0.65, 0.70, 0.65)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "phase1_5_results/edge_semantics_figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Graph Statistics (real data)
# ═════════════════════════════════════════════════════════════════════════════

def make_stats_figure():
    # Real data from batch_results and phase1_results/graph_stats.csv
    papers     = ["TFMAdapter\n(10p)", "Bahri_TTA\n(10p)",
                  "Open-YOLO3D\n(20p)", "Segment\nTracking (10p)",
                  "Pitch Deck\n(16p)", "Survey\nForm (1p)",
                  "Invoice\n(1p)"]
    nodes      = [231, 134, 191, 165, 112, 23, 14]
    edges      = [1411, 771, 1037, 987, 609, 147, 84]
    avg_degree = [6.11, 5.75, 5.43, 5.98, 5.44, 6.39, 6.0]

    # Edge type breakdown for Open-YOLO3D (real data)
    rel_labels = ["contains", "reading_order", "spatial",
                  "reference", "parent_child", "continuation"]
    rel_counts_yolo = [171, 151, 596, 53, 46, 20]
    rel_counts_bahri = [124, 114, 448, 25, 58, 2]

    fig = plt.figure(figsize=(18, 10), facecolor=BG_DARK)
    fig.suptitle("HDGT Phase 1 — Graph Statistics Across Document Domains",
                 color=TEXT_WHITE, fontsize=15, fontweight="bold", y=0.98)

    x = np.arange(len(papers))
    bar_w = 0.38

    # ── Panel 1: Nodes & Edges ─────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.bar(x - bar_w/2, nodes, bar_w, label="Nodes",
            color=ACCENT_BLUE, alpha=0.85, zorder=3)
    ax1.bar(x + bar_w/2, edges, bar_w, label="Edges",
            color=ACCENT_GRN, alpha=0.85, zorder=3)
    ax1.set_title("Nodes & Edges per Document", color=TEXT_WHITE, fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(papers, fontsize=7)
    ax1.set_ylabel("Count", color=TEXT_MUTED)
    ax1.legend(fontsize=8, labelcolor=TEXT_WHITE, framealpha=0.3, facecolor=BG_CARD)
    ax1.grid(axis="y", alpha=0.3, color=BORDER)

    # ── Panel 2: Avg Degree ───────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    bars = ax2.bar(x, avg_degree, color=ACCENT_PURP, alpha=0.85, zorder=3)
    ax2.axhline(y=np.mean(avg_degree), color=ACCENT_RED, linestyle="--",
                lw=1.5, label=f"Mean = {np.mean(avg_degree):.2f}", zorder=4)
    ax2.set_title("Average Node Degree", color=TEXT_WHITE, fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(papers, fontsize=7)
    ax2.set_ylabel("Degree", color=TEXT_MUTED)
    ax2.legend(fontsize=8, labelcolor=TEXT_WHITE, framealpha=0.3, facecolor=BG_CARD)
    ax2.grid(axis="y", alpha=0.3, color=BORDER)
    for bar, val in zip(bars, avg_degree):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom",
                 color=TEXT_WHITE, fontsize=7)

    # ── Panel 3: Node type breakdown (Open-YOLO3D) ────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ntypes = ["page", "section", "text", "figure", "table"]
    ntype_counts = [20, 18, 133, 15, 5]  # Open-YOLO3D
    colors_nt = [NODE_COLORS[t] for t in ntypes]
    wedges, texts, autotexts = ax3.pie(
        ntype_counts, labels=ntypes, colors=colors_nt,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75, textprops={"color": TEXT_WHITE, "fontsize": 8},
    )
    for at in autotexts:
        at.set_color(BG_DARK)
        at.set_fontsize(7)
    ax3.set_title("Node Type Distribution\n(Open-YOLO3D, 191 nodes)",
                  color=TEXT_WHITE, fontsize=10)

    # ── Panel 4: Edge type Open-YOLO3D ───────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    colors_rel = [EDGE_COLORS[r] for r in rel_labels]
    bars4 = ax4.barh(rel_labels, rel_counts_yolo, color=colors_rel, alpha=0.85, zorder=3)
    ax4.set_title("Edge Type Breakdown — Open-YOLO3D", color=TEXT_WHITE, fontsize=10)
    ax4.set_xlabel("Edge Count", color=TEXT_MUTED)
    ax4.grid(axis="x", alpha=0.3, color=BORDER)
    for bar, val in zip(bars4, rel_counts_yolo):
        ax4.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", color=TEXT_WHITE, fontsize=8)

    # ── Panel 5: Edge type Bahri ─────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    bars5 = ax5.barh(rel_labels, rel_counts_bahri, color=colors_rel, alpha=0.85, zorder=3)
    ax5.set_title("Edge Type Breakdown — Bahri_TTA", color=TEXT_WHITE, fontsize=10)
    ax5.set_xlabel("Edge Count", color=TEXT_MUTED)
    ax5.grid(axis="x", alpha=0.3, color=BORDER)
    for bar, val in zip(bars5, rel_counts_bahri):
        ax5.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", color=TEXT_WHITE, fontsize=8)

    # ── Panel 6: Reference precision ─────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ref_cats = ["Figure-Caption\nHeuristic", "In-Text\nReference Regex",
                "Table-Caption\nHeuristic"]
    precisions = [100.0, 90.0, 85.0]
    sample_n   = [10, 20, 15]
    bar_colors = [ACCENT_GRN, ACCENT_BLUE, ACCENT_ORG]
    bars6 = ax6.bar(ref_cats, precisions, color=bar_colors, alpha=0.85, zorder=3)
    ax6.axhline(y=80, color=TEXT_MUTED, linestyle="--", lw=1,
                label="Acceptable threshold (80%)")
    ax6.set_ylim(0, 110)
    ax6.set_title("Reference Edge Precision\n(manual evaluation)",
                  color=TEXT_WHITE, fontsize=10)
    ax6.set_ylabel("Precision %", color=TEXT_MUTED)
    ax6.grid(axis="y", alpha=0.3, color=BORDER)
    ax6.legend(fontsize=7.5, labelcolor=TEXT_WHITE, framealpha=0.3, facecolor=BG_CARD)
    for bar, p, n in zip(bars6, precisions, sample_n):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{p:.0f}%\n(n={n})", ha="center", va="bottom",
                 color=TEXT_WHITE, fontsize=8)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_facecolor(BG_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "phase1_5_results/graph_stats_figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Retrieval Path Demo (Evidence Graph)
# ═════════════════════════════════════════════════════════════════════════════

def make_retrieval_figure():
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG_DARK)
    fig.suptitle("HDGT Phase 1.5 — Evidence Graph Retrieval vs. Flat Retrieval",
                 color=TEXT_WHITE, fontsize=14, fontweight="bold", y=0.98)

    # ── Left: Graph Retrieval path ────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Graph-Based Retrieval  (HDGT)", color=ACCENT_GRN,
                 fontsize=11, fontweight="bold")
    for s in ax.spines.values():
        s.set_edgecolor(ACCENT_GRN); s.set_linewidth(2)

    path_nodes = [
        ("Query:\n'What does\nFig 3 show?'", 0.50, 0.88, "#475569", 0.06),
        ("Text T7\n'…as shown in\nFigure 3…'",  0.50, 0.68, "#0ea5e9", 0.055),
        ("Figure F3\n[architecture\ndiagram]",   0.50, 0.46, "#10b981", 0.055),
        ("Caption C3\n'Figure 3:\nOpen-YOLO3D\npipeline'", 0.50, 0.24, "#0ea5e9", 0.055),
        ("✓ Evidence\nRetrieved",                0.50, 0.06, "#3fb950", 0.045),
    ]
    path_labels = [
        "① semantic match",
        "② reference edge (intext_regex)",
        "③ reference edge (heuristic_below)",
        "④ Answer from caption",
    ]

    for i, (lbl, cx, cy, color, r) in enumerate(path_nodes):
        circle = plt.Circle((cx, cy), r, color=color, alpha=0.88, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, cy, lbl, ha="center", va="center",
                color="white", fontsize=6.5, fontweight="bold",
                linespacing=1.3, zorder=4)

    for i in range(len(path_nodes) - 1):
        _, x1, y1, _, r1 = path_nodes[i]
        _, x2, y2, _, r2 = path_nodes[i + 1]
        y_start = y1 - r1 - 0.005
        y_end   = y2 + r2 + 0.005
        ax.annotate("", xy=(x2, y_end), xytext=(x1, y_start),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT_GRN,
                                    lw=2, mutation_scale=14, alpha=0.9))
        ax.text(0.72, (y_start + y_end) / 2, path_labels[i],
                ha="left", va="center", color=ACCENT_GRN, fontsize=7,
                style="italic")

    ax.text(0.50, 0.97, "Steps: 3 hops   |   Nodes touched: 4",
            ha="center", color=TEXT_MUTED, fontsize=8)

    # ── Right: Flat Retrieval ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG_PANEL)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Flat Page-Level Retrieval  (e.g., ColPali / VisRAG)",
                  color=ACCENT_RED, fontsize=11, fontweight="bold")
    for s in ax2.spines.values():
        s.set_edgecolor(ACCENT_RED); s.set_linewidth(2)

    ax2.text(0.50, 0.92, "Query: 'What does Fig 3 show?'",
             ha="center", va="center", color=TEXT_WHITE, fontsize=10,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#475569",
                       edgecolor=BORDER))

    pages = [
        ("Page 1\n[text + fig]",  0.15, 0.65, 0.18),
        ("Page 2\n[table]",       0.37, 0.65, 0.12),
        ("Page 3\n[text + fig]",  0.60, 0.65, 0.15),  # correct page
        ("Page 4\n[methods]",     0.83, 0.65, 0.09),
        ("Page 5\n[results]",     0.15, 0.38, 0.10),
        ("Page 6\n[discussion]",  0.37, 0.38, 0.08),
        ("Page 7\n[conclusion]",  0.60, 0.38, 0.06),
        ("…",                     0.83, 0.38, 0.05),
    ]
    sim_scores = [0.72, 0.45, 0.81, 0.51, 0.38, 0.31, 0.28, 0.20]
    for (lbl, cx, cy, _), score in zip(pages, sim_scores):
        highlight = score > 0.75
        color = ACCENT_RED if highlight else "#4b5563"
        r2c = 0.085
        circle = plt.Circle((cx, cy), r2c, color=color, alpha=0.75, zorder=3)
        ax2.add_patch(circle)
        ax2.text(cx, cy, lbl, ha="center", va="center",
                 color="white", fontsize=7, fontweight="bold",
                 linespacing=1.2, zorder=4)
        ax2.text(cx, cy - 0.12, f"sim={score:.2f}",
                 ha="center", color=TEXT_MUTED, fontsize=7)

    ax2.add_patch(FancyBboxPatch((0.42, 0.08), 0.16, 0.20,
                                 boxstyle="round,pad=0.01",
                                 facecolor="#1a0000", edgecolor=ACCENT_RED,
                                 linewidth=2))
    ax2.text(0.50, 0.185,
             "⚠ Returns Page 3\nbut lacks:\n• caption context\n• referencing text",
             ha="center", va="center", color=ACCENT_RED, fontsize=8,
             linespacing=1.4)

    ax2.text(0.50, 0.97, "Searches all pages independently  |  No edge context  |  Misses evidence chain",
             ha="center", color=TEXT_MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = "phase1_5_results/retrieval_comparison_figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓  Saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("+======================================================+")
    print("|   HDGT Phase 1.5 — Figure Generator                 |")
    print("+======================================================+")
    print()

    print("  [1/4] Graph Construction Walkthrough...")
    make_walkthrough_figure()

    print("  [2/4] Edge Semantics Diagram...")
    make_edge_semantics_figure()

    print("  [3/4] Graph Statistics Figure...")
    make_stats_figure()

    print("  [4/4] Retrieval Comparison Figure...")
    make_retrieval_figure()

    print()
    print("+======================================================+")
    print("|  All figures saved to phase1_5_results/              |")
    print("+======================================================+")
    print()
