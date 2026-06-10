#!/usr/bin/env python
"""
build_graph.py — Phase 1 CLI entry point for the HDGT pipeline.

Usage
-----
  python build_graph.py paper.pdf
  python build_graph.py paper.pdf --output experiments/ --visualize
  python build_graph.py paper.pdf --doc-id arxiv_2401.00001 --config hdgt/configs/default.yaml
  python build_graph.py paper.pdf --verbose

Output
------
  experiments/<doc_id>_graph.pt    — Saved PyG HeteroData object
  experiments/<doc_id>_graph.png   — Visualization (if --visualize)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_header(doc_id: str, pdf_path: Path) -> None:
    print()
    print("+" + "=" * 54 + "+")
    print("|       HDGT -- Phase 1: Document Graph Builder        |")
    print("+" + "=" * 54 + "+")
    print(f"  Document : {pdf_path.name}")
    print(f"  Doc ID   : {doc_id}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HDGT Phase 1 — Build a heterogeneous document graph from a PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf_path",        type=str,  help="Path to the input PDF file.")
    parser.add_argument("--output",   "-o", type=str,  default="experiments",
                        help="Output directory (default: experiments/)")
    parser.add_argument("--config",   "-c", type=str,
                        default="hdgt/configs/default.yaml",
                        help="Path to YAML config (default: hdgt/configs/default.yaml)")
    parser.add_argument("--doc-id",        type=str,  default=None,
                        help="Document identifier (default: PDF filename stem)")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Render and save a graph visualization PNG.")
    parser.add_argument("--save-figures", action="store_true",
                        help="Extract and save figure images to <output>/<doc_id>_figures/. "
                             "Populates node.image_path for Phase 2 Qwen encoding.")
    parser.add_argument("--no-save",       action="store_true",
                        help="Skip saving graph.pt (useful for quick testing).")
    parser.add_argument("--verbose",       action="store_true",
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # ── Validate inputs ────────────────────────────────────────────────
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    doc_id = args.doc_id or pdf_path.stem
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[WARN] Config not found at {config_path}, using defaults.")
        config = {}
    else:
        config = load_config(str(config_path))

    print_header(doc_id, pdf_path)

    t0 = time.perf_counter()

    # ── Step 1: Parse PDF ──────────────────────────────────────────────
    print("  [1/5] Parsing PDF with Docling...")
    from hdgt.parsers.docling_parser import DoclingParser
    parser_obj = DoclingParser(verbose=args.verbose, save_figures=args.save_figures)
    figures_dir = None
    if args.save_figures:
        figures_dir = output_dir / f"{doc_id}_figures"
        print(f"        → Figure images will be saved to: {figures_dir}")
    nodes = parser_obj.parse(pdf_path, document_id=doc_id, figures_dir=figures_dir)

    num_pages = max((n.page for n in nodes), default=0) + 1
    print(f"        → {len(nodes)} elements extracted from {num_pages} page(s)")

    # ── Step 2: Build node feature tensors ────────────────────────────
    print("  [2/5] Building node feature tensors...")
    from hdgt.graph.node_builder import NodeBuilder
    node_builder = NodeBuilder()
    node_builder.build(nodes)
    node_builder.print_summary()

    # ── Step 3: Build edges ────────────────────────────────────────────
    print("  [3/5] Building graph edges...")
    from hdgt.graph.edge_builder import EdgeBuilder
    edge_cfg = config.get("edges", {})
    edge_builder = EdgeBuilder(edge_cfg)
    edges = edge_builder.build(nodes, node_builder)
    edge_builder.print_summary(edges)

    # ── Step 4: Assemble PyG HeteroData ───────────────────────────────
    print("  [4/5] Assembling HeteroData...")
    from hdgt.graph.hetero_graph import build_hetero_data, print_heterodata_summary
    data = build_hetero_data(node_builder, edges)
    print_heterodata_summary(data)

    # Validate
    try:
        data.validate()
        print("  ✓  data.validate() passed\n")
    except Exception as exc:
        print(f"  ✗  data.validate() FAILED: {exc}\n", file=sys.stderr)

    # ── Step 5: Save ───────────────────────────────────────────────────
    print("  [5/5] Saving outputs...")
    if not args.no_save:
        graph_path = output_dir / f"{doc_id}_graph.pt"
        torch.save(data, graph_path)
        print(f"  ✓  Graph saved  : {graph_path}")

        # Quick reload sanity check
        reloaded = torch.load(graph_path, weights_only=False)
        assert len(reloaded.node_types) > 0, "Reloaded graph has no node types!"
        print(f"  ✓  Reload check : OK ({len(reloaded.node_types)} node types)")

    if args.visualize:
        print("\n  Rendering visualization (this may take a moment for large graphs)...")
        from hdgt.visualization.graph_viz import GraphVisualizer
        viz = GraphVisualizer(config)
        viz_path = output_dir / f"{doc_id}_graph.png"
        viz.render(
            data,
            output_path=viz_path,
            title=f"HDGT Document Graph — {doc_id}",
        )

    elapsed = time.perf_counter() - t0
    print()
    print("+" + "=" * 54 + "+")
    print(f"|  Done in {elapsed:.1f}s".ljust(55) + "|")
    print("+" + "=" * 54 + "+")
    print()


if __name__ == "__main__":
    main()
