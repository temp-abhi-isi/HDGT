import os
import sys
import json
import gc
import yaml
from pathlib import Path
from tqdm import tqdm
import torch

# Ensure the root directory is in the path
sys.path.append(str(Path(__file__).parent.absolute()))

from hdgt.parsers.docling_parser import DoclingParser
from hdgt.graph.node_builder import NodeBuilder
from hdgt.graph.edge_builder import EdgeBuilder
from hdgt.graph.hetero_graph import build_hetero_data

def main():
    print("====================================================")
    print("   HDGT -- MP-DocVQA Graph Construction Loop")
    print("====================================================")

    # 1. Config & Paths
    config_path = Path("hdgt/configs/default.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    edge_cfg = config.get("edges", {})
    
    root_dir = Path("data/MP-DocVQA")
    mapping_file = root_dir / "context_map.json"
    output_dir = Path("experiments/mpdocvqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mapping_file.exists():
        print(f"[ERROR] Mapping file not found at: {mapping_file}")
        print("Please run 'python prepare_mpdocvqa.py' first.")
        sys.exit(1)

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    contexts = mapping_data["contexts"]
    print(f"Loaded {len(contexts)} contexts from mapping file.")

    # Initialize the parser
    # We set save_figures=False for MP-DocVQA to avoid disk overflow (6500+ pages!)
    # since we already have the raw page images and we don't need to re-extract them.
    # In Phase 2, we can just load the raw page images directly.
    parser_obj = DoclingParser(verbose=False, save_figures=False)

    success_count = 0
    fail_count = 0

    for context_id, ctx_info in tqdm(contexts.items(), desc="Building Graphs"):
        pdf_path = root_dir / "pdfs" / f"{context_id}.pdf"
        graph_path = output_dir / f"{context_id}_graph.pt"

        if not pdf_path.exists():
            # Skip if PDF was not compiled (due to missing images)
            continue

        if graph_path.exists():
            success_count += 1
            continue

        try:
            # Step 1: Parse PDF into nodes
            nodes = parser_obj.parse(pdf_path, document_id=context_id)

            # Step 2: Build node feature tensors
            node_builder = NodeBuilder()
            node_builder.build(nodes)

            # Step 3: Build edges
            edge_builder = EdgeBuilder(edge_cfg)
            edges = edge_builder.build(nodes, node_builder)

            # Step 4: Assemble PyG HeteroData
            data = build_hetero_data(node_builder, edges)

            # Validate
            data.validate()

            # Save
            torch.save(data, graph_path)
            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to build graph for {context_id}: {e}")
            fail_count += 1

        # Periodically clean up memory to prevent bloat
        if success_count % 50 == 0:
            gc.collect()

    print("\n" + "=" * 52)
    print("   MP-DocVQA Graph Building Complete")
    print("=" * 52)
    print(f"  Successfully built graphs: {success_count}")
    print(f"  Failed builds             : {fail_count}")
    print(f"  Graphs saved to           : {output_dir}")
    print("=" * 52)

if __name__ == "__main__":
    main()
