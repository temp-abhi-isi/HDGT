import json
import logging
from hdgt.parsers.docling_parser import DoclingParser
from hdgt.graph.edge_builder import EdgeBuilder
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
logging.basicConfig(level=logging.INFO)

print("Parsing Open-YOLO 3D...", flush=True)
parser = DoclingParser(verbose=False)
nodes = parser.parse("data/Open-YOLO 3D Towards Fast and Accurate.pdf", document_id="Open_YOLO_3D")

print("Building Edges...", flush=True)
edge_builder = EdgeBuilder({"k_spatial": 3})
# node_builder is required for edge_builder.build but actually it just uses the nodes for reference edges.
# wait, edge_builder.build expects node_builder to be passed.
from hdgt.graph.node_builder import NodeBuilder
nb = NodeBuilder()
nb.build(nodes)
edges = edge_builder.build(nodes, nb)

print("--- FIGURE-CAPTION LINKS ---")
caption_edges = [e for e in edges if e.relation == "reference" and e.dst_type == "text" and "heuristic" in e.metadata.get("source", "")]
sampled_caps = caption_edges[:10]
for e in sampled_caps:
    src_node = next((n for n in nodes if n.node_id == e.src_id), None)
    dst_node = next((n for n in nodes if n.node_id == e.dst_id), None)
    if src_node and dst_node:
        print(f"[{src_node.type}] (page {src_node.page}) --> [caption] (page {dst_node.page}): {dst_node.content[:100]}...")

print("\n--- IN-TEXT REFERENCE LINKS ---")
ref_edges = [e for e in edges if e.relation == "reference" and e.metadata.get("source") == "intext_regex"]
sampled_refs = ref_edges[:20]
for e in sampled_refs:
    src_node = next((n for n in nodes if n.node_id == e.src_id), None)
    dst_node = next((n for n in nodes if n.node_id == e.dst_id), None)
    if src_node and dst_node:
        # For an in-text reference: text node references a figure/table node
        fig_num = e.metadata.get("figure_num", e.metadata.get("table_num", "?"))
        print(f"[text] (page {src_node.page}): \"{src_node.content[:80]}...\" --> [{dst_node.type} {fig_num}] (page {dst_node.page})")
