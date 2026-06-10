import os
import glob
import torch
import yaml
from hdgt.visualization.graph_viz import GraphVisualizer

os.makedirs("phase1_results/sample_graphs", exist_ok=True)

with open("hdgt/configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

viz = GraphVisualizer(config)

pt_files = glob.glob("experiments/*_graph.pt")
for pt_file in pt_files:
    # Handle the fact that TFMAdapter_k3_graph.pt uses k3 but we just want to load the graph
    doc_id = os.path.basename(pt_file).replace("_graph.pt", "")
    out_png = f"phase1_results/sample_graphs/{doc_id}_graph.png"
    if os.path.exists(out_png):
        print(f"Skipping {doc_id}, image already exists.")
        continue
    
    print(f"Rendering {doc_id}...", flush=True)
    try:
        data = torch.load(pt_file, weights_only=False)
        viz.render(data, output_path=out_png, title=f"HDGT Document Graph — {doc_id}")
        print(f"Saved {out_png}")
    except Exception as e:
        print(f"Failed on {doc_id}: {e}")
