import torch

data = torch.load("experiments/Open_YOLO_3D_graph.pt", weights_only=False)

print("--- Checking Reference Edges (Text -> Figure) ---")
if ("text", "reference", "figure") in data.edge_types:
    edge_index = data["text", "reference", "figure"].edge_index
    num_edges = edge_index.shape[1]
    
    texts = data["text"].x  # We actually don't have raw string content in HeteroData!
    # Ah, the content is lost when converting to tensor. We need to get it from the parser.
print("NOTE: HeteroData only has tensors. We cannot verify text from here directly.")
