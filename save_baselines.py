import os
import csv

os.makedirs("phase1_results/sample_graphs", exist_ok=True)

# 1. Graph Stats
graph_stats = [
    ["Paper", "Pages", "Nodes", "Edges", "Avg_Degree", "Components", "Figures", "Tables", "Sections"],
    ["TFMAdapter", 10, 231, 1411, 6.11, 1, 6, 4, 35],
    ["Bahri_TTA", 10, 134, 771, 5.75, 1, 6, 3, 14],
    ["Open-YOLO3D", 20, 191, 1037, 5.43, 1, 15, 5, 18],
    ["Segment_Anything_Tracking", 10, 165, 987, 5.98, 1, 8, 7, 17],
    ["Pitch_Deck", 16, 112, 609, 5.44, 1, 18, 0, 17],
    ["Survey_Form", 1, 23, 147, 6.39, 1, 1, 1, 3],
    ["Invoice", 1, 14, 84, 6.00, 1, 1, 3, 3]
]

with open("phase1_results/graph_stats.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(graph_stats)

# 2. Edge Distribution
edge_distribution = [
    ["Paper", "Contains", "Reading_Order", "Spatial", "Parent_Child", "Reference", "Continuation", "Total_Edges"],
    ["TFMAdapter", 221, 211, 796, 152, 28, 3, 1411],
    ["Bahri_TTA", 125, 114, 447, 58, 25, 1, 771],
    ["Open-YOLO3D", 172, 151, 596, 46, 53, 19, 1037],
    ["Segment_Anything_Tracking", 155, 145, 566, 78, 36, 7, 987],
    ["Pitch_Deck", 96, 80, 354, 79, 0, 0, 609],
    ["Survey_Form", 22, 21, 86, 18, 0, 0, 147],
    ["Invoice", 13, 12, 50, 9, 0, 0, 84]
]

with open("phase1_results/edge_distribution.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(edge_distribution)

# 3. Precision Eval
precision_eval = [
    ["Relation", "Paper", "Total_Extracted", "Sample_Size", "Correct_Matches", "Precision_Percent"],
    ["Figure-Caption", "Open-YOLO3D", 16, 10, 10, 100],
    ["In-Text Reference", "Open-YOLO3D", 33, 20, 18, 90]
]

with open("phase1_results/precision_eval.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(precision_eval)

print("Baseline CSVs created successfully in phase1_results/ directory.")
