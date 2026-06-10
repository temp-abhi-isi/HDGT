import os
import subprocess
import sys

papers = [
    "data/Bahri_Test-Time_Adaptation_in_Point_Clouds_Leveraging_Sampling_Variation_with_Weight_WACV_2025_paper.pdf",
    "data/Open-YOLO 3D Towards Fast and Accurate.pdf",
    "data/Rajic_Segment_Anything_Meets_Point_Tracking_WACV_2025_paper.pdf"
]

doc_ids = ["Bahri_WACV", "Open_YOLO_3D", "Rajic_Segment"]

with open("batch_results.txt", "w", encoding="utf-8") as f:
    for i, paper in enumerate(papers):
        doc_id = doc_ids[i]
        print(f"Processing {doc_id}...", flush=True)
        f.write(f"--- {doc_id} ---\n")
        cmd = [sys.executable, "build_graph.py", paper, "--doc-id", doc_id]
        
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        
        f.write(result.stdout)
        f.write("\n")
        if result.returncode != 0:
            f.write(f"ERROR: {result.stderr}\n")
            print(f"ERROR on {doc_id}!")
        else:
            print(f"Success {doc_id}!")
        f.write("\n")
