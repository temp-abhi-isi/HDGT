import os
import subprocess
import sys

papers = [
    "data/Green Gray Simple Survey Form A4 Document.pdf",
    "data/NVCPitchDeckTemplate.pdf",
    "data/wordpress-pdf-invoice-plugin-sample.pdf"
]

doc_ids = ["Form_Survey", "Pitch_Deck", "Invoice_Sample"]

with open("batch_results_non_academic.txt", "w", encoding="utf-8") as f:
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
