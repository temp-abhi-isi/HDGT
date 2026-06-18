# MP-DocVQA Dataset & Graph Construction Instructions
This guide provides step-by-step instructions for running the **HDGT Phase 1 Graph Construction** experiment on a high-compute system.

---

## 1. System & Environment Setup

Run these setup steps on your high-compute system.

### A. Environment Check
First, clone the repository to the compute system and run the diagnostic check:
```bash
python check_env.py
```
This script will confirm if PyTorch is correctly configured with CUDA.

### B. Installation
Ensure you install PyTorch with the correct CUDA platform first. For example, if your system uses CUDA 12.1/12.4:
```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies from requirements.txt
pip install -r requirements.txt
```
*(Requirements include `torch-geometric`, `docling`, `pillow`, `tqdm`, `pyyaml`, `shapely`, `scikit-learn`, etc.)*

---

## 2. Dataset Download & Organization

1. Go to the [DocVQA Task 4 (Multipage DocVQA) Challenge Portal](https://rrc.cvc.uab.es/?ch=17) and log in.
2. Download the following archives:
   * **Questions and Answers** (e.g. `qas.zip`) — *Note: The validation QA file is already provided under `data/MP-DocVQA/qas.zip` in this repository, but you should download the full set.*
   * **Images** (e.g. `images.zip` / `documents.zip`) — Containing all page images.
3. Organize the directories on the compute system as follows:
   ```text
   data/
   └── MP-DocVQA/
       ├── qas.zip          # The zip file containing train.json, val.json, test.json
       └── images/          # Extract the contents of images.zip directly here
           ├── pybv0228_p80.png
           ├── nkbl0226_p0.png
           ├── snbx0223_p11.png
           └── ...
   ```

---

## 3. Step 1: Preprocessing & PDF Reconstruction

MP-DocVQA represents documents as separate page images, but each QA context consists of a contiguous range of page images (e.g., pages 11-22). We reconstruct these page ranges into multi-page PDFs to match the PDF-centric parsing of Docling.

Run the preprocessing script:
```bash
python prepare_mpdocvqa.py
```

### What this script does:
1. **Unzips the QAs**: Extracts `train.json`, `val.json`, and `test.json` from `data/MP-DocVQA/qas.zip` to `data/MP-DocVQA/`.
2. **Identifies unique contexts**: Scans the QA splits to group questions by unique, contiguous page range contexts (e.g. `snbx0223_p11_p22`).
3. **Compiles PDFs**: Locates the page images corresponding to each context, compiles them into a single multi-page PDF using Pillow, and saves them to `data/MP-DocVQA/pdfs/{context_id}.pdf`.
4. **Outputs Map**: Saves `data/MP-DocVQA/context_map.json` detailing the mapping of each question ID to its PDF path and the target `answer_page_idx`.

---

## 4. Step 2: Batch Graph Construction

Once the multi-page PDFs are constructed, we parse them to generate PyTorch Geometric heterogeneous graphs (`HeteroData` objects).

Run the batch graph construction loop:
```bash
python build_mpdocvqa_graphs.py
```

### What this script does:
1. Loads the configs from [default.yaml](file:///c:/Users/abhin/OneDrive/Documents/GitHub/HDGT/hdgt/configs/default.yaml).
2. Reads the `context_map.json` mapping.
3. Loops through all unique contexts in-process, parses them with `DoclingParser` (running OCR on the page images inside the compiled PDF), and uses the HDGT graph builder to assemble the nodes and edges.
4. Validates each graph using `data.validate()`.
5. Saves the resulting PyG graph objects to `experiments/mpdocvqa/{context_id}_graph.pt`.

---

## 5. Verification & Output Structure

Once the graph construction loop completes:
1. Check the `experiments/mpdocvqa/` directory. You should see a `.pt` file for each context, e.g.:
   ```text
   experiments/
   └── mpdocvqa/
       ├── pybv0228_p80_p80_graph.pt
       ├── nkbl0226_p0_p0_graph.pt
       ├── snbx0223_p11_p22_graph.pt
       └── ...
   ```
2. You can verify a single graph using PyTorch:
   ```python
   import torch
   data = torch.load("experiments/mpdocvqa/nkbl0226_p0_p0_graph.pt")
   print(data)
   ```

---

## 6. Next Steps: Phase 2 and Phase 3

Building these layout graphs sets the foundation for Phase 2 and Phase 3:
1. **Phase 2 (Embeddings)**: Loop over the nodes in `experiments/mpdocvqa/*_graph.pt` and use a multimodal encoder like Qwen2.5-VL to populate `data[node_type].x` with vision-language embeddings.
2. **Phase 3 (GNN & Traversal)**: Load the embedded graphs to run GNN message passing and perform $l$-hop traversal for context-aware QA retrieval.
