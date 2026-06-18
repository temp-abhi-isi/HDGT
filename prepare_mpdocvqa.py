import os
import sys
import zipfile
import json
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def main():
    print("====================================================")
    print("   HDGT -- MP-DocVQA Dataset Preparation Script")
    print("====================================================")

    # 1. Paths
    root_dir = Path("data/MP-DocVQA")
    qa_zip_path = root_dir / "qas.zip"
    images_dir = root_dir / "images"  # User needs to extract images.zip here
    pdfs_dir = root_dir / "pdfs"
    
    # Check if images directory exists
    if not images_dir.exists():
        print(f"\n[ERROR] Images directory not found at: {images_dir.resolve()}")
        print("Please download 'images.zip' from the RRC portal, extract it, and place")
        print(f"the images folder under 'data/MP-DocVQA/images/'.")
        sys.exit(1)

    # 2. Extract qas.zip if JSON files do not exist
    json_splits = ["train.json", "val.json", "test.json"]
    need_extraction = any(not (root_dir / split).exists() for split in json_splits)
    if need_extraction:
        if qa_zip_path.exists():
            print(f"Extracting {qa_zip_path}...")
            with zipfile.ZipFile(qa_zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_dir)
            print("Extraction complete.")
        else:
            print(f"[ERROR] QA zip file not found at: {qa_zip_path}")
            sys.exit(1)

    # 3. Read JSON splits and gather unique contexts
    # A context is defined by a unique (doc_id, tuple(page_ids))
    unique_contexts = {} # context_id -> {"doc_id": ..., "page_ids": [...]}
    questions_map = {}   # question_id -> {"context_id": ..., "answer_page_idx": ..., "answers": ...}
    
    print("\nReading QA JSON files...")
    for split in ["train.json", "val.json", "test.json"]:
        split_file = root_dir / split
        if not split_file.exists():
            print(f"Skipping split '{split}' (file not found).")
            continue
            
        with open(split_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in tqdm(data["data"], desc=f"Processing {split}"):
            q_id = str(item["questionId"])
            doc_id = item["doc_id"]
            page_ids = item["page_ids"]
            
            # Sort page IDs to ensure they are in numerical order of pages
            # Format: {doc_id}_p{page_num}
            pnums = []
            for p in page_ids:
                match = re.search(r'_p(\d+)', p)
                if not match:
                    print(f"[WARN] Could not parse page number from page ID: {p}")
                    pnums.append(0)
                else:
                    pnums.append(int(match.group(1)))
            
            # Zip and sort page_ids based on page numbers
            sorted_page_ids = [p for _, p in sorted(zip(pnums, page_ids))]
            
            # Generate a unique context ID
            start_p = pnums[0] if pnums else 0
            end_p = pnums[-1] if pnums else 0
            context_id = f"{doc_id}_p{start_p}_p{end_p}"
            
            if context_id not in unique_contexts:
                unique_contexts[context_id] = {
                    "doc_id": doc_id,
                    "page_ids": sorted_page_ids,
                    "split": split.split(".")[0]
                }
                
            questions_map[q_id] = {
                "context_id": context_id,
                "answer_page_idx": item.get("answer_page_idx", None),
                "answers": item.get("answers", [])
            }

    print(f"\nFound {len(unique_contexts)} unique page contexts across splits.")
    print(f"Found {len(questions_map)} questions mapped to these contexts.")

    # 4. Compile PDFs for each unique context
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    print("\nCompiling contexts into multi-page PDFs...")
    
    compiled_count = 0
    missing_images_count = 0
    
    for context_id, ctx_info in tqdm(unique_contexts.items(), desc="Compiling PDFs"):
        pdf_path = pdfs_dir / f"{context_id}.pdf"
        
        # Check if PDF already exists
        if pdf_path.exists():
            compiled_count += 1
            continue
            
        # Get paths of all page images in this context
        img_paths = []
        missing = False
        for pid in ctx_info["page_ids"]:
            img_path = images_dir / f"{pid}.png"
            if not img_path.exists():
                img_path = images_dir / f"{pid}.jpg"
                
            if not img_path.exists():
                # Search recursively in case they are nested (e.g. by split folder)
                found_paths = list(images_dir.glob(f"**/{pid}.*"))
                if found_paths:
                    img_path = found_paths[0]
                else:
                    print(f"\n[WARN] Image not found for page ID: {pid}")
                    missing = True
                    break
            img_paths.append(img_path)
            
        if missing:
            missing_images_count += 1
            continue
            
        # Compile images into a single PDF using Pillow
        try:
            pil_images = [Image.open(p).convert("RGB") for p in img_paths]
            pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:])
            compiled_count += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to compile PDF for {context_id}: {e}")

    # 5. Save the mappings to context_map.json
    mapping_data = {
        "contexts": unique_contexts,
        "questions": questions_map
    }
    mapping_file = root_dir / "context_map.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)
        
    print("\n" + "=" * 52)
    print("   MP-DocVQA Preparation Complete")
    print("=" * 52)
    print(f"  Total compiled PDFs  : {compiled_count}/{len(unique_contexts)}")
    if missing_images_count > 0:
        print(f"  Missing image errors : {missing_images_count}")
    print(f"  Mapping saved to     : {mapping_file}")
    print("=" * 52)

if __name__ == "__main__":
    main()
