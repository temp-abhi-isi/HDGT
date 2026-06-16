Here is the comprehensive, end-to-end project plan for building the **Heterogeneous Document Graph Transformer (HDGT)**, including the exact datasets, benchmarks, and state-of-the-art baselines we will use to prove our novelty.

### 1. Overall Project Plan
To build and validate the HDGT architecture efficiently, we will execute the project in five distinct phases:
*   **Phase 1: Multi-Modal Parsing & Graph Extraction:** We will use high-fidelity parsers like **MinerU** or **Docling** to process raw PDFs. These tools will extract text blocks, tables, images, and exact bounding box coordinates to form the initial nodes of our document graph.
*   **Phase 2: Vision-Language Encoding & Token Compression:** We will initialize **Qwen2.5-VL (7B)** as our frozen multimodal backbone. To handle token inflation, we will implement a High-resolution DocCompressor that downsizes page features into a compact, fixed set of layout-aware tokens (e.g., 324 tokens per page).
*   **Phase 3: HDGT Graph Neural Network Implementation:** Using PyTorch Geometric, we will code your custom GNN layer. This includes the **Disentangled Spatial-Semantic Attention** (which separates text meaning from 2D physical distances) and **Hourglass Token Merging** to aggregate dense local nodes dynamically.
*   **Phase 4: ALR Data Distillation & Supervised Fine-Tuning (SFT):** Because existing datasets only provide short answers, we will use a strong teacher model (like GPT-4o or Gemini-2.5-Flash) to distill training data into our structured **Analysis-Localization-Reasoning (ALR)** format. We will then fine-tune the generator module on this specific trace.
*   **Phase 5: Hybrid Retrieval & Benchmark Evaluation:** We will combine vector similarity search with $l$-hop graph traversal to retrieve the connected reasoning subgraphs, and then evaluate the entire pipeline against the SOTA benchmarks.

### 2. Datasets for Training & Distillation
We will train and fine-tune the model using a mixture of single-page layout datasets and complex multi-page document datasets:
*   **FUNSD, CORD, & PubLayNet:** Used for initial pipeline testing and single-page spatial relationship modeling to ensure the graph accurately captures forms, receipts, and scientific layouts.
*   **MP-DocVQA:** A multi-page document dataset focused on fine-grained information extraction and explicit reasoning over page indices. We will use this to train cross-page navigation.
*   **DUDE (Document Understanding Dataset and Evaluation):** Contains visually rich documents from diverse domains (medical, legal, financial) with strong emphasis on structured layouts like tables and forms. 

### 3. Benchmarks for Evaluation
To prove that our custom graph approach defeats the "Linear Adjacency Trap" and token inflation, we will benchmark against the heaviest long-document evaluation sets available:
*   **MMLongBench-Doc:** The ultimate stress test for long-context understanding. It features documents extending up to 468 pages and evaluates multi-hop reasoning across vast multimodal contexts.
*   **LongDocURL:** Evaluates long-range retrieval and the ability to locate and synthesize information across distant sections in complex web-based multi-modal documents (average length 30 pages).
*   **SlideVQA:** Tests structural and numerical reasoning across multi-image slide presentations, where visual cues and text are heavily interleaved.

### 4. Baselines for Comparison
To demonstrate that HDGT is SOTA for an IEEE Transactions submission, we will categorize and compare our results against the latest 2025/2026 architectures across three distinct paradigms:
*   **End-to-End MLLMs (The Sequence Models):** We will compare against models that process full document pages linearly, such as **mPLUG-DocOwl2** and **InternVL3**. We aim to show that our graph approach uses significantly less VRAM and latency than these massive sequence models.
*   **Visual RAG Methods (The "Bag-of-Patches" Models):** We will test against the latest visual retrievers like **ColPali**, **VisRAG**, and **URaG**. We will prove that our $l$-hop graph retrieval achieves higher accuracy than their isolated page-matching by retaining cross-page topological context.
*   **Agentic Frameworks (The Slow Reasoners):** We will compare against autonomous agent systems like **Doc-V\*** and **CogDoc**, proving that our pre-context graph subgraph retrieval achieves the same high-level reasoning accuracy without the massive inference latency of repeated multi-agent loops.
*   **Closed-Source APIs:** We will establish a ceiling by comparing our open-source, resource-efficient model against proprietary giants like **GPT-4o**, **Claude-3.5-Sonnet**, and **Gemini-1.5-Pro**.