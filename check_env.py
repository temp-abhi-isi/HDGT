#!/usr/bin/env python
"""
check_env.py — Verify PyTorch + CUDA environment before installing PyG.

Run this BEFORE installing torch-geometric and docling.

Usage:
    python check_env.py

Expected output (success):
    [✓] Python       3.11.x
    [✓] PyTorch      2.5.x
    [✓] CUDA avail   True
    [✓] CUDA version 12.x
    [✓] GPU name     NVIDIA GeForce RTX 2050
    [✓] Tensor ops   OK (GPU round-trip passed)

If any line shows [✗], fix PyTorch before installing PyG or docling.
"""

import sys


def check(label: str, ok: bool, detail: str = "") -> None:
    icon  = "✓" if ok else "✗"
    color = "\033[92m" if ok else "\033[91m"
    reset = "\033[0m"
    line  = f"  [{color}{icon}{reset}] {label:<18} {detail}"
    print(line)
    return ok


def main() -> None:
    print()
    print("═" * 52)
    print("   HDGT — Environment Check")
    print("═" * 52)
    print()

    all_ok = True

    # ── Python version ─────────────────────────────────────────────────
    py = sys.version_info
    py_ok = py.major == 3 and py.minor >= 10
    all_ok &= check("Python",      py_ok, f"{py.major}.{py.minor}.{py.micro}")

    # ── PyTorch ────────────────────────────────────────────────────────
    try:
        import torch
        torch_ok = True
        torch_ver = torch.__version__
    except ImportError:
        torch_ok = False
        torch_ver = "NOT INSTALLED"
    all_ok &= check("PyTorch",     torch_ok, torch_ver)

    if not torch_ok:
        print()
        print("  PyTorch is not installed. Run:")
        print("  pip install torch torchvision torchaudio \\")
        print("      --index-url https://download.pytorch.org/whl/cu124")
        print()
        sys.exit(1)

    # ── CUDA availability ──────────────────────────────────────────────
    cuda_avail = torch.cuda.is_available()
    all_ok &= check("CUDA avail",  cuda_avail, str(cuda_avail))

    cuda_ver_str = torch.version.cuda or "N/A"
    cuda_ver_ok  = cuda_avail
    all_ok &= check("CUDA version", cuda_ver_ok, cuda_ver_str)

    # ── GPU name ───────────────────────────────────────────────────────
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_ok   = True
    else:
        gpu_name = "No GPU found"
        gpu_ok   = False
    all_ok &= check("GPU name",    gpu_ok, gpu_name)

    # ── GPU memory ─────────────────────────────────────────────────────
    if cuda_avail:
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem_ok    = mem_total > 1.0
        all_ok   &= check("GPU memory", mem_ok, f"{mem_total:.1f} GB")

    # ── Tensor round-trip on GPU ───────────────────────────────────────
    if cuda_avail:
        try:
            x   = torch.randn(64, 64, device="cuda")
            y   = (x @ x.T).sum()
            rtt_ok = y.is_cuda and not torch.isnan(y)
        except Exception as exc:
            rtt_ok = False
        all_ok &= check("Tensor ops", rtt_ok, "GPU round-trip passed" if rtt_ok else "FAILED")

    # ── Optional: torch-geometric ──────────────────────────────────────
    try:
        import torch_geometric as pyg
        pyg_ver = pyg.__version__
        pyg_ok  = True
    except ImportError:
        pyg_ver = "NOT INSTALLED"
        pyg_ok  = False
    check("torch-geometric", pyg_ok, pyg_ver)   # not required, don't affect all_ok

    # ── Optional: docling ──────────────────────────────────────────────
    try:
        import docling
        dl_ver = getattr(docling, "__version__", "installed")
        dl_ok  = True
    except ImportError:
        dl_ver = "NOT INSTALLED"
        dl_ok  = False
    check("docling", dl_ok, dl_ver)   # not required, don't affect all_ok

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("═" * 52)
    if all_ok:
        print("  \033[92m✓ All checks passed — safe to install PyG and docling.\033[0m")
        print()
        print("  Next steps:")
        print("    pip install -r requirements.txt")
    else:
        print("  \033[91m✗ Some checks failed — fix PyTorch/CUDA first.\033[0m")
        print()
        print("  Fix with:")
        print("    pip install torch torchvision torchaudio \\")
        print("        --index-url https://download.pytorch.org/whl/cu124")
    print("═" * 52)
    print()


if __name__ == "__main__":
    main()
