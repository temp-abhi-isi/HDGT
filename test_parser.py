#!/usr/bin/env python
"""
test_parser.py — Raw Docling diagnostic script.

Run this on 3–5 PDFs BEFORE using build_graph.py.
It directly inspects what Docling actually extracts so you can verify:

  ✓  Are bounding boxes populated (non-zero)?
  ✓  Are section headings correctly labeled (not confused with captions)?
  ✓  Are figures / tables detected?
  ✓  Are captions extracted and separated from body text?

Usage
-----
  python test_parser.py paper.pdf
  python test_parser.py paper.pdf --max-items 60
  python test_parser.py paper.pdf --page 3
  python test_parser.py paper.pdf --sections-only
  python test_parser.py paper.pdf --export-json diag_paper.json

Output
------
  Prints a table of all extracted items with label, bbox, and content.
  Flags items with zero/missing bboxes (⚠).
  Summarises label distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ── Label → (type, role) map (mirrors schema.py) ─────────────────────────────
LABEL_MAP: dict[str, tuple[str, str]] = {
    "title":           ("section", "title"),
    "section_header":  ("section", "section_header"),
    "text":            ("text",    "paragraph"),
    "paragraph":       ("text",    "paragraph"),
    "list_item":       ("text",    "paragraph"),
    "page_header":     ("text",    "header"),
    "page_footer":     ("text",    "footer"),
    "footnote":        ("text",    "paragraph"),
    "formula":         ("text",    "paragraph"),
    "code":            ("text",    "paragraph"),
    "caption":         ("text",    "caption"),
    "table":           ("table",   "table"),
    "picture":         ("figure",  "figure"),
    "figure":          ("figure",  "figure"),
    "chart":           ("figure",  "figure"),
}

COL_W = (6, 20, 10, 10, 38, 5)   # widths for table columns


def hr(char: str = "─") -> str:
    return char * (sum(COL_W) + len(COL_W) * 3 + 1)


def row(*cells, widths=COL_W) -> str:
    parts = []
    for cell, w in zip(cells, widths):
        s = str(cell)
        parts.append(s[:w].ljust(w))
    return "│ " + " │ ".join(parts) + " │"


def _get_raw_label(item) -> str:
    try:
        return item.label.value.lower()
    except AttributeError:
        try:
            return str(item.label).lower()
        except Exception:
            return type(item).__name__.lower().replace("item", "")


def _get_page_no(item) -> int:
    try:
        if item.prov:
            return item.prov[0].page_no
    except Exception:
        pass
    return -1


def _get_bbox_raw(item):
    """Return raw bbox dict or None."""
    try:
        if item.prov and item.prov[0].bbox:
            b = item.prov[0].bbox
            return {"l": b.l, "t": b.t, "r": b.r, "b": b.b}
    except Exception:
        pass
    return None


def _bbox_is_valid(bbox_dict: Optional[dict]) -> bool:
    if bbox_dict is None:
        return False
    vals = [bbox_dict["l"], bbox_dict["t"], bbox_dict["r"], bbox_dict["b"]]
    return not all(v == 0.0 for v in vals)


def _get_content(item) -> str:
    try:
        return (item.text or "").strip()
    except AttributeError:
        try:
            return (item.export_to_markdown() or "").strip()
        except Exception:
            return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Docling diagnostic — inspect raw parser output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pdf_path",         type=str, help="PDF to inspect.")
    parser.add_argument("--max-items",      type=int, default=80,
                        help="Max items to print in the detail table (default: 80).")
    parser.add_argument("--page",           type=int, default=None,
                        help="Filter to a specific page number (1-indexed).")
    parser.add_argument("--sections-only",  action="store_true",
                        help="Only print items mapped to type='section'.")
    parser.add_argument("--figures-only",   action="store_true",
                        help="Only print items mapped to type='figure'.")
    parser.add_argument("--export-json",    type=str, default=None,
                        help="Export full item list to a JSON file.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load Docling ──────────────────────────────────────────────────
    print(f"\n  Loading Docling...")
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("[ERROR] docling not installed. Run: pip install docling", file=sys.stderr)
        sys.exit(1)

    converter = DocumentConverter()
    print(f"  Parsing: {pdf_path.name}  (this may take 30–120s on first run)\n")
    result   = converter.convert(str(pdf_path))
    doc      = result.document

    # ── Collect page sizes ────────────────────────────────────────────
    page_sizes: dict[int, tuple[float, float]] = {}
    try:
        for pg in result.pages:
            w = float(pg.size.width)  if (pg.size and pg.size.width)  else 612.0
            h = float(pg.size.height) if (pg.size and pg.size.height) else 792.0
            page_sizes[pg.page_no] = (w, h)
    except Exception:
        pass

    # ── Gather all items ──────────────────────────────────────────────
    all_items = []
    for item, _ in doc.iterate_items():
        raw_label = _get_raw_label(item)
        node_type, role = LABEL_MAP.get(raw_label, ("text", "paragraph"))
        page_no   = _get_page_no(item)
        bbox_raw  = _get_bbox_raw(item)
        bbox_ok   = _bbox_is_valid(bbox_raw)
        content   = _get_content(item)

        # Normalise bbox if page size known
        bbox_norm = None
        if bbox_raw and page_no in page_sizes:
            w, h = page_sizes[page_no]
            x1 = max(0.0, min(1.0, bbox_raw["l"] / w))
            y1 = max(0.0, min(1.0, 1.0 - bbox_raw["t"] / h))
            x2 = max(0.0, min(1.0, bbox_raw["r"] / w))
            y2 = max(0.0, min(1.0, 1.0 - bbox_raw["b"] / h))
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            bbox_norm = [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]

        all_items.append({
            "raw_label":  raw_label,
            "node_type":  node_type,
            "role":       role,
            "page_no":    page_no,
            "bbox_raw":   bbox_raw,
            "bbox_norm":  bbox_norm,
            "bbox_ok":    bbox_ok,
            "content":    content,
        })

    # ── Filters ───────────────────────────────────────────────────────
    filtered = all_items
    if args.page is not None:
        filtered = [x for x in filtered if x["page_no"] == args.page]
    if args.sections_only:
        filtered = [x for x in filtered if x["node_type"] == "section"]
    if args.figures_only:
        filtered = [x for x in filtered if x["node_type"] == "figure"]

    # ── Print label distribution ──────────────────────────────────────
    from collections import Counter
    label_counts = Counter(x["raw_label"] for x in all_items)
    type_counts  = Counter(x["node_type"] for x in all_items)

    print("╔" + "═" * 50 + "╗")
    print(f"║  {pdf_path.name[:46]:<46}  ║")
    print("╚" + "═" * 50 + "╝")
    print(f"\n  Total pages : {max(page_sizes.keys(), default='?')}")
    print(f"  Total items : {len(all_items)}")
    print(f"  Items shown : {min(len(filtered), args.max_items)}")

    print("\n  ── Raw label distribution (Docling) ──")
    for label, count in label_counts.most_common():
        mapped = LABEL_MAP.get(label, ("?", "?"))
        bar = "█" * min(count, 40)
        print(f"    {label:<20} {count:>5}  → {mapped[0]}+{mapped[1]}  {bar}")

    print("\n  ── Mapped type distribution (HDGT) ──")
    for ntype, count in type_counts.most_common():
        bar = "█" * min(count, 40)
        print(f"    {ntype:<14} {count:>5}  {bar}")

    # ── Bbox validity ─────────────────────────────────────────────────
    valid_bbox  = sum(1 for x in all_items if x["bbox_ok"])
    invalid_bbox = len(all_items) - valid_bbox
    pct = valid_bbox / max(len(all_items), 1) * 100
    print(f"\n  ── Bounding box validity ──")
    print(f"    Valid   : {valid_bbox}/{len(all_items)} ({pct:.1f}%)")
    if invalid_bbox > 0:
        print(f"    ⚠ {invalid_bbox} items have missing/zero bboxes")
        print(f"    Labels without bbox:")
        missing_labels = Counter(
            x["raw_label"] for x in all_items if not x["bbox_ok"]
        )
        for label, count in missing_labels.most_common(10):
            print(f"      {label}: {count}")

    # ── Section detection spot-check ──────────────────────────────────
    sections = [x for x in all_items if x["node_type"] == "section"]
    print(f"\n  ── Section nodes ({len(sections)} found) ──")
    if sections:
        for s in sections[:20]:
            flag = "" if s["bbox_ok"] else "  ⚠ no bbox"
            print(f"    p{s['page_no']:>3}  [{s['role']:<15}]  {s['content'][:60]!r}{flag}")
        if len(sections) > 20:
            print(f"    ... and {len(sections) - 20} more")
    else:
        print("    ⚠ No section nodes detected!")
        print("    Check if Docling is labeling headings as 'section_header' or 'title'.")
        print("    If not, you may need a custom heading heuristic.")

    # ── Figure detection spot-check ───────────────────────────────────
    figures = [x for x in all_items if x["node_type"] == "figure"]
    print(f"\n  ── Figure nodes ({len(figures)} found) ──")
    if figures:
        for f in figures[:10]:
            bbox_str = str(f["bbox_norm"]) if f["bbox_norm"] else "⚠ no bbox"
            print(f"    p{f['page_no']:>3}  bbox={bbox_str}  {f['content'][:40]!r}")
    else:
        print("    ⚠ No figure nodes detected!")

    # ── Caption spot-check ────────────────────────────────────────────
    captions = [x for x in all_items if x["role"] == "caption"]
    print(f"\n  ── Caption nodes (type=text, role=caption) — {len(captions)} found ──")
    for c in captions[:10]:
        print(f"    p{c['page_no']:>3}  {c['content'][:70]!r}")

    # ── Detail table ──────────────────────────────────────────────────
    to_show = filtered[:args.max_items]
    print(f"\n  ── Item detail table (first {len(to_show)} of {len(filtered)}) ──")
    print()
    print(row("#", "raw_label", "→ type", "role", "content (preview)", "bbox?",
              widths=COL_W))
    print(row(*["─" * w for w in COL_W], widths=COL_W))

    for i, it in enumerate(to_show):
        bbox_flag = "✓" if it["bbox_ok"] else "⚠"
        content_preview = it["content"].replace("\n", " ")[:38]
        print(row(
            i,
            it["raw_label"],
            it["node_type"],
            it["role"],
            content_preview,
            bbox_flag,
            widths=COL_W,
        ))

    # ── Export JSON ───────────────────────────────────────────────────
    if args.export_json:
        out = Path(args.export_json)
        # Make serialisable
        export_data = []
        for it in all_items:
            export_data.append({k: v for k, v in it.items() if k != "bbox_raw"})
        with open(out, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n  ✓ Exported {len(export_data)} items → {out}")

    print()


if __name__ == "__main__":
    main()
