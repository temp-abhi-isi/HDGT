"""
parsers/docling_parser.py

Converts a PDF into a flat list of DocumentNode objects using Docling.

Key decisions:
  - Bbox normalisation: Docling uses bottom-left origin; we convert to top-left.
  - Caption nodes map to type='text', role='caption' (not a separate node type).
  - Section headings map to type='section' with appropriate role.
  - Each page gets one synthetic 'page' node (bbox covering full page).
  - document_id is derived from the PDF filename stem.
  - node_uid = "{document_id}_p{page}_n{node_id}" for cross-document uniqueness.
  - Figure images are saved to figures_dir if save_figures=True.
    Phase 2 will read these paths for Qwen2.5-VL visual encoding.
    Saving them now avoids re-parsing the entire PDF corpus in Phase 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from hdgt.graph.schema import DocumentNode, DOCLING_LABEL_MAP

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Wraps Docling's DocumentConverter and maps its output to DocumentNode objects.

    Usage
    -----
    >>> parser = DoclingParser()
    >>> nodes = parser.parse("path/to/document.pdf")
    >>> nodes = parser.parse("path/to/document.pdf", document_id="doc_001")
    >>> # Save figure images for Phase 2 Qwen encoding:
    >>> parser = DoclingParser(save_figures=True)
    >>> nodes = parser.parse("paper.pdf", figures_dir=Path("experiments/paper_figures"))
    """

    def __init__(self, verbose: bool = False, save_figures: bool = False):
        self.verbose      = verbose
        self.save_figures = save_figures
        self._converter   = None   # lazy-loaded on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        pdf_path: str | Path,
        document_id: Optional[str] = None,
        figures_dir: Optional[Path] = None,
    ) -> List[DocumentNode]:
        """
        Parse a PDF and return a flat list of DocumentNode objects.

        Parameters
        ----------
        pdf_path : str or Path
        document_id : str, optional
            Identifier for this document. Defaults to the PDF filename stem.
        figures_dir : Path, optional
            Directory to save extracted figure images.
            Required when save_figures=True (set at __init__).
            Each figure is saved as: {figures_dir}/page{page}_n{node_id}.png
            The path is stored in node.image_path for Phase 2 Qwen encoding.

        Returns
        -------
        List[DocumentNode]
            Page nodes first, then all content nodes in document order.
        """
        if self.save_figures and figures_dir is not None:
            figures_dir = Path(figures_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = document_id or pdf_path.stem
        logger.info(f"Parsing [{doc_id}]: {pdf_path.name}")

        converter = self._get_converter()
        result    = converter.convert(str(pdf_path))
        doc       = result.document

        # Track figure count per page for naming saved images
        fig_count_per_page: dict[int, int] = {}

        nodes: List[DocumentNode] = []
        node_id_counter = 0

        # ── Collect page dimensions for bbox normalisation ─────────────
        page_sizes: dict[int, Tuple[float, float]] = {}
        try:
            for pg in result.pages:
                w = float(pg.size.width)  if (pg.size and pg.size.width)  else 612.0
                h = float(pg.size.height) if (pg.size and pg.size.height) else 792.0
                page_sizes[pg.page_no] = (w, h)
        except Exception:
            logger.warning("Could not read page sizes — using A4 default (612×792)")

        def get_page_size(page_no: int) -> Tuple[float, float]:
            return page_sizes.get(page_no, (612.0, 792.0))

        num_pages = max(page_sizes.keys(), default=1)

        # ── One 'page' node per page ───────────────────────────────────
        page_node_map: dict[int, int] = {}   # 1-indexed page_no → node_id
        pages_to_create = sorted(page_sizes.keys()) if page_sizes else range(1, num_pages + 1)

        for pg_no in pages_to_create:
            w, h = get_page_size(pg_no)
            uid = f"{doc_id}_p{pg_no - 1}_n{node_id_counter}"
            node = DocumentNode(
                node_id=node_id_counter,
                node_uid=uid,
                document_id=doc_id,
                page=pg_no - 1,
                type="page",
                role="page",
                bbox=[0.0, 0.0, 1.0, 1.0],
                content=f"Page {pg_no}",
                metadata={"width_pt": w, "height_pt": h},
            )
            page_node_map[pg_no] = node_id_counter
            nodes.append(node)
            node_id_counter += 1

        # ── Iterate all document items ─────────────────────────────────
        for item, _ in doc.iterate_items():
            try:
                node = self._item_to_node(item, node_id_counter, doc_id, get_page_size)
                if node is not None:
                    # Save figure image if requested
                    if node.type == "figure" and self.save_figures and figures_dir is not None:
                        fig_idx = fig_count_per_page.get(node.page, 0)
                        fig_count_per_page[node.page] = fig_idx + 1
                        img_path = figures_dir / f"page{node.page}_fig{fig_idx}.png"
                        saved = self._try_save_figure(item, result, img_path)
                        if saved:
                            node.image_path = str(img_path)
                    nodes.append(node)
                    node_id_counter += 1
            except Exception as exc:
                logger.debug(f"Skipping item ({type(item).__name__}): {exc}")
                continue

        # Fallback: inject page nodes if Docling version didn't expose them
        if not any(n.type == "page" for n in nodes):
            pages_seen = sorted({n.page for n in nodes})
            fallback_pages = []
            for pg_idx in pages_seen:
                uid = f"{doc_id}_p{pg_idx}_page"
                fallback_pages.append(DocumentNode(
                    node_id=node_id_counter,
                    node_uid=uid,
                    document_id=doc_id,
                    page=pg_idx,
                    type="page",
                    role="page",
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    content=f"Page {pg_idx + 1}",
                ))
                node_id_counter += 1
            nodes = fallback_pages + nodes
            # Re-assign consecutive node_ids
            for idx, n in enumerate(nodes):
                n.node_id = idx

        logger.info(
            f"Parsed {len(nodes)} nodes from {num_pages} page(s) [{doc_id}]"
        )
        return nodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_converter(self):
        """Lazy-load Docling DocumentConverter, with picture extraction if requested."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            if self.save_figures:
                try:
                    from docling.datamodel.pipeline_options import PdfPipelineOptions
                    from docling.document_converter import PdfFormatOption
                    from docling.datamodel.base_models import InputFormat
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.generate_picture_images = True
                    self._converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_options=pipeline_options
                            )
                        }
                    )
                    logger.info("Docling configured with picture image extraction.")
                except Exception as exc:
                    logger.warning(
                        f"Could not enable picture extraction ({exc}). "
                        "Falling back to standard converter — image_path will be None."
                    )
                    self._converter = DocumentConverter()
            else:
                self._converter = DocumentConverter()
        return self._converter

    def _try_save_figure(self, item, result, out_path: Path) -> bool:
        """
        Attempt to save a figure item's image to disk.
        Returns True on success, False on any failure.

        Tries multiple Docling API approaches for compatibility across versions.
        """
        try:
            # Approach 1: item.image.pil_image (docling >= 2.5)
            if hasattr(item, "image") and item.image is not None:
                pil = getattr(item.image, "pil_image", None)
                if pil is not None:
                    pil.save(str(out_path))
                    return True

            # Approach 2: item.get_image(doc) (older docling versions)
            if hasattr(item, "get_image"):
                pil = item.get_image(result.document)
                if pil is not None:
                    pil.save(str(out_path))
                    return True

            logger.debug(f"No image extraction method available for {type(item).__name__}")
            return False

        except Exception as exc:
            logger.debug(f"Figure save failed for {out_path.name}: {exc}")
            return False

    def _item_to_node(
        self,
        item,
        node_id: int,
        doc_id: str,
        get_page_size,
    ) -> Optional[DocumentNode]:
        """Convert a single Docling item into a DocumentNode. Returns None to skip."""

        # ── Determine raw label ────────────────────────────────────────
        raw_label = ""
        try:
            raw_label = item.label.value.lower()
        except AttributeError:
            try:
                raw_label = str(item.label).lower()
            except Exception:
                raw_label = type(item).__name__.lower().replace("item", "")

        # Map to (node_type, role); default to text/paragraph for unknowns
        node_type, role = DOCLING_LABEL_MAP.get(raw_label, ("text", "paragraph"))

        # ── Extract page number (1-indexed from Docling) ───────────────
        page_no = 1
        try:
            if item.prov:
                page_no = item.prov[0].page_no
        except Exception:
            pass

        # ── Extract bounding box ───────────────────────────────────────
        bbox = [0.0, 0.0, 1.0, 1.0]
        try:
            if item.prov and item.prov[0].bbox:
                b = item.prov[0].bbox
                w, h = get_page_size(page_no)
                # Docling: bottom-left origin, points.
                # Convert: flip y to top-left origin, normalise.
                x1 = b.l / w
                y1 = 1.0 - (b.t / h)   # top in top-left = 1 - top_bottomleft/h
                x2 = b.r / w
                y2 = 1.0 - (b.b / h)
                # Ensure correct ordering & clamp
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                bbox = [
                    max(0.0, min(1.0, x1)),
                    max(0.0, min(1.0, y1)),
                    max(0.0, min(1.0, x2)),
                    max(0.0, min(1.0, y2)),
                ]
        except Exception as exc:
            logger.debug(f"Bbox extraction failed for {raw_label}: {exc}")

        # ── Extract text content ───────────────────────────────────────
        content = ""
        try:
            content = item.text or ""
        except AttributeError:
            try:
                content = item.export_to_markdown() or ""
            except Exception:
                pass

        # ── Extra metadata ─────────────────────────────────────────────
        metadata: dict = {"raw_label": raw_label}
        try:
            if hasattr(item, "data") and item.data is not None:
                tbl = item.data
                metadata["num_rows"] = getattr(tbl, "num_rows", 0)
                metadata["num_cols"] = getattr(tbl, "num_cols", 0)
        except Exception:
            pass

        uid = f"{doc_id}_p{page_no - 1}_n{node_id}"

        return DocumentNode(
            node_id=node_id,
            node_uid=uid,
            document_id=doc_id,
            page=page_no - 1,         # 0-indexed
            type=node_type,
            role=role,
            bbox=bbox,
            content=content[:2000],   # cap to avoid huge strings in memory
            metadata=metadata,
        )
