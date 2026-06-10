"""
graph/node_builder.py

Converts a flat List[DocumentNode] into per-type feature tensors for PyG HeteroData.

Phase 1 feature vector — 9 dimensions (purely geometric + role):
    [x1, y1, x2, y2, page_norm, width, height, area, role_id]

Phase 2 will concatenate semantic embeddings from Qwen2.5-VL (768/4096-dim)
to these geometric features before message-passing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from hdgt.graph.schema import DocumentNode, NODE_TYPES, ROLE_IDS

logger = logging.getLogger(__name__)

FEATURE_DIM = 9   # 8 geometric + 1 role_id


class NodeBuilder:
    """
    Groups DocumentNode objects by type and builds per-type feature tensors.

    After calling build():
      features         : Dict[str, Tensor]          shape (N_type, FEATURE_DIM)
      node_lists       : Dict[str, List[DocumentNode]]
      global_to_local  : Dict[int, Tuple[str, int]]  global_id → (type, local_idx)
      local_to_global  : Dict[Tuple[str,int], int]   (type, local_idx) → global_id
      num_pages        : int
    """

    def __init__(self) -> None:
        self.features: Dict[str, torch.Tensor] = {}
        self.node_lists: Dict[str, List[DocumentNode]] = defaultdict(list)
        self.global_to_local: Dict[int, Tuple[str, int]] = {}
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        self.num_pages: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, nodes: List[DocumentNode]) -> None:
        """
        Process a flat list of DocumentNode objects.

        Parameters
        ----------
        nodes : List[DocumentNode]
            Output from DoclingParser.parse().
        """
        self.node_lists.clear()
        self.global_to_local.clear()
        self.local_to_global.clear()
        self.features.clear()

        self.num_pages = max((n.page for n in nodes), default=0) + 1

        # ── Group by type ──────────────────────────────────────────────
        for node in nodes:
            if node.type not in NODE_TYPES:
                logger.warning(f"Unknown node type {node.type!r} — skipping.")
                continue
            local_idx = len(self.node_lists[node.type])
            self.node_lists[node.type].append(node)
            self.global_to_local[node.node_id] = (node.type, local_idx)
            self.local_to_global[(node.type, local_idx)] = node.node_id

        # ── Build feature tensors ─────────────────────────────────────
        for ntype, node_list in self.node_lists.items():
            feat = torch.zeros(len(node_list), FEATURE_DIM, dtype=torch.float32)
            for local_idx, node in enumerate(node_list):
                feat[local_idx] = self._node_to_feature(node)
            self.features[ntype] = feat

        counts = {t: len(lst) for t, lst in self.node_lists.items()}
        logger.info(f"NodeBuilder: {counts}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_to_feature(self, node: DocumentNode) -> torch.Tensor:
        """
        9-dim feature vector:
          0:x1  1:y1  2:x2  3:y2
          4:page_norm (position in document, 0→1)
          5:width   6:height   7:area
          8:role_id (categorical integer, see ROLE_IDS)
        """
        x1, y1, x2, y2 = node.bbox
        width     = x2 - x1
        height    = y2 - y1
        area      = width * height
        page_norm = node.page / max(self.num_pages - 1, 1)
        role_id   = float(ROLE_IDS.get(node.role, 0))

        return torch.tensor(
            [x1, y1, x2, y2, page_norm, width, height, area, role_id],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        print("\n" + "=" * 54)
        print("  NODE SUMMARY")
        print("=" * 54)
        total = 0
        for ntype in NODE_TYPES:
            count = len(self.node_lists.get(ntype, []))
            total += count
            # Show role breakdown for 'text' nodes
            if ntype == "text" and count > 0:
                role_counts: dict = defaultdict(int)
                for n in self.node_lists["text"]:
                    role_counts[n.role] += 1
                role_str = ", ".join(f"{r}:{c}" for r, c in sorted(role_counts.items()))
                print(f"  {ntype:<12} : {count:>6} nodes  [{role_str}]")
            else:
                print(f"  {ntype:<12} : {count:>6} nodes")
            total += 0   # already counted above
        print("-" * 54)
        # recount properly
        total = sum(len(lst) for lst in self.node_lists.values())
        print(f"  {'TOTAL':<12} : {total:>6} nodes")
        print("=" * 54 + "\n")

    def get_local_index(self, global_id: int) -> Tuple[str, int]:
        """Return (node_type, local_index) for a given global node_id."""
        return self.global_to_local[global_id]

    def get_global_id(self, node_type: str, local_idx: int) -> int:
        """Return global node_id for a (node_type, local_index) pair."""
        return self.local_to_global[(node_type, local_idx)]
