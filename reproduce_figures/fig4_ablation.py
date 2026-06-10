"""Reproduce main Figure 4 — module ablation (knockout matrix + dumbbells +
balance heatmap).  See `generators/plot_improved.py::make_main_fig4_ablation`.
"""

from __future__ import annotations
from pathlib import Path

from _common import attach_resource_root

RESOURCE_ROOT: Path | None = None
attach_resource_root(RESOURCE_ROOT)

import plot_improved as P  # noqa: E402


if __name__ == "__main__":
    P.make_main_fig4_ablation()
    print("Figure 4 generated under generators/figures/  (PDF + SVG + PNG + TIFF)")
