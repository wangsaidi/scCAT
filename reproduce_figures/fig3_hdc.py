"""Reproduce main Figure 3 — HDC dendritic-cell flagship figure
(scCAT hero pair + competitor failure modes + per-cell-type purity +
marker-gene dot plot).

See `generators/plot_improved.py::make_main_fig3_hdc` for the implementation.
"""

from __future__ import annotations
from pathlib import Path

from _common import attach_resource_root

RESOURCE_ROOT: Path | None = None
attach_resource_root(RESOURCE_ROOT)

import plot_improved as P  # noqa: E402


if __name__ == "__main__":
    P.make_main_fig3_hdc()
    print("Figure 3 generated under generators/figures/  (PDF + SVG + PNG + TIFF)")
