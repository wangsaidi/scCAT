"""Reproduce main Figure 2 — controlled-simulation hero panel
(Simulated 3 + Simulated 4, all-method UMAP grids + mechanism metrics).

The actual figure-builder code lives in `generators/plot_improved.py`
(`make_main_fig2_simulation`); this script is a thin wrapper that imports
it and writes the figure to `output/`.
"""

from __future__ import annotations
from pathlib import Path

from _common import attach_resource_root

# Edit RESOURCE_ROOT here if your data / plot_improved.py live elsewhere.
RESOURCE_ROOT: Path | None = None

attach_resource_root(RESOURCE_ROOT)

import plot_improved as P  # noqa: E402


if __name__ == "__main__":
    P.make_main_fig2_simulation()
    print("Figure 2 generated under generators/figures/  (PDF + SVG + PNG + TIFF)")
