"""Reproduce main Figure 6 — Mouse Lung (atlas-level) + cross-dataset
Integration Balance summary + scCAT scaling on the 10x 73k PBMC downsampling
benchmark.

See `generators/plot_improved.py::make_main_fig6_lung_summary`.
"""

from __future__ import annotations
from pathlib import Path

from _common import attach_resource_root

RESOURCE_ROOT: Path | None = None
attach_resource_root(RESOURCE_ROOT)

import plot_improved as P  # noqa: E402


if __name__ == "__main__":
    P.make_main_fig6_lung_summary()
    print("Figure 6 generated under generators/figures/  (PDF + SVG + PNG + TIFF)")
