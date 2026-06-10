"""Reproduce main Figure 5 — Sc_mixology (cross-platform) + PBMC
(condition-associated) combined figure with marker-gene panel.

See `generators/plot_improved.py::make_main_fig5_scmix_pbmc`.
"""

from __future__ import annotations
from pathlib import Path

from _common import attach_resource_root

RESOURCE_ROOT: Path | None = None
attach_resource_root(RESOURCE_ROOT)

import plot_improved as P  # noqa: E402


if __name__ == "__main__":
    P.make_main_fig5_scmix_pbmc()
    print("Figure 5 generated under generators/figures/  (PDF + SVG + PNG + TIFF)")
