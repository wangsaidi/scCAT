"""Shared utilities for the figure-reproduction wrappers.

Locates `plot_improved.py` (the canonical figure builders used in the
manuscript), preferring the `generators/` tree bundled in this release so the
package is self-contained, and inserts it (plus `generators/experiments/`) onto
sys.path. Falls back to an upstream `resource/` checkout only if the bundled
copy is absent.
"""

from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_resource_root() -> Path:
    """Locate the directory that contains plot_improved.py.
    Override by setting RESOURCE_ROOT in the calling script before importing
    this module."""
    # Prefer the generators/ tree bundled inside this release (self-contained);
    # fall back to an upstream <project>/resource/plot_improved.py checkout only
    # if the bundled copy is absent.
    candidates = [
        HERE.parent / "generators",
        HERE.parent.parent.parent / "resource",
        HERE.parent.parent / "resource",
        HERE.parent / "resource",
        Path.cwd() / "resource",
    ]
    for c in candidates:
        if (c / "plot_improved.py").exists():
            return c
    raise FileNotFoundError(
        "Could not locate plot_improved.py.  Set RESOURCE_ROOT explicitly at "
        "the top of your script (see README §3)."
    )


def attach_resource_root(resource_root: Path | None = None) -> Path:
    """Make plot_improved.py and plot_supp_figures.py importable. Returns
    the resolved resource_root."""
    root = resource_root or _find_resource_root()
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # plot_supp_figures.py lives one level deeper, under experiments/
    exp_dir = root / "experiments"
    if exp_dir.exists() and str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    print(f"[reproduce_figures] using plot code from: {root}")
    return root
