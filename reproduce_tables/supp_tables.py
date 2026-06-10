"""Reproduce Supplementary Tables S1 – S19 as a single xlsx workbook chain.

Runs the three table generators bundled in this release, in dependency order:

    build_supp_tables.py     -> Supplementary_Tables_S1-S17.xlsx   (builds S1 – S17)
    make_supp_table_S18.py   -> Supplementary_Tables_S1-S18.xlsx   (appends S18)
    make_supp_table_S19.py   -> Supplementary_Tables_S1-S19.xlsx   (appends S19)

All three write into `reproduce_tables/output/` next to this script; nothing is
written back into the shipped `Publish_version/` workbook.

See README.md for (a) the external-data dependency — the `results/` tree read by
these scripts is NOT bundled — and (b) the S16 caveat: the regenerated S16 sheet
is the legacy *average-rank* meta-summary, whereas the shipped
`Publish_version/Supplementary_Tables_S1-S19.xlsx` carries the authoritative
IB-based `S16_IB_results` (the Fig. 6e numerical source), which supersedes it.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the three table builders live in the bundled generators/experiments/ tree
GEN_EXP = HERE.parent / "generators" / "experiments"
if str(GEN_EXP) not in sys.path:
    sys.path.insert(0, str(GEN_EXP))
print(f"[reproduce_tables] using table code from: {GEN_EXP}")

import build_supp_tables as BST          # noqa: E402
import make_supp_table_S18 as PSF_S18    # noqa: E402
import make_supp_table_S19 as PSF_S19    # noqa: E402


# (label, callable) — must run in this order; each step reads the previous
# step's workbook out of reproduce_tables/output/.
CHAIN = [
    ("S1 – S17  (build_supp_tables)",        BST.main),
    ("S18       (scIB-standard robustness)", PSF_S18.main),
    ("S19       (HDC separability)",         PSF_S19.main),
]


if __name__ == "__main__":
    for label, fn in CHAIN:
        try:
            print(f"\n=== Supplementary Tables {label} ===")
            fn()
        except Exception as exc:
            print(f"  [ERROR] {label} failed: {exc}")
            print("  (later steps depend on this one and will also fail)")

    print("\nWorkbook chain written under reproduce_tables/output/")
