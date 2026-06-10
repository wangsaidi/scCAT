"""Helpers for reading per-method values out of a standard metric table.

A *metric table* here is a tidy ``DataFrame`` with one row per integration
method and one column per metric - the exact format shipped under
``data/metric/batch_remove/<dataset>.csv`` and
``data/metric/cluster/<dataset>.csv``.  These helpers make the lookups robust
to the small column-name and method-name variations that occur across the
upstream pipelines (e.g. ``method`` vs ``Method``, ``cLISI`` vs
``cLISI_purity``, the public name ``scCAT`` vs the internal key ``BTCA``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Internal codename used in the shipped data tree <-> public method name.
_METHOD_ALIASES = {
    "scCAT": "BTCA",
    "BTCA": "scCAT",
    "INSCT": "INSCT_Unsupervised",
    "INSCT_Unsupervised": "INSCT",
}


def resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first column in *candidates* present in *df* (case-insensitive)."""
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _method_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in ("method", "methods"):
            return c
    return df.columns[0]


def get_value(df: pd.DataFrame, method: str, candidates: Sequence[str]) -> float:
    """Look up a numeric metric value for *method* in *df*.

    *candidates* is a list of acceptable column names; the first that exists is
    used.  Method matching is exact, then falls back to the
    ``scCAT`` <-> ``BTCA`` / ``INSCT`` <-> ``INSCT_Unsupervised`` aliases so the
    same call works whether the table stores the public name or the internal
    codename.  Anything missing returns ``np.nan``.
    """
    col = resolve_column(df, candidates)
    if col is None:
        return float("nan")
    mcol = _method_column(df)
    names = df[mcol].astype(str).str.strip()
    row = df[names == str(method)]
    if row.empty and method in _METHOD_ALIASES:
        row = df[names == _METHOD_ALIASES[method]]
    if row.empty:
        return float("nan")
    return float(pd.to_numeric(row[col], errors="coerce").iloc[0])
