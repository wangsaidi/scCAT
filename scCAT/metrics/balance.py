"""Integration Balance (IB) - a single trade-off score proposed in scCAT.

::

    s_batch = mean(iLISI, 1 - kBET, ASW_batch_mixing)
    s_bio   = mean(ARI, NMI, ASW_celltype, cLISI_purity)
    IB      = sqrt(s_batch * s_bio)

IB is the geometric mean of a batch-removal score and a biology-conservation
score, so a method scores well only when it simultaneously mixes batches *and*
preserves biological structure.  Components missing from a particular table are
silently skipped (each side is averaged over the components that are present).

Verbatim re-implementation of the formula used to generate the manuscript's
Fig. 6 / Supplementary Table S16 numbers; see
``generators/plot_improved.py::compute_integration_balance``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from ._tables import get_value


class BalanceScore(NamedTuple):
    """``(s_batch, s_bio, balance)`` - tuple-compatible with the original API."""

    s_batch: float
    s_bio: float
    balance: float


def integration_balance(
    batch_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    method: str,
) -> BalanceScore:
    """Compute the Integration Balance for one *method*.

    Parameters
    ----------
    batch_table:
        Tidy table holding the batch-removal / biology columns
        (``iLISI``, ``kBET``, ``ASW_batch_mixing``, ``ASW_celltype``,
        ``cLISI_purity``) - e.g. ``data/metric/batch_remove/<dataset>.csv``.
    cluster_table:
        Tidy table holding the clustering columns (``ARI``, ``NMI``) -
        e.g. ``data/metric/cluster/<dataset>.csv``.
    method:
        Method name (``"scCAT"`` / ``"BTCA"`` / any baseline).

    Returns
    -------
    BalanceScore
        ``(s_batch, s_bio, balance)``; ``(nan, nan, nan)`` if either side has no
        usable component.
    """
    iLISI = get_value(batch_table, method, ["iLISI"])
    kBET = get_value(batch_table, method, ["kBET"])
    asw_batch = get_value(batch_table, method, ["ASW_batch_mixing", "ASW_batch"])
    ari = get_value(cluster_table, method, ["ARI"])
    nmi = get_value(cluster_table, method, ["NMI"])
    asw_ct = get_value(batch_table, method, ["ASW_celltype", "ASW_cell_type"])
    clisi = get_value(batch_table, method, ["cLISI_purity", "cLISI"])

    batch_components = []
    if np.isfinite(iLISI):
        batch_components.append(iLISI)
    if np.isfinite(kBET):
        batch_components.append(1.0 - kBET)
    if np.isfinite(asw_batch):
        batch_components.append(asw_batch)

    bio_components = [v for v in (ari, nmi, asw_ct, clisi) if np.isfinite(v)]

    if not batch_components or not bio_components:
        return BalanceScore(float("nan"), float("nan"), float("nan"))

    s_batch = max(float(np.mean(batch_components)), 0.0)
    s_bio = max(float(np.mean(bio_components)), 0.0)
    return BalanceScore(s_batch, s_bio, float(np.sqrt(s_batch * s_bio)))
