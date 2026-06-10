"""scCAT integration-quality metrics, callable independently of the figure code.

This submodule packages the metrics used to evaluate batch integration in the
scCAT paper so they can be imported and called on their own::

    from scCAT.metrics import (
        compute_oci, compute_bsrs, integration_balance,
        ari, nmi, asw_celltype, asw_batch_mixing,
    )

Novel metrics proposed in this work
-----------------------------------
* ``compute_oci``          - Overcorrection Index (lower is better)
* ``compute_bsrs``         - Batch-Specific Retention Score (higher is better)
* ``integration_balance``  - Integration Balance, ``sqrt(batch x bio)`` (higher is better)

These three are verbatim re-implementations of the formulas used to generate the
manuscript's figures and Supplementary Table S16
(``generators/plot_improved.py::compute_oci`` / ``compute_bsrs`` /
``compute_integration_balance``).

Standard metrics (re-exposed)
-----------------------------
* ``ari`` / ``nmi``                        - clustering agreement (scikit-learn)
* ``asw_celltype`` / ``asw_batch_mixing``  - silhouette-based biology / batch scores

The remaining neighbourhood-graph metrics (kBET, iLISI, cLISI) follow their
standard ``scib`` definitions (Luecken et al., *Nat. Methods* 2022) and feed
``integration_balance`` through the shipped per-method metric tables; use
``get_value`` / ``resolve_column`` to read any metric out of those tables.
"""
from __future__ import annotations

from ._tables import get_value, resolve_column
from .balance import BalanceScore, integration_balance
from .mechanism import compute_bsrs, compute_oci
from .standard import ari, asw_batch_mixing, asw_celltype, nmi

__all__ = [
    "compute_oci",
    "compute_bsrs",
    "integration_balance",
    "BalanceScore",
    "ari",
    "nmi",
    "asw_celltype",
    "asw_batch_mixing",
    "get_value",
    "resolve_column",
]
