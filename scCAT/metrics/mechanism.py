"""Mechanism-probing metrics proposed in scCAT.

Both operate directly on a 2-D embedding (e.g. a UMAP) plus its cell-type
labels, and were designed to probe scCAT's central mechanism - the protection
of *batch-specific* cell populations against over-correction:

* **OCI - Overcorrection Index** (``compute_oci``, lower is better): the
  fraction of batch-specific cells absorbed into a cluster whose dominant label
  is some *other* cell type.
* **BSRS - Batch-Specific Retention Score** (``compute_bsrs``, higher is
  better): the mean silhouette of the batch-specific cells against the full
  cell-type label space.

Verbatim re-implementation of ``generators/plot_improved.py::compute_oci`` /
``compute_bsrs``.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd  # noqa: F401  (kept for the DataFrame type hint / API parity)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

_COORDS = ("UMAP1", "UMAP2")
_CELLTYPE_COL = "cell_type"


def compute_oci(
    embedding: "pd.DataFrame",
    batch_specific_types: Sequence[str],
    n_clusters: int,
    *,
    coord_cols: Sequence[str] = _COORDS,
    celltype_col: str = _CELLTYPE_COL,
) -> float:
    """Overcorrection Index (lower is better).

    K-means (``k = n_clusters``, ``n_init=10``, ``random_state=0``) is run on the
    embedding coordinates and each cluster is assigned its majority cell type.
    For every batch-specific cell type we measure the fraction of its cells that
    land in a cluster whose majority label is *not* that type, then average over
    the batch-specific types.

    Parameters
    ----------
    embedding:
        DataFrame with the 2-D coordinate columns (default ``UMAP1``/``UMAP2``)
        and a cell-type column (default ``cell_type``).
    batch_specific_types:
        The cell types that are present in only a subset of batches.
    n_clusters:
        Number of k-means clusters (typically the number of cell types).
    """
    X = embedding[list(coord_cols)].values
    if len(X) < n_clusters:
        return float("nan")
    try:
        preds = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(X)
    except Exception:
        return float("nan")
    ct = embedding[celltype_col].astype(str).values
    cluster_dom = {}
    for c in np.unique(preds):
        members = ct[preds == c]
        if len(members) == 0:
            continue
        cluster_dom[c] = Counter(members).most_common(1)[0][0]
    per_type = []
    for bs in batch_specific_types:
        mask = ct == str(bs)
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        n_mis = 0
        for idx in np.where(mask)[0]:
            if cluster_dom.get(preds[idx], None) != str(bs):
                n_mis += 1
        per_type.append(n_mis / n_total)
    return float(np.mean(per_type)) if per_type else float("nan")


def compute_bsrs(
    embedding: "pd.DataFrame",
    batch_specific_types: Sequence[str],
    *,
    coord_cols: Sequence[str] = _COORDS,
    celltype_col: str = _CELLTYPE_COL,
) -> float:
    """Batch-Specific Retention Score (higher is better).

    The per-cell silhouette is computed over the full cell-type label space on
    the embedding coordinates, then averaged over the batch-specific cells.
    """
    ct = embedding[celltype_col].astype(str)
    codes = ct.astype("category").cat.codes.values
    if len(np.unique(codes)) < 2:
        return float("nan")
    X = embedding[list(coord_cols)].values
    try:
        sil = silhouette_samples(X, codes)
    except Exception:
        return float("nan")
    per_type = []
    for t in batch_specific_types:
        mask = ct.values == str(t)
        if mask.sum() == 0:
            continue
        per_type.append(float(sil[mask].mean()))
    return float(np.mean(per_type)) if per_type else float("nan")
