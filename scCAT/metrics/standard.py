"""Standard integration metrics, re-exposed for independent use.

The manuscript's benchmark numbers were produced with the community-standard
``scib`` pipeline (Luecken et al., *Nat. Methods* 2022).  These thin wrappers
re-expose the subset of standard metrics that have an unambiguous,
dependency-light definition, so each component of the Integration Balance can be
recomputed independently from labels + an embedding.

``ari`` / ``nmi`` are exact scikit-learn calls.  ``asw_celltype`` /
``asw_batch_mixing`` follow the standard silhouette-based scib definitions (and
the same [0, 1] rescaling), so they are *definitionally* equivalent but are not
guaranteed bit-identical to the upstream pipeline's stored numbers.  The
neighbourhood-graph metrics (kBET, iLISI, cLISI) are intentionally *not*
re-implemented here - they follow their scib definitions and are consumed by
``integration_balance`` through the shipped metric tables.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)


def _codes(labels: Sequence) -> np.ndarray:
    return pd.Series([str(x) for x in labels]).astype("category").cat.codes.values


def ari(labels_true: Sequence, labels_pred: Sequence) -> float:
    """Adjusted Rand Index between ground-truth and predicted cluster labels."""
    return float(adjusted_rand_score(labels_true, labels_pred))


def nmi(labels_true: Sequence, labels_pred: Sequence) -> float:
    """Normalized Mutual Information (arithmetic averaging - the scib default)."""
    return float(
        normalized_mutual_info_score(
            labels_true, labels_pred, average_method="arithmetic"
        )
    )


def asw_celltype(embedding, celltype_labels) -> float:
    """Cell-type ASW rescaled to [0, 1] (1 = cell types well separated).

    Standard scib normalization: ``(mean_silhouette + 1) / 2`` over the
    cell-type labels.
    """
    X = np.asarray(embedding)
    codes = _codes(celltype_labels)
    if len(np.unique(codes)) < 2:
        return float("nan")
    sil = silhouette_samples(X, codes)
    return float((np.mean(sil) + 1.0) / 2.0)


def asw_batch_mixing(embedding, batch_labels, celltype_labels) -> float:
    """Batch-mixing ASW rescaled so 1 = batches well mixed within a cell type.

    Standard scib definition: within each cell type, take ``1 - |silhouette|``
    of the batch labels, average over its cells, then average over cell types.
    """
    X = np.asarray(embedding)
    batch = _codes(batch_labels)
    ct = np.asarray([str(c) for c in celltype_labels])
    scores = []
    for t in np.unique(ct):
        mask = ct == t
        if mask.sum() < 2:
            continue
        b = batch[mask]
        if len(np.unique(b)) < 2:
            continue
        sil = silhouette_samples(X[mask], b)
        scores.append(float(np.mean(1.0 - np.abs(sil))))
    return float(np.mean(scores)) if scores else float("nan")
