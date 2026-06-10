"""generate_demo_data.py — Build a small synthetic dataset that mirrors the
structure of the HDC (Villani et al. 2017) dendritic-cell benchmark used in
Figure 3 of the paper:

    - 2 batches
    - 4 cell types: pDC, DoubleNeg, CD141, CD1C
    - **2 of the 4 cell types are batch-specific** (CD141 only in Batch 0,
      CD1C only in Batch 1) — this is the "partial sharing" scenario that
      scCAT was specifically designed to handle.

The data are generated with a simple negative-binomial gene-expression model
plus a multiplicative batch effect on a random subset of marker-like genes.
No external network access or downloads are required.

Usage::

    from generate_demo_data import generate
    X, gene_names, batch_labels, cell_types = generate()
    # X         : float32 array, shape (n_cells, n_genes)
    # gene_names: list[str]
    # batch_labels: object array of "B0" / "B1"
    # cell_types: object array of "pDC" / "DoubleNeg" / "CD141" / "CD1C"
"""

from __future__ import annotations
import numpy as np


def generate(
    cells_per_shared_type_per_batch: int = 80,  # pDC + DoubleNeg
    cells_per_specific_type: int = 80,         # CD141 only in B0, CD1C only in B1
    n_genes: int = 800,
    n_marker_genes_per_type: int = 40,
    seed: int = 0,
):
    """Return (X, gene_names, batch_labels, cell_types).

    Total cells under defaults:
        Batch 0: 80 pDC + 80 DoubleNeg + 80 CD141                = 240
        Batch 1: 80 pDC + 80 DoubleNeg + 80 CD1C                 = 240
        Total                                                    = 480
    """
    rng = np.random.default_rng(seed)

    # ── 1. Composition specification (matches the HDC partial-sharing pattern) ──
    composition = [
        # (batch, cell_type, n_cells)
        ("B0", "pDC",       cells_per_shared_type_per_batch),
        ("B0", "DoubleNeg", cells_per_shared_type_per_batch),
        ("B0", "CD141",     cells_per_specific_type),     # batch-specific to B0
        ("B1", "pDC",       cells_per_shared_type_per_batch),
        ("B1", "DoubleNeg", cells_per_shared_type_per_batch),
        ("B1", "CD1C",      cells_per_specific_type),     # batch-specific to B1
    ]

    # ── 2. Assign per-cell-type marker-gene blocks ──
    cell_types_unique = ["pDC", "DoubleNeg", "CD141", "CD1C"]
    # Disjoint marker blocks
    marker_blocks = {
        t: rng.choice(n_genes,
                      size=n_marker_genes_per_type,
                      replace=False)
        for t in cell_types_unique
    }
    # Ensure no overlap — re-draw any overlapping markers
    used = set()
    for t in cell_types_unique:
        block = []
        for g in marker_blocks[t]:
            if g not in used:
                block.append(g)
                used.add(g)
        # Top up if collisions
        while len(block) < n_marker_genes_per_type:
            cand = int(rng.integers(0, n_genes))
            if cand not in used:
                block.append(cand)
                used.add(cand)
        marker_blocks[t] = np.array(block)

    # ── 3. Sample expression matrix cell by cell ──
    total_cells = sum(n for _, _, n in composition)
    X = np.zeros((total_cells, n_genes), dtype=np.float32)
    batch_labels = np.empty(total_cells, dtype=object)
    cell_types   = np.empty(total_cells, dtype=object)

    # Baseline expression for every gene in every cell (Poisson noise)
    base_lambda = rng.uniform(0.5, 2.0, size=n_genes)
    X[:] = rng.poisson(base_lambda[None, :], size=(total_cells, n_genes)).astype(np.float32)

    # Marker boost per cell type
    pos = 0
    for batch, ct, n_cells in composition:
        # Marker genes for this cell type get a strong boost (lambda 8-15)
        boost_lambda = rng.uniform(8.0, 15.0, size=n_marker_genes_per_type)
        X[pos:pos + n_cells, marker_blocks[ct]] += rng.poisson(
            boost_lambda[None, :], size=(n_cells, n_marker_genes_per_type)
        ).astype(np.float32)
        batch_labels[pos:pos + n_cells] = batch
        cell_types[pos:pos + n_cells]   = ct
        pos += n_cells

    # ── 4. Inject a multiplicative batch effect on ~30% of genes ──
    n_batch_genes = int(0.30 * n_genes)
    batch_genes = rng.choice(n_genes, size=n_batch_genes, replace=False)
    # Batch 0 multiplier ~1.0; Batch 1 multiplier ~1.6 on selected genes
    b1_mask = batch_labels == "B1"
    X[np.ix_(b1_mask, batch_genes)] *= 1.6
    # Add a small additive batch-specific noise on these genes too
    X[b1_mask[:, None] & np.isin(np.arange(n_genes), batch_genes)[None, :]] += \
        rng.normal(0, 0.5, size=int(b1_mask.sum() * n_batch_genes))

    # Clip and round
    X = np.clip(X, 0, None).astype(np.float32)

    gene_names = [f"Gene{i + 1:04d}" for i in range(n_genes)]

    return X, gene_names, batch_labels, cell_types


if __name__ == "__main__":
    X, genes, b, c = generate()
    print(f"X shape: {X.shape}")
    print(f"Batches:    {np.unique(b, return_counts=True)}")
    print(f"Cell types: {np.unique(c, return_counts=True)}")
    print(f"Per-(batch, cell_type) counts:")
    import pandas as pd
    print(pd.crosstab(pd.Series(b, name='batch'),
                       pd.Series(c, name='cell_type')))
