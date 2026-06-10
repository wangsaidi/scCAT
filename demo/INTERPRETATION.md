# Demo interpretation guide

This document explains what the outputs of `python demo_quickstart.py`
mean, what the **expected results** look like, and how to verify scCAT is
behaving as it does in the paper.

---

## 1. What does the demo dataset look like?

The demo dataset is a small synthetic mimic of the real **HDC**
(dendritic-cell) benchmark from Figure 3 of the paper. It contains:

| Property | Value |
|---|---|
| Number of cells | ~480 |
| Number of genes | 800 |
| Number of batches | 2 (B0, B1) |
| Number of cell types | 4 (pDC, DoubleNeg, CD141, CD1C) |
| **Batch-specific cell types** | **CD141 only in B0, CD1C only in B1** |
| Shared cell types | pDC, DoubleNeg (both present in both batches) |

The composition matrix should look like this when you run the demo:

```
cell_type  CD141  CD1C  DoubleNeg  pDC
batch
B0            80     0         80   80
B1             0    80         80   80
```

The zeros highlight the partial-sharing structure: **CD141 has no counterpart
in Batch 1, and CD1C has no counterpart in Batch 0**. This is the
"compositionally unbalanced" scenario that scCAT is specifically designed
to handle.

---

## 2. What does the script do?

1. **Generate** the synthetic dataset (described above).
2. **Preprocess** — HVG selection, normalisation, PCA on z-scaled HVG features.
3. **Construct triplets** — confidence-weighted (anchor, positive, negative)
   triplets from cross-batch MNN, same-batch KNN, HVG similarity and local
   density.
4. **Train** the MLP encoder for 100 epochs (~10–20 s on a CPU).
5. **Project** the latent embedding to 2-D with UMAP.
6. **Evaluate** with per-cell-type cluster purity and kNN-based batch mixing.
7. **Save** outputs to `output/`.

---

## 3. What outputs do I get?

After running the demo, `demo/output/` contains:

| File | What it is | How to use it |
|---|---|---|
| `demo_metadata.csv` | One row per cell: `cell_id`, `batch`, `cell_type` | Reference for the embedding |
| `demo_embedding.csv` | scCAT latent embedding, `cells × 64` | Feed into your own downstream analysis |
| `demo_umap.csv` | 2-D UMAP coords of the embedding | Re-plot in your tool of choice |
| `demo_umap_celltype.png` | UMAP coloured by cell type | **See expectation in §4** |
| `demo_umap_batch.png` | UMAP coloured by batch | **See expectation in §4** |
| `demo_metrics.csv` | Per-cell-type purity + kNN batch-mixing | Quick numerical check |
| `demo_summary.txt` | Human-readable summary | Read this first |

---

## 4. What should I see?

### 4.1 `demo_umap_celltype.png` — coloured by cell type

You should see **four well-separated clusters**, one per cell type. The two
shared cell types (pDC, DoubleNeg) form clusters with cells from both
batches; the two batch-specific cell types (CD141, CD1C) form their own
clusters. **All four clusters should be visually distinct from each other.**

If instead you see CD141 or CD1C being absorbed into pDC/DoubleNeg clusters,
that is the *overcorrection failure mode* — the very thing scCAT is designed
to prevent. If you observe this in the demo, something is wrong with the
install (please open an issue).

### 4.2 `demo_umap_batch.png` — coloured by batch

You should see:

- **The pDC and DoubleNeg clusters mix B0 and B1 colours uniformly**
  → scCAT removed the batch effect on the shared cell types.
- **The CD141 cluster is entirely one colour (B0) and the CD1C cluster is
  entirely another colour (B1)**
  → scCAT correctly recognised these cells have no cross-batch counterpart
    and did NOT force them to mix with anything.

This "mix-where-you-should, don't-mix-where-you-shouldn't" pattern is the
core promise of scCAT. The same pattern is shown in main Figure 3a + 3b of
the paper on the real HDC dataset.

### 4.3 `demo_metrics.csv` — quantitative check

| Metric | Expected value | What it means if low |
|---|---|---|
| `cluster_purity_pDC` | ≥ 0.85 | scCAT mis-clustered pDC cells |
| `cluster_purity_DoubleNeg` | ≥ 0.85 | scCAT mis-clustered DoubleNeg cells |
| `cluster_purity_CD141` | **≥ 0.85** (key) | scCAT overcorrected CD141 |
| `cluster_purity_CD1C` | **≥ 0.85** (key) | scCAT overcorrected CD1C |
| `cluster_purity_mean` | ≥ 0.85 | Overall structure broken |
| `knn_batch_mixing` | 0.30 – 0.60 | Either no integration (≤ 0.20) or overcorrection (≥ 0.75) |

The two purity values for **CD141** and **CD1C** are the most important —
they directly measure whether scCAT preserves the batch-specific cell types
instead of merging them. The expected purity for these is ≥ 0.85.

The kNN batch-mixing score will NOT approach 1.0, and that is correct: half
of the cells (CD141 in B0, CD1C in B1) have no counterpart in the other
batch and should not be mixed with anything.

---

## 5. Comparing to the paper

This demo replicates, in miniature, the behaviour reported on the real HDC
dataset in **main Figure 3** of the paper:

| Paper element | Demo equivalent |
|---|---|
| Figure 3a (scCAT UMAP, cell-type colouring, CD141/CD1C as distinct clusters) | `demo_umap_celltype.png` |
| Figure 3b (scCAT UMAP, batch colouring, pDC/DoubleNeg mixed) | `demo_umap_batch.png` |
| Figure 3d per-cell-type cluster purity | `demo_metrics.csv` purity columns |
| Methods §4.6 batch / bio metrics | `demo_metrics.csv` kNN-mixing |

The demo uses synthetic data so the absolute numbers will not exactly match
the paper, but the *pattern* should be identical: high purity on
batch-specific cell types and moderate (not maximal) batch mixing.

---

## 6. Want to try your own data?

Replace the call to `generate()` in `demo_quickstart.py` with code that
loads your own data:

```python
import numpy as np
import pandas as pd

# Your data: expression matrix + cell metadata
expr   = pd.read_csv("my_counts.csv", index_col=0)   # cells × genes
meta   = pd.read_csv("my_metadata.csv", index_col=0) # must have 'batch' column

X            = expr.values.astype(np.float32)
gene_names   = expr.columns.astype(str).tolist()
batch_labels = meta["batch"].astype(str).to_numpy()
cell_types   = meta["cell_type"].astype(str).to_numpy()  # optional, only used for evaluation
```

Then the rest of the script (steps 2–8) runs unchanged.

For input data that is **already normalised + HVG-selected** (e.g. h5ad files
from the Luecken benchmark), set `input_data_state="preprocessed_hvg"` in the
call to `prepare_inputs`. See `docs/api_reference.md` for details.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: No module named scCAT` | Package not installed | `pip install -e ..` from the demo dir, or run from the repo root after `pip install -e .` |
| Demo runs but `umap-learn` missing | UMAP not installed | `pip install umap-learn` (auto-installed by `pip install -e .`) |
| Demo says "no matplotlib" | Plotting extras not installed | `pip install matplotlib` or `pip install -e ".[plot]"` |
| All four cell types end up in one cluster | Something is wrong with the install | Open a GitHub issue with the full `demo_summary.txt` |
| CD141 / CD1C purity < 0.5 | Possible reproducibility issue | Verify Python and PyTorch versions match the tested ones in README §Installation |

---

## 8. Next steps

- Read `docs/algorithm.md` for the mathematical formulation.
- Read `docs/parameters.md` to learn what every hyperparameter does.
- See `reproduce_figures/` to reproduce the full paper figures on the real
  datasets.
