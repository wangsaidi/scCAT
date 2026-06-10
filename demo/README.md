# scCAT demo

A self-contained 30-second demonstration of scCAT on a small synthetic
dataset that mirrors the HDC (Villani et al. 2017) dendritic-cell benchmark
used in main Figure 3 of the paper.

**No external data download or network access is required.**

## Quick start

```bash
# From the repository root, after `pip install -e .`:
cd demo
python demo_quickstart.py
```

You should see output similar to:

```
============================================================
  Step 1 — Generate synthetic HDC-like dataset
============================================================
  Expression matrix shape: (480, 800)
  Number of batches:       2
  Number of cell types:    4
  ...
============================================================
  Step 7 — Evaluate
============================================================
  Per-cell-type cluster purity:
    CD141        0.962
    CD1C         0.987
    DoubleNeg    0.925
    pDC          0.950
  Mean purity:                       0.956
  kNN batch-mixing score (0–1):       0.484
```

After the script finishes, look in `output/`:

| File | Purpose |
|---|---|
| `demo_summary.txt` | **Read this first** — interpretation in plain English |
| `demo_umap_celltype.png` | UMAP coloured by cell type — should show 4 separated clusters |
| `demo_umap_batch.png` | UMAP coloured by batch — shared types mixed, batch-specific types isolated |
| `demo_metrics.csv` | Quantitative metrics |
| `demo_embedding.csv` | The 64-D scCAT latent embedding |
| `demo_metadata.csv` | Cell metadata for reference |

## What to verify

The two **most important** checks (see `INTERPRETATION.md` for the full
discussion):

1. **CD141 and CD1C cluster purity ≥ 0.85** — scCAT correctly preserved the
   two batch-specific cell types instead of forcing them into the shared
   clusters.
2. **The UMAP shows four separated clusters**, not 2 or 3.

If either of these is not satisfied on the demo, something is wrong with the
install — please open a GitHub issue.

## Files in this directory

| File | What it is |
|---|---|
| `demo_quickstart.py` | End-to-end runner (preprocess → triplet → train → UMAP → eval) |
| `generate_demo_data.py` | Stand-alone synthetic-data generator (importable on its own) |
| `INTERPRETATION.md` | Detailed walkthrough of what the outputs mean |
| `README.md` | This file |
