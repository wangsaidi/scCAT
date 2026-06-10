"""demo_quickstart.py — Run scCAT end-to-end on a synthetic HDC-like dataset
and produce:

    output/
        demo_metadata.csv       (batch labels + cell types)
        demo_embedding.csv      (scCAT 64-D latent embedding)
        demo_umap.csv           (2-D UMAP of the embedding)
        demo_umap_celltype.png  (UMAP coloured by cell type)
        demo_umap_batch.png     (UMAP coloured by batch)
        demo_metrics.csv        (kNN-mixing + cell-type purity)
        demo_summary.txt        (human-readable interpretation)

Total runtime on a CPU laptop: < 30 s.

The synthetic data mirror the structure of the real HDC benchmark used in
Figure 3 of the manuscript: 2 batches, 4 cell types, with 2 of the 4 cell
types being batch-specific (CD141 only in Batch 0, CD1C only in Batch 1).
This is the "partial sharing" scenario that scCAT was specifically
designed to handle.

After running this script, read demo/INTERPRETATION.md to understand what
the outputs mean.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Make the local scCAT package importable when this is run from a checkout
# without a prior `pip install -e .`
sys.path.insert(0, str(HERE.parent))

from generate_demo_data import generate                                # noqa: E402

# scCAT public API
from scCAT import (                                                    # noqa: E402
    Config, set_seed,
    prepare_inputs, construct_triplets, train_full_batch,
)


OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate / load data
# ─────────────────────────────────────────────────────────────────────────────

def _section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


_section("Step 1 — Generate synthetic HDC-like dataset")
X, gene_names, batch_labels, cell_types = generate()
print(f"  Expression matrix shape: {X.shape}")
print(f"  Number of batches:       {len(np.unique(batch_labels))}")
print(f"  Number of cell types:    {len(np.unique(cell_types))}")
print(f"  Per-(batch, cell_type) composition:")
print(pd.crosstab(pd.Series(batch_labels, name='batch'),
                   pd.Series(cell_types, name='cell_type')).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Configure scCAT and run end-to-end
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 2 — Configure scCAT")
config = Config()
# For the demo we lower max_epochs from 300 to 100 to keep total runtime <30s.
# In real usage the defaults are recommended.
config.max_epochs = 100
print(f"  seed          = {config.seed}")
print(f"  max_epochs    = {config.max_epochs}")
print(f"  n_hvg         = {config.n_hvg}  (≤ n_genes, capped automatically)")
print(f"  n_pca         = {config.n_pca}")
print(f"  latent_dim    = {config.latent_dim}")
print(f"  min_c_pos     = {config.min_c_pos}  (positive-confidence threshold)")
print(f"  mu_rare       = {config.mu_rare}    (batch-specific protection weight)")
set_seed(config.seed)


_section("Step 3 — Preprocess (HVG selection + PCA on z-scaled HVGs)")
t0 = time.perf_counter()
prepared = prepare_inputs(
    expression_matrix=X,
    feature_names=gene_names,
    batch_labels=batch_labels,
    input_data_state="raw",          # the demo data are raw counts
    config=config,
)
print(f"  Encoder input shape (PCA on HVG):       "
      f"{prepared['model_input'].shape}")
print(f"  Triplet space shape (z-scaled + PCA):   "
      f"{prepared['triplet_input'].shape}")
print(f"  HVG matrix shape:                       "
      f"{prepared['hvg_matrix'].shape}")
print(f"  Elapsed: {time.perf_counter() - t0:.1f}s")


_section("Step 4 — Construct confidence-weighted triplets")
t0 = time.perf_counter()
triplets = construct_triplets(
    model_input=prepared["triplet_input"],
    hvg_matrix=prepared["hvg_matrix"],
    batch_labels=batch_labels,
    config=config,
)
summary = triplets.summary
print(f"  Cross-batch positive pairs:  {summary.get('cross_batch_positive_pairs', 0):,}")
print(f"  Same-batch positive pairs:   {summary.get('same_batch_positive_pairs', 0):,}")
print(f"  Total triplets after filter: {len(triplets.triplets):,}")
print(f"  Rare-cell count (BSP):       {int(triplets.rare_flags.sum()):,}")
print(f"  Elapsed: {time.perf_counter() - t0:.1f}s")


_section("Step 5 — Train the encoder (full-batch SGD)")
t0 = time.perf_counter()
model, embedding, history = train_full_batch(
    model_input=prepared["model_input"],
    triplet_bundle=triplets,
    config=config,
    device="cpu",
)
print(f"  Final embedding shape: {embedding.shape}")
print(f"  Elapsed: {time.perf_counter() - t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Project to 2-D with UMAP and compute simple evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 6 — Project to 2-D with UMAP")
try:
    import umap
    xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.2,
                    random_state=0).fit_transform(embedding)
except ImportError:
    print("  [warn] umap-learn not installed — falling back to PCA(2)")
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(embedding)


_section("Step 7 — Evaluate")
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# (i) Cell-type cluster purity (k-means with k = #true types)
n_types = len(np.unique(cell_types))
km = KMeans(n_clusters=n_types, n_init=10, random_state=0).fit(embedding)
purity_per_type = {}
for t in np.unique(cell_types):
    mask = cell_types == t
    counts = pd.Series(km.labels_[mask]).value_counts()
    purity_per_type[t] = float(counts.iloc[0] / mask.sum())

# (ii) kNN-based batch-mixing score (lower = batches isolated, higher = mixed)
knn = NearestNeighbors(n_neighbors=21).fit(embedding)
_, ind = knn.kneighbors(embedding)
ind = ind[:, 1:]
mixing_score = float((batch_labels[ind] != batch_labels[:, None]).mean())

print(f"  Per-cell-type cluster purity:")
for t in sorted(purity_per_type):
    print(f"    {t:12s} {purity_per_type[t]:.3f}")
print(f"  Mean purity (across cell types):  "
      f"{np.mean(list(purity_per_type.values())):.3f}")
print(f"  kNN batch-mixing score (0–1):     {mixing_score:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save outputs
# ─────────────────────────────────────────────────────────────────────────────

_section("Step 8 — Save outputs")

# Tabular outputs
meta_df = pd.DataFrame({
    "cell_id":   [f"cell_{i:04d}" for i in range(len(X))],
    "batch":     batch_labels,
    "cell_type": cell_types,
}).set_index("cell_id")
meta_df.to_csv(OUT_DIR / "demo_metadata.csv")

emb_df = pd.DataFrame(
    embedding, index=meta_df.index,
    columns=[f"latent_{i + 1}" for i in range(embedding.shape[1])],
)
emb_df.to_csv(OUT_DIR / "demo_embedding.csv")

umap_df = pd.DataFrame(
    xy, index=meta_df.index, columns=["UMAP1", "UMAP2"],
)
umap_df["batch"]     = batch_labels
umap_df["cell_type"] = cell_types
umap_df.to_csv(OUT_DIR / "demo_umap.csv")

metrics_rows = [
    {"metric": "cluster_purity_pDC",       "value": round(purity_per_type.get("pDC", np.nan), 3)},
    {"metric": "cluster_purity_DoubleNeg", "value": round(purity_per_type.get("DoubleNeg", np.nan), 3)},
    {"metric": "cluster_purity_CD141",     "value": round(purity_per_type.get("CD141", np.nan), 3)},
    {"metric": "cluster_purity_CD1C",      "value": round(purity_per_type.get("CD1C", np.nan), 3)},
    {"metric": "cluster_purity_mean",      "value": round(float(np.mean(list(purity_per_type.values()))), 3)},
    {"metric": "knn_batch_mixing",         "value": round(mixing_score, 3)},
]
pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "demo_metrics.csv", index=False)

# Two UMAP figures
try:
    import matplotlib.pyplot as plt
    palette_ct = {"pDC": "#4C72B0", "DoubleNeg": "#DD8452",
                  "CD141": "#55A868", "CD1C": "#C44E52"}
    palette_b  = {"B0": "#2D9E2D", "B1": "#9B59B6"}

    # (a) by cell type
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=140)
    for ct in sorted(np.unique(cell_types)):
        mask = cell_types == ct
        ax.scatter(xy[mask, 0], xy[mask, 1],
                    s=12, alpha=0.85, c=palette_ct.get(ct, "#888"),
                    edgecolors="white", linewidths=0.3, label=ct)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("scCAT — coloured by cell type", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="best")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demo_umap_celltype.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (b) by batch
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=140)
    for b in sorted(np.unique(batch_labels)):
        mask = batch_labels == b
        ax.scatter(xy[mask, 0], xy[mask, 1],
                    s=12, alpha=0.85, c=palette_b.get(b, "#888"),
                    edgecolors="white", linewidths=0.3, label=b)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title("scCAT — coloured by batch", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="best")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "demo_umap_batch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    plotted = True
except ImportError:
    print("  [warn] matplotlib not installed — skipping PNG figures")
    plotted = False

# Human-readable summary
with open(OUT_DIR / "demo_summary.txt", "w", encoding="utf-8") as f:
    f.write("scCAT demo — synthetic HDC-like dataset\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Cells: {X.shape[0]} | Genes: {X.shape[1]} | "
            f"Batches: {len(np.unique(batch_labels))} | "
            f"Cell types: {len(np.unique(cell_types))}\n\n")
    f.write("Per-(batch, cell type) composition:\n")
    f.write(pd.crosstab(pd.Series(batch_labels, name='batch'),
                         pd.Series(cell_types, name='cell_type')).to_string())
    f.write("\n\nNote: CD141 is present only in Batch 0 and CD1C only in "
            "Batch 1 — these are the 'batch-specific' cell types.\n\n")
    f.write("=" * 60 + "\n")
    f.write("RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Per-cell-type k-means cluster purity (1.0 = perfectly preserved):\n")
    for t in sorted(purity_per_type):
        marker = "  ← batch-specific" if t in ("CD141", "CD1C") else ""
        f.write(f"  {t:12s} {purity_per_type[t]:.3f}{marker}\n")
    f.write(f"\n  Mean: {np.mean(list(purity_per_type.values())):.3f}\n\n")
    f.write(f"kNN batch-mixing score (0 = isolated, 1 = perfectly mixed):\n")
    f.write(f"  {mixing_score:.3f}\n\n")
    f.write("=" * 60 + "\n")
    f.write("HOW TO READ THESE RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("If scCAT is working correctly you should see:\n\n")
    f.write("  1. Per-cell-type purity ≥ 0.85 for ALL four cell types, "
            "INCLUDING the two batch-specific ones (CD141, CD1C). "
            "This means scCAT preserves the batch-specific cells as "
            "distinct clusters rather than absorbing them into the shared "
            "clusters.\n\n")
    f.write("  2. kNN batch-mixing in the range 0.3 - 0.6. The score will "
            "not approach 1.0 because two of the four cell types are "
            "batch-specific — those cells are correctly NOT mixed across "
            "batches (there is nothing in the other batch to mix with).\n\n")
    f.write("  3. In demo_umap_celltype.png, you should see four well-"
            "separated clusters (pDC, DoubleNeg, CD141, CD1C).\n\n")
    f.write("  4. In demo_umap_batch.png, the shared cell types (pDC and "
            "DoubleNeg) should show mixed colours (B0 and B1 interleaved), "
            "while the batch-specific clusters (CD141, CD1C) will each be "
            "a single colour — exactly as expected.\n\n")
    f.write("See demo/INTERPRETATION.md for the full walkthrough.\n")

_section("Done")
print(f"  Outputs written to: {OUT_DIR}")
print(f"  Open:")
print(f"    - demo_umap_celltype.png   (cells by cell type)")
print(f"    - demo_umap_batch.png      (cells by batch)")
print(f"    - demo_summary.txt         (interpretation)")
print(f"  See demo/INTERPRETATION.md for what to look for.")
