# scCAT API reference

The public API is the **five symbols** re-exported from the top-level
package:

```python
from scCAT import Config, set_seed, prepare_inputs, construct_triplets, train_full_batch
```

Everything else under the top-level package is implementation detail. The
optional `scCAT.metrics` submodule (documented at the end of this file) exposes
the paper's evaluation metrics — OCI, BSRS and Integration Balance — for
independent use.

---

## `Config` — `scCAT.config.Config`

```python
from scCAT import Config

config = Config()                # everything at defaults
config.max_epochs = 100          # override individual fields
```

`Config` is a frozen `@dataclass` containing every hyperparameter. The
complete list with defaults, allowed ranges and qualitative effect is in
[`parameters.md`](parameters.md). The most common knobs:

| Field | Default | What it does |
|---|---|---|
| `seed` | 42 | RNG seed for numpy / torch — controls reproducibility |
| `n_hvg` | 1000 | Number of highly-variable genes to keep |
| `n_pca` | 50 | PCA dimensionality for the encoder + triplet space |
| `latent_dim` | 64 | Output embedding dimensionality |
| `max_epochs` | 300 | Maximum training epochs (early stop usually fires earlier) |
| `min_c_pos` | 0.7 | Drop positive pairs with confidence below this |
| `mu_rare` | 0.3 | Strength of the batch-specific protection regulariser |

---

## `set_seed(seed: int) -> None`

```python
from scCAT import set_seed
set_seed(42)
```

Seeds `random`, `numpy.random`, `torch.manual_seed` and
`torch.cuda.manual_seed_all`. Call this once before `prepare_inputs` for
fully reproducible runs.

---

## `prepare_inputs(...)`

```python
from scCAT import prepare_inputs

prepared = prepare_inputs(
    expression_matrix=X,             # np.ndarray (n_cells, n_genes), float32
    feature_names=gene_names,        # list[str], length n_genes
    batch_labels=batch_labels,       # np.ndarray (n_cells,), dtype=object
    input_data_state="raw",          # "raw" or "preprocessed_hvg"
    config=config,                   # Config instance
)
```

**`input_data_state="raw"`** (default for unprocessed counts):

The pipeline is
`library-size normalise → log1p → HVG (top n_hvg) → ...`

- `model_input`        = `PCA(n_pca)` on z-scaled HVG (encoder input)
- `triplet_input`      = `PCA(n_pca)` on **batch-wise** z-scaled HVG
  (triplet space — batch-aware z-scaling reduces batch-driven distances
  during MNN search)
- `hvg_matrix`         = the raw HVG matrix (for HVG-similarity computation)

**`input_data_state="preprocessed_hvg"`**: if you have already done
`normalise → log1p → HVG` upstream (e.g. from a Luecken benchmark h5ad),
this skips the normalisation step and only does the z-scaling + PCA.

**Returns** a dict with keys `model_input`, `triplet_input`, `hvg_matrix`.

---

## `construct_triplets(...)`

```python
from scCAT import construct_triplets

triplets = construct_triplets(
    model_input=prepared["triplet_input"],   # PCA(triplet space)
    hvg_matrix=prepared["hvg_matrix"],       # for HVG similarity
    batch_labels=batch_labels,
    config=config,
)
```

Builds the confidence-weighted triplet set:

1. Builds the same-batch KNN graph
2. Builds the cross-batch MNN graph
3. Computes per-cell local density `ρ_i`
4. Computes positive and negative confidences for every candidate pair
5. Filters by `(min_c_pos, min_c_neg)` thresholds
6. Marks rare cells (`rare_flag = 1` for cells with no confident cross-batch
   positive) — these get extra protection during training

**Returns** a `TripletBundle` dataclass with fields:

| Field | Type | Meaning |
|---|---|---|
| `triplets` | `np.ndarray (n_triplets, 5)` | columns: `[anchor, positive, negative, c⁺, c⁻]` |
| `omega` | `np.ndarray (n_cells,)` | per-cell triplet weight |
| `rho_bar` | `np.ndarray (n_cells,)` | per-cell mean local density (used for adaptive margin) |
| `same_batch_knn_edges` | `np.ndarray (n_edges, 2)` | edges used by the BSP regulariser |
| `rare_flags` | `np.ndarray (n_cells,)` | 1 if cell has no confident cross-batch positive |
| `batch_codes_per_cell` | `np.ndarray (n_cells,)` | integer batch IDs |
| `batch_names` | `list[str]` | original batch names in encoding order |
| `summary` | `dict` | filter / count statistics for logging |

Most users will not touch this object directly — pass it straight to
`train_full_batch`.

---

## `train_full_batch(...)`

```python
from scCAT import train_full_batch

model, embedding, history = train_full_batch(
    model_input=prepared["model_input"],   # PCA encoder input
    triplet_bundle=triplets,               # from construct_triplets
    config=config,
    device="cpu",                          # or "cuda"
)
```

Trains the MLP encoder for up to `config.max_epochs` epochs with full-batch
optimisation over all triplets. Early stops when the total loss is stable.

**Returns**:

| Name | Type | Meaning |
|---|---|---|
| `model` | `MLPEncoder` | the trained encoder (you can re-apply it to new cells) |
| `embedding` | `np.ndarray (n_cells, latent_dim)` | the final embedding; **this is the main output** |
| `history` | `list[dict[str, float]]` | per-epoch loss components, useful for diagnostic plots |

---

## End-to-end skeleton

```python
import numpy as np
from scCAT import (
    Config, set_seed,
    prepare_inputs, construct_triplets, train_full_batch,
)

# 1. Your data
X            = ...                       # (n_cells, n_genes), float32
gene_names   = [...]                     # list[str]
batch_labels = ...                       # (n_cells,), e.g. np.array(["B0","B1",...])

# 2. Configure + seed
config = Config()
set_seed(config.seed)

# 3. Preprocess
prepared = prepare_inputs(
    expression_matrix=X,
    feature_names=gene_names,
    batch_labels=batch_labels,
    input_data_state="raw",
    config=config,
)

# 4. Triplets
triplets = construct_triplets(
    model_input=prepared["triplet_input"],
    hvg_matrix=prepared["hvg_matrix"],
    batch_labels=batch_labels,
    config=config,
)

# 5. Train
model, embedding, history = train_full_batch(
    model_input=prepared["model_input"],
    triplet_bundle=triplets,
    config=config,
    device="cpu",
)

# 6. Use the embedding however you like
import umap
xy = umap.UMAP(n_components=2, random_state=0).fit_transform(embedding)
```

---

## Frequently asked questions

**Q: Can I apply a trained encoder to a new (held-out) batch?**
Yes. The returned `model` is a standard `torch.nn.Module` and accepts
arbitrary `(n_new_cells, n_pca)` input. **Caveat**: scCAT is a transductive
method by design (it uses the cross-batch MNN graph computed on the full
training set). For a true inductive use-case you should re-train including
the new batch.

**Q: How large can `n_cells` go on CPU?**
Up to ~30 000 cells comfortably (< 10 min on a modern Intel laptop, < 1 GB
RSS). Beyond that, GPU is strongly recommended.

**Q: Does scCAT need the cell-type labels?**
No. scCAT is **fully unsupervised** — it never sees the cell-type labels
during training. They are only used downstream for evaluation.

**Q: What if I have only one batch?**
scCAT requires at least two batches for the cross-batch MNN construction.
If you have only one batch you do not need batch integration — a standard
PCA / UMAP / scVI pipeline is sufficient.

---

## Metrics submodule — `scCAT.metrics`

The metrics used to evaluate batch integration in the paper are packaged in the
optional `scCAT.metrics` submodule so they can be called **independently** of the
figure-generation code. It depends on **scikit-learn** and is intentionally *not*
imported by `import scCAT`, keeping the core training API dependency-light.

```python
from scCAT.metrics import (
    compute_oci, compute_bsrs, integration_balance,   # novel metrics (this work)
    ari, nmi, asw_celltype, asw_batch_mixing,          # standard, re-exposed
    get_value, resolve_column,                         # metric-table helpers
)
```

### Novel metrics proposed in this work

| Function | Direction | Meaning |
|---|---|---|
| `compute_oci(embedding, batch_specific_types, n_clusters)` | lower is better | **Overcorrection Index** — fraction of batch-specific cells absorbed into a cluster whose majority label is a *different* cell type (k-means, `n_init=10`, `random_state=0`, on the 2-D embedding). |
| `compute_bsrs(embedding, batch_specific_types)` | higher is better | **Batch-Specific Retention Score** — mean silhouette of the batch-specific cells against the full cell-type label space. |
| `integration_balance(batch_table, cluster_table, method)` | higher is better | **Integration Balance** — the geometric mean of a batch-removal score and a biology-conservation score. |

`compute_oci` / `compute_bsrs` take a `DataFrame` with the 2-D coordinate
columns (`UMAP1`, `UMAP2` by default) and a `cell_type` column.
`integration_balance` reads the shipped per-method metric tables
(`data/metric/batch_remove/<dataset>.csv` and `cluster/<dataset>.csv`) and
returns a `BalanceScore(s_batch, s_bio, balance)` named tuple, where

```
s_batch = mean(iLISI, 1 - kBET, ASW_batch_mixing)
s_bio   = mean(ARI, NMI, ASW_celltype, cLISI_purity)
balance = sqrt(s_batch * s_bio)
```

These three are verbatim re-implementations of the formulas in
`generators/plot_improved.py`; they reproduce the manuscript's Fig. 6 /
Supplementary Table S16 numbers exactly.

### Standard metrics (re-exposed)

`ari` / `nmi` are exact scikit-learn calls; `asw_celltype` / `asw_batch_mixing`
follow the standard `scib` silhouette definitions (and the same `[0, 1]`
rescaling). The neighbourhood-graph metrics (kBET, iLISI, cLISI) are not
re-implemented here — they follow their `scib` definitions and are consumed by
`integration_balance` through the shipped metric tables. Use `get_value` /
`resolve_column` to read any metric value out of those tables (robust to
`method`/`Method` and `scCAT`/`BTCA` naming variations).
