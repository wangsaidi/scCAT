# scCAT — Confidence-weighted Adaptive Triplet learning for single-cell batch integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C.svg)](https://pytorch.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**scCAT** is a single-cell RNA-seq batch-integration method that combines
**confidence-weighted triplet construction**, **adaptive margins** and
**batch-specific local-topology protection**. It is designed for the common
real-world situation where some cell populations are present in only a subset
of the batches — and where forcing every cell to be aligned across batches
would erase real biology.

---

## What problem does scCAT solve?

When you integrate scRNA-seq data from multiple batches, two opposing goals
must be balanced simultaneously:

1. **Integrate** — cells of the same biological identity but from different
   batches should overlap in the embedding.
2. **Preserve** — cells that are genuinely batch-specific (rare populations,
   condition-specific states, batch-only cell types) must **not** be forced to
   match cells from other batches.

Existing MNN- and anchor-based methods (Scanorama, fastMNN, Seurat) tend to
violate goal 2 — they create false cross-batch correspondences for cells that
do not actually have a counterpart, and merge batch-specific populations into
shared clusters (**overcorrection**). Deep latent methods (scVI, DESC,
DeepBID) suffer the same problem because they lack an explicit mechanism to
identify and protect cells without reliable counterparts.

**scCAT** addresses this with three coupled mechanisms:

- **Confidence-weighted triplets** — every candidate positive / negative pair
  is scored by combining cross-batch MNN ranking, HVG expression similarity
  and local density consistency; low-confidence pairs are filtered out before
  training.
- **Adaptive margins** — the triplet-loss margin is enlarged for low-density
  (rare) cells and for cell pairs spanning high-divergence batches, giving
  reliable matches stronger pull and unreliable matches weaker pull.
- **Batch-specific local-topology protection** — cells without a confident
  cross-batch counterpart are constrained only by their same-batch
  neighbourhood, never forced to align with other batches.

---

## Installation

### From source (recommended for now)

```bash
git clone https://github.com/wangsaidi/scCAT.git
cd scCAT
pip install -e .
```

### With optional plotting extras (needed to reproduce paper figures)

```bash
pip install -e ".[plot]"
```

### Pip-style install of locked dependencies only

```bash
pip install -r requirements.txt           # core
pip install -r requirements-plot.txt      # + plotting (optional)
```

### Tested environments

| Component | Tested version(s) |
|---|---|
| Python | 3.9 — 3.12 |
| PyTorch | 1.13 — 2.4 (CPU and CUDA) |
| Operating system | Linux, macOS, Windows |
| Hardware | CPU is sufficient up to ~30 k cells; GPU recommended for atlas scale |

---

## 60-second quickstart

```python
import numpy as np
from scCAT import (
    Config, set_seed, prepare_inputs, construct_triplets, train_full_batch,
)

# Your data: raw counts (n_cells × n_genes), batch labels, optional cell-type labels
X = np.load("counts.npy")            # shape (n_cells, n_genes)
gene_names  = open("genes.txt").read().splitlines()
batch_labels = np.load("batches.npy") # shape (n_cells,), e.g. array(['B0','B1',...])

config = Config()                     # default hyperparameters
set_seed(config.seed)

prepared = prepare_inputs(
    expression_matrix=X,
    feature_names=gene_names,
    batch_labels=batch_labels,
    input_data_state="raw",           # "raw" or "preprocessed_hvg"
    config=config,
)

triplets = construct_triplets(
    model_input=prepared["triplet_input"],
    hvg_matrix=prepared["hvg_matrix"],
    batch_labels=batch_labels,
    config=config,
)

model, embedding, history = train_full_batch(
    model_input=prepared["model_input"],
    triplet_bundle=triplets,
    config=config,
    device="cpu",                     # or "cuda"
)

# embedding is a numpy array of shape (n_cells, latent_dim).
# Feed it straight into UMAP / Leiden / your downstream analysis.
import umap
xy = umap.UMAP(n_components=2, random_state=0).fit_transform(embedding)
```

---

## Run the demo (no data download required)

```bash
cd demo
python demo_quickstart.py
```

This will:

1. Generate a small synthetic dataset (600 cells × 2 batches × 4 cell types,
   with **2 cell types batch-specific** — mirroring the structure of the
   real HDC dendritic-cell dataset used in the paper).
2. Run scCAT end-to-end (< 30 s on a CPU laptop).
3. Save a **before / after** UMAP comparison and a metric summary to
   `demo/output/`.
4. Print interpretation notes (what the metrics mean, what to look for in
   the UMAPs).

See **[`demo/INTERPRETATION.md`](demo/INTERPRETATION.md)** for a detailed
walkthrough of how to read the demo outputs.

---

## Reproduce the manuscript figures

All main figures (Fig 2 – Fig 7) and supplementary figures (S1 – S14) can be
reproduced from the scripts in [`reproduce_figures/`](reproduce_figures/):

```bash
pip install -e ".[plot]"          # install plotting extras
cd reproduce_figures

# After downloading the datasets (see reproduce_figures/README.md):
python fig2_simulation.py         # Figure 2 — controlled simulations
python fig3_hdc.py                # Figure 3 — HDC batch-specific preservation
python fig4_ablation.py           # Figure 4 — module ablation
python fig5_scmix_pbmc.py         # Figure 5 — Sc_mixology + PBMC
python fig6_lung_summary.py       # Figure 6 — Mouse Lung + cross-dataset summary
python fig7_crossomics.py         # Figure 7 — cross-omics (scRNA-seq + scATAC-seq)
python supp_figures.py            # All supplementary figures S1 – S14
```

Figure 1 is a schematic and is hand-drawn (Adobe Illustrator); it is not
generated by these scripts.

**Data access** — every dataset used in the paper, with primary citation,
GEO/figshare accession and direct URL, is listed in
[`reproduce_figures/README.md`](reproduce_figures/README.md) and in
Supplementary Table S17 of the manuscript.

---

## Repository layout

```
scCAT/
├── README.md                          # ← you are here
├── LICENSE                            # MIT
├── pyproject.toml                     # pip-installable package config
├── requirements.txt                   # core dependency list
├── requirements-plot.txt              # extras for reproducing figures
├── CITATION.cff                       # machine-readable citation metadata
│
├── scCAT/                             # the Python package
│   ├── __init__.py                    # exports the public API
│   ├── config.py                      # all hyperparameters (one dataclass)
│   ├── preprocess.py                  # HVG selection, normalisation, PCA
│   ├── triplets.py                    # confidence-weighted triplet construction
│   ├── losses.py                      # weighted triplet loss + BSP regulariser
│   ├── model.py                       # 2-layer MLP encoder
│   ├── trainer.py                     # full-batch training loop
│   └── utils.py                       # seeding, helpers
│
├── demo/
│   ├── README.md
│   ├── demo_quickstart.py             # runs scCAT on synthetic HDC-like data
│   ├── generate_demo_data.py          # synthetic data generator (no downloads)
│   └── INTERPRETATION.md              # how to read the demo output
│
├── reproduce_figures/
│   ├── README.md                      # data download + reproduction guide
│   ├── fig2_simulation.py             # Main Fig 2
│   ├── fig3_hdc.py                    # Main Fig 3
│   ├── fig4_ablation.py               # Main Fig 4
│   ├── fig5_scmix_pbmc.py             # Main Fig 5
│   ├── fig6_lung_summary.py           # Main Fig 6
│   └── supp_figures.py                # All supplementary figures
│
├── data/                              # Tier 1 reproducibility data (~6 MB)
│   ├── README.md                      # ← Three-tier data strategy explained
│   ├── zenodo_manifest.md             # Tier 2 deposit manifest
│   ├── datasets/{ds}/cell_metadata.csv  # per-cell batch + cell-type labels
│   ├── metric/{batch_remove,cluster}/{ds}.csv  # per-method metric CSVs
│   └── results/                       # ablation / sensitivity / runtime / scaling CSVs
│
└── docs/
    ├── algorithm.md                   # math + intuition
    ├── api_reference.md               # function-by-function API
    └── parameters.md                  # every hyperparameter explained
```

**Data layout** — the repo ships only **Tier 1 (~6 MB)**: cell metadata,
per-method numerical metrics, and the experimental result CSVs needed to
rebuild every supplementary table and verify every figure's numerical
claims. The heavier per-method embedding files (~5 GB) will be deposited
on Zenodo at acceptance; raw source data lives in the original GEO /
figshare archives. See [`data/README.md`](data/README.md) for the full
three-tier strategy and download links.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/algorithm.md`](docs/algorithm.md) | The math: confidence scoring, adaptive margin, BSP regulariser |
| [`docs/api_reference.md`](docs/api_reference.md) | Function-by-function reference for `prepare_inputs`, `construct_triplets`, `train_full_batch` |
| [`docs/parameters.md`](docs/parameters.md) | Every hyperparameter in `Config`, with default, range and qualitative effect |
| [`demo/INTERPRETATION.md`](demo/INTERPRETATION.md) | How to read the demo outputs and metric summary |

---

## When should I use scCAT?

scCAT is most useful when:

- Your batches **do not share all cell populations** — some cell types or
  cell states are present in only a subset of the batches.
- You suspect existing methods are **overcorrecting** rare or
  condition-specific populations (cells from a single batch end up in a
  cluster dominated by other batches).
- You care more about **preserving genuine biological structure** than
  about maximising a single batch-mixing metric.

scCAT is **not** specifically optimised for:

- Datasets where every cell type is present in every batch and balanced
  across batches — in that regime the simpler MNN/anchor methods are usually
  competitive and faster.
- Very small datasets (< 200 cells) where there are too few cells to reliably
  estimate confidence weights.

For a more detailed comparison, see Figure 6 and Supplementary Table S16 of
the manuscript.

---


## License

MIT — see [`LICENSE`](LICENSE).

## Contact

Issues and questions are best raised on the GitHub issue tracker. For
research collaborations or pre-acceptance enquiries, contact the
corresponding author of the manuscript.
