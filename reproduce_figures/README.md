# Reproducing the manuscript figures

This folder contains thin wrapper scripts that re-generate every main figure
(Fig 2 – Fig 7) and every supplementary figure (S1 – S14) from the
manuscript using the embeddings and metric tables shipped with the paper.

**Figure 1** is a schematic and is hand-drawn (Adobe Illustrator); it is
not reproduced by these scripts.

---

## 1. Prerequisites

Install plotting extras on top of the core install:

```bash
pip install -e ".[plot]"
# or
pip install -r requirements-plot.txt
```

This adds `matplotlib`, `seaborn`, `adjustText`, `openpyxl`, `python-docx`
and `psutil`.

---

## 2. Data download

The manuscript uses 11 datasets in total. Data are organised in **three
tiers** (full details in [`../data/README.md`](../data/README.md)):

- **Tier 1** (already in this repo, `../data/`, ~6 MB) — cell metadata,
  per-method metric CSVs and the experimental result CSVs needed by these
  reproduction scripts and by `build_supp_tables.py`.
- **Tier 2** (Zenodo, ~5 GB, DOI assigned at acceptance) — per-method
  embeddings + full benchmark results. Extract into `../data/embedding/`
  and `../data/benchmark_result/` after download.
- **Tier 3** (original GEO / figshare) — raw counts; only needed if you
  want to re-run methods from scratch.

For the reviewer / pre-print period, raw datasets can be downloaded
directly from their original sources:

| Dataset | Original source | URL |
|---|---|---|
| Simulated 1–4 | Generated with Splatter (seed = 42); see `00_regenerate_simulations.py` | n/a |
| Sc_mixology | Tian et al. 2019 *Nat Methods* | [GSE118767](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118767) |
| HDC | Villani et al. 2017 *Science* | [GSE94820](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94820) |
| PBMC | Kang et al. 2018 *Nat Biotechnol* | [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583) |
| Mouse Lung | Luecken et al. 2022 *Nat Methods* benchmark | [figshare 12420968](https://figshare.com/articles/dataset/12420968) |
| Human Pancreas | Luecken et al. 2022 benchmark | [figshare 12420968](https://figshare.com/articles/dataset/12420968) |
| Human Immune | Luecken et al. 2022 benchmark | [figshare 12420968](https://figshare.com/articles/dataset/12420968) |
| Gut | Haber et al. 2017 *Nature* | [GSE92332](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92332) |

A complete machine-readable list is in Supplementary Table S17 of the
manuscript.

Once downloaded, organise the files under:

```
data/
├── embedding/                # method × dataset → 2-D UMAP csv or h5ad
│   ├── HDC/
│   │   ├── BTCA.csv          # scCAT
│   │   ├── Scanorama.csv
│   │   ├── fastMNN.csv
│   │   ├── INSCT_Unsupervised.h5ad
│   │   ├── DESC.csv
│   │   ├── scBCN.csv
│   │   ├── DeepBID.csv
│   │   └── SPDR.h5ad
│   ├── PBMC/
│   ├── Lung/
│   └── ...                   # one folder per dataset
├── datasets/                 # raw or processed source data
│   ├── HDC/cell_metadata.csv
│   ├── PBMC/cell_metadata.csv
│   └── ...
└── benchmark_result/         # per-dataset metric CSVs
    └── metric/
        ├── batch_remove/HDC.csv
        ├── batch_remove/PBMC.csv
        ├── cluster/HDC.csv
        └── ...
```

The scripts here assume this layout; if your data live elsewhere, edit the
`DATA_ROOT` constant near the top of each `figN_*.py`.

---

## 3. Reproducing each figure

```bash
python fig2_simulation.py       # Main Fig 2 — controlled simulations (Sim 3, Sim 4)
python fig3_hdc.py              # Main Fig 3 — HDC batch-specific preservation
python fig4_ablation.py         # Main Fig 4 — module ablation (Sim 4 + HDC)
python fig5_scmix_pbmc.py       # Main Fig 5 — Sc_mixology + PBMC
python fig6_lung_summary.py     # Main Fig 6 — Mouse Lung + cross-dataset summary
python fig7_crossomics.py       # Main Fig 7 — cross-omics (scRNA-seq + scATAC-seq)
python supp_figures.py          # All supplementary figures S1 – S14
```

Each script saves four file formats per figure (PDF, SVG, PNG @ 300 dpi,
TIFF/LZW) to the `output/` directory next to the script. The PDFs and SVGs
are vector-perfect for final journal upload; the PNGs are convenient for
preview; the TIFFs are for journals that explicitly require TIFF/LZW.

---

## 4. Reproducing every panel from raw data (full pipeline)

If you want to re-run every method on the raw datasets (rather than relying
on the pre-computed embeddings), follow these steps:

1. Install the seven baseline methods according to their own installation
   instructions:
   - Scanorama: `pip install scanorama`
   - INSCT: `pip install insct` (or from GitHub)
   - DESC: `pip install desc`
   - SPDR, DeepBID, scBCN: see the corresponding paper supplements for
     installation; these are not currently distributed via PyPI.
   - fastMNN: install in R via `BiocManager::install("batchelor")` and call
     from Python via `rpy2`.

2. Run the method runners from the upstream source repository's `experiments/`
   pipeline (the full method-running pipeline is **not bundled** in this
   figures-and-tables release). These write outputs into
   `embedding/{dataset}/{method}.csv|h5ad`.

3. Run the metric pipeline (also part of that upstream `experiments/` pipeline)
   to regenerate `metric/batch_remove/{dataset}.csv` and
   `metric/cluster/{dataset}.csv`.

4. Re-run the figure scripts above.

Note that wall-clock cross-method comparison is **out of scope** for this
study because the seven baselines are written in heterogeneous languages
(R / Python / TensorFlow / PyTorch); see Methods §4.7 of the manuscript.

---

## 5. Supplementary tables

The 19 supplementary tables (`Supplementary_Tables_S1-S19.xlsx`) are regenerated
by the bundled table code in [`../reproduce_tables/`](../reproduce_tables/):

```bash
cd ../reproduce_tables
python supp_tables.py              # builds S1–S17, then appends S18 and S19
```

This chains `build_supp_tables.py` → `make_supp_table_S18.py` →
`make_supp_table_S19.py` (all under `../generators/experiments/`) and writes the
workbook to `../reproduce_tables/output/`. See
[`../reproduce_tables/README.md`](../reproduce_tables/README.md) for the
external-data dependency and the S16 caveat (the regenerated S16 is the legacy
average-rank view; the shipped workbook's `S16_IB_results` is authoritative).

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: plot_improved` | `_common.py` resolves `plot_improved.py` from the bundled `../generators/` tree first (falling back to an external `resource/` checkout) | Ensure `../generators/plot_improved.py` is intact, or set `RESOURCE_ROOT` at the top of the script to the directory that contains `plot_improved.py` |
| `FileNotFoundError: ...HDC/BTCA.csv` | Embedding files missing | Download the data drop from Zenodo (DOI in Suppl Table S17) |
| Plot looks different from the paper | A different version of matplotlib / seaborn can shift glyph metrics by a pixel or two | Use the package versions listed in Supplementary Table S15 |
