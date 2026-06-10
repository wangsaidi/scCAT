# Reproducing the supplementary tables

This folder regenerates the 19-sheet supplementary workbook
(`Supplementary_Tables_S1-S19.xlsx`) from the table-generation code bundled in
this release (`../generators/experiments/`). No code here reaches back into an
external `resource/` checkout.

The scripts run as a **three-step chain** — each step appends to the workbook
produced by the previous step:

| Step | Script (`../generators/experiments/`) | Builds | Writes |
|---|---|---|---|
| 1 | `build_supp_tables.py` | S1 – S17 | `output/Supplementary_Tables_S1-S17.xlsx` |
| 2 | `make_supp_table_S18.py` | + S18 (scIB-standard robustness) | `output/Supplementary_Tables_S1-S18.xlsx` |
| 3 | `make_supp_table_S19.py` | + S19 (HDC resolution-free separability) | `output/Supplementary_Tables_S1-S19.xlsx` |

All output lands in `reproduce_tables/output/` next to these scripts. **Nothing
is written back into the shipped `../../Supplementary_Tables_S1-S19.xlsx`**, which
remains the authoritative deliverable (see the S16 caveat in §4).

---

## 1. Prerequisites

```bash
pip install -e ".[plot]"
# or
pip install -r requirements-plot.txt
```

The table builders need `numpy`, `pandas`, `scipy`, `scikit-learn`, `openpyxl`
and `anndata`; step 1 additionally imports the embedding loaders from
`../generators/plot_improved.py` (already on the path — the wrapper handles
this), so the plotting extras above are the simplest one-shot install.

---

## 2. Data download

These scripts read the **experimental result tables**, not raw counts. They
expect a `results/` tree next to the generators:

```
generators/experiments/results/
├── phase2/
│   ├── scib_standard_secondary.csv            # S18 (Friedman/Nemenyi, avg rank)
│   ├── scib_standard_per_method_dataset.csv   # S18 (per method × dataset scIB)
│   └── friedman_nemenyi_IBfull_secondary.csv  # S18 (IB rank, for Spearman rho)
└── phase3/
    └── hdc_dc/
        └── separability_diagnostic.csv         # S19 (1-NN / silhouette / ARI)
```

Step 1 (`build_supp_tables.py`) additionally reads the per-method metric CSVs
and embeddings described in
[`../reproduce_figures/README.md`](../reproduce_figures/README.md) §2 (the same
`data/` drop used to reproduce the figures).

This `results/` tree is **not bundled** in the GitHub release (it is part of the
~5 GB data archive, Zenodo DOI assigned at acceptance; a machine-readable
manifest is in Supplementary Table S17). Until it is present, the scripts will
stop with a `FileNotFoundError` — that is expected, not a bug. The shipped
`Supplementary_Tables_S1-S19.xlsx` already contains the final, verified numbers.

---

## 3. Running the chain

Run everything in order:

```bash
python supp_tables.py        # runs steps 1 → 2 → 3 into output/
```

…or run the steps individually (they must run in order — each reads the
previous workbook):

```bash
cd ../generators/experiments
python build_supp_tables.py     # S1 – S17
python make_supp_table_S18.py   # + S18
python make_supp_table_S19.py   # + S19
```

Either way the final workbook is `reproduce_tables/output/Supplementary_Tables_S1-S19.xlsx`.

---

## 4. The S16 caveat (read before comparing to the shipped workbook)

`build_supp_tables.py` writes an S16 sheet named **`S16_Average_rank`** — the
legacy average-rank meta-summary. During the Fig. 6e redesign this view was
**superseded** by an Integration-Balance ranking, and the shipped
`../../Supplementary_Tables_S1-S19.xlsx` instead carries a sheet named
**`S16_IB_results`** (the per-dataset IB rank distribution that is the numerical
source of main Fig. 6e). That IB master sheet was rebuilt ad hoc and has **no
standalone `.py` generator** in this release; Fig. 6e itself is drawn by
`../generators/plot_improved.py::make_main_fig6_lung_summary`, which reads
`results/phase2/summary_IBfull_mean_std.csv`.

Consequence: a from-scratch run here reproduces S1 – S15, S17 – S19 faithfully,
but its S16 sheet is the **legacy** average-rank view, not the IB master. When
validating, compare against the shipped workbook with this in mind — the shipped
file is authoritative for S16.

The Nemenyi critical difference printed for S18 is computed deterministically
from (k, N, α) — `CD = q_α · √(k(k+1)/(6N))` — and for k = 9, N = 9, α = 0.05
equals **4.00** (an earlier upstream CSV stored a rounded 4.16; the script no
longer trusts that field).

---

## 5. Method codenames

Two internal method keys are mapped to display names by the scripts and also
appear as `.h5ad` / `.csv` filenames in the data tree (they are structural, not
typos):

| Internal key | Displayed as |
|---|---|
| `BTCA` | **scCAT** |
| `INSCT_Unsupervised` | **INSCT** |

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: ...results/phase2/...` or `...phase3/hdc_dc/...` | The `results/` tree is not present | Download the data archive (DOI in Suppl Table S17) and extract it under `generators/experiments/results/` |
| `FileNotFoundError: ...Supplementary_Tables_S1-S17.xlsx` when running S18 alone | Step 1 has not been run yet | Run `build_supp_tables.py` first (or just use `supp_tables.py`, which chains all three) |
| `ModuleNotFoundError: plot_improved` | Step 1 imports the embedding loaders from `../generators/plot_improved.py` | Run via `supp_tables.py` (it puts `generators/experiments/` on the path), or run the scripts from inside `generators/experiments/` |
| S16 sheet differs from the shipped workbook | Expected — see §4 (legacy `S16_Average_rank` vs authoritative `S16_IB_results`) | None; the shipped workbook is authoritative for S16 |
