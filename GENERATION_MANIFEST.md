# Generation manifest — figures & tables

This file maps every manuscript figure and supplementary table to the exact
script and function that produces it, all bundled in this release. The package
is self-contained: no script reaches back into an external `resource/` checkout.

Two helper trees drive everything:

- `generators/` — the canonical figure/table builders (the real code).
- `reproduce_figures/` and `reproduce_tables/` — thin wrappers that put
  `generators/` on the path and call the builders.

> **Data dependency.** The builders ship as *code only*. The `data/` drop and
> the `results/` tree they read are not bundled (≈5 GB; Zenodo DOI assigned at
> acceptance; machine-readable manifest in Supplementary Table S17). Without that
> data the scripts stop with a `FileNotFoundError` — expected, not a bug. The
> shipped figures and `Supplementary_Tables_S1-S19.xlsx` are the final artifacts.

---

## Main figures

Builder: `generators/plot_improved.py` (`BASE_DIR = Path(__file__).parent`;
output → `generators/figures/` as PDF + SVG + PNG @ 300 dpi + TIFF/LZW).

| Figure | Function (`plot_improved.py`) | Wrapper (`reproduce_figures/`) |
|---|---|---|
| Fig 1 | — *schematic, hand-drawn in Adobe Illustrator; not code-generated* | — |
| Fig 2 | `make_main_fig2_simulation` | `fig2_simulation.py` |
| Fig 3 | `make_main_fig3_hdc` | `fig3_hdc.py` |
| Fig 4 | `make_main_fig4_ablation` | `fig4_ablation.py` |
| Fig 5 | `make_main_fig5_scmix_pbmc` | `fig5_scmix_pbmc.py` |
| Fig 6 | `make_main_fig6_lung_summary` | `fig6_lung_summary.py` |
| Fig 7 | `make_main_fig7_crossomics` | `fig7_crossomics.py` |

**Fig 6e** (per-dataset Integration-Balance rank box plot, 9 methods × 9
datasets) is drawn inside `make_main_fig6_lung_summary`, reading
`results/phase2/summary_IBfull_mean_std.csv`. This IB ranking superseded the
earlier lollipop / average-rank diagram (see the S16 note below).

**Fig 7** (cross-omics batch integration, scRNA-seq + scATAC-seq) is drawn by
`make_main_fig7_crossomics`, reading the cross-omics drop under
`generators/ATAC/` (`dataset/cell_metadata.csv`, the per-method embeddings in
`DR/`, and `metric/batch_remove.csv` + `metric/cluster.csv`). Like the other
builders it ships as code only; that data drop travels with the Zenodo archive.

---

## Supplementary figures (`reproduce_figures/supp_figures.py`)

| Suppl. fig | Function | Source module | Dataset / content |
|---|---|---|---|
| S1 | `make_supp_per_dataset("data1_scenario1")` | `plot_improved.py` | Simulated 1, scenario 1 |
| S2 | `make_supp_per_dataset("data1_scenario2")` | `plot_improved.py` | Simulated 1, scenario 2 |
| S3 | `make_supp_per_dataset("data2_scenario1")` | `plot_improved.py` | Simulated 2, scenario 1 |
| S4 | `make_supp_per_dataset("data2_scenario2")` | `plot_improved.py` | Simulated 2, scenario 2 |
| S5 | `make_supp_per_dataset("Sc_mixology")` | `plot_improved.py` | Sc_mixology |
| S6 | `make_supp_per_dataset("PBMC")` | `plot_improved.py` | PBMC |
| S7 | `make_supp_per_dataset("Human_Pancreas")` | `plot_improved.py` | Human Pancreas |
| S8 | `make_supp_per_dataset("Immune_human")` | `plot_improved.py` | Human Immune |
| S9 | `make_supp_per_dataset("gut")` | `plot_improved.py` | Gut |
| S10 | `plot_supp_fig_S7` → `FigS10_sensitivity` | `experiments/plot_supp_figures.py` | Parameter sensitivity |
| S11 | `plot_supp_fig_S8` → `FigS11_runtime` | `experiments/plot_supp_figures.py` | Runtime + peak memory |
| S12 | `plot_supp_fig_S6` → `FigS12_ext_ablation` | `experiments/plot_supp_figures.py` | Extended ablation |
| S13 | `make_supp_per_dataset("Lung")` | `plot_improved.py` | Mouse Lung |
| S14 | `main` → `FigS14_scIB_robustness` | `phase2_scib_robustness_fig.py` | scIB-aggregation robustness (P0-2) |

> The `plot_supp_fig_S6/S7/S8` function names are historical; the output
> filenames (`FigS12`/`FigS10`/`FigS11`) are authoritative. `supp_figures.py`
> documents this mapping at the top of the file.

---

## Supplementary tables (`reproduce_tables/supp_tables.py`)

A three-step append chain; each step reads the previous step's workbook and all
output lands in `reproduce_tables/output/`:

| Step | Script (`generators/experiments/`) | Entry | Builds | Output |
|---|---|---|---|---|
| 1 | `build_supp_tables.py` | `main` | S1 – S17 | `output/Supplementary_Tables_S1-S17.xlsx` |
| 2 | `make_supp_table_S18.py` | `main` | + S18 (`S18_scIB_standard`) | `output/Supplementary_Tables_S1-S18.xlsx` |
| 3 | `make_supp_table_S19.py` | `main` | + S19 (`S19_HDC_separability`) | `output/Supplementary_Tables_S1-S19.xlsx` |

The Supplementary **Figures** legend document is built separately by
`generators/experiments/build_supp_doc.py` (python-docx).

---

## S16 provenance caveat (important)

There are two different "S16" objects; do not conflate them:

- **`S16_IB_results`** — the per-dataset Integration-Balance rank table that is
  the numerical source of **Fig. 6e**. This is the sheet in the shipped
  `Publish_version/Supplementary_Tables_S1-S19.xlsx` and is **authoritative**. It
  was rebuilt ad hoc during the Fig. 6e redesign and has **no standalone `.py`
  generator** in this release. (Fig. 6e itself is drawn by
  `make_main_fig6_lung_summary` from `results/phase2/summary_IBfull_mean_std.csv`.)
- **`S16_Average_rank`** — the legacy average-rank meta-summary emitted by
  `build_supp_tables.py`. It is **superseded** by the IB ranking and is what a
  from-scratch run of the chain above will write into its S16 sheet. When
  validating a regenerated workbook, treat the shipped `S16_IB_results` as
  authoritative for S16.

## Nemenyi critical difference (S18 / Fig. S14)

The Nemenyi critical difference is deterministic in (k, N, α):
`CD = q_α · √(k(k+1)/(6N))`. For k = 9, N = 9, α = 0.05 (q₀.₀₅ = 3.102) this is
**4.00**. An earlier upstream CSV stored a rounded 4.16; `make_supp_table_S18.py`
now recomputes CD from the q-table and no longer trusts that field. The Friedman
χ² and p-values legitimately differ between the IB ranking (χ² = 44.4,
p = 4.8 × 10⁻⁷) and the community-standard scIB ranking (χ² = 43.3,
p = 7.7 × 10⁻⁷); those are not typos.

## Method codenames

Two internal method keys are mapped to display names by the builders
(`METHOD_DISPLAY` / `disp()`); they also appear as `.h5ad` / `.csv` filenames in
the data tree, so they are structural and must not be "corrected":

| Internal key | Displayed as |
|---|---|
| `BTCA` | **scCAT** |
| `INSCT_Unsupervised` | **INSCT** |

The benchmark compares **12 methods** = scCAT + 11 baselines (INSCT, SPDR,
DeepBID, Scanorama, DESC, fastMNN, scBCN, Harmony, scVI, scANVI, BBKNN).
