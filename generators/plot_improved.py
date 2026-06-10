#!/usr/bin/env python3
"""
plot_improved.py  —  Nature Methods-style figure generator for the scCAT manuscript.
====================================================================================

SIX FIGURES, SIX TRULY DISTINCT ARCHITECTURES.  Each figure leads with its own
chart family, has its own panel composition, and tells the specific story of
its dataset.

    Fig 2  data2_scenario1   "Comprehensive baseline"   — cell-type-first +
                                                          method ranking strip
    Fig 3  data2_scenario2   "Batch-specific failure"   — TOP-4 large UMAPs +
                                                          confusion matrix
    Fig 4  Sc_mixology       "Cross-platform leaders"   — LANDSCAPE +
                                                          scCAT/SPDR head-to-head +
                                                          silhouette ridge
    Fig 5  HDC               "Rare-cell rescue"         — schematic + scCAT
                                                          featured + 2-D trade-off
    Fig 6  PBMC              "Condition integration"    — KDE density overlay +
                                                          paired dot plot
    Fig 7  Lung              "Scaling robustness"       — scCAT large pair +
                                                          per-batch heatmap +
                                                          biology-vs-mixing curve

Usage
-----
    python plot_improved.py                   # generate all six figures
    python plot_improved.py HDC               # one figure
    python plot_improved.py Sc_mixology PBMC  # several figures
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Standard library + dependencies
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle, FancyBboxPatch

from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Global style (Nature Methods)
# ─────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "svg.fonttype":       "none",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi":         150,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "legend.frameon":     False,
})

DOUBLE_COL = 7.09
FS_PANEL   = 8
FS_TITLE   = 7
FS_LABEL   = 7
FS_TICK    = 6
FS_LEGEND  = 7
FS_ANNOT   = 5.5

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Method settings
# ─────────────────────────────────────────────────────────────────────────────
METHODS = [
    "Scanorama", "DeepBID", "DESC", "scBCN",
    "fastMNN",   "INSCT_Unsupervised", "SPDR",
    # Phase 1 (Nat Methods reviewer-response): three gold-standard baselines
    "Harmony", "scVI", "scANVI",
    # Phase 1+ (grid symmetry + close BBKNN-citation): graph-only MNN-family baseline
    "BBKNN",
    "BTCA",
]
RANK_METHODS = [
    "DeepBID", "scBCN", "DESC", "Scanorama",
    "fastMNN", "INSCT_Unsupervised", "SPDR",
    "Harmony", "scVI", "scANVI", "BBKNN",
    "BTCA",
]
METHOD_DISPLAY = {
    "INSCT_Unsupervised": "INSCT",
    "SPDR":      "SPDR",
    "Scanorama": "Scanorama",
    "DeepBID":   "DeepBID",
    "DESC":      "DESC",
    "scBCN":     "scBCN",
    "fastMNN":   "fastMNN",
    "Harmony":   "Harmony",
    "scVI":      "scVI",
    "scANVI":    "scANVI",
    "BBKNN":     "BBKNN",
    "BTCA":      "scCAT",
}
METHOD_COLORS = {
    "INSCT_Unsupervised": "#6D8FC7",
    "SPDR":               "#6AAE75",
    "Scanorama":          "#C592C9",
    "DeepBID":            "#9E7BB5",
    "DESC":               "#D99AB4",
    "scBCN":              "#CE6F9A",
    "fastMNN":            "#78B7C5",
    "Harmony":            "#DDA45A",
    "scVI":               "#9B84B7",
    "scANVI":             "#62B5A5",
    "BBKNN":              "#C27C55",
    "BTCA":               "#2A8F45",
}
H5AD_INFO = {
    "INSCT_Unsupervised": {"file": "INSCT_Unsupervised.h5ad", "key": "X_tnn",  "align": "obs_names"},
    "SPDR":               {"file": "SPDR.h5ad",               "key": "X_spdr", "align": "metadata_order"},
    "Harmony":            {"file": "Harmony.h5ad",            "key": "X_umap", "align": "obs_names"},
    "scVI":               {"file": "scVI.h5ad",               "key": "X_umap", "align": "obs_names"},
    "scANVI":             {"file": "scANVI.h5ad",             "key": "X_umap", "align": "obs_names"},
    "BBKNN":              {"file": "BBKNN.h5ad",              "key": "X_umap", "align": "obs_names"},
}
BATCH_COL, CELLTYPE_COL, METHOD_COL = "batch", "cell_type", "method"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-dataset configurations
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "data2_scenario1": {
        "label":       "Simulated Dataset 3",
        "emb_dir":     "embedding/data2_scenario1",
        "meta_csv":    "datasets/data2_scenario1/cell_metadata.csv",
        "batch_metric":"metric/batch_remove/data2_scenario1.csv",
        "clust_metric":"metric/cluster/data2_scenario1.csv",
        "n_cells": 25000, "point_size": 5, "alpha": 0.50,
        "batch_specific_types": [],
        "batch_palette": {
            "Batch1":"#1F77B4","Batch2":"#FF7F0E","Batch3":"#2CA02C",
            "Batch4":"#D62728","Batch5":"#9467BD","Batch6":"#8C564B",
            "Batch7":"#E377C2","Batch8":"#6F6F6F","Batch9":"#BCBD22",
            "Batch10":"#17BECF",
        },
        "celltype_palette": {
            "Group1":"#1F77B4","Group2":"#FF7F0E","Group3":"#2CA02C",
            "Group4":"#D62728","Group5":"#9467BD","Group6":"#8C564B",
            "Group7":"#E377C2","Group8":"#6F6F6F","Group9":"#BCBD22",
            "Group10":"#17BECF",
        },
    },
    "data2_scenario2": {
        "label":       "Simulated Dataset 4",
        "emb_dir":     "embedding/data2_scenario2",
        "meta_csv":    "datasets/data2_scenario2/cell_metadata.csv",
        "batch_metric":"metric/batch_remove/data2_scenario2.csv",
        "clust_metric":"metric/cluster/data2_scenario2.csv",
        "n_cells": 13950, "point_size": 7, "alpha": 0.60,
        "batch_specific_types": ["Group1", "Group3", "Group5", "Group7"],
        "batch_palette": {
            "Batch1":"#1F77B4","Batch2":"#FF7F0E","Batch3":"#2CA02C",
            "Batch4":"#D62728","Batch5":"#9467BD","Batch6":"#8C564B",
            "Batch7":"#E377C2","Batch8":"#6F6F6F","Batch9":"#BCBD22",
            "Batch10":"#17BECF",
        },
        "celltype_palette": {
            "Group1":"#1F77B4","Group2":"#FF7F0E","Group3":"#2CA02C",
            "Group4":"#D62728","Group5":"#9467BD","Group6":"#8C564B",
            "Group7":"#E377C2","Group8":"#6F6F6F","Group9":"#BCBD22",
            "Group10":"#17BECF",
        },
    },
    "Sc_mixology": {
        "label":       "Sc_mixology",
        "emb_dir":     "embedding/Sc_mixology",
        "meta_csv":    "datasets/Sc_mixology/cell_metadata.csv",
        "batch_metric":"metric/batch_remove/Sc_mixology.csv",
        "clust_metric":"metric/cluster/Sc_mixology.csv",
        "n_cells": 1401, "point_size": 22, "alpha": 0.85,
        "batch_specific_types": [],
        "batch_palette": {
            "10x":     "#2271B2",
            "celseq2": "#E69F00",
            "dropseq": "#009E73",
        },
        "celltype_palette": {
            "HCC827": "#E15759",
            "H1975":  "#4E79A7",
            "H2228":  "#F28E2B",
        },
    },
    "HDC": {
        "label":       "Human Dendritic Cells",
        "emb_dir":     "embedding/HDC",
        "meta_csv":    "datasets/HDC/cell_metadata.csv",
        "batch_metric":"metric/batch_remove/HDC.csv",
        "clust_metric":"metric/cluster/HDC.csv",
        "n_cells": 569, "point_size": 32, "alpha": 0.90,
        "batch_specific_types": ["CD141", "CD1C"],
        "batch_palette": {
            "0": "#4292C6",
            "1": "#D94801",
        },
        "celltype_palette": {
            "pDC":       "#4292C6",
            "DoubleNeg": "#41AE76",
            "CD141":     "#D94801",
            "CD1C":      "#8B2500",
        },
    },
    "PBMC": {
        "label":       "Human PBMC",
        "emb_dir":     "embedding/PBMC",
        "meta_csv":    "datasets/PBMC/cell_metadata.csv",
        "batch_metric":"metric/batch_remove/PBMC.csv",
        "clust_metric":"metric/cluster/PBMC.csv",
        "n_cells": 13576, "point_size": 6, "alpha": 0.65,
        "batch_specific_types": [],
        "batch_palette": {
            "control":    "#4393C3",
            "stimulated": "#D6604D",
        },
        "celltype_palette": {
            "B":          "#4DAF4A",
            "CD14 Mono":  "#FF7F00",
            "CD16 Mono":  "#F781BF",
            "CD4 T":      "#377EB8",
            "CD8 T":      "#E41A1C",
            "DC":         "#A65628",
            "NK":         "#984EA3",
            "T":          "#999999",
        },
    },
    "Lung": {
        "label":       "Mouse Lung",
        "emb_dir":     "embedding/Lung",
        "meta_csv":    "datasets/Lung/cell_metadata.csv",
        "batch_metric":"metric/batch_remove/Lung.csv",
        "clust_metric":"metric/cluster/Lung.csv",
        "n_cells": 32472, "point_size": 3, "alpha": 0.40,
        "batch_specific_types": [],
        "batch_palette": {
            "1":"#08306B","2":"#08519C","3":"#2171B5",
            "4":"#4292C6","5":"#6BAED6","6":"#9ECAE1",
            "A1":"#7F0000","A2":"#B30000","A3":"#D7301F",
            "A4":"#EF6548","A5":"#FC8D59","A6":"#FDCC8A",
            "B1":"#00441B","B2":"#006D2C","B3":"#238B45","B4":"#41AE76",
        },
        "celltype_palette": {
            "Basal 1":"#A6CEE3","Basal 2":"#1F78B4","Ciliated":"#33A02C",
            "Ionocytes":"#B2DF8A","Secretory":"#FB9A99","Type 1":"#E31A1C",
            "Type 2":"#FDBF6F","Endothelium":"#FF7F00","Fibroblast":"#CAB2D6",
            "Lymphatic":"#6A3D9A","B cell":"#FFFF99","Dendritic cell":"#B15928",
            "Macrophage":"#8DD3C7","Mast cell":"#FFFFB3",
            "Neutrophil_CD14_high":"#BEBADA","Neutrophils_IL1R2":"#FB8072",
            "T/NK cell":"#80B1D3",
        },
    },
}
ALL_DATASETS = list(DATASET_CONFIGS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 4b.  Auto-derived configs for the SUPPLEMENTARY datasets
#      Sim 1 / Sim 2 / Human Pancreas / Human Immune / Gut
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PALETTE = [
    "#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B",
    "#E377C2","#7F7F7F","#BCBD22","#17BECF","#AEC7E8","#FFBB78",
    "#98DF8A","#FF9896","#C5B0D5","#C49C94","#F7B6D2","#C7C7C7",
    "#DBDB8D","#9EDAE5",
]


def _auto_palette(values):
    cats = []
    seen = set()
    for v in values:
        s = str(v)
        if s not in seen:
            seen.add(s); cats.append(s)
    return {c: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i, c in enumerate(cats)}


def _make_supp_config(key, label, emb_subdir, n_cells_hint, point_size, alpha):
    """Lazy-build a per-dataset config the same shape as the main 6 datasets.
    Palettes are auto-derived from the actual metadata at first use."""
    cfg = {
        "label":       label,
        "emb_dir":     f"embedding/{emb_subdir}",
        "meta_csv":    f"datasets/{emb_subdir}/cell_metadata.csv",
        "batch_metric": f"metric/batch_remove/{emb_subdir}.csv",
        "clust_metric": f"metric/cluster/{emb_subdir}.csv",
        "n_cells": n_cells_hint, "point_size": point_size, "alpha": alpha,
        "batch_specific_types": [],
        "batch_palette": None,        # filled at runtime
        "celltype_palette": None,
    }
    try:
        meta = pd.read_csv(BASE_DIR / cfg["meta_csv"])
        meta = meta.rename(columns={meta.columns[0]: "cell"})
        batch_vals = meta["batch"].astype(str).values
        ct_vals    = meta["cell_type"].astype(str).values
        cfg["batch_palette"]    = _auto_palette(batch_vals)
        cfg["celltype_palette"] = _auto_palette(ct_vals)
        # Detect batch-specific cell types
        n_batches = meta["batch"].nunique()
        bs = []
        for t in pd.unique(ct_vals):
            t_batches = pd.unique(batch_vals[ct_vals == t])
            if len(t_batches) < n_batches:
                bs.append(str(t))
        cfg["batch_specific_types"] = bs
    except Exception as e:
        warnings.warn(f"[supp config] Could not read metadata for {key}: {e}")
    return cfg


DATASET_CONFIGS["data1_scenario1"] = _make_supp_config(
    "data1_scenario1", "Simulated Dataset 1", "data1_scenario1",
    n_cells_hint=3000, point_size=18, alpha=0.80,
)
DATASET_CONFIGS["data1_scenario2"] = _make_supp_config(
    "data1_scenario2", "Simulated Dataset 2", "data1_scenario2",
    n_cells_hint=3000, point_size=18, alpha=0.80,
)
DATASET_CONFIGS["Human_Pancreas"] = _make_supp_config(
    "Human_Pancreas", "Human Pancreas", "Human_Pancreas",
    n_cells_hint=16382, point_size=5, alpha=0.55,
)
DATASET_CONFIGS["Immune_human"] = _make_supp_config(
    "Immune_human", "Human Immune", "Immune_human",
    n_cells_hint=33506, point_size=3, alpha=0.40,
)
DATASET_CONFIGS["gut"] = _make_supp_config(
    "gut", "Gut", "gut",
    n_cells_hint=9842, point_size=7, alpha=0.65,
)

ALL_SUPP_DATASETS = ["data1_scenario1", "data1_scenario2", "Human_Pancreas",
                      "Immune_human", "gut"]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Data-loading utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_metadata(meta_path):
    meta = pd.read_csv(meta_path)
    meta = meta.rename(columns={meta.columns[0]: "cell"})
    for c in ["cell", BATCH_COL, CELLTYPE_COL]:
        meta[c] = meta[c].astype(str)
    return meta[["cell", BATCH_COL, CELLTYPE_COL]]


def _load_csv_embedding(method, emb_dir, meta):
    p = emb_dir / f"{method}.csv"
    if not p.exists():
        p = emb_dir / f"{METHOD_DISPLAY.get(method, method)}.csv"
    if not p.exists():
        hits = sorted(emb_dir.glob(f"*{method}*.csv"))
        if not hits:
            raise FileNotFoundError(f"No CSV embedding for {method} in {emb_dir}")
        p = hits[0]
    emb = pd.read_csv(p)
    emb.columns = ["cell", "UMAP1", "UMAP2"] + list(emb.columns[3:])
    emb["cell"] = emb["cell"].astype(str)
    df = emb[["cell","UMAP1","UMAP2"]].merge(meta, on="cell", how="inner")
    if df.empty:
        raise ValueError(f"{method}: no overlap between embedding and metadata")
    return df


def _load_h5ad_embedding(method, emb_dir, meta):
    info  = H5AD_INFO[method]
    path  = emb_dir / info["file"]
    if not path.exists():
        hits = sorted(emb_dir.glob(f"*{method}*.h5ad"))
        if not hits:
            raise FileNotFoundError(f"No h5ad embedding for {method} in {emb_dir}")
        path = hits[0]
    adata = ad.read_h5ad(path)
    key   = info["key"]
    if key not in adata.obsm:
        raise KeyError(f"{method}: key {key!r} not in adata.obsm")
    arr   = np.asarray(adata.obsm[key])
    if info["align"] == "metadata_order":
        if arr.shape[0] != len(meta):
            raise ValueError(f"{method}: row count mismatch with metadata")
        df = meta.copy()
        df["UMAP1"], df["UMAP2"] = arr[:,0], arr[:,1]
    else:
        emb_df = pd.DataFrame({
            "cell": adata.obs_names.astype(str),
            "UMAP1": arr[:,0], "UMAP2": arr[:,1],
        })
        df = emb_df.merge(meta, on="cell", how="inner")
    return df


def load_all_embeddings(methods, emb_dir, meta):
    out = {}
    for m in methods:
        try:
            out[m] = _load_h5ad_embedding(m, emb_dir, meta) if m in H5AD_INFO \
                     else _load_csv_embedding(m, emb_dir, meta)
        except Exception as e:
            warnings.warn(f"Skipping {m}: {e}")
    return out


def _canonical(x):
    rev = {v: k for k, v in METHOD_DISPLAY.items()}
    return rev.get(str(x), str(x))


def read_metric(csv_path, methods):
    df = pd.read_csv(csv_path)
    mc = next((c for c in df.columns if c.lower() in {"method","methods","name"}), None)
    if mc is None:
        if len(df) == len(methods):
            df.insert(0, METHOD_COL, methods)
        else:
            raise ValueError(f"Cannot find method column in {csv_path}")
    else:
        df = df.rename(columns={mc: METHOD_COL})
    df[METHOD_COL] = df[METHOD_COL].map(_canonical)
    df[METHOD_COL] = pd.Categorical(df[METHOD_COL], categories=methods, ordered=True)
    return df


def _resolve(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        lo = {x.lower(): x for x in df.columns}
        if c.lower() in lo:
            return lo[c.lower()]
    return None


def _val(df, method, col):
    if col is None:
        return np.nan
    row = df[df[METHOD_COL] == method]
    if row.empty:
        return np.nan
    return pd.to_numeric(row[col], errors="coerce").iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Common drawing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _panel_label(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")


def _nature_embed_style(ax, title=None, title_size=None):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title is not None:
        ax.set_title(title,
                     fontsize=title_size or FS_TITLE,
                     fontweight="bold", pad=3)


def plot_embedding(ax, df, color_col, palette, method,
                   batch_specific=None, pt_size=10, alpha=0.7):
    """Standard scatter; batch-specific cell types drawn last with edge ring."""
    title  = METHOD_DISPLAY.get(method, method)
    bs_set = set(batch_specific or [])

    bg = df[~df[color_col].astype(str).isin(bs_set)]
    for cat, sub in bg.groupby(color_col, sort=False):
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=pt_size,
                   c=palette.get(str(cat), "#AAA"), alpha=alpha,
                   linewidths=0, rasterized=True)

    for cat in bs_set:
        sub = df[df[color_col].astype(str) == str(cat)]
        if sub.empty:
            continue
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=pt_size * 1.8,
                   c=palette.get(str(cat), "#AAA"),
                   alpha=min(alpha + 0.1, 1.0),
                   linewidths=0.45, edgecolors="#222",
                   rasterized=True, zorder=5)

    _nature_embed_style(ax, title)


def plot_embedding_featured(ax, df, color_col, palette, method,
                             batch_specific=None, pt_size=22, alpha=0.85):
    """Large featured UMAP used in 'hero' panels; bigger title and points."""
    bs_set = set(batch_specific or [])
    bg = df[~df[color_col].astype(str).isin(bs_set)]
    for cat, sub in bg.groupby(color_col, sort=False):
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=pt_size,
                   c=palette.get(str(cat), "#AAA"), alpha=alpha,
                   linewidths=0, rasterized=True)
    for cat in bs_set:
        sub = df[df[color_col].astype(str) == str(cat)]
        if sub.empty:
            continue
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=pt_size * 1.6,
                   c=palette.get(str(cat), "#AAA"),
                   alpha=min(alpha + 0.05, 1.0),
                   linewidths=0.5, edgecolors="#222",
                   rasterized=True, zorder=5)
    title = METHOD_DISPLAY.get(method, method)
    _nature_embed_style(ax, title, title_size=FS_TITLE + 1.5)


def plot_embedding_focus(ax, df, focus_types, palette, method,
                         pt_size=10, alpha=0.85, bg_color="#DDDDDD"):
    """Show only focus_types in colour; other cells as grey context."""
    title = METHOD_DISPLAY.get(method, method)
    focus_set = set(map(str, focus_types))
    bg = df[~df[CELLTYPE_COL].astype(str).isin(focus_set)]
    if not bg.empty:
        ax.scatter(bg["UMAP1"], bg["UMAP2"], s=pt_size * 0.7,
                   c=bg_color, alpha=alpha * 0.4, linewidths=0,
                   rasterized=True)
    for ct in focus_types:
        sub = df[df[CELLTYPE_COL].astype(str) == str(ct)]
        if sub.empty:
            continue
        ax.scatter(sub["UMAP1"], sub["UMAP2"], s=pt_size * 1.6,
                   c=palette.get(str(ct), "#888"), alpha=alpha,
                   linewidths=0.35, edgecolors="#222",
                   rasterized=True, zorder=5)
    _nature_embed_style(ax, title)


def _figure_legend(fig, palette, title, anchor, ncol_max=12):
    handles = [
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor=c, markeredgecolor="none",
               markersize=5.5, label=str(lbl))
        for lbl, c in palette.items()
    ]
    ncol = 1 if len(handles) <= ncol_max else 2
    fig.legend(handles=handles, title=title,
               loc="center left", bbox_to_anchor=anchor,
               frameon=False, fontsize=FS_LEGEND,
               title_fontsize=FS_LEGEND + 0.5,
               ncol=ncol, handletextpad=0.4, borderaxespad=0)


def _lim(values, pad=0.22, minspan=0.16):
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-0.05, 1.05)
    lo, hi = v.min(), v.max()
    span = max(hi - lo, minspan)
    p = max(span * pad, minspan * 0.18)
    return (max(-0.05, lo - p), min(1.05, hi + p))


def plot_scatter_metrics(ax, batch_df, x_col, y_col, xlabel, ylabel, title):
    """Method-coloured scatter (e.g. LISI: cLISI on x, iLISI on y).

    Labels are drawn with a semi-transparent white box and routed by
    adjustText so they do not overlap each other or cover the dots.
    Axis limits use extra padding to give labels room.
    """
    pts = [(m, _val(batch_df, m, x_col), _val(batch_df, m, y_col)) for m in METHODS]
    pts = [(m, x, y) for m, x, y in pts if np.isfinite(x) and np.isfinite(y)]
    if pts:
        # Wider padding so adjustText has room to push labels outward
        ax.set_xlim(*_lim([p[1] for p in pts], pad=0.25))
        ax.set_ylim(*_lim([p[2] for p in pts], pad=0.25))

    # Dots first (drawn under labels)
    for m, x, y in pts:
        is_sccat = (m == "BTCA")
        ax.scatter(x, y, s=60 if is_sccat else 38,
                   color=METHOD_COLORS.get(m, "#888"),
                   edgecolors="white", linewidth=0.6,
                   alpha=0.95, zorder=4 if not is_sccat else 5,
                   clip_on=False)

    # Labels with white background to prevent visual merging with dots
    texts = []
    for m, x, y in pts:
        is_sccat = (m == "BTCA")
        t = ax.text(
            x, y, METHOD_DISPLAY.get(m, m),
            fontsize=FS_ANNOT + (1.0 if is_sccat else 0.5),
            color=METHOD_COLORS.get(m, "#888"),
            ha="center", va="center",
            fontweight="bold" if is_sccat else "normal",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor="white", edgecolor="none",
                      alpha=0.70),
        )
        texts.append(t)

    # Auto-route labels so they don't overlap each other or cover dots
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            expand=(1.3, 1.5),
            force_text=(0.7, 1.0),
            force_static=(0.5, 0.7),
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.4, alpha=0.7),
            max_move=25,
        )
    except ImportError:
        # Fallback: leave labels at point centre (still better than no fix)
        pass

    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.6)


def plot_kbet_lollipop(ax, batch_df, kbet_col):
    """Horizontal lollipop of kBET per method, with numeric label to the right
    of each dot.

    The numeric label is placed at a uniform x-offset proportional to the
    overall axis range so it never overlaps the dot.  The right-hand axis
    padding is enlarged so labels for values close to the maximum are not
    clipped by the spine.
    """
    vals    = [_val(batch_df, m, kbet_col) for m in RANK_METHODS]
    labels  = [METHOD_DISPLAY.get(m, m)     for m in RANK_METHODS]
    colors  = [METHOD_COLORS.get(m, "#888") for m in RANK_METHODS]
    y       = np.arange(len(RANK_METHODS))

    finite = [v for v in vals if np.isfinite(v)]
    # Compute uniform numeric-label offset = 6.5% of axis range (extra breathing
    # room so labels are visually well separated from the dot, not just clearing it)
    max_v = max(finite) if finite else 1.0
    text_offset = max(0.065 * max_v, 0.06)

    for i, (v, c) in enumerate(zip(vals, colors)):
        if not np.isfinite(v):
            continue
        ax.plot([0, v], [i, i], color=c, lw=1.0, alpha=0.7)
        ax.scatter(v, i, color=c, s=32, zorder=4,
                   edgecolors="white", linewidth=0.4)
        ax.text(v + text_offset, i, f"{v:.2f}",
                va="center", ha="left", fontsize=FS_ANNOT, color=c)
    sccat_idx = RANK_METHODS.index("BTCA")
    ax.axhspan(sccat_idx - 0.45, sccat_idx + 0.45,
               color=METHOD_COLORS["BTCA"], alpha=0.08, lw=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("kBET rejection rate (↓ better)", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("kBET", fontsize=FS_TITLE, pad=4)
    if finite:
        # Add ~55% headroom so the shifted numeric labels (e.g. "1.00") clear the right edge
        ax.set_xlim(-0.02, max_v * 1.55 + 0.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2.5, width=0.6)
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)


def plot_eval_heatmap(ax, batch_df, clust_df, title="Evaluation Summary"):
    """Compact evaluation heatmap, methods × 6 metrics."""
    col_spec = [
        ("iLISI",     batch_df, ["iLISI"],                       True),
        ("cLISI",     batch_df, ["cLISI_purity","cLISI"],        True),
        ("ASW\ncell", batch_df, ["ASW_celltype","ASW_cell_type"],True),
        ("ASW\nbatch",batch_df, ["ASW_batch_mixing","ASW_batch"],True),
        ("ARI",       clust_df, ["ARI"],                         True),
        ("NMI",       clust_df, ["NMI"],                         True),
    ]
    resolved = [(d, df, _resolve(df, cs), hi) for d, df, cs, hi in col_spec]
    n_m, n_c = len(RANK_METHODS), len(resolved)
    raw  = np.full((n_m, n_c), np.nan)
    for j, (_, df, col, _) in enumerate(resolved):
        for i, m in enumerate(RANK_METHODS):
            raw[i, j] = _val(df, m, col)
    norm = np.full_like(raw, 0.5)
    for j, (_, _, _, hib) in enumerate(resolved):
        col = raw[:, j]
        valid = np.isfinite(col)
        if valid.sum() > 1:
            vmin, vmax = col[valid].min(), col[valid].max()
            if vmax > vmin:
                norm[:, j] = (col - vmin) / (vmax - vmin)
            if not hib:
                norm[:, j] = 1 - norm[:, j]
    im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_m):
        for j in range(n_c):
            v = raw[i, j]
            if np.isfinite(v):
                b = norm[i, j]
                txt = "white" if b < 0.2 or b > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color=txt)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels([r[0] for r in resolved],
                       fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in RANK_METHODS],
                       fontsize=FS_TICK)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    sccat_idx = RANK_METHODS.index("BTCA")
    ax.add_patch(Rectangle((-0.5, sccat_idx - 0.5), n_c, 1,
                           fill=False, edgecolor=METHOD_COLORS["BTCA"],
                           linewidth=0.8, clip_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=18)
    cbar.ax.tick_params(labelsize=FS_TICK - 1, length=2)
    cbar.set_label("Norm.\nscore", fontsize=FS_ANNOT, labelpad=2)
    cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(["low","mid","high"])


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Reusable analytical panels carried over from previous version
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_specific_preservation(ax, emb_data, meta, methods,
                                     batch_specific_types, n_clusters=None):
    """Bar chart: modal-cluster purity of batch-specific cells, per method."""
    if n_clusters is None:
        n_clusters = meta[CELLTYPE_COL].nunique()
    scores = {}
    for m in methods:
        if m not in emb_data:
            continue
        df = emb_data[m]
        X  = df[["UMAP1","UMAP2"]].values
        if len(X) < n_clusters:
            continue
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            preds = km.fit_predict(X)
        except Exception:
            continue
        ct = df[CELLTYPE_COL].astype(str).values
        pur = []
        for bs in batch_specific_types:
            mask = ct == str(bs)
            if mask.sum() == 0:
                continue
            mode_c = pd.Series(preds[mask]).mode().iloc[0]
            cmask  = preds == mode_c
            pur.append((ct[cmask] == str(bs)).sum() / cmask.sum())
        if pur:
            scores[m] = float(np.mean(pur))
    method_list = [m for m in RANK_METHODS if m in scores]
    vals   = [scores[m] for m in method_list]
    colors = [METHOD_COLORS.get(m, "#888") for m in method_list]
    bars = ax.bar(range(len(method_list)), vals, color=colors,
                  alpha=0.88, edgecolor="white", linewidth=0.5, width=0.72)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=FS_ANNOT)
    ax.set_xticks(range(len(method_list)))
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_ylabel("Cluster purity of\nbatch-specific cells", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Batch-specific preservation", fontsize=FS_TITLE, pad=4)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]:
        ax.spines[sp].set_linewidth(0.6)
    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.axvspan(i - 0.5, i + 0.5, color=METHOD_COLORS["BTCA"], alpha=0.07, lw=0)


def plot_celltype_silhouette_bars(ax, emb_data, meta, methods=None,
                                  celltype_palette=None):
    """Per-cell-type mean silhouette, grouped bar (Fig 5)."""
    methods = methods or RANK_METHODS
    cell_types = list(celltype_palette.keys()) if celltype_palette else \
                 list(meta[CELLTYPE_COL].unique())
    per_m = {}
    for m in methods:
        if m not in emb_data:
            continue
        df = emb_data[m]
        labs = df[CELLTYPE_COL].astype(str)
        codes = labs.astype("category").cat.codes.values
        if len(np.unique(codes)) < 2:
            continue
        try:
            sil = silhouette_samples(df[["UMAP1","UMAP2"]].values, codes)
        except Exception:
            continue
        per_m[m] = {ct: float(sil[(labs == ct).values].mean()) if (labs == ct).any() else np.nan
                    for ct in cell_types}
    method_list = [m for m in methods if m in per_m]
    n_m, n_t = len(method_list), len(cell_types)
    if n_m == 0 or n_t == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center"); return
    x = np.arange(n_t)
    width = 0.86 / n_m
    for i, m in enumerate(method_list):
        offset = (i - (n_m - 1)/2) * width
        vals = [per_m[m].get(ct, 0.0) for ct in cell_types]
        ax.bar(x + offset, vals, width=width*0.92,
               color=METHOD_COLORS.get(m,"#888"),
               edgecolor="white", linewidth=0.3,
               label=METHOD_DISPLAY.get(m, m))
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, fontsize=FS_TICK, rotation=15, ha="right")
    ax.set_ylabel("Mean silhouette", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Per-cell-type separation", fontsize=FS_TITLE, pad=4)
    ax.axhline(0, color="#888", lw=0.5, ls="--", alpha=0.7)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=FS_LEGEND - 0.5, frameon=False,
              ncol=2, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


def plot_composition_stack(ax, emb_data, meta, methods, celltype_palette,
                           batches_order=None):
    """Stacked composition bars, per method × condition (Fig 5d).

    Layout: 12 methods × 2 conditions = 24 narrow stacked bars. Method
    labels are 45° rotated with smaller font and tighter grouping to keep
    them readable; condition initials sit immediately above each bar
    instead of in a separate row (saves vertical space).
    """
    if batches_order is None:
        batches_order = sorted(meta[BATCH_COL].unique())
    cell_types = list(celltype_palette.keys())
    method_list = [m for m in methods if m in emb_data]
    n_b   = len(batches_order)
    # Tighter spacing for 12-method universe: narrower bars + smaller gap
    bar_w     = 0.32
    group_pad = 0.22
    method_x  = np.arange(len(method_list)) * (bar_w * n_b + group_pad)
    for i, m in enumerate(method_list):
        df = emb_data[m]
        for b_idx, batch in enumerate(batches_order):
            sub  = df[df[BATCH_COL] == batch]
            tot  = len(sub)
            if tot == 0:
                continue
            counts = sub[CELLTYPE_COL].value_counts()
            bottom = 0
            x_pos  = method_x[i] + b_idx * bar_w
            for ct in cell_types:
                frac = counts.get(ct, 0) / tot
                if frac <= 0:
                    continue
                ax.bar(x_pos, frac, bottom=bottom, width=bar_w*0.9,
                       color=celltype_palette.get(str(ct), "#888"),
                       edgecolor="white", linewidth=0.25)
                bottom += frac
            # Compact condition initial directly above each bar (1.02 = above 100%)
            ax.text(x_pos, 1.015, batch[0].upper(),
                    ha="center", va="bottom",
                    fontsize=FS_ANNOT - 0.5, color="#555")
    ax.set_xticks(method_x + bar_w * (n_b - 1) / 2)
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK - 0.5, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Cell-type fraction", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Composition per condition  (" +
                 "  /  ".join(f"{b[0].upper()}={b}" for b in batches_order) + ")",
                 fontsize=FS_TITLE, pad=4)
    ax.set_ylim(0, 1.08)   # extra headroom for the condition-initial row
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", pad=1)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)


def plot_per_batch_mixing(ax, emb_data, meta, methods=None, k=30,
                          subsample_n=8000, random_state=0):
    """Per-batch kNN-based mixing score heatmap (Fig 7)."""
    methods = methods or RANK_METHODS
    batches = sorted(meta[BATCH_COL].unique())
    method_list = [m for m in methods if m in emb_data]
    n_b, n_m = len(batches), len(method_list)
    M = np.full((n_b, n_m), np.nan)
    rng = np.random.default_rng(random_state)
    for j, m in enumerate(method_list):
        df = emb_data[m]
        if len(df) > subsample_n:
            idx = rng.choice(len(df), subsample_n, replace=False)
            df = df.iloc[idx]
        X = df[["UMAP1","UMAP2"]].values
        bvec = df[BATCH_COL].astype(str).values
        try:
            knn = NearestNeighbors(n_neighbors=min(k+1, len(X))).fit(X)
            _, ind = knn.kneighbors(X)
            ind = ind[:, 1:]
        except Exception:
            continue
        for i, batch in enumerate(batches):
            mask = bvec == batch
            if mask.sum() == 0:
                continue
            M[i, j] = (bvec[ind[mask]] != batch).mean()
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_b):
        for j in range(n_m):
            v = M[i, j]
            if np.isfinite(v):
                txt = "white" if v < 0.2 or v > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT - 0.5, color=txt)
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_yticks(range(n_b))
    ax.set_yticklabels(batches, fontsize=FS_TICK - 0.5)
    ax.set_title(f"Per-batch local mixing (k={k} NN)", fontsize=FS_TITLE, pad=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    if "BTCA" in method_list:
        c = method_list.index("BTCA")
        ax.add_patch(Rectangle((c - 0.5, -0.5), 1, n_b,
                               fill=False, edgecolor=METHOD_COLORS["BTCA"],
                               linewidth=0.9, clip_on=False))
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=22)
    cbar.set_label("Mixing", fontsize=FS_ANNOT, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  NEW analytical panels  (one chart family per figure, mostly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_ranking_strip(ax, batch_df, clust_df):
    """Fig 2.  Summed normalised score across 7 metrics; horizontal lollipop."""
    metric_specs = [
        (batch_df, "iLISI",        True),
        (batch_df, "cLISI_purity", True),
        (batch_df, "ASW_celltype", True),
        (batch_df, "ASW_batch_mixing", True),
        (batch_df, "kBET",         False),
        (clust_df, "ARI",          True),
        (clust_df, "NMI",          True),
    ]
    scores = {m: 0.0 for m in RANK_METHODS}
    used = 0
    for df, name, hib in metric_specs:
        col = _resolve(df, [name])
        if col is None:
            continue
        vals = np.array([_val(df, m, col) for m in RANK_METHODS], dtype=float)
        valid = np.isfinite(vals)
        if valid.sum() < 2:
            continue
        vmin, vmax = vals[valid].min(), vals[valid].max()
        if vmax > vmin:
            normed = (vals - vmin) / (vmax - vmin)
        else:
            normed = np.full_like(vals, 0.5)
        if not hib:
            normed = 1 - normed
        normed = np.where(valid, normed, 0)
        for m, v in zip(RANK_METHODS, normed):
            scores[m] += float(v)
        used += 1

    order = sorted(RANK_METHODS, key=lambda m: scores[m])  # ascending so best on top
    y = np.arange(len(order))
    max_val = max(scores.values()) if scores else used

    for i, m in enumerate(order):
        s = scores[m]
        c = METHOD_COLORS.get(m, "#888")
        ax.plot([0, s], [i, i], color=c, lw=1.5, alpha=0.75)
        ax.scatter(s, i, color=c, s=50, zorder=4,
                   edgecolors="white", linewidth=0.6)
        ax.text(s + max_val * 0.025, i, f"{s:.2f}",
                va="center", ha="left", fontsize=FS_ANNOT, color=c)

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in order],
                       fontsize=FS_TICK)
    ax.set_xlabel(f"Summed normalised score (max = {used})",
                  fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Overall method ranking", fontsize=FS_TITLE, pad=4)
    ax.set_xlim(0, max_val * 1.18 + 0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    if "BTCA" in order:
        sccat_idx = order.index("BTCA")
        ax.axhspan(sccat_idx - 0.45, sccat_idx + 0.45,
                   color=METHOD_COLORS["BTCA"], alpha=0.1, lw=0)


def plot_bs_confusion_matrix(ax, emb_data, meta, methods,
                             bs_types, n_clusters=None):
    """Fig 3.  Methods × batch-specific types — modal-cluster purity heatmap."""
    if n_clusters is None:
        n_clusters = meta[CELLTYPE_COL].nunique()
    method_list = [m for m in RANK_METHODS if m in emb_data]
    n_m, n_t = len(method_list), len(bs_types)
    M = np.full((n_m, n_t), np.nan)
    for i, m in enumerate(method_list):
        df = emb_data[m]
        X = df[["UMAP1","UMAP2"]].values
        if len(X) < n_clusters:
            continue
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            preds = km.fit_predict(X)
        except Exception:
            continue
        ct = df[CELLTYPE_COL].astype(str).values
        for j, bs in enumerate(bs_types):
            mask = ct == str(bs)
            if mask.sum() == 0:
                continue
            mode_c = pd.Series(preds[mask]).mode().iloc[0]
            cmask = preds == mode_c
            M[i, j] = (ct[cmask] == str(bs)).sum() / cmask.sum()

    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_m):
        for j in range(n_t):
            v = M[i, j]
            if np.isfinite(v):
                txt = "white" if v < 0.2 or v > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color=txt)
    ax.set_xticks(range(n_t))
    ax.set_xticklabels(bs_types, fontsize=FS_TICK, rotation=20, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK)
    ax.set_xlabel("Batch-specific cell type", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Cluster purity of batch-specific cell types",
                 fontsize=FS_TITLE, pad=4)
    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.add_patch(Rectangle((-0.5, i - 0.5), n_t, 1,
                               fill=False, edgecolor=METHOD_COLORS["BTCA"],
                               linewidth=0.9, clip_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=18)
    cbar.set_label("Purity", fontsize=FS_ANNOT, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)


def plot_silhouette_ridge(ax, emb_data, methods=None,
                          subsample_n=4000, random_state=0):
    """Fig 4.  Vertically stacked KDE ridges of per-cell silhouette per method."""
    methods = methods or RANK_METHODS
    rng = np.random.default_rng(random_state)
    method_data = {}
    for m in methods:
        if m not in emb_data:
            continue
        df = emb_data[m]
        if len(df) > subsample_n:
            idx = rng.choice(len(df), subsample_n, replace=False)
            df = df.iloc[idx]
        codes = df[CELLTYPE_COL].astype("category").cat.codes.values
        if len(np.unique(codes)) < 2:
            continue
        try:
            sil = silhouette_samples(df[["UMAP1","UMAP2"]].values, codes)
        except Exception:
            continue
        method_data[m] = sil
    if not method_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return
    ordered = [m for m in RANK_METHODS if m in method_data]
    n = len(ordered)
    x_grid  = np.linspace(-1.0, 1.0, 220)
    overlap = 0.55
    row_h   = 1 - overlap
    for i, m in enumerate(ordered):
        sil = method_data[m]
        try:
            kde = gaussian_kde(sil, bw_method=0.18)
            density = kde(x_grid)
        except Exception:
            continue
        density = density / (density.max() + 1e-9) * 0.85
        y_base = (n - 1 - i) * row_h
        color = METHOD_COLORS.get(m, "#888")
        ax.fill_between(x_grid, y_base, y_base + density,
                        color=color, alpha=0.70, linewidth=0,
                        zorder=n - i)
        ax.plot(x_grid, y_base + density, color="#333", lw=0.45,
                zorder=n - i + 0.1)
        med = float(np.median(sil))
        idx_m = int(np.argmin(np.abs(x_grid - med)))
        ax.plot([med, med], [y_base, y_base + density[idx_m]],
                color="white", lw=1.1, zorder=n - i + 0.2)
        ax.text(-1.05, y_base + 0.22,
                METHOD_DISPLAY.get(m, m),
                ha="right", va="center", fontsize=FS_TICK, color=color,
                fontweight="bold" if m == "BTCA" else "normal")
        ax.text(1.04, y_base + 0.22, f"med={med:+.2f}",
                ha="left", va="center", fontsize=FS_ANNOT, color="#444")
    ax.set_xlim(-1.30, 1.25)
    ax.set_ylim(-0.05, (n - 1) * row_h + 1.0)
    ax.set_xlabel("Per-cell silhouette  (cell-type labels)",
                  fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Silhouette distribution per method",
                 fontsize=FS_TITLE, pad=4)
    ax.set_yticks([])
    ax.axvline(0, color="#888", lw=0.5, ls="--", alpha=0.6)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)


def plot_hdc_schematic(ax, batch_palette, celltype_palette):
    """Fig 5.  Illustrated diagram: CD141 only in batch 0, CD1C only in batch 1."""
    b0 = batch_palette.get("0", "#4292C6")
    b1 = batch_palette.get("1", "#D94801")
    # Two soft-edged boxes
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.20), 0.42, 0.62,
        boxstyle="round,pad=0.02", linewidth=1.2,
        facecolor=b0, edgecolor=b0, alpha=0.16))
    ax.add_patch(FancyBboxPatch(
        (0.54, 0.20), 0.42, 0.62,
        boxstyle="round,pad=0.02", linewidth=1.2,
        facecolor=b1, edgecolor=b1, alpha=0.16))
    ax.text(0.04 + 0.21, 0.85, "Batch 0",
            fontsize=FS_LABEL + 0.5, fontweight="bold",
            ha="center", color=b0)
    ax.text(0.54 + 0.21, 0.85, "Batch 1",
            fontsize=FS_LABEL + 0.5, fontweight="bold",
            ha="center", color=b1)
    # Cell-type circles
    pos_b0 = {"pDC": (0.13, 0.62), "DoubleNeg": (0.25, 0.42), "CD141": (0.37, 0.62)}
    pos_b1 = {"pDC": (0.63, 0.62), "DoubleNeg": (0.75, 0.42), "CD1C":  (0.87, 0.62)}
    for ct, (cx, cy) in pos_b0.items():
        col = celltype_palette.get(ct, "#888")
        ax.scatter(cx, cy, s=520, c=col,
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(cx, cy - 0.10, ct, ha="center", va="top",
                fontsize=FS_ANNOT + 0.5, color="#222", zorder=6)
    for ct, (cx, cy) in pos_b1.items():
        col = celltype_palette.get(ct, "#888")
        ax.scatter(cx, cy, s=520, c=col,
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(cx, cy - 0.10, ct, ha="center", va="top",
                fontsize=FS_ANNOT + 0.5, color="#222", zorder=6)
    # Batch-specific markers
    for cx, lbl in [(0.37, "batch-specific\nto Batch 0"),
                    (0.87, "batch-specific\nto Batch 1")]:
        ax.annotate("", xy=(cx, 0.18), xytext=(cx, 0.07),
                    arrowprops=dict(arrowstyle="->", color="#222", lw=0.9))
        ax.text(cx, 0.05, lbl, ha="center", va="top",
                fontsize=FS_ANNOT + 0.5, color="#222",
                fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Dataset structure (schematic)",
                 fontsize=FS_TITLE, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_tradeoff_2d(ax, x_vals_dict, y_vals_dict,
                     xlabel, ylabel, title,
                     optimal_corner="upper-right"):
    """Fig 5 / Fig 7.  Generic 2-D scatter of methods showing a trade-off."""
    pts = []
    for m in METHODS:
        x = x_vals_dict.get(m, np.nan)
        y = y_vals_dict.get(m, np.nan)
        if np.isfinite(x) and np.isfinite(y):
            pts.append((m, x, y))
    if pts:
        ax.set_xlim(*_lim([p[1] for p in pts], pad=0.20))
        ax.set_ylim(*_lim([p[2] for p in pts], pad=0.20))
    for m, x, y in pts:
        is_sccat = (m == "BTCA")
        ax.scatter(x, y, s=60 if is_sccat else 38,
                   color=METHOD_COLORS.get(m, "#888"),
                   edgecolors="white", linewidth=0.6,
                   alpha=0.95, zorder=5 if is_sccat else 4,
                   clip_on=False)
    if pts:
        xl, yl = ax.get_xlim(), ax.get_ylim()
        xr, yr = xl[1] - xl[0], yl[1] - yl[0]
        for m, x, y in pts:
            dx = -0.022 * xr if x > xl[0] + 0.7 * xr else 0.022 * xr
            dy = -0.022 * yr if y > yl[0] + 0.7 * yr else 0.022 * yr
            ha = "right" if x > xl[0] + 0.7 * xr else "left"
            va = "top"   if y > yl[0] + 0.7 * yr else "bottom"
            ax.text(x + dx, y + dy, METHOD_DISPLAY.get(m, m),
                    fontsize=FS_ANNOT + (0.6 if m == "BTCA" else 0),
                    color=METHOD_COLORS.get(m, "#888"),
                    ha=ha, va=va,
                    fontweight="bold" if m == "BTCA" else "normal",
                    clip_on=False, zorder=6)
        # Optimal-corner annotation
        corner_xy = {
            "upper-right": (xl[1], yl[1], "right", "top"),
            "upper-left":  (xl[0], yl[1], "left",  "top"),
            "lower-right": (xl[1], yl[0], "right", "bottom"),
            "lower-left":  (xl[0], yl[0], "left",  "bottom"),
        }[optimal_corner]
        ax.text(corner_xy[0], corner_xy[1],
                "optimal\n",
                ha=corner_xy[2], va=corner_xy[3],
                fontsize=FS_ANNOT, color="#555",
                style="italic", clip_on=False)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.6)


def plot_density_overlay(ax, df, batch_col, batch_values, batch_palette, method,
                         subsample_n=3000, random_state=0):
    """Fig 6.  KDE contour overlay of each batch value on the same axes."""
    rng = np.random.default_rng(random_state)
    all_x = df["UMAP1"].values
    all_y = df["UMAP2"].values
    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()
    xpad = (xmax - xmin) * 0.05; ypad = (ymax - ymin) * 0.05
    xmin -= xpad; xmax += xpad
    ymin -= ypad; ymax += ypad
    # Light background of all cells
    ax.scatter(all_x, all_y, s=2, c="#DDDDDD", alpha=0.4,
               linewidths=0, rasterized=True)
    xgrid, ygrid = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]
    grid_pts = np.vstack([xgrid.ravel(), ygrid.ravel()])
    for bv in batch_values:
        sub = df[df[batch_col].astype(str) == str(bv)]
        if len(sub) < 30:
            continue
        if len(sub) > subsample_n:
            idx = rng.choice(len(sub), subsample_n, replace=False)
            sub = sub.iloc[idx]
        try:
            xy = np.vstack([sub["UMAP1"].values, sub["UMAP2"].values])
            kde = gaussian_kde(xy, bw_method=0.22)
            z = kde(grid_pts).reshape(xgrid.shape)
        except Exception:
            continue
        color = batch_palette.get(str(bv), "#888")
        ax.contour(xgrid, ygrid, z,
                   levels=4, colors=[color], linewidths=0.9, alpha=0.9)
        ax.contourf(xgrid, ygrid, z,
                    levels=4, colors=[color], alpha=0.16)
    title = METHOD_DISPLAY.get(method, method) + " — condition density"
    _nature_embed_style(ax, title, title_size=FS_TITLE + 1)
    # Add an explicit legend for the two conditions, with a white
    # background so the labels remain readable on top of dense contours.
    handles = [Line2D([0], [0], linewidth=2.0,
                       color=batch_palette.get(str(bv), "#888"),
                       label=str(bv)) for bv in batch_values]
    leg = ax.legend(handles=handles, loc="upper left",
              fontsize=FS_LEGEND - 0.5, frameon=True,
              facecolor="white", edgecolor="#BBB",
              framealpha=0.92, handlelength=1.4,
              handletextpad=0.5, borderpad=0.4)
    # Bring legend to top so it's never hidden by contour fills
    leg.set_zorder(20)


def plot_paired_dotplot(ax, emb_data, meta, methods, celltype_palette,
                        batches_pair):
    """Fig 6.  Per-cell-type, per-method, two-dot paired plot showing how
    each cell type's proportion shifts between the two batches."""
    cell_types = list(celltype_palette.keys())
    method_list = [m for m in methods if m in emb_data]
    n_ct = len(cell_types)
    n_m  = len(method_list)
    if n_ct == 0 or n_m == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center"); return
    ct_x = np.arange(n_ct)
    dot_w = 0.86 / n_m
    for i, m in enumerate(method_list):
        df = emb_data[m]
        offset = (i - (n_m - 1) / 2) * dot_w
        for j, ct in enumerate(cell_types):
            mask_ct = df[CELLTYPE_COL].astype(str) == str(ct)
            tot = mask_ct.sum()
            if tot == 0:
                continue
            # Fraction of this cell type within each batch (out of that batch's
            # total cells – this captures relative density of the cell type)
            for b_val, marker in zip(batches_pair, ("o", "s")):
                pass  # actual logic below
            x_pos = ct_x[j] + offset
            # Use within-batch fraction of this cell type
            n_b0_total = (df[BATCH_COL].astype(str) == str(batches_pair[0])).sum()
            n_b1_total = (df[BATCH_COL].astype(str) == str(batches_pair[1])).sum()
            n_ct_b0 = ((df[BATCH_COL].astype(str) == str(batches_pair[0])) & mask_ct).sum()
            n_ct_b1 = ((df[BATCH_COL].astype(str) == str(batches_pair[1])) & mask_ct).sum()
            frac_b0 = n_ct_b0 / max(n_b0_total, 1)
            frac_b1 = n_ct_b1 / max(n_b1_total, 1)
            color = METHOD_COLORS.get(m, "#888")
            ax.plot([x_pos, x_pos], [frac_b0, frac_b1],
                    color=color, lw=0.7, alpha=0.6, zorder=2)
            ax.scatter(x_pos, frac_b0, s=18, color=color, marker="o",
                       edgecolors="white", linewidth=0.3, zorder=4)
            ax.scatter(x_pos, frac_b1, s=18, color=color, marker="s",
                       edgecolors="white", linewidth=0.3, zorder=4)
    handles = [
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor="#666", markeredgecolor="white",
               markersize=4, label=f"{batches_pair[0]} (○)"),
        Line2D([0],[0], marker="s", linestyle="None",
               markerfacecolor="#666", markeredgecolor="white",
               markersize=4, label=f"{batches_pair[1]} (□)"),
    ]
    ax.legend(handles=handles, fontsize=FS_LEGEND - 1, frameon=False,
              loc="upper right", ncol=2)
    method_handles = [
        Line2D([0],[0], marker="o", linestyle="None",
               markerfacecolor=METHOD_COLORS.get(m, "#888"),
               markeredgecolor="white",
               markersize=4, label=METHOD_DISPLAY.get(m, m))
        for m in method_list
    ]
    leg2 = ax.legend(handles=method_handles, fontsize=FS_LEGEND - 1, frameon=False,
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     borderaxespad=0, ncol=1, title="Method",
                     title_fontsize=FS_LEGEND - 0.5)
    # Re-add the first legend (matplotlib drops the previous one)
    ax.add_artist(leg2)
    h2 = ax.legend(handles=handles, fontsize=FS_LEGEND - 1, frameon=False,
                   loc="upper right", ncol=2)
    ax.set_xticks(ct_x)
    ax.set_xticklabels(cell_types, fontsize=FS_TICK, rotation=20, ha="right")
    ax.set_ylabel("Within-batch fraction", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Per-cell-type proportion: condition pair",
                 fontsize=FS_TITLE, pad=4)
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.axhline(0, color="#888", lw=0.4, ls="--", alpha=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# 8b-new.  Additional top-tier panels (new Fig 2/3/5/6 extensions)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ibfull_bar(ax, dataset_key: str, title: str) -> None:
    """Ranked IB bar from multi-seed phase-2 summary. Used in Fig 2g and Fig 5g."""
    csv = BASE_DIR / "experiments" / "results" / "phase2" / "summary_IBfull_mean_std.csv"
    if not csv.exists():
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                fontsize=FS_TICK, color="#888")
        ax.set_title(title, fontsize=FS_TITLE, pad=4)
        return
    df = pd.read_csv(csv)
    sub = df[df["dataset"] == dataset_key].copy()
    if sub.empty:
        ax.text(0.5, 0.5, f"No data: {dataset_key}", transform=ax.transAxes,
                ha="center", fontsize=FS_TICK, color="#888")
        return
    sub = sub.sort_values("IB_full_mean", ascending=True)
    methods = sub["method"].tolist()
    means   = sub["IB_full_mean"].values
    stds    = sub["IB_full_std"].fillna(0).values
    colors  = [METHOD_COLORS.get(m, "#aaa") for m in methods]
    labels  = [METHOD_DISPLAY.get(m, m) for m in methods]
    y = np.arange(len(methods))
    ax.barh(y, means, xerr=stds, color=colors, height=0.62,
            error_kw=dict(elinewidth=0.7, capsize=2, capthick=0.7, ecolor="#555"),
            linewidth=0)
    for i, (m, v) in enumerate(zip(methods, means)):
        # Offset each label past its OWN error-bar cap (not the global max std),
        # so high-variance methods like scVI don't collide with their cap.
        ax.text(v + stds[i] + 0.015, i, f"{v:.3f}",
                va="center", ha="left", fontsize=FS_ANNOT,
                fontweight="bold" if m == "BTCA" else "normal",
                color=METHOD_COLORS.get(m, "#555"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("Integration Balance (IB, range 0–1, higher = better)",
                  fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.set_xlim(0, min(1.0, means.max() + 0.15))
    ax.xaxis.set_major_locator(MaxNLocator(4, prune="lower"))
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _annotate_scatter(ax, pts, fontsize=FS_ANNOT, avoid_objs=None):
    """
    Annotate scatter points with overlap avoidance.
    pts: list of (x, y, label, color, bold)
    avoid_objs: optional list of existing Text artists that labels must not cover.
    Uses adjustText if available; falls back to position-aware offsets.
    """
    try:
        from adjustText import adjust_text
        texts = []
        for (x, y, lbl, c, bold) in pts:
            t = ax.text(x, y, lbl, fontsize=fontsize, color=c,
                        fontweight="bold" if bold else "normal")
            texts.append(t)
        adj_kw = dict(
            ax=ax,
            expand=(1.4, 1.6),
            force_text=(0.5, 0.7),
            force_points=(0.2, 0.3),
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
        )
        if avoid_objs:
            try:
                adj_kw["add_objects"] = avoid_objs
            except Exception:
                pass
        adjust_text(texts, **adj_kw)
    except ImportError:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2
        for (x, y, lbl, c, bold) in pts:
            dx = 4 if x <= xmid else -(4 + len(lbl) * 4)
            dy = 3 if y <= ymid else -10
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=fontsize, color=c,
                        fontweight="bold" if bold else "normal")


def _plot_oci_ilisi_scatter(ax, oci_dict: dict, bdf, title: str) -> None:
    """OCI (y) vs iLISI (x) scatter for Sim 4 — shows scCAT avoids overcorrection
    while maintaining batch integration. Used in Fig 2h."""
    ilisi_col = _resolve(bdf, ["iLISI", "iLISI_mean"])
    if ilisi_col is None:
        ax.text(0.5, 0.5, "No iLISI", transform=ax.transAxes,
                ha="center", fontsize=FS_TICK)
        ax.set_title(title, fontsize=FS_TITLE, pad=4)
        return
    pts = []
    for m in RANK_METHODS:
        if m not in oci_dict:
            continue
        x = _val(bdf, m, ilisi_col)
        y = oci_dict[m]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        c   = METHOD_COLORS.get(m, "#aaa")
        lbl = METHOD_DISPLAY.get(m, m)
        ax.scatter(x, y, color=c, s=36, zorder=3, edgecolors="white",
                   linewidths=0.4)
        pts.append((x, y, lbl, c, m == "BTCA"))
    ax.set_xlabel("iLISI (batch mixing, range 0–1 ↑)", fontsize=FS_LABEL)
    ax.set_ylabel("OCI (overcorrection index, range 0–1 ↓)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    _annotate_scatter(ax, pts, fontsize=FS_ANNOT)


def _plot_1nn_sep_bar(ax, title: str) -> None:
    """1-NN same-type consistency per method (HDC resolution-free diagnostic).
    Used in Fig 3g. Source: results/phase3/hdc_dc/separability_diagnostic.csv."""
    csv = BASE_DIR / "experiments" / "results" / "phase3" / "hdc_dc" / \
          "separability_diagnostic.csv"
    if not csv.exists():
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                fontsize=FS_TICK, color="#888")
        ax.set_title(title, fontsize=FS_TITLE, pad=4)
        return
    sep = pd.read_csv(csv).sort_values("loo_1nn", ascending=True)
    methods = sep["method"].tolist()
    vals    = sep["loo_1nn"].values
    colors  = [METHOD_COLORS.get(m, "#aaa") for m in methods]
    labels  = [METHOD_DISPLAY.get(m, m) for m in methods]
    y = np.arange(len(methods))
    bars = ax.barh(y, vals, color=colors, height=0.60, linewidth=0)
    # Shaded band for "well-behaved" group (> 0.80)
    ax.axvspan(0.80, 1.12, color="#DDEEDD", alpha=0.35, zorder=0)
    ax.text(0.81, -0.7, "well-behaved group", fontsize=FS_ANNOT - 0.5,
            color="#448844", va="bottom", ha="left", style="italic")
    # Threshold line separating over-correctors from the rest
    ax.axvline(0.80, color="#cc0000", lw=0.8, ls="--", alpha=0.7)
    ax.text(0.795, len(methods) - 0.3, "over-\ncorrector",
            va="top", ha="right", fontsize=FS_ANNOT - 0.5, color="#cc0000")
    for i, (m, v) in enumerate(zip(methods, vals)):
        ax.text(v + 0.005, i, f"{v:.2f}",
                va="center", ha="left", fontsize=FS_ANNOT,
                fontweight="bold" if m == "BTCA" else "normal",
                color=METHOD_COLORS.get(m, "#555"))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel("1-NN same-type consistency ↑", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_locator(MaxNLocator(4, prune="lower"))
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _plot_ilisi_1nn_scatter(ax, bdf, title: str) -> None:
    """iLISI (x) vs 1-NN same-type consistency (y) — batch-mixing vs bio-preservation
    trade-off for HDC. Used in Fig 3h."""
    csv = BASE_DIR / "experiments" / "results" / "phase3" / "hdc_dc" / \
          "separability_diagnostic.csv"
    if not csv.exists():
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                fontsize=FS_TICK); return
    sep = pd.read_csv(csv).set_index("method")
    ilisi_col = _resolve(bdf, ["iLISI", "iLISI_mean"])
    for m in RANK_METHODS:
        if m not in sep.index:
            continue
        y = sep.loc[m, "loo_1nn"]
        x = _val(bdf, m, ilisi_col) if ilisi_col else np.nan
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        c   = METHOD_COLORS.get(m, "#aaa")
        lbl = METHOD_DISPLAY.get(m, m)
        ax.scatter(x, y, color=c, s=36, zorder=3,
                   edgecolors="white", linewidths=0.4)
        label_offsets = {
            "INSCT_Unsupervised": (5, 7),
            "scANVI": (-22, -8),
            "scVI": (7, 7),
            "BTCA": (5, 5),
        }
        dx, dy = label_offsets.get(m, (3, 2))
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=FS_ANNOT,
                    ha="right" if dx < 0 else "left",
                    color=c, fontweight="bold" if m == "BTCA" else "normal",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.35,
                                    alpha=0.65, shrinkA=1, shrinkB=2))
    ax.axhline(0.80, color="#cc0000", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("iLISI (batch mixing ↑)", fontsize=FS_LABEL)
    ax.set_ylabel("1-NN consistency ↑", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _plot_de_concordance_bar(ax, title: str) -> None:
    """Per-cell-type IFN-β DE concordance bars (4 methods × 8 cell types).
    Used in Fig 5h. Source: results/phase3/pbmc_ifn/per_type_concordance.csv."""
    csv = BASE_DIR / "experiments" / "results" / "phase3" / "pbmc_ifn" / \
          "per_type_concordance.csv"
    if not csv.exists():
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                fontsize=FS_TICK, color="#888")
        ax.set_title(title, fontsize=FS_TITLE, pad=4)
        return
    raw = pd.read_csv(csv, index_col=0)
    # Canonical column names from phase3 output
    methods_plot = ["scCAT", "Harmony", "Scanorama", "INSCT_Unsupervised"]
    method_map   = {"scCAT": "BTCA", "INSCT_Unsupervised": "INSCT_Unsupervised"}
    display_cols = ["scCAT", "Harmony", "Scanorama", "INSCT"]
    ct_order = list(raw.index)
    x    = np.arange(len(ct_order))
    w    = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w
    bar_colors = [METHOD_COLORS.get("BTCA", "#2D9E2D"),
                  METHOD_COLORS.get("Harmony", "#F39C12"),
                  METHOD_COLORS.get("Scanorama", "#C474C4"),
                  METHOD_COLORS.get("INSCT_Unsupervised", "#5B8DD9")]
    ZERO_THRESH = 0.01  # values below this are treated as "not computable"
    for j, (col, disp, col_c) in enumerate(zip(methods_plot, display_cols, bar_colors)):
        if col in raw.columns:
            vals = raw[col].values.copy()
        else:
            vals = np.zeros(len(ct_order))
        for k_idx, v in enumerate(vals):
            bx = x[k_idx] + offsets[j]
            if v < ZERO_THRESH:
                # Hatched placeholder: indicates concordance could not be computed
                ax.bar(bx, 0.03, width=w, color="#CCCCCC",
                       linewidth=0.5, edgecolor="#999", alpha=0.7,
                       hatch="////")
                ax.text(bx, 0.05, "–", ha="center", va="bottom",
                        fontsize=FS_ANNOT - 1, color="#999")
            else:
                ax.bar(bx, v, width=w, color=col_c,
                       linewidth=0, alpha=0.88,
                       label=disp if k_idx == 0 else "")
    # Re-draw legend from a single labelled bar per method
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=d, alpha=0.88)
               for c, d in zip(bar_colors, display_cols)]
    handles.append(mpatches.Patch(facecolor="#CCCCCC", edgecolor="#999",
                                   hatch="////", label="not computable",
                                   alpha=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(ct_order, rotation=35, ha="right", fontsize=FS_TICK)
    ax.set_ylabel("DE concordance (ρ) ↑", fontsize=FS_LABEL)
    ax.set_ylim(0, 1.10)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    # Legend moved ABOVE the plot (single horizontal row) so it never
    # overlaps the bars; title raised to sit above the legend.
    ax.legend(handles=handles, fontsize=FS_ANNOT - 0.5, ncol=5,
              frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.9, handletextpad=0.35, handlelength=1.2)
    ax.set_title(title, fontsize=FS_TITLE, pad=16)
    ax.axhline(0, color="#888", lw=0.4, ls="--", alpha=0.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _plot_wilcoxon_summary(ax, title: str) -> None:
    """Compact Wilcoxon BH-corrected p-value summary (scCAT vs all methods).
    Used in Fig 6h. Source: wilcoxon_scCAT_IBfull.csv."""
    csv = BASE_DIR / "experiments" / "results" / "phase2" / "wilcoxon_scCAT_IBfull.csv"
    if not csv.exists():
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", fontsize=FS_TICK, color="#888")
        ax.set_title(title, fontsize=FS_TITLE, pad=4)
        return
    wil = pd.read_csv(csv).sort_values("p_bh")
    methods = wil["opponent"].tolist()
    pvals   = wil["p_bh"].values
    diffs   = wil["median_diff"].values
    labels  = [METHOD_DISPLAY.get(m, m) for m in methods]
    y = np.arange(len(methods))
    # Bar of -log10(p_bh); grey for non-significant (p_bh >= 0.05)
    log_p = -np.log10(np.clip(pvals, 1e-6, 1.0))
    bar_colors = []
    for p, d in zip(pvals, diffs):
        if p >= 0.05:
            bar_colors.append("#AAAAAA")          # grey = not significant
        elif d > 0:
            bar_colors.append("#2D9E2D")          # green = scCAT better (sig)
        else:
            bar_colors.append("#C47400")          # orange = scCAT worse (sig)
    ax.barh(y, log_p, color=bar_colors, height=0.60, linewidth=0, alpha=0.85)
    ax.axvline(-np.log10(0.05), color="#888", lw=0.8, ls="--")
    ax.text(-np.log10(0.05) + 0.05, len(methods) - 0.4,
            "p = 0.05", fontsize=FS_ANNOT - 0.5, color="#555")
    for i, (m, p, d) in enumerate(zip(methods, pvals, diffs)):
        if p >= 0.05:
            sig = "n.s."
            sig_color = "#777"
        elif p > 0.01:
            sig = "*"
            sig_color = "#333"
        else:
            sig = "**"
            sig_color = "#111"
        ax.text(log_p[i] + 0.05, i, sig, va="center", ha="left",
                fontsize=FS_ANNOT, color=sig_color,
                fontweight="bold" if sig != "n.s." else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.set_xlabel(r"$-\log_{10}(p_{\mathrm{BH}})$", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    # Bars are sorted shortest (n.s.) at top → the upper-right quadrant is
    # empty, so anchor the legend there instead of lower-right where it
    # previously overlapped the long significant bars.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2D9E2D", label="scCAT better (sig.)"),
                        Patch(color="#C47400", label="scCAT worse (sig.)"),
                        Patch(color="#AAAAAA", label="n.s.")],
              fontsize=FS_ANNOT - 0.5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.97), handlelength=1.2,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)


# ─────────────────────────────────────────────────────────────────────────────
# 8b.  Cross-dataset analytical panels (for the 3 main "claim" figures)
# ─────────────────────────────────────────────────────────────────────────────

def _composite_score(bdf, cdf, method):
    """Mean normalised score for one (method, dataset) across 7 metrics."""
    metric_specs = [
        (bdf, "iLISI",            True),
        (bdf, "cLISI_purity",     True),
        (bdf, "ASW_celltype",     True),
        (bdf, "ASW_batch_mixing", True),
        (bdf, "kBET",             False),
        (cdf, "ARI",              True),
        (cdf, "NMI",              True),
    ]
    parts = []
    for df, name, hib in metric_specs:
        col = _resolve(df, [name])
        if col is None:
            continue
        all_vals = np.array([_val(df, m, col) for m in RANK_METHODS], dtype=float)
        v = _val(df, method, col)
        valid = np.isfinite(all_vals)
        if valid.sum() < 2 or not np.isfinite(v):
            continue
        vmin, vmax = all_vals[valid].min(), all_vals[valid].max()
        if vmax > vmin:
            normed = (v - vmin) / (vmax - vmin)
        else:
            normed = 0.5
        if not hib:
            normed = 1 - normed
        parts.append(normed)
    return float(np.mean(parts)) if parts else np.nan


def plot_master_scorecard(ax, all_metrics, dataset_keys=None):
    """Methods × datasets heatmap of composite scores. Used in Fig 2 b."""
    dataset_keys = dataset_keys or ALL_DATASETS
    n_m, n_d = len(RANK_METHODS), len(dataset_keys)
    M = np.full((n_m, n_d), np.nan)
    for j, dk in enumerate(dataset_keys):
        bdf = all_metrics[dk]["batch"]
        cdf = all_metrics[dk]["cluster"]
        for i, m in enumerate(RANK_METHODS):
            M[i, j] = _composite_score(bdf, cdf, m)

    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_m):
        for j in range(n_d):
            v = M[i, j]
            if np.isfinite(v):
                txt = "white" if v < 0.20 or v > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color=txt)
    ax.set_xticks(range(n_d))
    ax.set_xticklabels([DATASET_CONFIGS[dk]["label"] for dk in dataset_keys],
                       fontsize=FS_TICK, rotation=22, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in RANK_METHODS],
                       fontsize=FS_TICK)
    ax.set_title("Composite benchmark score per dataset", fontsize=FS_TITLE, pad=4)

    if "BTCA" in RANK_METHODS:
        i = RANK_METHODS.index("BTCA")
        ax.add_patch(Rectangle((-0.5, i - 0.5), n_d, 1,
                               fill=False, edgecolor=METHOD_COLORS["BTCA"],
                               linewidth=0.9, clip_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=18)
    cbar.set_label("Score (↑)", fontsize=FS_ANNOT, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)


def plot_aggregated_ranking_lollipop(ax, all_metrics, dataset_keys=None):
    """Average composite score across all datasets per method. Used in Fig 2 d."""
    dataset_keys = dataset_keys or ALL_DATASETS
    means = {}
    for m in RANK_METHODS:
        scores = []
        for dk in dataset_keys:
            s = _composite_score(all_metrics[dk]["batch"],
                                  all_metrics[dk]["cluster"], m)
            if np.isfinite(s):
                scores.append(s)
        if scores:
            means[m] = float(np.mean(scores))

    order = sorted(RANK_METHODS, key=lambda m: means.get(m, 0))
    y = np.arange(len(order))
    max_val = max(means.values()) if means else 1.0
    for i, m in enumerate(order):
        s = means.get(m, 0)
        c = METHOD_COLORS.get(m, "#888")
        ax.plot([0, s], [i, i], color=c, lw=1.6, alpha=0.78)
        ax.scatter(s, i, color=c, s=60, zorder=4,
                   edgecolors="white", linewidth=0.7)
        ax.text(s + max_val * 0.015, i, f"{s:.2f}",
                va="center", ha="left",
                fontsize=FS_ANNOT + (0.8 if m == "BTCA" else 0),
                fontweight="bold" if m == "BTCA" else "normal",
                color=c)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in order], fontsize=FS_TICK)
    for tick, m in zip(ax.get_yticklabels(), order):
        if m == "BTCA":
            tick.set_fontweight("bold")
    ax.set_xlabel(f"Mean composite score across {len(dataset_keys)} datasets",
                  fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Overall benchmark ranking", fontsize=FS_TITLE, pad=4)
    ax.set_xlim(0, max_val * 1.18 + 0.04)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    if "BTCA" in order:
        idx = order.index("BTCA")
        ax.axhspan(idx - 0.45, idx + 0.45,
                   color=METHOD_COLORS["BTCA"], alpha=0.10, lw=0)


def plot_dataset_table(ax, dataset_keys=None):
    """Compact text-table of dataset characteristics. Used in Fig 2 a."""
    dataset_keys = dataset_keys or ALL_DATASETS
    headers = ["Dataset", "Cells", "Batches", "Types", "Challenge"]
    challenges = {
        "data2_scenario1": "Balanced large-scale",
        "data2_scenario2": "Batch-specific cells",
        "Sc_mixology":     "Cross-platform",
        "HDC":             "Rare batch-specific",
        "PBMC":            "Condition (IFN-β)",
        "Lung":            "16-batch scale",
    }
    rows = []
    for dk in dataset_keys:
        cfg = DATASET_CONFIGS[dk]
        meta = read_metadata(BASE_DIR / cfg["meta_csv"])
        rows.append([
            cfg["label"],
            f"{len(meta):,}",
            str(meta[BATCH_COL].nunique()),
            str(meta[CELLTYPE_COL].nunique()),
            challenges.get(dk, "—"),
        ])

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n_cols = len(headers)
    col_edges = np.array([0.02, 0.34, 0.46, 0.58, 0.68, 0.98])
    col_centers = (col_edges[:-1] + col_edges[1:]) / 2

    n_rows = len(rows)
    top_y = 0.90; bottom_y = 0.06
    row_edges = np.linspace(top_y, bottom_y, n_rows + 1)
    header_y = top_y + 0.045
    row_centers = (row_edges[:-1] + row_edges[1:]) / 2

    # Header row with subtle background bar
    ax.add_patch(Rectangle((col_edges[0], top_y), col_edges[-1] - col_edges[0],
                            0.075, facecolor="#EEEEEE", edgecolor="none"))
    for j, h in enumerate(headers):
        ha = "left" if j == 0 else "center"
        x_pos = col_edges[0] + 0.01 if j == 0 else col_centers[j]
        ax.text(x_pos, header_y, h, fontsize=FS_TICK + 0.6,
                fontweight="bold", ha=ha, va="center", color="#222")
    ax.plot(col_edges[[0, -1]], [top_y, top_y], color="#888", lw=0.5)

    for i, row in enumerate(rows):
        y_pos = row_centers[i]
        for j, val in enumerate(row):
            ha = "left" if j == 0 else "center"
            x_pos = col_edges[0] + 0.01 if j == 0 else col_centers[j]
            ax.text(x_pos, y_pos, val,
                    fontsize=FS_TICK - 0.3, ha=ha, va="center", color="#333")

    ax.plot(col_edges[[0, -1]], [bottom_y, bottom_y], color="#CCC", lw=0.4)
    ax.set_title("Datasets evaluated in this study", fontsize=FS_TITLE, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_combined_confusion_matrix(ax, emb_sim4, meta_sim4, bs_sim4,
                                    emb_hdc, meta_hdc, bs_hdc):
    """Methods × (Sim4 + HDC) batch-specific cell types. Used in Fig 3 d."""
    n_cl_sim4 = meta_sim4[CELLTYPE_COL].nunique()
    n_cl_hdc  = meta_hdc[CELLTYPE_COL].nunique()
    method_list = [m for m in RANK_METHODS if m in emb_sim4 and m in emb_hdc]
    n_m = len(method_list)
    cols = list(bs_sim4) + list(bs_hdc)
    n_g = len(cols)
    M = np.full((n_m, n_g), np.nan)

    def _purity(df, n_clusters, bs):
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            preds = km.fit_predict(df[["UMAP1","UMAP2"]].values)
            ct = df[CELLTYPE_COL].astype(str).values
            mask = ct == str(bs)
            if mask.sum() == 0:
                return np.nan
            mode_c = pd.Series(preds[mask]).mode().iloc[0]
            cmask = preds == mode_c
            return float((ct[cmask] == str(bs)).sum() / cmask.sum())
        except Exception:
            return np.nan

    for i, m in enumerate(method_list):
        for j, bs in enumerate(bs_sim4):
            M[i, j] = _purity(emb_sim4[m], n_cl_sim4, bs)
        for j, bs in enumerate(bs_hdc):
            M[i, len(bs_sim4) + j] = _purity(emb_hdc[m], n_cl_hdc, bs)

    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_m):
        for j in range(n_g):
            v = M[i, j]
            if np.isfinite(v):
                txt = "white" if v < 0.20 or v > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color=txt)

    sep_x = len(bs_sim4) - 0.5
    ax.axvline(sep_x, color="#000", lw=1.4, ymin=0, ymax=1)

    ax.set_xticks(range(n_g))
    ax.set_xticklabels(cols, fontsize=FS_TICK, rotation=25, ha="right")
    sim4_mid = (len(bs_sim4) - 1) / 2
    hdc_mid  = len(bs_sim4) + (len(bs_hdc) - 1) / 2
    ax.text(sim4_mid, -0.62, "Simulated 4", ha="center", va="bottom",
            fontsize=FS_TICK + 0.5, fontweight="bold", color="#222",
            transform=ax.transData)
    ax.text(hdc_mid,  -0.62, "HDC",         ha="center", va="bottom",
            fontsize=FS_TICK + 0.5, fontweight="bold", color="#222",
            transform=ax.transData)

    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK)
    ax.set_title("Cluster purity of batch-specific cell types (combined evidence)",
                 fontsize=FS_TITLE, pad=4)

    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.add_patch(Rectangle((-0.5, i - 0.5), n_g, 1,
                               fill=False, edgecolor=METHOD_COLORS["BTCA"],
                               linewidth=0.9, clip_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=18)
    cbar.set_label("Purity", fontsize=FS_ANNOT, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)


def plot_paired_preservation_bars(ax, emb_sim4, meta_sim4, bs_sim4,
                                    emb_hdc, meta_hdc, bs_hdc):
    """Per-method paired bars: Sim4 vs HDC batch-specific preservation. Fig 3 e."""
    n_cl_sim4 = meta_sim4[CELLTYPE_COL].nunique()
    n_cl_hdc  = meta_hdc[CELLTYPE_COL].nunique()
    method_list = [m for m in RANK_METHODS if m in emb_sim4 and m in emb_hdc]

    def _mean_purity(emb, n_clusters, bs_types):
        purs = []
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            preds = km.fit_predict(emb[["UMAP1","UMAP2"]].values)
            ct = emb[CELLTYPE_COL].astype(str).values
            for bs in bs_types:
                mask = ct == str(bs)
                if mask.sum() == 0:
                    continue
                mode_c = pd.Series(preds[mask]).mode().iloc[0]
                cmask = preds == mode_c
                purs.append((ct[cmask] == str(bs)).sum() / cmask.sum())
        except Exception:
            pass
        return float(np.mean(purs)) if purs else np.nan

    sim4_scores, hdc_scores = {}, {}
    for m in method_list:
        sim4_scores[m] = _mean_purity(emb_sim4[m], n_cl_sim4, bs_sim4)
        hdc_scores[m]  = _mean_purity(emb_hdc[m],  n_cl_hdc,  bs_hdc)

    x = np.arange(len(method_list))
    w = 0.38
    for i, m in enumerate(method_list):
        col = METHOD_COLORS.get(m, "#888")
        s_val = sim4_scores.get(m, np.nan)
        h_val = hdc_scores.get(m, np.nan)
        if np.isfinite(s_val):
            ax.bar(x[i] - w/2, s_val, width=w*0.92, color=col, alpha=0.95,
                   edgecolor="white", linewidth=0.4)
            ax.text(x[i] - w/2, s_val + 0.015, f"{s_val:.2f}",
                    ha="center", va="bottom",
                    fontsize=FS_ANNOT - 0.5, color=col)
        if np.isfinite(h_val):
            ax.bar(x[i] + w/2, h_val, width=w*0.92, color=col, alpha=0.55,
                   edgecolor="white", linewidth=0.4, hatch="///")
            ax.text(x[i] + w/2, h_val + 0.015, f"{h_val:.2f}",
                    ha="center", va="bottom",
                    fontsize=FS_ANNOT - 0.5, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_ylabel("Cluster purity (↑)", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Batch-specific cell preservation: cross-dataset evidence",
                 fontsize=FS_TITLE, pad=4)
    ax.set_ylim(0, 1.16)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="#666", markeredgecolor="white",
               markersize=6, label="Simulated 4"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="none", markeredgecolor="#666",
               markersize=6, label="HDC (hatched)"),
    ]
    ax.legend(handles=handles, fontsize=FS_LEGEND - 0.5, frameon=False,
              loc="upper left", handletextpad=0.4)
    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.axvspan(i - 0.5, i + 0.5,
                   color=METHOD_COLORS["BTCA"], alpha=0.08, lw=0)


def plot_cross_dataset_tradeoff(ax, all_metrics, dataset_keys,
                                 dataset_markers=None):
    """Trade-off scatter where each dataset uses a different marker. Fig 4 d."""
    dataset_markers = dataset_markers or ["o", "s", "^", "D", "P", "v"]

    all_x_vals, all_y_vals = [], []
    sccat_pts = []

    for k_idx, dk in enumerate(dataset_keys):
        bdf = all_metrics[dk]["batch"]
        cdf = all_metrics[dk]["cluster"]
        kbet_col = _resolve(bdf, ["kBET"])
        ari_col  = _resolve(cdf, ["ARI"])
        if kbet_col is None or ari_col is None:
            continue
        for m in METHODS:
            kbet = _val(bdf, m, kbet_col)
            ari  = _val(cdf, m, ari_col)
            if not np.isfinite(kbet) or not np.isfinite(ari):
                continue
            x = 1 - kbet; y = ari
            all_x_vals.append(x); all_y_vals.append(y)
            marker = dataset_markers[k_idx]
            col = METHOD_COLORS.get(m, "#888")
            is_sccat = (m == "BTCA")
            ax.scatter(x, y, marker=marker,
                       s=85 if is_sccat else 32,
                       color=col, alpha=0.95,
                       edgecolors="white", linewidth=0.55,
                       zorder=6 if is_sccat else 3,
                       clip_on=False)
            if is_sccat:
                sccat_pts.append((x, y, dk))

    if sccat_pts:
        sccat_pts_sorted = sorted(sccat_pts, key=lambda p: p[0])
        xs = [p[0] for p in sccat_pts_sorted]
        ys = [p[1] for p in sccat_pts_sorted]
        ax.plot(xs, ys, color=METHOD_COLORS["BTCA"],
                lw=1.0, linestyle="--", alpha=0.7, zorder=5)

    if all_x_vals:
        ax.set_xlim(*_lim(all_x_vals, pad=0.18))
        ax.set_ylim(*_lim(all_y_vals, pad=0.18))

    shape_handles = [
        Line2D([0], [0], marker=dataset_markers[i], linestyle="None",
               markerfacecolor="#888", markeredgecolor="white",
               markersize=6,
               label=DATASET_CONFIGS[dk]["label"])
        for i, dk in enumerate(dataset_keys)
    ]
    shape_leg = ax.legend(handles=shape_handles, fontsize=FS_LEGEND - 0.5,
                           frameon=False, loc="lower right",
                           title="Dataset", title_fontsize=FS_LEGEND,
                           handletextpad=0.3)
    ax.add_artist(shape_leg)
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=METHOD_COLORS.get(m, "#888"),
               markeredgecolor="white", markersize=5,
               label=METHOD_DISPLAY.get(m, m))
        for m in RANK_METHODS
    ]
    ax.legend(handles=method_handles, fontsize=FS_LEGEND - 0.5,
              frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              title="Method", title_fontsize=FS_LEGEND,
              handletextpad=0.3)

    ax.set_xlabel("Batch mixing  (1 − kBET, ↑)", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("Biology preservation  (ARI, ↑)", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Cross-dataset trade-off (scCAT dashed = consistency)",
                 fontsize=FS_TITLE, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.6)


# ─────────────────────────────────────────────────────────────────────────────
# 8c.  Mechanism-driven metrics  (introduced in the "small but precise" redesign)
#
# These three metrics directly probe scCAT's mechanism:
#   • OCI (Overcorrection Index)        — how many batch-specific cells get
#                                          absorbed into clusters dominated by
#                                          OTHER cell types (↓ better)
#   • BSRS (Batch-Specific Retention)   — silhouette of batch-specific cells
#                                          against the full label space (↑ better)
#   • Integration Balance               — sqrt(batch_removal × bio_conservation),
#                                          rewards methods that simultaneously
#                                          mix batches AND preserve biology
# ─────────────────────────────────────────────────────────────────────────────

from collections import Counter as _Counter


def compute_oci(emb_df, bs_types, n_clusters):
    """Overcorrection Index — fraction of batch-specific cells that end up in
    a k-means cluster whose dominant label is NOT that bs cell type."""
    X = emb_df[["UMAP1","UMAP2"]].values
    if len(X) < n_clusters:
        return np.nan
    try:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        preds = km.fit_predict(X)
    except Exception:
        return np.nan
    ct = emb_df[CELLTYPE_COL].astype(str).values
    cluster_dom = {}
    for c in np.unique(preds):
        members = ct[preds == c]
        if len(members) == 0:
            continue
        cluster_dom[c] = _Counter(members).most_common(1)[0][0]
    per_type = []
    for bs in bs_types:
        mask = ct == str(bs)
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        n_mis = 0
        for idx in np.where(mask)[0]:
            c = preds[idx]
            if cluster_dom.get(c, None) != str(bs):
                n_mis += 1
        per_type.append(n_mis / n_total)
    return float(np.mean(per_type)) if per_type else np.nan


def compute_bsrs(emb_df, bs_types):
    """Batch-Specific Retention Score — mean per-cell silhouette (against full
    cell-type label space) restricted to batch-specific cells."""
    ct = emb_df[CELLTYPE_COL].astype(str)
    codes = ct.astype("category").cat.codes.values
    if len(np.unique(codes)) < 2:
        return np.nan
    X = emb_df[["UMAP1","UMAP2"]].values
    try:
        sil = silhouette_samples(X, codes)
    except Exception:
        return np.nan
    per_type = []
    for t in bs_types:
        mask = ct.values == str(t)
        if mask.sum() == 0:
            continue
        per_type.append(float(sil[mask].mean()))
    return float(np.mean(per_type)) if per_type else np.nan


def compute_integration_balance(bdf, cdf, method):
    """Returns (s_batch, s_bio, balance) where
        s_batch = mean(iLISI, 1-kBET, ASW_batch_mixing)
        s_bio   = mean(ARI, NMI, ASW_celltype, cLISI_purity)
        balance = sqrt(s_batch * s_bio)
    Components missing from a dataset are silently skipped."""
    iLISI     = _val(bdf, method, _resolve(bdf, ["iLISI"]))
    kBET      = _val(bdf, method, _resolve(bdf, ["kBET"]))
    ASW_batch = _val(bdf, method, _resolve(bdf, ["ASW_batch_mixing","ASW_batch"]))
    ARI       = _val(cdf, method, _resolve(cdf, ["ARI"]))
    NMI       = _val(cdf, method, _resolve(cdf, ["NMI"]))
    ASW_ct    = _val(bdf, method, _resolve(bdf, ["ASW_celltype","ASW_cell_type"]))
    cLISI_p   = _val(bdf, method, _resolve(bdf, ["cLISI_purity","cLISI"]))

    batch_components = []
    if np.isfinite(iLISI):     batch_components.append(iLISI)
    if np.isfinite(kBET):      batch_components.append(1 - kBET)
    if np.isfinite(ASW_batch): batch_components.append(ASW_batch)

    bio_components = []
    for v in (ARI, NMI, ASW_ct, cLISI_p):
        if np.isfinite(v):
            bio_components.append(v)

    if not batch_components or not bio_components:
        return np.nan, np.nan, np.nan
    s_batch = max(float(np.mean(batch_components)), 0.0)
    s_bio   = max(float(np.mean(bio_components)),   0.0)
    return s_batch, s_bio, float(np.sqrt(s_batch * s_bio))


def plot_oci_bar(ax, oci_dict, dataset_label=""):
    """Bar chart of OCI per method — lower is better.

    Two fixes for the 12-method universe:
    (1) When the bar value is exactly 0, draw a visible 'sentinel' bar at
        height = 1.5% of ymax so the method is not lost on the x-axis (the
        true value is still shown above as '0.00').
    (2) X-tick labels rotated 45° with smaller font to avoid overlap when
        12 methods sit in a ~2-inch-wide panel.
    """
    method_list = [m for m in RANK_METHODS
                   if m in oci_dict and np.isfinite(oci_dict[m])]
    if not method_list:
        ax.text(0.5, 0.5, "OCI unavailable", transform=ax.transAxes,
                ha="center", va="center"); return
    vals = [oci_dict[m] for m in method_list]
    colors = [METHOD_COLORS.get(m, "#888") for m in method_list]

    ymax = max(max(vals), 0.5) * 1.22
    # Sentinel: minimum visible bar height = 1.5% of ymax (so 0 values are
    # still visible as a small bar, not an invisible line at y=0).
    min_visible = ymax * 0.015
    bar_heights = [max(v, min_visible) if v <= 0 else v for v in vals]
    bars = ax.bar(range(len(method_list)), bar_heights, color=colors,
                  alpha=0.88, edgecolor="white", linewidth=0.5, width=0.72)
    # Border the sentinel bars in a distinct way so the reader can tell
    # they represent zero (or near-zero) measurements
    for b, v in zip(bars, vals):
        if v <= 0:
            b.set_hatch("///")
            b.set_alpha(0.55)

    for b, v in zip(bars, vals):
        # Rotate value labels 90° (vertical) so 12 close-together labels
        # never overlap with each other regardless of values.
        # Use 3 decimals for very small values so "0.00" doesn't look like exact zero.
        if 0 < v < 0.01:
            label = "<0.01"
        elif v == 0:
            label = "0.00"
        else:
            label = f"{v:.2f}"
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + ymax * 0.015, label,
                ha="center", va="bottom",
                fontsize=FS_ANNOT - 0.5, rotation=90)
    ax.set_xticks(range(len(method_list)))
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK - 0.5, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Overcorrection index (↓)", fontsize=FS_LABEL, labelpad=2)
    ax.set_title(f"OCI{' — ' + dataset_label if dataset_label else ''}",
                 fontsize=FS_TITLE, pad=4)
    # Extra top headroom: rotated labels grow upward, need more space
    ax.set_ylim(0, ymax * 1.20)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", pad=1)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.axvspan(i - 0.5, i + 0.5,
                   color=METHOD_COLORS["BTCA"], alpha=0.10, lw=0)


def plot_bsrs_bar(ax, bsrs_dict, dataset_label=""):
    """Bar chart of Batch-Specific Retention Score per method — higher is better.

    Negative-value labels are placed slightly above the x-axis (so they sit
    above the visual baseline and never overlap the descending bar).
    No '+' prefix on positive numbers; extra headroom above max value so
    the topmost label doesn't collide with the panel title.
    """
    method_list = [m for m in RANK_METHODS
                   if m in bsrs_dict and np.isfinite(bsrs_dict[m])]
    if not method_list:
        ax.text(0.5, 0.5, "BSRS unavailable", transform=ax.transAxes,
                ha="center", va="center"); return
    vals = [bsrs_dict[m] for m in method_list]
    colors = [METHOD_COLORS.get(m, "#888") for m in method_list]
    bars = ax.bar(range(len(method_list)), vals, color=colors,
                  alpha=0.88, edgecolor="white", linewidth=0.5, width=0.72)
    span = max(abs(min(vals)), abs(max(vals)))
    for b, v in zip(bars, vals):
        if v >= 0:
            y_lab = v + span * 0.025
            va = "bottom"
        else:
            # Negative bar: place label just above x-axis baseline
            y_lab = span * 0.025
            va = "bottom"
        # Vertical (90°) labels so 12 adjacent values never overlap;
        # no '+' prefix on positive numbers (Nature Methods convention).
        ax.text(b.get_x() + b.get_width()/2, y_lab, f"{v:.2f}",
                ha="center", va=va,
                fontsize=FS_ANNOT - 0.5, rotation=90,
                color=("#666" if v < 0 else "black"))

    # Explicit ylim with extra headroom so vertical top labels don't crowd
    y_min = min(min(vals), 0) - span * 0.10
    y_max = max(vals) + span * 0.40
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(range(len(method_list)))
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK - 0.5, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("BSRS (batch-specific retention, range 0–1 ↑)",
                  fontsize=FS_LABEL, labelpad=2)
    ax.set_title(f"Batch-specific retention{' — ' + dataset_label if dataset_label else ''}",
                 fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", pad=1)
    ax.axhline(0, color="#888", lw=0.5, ls="--", alpha=0.5)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.axvspan(i - 0.5, i + 0.5,
                   color=METHOD_COLORS["BTCA"], alpha=0.10, lw=0)


def plot_integration_balance(ax, bdf, cdf, title="Integration Balance"):
    """Trade-off scatter: x = batch removal score, y = bio conservation score.
    Label = '<method>\\nIB=<balance>' so the geometric-mean score is visible.
    Uses adjustText to auto-route labels and avoid overlap with dots."""
    pts = []
    for m in METHODS:
        sb, sbio, ib = compute_integration_balance(bdf, cdf, m)
        if np.isfinite(sb) and np.isfinite(sbio):
            pts.append((m, sb, sbio, ib))
    if pts:
        # Slightly larger padding to give labels more room
        ax.set_xlim(*_lim([p[1] for p in pts], pad=0.30))
        ax.set_ylim(*_lim([p[2] for p in pts], pad=0.30))

    # Plot dots
    for m, x, y, ib in pts:
        is_sccat = (m == "BTCA")
        ax.scatter(x, y, s=80 if is_sccat else 44,
                   color=METHOD_COLORS.get(m, "#888"),
                   edgecolors="white", linewidth=0.7,
                   alpha=0.95, zorder=5 if is_sccat else 4,
                   clip_on=False)

    # Place labels with adjustText (auto-route to avoid overlap)
    texts = []
    for m, x, y, ib in pts:
        is_sccat = (m == "BTCA")
        t = ax.text(
            x, y, f"{METHOD_DISPLAY.get(m, m)}\nIB={ib:.2f}",
            fontsize=FS_ANNOT + (0.7 if is_sccat else 0),
            color=METHOD_COLORS.get(m, "#888"),
            ha="center", va="center",
            fontweight="bold" if is_sccat else "normal",
            linespacing=0.95, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor="white", edgecolor="none",
                      alpha=0.65),
        )
        texts.append(t)

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            expand=(1.4, 1.6),
            force_text=(0.6, 0.9),
            force_static=(0.4, 0.6),
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.4, alpha=0.7),
            max_move=30,
        )
    except ImportError:
        # Fallback to quadrant-based offsets if adjustText is missing
        pass

    ax.set_xlabel("Batch removal score (↑)", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("Bio conservation score (↑)", fontsize=FS_LABEL, labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.6)


def annotate_cluster_centroids(ax, df, celltype_col, palette,
                                fontsize=None, with_box=True,
                                only_types=None):
    """Annotate each cell-type cluster with its name at the centroid.
    Used as a top-tier finishing touch on featured UMAPs."""
    if fontsize is None:
        fontsize = FS_TICK + 0.5
    types = only_types if only_types else df[celltype_col].astype(str).unique()
    for ct in types:
        sub = df[df[celltype_col].astype(str) == str(ct)]
        if sub.empty:
            continue
        cx = float(np.median(sub["UMAP1"]))
        cy = float(np.median(sub["UMAP2"]))
        color = palette.get(str(ct), "#333")
        if with_box:
            ax.text(cx, cy, str(ct),
                     fontsize=fontsize, color="white",
                     fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.25",
                               facecolor=color, edgecolor="white",
                               linewidth=0.5, alpha=0.90),
                     zorder=20, clip_on=False)
        else:
            ax.text(cx, cy, str(ct),
                     fontsize=fontsize, color=color,
                     fontweight="bold", ha="center", va="center",
                     zorder=20, clip_on=False)


def plot_per_celltype_purity_bars(ax, emb_data, meta, methods=None,
                                   celltype_palette=None,
                                   celltype_order=None,
                                   n_clusters=None):
    """Per-cell-type cluster purity as a heatmap (cell types × methods).

    Why heatmap rather than grouped bars: with 12 methods × N cell types
    the grouped-bar variant produces 48+ thin bars that cannot show value
    labels and that are hard to distinguish by colour. The heatmap form
    scales to any (N_methods, M_cell_types) and is the standard Nature-style
    quantitative grid for this kind of comparison; each cell shows the
    exact purity value, and scCAT's row is highlighted with a thick green
    border so the reader's eye is led to the comparison.

    Purity = fraction of cells of a given true type whose assigned k-means
    cluster is dominated by that same true type. 1.0 = perfectly preserved.
    """
    from matplotlib.patches import Rectangle
    methods = methods or RANK_METHODS
    method_list = [m for m in methods if m in emb_data]
    if celltype_order is None:
        celltype_order = list(celltype_palette.keys()) if celltype_palette \
                         else sorted(meta[CELLTYPE_COL].unique())
    if n_clusters is None:
        n_clusters = meta[CELLTYPE_COL].nunique()

    # Compute purity matrix
    purity = np.full((len(method_list), len(celltype_order)), np.nan)
    for j, m in enumerate(method_list):
        df = emb_data[m]
        X = df[["UMAP1","UMAP2"]].values
        if len(X) < n_clusters:
            continue
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            preds = km.fit_predict(X)
        except Exception:
            continue
        ct = df[CELLTYPE_COL].astype(str).values
        cluster_dom = {}
        for c in np.unique(preds):
            members = ct[preds == c]
            if len(members):
                from collections import Counter
                cluster_dom[c] = Counter(members).most_common(1)[0][0]
        for i, t in enumerate(celltype_order):
            mask = ct == str(t)
            if mask.sum() == 0:
                continue
            n_correct = 0
            for idx in np.where(mask)[0]:
                if cluster_dom.get(preds[idx], None) == str(t):
                    n_correct += 1
            purity[j, i] = n_correct / mask.sum()

    # Heatmap: methods × cell types (Nature convention — rows = condition / method)
    im = ax.imshow(purity, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Annotate every cell with the purity value
    for j, m in enumerate(method_list):
        for i, t in enumerate(celltype_order):
            v = purity[j, i]
            if np.isfinite(v):
                # White text on dark (low purity) cells; black on light (high)
                txt_col = "white" if v < 0.35 or v > 0.85 else "black"
                ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color=txt_col)

    # Axes
    ax.set_xticks(range(len(celltype_order)))
    ax.set_xticklabels(celltype_order, fontsize=FS_TICK + 0.3, rotation=0)
    if celltype_palette:
        for tick, ct in zip(ax.get_xticklabels(), celltype_order):
            tick.set_color(celltype_palette.get(ct, "black"))
            tick.set_fontweight("bold")
    ax.set_yticks(range(len(method_list)))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK)
    ax.set_title("Per-cell-type cluster purity",
                 fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Highlight scCAT row with thick green border
    if "BTCA" in method_list:
        j = method_list.index("BTCA")
        ax.add_patch(Rectangle((-0.5, j - 0.5), len(celltype_order), 1,
                               fill=False, edgecolor=METHOD_COLORS["BTCA"],
                               linewidth=1.2, clip_on=False))

    # Compact colour bar (anchored to right)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, aspect=22)
    cbar.set_label("Cluster purity (↑)", fontsize=FS_ANNOT, labelpad=2)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=FS_TICK - 1, length=2)


def plot_marker_dotplot(ax, expression_dict, cell_type_labels,
                         markers, cell_type_order,
                         celltype_palette=None,
                         min_expr_threshold=0.0,
                         title="Marker gene expression",
                         max_dot_size=240):
    """Classic scRNA-seq dot plot.

    For each (marker, cell_type) pair:
        - dot AREA  proportional to fraction of cells expressing
        - dot COLOR proportional to mean expression within the cell type
    """
    ct_arr = np.asarray([str(x) for x in cell_type_labels])
    n_t = len(cell_type_order)
    n_m = len(markers)

    pct  = np.zeros((n_m, n_t))
    mean = np.zeros((n_m, n_t))

    for j, ct in enumerate(cell_type_order):
        mask = ct_arr == str(ct)
        if mask.sum() == 0:
            continue
        for i, mk in enumerate(markers):
            expr = expression_dict.get(mk)
            if expr is None:
                pct[i, j] = np.nan
                mean[i, j] = np.nan
                continue
            sub = expr[mask]
            sub = sub[np.isfinite(sub)]
            if len(sub) == 0:
                continue
            pct[i, j] = float((sub > min_expr_threshold).mean())
            mean[i, j] = float(sub.mean())

    # Build dot data
    xs, ys, sizes, colors = [], [], [], []
    vmax = float(np.nanmax(mean)) if np.isfinite(mean).any() else 1.0
    for i in range(n_m):
        for j in range(n_t):
            if not (np.isfinite(pct[i, j]) and np.isfinite(mean[i, j])):
                continue
            xs.append(j); ys.append(i)
            sizes.append(max(10, pct[i, j] * max_dot_size))
            colors.append(mean[i, j])

    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="Reds",
                     vmin=0, vmax=vmax, edgecolors="#222",
                     linewidth=0.4, zorder=3)

    ax.set_xticks(range(n_t))
    ax.set_xticklabels(cell_type_order, fontsize=FS_TICK + 0.3, rotation=0)
    if celltype_palette:
        for tick, ct in zip(ax.get_xticklabels(), cell_type_order):
            tick.set_color(celltype_palette.get(ct, "black"))
            tick.set_fontweight("bold")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(markers, fontsize=FS_TICK + 0.3,
                        fontstyle="italic")
    ax.set_ylim(-0.6, n_m - 0.4)
    ax.set_xlim(-0.6, n_t - 0.4)
    ax.invert_yaxis()
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)

    # Color bar for mean expression
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, aspect=15)
    cbar.set_label("Mean expression", fontsize=FS_ANNOT, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)

    # Size legend (% expressing) — draw separate handles
    pct_legend_vals = [0.25, 0.50, 0.75, 1.00]
    legend_handles = [
        plt.scatter([], [], s=max(10, p * max_dot_size),
                     c="#888", edgecolors="#222", linewidth=0.4,
                     label=f"{int(p*100)}%")
        for p in pct_legend_vals
    ]
    ax.legend(handles=legend_handles, title="% expressing",
              loc="upper left", bbox_to_anchor=(1.18, 1.0),
              fontsize=FS_LEGEND - 1, title_fontsize=FS_LEGEND - 0.5,
              frameon=False, borderaxespad=0,
              labelspacing=0.6, handletextpad=0.6)


def plot_composition_matrix(ax, meta, celltype_palette, batch_palette,
                             title="Dataset composition"):
    """Cell type × batch composition matrix.
    Cells with zero count are shown as light-grey '0' (missing entries).
    Cells with positive count get a colored fill proportional to log(count).
    Used in Fig 3 panel a as a professional alternative to schematic art.
    """
    cell_types = list(celltype_palette.keys())
    # Sort batches naturally
    batches = sorted(str(b) for b in meta[BATCH_COL].unique())

    # Build count matrix
    M = np.zeros((len(cell_types), len(batches)), dtype=int)
    for i, ct in enumerate(cell_types):
        for j, b in enumerate(batches):
            M[i, j] = ((meta[CELLTYPE_COL].astype(str) == str(ct)) &
                       (meta[BATCH_COL].astype(str) == str(b))).sum()

    # Use log-colored fill so the visual contrast works with skewed counts
    M_color = np.where(M > 0, np.log1p(M), np.nan)
    vmax = np.nanmax(M_color) if np.isfinite(M_color).any() else 1.0
    im = ax.imshow(M_color, cmap="Blues", vmin=0, vmax=vmax * 1.05,
                    aspect="auto")
    # Overlay text counts
    for i in range(len(cell_types)):
        for j in range(len(batches)):
            v = M[i, j]
            if v == 0:
                ax.text(j, i, "—", ha="center", va="center",
                         fontsize=FS_ANNOT + 0.5, color="#AAA",
                         fontweight="bold")
            else:
                # White text on dark cells, black on light cells
                color_norm = M_color[i, j] / (vmax * 1.05) if vmax > 0 else 0.5
                txt_color = "white" if color_norm > 0.55 else "#222"
                ax.text(j, i, f"{v}", ha="center", va="center",
                         fontsize=FS_ANNOT + 0.5, color=txt_color,
                         fontweight="bold")

    # Mark batch-specific rows with a colored border on the empty cell
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(
        [f"Batch {b}" if not str(b).startswith("Batch") else str(b)
         for b in batches],
        fontsize=FS_TICK, rotation=0, ha="center",
    )
    ax.set_yticks(range(len(cell_types)))
    # Color the y-tick labels by cell-type palette
    ax.set_yticklabels(cell_types, fontsize=FS_TICK)
    for tick, ct in zip(ax.get_yticklabels(), cell_types):
        tick.set_color(celltype_palette.get(ct, "black"))
        tick.set_fontweight("bold")

    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    # Light grid between cells
    ax.set_xticks(np.arange(len(batches)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(cell_types)) + 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_per_celltype_silhouette_heatmap(ax, emb_data, meta,
                                         methods=None,
                                         celltypes=None,
                                         celltype_palette=None,
                                         title="Per-cell-type silhouette"):
    """Heatmap: cell types × methods, value = mean per-cell silhouette for
    that cell type using full cell-type labels. Negative values are red,
    positive are green (RdYlGn diverging colormap centered at 0).

    Used in Fig 3 panel d as a per-cell-type breakdown that's clearly
    distinct from the aggregate BSRS bar used in Fig 2.
    """
    methods = methods or RANK_METHODS
    celltypes = celltypes or (list(celltype_palette.keys()) if celltype_palette
                              else sorted(meta[CELLTYPE_COL].unique()))

    method_list = [m for m in methods if m in emb_data]
    n_ct, n_m = len(celltypes), len(method_list)
    M = np.full((n_ct, n_m), np.nan)

    for j, m in enumerate(method_list):
        df = emb_data[m]
        labs = df[CELLTYPE_COL].astype(str)
        codes = labs.astype("category").cat.codes.values
        if len(np.unique(codes)) < 2:
            continue
        try:
            sil = silhouette_samples(df[["UMAP1","UMAP2"]].values, codes)
        except Exception:
            continue
        for i, ct in enumerate(celltypes):
            mask = labs.values == str(ct)
            if mask.sum() > 0:
                M[i, j] = float(sil[mask].mean())

    # Diverging colormap centered at 0 (negative = red, positive = green)
    vmax = max(0.5, np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(n_ct):
        for j in range(n_m):
            v = M[i, j]
            if np.isfinite(v):
                # Pick text color based on background lightness
                norm = (v + vmax) / (2 * vmax)
                txt = "white" if (norm < 0.18 or norm > 0.82) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=FS_ANNOT, color=txt)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                       fontsize=FS_TICK, rotation=30, ha="right")
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(celltypes, fontsize=FS_TICK)
    if celltype_palette:
        for tick, ct in zip(ax.get_yticklabels(), celltypes):
            tick.set_color(celltype_palette.get(ct, "black"))
            tick.set_fontweight("bold")

    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    # Mark scCAT column
    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.add_patch(Rectangle((i - 0.5, -0.5), 1, n_ct,
                                fill=False, edgecolor=METHOD_COLORS["BTCA"],
                                linewidth=1.0, clip_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=15)
    cbar.set_label("Silhouette", fontsize=FS_ANNOT, labelpad=2)
    cbar.ax.tick_params(labelsize=FS_TICK - 1)
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f"-{vmax:.1f}", "0", f"+{vmax:.1f}"])


def load_dataset_expression(dataset_key, genes, h5ad_source="INSCT_Unsupervised"):
    """Load expression matrix for given gene names from a method's h5ad file.
    Returns {"_cell_names": array, gene_name: expression_array_or_None, ...}."""
    cfg = DATASET_CONFIGS[dataset_key]
    path = BASE_DIR / cfg["emb_dir"] / f"{h5ad_source}.h5ad"
    if not path.exists():
        return None
    adata = ad.read_h5ad(path)
    cell_names = adata.obs_names.astype(str).values
    expr_dict = {"_cell_names": cell_names}
    var_names = list(adata.var_names)
    for g in genes:
        if g in var_names:
            col_idx = var_names.index(g)
            arr = adata.X[:, col_idx]
            if hasattr(arr, "toarray"):
                arr = arr.toarray().ravel()
            else:
                arr = np.asarray(arr).ravel()
            expr_dict[g] = arr.astype(float)
        else:
            expr_dict[g] = None
    return expr_dict


def plot_marker_feature(ax, df, expr_arr, gene_name, pt_size=5):
    """Feature plot: cells coloured by gene expression on UMAP."""
    if expr_arr is None:
        ax.text(0.5, 0.5, f"{gene_name}\n(not in HVGs)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_TICK)
        _nature_embed_style(ax, gene_name)
        return
    valid = np.isfinite(expr_arr)
    if valid.sum() == 0:
        ax.text(0.5, 0.5, f"{gene_name}\n(no expr)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_TICK)
        _nature_embed_style(ax, gene_name)
        return
    finite_expr = expr_arr[valid]
    vmax = max(float(np.quantile(finite_expr, 0.97)), 0.3)
    # Plot zero-expressing cells first (background grey)
    expr_for_color = np.where(valid, expr_arr, 0.0)
    order = np.argsort(expr_for_color)
    sc = ax.scatter(df["UMAP1"].values[order], df["UMAP2"].values[order],
                    s=pt_size, c=expr_for_color[order], cmap="viridis",
                    vmin=0, vmax=vmax,
                    alpha=0.85, linewidths=0, rasterized=True)
    _nature_embed_style(ax, gene_name, title_size=FS_TITLE + 0.5)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, aspect=12)
    cbar.set_label("expr", fontsize=FS_ANNOT, labelpad=1)
    cbar.ax.tick_params(labelsize=FS_TICK - 1, length=2)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Shared helpers for figure layout
# ─────────────────────────────────────────────────────────────────────────────

def _load_figure_data(dataset_key):
    cfg  = DATASET_CONFIGS[dataset_key]
    meta = read_metadata(BASE_DIR / cfg["meta_csv"])
    emb  = load_all_embeddings(METHODS, BASE_DIR / cfg["emb_dir"], meta)
    bdf  = read_metric(BASE_DIR / cfg["batch_metric"], METHODS)
    cdf  = read_metric(BASE_DIR / cfg["clust_metric"], METHODS)
    return cfg, meta, emb, bdf, cdf


def _resolved_metric_cols(bdf):
    return dict(
        ilisi  = _resolve(bdf, ["iLISI"]),
        clisi  = _resolve(bdf, ["cLISI_purity","cLISI","cLISI_celltype"]),
        asw_ct = _resolve(bdf, ["ASW_celltype","ASW_cell_type"]),
        asw_ba = _resolve(bdf, ["ASW_batch_mixing","ASW_batch"]),
        kbet   = _resolve(bdf, ["kBET","kbet"]),
    )


def _draw_umap_grid(fig, gs_slot, emb_data, color_col, palette,
                    batch_specific, pt_size, alpha,
                    panel_letter=None, label_x=-0.22, label_y=1.10,
                    n_cols=4):
    """Draw a method-by-method UMAP grid. Number of rows is determined
    dynamically from len(METHODS) so the layout auto-adapts when methods
    are added (e.g. 8 → 11 methods becomes 3 × 4 with one empty slot)."""
    n_meth = len(METHODS)
    n_rows = (n_meth + n_cols - 1) // n_cols
    gs = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs_slot,
                                          wspace=0.28, hspace=0.45)
    first_ax = None
    for i, m in enumerate(METHODS):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        if first_ax is None:
            first_ax = ax
        if m in emb_data:
            plot_embedding(ax, emb_data[m], color_col, palette, m,
                           batch_specific=batch_specific,
                           pt_size=pt_size, alpha=alpha)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS_TICK)
            _nature_embed_style(ax, METHOD_DISPLAY.get(m, m))
    # Fill empty trailing slots with blank axes (so the grid looks clean)
    for j in range(n_meth, n_rows * n_cols):
        ax = fig.add_subplot(gs[j // n_cols, j % n_cols])
        ax.axis("off")
    if panel_letter and first_ax is not None:
        _panel_label(first_ax, panel_letter, x=label_x, y=label_y)


def _draw_umap_row(fig, gs_slot, emb_data, plot_fn,
                   methods_subset=None,
                   panel_letter=None, label_x=-0.20, wspace=0.18, **kwargs):
    methods_subset = methods_subset or METHODS
    n = len(methods_subset)
    gs = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=gs_slot,
                                          wspace=wspace)
    first_ax = None
    for i, m in enumerate(methods_subset):
        ax = fig.add_subplot(gs[0, i])
        if first_ax is None:
            first_ax = ax
        if m in emb_data:
            plot_fn(ax, emb_data[m], method=m, **kwargs)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS_TICK)
            _nature_embed_style(ax, METHOD_DISPLAY.get(m, m))
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("#cccccc")
            sp.set_linewidth(0.5)
    if panel_letter and first_ax is not None:
        _panel_label(first_ax, panel_letter, x=label_x)


def _save(fig, key, dpi_raster=300):
    """Save a figure as PDF + SVG (vector) + PNG + TIFF (raster, 300 dpi).

    PNG and TIFF use 300 dpi at the figure's intrinsic size, which gives
    publication-grade resolution for both screen and print (Nature Methods
    accepts both PDF/EPS for vector and PNG/TIFF for raster).
    TIFF uses LZW compression to keep files reasonably sized.

    If the target files are locked (open in a viewer), the function
    automatically falls back to versioned filenames so the run never crashes.
    """
    # Standard export targets — vector first, then raster
    targets = [
        (OUT_DIR / f"{key}.pdf",  "pdf",  dict(dpi=dpi_raster)),
        (OUT_DIR / f"{key}.svg",  "svg",  {}),
        (OUT_DIR / f"{key}.png",  "png",  dict(dpi=dpi_raster)),
        (OUT_DIR / f"{key}.tiff", "tiff", dict(dpi=dpi_raster,
                                                pil_kwargs={"compression": "tiff_lzw"})),
    ]
    common = dict(bbox_inches="tight", facecolor="white")

    # First try to write all to the canonical names
    locked = False
    saved_paths = []
    try:
        for path, fmt, extra in targets:
            fig.savefig(path, format=fmt, **common, **extra)
            saved_paths.append(path.name)
    except PermissionError:
        locked = True

    # If anything was locked, fall back to a versioned suffix for ALL formats
    if locked:
        import time
        suffix = time.strftime("%H%M%S")
        saved_paths = []
        for path, fmt, extra in targets:
            stem = path.stem
            new_path = path.parent / f"{stem}_v{suffix}{path.suffix}"
            fig.savefig(new_path, format=fmt, **common, **extra)
            saved_paths.append(new_path.name)
        print(f"  [LOCKED] some {key}.* files were locked")
        print(f"           saved as: {' | '.join(saved_paths)}")
        print(f"           (close any open viewer to overwrite the originals)")
    else:
        print(f"  -> {' | '.join(saved_paths)}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PER-FIGURE BUILDERS — six unique architectures
# ─────────────────────────────────────────────────────────────────────────────

def make_fig_data2_s1(key):
    """Fig 2 — Comprehensive baseline; leads with cell-type UMAPs; closes with
    method ranking strip across all 7 metrics."""
    cfg, meta, emb, bdf, cdf = _load_figure_data(key)
    cols = _resolved_metric_cols(bdf)
    pt, al = cfg["point_size"], cfg["alpha"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.3, 12.6), dpi=150)
    fig.subplots_adjust(right=0.855)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.8, 2.3, 1.85], hspace=0.30)

    # a — leads with cell-type UMAPs (the story)
    _draw_umap_grid(fig, outer[0], emb, CELLTYPE_COL,
                    cfg["celltype_palette"], cfg["batch_specific_types"],
                    pt, al, panel_letter="a")
    # b — batch UMAPs follow (secondary)
    _draw_umap_grid(fig, outer[1], emb, BATCH_COL,
                    cfg["batch_palette"], None,
                    max(pt - 1, 2), al, panel_letter="b")

    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.862, 0.80))
    _figure_legend(fig, cfg["batch_palette"], "Batch",
                   anchor=(0.862, 0.43))

    # c — LISI scatter | ASW scatter | ranking strip
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2],
        width_ratios=[1.0, 1.0, 1.55], wspace=0.55,
    )
    ax_lisi = fig.add_subplot(gs_c[0, 0]); _panel_label(ax_lisi, "c", x=-0.30)
    plot_scatter_metrics(ax_lisi, bdf, cols["clisi"], cols["ilisi"],
                         "cLISI (cell type↑)", "iLISI (batch↑)", "LISI")
    ax_asw = fig.add_subplot(gs_c[0, 1])
    plot_scatter_metrics(ax_asw, bdf, cols["asw_ct"], cols["asw_ba"],
                         "ASW cell type (↑)", "ASW batch (↑)", "ASW")
    ax_rank = fig.add_subplot(gs_c[0, 2]); _panel_label(ax_rank, "d", x=-0.18)
    plot_method_ranking_strip(ax_rank, bdf, cdf)

    _save(fig, key)


def make_fig_data2_s2(key):
    """Fig 3 — Batch-specific failure mode; TOP-4 large UMAPs + 1×8 thumbnails
    + confusion-matrix heatmap + preservation bar / kBET lollipop."""
    cfg, meta, emb, bdf, cdf = _load_figure_data(key)
    cols = _resolved_metric_cols(bdf)
    pt, al = cfg["point_size"], cfg["alpha"]
    bs_types = cfg["batch_specific_types"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.3, 14.2), dpi=150)
    fig.subplots_adjust(right=0.855)
    outer = gridspec.GridSpec(4, 1, figure=fig,
                              height_ratios=[2.7, 1.30, 2.9, 1.75],
                              hspace=0.34)

    # a — 1×4 LARGE UMAPs of top-4 methods (closest competitors + scCAT)
    top4 = ["BTCA", "fastMNN", "INSCT_Unsupervised", "SPDR"]
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0],
                                            wspace=0.22)
    for i, m in enumerate(top4):
        ax = fig.add_subplot(gs_a[0, i])
        if i == 0:
            _panel_label(ax, "a", x=-0.14)
        if m in emb:
            plot_embedding_featured(ax, emb[m], CELLTYPE_COL,
                                    cfg["celltype_palette"], m,
                                    batch_specific=bs_types,
                                    pt_size=pt * 1.4, alpha=al)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center")
            _nature_embed_style(ax, METHOD_DISPLAY.get(m, m))

    # b — 1×8 thumbnail row of ALL methods so the full comparison stays present
    _draw_umap_row(
        fig, outer[1], emb,
        plot_fn=lambda ax, df, method:
            plot_embedding(ax, df, CELLTYPE_COL, cfg["celltype_palette"],
                           method, batch_specific=bs_types,
                           pt_size=pt * 0.6, alpha=al * 0.85),
        panel_letter="b", label_x=-0.22,
    )

    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.862, 0.68))

    # c — confusion matrix (8 methods × 4 batch-specific groups)
    ax_c = fig.add_subplot(outer[2]); _panel_label(ax_c, "c", x=-0.09)
    plot_bs_confusion_matrix(ax_c, emb, meta, METHODS, bs_types)

    # d — preservation bar  |  kBET lollipop
    gs_d = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[3],
        width_ratios=[1.0, 1.0], wspace=0.55,
    )
    ax_d1 = fig.add_subplot(gs_d[0, 0]); _panel_label(ax_d1, "d", x=-0.18)
    plot_batch_specific_preservation(ax_d1, emb, meta, METHODS, bs_types)
    ax_d2 = fig.add_subplot(gs_d[0, 1]); _panel_label(ax_d2, "e", x=-0.20)
    plot_kbet_lollipop(ax_d2, bdf, cols["kbet"])

    _save(fig, key)


def make_fig_sc_mixology(key):
    """Fig 4 — Cross-platform; LANDSCAPE aspect; scCAT vs SPDR head-to-head;
    1×6 small comparison row; silhouette ridge + compact eval heatmap."""
    cfg, meta, emb, bdf, cdf = _load_figure_data(key)
    pt, al = cfg["point_size"], cfg["alpha"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.6, 10.5), dpi=150)
    fig.subplots_adjust(right=0.862)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.4, 1.35, 2.4],
                              hspace=0.34)

    # a — scCAT vs SPDR head-to-head (cell-type coloured)
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            wspace=0.20)
    for i, m in enumerate(["BTCA", "SPDR"]):
        ax = fig.add_subplot(gs_a[0, i])
        if i == 0:
            _panel_label(ax, "a", x=-0.10)
        if m in emb:
            plot_embedding_featured(ax, emb[m], CELLTYPE_COL,
                                    cfg["celltype_palette"], m,
                                    pt_size=pt * 1.4, alpha=al)

    # b — small UMAPs of the remaining 6 methods (cell-type coloured)
    others = [m for m in METHODS if m not in ("BTCA", "SPDR")]
    _draw_umap_row(
        fig, outer[1], emb,
        plot_fn=lambda ax, df, method:
            plot_embedding(ax, df, CELLTYPE_COL, cfg["celltype_palette"],
                           method, pt_size=pt * 0.55, alpha=al * 0.9),
        methods_subset=others,
        panel_letter="b", label_x=-0.20, wspace=0.20,
    )

    _figure_legend(fig, cfg["celltype_palette"], "Cell line",
                   anchor=(0.864, 0.75))
    _figure_legend(fig, cfg["batch_palette"], "Platform",
                   anchor=(0.864, 0.56))

    # c — silhouette ridge | compact vertical eval heatmap
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[1.55, 1.30], wspace=0.45,
    )
    ax_c = fig.add_subplot(gs_c[0, 0]); _panel_label(ax_c, "c", x=-0.16)
    plot_silhouette_ridge(ax_c, emb)
    ax_d = fig.add_subplot(gs_c[0, 1]); _panel_label(ax_d, "d", x=-0.20)
    plot_eval_heatmap(ax_d, bdf, cdf)

    _save(fig, key)


def make_fig_hdc(key):
    """Fig 5 — Rare-cell rescue; schematic + scCAT featured + 1×7 focus row +
    per-cell-type silhouette bars + kBET-vs-preservation trade-off scatter."""
    cfg, meta, emb, bdf, cdf = _load_figure_data(key)
    cols = _resolved_metric_cols(bdf)
    pt, al = cfg["point_size"], cfg["alpha"]
    bs_types = cfg["batch_specific_types"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.3, 13.0), dpi=150)
    fig.subplots_adjust(right=0.855)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.4, 1.45, 2.5],
                              hspace=0.34)

    # Top row: schematic (left) | scCAT featured large UMAP (right)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.35, 1.0], wspace=0.22,
    )
    ax_sch = fig.add_subplot(gs_top[0, 0]); _panel_label(ax_sch, "a", x=-0.08)
    plot_hdc_schematic(ax_sch, cfg["batch_palette"], cfg["celltype_palette"])

    ax_btca = fig.add_subplot(gs_top[0, 1]); _panel_label(ax_btca, "b", x=-0.14)
    if "BTCA" in emb:
        plot_embedding_featured(ax_btca, emb["BTCA"], CELLTYPE_COL,
                                cfg["celltype_palette"], "BTCA",
                                batch_specific=bs_types,
                                pt_size=pt * 1.6, alpha=al)

    # Row c: 1×7 focus-only UMAPs of competitors
    other_methods = [m for m in METHODS if m != "BTCA"]
    _draw_umap_row(
        fig, outer[1], emb,
        plot_fn=lambda ax, df, method:
            plot_embedding_focus(ax, df, bs_types, cfg["celltype_palette"],
                                 method, pt_size=pt * 0.65, alpha=al * 0.95),
        methods_subset=other_methods,
        panel_letter="c", label_x=-0.22, wspace=0.20,
    )

    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.862, 0.66))

    # Bottom row d: silhouette bars  |  trade-off scatter
    gs_d = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[1.55, 1.20], wspace=0.50,
    )
    ax_d = fig.add_subplot(gs_d[0, 0]); _panel_label(ax_d, "d", x=-0.14)
    plot_celltype_silhouette_bars(ax_d, emb, meta,
                                  celltype_palette=cfg["celltype_palette"])

    ax_e = fig.add_subplot(gs_d[0, 1]); _panel_label(ax_e, "e", x=-0.20)
    # Compute preservation rates for the trade-off scatter
    n_clusters = meta[CELLTYPE_COL].nunique()
    preservation = {}
    for m in METHODS:
        if m not in emb:
            continue
        df = emb[m]
        try:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            preds = km.fit_predict(df[["UMAP1","UMAP2"]].values)
            ct = df[CELLTYPE_COL].astype(str).values
            pur = []
            for bs in bs_types:
                mask = ct == str(bs)
                if mask.sum() == 0:
                    continue
                mode_c = pd.Series(preds[mask]).mode().iloc[0]
                cmask = preds == mode_c
                pur.append((ct[cmask] == str(bs)).sum() / cmask.sum())
            if pur:
                preservation[m] = float(np.mean(pur))
        except Exception:
            pass
    kbet_vals = {m: _val(bdf, m, cols["kbet"]) for m in METHODS}
    plot_tradeoff_2d(
        ax_e, kbet_vals, preservation,
        xlabel="kBET (↓ = better mixing)",
        ylabel="Batch-specific preservation (↑)",
        title="Mixing vs rare-cell rescue",
        optimal_corner="upper-left",
    )

    _save(fig, key)


def make_fig_pbmc(key):
    """Fig 6 — Condition integration; KDE density overlay (scCAT) +
    2×4 cell-type UMAPs + paired dot plot + composition stack."""
    cfg, meta, emb, bdf, cdf = _load_figure_data(key)
    pt, al = cfg["point_size"], cfg["alpha"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.4, 13.5), dpi=150)
    fig.subplots_adjust(right=0.862)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.2, 2.6, 2.4],
                              hspace=0.34)

    # a + b: density overlay (left) | scCAT featured cell-type UMAP (right)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.4, 1.0], wspace=0.22,
    )
    ax_a = fig.add_subplot(gs_top[0, 0]); _panel_label(ax_a, "a", x=-0.10)
    if "BTCA" in emb:
        plot_density_overlay(ax_a, emb["BTCA"], BATCH_COL,
                             ["control", "stimulated"],
                             cfg["batch_palette"], "BTCA")
    ax_b_inset = fig.add_subplot(gs_top[0, 1]); _panel_label(ax_b_inset, "b", x=-0.15)
    if "BTCA" in emb:
        plot_embedding_featured(ax_b_inset, emb["BTCA"], CELLTYPE_COL,
                                cfg["celltype_palette"], "BTCA",
                                pt_size=pt * 1.4, alpha=al)

    # c: 2×4 small UMAPs of ALL methods, cell-type coloured
    _draw_umap_grid(fig, outer[1], emb, CELLTYPE_COL,
                    cfg["celltype_palette"], None,
                    pt, al, panel_letter="c", label_x=-0.22)

    _figure_legend(fig, cfg["batch_palette"], "Condition",
                   anchor=(0.866, 0.85))
    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.866, 0.46))

    # d + e: paired dot plot | composition stack
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[1.25, 1.55], wspace=0.55,
    )
    ax_d = fig.add_subplot(gs_bot[0, 0]); _panel_label(ax_d, "d", x=-0.16)
    plot_paired_dotplot(ax_d, emb, meta, METHODS,
                        cfg["celltype_palette"], ["control", "stimulated"])
    ax_e = fig.add_subplot(gs_bot[0, 1]); _panel_label(ax_e, "e", x=-0.13)
    plot_composition_stack(ax_e, emb, meta, METHODS, cfg["celltype_palette"])

    _save(fig, key)


def make_fig_lung(key):
    """Fig 7 — Scaling robustness; scCAT featured pair + 1×7 thumbnails +
    per-batch mixing heatmap + biology-vs-mixing trade-off curve."""
    cfg, meta, emb, bdf, cdf = _load_figure_data(key)
    cols = _resolved_metric_cols(bdf)
    pt, al = cfg["point_size"], cfg["alpha"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.6, 15.0), dpi=150)
    fig.subplots_adjust(right=0.858)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.6, 1.50, 3.4],
                              hspace=0.32)

    # a: scCAT featured pair (batch | cell type)
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            wspace=0.22)
    ax_a1 = fig.add_subplot(gs_a[0, 0]); _panel_label(ax_a1, "a", x=-0.10)
    if "BTCA" in emb:
        plot_embedding_featured(ax_a1, emb["BTCA"], BATCH_COL,
                                cfg["batch_palette"], "BTCA",
                                pt_size=pt * 1.6, alpha=al)
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    if "BTCA" in emb:
        plot_embedding_featured(ax_a2, emb["BTCA"], CELLTYPE_COL,
                                cfg["celltype_palette"], "BTCA",
                                pt_size=pt * 1.6, alpha=al)

    # b: 1×7 small UMAPs of other 7 methods (cell-type coloured)
    other_methods = [m for m in METHODS if m != "BTCA"]
    _draw_umap_row(
        fig, outer[1], emb,
        plot_fn=lambda ax, df, method:
            plot_embedding(ax, df, CELLTYPE_COL, cfg["celltype_palette"],
                           method, pt_size=pt * 0.55, alpha=al * 0.85),
        methods_subset=other_methods,
        panel_letter="b", label_x=-0.22, wspace=0.20,
    )

    # Batch legend grouped by series
    bp = cfg["batch_palette"]
    _figure_legend(fig,
                   {k: v for k, v in bp.items() if not k.startswith(("A","B"))},
                   "Numeric", anchor=(0.860, 0.88))
    _figure_legend(fig,
                   {k: v for k, v in bp.items() if k.startswith("A")},
                   "A-series", anchor=(0.860, 0.80))
    _figure_legend(fig,
                   {k: v for k, v in bp.items() if k.startswith("B")},
                   "B-series", anchor=(0.860, 0.72))
    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.860, 0.50), ncol_max=18)

    # c: per-batch heatmap | biology-vs-mixing trade-off curve
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[1.55, 1.15], wspace=0.45,
    )
    ax_c = fig.add_subplot(gs_c[0, 0]); _panel_label(ax_c, "c", x=-0.16)
    plot_per_batch_mixing(ax_c, emb, meta, methods=RANK_METHODS,
                          k=30, subsample_n=8000)

    ax_d = fig.add_subplot(gs_c[0, 1]); _panel_label(ax_d, "d", x=-0.20)
    kbet_vals = {m: _val(bdf, m, cols["kbet"]) for m in METHODS}
    ari_col = _resolve(cdf, ["ARI"])
    ari_vals = {m: _val(cdf, m, ari_col) for m in METHODS}
    # Convert kBET to a mixing score (higher = better) so trade-off plot reads
    # "upper-right = optimal" consistently
    mixing_vals = {m: (1 - kbet_vals[m]) if np.isfinite(kbet_vals[m]) else np.nan
                   for m in METHODS}
    plot_tradeoff_2d(
        ax_d, mixing_vals, ari_vals,
        xlabel="Batch mixing  (1 − kBET, ↑)",
        ylabel="Biology preservation  (ARI, ↑)",
        title="Biology vs mixing trade-off",
        optimal_corner="upper-right",
    )

    _save(fig, key)


# ─────────────────────────────────────────────────────────────────────────────
# 10b.  MAIN FIGURE BUILDERS  (claim-organised; Nature Methods style)
#
# The three "main" figures consolidate evidence across multiple datasets so
# that each figure answers a single scientific claim, while the per-dataset
# builders (section 10) become Supplementary Figures.
# ─────────────────────────────────────────────────────────────────────────────

def _load_metric_only(dataset_key):
    """Load only the metric CSVs (no embeddings) — cheap for Fig 2."""
    cfg = DATASET_CONFIGS[dataset_key]
    return {
        "batch":   read_metric(BASE_DIR / cfg["batch_metric"], METHODS),
        "cluster": read_metric(BASE_DIR / cfg["clust_metric"], METHODS),
    }


def make_main_fig2_simulation(key=None):
    """
    Fig 2 — "scCAT reduces overcorrection in controlled simulations with
            partially shared cell populations"

    8 panels (top-tier target):
    a  Sim 3 (balanced)        — 1×5 cell-type UMAPs of representative methods
    b  Sim 4 (batch-specific)  — 1×5 cell-type UMAPs, bs cells outlined
    c  Sim 4                   — 1×5 BATCH-coloured UMAPs (batch correction maintained)
    d  OCI bar      (Sim 4, all methods)
    e  BSRS bar     (Sim 4, all methods)
    f  Integration Balance scatter (Sim 4, all methods)
    g  Sim 3 IB ranked bar     (competitive on balanced datasets)
    h  OCI vs iLISI scatter    (overcorrection vs batch-mixing trade-off)
    """
    cfg3, meta3, emb3, bdf3, cdf3 = _load_figure_data("data2_scenario1")
    cfg4, meta4, emb4, bdf4, cdf4 = _load_figure_data("data2_scenario2")
    bs4 = cfg4["batch_specific_types"]

    SEL4 = ["Scanorama", "Harmony", "scVI", "INSCT_Unsupervised", "BTCA"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.3, 12.2), dpi=150)
    fig.subplots_adjust(left=0.055, right=0.875, bottom=0.035, top=0.965)
    outer = gridspec.GridSpec(5, 1, figure=fig,
                              height_ratios=[1.65, 1.65, 1.65, 2.05, 1.45],
                              hspace=0.23)

    # a — Sim 3 cell-type UMAPs
    _draw_umap_row(
        fig, outer[0], emb3,
        plot_fn=lambda ax, df, method:
            plot_embedding_featured(ax, df, CELLTYPE_COL,
                                     cfg3["celltype_palette"], method,
                                     pt_size=cfg3["point_size"] * 1.4,
                                     alpha=cfg3["alpha"]),
        methods_subset=SEL4,
        panel_letter="a", label_x=-0.16, wspace=0.24,
    )
    fig.text(0.022, 0.86, "Sim 3  ·  balanced",
             fontsize=FS_TITLE + 1.5, fontweight="bold",
             color="#1a1a1a", rotation=90, va="center", ha="center")

    # b — Sim 4 cell-type UMAPs (batch-specific cells outlined)
    _draw_umap_row(
        fig, outer[1], emb4,
        plot_fn=lambda ax, df, method:
            plot_embedding_featured(ax, df, CELLTYPE_COL,
                                     cfg4["celltype_palette"], method,
                                     batch_specific=bs4,
                                     pt_size=cfg4["point_size"] * 1.4,
                                     alpha=cfg4["alpha"]),
        methods_subset=SEL4,
        panel_letter="b", label_x=-0.16, wspace=0.24,
    )
    fig.text(0.022, 0.64, "Sim 4  ·  batch-specific\n(cell type)",
             fontsize=FS_TITLE + 1.2, fontweight="bold",
             color="#1a1a1a", rotation=90, va="center", ha="center")

    # c — Sim 4 BATCH-coloured UMAPs (same 5 methods, proves batch correction maintained)
    _draw_umap_row(
        fig, outer[2], emb4,
        plot_fn=lambda ax, df, method:
            plot_embedding_featured(ax, df, BATCH_COL,
                                     cfg4["batch_palette"], method,
                                     pt_size=cfg4["point_size"] * 1.4,
                                     alpha=cfg4["alpha"]),
        methods_subset=SEL4,
        panel_letter="c", label_x=-0.16, wspace=0.24,
    )
    fig.text(0.022, 0.455, "Sim 4  ·  batch-specific\n(batch)",
             fontsize=FS_TITLE + 1.2, fontweight="bold",
             color="#1a1a1a", rotation=90, va="center", ha="center")

    _figure_legend(fig, cfg3["celltype_palette"], "Sim 3 types",
                   anchor=(0.882, 0.89))
    _figure_legend(fig, cfg4["celltype_palette"], "Sim 4 types",
                   anchor=(0.882, 0.66))
    _figure_legend(fig, cfg4["batch_palette"], "Batch",
                   anchor=(0.882, 0.49))

    # Compute OCI + BSRS for all methods on Sim 4
    n_cl4 = meta4[CELLTYPE_COL].nunique()
    oci_dict, bsrs_dict = {}, {}
    for m in METHODS:
        if m not in emb4:
            continue
        oci_dict[m]  = compute_oci(emb4[m], bs4, n_cl4)
        bsrs_dict[m] = compute_bsrs(emb4[m], bs4)

    # d, e, f — OCI / BSRS / Integration Balance
    gs_def = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[3],
        width_ratios=[1.0, 1.0, 1.5], wspace=0.55,
    )
    ax_d = fig.add_subplot(gs_def[0, 0]); _panel_label(ax_d, "d", x=-0.20)
    plot_oci_bar(ax_d, oci_dict, dataset_label="Sim 4")
    ax_e = fig.add_subplot(gs_def[0, 1]); _panel_label(ax_e, "e", x=-0.22)
    plot_bsrs_bar(ax_e, bsrs_dict, dataset_label="Sim 4")
    ax_f = fig.add_subplot(gs_def[0, 2]); _panel_label(ax_f, "f", x=-0.20)
    plot_integration_balance(ax_f, bdf4, cdf4,
                             title="Integration Balance (Sim 4)")

    # g, h — Sim 3 IB bar + OCI vs iLISI trade-off scatter
    gs_gh = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[4],
        width_ratios=[1.2, 1.3], wspace=0.55,
    )
    ax_g = fig.add_subplot(gs_gh[0, 0]); _panel_label(ax_g, "g", x=-0.20)
    _plot_ibfull_bar(ax_g, "data2_scenario1",
                     "IB ranking — Sim 3 (balanced)")
    ax_h = fig.add_subplot(gs_gh[0, 1]); _panel_label(ax_h, "h", x=-0.22)
    _plot_oci_ilisi_scatter(ax_h, oci_dict, bdf4,
                             "Overcorrection vs batch mixing (Sim 4)")

    _save(fig, "Fig2_simulation")


def make_main_fig3_hdc(key=None):
    """
    Fig 3 — "scCAT preserves batch-specific dendritic cell populations while
            maintaining batch mixing"

    8 panels: hero pair + dual failure-mode contrast (CT + batch) +
    quantitative + biological validation + resolution-free separability:

        a   scCAT integrated UMAP coloured by cell type, with cluster
            centroids annotated and CD141 / CD1C outlined
        b   scCAT integrated UMAP coloured by batch (same coordinates),
            proving shared cell types are well-mixed
        c   1×4 failure-mode UMAPs (Scanorama / Harmony / scVI / INSCT),
            cell type coloured, CD141 / CD1C outlined
        d   Same 1×4 failure-mode UMAPs coloured by batch — shows
            overcorrectors (Scanorama / fastMNN) merge the two donors
        e   Per-cell-type cluster purity — grouped bars (4 cell types ×
            8 methods)
        f   Marker gene DOT PLOT (4 markers × 4 cell types) on scCAT
            embedding — biological validation that cell identity is preserved
        g   1-NN same-type consistency (resolution-free separability) per
            method — only Scanorama / fastMNN fall below 0.80
        h   Trade-off scatter: iLISI (batch mixing) vs 1-NN consistency
            (bio preservation) — scCAT sits in the good-balance region
    """
    cfg, meta, emb, bdf, cdf = _load_figure_data("HDC")
    bs = cfg["batch_specific_types"]
    pt, al = cfg["point_size"], cfg["alpha"]
    celltype_order = list(cfg["celltype_palette"].keys())

    fig = plt.figure(figsize=(DOUBLE_COL + 1.3, 12.3), dpi=150)
    fig.subplots_adjust(left=0.055, right=0.875, bottom=0.035, top=0.955)
    outer = gridspec.GridSpec(5, 1, figure=fig,
                              height_ratios=[2.25, 1.35, 1.35, 2.05, 1.45],
                              hspace=0.24)

    # ── Row 0 (a, b): scCAT hero pair ─────────────────────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.20,
    )
    ax_a = fig.add_subplot(gs_top[0, 0]); _panel_label(ax_a, "a", x=-0.10)
    if "BTCA" in emb:
        plot_embedding_featured(ax_a, emb["BTCA"], CELLTYPE_COL,
                                 cfg["celltype_palette"], "BTCA",
                                 batch_specific=bs,
                                 pt_size=pt * 1.6, alpha=al)
        annotate_cluster_centroids(ax_a, emb["BTCA"], CELLTYPE_COL,
                                    cfg["celltype_palette"],
                                    fontsize=FS_TICK + 0.5)
        ax_a.set_title("scCAT — coloured by cell type",
                        fontsize=FS_TITLE + 1.5, fontweight="bold", pad=5)
    ax_b = fig.add_subplot(gs_top[0, 1]); _panel_label(ax_b, "b", x=-0.10)
    if "BTCA" in emb:
        plot_embedding_featured(ax_b, emb["BTCA"], BATCH_COL,
                                 cfg["batch_palette"], "BTCA",
                                 pt_size=pt * 1.6, alpha=al)
        annotate_cluster_centroids(ax_b, emb["BTCA"], CELLTYPE_COL,
                                    cfg["celltype_palette"],
                                    fontsize=FS_TICK + 0.5,
                                    with_box=True)
        ax_b.set_title("scCAT — coloured by batch",
                        fontsize=FS_TITLE + 1.5, fontweight="bold", pad=5)

    _figure_legend(fig, cfg["batch_palette"], "Batch",
                   anchor=(0.882, 0.93))
    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.882, 0.81))

    # ── Row 1 (c): failure-mode UMAPs coloured by cell type ──────────
    failure_methods = ["Scanorama", "Harmony", "scVI", "INSCT_Unsupervised"]
    comparison_methods = ["BTCA"] + failure_methods
    _draw_umap_row(
        fig, outer[1], emb,
        plot_fn=lambda ax, df, method:
            plot_embedding(ax, df, CELLTYPE_COL, cfg["celltype_palette"],
                            method, batch_specific=bs,
                            pt_size=pt * 0.95, alpha=al * 0.95),
        methods_subset=comparison_methods,
        panel_letter="c", label_x=-0.26, wspace=0.16,
    )
    fig.text(0.858, 0.585, "Cell type", fontsize=FS_LABEL + 0.5,
             fontweight="bold", rotation=90, va="center", ha="center",
             color="#333333")

    # ── Row 2 (d): scCAT + 4 failure-mode UMAPs, coloured by batch ──
    # scCAT is shown first as a reference so the reader can directly judge
    # whether the donor separation visible in the competitor panels is
    # typical or pathological.  Scanorama / fastMNN nearly fuse the two
    # donors (overcorrection); Harmony / scVI / INSCT preserve separation.
    def _batch_row_fn(ax, df, method):
        plot_embedding(ax, df, BATCH_COL, cfg["batch_palette"],
                       method, pt_size=pt * 0.95, alpha=al * 0.95)
        if method == "BTCA":
            disp = METHOD_DISPLAY.get("BTCA", "scCAT")
            ax.set_title(disp, fontsize=FS_TITLE + 0.5,
                         fontweight="bold", color="#000000", pad=3)

    _draw_umap_row(
        fig, outer[2], emb,
        plot_fn=_batch_row_fn,
        methods_subset=comparison_methods,
        panel_letter="d", label_x=-0.26, wspace=0.16,
    )
    fig.text(0.858, 0.468, "Batch", fontsize=FS_LABEL + 0.5,
             fontweight="bold", rotation=90, va="center", ha="center",
             color="#333333")

    # ── Row 3 (e, f): per-celltype purity bars | marker dot plot ─────
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[3],
        width_ratios=[1.65, 1.20], wspace=0.55,
    )
    ax_e = fig.add_subplot(gs_mid[0, 0]); _panel_label(ax_e, "e", x=-0.12)
    plot_per_celltype_purity_bars(
        ax_e, emb, meta,
        celltype_palette=cfg["celltype_palette"],
        celltype_order=celltype_order,
    )

    ax_f = fig.add_subplot(gs_mid[0, 1]); _panel_label(ax_f, "f", x=-0.30)
    dc_markers = ["CLEC9A", "CD1C", "LILRA4", "AXL"]
    expr_dict = load_dataset_expression("HDC", dc_markers, "INSCT_Unsupervised")
    sccat_df = emb.get("BTCA")
    if sccat_df is not None and expr_dict is not None:
        cell_to_idx = {c: i for i, c in enumerate(expr_dict["_cell_names"])}
        sccat_cells = sccat_df["cell"].astype(str).values
        match_idx = np.array([cell_to_idx.get(c, -1) for c in sccat_cells])
        valid = match_idx >= 0
        aligned_expr = {}
        for g in dc_markers:
            if expr_dict.get(g) is None:
                aligned_expr[g] = None
                continue
            arr = np.full(len(sccat_df), np.nan)
            arr[valid] = expr_dict[g][match_idx[valid]]
            aligned_expr[g] = arr
        plot_marker_dotplot(
            ax_f, aligned_expr,
            cell_type_labels=sccat_df[CELLTYPE_COL].values,
            markers=dc_markers,
            cell_type_order=celltype_order,
            celltype_palette=cfg["celltype_palette"],
            title="Marker gene expression (scCAT)",
        )

    # ── Row 4 (g, h): 1-NN separability bar | iLISI vs 1-NN scatter ──
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[4],
        width_ratios=[1.20, 1.30], wspace=0.50,
    )
    ax_g = fig.add_subplot(gs_bot[0, 0]); _panel_label(ax_g, "g", x=-0.14)
    _plot_1nn_sep_bar(ax_g, "1-NN same-type consistency (HDC)")
    ax_h = fig.add_subplot(gs_bot[0, 1]); _panel_label(ax_h, "h", x=-0.18)
    _plot_ilisi_1nn_scatter(ax_h, bdf, "iLISI vs 1-NN consistency")

    _save(fig, "Fig3_hdc")


def make_main_fig4_sc_mixology(key=None):
    """
    Fig 4 — "scCAT maintains cell-line identity during cross-platform integration"

    a  scCAT vs SPDR head-to-head  (2 large UMAPs)
    b  1×6 small UMAPs of remaining methods
    c  Silhouette ridge per method
    d  Integration Balance scatter
    """
    cfg, meta, emb, bdf, cdf = _load_figure_data("Sc_mixology")
    pt, al = cfg["point_size"], cfg["alpha"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.4, 10.5), dpi=150)
    fig.subplots_adjust(right=0.862)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.4, 1.35, 2.4],
                              hspace=0.34)

    # a — scCAT + SPDR
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            wspace=0.20)
    for i, m in enumerate(["BTCA", "SPDR"]):
        ax = fig.add_subplot(gs_a[0, i])
        if i == 0:
            _panel_label(ax, "a", x=-0.10)
        if m in emb:
            plot_embedding_featured(ax, emb[m], CELLTYPE_COL,
                                    cfg["celltype_palette"], m,
                                    pt_size=pt * 1.4, alpha=al)

    # b — small UMAPs of others
    others = [m for m in METHODS if m not in ("BTCA", "SPDR")]
    _draw_umap_row(
        fig, outer[1], emb,
        plot_fn=lambda ax, df, method:
            plot_embedding(ax, df, CELLTYPE_COL, cfg["celltype_palette"],
                           method, pt_size=pt * 0.55, alpha=al * 0.9),
        methods_subset=others,
        panel_letter="b", label_x=-0.20, wspace=0.20,
    )

    _figure_legend(fig, cfg["celltype_palette"], "Cell line",
                   anchor=(0.864, 0.75))
    _figure_legend(fig, cfg["batch_palette"], "Platform",
                   anchor=(0.864, 0.56))

    # c — silhouette ridge | d — integration balance
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                            width_ratios=[1.45, 1.40],
                                            wspace=0.45)
    ax_c = fig.add_subplot(gs_c[0, 0]); _panel_label(ax_c, "c", x=-0.16)
    plot_silhouette_ridge(ax_c, emb)
    ax_d = fig.add_subplot(gs_c[0, 1]); _panel_label(ax_d, "d", x=-0.20)
    plot_integration_balance(ax_d, bdf, cdf, title="Integration Balance")

    _save(fig, "Fig4_sc_mixology")


def make_main_fig5_pbmc(key=None):
    """
    Fig 5 — "scCAT balances condition mixing and immune cell identity in PBMC"

    a  scCAT density overlay (control vs stim)
    b  scCAT featured cell-type UMAP
    c  2×4 small UMAPs of all methods (cell-type coloured)
    d  Composition stack (per condition)
    e  Integration Balance scatter
    f  Marker gene preservation (ISG15, IFIT1, MX1, OAS1 on scCAT UMAP)
    """
    cfg, meta, emb, bdf, cdf = _load_figure_data("PBMC")
    pt, al = cfg["point_size"], cfg["alpha"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.4, 16.5), dpi=150)
    fig.subplots_adjust(right=0.862)
    outer = gridspec.GridSpec(4, 1, figure=fig,
                              height_ratios=[2.3, 2.4, 2.3, 2.1],
                              hspace=0.34)

    # a + b — scCAT density overlay | scCAT cell-type UMAP
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.4, 1.0], wspace=0.30,
    )
    ax_a = fig.add_subplot(gs_top[0, 0]); _panel_label(ax_a, "a", x=-0.10)
    if "BTCA" in emb:
        plot_density_overlay(ax_a, emb["BTCA"], BATCH_COL,
                             ["control", "stimulated"],
                             cfg["batch_palette"], "BTCA")
    ax_b = fig.add_subplot(gs_top[0, 1]); _panel_label(ax_b, "b", x=-0.15)
    if "BTCA" in emb:
        plot_embedding_featured(ax_b, emb["BTCA"], CELLTYPE_COL,
                                cfg["celltype_palette"], "BTCA",
                                pt_size=pt * 1.4, alpha=al)

    # c — 2×4 small UMAPs by cell type
    _draw_umap_grid(fig, outer[1], emb, CELLTYPE_COL,
                    cfg["celltype_palette"], None,
                    pt, al, panel_letter="c", label_x=-0.22)

    _figure_legend(fig, cfg["batch_palette"], "Condition",
                   anchor=(0.866, 0.83))
    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.866, 0.50))

    # d + e — composition stack | Integration Balance
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[1.55, 1.20], wspace=0.55,
    )
    ax_d = fig.add_subplot(gs_mid[0, 0]); _panel_label(ax_d, "d", x=-0.12)
    plot_composition_stack(ax_d, emb, meta, METHODS, cfg["celltype_palette"])
    ax_e = fig.add_subplot(gs_mid[0, 1]); _panel_label(ax_e, "e", x=-0.20)
    plot_integration_balance(ax_e, bdf, cdf, title="Integration Balance")

    # f — IFN-response marker gene preservation on scCAT UMAP
    ifn_markers = ["ISG15", "IFIT1", "MX1", "OAS1"]
    expr_dict = load_dataset_expression("PBMC", ifn_markers, "INSCT_Unsupervised")
    sccat_df = emb.get("BTCA")
    match_idx = None
    if sccat_df is not None and expr_dict is not None:
        cell_to_idx = {c: i for i, c in enumerate(expr_dict["_cell_names"])}
        sccat_cells = sccat_df["cell"].astype(str).values
        match_idx = np.array([cell_to_idx.get(c, -1) for c in sccat_cells])

    gs_f = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[3],
                                            wspace=0.32)
    for i, g in enumerate(ifn_markers):
        ax = fig.add_subplot(gs_f[0, i])
        if i == 0:
            _panel_label(ax, "f", x=-0.20)
        if sccat_df is None or expr_dict is None or match_idx is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center"); _nature_embed_style(ax, g)
            continue
        if expr_dict.get(g) is None:
            plot_marker_feature(ax, sccat_df, None, g); continue
        expr_arr = np.full(len(sccat_df), np.nan)
        valid = match_idx >= 0
        expr_arr[valid] = expr_dict[g][match_idx[valid]]
        plot_marker_feature(ax, sccat_df, expr_arr, g, pt_size=4)

    _save(fig, "Fig5_pbmc")


def make_main_fig6_lung(key=None):
    """
    Fig 6 — "scCAT scales to complex multi-batch atlas-level integration"

    a  scCAT featured pair (batch | cell type)
    b  1×7 small UMAPs of other methods (cell-type coloured)
    c  Per-batch local mixing heatmap (16 batches × 8 methods)
    d  Integration Balance scatter
    """
    cfg, meta, emb, bdf, cdf = _load_figure_data("Lung")
    pt, al = cfg["point_size"], cfg["alpha"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.5, 14.5), dpi=150)
    fig.subplots_adjust(right=0.858)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.5, 1.45, 3.2],
                              hspace=0.34)

    # a — scCAT pair
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            wspace=0.22)
    ax_a1 = fig.add_subplot(gs_a[0, 0]); _panel_label(ax_a1, "a", x=-0.10)
    if "BTCA" in emb:
        plot_embedding_featured(ax_a1, emb["BTCA"], BATCH_COL,
                                cfg["batch_palette"], "BTCA",
                                pt_size=pt * 1.6, alpha=al)
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    if "BTCA" in emb:
        plot_embedding_featured(ax_a2, emb["BTCA"], CELLTYPE_COL,
                                cfg["celltype_palette"], "BTCA",
                                pt_size=pt * 1.6, alpha=al)

    # b — 1×7 thumbnails of other methods
    other_methods = [m for m in METHODS if m != "BTCA"]
    _draw_umap_row(
        fig, outer[1], emb,
        plot_fn=lambda ax, df, method:
            plot_embedding(ax, df, CELLTYPE_COL, cfg["celltype_palette"],
                           method, pt_size=pt * 0.55, alpha=al * 0.85),
        methods_subset=other_methods,
        panel_letter="b", label_x=-0.22, wspace=0.20,
    )

    bp = cfg["batch_palette"]
    _figure_legend(fig,
                   {k: v for k, v in bp.items() if not k.startswith(("A","B"))},
                   "Numeric", anchor=(0.860, 0.87))
    _figure_legend(fig,
                   {k: v for k, v in bp.items() if k.startswith("A")},
                   "A-series", anchor=(0.860, 0.79))
    _figure_legend(fig,
                   {k: v for k, v in bp.items() if k.startswith("B")},
                   "B-series", anchor=(0.860, 0.71))
    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.860, 0.50), ncol_max=18)

    # c — per-batch heatmap | d — Integration Balance
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                            width_ratios=[1.5, 1.15], wspace=0.45)
    ax_c = fig.add_subplot(gs_c[0, 0]); _panel_label(ax_c, "c", x=-0.16)
    plot_per_batch_mixing(ax_c, emb, meta, methods=RANK_METHODS,
                          k=30, subsample_n=8000)
    ax_d = fig.add_subplot(gs_c[0, 1]); _panel_label(ax_d, "d", x=-0.20)
    plot_integration_balance(ax_d, bdf, cdf, title="Integration Balance")

    _save(fig, "Fig6_lung")


# ─────────────────────────────────────────────────────────────────────────────
# 10c.  NEW MAIN FIGURES (per the 6-figure mechanism-driven layout)
#       Fig 4 = Ablation analysis as a main figure
#       Fig 5 = Sc_mixology + PBMC combined
#       Fig 6 = Mouse Lung + cross-dataset overall summary
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CSV = BASE_DIR / "experiments" / "results" / "ablation_results.csv"
RUNTIME_CSV  = BASE_DIR / "experiments" / "results" / "runtime_results.csv"

ABL_ORDER = ["full", "noConf", "noFilter", "fixedMargin", "noBSP"]
ABL_DISPLAY = {
    "full":        "scCAT-full",
    "noConf":      "−Conf",
    "noFilter":    "−Filter",
    "fixedMargin": "−AdaptMargin",
    "noBSP":       "−BSP",
}
ABL_COLORS = {
    "full":        METHOD_COLORS["BTCA"],
    "noConf":      "#5B8DD9",
    "noFilter":    "#64B5CD",
    "fixedMargin": "#DDA0DD",
    "noBSP":       "#E06C9F",
}


def _ablation_subset(ablation_df, dataset):
    sub = ablation_df[ablation_df["dataset"] == dataset]
    if sub.empty:
        return None
    return sub.set_index("config").reindex(ABL_ORDER).reset_index()


def _plot_ablation_metric_bars(ax, sub, metric_col, ylabel, title,
                                higher_better=True, ylim=None):
    cfgs = sub["config"].tolist()
    vals = sub[metric_col].astype(float).values
    xs   = np.arange(len(cfgs))
    cols = [ABL_COLORS.get(c, "#888") for c in cfgs]
    # 'full' bar gets a thicker dark edge for emphasis (replaces floating star)
    edges  = ["#1A6E1A" if c == "full" else "white" for c in cfgs]
    lws    = [1.6 if c == "full" else 0.5 for c in cfgs]
    bars = ax.bar(xs, vals, color=cols, alpha=0.92,
                   edgecolor=edges, linewidth=lws, width=0.72)
    span = max(abs(vals.max() if higher_better else vals.max()),
                abs(vals.min())) if len(vals) else 1.0
    for b, v in zip(bars, vals):
        if not np.isfinite(v):
            continue
        yoff = abs(span * 0.03)
        y = v + yoff if v >= 0 else v - yoff
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, y, f"{v:.2f}",
                ha="center", va=va, fontsize=FS_ANNOT)
    ax.set_xticks(xs)
    ax.set_xticklabels([ABL_DISPLAY.get(c, c) for c in cfgs],
                        fontsize=FS_TICK, rotation=30, ha="right")
    # Bold + color the 'full' tick label so the baseline stands out at a glance
    for tick, c in zip(ax.get_xticklabels(), cfgs):
        if c == "full":
            tick.set_fontweight("bold")
            tick.set_color(ABL_COLORS["full"])
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif higher_better and vals.min() >= 0:
        ax.set_ylim(0, max(vals.max() * 1.30, 0.1))


def _plot_ablation_design(ax, datasets_used):
    """Knockout matrix: 5 ablation configurations × 4 scCAT modules.
    Green ✓ = module enabled, red ✗ = module disabled.
    Uses mathtext for Greek-letter parameters (matplotlib-rendered, so
    rendering is independent of system font Unicode support).
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Module column headers (clean, no parameter symbol — moved to footer key)
    modules = [
        "Confidence\nweighting",
        "Confidence\nfilter",
        "Adaptive\nmargin",
        "Batch-specific\nprotection",
    ]
    # Each config's disabled module index (None = nothing disabled, i.e. full)
    # Effect text uses matplotlib mathtext so symbols render reliably
    configs = [
        ("scCAT-full",     "full",        None, "baseline (all four modules)"),
        ("−Conf",          "noConf",      0,    r"$\gamma = 0$"),
        ("−Filter",        "noFilter",    1,    r"$\tau^+ = \tau^- = 0$"),
        ("−AdaptMargin",   "fixedMargin", 2,    r"$\lambda_\rho = \lambda_b = 0$"),
        ("−BSP",           "noBSP",       3,    r"$\mu_{\mathrm{rare}} = 0$"),
    ]

    # Title (top, very tight)
    ax.text(0.50, 0.97, "Module knockout design",
            ha="center", va="top", fontsize=FS_TITLE + 1.5,
            fontweight="bold", color="#222")

    # Use FULL vertical space — minimal internal padding so the perceived
    # gap to the next gridspec row matches all other inter-row gaps
    n_cfg = len(configs); n_mod = len(modules)
    table_top = 0.82; table_bot = 0.14      # leave room for footer key dots
    table_left = 0.02; table_right = 0.78
    extra_col = 0.21                          # right margin for effect text

    # 1 column for config name + n_mod columns for modules + 1 for effect
    col_widths = [(table_right - table_left) / (n_mod + 1.4)] * (n_mod + 1)
    col_widths[0] *= 1.4
    col_x = [table_left + sum(col_widths[:i]) + col_widths[i] / 2
              for i in range(n_mod + 1)]

    row_h = (table_top - table_bot) / (n_cfg + 1)
    row_y_header = table_top - row_h * 0.5
    row_y = [table_top - row_h * (i + 1.5) for i in range(n_cfg)]

    # Light alternating row background
    for i in range(n_cfg):
        if i == 0:
            # Highlight full-model row
            ax.add_patch(Rectangle(
                (table_left, row_y[i] - row_h * 0.42),
                table_right + extra_col - table_left, row_h * 0.85,
                facecolor=ABL_COLORS["full"], edgecolor=ABL_COLORS["full"],
                alpha=0.10, lw=0,
            ))
        elif i % 2 == 0:
            ax.add_patch(Rectangle(
                (table_left, row_y[i] - row_h * 0.42),
                table_right + extra_col - table_left, row_h * 0.85,
                facecolor="#F4F4F4", edgecolor="none",
            ))

    # ── Header row ── (cleaner — just module names, no parameter symbol)
    ax.text(col_x[0], row_y_header, "Configuration",
            ha="center", va="center", fontsize=FS_ANNOT + 1,
            fontweight="bold", color="#222")
    for j, mod_name in enumerate(modules):
        ax.text(col_x[j + 1], row_y_header,
                mod_name, ha="center", va="center",
                fontsize=FS_ANNOT + 0.5, fontweight="bold", color="#222",
                linespacing=1.05)
    ax.text(table_right + extra_col / 2, row_y_header, "Effect tested",
            ha="center", va="center", fontsize=FS_ANNOT + 1,
            fontweight="bold", color="#222")

    # Horizontal separator under header
    ax.plot([table_left, table_right + extra_col],
            [table_top - row_h * 0.98] * 2,
            color="#888", lw=0.6)

    # ── Config rows ──
    for i, (cfg_label, cfg_key, ablated_idx, effect) in enumerate(configs):
        y = row_y[i]
        # Config name (colored)
        ax.text(col_x[0], y, cfg_label, ha="center", va="center",
                fontsize=FS_ANNOT + 0.5, fontweight="bold",
                color=ABL_COLORS[cfg_key])

        # Module on/off cells
        for k in range(n_mod):
            xk = col_x[k + 1]
            if ablated_idx is None or k != ablated_idx:
                # Module enabled
                ax.scatter(xk, y, s=95, marker="o",
                            color="#2D9E2D", edgecolors="#1A6E1A",
                            linewidths=0.6, zorder=4)
                ax.text(xk, y, "✓", ha="center", va="center",
                        fontsize=FS_ANNOT + 0.5, fontweight="bold",
                        color="white", zorder=5)
            else:
                # Module disabled
                ax.scatter(xk, y, s=95, marker="o",
                            color="#E15759", edgecolors="#9E1E20",
                            linewidths=0.6, zorder=4)
                ax.text(xk, y, "✗", ha="center", va="center",
                        fontsize=FS_ANNOT + 1, fontweight="bold",
                        color="white", zorder=5)

        # Effect text on the right (rendered with mathtext)
        ax.text(table_right + extra_col / 2, y, effect,
                ha="center", va="center", fontsize=FS_ANNOT + 0.5,
                color="#444")

    # ── Footer key with colored markers (placed near panel bottom) ──
    # key_y is set high enough that the scatter dot (radius ≈ 0.07 in
    # axis-y units at s=85) fits fully inside the panel.  clip_on=False
    # is added as a safety net so the dot renders fully even if it would
    # otherwise be clipped at the axes boundary.
    key_y = 0.075
    # Green active marker
    ax.scatter(0.08, key_y, s=85, marker="o",
                color="#2D9E2D", edgecolors="#1A6E1A",
                linewidths=0.6, zorder=4, clip_on=False)
    ax.text(0.08, key_y, "✓", ha="center", va="center",
            fontsize=FS_ANNOT + 0.5, fontweight="bold",
            color="white", zorder=5, clip_on=False)
    ax.text(0.105, key_y, "module active",
            ha="left", va="center", fontsize=FS_ANNOT,
            color="#333", clip_on=False)
    # Red disabled marker
    ax.scatter(0.32, key_y, s=85, marker="o",
                color="#E15759", edgecolors="#9E1E20",
                linewidths=0.6, zorder=4, clip_on=False)
    ax.text(0.32, key_y, "✗", ha="center", va="center",
            fontsize=FS_ANNOT + 1, fontweight="bold",
            color="white", zorder=5, clip_on=False)
    ax.text(0.345, key_y, "module disabled",
            ha="left", va="center", fontsize=FS_ANNOT,
            color="#333", clip_on=False)
    # Datasets footer (right side)
    ax.text(0.99, key_y,
            f"Tested on: {'  +  '.join(DATASET_CONFIGS[d]['label'] for d in datasets_used)}",
            ha="right", va="center", fontsize=FS_ANNOT,
            color="#666", style="italic", clip_on=False)


def _plot_balance_score_heatmap(ax, ablation_df, datasets):
    """rows = configs, columns = (S_batch, S_bio, Balance) per dataset."""
    # Build matrix of [n_configs, 3*n_datasets]
    DS_SHORT = {
        "data2_scenario2": "Sim 4",
        "HDC": "HDC",
    }
    col_groups = []
    col_labels = []
    for ds in datasets:
        ds_short = DS_SHORT.get(ds, DATASET_CONFIGS[ds]["label"])
        col_groups.extend([(ds, "batch"), (ds, "bio"), (ds, "balance")])
        col_labels.extend([
            f"$S_{{batch}}$\n({ds_short})",
            f"$S_{{bio}}$\n({ds_short})",
            f"IB\n({ds_short})",
        ])

    n_rows = len(ABL_ORDER); n_cols = len(col_groups)
    M = np.full((n_rows, n_cols), np.nan)

    for i, cfg in enumerate(ABL_ORDER):
        for j, (ds, key) in enumerate(col_groups):
            row = ablation_df[(ablation_df["dataset"] == ds) &
                              (ablation_df["config"] == cfg)]
            if row.empty:
                continue
            r = row.iloc[0]
            ari = float(r.get("ARI", np.nan))
            nmi = float(r.get("NMI", np.nan))
            asw_ct = float(r.get("ASW_cell_type", np.nan))
            mix = float(r.get("knn_mixing", np.nan))
            ib = float(r.get("IB", np.nan))
            s_batch = mix
            s_bio_vals = [v for v in (ari, nmi, asw_ct) if np.isfinite(v)]
            s_bio = float(np.mean(s_bio_vals)) if s_bio_vals else np.nan
            if key == "batch":   M[i, j] = s_batch
            elif key == "bio":    M[i, j] = s_bio
            else:                  M[i, j] = ib

    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            if np.isfinite(v):
                txt = "white" if v < 0.20 or v > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=FS_ANNOT, color=txt)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=FS_TICK, rotation=0, ha="center",
                        linespacing=1.2)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([ABL_DISPLAY[c] for c in ABL_ORDER],
                        fontsize=FS_TICK)
    # Vertical separator between dataset groups (more prominent)
    for ds_idx in range(len(datasets) - 1):
        ax.axvline(ds_idx * 3 + 2.5, color="#555", lw=1.2)
    ax.set_title("Balance scores per ablation configuration",
                  fontsize=FS_TITLE, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def _plot_module_contribution(ax, ablation_df, dataset):
    """Horizontal dumbbell chart showing each module's contribution to IB.

    For each ablated config, one horizontal dumbbell:
        gray dot     = full-model IB
        coloured dot = ablated IB
        line connecting = module contribution magnitude

    OCI is shown directly in the OCI-bar panels (b/e), so this dumbbell
    is kept IB-only for clarity (no risk of label overlap between metrics)."""
    sub = _ablation_subset(ablation_df, dataset)
    if sub is None:
        return
    full_ib = float(sub[sub["config"] == "full"]["IB"].iloc[0])

    ablated_cfgs = [c for c in ABL_ORDER if c != "full"]
    n = len(ablated_cfgs)

    # Pre-compute everything so we can pick a safe label position
    rows = []
    for c in ablated_cfgs:
        r = sub[sub["config"] == c]
        if r.empty:
            rows.append(None); continue
        rows.append(float(r["IB"].iloc[0]))

    finite_vals = [v for v in rows if v is not None] + [full_ib]
    x_max_data  = max(finite_vals)

    # Dumbbells
    for i, (c, abl_ib) in enumerate(zip(ablated_cfgs, rows)):
        if abl_ib is None:
            continue
        color = ABL_COLORS[c]

        # Connection line
        ax.plot([full_ib, abl_ib], [i, i],
                color=color, lw=2.4, alpha=0.75, zorder=2,
                solid_capstyle="round")
        # full marker (gray, smaller)
        ax.scatter(full_ib, i, color="#888", s=80, zorder=3,
                    edgecolor="white", linewidth=0.6)
        # ablated marker (color, larger)
        ax.scatter(abl_ib, i, color=color, s=130, zorder=4,
                    edgecolor="#222", linewidth=0.7)
        # Δ label — placed at right of axis (always in same column, no overlap)
        d_ib = abl_ib - full_ib
        sign = "+" if d_ib >= 0 else "−"
        ax.text(1.02, i, f"Δ = {sign}{abs(d_ib):.2f}",
                 color=color, va="center", ha="left",
                 fontsize=FS_ANNOT + 0.5, fontweight="bold",
                 transform=ax.get_yaxis_transform(),
                 clip_on=False)

    # Reference vertical line at full IB (subtle dashed)
    ax.axvline(full_ib, color="#888", lw=0.6, ls="--", alpha=0.7, zorder=1)

    ax.set_yticks(range(n))
    ax.set_yticklabels([ABL_DISPLAY[c] for c in ablated_cfgs],
                        fontsize=FS_TICK + 0.3, fontweight="bold")
    for tick, c in zip(ax.get_yticklabels(), ablated_cfgs):
        tick.set_color(ABL_COLORS[c])
    # Auto x range with a bit of left/right padding (Δ labels live OUTSIDE
    # the right axis edge so we don't need to allocate xlim space for them)
    x_lo = min(0, min([v for v in rows if v is not None] + [full_ib]) - 0.05)
    x_hi = max(1.0, x_max_data + 0.05)
    ax.set_xlim(x_lo, x_hi)
    # Small top headroom for the single 'full = X.XX' label on the dashed line
    ax.set_ylim(-0.55, n - 0.35)
    ax.invert_yaxis()
    ax.set_xlabel("Integration Balance  (full → ablated)",
                   fontsize=FS_LABEL, labelpad=2)
    ax.set_title(f"Module contribution to IB — "
                  f"{DATASET_CONFIGS[dataset]['label']}",
                  fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ── Single 'full = X.XX' label at the top of the dashed reference line ──
    # The visual convention is self-explanatory: gray dot (always on the
    # dashed line) = full, coloured dot = ablated value.
    ax.text(full_ib, -0.42, f"full = {full_ib:.2f}",
            color="#555", fontsize=FS_ANNOT + 0.5, fontweight="bold",
            ha="center", va="bottom", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                      edgecolor="#888", linewidth=0.5, alpha=0.95),
            zorder=10)


def make_main_fig4_ablation(key=None):
    """
    Fig 4 — "Confidence weighting, adaptive margins and batch-specific
            protection jointly contribute to robust integration"

    Promoted ablation analysis (previously Suppl S6) as a main figure
    because module ablations are the strongest evidence that scCAT's
    design choices matter mechanistically.

    a  Ablation design schematic (5 variants)
    b  Sim 4 — OCI bar across 5 configs
    c  Sim 4 — BSRS bar across 5 configs
    d  HDC  — OCI bar across 5 configs
    e  HDC  — BSRS bar across 5 configs
    f  Balance score heatmap (5 configs × {S_batch, S_bio, IB} × 2 datasets)
    g  Module contribution summary (ΔIB and ΔOCI per ablated module on Sim 4)
    """
    if not ABLATION_CSV.exists():
        print(f"[Fig 4] {ABLATION_CSV.name} not found — run experiments first")
        return
    abl = pd.read_csv(ABLATION_CSV)
    datasets = ["data2_scenario2", "HDC"]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.0, 11.5), dpi=150)

    # Two independent GridSpecs give us explicit control over the gap
    # between panel a and the metric rows below — guaranteeing it matches
    # the gap between rows e/f/g (HDC) and h (heatmap).
    #
    # Layout in figure coordinates (0 = bottom, 1 = top):
    #   panel a:        top=0.965, bottom=0.86   (height = 0.105)
    #   --- gap of 0.05 ---
    #   row b/c/d:      top=0.81,  bottom=~0.575
    #   --- gap of 0.05 ---  (controlled by hspace inside gs_rest)
    #   row e/f/g:      ~0.525 to ~0.290
    #   --- gap of 0.05 ---
    #   row h:          ~0.240 to 0.045
    gs_a = gridspec.GridSpec(1, 1, figure=fig,
                              top=0.965, bottom=0.81,
                              left=0.06, right=0.97)
    gs_rest = gridspec.GridSpec(3, 1, figure=fig,
                                 top=0.76, bottom=0.045,
                                 left=0.06, right=0.97,
                                 hspace=0.28)   # ≈ 0.05 of fig between rows
    # NOTE: a-b gap = 0.81 − 0.76 = 0.05 of figure height = e-h gap ✓

    # Row a — design
    ax_a = fig.add_subplot(gs_a[0]); _panel_label(ax_a, "a", x=-0.04)
    _plot_ablation_design(ax_a, datasets)

    # Row b/c/d — Sim 4 OCI + BSRS + module contribution (3 panels)
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_rest[0],
        width_ratios=[1.0, 1.0, 1.4], wspace=0.45,
    )
    s4 = _ablation_subset(abl, "data2_scenario2")
    if s4 is not None:
        ax_b = fig.add_subplot(gs_row2[0, 0]); _panel_label(ax_b, "b", x=-0.22)
        _plot_ablation_metric_bars(ax_b, s4, "OCI",
                                    "Overcorrection index (↓)",
                                    "Sim 4 — OCI",
                                    higher_better=False, ylim=(0, 1.05))
        ax_c = fig.add_subplot(gs_row2[0, 1]); _panel_label(ax_c, "c", x=-0.22)
        _plot_ablation_metric_bars(ax_c, s4, "BSRS",
                                    "Batch-specific retention (↑)",
                                    "Sim 4 — BSRS",
                                    higher_better=True)
        ax_d = fig.add_subplot(gs_row2[0, 2]); _panel_label(ax_d, "d", x=-0.16)
        _plot_module_contribution(ax_d, abl, "data2_scenario2")

    # Row e/f/g — HDC OCI + BSRS + module contribution
    gs_row3 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_rest[1],
        width_ratios=[1.0, 1.0, 1.4], wspace=0.45,
    )
    hdc = _ablation_subset(abl, "HDC")
    if hdc is not None:
        ax_e = fig.add_subplot(gs_row3[0, 0]); _panel_label(ax_e, "e", x=-0.22)
        _plot_ablation_metric_bars(ax_e, hdc, "OCI",
                                    "Overcorrection index (↓)",
                                    "HDC — OCI",
                                    higher_better=False, ylim=(0, 1.05))
        ax_f = fig.add_subplot(gs_row3[0, 1]); _panel_label(ax_f, "f", x=-0.22)
        _plot_ablation_metric_bars(ax_f, hdc, "BSRS",
                                    "Batch-specific retention (↑)",
                                    "HDC — BSRS",
                                    higher_better=True)
        ax_g = fig.add_subplot(gs_row3[0, 2]); _panel_label(ax_g, "g", x=-0.16)
        _plot_module_contribution(ax_g, abl, "HDC")

    # Row h — balance score heatmap (full width)
    ax_h = fig.add_subplot(gs_rest[2]); _panel_label(ax_h, "h", x=-0.04)
    _plot_balance_score_heatmap(ax_h, abl, datasets)

    _save(fig, "Fig4_ablation")


def make_main_fig5_scmix_pbmc(key=None):
    """
    Fig 5 — "scCAT maintains biological identity in cross-platform and
            condition-associated integration tasks"

    Sc_mixology + PBMC combined into one main figure, 8 panels.

    a   Sc_mixology — scCAT vs SPDR head-to-head (1×2 large, cell-line coloured)
    b   Sc_mixology — silhouette ridge per method (all 8 methods, compact)
    c   PBMC        — scCAT density overlay (control vs stim)
    d   PBMC        — cell-type composition stack per condition
    e   PBMC        — Integration Balance scatter (all 8 methods)
    f   PBMC        — 1×4 marker preservation on scCAT UMAP (ISG15/IFIT1/IFIT3/GNLY)
    g   Sc_mixology — ranked IB bar (multi-seed phase-2 mean ± SD)
    h   PBMC        — per-cell-type IFN-β DE concordance (4 methods × 8 cell types)
    """
    cfg_m, meta_m, emb_m, bdf_m, cdf_m = _load_figure_data("Sc_mixology")
    cfg_p, meta_p, emb_p, bdf_p, cdf_p = _load_figure_data("PBMC")

    fig = plt.figure(figsize=(DOUBLE_COL + 1.4, 12.4), dpi=150)
    fig.subplots_adjust(left=0.055, right=0.875, bottom=0.035, top=0.965)
    outer = gridspec.GridSpec(5, 1, figure=fig,
                              height_ratios=[2.15, 1.35, 1.74, 1.65, 1.35],
                              hspace=0.32)

    # ── Top: Sc_mixology ─────────────────────────────────────────────
    # Row a: scCAT vs SPDR head-to-head (1×2 LARGE)
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            wspace=0.22)
    for i, m in enumerate(["BTCA", "SPDR"]):
        ax = fig.add_subplot(gs_a[0, i])
        if i == 0:
            _panel_label(ax, "a", x=-0.10)
        if m in emb_m:
            plot_embedding_featured(ax, emb_m[m], CELLTYPE_COL,
                                     cfg_m["celltype_palette"], m,
                                     pt_size=cfg_m["point_size"] * 1.4,
                                     alpha=cfg_m["alpha"])

    # Row b: silhouette ridge (all 8 methods, full width)
    ax_b = fig.add_subplot(outer[1]); _panel_label(ax_b, "b", x=-0.05)
    plot_silhouette_ridge(ax_b, emb_m)

    _figure_legend(fig, cfg_m["celltype_palette"], "Cell line",
                   anchor=(0.883, 0.88))
    _figure_legend(fig, cfg_m["batch_palette"], "Platform",
                   anchor=(0.883, 0.72))

    # ── Bottom: PBMC ─────────────────────────────────────────────────
    # Row c/d: density overlay + composition stack
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                            width_ratios=[1.0, 1.4],
                                            wspace=0.32)
    ax_c = fig.add_subplot(gs_c[0, 0]); _panel_label(ax_c, "c", x=-0.16)
    if "BTCA" in emb_p:
        plot_density_overlay(ax_c, emb_p["BTCA"], BATCH_COL,
                              ["control", "stimulated"],
                              cfg_p["batch_palette"], "BTCA")
    ax_d = fig.add_subplot(gs_c[0, 1]); _panel_label(ax_d, "d", x=-0.16)
    plot_composition_stack(ax_d, emb_p, meta_p, METHODS,
                            cfg_p["celltype_palette"])

    # Row e/f: Integration Balance + 4 marker plots
    markers = ["ISG15", "IFIT1", "IFIT3", "GNLY"]
    marker_titles = {
        "ISG15": "ISG15  (IFN)",
        "IFIT1": "IFIT1  (IFN)",
        "IFIT3": "IFIT3  (IFN)",
        "GNLY":  "GNLY  (NK cell)",
    }
    gs_e = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[3],
        width_ratios=[1.30, 1.00, 1.00, 1.00, 1.00],
        wspace=0.32,
    )
    ax_e = fig.add_subplot(gs_e[0, 0]); _panel_label(ax_e, "e", x=-0.22)
    plot_integration_balance(ax_e, bdf_p, cdf_p,
                              title="PBMC Integration Balance")

    expr_dict = load_dataset_expression("PBMC", markers, "INSCT_Unsupervised")
    sccat_df = emb_p.get("BTCA")
    match_idx = None
    if sccat_df is not None and expr_dict is not None:
        cell_to_idx = {c: i for i, c in enumerate(expr_dict["_cell_names"])}
        sccat_cells = sccat_df["cell"].astype(str).values
        match_idx = np.array([cell_to_idx.get(c, -1) for c in sccat_cells])

    for i, g in enumerate(markers):
        ax = fig.add_subplot(gs_e[0, i + 1])
        if i == 0:
            _panel_label(ax, "f", x=-0.28)
        if sccat_df is None or expr_dict is None or match_idx is None \
           or expr_dict.get(g) is None:
            plot_marker_feature(ax, sccat_df if sccat_df is not None else pd.DataFrame(),
                                 None, marker_titles.get(g, g))
            continue
        expr_arr = np.full(len(sccat_df), np.nan)
        valid = match_idx >= 0
        expr_arr[valid] = expr_dict[g][match_idx[valid]]
        plot_marker_feature(ax, sccat_df, expr_arr,
                             marker_titles.get(g, g), pt_size=4)

    _figure_legend(fig, cfg_p["batch_palette"], "Condition",
                   anchor=(0.883, 0.44))

    # ── Row 4 (g, h): Sc_mix IB bar | PBMC DE concordance ───────────
    gs_gh = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[4],
        width_ratios=[1.10, 1.50], wspace=0.48,
    )
    ax_g = fig.add_subplot(gs_gh[0, 0]); _panel_label(ax_g, "g", x=-0.14)
    _plot_ibfull_bar(ax_g, "Sc_mixology",
                     "Sc_mixology IB (multi-seed)")
    ax_h = fig.add_subplot(gs_gh[0, 1]); _panel_label(ax_h, "h", x=-0.14)
    _plot_de_concordance_bar(ax_h, "IFN-β DE concordance (PBMC)")

    _save(fig, "Fig5_scmix_pbmc")


def _compute_cross_dataset_balance(main_datasets):
    """Return DataFrame with columns [method, dataset, s_batch, s_bio, IB]."""
    rows = []
    for dk in main_datasets:
        cfg = DATASET_CONFIGS[dk]
        try:
            bdf = read_metric(BASE_DIR / cfg["batch_metric"], METHODS)
            cdf = read_metric(BASE_DIR / cfg["clust_metric"], METHODS)
        except Exception:
            continue
        for m in METHODS:
            sb, sbio, ib = compute_integration_balance(bdf, cdf, m)
            rows.append({
                "method": m, "dataset": dk,
                "s_batch": sb, "s_bio": sbio, "IB": ib,
            })
    return pd.DataFrame(rows)


def _plot_overall_balance_heatmap(ax, balance_df, main_datasets):
    method_list = RANK_METHODS
    n_m, n_d = len(method_list), len(main_datasets)
    M = np.full((n_m, n_d), np.nan)
    for j, ds in enumerate(main_datasets):
        for i, m in enumerate(method_list):
            row = balance_df[(balance_df["method"] == m) &
                              (balance_df["dataset"] == ds)]
            if not row.empty and np.isfinite(row["IB"].iloc[0]):
                M[i, j] = row["IB"].iloc[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(n_m):
        for j in range(n_d):
            v = M[i, j]
            if np.isfinite(v):
                txt = "white" if v < 0.20 or v > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=FS_ANNOT, color=txt)
    ax.set_xticks(range(n_d))
    ax.set_xticklabels([DATASET_CONFIGS[d]["label"] for d in main_datasets],
                        fontsize=FS_TICK, rotation=22, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in method_list],
                        fontsize=FS_TICK)
    ax.set_title("Integration Balance across main datasets",
                  fontsize=FS_TITLE, pad=4)
    if "BTCA" in method_list:
        i = method_list.index("BTCA")
        ax.add_patch(Rectangle((-0.5, i - 0.5), n_d, 1,
                                fill=False, edgecolor=METHOD_COLORS["BTCA"],
                                linewidth=0.9, clip_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def _plot_average_rank(ax, balance_df, main_datasets):
    """Per-method average rank of IB across main datasets (1 = best).
    All numeric labels are placed at a fixed x-column to the RIGHT of the
    longest bar — this prevents the previous behaviour where a short label
    was rendered directly on top of a long lollipop line."""
    pivot = balance_df.pivot(index="method", columns="dataset", values="IB")
    # rank within each dataset (ascending=False so higher IB = rank 1)
    ranks = pivot.rank(axis=0, ascending=False, method="average")
    mean_rank = ranks[main_datasets].mean(axis=1)
    order = mean_rank.sort_values().index.tolist()
    vals = [mean_rank.loc[m] for m in order]
    colors = [METHOD_COLORS.get(m, "#888") for m in order]
    ys = np.arange(len(order))

    # Lollipops first (no labels yet)
    for i, (mthd, v, c) in enumerate(zip(order, vals, colors)):
        ax.plot([0, v], [i, i], color=c, lw=1.6, alpha=0.78)
        ax.scatter(v, i, color=c, s=60, zorder=4,
                    edgecolors="white", linewidth=0.7)

    # All numeric labels in one right-aligned column (well past longest bar)
    max_v = max(vals)
    label_x = max_v + 0.6   # fixed offset past the longest bar
    for i, (mthd, v, c) in enumerate(zip(order, vals, colors)):
        ax.text(label_x, i, f"{v:.2f}",
                 va="center", ha="left",
                 fontsize=FS_ANNOT + 0.3, color=c,
                 fontweight="bold" if mthd == "BTCA" else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels([METHOD_DISPLAY.get(m, m) for m in order],
                        fontsize=FS_TICK)
    ax.set_xlabel(f"Mean rank across {len(main_datasets)} datasets (lower = better)",
                   fontsize=FS_LABEL, labelpad=2)
    ax.set_title("Average rank of Integration Balance",
                  fontsize=FS_TITLE, pad=4)
    # xlim leaves room for the right-aligned label column
    ax.set_xlim(0, max_v + 1.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if "BTCA" in order:
        sccat_idx = order.index("BTCA")
        ax.axhspan(sccat_idx - 0.45, sccat_idx + 0.45,
                    color=METHOD_COLORS["BTCA"], alpha=0.10, lw=0)


def _plot_runtime_summary(ax):
    """Mini runtime bar chart from runtime_results.csv (legacy version,
    kept for backwards compatibility; superseded by _plot_efficiency_8method)."""
    if not RUNTIME_CSV.exists():
        ax.text(0.5, 0.5, "Runtime data\nnot available",
                 transform=ax.transAxes, ha="center", va="center",
                 fontsize=FS_TICK)
        ax.axis("off")
        return
    rt = pd.read_csv(RUNTIME_CSV).sort_values("n_cells").reset_index(drop=True)
    xs = np.arange(len(rt))
    ax.bar(xs, rt["runtime_sec"], color=METHOD_COLORS["BTCA"],
            alpha=0.88, edgecolor="white", linewidth=0.5, width=0.7)
    for x, v in zip(xs, rt["runtime_sec"]):
        ax.text(x, v + rt["runtime_sec"].max() * 0.025, f"{v:.0f}s",
                ha="center", va="bottom", fontsize=FS_ANNOT)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [DATASET_CONFIGS.get(d, {}).get("label", d) for d in rt["dataset"]],
        fontsize=FS_TICK, rotation=20, ha="right",
    )
    ax.set_ylabel("Runtime (s, CPU)", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("scCAT runtime", fontsize=FS_TITLE, pad=4)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# scCAT scaling on the 10x 73k PBMC downsampling benchmark (panel f, g).
# We deliberately do NOT plot cross-method wall-clock here because the seven
# integration baselines (Scanorama / fastMNN / INSCT / DESC / scBCN / DeepBID /
# SPDR) are implemented in mixed languages (Python / R / TensorFlow / PyTorch)
# and require heterogeneous hardware setups; a fair single-machine wall-clock
# comparison across all seven is out of scope for this study.  Instead, Methods
# §4.7 cites Luecken 2022 Nat Methods for a literature-based comparison of
# integration-method runtimes, and this panel reports scCAT's empirical
# scaling on a uniformly subsampled 10x 73k PBMC dataset (CPU only).
# ─────────────────────────────────────────────────────────────────────────────

SCCAT_DOWN_CSV = HERE.parent / "experiments" / "results" / "scCAT_downsampling.csv" \
    if False else None  # populated below after HERE is defined elsewhere

# Resolve the scCAT scaling CSV path lazily — the experiments dir lives next to BASE_DIR
SCCAT_SCALING_CSV = (BASE_DIR / "experiments" / "results" / "scCAT_downsampling.csv")


def _load_scCAT_scaling():
    """Load scCAT-only scaling data from run_scCAT_downsampling.py output.
    Returns DataFrame with columns n_cells, Time(s), PeakMemory(gb), or
    empty DataFrame if file missing."""
    if not SCCAT_SCALING_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SCCAT_SCALING_CSV).sort_values("n_cells").reset_index(drop=True)
    return df


def plot_scCAT_scaling(ax, metric="time"):
    """Log-log scaling curve for scCAT on the 10x 73k PBMC downsampling
    benchmark.  metric = 'time' (panel f) or 'memory' (panel g).

    Annotates the empirical fit O(N^alpha) on the plot so reviewers can read
    the scaling exponent directly.
    """
    df = _load_scCAT_scaling()
    if df.empty:
        ax.text(0.5, 0.5, "scCAT scaling data\nnot available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_TICK)
        ax.axis("off")
        return

    col = "Time(s)" if metric == "time" else "PeakMemory(gb)"
    y_label = "Runtime (s, CPU)" if metric == "time" else "Peak memory (GB)"
    title   = "scCAT runtime scaling" if metric == "time" \
              else "scCAT peak-memory scaling"

    xs = df["n_cells"].values.astype(float)
    ys = df[col].values.astype(float)

    color = METHOD_COLORS.get("BTCA", "#2ca02c")
    ax.plot(xs, ys, marker="o", markersize=5.5, linewidth=2.0,
            color=color, zorder=5, label="scCAT (measured)")

    # Annotate each point with the cell count for clarity
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{int(x):,}",
            xy=(x, y), xytext=(4, -10), textcoords="offset points",
            fontsize=FS_ANNOT - 1, color="#555", alpha=0.85,
        )

    # Empirical power-law fit on log-log (skip if <3 points)
    if len(xs) >= 3:
        # Fit log(y) = alpha * log(x) + beta on the upper-half of points where
        # the power-law regime is cleaner (warm-up dominates the small end).
        mask = xs >= 1000
        if mask.sum() >= 2:
            lx = np.log10(xs[mask])
            ly = np.log10(ys[mask])
            alpha, beta = np.polyfit(lx, ly, 1)
            # Plot the fitted line as a thin dashed reference
            xfit = np.array([xs[mask].min(), xs[mask].max()])
            yfit = 10 ** (alpha * np.log10(xfit) + beta)
            ax.plot(xfit, yfit, ls="--", lw=1.0, color=color, alpha=0.55,
                    label=f"power-law fit  O(N^{alpha:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cells (10x 73k PBMC subsample)",
                  fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel(y_label, fontsize=FS_LABEL, labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, pad=4)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.legend(
        fontsize=FS_LEGEND - 1, frameon=False,
        loc="upper left", handlelength=1.6, handletextpad=0.5,
        borderaxespad=0.3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Corrected multi-seed Integration-Balance source for Fig 6 d/e.
# Panels d (per-dataset IB heatmap) and e (critical-difference diagram) must
# reflect the audited 3-seed full-IB values in phase2/, NOT the legacy
# single-seed metric tables.  phase2 tables key scCAT as 'scCAT'; the figure
# uses the internal key 'BTCA', so we remap on load.
# ─────────────────────────────────────────────────────────────────────────────
PHASE2_DIR = BASE_DIR / "experiments" / "results" / "phase2"


def _load_corrected_balance(main_datasets):
    """Return DataFrame [method, dataset, s_batch, s_bio, IB] from the audited
    3-seed full-IB summary (phase2/summary_IBfull_mean_std.csv).

    IB is taken directly from IB_full_mean — the mean of per-seed full
    Integration Balance, i.e. the SAME quantity used in the leaderboard and
    Friedman/Nemenyi analysis.  We deliberately do NOT recompute
    sqrt(S_batch * S_bio) from seed-averaged components, because
    mean(sqrt(.)) != sqrt(mean(.)).  Methods/datasets absent from a cell stay
    NaN (e.g. scCAT was not run on Lung), which the heatmap renders blank."""
    src = PHASE2_DIR / "summary_IBfull_mean_std.csv"
    if not src.exists():
        # Fallback to the legacy single-seed recompute if the audited table
        # is missing, so the figure still builds.
        return _compute_cross_dataset_balance(main_datasets)
    s = pd.read_csv(src)
    s["method"] = s["method"].replace({"scCAT": "BTCA"})
    rows = []
    for dk in main_datasets:
        sub = s[s["dataset"] == dk]
        for m in RANK_METHODS:
            r = sub[sub["method"] == m]
            ib = float(r["IB_full_mean"].iloc[0]) if not r.empty else np.nan
            rows.append({"method": m, "dataset": dk,
                         "s_batch": np.nan, "s_bio": np.nan, "IB": ib})
    return pd.DataFrame(rows)


def _plot_critical_difference(ax):
    """Friedman–Nemenyi critical-difference (CD) diagram for Fig 6e.

    Uses the audited SECONDARY analysis: the 9 methods with complete coverage
    across scCAT's 9 evaluation datasets (Demšar 2006).  Methods linked by a
    horizontal crossbar are NOT statistically distinguishable at alpha = 0.05.
    scANVI (semi-supervised) and scCAT occupy the two best average ranks and
    are linked — i.e. scCAT matches the semi-supervised gold standard without
    using cell-type labels."""
    rank_src = PHASE2_DIR / "friedman_nemenyi_IBfull_secondary.csv"
    pm_src   = PHASE2_DIR / "nemenyi_pmatrix_IBfull_secondary.csv"
    try:
        import scikit_posthocs as sp
        cd_df = pd.read_csv(rank_src)
        pm = pd.read_csv(pm_src, index_col=0)
        ren = {"INSCT_Unsupervised": "INSCT"}
        ranks = cd_df.set_index("method")["avg_rank"].rename(index=ren)
        pm = pm.rename(index=ren, columns=ren)
        CD = float(cd_df["CD"].iloc[0])
        pal = {}
        for m in ranks.index:
            base = "INSCT_Unsupervised" if m == "INSCT" else \
                   ("BTCA" if m == "scCAT" else m)
            pal[m] = METHOD_COLORS.get(base, "#888")
        sp.critical_difference_diagram(
            ranks, pm, ax=ax, color_palette=pal,
            label_fmt_left="{label}  ({rank:.2f})",
            label_fmt_right="({rank:.2f})  {label}",
        )
        # Harmonise label fonts; bold the scCAT label so it reads at a glance.
        for t in ax.texts:
            t.set_fontsize(FS_TICK)
            if "scCAT" in t.get_text():
                t.set_fontweight("bold")
        ax.set_title(
            f"Critical-difference diagram  (CD = {CD:.2f}, α = 0.05)",
            fontsize=FS_TITLE, pad=6)
    except Exception as exc:  # never break the whole figure on a plot error
        ax.text(0.5, 0.5, f"CD diagram unavailable\n({type(exc).__name__})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_TICK)
        ax.axis("off")


def _plot_rank_distribution(ax):
    """Per-dataset Integration-Balance rank distribution for Fig 6e.

    Presents the same omnibus analysis as a Friedman–Nemenyi critical-difference
    diagram, but in the idiom of the scIB benchmark (Luecken 2022): for each
    method the box gives the IQR + median of its IB rank across the datasets
    with complete coverage, and the dots are the individual per-dataset ranks.
    Methods are ordered best-to-worst by mean rank (diamond marker); scCAT is
    highlighted and its Nemenyi tie with the semi-supervised scANVI is stated.

    Source: summary_IBfull_mean_std.csv, re-ranked within each dataset.  Mean
    ranks and the Friedman statistic reconcile exactly with
    friedman_nemenyi_IBfull_secondary.csv; the scCAT–scANVI post-hoc p-value is
    read from nemenyi_pmatrix_IBfull_secondary.csv."""
    src   = BASE_DIR / "experiments" / "results" / "phase2" / \
            "summary_IBfull_mean_std.csv"
    fried = PHASE2_DIR / "friedman_nemenyi_IBfull_secondary.csv"
    pmsrc = PHASE2_DIR / "nemenyi_pmatrix_IBfull_secondary.csv"
    try:
        methods9 = ["scANVI", "scCAT", "scVI", "fastMNN", "Harmony",
                    "INSCT_Unsupervised", "Scanorama", "BBKNN", "DESC"]
        df  = pd.read_csv(src)
        sub = df[df["method"].isin(methods9)].copy()
        # keep only datasets where every one of the 9 methods has a score
        cnt  = sub.groupby("dataset")["method"].nunique()
        keep = cnt[cnt == len(methods9)].index
        sub  = sub[sub["dataset"].isin(keep)].copy()
        n_ds = len(keep)
        # rank within each dataset: 1 = highest IB_full (best)
        sub["rank"] = sub.groupby("dataset")["IB_full_mean"] \
                         .rank(ascending=False, method="average")
        rank_by_m = {m: sub.loc[sub["method"] == m, "rank"].values
                     for m in methods9}
        mean_rank = {m: float(np.mean(rank_by_m[m])) for m in methods9}
        order = sorted(methods9, key=lambda m: mean_rank[m])     # best first

        disp = {"INSCT_Unsupervised": "INSCT"}
        rng  = np.random.RandomState(7)
        for i, m in enumerate(order):
            col  = METHOD_COLORS.get("BTCA" if m == "scCAT" else m, "#888")
            vals = rank_by_m[m]
            if m == "scCAT":                     # faint band behind the row
                ax.axhspan(i - 0.45, i + 0.45, color=col, alpha=0.10, zorder=0)
            ax.boxplot([vals], positions=[i], vert=False, widths=0.55,
                       patch_artist=True, showfliers=False, zorder=2,
                       medianprops=dict(color="#222", lw=1.0),
                       whiskerprops=dict(color=col, lw=0.8),
                       capprops=dict(color=col, lw=0.8),
                       boxprops=dict(facecolor=col, edgecolor=col,
                                     alpha=0.28, lw=0.8))
            jit = (rng.rand(len(vals)) - 0.5) * 0.30
            ax.scatter(vals, np.full(len(vals), i) + jit, s=7, color=col,
                       edgecolor="white", linewidth=0.3, alpha=0.9, zorder=4)
            ax.scatter([mean_rank[m]], [i], marker="D", s=16, color=col,
                       edgecolor="#222", linewidth=0.5, zorder=5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([disp.get(m, m) for m in order], fontsize=FS_TICK)
        for t in ax.get_yticklabels():
            if t.get_text() == "scCAT":
                t.set_fontweight("bold")
        ax.invert_yaxis()                        # best method at the top
        ax.set_xlim(0.4, len(methods9) + 0.6)
        ax.set_xticks(range(1, len(methods9) + 1))
        ax.set_xlabel("IB rank across datasets (1 = best)", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(axis="x", color="#e8e8e8", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        # statistics: Friedman omnibus + the key scCAT–scANVI post-hoc tie.
        # Anchored top-right, the empty quadrant (the top methods all sit at low
        # ranks, so nothing occupies the high-rank end of their rows).
        fr   = pd.read_csv(fried)
        chi2 = float(fr["chi2"].iloc[0]); pval = float(fr["p"].iloc[0])
        stat = f"Friedman $\\chi^2$ = {chi2:.1f}, p = {pval:.1e}"
        try:
            pm = pd.read_csv(pmsrc, index_col=0)
            stat += f"\nscCAT vs scANVI: Nemenyi p = " \
                    f"{float(pm.loc['scCAT', 'scANVI']):.2f} (n.s.)"
        except Exception:
            pass
        ax.set_title(f"Per-dataset IB rank  (9 methods × {n_ds} datasets)",
                     fontsize=FS_TITLE, pad=6)
        ax.text(0.97, 0.94, stat, transform=ax.transAxes, ha="right",
                va="top", fontsize=FS_ANNOT - 0.5, color="#555",
                linespacing=1.3)
    except Exception as exc:  # never break the whole figure on a plot error
        ax.text(0.5, 0.5, f"Rank plot unavailable\n({type(exc).__name__})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_TICK)
        ax.axis("off")


def make_main_fig6_lung_summary(key=None):
    """
    Fig 6 — "scCAT achieves a favorable batch-biology balance in complex
            multi-batch integration and across benchmark datasets"

    a   Mouse Lung — scCAT featured pair (batch | cell type)
    b   Mouse Lung — 1×4 small UMAPs of other key methods
    c   Mouse Lung — per-batch local mixing heatmap (16 × 8)
    d   Integration Balance heatmap across the 5 benchmarked showcase datasets
        (3-seed mean; Lung is qualitative-only, see panels a–c)
    e   Per-dataset IB rank distribution (9 methods × 9 datasets; box = IQR,
        dots = per-dataset ranks, diamond = mean rank; Friedman + Nemenyi)
    f   scCAT runtime scaling on 10x 73k PBMC downsampling (log-log)
    g   scCAT peak-memory scaling (log-log)
    h   Wilcoxon BH-corrected p-value summary (scCAT vs all methods)

    Note: cross-method wall-clock comparison across the seven integration
    baselines (mixed R / Python / TensorFlow / PyTorch implementations) is
    out of scope for this study; Methods §4.7 cites Luecken 2022 for a
    literature-based comparison.  Panel f, g report scCAT's empirical
    scaling on a uniformly subsampled 10x 73k PBMC dataset (CPU only).
    """
    cfg, meta, emb, bdf, cdf = _load_figure_data("Lung")
    pt, al = cfg["point_size"], cfg["alpha"]

    # Panel d (quantitative 3-seed IB heatmap) covers the showcase datasets
    # that are part of the multi-seed benchmark.  Lung is NOT included here:
    # it is the qualitative showcase (panels a–c) and its per-method local
    # mixing is quantified in panel c; scCAT/scANVI have no 3-seed Lung IB, so
    # a Lung column would leave the two top methods blank and invite
    # mis-reading.  Panel e uses its own audited 9×9 analysis directly.
    main_datasets = ["data2_scenario1", "data2_scenario2", "Sc_mixology",
                     "HDC", "PBMC"]
    balance_df = _load_corrected_balance(main_datasets)

    fig = plt.figure(figsize=(DOUBLE_COL + 1.4, 12.4), dpi=150)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.035, top=0.965)
    # Row heights: a(big pair), b(small competitor strip), c(per-batch heatmap),
    # d/e (cross-dataset summary), f/g (scaling — smaller since scCAT-only).
    # hspace 0.40 → 0.30 after removing the row-b header text.
    outer = gridspec.GridSpec(5, 1, figure=fig,
                              height_ratios=[2.15, 1.20, 2.35, 1.80, 1.35],
                              hspace=0.24)

    # Row a — scCAT large pair, each with an explicit colour-key in title
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                            wspace=0.22)
    ax_a1 = fig.add_subplot(gs_a[0, 0]); _panel_label(ax_a1, "a", x=-0.10)
    if "BTCA" in emb:
        plot_embedding_featured(ax_a1, emb["BTCA"], BATCH_COL,
                                cfg["batch_palette"], "BTCA",
                                pt_size=pt * 1.6, alpha=al)
        ax_a1.set_title(
            f"scCAT — coloured by batch  ({cfg.get('n_cells_actual', 32472):,} cells, "
            f"{len(cfg['batch_palette'])} batches)",
            fontsize=FS_TITLE + 1, fontweight="bold", pad=5,
        )
    ax_a2 = fig.add_subplot(gs_a[0, 1])
    if "BTCA" in emb:
        plot_embedding_featured(ax_a2, emb["BTCA"], CELLTYPE_COL,
                                cfg["celltype_palette"], "BTCA",
                                pt_size=pt * 1.6, alpha=al)
        ax_a2.set_title(
            f"scCAT — coloured by cell type  "
            f"({len(cfg['celltype_palette'])} cell types)",
            fontsize=FS_TITLE + 1, fontweight="bold", pad=5,
        )

    # Row b — 1×4 thumbnails of key competing methods.
    # Standard C 4-competitor set (consistent with Fig 2 / Fig 3c):
    # Scanorama (MNN), Harmony (linear), scVI (deep VAE), INSCT (triplet).
    competitors4 = ["Scanorama", "Harmony", "scVI", "INSCT_Unsupervised"]
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                            wspace=0.20)
    for i, m in enumerate(competitors4):
        ax = fig.add_subplot(gs_b[0, i])
        if i == 0:
            _panel_label(ax, "b", x=-0.22)
        if m in emb:
            plot_embedding(ax, emb[m], CELLTYPE_COL,
                            cfg["celltype_palette"], m,
                            pt_size=pt * 0.65, alpha=al * 0.85)

    # (Row-b header text removed — caption belongs in the manuscript legend.)

    # Row c — per-batch local mixing heatmap (Lung's distinctive panel)
    ax_c = fig.add_subplot(outer[2]); _panel_label(ax_c, "c", x=-0.05)
    plot_per_batch_mixing(ax_c, emb, meta, methods=RANK_METHODS,
                          k=30, subsample_n=8000)

    # Row d — 2 panels: overall IB heatmap | avg rank (no more runtime mini)
    gs_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3],
                                            width_ratios=[1.55, 1.30],
                                            wspace=0.50)
    ax_d = fig.add_subplot(gs_d[0, 0]); _panel_label(ax_d, "d", x=-0.18)
    _plot_overall_balance_heatmap(ax_d, balance_df, main_datasets)
    ax_e = fig.add_subplot(gs_d[0, 1]); _panel_label(ax_e, "e", x=-0.16)
    _plot_rank_distribution(ax_e)

    # Row 5 — scCAT scaling + Wilcoxon summary.
    # f = runtime vs cells (log-log), g = peak memory vs cells,
    # h = Wilcoxon BH p-value summary (scCAT vs all).
    gs_f = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[4],
        width_ratios=[1.00, 1.00, 1.20], wspace=0.42,
    )
    ax_f = fig.add_subplot(gs_f[0, 0]); _panel_label(ax_f, "f", x=-0.16)
    plot_scCAT_scaling(ax_f, metric="time")
    ax_g = fig.add_subplot(gs_f[0, 1]); _panel_label(ax_g, "g", x=-0.16)
    plot_scCAT_scaling(ax_g, metric="memory")
    ax_h = fig.add_subplot(gs_f[0, 2]); _panel_label(ax_h, "h", x=-0.14)
    _plot_wilcoxon_summary(ax_h, "Wilcoxon signed-rank (scCAT vs others)")

    _save(fig, "Fig6_lung_summary")


# ─────────────────────────────────────────────────────────────────────────────
# 10d.  Generic supplementary per-dataset builder
#       Used for S1 (Sim 1), S2 (Sim 2), S7 (Pancreas), S8 (Immune), S9 (Gut)
# ─────────────────────────────────────────────────────────────────────────────

def make_supp_per_dataset(dataset_key: str):
    """Standard supplementary view: N×4 batch + N×4 cell-type UMAPs + metrics,
    where N = ceil(len(METHODS)/4). For the 11-method universe used in this
    paper this gives a 3×4 grid with one trailing empty slot per row."""
    cfg, meta, emb, bdf, cdf = _load_figure_data(dataset_key)
    cols = _resolved_metric_cols(bdf)
    pt, al = cfg["point_size"], cfg["alpha"]
    bs_types = cfg.get("batch_specific_types", [])

    # Scale figure height so each UMAP grid row gets ~2.0 inches
    n_rows_grid = (len(METHODS) + 3) // 4
    grid_unit_h = 1.4 * n_rows_grid          # 2.8 for 2-row, 4.2 for 3-row
    fig_h = 2 * grid_unit_h + 1.85 + 1.5     # both grids + metric row + padding
    fig = plt.figure(figsize=(DOUBLE_COL + 1.2, fig_h), dpi=150)
    fig.subplots_adjust(right=0.862)
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[grid_unit_h, grid_unit_h, 1.85],
                              hspace=0.30)

    _draw_umap_grid(fig, outer[0], emb, BATCH_COL,
                    cfg["batch_palette"], None,
                    pt, al, panel_letter="a")
    _draw_umap_grid(fig, outer[1], emb, CELLTYPE_COL,
                    cfg["celltype_palette"], bs_types,
                    pt, al, panel_letter="b")

    _figure_legend(fig, cfg["batch_palette"], "Batch",
                   anchor=(0.864, 0.83), ncol_max=15)
    _figure_legend(fig, cfg["celltype_palette"], "Cell type",
                   anchor=(0.864, 0.48), ncol_max=18)

    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2],
        width_ratios=[1.05, 1.05, 1.50], wspace=0.55,
    )
    ax_c1 = fig.add_subplot(gs_c[0, 0]); _panel_label(ax_c1, "c", x=-0.28)
    plot_scatter_metrics(ax_c1, bdf, cols["clisi"], cols["ilisi"],
                          "cLISI (cell type↑)", "iLISI (batch↑)", "LISI")
    plot_kbet_lollipop(fig.add_subplot(gs_c[0, 1]), bdf, cols["kbet"])
    ax_c3 = fig.add_subplot(gs_c[0, 2]); _panel_label(ax_c3, "d", x=-0.18)
    plot_eval_heatmap(ax_c3, bdf, cdf,
                      title=f"Evaluation Summary — {cfg['label']}")

    fname_map = {
        "data1_scenario1": "FigS1_sim1",
        "data1_scenario2": "FigS2_sim2",
        "Human_Pancreas":  "FigS7_pancreas",
        "Immune_human":    "FigS8_immune",
        "gut":             "FigS9_gut",
        "data2_scenario1": "FigS3_sim3",
        "data2_scenario2": "FigS4_sim4",
        "Sc_mixology":     "FigS5_scmixology",
        "PBMC":            "FigS6_pbmc",
        "Lung":            "FigS13_lung_full",   # all-8-method view of Lung
    }
    _save(fig, fname_map.get(dataset_key, f"FigS_{dataset_key}"))


# ─────────────────────────────────────────────────────────────────────────────
# 10b.  Fig 7 — Cross-omics batch correction (scRNA-seq + scATAC-seq)
# ─────────────────────────────────────────────────────────────────────────────

def make_main_fig7_crossomics(key=None):
    """Fig 7 — Cross-omics batch correction.

    Lengyel et al. fallopian-tube/ovarian dataset; 14,605 subsampled cells;
    RNA and ATAC merged into 2 batches.

    Panel a — UMAP grid coloured by omics origin (batch)
    Panel b — UMAP grid coloured by cell type
    Panel c — Bubble chart: Bio conservation (left) + Batch mixing (right)
    """
    # ── paths ────────────────────────────────────────────────────────────────
    ATAC_DIR  = BASE_DIR / "ATAC"
    EMB_DIR   = ATAC_DIR / "DR"
    META_CSV  = ATAC_DIR / "dataset" / "cell_metadata.csv"
    BR_CSV    = ATAC_DIR / "metric" / "batch_remove.csv"
    CL_CSV    = ATAC_DIR / "metric" / "cluster.csv"

    # ── method order (display left-to-right, top-to-bottom in grid) ──────────
    ATAC_METHODS = [
        "Scanorama", "DeepBID", "DESC", "scBCN",
        "fastMNN", "INSCT_Unsupervised", "SPDR", "BTCA",
    ]
    ATAC_H5AD = {
        "INSCT_Unsupervised": ("INSCT_Unsupervised.h5ad", "X_tnn",  "obs_names"),
        "SPDR":               ("SPDR.h5ad",               "X_spdr", "metadata_order"),
    }
    # bubble chart display order (bottom = scCAT)
    BUBBLE_METHODS = [
        "Scanorama", "DeepBID", "DESC", "scBCN",
        "fastMNN", "INSCT_Unsupervised", "SPDR", "BTCA",
    ]

    BATCH_PAL = {"RNA": "#2D9E2D", "ATAC": "#DD2477"}
    CELLTYPE_PAL = {
        "B cell":                              "#31CC31",
        "ciliated epithelial cell":            "#8FBC8B",
        "endothelial cell":                    "#B954D2",
        "endothelial cell of lymphatic vessel":"#86CDEB",
        "macrophage":                          "#F08080",
        "mast cell":                           "#6395ED",
        "natural killer cell":                 "#DDA0DD",
        "pericyte":                            "#FF67B3",
        "secretory cell":                      "#F0E68C",
        "smooth muscle cell":                  "#40E0D0",
        "stromal cell":                        "#FFAA44",
        "leukocyte":                           "#A3005B",
    }
    PT, AL = 1.8, 0.65

    # ── 1. Load metadata ──────────────────────────────────────────────────────
    meta = pd.read_csv(META_CSV)
    meta["cell"] = meta["cell"].astype(str)
    meta_sub = meta[["cell", "batch", "cell_type"]].copy()

    # ── 2. Load embeddings ────────────────────────────────────────────────────
    emb = {}
    for m in ATAC_METHODS:
        if m in ATAC_H5AD:
            fname, obsm_key, align = ATAC_H5AD[m]
            adata = ad.read_h5ad(EMB_DIR / fname)
            arr = np.asarray(adata.obsm[obsm_key])
            if align == "obs_names":
                df = pd.DataFrame({
                    "cell": adata.obs_names.astype(str),
                    "UMAP1": arr[:, 0], "UMAP2": arr[:, 1],
                })
                df = df.merge(meta_sub, on="cell", how="inner")
            else:  # metadata_order — SPDR preserves meta row order
                df = meta_sub.copy()
                df["UMAP1"] = arr[:len(meta_sub), 0]
                df["UMAP2"] = arr[:len(meta_sub), 1]
        else:
            raw = pd.read_csv(EMB_DIR / f"{m}.csv")
            raw.columns = ["cell", "UMAP1", "UMAP2"]
            raw["cell"] = raw["cell"].astype(str)
            df = raw.merge(meta_sub, on="cell", how="inner")
        emb[m] = df

    # ── 3. Load metrics ───────────────────────────────────────────────────────
    br = pd.read_csv(BR_CSV)
    cl = pd.read_csv(CL_CSV)

    def _get(df, method, *col_candidates):
        col_map = {str(c).lower(): c for c in df.columns}
        m_col = next((c for c in df.columns
                      if str(c).strip().lower() in ("method", "methods")), df.columns[0])
        row = df[df[m_col].astype(str).str.strip() == method]
        if row.empty:
            return np.nan
        for cand in col_candidates:
            if cand in df.columns:
                v = row[cand].values[0]
                return float(v) if pd.notna(v) else np.nan
            if cand.lower() in col_map:
                v = row[col_map[cand.lower()]].values[0]
                return float(v) if pd.notna(v) else np.nan
        return np.nan

    bio_cols  = ["ARI", "NMI", "cLISI_purity", "ASW_celltype"]
    bat_cols  = ["iLISI", "ASW_batch_mixing", "kBET"]
    bubble_labels = ["ARI", "NMI", "cLISI", "ASW\ncell type", "Bio\noverall",
                     "iLISI", "ASW\nbatch", "kBET", "Batch\noverall"]

    rows = []
    for m in BUBBLE_METHODS:
        ari  = _get(cl, m, "ARI")
        nmi  = _get(cl, m, "NMI")
        clisi= _get(br, m, "cLISI_purity", "cLISI")
        aswc = _get(br, m, "ASW_celltype", "ASW_cell_type")
        ilisi= _get(br, m, "iLISI")
        aswb = _get(br, m, "ASW_batch_mixing", "ASW_batch")
        kbet = _get(br, m, "kBET")
        bio_overall  = np.nanmean([ari, nmi, clisi, aswc])
        bat_overall  = np.nanmean([ilisi, aswb, 1.0 - kbet if np.isfinite(kbet) else np.nan])
        rows.append({
            "method": m,
            "ARI": ari, "NMI": nmi, "cLISI": clisi, "ASW_cell": aswc,
            "Bio_ov": bio_overall,
            "iLISI": ilisi, "ASW_batch": aswb, "kBET": kbet,
            "Bat_ov": bat_overall,
        })
    bub_df = pd.DataFrame(rows)
    # value columns in x order
    val_cols = ["ARI", "NMI", "cLISI", "ASW_cell", "Bio_ov",
                "iLISI", "ASW_batch", "kBET", "Bat_ov"]

    # ── 4. Layout ─────────────────────────────────────────────────────────────
    n_rows_grid = 2   # 8 methods → 2 rows × 4 cols
    grid_unit_h = 1.55 * n_rows_grid          # 3.1 in per UMAP block
    bub_h       = 2.4                         # bubble panel height
    fig_w       = DOUBLE_COL + 1.5           # right margin for legends
    fig_h       = 2 * grid_unit_h + bub_h + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    fig.subplots_adjust(right=0.84)
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[grid_unit_h, grid_unit_h, bub_h],
        hspace=0.35,
    )

    # ── 5. UMAP grids ─────────────────────────────────────────────────────────
    def _umap_grid(gs_slot, color_col, palette, panel_letter):
        gs = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=gs_slot, wspace=0.28, hspace=0.50
        )
        first_ax = None
        for i, m in enumerate(ATAC_METHODS):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            if first_ax is None:
                first_ax = ax
            if m in emb:
                plot_embedding(ax, emb[m], color_col, palette, m,
                               pt_size=PT, alpha=AL)
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=FS_TICK)
                _nature_embed_style(ax, METHOD_DISPLAY.get(m, m))
        _panel_label(first_ax, panel_letter, x=-0.28, y=1.12)

    _umap_grid(outer[0], "batch",     BATCH_PAL,    "a")
    _umap_grid(outer[1], "cell_type", CELLTYPE_PAL, "b")

    # legends placed outside right edge
    _figure_legend(fig, BATCH_PAL,    "Batch",     anchor=(0.854, 0.88), ncol_max=4)
    _figure_legend(fig, CELLTYPE_PAL, "Cell type", anchor=(0.854, 0.55), ncol_max=15)

    # ── 6. Bubble chart ───────────────────────────────────────────────────────
    ax_bub = fig.add_subplot(outer[2])
    _panel_label(ax_bub, "c", x=-0.065, y=1.28)

    n_m  = len(BUBBLE_METHODS)
    n_x  = len(val_cols)          # 9 columns
    BMIN, BMAX = 8, 220           # bubble size range (pts²)
    OV_COLS = {4, 8}              # "Overall" column x-indices

    # Grid background
    for i in range(n_m):
        for j in range(n_x):
            fc = "#E8E8E8" if j in OV_COLS else "white"
            ax_bub.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=fc, edgecolor="#BBBBBB", linewidth=0.4, zorder=0
            ))

    # Bubbles
    for iy, row in bub_df.iterrows():
        m = row["method"]
        col = METHOD_COLORS.get(m, "#888888")
        for jx, vc in enumerate(val_cols):
            v = row[vc]
            if not np.isfinite(v):
                continue
            v_plot = np.clip(v, 0, 1)
            sz = BMIN + v_plot * (BMAX - BMIN)
            ax_bub.scatter(jx, iy, s=sz, c=col, alpha=0.85,
                           edgecolors="white", linewidths=0.5, zorder=3)
            # Only label when bubble is large enough to contain the text
            if v_plot >= 0.18:
                ax_bub.text(jx, iy, f"{v:.2f}",
                            ha="center", va="center",
                            fontsize=FS_ANNOT - 0.5, color="white",
                            fontweight="bold", zorder=4)
            else:
                # Small bubble: place value just above in dark text
                ax_bub.text(jx, iy - 0.38, f"{v:.2f}",
                            ha="center", va="bottom",
                            fontsize=FS_ANNOT - 1.0, color="#333333",
                            zorder=4)

    # Method labels (y-axis)
    ax_bub.set_yticks(range(n_m))
    ax_bub.set_yticklabels(
        [METHOD_DISPLAY.get(m, m) for m in BUBBLE_METHODS],
        fontsize=FS_TICK
    )

    # x-axis labels
    ax_bub.set_xticks(range(n_x))
    ax_bub.set_xticklabels(bubble_labels, fontsize=FS_TICK - 0.5)

    ax_bub.set_xlim(-0.5, n_x - 0.5)
    ax_bub.set_ylim(n_m - 0.5, -0.5)
    ax_bub.tick_params(length=0)
    for sp in ax_bub.spines.values():
        sp.set_visible(False)

    # Group header bars (Bio / Batch)
    top_y_data = -1.55
    bar_h      = 0.85
    for (start, length, label, color) in [
        (0, 5, "Biological conservation", "#2D9E2D"),
        (5, 4, "Batch mixing",            "#4B8EC8"),
    ]:
        ax_bub.add_patch(FancyBboxPatch(
            (start - 0.5, top_y_data - 0.25), length, bar_h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=color, edgecolor="none", clip_on=False, zorder=4
        ))
        ax_bub.text(
            start + length / 2 - 0.5, top_y_data + 0.15, label,
            ha="center", va="center",
            fontsize=FS_LABEL - 0.5, color="white", zorder=5
        )

    # kBET footnote
    ax_bub.text(
        0.0, -0.22,
        "kBET: lower is better; 1 − kBET used for Batch overall",
        transform=ax_bub.transAxes,
        fontsize=FS_ANNOT - 0.5, color="#555", style="italic"
    )

    # Bubble size legend
    for v_leg, leg_label in [(0.2, "0.2"), (0.6, "0.6"), (1.0, "1.0")]:
        ax_bub.scatter([], [], s=BMIN + v_leg * (BMAX - BMIN),
                       c="#888", alpha=0.85, edgecolors="none",
                       label=leg_label)
    ax_bub.legend(title="Score", loc="center left",
                  bbox_to_anchor=(1.02, 0.5),
                  frameon=True, fontsize=FS_LEGEND,
                  title_fontsize=FS_LEGEND,
                  labelspacing=1.0, borderpad=0.5,
                  handletextpad=0.8)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    _save(fig, "Fig7_crossomics")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

# Per the new 6-figure mechanism-driven layout (Fig 1 is the schematic, kept by user)
MAIN_FIGURE_BUILDERS = {
    "main_fig2": make_main_fig2_simulation,     # Sim 3 + Sim 4
    "main_fig3": make_main_fig3_hdc,            # HDC flagship
    "main_fig4": make_main_fig4_ablation,       # NEW: ablation as main figure
    "main_fig5": make_main_fig5_scmix_pbmc,     # NEW: Sc_mixology + PBMC combined
    "main_fig6": make_main_fig6_lung_summary,   # NEW: Lung + cross-dataset summary
    "main_fig7": make_main_fig7_crossomics,     # NEW: Cross-omics (scRNA + scATAC)
}

# Supplementary per-dataset views (S1–S9)
SUPPL_PER_DATASET = {
    # S1 / S2 — extra simulated
    "supp_sim1":     lambda: make_supp_per_dataset("data1_scenario1"),
    "supp_sim2":     lambda: make_supp_per_dataset("data1_scenario2"),
    # S3 / S4 — main simulated, full 8-method view
    "supp_sim3":     lambda: make_supp_per_dataset("data2_scenario1"),
    "supp_sim4":     lambda: make_supp_per_dataset("data2_scenario2"),
    # S5 — Sc_mixology full 8 methods
    "supp_scmix":    lambda: make_supp_per_dataset("Sc_mixology"),
    # S6 — PBMC full 8 methods
    "supp_pbmc":     lambda: make_supp_per_dataset("PBMC"),
    # S7 / S8 / S9 — additional real datasets
    "supp_pancreas": lambda: make_supp_per_dataset("Human_Pancreas"),
    "supp_immune":   lambda: make_supp_per_dataset("Immune_human"),
    "supp_gut":      lambda: make_supp_per_dataset("gut"),
    # S13 — full 8-method view of Lung (batch + cell-type colouring)
    "supp_lung":     lambda: make_supp_per_dataset("Lung"),
}

# Existing custom per-dataset builders (alternative detailed views, kept for reference)
LEGACY_PER_DATASET = {
    "data2_scenario1": make_fig_data2_s1,
    "data2_scenario2": make_fig_data2_s2,
    "Sc_mixology":     make_fig_sc_mixology,
    "HDC":             make_fig_hdc,
    "PBMC":            make_fig_pbmc,
    "Lung":            make_fig_lung,
}

FIGURE_BUILDERS = {**MAIN_FIGURE_BUILDERS, **SUPPL_PER_DATASET, **LEGACY_PER_DATASET}

# Group aliases — convenient shorthands
GROUP_ALIASES = {
    "main": list(MAIN_FIGURE_BUILDERS),
    "supp": list(SUPPL_PER_DATASET),
    "legacy": list(LEGACY_PER_DATASET),
    "all":  list(MAIN_FIGURE_BUILDERS) + list(SUPPL_PER_DATASET),
}


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["main"]

    # Expand group aliases
    keys = []
    for a in args:
        if a in GROUP_ALIASES:
            keys.extend(GROUP_ALIASES[a])
        else:
            keys.append(a)

    invalid = [k for k in keys if k not in FIGURE_BUILDERS]
    if invalid:
        print(f"Unknown key(s): {invalid}")
        print(f"Valid keys: {sorted(FIGURE_BUILDERS)}")
        print(f"Aliases:    {sorted(GROUP_ALIASES)}")
        sys.exit(1)

    label_map = {
        "main_fig2": "Main Fig 2 — Controlled simulation (Sim 3 + Sim 4)",
        "main_fig3": "Main Fig 3 — HDC (batch-specific cell preservation)",
        "main_fig4": "Main Fig 4 — Ablation analysis (mechanism contribution)",
        "main_fig5": "Main Fig 5 — Sc_mixology + PBMC (cross-platform + condition)",
        "main_fig6": "Main Fig 6 — Lung + cross-dataset summary",
        "main_fig7": "Main Fig 7 — Cross-omics batch correction (scRNA + scATAC)",
    }
    for k in keys:
        if k in LEGACY_PER_DATASET:
            label = DATASET_CONFIGS[k]["label"] + " (legacy detailed view)"
        elif k in SUPPL_PER_DATASET:
            label = "Suppl: " + k.replace("supp_", "")
        else:
            label = label_map.get(k, k)
        print(f"\n{'='*60}")
        print(f"  {label}  ({k})")
        print(f"{'='*60}")
        try:
            builder = FIGURE_BUILDERS[k]
            # Builders are either main_fig*(key) or supp_*() lambdas
            try:
                builder(k)
            except TypeError:
                builder()
        except Exception as exc:
            print(f"[ERROR] {k}: {exc}")
            import traceback; traceback.print_exc()

    print("\nAll figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
