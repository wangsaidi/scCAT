"""
plot_supp_figures.py — Generate Supplementary Figures S6, S7, S8 from the
ablation / sensitivity / runtime CSVs produced by run_experiments.py.

    S6  Ablation analysis    — bars of OCI / BSRS / IB / ARI for each config
                                across HDC + Sim 4
    S7  Parameter sensitivity — line plots of IB vs parameter value for the
                                5 swept parameters
    S8  Runtime / memory     — wall-time and peak-memory bars across datasets,
                                with per-cell normalized scaling overlay
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterMathtext, NullFormatter

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURE_DIR  = HERE.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Style (matches plot_improved.py for figure consistency)
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
FS_PANEL = 8; FS_TITLE = 7; FS_LABEL = 7; FS_TICK = 6
FS_LEGEND = 7; FS_ANNOT = 5.5

# Config order (matches the ablation script)
CONFIG_ORDER = ["full", "noConf", "noFilter", "fixedMargin", "noBSP"]
CONFIG_DISPLAY = {
    "full":        "scCAT-full",
    "noConf":      "−Conf",
    "noFilter":    "−Filter",
    "fixedMargin": "−AdaptMargin",
    "noBSP":       "−BSP",
    "MNNonly":     "MNN-only",
}
CONFIG_COLORS = {
    "full":        "#2D9E2D",   # scCAT green
    "noConf":      "#5B8DD9",
    "noFilter":    "#64B5CD",
    "fixedMargin": "#DDA0DD",
    "noBSP":       "#E06C9F",
    "MNNonly":     "#999999",
}

PARAM_DISPLAY = {
    # Use mathtext for Greek letters / subscripts so glyphs render correctly
    # under the default sans-serif font (otherwise Unicode subscripts like
    # ₀ render as "?" or solid blocks).
    "min_c_pos":  r"$\mathrm{min\_c\_pos}$ (positive conf. threshold)",
    "m0":         r"$m_0$ (base margin)",
    "mu_rare":    r"$\mu_\mathrm{rare}$ (BSP weight)",
    "gamma":      r"$\gamma$ (conf. weight scale)",
    "knn_k":      r"$k$ (same-batch KNN)",
}
PARAM_DEFAULTS = {
    "min_c_pos": 0.7,
    "m0":        0.5,
    "mu_rare":   0.3,
    "gamma":     0.5,
    "knn_k":     5,
}

DATASET_DISPLAY = {
    "HDC":             "HDC",
    "data2_scenario1": "Sim 3",
    "data2_scenario2": "Sim 4",
    "Sc_mixology":     "Sc_mixology",
    "PBMC":            "PBMC",
    "Lung":            "Mouse Lung",
}


def _save(fig, name, dpi_raster=300):
    """Save as PDF + SVG + PNG + TIFF at 300 dpi (LZW-compressed TIFF)."""
    common = dict(bbox_inches="tight", facecolor="white")
    targets = [
        (FIGURE_DIR / f"{name}.pdf",  "pdf",  dict(dpi=dpi_raster)),
        (FIGURE_DIR / f"{name}.svg",  "svg",  {}),
        (FIGURE_DIR / f"{name}.png",  "png",  dict(dpi=dpi_raster)),
        (FIGURE_DIR / f"{name}.tiff", "tiff", dict(dpi=dpi_raster,
                                                    pil_kwargs={"compression": "tiff_lzw"})),
    ]
    saved = []
    try:
        for p, fmt, extra in targets:
            fig.savefig(p, format=fmt, **common, **extra)
            saved.append(p.name)
        print(f"  -> {' | '.join(saved)}")
    except PermissionError:
        import time
        suffix = time.strftime("%H%M%S")
        saved = []
        for p, fmt, extra in targets:
            np_ = p.parent / f"{p.stem}_v{suffix}{p.suffix}"
            fig.savefig(np_, format=fmt, **common, **extra)
            saved.append(np_.name)
        print(f"  [LOCKED] saved as: {' | '.join(saved)}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Supplementary Figure S6 — Ablation analysis
# ═════════════════════════════════════════════════════════════════════════════

def plot_supp_fig_S6():
    csv_path = RESULTS_DIR / "ablation_results.csv"
    if not csv_path.exists():
        print(f"[skip S6] {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[skip S6] empty results"); return

    datasets = ["HDC", "data2_scenario2"]
    datasets = [d for d in datasets if d in df["dataset"].unique()]
    metrics = [
        ("OCI",  "Overcorrection (↓)",   False),
        ("BSRS", "Batch-specific retention (↑)", True),
        ("IB",   "Integration Balance (↑)", True),
        ("ARI",  "Clustering ARI (↑)",   True),
    ]

    fig = plt.figure(figsize=(DOUBLE_COL + 1.0, 7.5), dpi=150)
    fig.suptitle("Module ablations on Sim 4 and HDC",
                 fontsize=FS_TITLE + 2, fontweight="bold", y=0.995)
    outer = gridspec.GridSpec(
        len(datasets), len(metrics), figure=fig,
        wspace=0.45, hspace=0.55,
        left=0.10, right=0.97, top=0.93, bottom=0.10,
    )

    for d_i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].copy()
        sub = sub.set_index("config").reindex(
            [c for c in CONFIG_ORDER if c in sub["config"].values
             or c in sub.index]
        ).reset_index().rename(columns={"index": "config"})
        sub = sub[sub["config"].isin(CONFIG_ORDER)]

        for m_i, (m_col, m_lbl, higher_better) in enumerate(metrics):
            ax = fig.add_subplot(outer[d_i, m_i])
            vals = sub[m_col].astype(float).values
            cfgs = sub["config"].values
            colors = [CONFIG_COLORS.get(c, "#888") for c in cfgs]
            xs = np.arange(len(cfgs))

            bars = ax.bar(xs, vals, color=colors, alpha=0.92,
                          edgecolor="white", linewidth=0.5, width=0.7)
            for b, v in zip(bars, vals):
                if not np.isfinite(v):
                    continue
                yoff = abs(max(vals.max(), 0.1) * 0.02)
                y = v + yoff if v >= 0 else v - yoff
                va = "bottom" if v >= 0 else "top"
                ax.text(b.get_x() + b.get_width() / 2, y, f"{v:.2f}",
                        ha="center", va=va, fontsize=FS_ANNOT)

            ax.set_xticks(xs)
            ax.set_xticklabels([CONFIG_DISPLAY.get(c, c) for c in cfgs],
                                fontsize=FS_TICK, rotation=30, ha="right")
            if d_i == 0:
                ax.set_title(m_lbl, fontsize=FS_TITLE, pad=4)
            if m_i == 0:
                ax.set_ylabel(f"{DATASET_DISPLAY.get(ds, ds)}\n{m_col}",
                              fontsize=FS_LABEL, labelpad=3)
            else:
                ax.set_ylabel(m_col, fontsize=FS_LABEL, labelpad=2)
            ax.tick_params(axis="y", labelsize=FS_TICK)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

            # Annotate "full" with a star marker
            if "full" in list(cfgs):
                full_i = list(cfgs).index("full")
                ax.scatter(full_i, vals[full_i] + abs(vals.max() * 0.08),
                            marker="*", color=CONFIG_COLORS["full"],
                            s=70, zorder=10, clip_on=False,
                            edgecolors="none")

            if higher_better:
                ax.set_ylim(min(0, vals.min() * 1.18 if vals.min() < 0 else 0),
                            vals.max() * 1.30 + 0.02)
            else:
                ax.set_ylim(0, max(vals.max() * 1.30, 0.1))

    fig.text(0.02, 0.50, "scCAT-full is starred; bars to its right are ablated versions.",
             fontsize=FS_ANNOT + 0.5, color="#444", rotation=90, va="center", ha="left")
    _save(fig, "FigS12_ext_ablation")


# ═════════════════════════════════════════════════════════════════════════════
# Supplementary Figure S7 — Parameter sensitivity
# ═════════════════════════════════════════════════════════════════════════════

def plot_supp_fig_S7():
    csv_path = RESULTS_DIR / "sensitivity_results.csv"
    if not csv_path.exists():
        print(f"[skip S7] {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[skip S7] empty results"); return

    params = list(PARAM_DISPLAY.keys())
    params = [p for p in params if p in df["param"].unique()]

    metrics = [
        ("OCI",  "OCI (↓)",   False, "#E06C9F"),
        ("BSRS", "BSRS (↑)",  True,  "#64B5CD"),
        ("IB",   "IB (↑)",    True,  "#2D9E2D"),
    ]

    n_p = len(params); n_m = len(metrics)
    fig = plt.figure(figsize=(DOUBLE_COL + 1.0, 1.7 + 1.7 * n_p), dpi=150)
    fig.suptitle("Sensitivity on HDC",
                 fontsize=FS_TITLE + 2, fontweight="bold", y=0.995)
    outer = gridspec.GridSpec(
        n_p, n_m, figure=fig,
        wspace=0.40, hspace=0.55,
        left=0.10, right=0.97, top=0.94, bottom=0.06,
    )

    for p_i, param in enumerate(params):
        sub = df[df["param"] == param].copy()
        sub["value_num"] = pd.to_numeric(sub["value_num"], errors="coerce")
        if sub["value_num"].isna().any():
            sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.sort_values("value_num").reset_index(drop=True)
        xvals = sub["value_num"].values

        for m_i, (m_col, m_lbl, higher_better, color) in enumerate(metrics):
            ax = fig.add_subplot(outer[p_i, m_i])
            yvals = sub[m_col].astype(float).values

            ax.plot(xvals, yvals, marker="o", color=color, lw=1.4, markersize=5,
                     markeredgecolor="white", markeredgewidth=0.5)

            # Number annotations: use offset-points (constant pixel distance)
            # so the gap looks identical at any y-range; combine with explicit
            # ylim padding below so labels never spill outside the axes.
            for x, y in zip(xvals, yvals):
                if not np.isfinite(y):
                    continue
                ax.annotate(
                    f"{y:.2f}", xy=(x, y),
                    xytext=(0, 6 if higher_better else -6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if higher_better else "top",
                    fontsize=FS_ANNOT, color=color,
                    clip_on=False,
                )

            # Mark default
            d_val = PARAM_DEFAULTS.get(param, None)
            if d_val is not None and d_val in xvals:
                d_idx = np.where(xvals == d_val)[0][0]
                ax.axvline(d_val, color="#888", lw=0.5, ls="--", alpha=0.6)
                ax.scatter(d_val, yvals[d_idx], marker="*",
                            color="#222", s=80, zorder=10, clip_on=False)

            # Add explicit ylim padding so annotation labels stay inside.
            y_finite = yvals[np.isfinite(yvals)]
            if len(y_finite) > 0:
                y_min, y_max = float(y_finite.min()), float(y_finite.max())
                y_span = max(y_max - y_min, 0.05)
                # 22% top headroom (labels go up for "higher_better"),
                # 18% bottom headroom (labels go down for OCI),
                # extra top room when value is near 1.0
                top_pad = y_span * 0.22 if higher_better else y_span * 0.10
                bot_pad = y_span * 0.10 if higher_better else y_span * 0.22
                ax.set_ylim(y_min - bot_pad, y_max + top_pad)

            ax.set_xlabel(PARAM_DISPLAY[param] if m_i == 0 else "",
                           fontsize=FS_LABEL, labelpad=2)
            ax.set_ylabel(m_lbl, fontsize=FS_LABEL, labelpad=2)
            ax.tick_params(labelsize=FS_TICK)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(True, axis="y", lw=0.3, alpha=0.4)
            if p_i == 0:
                ax.set_title(m_lbl, fontsize=FS_TITLE, pad=4)

    _save(fig, "FigS10_sensitivity")


# ═════════════════════════════════════════════════════════════════════════════
# Supplementary Figure S8 — Runtime + memory benchmark
# ═════════════════════════════════════════════════════════════════════════════

def plot_supp_fig_S8():
    # Prefer the 4-method (scCAT + Harmony + scVI + scANVI) CSV
    new_csv = RESULTS_DIR / "runtime_4methods.csv"
    legacy  = RESULTS_DIR / "runtime_results.csv"
    if new_csv.exists():
        df = pd.read_csv(new_csv)
        df["method"] = df["method"].astype(str)
        multi_method = True
    elif legacy.exists():
        df = pd.read_csv(legacy)
        df["method"] = "scCAT"
        multi_method = False
    else:
        print("[skip S8] no runtime CSV found"); return
    if df.empty:
        print("[skip S8] empty results"); return

    df = df.sort_values(["n_cells", "method"]).reset_index(drop=True)
    df["ds_label"] = df["dataset"].map(lambda d: DATASET_DISPLAY.get(d, d))

    fig = plt.figure(figsize=(DOUBLE_COL + 1.0, 7.5), dpi=150)
    title = ("Runtime and peak memory: scCAT vs Harmony / scVI / scANVI (CPU)"
              if multi_method else
              "Runtime and peak memory of scCAT (CPU)")
    fig.suptitle(title, fontsize=FS_TITLE + 2, fontweight="bold", y=0.995)
    outer = gridspec.GridSpec(2, 2, figure=fig,
                              wspace=0.38, hspace=0.50,
                              left=0.10, right=0.97, top=0.92, bottom=0.12)

    method_colors = {
        "scCAT":   "#2D9E2D",
        "Harmony": "#F39C12",
        "scVI":    "#8E44AD",
        "scANVI":  "#16A085",
    }
    methods = sorted(df["method"].unique(),
                     key=lambda m: ["Harmony","scCAT","scVI","scANVI"].index(m)
                     if m in ["Harmony","scCAT","scVI","scANVI"] else 99)
    datasets = sorted(df["ds_label"].unique(),
                       key=lambda d: df[df["ds_label"]==d]["n_cells"].iloc[0])
    n_ds, n_m = len(datasets), len(methods)

    # a — grouped runtime bars (log-y to handle Harmony 1s vs scANVI 4000s)
    ax_a = fig.add_subplot(outer[0, 0])
    bar_w = 0.8 / n_m
    for j, m in enumerate(methods):
        vals = [df[(df["ds_label"]==d) & (df["method"]==m)]["runtime_sec"].iloc[0]
                if len(df[(df["ds_label"]==d) & (df["method"]==m)]) > 0 else 0
                for d in datasets]
        xs = np.arange(n_ds) + (j - (n_m-1)/2) * bar_w
        ax_a.bar(xs, vals, width=bar_w, color=method_colors.get(m, "#888"),
                  alpha=0.88, edgecolor="white", linewidth=0.4, label=m)
    ax_a.set_yscale("log")
    ax_a.set_xticks(np.arange(n_ds))
    ax_a.set_xticklabels(datasets, fontsize=FS_TICK, rotation=20, ha="right")
    ax_a.set_ylabel("Wall-clock training time (s, log)", fontsize=FS_LABEL)
    ax_a.set_title("a   Runtime per dataset (4 methods × 4 datasets)",
                    fontsize=FS_TITLE, pad=4, loc="left", fontweight="bold")
    ax_a.legend(fontsize=FS_LEGEND - 0.5, frameon=False, ncol=4,
                 loc="upper left", handlelength=1.2, handletextpad=0.4,
                 columnspacing=0.6, borderaxespad=0.2)
    for sp in ("top", "right"): ax_a.spines[sp].set_visible(False)
    ax_a.tick_params(labelsize=FS_TICK)
    ax_a.grid(True, axis="y", which="major", lw=0.3, alpha=0.4)

    # b — peak memory grouped bar (linear, MB)
    ax_b = fig.add_subplot(outer[0, 1])
    for j, m in enumerate(methods):
        vals = [df[(df["ds_label"]==d) & (df["method"]==m)]["peak_mb"].iloc[0]
                if len(df[(df["ds_label"]==d) & (df["method"]==m)]) > 0 else 0
                for d in datasets]
        xs = np.arange(n_ds) + (j - (n_m-1)/2) * bar_w
        ax_b.bar(xs, vals, width=bar_w, color=method_colors.get(m, "#888"),
                  alpha=0.88, edgecolor="white", linewidth=0.4, label=m)
    ax_b.set_xticks(np.arange(n_ds))
    ax_b.set_xticklabels(datasets, fontsize=FS_TICK, rotation=20, ha="right")
    ax_b.set_ylabel("Peak Python memory (MB)", fontsize=FS_LABEL)
    ax_b.set_title("b   Peak memory per dataset",
                    fontsize=FS_TITLE, pad=4, loc="left", fontweight="bold")
    for sp in ("top", "right"): ax_b.spines[sp].set_visible(False)
    ax_b.tick_params(labelsize=FS_TICK)
    ax_b.grid(True, axis="y", which="major", lw=0.3, alpha=0.4)

    # c — runtime vs cells (scaling), log-log, ONE LINE PER METHOD
    ax_c = fig.add_subplot(outer[1, 0])
    for m in methods:
        sub = df[df["method"] == m].sort_values("n_cells")
        ax_c.plot(sub["n_cells"], sub["runtime_sec"],
                   "-o", color=method_colors.get(m, "#888"),
                   markersize=5.5, linewidth=1.8,
                   markeredgecolor="white", markeredgewidth=0.6,
                   label=m, alpha=0.95)
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.legend(fontsize=FS_LEGEND - 0.5, frameon=False, loc="upper left",
                 handlelength=1.4, handletextpad=0.5, borderaxespad=0.3)
    _scaling_texts = []   # no per-point labels needed — methods are legend-distinguished

    # Explicit decade-padded limits + force every decade to be ticked.
    # The default LogLocator caps at ~5 ticks and silently drops decades when
    # the data span is small, so 7→590 collapses to just 10^1 and 10^2.
    x_min, x_max = float(df["n_cells"].min()), float(df["n_cells"].max())
    y_min, y_max = float(df["runtime_sec"].min()), float(df["runtime_sec"].max())
    # Round limits to enclosing full decades
    x_lo = 10 ** np.floor(np.log10(x_min)) * 0.6
    x_hi = 10 ** np.ceil(np.log10(x_max)) * 1.5
    y_lo = 10 ** np.floor(np.log10(y_min)) * 0.6
    y_hi = 10 ** np.ceil(np.log10(y_max)) * 1.5
    ax_c.set_xlim(x_lo, x_hi)
    ax_c.set_ylim(y_lo, y_hi)

    # Major ticks = every decade (10^n), minor ticks = 2,3,...,9 within each decade
    ax_c.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax_c.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax_c.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(np.arange(2, 10) / 10.0), numticks=12)
    )
    ax_c.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(np.arange(2, 10) / 10.0), numticks=12)
    )
    ax_c.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_c.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax_c.xaxis.set_minor_formatter(NullFormatter())
    ax_c.yaxis.set_minor_formatter(NullFormatter())

    ax_c.set_xlabel("Number of cells", fontsize=FS_LABEL)
    ax_c.set_ylabel("Wall-clock time (s)", fontsize=FS_LABEL)
    ax_c.set_title("c   Scaling of runtime with cell count",
                    fontsize=FS_TITLE, pad=4, loc="left", fontweight="bold")
    for sp in ("top", "right"): ax_c.spines[sp].set_visible(False)
    ax_c.tick_params(axis="both", which="major", labelsize=FS_TICK,
                     length=3.5, width=0.7)
    ax_c.tick_params(axis="both", which="minor", length=2.0, width=0.5)
    ax_c.grid(True, which="major", lw=0.45, alpha=0.55)
    ax_c.grid(True, which="minor", lw=0.25, alpha=0.30)

    # Route the dataset labels with adjustText so they don't sit on top of dots.
    # Must run AFTER set_xlim / set_ylim / scale so coordinates are final.
    try:
        from adjustText import adjust_text
        adjust_text(
            _scaling_texts, ax=ax_c,
            expand=(1.4, 1.6),
            force_text=(0.7, 0.9),
            force_static=(0.6, 0.8),
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.4, alpha=0.7),
            max_move=30,
        )
    except ImportError:
        pass

    # d — table: 4 methods × 4 datasets compact runtime grid (sec)
    ax_d = fig.add_subplot(outer[1, 1])
    ax_d.axis("off")
    if multi_method:
        cols = ["Method"] + datasets
        rows = []
        for m in methods:
            row = [m]
            for d in datasets:
                sub = df[(df["ds_label"]==d) & (df["method"]==m)]
                row.append(f"{sub['runtime_sec'].iloc[0]:.0f}s" if len(sub) else "—")
            rows.append(row)
        table = ax_d.table(cellText=rows, colLabels=cols, loc="center",
                            cellLoc="center", colLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(FS_TICK)
        table.scale(1.05, 1.45)
        ax_d.set_title("d   Runtime (s) — 4 methods × 4 datasets",
                        fontsize=FS_TITLE, pad=4, loc="left",
                        fontweight="bold")
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor("#CCCCCC"); cell.set_linewidth(0.4)
            if i == 0:
                cell.set_facecolor("#EEEEEE")
                cell.set_text_props(weight="bold")
            elif j == 0:  # method column
                # Colour-code the row label by method
                m = rows[i-1][0]
                col = method_colors.get(m, "#888")
                cell.set_text_props(color=col, weight="bold")
    else:
        # Legacy scCAT-only table
        cols = ["Dataset", "Cells", "Batches", "Time (s)", "Mem (MB)", "Epochs"]
        rows = []
        for _, row in df.iterrows():
            rows.append([row["ds_label"], f"{int(row['n_cells']):,}",
                          f"{int(row['n_batches'])}", f"{row['runtime_sec']:.0f}",
                          f"{row['peak_mb']:.0f}",
                          f"{int(row['n_epochs'])}" if pd.notna(row.get('n_epochs')) else "—"])
        table = ax_d.table(cellText=rows, colLabels=cols, loc="center",
                            cellLoc="center", colLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(FS_TICK)
        table.scale(1.0, 1.4)
        ax_d.set_title("d   Numerical summary",
                        fontsize=FS_TITLE, pad=4, loc="left",
                        fontweight="bold")
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor("#CCCCCC"); cell.set_linewidth(0.4)
            if i == 0:
                cell.set_facecolor("#EEEEEE")
                cell.set_text_props(weight="bold")

    _save(fig, "FigS11_runtime")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["S6", "S7", "S8"]
    if "S6" in targets: plot_supp_fig_S6()
    if "S7" in targets: plot_supp_fig_S7()
    if "S8" in targets: plot_supp_fig_S8()
