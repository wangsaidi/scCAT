"""
phase2_scib_robustness_fig.py — Supplementary figure for reviewer concern P0-2
("the Integration Balance metric is self-defined / circular").

Two panels, 9 full-coverage methods x 9 datasets (= scCAT's evaluation set):
  a  Community-standard scIB overall score (Luecken 2022): 0.4*batch + 0.6*bio,
     using the SAME component metrics as IB but the standard LINEAR weighting.
     scCAT is #1 among unsupervised methods and #2 overall (behind the
     semi-supervised scANVI) — i.e. its standing is not an IB artefact.
  b  Rank agreement between our IB ranking and the standard-scIB ranking
     (Spearman rho = 0.98): the method ordering is essentially invariant to
     the choice of aggregation.

Inputs (results/phase2/):
  scib_standard_secondary.csv          (scIB avg-rank + mean per method)
  friedman_nemenyi_IBfull_secondary.csv (IB  avg-rank per method)
Outputs (figures/):
  FigS14_scIB_robustness.{pdf,png,svg,tiff}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
PHASE2 = BASE / "experiments" / "results" / "phase2"
FIGDIR = BASE / "figures"

FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 7, 7, 6, 5.5
METHOD_COLORS = {
    "INSCT_Unsupervised": "#5B8DD9", "SPDR": "#55A868", "Scanorama": "#C474C4",
    "DeepBID": "#A63BB5", "DESC": "#E06C9F", "scBCN": "#DD2477",
    "fastMNN": "#64B5CD", "Harmony": "#F39C12", "scVI": "#8E44AD",
    "scANVI": "#16A085", "BBKNN": "#D35400", "scCAT": "#2D9E2D",
}
DISP = {"INSCT_Unsupervised": "INSCT"}
def disp(m): return DISP.get(m, m)

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "axes.linewidth": 0.6})


def main() -> None:
    sc = pd.read_csv(PHASE2 / "scib_standard_secondary.csv")
    ib = pd.read_csv(PHASE2 / "friedman_nemenyi_IBfull_secondary.csv")
    CD = float(sc["CD"].iloc[0]); chi2 = float(sc["chi2"].iloc[0])
    pval = float(sc["p_value"].iloc[0]); k = int(sc["k"].iloc[0]); N = int(sc["N"].iloc[0])

    sc_sorted = sc.sort_values("mean_scIB_overall", ascending=True)  # for barh
    methods = list(sc_sorted["method"])
    vals = list(sc_sorted["mean_scIB_overall"])
    semi = dict(zip(sc["method"], sc["semi_supervised"]))

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- Panel a: standard-scIB bar ----
    y = np.arange(len(methods))
    for i, (m, v) in enumerate(zip(methods, vals)):
        highlight = m in ("scCAT", "scANVI")
        axA.barh(i, v, color=METHOD_COLORS.get(m, "#888"),
                 alpha=1.0 if highlight else 0.55,
                 edgecolor="black" if highlight else "none",
                 linewidth=0.9 if highlight else 0, height=0.72, zorder=3)
        axA.text(v + 0.008, i, f"{v:.3f}", va="center", ha="left",
                 fontsize=FS_ANNOT, zorder=4)
    labels = [disp(m) + (" †" if semi.get(m) else "") for m in methods]
    axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=FS_TICK)
    # bold scCAT tick label
    for tk, m in zip(axA.get_yticklabels(), methods):
        if m == "scCAT": tk.set_fontweight("bold")
    axA.set_xlabel("Standard scIB overall  (0.4·batch + 0.6·bio)", fontsize=FS_LABEL)
    axA.set_xlim(0, max(vals) * 1.18)
    axA.tick_params(axis="x", labelsize=FS_TICK)
    axA.set_title("a   Community-standard scIB score (Luecken 2022)",
                  fontsize=FS_TITLE, loc="left", pad=4)
    axA.text(0.98, 0.04, "† semi-supervised (uses labels)", transform=axA.transAxes,
             ha="right", va="bottom", fontsize=FS_ANNOT, style="italic", color="#444")
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)

    # ---- Panel b: IB-rank vs scIB-rank agreement ----
    merged = pd.merge(ib[["method", "avg_rank"]].rename(columns={"avg_rank": "ib_rank"}),
                      sc[["method", "avg_rank"]].rename(columns={"avg_rank": "scib_rank"}),
                      on="method")
    rho, _ = spearmanr(merged["ib_rank"], merged["scib_rank"])
    axB.plot([0.5, 8.5], [0.5, 8.5], ls="--", lw=0.8, color="#999", zorder=1)
    # per-label pixel offsets to avoid collisions (esp. the worst-method cluster)
    LBL_OFF = {"scCAT": (5, 5), "scANVI": (5, 5), "scVI": (5, 4),
               "DESC": (6, 2), "BBKNN": (-4, -10)}
    for _, r in merged.iterrows():
        m = r["method"]; highlight = m in ("scCAT", "scANVI")
        axB.scatter(r["ib_rank"], r["scib_rank"], s=46 if highlight else 28,
                    color=METHOD_COLORS.get(m, "#888"),
                    edgecolor="black" if highlight else "white",
                    linewidth=0.8 if highlight else 0.4, zorder=3)
        if m in LBL_OFF:
            axB.annotate(disp(m), (r["ib_rank"], r["scib_rank"]),
                         textcoords="offset points", xytext=LBL_OFF[m],
                         ha="right" if m == "BBKNN" else "left",
                         fontsize=FS_ANNOT,
                         fontweight="bold" if m == "scCAT" else "normal")
    axB.set_xlabel("Avg. rank — our IB", fontsize=FS_LABEL)
    axB.set_ylabel("Avg. rank — standard scIB", fontsize=FS_LABEL)
    axB.set_xlim(0.5, 8.7); axB.set_ylim(0.5, 8.7)
    axB.invert_xaxis(); axB.invert_yaxis()   # rank 1 (best) at top-right
    axB.tick_params(labelsize=FS_TICK)
    axB.set_title("b   Ranking is robust to the aggregation choice",
                  fontsize=FS_TITLE, loc="left", pad=4)
    axB.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}\n(rank 1 = best)",
             transform=axB.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.5))
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)

    fig.suptitle(
        f"scCAT standing is not an artefact of the IB metric  "
        f"(Friedman χ² = {chi2:.1f}, p = {pval:.1e}, k = {k}, N = {N})",
        fontsize=FS_TITLE, y=1.02)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg", "tiff"):
        fig.savefig(FIGDIR / f"FigS14_scIB_robustness.{ext}", dpi=400,
                    bbox_inches="tight")
    print("saved →", FIGDIR / "FigS14_scIB_robustness.pdf")
    print(f"Spearman rho = {rho:.4f}")


if __name__ == "__main__":
    main()
