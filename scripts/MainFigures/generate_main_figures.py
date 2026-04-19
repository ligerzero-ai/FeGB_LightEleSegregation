#!/cmmc/ptmp/hmai/mambaforge/envs/pymatgen/bin/python
"""
Generate manuscript figures using the KP dataset (excluding suspicious SCF jobs).
Produces Figures 3, 4, 5, 6, 7, and 10 from the manuscript (KP data).

Figures 8 (Wsep) and 9 (ANSBO) require cleavage/Chargemol data only available
for KS and are produced from df_main_final.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from plotters import scatter_by_gb

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

LIGHT_ELEMENTS = ["H", "He", "B", "C", "N", "O", "P", "S"]

gb_latex_labels = {
    "S11_RA110_S3_32": r"$\Sigma11[110](3\bar{3}2)$",
    "S3_RA110_S1_11": r"$\Sigma3[110](1\bar{1}1)$",
    "S3_RA110_S1_12": r"$\Sigma3[110](1\bar{1}2)$",
    "S5_RA001_S210": r"$\Sigma5[001](210)$",
    "S5_RA001_S310": r"$\Sigma5[001](310)$",
    "S9_RA110_S2_21": r"$\Sigma9[110](2\bar{2}1)$",
}
gb_order = [
    "S3_RA110_S1_11", "S3_RA110_S1_12", "S5_RA001_S210",
    "S5_RA001_S310", "S9_RA110_S2_21", "S11_RA110_S3_32",
]
dict_GB_gbenergy = {
    "S3_RA110_S1_11": 1.58, "S3_RA110_S1_12": 0.45,
    "S5_RA001_S310": 1.69, "S5_RA001_S210": 1.62,
    "S9_RA110_S2_21": 1.75, "S11_RA110_S3_32": 1.45,
}
custom_colors = {
    "S3_RA110_S1_11": 'blue', "S3_RA110_S1_12": "orange",
    "S5_RA001_S310": "green", "S5_RA001_S210": "purple",
    "S9_RA110_S2_21": "brown", "S11_RA110_S3_32": 'red',
}
gb_marker_dict = {
    'S5_RA001_S210': "s", 'S11_RA110_S3_32': "o", 'S3_RA110_S1_12': "D",
    'S3_RA110_S1_11': "^", 'S9_RA110_S2_21': "v", 'S5_RA001_S310': "h",
}
gb_sigma_mapping = {
    "S3_RA110_S1_11": 3, "S3_RA110_S1_12": 3,
    "S5_RA001_S210": 5, "S5_RA001_S310": 5,
    "S9_RA110_S2_21": 9, "S11_RA110_S3_32": 11,
}

# ---- Load data ----
print("Loading checkpoints...")
# Pre-dedup data with Voronoi features (for Figures 4, 6, 7)
df_vor = pd.read_pickle(CHECKPOINT_DIR / "04_df_KP_voronoi.pkl.gz")
# Post-dedup data (for Figure 10 site types, Figure 3 min Eseg, and Figure 5 histograms)
df_filt = pd.read_pickle(CHECKPOINT_DIR / "07_df_KP_filtered.pkl.gz")

# ---- SCF filter ----
def last_scf(x):
    if isinstance(x, (list, np.ndarray)):
        return int(x[-1]) if len(x) else np.nan
    return np.nan

if "scf_steps" in df_vor.columns:
    df_vor["_last_scf"] = df_vor["scf_steps"].apply(last_scf)
    bad = (df_vor["_last_scf"] < 15) | (df_vor["_last_scf"] > 150)
    print(f"Filtering {bad.sum()} suspicious SCF jobs from Voronoi data ({len(df_vor)} -> {(~bad).sum()})")
    df_vor = df_vor[~bad].copy()

if "scf_steps" in df_filt.columns:
    df_filt["_last_scf"] = df_filt["scf_steps"].apply(last_scf)
    bad = (df_filt["_last_scf"] < 15) | (df_filt["_last_scf"] > 150)
    print(f"Filtering {bad.sum()} suspicious SCF jobs from filtered data ({len(df_filt)} -> {(~bad).sum()})")
    df_filt = df_filt[~bad].copy()

# Ensure GB_string column exists for scatter_by_gb
for df in [df_vor, df_filt]:
    df["GB_string"] = df["GB"]

# Common scatter_by_gb kwargs — matched to original notebook
SCATTER_KW = dict(
    gb_col="GB",
    gb_colors=custom_colors,
    gb_markers=gb_marker_dict,
    gb_label_map=gb_latex_labels,
    gb_sigma_mapping=gb_sigma_mapping,
    figsize=(8, 6),
    s=200,
    tick_fontsize=20,
    label_fontsize=24,
    legend_loc="lower left",
    legend_anchor=(0.641, -0.01),
    legend_markersize=14,
    style_legend_loc="lower left",
    style_legend_anchor=(0.325, 0.00),
    legend_fontsize=16,
    style_legend_fontsize=16,
    marker_edgewidth=3.0,
    fill_by_col="calculation_description",
    hollow_values=("GB_int",),
    legend_title="",
    legend_borderpad=0.1,
    legend_handletextpad=0.1,
    style_legend_borderpad=0.1,
    style_legend_handletextpad=0.1,
)

from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

# ======================================================================
# Figure 3: min(Eseg) vs GB energy — single plot
#   colour = element, marker shape = GB, lines connecting same element
# ======================================================================
print("\nFigure 3: min Eseg vs GB energy...")

cmap_ele = get_cmap("tab10")
element_colors = {ele: cmap_ele(i % 10) for i, ele in enumerate(LIGHT_ELEMENTS)}

min_eseg_df = df_vor.groupby(["GB", "element"])["Eseg"].min().reset_index()

fig, ax = plt.subplots(figsize=(9, 6))

all_xs, all_ys = [], []
for ele in LIGHT_ELEMENTS:
    ele_data = min_eseg_df[min_eseg_df["element"] == ele]
    gb_names = [gb for gb in ele_data["GB"].values if gb in dict_GB_gbenergy]
    pts = [(dict_GB_gbenergy[gb], float(ele_data[ele_data["GB"] == gb]["Eseg"].iloc[0]), gb)
           for gb in gb_names]
    pts.sort(key=lambda t: t[0])
    if pts:
        xs, ys, gbs = zip(*pts)
        all_xs.extend(xs); all_ys.extend(ys)
        ax.plot(xs, ys, linestyle="-", linewidth=4, alpha=0.9, color=element_colors[ele], zorder=10)
        for x, y, gb in pts:
            ax.plot(x, y, marker=gb_marker_dict.get(gb, "o"), markersize=13,
                    markeredgecolor="k", markerfacecolor=element_colors[ele],
                    linestyle="None", zorder=11)

# Force axis limits to show the data
if all_ys:
    ax.set_ylim(min(all_ys) - 0.2, max(all_ys) + 0.2)
    ax.set_xlim(min(all_xs) - 0.1, max(all_xs) + 0.1)

# Element legend (colour, lines)
ele_handles = [
    Line2D([0], [0], color=element_colors[e], linewidth=3, marker="o",
           markeredgecolor="k", markersize=8, label=e)
    for e in LIGHT_ELEMENTS
]
# GB legend (marker shape)
gb_handles = [
    Line2D([0], [0], marker=gb_marker_dict[gb], linestyle="", color="gray",
           markeredgecolor="k", markersize=10, label=gb_latex_labels[gb])
    for gb in gb_order
]
leg1 = ax.legend(handles=gb_handles, title="Grain boundary", loc="lower left",
                 fontsize=14, title_fontsize=14, frameon=True,
                 bbox_to_anchor=(0.0, 0.0))
ax.add_artist(leg1)
ax.legend(handles=ele_handles, title="Element", loc="lower center",
          fontsize=14, title_fontsize=14, frameon=True, ncol=3,
          bbox_to_anchor=(0.5, 0.0))

ax.set_xlabel(r"$\gamma_{GB}$ (J/m$^2$)", fontsize=24)
ax.set_ylabel(r"min($E_{seg}$) (eV)", fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "min_eseg_vs_gb_energy_per_element.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved min_eseg_vs_gb_energy_per_element.png")


# ======================================================================
# Figure 4: Eseg vs distance from GB
# ======================================================================
print("\nFigure 4: Eseg vs dist_GB...")
element_label_position_fig4 = {
    "H": (0.82, 0.82), "He": (0.75, 0.97), "B": (0.82, 0.82), "C": (0.82, 0.80),
    "N": (0.82, 0.92), "O": (0.82, 0.76), "P": (0.82, 0.965), "S": (0.82, 0.76),
}
for ele in LIGHT_ELEMENTS:
    ele_data = df_vor[df_vor["element"] == ele].copy()
    show = ele == "C"
    _, (fig, ax) = scatter_by_gb(
        ele_data, x="dist_GB", y="Eseg",
        x_label=r"distance to GB ($\rm{\AA}$)",
        y_label=r"E$_{\rm{seg}}$ (eV/atom)",
        xlim=[-0.5, 7], ylim=[None, 0.5],
        show_legend=show, show_style_legend=show,
        **SCATTER_KW,
    )
    ax.axhline(0, linestyle="--", color="k")
    ax.axvline(0, linestyle="--", color="k")
    pos = element_label_position_fig4[ele]
    ax.text(pos[0], pos[1], ele, transform=ax.transAxes, fontsize=70, ha="left", va="top")
    fig.savefig(FIG_DIR / f"Eseg_dist_GB_{ele}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
print(f"  Saved 8 Eseg_dist_GB plots")


# ======================================================================
# Figure 6: Eseg vs Voronoi volume
# ======================================================================
print("\nFigure 6: Eseg vs Voronoi volume...")
element_label_position_fig6 = {
    "H": (0.82, 0.75), "He": (0.75, 0.90), "B": (0.85, 0.92), "C": (0.82, 0.92),
    "N": (0.82, 0.92), "O": (0.82, 0.90), "P": (0.82, 0.96), "S": (0.82, 0.85),
}
SCATTER_KW_FIG6 = dict(SCATTER_KW)
SCATTER_KW_FIG6["legend_fontsize"] = 18
SCATTER_KW_FIG6["style_legend_fontsize"] = 18
SCATTER_KW_FIG6["legend_anchor"] = (0.60, 0.01)
SCATTER_KW_FIG6["legend_markersize"] = 14
SCATTER_KW_FIG6["style_legend_loc"] = "lower left"
SCATTER_KW_FIG6["style_legend_anchor"] = (0.25, 0.00)
for ele in LIGHT_ELEMENTS:
    ele_data = df_vor[(df_vor["element"] == ele) & (df_vor["VorNN_tot_vol"] < 20)].copy()
    show = ele == "C"
    _, (fig, ax) = scatter_by_gb(
        ele_data, x="VorNN_tot_vol", y="Eseg",
        x_label=r"Voronoi volume ($\rm{\AA}^3$)",
        y_label=r"E$_{\rm{seg}}$ (eV/atom)",
        show_legend=show, show_style_legend=show,
        **SCATTER_KW_FIG6,
    )
    ax.axhline(0, linestyle="--", color="k")
    pos = element_label_position_fig6[ele]
    ax.text(pos[0], pos[1], ele, transform=ax.transAxes, fontsize=70, ha="left", va="top")
    fig.savefig(FIG_DIR / f"Eseg_Voronoi_NN_dist_{ele}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
print(f"  Saved 8 Eseg_Voronoi plots")


# ======================================================================
# Figure 7: Eseg vs min nearest-neighbour distance
# ======================================================================
print("\nFigure 7: Eseg vs min NN distance...")
element_label_position_fig7 = {
    "H": (0.82, 0.85), "He": (0.75, 0.88), "B": (0.82, 0.90), "C": (0.82, 0.90),
    "N": (0.82, 0.90), "O": (0.82, 0.90), "P": (0.82, 0.82), "S": (0.82, 0.90),
}
for ele in LIGHT_ELEMENTS:
    ele_data = df_vor[(df_vor["element"] == ele) & (df_vor["VorNN_tot_vol"] < 20) & (df_vor["Eseg"] < 0.1)].copy()
    show = ele == "S"
    _, (fig, ax) = scatter_by_gb(
        ele_data, x="min_nn_dist", y="Eseg",
        x_label=r"Min nearest neighbour dist ($\rm{\AA}$)",
        y_label=r"E$_{\rm{seg}}$ (eV/atom)",
        show_legend=show, show_style_legend=show,
        legend_anchor=(0.641, -0.001),
        style_legend_anchor=(0.641, 0.45),
        **{k: v for k, v in SCATTER_KW.items() if k not in ("legend_anchor", "style_legend_anchor")},
    )
    ax.axhline(0, linestyle="--", color="k")
    pos = element_label_position_fig7[ele]
    ax.text(pos[0], pos[1], ele, transform=ax.transAxes, fontsize=70, ha="left", va="top")
    fig.savefig(FIG_DIR / f"Eseg_min_nn_dist_{ele}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
print(f"  Saved 8 Eseg_min_nn_dist plots")


# ======================================================================
# Figure 10: Site type bar chart
# ======================================================================
print("\nFigure 10: Site type classification...")
if "site_types_mixed" in df_filt.columns:
    from pymatgen.core import Element
    elements = sorted(df_filt["element"].unique(), key=lambda x: Element(x).Z)

    counts_sub, counts_mixed, counts_int = [], [], []
    percents_sub, percents_mixed, percents_int = [], [], []
    for ele in elements:
        df_ele = df_filt[df_filt["element"] == ele]
        ns = (df_ele["site_types_mixed"] == "sub").sum()
        nm = (df_ele["site_types_mixed"] == "mixed").sum()
        ni = (df_ele["site_types_mixed"] == "int").sum()
        nt = ns + nm + ni
        counts_sub.append(ns); counts_mixed.append(nm); counts_int.append(ni)
        if nt > 0:
            percents_sub.append(ns / nt * 100)
            percents_mixed.append(nm / nt * 100)
            percents_int.append(ni / nt * 100)
        else:
            percents_sub.append(0); percents_mixed.append(0); percents_int.append(0)

    x = np.arange(len(elements))
    width = 0.6

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, counts_sub, width, label="substitutional", color="tab:orange")
    ax.bar(x, counts_mixed, width, bottom=counts_sub, label="mixed", color="tab:green")
    ax.bar(x, counts_int, width,
           bottom=(np.array(counts_sub) + np.array(counts_mixed)), label="interstitial", color="tab:blue")

    ax.set_xticks(x)
    ax.set_xticklabels(elements, fontsize=20)
    ax.set_ylabel("Unique sites in GB set", fontsize=24)
    ax.legend(fontsize=17.5, loc="upper left", bbox_to_anchor=(0.0, 1.15), ncol=3, frameon=True, borderaxespad=0.1)
    ax.tick_params(axis="y", labelsize=20)

    # Annotate percentages
    for i, (ns, nm, ni, ps, pm, pi) in enumerate(
            zip(counts_sub, counts_mixed, counts_int, percents_sub, percents_mixed, percents_int)):
        y = 0
        if ns > 0:
            ax.text(i, y + ns / 2, f"{ps:.0f}%", ha="center", va="center", color="k", fontsize=14, fontweight="bold")
        y += ns
        if nm > 0:
            ax.text(i, y + nm / 2, f"{pm:.0f}%", ha="center", va="center", color="k", fontsize=14, fontweight="bold")
        y += nm
        if ni > 0:
            ax.text(i, y + ni / 2, f"{pi:.0f}%", ha="center", va="center", color="w", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "element_site_types.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved element_site_types.png")
    print(f"  Total unique sites: {len(df_filt)}")
else:
    print("  WARNING: site_types_mixed column not found")


# ======================================================================
# Figure 8: Eseg vs R_Wsep_RGS (Rice-Wang work of separation)
#   Uses KS data from pre-computed df_main (cleavage calcs are KS-only)
# ======================================================================
print("\nFigure 8: Eseg vs R_Wsep (KS, from df_main)...")
df_main_path = CHECKPOINT_DIR / "09_df_main_final.pkl.gz"
if df_main_path.exists():
    df_main = pd.read_pickle(df_main_path)
    df_main["GB_string"] = df_main["GB"]

    element_label_position_fig8 = {
        "H": (0.78, 0.93), "He": (0.05, 0.25), "B": (0.78, 0.93), "C": (0.78, 0.93),
        "N": (0.78, 0.93), "O": (0.78, 0.93), "P": (0.78, 0.93), "S": (0.78, 0.93),
    }
    SCATTER_KW_WSEP = dict(SCATTER_KW)
    SCATTER_KW_WSEP["label_fontsize"] = 30
    SCATTER_KW_WSEP["legend_fontsize"] = 17.5
    SCATTER_KW_WSEP["style_legend_fontsize"] = 17.5
    SCATTER_KW_WSEP["legend_anchor"] = (0.001, 0.00)
    SCATTER_KW_WSEP["style_legend_anchor"] = (0.35, 0.00)

    for ele in LIGHT_ELEMENTS:
        df_ele = df_main[(df_main["element"] == ele) & (df_main["VorNN_tot_vol"] < 20) & (df_main["eta_Wsep_RGS_min"] < 3)]
        show = ele == "H"
        _, (fig, ax) = scatter_by_gb(
            df_ele, x="Eseg", y="R_Wsep_RGS_min",
            x_label=r"E$_{\rm{seg}}$ (eV/atom)",
            y_label=r"R$_{\rm{Wsep}}$",
            xlim=[-3, 0.2], ylim=[0.4, 1.5],
            show_legend=show, show_style_legend=show,
            **SCATTER_KW_WSEP,
        )
        ax.axhline(1, linestyle="--", color="k")
        ax.axvline(0, linestyle="--", color="k")
        pos = element_label_position_fig8[ele]
        ax.text(pos[0], pos[1], ele, transform=ax.transAxes, fontsize=70, ha="left", va="top")
        fig.savefig(FIG_DIR / f"Eseg_RWsepRGS_{ele}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved 8 Eseg_RWsepRGS plots")

    # ======================================================================
    # Figure 9: Eseg vs R_ANSBO (area-normalised summed bond orders)
    # ======================================================================
    print("\nFigure 9: Eseg vs R_ANSBO (KS, from df_main)...")
    element_label_position_fig9 = {
        "H": (0.05, 0.92), "He": (0.05, 0.25), "B": (0.05, 0.92), "C": (0.05, 0.92),
        "N": (0.05, 0.92), "O": (0.05, 0.92), "P": (0.05, 0.92), "S": (0.78, 0.96),
    }
    for ele in LIGHT_ELEMENTS:
        df_ele = df_main[(df_main["element"] == ele) & (df_main["VorNN_tot_vol"] < 20) & (df_main["eta_Wsep_RGS_min"] < 3)]
        show = ele == "S"
        _, (fig, ax) = scatter_by_gb(
            df_ele, x="Eseg", y="R_ANSBO_min",
            x_label=r"E$_{\rm{seg}}$ (eV/atom)",
            y_label=r"R$_{\rm{ANSBO}}$",
            xlim=[-3, 0.2], ylim=[0.6, 1.3],
            show_legend=show, show_style_legend=show,
            **SCATTER_KW_WSEP,
        )
        ax.axhline(1.0, linestyle="--", color="k")
        ax.axvline(0, linestyle="--", color="k")
        pos = element_label_position_fig9[ele]
        ax.text(pos[0], pos[1], ele, transform=ax.transAxes, fontsize=70, ha="left", va="top")
        fig.savefig(FIG_DIR / f"Eseg_R_ANSBO_{ele}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved 8 Eseg_R_ANSBO plots")

    # ======================================================================
    # Extra: Eseg vs R_BO_mean (average bond order)
    # ======================================================================
    print("\nExtra: Eseg vs R_averageBO (KS, from df_main)...")
    element_label_position_extra = {
        "H": (0.02, 0.80), "He": (0.02, 0.20), "B": (0.02, 0.20), "C": (0.02, 0.20),
        "N": (0.02, 0.20), "O": (0.02, 0.20), "P": (0.02, 0.20), "S": (0.02, 0.20),
    }
    for ele in LIGHT_ELEMENTS:
        df_ele = df_main[(df_main["element"] == ele) & (df_main["VorNN_tot_vol"] < 20) & (df_main["eta_Wsep_RGS_min"] < 3)]
        show = ele == "H"
        _, (fig, ax) = scatter_by_gb(
            df_ele, x="Eseg", y="R_BO_mean",
            x_label=r"E$_{\rm{seg}}$ (eV/atom)",
            y_label=r"$\Delta$average(BO)",
            show_legend=show, show_style_legend=show,
            **{k: v for k, v in SCATTER_KW_WSEP.items() if k not in ("style_legend_anchor",)},
            style_legend_anchor=(0.58, 0.00),
        )
        ax.axhline(1, linestyle="--", color="k")
        ax.axvline(0, linestyle="--", color="k")
        pos = element_label_position_extra[ele]
        ax.text(pos[0], pos[1], ele, transform=ax.transAxes, fontsize=70, ha="left", va="top")
        fig.savefig(FIG_DIR / f"Eseg_R_averageBO_{ele}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved 8 Eseg_R_averageBO plots")
else:
    print(f"  WARNING: {df_main_path} not found — skipping Wsep/ANSBO figures")


# ======================================================================
# Figure 5: Eseg histograms per element, split by site type
# ======================================================================
print("\nFigure 5: Eseg histograms by site type...")

site_colors = {"int": "tab:blue", "sub": "tab:orange", "mixed": "tab:green"}
site_order_hist = ["int", "sub", "mixed"]
site_labels_hist = {"int": "interstitial", "sub": "substitutional", "mixed": "mixed"}

for ele in LIGHT_ELEMENTS:
    ele_data = df_filt[df_filt["element"] == ele].copy()
    if len(ele_data) == 0:
        continue

    bin_width = 0.1 if ele == "H" else 0.2
    emin = np.floor(ele_data["Eseg"].min() / bin_width) * bin_width
    emax = np.ceil(ele_data["Eseg"].max() / bin_width) * bin_width + bin_width
    bins = np.arange(emin, emax, bin_width)

    show_legend = ele == "P"

    fig, ax = plt.subplots(figsize=(8, 6))
    data_by_type = []
    colors = []
    labels = []
    for st in site_order_hist:
        subset = ele_data[ele_data["site_types_mixed"] == st]["Eseg"]
        if len(subset) > 0:
            data_by_type.append(subset.values)
            colors.append(site_colors[st])
            labels.append(site_labels_hist[st])

    ax.hist(data_by_type, bins=bins, stacked=True, color=colors, label=labels,
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xlabel(r"E$_{\rm{seg}}$ (eV)", fontsize=24)
    ax.set_ylabel("Count", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    if show_legend:
        ax.legend(fontsize=20, loc="upper center", bbox_to_anchor=(0.5, 1.0))
   
    ax.text(0.08, 0.92, ele, transform=ax.transAxes, fontsize=70,
            ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"Eseg_histogram_{ele}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
print(f"  Saved 8 Fig 5 Eseg_histogram plots")


# ======================================================================
# Summary
# ======================================================================
print(f"\n{'='*60}")
print(f"All figures saved to: {FIG_DIR}")
print(f"Figures 3, 4, 6, 7, 10: KP data")
print(f"Figures 8-9 + averageBO: KS data (cleavage/Chargemol)")
print(f"Figure 5 (histograms): KP data (dedup'd)")
print(f"{'='*60}")
