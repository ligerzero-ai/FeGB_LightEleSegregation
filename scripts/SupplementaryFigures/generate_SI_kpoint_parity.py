#!/cmmc/ptmp/hmai/mambaforge/envs/pymatgen/bin/python
"""Generate SI k-point convergence figures: parity scatter, dEseg histogram, dEseg boxplot.

Requires: checkpoints/08_df_compare_pairwise.pkl.gz and 08_df_sub_compare_pairwise.pkl.gz
from the 2026_04_01_KP_vs_KS_Analysis pipeline.

Outputs to manuscript/Figures/:
  - SI_KP_vs_KS_Eseg_scatter.png
  - SI_KP_vs_KS_dEseg_histogram.png
  - SI_KP_vs_KS_dEseg_boxplot.png
"""
import pickle, gzip
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CHK = REPO / 'data' / 'checkpoints'
OUT = REPO / 'figures'

with gzip.open(CHK / '08_df_compare_pairwise.pkl.gz', 'rb') as f:
    df_int = pickle.load(f)
with gzip.open(CHK / '08_df_sub_compare_pairwise.pkl.gz', 'rb') as f:
    df_sub = pickle.load(f)

common = [c for c in ('element','Eseg_KS','Eseg_KP','dEseg') if c in df_int.columns and c in df_sub.columns]
df = pd.concat([df_int[common], df_sub[common]], ignore_index=True)

LIGHT = ['H','He','B','C','N','O','P','S']
colors = {'H':'#1f77b4','He':'#ff7f0e','B':'#2ca02c','C':'#d62728',
          'N':'#9467bd','O':'#8c564b','P':'#e377c2','S':'#7f7f7f'}

# --- Parity scatter ---
fig, ax = plt.subplots(figsize=(7,7))
for el in LIGHT:
    sub = df[df['element']==el]
    ax.scatter(sub['Eseg_KS'], sub['Eseg_KP'], s=8, alpha=0.4, label=el, color=colors[el])
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
ax.set_xlabel(r'$E_\mathrm{seg}$ (KS) / eV', fontsize=20)
ax.set_ylabel(r'$E_\mathrm{seg}$ (KP) / eV', fontsize=20)
ax.tick_params(labelsize=16)
ax.legend(fontsize=14)
ax.set_aspect('equal')
plt.tight_layout()
fig.savefig(OUT / 'SI_KP_vs_KS_Eseg_scatter.png', dpi=300, bbox_inches='tight')
print('Saved scatter')

# --- Histogram ---
fig, ax = plt.subplots(figsize=(8,6))
ax.hist(df['dEseg'], bins=80, edgecolor='black', linewidth=0.3, alpha=0.7, color='#4c72b0')
ax.axvline(df['dEseg'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"mean = {df['dEseg'].mean():.3f} eV")
ax.set_xlabel(r'$\Delta E_\mathrm{seg} = E_\mathrm{seg}(\mathrm{KP}) - E_\mathrm{seg}(\mathrm{KS})$  (eV)', fontsize=20)
ax.set_ylabel('Count', fontsize=20)
ax.tick_params(labelsize=16)
ax.legend(fontsize=16)
plt.tight_layout()
fig.savefig(OUT / 'SI_KP_vs_KS_dEseg_histogram.png', dpi=300, bbox_inches='tight')
print('Saved histogram')

# --- Boxplot ---
fig, ax = plt.subplots(figsize=(8,6))
data = [df.loc[df['element']==el, 'dEseg'].values for el in LIGHT]
bp = ax.boxplot(data, labels=LIGHT, patch_artist=True, showfliers=True,
                flierprops=dict(marker='.', markersize=4, alpha=0.5))
for patch, el in zip(bp['boxes'], LIGHT):
    patch.set_facecolor(colors[el]); patch.set_alpha(0.6)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)
ax.set_ylabel(r'$\Delta E_\mathrm{seg}$  (eV)', fontsize=20)
ax.set_xlabel('Element', fontsize=20)
ax.tick_params(labelsize=16)
plt.tight_layout()
fig.savefig(OUT / 'SI_KP_vs_KS_dEseg_boxplot.png', dpi=300, bbox_inches='tight')
print('Saved boxplot')
