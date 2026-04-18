#!/cmmc/ptmp/hmai/mambaforge/envs/pymatgen/bin/python
"""Generate ±0.2 eV filtered (zoom) KP vs KS dEseg histogram + boxplot.

The source dataframe 08_df_compare_pairwise already has the anomalous-SCF
filter applied (see 2026_04_01_KP_vs_KS_Eseg_Analysis.py line 245). Here
we additionally restrict to |dEseg| <= 0.2 eV so the bulk of the
distribution can be read without compression by a handful of outliers.
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
df_int['site_type'] = 'int'
df_sub['site_type'] = 'sub'
common = [c for c in ('element', 'GB', 'Eseg_KS', 'Eseg_KP', 'dEseg', 'site_type') if c in df_int.columns and c in df_sub.columns]
df = pd.concat([df_int[common], df_sub[common]], ignore_index=True)
print(f'int: {len(df_int)}, sub: {len(df_sub)}, combined: {len(df)}')

THRESH = 0.2
n_total = len(df)
mask = df['dEseg'].abs() <= THRESH
df_in = df[mask].copy()
df_out = df[~mask].copy()
print(f'Total (SCF-filtered) pairwise: {n_total}')
print(f'Within |dEseg|<={THRESH} eV: {mask.sum()} ({100*mask.mean():.1f}%)')
print(f'Outside |dEseg|>{THRESH} eV: {(~mask).sum()} ({100*(~mask).mean():.1f}%)')
print('\nOutliers by element:')
print(df_out['element'].value_counts())
print('\nOutliers head:')
print(df_out.sort_values('dEseg', key=abs, ascending=False)[['element','GB','Eseg_KS','Eseg_KP','dEseg']].head(15))

LIGHT = ['H','He','B','C','N','O','P','S']
colors = {'H': '#1f77b4', 'He': '#ff7f0e', 'B': '#2ca02c', 'C': '#d62728',
          'N': '#9467bd', 'O': '#8c564b', 'P': '#e377c2', 'S': '#7f7f7f'}

# --- Histogram ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(df_in['dEseg'], bins=60, edgecolor='black', linewidth=0.3,
        alpha=0.75, color='#4c72b0')
ax.axvline(df_in['dEseg'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"mean = {df_in['dEseg'].mean():.3f} eV")
ax.axvline(0, color='black', linewidth=0.8, alpha=0.6)
ax.set_xlabel(r'$\Delta E_\mathrm{seg} = E_\mathrm{seg}(\mathrm{KP}) - E_\mathrm{seg}(\mathrm{KS})$  (eV)', fontsize=22)
ax.set_ylabel('Count', fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.set_xlim(-THRESH, THRESH)
ax.legend(fontsize=16)
plt.tight_layout()
fig.savefig(OUT / 'SI_KP_vs_KS_dEseg_histogram_filt_pm02.png', dpi=300, bbox_inches='tight')
print(f"Saved {OUT / 'SI_KP_vs_KS_dEseg_histogram_filt_pm02.png'}")

# --- Boxplot ---
fig, ax = plt.subplots(figsize=(8, 6))
data = [df_in.loc[df_in['element'] == el, 'dEseg'].values for el in LIGHT]
bp = ax.boxplot(data, labels=LIGHT, patch_artist=True, showfliers=True,
                flierprops=dict(marker='.', markersize=4, alpha=0.5))
for patch, el in zip(bp['boxes'], LIGHT):
    patch.set_facecolor(colors[el])
    patch.set_alpha(0.6)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)
ax.set_ylabel(r'$\Delta E_\mathrm{seg}$  (eV)', fontsize=22)
ax.set_xlabel('Element', fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.set_ylim(-THRESH, THRESH)
plt.tight_layout()
fig.savefig(OUT / 'SI_KP_vs_KS_dEseg_boxplot_filt_pm02.png', dpi=300, bbox_inches='tight')
print(f"Saved {OUT / 'SI_KP_vs_KS_dEseg_boxplot_filt_pm02.png'}")
