#!/cmmc/ptmp/hmai/mambaforge/envs/pymatgen/bin/python
"""Plot Spearman rho vs NN shell index for all 8 elements on a single panel."""
import matplotlib.pyplot as plt
import numpy as np

rho = {
    'H':  [-0.88, -0.86, -0.81, -0.85, +0.54, +0.41],
    'He': [+0.53, -0.61, -0.64, -0.64, -0.30, -0.13],
    'B':  [-0.90, -0.84, -0.78, -0.80, -0.82, -0.69],
    'C':  [-0.80, -0.81, -0.13, +0.01, +0.09, +0.03],
    'N':  [-0.79, -0.80, -0.34, -0.19, -0.26, +0.48],
    'O':  [-0.53, -0.57, -0.41, -0.28, -0.06, -0.17],
    'P':  [-0.61, -0.12, -0.32, -0.16, -0.59, -0.44],
    'S':  [-0.72, -0.49, -0.67, +0.17, +0.02, -0.53],
}

colors = {
    'H': '#1f77b4', 'He': '#ff7f0e', 'B': '#2ca02c', 'C': '#d62728',
    'N': '#9467bd', 'O': '#8c564b', 'P': '#e377c2', 'S': '#7f7f7f',
}
markers = {'H': 'o', 'He': 's', 'B': '^', 'C': 'D', 'N': 'v', 'O': 'P', 'P': 'X', 'S': '*'}

k = np.arange(1, 7)
fig, ax = plt.subplots(figsize=(8, 6))

for el, vals in rho.items():
    ax.plot(k, vals, color=colors[el], marker=markers[el], markersize=12,
            markeredgewidth=1.5, markeredgecolor='black',
            linewidth=2, label=el)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.set_xlabel(r'Nearest-neighbour shell index $k$', fontsize=24)
ax.set_ylabel(r'Spearman $\rho_\mathrm{full}$', fontsize=24)
ax.set_xticks(k)
ax.tick_params(axis='both', labelsize=20)
ax.set_ylim(-1.0, 1.0)
ax.legend(fontsize=16, ncol=2, loc='lower right', frameon=True)
ax.grid(alpha=0.3)

plt.tight_layout()
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
out = REPO / 'figures' / 'SI_rho_vs_kNN_shell.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
