# Main Manuscript Figure Scripts

Scripts to regenerate all figures appearing in the main manuscript.

## Scripts

| Script | Figures Generated | Dependencies |
|---|---|---|
| `generate_main_figures.py` | All main-text figures (see below) | Checkpoints from `2026_04_01_KP_vs_KS_Analysis/` |

## Figures produced

The monolithic `generate_main_figures.py` produces all per-element figure sets:

| Figure set | Pattern | Count |
|---|---|---|
| Min Eseg vs GB energy | `min_eseg_vs_gb_energy_per_element.png` | 1 |
| Eseg vs distance from GB | `Eseg_dist_GB_{ele}.png` | 8 |
| Voronoi volume + NN dist vs Eseg | `Eseg_Voronoi_NN_dist_{ele}.png` | 8 |
| Min NN distance vs Eseg | `Eseg_min_nn_dist_{ele}.png` | 8 |
| Site type classification | `element_site_types.png` | 1 |
| Rice-Wang cohesion maps | `Eseg_RWsepRGS_{ele}.png` | 8 |
| ANSBO cohesion maps | `Eseg_R_ANSBO_{ele}.png` | 8 |
| Eseg histograms | `Eseg_histogram_{ele}.png` | 8 |

Elements: H, He, B, C, N, O, P, S.

## Usage

```bash
/cmmc/ptmp/hmai/mambaforge/envs/pymatgen/bin/python generate_main_figures.py
```

The script reads checkpoints from `/u/hmai/2025_06_18_FeInterstitials/2026_04_01_KP_vs_KS_Analysis/checkpoints/` and writes figures to its `figures_KP/` output directory. Copy the desired figures into `manuscript/Figures/` for the publication build.
