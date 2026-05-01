# Cached analysis checkpoints — full schema

Pre-computed pandas DataFrames consumed by the figure-generation scripts. All files are gzip-compressed pandas pickles. Each row corresponds to one segregation site (or vacancy site for `KP_Vacancy_Formation_Energy.pkl.gz`).

These checkpoints are derived from the raw VASP DataFrames in `data/raw_data/`. The figure scripts read these checkpoints directly so the figures reproduce without re-running the analysis pipeline.

## File index

| File | Rows | Cols | Loaded by |
|---|---|---|---|
| `04_df_KP_voronoi.pkl.gz` | 4,528 | 57 | analysis (input to `07_df_KP_filtered`) |
| `04_df_KP_voronoi_with_nn.pkl.gz` | 4,528 | 63 | `MainFigures/generate_main_figures.py` (Fig. 4 NN-correlation) |
| `07_df_KP_filtered.pkl.gz` | 416 | 67 | `MainFigures/generate_main_figures.py` (Fig. 9), input to `09_df_main_final` |
| `08_df_compare_pairwise.pkl.gz` | 4,129 | 6 | `SupplementaryFigures/generate_SI_kpoint_*.py` |
| `08_df_sub_compare_pairwise.pkl.gz` | 382 | 6 | `SupplementaryFigures/generate_SI_kpoint_*.py` |
| `09_df_main_final.pkl.gz` | 408 | 116 | `MainFigures/generate_main_figures.py` (Figs. 6–10) |
| `KP_Vacancy_Formation_Energy.pkl.gz` | 56 | 8 | `SupplementaryFigures/generate_SI_vacancy_tables.py` |

## Loading

```python
import pandas as pd
df = pd.read_pickle("data/checkpoints/09_df_main_final.pkl.gz")
print(df.iloc[0])
```

---

## Common segregation-row columns

`04_df_KP_voronoi.pkl.gz`, `04_df_KP_voronoi_with_nn.pkl.gz`, `07_df_KP_filtered.pkl.gz`, and `09_df_main_final.pkl.gz` all start from the raw VASP-derived schema (see `data/raw_data/README.md`) and add per-site descriptors.

The columns listed below appear in all of them. Per-file additions are documented in the file-specific sections.

### Inherited from the raw VASP DataFrame

| Column | dtype | Description |
|---|---|---|
| `job_name` | `str` | Unique site identifier. Pattern `<GB>-i<X>-site-<idx>` (interstitial) or `<GB>-<X>-<idx>` (substitutional). |
| `filepath` | `str` | Original VASP run directory (provenance). |
| `structures` | `np.ndarray[str]` | Per-ionic-step pymatgen Structures, JSON-serialised. Last entry is the converged geometry. |
| `energy` | `float` | Final total DFT energy (eV). Reduced from the per-step array in the raw DataFrame. |
| `energy_zero` | `float` | Final entropy-extrapolated energy `E(σ→0)` (eV). |
| `forces` | `np.ndarray` | Per-step forces (eV/Å), shape `(n_steps, n_atoms, 3)`. |
| `stresses` | `np.ndarray` | Per-step stress tensor, shape `(n_steps, 3, 3)`. |
| `magmoms` | `np.ndarray` | Per-step per-atom magnetic moments (μ_B), shape `(n_steps, n_atoms)`. |
| `scf_steps` | `list[int]` | Electronic SCF iterations per ionic step. |
| `scf_convergence` | `list[bool]` *or* NaN | Per-step electronic convergence flag. (Some early checkpoints store this in `kpoints`/`incar` instead of the raw-style columns; see notes below.) |
| `convergence` | `bool` | Overall calculation convergence. |
| `kpoints` | `dict` | Parsed KPOINTS file (mesh, generation scheme). |
| `incar` | `dict` | Full INCAR for the run. |
| `KPOINTS`, `INCAR`, `consumed_time`, `calc_start_time`, `electron_count`, `scf_convergence` | (NaN in 04/07; populated in 09) | Raw-schema duplicates that were re-attached late in the pipeline. In `09_df_main_final.pkl.gz` these are the canonical raw-schema columns; in `04_*` and `07_*` they are NaN — use the lower-case variants instead. |
| `element_list` | `list[str]` | Element blocks in POSCAR order, e.g. `['Fe','P','Fe']`. |
| `element_count` | `list[int]` | Atom count per block, parallel to `element_list`. |
| `potcar_electron_count` | `list[float]` | POTCAR valence per element. |
| `total_electron_count` | `float` | Total electron count. |
| `final_structure` | `pymatgen.core.Structure` | Converged geometry (already deserialised). |

### Per-site descriptors (added at the 04 stage)

| Column | dtype | Description |
|---|---|---|
| `calculation_description` | `str` | `'GB_int'` or `'GB_sub'`. |
| `GB` | `str` | Underscore-form GB key (e.g. `S5_RA001_S310`). |
| `element` | `str` | Solute element. |
| `gb_energy` | `float` | GB excess energy (J/m²) of the underlying pure GB — used for plotting only. |
| `dataset` | `str` | `'KP'` (these checkpoints are for explicit Γ-mesh runs only). |
| `Eseg` | `float` | Production segregation energy (eV). Negative ⇒ favourable to GB. Computed via the formula in the top-level README; `scripts/example_compute_eseg.py` reproduces this column to machine precision from the raw data. |
| `midpoint` | `float` | Fractional z of the GB plane in the supercell (mean of the two GBs in a periodic slab). |
| `non_Fe_idx` | `int` | Index of the solute atom in the converged structure. |
| `solute_z_frac` | `float` | Fractional z of the solute. |
| `c_offset_frac` | `float` | `solute_z_frac - midpoint`. |
| `c_len_A` | `float` | c-axis length (Å). |
| `c_signed_A` | `float` | `c_offset_frac × c_len_A` — signed distance from the GB plane (Å). |
| `dist_GB` | `float` | `\|c_signed_A\|` — unsigned distance from GB plane (Å). |
| `site_idx` | `int` | Same as `non_Fe_idx`; retained for legacy callers. |
| `VorNN_CoordNo` | `int` | Voronoi coordination number of the solute. |
| `VorNN_tot_vol` | `float` | Total volume (Å³) of the solute's Voronoi cell. |
| `VorNN_tot_area` | `float` | Total surface area (Å²) of the Voronoi cell. |
| `VorNN_volumes_{std,mean,min,max}` | `float` | Statistics of per-face volume contributions of NN polyhedra. |
| `VorNN_vertices_{std,mean,min,max}` | `float`/`int` | Statistics of vertex counts on the Voronoi faces. |
| `VorNN_areas_{std,mean,min,max}` | `float` | Per-face area statistics. |
| `VorNN_distances_{std,mean,min,max}` | `float` | Distance from solute to each Voronoi face (Å). |
| `min_nn_dist` | `float` | Distance to nearest neighbour (Å). |

The Voronoi descriptors are computed by `scripts/featurisers.py:add_voronoi_descriptors`.

---

## `04_df_KP_voronoi.pkl.gz` (57 cols, 4,528 rows)

All 4,528 KP-mesh segregation calculations (interstitials + substitutionals across all 6 GBs and 8 elements) annotated with Voronoi descriptors. **No SOAP-based deduplication has been applied.** This is the entry point for the deduplication pipeline that produces `07_df_KP_filtered`.

Columns: the **Inherited** + **Per-site descriptors** sections above (57 total).

## `04_df_KP_voronoi_with_nn.pkl.gz` (63 cols, 4,528 rows)

Identical to `04_df_KP_voronoi.pkl.gz` plus six explicit nearest-neighbour distances:

| Column | dtype | Description |
|---|---|---|
| `nn_dist_1` … `nn_dist_6` | `float` | 1st through 6th nearest-neighbour distance from the solute (Å), sorted ascending. |

Used by the SI nearest-neighbour correlation figure.

## `07_df_KP_filtered.pkl.gz` (67 cols, 416 rows)

`04_df_KP_voronoi.pkl.gz` after **SOAP-based deduplication** of symmetry-equivalent sites and assignment of a coarse site-type label. The 4,528 → 416 reduction collapses sites whose SOAP fingerprints are within a fixed cutoff into a single representative row (the lowest-`Eseg` member of each cluster).

Adds:

| Column | dtype | Description |
|---|---|---|
| `SOAP` | `np.ndarray[float]` | SOAP fingerprint of the solute neighbourhood (averaged over species). |
| `SOAP_pca` | `np.ndarray[float]` | First few PCA components of `SOAP` (used for plotting / clustering). |
| `dropped_job_names` | `list[str]` | `job_name`s of the duplicate sites that were merged into this representative row. |
| `dropped_Esegs` | `np.ndarray[float]` | `Eseg` values of the dropped duplicates (eV). Useful for verifying the deduplication didn't discard a non-trivial site. |
| `site_type` | `str` | `'int'` (interstitial) or `'sub'` (substitutional). |
| `site_type_list` | `list[str]` | Site types of the dropped duplicates (empty when no duplicates). |
| `mixed_site_types` | `bool` | True when the dedup cluster contains both `int` and `sub` sites. |
| `site_types_mixed` | `str` | `'int'`, `'sub'`, or `'mixed'` — convenience flag for plotting. |
| `n_sites` | `int` | Cluster size (1 + len(`dropped_job_names`)). |
| `GB_string` | `str` | Same value as `GB`; used by some plotting helpers. |

## `08_df_compare_pairwise.pkl.gz` (6 cols, 4,129 rows)

KS↔KP pairwise comparison for **interstitial** sites. Each row is one solute site for which both a KSPACING-relaxed (`KS`) and an explicit-Γ-mesh (`KP`) static calculation exist.

| Column | dtype | Description |
|---|---|---|
| `job_name` | `str` | Site identifier (KS-form name, e.g. `S5-RA100-S310-iP-site-92`). |
| `Eseg_KS` | `float` | Production segregation energy from the KS calculation (eV). |
| `Eseg_KP` | `float` | Segregation energy from the KP calculation (eV). |
| `element` | `str` | Solute element. |
| `GB` | `str` | Underscore-form GB key. |
| `dEseg` | `float` | `Eseg_KP - Eseg_KS` (eV). |

`scripts/example_compute_eseg.py --verify` cross-checks `Eseg_KS` against the raw VASP DataFrames.

## `08_df_sub_compare_pairwise.pkl.gz` (6 cols, 382 rows)

Same schema as `08_df_compare_pairwise.pkl.gz` but for **substitutional** sites. The first column is `job_name_norm` (KS-form name canonicalised against the substitutional pickle's naming) instead of `job_name`.

## `09_df_main_final.pkl.gz` (116 cols, 408 rows)

The master post-deduplication DataFrame with **cohesion descriptors attached**. Starts from `07_df_KP_filtered.pkl.gz` (with the few rows that lack a Wsep/Chargemol match dropped, 416 → 408) and joins:

1. Rigid Rice-Wang cleavage (`Wsep_RGS_*`, `eta_Wsep_RGS_*`, `R_Wsep_RGS_min`).
2. Per-site Chargemol DDEC6 analysis (all `cm_*` columns, sourced from `chargemol_FeGBInterstitial.pkl.gz` in `data/raw_data/`).
3. Pure-GB Chargemol references (`pure_cm_*` and `pure_Wsep_RGS_min`).
4. ANSBO and bond-order ratios (`ANSBO_min`, `R_ANSBO_min`, `eta_ANSBO_min`, `R_BO_min`, `R_BO_mean`, `eta_BO_min`, `eta_BO_mean`).

Columns in addition to the **Inherited** + **Per-site descriptors** + **`07_*` additions** (see above):

### Cleaved-supercell totals (per-site)

| Column | dtype | Description |
|---|---|---|
| `cleavage_plane_lst` | `np.ndarray[float]` | Fractional z of each candidate cleavage plane considered for this site. |
| `cleavage_plane_idx_lst` | `np.ndarray[int]` | Plane index per entry of `cleavage_plane_lst`. |
| `cleavage_total_energy_lst` | `np.ndarray[float]` | Total DFT energy of each cleaved cell (eV). |
| `cleavage_final_structure_lst` | `np.ndarray[Structure]` | Converged cleaved geometries (one per plane). |

### Rigid Rice-Wang work of separation

| Column | dtype | Description |
|---|---|---|
| `Wsep_RGS_lst` | `np.ndarray[float]` | Per-plane rigid Rice-Wang work of separation (J/m²). |
| `Wsep_RGS_min` | `float` | Minimum-`W_sep` plane (the weakest plane for this site). |
| `eta_Wsep_RGS` | `np.ndarray[float]` | Per-plane embrittlement potency `W_sep^seg − W_sep^pure` (J/m²). |
| `eta_Wsep_RGS_min` | `float` | Embrittlement potency at the weakest plane. Negative ⇒ embrittling. |
| `pure_Wsep_RGS_min` | `float` | Reference `W_sep` of the corresponding pure-GB cleaved cell at the same plane (J/m²). |
| `R_Wsep_RGS_min` | `float` | `Wsep_RGS_min / pure_Wsep_RGS_min` — segregant cohesion ratio. <1 ⇒ embrittling. |

### Chargemol DDEC6 descriptors at the segregant

Joined from `data/raw_data/chargemol_FeGBInterstitial.pkl.gz`. All column meanings match the raw-data README; the `cm_` prefix marks them as joined columns.

| Column | dtype | Description |
|---|---|---|
| `cm_match_type` | `str` | How the Chargemol row was matched (`'original'`, `'doubled'`, `'failed'`). |
| `cm_index` | `int` | Atom index of the segregant in `cm_structure`. |
| `cm_directory`, `cm_filepath` | `str` | Provenance paths. |
| `cm_GB`, `cm_element` | `str` | Echoed for joins. |
| `cm_min_max_bo`, `cm_plane_min_max` | `float` | Maximum bond order at the weakest plane and its z-coordinate. |
| `cm_min_mean_bo`, `cm_plane_min_mean` | `float` | Mean bond order at the plane that minimises the mean, and its z-coordinate. |
| `cm_bond_order_{min,max,mean,std}` | `float` | Bond-order statistics across the weakest plane. |
| `cm_n_bonds` | `int` | Number of bonds crossing the weakest plane. |
| `cm_bond_order_sums` | `float` | Sum of bond orders across the weakest plane. |
| `cm_ddec_charges`, `cm_cm5_charges` | `float` | Atomic charges (DDEC6, CM5) on the segregant. |
| `cm_ddec_spin_moments` | `float` | DDEC6 spin moment on the segregant. |
| `cm_ddec_rcubed_moments`, `cm_ddec_rfourth_moments` | `float` | Higher-order DDEC6 atomic moments (size descriptors). |
| `cm_dipoles` | `list[float]` | Atomic dipole vector components on the segregant. |
| `cm_charge_transfer`, `cm_partial_charge` | `float` | Net and partial charge on the segregant. |
| `cm_layer_boundaries` | `np.ndarray[float]` | z-coordinates of atomic layers along the GB normal (Å). |
| `cm_cleavage_coord` | `list[float]` | Candidate cleavage plane z-coordinates (Å). |
| `cm_ANSBO_profile` | `list[float]` | Area-normalised summed bond order (Å⁻²) at each cleavage plane. |
| `cm_structure` | `pymatgen.core.Structure` | Final relaxed segregant cell used by Chargemol. |
| `cm_bonding_additional_df` | `pd.DataFrame` | Embedded bond-by-bond breakdown per cleavage plane. |

### Pure-GB Chargemol references

| Column | dtype | Description |
|---|---|---|
| `pure_cm_min_max_bo`, `pure_cm_plane_min_max` | `float` | As above for the matching pure GB. |
| `pure_cm_min_mean_bo`, `pure_cm_plane_min_mean` | `float` | As above for the matching pure GB. |
| `pure_cm_layer_boundaries`, `pure_cm_cleavage_coord`, `pure_cm_ANSBO_profile` | as above | Pure-GB layer/plane geometry. |
| `pure_cm_ANSBO_min` | `float` | Minimum ANSBO across the pure-GB profile (denominator of `R_ANSBO_min`). |

### ANSBO / bond-order embrittlement metrics

| Column | dtype | Description |
|---|---|---|
| `ANSBO_min` | `float` | Minimum ANSBO across `cm_ANSBO_profile` for the segregated cell (Å⁻²). |
| `eta_ANSBO_min` | `float` | `ANSBO_min - pure_cm_ANSBO_min` (Å⁻²). Negative ⇒ embrittling. |
| `R_ANSBO_min` | `float` | `ANSBO_min / pure_cm_ANSBO_min`. <1 ⇒ embrittling. |
| `eta_BO_min` | `float` | `cm_min_max_bo - pure_cm_min_max_bo`. |
| `R_BO_min` | `float` | `cm_min_max_bo / pure_cm_min_max_bo`. |
| `eta_BO_mean` | `float` | `cm_min_mean_bo - pure_cm_min_mean_bo`. |
| `R_BO_mean` | `float` | `cm_min_mean_bo / pure_cm_min_mean_bo`. |

---

## `KP_Vacancy_Formation_Energy.pkl.gz` (8 cols, 56 rows)

Per-site relaxed vacancy formation energies for a representative subset of GB sites (the deduplicated KP `site_type='sub'` set from `07_df_KP_filtered.pkl.gz`). Used by `scripts/SupplementaryFigures/generate_SI_vacancy_tables.py`.

| Column | dtype | Description |
|---|---|---|
| `job_name` | `str` | Vacancy run identifier, pattern `<GB_underscore>_vac_<site_idx>`. |
| `GB` | `str` | Underscore-form GB key. |
| `site` | `int` | Atom index of the vacated atom in the pure-GB cell. |
| `E_vac` | `float` | Total energy of the GB cell with one Fe removed (eV). |
| `E_pure_GB` | `float` | Total energy of the corresponding pure GB cell (eV). |
| `n_atoms_vac` | `int` | Atom count of the vacancy cell (= `n_atoms_pure - 1`). |
| `n_atoms_pure` | `int` | Atom count of the pure GB cell. |
| `E_vf` | `float` | Vacancy formation energy (eV): `E_vac - (n_atoms_vac / n_atoms_pure) × E_pure_GB`. |

Negative values would indicate spontaneous vacancy formation; in this set all `E_vf > 0` as expected.
