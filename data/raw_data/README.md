# Raw DFT DataFrames — full schema

VASP-derived pandas DataFrames containing per-calculation outputs (energies, forces, stresses, magnetic moments, ionic-step structures, job names, INCARs, KPOINTS, etc.) for every job that contributes to the figures in the manuscript. All files are gzip-compressed pandas pickles.

## File index

| File | Rows | Cols | Contents |
|---|---|---|---|
| `FeGB_Int_vasp_database.pkl.gz` | 4,941 | 19 | Interstitial GB segregation calculations |
| `FeGB_Sub_vasp_database.pkl.gz` | 4,183 | 19 | Substitutional GB segregation calculations |
| `Fe_Pure_GB_SLAB_vasp_database.pkl.gz` | 22 | 19 | Pure Fe GB and slab references |
| `FeBulk_Int_vasp_database.pkl.gz` | 28 | 19 | Bulk interstitial reference cells |
| `FeBulk_Sub_vasp_database.pkl.gz` | 120 | 19 | Bulk substitutional reference cells (incl. pure-Fe `bulk_128_atom_bcc_Fe_Fe`) |
| `FeGB_Int_Cleaved_vasp_database.pkl.gz` | 7,415 | 19 | Cleaved interstitial GBs for Wsep (combined from three submission batches; deduplicated by `job_name`; 21 unconverged retained) |
| `FeGB_Pure_Cleaved_vasp_database.pkl.gz` | 71 | 19 | Cleaved pure GB references for Wsep |
| `chargemol_FeGBInterstitial.pkl.gz` | 666 | 28 | DDEC6 bond orders for interstitial-segregated GBs |
| `chargemol_FeGBPure.pkl.gz` | 22 | 11 | DDEC6 bond orders for pure GBs |

## Loading

```python
import pandas as pd
df = pd.read_pickle("data/raw_data/FeGB_Int_vasp_database.pkl.gz")  # gzip auto-detected
print(df.iloc[0])
```

---

## Common schema — VASP-derived files (19 columns)

The seven files `FeGB_Int_*`, `FeGB_Sub_*`, `Fe_Pure_GB_SLAB_*`, `FeBulk_Int_*`, `FeBulk_Sub_*`, `FeGB_Int_Cleaved_*`, `FeGB_Pure_Cleaved_*` share the same 19-column schema. Each row is one VASP calculation.

| Column | dtype | Description |
|---|---|---|
| `job_name` | `str` | Unique job identifier. Patterns: `<GB>-i<X>-site-<idx>` (interstitial GB seg), `<GB>-<X>-<idx>` (substitutional GB seg), `<GB>-cleaved-<plane>-<frac>` (cleaved), `Fe_bulk_interstitial_<X>_<idx>` (bulk int), `bulk_128_atom_bcc_Fe_<X>` (bulk sub), `<GB>-GB`/`<GB>-SLAB` (pure refs). |
| `filepath` | `str` | Original path on disk where the calculation was run (for provenance only; not needed to load the data). |
| `calc_start_time` | `pd.Timestamp` | Wall-clock submission/start time. |
| `consumed_time` | `dict` | `{cpu_time, user_time, system_time}` in seconds. |
| `structures` | `np.ndarray[str]` | Per-ionic-step pymatgen Structure objects, JSON-serialised. The final entry (`structures[-1]`) is the converged geometry. Deserialise with `pymatgen.core.Structure.from_str(str(s), fmt="json")`. |
| `energy` | `np.ndarray[float]` | Per-ionic-step total DFT energy (eV). The final entry is the converged total energy. |
| `energy_zero` | `np.ndarray[float]` | Per-ionic-step `E(σ→0)` (entropy-extrapolated energy, eV). |
| `forces` | `np.ndarray` | Per-ionic-step per-atom forces (eV/Å), shape `(n_steps, n_atoms, 3)`. |
| `stresses` | `np.ndarray` | Per-ionic-step stress tensor, shape `(n_steps, 3, 3)`. |
| `magmoms` | `np.ndarray` | Per-ionic-step per-atom magnetic moments (μ_B), shape `(n_steps, n_atoms)`. |
| `scf_steps` | `list[int]` | Number of electronic SCF iterations per ionic step. |
| `scf_convergence` | `list[bool]` | Per-ionic-step electronic convergence flag. |
| `convergence` | `bool` | Overall calculation convergence (forces below `EDIFFG`, energies converged). |
| `KPOINTS` | `str` | Either an explicit Γ-centred mesh string (e.g. `"4 3 1"`) or `"KSPACING: 0.5"` when VASP's `KSPACING` was used instead of an explicit `KPOINTS` file. |
| `INCAR` | `dict` | Full INCAR for the run, e.g. `{'ALGO': 'Fast', 'ENCUT': 400, 'ISMEAR': 1, 'SIGMA': 0.2, 'NSW': 300, 'EDIFF': 1e-5, 'EDIFFG': -0.01, …}`. |
| `element_list` | `list[str]` | Element symbols per contiguous block in the POSCAR (e.g. `['Fe', 'X', 'Fe']`). |
| `element_count` | `list[int]` | Atom count for each block in `element_list` (e.g. `[1, 1, 126]` ⇒ 1 Fe + 1 X + 126 Fe = 127 Fe + 1 X). |
| `electron_count` | `float` | Total electron count (sum over all atoms × `potcar_electron_count`). |
| `potcar_electron_count` | `list[float]` | Number of valence electrons per element in `element_list` from the POTCAR (e.g. `[8.0, 5.0, 8.0]` for `[Fe, P, Fe]`). |

### Counting Fe atoms

The Fe count is the sum of `element_count[i]` for indices where `element_list[i] == 'Fe'`. See `scripts/example_compute_eseg.py:add_E_and_nFe` for a reference implementation.

### Per-file naming nuances

- **`FeGB_Int_vasp_database.pkl.gz`**: GB prefix uses `RA100` for S5 GBs (a typo for `RA001`), and the seg cells use the doubled supercell. Map `S5-RA100-<X>` → `S5-RA001-<X>-d` to match the pure-GB reference.
- **`FeGB_Int_Cleaved_vasp_database.pkl.gz`**: Same `RA100` typo. `job_name` extends with a `-cleaved-<plane_idx>-<frac>` suffix.
- **`FeBulk_Sub_vasp_database.pkl.gz`**: Includes `bulk_128_atom_bcc_Fe_Fe` (the **pure-Fe 128-atom reference**, used for μ_Fe). Total energy: −1054.3935 eV → μ_Fe = −8.2374 eV/atom.
- **`FeBulk_Int_vasp_database.pkl.gz`**: Four sites per element (sub-indexed `_0`..`_3`); the lowest-energy entry is the canonical reference.
- **`Fe_Pure_GB_SLAB_vasp_database.pkl.gz`**: Two row types per GB — `<GB>-GB` and `<GB>-SLAB`. The `*-GB` rows are the pure-GB reference for `E_pure_GB`. S5 entries appear in both single (`S5-RA001-S210`) and doubled (`S5-RA001-S210-d`) cells.

---

## `chargemol_FeGBInterstitial.pkl.gz` (28 cols, 666 rows)

DDEC6 bond-order analysis for each interstitial-segregated GB, computed with [Chargemol](https://sourceforge.net/p/ddec/wiki/Home/) on the relaxed VASP outputs.

| Column | dtype | Description |
|---|---|---|
| `job_name` | `str` | Identifier matching the segregation row in `FeGB_Int_vasp_database.pkl.gz` (template `<GB>-i<X>-site-<idx>`). |
| `directory`, `filepath` | `str` | Source directory paths (provenance). |
| `GB` | `str` | Underscore-form GB key (e.g. `S11_RA110_S3`). |
| `element` | `str` | Solute element (`B`, `C`, `H`, `He`, `N`, `O`, `P`, `S`, …). |
| `structure` | `pymatgen.core.Structure` | Final relaxed structure (already deserialised). |
| `layer_boundaries` | `np.ndarray[float]` | z-coordinates (Å) of the atomic layers along the GB normal. |
| `cleavage_coord` | `list[float]` | z-coordinates of the candidate cleavage planes (between layers). |
| `ANSBO_profile` | `list[float]` | Area-normalised summed bond order (Å⁻²) for each cleavage plane (parallel to `cleavage_coord`). |
| `bond_order_min/max/mean/std` | `float` | Statistics of bond orders crossing the **weakest** cleavage plane. |
| `bond_order_sums` | `float` | Sum of bond orders crossing the weakest cleavage plane. |
| `n_bonds` | `int` | Number of bonds crossing the weakest cleavage plane. |
| `min_max_bo`, `plane_min_max` | `float` | Maximum bond order at the weakest plane and its z-coordinate. |
| `min_mean_bo`, `plane_min_mean` | `float` | Per-plane mean bond order at the plane that minimises the mean (and its z-coordinate). |
| `ddec_charges`, `cm5_charges` | `float` | DDEC6 and CM5 charges on the segregant atom. |
| `ddec_spin_moments` | `float` | DDEC6-derived spin moment on the segregant. |
| `ddec_rcubed_moments`, `ddec_rfourth_moments` | `float` | Higher-order DDEC6 atomic moments (size descriptors) on the segregant. |
| `dipoles` | `list[float]` | Atomic dipole vector components on the segregant. |
| `charge_transfer`, `partial_charge` | `float` | Net charge transfer to/from the segregant. |
| `bonding_additional_df` | `pd.DataFrame` | Per-cleavage-plane breakdown (bond-by-bond table) — embedded DataFrame with one row per bond crossing each candidate plane. |

## `chargemol_FeGBPure.pkl.gz` (11 cols, 22 rows)

Same Chargemol analysis, applied to the pure GB / slab references.

| Column | dtype | Description |
|---|---|---|
| `job_name` | `str` | One of `<GB>-GB` or `<GB>-SLAB`. |
| `directory` | `str` | Source directory. |
| `structure` | `pymatgen.core.Structure` | Final relaxed pure structure. |
| `layer_boundaries`, `cleavage_coord`, `ANSBO_profile` | as above | Layer/plane geometry and per-plane ANSBO for the pure cell (used as the denominator in $R_\mathrm{ANSBO}$). |
| `min_max_bo`, `plane_min_max`, `min_mean_bo`, `plane_min_mean` | `float` | As in the segregated table. |
| `bonding_additional_df` | `pd.DataFrame` | Per-plane bond breakdown. |
