import pandas as pd
import numpy as np

from pymatgen.core import Element
from pymatgen.core import Structure
from ase import Atoms

def count_fe_safe(a):
    if isinstance(a, Atoms):
        return sum(sym == "Fe" for sym in a.get_chemical_symbols())
    elif isinstance(a, Structure):
        return sum(site.species_string == "Fe" for site in a)
    return np.nan
    
def get_non_element(structure, host: str = "Fe", *, return_symbol: bool = True, strict: bool = True):
    """
    Return the non-host element present in a pymatgen Structure.

    - If exactly one non-host element exists: return it (symbol or Element).
    - If none: return None.
    - If multiple and strict=True: raise ValueError. If strict=False: return a sorted list.

    Parameters
    ----------
    structure : pymatgen.core.Structure
    host : str
        Host element symbol to ignore (default "Fe").
    return_symbol : bool
        If True, return element symbol(s) as str; else return Element object(s).
    strict : bool
        If True, require exactly one non-host element.

    """
    # Unique element symbols in the structure
    elems = {el.symbol for el in structure.composition.elements}
    non_host = sorted(e for e in elems if e != host)

    if not non_host:
        return None
    if strict and len(non_host) != 1:
        raise ValueError(f"Expected exactly one non-{host} element, found: {non_host}")

    if len(non_host) == 1:
        return non_host[0] if return_symbol else Element(non_host[0])
    # multiple non-hosts and strict=False
    return non_host if return_symbol else [Element(e) for e in non_host]

def add_eseg_columns(
    df_seg: pd.DataFrame,
    df_pure_gb: pd.DataFrame,
    df_elem_bulk: pd.DataFrame,   # per-element bulk table (e.g., soln_df_DFT)
    df_host_bulk: pd.DataFrame,   # host-only bulk table (e.g., df_bulk)
    *,
    key_gb: str = "GB",
    key_element: str = "element",
    nfe_col: str = "n_Fe",
    energy_col: str = "energy",   # <-- single energy column now
    host_element: str = "Fe",
    host_supercell_natoms: int = 128,
    exclude_elements: tuple[str, ...] = ("Yb",),
    validate_unique: bool = True,
    overwrite: bool = True,       # kept for API compatibility (unused)
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Vectorized segregation energy with host-per-atom correction (single energy column):
        n_Fe_diff = nFe_seg - nFe_pureGB - (nFe_elem_bulk - nFe_host_bulk)
        Eseg      = E_seg - E_pureGB - (E_elem_bulk - E_host_bulk)
                    - n_Fe_diff * [E_host_bulk / host_supercell_natoms]

    Requires `energy_col` and `nfe_col` in df_seg, df_pure_gb, df_elem_bulk and host row in df_host_bulk.
    """
    # 1) filter unwanted elements & copy
    seg = df_seg.copy()
    if exclude_elements:
        seg = seg[~seg[key_element].isin(exclude_elements)].copy()

    # 2) sanity: required columns
    for name, df in [("df_seg", seg), ("df_pure_gb", df_pure_gb),
                     ("df_elem_bulk", df_elem_bulk), ("df_host_bulk", df_host_bulk)]:
        if nfe_col not in df.columns:
            raise KeyError(f"'{nfe_col}' not in {name}.")
        if energy_col not in df.columns:
            needed = list(df.columns)
            raise KeyError(f"'{energy_col}' not in {name}. Available: {needed}")

    # 3) prepare lookups (unique keys)
    pure = df_pure_gb.loc[:, [key_gb, nfe_col, energy_col]].drop_duplicates(key_gb)
    if validate_unique and pure.duplicated(key_gb).any():
        raise ValueError("df_pure_gb has non-unique GB keys.")

    elem = df_elem_bulk.loc[:, [key_element, nfe_col, energy_col]].drop_duplicates(key_element)
    if validate_unique and elem.duplicated(key_element).any():
        raise ValueError("df_elem_bulk has non-unique element keys.")

    # check coverage of elements
    missing_elems = set(seg[key_element].unique()) - set(elem[key_element].unique())
    if missing_elems:
        raise KeyError(f"Missing element-bulk rows for: {sorted(missing_elems)}")

    # host bulk (single row)
    host_rows = df_host_bulk[df_host_bulk[key_element] == host_element]
    # print(df_host_bulk[df_host_bulk[key_element] == host_element])
    if host_rows.empty:
        raise KeyError(f"Host element '{host_element}' not found in df_host_bulk[{key_element!r}].")
    host = host_rows.iloc[[0]].loc[:, [key_element, nfe_col, energy_col]]

    # 4) merges
    seg = seg.merge(
        pure.rename(columns={nfe_col: "_pure_nFe", energy_col: "_pure_energy"}),
        on=key_gb, how="left", validate="m:1",
    )
    seg = seg.merge(
        elem.rename(columns={nfe_col: "_elem_nFe", energy_col: "_elem_energy"}),
        on=key_element, how="left", validate="m:1",
    )
    if seg["_pure_nFe"].isna().any():
        missing_gb = seg.loc[seg["_pure_nFe"].isna(), key_gb].unique()
        raise KeyError(f"Missing pure GB rows for GB={missing_gb.tolist()}")

    # 5) host scalars
    host_nFe = float(host[nfe_col].iloc[0])
    host_energy = float(host[energy_col].iloc[0])

    # 6) n_Fe_diff
    seg["n_Fe_diff"] = seg[nfe_col] - seg["_pure_nFe"] - (seg["_elem_nFe"] - host_nFe)

    # 7) segregation energy (single output)
    e_host_per_atom = host_energy / float(host_supercell_natoms)
    seg["Eseg"] = (
        seg[energy_col]
        - seg["_pure_energy"]
        - (seg["_elem_energy"] - host_energy)
        - seg["n_Fe_diff"] * e_host_per_atom
    )

    # 8) cleanup helper cols (keep n_Fe_diff)
    seg = seg.drop(columns=["_pure_nFe", "_elem_nFe", "_pure_energy", "_elem_energy"])

    # if verbose:
    #     print(f"Added columns: ['Eseg', 'n_Fe_diff'] (host={host_element}, N={host_supercell_natoms}).")

    return seg

    
def get_soln_df(
    bulk_sub_df: pd.DataFrame,
    bulk_int_df: pd.DataFrame,
    *,
    host_element: str = "Fe",
    host_supercell_natoms: int = 128,
    element_col: str = "element",
    sub_energy_col: str = "energy",   # total energy (N atoms, substitutional)
    int_energy_col: str = "energy",   # total energy (N+1 atoms, interstitial)
    structure_col: str = "structure", # structure object/identifier
) -> pd.DataFrame:
    """
    For each element, pick the most energetically favourable configuration:
      - Compare best interstitial (N+1 atoms) vs substitutional (N atoms + one host atom).
      - If interstitial is preferred, return that interstitial structure; otherwise the substitutional structure.
    Also returns solution energy relative to pure host supercell.
    """
    # Clean + ensure numeric
    sub_df = bulk_sub_df.dropna(subset=[element_col, sub_energy_col]).copy()
    int_df = bulk_int_df.dropna(subset=[element_col, int_energy_col]).copy()
    # sub_df[sub_energy_col] = pd.to_numeric(sub_df[sub_energy_col], errors="coerce")
    # int_df[int_energy_col] = pd.to_numeric(int_df[int_energy_col], errors="coerce")
    sub_df = sub_df.dropna(subset=[sub_energy_col])
    int_df = int_df.dropna(subset=[int_energy_col])

    # Host reference energy (N atoms)
    host_mask = sub_df[element_col] == host_element
    if not host_mask.any():
        raise ValueError(f"No host element {host_element!r} found in bulk_sub_df.")
    host_ref_energy = float(sub_df.loc[host_mask, sub_energy_col].iloc[0])
    host_per_atom_energy = host_ref_energy / float(host_supercell_natoms)

    rows_out = []
    for _, row in sub_df.iterrows():
        elem = row[element_col]
        e_sub = float(row[sub_energy_col])  # substitutional (N atoms)
        sub_struct = row.get(structure_col, np.nan)

        # Best interstitial for this element (if any)
        corr_int = int_df[int_df[element_col] == elem]
        best_int_energy = np.nan
        best_int_struct = np.nan
        if not corr_int.empty:
            best_int_row = corr_int.sort_values(int_energy_col, kind="mergesort").iloc[0]
            best_int_energy = float(best_int_row[int_energy_col])
            best_int_struct = best_int_row.get(structure_col, best_int_row.get("site", np.nan))

        # Equalize atom count for comparison: compare (E_sub + E_host_atom) vs E_int(N+1)
        bulk_toten = e_sub + host_per_atom_energy

        if not np.isnan(best_int_energy) and best_int_energy < bulk_toten:
            # Interstitial preferred
            chosen_config = "interstitial"
            chosen_energy = best_int_energy
            chosen_structure = best_int_struct
            soln_energy = best_int_energy - host_ref_energy     # relative to pure host (N atoms)
            preference_by = bulk_toten - best_int_energy        # positive margin
        else:
            # Substitutional preferred
            chosen_config = "substitutional"
            chosen_energy = e_sub
            chosen_structure = sub_struct
            soln_energy = e_sub - host_ref_energy
            preference_by = np.nan

        rows_out.append({
            element_col: elem,
            "chosen_config": chosen_config,     # "interstitial" or "substitutional"
            "soln_en": soln_energy,
            "energy": chosen_energy,     # total energy of the chosen configuration
            "preference_by": preference_by,     # only meaningful if interstitial is chosen
            "structure": chosen_structure,      # the requested single "best" structure
            "n_Fe": count_fe_safe(chosen_structure)
        })

    return pd.DataFrame(rows_out).reset_index(drop=True)